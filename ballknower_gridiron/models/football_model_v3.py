"""
ballknower_gridiron.models.football_model_v3
============================================

v3 NFL model. Inherits v2's rolling QB pipeline and adds **rolling
team efficiency** signals — also leakage-free.

What's new vs v2
----------------
Five new features in the feature vector:

  * net_pts_diff       home_net_pts_pg − away_net_pts_pg
                         (the user's "NET rating" — sign favors home)
  * net_epa_diff       home_net_epa_per_play − away_net_epa_per_play
                         (gold-standard efficiency signal for modern NFL)
  * off_epa_diff       home_off_epa − away_off_epa
                         (separates offense from defense)
  * def_epa_diff       away_def_epa − home_def_epa
                         (lower def_epa is better defense, so this
                         convention puts "positive favors home")
  * mean_pace          (home_plays_pg + away_plays_pg) / 2
                         (pace context — fast-paced games tend to be
                         higher-variance / less predictable)

How leakage-free works
----------------------
For any game on date D in season S, a team's team-efficiency features
are computed from PBP in games STRICTLY BEFORE D in S. When few or no
games have been played yet, the values are blended with the team's
END-OF-PRIOR-SEASON snapshot using a games-played-based interpolation:

  games_played <= settings.blend_start_games  ->  100% prior
  games_played >= settings.blend_end_games    ->  100% current rolling
  in between                                  ->  linear

The defaults (1, 4) reproduce the user's original "weeks 1-2 prior,
3-4 blend, 5+ current" intent, but use games-played so byes are
handled correctly.

QB rating composite (inherited from v2)
---------------------------------------
v3 doesn't redefine the QB rating — it inherits v2's rolling QB
pipeline verbatim. The composite z-scores 9 stats (pass YPG, pass
TDs/G, rush YPG, rush TDs/G, INTs, fumbles, completion %, traditional
passer rating, and ESPN QBR) against the prior-season distribution.

Persistence
-----------
Saves model.pkl + scaler.pkl + elo_state.json + feature_state.json +
qb_ratings.json (inherited) + team_metrics.json + metadata.json into
`models_artifacts/football/v3/`. The team_metrics.json stores per-team
LATEST rolling snapshot + PRIOR-SEASON end-of-season snapshot — not
the full rolling table.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.data.team_efficiency_loader import (
    blend_rolling_with_prior,
    compute_end_of_season_team_metrics,
    compute_rolling_team_metrics,
    get_default_team_metrics,
    latest_team_metrics_snapshot,
    load_pbp_for_seasons,
)
from ballknower_gridiron.models.football_elo import NFLEloSystem
from ballknower_gridiron.models.football_model_v2 import (
    FEATURE_COLUMNS_V2,
    NFLFeatureBuilderV2,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


FEATURE_COLUMNS_V3: List[str] = FEATURE_COLUMNS_V2 + [
    "net_pts_diff",     # sign-flips (favors home)
    "net_epa_diff",     # sign-flips (favors home)
    "off_epa_diff",     # sign-flips (favors home offense)
    "def_epa_diff",     # sign-flips (away_def - home_def; +ve = home D better)
    "mean_pace",        # symmetric context
]


# ---------------------------------------------------------------------------
# Feature builder v3 — ROLLING team efficiency
# ---------------------------------------------------------------------------
class NFLFeatureBuilderV3(NFLFeatureBuilderV2):
    """
    v2 builder (rolling QB) + ROLLING per-(team, season, game_date)
    team-efficiency metrics.

    Parallel to v2's QB structure:
      * `team_metrics_by_date` — full chronological lookup (training)
      * `team_metrics_latest`  — per-team latest snapshot (inference)
      * `team_metrics_prior`   — end-of-prior-season per team (early-week blend)
    """

    def __init__(self) -> None:
        super().__init__()
        # Rolling lookup populated from compute_rolling_team_metrics()
        self.team_metrics_by_date: Dict[
            Tuple[str, int, date_cls], Dict[str, float]
        ] = {}
        # Latest snapshot per team (for inference)
        self.team_metrics_latest: Dict[str, Dict[str, float]] = {}
        # Per-season prior end-of-season state (for early-week blend in training)
        self.end_of_season_by_team: Dict[int, Dict[str, Dict[str, float]]] = {}

    def load_rolling_team_metrics(
        self,
        seasons: List[int],
        games: pd.DataFrame,
        force_refresh: bool = False,
    ) -> None:
        """
        Pull PBP, compute rolling per-game team metrics, cache them on
        the builder. Should be called BEFORE building the training frame.

        Loads one prior season too so early-week games in the training
        period have a valid prior-season anchor.
        """
        seasons_with_prior = sorted(set(seasons) | {min(seasons) - 1})
        log.info(
            "[v3] Loading PBP for %d seasons (including prior anchor) …",
            len(seasons_with_prior),
        )
        pbp = load_pbp_for_seasons(seasons_with_prior, force_refresh=force_refresh)
        if pbp.empty:
            log.warning("[v3] No PBP — team efficiency features will be zero.")
            return

        # The rolling table is indexed by every scheduled (team, season, game_date).
        # We need games for ALL seasons we have PBP for, so we can compute the
        # prior-season game-by-game state too. Pull schedule extension if needed.
        log.info("[v3] Computing rolling team metrics …")
        self.team_metrics_by_date = compute_rolling_team_metrics(pbp, games)
        self.end_of_season_by_team = compute_end_of_season_team_metrics(pbp, games)

        # Build latest snapshot for the most recent training season.
        if self.latest_season is None:
            self.latest_season = max(seasons) if seasons else None
        if self.latest_season is not None:
            self.team_metrics_latest = latest_team_metrics_snapshot(
                self.team_metrics_by_date,
                self.end_of_season_by_team,
                self.latest_season,
            )
            log.info("[v3] Latest team-metrics snapshot — %d teams in season %s.",
                     len(self.team_metrics_latest), self.latest_season)

    def _team_metrics_at(
        self, team: str, season: int, game_date: date_cls,
    ) -> Dict[str, float]:
        """
        Look up (team, season, game_date) in the rolling table and
        blend with prior-season anchor by games_played. Falls back to
        latest snapshot or defaults if nothing rolling is available
        (inference time for future games).
        """
        key = (str(team), int(season), game_date)
        defaults = get_default_team_metrics()

        current = self.team_metrics_by_date.get(key)
        prior = self.end_of_season_by_team.get(season - 1, {}).get(team)

        if current is not None:
            # Training-time path: use blend based on games_played_before.
            games_played = current.get("games_played", 0.0)
            return blend_rolling_with_prior(current, prior, games_played)

        # Inference-time path: not in rolling table (predicting a future game).
        if self.team_metrics_latest:
            return {**defaults, **self.team_metrics_latest.get(team, {})}
        if prior is not None:
            return {**defaults, **prior}
        return defaults

    def features_for_matchup(
        self,
        home_team: str,
        away_team: str,
        game_date,
        season: int,
        week: int,
        home_rest: float,
        away_rest: float,
        is_playoff: bool = False,
        is_international: bool = False,
        is_div_game: bool = False,
        home_qb_out: bool = False,
        away_qb_out: bool = False,
        home_backup_qb_rating: Optional[float] = None,
        away_backup_qb_rating: Optional[float] = None,
    ) -> Dict[str, float]:
        base = super().features_for_matchup(
            home_team, away_team, game_date, season, week,
            home_rest, away_rest, is_playoff, is_international, is_div_game,
            home_qb_out=home_qb_out, away_qb_out=away_qb_out,
            home_backup_qb_rating=home_backup_qb_rating,
            away_backup_qb_rating=away_backup_qb_rating,
        )

        if hasattr(game_date, "date"):
            gd = game_date.date()
        else:
            gd = game_date

        home_m = self._team_metrics_at(home_team, season, gd)
        away_m = self._team_metrics_at(away_team, season, gd)

        base["net_pts_diff"] = home_m["net_pts_pg"] - away_m["net_pts_pg"]
        base["net_epa_diff"] = home_m["net_epa_per_play"] - away_m["net_epa_per_play"]
        base["off_epa_diff"] = home_m["off_epa_per_play"] - away_m["off_epa_per_play"]
        # Lower def_epa is better, so positive favors home.
        base["def_epa_diff"] = away_m["def_epa_per_play"] - home_m["def_epa_per_play"]
        base["mean_pace"] = (home_m["plays_per_game"] + away_m["plays_per_game"]) / 2.0
        return base

    def build_training_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        """Same shape as v2 but with team-efficiency features attached."""
        rows: List[Dict] = []
        log.info("Replaying NFL history to build v3 features (rolling team EPA) …")
        for row in games.itertuples(index=False):
            game_date = row.game_date
            if hasattr(game_date, "date"):
                game_date = game_date.date()

            feats = self.features_for_matchup(
                home_team=row.home_team,
                away_team=row.away_team,
                game_date=game_date,
                season=int(row.season),
                week=int(row.week),
                home_rest=float(row.home_rest),
                away_rest=float(row.away_rest),
                is_playoff=bool(getattr(row, "is_playoff", 0)),
                is_international=bool(getattr(row, "is_international", 0)),
                is_div_game=bool(getattr(row, "div_game", 0) or 0),
                home_qb_out=False, away_qb_out=False,
            )
            feats["home_won"] = int(row.home_won)
            feats["game_date"] = row.game_date
            feats["season"] = int(row.season)
            feats["week"] = int(row.week)
            rows.append(feats)

            self.record_game(
                home_team=row.home_team,
                away_team=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                game_date=game_date,
                season=int(row.season),
                is_playoff=bool(getattr(row, "is_playoff", 0)),
                is_international=bool(getattr(row, "is_international", 0)),
            )

        df = pd.DataFrame(rows)
        log.info("Built NFL v3 feature frame with shape %s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Trained v3 bundle
# ---------------------------------------------------------------------------
@dataclass
class NFLModelV3:
    """Trained v3 NFL model bundle. Same interface as v1 / v2."""

    clf: CalibratedClassifierCV
    scaler: StandardScaler
    elo: NFLEloSystem
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float]
    version: str = "v3"

    form_state: Dict[str, List[int]] = field(default_factory=dict)
    last_game_date: Dict[str, str] = field(default_factory=dict)

    # QB (inherited from v2 bundle shape)
    team_qb_ratings_latest: Dict[str, float] = field(default_factory=dict)
    team_qb_starters_latest: Dict[str, str] = field(default_factory=dict)
    # Team metrics (v3 additions)
    team_metrics_latest: Dict[str, Dict[str, float]] = field(default_factory=dict)
    team_metrics_prior: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latest_season: Optional[int] = None

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[date_cls] = None,
        season: Optional[int] = None,
        week: int = 1,
        home_rest: float = 7.0,
        away_rest: float = 7.0,
        is_playoff: bool = False,
        is_international: bool = False,
        is_div_game: bool = False,
        home_qb_out: bool = False,
        away_qb_out: bool = False,
        home_backup_qb_rating: Optional[float] = None,
        away_backup_qb_rating: Optional[float] = None,
        override_qb_ratings: Optional[Dict[str, float]] = None,
        override_team_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[float, float]:
        """Return (P(home wins), confidence)."""
        fb = NFLFeatureBuilderV3()
        fb.elo = self.elo
        if self.form_state:
            for team, vals in self.form_state.items():
                fb._form[team] = deque(vals, maxlen=5)
        if self.last_game_date:
            for team, iso in self.last_game_date.items():
                try:
                    fb._last_game_date[team] = date_cls.fromisoformat(iso)
                except Exception:  # noqa: BLE001
                    pass

        # QB snapshot (inference)
        fb.team_qb_ratings_latest = dict(self.team_qb_ratings_latest)
        fb.team_qb_starters_latest = dict(self.team_qb_starters_latest)
        fb.latest_season = self.latest_season
        if override_qb_ratings is not None:
            fb.team_qb_ratings_latest.update(override_qb_ratings)

        # Team-metrics snapshot (inference)
        fb.team_metrics_latest = dict(self.team_metrics_latest)
        if override_team_metrics is not None:
            for t, m in override_team_metrics.items():
                fb.team_metrics_latest[t] = {**get_default_team_metrics(), **m}
        # Prior-season as fallback for season-1 lookups (rare at inference).
        if self.latest_season is not None and self.team_metrics_prior:
            fb.end_of_season_by_team[self.latest_season - 1] = dict(self.team_metrics_prior)

        game_date = game_date or date_cls.today()
        season = season if season is not None else (
            self.latest_season or self.elo.last_season or game_date.year
        )

        feats = fb.features_for_matchup(
            home_team, away_team,
            game_date=game_date,
            season=int(season),
            week=int(week),
            home_rest=home_rest,
            away_rest=away_rest,
            is_playoff=is_playoff,
            is_international=is_international,
            is_div_game=is_div_game,
            home_qb_out=home_qb_out,
            away_qb_out=away_qb_out,
            home_backup_qb_rating=home_backup_qb_rating,
            away_backup_qb_rating=away_backup_qb_rating,
        )
        X = np.array([[feats[c] for c in self.feature_columns]], dtype=float)
        Xs = self.scaler.transform(X)
        p_home = float(self.clf.predict_proba(Xs)[0, 1])
        return p_home, abs(p_home - 0.5) * 2.0

    # ------ Accessors ----------------------------------------------------
    def get_qb_rating(self, team: str, season: Optional[int] = None) -> float:
        return float(self.team_qb_ratings_latest.get(team, 0.0))

    def get_starter_name(self, team: str, season: Optional[int] = None) -> str:
        return self.team_qb_starters_latest.get(team, "")

    def get_team_metrics(
        self, team: str, season: Optional[int] = None,
    ) -> Dict[str, float]:
        """Latest rolling snapshot of team metrics (no blend)."""
        if team in self.team_metrics_latest:
            return dict(self.team_metrics_latest[team])
        return get_default_team_metrics()

    # ------ Persistence --------------------------------------------------
    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        with open(dir_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        (dir_path / "elo_state.json").write_text(
            json.dumps(self.elo.to_dict(), indent=2)
        )
        (dir_path / "feature_state.json").write_text(json.dumps({
            "form_state": self.form_state or {},
            "last_game_date": self.last_game_date or {},
        }, indent=2))
        (dir_path / "qb_ratings.json").write_text(json.dumps({
            "team_qb_ratings_latest": self.team_qb_ratings_latest or {},
            "team_qb_starters_latest": self.team_qb_starters_latest or {},
            "latest_season": self.latest_season,
            "schema": "rolling_v1",
        }, indent=2))
        (dir_path / "team_metrics.json").write_text(json.dumps({
            "team_metrics_latest": self.team_metrics_latest or {},
            "team_metrics_prior": self.team_metrics_prior or {},
            "schema": "rolling_v1",
        }, indent=2))
        (dir_path / "metadata.json").write_text(json.dumps({
            "version": self.version,
            "feature_columns": self.feature_columns,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }, indent=2))
        log.info("Saved NFLModelV3 bundle -> %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "NFLModelV3":
        with open(dir_path / "model.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(dir_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        elo = NFLEloSystem.from_dict(
            json.loads((dir_path / "elo_state.json").read_text())
        )
        feat_state = json.loads((dir_path / "feature_state.json").read_text())
        qb_state = json.loads((dir_path / "qb_ratings.json").read_text())
        team_state = json.loads((dir_path / "team_metrics.json").read_text())
        meta = json.loads((dir_path / "metadata.json").read_text())

        if qb_state.get("schema") != "rolling_v1" or team_state.get("schema") != "rolling_v1":
            raise RuntimeError(
                "This v3 bundle was trained with the old season-aggregate "
                "pipeline (pre-rolling). Retrain with:\n"
                "    python -m ballknower_gridiron.scripts.train_football_model --version v3"
            )

        return cls(
            clf=clf, scaler=scaler, elo=elo,
            feature_columns=meta["feature_columns"],
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
            version=meta.get("version", "v3"),
            form_state=feat_state.get("form_state", {}),
            last_game_date=feat_state.get("last_game_date", {}),
            team_qb_ratings_latest=qb_state.get("team_qb_ratings_latest", {}),
            team_qb_starters_latest=qb_state.get("team_qb_starters_latest", {}),
            team_metrics_latest=team_state.get("team_metrics_latest", {}),
            team_metrics_prior=team_state.get("team_metrics_prior", {}),
            latest_season=qb_state.get("latest_season"),
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_nfl_model_v3(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
    force_refresh_team_metrics: bool = False,
    feature_columns: Optional[List[str]] = None,
    version_label: str = "v3",
) -> NFLModelV3:
    """Train v3 with leakage-free rolling QB ratings + rolling team metrics.

    Parameters
    ----------
    feature_columns
        If provided, train on only this subset of FEATURE_COLUMNS_V3. The
        feature builder still computes all features; we just select these
        when building X. Used by v3.1 (pruned version) to train with 16
        features instead of the full 21.
    version_label
        Stored in the saved bundle as `version`. Defaults to "v3"; v3.1
        passes "v3.1" so loaders can tell them apart.
    """
    if games.empty:
        raise ValueError("Cannot train on an empty games dataframe.")
    games = games.sort_values("game_date").reset_index(drop=True)

    feat_cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS_V3)

    fb = NFLFeatureBuilderV3()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games, force_refresh=force_refresh_qb_stats)
    fb.load_rolling_team_metrics(seasons, games, force_refresh=force_refresh_team_metrics)

    feat_df = fb.build_training_frame(games)
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    log.info(
        "[%s] Train rows: %d (%s -> %s) | Test rows: %d (%s -> %s) | "
        "Training on %d features.",
        version_label,
        len(train_df),
        pd.to_datetime(train_df["game_date"].min()).date(),
        pd.to_datetime(train_df["game_date"].max()).date(),
        len(test_df),
        pd.to_datetime(test_df["game_date"].min()).date(),
        pd.to_datetime(test_df["game_date"].max()).date(),
        len(feat_cols),
    )

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["home_won"].to_numpy(dtype=int)
    X_test = test_df[feat_cols].to_numpy(dtype=float)
    y_test = test_df["home_won"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = XGBClassifier(
        n_estimators=settings.xgb_n_estimators,
        max_depth=settings.xgb_max_depth,
        learning_rate=settings.xgb_learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=settings.random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    log.info("[%s] Fitting calibrated XGBoost classifier …", version_label)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train_s, y_train)

    proba_test = clf.predict_proba(X_test_s)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)
    metrics = {
        "accuracy":      float(accuracy_score(y_test, preds_test)),
        "roc_auc":       float(roc_auc_score(y_test, proba_test)),
        "log_loss":      float(log_loss(y_test, proba_test, labels=[0, 1])),
        "brier":         float(brier_score_loss(y_test, proba_test)),
        "n_train":       int(len(y_train)),
        "n_test":        int(len(y_test)),
        "home_win_rate": float(y_test.mean()),
    }
    log.info("[%s] Holdout metrics: %s", version_label, metrics)

    form_state = {team: list(d) for team, d in fb._form.items()}
    last_game_date = {team: dt.isoformat() for team, dt in fb._last_game_date.items()}

    # Save prior-season state for the latest training season's prior.
    team_metrics_prior = fb.end_of_season_by_team.get(
        (fb.latest_season - 1) if fb.latest_season else 0, {}
    )

    return NFLModelV3(
        clf=clf, scaler=scaler, elo=fb.elo,
        feature_columns=feat_cols,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        metrics=metrics,
        version=version_label,
        form_state=form_state,
        last_game_date=last_game_date,
        team_qb_ratings_latest=fb.team_qb_ratings_latest,
        team_qb_starters_latest=fb.team_qb_starters_latest,
        team_metrics_latest=fb.team_metrics_latest,
        team_metrics_prior=team_metrics_prior,
        latest_season=fb.latest_season,
    )
