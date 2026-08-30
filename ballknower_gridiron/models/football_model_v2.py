"""
ballknower_gridiron.models.football_model_v2
============================================

v2 NFL model. Inherits v1's features and adds **QB rating + QB-out toggle**
— now computed with a **leakage-free rolling pipeline**.

What's new vs v1
----------------
Three new features in the feature vector (qb_rating_diff,
mean_qb_rating, qb_healthy_diff) computed from each team's STARTING QB
rating, identified as the QB with the most cumulative passing attempts
on the team UP TO the game date. The rating itself is a z-scored
composite of 9 stats — see `models.qb_rating` for the formula. Crucially,
those stats are computed from games STRICTLY BEFORE each prediction's
date, eliminating the feature-leakage that inflated v2's pre-rolling
numbers.

The QB rating itself is a z-scored composite of 9 stats (PPG, TDs/G,
rushing, INTs, fumbles, completion %, **traditional passer rating**,
and **ESPN QBR**). See `ballknower_gridiron.models.qb_rating` for the
formula and weights — both passer rating and ESPN QBR are included
because they measure related but distinct things (passer rating is
unadjusted; QBR is opponent-adjusted, garbage-time-discounted, and
includes rushing/sacks).

Starter-out toggle behavior is unchanged:
  * Known backup rating → backup_rating − backup_qb_penalty
  * Unknown backup     → starter_rating − (2.5 + backup_qb_penalty)
  * Both starters out  → both diffs zero out

Persistence
-----------
The bundle saves the latest rolling snapshot per team (for inference)
+ the prior-season end-of-season snapshot (for early-season fallback).
The full rolling table is NOT saved — it's rebuilt from PBP/weekly
stats during backtesting via the same pipeline as training.

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
from ballknower_gridiron.data.qb_stats_loader import load_qb_weekly_stats
from ballknower_gridiron.models.football_elo import NFLEloSystem
from ballknower_gridiron.models.football_model import (
    FEATURE_COLUMNS,
    NFLFeatureBuilder,
    NFLModel,
)
from ballknower_gridiron.models.qb_rating import (
    compute_rolling_qb_ratings,
    latest_qb_ratings_snapshot,
    team_qb_rating_with_override,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


FEATURE_COLUMNS_V2: List[str] = FEATURE_COLUMNS + [
    "qb_rating_diff",   # home_qb - away_qb (signed; favors home if positive)
    "mean_qb_rating",   # symmetric: indicates overall QB quality in the game
    "qb_healthy_diff",  # float(away_qb_out) - float(home_qb_out)
]


# ---------------------------------------------------------------------------
# Feature builder v2 — ROLLING
# ---------------------------------------------------------------------------
class NFLFeatureBuilderV2(NFLFeatureBuilder):
    """
    v1 builder + ROLLING per-(team, season, game_date) QB ratings.

    The class holds two parallel data structures:
      * `team_qb_ratings_by_date` — full chronological lookup, populated
        at training time and used during chronological feature emission.
      * `team_qb_ratings_latest` — per-team latest snapshot, used at
        inference time when we don't have rolling data for a future date.

    Features are emitted by looking up (team, season, game_date) in the
    rolling table first; if missing (or zero), fall back to the latest
    snapshot.
    """

    def __init__(self) -> None:
        super().__init__()
        # Rolling lookup populated from compute_rolling_qb_ratings()
        self.team_qb_ratings_by_date: Dict[Tuple[str, int, date_cls], float] = {}
        self.team_qb_starters_by_date: Dict[Tuple[str, int, date_cls], str] = {}
        # Latest snapshot per team (for inference)
        self.team_qb_ratings_latest: Dict[str, float] = {}
        self.team_qb_starters_latest: Dict[str, str] = {}
        # Latest training season — used by inference fallback
        self.latest_season: Optional[int] = None

    def load_rolling_qb_ratings(
        self,
        seasons: List[int],
        games: pd.DataFrame,
        force_refresh: bool = False,
    ) -> None:
        """
        Pull weekly QB stats, compute rolling per-game ratings, and
        cache them on the builder. Should be called BEFORE building
        the training frame.
        """
        log.info("[v2] Loading weekly QB stats for %d seasons …", len(seasons))
        weekly = load_qb_weekly_stats(seasons, force_refresh=force_refresh)
        if weekly.empty:
            log.warning("[v2] No weekly QB stats — QB features will be zero.")
            return
        log.info(
            "[v2] Computing rolling QB ratings (replaying %d weekly rows) …",
            len(weekly),
        )
        ratings, starters = compute_rolling_qb_ratings(weekly, games)
        self.team_qb_ratings_by_date = ratings
        self.team_qb_starters_by_date = starters

        # Build the latest snapshot for the most recent season we trained on.
        self.latest_season = max(seasons) if seasons else None
        if self.latest_season is not None:
            snap_r, snap_s = latest_qb_ratings_snapshot(
                ratings, starters, self.latest_season,
            )
            self.team_qb_ratings_latest = snap_r
            self.team_qb_starters_latest = snap_s
            log.info("[v2] Latest QB snapshot — %d teams in season %s.",
                     len(snap_r), self.latest_season)

    def _team_rating_at(
        self, team: str, season: int, game_date: date_cls,
    ) -> float:
        """
        Look up (team, season, game_date) in the rolling table. Returns
        0.0 if no entry exists (which is what the feature builder
        substitutes for "no signal yet").
        """
        key = (str(team), int(season), game_date)
        if key in self.team_qb_ratings_by_date:
            return float(self.team_qb_ratings_by_date[key])
        # Inference-time fallback: use the latest snapshot.
        if team in self.team_qb_ratings_latest:
            return float(self.team_qb_ratings_latest[team])
        return 0.0

    def _team_starter_at(
        self, team: str, season: int, game_date: date_cls,
    ) -> str:
        key = (str(team), int(season), game_date)
        if key in self.team_qb_starters_by_date:
            return str(self.team_qb_starters_by_date[key])
        return self.team_qb_starters_latest.get(team, "")

    def get_team_qb_rating(self, team: str, season: int = 0) -> float:
        """Inference-time accessor: latest rating snapshot per team."""
        return float(self.team_qb_ratings_latest.get(team, 0.0))

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
        )

        if hasattr(game_date, "date"):
            gd = game_date.date()
        else:
            gd = game_date

        home_starter_r = self._team_rating_at(home_team, season, gd)
        away_starter_r = self._team_rating_at(away_team, season, gd)

        # Apply QB-out override (uses the helper which is unchanged).
        team_ratings_pair = {home_team: home_starter_r, away_team: away_starter_r}
        home_qb = team_qb_rating_with_override(
            home_team, team_ratings_pair,
            starter_out=home_qb_out, backup_rating=home_backup_qb_rating,
        )
        away_qb = team_qb_rating_with_override(
            away_team, team_ratings_pair,
            starter_out=away_qb_out, backup_rating=away_backup_qb_rating,
        )

        base["qb_rating_diff"] = home_qb - away_qb
        base["mean_qb_rating"] = (home_qb + away_qb) / 2.0
        base["qb_healthy_diff"] = float(away_qb_out) - float(home_qb_out)
        return base

    def build_training_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Same shape as v1 but with QB rating features attached. The
        rolling QB ratings table must already be loaded.
        """
        rows: List[Dict] = []
        log.info("Replaying NFL history to build v2 features (rolling QB) …")
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
        log.info("Built NFL v2 feature frame with shape %s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Trained v2 bundle
# ---------------------------------------------------------------------------
@dataclass
class NFLModelV2:
    """
    Trained v2 NFL model bundle. Inference-time uses the saved latest
    snapshot per team — no rolling table needed at predict time.
    """

    clf: CalibratedClassifierCV
    scaler: StandardScaler
    elo: NFLEloSystem
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float]
    version: str = "v2"

    form_state: Dict[str, List[int]] = field(default_factory=dict)
    last_game_date: Dict[str, str] = field(default_factory=dict)

    # Latest rolling snapshot per team (most recent training game's pre-state)
    team_qb_ratings_latest: Dict[str, float] = field(default_factory=dict)
    team_qb_starters_latest: Dict[str, str] = field(default_factory=dict)
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
    ) -> Tuple[float, float]:
        """
        Return (P(home wins), confidence).

        `override_qb_ratings` lets you pass fresh team-level QB ratings
        (e.g., after a starter QB has been changed mid-season).
        """
        fb = NFLFeatureBuilderV2()
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

        # Wire up the saved latest snapshot for inference.
        fb.team_qb_ratings_latest = dict(self.team_qb_ratings_latest)
        fb.team_qb_starters_latest = dict(self.team_qb_starters_latest)
        fb.latest_season = self.latest_season

        if override_qb_ratings is not None:
            fb.team_qb_ratings_latest.update(override_qb_ratings)

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
            "schema": "rolling_v1",  # so loaders can detect format
        }, indent=2))
        (dir_path / "metadata.json").write_text(json.dumps({
            "version": self.version,
            "feature_columns": self.feature_columns,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }, indent=2))
        log.info("Saved NFLModelV2 bundle -> %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "NFLModelV2":
        with open(dir_path / "model.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(dir_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        elo = NFLEloSystem.from_dict(
            json.loads((dir_path / "elo_state.json").read_text())
        )
        feat_state = json.loads((dir_path / "feature_state.json").read_text())
        qb_state = json.loads((dir_path / "qb_ratings.json").read_text())
        meta = json.loads((dir_path / "metadata.json").read_text())

        # Schema detection — newer bundles include `schema: rolling_v1`.
        if qb_state.get("schema") != "rolling_v1":
            raise RuntimeError(
                "This v2 bundle was trained with the old season-aggregate "
                "QB pipeline (pre-rolling). Retrain with:\n"
                "    python -m ballknower_gridiron.scripts.train_football_model --version v2"
            )

        return cls(
            clf=clf, scaler=scaler, elo=elo,
            feature_columns=meta["feature_columns"],
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
            version=meta.get("version", "v2"),
            form_state=feat_state.get("form_state", {}),
            last_game_date=feat_state.get("last_game_date", {}),
            team_qb_ratings_latest=qb_state.get("team_qb_ratings_latest", {}),
            team_qb_starters_latest=qb_state.get("team_qb_starters_latest", {}),
            latest_season=qb_state.get("latest_season"),
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_nfl_model_v2(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
) -> NFLModelV2:
    """Train v2 with leakage-free rolling QB ratings."""
    if games.empty:
        raise ValueError("Cannot train on an empty games dataframe.")
    games = games.sort_values("game_date").reset_index(drop=True)

    fb = NFLFeatureBuilderV2()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games, force_refresh=force_refresh_qb_stats)

    feat_df = fb.build_training_frame(games)
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    log.info(
        "[v2] Train rows: %d (%s -> %s) | Test rows: %d (%s -> %s)",
        len(train_df),
        pd.to_datetime(train_df["game_date"].min()).date(),
        pd.to_datetime(train_df["game_date"].max()).date(),
        len(test_df),
        pd.to_datetime(test_df["game_date"].min()).date(),
        pd.to_datetime(test_df["game_date"].max()).date(),
    )

    X_train = train_df[FEATURE_COLUMNS_V2].to_numpy(dtype=float)
    y_train = train_df["home_won"].to_numpy(dtype=int)
    X_test = test_df[FEATURE_COLUMNS_V2].to_numpy(dtype=float)
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
    log.info("[v2] Fitting calibrated XGBoost classifier …")
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
    log.info("[v2] Holdout metrics: %s", metrics)

    form_state = {team: list(d) for team, d in fb._form.items()}
    last_game_date = {team: dt.isoformat() for team, dt in fb._last_game_date.items()}

    return NFLModelV2(
        clf=clf, scaler=scaler, elo=fb.elo,
        feature_columns=FEATURE_COLUMNS_V2,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        metrics=metrics,
        form_state=form_state,
        last_game_date=last_game_date,
        team_qb_ratings_latest=fb.team_qb_ratings_latest,
        team_qb_starters_latest=fb.team_qb_starters_latest,
        latest_season=fb.latest_season,
    )


# ---------------------------------------------------------------------------
# Polymorphic loader (v1 / v2 / v3 / v3.1 / v4 / v4.1)
# ---------------------------------------------------------------------------
def load_active_nfl_model(version: Optional[str] = None):
    """
    Polymorphic loader for any trained NFL model version. Pass one of
    "v1", "v2", "v3", "v3.1", "v4", "v4.1" (or omit to use
    settings.active_model_version). Both "v3.1" and "v3_1" are accepted
    (we normalize internally).
    """
    raw = (version or settings.active_model_version).lower()
    normalized = raw.replace(".", "_")
    if normalized == "v1":
        return NFLModel.load(settings.models_dir)
    if normalized == "v2":
        return NFLModelV2.load(settings.models_dir_v2)
    if normalized == "v3":
        # Forward imports to avoid circular references at module load time.
        from ballknower_gridiron.models.football_model_v3 import NFLModelV3
        return NFLModelV3.load(settings.models_dir_v3)
    if normalized == "v3_1":
        # v3.1 saves an NFLModelV3 bundle to the v3_1 directory.
        from ballknower_gridiron.models.football_model_v3 import NFLModelV3
        return NFLModelV3.load(settings.models_dir_v3_1)
    if normalized == "v4":
        from ballknower_gridiron.models.football_model_v4 import NFLModelV4
        return NFLModelV4.load(settings.models_dir_v4)
    if normalized == "v4_1":
        from ballknower_gridiron.models.football_model_v4 import NFLModelV4
        return NFLModelV4.load(settings.models_dir_v4_1)
    if normalized == "v5":
        # v5 is a SPREAD regression model — different class/interface.
        from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
        return NFLSpreadModelV5.load(settings.models_dir_v5)
    raise ValueError(f"Unknown NFL model version: {version!r}")
