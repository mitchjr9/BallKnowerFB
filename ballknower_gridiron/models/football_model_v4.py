"""
ballknower_gridiron.models.football_model_v4
============================================

v4 NFL model. Inherits v3's rolling pipeline (QB ratings, team EPA) and
adds features that are GENUINELY ORTHOGONAL to ELO:

New v4 features (4)
-------------------
  * is_cold        — outdoor game with temp < 32°F. Cold weather suppresses
                      passing accuracy, slows down pass-protection, and
                      historically favors run-heavy teams. ELO is
                      season-average; it cannot anticipate "Week 13 in Buffalo
                      in 18°F." This is per-game information ELO can't have.
  * is_windy       — outdoor game with wind > 15 mph. Wind devastates passing
                      games and field goals. Same orthogonality argument.
  * is_dome        — game played indoors (closed/dome roof). Some teams play
                      drastically better indoors; the binary flag lets the
                      model condition on the broad environment shift.
  * qb_change_diff — `(home_qb_changed) - (away_qb_changed)`, where each side
                      is 1 if THIS WEEK's starter differs from THAT TEAM's
                      most recent prior game's starter in the same season.
                      Captures mid-season QB changes BEFORE ELO has adjusted
                      to the new starter's actual performance. Designed
                      specifically to dodge the "QB rating is redundant with
                      ELO" trap that sank v2's QB features.

These are added on top of the 21 features inherited from v3, so v4 has
**25 features total**.

Why this design beats v2's QB rating features
---------------------------------------------
The lesson from v2 was: any feature that just measures team-or-QB quality
through a different lens will be redundant with ELO (which measures
exactly that, via wins/losses). To beat the ELO ceiling, new features
need to capture **game-specific information that ELO cannot know in
advance.** Weather is game-specific. A Week-N QB change happens AFTER
ELO has already credited the prior QB's outcomes — so it's a forward-
looking disruption signal, not a backward-looking quality signal.

Persistence
-----------
Same bundle format as v3 (model.pkl, scaler.pkl, elo_state.json,
feature_state.json, qb_ratings.json, team_metrics.json, metadata.json),
saved into `settings.models_dir_v4`. The metadata records version="v4"
and feature_columns=FEATURE_COLUMNS_V4.

DISCLAIMER: For entertainment and educational purposes only.
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
    get_default_team_metrics,
)
from ballknower_gridiron.models.football_elo import NFLEloSystem
from ballknower_gridiron.models.football_model_v3 import (
    FEATURE_COLUMNS_V3,
    NFLFeatureBuilderV3,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# v4 adds 4 features on top of v3's 21 → 25 total.
V4_NEW_FEATURES: List[str] = [
    "is_cold",
    "is_windy",
    "is_dome",
    "qb_change_diff",
]
FEATURE_COLUMNS_V4: List[str] = FEATURE_COLUMNS_V3 + V4_NEW_FEATURES


# Thresholds for weather binarization. Chosen for "noticeably affects game"
# rather than "merely uncomfortable":
#   - 32°F: water freezes; passing accuracy drops measurably below this
#   - 15 mph: sustained winds at which kickers and deep balls suffer
_COLD_THRESHOLD_F = 32.0
_WIND_THRESHOLD_MPH = 15.0
_DOME_ROOF_VALUES = {"closed", "dome", "indoors"}


# ---------------------------------------------------------------------------
# Feature builder v4
# ---------------------------------------------------------------------------
class NFLFeatureBuilderV4(NFLFeatureBuilderV3):
    """
    v3 builder + weather features + QB change indicator.

    The starter-history cache (used to detect "this week's starter !=
    last game's starter for this team") is populated automatically when
    `load_rolling_qb_ratings()` runs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-(team, season), a date-sorted list of (game_date, starter_name)
        # tuples. Built once after rolling QB ratings load.
        self._team_starter_history: Dict[
            Tuple[str, int], List[Tuple[date_cls, str]]
        ] = {}

    def load_rolling_qb_ratings(
        self,
        seasons: List[int],
        games: pd.DataFrame,
        force_refresh: bool = False,
    ) -> None:
        super().load_rolling_qb_ratings(seasons, games, force_refresh=force_refresh)
        self._build_starter_history()

    def _build_starter_history(self) -> None:
        self._team_starter_history.clear()
        for (team, season, gd), name in self.team_qb_starters_by_date.items():
            self._team_starter_history.setdefault((team, season), []).append(
                (gd, name)
            )
        for k in self._team_starter_history:
            self._team_starter_history[k].sort(key=lambda x: x[0])

    def _did_qb_change(
        self, team: str, season: int, game_date: date_cls,
    ) -> int:
        """
        Return 1 iff this team's starter at `game_date` differs from its
        starter at the most recent prior game IN THE SAME SEASON. Returns
        0 if there's no prior game in-season (e.g., Week 1) or if no
        rolling history is available (rare — only at inference for future
        games beyond the training horizon).
        """
        hist = self._team_starter_history.get((str(team), int(season)))
        if not hist:
            return 0
        prev_starter = None
        current_starter = None
        for gd, name in hist:
            if gd < game_date:
                prev_starter = name
            elif gd == game_date:
                current_starter = name
                break
            else:
                break
        if not prev_starter or not current_starter:
            return 0
        return 1 if current_starter != prev_starter else 0

    @staticmethod
    def _weather_flags(
        temp: Optional[float],
        wind: Optional[float],
        roof: Optional[str],
    ) -> Tuple[float, float, float]:
        """Return (is_cold, is_windy, is_dome) as floats in {0.0, 1.0}.

        Domes/closed roofs override outdoor weather: a closed-roof game in
        a cold market still reads as is_cold=0, is_windy=0.
        """
        roof_norm = (roof or "").strip().lower()
        is_dome = 1.0 if roof_norm in _DOME_ROOF_VALUES else 0.0
        if is_dome:
            return 0.0, 0.0, is_dome
        is_cold = 0.0
        if temp is not None and not pd.isna(temp):
            try:
                is_cold = 1.0 if float(temp) < _COLD_THRESHOLD_F else 0.0
            except (TypeError, ValueError):
                pass
        is_windy = 0.0
        if wind is not None and not pd.isna(wind):
            try:
                is_windy = 1.0 if float(wind) > _WIND_THRESHOLD_MPH else 0.0
            except (TypeError, ValueError):
                pass
        return is_cold, is_windy, is_dome

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
        # New v4 kwargs --------------------------------------------------
        temp: Optional[float] = None,
        wind: Optional[float] = None,
        roof: Optional[str] = None,
        home_qb_changed: Optional[int] = None,
        away_qb_changed: Optional[int] = None,
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

        # Weather flags
        is_cold, is_windy, is_dome = self._weather_flags(temp, wind, roof)
        base["is_cold"] = is_cold
        base["is_windy"] = is_windy
        base["is_dome"] = is_dome

        # QB change indicator. If the caller didn't override, derive from
        # the starter history we built during training.
        h_changed = (
            int(home_qb_changed)
            if home_qb_changed is not None
            else self._did_qb_change(home_team, season, gd)
        )
        a_changed = (
            int(away_qb_changed)
            if away_qb_changed is not None
            else self._did_qb_change(away_team, season, gd)
        )
        # Sign convention matches other diff features: positive means the
        # home team is the LESS disrupted of the two (so it should help
        # home's win probability). away_changed - home_changed:
        #   home stable, away changed  ->  +1  (good for home)
        #   home changed, away stable  ->  -1  (bad for home)
        #   neither / both             ->   0
        base["qb_change_diff"] = float(a_changed) - float(h_changed)
        return base

    def build_training_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        """Same shape as v3 but pulls temp/wind/roof from each game row."""
        rows: List[Dict] = []
        log.info("Replaying NFL history to build v4 features (v3 + weather + QB change) …")
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
                home_qb_out=False,
                away_qb_out=False,
                temp=getattr(row, "temp", None),
                wind=getattr(row, "wind", None),
                roof=getattr(row, "roof", None),
                # qb_changed left None -> derived from starter history
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
        log.info("Built NFL v4 feature frame with shape %s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Trained v4 bundle
# ---------------------------------------------------------------------------
@dataclass
class NFLModelV4:
    """Trained v4 NFL model bundle. Same shape as v3 + weather/QB-change."""

    clf: CalibratedClassifierCV
    scaler: StandardScaler
    elo: NFLEloSystem
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float]
    version: str = "v4"

    form_state: Dict[str, List[int]] = field(default_factory=dict)
    last_game_date: Dict[str, str] = field(default_factory=dict)

    # Same QB + team metrics state as v3
    team_qb_ratings_latest: Dict[str, float] = field(default_factory=dict)
    team_qb_starters_latest: Dict[str, str] = field(default_factory=dict)
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
        # v4 prediction kwargs
        temp: Optional[float] = None,
        wind: Optional[float] = None,
        roof: Optional[str] = None,
        home_qb_changed: Optional[int] = None,
        away_qb_changed: Optional[int] = None,
        # Allow user to override learned snapshots
        override_qb_ratings: Optional[Dict[str, float]] = None,
        override_team_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[float, float]:
        """Return (P(home wins), confidence)."""
        fb = NFLFeatureBuilderV4()
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

        fb.team_qb_ratings_latest = dict(self.team_qb_ratings_latest)
        fb.team_qb_starters_latest = dict(self.team_qb_starters_latest)
        fb.latest_season = self.latest_season
        if override_qb_ratings is not None:
            fb.team_qb_ratings_latest.update(override_qb_ratings)

        fb.team_metrics_latest = dict(self.team_metrics_latest)
        if override_team_metrics is not None:
            for t, m in override_team_metrics.items():
                fb.team_metrics_latest[t] = {**get_default_team_metrics(), **m}
        if self.latest_season is not None and self.team_metrics_prior:
            fb.end_of_season_by_team[self.latest_season - 1] = dict(
                self.team_metrics_prior
            )

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
            temp=temp, wind=wind, roof=roof,
            home_qb_changed=home_qb_changed,
            away_qb_changed=away_qb_changed,
        )
        X = np.array([[feats[c] for c in self.feature_columns]], dtype=float)
        Xs = self.scaler.transform(X)
        p_home = float(self.clf.predict_proba(Xs)[0, 1])
        return p_home, abs(p_home - 0.5) * 2.0

    # ------ Accessors (same as v3 for cross-version compatibility) -----
    def get_qb_rating(self, team: str, season: Optional[int] = None) -> float:
        return float(self.team_qb_ratings_latest.get(team, 0.0))

    def get_starter_name(self, team: str, season: Optional[int] = None) -> str:
        return self.team_qb_starters_latest.get(team, "")

    def get_team_metrics(
        self, team: str, season: Optional[int] = None,
    ) -> Dict[str, float]:
        if team in self.team_metrics_latest:
            return dict(self.team_metrics_latest[team])
        return get_default_team_metrics()

    # ------ Persistence ------------------------------------------------
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
        log.info("Saved NFLModelV4 bundle -> %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "NFLModelV4":
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
                "This v4 bundle was trained with an older pipeline. Retrain with:\n"
                "    python -m ballknower_gridiron.scripts.train_football_model --version v4"
            )

        return cls(
            clf=clf, scaler=scaler, elo=elo,
            feature_columns=meta["feature_columns"],
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
            version=meta.get("version", "v4"),
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
def train_nfl_model_v4(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
    force_refresh_team_metrics: bool = False,
    feature_columns: Optional[List[str]] = None,
    version_label: str = "v4",
) -> NFLModelV4:
    """Train v4 with leakage-free rolling features + weather + QB change.

    Parameters
    ----------
    feature_columns
        If provided, train on only this subset. Used by v4.1 to train
        with FEATURE_COLUMNS_V4_1 (pruned).
    version_label
        Stored in the saved bundle as `version`. Defaults to "v4"; v4.1
        passes "v4.1".
    """
    if games.empty:
        raise ValueError("Cannot train on an empty games dataframe.")
    games = games.sort_values("game_date").reset_index(drop=True)

    feat_cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS_V4)

    fb = NFLFeatureBuilderV4()
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

    # Quick weather-coverage diagnostic so the user can see how many
    # historical games actually have temp/wind data attached.
    if "is_cold" in feat_cols or "is_windy" in feat_cols:
        n_cold = int(feat_df["is_cold"].sum())
        n_windy = int(feat_df["is_windy"].sum())
        n_dome = int(feat_df["is_dome"].sum())
        log.info(
            "[%s] Weather coverage: %d cold games, %d windy, %d dome (of %d total).",
            version_label, n_cold, n_windy, n_dome, len(feat_df),
        )
    if "qb_change_diff" in feat_cols:
        n_changes = int((feat_df["qb_change_diff"] != 0).sum())
        log.info("[%s] QB-change events flagged in %d games.", version_label, n_changes)

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

    team_metrics_prior = fb.end_of_season_by_team.get(
        (fb.latest_season - 1) if fb.latest_season else 0, {}
    )

    return NFLModelV4(
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
