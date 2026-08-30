"""
ballknower_gridiron.models.football_model
=========================================

NFL game-outcome predictor — v1 baseline.

Pipeline (mirrors the basketball v1 architecture):

  1. **Feature engineering** — replay every game chronologically. For
     each game, snapshot pre-game state (ELO diff, rest days,
     short-week / bye flags, recent form, playoff flag, international
     flag, divisional matchup flag). The ELO + state then update with
     the actual result. No leakage.

  2. **Features used** (home-team perspective; positive favors home):
       - elo_diff             : home_elo − away_elo (raw, no HCA)
       - elo_diff_with_hca    : (home_elo + HCA) − away_elo
                                  (HCA off for international games)
       - rest_diff            : home_rest − away_rest (in days)
       - short_week_diff      : away_short − home_short (positive → home rested)
       - bye_diff             : home_bye − away_bye   (positive → home off bye)
       - either_short_week    : 1 if either team on short week (fatigue ctx)
       - either_bye           : 1 if either team off a bye
       - form_diff_5          : home form (last-5 W%) − away form
       - is_playoff           : 1 if POST game
       - is_international     : 1 if neutral-site (e.g., London/Munich)
       - is_div_game          : 1 if divisional matchup (close games tend)
       - week_in_season       : 1..22 (regression target for "context")
       - mean_rest_days       : (home_rest + away_rest)/2 (abs fatigue ctx)

  3. **Time-series split** — last `test_fraction` of games chronological
     becomes the test set. No data from after the train cutoff touches
     the training fold.

  4. **XGBoost + isotonic calibration** — calibrated probabilities.
     Smaller max_depth than NBA because the NFL has way less data
     (only ~270 games/season × 12 seasons ≈ 3,200 games vs NBA's
     ~1,200 × 8 ≈ 10,000).

  5. **Persistence** — model.pkl + scaler.pkl + elo_state.json +
     feature_state.json + metadata.json saved under
     `models_artifacts/football/`.

NFL-specific notes vs NBA
-------------------------
  * No "back-to-back" feature — NFL doesn't have those. Replaced by
    short-week / bye flags.
  * `form_diff_5` instead of `form_diff_10` — only 17 regular-season
    games means a 10-game rolling window covers ~60% of the season,
    which is too sluggish to react to mid-season changes.
  * `is_international` is a new feature that flags neutral-site games
    (HCA is already neutralized in ELO for these; the flag captures
    additional travel/jet-lag effects the model can learn).

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.models.football_elo import NFLEloSystem
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


FEATURE_COLUMNS: List[str] = [
    "elo_diff",
    "elo_diff_with_hca",      # intentionally asymmetric (HCA → home)
    "rest_diff",
    "short_week_diff",
    "bye_diff",
    "either_short_week",
    "either_bye",
    "form_diff_5",
    "is_playoff",
    "is_international",
    "is_div_game",
    "week_in_season",
    "mean_rest_days",
]


# ---------------------------------------------------------------------------
# Feature builder — replays history producing pre-game features
# ---------------------------------------------------------------------------
class NFLFeatureBuilder:
    """Walks chronological games, emitting features then updating state."""

    def __init__(self) -> None:
        self.elo = NFLEloSystem()
        # Last-5 form: 1=W, 0=L. NFL has only 17 reg-season games so a
        # 10-game window (NBA default) would be too slow to update.
        self._form: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=5))
        # Last game date per team (for sanity checks / rest fallbacks).
        self._last_game_date: Dict[str, date_cls] = {}

    # -- helpers ----------------------------------------------------------
    def _form_pct(self, team: str) -> float:
        d = self._form[team]
        return sum(d) / len(d) if d else 0.5

    # -- feature emission -------------------------------------------------
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
    ) -> Dict[str, float]:
        """
        Emit features for a single matchup. The caller is responsible for
        passing pre-game `home_rest` and `away_rest` (the schedule data
        already gives us these directly — no need to compute from
        last-game-date math like the NBA pipeline does).
        """
        if hasattr(game_date, "date"):
            game_date = game_date.date()

        # Read pre-game ratings; HCA is neutralized at the ELO level for
        # international games, so we re-apply the convention here for
        # the feature value.
        elo_home = self.elo.get_rating(home_team)
        elo_away = self.elo.get_rating(away_team)
        hca = 0.0 if is_international else settings.elo_hca

        home_rest_f = float(min(home_rest, settings.rest_cap_days))
        away_rest_f = float(min(away_rest, settings.rest_cap_days))

        home_short = 1.0 if home_rest_f <= settings.short_week_threshold else 0.0
        away_short = 1.0 if away_rest_f <= settings.short_week_threshold else 0.0
        home_bye   = 1.0 if home_rest_f >= settings.bye_week_threshold   else 0.0
        away_bye   = 1.0 if away_rest_f >= settings.bye_week_threshold   else 0.0

        return {
            # Asymmetric (intentional — HCA goes to whoever is home)
            "elo_diff": elo_home - elo_away,
            "elo_diff_with_hca": (elo_home + hca) - elo_away,

            # Symmetric diffs (sign-flip cleanly under home/away swap;
            # convention: positive = favors home)
            "rest_diff": home_rest_f - away_rest_f,
            "short_week_diff": away_short - home_short,  # +ve when home rested
            "bye_diff": home_bye - away_bye,
            "form_diff_5": self._form_pct(home_team) - self._form_pct(away_team),

            # Symmetric context (unchanged under swap)
            "either_short_week": max(home_short, away_short),
            "either_bye": max(home_bye, away_bye),
            "mean_rest_days": (home_rest_f + away_rest_f) / 2.0,

            # Game-level (don't move under swap)
            "is_playoff": 1.0 if is_playoff else 0.0,
            "is_international": 1.0 if is_international else 0.0,
            "is_div_game": 1.0 if is_div_game else 0.0,
            "week_in_season": float(week),
        }

    # -- state update after a game ----------------------------------------
    def record_game(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_date,
        season: int,
        is_playoff: bool = False,
        is_international: bool = False,
    ) -> None:
        if hasattr(game_date, "date"):
            game_date = game_date.date()

        # ELO update (handles season regression internally)
        self.elo.update_game(
            home_team, away_team, home_score, away_score,
            game_date, season, is_playoff, is_international,
        )

        # Form deques
        if home_score > away_score:
            self._form[home_team].append(1)
            self._form[away_team].append(0)
        else:
            self._form[home_team].append(0)
            self._form[away_team].append(1)

        self._last_game_date[home_team] = game_date
        self._last_game_date[away_team] = game_date

    # -- training-frame builder -------------------------------------------
    def build_training_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Walk a chronologically-sorted games DataFrame, emitting one row
        per game with FEATURE_COLUMNS + label `home_won` + `game_date`.

        Features are snapshotted BEFORE the game's result updates ELO /
        form state.
        """
        rows: List[Dict] = []
        log.info("Replaying NFL history to build features …")
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
        log.info("Built NFL feature frame with shape %s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Trained-model container
# ---------------------------------------------------------------------------
@dataclass
class NFLModel:
    """Trained NFL model bundle — predict + persist."""

    clf: CalibratedClassifierCV
    scaler: StandardScaler
    elo: NFLEloSystem
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float]
    version: str = "v1"

    # Embedded feature-builder state for inference. Stored separately so
    # we can replay-with-current-state when predicting upcoming games.
    form_state: Dict[str, List[int]] = None  # type: ignore
    last_game_date: Dict[str, str] = None  # type: ignore

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
    ) -> Tuple[float, float]:
        """
        Return (P(home wins), confidence). Confidence = 2 × |p − 0.5|.

        Uses the trained ELO + saved form state. If a team hasn't been
        seen, falls back to neutral defaults (rating=1500, form=0.5).
        """
        fb = NFLFeatureBuilder()
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

        game_date = game_date or date_cls.today()
        season = season if season is not None else (self.elo.last_season or game_date.year)

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
        )
        X = np.array([[feats[c] for c in self.feature_columns]], dtype=float)
        Xs = self.scaler.transform(X)
        p_home = float(self.clf.predict_proba(Xs)[0, 1])
        return p_home, abs(p_home - 0.5) * 2.0

    # -- persistence ------------------------------------------------------
    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        with open(dir_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        (dir_path / "elo_state.json").write_text(
            json.dumps(self.elo.to_dict(), indent=2)
        )
        (dir_path / "feature_state.json").write_text(
            json.dumps({
                "form_state": self.form_state or {},
                "last_game_date": self.last_game_date or {},
            }, indent=2)
        )
        (dir_path / "metadata.json").write_text(json.dumps({
            "version": self.version,
            "feature_columns": self.feature_columns,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }, indent=2))
        log.info("Saved NFLModel bundle -> %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "NFLModel":
        with open(dir_path / "model.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(dir_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        elo = NFLEloSystem.from_dict(
            json.loads((dir_path / "elo_state.json").read_text())
        )
        feat_state = json.loads((dir_path / "feature_state.json").read_text())
        meta = json.loads((dir_path / "metadata.json").read_text())
        return cls(
            clf=clf,
            scaler=scaler,
            elo=elo,
            feature_columns=meta["feature_columns"],
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
            version=meta.get("version", "v1"),
            form_state=feat_state.get("form_state", {}),
            last_game_date=feat_state.get("last_game_date", {}),
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_nfl_model(games: pd.DataFrame) -> NFLModel:
    """
    Build features, do a chronological train/test split, fit a calibrated
    XGBoost classifier, log metrics, and return an `NFLModel` bundle.

    No future leakage:
      * Features are emitted BEFORE each game's result updates ELO/form.
      * Test set is the chronological tail of the data.
    """
    if games.empty:
        raise ValueError("Cannot train on an empty games dataframe.")
    games = games.sort_values("game_date").reset_index(drop=True)

    fb = NFLFeatureBuilder()
    feat_df = fb.build_training_frame(games)
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    log.info(
        "Train rows: %d (%s -> %s) | Test rows: %d (%s -> %s)",
        len(train_df),
        pd.to_datetime(train_df["game_date"].min()).date(),
        pd.to_datetime(train_df["game_date"].max()).date(),
        len(test_df),
        pd.to_datetime(test_df["game_date"].min()).date(),
        pd.to_datetime(test_df["game_date"].max()).date(),
    )

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["home_won"].to_numpy(dtype=int)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
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
    log.info("Fitting calibrated XGBoost classifier for NFL v1 …")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train_s, y_train)

    proba_test = clf.predict_proba(X_test_s)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_test)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "log_loss": float(log_loss(y_test, proba_test, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, proba_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        # NFL home win rate baseline: historically ~57%, declining
        # post-COVID toward ~55%. Anything above that = signal beyond
        # home advantage.
        "home_win_rate": float(y_test.mean()),
    }
    log.info("Holdout metrics: %s", metrics)

    form_state = {team: list(d) for team, d in fb._form.items()}
    last_game_date = {team: dt.isoformat() for team, dt in fb._last_game_date.items()}

    return NFLModel(
        clf=clf,
        scaler=scaler,
        elo=fb.elo,
        feature_columns=FEATURE_COLUMNS,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        metrics=metrics,
        form_state=form_state,
        last_game_date=last_game_date,
    )
