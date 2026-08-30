"""
ballknower_gridiron.models.football_model_v5
============================================

v5 NFL model — SPREAD prediction.

What changed vs. v1-v4
----------------------
v1-v4 predicted **P(home wins)**, a binary classification target. v5
predicts the **point margin** (home_score - away_score) as a regression.
Why this matters:

  * A single regression model can answer all three relevant questions:
        - Who wins?            sign of predicted_margin
        - By how much?         |predicted_margin|
        - Against the spread?  predicted_margin vs Vegas spread_line
  * NFL spreads have meaningful pre-game inefficiencies the closing line
    eventually captures, but at the open or mid-week you can find spots
    where the model meaningfully disagrees with Vegas. THAT is
    publishable newsletter content.
  * Margin prediction surfaces different feature value than win
    probability did. Weather (a smaller-margin compresser) and QB
    rating (a points-per-game driver) may matter MORE here than they
    did for binary outcomes. Worth checking with permutation importance
    after first training run.

Architecture
------------
  * Same feature builder as v3 (NFLFeatureBuilderV3 with rolling QB
    ratings + rolling team EPA). Same 21 features. v5.1 can experiment
    with v4's weather features later — establish the clean baseline first.
  * **XGBRegressor** instead of XGBClassifier. Loss = squared error.
    No calibration step (we're not producing probabilities).
  * Output is a continuous margin in points. Sign indicates winner,
    magnitude indicates margin, comparison to Vegas spread gives the
    ATS pick.

Metrics
-------
  * **MAE**  — mean absolute error in points (Vegas ≈ 10.5, top public
    models ≈ 10.5-11.0).
  * **RMSE** — penalizes blowout misses more than MAE.
  * **R²**   — variance explained. NFL is noisy; R² around 0.10-0.20 is
    realistic, not a sign of failure.
  * **ATS hit rate** — pct of games where the side our predicted margin
    favors actually covered the Vegas spread. Above 52.4% beats the
    standard vig; 54-55% is "real edge"; > 55% sustained is suspicious.
    Computed in v5 training output as a quick sanity check; the
    `ats_diagnostics.py` script does the proper tier-by-tier analysis.

What v5 does NOT include
------------------------
  * Injury data (V5.1 candidate via Pro Football Reference scrape)
  * v4's weather + QB-change features (V5.2 candidate, contingent on
    importance findings)
  * Total prediction (separate model — would be v5_total or v6)
  * Uncertainty quantification (would require quantile regression or
    bootstrap; future enhancement)

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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


# v5 uses v3's feature set unchanged. Re-exported here for convenience and
# so v5.x variants can mutate it without touching v3's namespace.
FEATURE_COLUMNS_V5: List[str] = list(FEATURE_COLUMNS_V3)


# ---------------------------------------------------------------------------
# ATS evaluation helper (used both in training and by ats_diagnostics)
# ---------------------------------------------------------------------------
def ats_hit_rate(
    predicted_margins: np.ndarray,
    spread_lines: np.ndarray,
    actual_margins: np.ndarray,
    min_gap: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute ATS hit rate given predicted margins, Vegas spreads, and
    actual point margins.

    **nflverse `spread_line` convention** (verified against actual data
    in scripts/leakage_audit.py): `spread_line` is the HOME team's
    EXPECTED MARGIN — positive when home is favored. So
    `spread_line = 7` means home is favored by 7 (not that home is
    giving 7 to the bookmaker in the conventional Vegas/oddsmaker sense).
    Mean spread_line in the training data is ~+1.9 because NFL home
    teams average ~+2 point HFA.

    Given that convention:
        pick_signal = predicted_margin − spread_line
        actual_signal = actual_margin − spread_line
        > 0 → home covered (won by more than the spread)
        < 0 → away covered (home lost OR didn't cover)
        = 0 → push (excluded from hit-rate denominator)

    Parameters
    ----------
    min_gap
        Only count picks where |predicted_margin − spread_line| > min_gap.
        Lets you filter out coinflip disagreements (which the model isn't
        confident on) and isolate the "real opinion" subset.

    Returns
    -------
    (hit_rate, n_picks) — hit_rate is NaN if n_picks == 0.
    """
    valid = ~np.isnan(spread_lines) & ~np.isnan(predicted_margins) & ~np.isnan(actual_margins)
    if not valid.any():
        return float("nan"), 0
    pm = predicted_margins[valid]
    sl = spread_lines[valid]
    am = actual_margins[valid]

    pick_signal = pm - sl       # >0 -> bet home (we predict home outperforms spread)
    actual_signal = am - sl     # >0 -> home covered, <0 -> away covered

    picked_mask = np.abs(pick_signal) > min_gap
    push_mask = actual_signal == 0
    eligible = picked_mask & ~push_mask
    n = int(eligible.sum())
    if n == 0:
        return float("nan"), 0

    correct = (np.sign(pick_signal[eligible]) == np.sign(actual_signal[eligible]))
    return float(correct.mean()), n


# ---------------------------------------------------------------------------
# Trained v5 bundle
# ---------------------------------------------------------------------------
@dataclass
class NFLSpreadModelV5:
    """
    Trained v5 NFL spread-prediction bundle.

    NOTE: This is a REGRESSOR, not a classifier. The primary inference
    method is `predict_margin()`, which returns predicted points (positive
    = home wins by N). Cast to a probability via `margin_to_p_home()` if
    you need a win-probability scalar for newsletter copy.
    """

    regressor: XGBRegressor
    scaler: StandardScaler
    elo: NFLEloSystem
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float]
    version: str = "v5"

    # Inherited state from the V3-style feature builder
    form_state: Dict[str, List[int]] = field(default_factory=dict)
    last_game_date: Dict[str, str] = field(default_factory=dict)
    team_qb_ratings_latest: Dict[str, float] = field(default_factory=dict)
    team_qb_starters_latest: Dict[str, str] = field(default_factory=dict)
    team_metrics_latest: Dict[str, Dict[str, float]] = field(default_factory=dict)
    team_metrics_prior: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latest_season: Optional[int] = None

    # Calibration scalar used by margin_to_p_home (fit during training)
    margin_to_logit_scale: float = 0.15  # placeholder default; overridden by training

    def predict_margin(
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
    ) -> float:
        """
        Predict the point margin (home_score - away_score). Positive →
        home favored; negative → away favored. Magnitude indicates the
        expected margin in points.
        """
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
        )
        X = np.array([[feats[c] for c in self.feature_columns]], dtype=float)
        Xs = self.scaler.transform(X)
        return float(self.regressor.predict(Xs)[0])

    def margin_to_p_home(self, margin: float) -> float:
        """
        Convert a predicted point margin to an approximate win probability.

        Uses a logistic transform fit during training. Not a substitute
        for v3.1's calibrated classifier (the regression target is
        different and noisier); but useful when newsletter copy wants
        a single 'X% confidence' number alongside the predicted margin.
        """
        z = margin * self.margin_to_logit_scale
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_with_spread(
        self,
        home_team: str,
        away_team: str,
        vegas_spread_line: Optional[float],
        **predict_kwargs,
    ) -> Dict[str, float]:
        """
        One-shot helper that returns predicted margin, ATS pick, gap vs
        Vegas spread, and an approximate win probability. Useful for
        newsletter row generation.

        nflverse spread_line convention (per leakage_audit.py findings):
        POSITIVE when home is favored (spread_line = expected home
        margin). A spread_line of +7 means home is favored by 7.
        """
        margin = self.predict_margin(home_team, away_team, **predict_kwargs)
        p_home = self.margin_to_p_home(margin)
        result = {
            "predicted_margin": margin,
            "p_home": p_home,
            "pick_winner": home_team if margin > 0 else away_team,
        }
        if vegas_spread_line is not None and not np.isnan(vegas_spread_line):
            gap = margin - vegas_spread_line  # positive -> home outperforms spread
            result["vegas_spread_line"] = float(vegas_spread_line)
            result["ats_gap"] = float(gap)
            result["ats_pick"] = home_team if gap > 0 else away_team
            result["ats_confidence"] = abs(gap)
        return result

    # ------ Persistence ----------------------------------------------------
    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "regressor.pkl", "wb") as f:
            pickle.dump(self.regressor, f)
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
            "margin_to_logit_scale": self.margin_to_logit_scale,
            "kind": "regression_spread",
        }, indent=2))
        log.info("Saved NFLSpreadModelV5 bundle -> %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "NFLSpreadModelV5":
        with open(dir_path / "regressor.pkl", "rb") as f:
            regressor = pickle.load(f)
        with open(dir_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        elo = NFLEloSystem.from_dict(
            json.loads((dir_path / "elo_state.json").read_text())
        )
        feat_state = json.loads((dir_path / "feature_state.json").read_text())
        qb_state = json.loads((dir_path / "qb_ratings.json").read_text())
        team_state = json.loads((dir_path / "team_metrics.json").read_text())
        meta = json.loads((dir_path / "metadata.json").read_text())

        if meta.get("kind") != "regression_spread":
            raise RuntimeError(
                "This bundle is not a v5 spread model. "
                "Use NFLModelV3.load() for win-probability bundles."
            )

        return cls(
            regressor=regressor, scaler=scaler, elo=elo,
            feature_columns=meta["feature_columns"],
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
            version=meta.get("version", "v5"),
            form_state=feat_state.get("form_state", {}),
            last_game_date=feat_state.get("last_game_date", {}),
            team_qb_ratings_latest=qb_state.get("team_qb_ratings_latest", {}),
            team_qb_starters_latest=qb_state.get("team_qb_starters_latest", {}),
            team_metrics_latest=team_state.get("team_metrics_latest", {}),
            team_metrics_prior=team_state.get("team_metrics_prior", {}),
            latest_season=qb_state.get("latest_season"),
            margin_to_logit_scale=float(meta.get("margin_to_logit_scale", 0.15)),
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def _fit_margin_to_logit_scale(margins: np.ndarray, won: np.ndarray) -> float:
    """
    Fit a single-parameter logistic mapping from predicted margin to
    win probability: P(win) = sigmoid(scale * margin).

    Estimated from training-set actual margins and outcomes. Used at
    inference for the optional `margin_to_p_home()` helper.

    Returns the `scale` parameter. Typical NFL value is ~0.10-0.20 (a
    7-point favorite has roughly 67-70% win probability).
    """
    from scipy.optimize import minimize_scalar  # noqa: WPS433 (lazy import to avoid hard dep)

    def neg_loglik(scale: float) -> float:
        p = 1.0 / (1.0 + np.exp(-scale * margins))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -float(np.sum(won * np.log(p) + (1 - won) * np.log(1 - p)))

    try:
        res = minimize_scalar(neg_loglik, bounds=(0.001, 1.0), method="bounded")
        return float(res.x)
    except Exception:  # noqa: BLE001
        return 0.15  # fallback to NFL-typical scale


def train_nfl_spread_v5(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
    force_refresh_team_metrics: bool = False,
    feature_columns: Optional[List[str]] = None,
    version_label: str = "v5",
) -> NFLSpreadModelV5:
    """
    Train v5 — spread prediction (regression on home_score - away_score).

    Uses v3's leakage-free rolling feature builder. Same chronological
    train/test split as the win-probability models, so the holdout
    games match exactly across versions for cross-comparison.
    """
    if games.empty:
        raise ValueError("Cannot train on an empty games dataframe.")
    games = games.sort_values("game_date").reset_index(drop=True)

    if "point_diff" not in games.columns:
        # Belt-and-suspenders — football_loader computes it, but in case
        # an upstream caller skipped that step, derive it here.
        games = games.copy()
        games["point_diff"] = games["home_score"].astype(float) - games["away_score"].astype(float)

    feat_cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS_V5)

    fb = NFLFeatureBuilderV3()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games, force_refresh=force_refresh_qb_stats)
    fb.load_rolling_team_metrics(seasons, games, force_refresh=force_refresh_team_metrics)

    feat_df = fb.build_training_frame(games)
    # Attach the regression target + spread line for ATS sanity checks.
    # build_training_frame already preserves game_date/season/week + home_won,
    # but not point_diff or spread_line — pull them in by game date alignment.
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    feat_df["point_diff"] = games_sorted["point_diff"].values
    if "spread_line" in games_sorted.columns:
        feat_df["spread_line"] = games_sorted["spread_line"].values
    else:
        feat_df["spread_line"] = np.nan

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
    # Spread-line coverage diagnostic — historical games sometimes miss this.
    n_spread = int(feat_df["spread_line"].notna().sum())
    log.info("[%s] Vegas spread_line available for %d / %d games (%.1f%%).",
             version_label, n_spread, len(feat_df), 100.0 * n_spread / len(feat_df))

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["point_diff"].to_numpy(dtype=float)
    X_test = test_df[feat_cols].to_numpy(dtype=float)
    y_test = test_df["point_diff"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    regressor = XGBRegressor(
        n_estimators=settings.xgb_n_estimators,
        max_depth=settings.xgb_max_depth,
        learning_rate=settings.xgb_learning_rate,
        objective="reg:squarederror",
        random_state=settings.random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    log.info("[%s] Fitting XGBoost regressor on point margin …", version_label)
    regressor.fit(X_train_s, y_train)

    # Holdout predictions
    pred_test = regressor.predict(X_test_s)

    mae = float(mean_absolute_error(y_test, pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    r2 = float(r2_score(y_test, pred_test))
    # Quick ATS hit rate sanity-check on the holdout (full ATS analysis
    # lives in scripts/ats_diagnostics.py).
    spread_test = test_df["spread_line"].to_numpy(dtype=float)
    ats_overall, n_ats = ats_hit_rate(pred_test, spread_test, y_test, min_gap=0.0)
    ats_3pt, n_3pt = ats_hit_rate(pred_test, spread_test, y_test, min_gap=3.0)

    # Implied win-prob "accuracy" — just for cross-comparison with v1-v4.
    pred_winners = (pred_test > 0).astype(int)
    actual_winners = (y_test > 0).astype(int)
    implied_acc = float((pred_winners == actual_winners).mean())

    metrics = {
        "mae":             mae,
        "rmse":            rmse,
        "r2":              r2,
        "implied_accuracy": implied_acc,
        "ats_hit_rate":    ats_overall,
        "ats_n_picks":     int(n_ats),
        "ats_hit_rate_3pt": ats_3pt,
        "ats_n_picks_3pt": int(n_3pt),
        "n_train":         int(len(y_train)),
        "n_test":          int(len(y_test)),
    }
    log.info("[%s] Holdout metrics: %s", version_label, metrics)

    # Fit margin->probability mapping on TRAINING data (no leakage into test).
    train_won = (y_train > 0).astype(int)
    train_pred = regressor.predict(X_train_s)
    scale = _fit_margin_to_logit_scale(train_pred, train_won)
    log.info("[%s] Fitted margin->probability scale = %.4f "
             "(a 7-point favorite implies p_home = %.3f).",
             version_label, scale, 1.0 / (1.0 + np.exp(-7 * scale)))

    form_state = {team: list(d) for team, d in fb._form.items()}
    last_game_date = {team: dt.isoformat() for team, dt in fb._last_game_date.items()}
    team_metrics_prior = fb.end_of_season_by_team.get(
        (fb.latest_season - 1) if fb.latest_season else 0, {}
    )

    return NFLSpreadModelV5(
        regressor=regressor, scaler=scaler, elo=fb.elo,
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
        margin_to_logit_scale=scale,
    )
