"""
ballknower_gridiron.scripts.backtest_nfl
========================================

Compare a trained NFL model against a pure-ELO baseline (and against
each other version, if multiple are trained), sweep blend weights to
find the optimum, and emit per-ELO-gap accuracy buckets + calibration
buckets.

Designed to mirror `scripts.backtest_nba` in shape so the newsletter
copy you write for basketball and football reads the same way.

Examples
--------
    python -m ballknower_gridiron.scripts.backtest_nfl
    python -m ballknower_gridiron.scripts.backtest_nfl --version v3
    python -m ballknower_gridiron.scripts.backtest_nfl --markdown report.md
    python -m ballknower_gridiron.scripts.backtest_nfl --holdout-seasons 2

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date as date_cls
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make runnable as `python scripts/backtest_nfl.py` too.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ballknower_gridiron.config.settings import settings  # noqa: E402
from ballknower_gridiron.data.football_loader import load_nfl_games  # noqa: E402
from ballknower_gridiron.models.football_model import (  # noqa: E402
    NFLModel,
)
from ballknower_gridiron.models.football_model_v2 import (  # noqa: E402
    NFLModelV2,
    load_active_nfl_model,
)
from ballknower_gridiron.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("backtest_nfl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _elo_prob_only(home_rating: float, away_rating: float, hca: float) -> float:
    """Pure-ELO win probability for home team (logistic, 400-point scale)."""
    diff = (home_rating + hca) - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _safe_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(int) == y))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _model_predict(model, row) -> float:
    """Call predict_proba with whichever kwargs the model accepts."""
    return model.predict_proba(
        home_team=row.home_team,
        away_team=row.away_team,
        game_date=row.game_date.date()
        if hasattr(row.game_date, "date")
        else row.game_date,
        season=int(row.season),
        week=int(row.week),
        home_rest=float(row.home_rest),
        away_rest=float(row.away_rest),
        is_playoff=bool(getattr(row, "is_playoff", 0)),
        is_international=bool(getattr(row, "is_international", 0)),
        is_div_game=bool(getattr(row, "div_game", 0) or 0),
    )[0]


def _build_holdout(games: pd.DataFrame, holdout_seasons: int) -> pd.DataFrame:
    seasons_sorted = sorted(games["season"].astype(int).unique().tolist())
    cutoff_season = seasons_sorted[-holdout_seasons]
    holdout = games[games["season"] >= cutoff_season].copy()
    log.info(
        "Holdout: seasons %s..%s, %d games",
        cutoff_season, seasons_sorted[-1], len(holdout),
    )
    return holdout


def _model_probs_from_replay(model, games_all: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """
    Re-replay games chronologically using a *fresh* feature builder for the
    appropriate model version, taking the model's calibrated probability
    on each holdout game.
    """
    # Use the trained model's feature columns to decide which builder to use.
    feat_cols = set(model.feature_columns)
    seasons_int = sorted(games_all["season"].astype(int).unique().tolist())

    if "net_pts_diff" in feat_cols:
        from ballknower_gridiron.models.football_model_v3 import (
            NFLFeatureBuilderV3,
        )
        fb = NFLFeatureBuilderV3()
        # Rebuild both rolling tables from PBP/weekly stats — same pipeline
        # as training. This guarantees the backtest sees identical feature
        # values to what the model was trained on (no inference-fallback
        # cheating).
        fb.load_rolling_qb_ratings(seasons_int, games_all)
        fb.load_rolling_team_metrics(seasons_int, games_all)
    elif "qb_rating_diff" in feat_cols:
        from ballknower_gridiron.models.football_model_v2 import (
            NFLFeatureBuilderV2,
        )
        fb = NFLFeatureBuilderV2()
        fb.load_rolling_qb_ratings(seasons_int, games_all)
    else:
        from ballknower_gridiron.models.football_model import NFLFeatureBuilder
        fb = NFLFeatureBuilder()

    games_sorted = games_all.sort_values("game_date").reset_index(drop=True)
    holdout_index = set(holdout.index.tolist())  # noqa: F841  (kept for symmetry)
    # We need to match rows from `games_sorted` back to `holdout` rows. The
    # cleanest way is to match by (home_team, away_team, game_date).
    holdout_keys = {
        (r.home_team, r.away_team, pd.Timestamp(r.game_date).date()): None
        for r in holdout.itertuples(index=False)
    }

    log.info("Replaying %d games to score holdout via %s …",
             len(games_sorted), model.version)
    probs: List[Tuple] = []
    for row in games_sorted.itertuples(index=False):
        game_date = row.game_date
        if hasattr(game_date, "date"):
            game_date = game_date.date()
        key = (row.home_team, row.away_team, game_date)
        if key in holdout_keys:
            # Score BEFORE updating ELO/state for this game.
            feats = fb.features_for_matchup(
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
            X = np.array([[feats[c] for c in model.feature_columns]], dtype=float)
            Xs = model.scaler.transform(X)
            p_home = float(model.clf.predict_proba(Xs)[0, 1])
            holdout_keys[key] = p_home
            probs.append((key, p_home))

        # Update state with the actual result.
        fb.record_game(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=int(row.home_score),
            away_score=int(row.away_score),
            game_date=game_date,
            season=int(row.season),
            is_playoff=bool(getattr(row, "is_playoff", 0)),
            is_international=bool(getattr(row, "is_international", 0)),
        )

    # Build aligned arrays in holdout order
    out = np.zeros(len(holdout), dtype=float)
    for i, r in enumerate(holdout.itertuples(index=False)):
        gd = r.game_date
        if hasattr(gd, "date"):
            gd = gd.date()
        out[i] = holdout_keys.get((r.home_team, r.away_team, gd), 0.5)
    return out


def _elo_replay_probs(games_all: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """
    Pure-ELO baseline: replay all games chronologically, snapshot the
    pre-game logistic probability for each holdout game.
    """
    from ballknower_gridiron.models.football_elo import NFLEloSystem
    elo = NFLEloSystem()
    games_sorted = games_all.sort_values("game_date").reset_index(drop=True)

    holdout_keys: Dict = {}
    for r in holdout.itertuples(index=False):
        gd = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        holdout_keys[(r.home_team, r.away_team, gd)] = None

    for row in games_sorted.itertuples(index=False):
        gd = row.game_date.date() if hasattr(row.game_date, "date") else row.game_date
        key = (row.home_team, row.away_team, gd)
        if key in holdout_keys:
            hca = 0.0 if bool(getattr(row, "is_international", 0)) else settings.elo_hca
            p = _elo_prob_only(
                elo.get_rating(row.home_team),
                elo.get_rating(row.away_team),
                hca,
            )
            holdout_keys[key] = p
        elo.update_game(
            row.home_team, row.away_team,
            int(row.home_score), int(row.away_score),
            gd, int(row.season),
            bool(getattr(row, "is_playoff", 0)),
            bool(getattr(row, "is_international", 0)),
        )

    out = np.zeros(len(holdout), dtype=float)
    for i, r in enumerate(holdout.itertuples(index=False)):
        gd = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        out[i] = holdout_keys.get((r.home_team, r.away_team, gd), 0.5)
    return out


def _blend_sweep(
    model_p: np.ndarray, elo_p: np.ndarray, y: np.ndarray, steps: int = 11,
) -> Tuple[float, float, pd.DataFrame]:
    """Sweep w_elo in [0,1], return (best_w, best_logloss, full_table)."""
    rows = []
    weights = np.linspace(0.0, 1.0, steps)
    for w in weights:
        blended = w * elo_p + (1 - w) * model_p
        rows.append({
            "w_elo": float(w),
            "accuracy": _accuracy(y, blended),
            "log_loss": _safe_log_loss(y, blended),
            "brier": _brier(y, blended),
        })
    df = pd.DataFrame(rows)
    best = df.loc[df["log_loss"].idxmin()]
    return float(best["w_elo"]), float(best["log_loss"]), df


def _elo_gap_buckets(
    home_elo: np.ndarray, away_elo: np.ndarray,
    p_model: np.ndarray, y: np.ndarray,
    bin_edges=(-200, -100, -50, -25, 0, 25, 50, 100, 200),
) -> pd.DataFrame:
    gap = home_elo - away_elo
    edges = list(bin_edges)
    edges = [-math.inf] + edges + [math.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gap > lo) & (gap <= hi)
        if not mask.any():
            continue
        rows.append({
            "elo_gap_range": f"({lo:.0f}, {hi:.0f}]",
            "n_games": int(mask.sum()),
            "model_acc": _accuracy(y[mask], p_model[mask]),
            "elo_baseline_acc": _accuracy(y[mask], (gap[mask] > 0).astype(float)),
            "mean_model_p": float(p_model[mask].mean()),
            "actual_home_win_rate": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def _calibration_buckets(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            rows.append({
                "bucket": f"[{lo:.1f}, {hi:.1f})",
                "n": 0, "mean_p": float("nan"), "actual_p": float("nan"),
                "gap": float("nan"),
            })
            continue
        mp = float(p[mask].mean())
        ap = float(y[mask].mean())
        rows.append({
            "bucket": f"[{lo:.1f}, {hi:.1f})",
            "n": int(mask.sum()),
            "mean_p": mp,
            "actual_p": ap,
            "gap": ap - mp,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Backtest a trained BallKnower Gridiron NFL model.",
    )
    ap.add_argument(
        "--version",
        choices=["v1", "v2", "v3"],
        default=settings.active_model_version,
        help="Trained model version to backtest (default: %(default)s).",
    )
    ap.add_argument(
        "--seasons-back",
        type=int,
        default=settings.nfl_seasons_back,
        help="How many seasons to load (default: %(default)s).",
    )
    ap.add_argument(
        "--holdout-seasons",
        type=int,
        default=2,
        help="Last N seasons treated as the holdout (default: %(default)s).",
    )
    ap.add_argument(
        "--no-playoffs",
        dest="include_playoffs",
        action="store_false",
        help="Exclude postseason from the holdout.",
    )
    ap.set_defaults(include_playoffs=True)
    ap.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Optional path to write a Markdown report (for newsletter copy).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    settings.ensure_dirs()
    log.info("=" * 72)
    log.info("Backtest NFL %s | seasons_back=%d | holdout=%d seasons",
             args.version, args.seasons_back, args.holdout_seasons)
    log.info("=" * 72)

    games = load_nfl_games(
        seasons_back=args.seasons_back,
        include_playoffs=args.include_playoffs,
        completed_only=True,
        force_refresh=False,
    )
    if games.empty:
        log.error("No games loaded — aborting.")
        return 2

    holdout = _build_holdout(games, args.holdout_seasons)

    model = load_active_nfl_model(args.version)
    log.info("Loaded model %s — trained_at=%s", model.version, model.trained_at)

    p_model = _model_probs_from_replay(model, games, holdout)
    p_elo = _elo_replay_probs(games, holdout)
    y = holdout["home_won"].to_numpy(dtype=int)

    # Headline metrics
    print("\n=== Holdout metrics ===")
    print(f"{'model':<10} acc={_accuracy(y, p_model):.4f}  "
          f"logloss={_safe_log_loss(y, p_model):.4f}  "
          f"brier={_brier(y, p_model):.4f}")
    print(f"{'elo-only':<10} acc={_accuracy(y, p_elo):.4f}  "
          f"logloss={_safe_log_loss(y, p_elo):.4f}  "
          f"brier={_brier(y, p_elo):.4f}")

    # Blend sweep
    best_w, best_ll, sweep = _blend_sweep(p_model, p_elo, y, steps=11)
    print("\n=== Blend sweep (w_elo · ELO + (1-w_elo) · model) ===")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest blend: w_elo={best_w:.2f}, logloss={best_ll:.4f}")

    # ELO-gap buckets — need pre-game ELOs. Re-replay just for snapshots.
    from ballknower_gridiron.models.football_elo import NFLEloSystem
    elo_snap = NFLEloSystem()
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    holdout_keys = {}
    for r in holdout.itertuples(index=False):
        gd = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        holdout_keys[(r.home_team, r.away_team, gd)] = (None, None)
    for row in games_sorted.itertuples(index=False):
        gd = row.game_date.date() if hasattr(row.game_date, "date") else row.game_date
        key = (row.home_team, row.away_team, gd)
        if key in holdout_keys:
            holdout_keys[key] = (
                elo_snap.get_rating(row.home_team),
                elo_snap.get_rating(row.away_team),
            )
        elo_snap.update_game(
            row.home_team, row.away_team,
            int(row.home_score), int(row.away_score),
            gd, int(row.season),
            bool(getattr(row, "is_playoff", 0)),
            bool(getattr(row, "is_international", 0)),
        )

    home_elos = np.zeros(len(holdout))
    away_elos = np.zeros(len(holdout))
    for i, r in enumerate(holdout.itertuples(index=False)):
        gd = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        he, ae = holdout_keys.get((r.home_team, r.away_team, gd), (1500.0, 1500.0))
        home_elos[i] = he if he is not None else 1500.0
        away_elos[i] = ae if ae is not None else 1500.0

    gap_df = _elo_gap_buckets(home_elos, away_elos, p_model, y)
    print("\n=== Accuracy by pre-game ELO gap ===")
    print(gap_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    calib_df = _calibration_buckets(p_model, y, n_bins=10)
    print("\n=== Calibration (model) ===")
    print(calib_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Optional markdown export
    if args.markdown:
        _write_markdown(
            Path(args.markdown), model.version, model.trained_at,
            y, p_model, p_elo, sweep, best_w, best_ll, gap_df, calib_df,
            n_games=len(holdout),
        )
        print(f"\n✓ Wrote Markdown report -> {args.markdown}")

    return 0


def _write_markdown(
    path: Path,
    version: str,
    trained_at: str,
    y: np.ndarray,
    p_model: np.ndarray,
    p_elo: np.ndarray,
    sweep: pd.DataFrame,
    best_w: float,
    best_ll: float,
    gap_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    n_games: int,
) -> None:
    """Emit a one-page Markdown report you can paste into a newsletter."""
    lines: List[str] = []
    lines.append(f"# BallKnower Gridiron — {version} backtest\n")
    lines.append(f"_Trained at: {trained_at}; holdout games: {n_games}_\n")
    lines.append("\n## Headline\n")
    lines.append(
        f"- **Model accuracy:** {_accuracy(y, p_model):.1%}  "
        f"(ELO baseline: {_accuracy(y, p_elo):.1%})"
    )
    lines.append(
        f"- **Log-loss:** {_safe_log_loss(y, p_model):.4f}  "
        f"(ELO: {_safe_log_loss(y, p_elo):.4f})"
    )
    lines.append(
        f"- **Brier:** {_brier(y, p_model):.4f}  "
        f"(ELO: {_brier(y, p_elo):.4f})"
    )
    lines.append(
        f"- **Best blend:** w_elo = {best_w:.2f} → log-loss {best_ll:.4f}\n"
    )
    lines.append("\n## Blend sweep\n")
    lines.append(sweep.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Accuracy by ELO gap\n")
    lines.append(gap_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Calibration\n")
    lines.append(calib_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n---\n")
    lines.append(
        "_For entertainment and educational purposes only — not financial or "
        "betting advice._\n"
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
