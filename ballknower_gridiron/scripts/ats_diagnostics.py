"""
ballknower_gridiron.scripts.ats_diagnostics
===========================================

Detailed against-the-spread (ATS) analysis for a trained v5 spread
model. Where the training output gives a single overall ATS hit rate,
this script slices the holdout by **confidence tier** — the gap between
predicted margin and Vegas spread — and shows hit rate per tier.

Why tiering matters
-------------------
A model that disagrees with Vegas by 0.5 points isn't really expressing
an opinion. A model that disagrees by 5+ points IS. Newsletter content
should be driven by the high-confidence tiers, not the overall hit rate.
Concretely:
  * Overall ATS hit rate of 51% might hide a TIER of picks where the
    model is 56% — and those are the picks worth publishing.
  * Conversely, a "look at this 55% ATS hit rate!" headline might
    collapse if you discover all the lift came from one fluky tier.

Output sections
---------------
  1. Headline metrics on the holdout (MAE / RMSE / R² / overall ATS).
  2. ATS hit rate broken down by confidence-gap tier:
       - Strong   (gap ≥ 5.0 pts)
       - Medium   (3.0-5.0)
       - Light    (1.5-3.0)
       - Coinflip (< 1.5)
  3. Threshold sweep — for each cutoff X, "if we only pick gap ≥ X,
     what's our hit rate and how many picks per season?"
  4. Side breakdown — does the model do better picking home or away?
  5. Predicted-vs-actual scatter plot (PNG) if matplotlib available.

Profitability reminder
----------------------
At standard -110 vig, the break-even ATS hit rate is 52.38%. A tier
that hits 53-54% is real but tiny edge; 55-57% is meaningful; > 57%
sustained is either world-class or a measurement artifact.

Usage
-----
    python -m ballknower_gridiron.scripts.ats_diagnostics --version v5
    python -m ballknower_gridiron.scripts.ats_diagnostics --version v5 --no-plots

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# Standard -110 vig break-even: 110 / 210 = 52.38%
ATS_BREAKEVEN = 110.0 / 210.0


# Confidence tiers (gap = |predicted_margin + spread_line|)
TIER_BOUNDS = [
    ("Strong  ", 5.0, float("inf")),
    ("Medium  ", 3.0, 5.0),
    ("Light   ", 1.5, 3.0),
    ("Coinflip", 0.0, 1.5),
]


@dataclass
class HoldoutData:
    predicted_margin: np.ndarray
    spread_line:      np.ndarray
    actual_margin:    np.ndarray
    is_home_picked:   np.ndarray  # bool
    is_correct:       np.ndarray  # bool
    ats_gap:          np.ndarray  # signed; sign = pick direction
    abs_gap:          np.ndarray  # magnitude
    is_push:          np.ndarray  # bool — pushes excluded from hit rates
    n_total:          int
    n_valid:          int         # excludes NaN spread, NaN margin, pushes


def rebuild_v5_holdout(model) -> HoldoutData:
    """
    Rebuild the v5 holdout using the same rolling-feature pipeline as
    training. Returns predicted margins, actual margins, and Vegas
    spreads aligned game-by-game with the chronological holdout split.
    """
    from ballknower_gridiron.data.football_loader import load_nfl_games
    from ballknower_gridiron.models.football_model_v3 import NFLFeatureBuilderV3

    log.info("Loading NFL games and rebuilding v5 features …")
    games = load_nfl_games(seasons_back=settings.nfl_seasons_back)
    games = games.sort_values("game_date").reset_index(drop=True)
    if "point_diff" not in games.columns:
        games["point_diff"] = games["home_score"].astype(float) - games["away_score"].astype(float)

    fb = NFLFeatureBuilderV3()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games)
    fb.load_rolling_team_metrics(seasons, games)

    feat_df = fb.build_training_frame(games)
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    feat_df["point_diff"] = games_sorted["point_diff"].values
    if "spread_line" in games_sorted.columns:
        feat_df["spread_line"] = games_sorted["spread_line"].values
    else:
        feat_df["spread_line"] = np.nan

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    test_df = feat_df.iloc[split_idx:].copy()

    X_test = test_df[model.feature_columns].to_numpy(dtype=float)
    X_test_s = model.scaler.transform(X_test)
    pred = model.regressor.predict(X_test_s)
    spreads = test_df["spread_line"].to_numpy(dtype=float)
    actuals = test_df["point_diff"].to_numpy(dtype=float)

    # ATS pick: predicted_margin > spread → bet home outperforms.
    # nflverse spread_line is POSITIVE when home is favored (= home's
    # expected margin), so the comparison is (predicted - spread), NOT
    # (predicted + spread). See leakage_audit.py findings.
    gap = pred - spreads
    actual_signal = actuals - spreads
    is_push = actual_signal == 0
    is_home_picked = gap > 0
    is_correct = np.sign(gap) == np.sign(actual_signal)

    valid_mask = ~np.isnan(pred) & ~np.isnan(spreads) & ~np.isnan(actuals) & ~is_push

    return HoldoutData(
        predicted_margin=pred,
        spread_line=spreads,
        actual_margin=actuals,
        is_home_picked=is_home_picked,
        is_correct=is_correct,
        ats_gap=gap,
        abs_gap=np.abs(gap),
        is_push=is_push,
        n_total=int(len(pred)),
        n_valid=int(valid_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_pct_with_edge(rate: float, n: int) -> str:
    """Format hit rate, marking it relative to the -110 break-even."""
    if n == 0 or np.isnan(rate):
        return "  -   "
    edge = (rate - ATS_BREAKEVEN) * 100
    edge_mark = f"  ({edge:+.1f}pp vs break-even)" if not np.isnan(edge) else ""
    return f"{rate*100:.1f}%{edge_mark}"


def print_headline(model, h: HoldoutData) -> None:
    print()
    print("=" * 78)
    print(f"  v5 spread model — holdout diagnostics")
    print("=" * 78)
    print(f"  Trained: {model.trained_at}")
    print(f"  Holdout: {h.n_total} games ({h.n_valid} ATS-eligible after"
          f" excluding pushes / missing spreads)")
    m = model.metrics
    if m:
        print(f"  MAE: {m.get('mae', 0):.3f} pts   "
              f"RMSE: {m.get('rmse', 0):.3f} pts   "
              f"R²: {m.get('r2', 0):.4f}")
        print(f"  Implied accuracy (sign of margin): {m.get('implied_accuracy', 0):.4f}")
        ats = m.get("ats_hit_rate", float("nan"))
        n_picks = m.get("ats_n_picks", 0)
        print(f"  Overall ATS: {_format_pct_with_edge(ats, n_picks)} ({n_picks} picks)")
    print()
    print(f"  Profitability reminder: standard -110 vig requires "
          f"{ATS_BREAKEVEN*100:.2f}% to break even.")
    print()


def print_tier_breakdown(h: HoldoutData) -> None:
    """
    For each confidence tier, show: # of picks, hit rate, edge vs vig,
    and side distribution.
    """
    print("=" * 78)
    print("  ATS HIT RATE BY CONFIDENCE TIER")
    print("  Gap = |predicted_margin + spread_line|. Larger = stronger opinion.")
    print("=" * 78)
    print(f"  {'Tier':<10} {'Range':<12} {'Picks':>6} {'Hit %':>8}  {'Edge':>9}  Distribution")
    print("  " + "-" * 74)

    valid = ~np.isnan(h.spread_line) & ~np.isnan(h.predicted_margin) & ~np.isnan(h.actual_margin)
    valid &= ~h.is_push

    for label, lo, hi in TIER_BOUNDS:
        in_tier = valid & (h.abs_gap >= lo) & (h.abs_gap < hi)
        n = int(in_tier.sum())
        if n == 0:
            print(f"  {label} {f'{lo:.1f}-{hi:.1f}' if hi != float('inf') else f'≥{lo:.1f}':<12} "
                  f"{0:>6}   -      -          (no picks)")
            continue
        wins = int(h.is_correct[in_tier].sum())
        rate = wins / n
        edge_pp = (rate - ATS_BREAKEVEN) * 100
        # Distribution: how many home vs away picks?
        n_home = int(h.is_home_picked[in_tier].sum())
        n_away = n - n_home
        range_str = f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else f"≥{lo:.1f}"
        edge_str = f"{edge_pp:+5.1f}pp"
        print(f"  {label} {range_str:<12} "
              f"{n:>6}  {rate*100:>5.1f}%  {edge_str:>9}  "
              f"home={n_home}/{n}, away={n_away}/{n}")
    print()


def print_threshold_sweep(h: HoldoutData) -> None:
    """
    "If we only bet picks with gap >= X, what's our hit rate?"
    A cleaner way to find the right confidence cutoff for newsletter
    publishing.
    """
    print("=" * 78)
    print("  THRESHOLD SWEEP — 'only publish picks where gap ≥ X'")
    print("=" * 78)
    print(f"  {'Cutoff':>8}  {'Picks':>6}  {'Per season*':>12}  {'Hit %':>8}  {'Edge':>9}")
    print("  " + "-" * 60)

    valid = ~np.isnan(h.spread_line) & ~np.isnan(h.predicted_margin) & ~np.isnan(h.actual_margin)
    valid &= ~h.is_push
    # Estimate "per season" by assuming the holdout spans ~272 games per
    # season (17 weeks × 16 games). The holdout fraction is typically
    # ~1.5-2 seasons.
    seasons_in_holdout = max(1.0, h.n_total / 272.0)

    for cutoff in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        in_set = valid & (h.abs_gap >= cutoff)
        n = int(in_set.sum())
        if n == 0:
            print(f"  ≥{cutoff:>4.1f}    {0:>6}    -          -        -")
            continue
        rate = float(h.is_correct[in_set].mean())
        edge_pp = (rate - ATS_BREAKEVEN) * 100
        per_season = n / seasons_in_holdout
        print(f"  ≥{cutoff:>4.1f}    {n:>6}    "
              f"{per_season:>10.1f}  {rate*100:>6.1f}%   "
              f"{edge_pp:+5.1f}pp")
    print()
    print(f"  *per season estimate assumes ~272 games/season; "
          f"holdout spans ~{seasons_in_holdout:.1f} seasons.")
    print()


def print_side_breakdown(h: HoldoutData) -> None:
    """Does the model do better picking home or away?"""
    valid = ~np.isnan(h.spread_line) & ~np.isnan(h.predicted_margin) & ~np.isnan(h.actual_margin)
    valid &= ~h.is_push

    home_picks = valid & h.is_home_picked
    away_picks = valid & ~h.is_home_picked

    print("=" * 78)
    print("  SIDE BREAKDOWN — picking home vs picking away")
    print("=" * 78)
    for label, mask in [("Home picks", home_picks), ("Away picks", away_picks)]:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<14} 0 picks")
            continue
        wins = int(h.is_correct[mask].sum())
        rate = wins / n
        edge_pp = (rate - ATS_BREAKEVEN) * 100
        print(f"  {label:<14} {n:>4} picks   "
              f"{rate*100:>5.1f}% hit rate   "
              f"{edge_pp:+5.1f}pp vs break-even")
    print()


def maybe_save_scatter_plot(
    h: HoldoutData,
    version: str,
    output_dir: Path,
) -> bool:
    """Predicted-vs-actual margin scatter plot (with diagonal reference)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed — skipping scatter plot.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = ~np.isnan(h.predicted_margin) & ~np.isnan(h.actual_margin)
    pm = h.predicted_margin[valid]
    am = h.actual_margin[valid]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pm, am, alpha=0.5, s=20)
    lim = max(abs(pm).max(), abs(am).max()) + 2
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, label="Perfect prediction")
    ax.axhline(0, color="grey", alpha=0.3, linewidth=0.8)
    ax.axvline(0, color="grey", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Predicted margin (home − away)")
    ax.set_ylabel("Actual margin (home − away)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Predicted vs actual margin — {version}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"spread_scatter_{version.replace('.', '_')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved scatter plot -> %s", out_path)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="ATS diagnostics for a trained v5 spread model.",
    )
    parser.add_argument(
        "--version", default="v5",
        help="Model version (currently only v5 produces spread models).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip PNG even if matplotlib is available.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Where to save PNGs (default: settings.logs_dir).",
    )
    args = parser.parse_args()

    from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
    bundle_dir = settings.models_dir_for(args.version)
    log.info("Loading %s spread model from %s …", args.version, bundle_dir)
    try:
        model = NFLSpreadModelV5.load(bundle_dir)
    except FileNotFoundError:
        log.error(
            "No trained %s bundle. Train it first with:\n"
            "    python -m ballknower_gridiron.scripts.train_football_model --version %s",
            args.version, args.version,
        )
        return 2

    h = rebuild_v5_holdout(model)
    print_headline(model, h)
    print_tier_breakdown(h)
    print_threshold_sweep(h)
    print_side_breakdown(h)

    if not args.no_plots:
        out_dir = Path(args.plot_dir) if args.plot_dir else settings.logs_dir
        maybe_save_scatter_plot(h, args.version, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
