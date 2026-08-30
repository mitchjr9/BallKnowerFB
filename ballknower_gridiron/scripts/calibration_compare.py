"""
ballknower_gridiron.scripts.calibration_compare
===============================================

Side-by-side calibration comparison of two or more trained NFL models.

When V3 has slightly better accuracy and V3.1 has slightly better log
loss / Brier, you need to compare their CALIBRATION directly to pick a
production model. Raw accuracy and the saved Brier give you a single
number; this script gives you the bucket-by-bucket reliability story
and a head-to-head ECE/MCE/Brier comparison.

What you get
------------
  * Per-version reliability table (predicted vs. actual win rate in each
    probability bucket).
  * Side-by-side ECE/MCE/Brier/log_loss/accuracy summary with the BEST
    score in each column marked with a ✓.
  * Per-version probability distribution counts (high-confidence pick
    volume above various thresholds).
  * Overlay reliability diagram saved to PNG if matplotlib is installed.

Usage
-----
    # Compare two
    python -m ballknower_gridiron.scripts.calibration_compare --versions v3 v3.1

    # Compare any number
    python -m ballknower_gridiron.scripts.calibration_compare --versions v3 v3.1 v4.1

    # Custom bin count and plot directory
    python -m ballknower_gridiron.scripts.calibration_compare \\
        --versions v3 v3.1 --bins 12 --plot-dir ~/Desktop

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.scripts.calibration_diagnostics import (
    compute_calibration_bins,
    expected_calibration_error,
    max_calibration_error,
)
from ballknower_gridiron.scripts.feature_importance import rebuild_holdout
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class VersionStats:
    """One row of the side-by-side comparison."""
    version: str
    n_features: int
    y_prob: np.ndarray
    y_true: np.ndarray
    bins: list
    ece: float
    mce: float
    brier: float
    log_loss_val: float
    accuracy: float
    roc_auc: float


def compute_version_stats(version: str, n_bins: int) -> Optional[VersionStats]:
    """Load model, rebuild its holdout, compute all comparison metrics."""
    from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model
    log.info("Loading %s …", version)
    try:
        model = load_active_nfl_model(version)
    except FileNotFoundError:
        log.error(
            "No trained %s bundle at %s — skip (train it first).",
            version, settings.models_dir_for(version),
        )
        return None

    log.info("Rebuilding holdout for %s …", version)
    try:
        X_test_s, y_test, _ = rebuild_holdout(version, model)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not rebuild holdout for %s: %s", version, exc)
        return None

    y_prob = model.clf.predict_proba(X_test_s)[:, 1]
    bins = compute_calibration_bins(y_prob, y_test, n_bins=n_bins)
    return VersionStats(
        version=version,
        n_features=len(model.feature_columns),
        y_prob=y_prob,
        y_true=y_test,
        bins=bins,
        ece=expected_calibration_error(bins, len(y_test)),
        mce=max_calibration_error(bins),
        brier=float(brier_score_loss(y_test, y_prob)),
        log_loss_val=float(log_loss(y_test, y_prob, labels=[0, 1])),
        accuracy=float(accuracy_score(y_test, (y_prob >= 0.5).astype(int))),
        roc_auc=float(roc_auc_score(y_test, y_prob)),
    )


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------
def print_summary_table(stats_list: List[VersionStats]) -> None:
    """
    Print the head-to-head metric table. Marks the BEST value in each
    column with a ✓. For ECE/MCE/Brier/log_loss, lower is better; for
    accuracy/ROC, higher is better.
    """
    print()
    print("=" * 88)
    print("  SUMMARY — head-to-head metric comparison")
    print("=" * 88)
    print(f"  {'Version':<8} {'#Feat':>6}  "
          f"{'Accuracy':>10} {'ROC AUC':>9} "
          f"{'LogLoss':>9} {'Brier':>9} "
          f"{'ECE':>8} {'MCE':>8}")
    print("  " + "-" * 84)

    # Identify best per column
    best_acc = max(s.accuracy for s in stats_list)
    best_roc = max(s.roc_auc for s in stats_list)
    best_ll = min(s.log_loss_val for s in stats_list)
    best_brier = min(s.brier for s in stats_list)
    best_ece = min(s.ece for s in stats_list)
    best_mce = min(s.mce for s in stats_list)

    def _mark(value: float, best_value: float, fmt: str) -> str:
        is_best = abs(value - best_value) < 1e-9
        return (f"{value:{fmt}}✓" if is_best else f"{value:{fmt}} ")

    for s in stats_list:
        print(f"  {s.version:<8} {s.n_features:>6}  "
              f"{_mark(s.accuracy, best_acc, '9.4f'):>10} "
              f"{_mark(s.roc_auc, best_roc, '8.4f'):>9} "
              f"{_mark(s.log_loss_val, best_ll, '8.4f'):>9} "
              f"{_mark(s.brier, best_brier, '8.4f'):>9} "
              f"{_mark(s.ece, best_ece, '7.4f'):>8} "
              f"{_mark(s.mce, best_mce, '7.4f'):>8}")
    print()
    print("  ✓ marks the best value in each column.")
    print("  Lower is better: LogLoss, Brier, ECE, MCE.")
    print("  Higher is better: Accuracy, ROC AUC.")
    print()


def print_reliability_table_combined(stats_list: List[VersionStats]) -> None:
    """
    For each probability bucket, print every version's predicted vs.
    actual side by side. Lets you spot patterns like 'V3 over-predicts
    at the high end but V3.1 doesn't'.
    """
    print("=" * 88)
    print("  RELIABILITY by probability bucket")
    print("  (predicted -> actual)  — miss column = actual − predicted")
    print("=" * 88)

    # All versions should have the same bin edges (same n_bins).
    n_bins = len(stats_list[0].bins)
    bin_ranges = [(b["lo"], b["hi"]) for b in stats_list[0].bins]

    # Header
    header = f"  {'Range':<14}"
    for s in stats_list:
        header += f" │ {s.version:<13} (n  pred  act  miss)"
    print(header)
    print("  " + "-" * 86)

    for i in range(n_bins):
        lo, hi = bin_ranges[i]
        row = f"  {lo:.2f}-{hi:.2f}     "
        for s in stats_list:
            b = s.bins[i]
            if b["n"] == 0:
                row += f" │ {'':<13}    0    -     -      -  "
            else:
                miss = b["miss"]
                sign = "+" if miss >= 0 else "-"
                row += (f" │ {'':<13} {b['n']:>3} "
                        f"{b['predicted']:>5.3f} {b['actual']:>5.3f} "
                        f"{sign}{abs(miss):>4.3f}")
        print(row)
    print()


def print_confidence_volume(stats_list: List[VersionStats]) -> None:
    """
    For each version, show how many high-confidence picks per threshold.
    Critical for newsletter content — confident-pick volume is what
    drives a weekly column.
    """
    print("=" * 88)
    print("  CONFIDENT-PICK VOLUME (picks ≥ threshold OR ≤ 1-threshold)")
    print("  Tells you how many strong opinions per holdout the model produces.")
    print("=" * 88)

    header = f"  {'Threshold':<14}"
    for s in stats_list:
        header += f" │ {s.version:<13} (count  share)"
    print(header)
    print("  " + "-" * 86)

    for thresh in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        row = f"  ≥{thresh:.2f} or ≤{1-thresh:.2f}"
        for s in stats_list:
            total = len(s.y_prob)
            n = int(((s.y_prob >= thresh) | (s.y_prob <= 1 - thresh)).sum())
            share = (n / total) * 100 if total else 0.0
            row += f" │ {'':<13} {n:>4}  {share:>5.1f}%"
        print(row)
    print()


def print_winner_interpretation(stats_list: List[VersionStats]) -> None:
    """
    Plain-English interpretation: which version is best for which use case.
    """
    print("=" * 88)
    print("  RECOMMENDATION")
    print("=" * 88)
    # Find best on each axis
    best_acc = max(stats_list, key=lambda s: s.accuracy)
    best_calib = min(stats_list, key=lambda s: s.ece)
    best_brier = min(stats_list, key=lambda s: s.brier)

    print(f"  Highest accuracy:        {best_acc.version}  ({best_acc.accuracy:.4f})")
    print(f"  Best calibration (ECE):  {best_calib.version}  ({best_calib.ece:.4f})")
    print(f"  Best Brier:              {best_brier.version}  ({best_brier.brier:.4f})")
    print()
    if best_acc.version == best_calib.version == best_brier.version:
        print(f"  → {best_acc.version} wins on all three. Clear production choice.")
    else:
        print(f"  → No single winner. For a NEWSLETTER use case where probability")
        print(f"    quality matters more than raw accuracy (e.g., publishing")
        print(f"    confidence-tiered picks), prefer {best_calib.version} for its")
        print(f"    better calibration. For a pure 'who wins' use case, prefer")
        print(f"    {best_acc.version}.")
    print()


# ---------------------------------------------------------------------------
# Optional overlay plot
# ---------------------------------------------------------------------------
def maybe_save_overlay_plot(
    stats_list: List[VersionStats],
    output_dir: Path,
) -> bool:
    """Overlay reliability diagram across versions in one PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed — skipping overlay plot.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, s in enumerate(stats_list):
        xs = [b["predicted"] for b in s.bins if b["n"] > 0]
        ys = [b["actual"] for b in s.bins if b["n"] > 0]
        sizes = [max(20.0, b["n"] / 2.0) for b in s.bins if b["n"] > 0]
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, marker="o", markersize=0, linewidth=1.5,
                alpha=0.7, label=f"{s.version} (ECE={s.ece:.3f})")
        ax.scatter(xs, ys, s=sizes, color=c, alpha=0.7)

    ax.set_xlabel("Predicted probability (mean per bin)")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability comparison")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "calibration_compare.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved overlay reliability plot -> %s", out_path)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side calibration comparison of trained NFL models.",
    )
    parser.add_argument(
        "--versions", nargs="+", required=True,
        help="Model versions to compare (e.g. v3 v3.1 v4.1).",
    )
    parser.add_argument(
        "--bins", type=int, default=10,
        help="Number of probability bins (default 10).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip overlay PNG even if matplotlib is available.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Where to save PNGs (default: settings.logs_dir).",
    )
    args = parser.parse_args()

    valid = {"v1", "v2", "v3", "v3.1", "v3_1", "v4", "v4.1", "v4_1"}
    unknown = [v for v in args.versions if v.lower() not in valid]
    if unknown:
        log.error("Unknown version(s): %s — choose from %s.",
                  unknown, sorted(valid))
        return 2

    stats_list: List[VersionStats] = []
    for v in args.versions:
        s = compute_version_stats(v.lower(), n_bins=args.bins)
        if s is not None:
            stats_list.append(s)

    if len(stats_list) < 2:
        log.error("Need at least 2 successfully loaded models to compare; got %d.",
                  len(stats_list))
        return 1

    # Verify all versions used the same holdout size (sanity check).
    holdout_sizes = {len(s.y_true) for s in stats_list}
    if len(holdout_sizes) > 1:
        log.warning(
            "Holdout sizes differ across versions: %s. Comparison still works "
            "but suggests the chronological split is hitting different game sets.",
            holdout_sizes,
        )

    print_summary_table(stats_list)
    print_reliability_table_combined(stats_list)
    print_confidence_volume(stats_list)
    print_winner_interpretation(stats_list)

    if not args.no_plots:
        out_dir = Path(args.plot_dir) if args.plot_dir else settings.logs_dir
        maybe_save_overlay_plot(stats_list, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
