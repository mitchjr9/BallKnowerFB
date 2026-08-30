"""
ballknower_gridiron.scripts.calibration_diagnostics
===================================================

Two diagnostics that matter for newsletter content more than raw accuracy:

1. **Reliability curve (calibration plot)** — bucket holdout predictions
   by predicted probability and check whether actual win rate in each
   bucket matches. If 0.75-confidence picks actually win 75% of the
   time, your probabilities are trustworthy and the "confident picks"
   angle for the newsletter works. If 0.75-confidence picks only win
   55% of the time, the model's confidence is inflated.

2. **Predicted-probability distribution** — histogram of predicted
   probabilities. Tells you how many "confident pick" candidates you
   produce per game. If the model rarely outputs > 0.7, you have few
   high-conviction picks to feature; if it often does, you have
   plenty of newsletter material. Also reveals whether the model is
   "narrow" (clustered around 0.5) or "spread" (lots of confident
   predictions).

Both are computed on the same chronological holdout that produced the
saved accuracy/AUC metrics — so they reflect actual model behavior on
games the model didn't train on.

Quality metrics reported
------------------------
  * Expected Calibration Error (ECE) — weighted average of |predicted -
    actual| across bins, weighted by bin count. Lower is better.
    Well-calibrated models target < 0.05.
  * Maximum Calibration Error — worst single-bin miss. Lower is better.
  * Brier score and log loss for reference.

Optional plotting
-----------------
If matplotlib is installed, the script will also save:
  * `calibration_<version>.png` — reliability diagram with histogram inset
  * `proba_dist_<version>.png` — predicted probability histogram

Without matplotlib, the text output is fully usable on its own.

Usage
-----
    python -m ballknower_gridiron.scripts.calibration_diagnostics --version v3
    python -m ballknower_gridiron.scripts.calibration_diagnostics --version v4 --bins 20
    python -m ballknower_gridiron.scripts.calibration_diagnostics --version v3 --no-plots

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.scripts.feature_importance import rebuild_holdout
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Calibration computation
# ---------------------------------------------------------------------------
def compute_calibration_bins(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> List[dict]:
    """
    Group predictions into equal-width probability bins and compute:
        - bin range
        - count of predictions in this bin
        - mean predicted probability
        - actual win rate
        - difference (calibration miss)
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin includes the upper edge; others are [lo, hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({
                "lo": float(lo), "hi": float(hi),
                "n": 0, "predicted": float("nan"), "actual": float("nan"),
                "miss": float("nan"),
            })
            continue
        mean_pred = float(y_prob[mask].mean())
        actual = float(y_true[mask].mean())
        bins.append({
            "lo": float(lo), "hi": float(hi),
            "n": n, "predicted": mean_pred, "actual": actual,
            "miss": actual - mean_pred,
        })
    return bins


def expected_calibration_error(bins: List[dict], total_n: int) -> float:
    """
    ECE = sum over bins of (bin_count / total) * |actual - predicted|.
    Only non-empty bins contribute.
    """
    if total_n <= 0:
        return 0.0
    return float(sum(
        (b["n"] / total_n) * abs(b["miss"])
        for b in bins if b["n"] > 0
    ))


def max_calibration_error(bins: List[dict]) -> float:
    """Largest absolute miss in any non-empty bin."""
    non_empty = [abs(b["miss"]) for b in bins if b["n"] > 0]
    return float(max(non_empty)) if non_empty else 0.0


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
_BAR = "█"


def _hbar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    return _BAR * max(0, min(width, int(round((value / max_value) * width))))


def print_calibration_table(
    bins: List[dict], total_n: int, version: str,
) -> None:
    print()
    print("=" * 78)
    print(f"  Reliability (calibration) — {version}")
    print(f"  Bucket holdout predictions by predicted probability and check")
    print(f"  whether actual win rate matches. Miss should be small in every bin.")
    print("=" * 78)
    print(f"  {'Range':<14} {'Count':>6} {'Pred':>7} {'Actual':>7} {'Miss':>7}  Reliability")
    print("  " + "-" * 70)

    max_count = max((b["n"] for b in bins), default=1)
    for b in bins:
        rng = f"{b['lo']:.2f}-{b['hi']:.2f}"
        if b["n"] == 0:
            print(f"  {rng:<14} {0:>6}  {'-':>6}  {'-':>6}  {'-':>6}")
            continue
        # Reliability bar: green-ish when miss is small (we'll just use a single bar)
        miss = b["miss"]
        sign = "+" if miss >= 0 else "-"
        print(f"  {rng:<14} {b['n']:>6} {b['predicted']:>6.3f} "
              f"{b['actual']:>6.3f}  {sign}{abs(miss):>5.3f}  "
              f"{_hbar(b['n'], max_count, width=20)}")
    print()
    ece = expected_calibration_error(bins, total_n)
    mce = max_calibration_error(bins)
    print(f"  Expected Calibration Error (ECE): {ece:.4f}   "
          f"(lower is better; < 0.05 is well-calibrated)")
    print(f"  Maximum Calibration Error (MCE):  {mce:.4f}   "
          f"(largest single-bin miss)")
    print()


def print_probability_histogram(
    y_prob: np.ndarray, version: str, n_bins: int = 20,
) -> None:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(y_prob, bins=edges)
    total = int(len(y_prob))

    print("=" * 78)
    print(f"  Predicted-probability distribution — {version}")
    print(f"  How spread are the model's predictions? (n = {total})")
    print("=" * 78)
    print(f"  {'Range':<14} {'Count':>6} {'Share':>7}  Histogram")
    print("  " + "-" * 70)
    max_count = int(max(counts.max(), 1))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        rng = f"{lo:.2f}-{hi:.2f}"
        n = int(counts[i])
        share = (n / total) * 100 if total else 0.0
        print(f"  {rng:<14} {n:>6} {share:>6.1f}%  {_hbar(n, max_count, width=40)}")

    # Confidence thresholds — common newsletter use cases.
    print()
    print(f"  {'Threshold':<18} {'Count':>6} {'Share':>7}")
    print("  " + "-" * 35)
    for thresh in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        # Either side of 0.5 counts — picks are "either team wins" not just home.
        n_above = int(((y_prob >= thresh) | (y_prob <= 1 - thresh)).sum())
        share = (n_above / total) * 100 if total else 0.0
        print(f"  ≥{thresh:.2f} or ≤{1-thresh:.2f}     {n_above:>6} {share:>6.1f}%")
    print()


# ---------------------------------------------------------------------------
# Optional plotting (matplotlib)
# ---------------------------------------------------------------------------
def maybe_save_plots(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    bins: List[dict],
    version: str,
    output_dir: Path,
) -> bool:
    """Save reliability + histogram plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed — skipping plots. Install with `pip install matplotlib` if you want PNGs.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Reliability diagram
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [b["predicted"] for b in bins if b["n"] > 0]
    ys = [b["actual"] for b in bins if b["n"] > 0]
    sizes = [max(20.0, b["n"] / 2.0) for b in bins if b["n"] > 0]
    ax.scatter(xs, ys, s=sizes, alpha=0.7, label="Holdout bin")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted probability (mean per bin)")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — {version}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    rel_path = output_dir / f"calibration_{version.replace('.', '_')}.png"
    fig.savefig(rel_path, dpi=120)
    plt.close(fig)
    log.info("Saved reliability diagram -> %s", rel_path)

    # 2) Probability histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob, bins=np.linspace(0, 1, 21), edgecolor="black", alpha=0.85)
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.7, label="0.5 (coinflip)")
    for thresh, color in [(0.65, "orange"), (0.75, "red")]:
        ax.axvline(thresh, color=color, linestyle=":", alpha=0.7,
                   label=f"≥{thresh:.2f}")
        ax.axvline(1 - thresh, color=color, linestyle=":", alpha=0.7)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted-probability distribution — {version}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = output_dir / f"proba_dist_{version.replace('.', '_')}.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    log.info("Saved probability distribution -> %s", hist_path)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reliability / calibration diagnostics for a trained NFL model.",
    )
    parser.add_argument(
        "--version", default=None,
        help="Model version (v1, v2, v3, v3.1, v4, v4.1). Defaults to settings.active_model_version.",
    )
    parser.add_argument(
        "--bins", type=int, default=10,
        help="Number of probability bins for the reliability table (default 10).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip PNG plots even if matplotlib is available.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Where to save PNGs (default: settings.logs_dir).",
    )
    args = parser.parse_args()

    version = (args.version or settings.active_model_version).lower()
    valid = {"v1", "v2", "v3", "v3.1", "v3_1", "v4", "v4.1", "v4_1"}
    if version not in valid:
        log.error("Unknown version %r — choose from %s.", version, sorted(valid))
        return 2

    log.info("Loading %s model …", version)
    from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model
    try:
        model = load_active_nfl_model(version)
    except FileNotFoundError:
        log.error(
            "No trained %s bundle at %s. Train it first with:\n"
            "    python -m ballknower_gridiron.scripts.train_football_model --version %s",
            version, settings.models_dir_for(version), version,
        )
        return 2

    print(f"\nLoaded {version} ({len(model.feature_columns)} features).")
    if model.metrics:
        m = model.metrics
        print(f"Saved metrics: accuracy={m.get('accuracy', 0):.4f}  "
              f"ROC AUC={m.get('roc_auc', 0):.4f}  "
              f"log_loss={m.get('log_loss', 0):.4f}  "
              f"brier={m.get('brier', 0):.4f}\n")

    log.info("Rebuilding holdout (this can take a few minutes for v3+) …")
    try:
        X_test_s, y_test, _ = rebuild_holdout(version, model)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not rebuild holdout: %s", exc)
        return 1

    # Get probability predictions on the same holdout.
    y_prob = model.clf.predict_proba(X_test_s)[:, 1]

    # Sanity check against saved metrics — should match within rounding.
    ll = float(log_loss(y_test, y_prob, labels=[0, 1]))
    brier = float(brier_score_loss(y_test, y_prob))
    log.info("Rebuilt holdout reproduces log_loss=%.4f, brier=%.4f", ll, brier)

    bins = compute_calibration_bins(y_prob, y_test, n_bins=args.bins)
    print_calibration_table(bins, total_n=int(len(y_test)), version=version)
    print_probability_histogram(y_prob, version=version)

    if not args.no_plots:
        out_dir = Path(args.plot_dir) if args.plot_dir else settings.logs_dir
        maybe_save_plots(y_prob, y_test, bins, version, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
