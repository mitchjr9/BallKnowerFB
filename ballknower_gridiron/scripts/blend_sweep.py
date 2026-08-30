"""
ballknower_gridiron.scripts.blend_sweep
=======================================

Find the ELO blend weight that actually belongs in production, by measuring
it instead of guessing.

Why this exists
---------------
Every calibration number we trust about v3.1 — ECE 0.0352, the 0.70-0.80
bucket landing at 0.741 against a 0.738 prediction — was computed on the
model's **raw** probabilities. But the newsletter publishes a *blended*
probability:

    p_blend = (1 - w) * p_model + w * p_elo

which is a different distribution with its own calibration that nobody has
measured. Shipping `w = 0.50` because it happens to be the scaffold default
means publishing numbers whose calibration is unverified — which undercuts
the entire reason v3.1 was chosen over v3.

This sweeps w from 0.00 to 1.00 and reports ECE, MCE, Brier, log loss,
accuracy, ROC AUC, and confident-pick volume at each step, on the same
chronological holdout the saved metrics came from.

What to expect, so a surprising result gets a second look
--------------------------------------------------------
Blending a well-calibrated forecaster with a second, *differently*
calibrated one usually costs calibration and buys a little robustness. Three
plausible shapes:

  * **Minimum at or near w = 0.0** — v3.1's isotonic calibration is already
    doing the work and ELO only dilutes it. Ship pure model.
  * **Shallow minimum around w = 0.2-0.4** — ELO is correcting the model in
    the tails where its training data is thin. Ship the measured weight.
  * **Minimum at high w** — would mean ELO alone is better calibrated than
    the calibrated classifier. Treat that as suspicious rather than as a
    result: check that the holdout is the one you think it is before acting.

Read the whole curve, not just the argmin. ECE on ~495 games has real
sampling noise, so a minimum that is 0.002 below its neighbours is a tie,
not a winner — the script says so explicitly rather than making you eyeball it.

Usage
-----
    # Default: sweep the active model in 0.05 steps
    python -m ballknower_gridiron.scripts.blend_sweep --version v3.1

    # Coarser grid, more bins
    python -m ballknower_gridiron.scripts.blend_sweep --version v3.1 \\
        --step 0.10 --bins 12

    # Compare two candidates head to head
    python -m ballknower_gridiron.scripts.blend_sweep --version v3.1 v4.1

    # Write the curve to CSV for plotting
    python -m ballknower_gridiron.scripts.blend_sweep --version v3.1 \\
        --csv logs/blend_sweep_v3_1.csv

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.scripts.calibration_diagnostics import (
    compute_calibration_bins,
    expected_calibration_error,
    max_calibration_error,
)
from ballknower_gridiron.scripts.feature_importance import rebuild_holdout
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)

# An ECE difference smaller than this is noise on a ~500-game holdout, not a
# result. Used to decide whether to report a winner or a plateau.
ECE_TIE_BAND = 0.004


@dataclass
class BlendPoint:
    """One row of the sweep: all metrics at a single blend weight."""
    w: float
    ece: float
    mce: float
    brier: float
    log_loss_val: float
    accuracy: float
    roc_auc: float
    n_conf_65: int      # picks at >= 0.65 or <= 0.35
    n_conf_75: int      # picks at >= 0.75 or <= 0.25


def elo_probabilities(X_test_raw: np.ndarray,
                      feature_columns: List[str]) -> Optional[np.ndarray]:
    """
    Recover the ELO baseline probability for each holdout game.

    `elo_diff_with_hca` is already a model feature — (home_elo + HCA) −
    away_elo — so the baseline is recoverable from the raw (unscaled)
    holdout matrix without re-walking ELO history. Pulling it from the same
    frame the model was scored on also guarantees the two probabilities are
    aligned game-for-game, which a separate ELO replay would not.
    """
    if "elo_diff_with_hca" in feature_columns:
        idx = feature_columns.index("elo_diff_with_hca")
        diff = X_test_raw[:, idx].astype(float)
    elif "elo_diff" in feature_columns:
        # Fall back to the HCA-free column and add the setting back in.
        idx = feature_columns.index("elo_diff")
        diff = X_test_raw[:, idx].astype(float) + settings.elo_hca
        log.warning("elo_diff_with_hca not in feature set — reconstructing "
                    "from elo_diff + settings.elo_hca (%.1f).", settings.elo_hca)
    else:
        return None
    return 1.0 / (1.0 + np.power(10.0, -diff / 400.0))


def evaluate_blend(p_model: np.ndarray, p_elo: np.ndarray, y_true: np.ndarray,
                   w: float, n_bins: int) -> BlendPoint:
    """Compute every metric at one blend weight."""
    p = (1.0 - w) * p_model + w * p_elo
    # Clip only to keep log_loss finite; the blend itself cannot leave [0,1].
    p_safe = np.clip(p, 1e-9, 1 - 1e-9)
    bins = compute_calibration_bins(p, y_true, n_bins=n_bins)
    return BlendPoint(
        w=w,
        ece=expected_calibration_error(bins, len(y_true)),
        mce=max_calibration_error(bins),
        brier=float(brier_score_loss(y_true, p)),
        log_loss_val=float(log_loss(y_true, p_safe, labels=[0, 1])),
        accuracy=float(accuracy_score(y_true, (p >= 0.5).astype(int))),
        roc_auc=float(roc_auc_score(y_true, p)),
        n_conf_65=int(((p >= 0.65) | (p <= 0.35)).sum()),
        n_conf_75=int(((p >= 0.75) | (p <= 0.25)).sum()),
    )


def sweep_version(version: str, step: float, n_bins: int
                  ) -> Optional[Tuple[List[BlendPoint], int]]:
    """Run the full sweep for one model version. Returns (points, n_holdout)."""
    from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model

    log.info("Loading %s …", version)
    try:
        model = load_active_nfl_model(version)
    except FileNotFoundError:
        log.error("No trained %s bundle at %s — train it first.",
                  version, settings.models_dir_for(version))
        return None

    log.info("Rebuilding holdout for %s …", version)
    X_test_s, y_test, X_test_raw = rebuild_holdout(version, model)

    p_model = model.clf.predict_proba(X_test_s)[:, 1]
    p_elo = elo_probabilities(X_test_raw, model.feature_columns)
    if p_elo is None:
        log.error("Neither elo_diff_with_hca nor elo_diff is in %s's feature "
                  "set — cannot reconstruct the ELO baseline.", version)
        return None

    weights = [round(i * step, 4) for i in range(int(round(1.0 / step)) + 1)]
    points = [evaluate_blend(p_model, p_elo, y_test, w, n_bins) for w in weights]
    return points, int(len(y_test))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_sweep_table(version: str, points: List[BlendPoint], n_holdout: int) -> None:
    best_ece = min(p.ece for p in points)
    best_brier = min(p.brier for p in points)
    best_ll = min(p.log_loss_val for p in points)
    best_acc = max(p.accuracy for p in points)

    print()
    print("=" * 88)
    print(f"  BLEND SWEEP — {version}   (holdout: {n_holdout} games)")
    print(f"  p_blend = (1 - w) * p_model + w * p_elo")
    print("=" * 88)
    print(f"  {'w':>6}  {'ECE':>9} {'MCE':>8} {'Brier':>9} {'LogLoss':>9} "
          f"{'Acc':>8} {'ROC':>8}  {'≥.65':>5} {'≥.75':>5}")
    print("  " + "-" * 84)
    for p in points:
        marks = ""
        if abs(p.ece - best_ece) < 1e-12:
            marks += "  ← best ECE"
        if abs(p.brier - best_brier) < 1e-12:
            marks += "  ← best Brier"
        if abs(p.log_loss_val - best_ll) < 1e-12:
            marks += "  ← best LogLoss"
        if abs(p.accuracy - best_acc) < 1e-12:
            marks += "  ← best Acc"
        print(f"  {p.w:>6.2f}  {p.ece:>9.4f} {p.mce:>8.4f} {p.brier:>9.4f} "
              f"{p.log_loss_val:>9.4f} {p.accuracy:>8.4f} {p.roc_auc:>8.4f}  "
              f"{p.n_conf_65:>5} {p.n_conf_75:>5}{marks}")
    print()
    print("  ≥.65 / ≥.75 = confident-pick volume (picks at or beyond that")
    print("  probability on either side). Blending toward ELO usually moves")
    print("  these, which changes how much publishable content a week yields.")
    print()


def print_verdict(version: str, points: List[BlendPoint], n_holdout: int) -> float:
    """
    Interpret the curve and return the recommended weight.

    Reports a plateau rather than an argmin when the minimum isn't separated
    from its neighbours by more than sampling noise. On ~500 games an ECE
    gap of a few thousandths is not a real difference, and picking the exact
    argmin off a noisy curve is its own kind of overfitting — to the holdout.
    """
    best = min(points, key=lambda p: p.ece)
    tied = [p for p in points if p.ece <= best.ece + ECE_TIE_BAND]
    baseline = next((p for p in points if abs(p.w - 0.0) < 1e-9), None)
    current = next((p for p in points
                    if abs(p.w - settings.default_blend_elo) < 1e-9), None)

    print("=" * 88)
    print("  VERDICT")
    print("=" * 88)
    print(f"  Lowest ECE:        w = {best.w:.2f}  (ECE {best.ece:.4f}, "
          f"Brier {best.brier:.4f}, acc {best.accuracy:.4f})")
    if baseline is not None:
        print(f"  Pure model (w=0):  ECE {baseline.ece:.4f}, "
              f"Brier {baseline.brier:.4f}, acc {baseline.accuracy:.4f}")
    if current is not None:
        print(f"  Current default:   w = {settings.default_blend_elo:.2f}  "
              f"(ECE {current.ece:.4f}, Brier {current.brier:.4f}, "
              f"acc {current.accuracy:.4f})")
    print()

    if len(tied) > 1:
        lo, hi = min(p.w for p in tied), max(p.w for p in tied)
        print(f"  ► The minimum is NOT well separated. Every weight in "
              f"[{lo:.2f}, {hi:.2f}] is within")
        print(f"    {ECE_TIE_BAND:.3f} ECE of the best — that band is sampling "
              f"noise on {n_holdout} games,")
        print(f"    not a difference you can act on. Picking the exact argmin "
              f"off this curve")
        print(f"    would be overfitting to the holdout.")
        print()
        # Within a tie, prefer the simplest defensible choice.
        if any(abs(p.w) < 1e-9 for p in tied):
            rec = 0.0
            why = ("w = 0.00 is inside the tied band, and it is the only "
                   "weight whose calibration\n    we have already validated "
                   "end to end. Prefer the simpler configuration when the\n"
                   "    measurement cannot distinguish them.")
        else:
            rec = float(np.median([p.w for p in tied]))
            rec = min(tied, key=lambda p: abs(p.w - rec)).w
            why = (f"w = {rec:.2f} sits in the middle of the tied band, which "
                   f"is more robust to\n    resampling than either edge.")
        print(f"  ► RECOMMENDED: w = {rec:.2f}")
        print(f"    {why}")
    else:
        rec = best.w
        print(f"  ► RECOMMENDED: w = {rec:.2f}")
        print(f"    The minimum is separated from its neighbours by more than "
              f"{ECE_TIE_BAND:.3f} ECE,")
        print(f"    so this is a real result rather than noise.")

    # Flag the case where blending buys calibration but costs picks.
    rec_pt = min(points, key=lambda p: abs(p.w - rec))
    if baseline is not None and rec_pt.n_conf_75 < baseline.n_conf_75 * 0.8:
        print()
        print(f"  ⚠ Note: at w = {rec:.2f} the ≥0.75 pick volume drops from "
              f"{baseline.n_conf_75} to {rec_pt.n_conf_75}.")
        print(f"    Better calibration, fewer strong opinions per week. That's "
              f"a real editorial")
        print(f"    trade-off, not a free win — decide it deliberately.")

    print()
    print(f"  Set it with:  export NFL_BLEND_ELO_DEFAULT={rec:.2f}")
    print()
    return rec


def write_csv(path: Path, rows: Dict[str, List[BlendPoint]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["version", "w", "ece", "mce", "brier", "log_loss",
                      "accuracy", "roc_auc", "n_conf_65", "n_conf_75"])
        for version, points in rows.items():
            for p in points:
                wtr.writerow([version, f"{p.w:.4f}", f"{p.ece:.6f}",
                              f"{p.mce:.6f}", f"{p.brier:.6f}",
                              f"{p.log_loss_val:.6f}", f"{p.accuracy:.6f}",
                              f"{p.roc_auc:.6f}", p.n_conf_65, p.n_conf_75])
    log.info("Wrote sweep curve -> %s", path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the ELO blend weight and pick it by measurement.")
    parser.add_argument("--version", nargs="+", default=["v3.1"],
                        help="Model version(s) to sweep (default: v3.1).")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Blend-weight grid step (default: 0.05).")
    parser.add_argument("--bins", type=int, default=10,
                        help="Calibration bins (default: 10).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write the sweep curve as CSV.")
    args = parser.parse_args(argv)

    if not (0.001 <= args.step <= 0.5):
        log.error("--step must be between 0.001 and 0.5.")
        return 2

    results: Dict[str, List[BlendPoint]] = {}
    for version in args.version:
        out = sweep_version(version.lower(), args.step, args.bins)
        if out is None:
            continue
        points, n_holdout = out
        results[version] = points
        print_sweep_table(version, points, n_holdout)
        print_verdict(version, points, n_holdout)

    if not results:
        log.error("No versions could be swept.")
        return 1

    if args.csv:
        write_csv(Path(args.csv), results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
