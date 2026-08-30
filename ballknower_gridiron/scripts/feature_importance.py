"""
ballknower_gridiron.scripts.feature_importance
==============================================

Inspect what features a trained NFL model is actually relying on.

Why this matters
----------------
After training, the holdout accuracy number tells you the model works
overall — but not WHY. Feature importance answers two questions:

  1. Which features carry the predictive signal?
  2. Which features are noise (or redundant with stronger features)?

For NFL models, the practical questions tend to be:
  * Is ELO doing 80% of the work, or have the new features earned their seats?
  * Are the team-efficiency features (V3) actually informative, or are
    they just correlated noise on top of ELO?
  * Is the QB rating differentiating starters, or has it collapsed to
    near-zero importance?

Two importance metrics computed
-------------------------------
  * **Native XGBoost gain importance** — for each feature, the mean
    reduction in training loss when that feature is used in a tree
    split, averaged across the calibration folds. Fast, comes free
    with the trained booster. Tends to OVER-attribute to correlated
    features because the tree only needs to split on one of them.

  * **Permutation importance** (optional, slower) — for each feature,
    shuffle its values in the holdout set, predict again, measure
    drop in accuracy. This is the gold-standard signal: it tells you
    what would happen if a feature were replaced with random noise.
    Penalizes correlated features properly because shuffling one
    when its twin is still intact yields little drop.

Both views are useful together. If a feature is high on gain but low
on permutation, it's redundant with a correlated feature. If it's
high on both, it's pulling its own weight.

Usage
-----
    python -m ballknower_gridiron.scripts.feature_importance --version v3
    python -m ballknower_gridiron.scripts.feature_importance --version v2 --permutation
    python -m ballknower_gridiron.scripts.feature_importance --version v3 --permutation --repeats 10

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature -> category mapping for the summary view
# ---------------------------------------------------------------------------
# Patterns are substring matches; the FIRST pattern that matches wins.
CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("ELO",                 ["elo"]),
    ("Team efficiency",     ["net_pts", "net_epa", "off_epa", "def_epa", "mean_pace"]),
    ("Weather (v4)",        ["is_cold", "is_windy", "is_dome"]),
    ("QB change (v4)",      ["qb_change"]),
    ("QB rating",           ["qb_rating", "mean_qb", "qb_healthy"]),
    ("Form (last 5)",       ["form"]),
    ("Rest & schedule",     ["rest", "short_week", "bye_week"]),
    ("Context flags",       ["playoff", "international", "div_game", "hca"]),
]


def categorize_feature(name: str) -> str:
    """Return the high-level category for a feature column name."""
    n = name.lower()
    for cat, patterns in CATEGORY_PATTERNS:
        for p in patterns:
            if p in n:
                return cat
    return "Other"


# ---------------------------------------------------------------------------
# Native XGBoost gain importance
# ---------------------------------------------------------------------------
def get_xgb_importance_avg(
    model, importance_type: str = "gain",
) -> List[Tuple[str, float]]:
    """
    Extract XGBoost importance from a CalibratedClassifierCV wrapper,
    averaged across calibration folds. Returns list of (feature_name,
    importance) sorted descending. Importances are normalized within
    each fold to sum to 1.0 so cross-fold averaging is meaningful.
    """
    cc = getattr(model, "clf", None)
    if cc is None or not hasattr(cc, "calibrated_classifiers_"):
        log.error("Model classifier has no calibrated_classifiers_; cannot extract importance.")
        return []

    feat_cols = list(model.feature_columns)
    n_feats = len(feat_cols)
    fold_arrays: List[np.ndarray] = []

    for calc in cc.calibrated_classifiers_:
        est = getattr(calc, "estimator", None) or getattr(calc, "base_estimator", None)
        if est is None or not hasattr(est, "get_booster"):
            continue
        try:
            booster = est.get_booster()
            score_dict = booster.get_score(importance_type=importance_type)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read %s importance from a fold: %s", importance_type, exc)
            continue

        arr = np.zeros(n_feats, dtype=float)
        for key, val in score_dict.items():
            # Trained on numpy arrays -> XGBoost uses 'f0', 'f1', ... names.
            # Trained on DataFrame -> column names. Handle both.
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < n_feats:
                    arr[idx] = float(val)
            elif key in feat_cols:
                arr[feat_cols.index(key)] = float(val)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        fold_arrays.append(arr)

    if not fold_arrays:
        return []

    avg = np.mean(fold_arrays, axis=0)
    pairs = list(zip(feat_cols, avg.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Permutation importance on the holdout set
# ---------------------------------------------------------------------------
def rebuild_holdout(version: str, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-run the training pipeline up to the chronological train/test split
    to recover the exact holdout (X_test_scaled, y_test, X_test_raw).

    This matches what train_nfl_model_v*() does, so permutation results
    are over the SAME games the saved metrics were computed on. v3.1 and
    v4.1 use their parent's builder (the bundle's feature_columns drives
    which columns are sliced out for prediction).
    """
    from ballknower_gridiron.data.football_loader import load_nfl_games

    games = load_nfl_games(seasons_back=settings.nfl_seasons_back)
    games = games.sort_values("game_date").reset_index(drop=True)

    normalized = version.lower().replace(".", "_")
    log.info("Rebuilding holdout for %s …", version)
    if normalized == "v1":
        from ballknower_gridiron.models.football_model import NFLFeatureBuilder
        fb = NFLFeatureBuilder()
    elif normalized == "v2":
        from ballknower_gridiron.models.football_model_v2 import NFLFeatureBuilderV2
        fb = NFLFeatureBuilderV2()
        seasons = sorted(games["season"].astype(int).unique().tolist())
        fb.load_rolling_qb_ratings(seasons, games)
    elif normalized in ("v3", "v3_1"):
        # v3.1 uses the v3 builder; only the trained feature_columns differ.
        from ballknower_gridiron.models.football_model_v3 import NFLFeatureBuilderV3
        fb = NFLFeatureBuilderV3()
        seasons = sorted(games["season"].astype(int).unique().tolist())
        fb.load_rolling_qb_ratings(seasons, games)
        fb.load_rolling_team_metrics(seasons, games)
    elif normalized in ("v4", "v4_1"):
        # v4.1 uses the v4 builder; only the trained feature_columns differ.
        from ballknower_gridiron.models.football_model_v4 import NFLFeatureBuilderV4
        fb = NFLFeatureBuilderV4()
        seasons = sorted(games["season"].astype(int).unique().tolist())
        fb.load_rolling_qb_ratings(seasons, games)
        fb.load_rolling_team_metrics(seasons, games)
    else:
        raise ValueError(f"Unknown version: {version!r}")

    feat_df = fb.build_training_frame(games)
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    test_df = feat_df.iloc[split_idx:]
    X_test = test_df[model.feature_columns].to_numpy(dtype=float)
    y_test = test_df["home_won"].to_numpy(dtype=int)
    X_test_s = model.scaler.transform(X_test)
    return X_test_s, y_test, X_test


def compute_permutation_importance(
    model, X_test_s: np.ndarray, y_test: np.ndarray, n_repeats: int = 5,
) -> List[Tuple[str, float, float, float]]:
    """
    For each feature, shuffle its column N times and measure how much
    accuracy drops vs. the baseline (unshuffled) prediction.

    Returns: list of (feature_name, mean_acc_drop, mean_logloss_increase,
                      mean_roc_auc_drop), sorted by acc_drop descending.

    A feature with mean_acc_drop near zero contributes nothing actionable
    to the holdout predictions. A feature with a large drop is genuinely
    pulling the model toward correct answers.
    """
    feat_cols = list(model.feature_columns)
    proba_base = model.clf.predict_proba(X_test_s)[:, 1]
    baseline_acc = accuracy_score(y_test, (proba_base >= 0.5).astype(int))
    baseline_ll = log_loss(y_test, proba_base, labels=[0, 1])
    baseline_auc = roc_auc_score(y_test, proba_base)
    log.info(
        "Baseline holdout: accuracy=%.4f, log_loss=%.4f, ROC AUC=%.4f, n=%d",
        baseline_acc, baseline_ll, baseline_auc, len(y_test),
    )

    rng = np.random.default_rng(settings.random_state)
    results: List[Tuple[str, float, float, float]] = []
    for j, feat in enumerate(feat_cols):
        acc_drops, ll_incs, auc_drops = [], [], []
        for _ in range(n_repeats):
            X_perm = X_test_s.copy()
            rng.shuffle(X_perm[:, j])
            proba_perm = model.clf.predict_proba(X_perm)[:, 1]
            acc = accuracy_score(y_test, (proba_perm >= 0.5).astype(int))
            ll = log_loss(y_test, proba_perm, labels=[0, 1])
            try:
                auc = roc_auc_score(y_test, proba_perm)
            except ValueError:
                auc = baseline_auc
            acc_drops.append(baseline_acc - acc)
            ll_incs.append(ll - baseline_ll)
            auc_drops.append(baseline_auc - auc)
        results.append((
            feat,
            float(np.mean(acc_drops)),
            float(np.mean(ll_incs)),
            float(np.mean(auc_drops)),
        ))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
_BAR_CHARS = "█"
_BAR_WIDTH = 40


def _bar(value: float, max_value: float, width: int = _BAR_WIDTH) -> str:
    if max_value <= 0:
        return ""
    n = int(round((value / max_value) * width))
    return _BAR_CHARS * max(0, min(width, n))


def print_native_chart(pairs: List[Tuple[str, float]], version: str) -> None:
    print()
    print("=" * 78)
    print(f"  XGBoost native GAIN importance — {version}")
    print(f"  ({len(pairs)} features; values normalized to sum=1 per fold, averaged)")
    print("=" * 78)
    if not pairs:
        print("  (no importance data available)")
        return
    max_imp = max(p[1] for p in pairs)
    for i, (feat, imp) in enumerate(pairs, 1):
        cat = categorize_feature(feat)
        print(f"  {i:2}. {feat:<22} [{cat:<17}] {imp*100:6.2f}%  {_bar(imp, max_imp)}")
    print()


def print_permutation_chart(
    results: List[Tuple[str, float, float, float]], version: str,
) -> None:
    print()
    print("=" * 78)
    print(f"  Permutation importance on holdout — {version}")
    print("  (mean accuracy drop when feature is shuffled; bigger = more important)")
    print("=" * 78)
    if not results:
        print("  (no permutation data available)")
        return
    max_drop = max(abs(r[1]) for r in results) if results else 1.0
    for i, (feat, acc_drop, ll_inc, auc_drop) in enumerate(results, 1):
        cat = categorize_feature(feat)
        sign = "+" if acc_drop >= 0 else "-"
        print(f"  {i:2}. {feat:<22} [{cat:<17}] "
              f"Δacc={sign}{abs(acc_drop)*100:5.2f}pp  "
              f"ΔAUC={auc_drop*100:+5.2f}pp  "
              f"{_bar(max(0.0, acc_drop), max_drop)}")
    print()


def print_category_summary(
    pairs: List[Tuple[str, float]],
    label: str,
) -> None:
    """Roll up per-feature importance into the higher-level categories."""
    by_cat: Dict[str, float] = {}
    for feat, val in pairs:
        cat = categorize_feature(feat)
        by_cat[cat] = by_cat.get(cat, 0.0) + max(val, 0.0)

    print("-" * 78)
    print(f"  By category ({label}):")
    print("-" * 78)
    items = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)
    max_v = max(by_cat.values()) if by_cat else 1.0
    for cat, total in items:
        print(f"  {cat:<22} {total*100:6.2f}%  {_bar(total, max_v)}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect what features a trained NFL model relies on.",
    )
    parser.add_argument(
        "--version", default=None,
        help="Model version to inspect (v1, v2, or v3). "
             "Defaults to settings.active_model_version.",
    )
    parser.add_argument(
        "--permutation", action="store_true",
        help="Also compute permutation importance on the holdout set. "
             "Slower (rebuilds features) but more robust.",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of shuffles per feature for permutation importance "
             "(default 5; more = more stable estimates).",
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

    n_feats = len(model.feature_columns)
    print(f"\nLoaded {version} model — {n_feats} features.")
    if model.metrics:
        m = model.metrics
        print(f"Holdout: accuracy={m.get('accuracy', 0):.4f}  "
              f"ROC AUC={m.get('roc_auc', 0):.4f}  "
              f"log_loss={m.get('log_loss', 0):.4f}  "
              f"brier={m.get('brier', 0):.4f}")

    # 1) Native gain importance — always run, it's free.
    native = get_xgb_importance_avg(model, importance_type="gain")
    print_native_chart(native, version)
    print_category_summary(native, "gain-based")

    # 2) Permutation importance — opt-in.
    if args.permutation:
        log.info("Rebuilding holdout features (this can take a few minutes for v3) …")
        try:
            X_test_s, y_test, _ = rebuild_holdout(version, model)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not rebuild holdout: %s", exc)
            return 1
        perm_results = compute_permutation_importance(
            model, X_test_s, y_test, n_repeats=args.repeats,
        )
        print_permutation_chart(perm_results, version)
        # Re-shape for category roll-up
        perm_for_cat = [(feat, acc) for feat, acc, _, _ in perm_results]
        print_category_summary(perm_for_cat, "permutation Δaccuracy")
    else:
        print("Run with --permutation for the slower but more robust permutation view.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
