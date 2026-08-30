"""
ballknower_gridiron.models.football_model_v3_1
==============================================

v3.1 = v3 with five dead features pruned from the training feature list.

Why
---
Feature-importance analysis on v3 (run via
`scripts.feature_importance --version v3 --permutation`) showed that five
features had zero or NEGATIVE permutation accuracy on the holdout —
meaning the model performed BETTER on average when those features were
shuffled to random values. That's overfitting room with no upside.

Pruned features
---------------
  * qb_healthy_diff   — constant 0 in training (no historical injury data)
  * short_week_diff   — exactly 0.00pp permutation Δacc
  * qb_rating_diff    — negative permutation Δacc (-0.24pp)
  * form_diff_5       — 0.00pp Δacc; redundant with ELO's rolling form signal
  * is_international  — slightly negative on both Δacc and ΔAUC

Result: 16 features instead of 21.

Same builder, same model class, same bundle format as v3 — the only
differences are the training feature list, the saved version string
("v3.1"), and the artifact directory (settings.models_dir_v3_1).

What's NOT changed
------------------
The feature builder still computes all 21 features (it has to, since v3
inference paths may want them). v3.1 just slices a smaller column set
when feeding the model. So you can use the v3 builder for either
version — the model itself records which columns it uses in
`feature_columns` and selects them at predict time.

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import pandas as pd

from ballknower_gridiron.models.football_model_v3 import (
    FEATURE_COLUMNS_V3,
    NFLFeatureBuilderV3,
    NFLModelV3,
    train_nfl_model_v3,
)


# The 5 features identified as dead-weight from v3 permutation importance.
PRUNED_FEATURES = frozenset({
    "qb_healthy_diff",
    "short_week_diff",
    "qb_rating_diff",
    "form_diff_5",
    "is_international",
})

# Lean training feature list — preserves V3's column ORDER for the
# survivors, which matters for downstream feature-importance comparison.
FEATURE_COLUMNS_V3_1 = [
    f for f in FEATURE_COLUMNS_V3 if f not in PRUNED_FEATURES
]


def train_nfl_model_v3_1(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
    force_refresh_team_metrics: bool = False,
) -> NFLModelV3:
    """Train v3.1 — same pipeline as v3, just a leaner feature list.

    Returns an NFLModelV3 instance (the bundle layout is identical to v3).
    The saved model's `feature_columns` attribute will have 16 entries
    instead of 21, and its `version` field will say "v3.1".
    """
    return train_nfl_model_v3(
        games=games,
        force_refresh_qb_stats=force_refresh_qb_stats,
        force_refresh_team_metrics=force_refresh_team_metrics,
        feature_columns=FEATURE_COLUMNS_V3_1,
        version_label="v3.1",
    )
