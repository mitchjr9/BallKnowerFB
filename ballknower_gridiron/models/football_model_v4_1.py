"""
ballknower_gridiron.models.football_model_v4_1
==============================================

v4.1 = v4 with the same five dead features pruned that produced v3.1.

Why
---
Per the user request, v4.1 applies v3.1's pruning logic to v4. We don't
re-derive the pruning set from v4's own permutation importance (we'd
need to train v4 first and run importance on it); instead, we assume
the same features that were dead in v3 are still dead in v4. Once v4 is
trained, running `feature_importance --version v4 --permutation` will
verify this and may reveal a different pruning candidate set worth
trying as a future v4.2.

Pruned (same as v3.1)
---------------------
  * qb_healthy_diff
  * short_week_diff
  * qb_rating_diff
  * form_diff_5
  * is_international

Result: 20 features instead of 25 (v4's 21 + 4 new − 5 dead).

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import pandas as pd

from ballknower_gridiron.models.football_model_v3_1 import PRUNED_FEATURES
from ballknower_gridiron.models.football_model_v4 import (
    FEATURE_COLUMNS_V4,
    NFLModelV4,
    train_nfl_model_v4,
)


# Apply v3.1's pruning set to v4's full feature list, preserving order.
FEATURE_COLUMNS_V4_1 = [
    f for f in FEATURE_COLUMNS_V4 if f not in PRUNED_FEATURES
]


def train_nfl_model_v4_1(
    games: pd.DataFrame,
    force_refresh_qb_stats: bool = False,
    force_refresh_team_metrics: bool = False,
) -> NFLModelV4:
    """Train v4.1 — v4 pipeline, leaner feature list.

    Returns an NFLModelV4 instance whose `feature_columns` has 20 entries
    (instead of v4's 25) and whose `version` field is "v4.1".
    """
    return train_nfl_model_v4(
        games=games,
        force_refresh_qb_stats=force_refresh_qb_stats,
        force_refresh_team_metrics=force_refresh_team_metrics,
        feature_columns=FEATURE_COLUMNS_V4_1,
        version_label="v4.1",
    )
