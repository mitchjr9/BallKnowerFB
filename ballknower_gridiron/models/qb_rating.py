"""
ballknower_gridiron.models.qb_rating
====================================

NFL QB composite rating — the football analog of basketball's
`star_player.py`, but built around the QB position since QBs dominate
NFL outcomes more than any single position in any other sport.

The composite
-------------
For each qualified QB (≥ NFL_QB_MIN_GAMES games AND
≥ NFL_QB_MIN_ATT_PG attempts/game per `settings`), every stat below
is z-scored across all qualified QBs in that season, then summed with
the weights below to produce the QB's rating:

    +2.0 × z(pass_ypg)         Passing yards per game
                                 (250+ YPG is starter-level, per user spec)
    +2.5 × z(pass_tds_pg)      Passing TDs per game
                                 (2.0+ is the elite threshold)
    +1.5 × z(rush_ypg)         Rushing yards per game
                                 (25+ YPG flags dual-threat QBs)
    +1.5 × z(rush_tds_pg)      Rushing TDs per game
    -2.0 × z(ints_pg)          Interceptions per game  (negative)
    -1.0 × z(fumbles_lost_pg)  Fumbles lost per game   (negative)
    +1.5 × z(completion_pct)   Completion %
                                 (62%+ is the modern starter floor)
    +1.5 × z(passer_rating)    NFL passer rating (1973 formula, max 158.3)
                                 (90+ is solid; 100+ is excellent)
    +2.0 × z(espn_qbr)         ESPN's Total Quarterback Rating (0-100)
                                 (60+ is good; 70+ is elite)

Why BOTH passer rating and ESPN QBR
-----------------------------------
They measure related but distinct things:
  * Passer rating is volume-friendly and equally weights TDs, INTs,
    yards, and completion %. It does not penalize sacks or credit
    rushing.
  * ESPN QBR opponent-adjusts EVERY play, down-weights garbage time,
    and credits rushing/penalty plays the QB is responsible for. It's
    a more modern, more predictive metric — but it's proprietary and
    sometimes the source is slow to update.

Keeping both means: when QBR is available, the model leans on it
(weight 2.0); when it isn't (a season's data hasn't been scraped yet,
or the source is down), the passer-rating z-score still carries
useful signal. They're correlated (~0.7) so they don't double-count
— they correct each other on the margins.

Weights are biased toward TDs and turnovers because those map most
directly to scoring/winning — they're the "big plays" of the position.
Pure YPG matters less than YPG-per-attempt would, but per-game stats
are what the user listed explicitly so we honor that.

Team-level aggregation
----------------------
For each team in a given season, identify the QB who threw the most
attempts ("the starter") and use his rating. If a starter is flagged
as out (the `--qb-out` toggle in `predict_nfl.py`), the backup's
rating is used minus `settings.backup_qb_penalty`.

This is intentionally less smooth than basketball's "top-3 average":
NFL games run almost entirely through one player at the QB position,
so a single-QB rating is the right abstraction. The backup penalty
captures system-fit uncertainty that a pure rating swap misses.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# Stat -> weight in composite (positive = more is better; negative = penalty).
# Passer rating + ESPN QBR are both included; total positive "QB efficiency"
# weight (passer_rating 1.5 + espn_qbr 2.0 = 3.5) > old single-passer-rating
# (2.0), reflecting that we now have two complementary efficiency signals.
QB_STAT_WEIGHTS: Dict[str, float] = {
    "pass_ypg":        2.0,
    "pass_tds_pg":     2.5,
    "rush_ypg":        1.5,
    "rush_tds_pg":     1.5,
    "ints_pg":        -2.0,
    "fumbles_lost_pg":-1.0,
    "completion_pct":  1.5,
    "passer_rating":   1.5,   # NFL 1973 formula (max 158.3)
    "espn_qbr":        2.0,   # ESPN QBR (0-100, opponent-adjusted)
}


def _z_score(s: pd.Series) -> pd.Series:
    """Z-score a series; return zeros if variance is degenerate."""
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def compute_qb_ratings(qb_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `qb_rating` column to a QB stats DataFrame (output of
    `load_qb_season_stats`).

    Qualified QBs (per `settings.qb_min_games` + min attempts/game) get
    a real composite; non-qualified QBs get 0.0 so they don't affect
    team-level aggregation when picked up as backups.
    """
    df = qb_stats.copy()
    if df.empty:
        df["qb_rating"] = []
        return df

    qualified = df[df.get("qualified", True) == True].copy()  # noqa: E712
    df["qb_rating"] = 0.0
    if len(qualified) < 5:
        log.warning("Only %d qualified QBs — skipping QB rating computation.",
                    len(qualified))
        return df

    rating = pd.Series(np.zeros(len(qualified)), index=qualified.index)
    for stat, weight in QB_STAT_WEIGHTS.items():
        if stat in qualified.columns:
            z = _z_score(qualified[stat].fillna(qualified[stat].median()))
            rating = rating + z * weight

    df.loc[qualified.index, "qb_rating"] = rating
    return df


def compute_team_qb_ratings(
    qb_stats: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    For each team, identify "the starting QB" as the one with the most
    pass attempts on that team for the season, then return that QB's
    rating as the team's QB rating.

    Returns
    -------
    (team_ratings, team_starter_names)
      team_ratings : {team_abbr -> qb_rating}
      team_starter_names : {team_abbr -> player_display_name}
        Display-only — used by predict_nfl.py output and newsletter copy.
    """
    if qb_stats.empty:
        return {}, {}

    rated = compute_qb_ratings(qb_stats)

    # Pick the most-attempts QB per team as the starter.
    team_ratings: Dict[str, float] = {}
    team_starters: Dict[str, str] = {}
    # Group by `team` (nflverse player_stats uses `team`, sometimes `recent_team`)
    team_col = "team" if "team" in rated.columns else "recent_team"
    if team_col not in rated.columns:
        log.warning("QB stats have no team column — can't aggregate to team level.")
        return {}, {}

    name_col = "player_display_name" if "player_display_name" in rated.columns else "player_name"
    if name_col not in rated.columns:
        name_col = "player_id"

    for team, group in rated.groupby(team_col):
        starter = group.sort_values("attempts", ascending=False).iloc[0]
        team_ratings[str(team)] = float(starter["qb_rating"])
        team_starters[str(team)] = str(starter.get(name_col, "Unknown"))

    return team_ratings, team_starters


def get_qb_rating_for_player(
    qb_stats: pd.DataFrame,
    player_id: Optional[str] = None,
    player_name: Optional[str] = None,
) -> Optional[float]:
    """
    Look up an individual QB's rating from a rated DataFrame. Used by
    the backup-QB substitution logic in `predict_nfl.py --qb-out`.

    Match by player_id if provided; otherwise by case-insensitive
    player name. Returns None if not found.
    """
    if qb_stats.empty:
        return None
    rated = compute_qb_ratings(qb_stats)

    if player_id is not None and "player_id" in rated.columns:
        m = rated[rated["player_id"].astype(str) == str(player_id)]
        if not m.empty:
            return float(m.iloc[0]["qb_rating"])

    if player_name is not None:
        name_col = "player_display_name" if "player_display_name" in rated.columns \
            else ("player_name" if "player_name" in rated.columns else None)
        if name_col is not None:
            target = player_name.strip().lower()
            m = rated[rated[name_col].astype(str).str.strip().str.lower() == target]
            if not m.empty:
                return float(m.iloc[0]["qb_rating"])
    return None


def summarize_top_qbs(qb_stats: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Diagnostic helper: top N QBs league-wide by rating."""
    rated = compute_qb_ratings(qb_stats)
    cols = ["player_display_name", "team", "games_played",
            "pass_ypg", "pass_tds_pg", "rush_ypg", "ints_pg",
            "completion_pct", "passer_rating", "espn_qbr", "qb_rating"]
    cols = [c for c in cols if c in rated.columns]
    return rated.nlargest(n, "qb_rating")[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference-time team rating with QB-healthy override
# ---------------------------------------------------------------------------
def team_qb_rating_with_override(
    team: str,
    team_ratings: Dict[str, float],
    *,
    starter_out: bool = False,
    backup_rating: Optional[float] = None,
) -> float:
    """
    Resolve a team's effective QB rating given an optional "starter out"
    override.

    Parameters
    ----------
    team : team abbreviation.
    team_ratings : the {team -> qb_rating} dict produced at train time
                   (snapshot of the starting QB per team that season).
    starter_out : True if the user has toggled the starter as out.
    backup_rating : the backup QB's rating if known. If None and
                    `starter_out` is True, we apply a generic "backup
                    penalty" relative to the starter (handles the case
                    where we don't know who the backup is yet).

    Returns
    -------
    Effective QB rating to feed into the v2 model.
    """
    starter_r = team_ratings.get(team, 0.0)
    if not starter_out:
        return starter_r
    # Starter is OUT.
    if backup_rating is not None:
        # Backup is known — use their rating but still apply the system-fit
        # penalty (a backup playing for a new team mid-season is less
        # effective than their isolated rating implies).
        return backup_rating - settings.backup_qb_penalty
    # Unknown backup — apply a heuristic penalty relative to the starter.
    # Empirically (per FiveThirtyEight's QB Elo model), starter → backup
    # ELO drops are typically 30–60 points; in QB-rating-z-score space
    # that's roughly a 2.5–3 drop. We use 2.5 as the default penalty
    # (settings.backup_qb_penalty defaults to 1.0, applied on top of the
    # named "starter - 2.5" anchor).
    return starter_r - (2.5 + settings.backup_qb_penalty)


# ===========================================================================
# ROLLING (leakage-free) QB rating pipeline
# ===========================================================================
"""
The functions below replace the season-aggregate `compute_team_qb_ratings`
with per-(team, season, game_date) ratings computed from weekly stats
strictly BEFORE the game date.

Z-score reference distribution
------------------------------
For each season S, we z-score against season (S-1)'s end-of-season
qualified-QB distribution. For the very first season in training we use
that season's own distribution (one-time concession — slight leak but
only affects the rating scale, not relative rankings).

Starter identification with mid-season changes
----------------------------------------------
At each game date, the team's "starter" is the QB with the most
cumulative passing ATTEMPTS on that team SO FAR THIS SEASON. This
handles benchings/injuries naturally: if a Week 1-9 starter is benched
in Week 10, by Week 12-13 the backup has more attempts and becomes
"the starter" in the model.
"""
from datetime import date as date_cls
from typing import Tuple as _Tuple


def _cumulative_qb_stats(weekly_qb: pd.DataFrame) -> pd.DataFrame:
    """
    Take weekly QB stats and return per-(player, team, season, week) rows
    with PRE-GAME cumulative counters. Each row's `cum_*_pre` values
    reflect the player's totals through games BEFORE that week.
    """
    if weekly_qb.empty:
        return pd.DataFrame()

    df = weekly_qb.copy()
    df = df.sort_values(["player_id", "team", "season", "week"]).reset_index(drop=True)
    grp = df.groupby(["player_id", "team", "season"], sort=False)

    raw_cols = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered",
        "carries", "rushing_yards", "rushing_tds", "fumbles_lost",
    ]
    for c in raw_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[f"cum_{c}_pre"] = grp[c].cumsum().shift(1)

    # Weekly espn_qbr — we cumulate QB-plays-weighted average via
    # numerator (qbr_total * qb_plays) and denominator (qb_plays).
    if "espn_qbr" in df.columns and "qb_plays" in df.columns:
        df["_qbr_num_week"] = pd.to_numeric(df["espn_qbr"], errors="coerce") * \
                              pd.to_numeric(df["qb_plays"], errors="coerce")
        df["_qbr_den_week"] = pd.to_numeric(df["qb_plays"], errors="coerce")
        df["cum_qbr_num_pre"] = grp["_qbr_num_week"].cumsum().shift(1)
        df["cum_qbr_den_pre"] = grp["_qbr_den_week"].cumsum().shift(1)

    df["games_played_pre"] = grp.cumcount()  # 0 for first week of each (qb, team, season)

    # First row of each group has NaN for cum_* (shift past boundary). Fill 0.
    first_mask = df["games_played_pre"] == 0
    for c in [c for c in df.columns if c.startswith("cum_")]:
        df.loc[first_mask, c] = 0.0
    return df


def _scaler_from_season(
    weekly_qb: pd.DataFrame,
    season: int,
    settings_obj,
) -> Dict[str, _Tuple[float, float]]:
    """
    Compute the league-wide (mu, sigma) per stat from end-of-season
    aggregates of qualified QBs in `season`. Used as the z-score
    reference for the NEXT season's rolling ratings.
    """
    if weekly_qb.empty:
        return {}
    season_rows = weekly_qb[weekly_qb["season"] == season]
    if season_rows.empty:
        return {}

    # Step 1: Aggregate the basic counter stats per (player, team) — these
    # are all simple sums and don't depend on optional columns like
    # `espn_qbr` / `qb_plays`. Keeping this `.agg()` call free of
    # conditional dict items avoids the bug where a missing column inside
    # an embedded lambda raises mid-aggregation.
    basic_agg = {
        "completions":           ("completions", "sum"),
        "attempts":              ("attempts", "sum"),
        "passing_yards":         ("passing_yards", "sum"),
        "passing_tds":           ("passing_tds", "sum"),
        "passing_interceptions": ("passing_interceptions", "sum"),
        "rushing_yards":         ("rushing_yards", "sum"),
        "rushing_tds":           ("rushing_tds", "sum"),
        "fumbles_lost":          ("fumbles_lost", "sum"),
        "games_played":          ("week", "nunique"),
    }
    agg = (
        season_rows.groupby(["player_id", "team"])
        .agg(**basic_agg)
        .reset_index()
    )

    # Step 2: Handle ESPN QBR separately. Three cases:
    #   (a) both espn_qbr and qb_plays present -> properly weight QBR by plays
    #   (b) only espn_qbr present              -> simple mean of weekly QBR
    #   (c) neither present                    -> espn_qbr column = NaN
    has_qbr = "espn_qbr" in season_rows.columns
    has_qb_plays = "qb_plays" in season_rows.columns

    if has_qbr and has_qb_plays:
        srw = season_rows.copy()
        srw["_qbr_num"] = (
            pd.to_numeric(srw["espn_qbr"], errors="coerce")
            * pd.to_numeric(srw["qb_plays"], errors="coerce")
        )
        srw["_qbr_den"] = pd.to_numeric(srw["qb_plays"], errors="coerce")
        qbr_agg = (
            srw.groupby(["player_id", "team"])
            .agg(qbr_num=("_qbr_num", "sum"), qbr_den=("_qbr_den", "sum"))
            .reset_index()
        )
        agg = agg.merge(qbr_agg, on=["player_id", "team"], how="left")
        agg["espn_qbr"] = agg["qbr_num"] / agg["qbr_den"].replace(0, np.nan)
    elif has_qbr:
        # Fallback: unweighted mean of weekly QBR
        qbr_agg = (
            season_rows.groupby(["player_id", "team"])
            .agg(espn_qbr=("espn_qbr",
                           lambda s: pd.to_numeric(s, errors="coerce").mean()))
            .reset_index()
        )
        agg = agg.merge(qbr_agg, on=["player_id", "team"], how="left")
    else:
        agg["espn_qbr"] = np.nan

    games_safe = agg["games_played"].replace(0, 1)
    att_safe = agg["attempts"].replace(0, 1)
    agg["pass_ypg"]        = agg["passing_yards"] / games_safe
    agg["pass_tds_pg"]     = agg["passing_tds"] / games_safe
    agg["rush_ypg"]        = agg["rushing_yards"] / games_safe
    agg["rush_tds_pg"]     = agg["rushing_tds"] / games_safe
    agg["ints_pg"]         = agg["passing_interceptions"] / games_safe
    agg["fumbles_lost_pg"] = agg["fumbles_lost"] / games_safe
    agg["completion_pct"]  = (agg["completions"] / att_safe) * 100.0

    # Passer rating
    a = ((agg["completions"] / att_safe) - 0.3) * 5.0
    b = ((agg["passing_yards"] / att_safe) - 3.0) * 0.25
    c = (agg["passing_tds"] / att_safe) * 20.0
    d = 2.375 - ((agg["passing_interceptions"] / att_safe) * 25.0)
    a = a.clip(lower=0.0, upper=2.375)
    b = b.clip(lower=0.0, upper=2.375)
    c = c.clip(lower=0.0, upper=2.375)
    d = d.clip(lower=0.0, upper=2.375)
    agg["passer_rating"] = ((a + b + c + d) / 6.0) * 100.0

    # Qualification filter
    qualified = agg[
        (agg["games_played"] >= settings_obj.qb_min_games)
        & ((agg["attempts"] / games_safe) >= settings_obj.qb_min_attempts_per_game)
    ]
    if len(qualified) < 5:
        log.warning("[scaler] only %d qualified QBs in %s — z-scoring may be noisy.",
                    len(qualified), season)
        if len(qualified) < 2:
            return {}

    scaler: Dict[str, _Tuple[float, float]] = {}
    for stat in QB_STAT_WEIGHTS.keys():
        if stat in qualified.columns:
            vals = pd.to_numeric(qualified[stat], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma == 0 or pd.isna(sigma) or pd.isna(mu):
                continue
            scaler[stat] = (mu, sigma)
    return scaler


def _rolling_rating_from_cumulative(
    cum_row: pd.Series,
    scaler: Dict[str, _Tuple[float, float]],
) -> float:
    """
    Compute a single QB rating from a row of cumulative-pre stats
    using the supplied (mu, sigma) z-score scaler.

    `cum_row` must contain cum_*_pre fields and games_played_pre.
    """
    games = float(cum_row.get("games_played_pre", 0))
    if games <= 0:
        return 0.0
    attempts = float(cum_row.get("cum_attempts_pre", 0))
    if attempts <= 0:
        return 0.0

    # Derive per-game / per-attempt stats
    derived: Dict[str, float] = {
        "pass_ypg":         float(cum_row.get("cum_passing_yards_pre", 0)) / games,
        "pass_tds_pg":      float(cum_row.get("cum_passing_tds_pre", 0)) / games,
        "rush_ypg":         float(cum_row.get("cum_rushing_yards_pre", 0)) / games,
        "rush_tds_pg":      float(cum_row.get("cum_rushing_tds_pre", 0)) / games,
        "ints_pg":          float(cum_row.get("cum_passing_interceptions_pre", 0)) / games,
        "fumbles_lost_pg":  float(cum_row.get("cum_fumbles_lost_pre", 0)) / games,
        "completion_pct":   (float(cum_row.get("cum_completions_pre", 0)) / attempts) * 100.0,
    }
    # Passer rating from cumulative
    comp_pct = derived["completion_pct"] / 100.0  # back to fraction
    ypa = float(cum_row.get("cum_passing_yards_pre", 0)) / attempts
    tdpa = float(cum_row.get("cum_passing_tds_pre", 0)) / attempts
    intpa = float(cum_row.get("cum_passing_interceptions_pre", 0)) / attempts
    a = max(0.0, min(2.375, (comp_pct - 0.3) * 5.0))
    b = max(0.0, min(2.375, (ypa - 3.0) * 0.25))
    c = max(0.0, min(2.375, tdpa * 20.0))
    d = max(0.0, min(2.375, 2.375 - intpa * 25.0))
    derived["passer_rating"] = ((a + b + c + d) / 6.0) * 100.0
    # ESPN QBR
    qbr_num = float(cum_row.get("cum_qbr_num_pre", 0) or 0)
    qbr_den = float(cum_row.get("cum_qbr_den_pre", 0) or 0)
    if qbr_den > 0:
        derived["espn_qbr"] = qbr_num / qbr_den
    else:
        derived["espn_qbr"] = float("nan")  # let scaler/median-impute handle

    rating = 0.0
    for stat, weight in QB_STAT_WEIGHTS.items():
        if stat not in scaler:
            continue
        v = derived.get(stat, float("nan"))
        if pd.isna(v):
            continue  # missing QBR → contribute 0 (effectively median-imputed in z-space)
        mu, sigma = scaler[stat]
        if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            continue
        rating += weight * (v - mu) / sigma
    return float(rating)


def _player_full_season_ratings(
    weekly_qb: pd.DataFrame,
    self_scalers: Dict[int, Dict[str, _Tuple[float, float]]],
) -> Dict[_Tuple[str, int], float]:
    """
    For each (player_id, season), compute the player's END-OF-SEASON
    full rating using THAT SEASON'S OWN distribution as the scaler.

    The result is used as the prior anchor in the rolling blend: a QB
    who was elite last year starts THIS year's rolling rating already
    anchored to "elite," not to zero.

    We use each season's self-scaler (instead of the (season-1) scaler
    we use mid-season) because we want this rating to be interpretable
    as "how this player ranked among their peers that year." That's
    the same scale the model sees when the rolling rating is fully
    formed by mid-season, so the blend is smooth.

    Players with very few starts get filtered out — a 1-game cameo
    isn't a meaningful "this is who you are" signal.
    """
    if weekly_qb.empty:
        return {}
    out: Dict[_Tuple[str, int], float] = {}

    for season in sorted(weekly_qb["season"].astype(int).unique().tolist()):
        season_rows = weekly_qb[weekly_qb["season"] == season]
        if season_rows.empty:
            continue
        scaler = self_scalers.get(season)
        if not scaler:
            continue

        # Aggregate per-player full-season stats (sum across teams if
        # the QB was traded mid-season — we want the player's overall
        # production, not just their last team's snapshot).
        basic_agg = {
            "completions":           ("completions", "sum"),
            "attempts":              ("attempts", "sum"),
            "passing_yards":         ("passing_yards", "sum"),
            "passing_tds":           ("passing_tds", "sum"),
            "passing_interceptions": ("passing_interceptions", "sum"),
            "rushing_yards":         ("rushing_yards", "sum"),
            "rushing_tds":           ("rushing_tds", "sum"),
            "fumbles_lost":          ("fumbles_lost", "sum"),
            "games_played":          ("week", "nunique"),
        }
        agg = season_rows.groupby("player_id").agg(**basic_agg).reset_index()

        # ESPN QBR via the same three-case fallback used in _scaler_from_season.
        has_qbr = "espn_qbr" in season_rows.columns
        has_qb_plays = "qb_plays" in season_rows.columns
        if has_qbr and has_qb_plays:
            srw = season_rows.copy()
            srw["_qbr_num"] = (
                pd.to_numeric(srw["espn_qbr"], errors="coerce")
                * pd.to_numeric(srw["qb_plays"], errors="coerce")
            )
            srw["_qbr_den"] = pd.to_numeric(srw["qb_plays"], errors="coerce")
            qbr_agg = (
                srw.groupby("player_id")
                .agg(qbr_num=("_qbr_num", "sum"), qbr_den=("_qbr_den", "sum"))
                .reset_index()
            )
            agg = agg.merge(qbr_agg, on="player_id", how="left")
            agg["espn_qbr"] = agg["qbr_num"] / agg["qbr_den"].replace(0, np.nan)
        elif has_qbr:
            qbr_agg = (
                season_rows.groupby("player_id")
                .agg(espn_qbr=("espn_qbr",
                               lambda s: pd.to_numeric(s, errors="coerce").mean()))
                .reset_index()
            )
            agg = agg.merge(qbr_agg, on="player_id", how="left")
        else:
            agg["espn_qbr"] = np.nan

        # Derived rates (same shape as _scaler_from_season).
        games_safe = agg["games_played"].replace(0, 1)
        att_safe = agg["attempts"].replace(0, 1)
        agg["pass_ypg"]        = agg["passing_yards"] / games_safe
        agg["pass_tds_pg"]     = agg["passing_tds"] / games_safe
        agg["rush_ypg"]        = agg["rushing_yards"] / games_safe
        agg["rush_tds_pg"]     = agg["rushing_tds"] / games_safe
        agg["ints_pg"]         = agg["passing_interceptions"] / games_safe
        agg["fumbles_lost_pg"] = agg["fumbles_lost"] / games_safe
        agg["completion_pct"]  = (agg["completions"] / att_safe) * 100.0
        a = ((agg["completions"] / att_safe) - 0.3) * 5.0
        b = ((agg["passing_yards"] / att_safe) - 3.0) * 0.25
        c = (agg["passing_tds"] / att_safe) * 20.0
        d = 2.375 - ((agg["passing_interceptions"] / att_safe) * 25.0)
        a = a.clip(lower=0.0, upper=2.375)
        b = b.clip(lower=0.0, upper=2.375)
        c = c.clip(lower=0.0, upper=2.375)
        d = d.clip(lower=0.0, upper=2.375)
        agg["passer_rating"] = ((a + b + c + d) / 6.0) * 100.0

        # Filter to QBs with enough participation to anchor on.
        min_games_for_anchor = max(2, settings.qb_min_games // 2)
        eligible = agg[agg["games_played"] >= min_games_for_anchor]

        for r in eligible.itertuples(index=False):
            rating = 0.0
            contributed = False
            for stat, weight in QB_STAT_WEIGHTS.items():
                if stat not in scaler:
                    continue
                v_raw = getattr(r, stat, float("nan"))
                v = float(v_raw) if v_raw is not None else float("nan")
                if pd.isna(v):
                    continue
                mu, sigma = scaler[stat]
                if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
                    continue
                rating += weight * (v - mu) / sigma
                contributed = True
            if contributed:
                out[(str(r.player_id), int(season))] = float(rating)

    log.info("Computed prior anchors for %d (player, season) pairs.", len(out))
    return out


def compute_rolling_qb_ratings(
    weekly_qb: pd.DataFrame,
    games: pd.DataFrame,
) -> _Tuple[
    Dict[_Tuple[str, int, date_cls], float],
    Dict[_Tuple[str, int, date_cls], str],
]:
    """
    For each scheduled game in `games`, compute each team's pre-game
    starter QB rating using only weekly stats BEFORE that game date.

    Starter = QB with most cumulative attempts on the team up to that
    date in that season.

    Returns
    -------
    (team_ratings_by_date, team_starters_by_date)
      team_ratings_by_date  : {(team, season, game_date) -> rating}
      team_starters_by_date : {(team, season, game_date) -> player_name}
    """
    if weekly_qb.empty or games.empty:
        return {}, {}

    # 1. Build pre-game cumulative stats per (player, team, season, week).
    cum = _cumulative_qb_stats(weekly_qb)
    if cum.empty:
        return {}, {}

    # 2. Build z-score reference scalers.
    #
    # `self_scalers[s]`  : each season's OWN distribution of qualified QBs.
    #                      Used both to compute end-of-season anchor ratings
    #                      AND as the in-season reference for the NEXT season.
    # `in_season_scalers[s]` : what to use when rating QBs DURING season s.
    #                      Always (s-1)'s distribution; (s-1) absent → s.
    seasons_sorted = sorted(weekly_qb["season"].astype(int).unique().tolist())
    self_scalers: Dict[int, Dict[str, _Tuple[float, float]]] = {}
    for s in seasons_sorted:
        self_scalers[s] = _scaler_from_season(weekly_qb, s, settings)

    in_season_scalers: Dict[int, Dict[str, _Tuple[float, float]]] = {}
    for s in seasons_sorted:
        if (s - 1) in self_scalers and self_scalers[s - 1]:
            in_season_scalers[s] = self_scalers[s - 1]
        else:
            in_season_scalers[s] = self_scalers[s]

    # Prior anchors: {(player_id, season) -> end-of-season rating}.
    # A QB's "prior" for season s is their full season (s-1) rating.
    prior_anchors = _player_full_season_ratings(weekly_qb, self_scalers)

    # 3. Build a (team, season, week) -> {qb_id -> (cum_attempts_pre, cum_row, name)} map.
    # We'll use it to identify the starter for any game by mapping its
    # week back to the week-cumulatives we have.
    name_col = "player_display_name" if "player_display_name" in cum.columns else "player_name"
    if "week" not in cum.columns:
        log.warning("Weekly QB stats missing `week` column — cannot do rolling.")
        return {}, {}

    cum["week_int"] = pd.to_numeric(cum["week"], errors="coerce").astype("Int64")

    # 4. Process each game in `games`.
    out_ratings: Dict[_Tuple[str, int, date_cls], float] = {}
    out_starters: Dict[_Tuple[str, int, date_cls], str] = {}

    # Pre-sort cum by week so we can binary-search-via-mask quickly.
    cum_by_team_season = dict(list(cum.groupby(["team", "season"], sort=False)))

    blend_start = settings.blend_start_games
    blend_end = settings.blend_end_games
    blend_denom = max(1, (blend_end - blend_start))

    # Stats for logging (so user can see the blend is actually firing)
    n_blended = 0
    n_no_prior = 0

    for r in games.itertuples(index=False):
        try:
            season = int(r.season)
            wk = int(r.week)
            gd = r.game_date
            if hasattr(gd, "date"):
                gd = gd.date()
        except (AttributeError, ValueError, TypeError):
            continue

        scaler = in_season_scalers.get(season, {})

        for team in (r.home_team, r.away_team):
            team_key = (str(team), season)
            sub = cum_by_team_season.get(team_key)
            if sub is None or sub.empty:
                continue
            # Find QBs whose week_int <= wk and pick the row with greatest
            # cum_attempts_pre. (Use ≤ wk since "pre" means before THIS week's game.)
            sub_to_week = sub[sub["week_int"] <= wk]
            if sub_to_week.empty:
                continue
            # For each player_id, take the latest week we have ≤ wk.
            sub_to_week = sub_to_week.sort_values(["player_id", "week_int"]).groupby(
                "player_id", as_index=False, sort=False,
            ).tail(1)
            if sub_to_week.empty:
                continue
            # Starter = QB with max cum_attempts_pre on this team up to this week.
            starter_row = sub_to_week.sort_values("cum_attempts_pre", ascending=False).iloc[0]
            current_rating = _rolling_rating_from_cumulative(starter_row, scaler)
            games_played = float(starter_row.get("games_played_pre", 0))
            starter_pid = str(starter_row.get("player_id", ""))

            # Prior anchor = starter's end-of-prior-season rating.
            # If the starter is a rookie or wasn't a starter last year,
            # there's no anchor — fall back to current_rating as-is.
            prior_rating = prior_anchors.get((starter_pid, season - 1))

            if prior_rating is not None:
                # Same blend curve as team efficiency:
                #   games_played <= start -> all prior
                #   games_played >= end   -> all current
                #   in between            -> linear
                if games_played <= blend_start:
                    blended = prior_rating
                elif games_played >= blend_end:
                    blended = current_rating
                else:
                    t = (games_played - blend_start) / blend_denom
                    blended = (1.0 - t) * prior_rating + t * current_rating
                n_blended += 1
            else:
                blended = current_rating
                n_no_prior += 1

            out_ratings[(str(team), season, gd)] = float(blended)
            out_starters[(str(team), season, gd)] = str(starter_row.get(name_col, "Unknown"))

    log.info(
        "Computed rolling QB ratings: %s entries (%s blended with prior anchor, "
        "%s without prior — rookies/new starters).",
        f"{len(out_ratings):,}", f"{n_blended:,}", f"{n_no_prior:,}",
    )
    return out_ratings, out_starters


def latest_qb_ratings_snapshot(
    ratings_by_date: Dict[_Tuple[str, int, date_cls], float],
    starters_by_date: Dict[_Tuple[str, int, date_cls], str],
    as_of_season: int,
) -> _Tuple[Dict[str, float], Dict[str, str]]:
    """
    For each team, return the most recent rolling QB rating in
    `as_of_season`. Used to save into the trained-model bundle for
    inference time.
    """
    by_team_latest_r: Dict[str, _Tuple[date_cls, float]] = {}
    by_team_latest_s: Dict[str, _Tuple[date_cls, str]] = {}
    for (team, season, gd), rating in ratings_by_date.items():
        if season != as_of_season:
            continue
        if team not in by_team_latest_r or gd > by_team_latest_r[team][0]:
            by_team_latest_r[team] = (gd, rating)
    for (team, season, gd), name in starters_by_date.items():
        if season != as_of_season:
            continue
        if team not in by_team_latest_s or gd > by_team_latest_s[team][0]:
            by_team_latest_s[team] = (gd, name)

    out_r = {team: r for team, (_, r) in by_team_latest_r.items()}
    out_s = {team: n for team, (_, n) in by_team_latest_s.items()}
    return out_r, out_s
