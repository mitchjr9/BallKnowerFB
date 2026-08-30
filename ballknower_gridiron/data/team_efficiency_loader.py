"""
ballknower_gridiron.data.team_efficiency_loader
===============================================

Per-(team, season, game_date) NFL team efficiency metrics, derived
from nflverse play-by-play. **No leakage**: for any game on date D in
season S, a team's metrics are computed from plays in games that
occurred STRICTLY BEFORE D in season S.

This replaces the older season-aggregate API (which used the whole
season's PBP to compute a single per-team metric — fine for
end-of-season analytics, but leaky when used as a training feature
for in-season games).

Public API
----------
    load_pbp_for_seasons(seasons)
        Pull raw PBP from nflverse (cached on disk via nflreadpy).

    compute_rolling_team_metrics(pbp, games)
        Return a {(team, season, game_date) -> metrics_dict} lookup.
        Each value is the team's *pre-game* expanding-window metrics:
        EPA-per-play, net points per game, plays per game, plus the
        games_played counter so callers can blend with prior season.

    compute_end_of_season_team_metrics(rolling_table)
        Return {season -> {team -> metrics}} reflecting each team's
        FINAL state at the end of each season. Used as the "prior
        season" baseline for early-week games.

    blend_rolling_with_prior(current, prior, games_played, settings)
        Linear interpolation: low games_played → mostly prior; once
        games_played reaches settings.blend_end_games → mostly current.

    get_default_team_metrics()
        Neutral defaults for teams with no data at all.

Computation strategy
--------------------
1. Aggregate PBP into one row per (game_id, team) with per-game
   counters (offensive plays, off-EPA sum, defensive plays, def-EPA
   sum, points scored, points allowed).
2. Join the team-game rows onto the schedule to get `game_date`.
3. Sort within each (team, season) by game_date and compute
   shifted-cumulative sums — so row N contains the cumulative through
   row N-1, which represents the team's "pre-game" state at row N's
   date. This is one vectorized pandas pass; total cost is dominated
   by the PBP pull, not the aggregation.

Caching
-------
PBP is huge (~50k rows × ~370 columns per season). This module writes
each downloaded season to `<data_dir>/pbp_cache/pbp_<season>.parquet`,
so retrains never re-download already-cached seasons. Downloads also
retry up to 4 times with exponential backoff (2s/4s/8s) to handle
transient GitHub 502 errors that occasionally interrupt nflverse pulls
mid-fetch. Per-season caching is critical because nflverse's bulk-load
will fail the entire call on a single bad season; with our cache, a
502 in season 13 of 13 only loses *that one* season, recovered on the
next run.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# Lookup keys are (team, season, game_date). Use date_cls (not pd.Timestamp)
# so JSON serialization is straightforward.
RollingKey = Tuple[str, int, date_cls]
TeamMetrics = Dict[str, float]


# Neutral fallbacks for teams with no data at all (e.g., very first
# season in training history with no prior).
_DEFAULTS: TeamMetrics = {
    "pts_for_pg":      21.0,
    "pts_against_pg":  21.0,
    "net_pts_pg":      0.0,
    "off_epa_per_play": 0.0,
    "def_epa_per_play": 0.0,
    "net_epa_per_play": 0.0,
    "plays_per_game":  62.0,
    "games_played":    0.0,
}


def get_default_team_metrics() -> TeamMetrics:
    """Neutral defaults for a team with no data."""
    return dict(_DEFAULTS)


# ---------------------------------------------------------------------------
# Raw PBP loading — with per-season parquet caching + retry on transient
# GitHub 502 / connection errors. nflverse PBP downloads occasionally
# fail mid-bulk-fetch, which used to blow up training runs. The cache
# layer here means a successful download is permanent (no re-fetch on
# subsequent runs) and retries handle GitHub's flaky CDN.
# ---------------------------------------------------------------------------
def _pbp_cache_dir() -> Path:
    """Where per-season cached PBP parquet files live."""
    p = settings.data_dir / "pbp_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_one_season_pbp(season: int, max_attempts: int = 4) -> pd.DataFrame:
    """
    Download one season of PBP from nflverse with exponential-backoff
    retry. Handles the most common transient failure: GitHub 502 Bad
    Gateway on the parquet download URL.

    Backoff schedule: 2s, 4s, 8s between attempts (so 4 attempts = up
    to ~14s total wait). After max_attempts, raises with a useful
    message that distinguishes 'transient infra issue' from 'permanent
    data issue'.
    """
    import time
    try:
        import nflreadpy as nfl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            df_pl = nfl.load_pbp(seasons=[season])
            df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl
            log.info("Downloaded season %d PBP: %s rows (attempt %d/%d).",
                     season, f"{len(df):,}", attempt, max_attempts)
            return df
        except (ConnectionError, OSError, Exception) as exc:  # noqa: BLE001
            # We catch broadly because nflreadpy wraps urllib/requests errors
            # in its own exception types. Anything network-shaped → retry.
            # If the exception is clearly NOT network-shaped (e.g., ImportError),
            # the broad except still catches it but the message helps the user.
            last_err = exc
            if attempt < max_attempts:
                delay = 2.0 * (2 ** (attempt - 1))  # 2s, 4s, 8s
                log.warning(
                    "Season %d PBP download attempt %d/%d failed: %s. "
                    "Retrying in %.0fs …",
                    season, attempt, max_attempts, str(exc).splitlines()[0], delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "Season %d PBP download FINAL FAILURE after %d attempts: %s",
                    season, max_attempts, exc,
                )

    raise RuntimeError(
        f"Failed to download PBP for season {season} after {max_attempts} attempts. "
        f"This is almost always a transient nflverse/GitHub CDN issue (502 Bad "
        f"Gateway). Wait a minute and retry, or check https://github.com/nflverse/"
        f"nflverse-data/releases to verify the parquet is reachable. "
        f"Last error: {last_err}"
    ) from last_err


def load_pbp_for_seasons(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load nflverse PBP for one or more seasons, with a robust local
    parquet cache and per-season retry.

    Strategy
    --------
    For each requested season:
      - If `<data_dir>/pbp_cache/pbp_<season>.parquet` exists and
        `force_refresh=False`, read from disk.
      - Otherwise, download THAT season alone with up to 4 retries
        (exponential backoff). Save to the cache on success.

    Per-season caching matters because nfl.load_pbp(seasons=[a,b,c,...])
    fails the whole call on a single bad season — meaning a 502 in the
    13th of 13 seasons throws away the first 12 successful downloads.
    Cache + per-season retry recovers cleanly on the next run.

    Parameters
    ----------
    seasons
        Seasons to load (any order; output is concatenated in input order).
    force_refresh
        If True, re-download all seasons even if cached.
    """
    if not seasons:
        return pd.DataFrame()

    cache_dir = _pbp_cache_dir()
    log.info("Loading PBP for %d seasons %s (cache: %s) …",
             len(seasons), seasons, cache_dir)

    dfs: List[pd.DataFrame] = []
    n_from_cache = 0
    n_downloaded = 0
    for season in seasons:
        cache_path = cache_dir / f"pbp_{season}.parquet"
        if not force_refresh and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                dfs.append(df)
                n_from_cache += 1
                continue
            except Exception as exc:  # noqa: BLE001 — corrupt cache → redownload
                log.warning(
                    "Cached PBP for season %d unreadable (%s) — re-downloading.",
                    season, exc,
                )
        df = _download_one_season_pbp(season)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:  # noqa: BLE001 — cache write failure is non-fatal
            log.warning(
                "Couldn't write PBP cache for season %d (%s) — continuing.",
                season, exc,
            )
        dfs.append(df)
        n_downloaded += 1

    combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    log.info("Loaded %s PBP rows. (cache hits: %d, downloads: %d)",
             f"{len(combined):,}", n_from_cache, n_downloaded)
    return combined


# ---------------------------------------------------------------------------
# Per-game team aggregation (the building block for rolling)
# ---------------------------------------------------------------------------
def _per_game_team_aggregates(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw PBP into one row per (game_id, team) with the per-game
    counters we'll later cumsum:

        off_epa_sum, off_plays_count,
        def_epa_sum, def_plays_count,
        pts_for, pts_against

    "Plays" = passing or rushing attempts only (special teams excluded
    from EPA because their EPA distribution is very different and
    would bias the per-play metric).
    """
    if pbp.empty:
        return pd.DataFrame()

    needed = {"posteam", "defteam", "home_team", "away_team", "game_id", "epa"}
    missing = needed - set(pbp.columns)
    if missing:
        log.warning("PBP missing columns %s — efficiency calc may be degraded.", missing)

    # Filter to actual plays (pass + rush attempts with non-null EPA)
    actual_play = pd.Series(False, index=pbp.index)
    if "pass_attempt" in pbp.columns:
        actual_play |= pbp["pass_attempt"].fillna(0).astype(int) == 1
    if "rush_attempt" in pbp.columns:
        actual_play |= pbp["rush_attempt"].fillna(0).astype(int) == 1
    if not actual_play.any() and "play_type" in pbp.columns:
        actual_play = pbp["play_type"].isin(["pass", "run"])
    plays = pbp[actual_play & pbp["epa"].notna()].copy()
    if plays.empty:
        return pd.DataFrame()

    # Offensive aggregates: group by (game_id, posteam)
    off = (
        plays.groupby(["game_id", "posteam"])
        .agg(off_epa_sum=("epa", "sum"), off_plays_count=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defensive aggregates: group by (game_id, defteam)
    deff = (
        plays.groupby(["game_id", "defteam"])
        .agg(def_epa_sum=("epa", "sum"), def_plays_count=("epa", "size"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    merged = off.merge(deff, on=["game_id", "team"], how="outer")
    for col in ("off_epa_sum", "off_plays_count", "def_epa_sum", "def_plays_count"):
        merged[col] = merged[col].fillna(0.0)

    # Points per (game_id, team) — extract from the last PBP row of each
    # game and assign to both home & away teams.
    pts_rows: List[Dict] = []
    if {"total_home_score", "total_away_score"}.issubset(pbp.columns):
        sort_col = "play_id" if "play_id" in pbp.columns else "epa"
        last_per_game = (
            pbp.sort_values(["game_id", sort_col]).groupby("game_id").tail(1)
        )
        for r in last_per_game.itertuples(index=False):
            ht, at = getattr(r, "home_team", None), getattr(r, "away_team", None)
            hs, asc = getattr(r, "total_home_score", None), getattr(r, "total_away_score", None)
            if pd.isna(hs) or pd.isna(asc):
                continue
            gid = getattr(r, "game_id", None)
            if ht is not None:
                pts_rows.append({"game_id": gid, "team": ht,
                                 "pts_for": float(hs), "pts_against": float(asc)})
            if at is not None:
                pts_rows.append({"game_id": gid, "team": at,
                                 "pts_for": float(asc), "pts_against": float(hs)})
    pts_df = pd.DataFrame(pts_rows)
    if not pts_df.empty:
        merged = merged.merge(pts_df, on=["game_id", "team"], how="left")
    else:
        merged["pts_for"] = 0.0
        merged["pts_against"] = 0.0
    merged["pts_for"] = merged["pts_for"].fillna(0.0)
    merged["pts_against"] = merged["pts_against"].fillna(0.0)

    return merged.dropna(subset=["team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling lookup table
# ---------------------------------------------------------------------------
def compute_rolling_team_metrics(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
) -> Dict[RollingKey, TeamMetrics]:
    """
    Return {(team, season, game_date) -> metrics} where each value is
    the team's expanding-window metrics from games BEFORE that date in
    the same season.

    Keys correspond to every game in `games` that has a matching team
    in PBP. Early-season games (where the team has 0 prior games this
    season) return defaults with games_played=0 — the caller is
    responsible for blending with prior season.
    """
    if pbp.empty or games.empty:
        return {}

    per_game = _per_game_team_aggregates(pbp)
    if per_game.empty:
        return {}

    # Need game_date + season per game_id. Source from the schedule.
    if not {"game_id", "game_date", "season"}.issubset(games.columns):
        log.warning("games table is missing game_id/game_date/season — "
                    "can't build rolling table.")
        return {}

    schedule_cols = games[["game_id", "game_date", "season"]].copy()
    schedule_cols["game_date"] = pd.to_datetime(
        schedule_cols["game_date"], errors="coerce"
    )

    df = per_game.merge(schedule_cols, on="game_id", how="inner")
    df = df.dropna(subset=["game_date", "season", "team"])
    df["season"] = df["season"].astype(int)

    # Sort chronologically within (team, season).
    df = df.sort_values(["team", "season", "game_date"]).reset_index(drop=True)

    # Cumulative-through-this-row, then shift by 1 to get cumulative-BEFORE.
    grp = df.groupby(["team", "season"], sort=False)
    for col in ("off_epa_sum", "off_plays_count",
                "def_epa_sum", "def_plays_count",
                "pts_for", "pts_against"):
        df[f"cum_{col}_pre"] = grp[col].cumsum().shift(1)

    # Games-played-before counter (0 for the first game of each team-season)
    df["games_played_before"] = grp.cumcount()

    # First row of each team-season has NaN for cum_* (we shifted past the
    # group boundary). Fill with 0.
    first_mask = df["games_played_before"] == 0
    cum_cols = [c for c in df.columns if c.startswith("cum_")]
    df.loc[first_mask, cum_cols] = 0.0

    # Compute pre-game metrics from cumulative counters.
    off_plays = df["cum_off_plays_count_pre"].replace(0, np.nan)
    def_plays = df["cum_def_plays_count_pre"].replace(0, np.nan)
    games_safe = df["games_played_before"].replace(0, np.nan)
    df["off_epa_per_play"] = (df["cum_off_epa_sum_pre"] / off_plays).fillna(0.0)
    df["def_epa_per_play"] = (df["cum_def_epa_sum_pre"] / def_plays).fillna(0.0)
    df["net_epa_per_play"] = df["off_epa_per_play"] - df["def_epa_per_play"]
    df["pts_for_pg"] = (df["cum_pts_for_pre"] / games_safe).fillna(_DEFAULTS["pts_for_pg"])
    df["pts_against_pg"] = (df["cum_pts_against_pre"] / games_safe).fillna(_DEFAULTS["pts_against_pg"])
    df["net_pts_pg"] = df["pts_for_pg"] - df["pts_against_pg"]
    df["plays_per_game"] = (df["cum_off_plays_count_pre"] / games_safe).fillna(_DEFAULTS["plays_per_game"])

    # Build the dict.
    rolling: Dict[RollingKey, TeamMetrics] = {}
    for r in df.itertuples(index=False):
        gd = r.game_date
        if hasattr(gd, "date"):
            gd = gd.date()
        key = (str(r.team), int(r.season), gd)
        rolling[key] = {
            "pts_for_pg":       float(r.pts_for_pg),
            "pts_against_pg":   float(r.pts_against_pg),
            "net_pts_pg":       float(r.net_pts_pg),
            "off_epa_per_play": float(r.off_epa_per_play),
            "def_epa_per_play": float(r.def_epa_per_play),
            "net_epa_per_play": float(r.net_epa_per_play),
            "plays_per_game":   float(r.plays_per_game),
            "games_played":     float(r.games_played_before),
        }
    log.info("Built rolling team-metrics table: %s entries across %s teams.",
             f"{len(rolling):,}", df["team"].nunique())
    return rolling


# ---------------------------------------------------------------------------
# End-of-season snapshot (prior-season baseline for early weeks)
# ---------------------------------------------------------------------------
def compute_end_of_season_team_metrics(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
) -> Dict[int, Dict[str, TeamMetrics]]:
    """
    Return {season -> {team -> final_metrics}} reflecting each team's
    END-OF-SEASON state. This is what gets used as the "prior season"
    fallback for early-week games next year.
    """
    if pbp.empty or games.empty:
        return {}

    per_game = _per_game_team_aggregates(pbp)
    if per_game.empty:
        return {}

    schedule_cols = games[["game_id", "game_date", "season"]].copy()
    schedule_cols["game_date"] = pd.to_datetime(
        schedule_cols["game_date"], errors="coerce"
    )
    df = per_game.merge(schedule_cols, on="game_id", how="inner")
    df = df.dropna(subset=["game_date", "season", "team"])
    df["season"] = df["season"].astype(int)

    # Aggregate ENTIRE season per (team, season) — this IS the season-aggregate
    # we used to use, but now it's correctly reserved for prior-season anchoring.
    agg = (
        df.groupby(["team", "season"])
        .agg(
            off_epa_sum=("off_epa_sum", "sum"),
            off_plays_count=("off_plays_count", "sum"),
            def_epa_sum=("def_epa_sum", "sum"),
            def_plays_count=("def_plays_count", "sum"),
            pts_for=("pts_for", "sum"),
            pts_against=("pts_against", "sum"),
            games_played=("game_id", "nunique"),
        )
        .reset_index()
    )

    off_plays = agg["off_plays_count"].replace(0, np.nan)
    def_plays = agg["def_plays_count"].replace(0, np.nan)
    games_safe = agg["games_played"].replace(0, np.nan)
    agg["off_epa_per_play"] = (agg["off_epa_sum"] / off_plays).fillna(0.0)
    agg["def_epa_per_play"] = (agg["def_epa_sum"] / def_plays).fillna(0.0)
    agg["net_epa_per_play"] = agg["off_epa_per_play"] - agg["def_epa_per_play"]
    agg["pts_for_pg"] = (agg["pts_for"] / games_safe).fillna(_DEFAULTS["pts_for_pg"])
    agg["pts_against_pg"] = (agg["pts_against"] / games_safe).fillna(_DEFAULTS["pts_against_pg"])
    agg["net_pts_pg"] = agg["pts_for_pg"] - agg["pts_against_pg"]
    agg["plays_per_game"] = (agg["off_plays_count"] / games_safe).fillna(_DEFAULTS["plays_per_game"])

    out: Dict[int, Dict[str, TeamMetrics]] = {}
    for r in agg.itertuples(index=False):
        s = int(r.season)
        out.setdefault(s, {})[str(r.team)] = {
            "pts_for_pg":       float(r.pts_for_pg),
            "pts_against_pg":   float(r.pts_against_pg),
            "net_pts_pg":       float(r.net_pts_pg),
            "off_epa_per_play": float(r.off_epa_per_play),
            "def_epa_per_play": float(r.def_epa_per_play),
            "net_epa_per_play": float(r.net_epa_per_play),
            "plays_per_game":   float(r.plays_per_game),
            "games_played":     float(r.games_played),
        }
    return out


# ---------------------------------------------------------------------------
# Blend helper — called at feature-emission time
# ---------------------------------------------------------------------------
def blend_rolling_with_prior(
    current: Optional[TeamMetrics],
    prior: Optional[TeamMetrics],
    games_played: float,
) -> TeamMetrics:
    """
    Linear blend of current-rolling and prior-season metrics, weighted
    by games_played. Reproduces the original "weeks 1-2 prior, 3-4
    blend, 5+ current" intent but using games-played so it handles
    byes correctly.

        games_played <= blend_start_games  -> 100% prior
        games_played >= blend_end_games    -> 100% current (rolling)
        in between                         -> linear interpolation

    With defaults (1, 4): games_played 0,1 -> all prior;
    games_played 2 -> 1/3 current; games_played 3 -> 2/3 current;
    games_played 4+ -> all current.
    """
    defaults = get_default_team_metrics()
    if not current and not prior:
        return defaults
    if not current:
        return {**defaults, **(prior or {})}
    if not prior:
        return {**defaults, **current}

    start = settings.blend_start_games
    end = settings.blend_end_games

    if games_played <= start:
        return {**defaults, **prior}
    if games_played >= end:
        return {**defaults, **current}

    denom = max(1, (end - start))
    t = (games_played - start) / denom  # 0..1
    blended: TeamMetrics = {}
    for k, dv in defaults.items():
        pv = float(prior.get(k, dv))
        cv = float(current.get(k, dv))
        blended[k] = (1.0 - t) * pv + t * cv
    return blended


# ---------------------------------------------------------------------------
# Convenience: latest snapshot per team for inference-time bundle
# ---------------------------------------------------------------------------
def latest_team_metrics_snapshot(
    rolling: Dict[RollingKey, TeamMetrics],
    end_of_season: Dict[int, Dict[str, TeamMetrics]],
    as_of_season: int,
) -> Dict[str, TeamMetrics]:
    """
    For each team, return the most recent rolling state IN `as_of_season`.
    Falls back to end_of_season[as_of_season-1] if the team has no
    games in `as_of_season` (e.g., training cutoff is mid-season and
    the team has been on bye).

    This is what we save in the trained-model bundle — used at predict
    time when we don't have rolling data for the future game's date.
    """
    by_team_latest: Dict[str, Tuple[date_cls, TeamMetrics]] = {}
    for (team, season, gd), m in rolling.items():
        if season != as_of_season:
            continue
        if team not in by_team_latest or gd > by_team_latest[team][0]:
            by_team_latest[team] = (gd, m)

    out: Dict[str, TeamMetrics] = {}
    for team, (_, m) in by_team_latest.items():
        out[team] = dict(m)

    # Backfill missing teams from prior season's end-of-season state.
    prior = end_of_season.get(as_of_season - 1, {})
    for team, m in prior.items():
        out.setdefault(team, dict(m))
    return out
