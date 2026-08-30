"""
ballknower_gridiron.data.qb_stats_loader
========================================

Pulls per-season NFL QB statistics + weekly depth charts via
`nflreadpy`. Two artifacts:

  * **QB season stats** — one row per (player_id, team, season) with the
    stats needed for the QB composite (see `models.qb_rating`):
        completions, attempts, passing_yards, passing_tds, interceptions,
        sacks_taken, sack_yards, rushing_yards, rushing_tds, fumbles_lost,
        games_played, plus derived per-game and rate stats.
  * **Weekly depth charts** — used to identify *who* the starting QB is
    on a given team for a given week, and to identify the backup if the
    starter is out (per the user's QB-healthy toggle requirement).

Why this loader is here (separate from `football_loader.py`)
------------------------------------------------------------
Schedule data is small and pulled once; player stats are bigger and
pulled per season. Keeping them separate matches the basketball pattern
(`player_stats_loader.py` is separate from `basketball_loader.py`).

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.data.espn_qbr_loader import load_espn_qbr_for_seasons
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# ESPN QBR uses slightly different team codes in some years (notably the
# Rams: LAR vs LA; Chargers: SD/LAC). Mirror espn_qbr_loader's mapping so
# we can match QBR rows onto nflverse stats rows.
_TEAM_NORM = {
    "JAX": "JAX", "JAC": "JAX",
    "WSH": "WAS", "WAS": "WAS",
    "LAR": "LA", "LA": "LA",
    "SD": "LAC", "LAC": "LAC",
    "OAK": "LV", "LV": "LV",
    "STL": "LA",
}


def _norm_team(t: str) -> str:
    return _TEAM_NORM.get(str(t or "").strip().upper(), str(t or "").strip().upper())


def _merge_espn_qbr(qb_df: pd.DataFrame, season: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Add an `espn_qbr` column to a QB-season-stats DataFrame.

    Match strategy
    --------------
    Join on (season, normalized team, lowercase last-name). NFL teams have
    very few QBs per season, so collisions are practically impossible.
    Rows with no QBR match get NaN, which the rating composite handles by
    filling with the median before z-scoring.
    """
    if qb_df.empty:
        qb_df["espn_qbr"] = pd.NA
        return qb_df

    try:
        qbr = load_espn_qbr_for_seasons([season], force_refresh=force_refresh)
    except Exception as exc:  # noqa: BLE001
        log.warning("ESPN QBR fetch failed for %s: %s — falling back to passer rating only.",
                    season, exc)
        qb_df["espn_qbr"] = pd.NA
        return qb_df

    if qbr.empty:
        qb_df["espn_qbr"] = pd.NA
        return qb_df

    # Build the join key on both sides.
    left = qb_df.copy()
    name_col = "player_display_name" if "player_display_name" in left.columns else "player_name"
    if name_col not in left.columns:
        log.warning("QB stats have no name column — skipping QBR merge.")
        left["espn_qbr"] = pd.NA
        return left

    left["_team_norm"] = left["team"].map(_norm_team) if "team" in left.columns else ""
    # Extract last name (everything after the last space).
    left["_last"] = (
        left[name_col].astype(str).str.strip().str.lower().str.split().str[-1]
    )

    right = qbr.copy()
    right["_team_norm"] = right["team_abb"].map(_norm_team)
    right["_last"] = right["name_last"].astype(str).str.strip().str.lower()
    right = right.rename(columns={"qbr_total": "espn_qbr"})

    # Keep only the join keys + value cols on the right side to avoid
    # column collisions.
    right_small = right[["_team_norm", "_last", "espn_qbr", "qb_plays"]]
    # Some QBs change teams mid-season — nflverse keeps separate rows per
    # team, ESPN QBR uses the team where the QB had most snaps. Take the
    # max QBR within a (team, last) group to be safe.
    right_small = (
        right_small
        .sort_values("qb_plays", ascending=False)
        .drop_duplicates(subset=["_team_norm", "_last"], keep="first")
    )

    merged = left.merge(
        right_small, on=["_team_norm", "_last"], how="left", suffixes=("", "_qbr"),
    )
    merged = merged.drop(columns=["_team_norm", "_last"], errors="ignore")
    if "qb_plays" in merged.columns:
        merged = merged.drop(columns=["qb_plays"])

    n_matched = int(merged["espn_qbr"].notna().sum())
    log.info("Merged ESPN QBR for %s: matched %d / %d QB rows.",
             season, n_matched, len(merged))
    return merged


# Columns we want from nflverse player_stats (passing position).
# Note: nflverse uses snake_case. The schema is documented at
# https://nflreadr.nflverse.com/articles/dictionary_player_stats.html
_QB_STAT_COLUMNS = [
    "player_id", "player_name", "player_display_name",
    "team", "season", "season_type",
    "position", "position_group",
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions",  # name varies by version; we handle both
    "interceptions",
    "sacks_suffered", "sacks", "sack_yards_lost", "sack_yards",
    "carries", "rushing_yards", "rushing_tds",
    "rushing_fumbles_lost", "sack_fumbles_lost", "fumbles_lost",
    "games",  # GP — varies in newer versions
    "pacr", "dakota",  # advanced passer rating proxies (if present)
    "passer_rating",
    "completion_pct",
]


def _cache_path(season: int, stat_type: str = "qb_season") -> Path:
    return settings.data_dir / f"nfl_{stat_type}_{season}.csv"


# ---------------------------------------------------------------------------
# QB season stats
# ---------------------------------------------------------------------------
def _fetch_player_stats_season(
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull season-aggregated player stats for `season`, cached on disk.
    Returns ALL positions; QB filtering happens downstream.
    """
    path = _cache_path(season, "player_stats_season")
    if path.exists() and not force_refresh:
        log.debug("Player stats cache hit: %s", path.name)
        return pd.read_csv(path, low_memory=False)

    try:
        import nflreadpy as nfl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    log.info("Fetching NFL player season stats for %s …", season)
    # summary_level="reg+post" pulls regular season + postseason aggregated.
    # If a version of nflreadpy doesn't support that mode, fall back to "reg".
    try:
        df_pl = nfl.load_player_stats(
            seasons=[season], summary_level="reg+post",
        )
    except (TypeError, ValueError) as exc:
        log.debug("reg+post mode unavailable (%s); using 'reg'.", exc)
        df_pl = nfl.load_player_stats(seasons=[season], summary_level="reg")

    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl
    if df.empty:
        log.warning("No player stats returned for season %s.", season)
        return df

    df.to_csv(path, index=False)
    log.info("Cached %d player rows -> %s", len(df), path.name)
    return df


def _normalize_qb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    nflverse has renamed several columns across versions
    (e.g., `interceptions` → `passing_interceptions`, `sacks_suffered`
    → `sacks`). Normalize to a single canonical schema so the rest of
    the codebase doesn't care which version is installed.
    """
    df = df.copy()

    # Interceptions
    if "passing_interceptions" not in df.columns and "interceptions" in df.columns:
        df["passing_interceptions"] = df["interceptions"]

    # Sacks taken
    if "sacks_suffered" not in df.columns and "sacks" in df.columns:
        df["sacks_suffered"] = df["sacks"]

    # Sack yards
    if "sack_yards_lost" not in df.columns and "sack_yards" in df.columns:
        df["sack_yards_lost"] = df["sack_yards"]

    # Total fumbles_lost may be split into rushing_fumbles_lost +
    # sack_fumbles_lost in newer versions.
    if "fumbles_lost" not in df.columns:
        rush_fl = df.get("rushing_fumbles_lost", 0)
        sack_fl = df.get("sack_fumbles_lost", 0)
        df["fumbles_lost"] = pd.to_numeric(rush_fl, errors="coerce").fillna(0) + \
                             pd.to_numeric(sack_fl, errors="coerce").fillna(0)

    # Games played
    if "games" not in df.columns:
        if "games_played" in df.columns:
            df["games"] = df["games_played"]
        else:
            df["games"] = 1  # safe default; per-game derivations handle this

    return df


def load_qb_season_stats(
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return one row per QB-season with the columns needed for the QB
    composite. Filters to position == "QB" and applies minimum-games
    /attempts qualification thresholds from settings.

    Derived columns added:
      * games_played            (int — coerced from `games`)
      * pass_ypg                (passing_yards / games_played)
      * pass_tds_pg             (passing_tds / games_played)
      * rush_ypg                (rushing_yards / games_played)
      * rush_tds_pg             (rushing_tds / games_played)
      * total_tds_pg            (pass_tds_pg + rush_tds_pg)
      * ints_pg                 (passing_interceptions / games_played)
      * fumbles_lost_pg
      * turnovers_pg            (ints_pg + fumbles_lost_pg)
      * sacks_pg
      * attempts_pg
      * completion_pct          (re-derived from completions / attempts;
                                 some upstream versions miss this column)
      * passer_rating           (NFL passer rating; computed here if absent
                                 to avoid relying on nflreadpy's column)
      * qualified               (boolean — meets min games + attempts)
    """
    raw = _fetch_player_stats_season(season, force_refresh=force_refresh)
    if raw.empty:
        return pd.DataFrame()

    df = _normalize_qb_columns(raw)

    # Filter to QBs. nflverse uses both `position` and `position_group`;
    # match against either.
    pos = df.get("position", pd.Series(dtype=str)).astype(str).str.upper()
    pg = df.get("position_group", pd.Series(dtype=str)).astype(str).str.upper()
    df = df[(pos == "QB") | (pg == "QB")].copy()
    if df.empty:
        log.warning("No QB rows after position filter for season %s.", season)
        return df

    # Coerce numeric columns
    num_cols = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered", "sack_yards_lost",
        "carries", "rushing_yards", "rushing_tds", "fumbles_lost",
        "games", "passer_rating",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df["games_played"] = df["games"].astype(int)
    # Guard against zero-games rows producing inf/nan.
    games_safe = df["games_played"].replace(0, 1).astype(float)

    df["pass_ypg"] = df["passing_yards"] / games_safe
    df["pass_tds_pg"] = df["passing_tds"] / games_safe
    df["rush_ypg"] = df["rushing_yards"] / games_safe
    df["rush_tds_pg"] = df["rushing_tds"] / games_safe
    df["total_tds_pg"] = df["pass_tds_pg"] + df["rush_tds_pg"]
    df["ints_pg"] = df["passing_interceptions"] / games_safe
    df["fumbles_lost_pg"] = df["fumbles_lost"] / games_safe
    df["turnovers_pg"] = df["ints_pg"] + df["fumbles_lost_pg"]
    df["sacks_pg"] = df["sacks_suffered"] / games_safe
    df["attempts_pg"] = df["attempts"] / games_safe

    # Completion % (re-derive; some versions don't ship it cleanly)
    df["completion_pct"] = (df["completions"] / df["attempts"].replace(0, 1)) * 100.0

    # NFL passer rating — compute if missing. The classic 1973 formula:
    #   a = ((completions/attempts) - 0.3) * 5
    #   b = ((yards/attempts) - 3) * 0.25
    #   c = (TDs/attempts) * 20
    #   d = 2.375 - ((INTs/attempts) * 25)
    # Each component clipped to [0, 2.375]. Final rating = ((a+b+c+d)/6) * 100
    if (df["passer_rating"] == 0).all():
        att_safe = df["attempts"].replace(0, 1)
        a = ((df["completions"] / att_safe) - 0.3) * 5.0
        b = ((df["passing_yards"] / att_safe) - 3.0) * 0.25
        c = (df["passing_tds"] / att_safe) * 20.0
        d = 2.375 - ((df["passing_interceptions"] / att_safe) * 25.0)
        for col in (a, b, c, d):
            col.clip(lower=0.0, upper=2.375, inplace=True)
        df["passer_rating"] = ((a + b + c + d) / 6.0) * 100.0

    # Merge in ESPN QBR (best-effort — falls back to NaN on any failure
    # so the model can keep training even if the QBR feed is down).
    df = _merge_espn_qbr(df, season, force_refresh=force_refresh)

    # Qualification: minimum games + minimum attempts/game. This filters
    # out scout-team appearances, gimmick wildcat snaps, etc.
    df["qualified"] = (
        (df["games_played"] >= settings.qb_min_games)
        & (df["attempts_pg"] >= settings.qb_min_attempts_per_game)
    )

    return df.reset_index(drop=True)


def load_qb_stats_for_seasons(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Concatenated QB stats across multiple seasons (each tagged with `season`)."""
    frames = []
    for s in seasons:
        try:
            frames.append(load_qb_season_stats(s, force_refresh=force_refresh))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load QB stats for %s: %s", s, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Depth charts — for the QB-healthy toggle
# ---------------------------------------------------------------------------
def _depth_chart_cache_path(season: int) -> Path:
    return settings.data_dir / f"nfl_depth_charts_{season}.csv"


def load_depth_charts(
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load NFL depth charts for `season` from nflreadpy. Columns of
    interest:
      * team, week, depth_team (1=starter, 2=backup, ...), position,
      * player_name, gsis_id (== player_id in player_stats).

    Note from nflverse (2026 dictionary): after the 2024 season, depth
    charts are no longer assigned a week — each update gets an ISO
    timestamp instead. This loader handles both schemas.
    """
    path = _depth_chart_cache_path(season)
    if path.exists() and not force_refresh:
        log.debug("Depth chart cache hit: %s", path.name)
        return pd.read_csv(path, low_memory=False)

    try:
        import nflreadpy as nfl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    log.info("Fetching depth charts for %s …", season)
    try:
        df_pl = nfl.load_depth_charts(seasons=[season])
        df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl
    except Exception as exc:  # noqa: BLE001
        # Depth charts can be unavailable for very recent dates if nflverse
        # hasn't backfilled yet. Don't blow up the whole pipeline.
        log.warning("Could not load depth charts for %s: %s — proceeding without.", season, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df.to_csv(path, index=False)
    log.info("Cached %d depth-chart rows -> %s", len(df), path.name)
    return df


def get_starting_qb(
    team: str,
    season: int,
    week: Optional[int] = None,
) -> Optional[dict]:
    """
    Find the starting QB for `team` in `season` (optionally `week`).
    Returns {player_id, player_name} or None if not found.

    Logic:
      1. Load depth chart for the season.
      2. Filter to (team, position=QB, depth_team=1, week<=requested week).
      3. Pick the most recent row.

    Robust to schema variance — older nflverse rows have explicit `week`;
    newer rows have ISO timestamps in `last_updated` or similar.
    """
    df = load_depth_charts(season)
    if df.empty:
        return None

    df = df.copy()
    # Find the canonical column names. We accept several aliases.
    team_col = next((c for c in ("team", "club_code", "team_abbr") if c in df.columns), None)
    pos_col = next((c for c in ("position", "depth_position", "pos_abb") if c in df.columns), None)
    rank_col = next((c for c in ("depth_team", "depth_chart_position", "pos_rank") if c in df.columns), None)
    name_col = next((c for c in (
        "player_name", "full_name", "football_name", "gsis_full_name", "player",
    ) if c in df.columns), None)
    id_col = next((c for c in ("gsis_id", "player_id", "gsis_it_id") if c in df.columns), None)
    week_col = "week" if "week" in df.columns else None
    ts_col = next((c for c in ("last_updated", "as_of", "timestamp") if c in df.columns), None)

    if not (team_col and pos_col and name_col):
        log.warning("Depth chart schema unrecognized; can't extract starting QB.")
        return None

    mask = (df[team_col].astype(str) == team) & \
           (df[pos_col].astype(str).str.upper() == "QB")
    if rank_col is not None:
        mask &= pd.to_numeric(df[rank_col], errors="coerce") == 1
    sub = df[mask].copy()
    if sub.empty:
        return None

    # Sort to grab the most recent assignment, then pick row 0.
    if week_col is not None:
        sub[week_col] = pd.to_numeric(sub[week_col], errors="coerce")
        if week is not None:
            sub = sub[sub[week_col] <= week]
        sub = sub.sort_values(week_col, ascending=False)
    elif ts_col is not None:
        sub[ts_col] = pd.to_datetime(sub[ts_col], errors="coerce")
        sub = sub.sort_values(ts_col, ascending=False)

    if sub.empty:
        return None
    row = sub.iloc[0]
    return {
        "player_id": row[id_col] if id_col else None,
        "player_name": row[name_col],
    }


def get_backup_qb(
    team: str,
    season: int,
    week: Optional[int] = None,
) -> Optional[dict]:
    """
    Return the depth-chart #2 QB for `team`. Same robustness as
    `get_starting_qb`; used when the user toggles the starter as "out".
    """
    df = load_depth_charts(season)
    if df.empty:
        return None

    df = df.copy()
    team_col = next((c for c in ("team", "club_code", "team_abbr") if c in df.columns), None)
    pos_col = next((c for c in ("position", "depth_position", "pos_abb") if c in df.columns), None)
    rank_col = next((c for c in ("depth_team", "depth_chart_position", "pos_rank") if c in df.columns), None)
    name_col = next((c for c in (
        "player_name", "full_name", "football_name", "gsis_full_name", "player",
    ) if c in df.columns), None)
    id_col = next((c for c in ("gsis_id", "player_id", "gsis_it_id") if c in df.columns), None)
    week_col = "week" if "week" in df.columns else None

    if not (team_col and pos_col and name_col and rank_col):
        return None

    mask = (df[team_col].astype(str) == team) & \
           (df[pos_col].astype(str).str.upper() == "QB") & \
           (pd.to_numeric(df[rank_col], errors="coerce") == 2)
    sub = df[mask].copy()
    if sub.empty:
        return None
    if week_col is not None and week is not None:
        sub[week_col] = pd.to_numeric(sub[week_col], errors="coerce")
        sub = sub[sub[week_col] <= week]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {
        "player_id": row[id_col] if id_col else None,
        "player_name": row[name_col],
    }


# ---------------------------------------------------------------------------
# Weekly QB stats — for the ROLLING (leakage-free) QB rating pipeline
# ---------------------------------------------------------------------------
def _fetch_player_stats_weekly(
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull WEEKLY (not season-aggregated) player stats for `season`,
    cached on disk. Returns ALL positions; QB filtering downstream.

    This is the rolling analog of `_fetch_player_stats_season`. We use
    nflreadpy.load_player_stats(summary_level="week").
    """
    path = _cache_path(season, "player_stats_weekly")
    if path.exists() and not force_refresh:
        log.debug("Weekly player stats cache hit: %s", path.name)
        return pd.read_csv(path, low_memory=False)

    try:
        import nflreadpy as nfl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    log.info("Fetching NFL weekly player stats for %s …", season)
    df_pl = nfl.load_player_stats(seasons=[season], summary_level="week")
    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl
    if df.empty:
        log.warning("No weekly player stats for season %s.", season)
        return df

    df.to_csv(path, index=False)
    log.info("Cached %d weekly player rows -> %s", len(df), path.name)
    return df


def load_qb_weekly_stats(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return per-(QB, season, week) weekly stats — the building block for
    rolling QB rating computation. Filters to position == QB.

    Columns
    -------
    player_id, player_display_name, team, season, week,
    completions, attempts, passing_yards, passing_tds,
    passing_interceptions, sacks_suffered,
    rushing_yards, rushing_tds, fumbles_lost,
    plus any other passing/rushing fields nflverse ships per week.

    These are RAW counts (not per-game rates). The rolling QB rating
    function will cumulative-sum them chronologically.
    """
    if not seasons:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for s in seasons:
        try:
            raw = _fetch_player_stats_weekly(s, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("Weekly stats fetch failed for %s: %s", s, exc)
            continue
        if raw.empty:
            continue
        df = _normalize_qb_columns(raw)

        # Position filter
        pos = df.get("position", pd.Series(dtype=str)).astype(str).str.upper()
        pg = df.get("position_group", pd.Series(dtype=str)).astype(str).str.upper()
        df = df[(pos == "QB") | (pg == "QB")].copy()
        if df.empty:
            continue

        # Coerce numeric counts
        num_cols = [
            "completions", "attempts", "passing_yards", "passing_tds",
            "passing_interceptions", "sacks_suffered", "sack_yards_lost",
            "carries", "rushing_yards", "rushing_tds", "fumbles_lost",
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            else:
                df[c] = 0.0

        # Ensure season + week present and integer
        if "season" not in df.columns:
            df["season"] = s
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(s).astype(int)
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        else:
            df["week"] = pd.NA

        # Merge weekly ESPN QBR if available
        df = _merge_espn_qbr_weekly(df, s, force_refresh=force_refresh)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    log.info(
        "Loaded weekly QB stats: %d rows across %d seasons.",
        len(out), out["season"].nunique(),
    )
    return out


def _merge_espn_qbr_weekly(
    weekly_qb: pd.DataFrame,
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Add weekly `espn_qbr` to a weekly QB stats DataFrame. Joins on
    (season, normalized team, lowercase last name, week). Missing
    matches get NaN; the rolling rating handles those gracefully.
    """
    from ballknower_gridiron.data.espn_qbr_loader import (
        load_espn_qbr_weekly_for_seasons,
    )

    if weekly_qb.empty:
        weekly_qb["espn_qbr"] = pd.NA
        return weekly_qb

    try:
        qbr = load_espn_qbr_weekly_for_seasons([season], force_refresh=force_refresh)
    except Exception as exc:  # noqa: BLE001
        log.warning("Weekly QBR fetch failed for %s: %s — using NaN.", season, exc)
        weekly_qb["espn_qbr"] = pd.NA
        return weekly_qb

    if qbr.empty:
        weekly_qb["espn_qbr"] = pd.NA
        return weekly_qb

    left = weekly_qb.copy()
    name_col = "player_display_name" if "player_display_name" in left.columns else "player_name"
    if name_col not in left.columns:
        left["espn_qbr"] = pd.NA
        return left

    left["_team_norm"] = left["team"].map(_norm_team) if "team" in left.columns else ""
    left["_last"] = (
        left[name_col].astype(str).str.strip().str.lower().str.split().str[-1]
    )
    left["_week_int"] = pd.to_numeric(left["week"], errors="coerce").astype("Int64")

    right = qbr.copy()
    right["_team_norm"] = right["team_abb"].map(_norm_team)
    right["_last"] = right["name_last"].astype(str).str.strip().str.lower()
    right["_week_int"] = pd.to_numeric(right["week"], errors="coerce").astype("Int64")
    right = right.rename(columns={"qbr_total": "espn_qbr"})
    # Bring `qb_plays` through too — downstream code uses it to weight
    # QBR averages when aggregating across weeks. Fill missing with NaN
    # so it doesn't accidentally zero-out a per-play denominator.
    qbr_cols = ["_team_norm", "_last", "_week_int", "espn_qbr"]
    if "qb_plays" in right.columns:
        qbr_cols.append("qb_plays")
    right_small = right[qbr_cols]
    right_small = right_small.drop_duplicates(
        subset=["_team_norm", "_last", "_week_int"], keep="last",
    )

    merged = left.merge(
        right_small, on=["_team_norm", "_last", "_week_int"],
        how="left", suffixes=("", "_qbr"),
    )
    merged = merged.drop(columns=["_team_norm", "_last", "_week_int"], errors="ignore")

    n_match = int(merged["espn_qbr"].notna().sum())
    log.info("Merged weekly ESPN QBR for %s: matched %d / %d QB-week rows.",
             season, n_match, len(merged))
    return merged
