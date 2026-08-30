"""
ballknower_gridiron.data.football_loader
========================================

Loads NFL schedule + game data via `nflreadpy` — the official Python
port of nflverse's R library (`nfl_data_py` was archived September 2025;
`nflreadpy` replaced it).

Why `nflreadpy` is the right choice
-----------------------------------
  * **Maintained**: official nflverse package, updated nightly during season.
  * **Schedule already has rest, weather, and neutral-site flags**:
    no need to compute them ourselves. `home_rest`, `away_rest`,
    `temp`, `wind`, `roof`, `surface`, `location` ("Home" vs "Neutral"
    — the latter flags every international game).
  * **One function for past + future**: `load_schedules(seasons)` returns
    *all* games of those seasons including unplayed ones — completed
    games have scores, future games have NaN scores.

Two main artifacts this module produces:
  * **Completed games** (one row per game): used for training. Filter
    to `~np.isnan(home_score)`.
  * **Upcoming games**: filter to NaN scores in the current season
    (week >= current week of season).

All pulls are cached on disk in `data_cache/football/` so repeat
training runs don't hammer github.com/nflverse/nflverse-data.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------
def _current_season() -> int:
    """
    Return the calendar year that the *current* NFL season started.

    The NFL season starts in September. Before September we're either in
    the offseason (May–Aug — still tied to previous season's data) or the
    playoffs of the season that started the previous calendar year.
    """
    today = date_cls.today()
    return today.year if today.month >= 9 else today.year - 1


def list_seasons(seasons_back: Optional[int] = None) -> List[int]:
    """Return the list of seasons to pull (most-recent-first)."""
    n = seasons_back if seasons_back is not None else settings.nfl_seasons_back
    current = _current_season()
    return [current - i for i in range(n)]


# ---------------------------------------------------------------------------
# Per-season schedule fetch + cache
# ---------------------------------------------------------------------------
def _cache_path(seasons: List[int]) -> Path:
    """Local cache file for a set of seasons. Sorted for cache stability."""
    tag = "_".join(str(s) for s in sorted(seasons))
    return settings.data_dir / f"nfl_schedule_{tag}.csv"


def _fetch_schedules(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull schedules for the given seasons via nflreadpy. Cached on disk
    as one big CSV (schedules are small — ~270 rows per season).

    The nflverse schedules dataset is built and maintained by Lee Sharpe;
    it includes:

        season, week, game_type (REG/WC/DIV/CON/SB),
        gameday, weekday, gametime, game_id,
        home_team, away_team, home_score, away_score,
        home_rest, away_rest,            # days of rest before game
        temp, wind,                       # outdoor games only
        roof, surface,                    # stadium info
        location,                         # "Home" or "Neutral" (international)
        stadium,
        spread_line, total_line,          # Vegas info (display only)
        away_moneyline, home_moneyline,
        div_game,                         # division matchup flag
        referee,

    Polars is the native format; we convert to pandas immediately to
    match the basketball codebase.
    """
    path = _cache_path(seasons)
    if path.exists() and not force_refresh:
        log.debug("Schedule cache hit: %s", path.name)
        return pd.read_csv(path, low_memory=False)

    try:
        import nflreadpy as nfl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    log.info("Fetching NFL schedules for seasons %s …", seasons)
    df_pl = nfl.load_schedules(seasons=seasons)
    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl
    if df.empty:
        log.warning("nflreadpy returned no schedule data for %s.", seasons)
        return df

    # Persist immediately so we can iterate on downstream code without
    # re-hitting GitHub.
    df.to_csv(path, index=False)
    log.info("Cached %d schedule rows -> %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Schema normalization — map nflverse columns onto the same shape the
# basketball pipeline uses (so feature builders, ELO, and scripts can
# share patterns across sports).
# ---------------------------------------------------------------------------
_REQUIRED_RAW_COLUMNS = [
    "season", "week", "gameday", "game_id",
    "home_team", "away_team", "home_score", "away_score",
    "home_rest", "away_rest",
]
# Either of these tells us regular-season vs playoffs. nflverse schedules
# use `game_type` (REG/WC/DIV/CON/SB/PRE). Some older / alternate sources
# use `season_type` (REG/POST). We accept whichever is present.
_PLAYOFF_FLAG_COLUMNS = ["game_type", "season_type"]
_OPTIONAL_RAW_COLUMNS = [
    "temp", "wind", "roof", "surface", "location", "stadium",
    "spread_line", "total_line", "weekday", "gametime",
    "div_game", "home_moneyline", "away_moneyline", "referee",
]


def _normalize_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape nflverse schedule rows into the standard ballknower
    one-row-per-game schema. Adds derived columns:

      * game_date        : datetime version of `gameday`
      * is_playoff       : 1 if game is in the postseason
                            (derived from `game_type` ∈ {WC,DIV,CON,SB},
                            or `season_type` == "POST" if game_type missing)
      * is_international : 1 if location == "Neutral" (covers London,
                            Munich, Mexico City, São Paulo, Madrid, etc.)
      * is_dome          : 1 if roof in {"dome", "closed"}
      * home_won         : 1 if home_score > away_score (NaN for unplayed)
      * point_diff       : home_score - away_score (NaN for unplayed)
      * completed        : True if both scores are non-null
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # Make sure required columns exist; missing optionals get filled.
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        log.warning("nflverse schedule missing required columns: %s — "
                    "downstream features may be degraded.", missing)
    for col in _OPTIONAL_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce")

    # Playoff flag — handle both `game_type` (current nflverse) and
    # `season_type` (older / alternate sources). Fall back to 0 if neither.
    if "game_type" in df.columns:
        gt = df["game_type"].astype(str).str.upper()
        df["is_playoff"] = gt.isin(["WC", "DIV", "CON", "SB"]).astype(int)
    elif "season_type" in df.columns:
        df["is_playoff"] = (
            df["season_type"].astype(str).str.upper() == "POST"
        ).astype(int)
    else:
        log.warning(
            "Schedule has neither `game_type` nor `season_type` — treating "
            "all games as regular season (is_playoff=0)."
        )
        df["is_playoff"] = 0

    # Neutral-site games (international Series, Super Bowl). The Super Bowl
    # is on a neutral field but happens at season's end — we still capture
    # it. For training, the model will learn it from the playoff flag too.
    loc = df["location"].astype(str).str.lower().fillna("home")
    df["is_international"] = (loc == "neutral").astype(int)

    roof = df["roof"].astype(str).str.lower().fillna("outdoors")
    df["is_dome"] = roof.isin(["dome", "closed"]).astype(int)

    # Wind / temp may be NaN for dome games (no weather). Replace with
    # neutral values for downstream models (v4 weather feature).
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    # In domes, NFL convention is temp=68, wind=0 — use this if missing.
    dome_mask = df["is_dome"] == 1
    df.loc[dome_mask & df["temp"].isna(), "temp"] = 68.0
    df.loc[dome_mask & df["wind"].isna(), "wind"] = 0.0

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["completed"] = df["home_score"].notna() & df["away_score"].notna()
    df["home_won"] = (df["home_score"] > df["away_score"]).where(df["completed"]).astype("Int64")
    df["point_diff"] = (df["home_score"] - df["away_score"]).where(df["completed"])

    # Rest columns: cap at the configured maximum so bye-week rest doesn't
    # blow up scale-sensitive features.
    df["home_rest"] = pd.to_numeric(df["home_rest"], errors="coerce").fillna(7).clip(
        lower=0, upper=settings.rest_cap_days,
    )
    df["away_rest"] = pd.to_numeric(df["away_rest"], errors="coerce").fillna(7).clip(
        lower=0, upper=settings.rest_cap_days,
    )

    # Sort chronologically; downstream replay assumes this.
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API: historical
# ---------------------------------------------------------------------------
def load_nfl_games(
    seasons_back: Optional[int] = None,
    include_playoffs: bool = True,
    completed_only: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a chronologically sorted DataFrame, one row per NFL game.

    Parameters
    ----------
    seasons_back : how many seasons of history (default: settings.nfl_seasons_back).
    include_playoffs : include POST games as well as REG.
    completed_only : if True (default), filter to games with final scores.
                     Set False when you want past + future schedule together.
    force_refresh : ignore disk cache and re-pull from nflverse.

    Columns (after normalization)
    -----------------------------
    game_id, game_date, season, week, game_type, is_playoff,
    home_team, away_team, home_score, away_score, home_won, point_diff,
    home_rest, away_rest,
    is_international, is_dome, temp, wind, roof, surface, location, stadium,
    spread_line, total_line, div_game, completed.
    """
    seasons = list_seasons(seasons_back)
    log.info("Loading NFL games for seasons: %s", seasons)

    raw = _fetch_schedules(seasons, force_refresh=force_refresh)
    if raw.empty:
        raise RuntimeError(
            "No NFL games could be loaded. Check your network connection "
            "and that `nflreadpy` is installed."
        )

    df = _normalize_schedule(raw)
    if not include_playoffs:
        df = df[df["is_playoff"] == 0].reset_index(drop=True)
    if completed_only:
        df = df[df["completed"]].reset_index(drop=True)

    # Coerce label column back to plain int (only safe AFTER completed_only
    # filter since unplayed games have <NA> there).
    if completed_only:
        df["home_won"] = df["home_won"].astype(int)
        df["point_diff"] = df["point_diff"].astype(float)

    log.info("Loaded %d NFL games across %d season(s) (completed_only=%s).",
             len(df), len(seasons), completed_only)
    return df


# ---------------------------------------------------------------------------
# Public API: upcoming games
# ---------------------------------------------------------------------------
def get_upcoming_games(
    days_ahead: int = 10,
    season: Optional[int] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return all scheduled games in the next `days_ahead` days, starting
    today. Filtered out of the nflverse schedule, which contains every
    game of a season including unplayed ones.

    Season-boundary handling
    ------------------------
    `_current_season()` is a *training* concept: from January through
    August it returns the season that started the previous September,
    because that's the season with completed games to learn from. But
    "what's coming up" in, say, late August is Week 1 of the season
    that hasn't started yet — a different season label entirely.

    So this function looks at BOTH `_current_season()` and the following
    season, then filters by date. In midseason the next-season fetch
    returns an empty or unplayed schedule and costs nothing; in the
    offseason it's the only place upcoming games can be found. Passing
    an explicit `season` overrides this and searches that season alone.

    Parameters
    ----------
    days_ahead
        Horizon in days from today.
    season
        Search only this season instead of the current/next pair.
    force_refresh
        Re-download the schedule instead of using the local cache. Worth
        setting when a season is newly published or a spread line moved.
    """
    if season is not None:
        seasons = [int(season)]
    else:
        current = _current_season()
        seasons = [current, current + 1]

    frames: List[pd.DataFrame] = []
    for s in seasons:
        try:
            raw = _fetch_schedules([s], force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            # A not-yet-published season 404s at nflverse. That is expected
            # early in an offseason and must not break the lookup.
            log.info("No schedule available yet for season %d (%s) — skipping.", s, exc)
            continue
        if raw is None or raw.empty:
            continue
        frames.append(_normalize_schedule(raw))

    if not frames:
        log.warning("No schedule data available for seasons %s.", seasons)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    today = pd.Timestamp(date_cls.today())
    horizon = today + pd.Timedelta(days=days_ahead)
    upcoming = df[
        (~df["completed"]) & (df["game_date"] >= today) & (df["game_date"] <= horizon)
    ].sort_values("game_date").reset_index(drop=True)

    if upcoming.empty:
        # Give the caller something actionable instead of a bare zero.
        future = df[(~df["completed"]) & (df["game_date"] >= today)]
        if not future.empty:
            next_date = pd.to_datetime(future["game_date"].min()).date()
            days_out = (next_date - date_cls.today()).days
            log.info(
                "Found 0 games in the next %d day(s). Next scheduled game is %s "
                "(%d days out) — try --days-ahead %d.",
                days_ahead, next_date, days_out, days_out + 1,
            )
        else:
            log.info("Found 0 upcoming games in seasons %s — no unplayed games "
                     "remain in the fetched schedules.", seasons)
        return upcoming

    seasons_found = sorted(upcoming["season"].unique().tolist())
    log.info("Found %d upcoming games in the next %d day(s) (season(s): %s).",
             len(upcoming), days_ahead, seasons_found)
    return upcoming


def get_games_for_week(season: int, week: int) -> pd.DataFrame:
    """Return all games for one (season, week) combination."""
    raw = _fetch_schedules([season], force_refresh=False)
    df = _normalize_schedule(raw)
    return df[(df["season"] == season) & (df["week"] == week)].reset_index(drop=True)


def refresh_current_season() -> dict:
    """
    Force re-download of the current AND next season's schedules — the
    same pair `get_upcoming_games` searches. Refreshing only the current
    season would be useless in the offseason, when the schedule you
    actually want is next season's.

    Used as the "🔄 Refresh" hook in the Streamlit app. Also worth running
    midseason: nflverse updates `spread_line` as lines move, so a refresh
    picks up current numbers for the Vegas-disagreement section.
    """
    current = _current_season()
    refreshed, skipped = [], []
    for season in (current, current + 1):
        try:
            raw = _fetch_schedules([season], force_refresh=True)
            n = 0 if raw is None else len(raw)
            log.info("Refreshed NFL season %s (%d rows).", season, n)
            refreshed.append({"season": season, "rows": n})
        except Exception as exc:  # noqa: BLE001
            log.info("Season %s not available to refresh (%s).", season, exc)
            skipped.append(season)

    return {
        "seasons_refreshed": refreshed,
        "seasons_skipped": skipped,
        "refreshed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
