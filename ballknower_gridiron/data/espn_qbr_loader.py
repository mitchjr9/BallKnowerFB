"""
ballknower_gridiron.data.espn_qbr_loader
========================================

Fetches ESPN's Total Quarterback Rating (QBR, 0–100 scale) directly
from the `nflverse/espnscrapeR-data` GitHub repo. This is the same
data that the R package `nflreadr` ships via `load_espn_qbr()` — the
Python port (`nflreadpy`) deliberately doesn't wrap it (the nflverse
team flagged it as "not sure of long-term fit"), so we fetch the raw
CSV ourselves.

Why include QBR
---------------
Traditional passer rating (the 1973 NFL formula, max 158.3) was
designed in an era of much lower passing volume and weights
completions/yards/TDs/INTs equally. ESPN QBR was designed to address
its weaknesses:

  * Adjusts for *down and distance* (a 5-yard pass on 3rd-and-4 is
    worth more than a 5-yard pass on 1st-and-10).
  * Discounts garbage-time stats (a TD when the game's already decided
    counts less).
  * Adjusts for *opponent strength* (yards against the '85 Bears'
    descendants count more than against bottom-five defenses).
  * Includes rushing, sacks, fumbles, and penalties — passer rating
    only credits passing.
  * Centered at 50 with a 0–100 scale (intuitive).

Caveats: QBR is proprietary (ESPN does the EPA modeling internally,
not the open-source nflfastR EPA). Treating it as a complement to
passer rating rather than a replacement is the right call.

Data source
-----------
  * URL: https://raw.githubusercontent.com/nflverse/espnscrapeR-data/
          master/data/qbr-nfl-season.csv  (one row per QB-season)
  * Available: 2006 → present
  * Updates: weekly during the NFL season (automated by nflverse)

Schema (relevant columns from a ~23-column CSV):
  season, season_type, team_abb, player_id (ESPN), name_first, name_last,
  name_display, name_short, rank, qbr_total, qb_plays, qualified

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


# Pinning to the nflverse/espnscrapeR-data master branch. This repo is the
# automated nightly mirror of ESPN's public QBR feed.
_QBR_SEASON_URL: str = (
    "https://raw.githubusercontent.com/nflverse/espnscrapeR-data/"
    "master/data/qbr-nfl-season.csv"
)
_QBR_WEEKLY_URL: str = (
    "https://raw.githubusercontent.com/nflverse/espnscrapeR-data/"
    "master/data/qbr-nfl-weekly.csv"
)


# Team abbreviation normalization — ESPN occasionally uses different codes
# than nflverse. Map ESPN values to nflverse canonical values.
_TEAM_ABBR_FIX = {
    "JAX": "JAX", "JAC": "JAX",
    "WSH": "WAS", "WAS": "WAS",
    "LAR": "LA", "LA": "LA",       # nflverse uses "LA" for the Rams 2016+
    "SD": "LAC", "LAC": "LAC",
    "OAK": "LV", "LV": "LV",
    "STL": "LA",
}


def _normalize_team(team: str) -> str:
    t = str(team or "").strip().upper()
    return _TEAM_ABBR_FIX.get(t, t)


def _cache_path() -> Path:
    return settings.data_dir / "espn_qbr_season_all.csv"


def _fetch_qbr_season_csv(force_refresh: bool = False) -> pd.DataFrame:
    """Download the full multi-season QBR CSV (small — ~1MB)."""
    cache = _cache_path()
    if cache.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache, low_memory=False)
            if not df.empty:
                log.debug("ESPN QBR cache hit: %s (%d rows)", cache.name, len(df))
                return df
        except Exception as exc:  # noqa: BLE001
            log.warning("Bad QBR cache, re-fetching: %s", exc)

    try:
        import requests  # local import; only needed at fetch time
    except ImportError:
        log.error("`requests` not installed — cannot fetch ESPN QBR. "
                  "Run: pip install requests")
        return pd.DataFrame()

    log.info("Fetching ESPN QBR season CSV from nflverse/espnscrapeR-data …")
    try:
        resp = requests.get(_QBR_SEASON_URL, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not fetch ESPN QBR (%s). The model will continue without "
            "QBR — passer rating alone will be used.", exc,
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse ESPN QBR CSV: %s", exc)
        return pd.DataFrame()

    if df.empty:
        log.warning("ESPN QBR CSV came back empty.")
        return df

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    log.info("Cached ESPN QBR -> %s (%d rows, %d seasons)",
             cache.name, len(df), df["season"].nunique() if "season" in df else 0)
    return df


def load_espn_qbr_for_seasons(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a tidy QBR-season DataFrame for the requested seasons.

    Columns
    -------
      * season         (int)
      * team_abb       (str, normalized to nflverse abbreviations)
      * player_id_espn (str, ESPN's internal ID — different from gsis_id)
      * name_display   (e.g. "Patrick Mahomes")
      * name_last      (lowercase, for joining)
      * name_first     (lowercase, for joining)
      * qbr_total      (float, 0–100, the ESPN QBR headline number)
      * qb_plays       (int — analogous to attempts; useful for qualification)
      * is_regular_season (bool — True if season_type == "Regular Season")

    On any failure (network down, schema drift, etc.) returns an empty
    DataFrame and logs a warning. The QB rating composite handles
    missing QBR by falling back to passer-rating-only z-scores.
    """
    if not seasons:
        return pd.DataFrame()

    raw = _fetch_qbr_season_csv(force_refresh=force_refresh)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # The CSV ships every season since 2006 — slice to what we need.
    if "season" not in df.columns:
        log.warning("ESPN QBR CSV missing `season` column — bailing.")
        return pd.DataFrame()
    df = df[df["season"].astype(int).isin(set(int(s) for s in seasons))].copy()
    if df.empty:
        log.warning("No ESPN QBR rows for requested seasons %s.", seasons)
        return df

    # Keep only regular-season totals. Some CSV versions use
    # "Regular Season", others "Regular".
    if "season_type" in df.columns:
        st = df["season_type"].astype(str).str.lower()
        df["is_regular_season"] = st.str.contains("regular")
        df = df[df["is_regular_season"]].copy()
    else:
        df["is_regular_season"] = True

    # Normalize team abbreviations.
    team_col = "team_abb" if "team_abb" in df.columns else (
        "team" if "team" in df.columns else None
    )
    if team_col is None:
        log.warning("ESPN QBR has no team column — bailing.")
        return pd.DataFrame()
    df["team_abb"] = df[team_col].map(_normalize_team)

    # Standardize name columns.
    if "name_last" in df.columns:
        df["name_last"] = df["name_last"].astype(str).str.strip().str.lower()
    else:
        df["name_last"] = ""
    if "name_first" in df.columns:
        df["name_first"] = df["name_first"].astype(str).str.strip().str.lower()
    else:
        df["name_first"] = ""
    if "name_display" in df.columns:
        df["name_display"] = df["name_display"].astype(str).str.strip()
    else:
        # Fallback: build from first+last.
        df["name_display"] = (
            df["name_first"].str.title() + " " + df["name_last"].str.title()
        ).str.strip()

    # ESPN ID — kept for downstream join with nflverse's player table if
    # we ever decide to upgrade the join.
    if "player_id" in df.columns:
        df["player_id_espn"] = df["player_id"].astype(str)
    else:
        df["player_id_espn"] = ""

    # Numeric columns.
    for col in ("qbr_total", "qb_plays"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    keep = [
        "season", "team_abb", "player_id_espn",
        "name_display", "name_first", "name_last",
        "qbr_total", "qb_plays", "is_regular_season",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["season"] = out["season"].astype(int)
    log.info(
        "Loaded ESPN QBR for %d seasons — %d QB rows (avg QBR=%.1f).",
        out["season"].nunique(), len(out),
        float(out["qbr_total"].mean()) if not out.empty else 0.0,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weekly QBR — for the ROLLING (leakage-free) QB rating pipeline
# ---------------------------------------------------------------------------
_QBR_WEEKLY_CACHE_NAME = "espn_qbr_weekly_all.csv"


def _fetch_qbr_weekly_csv(force_refresh: bool = False) -> pd.DataFrame:
    """Download the multi-season weekly QBR CSV (~5MB)."""
    cache = settings.data_dir / _QBR_WEEKLY_CACHE_NAME
    if cache.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache, low_memory=False)
            if not df.empty:
                log.debug("Weekly QBR cache hit: %s (%d rows)", cache.name, len(df))
                return df
        except Exception as exc:  # noqa: BLE001
            log.warning("Bad weekly QBR cache, re-fetching: %s", exc)

    try:
        import requests
    except ImportError:
        log.error("`requests` not installed — cannot fetch weekly QBR.")
        return pd.DataFrame()

    log.info("Fetching weekly ESPN QBR from nflverse/espnscrapeR-data …")
    try:
        resp = requests.get(_QBR_WEEKLY_URL, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch weekly QBR (%s) — using empty.", exc)
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse weekly QBR CSV: %s", exc)
        return pd.DataFrame()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    log.info("Cached weekly QBR -> %s (%d rows)", cache.name, len(df))
    return df


def load_espn_qbr_weekly_for_seasons(
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Per-week ESPN QBR for the requested seasons. Same column structure
    as the season loader but with a `week` column added.

    Returned columns
    ----------------
    season, week, team_abb, player_id_espn,
    name_display, name_first, name_last,
    qbr_total, qb_plays, is_regular_season
    """
    if not seasons:
        return pd.DataFrame()

    raw = _fetch_qbr_weekly_csv(force_refresh=force_refresh)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if "season" not in df.columns:
        log.warning("Weekly QBR has no `season` column.")
        return pd.DataFrame()
    df = df[df["season"].astype(int).isin(set(int(s) for s in seasons))].copy()
    if df.empty:
        return df

    if "season_type" in df.columns:
        st = df["season_type"].astype(str).str.lower()
        df["is_regular_season"] = st.str.contains("regular")
        df = df[df["is_regular_season"]].copy()
    else:
        df["is_regular_season"] = True

    team_col = "team_abb" if "team_abb" in df.columns else (
        "team" if "team" in df.columns else None
    )
    if team_col is None:
        return pd.DataFrame()
    df["team_abb"] = df[team_col].map(_normalize_team)

    # Weekly QBR uses `game_week` (e.g., "1", "2", ...). Coerce to int.
    if "game_week" in df.columns:
        df["week"] = pd.to_numeric(df["game_week"], errors="coerce").astype("Int64")
    elif "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    else:
        log.warning("Weekly QBR has no week column.")
        return pd.DataFrame()
    df = df[df["week"].notna()]

    if "name_last" in df.columns:
        df["name_last"] = df["name_last"].astype(str).str.strip().str.lower()
    else:
        df["name_last"] = ""
    if "name_first" in df.columns:
        df["name_first"] = df["name_first"].astype(str).str.strip().str.lower()
    else:
        df["name_first"] = ""
    if "name_display" in df.columns:
        df["name_display"] = df["name_display"].astype(str).str.strip()
    else:
        df["name_display"] = (
            df["name_first"].str.title() + " " + df["name_last"].str.title()
        ).str.strip()

    df["player_id_espn"] = df["player_id"].astype(str) if "player_id" in df.columns else ""

    for col in ("qbr_total", "qb_plays"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    keep = [
        "season", "week", "team_abb", "player_id_espn",
        "name_display", "name_first", "name_last",
        "qbr_total", "qb_plays", "is_regular_season",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    log.info("Loaded weekly ESPN QBR for %d seasons — %d rows.",
             out["season"].nunique(), len(out))
    return out.reset_index(drop=True)
