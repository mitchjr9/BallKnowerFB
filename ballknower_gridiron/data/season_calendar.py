"""
ballknower_gridiron.data.season_calendar
========================================

One rule, shared by every loader that caches to disk:

    **A season that is still being played is never served from cache.**

Why this module exists
----------------------
Every nflverse loader in this package caches to disk so historical data isn't
re-downloaded on every run. That is correct for finished seasons and wrong for
the current one, whose data changes every week. Each loader got this wrong
independently, and each produced its own quiet failure:

  * the grader read a schedule cached before kickoff, found zero finished
    games, and reported the whole week as "still open";
  * the weekly pipeline counted games played from a pre-season cache, so data
    depth read "none" in Week 3;
  * play-by-play and weekly QB stats cached on first download would have frozen
    the in-season team EPA and QB ratings at whatever week they were first
    pulled.

Fixing each call site separately is how the bug kept coming back. Centralising
the rule here means a new loader gets it right by calling `is_in_progress`.

The cost is one small re-download per run for the current season only. The
alternative — silently stale ratings in a pre-registered forecast — is not
something you can fix after the fact.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional


def current_nfl_season(today: Optional[date] = None) -> int:
    """
    The season label that is current on `today`.

    NFL seasons are labelled by the year they START in, and run September
    through early February. So January 2027 is still the 2026 season.
    """
    d = today or date.today()
    return d.year if d.month >= 9 else d.year - 1


def is_in_progress(season: int, today: Optional[date] = None) -> bool:
    """
    True if `season`'s data may still change, so a cached copy can't be trusted.

    Deliberately conservative: from March to August the just-finished season is
    still treated as live. That costs one unnecessary re-download of a season
    that won't change. Erring the other way would serve stale data for the
    season that matters.
    """
    return int(season) >= current_nfl_season(today)


def any_in_progress(seasons: Iterable[int], today: Optional[date] = None) -> bool:
    return any(is_in_progress(s, today) for s in seasons)
