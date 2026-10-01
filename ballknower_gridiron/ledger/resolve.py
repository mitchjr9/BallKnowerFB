"""
ballknower_gridiron.ledger.resolve
==================================

Grades emitted forecasts against actual NFL results. Closes the loop on a
ledger that could previously only write.

    python -m ballknower_gridiron.ledger.resolve
    python -m ballknower_gridiron.ledger.resolve --date 2026-09-08
    python -m ballknower_gridiron.ledger.resolve --all

Design rules, each of which exists to survive an auditor rather than to be
convenient:

* **Append-only, in a separate file.** Resolutions go to `resolutions.jsonl`
  beside `forecasts.jsonl`, which is never opened for writing. A forecast row
  is immutable the moment it is committed, so the manifest's `chain_head` stays
  verifiable forever and "did he quietly delete the bad picks?" has a
  mechanical answer.

* **Idempotent, keyed on forecast_id.** Re-running skips rows already graded.
  Games that aren't final yet stay open and get picked up on a later run — no
  regrade, no overwrite. That matters because a Monday run will find Sunday
  final and Monday night still in progress.

* **Void is narrow, and a tie is not void.** See the two decisions below.

Two NFL resolution decisions, made deliberately and in advance
--------------------------------------------------------------
**1. A tie resolves the moneyline as FALSE, not void.**

NFL regular-season games can end tied (roughly one per season under current
overtime rules). Sportsbooks void moneylines on a tie. We don't, for two
reasons. The model's target is `home_score > away_score`, so its probability
is literally P(home outscores away) — a tie is a "no" in exactly the sense the
model was trained on. And voiding would quietly delete the hardest games from
the record, which is the same move as deleting bad picks. Zero is a real
outcome; see the sibling decision in the fantasy emitter.

**2. Void is reserved for games that never produce a result.**

Cancelled or abandoned games (the 2022 Bills-Bengals no-contest is the real
precedent) can't be graded and are marked `void`. A *postponed* game that is
later played keeps the same nflverse `game_id`, so it resolves normally
whenever it finishes — it simply stays open until then.

DISCLAIMER: For entertainment and educational purposes only — not financial or
betting advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.ledger.emit_records import (
    RESOLUTION_POLICY_VERSION, event_id, now_iso,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger("ledger.resolve")

CONTENT_ROOT = settings.project_root / "content" / "football"
DISCLAIMER = ("For entertainment and educational purposes only — "
              "not financial or betting advice.")


# --------------------------------------------------------------------------- #
# Locating batches
# --------------------------------------------------------------------------- #
def batch_dirs(date: Optional[str], every: bool) -> List[Path]:
    """Return the ledger directories to grade, newest last."""
    if every:
        return sorted(p for p in CONTENT_ROOT.glob("*/ledger")
                      if (p / "forecasts.jsonl").exists())
    d = CONTENT_ROOT / (date or dt.date.today().isoformat()) / "ledger"
    return [d] if (d / "forecasts.jsonl").exists() else []


# --------------------------------------------------------------------------- #
# Results index
# --------------------------------------------------------------------------- #
def build_results_index(seasons: List[int],
                        force_refresh: bool = True) -> Dict[str, dict]:
    """
    Map `event_id` -> final result, keyed exactly the way the emitter keyed it
    so the join can't silently miss.

    Both key forms are registered: the nflverse `game_id` form the emitter
    prefers, and the composite fallback for any row emitted without one. A row
    that matches neither stays open rather than being graded against the wrong
    game — the failure mode we want is "ungraded", never "graded wrong".

    `force_refresh` defaults to **True**, and that default is load-bearing.
    The schedule is cached to CSV, and the copy written when the slate was
    generated has empty score columns for every game that had not yet been
    played. Grading against that cache finds zero completed games and reports
    the entire week as "still open" — which looks like a broken grader rather
    than stale data, and would quietly stall the whole record. Re-pulling costs
    one small request; getting this wrong costs a week.
    """
    from ballknower_gridiron.data.football_loader import load_nfl_games

    seasons_back = max(1, dt.date.today().year - min(seasons) + 1)
    games = load_nfl_games(seasons_back=seasons_back, completed_only=True,
                           force_refresh=force_refresh)
    games = games[games["season"].astype(int).isin([int(s) for s in seasons])]

    index: Dict[str, dict] = {}
    for r in games.itertuples(index=False):
        home = str(getattr(r, "home_team", "")).upper()
        away = str(getattr(r, "away_team", "")).upper()
        hs = getattr(r, "home_score", None)
        as_ = getattr(r, "away_score", None)
        if hs is None or as_ is None:
            continue
        try:
            hs, as_ = int(hs), int(as_)
        except (TypeError, ValueError):
            continue

        payload = {
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "margin": hs - as_,
            "winner": home if hs > as_ else (away if as_ > hs else None),
            "is_tie": hs == as_,
            "game_date": str(getattr(r, "game_date", ""))[:10],
            "season": int(getattr(r, "season", 0)),
            "week": int(getattr(r, "week", 0) or 0),
        }
        gid = getattr(r, "game_id", None)
        for key in {
            event_id(gid, payload["season"], payload["week"], home, away),
            event_id(None, payload["season"], payload["week"], home, away),
        }:
            index[key] = payload
    return index


# --------------------------------------------------------------------------- #
# Grading one forecast
# --------------------------------------------------------------------------- #
def grade(fc: dict, result: dict, source_asof: str) -> dict:
    """
    Produce one resolution row. Stores the raw observation alongside the graded
    outcome so an auditor can recompute the grade rather than trust it.
    """
    base = {
        "forecast_id": fc["forecast_id"],
        "event_id": fc["event_id"],
        "record_family": fc["record_family"],
        "market_type": fc["market_type"],
        "resolved_ts": now_iso(),
        "source": "nflverse",
        "source_asof_ts": source_asof,
        "policy_version": RESOLUTION_POLICY_VERSION,
        "observed": {
            "home_team": result["home_team"], "away_team": result["away_team"],
            "home_score": result["home_score"], "away_score": result["away_score"],
            "margin": result["margin"],
        },
    }

    if fc["record_family"] == "probabilistic":
        # The proposition is "<subject> defeats <opponent>". Resolve on whether
        # the SUBJECT actually outscored the opponent — reading the stored side
        # rather than re-deriving from probability, so a grader bug can't be
        # masked by agreeing with the forecast.
        subject_is_home = fc.get("side") == "home"
        subject = result["home_team"] if subject_is_home else result["away_team"]
        if result["is_tie"]:
            # Decision 1: a tie means the subject did not defeat the opponent.
            outcome = 0
            note = "tie — proposition false (see module docstring)"
        else:
            outcome = 1 if result["winner"] == subject else 0
            note = None
        base.update({"resolution_status": "resolved", "outcome": outcome,
                     "subject_won": bool(outcome)})
        if note:
            base["note"] = note
        return base

    # Distributional: the margin record is home minus away, always.
    base.update({"resolution_status": "resolved",
                 "actual_value": float(result["margin"])})
    return base


def void(fc: dict, reason: str, source_asof: str) -> dict:
    return {
        "forecast_id": fc["forecast_id"],
        "event_id": fc["event_id"],
        "record_family": fc["record_family"],
        "market_type": fc["market_type"],
        "resolved_ts": now_iso(),
        "source": "nflverse",
        "source_asof_ts": source_asof,
        "policy_version": RESOLUTION_POLICY_VERSION,
        "resolution_status": "void",
        "note": reason,
    }


# --------------------------------------------------------------------------- #
# Batch resolution
# --------------------------------------------------------------------------- #
def resolve_batch(ledger_dir: Path, results: Dict[str, dict],
                  source_asof: str) -> Dict[str, int]:
    forecasts = [json.loads(l) for l in
                 (ledger_dir / "forecasts.jsonl").read_text().splitlines() if l.strip()]

    res_path = ledger_dir / "resolutions.jsonl"
    already: set = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            if line.strip():
                already.add(json.loads(line)["forecast_id"])

    new_rows: List[dict] = []
    stats = {"forecasts": len(forecasts), "already": 0, "new": 0,
             "resolved": 0, "void": 0, "open": 0}
    open_events: Dict[str, str] = {}

    for fc in forecasts:
        if fc["forecast_id"] in already:
            stats["already"] += 1
            continue
        result = results.get(fc["event_id"])
        if result is None:
            # Not final upstream yet — leave it open. Never guess.
            stats["open"] += 1
            # Name it, so "3 still open" is actionable instead of alarming:
            # one unplayed Monday night game and a genuine join failure produce
            # the same count but need completely different responses.
            open_events[fc["event_id"]] = str(fc.get("event_start_ts", ""))[:16]
            continue
        row = grade(fc, result, source_asof)
        new_rows.append(row)
        stats["new"] += 1
        stats["void" if row["resolution_status"] == "void" else "resolved"] += 1

    if new_rows:
        with open(res_path, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    stats["open_events"] = open_events
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Grade emitted NFL forecasts against actual results.")
    ap.add_argument("--date", default=None,
                    help="batch folder to grade (default: today)")
    ap.add_argument("--all", action="store_true", dest="every",
                    help="grade every batch that still has open forecasts")
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="seasons to pull results for (default: current NFL season)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="grade against the cached schedule instead of "
                         "re-pulling. Only for offline replay — the cache "
                         "written when a slate was generated has no scores in "
                         "it, so this will report every game as still open.")
    args = ap.parse_args(argv)

    dirs = batch_dirs(args.date, args.every)
    if not dirs:
        raise SystemExit(
            "No forecast batch found. Expected "
            f"{CONTENT_ROOT}/<date>/ledger/forecasts.jsonl\n"
            "Run: python -m ballknower_gridiron.ledger.emit")

    if args.seasons:
        seasons = args.seasons
    else:
        today = dt.date.today()
        # An NFL season is labelled by the calendar year it STARTS in, so
        # January and February belong to the previous label.
        seasons = [today.year if today.month >= 9 else today.year - 1]

    print(f"Pulling NFL results for {seasons} "
          f"({'CACHED' if args.no_refresh else 'fresh from nflverse'}) …")
    results = build_results_index(seasons, force_refresh=not args.no_refresh)
    source_asof = now_iso()
    print(f"  {len(results):,} index keys across {len(seasons)} season(s)")

    if not results:
        raise SystemExit(
            "  The results index is EMPTY — nothing can be graded.\n"
            "  Either no game in these seasons has finished yet, or the "
            "schedule pull failed.\n"
            "  Grading is skipped entirely rather than writing rows against "
            "missing data.")
    print()

    print("=" * 64)
    print("  RESOLUTION")
    print("=" * 64)
    totals = {"new": 0, "resolved": 0, "void": 0, "open": 0}
    all_open: Dict[str, str] = {}
    for d in dirs:
        s = resolve_batch(d, results, source_asof)
        for k in totals:
            totals[k] += s[k]
        all_open.update(s.get("open_events", {}))
        print(f"  {d.parent.name}: {s['forecasts']:3d} forecasts · "
              f"{s['new']:3d} newly graded "
              f"({s['resolved']} resolved, {s['void']} void) · "
              f"{s['open']:3d} still open"
              + (f" · {s['already']} previously graded" if s["already"] else ""))

    print("-" * 64)
    print(f"  newly graded {totals['new']} "
          f"({totals['resolved']} resolved, {totals['void']} void), "
          f"{totals['open']} still open")
    if totals["open"]:
        print()
        print(f"  {len(all_open)} event(s) still open:")
        for ev, when in sorted(all_open.items(), key=lambda kv: kv[1]):
            print(f"    {ev:<44} kickoff {when}")
        print()
        print("  An open event is one the results feed does not yet show as "
              "final. If a game")
        print("  HAS finished and still shows open, the join failed rather "
              "than the data being")
        print("  late — check that the event_id matches nflverse's game_id. "
              "Re-run after they")
        print("  finish; grading is append-only, so nothing regrades.")
    print("\n  forecasts.jsonl was NOT modified — resolutions live beside it.")
    print("\n  Next:")
    print("    python -m ballknower_gridiron.ledger.score")
    print(f"    git add content/football && "
          f"git commit -m 'ledger: resolve {dt.date.today()}' && git push")
    print(f"\n  {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
