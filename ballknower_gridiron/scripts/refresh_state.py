"""
ballknower_gridiron.scripts.refresh_state
=========================================

Fold completed games into the ratings, **without** touching the model.

    python -m ballknower_gridiron.scripts.refresh_state            # v3.1 + v5
    python -m ballknower_gridiron.scripts.refresh_state --check    # report only
    python -m ballknower_gridiron.scripts.refresh_state --versions v3.1

The problem this solves
-----------------------
A trained bundle carries two very different kinds of thing:

  * **weights** — the fitted XGBoost classifier/regressor, its scaler, and the
    isotonic calibrator. These were validated (ECE, Brier, blend sweep) and
    should change only in the offseason, deliberately.
  * **state** — ELO ratings, recent form, rolling team EPA, rolling QB ratings.
    These are *inputs*, and they should change after every game.

Until now the weekly pipeline served the state frozen at training time, so
Week 3's forecasts did not know Weeks 1 and 2 had happened. This script
replays every completed game through the same feature builder training uses,
and writes back only the state.

Why not just retrain weekly
---------------------------
Retraining would also bring the state current, but it refits the weights too:
it replaces the artifact whose calibration we measured, every week, and ledger
rows would carry an identical `model_version` / `config_hash` / `code_commit`
across those different artifacts, so nobody could tell which one produced a
forecast. Separating state from weights keeps the validated model fixed and
makes the one thing that changes weekly — the state — explicit and timestamped.

Integrity checks (a failure refuses to write)
--------------------------------------------
1. **Faithfulness.** Before trusting the full replay, the script replays only
   the games the model was trained on and compares the result to the ELO and
   form saved at training. They must match. A mismatch means the replay isn't
   reproducing training — most often because an ELO setting (K, home-field,
   off-season regression) changed in `.env` after the model was trained, in
   which case the refreshed ratings would be computed on a different scale
   from the one the model learned.
2. **Weights untouched.** Weight files are hashed before and after. The script
   only ever writes the four state files, and verifies it.

Every refresh backs up the previous state and writes `state_meta.json`, whose
`refreshed_at` becomes the `asof_ts` of the forecasts built from it.

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.data.season_calendar import current_nfl_season
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger("refresh_state")

# Versions whose state is built by NFLFeatureBuilderV3. v4/v4.1 use the v4
# builder (weather features) and are not production, so they are refused
# rather than refreshed with the wrong builder.
SUPPORTED = ("v3", "v3.1", "v5")
DEFAULT_VERSIONS = ("v3.1", "v5")

STATE_FILES = ("elo_state.json", "feature_state.json",
               "qb_ratings.json", "team_metrics.json")
WEIGHT_FILES = ("model.pkl", "regressor.pkl", "scaler.pkl", "metadata.json")

# Replaying the training games must reproduce the saved ELO to within this.
# Exact float reproduction is expected; the tolerance only absorbs nflverse's
# occasional retroactive score corrections, which move ratings by a point or so.
ELO_TOLERANCE = 1.0


class RefreshError(RuntimeError):
    """Raised when a refresh would write state that can't be trusted."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_hashes(bundle_dir: Path) -> Dict[str, str]:
    return {f: h for f in WEIGHT_FILES
            if (h := sha256_of(bundle_dir / f)) is not None}


def load_bundle(version: str):
    v = version.lower().replace("_", ".")
    if v == "v5":
        from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
        return NFLSpreadModelV5.load(settings.models_dir_for(v))
    from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model
    return load_active_nfl_model(v)


def training_cutoff(bundle) -> Optional[pd.Timestamp]:
    """Latest game the saved state had absorbed — i.e. the end of training."""
    dates = [d for d in (bundle.last_game_date or {}).values() if d]
    return pd.to_datetime(max(dates)) if dates else None


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def default_builder_factory(games: pd.DataFrame):
    """Build a V3 feature builder with its rolling QB and team-EPA tables loaded.

    Uses the in-progress-season cache rule, so the current season's play-by-play
    and weekly QB stats are re-pulled rather than read from a stale copy.
    """
    from ballknower_gridiron.models.football_model_v3 import NFLFeatureBuilderV3
    fb = NFLFeatureBuilderV3()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games)
    fb.load_rolling_team_metrics(seasons, games)
    return fb


def replay(games: pd.DataFrame, builder_factory: Callable
           ) -> Tuple[object, object]:
    """
    Return (loaded_builder_template, fully_replayed_builder).

    Loads the data tables once, then deep-copies the loaded-but-unreplayed
    builder so the faithfulness replay and the full replay start from the same
    point without downloading everything twice.
    """
    template = builder_factory(games)
    try:
        full = copy.deepcopy(template)
    except Exception as exc:  # noqa: BLE001 — correctness over speed
        # Some loaded tables may not deep-copy cleanly. Rebuilding costs a
        # second data load but can't produce a different answer.
        log.warning("Builder did not deep-copy (%s); reloading instead.", exc)
        template = builder_factory(games)
        full = builder_factory(games)
    full.build_training_frame(games)
    return template, full


def check_faithfulness(template, games: pd.DataFrame, bundle) -> Dict:
    """
    Replay only the games the model was trained on and compare to the saved
    state. Raises RefreshError on a mismatch.
    """
    cutoff = training_cutoff(bundle)
    if cutoff is None:
        raise RefreshError("Bundle has no last_game_date — cannot locate the "
                           "training cutoff to verify the replay against.")
    train_games = games[pd.to_datetime(games["game_date"]) <= cutoff]
    try:
        fb = copy.deepcopy(template)
    except Exception:  # noqa: BLE001
        fb = default_builder_factory(games)
    fb.build_training_frame(train_games)

    saved = bundle.elo.ratings
    replayed = fb.elo.ratings
    teams = sorted(set(saved) | set(replayed))
    diffs = {t: abs(float(replayed.get(t, 0.0)) - float(saved.get(t, 0.0)))
             for t in teams}
    worst_team = max(diffs, key=diffs.get) if diffs else None
    max_diff = diffs[worst_team] if worst_team else 0.0

    saved_form = {t: list(v) for t, v in (bundle.form_state or {}).items()}
    replay_form = {t: list(d) for t, d in fb._form.items()}
    form_match = saved_form == replay_form

    result = {"training_cutoff": str(cutoff.date()),
              "games_replayed": int(len(train_games)),
              "max_elo_diff": round(max_diff, 4),
              "worst_team": worst_team,
              "form_match": form_match}

    if max_diff > ELO_TOLERANCE or not form_match:
        raise RefreshError(
            "Replay does NOT reproduce the trained state "
            f"(max ELO diff {max_diff:.2f} on {worst_team}; "
            f"form {'matches' if form_match else 'differs'}).\n"
            "The refreshed ratings would be on a different scale from the one "
            "the model learned. Most likely an ELO setting in .env "
            "(NFL_ELO_K, NFL_ELO_HCA, NFL_ELO_SEASON_REG, NFL_MOV_MAX) changed "
            "after training, or NFL_SEASONS_BACK changed where history starts. "
            "Restore the training settings, or retrain deliberately.")
    return result


# --------------------------------------------------------------------------- #
# Write-back
# --------------------------------------------------------------------------- #
def apply_state(bundle, fb) -> None:
    """Copy the eight state fields from the replayed builder onto the bundle.

    Mirrors how train_nfl_model_v3 assembles them, field for field, so a
    refreshed bundle is indistinguishable from one trained at this moment —
    except that its weights are the validated ones.
    """
    bundle.elo = fb.elo
    bundle.form_state = {t: list(d) for t, d in fb._form.items()}
    bundle.last_game_date = {t: d.isoformat() for t, d in fb._last_game_date.items()}
    bundle.team_qb_ratings_latest = dict(fb.team_qb_ratings_latest)
    bundle.team_qb_starters_latest = dict(fb.team_qb_starters_latest)
    bundle.team_metrics_latest = dict(fb.team_metrics_latest)
    bundle.latest_season = fb.latest_season
    prior_key = (fb.latest_season - 1) if fb.latest_season else 0
    bundle.team_metrics_prior = dict(fb.end_of_season_by_team.get(prior_key, {}))


def games_played_by_team(games: pd.DataFrame, season: int) -> Dict[str, int]:
    cur = games[games["season"].astype(int) == int(season)]
    out: Dict[str, int] = {}
    for col in ("home_team", "away_team"):
        for team, n in cur[col].value_counts().items():
            out[str(team)] = out.get(str(team), 0) + int(n)
    return dict(sorted(out.items()))


def write_state(bundle, bundle_dir: Path, meta: Dict) -> Dict[str, str]:
    """
    Write only the state files, verify the weights didn't change, back up the
    previous state. Returns the (unchanged) weight hashes.
    """
    before = weight_hashes(bundle_dir)

    # Serialise through the bundle's own save() into a scratch dir, then copy
    # ONLY the state files across. Reusing save() means the state format can't
    # drift from what load() expects; copying selectively means the weight
    # files on disk are never opened for writing at all.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        bundle.save(tmp_dir)

        backup = bundle_dir / "state_backup" / datetime.now(
            timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup.mkdir(parents=True, exist_ok=True)
        for f in STATE_FILES + ("state_meta.json",):
            if (bundle_dir / f).exists():
                shutil.copy2(bundle_dir / f, backup / f)

        for f in STATE_FILES:
            shutil.copy2(tmp_dir / f, bundle_dir / f)

    after = weight_hashes(bundle_dir)
    if before != after:
        raise RefreshError(f"Weight files changed during a state refresh in "
                           f"{bundle_dir} — this must never happen. "
                           f"before={before} after={after}")

    meta = {**meta, "weights_sha256": after}
    (bundle_dir / "state_meta.json").write_text(json.dumps(meta, indent=2))
    return after


# --------------------------------------------------------------------------- #
# Staleness (also used by the weekly pipeline's guard)
# --------------------------------------------------------------------------- #
def read_state_meta(version: str) -> Optional[Dict]:
    p = settings.models_dir_for(version) / "state_meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def unabsorbed_games(version: str, completed: pd.DataFrame) -> pd.DataFrame:
    """
    Completed games the bundle's state has NOT absorbed yet.

    Falls back to the training cutoff for a bundle that has never been
    refreshed, so a legacy bundle reads as stale the moment a game finishes.
    """
    meta = read_state_meta(version)
    if meta and meta.get("last_game_absorbed"):
        cutoff = pd.to_datetime(meta["last_game_absorbed"])
    else:
        cutoff = training_cutoff(load_bundle(version))
    if cutoff is None:
        return completed
    return completed[pd.to_datetime(completed["game_date"]) > cutoff]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_completed_games(first_season: int) -> pd.DataFrame:
    """All completed games from `first_season` through the current season."""
    from ballknower_gridiron.data.football_loader import load_nfl_games
    seasons_back = current_nfl_season() - int(first_season) + 1
    games = load_nfl_games(seasons_back=seasons_back, completed_only=True)
    games = games[games["season"].astype(int) >= int(first_season)]
    return games.sort_values("game_date").reset_index(drop=True)


def refresh(versions: List[str], builder_factory: Callable = default_builder_factory,
            games: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
    bundles = {v: load_bundle(v) for v in versions}

    # Start the replay where training started, so the ELO trajectory is the
    # one the model learned. Training pulled NFL_SEASONS_BACK seasons ending at
    # the bundle's latest season.
    ref = bundles[versions[0]]
    first_season = int(ref.latest_season) - settings.nfl_seasons_back + 1

    if games is None:
        games = load_completed_games(first_season)
    log.info("Replaying %d completed games (%s → %s) …", len(games),
             pd.to_datetime(games["game_date"]).min().date(),
             pd.to_datetime(games["game_date"]).max().date())

    template, full = replay(games, builder_factory)
    season = current_nfl_season()
    last_absorbed = pd.to_datetime(games["game_date"]).max()
    refreshed_at = now_iso()

    results: Dict[str, Dict] = {}
    for v, bundle in bundles.items():
        faith = check_faithfulness(template, games, bundle)
        # Baseline for "movement this season" is last season's ratings AFTER
        # the off-season regression — the ratings the season actually started
        # from. Comparing against the un-regressed end-of-season numbers would
        # fold the regression toward 1500 into "movement" and show a 3-0 team
        # as having dropped.
        start = copy.deepcopy(bundle.elo)
        start.roll_to_season(season)
        old_ratings = dict(start.ratings)
        apply_state(bundle, full)
        meta = {
            "version": v,
            "refreshed_at": refreshed_at,
            "last_game_absorbed": str(last_absorbed.date()),
            "games_absorbed": int(len(games)),
            "replay_first_season": first_season,
            "season": season,
            "games_played": games_played_by_team(games, season),
            "faithfulness": faith,
        }
        hashes = write_state(bundle, settings.models_dir_for(v), meta)
        results[v] = {"meta": meta, "hashes": hashes,
                      "old_ratings": old_ratings,
                      "new_ratings": dict(bundle.elo.ratings)}
    return results


def print_report(results: Dict[str, Dict]) -> None:
    print()
    print("=" * 66)
    print("  STATE REFRESH")
    print("=" * 66)
    for v, r in results.items():
        m = r["meta"]
        f = m["faithfulness"]
        print(f"  {v}")
        print(f"    faithfulness : replayed {f['games_replayed']} training games → "
              f"max ELO diff {f['max_elo_diff']:.3f}, "
              f"form {'matches' if f['form_match'] else 'DIFFERS'}  ✓")
        print(f"    absorbed     : {m['games_absorbed']} games through "
              f"{m['last_game_absorbed']}")
        print(f"    weights      : unchanged "
              f"({', '.join(k for k in r['hashes'] if k.endswith('.pkl'))})  ✓")
        print(f"    asof_ts      : {m['refreshed_at']}")

    # Who moved since training — the thing the frozen state was hiding.
    first = next(iter(results.values()))
    old, new = first["old_ratings"], first["new_ratings"]
    moves = sorted(((t, new[t] - old.get(t, new[t])) for t in new),
                   key=lambda x: x[1])
    gp = first["meta"]["games_played"]
    if moves and gp:
        print()
        print(f"  ELO movement this season (what the frozen state was hiding)")
        for t, d in moves[-3:][::-1]:
            print(f"    ↑ {t:<4} {d:+6.1f}   ({gp.get(t, 0)} games)")
        for t, d in moves[:3]:
            print(f"    ↓ {t:<4} {d:+6.1f}   ({gp.get(t, 0)} games)")
    print()
    print("  Next: generate the slate. It will now serve these ratings, and its")
    print("  asof_ts will be this refresh's timestamp.")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fold completed games into model state; weights stay frozen.")
    ap.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS),
                    help=f"Bundles to refresh (default: {' '.join(DEFAULT_VERSIONS)}).")
    ap.add_argument("--check", action="store_true",
                    help="Report how many completed games each bundle has not "
                         "absorbed yet, and write nothing.")
    args = ap.parse_args(argv)

    versions = [v.lower().replace("_", ".") for v in args.versions]
    bad = [v for v in versions if v not in SUPPORTED]
    if bad:
        print(f"Unsupported version(s) {bad}. The refresh replays through the V3 "
              f"feature builder, which serves {', '.join(SUPPORTED)} only.")
        return 2

    if args.check:
        from ballknower_gridiron.data.football_loader import load_nfl_games
        completed = load_nfl_games(seasons_back=1, completed_only=True)
        for v in versions:
            behind = unabsorbed_games(v, completed)
            meta = read_state_meta(v)
            since = meta["last_game_absorbed"] if meta else "training (never refreshed)"
            status = "current" if behind.empty else f"{len(behind)} game(s) behind"
            print(f"  {v:<5} state as of {since} — {status}")
        return 0

    try:
        results = refresh(versions)
    except RefreshError as exc:
        print(f"\n  ✗ REFRESH REFUSED — nothing was written.\n\n  {exc}\n")
        return 1
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
