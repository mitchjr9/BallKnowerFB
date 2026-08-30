"""
ballknower_gridiron.scripts.train_football_model
================================================

CLI to train an NFL win-probability model and save the bundle to disk.

Examples
--------
    # Train v2 (default — QB rating layer) with the last 12 seasons
    python -m ballknower_gridiron.scripts.train_football_model

    # Explicitly choose a version
    python -m ballknower_gridiron.scripts.train_football_model --version v3

    # Force a fresh data pull (skip caches)
    python -m ballknower_gridiron.scripts.train_football_model --refresh-data

    # Train on fewer seasons (faster iteration)
    python -m ballknower_gridiron.scripts.train_football_model --seasons-back 6

The bundle is saved under:
    models_artifacts/football/         (v1)
    models_artifacts/football/v2/
    models_artifacts/football/v3/

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make this script runnable both as a module (-m) and as a direct file.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ballknower_gridiron.config.settings import settings  # noqa: E402
from ballknower_gridiron.data.football_loader import load_nfl_games  # noqa: E402
from ballknower_gridiron.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("train_football_model")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train a BallKnower Gridiron NFL win-probability model.",
    )
    ap.add_argument(
        "--version",
        choices=["v1", "v2", "v3", "v3.1", "v3_1", "v4", "v4.1", "v4_1", "v5"],
        default=settings.active_model_version,
        help="Model version to train (default: %(default)s).",
    )
    ap.add_argument(
        "--seasons-back",
        type=int,
        default=settings.nfl_seasons_back,
        help="How many recent seasons of history to use (default: %(default)s).",
    )
    ap.add_argument(
        "--include-playoffs",
        action="store_true",
        default=True,
        help="Include postseason games in training (default: True).",
    )
    ap.add_argument(
        "--no-playoffs",
        dest="include_playoffs",
        action="store_false",
        help="Train on regular-season games only.",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force fresh nflverse pulls (ignore on-disk caches).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    settings.ensure_dirs()
    log.info("=" * 72)
    log.info(
        "Training NFL %s on last %d seasons (playoffs=%s, refresh=%s)",
        args.version, args.seasons_back, args.include_playoffs, args.refresh_data,
    )
    log.info("=" * 72)

    games = load_nfl_games(
        seasons_back=args.seasons_back,
        include_playoffs=args.include_playoffs,
        completed_only=True,
        force_refresh=args.refresh_data,
    )
    if games.empty:
        log.error("No games loaded — aborting.")
        return 2
    log.info("Loaded %d completed NFL games for training.", len(games))

    # Accept either dotted CLI form ("v3.1") or filesystem-safe form ("v3_1").
    version = args.version.lower()
    normalized = version.replace(".", "_")
    if normalized == "v1":
        from ballknower_gridiron.models.football_model import train_nfl_model
        model = train_nfl_model(games)
    elif normalized == "v2":
        from ballknower_gridiron.models.football_model_v2 import train_nfl_model_v2
        model = train_nfl_model_v2(games, force_refresh_qb_stats=args.refresh_data)
    elif normalized == "v3":
        from ballknower_gridiron.models.football_model_v3 import train_nfl_model_v3
        model = train_nfl_model_v3(
            games,
            force_refresh_qb_stats=args.refresh_data,
            force_refresh_team_metrics=args.refresh_data,
        )
    elif normalized == "v3_1":
        from ballknower_gridiron.models.football_model_v3_1 import train_nfl_model_v3_1
        model = train_nfl_model_v3_1(
            games,
            force_refresh_qb_stats=args.refresh_data,
            force_refresh_team_metrics=args.refresh_data,
        )
    elif normalized == "v4":
        from ballknower_gridiron.models.football_model_v4 import train_nfl_model_v4
        model = train_nfl_model_v4(
            games,
            force_refresh_qb_stats=args.refresh_data,
            force_refresh_team_metrics=args.refresh_data,
        )
    elif normalized == "v4_1":
        from ballknower_gridiron.models.football_model_v4_1 import train_nfl_model_v4_1
        model = train_nfl_model_v4_1(
            games,
            force_refresh_qb_stats=args.refresh_data,
            force_refresh_team_metrics=args.refresh_data,
        )
    elif normalized == "v5":
        from ballknower_gridiron.models.football_model_v5 import train_nfl_spread_v5
        model = train_nfl_spread_v5(
            games,
            force_refresh_qb_stats=args.refresh_data,
            force_refresh_team_metrics=args.refresh_data,
        )
    else:
        log.error("Unknown version: %s (valid: v1, v2, v3, v3.1, v4, v4.1, v5)", version)
        return 2

    out_dir = settings.models_dir_for(version)
    model.save(out_dir)

    log.info("=" * 72)
    log.info("DONE — %s saved to %s", version, out_dir)
    for k, v in model.metrics.items():
        log.info("  %-15s %s", k, v)
    log.info("=" * 72)
    print(f"\n✓ Trained NFL {version}. Metrics:")
    for k, v in model.metrics.items():
        print(f"    {k:<15} {v}")
    print(f"\nBundle: {out_dir}")
    print(f"\n{settings.__class__.__name__} note: {model.version} is active "
          f"if NFL_MODEL_VERSION={version!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
