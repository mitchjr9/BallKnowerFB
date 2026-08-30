"""
ballknower_gridiron.scripts.weekly_football_pipeline
====================================================

The weekly NFL newsletter-content generator. Pulls the upcoming game
slate, runs V3.1 for calibrated win probabilities + V5 for predicted
margins, and writes a complete markdown + HTML newsletter to
`content/football/<YYYY-MM-DD>/`.

NFL-specific notes vs. the basketball daily pipeline:

  * **Weekly cadence, not daily.** NFL plays Thurs/Sat/Sun/Mon. Default
    horizon is 10 days to capture the whole week including TNF and MNF
    of both this week and next week's TNF.
  * **Two models, not one.** V3.1 is the win-probability classifier;
    V5 is the spread/margin regressor. Both are loaded; V5 is optional
    and the pipeline degrades gracefully without it.
  * **Vegas spread comparison.** When V5 outputs a margin and the
    schedule has a `spread_line`, the newsletter includes a "Where We
    Disagree with Vegas Most" section. This is editorial context, NOT
    a betting recommendation.

Usage
-----
    # Standard weekly run (V3.1 wins + V5 margins, 10-day horizon)
    python -m ballknower_gridiron.scripts.weekly_football_pipeline

    # Custom model versions
    python -m ballknower_gridiron.scripts.weekly_football_pipeline \\
        --wp-version v3.1 --margin-version v5 --days-ahead 14

    # Skip V5 entirely (only win probabilities)
    python -m ballknower_gridiron.scripts.weekly_football_pipeline --no-margins

    # Specify output format
    python -m ballknower_gridiron.scripts.weekly_football_pipeline --formats md html

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.data.football_loader import get_upcoming_games
from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model
from ballknower_gridiron.utils.content_utils import (
    GamePrediction,
    render_blog,
    render_full_newsletter,
    render_html_newsletter,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger("weekly_football_pipeline")


def _elo_baseline_probability(elo_home: float, elo_away: float, hca: float) -> float:
    diff = (elo_home + hca) - elo_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _season_context(slate_seasons: List[int]) -> Tuple[str, Dict[str, int]]:
    """
    Return (asof_ts, {team: completed games this season}).

    `asof_ts` is the kickoff of the most recent COMPLETED game anywhere in the
    schedule — the honest answer to "what is the newest result this forecast
    could possibly have been built from". In Week 1 that is last season's Super
    Bowl, which is correct and worth seeing in the ledger rather than papering
    over with `now()`.

    The games-played count is taken from the schedule rather than from model
    internals, because it has to describe the *slate's* season specifically:
    a team with 17 games last year and 0 this year has `data_depth == "none"`,
    and reading a counter that still holds last season's tally is exactly the
    bug that let a sibling package publish six Locks in a week where nobody had
    played.
    """
    from ballknower_gridiron.data.football_loader import load_nfl_games

    games_played: Dict[str, int] = {}
    asof_ts = now_utc_iso()
    try:
        hist = load_nfl_games(seasons_back=2, completed_only=True)
    except Exception as exc:  # noqa: BLE001 — no history is survivable here
        log.warning("Could not load completed games for asof/depth (%s); "
                    "falling back to now() and depth=0.", exc)
        return asof_ts, games_played

    if not hist.empty:
        latest = pd.to_datetime(hist["game_date"]).max()
        if pd.notna(latest):
            asof_ts = latest.tz_localize("UTC").isoformat() \
                if latest.tzinfo is None else latest.isoformat()

        if slate_seasons:
            cur = hist[hist["season"].astype(int) == int(slate_seasons[0])]
            for col in ("home_team", "away_team"):
                for team, n in cur[col].value_counts().items():
                    games_played[str(team)] = games_played.get(str(team), 0) + int(n)

    return asof_ts, games_played


def now_utc_iso() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_load_margin_model(version: str):
    """Try to load V5; return None on failure (degrade gracefully)."""
    try:
        from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
        return NFLSpreadModelV5.load(settings.models_dir_for(version))
    except FileNotFoundError:
        log.warning("No trained %s bundle found — newsletter will show win "
                    "probabilities only (no margins).", version)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s margin model (%s) — continuing without margins.",
                    version, exc)
        return None


def predict_upcoming_slate(
    wp_version: Optional[str] = None,
    margin_version: Optional[str] = "v5",
    days_ahead: int = 10,
    blend_elo_weight: Optional[float] = None,
) -> List[GamePrediction]:
    """
    Run both models over the upcoming game slate.

    Parameters
    ----------
    wp_version
        Win-probability classifier version (defaults to settings.active_model_version,
        but V3.1 is recommended in production).
    margin_version
        Margin regressor version. Pass None or "" to skip margin prediction.
    days_ahead
        Schedule horizon in days (10 covers a typical NFL week including TNF
        of the following week).
    blend_elo_weight
        Blend weight for the ELO baseline against the model's probability.
        Defaults to settings.default_blend_elo (0.50 for NFL).
    """
    wp_version = wp_version or settings.active_model_version
    blend_elo_weight = (
        blend_elo_weight if blend_elo_weight is not None
        else settings.default_blend_elo
    )

    log.info("Loading win-probability model: %s", wp_version)
    wp_model = load_active_nfl_model(version=wp_version)

    margin_model = None
    if margin_version:
        log.info("Loading margin model: %s", margin_version)
        margin_model = _safe_load_margin_model(margin_version)

    log.info("Fetching upcoming schedule (next %d days) …", days_ahead)
    try:
        schedule = get_upcoming_games(days_ahead=days_ahead)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch upcoming schedule: %s", exc)
        return []

    if schedule.empty:
        log.warning("No upcoming games in the next %d days.", days_ahead)
        return []

    # ---- season roll (train/serve skew fix) -------------------------------
    # Off-season ELO regression only fires inside update_game, which is only
    # called while fitting. A model trained through February and asked about
    # Week 1 would otherwise serve un-regressed end-of-last-season ratings,
    # while every training example it learned from had the 33% regression
    # applied at exactly this point in the calendar. Roll explicitly.
    slate_seasons = sorted({int(s) for s in schedule["season"].dropna().unique()})
    if slate_seasons:
        target_season = slate_seasons[0]
        if wp_model.elo.roll_to_season(target_season):
            log.info("Rolled %s ELO into season %d before serving.",
                     wp_version, target_season)
        if margin_model is not None:
            margin_model.elo.roll_to_season(target_season)

    # ---- asof_ts + games played this season -------------------------------
    # asof_ts is the latest data the model was permitted to see. Stamping it on
    # every row is what lets the ledger enforce the leakage invariant instead of
    # trusting that we ran the pipeline at a sensible time.
    asof_ts, games_played = _season_context(slate_seasons)

    log.info("Predicting %d upcoming games …", len(schedule))
    hca = settings.elo_hca
    predictions: List[GamePrediction] = []

    for row in schedule.itertuples(index=False):
        home = str(getattr(row, "home_team", "")).upper()
        away = str(getattr(row, "away_team", "")).upper()
        if not home or not away:
            continue
        game_date = pd.to_datetime(getattr(row, "game_date")).date()
        season = int(getattr(row, "season", 0)) or None
        week = int(getattr(row, "week", 1))
        is_playoff = bool(getattr(row, "is_playoff", False))
        is_international = bool(getattr(row, "is_international", False))
        is_divisional = bool(getattr(row, "div_game", False))
        home_rest = float(getattr(row, "home_rest", 7.0) or 7.0)
        away_rest = float(getattr(row, "away_rest", 7.0) or 7.0)
        spread_line = getattr(row, "spread_line", np.nan)
        spread_line = float(spread_line) if pd.notna(spread_line) else None
        game_id = getattr(row, "game_id", None)
        game_id = str(game_id) if game_id is not None and pd.notna(game_id) else None

        # Prefer the real kickoff instant over the date. nflverse carries
        # `gametime` (ET clock) alongside `game_date`; without it the ledger
        # would compare a Sunday-morning commit against midnight and wrongly
        # accept a row for a 1pm game.
        kickoff_ts = _kickoff_iso(row, game_date)

        # ---- V3.1: win probability ----
        try:
            p_model, _ = wp_model.predict_proba(
                home_team=home, away_team=away,
                game_date=game_date, season=season, week=week,
                home_rest=home_rest, away_rest=away_rest,
                is_playoff=is_playoff,
                is_international=is_international,
                is_div_game=is_divisional,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("WP prediction failed for %s @ %s: %s", away, home, exc)
            continue

        # ELO baseline + blend
        elo_home = wp_model.elo.get_rating(home)
        elo_away = wp_model.elo.get_rating(away)
        p_elo = _elo_baseline_probability(elo_home, elo_away, hca)
        p_blend = (1.0 - blend_elo_weight) * p_model + blend_elo_weight * p_elo

        # Optional QB rating context
        extras = {}
        if hasattr(wp_model, "get_qb_rating"):
            try:
                extras["qb_rating_home"] = wp_model.get_qb_rating(home)
                extras["qb_rating_away"] = wp_model.get_qb_rating(away)
            except Exception:  # noqa: BLE001
                pass

        pred = GamePrediction.from_probs(
            game_date=game_date.isoformat(),
            home_team=home, away_team=away,
            p_blended=p_blend, p_model=p_model, p_elo=p_elo,
            elo_home=elo_home, elo_away=elo_away,
            is_playoff=is_playoff,
            season=season, week=week,
            is_divisional=is_divisional, is_international=is_international,
            **extras,
        )
        pred.asof_ts = asof_ts
        pred.event_start_ts = kickoff_ts
        pred.game_id = game_id

        # How much played football backs the thinner-resumed side. Governs the
        # ledger's data_depth, which caps how confident a tier may be — in
        # Week 1 nobody has played, so nothing should publish as a Lock.
        pred.attach_depth(min(games_played.get(home, 0),
                              games_played.get(away, 0)))

        # Feature values behind the pick, so the blog can explain WHY.
        if hasattr(wp_model, "get_team_metrics"):
            try:
                tm_h = wp_model.get_team_metrics(home)
                tm_a = wp_model.get_team_metrics(away)
                pred.drivers["net_epa_diff"] = float(
                    tm_h.get("net_epa_per_play", 0.0)
                    - tm_a.get("net_epa_per_play", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("No team metrics for %s/%s: %s", home, away, exc)

        # ---- V5: margin prediction (optional) ----
        if margin_model is not None:
            try:
                margin = margin_model.predict_margin(
                    home_team=home, away_team=away,
                    game_date=game_date, season=season, week=week,
                    home_rest=home_rest, away_rest=away_rest,
                    is_playoff=is_playoff,
                    is_international=is_international,
                    is_div_game=is_divisional,
                )
                pred.attach_margin(margin, spread_line)
            except Exception as exc:  # noqa: BLE001
                log.warning("Margin prediction failed for %s @ %s: %s", away, home, exc)

        predictions.append(pred)

    return predictions


def run_weekly_pipeline(
    wp_version: Optional[str] = None,
    margin_version: Optional[str] = "v5",
    days_ahead: int = 10,
    blend_elo_weight: Optional[float] = None,
    output_dir: Optional[Path] = None,
    formats: Tuple[str, ...] = ("md", "html"),
) -> dict:
    """
    Run the full weekly pipeline and write output files. Returns paths
    of written files keyed by format.
    """
    preds = predict_upcoming_slate(
        wp_version=wp_version,
        margin_version=margin_version,
        days_ahead=days_ahead,
        blend_elo_weight=blend_elo_weight,
    )

    today = date_cls.today().isoformat()
    out_dir = Path(output_dir) if output_dir else (
        settings.project_root / "content" / "football" / today
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing newsletter outputs to %s", out_dir)

    # Determine week label from predictions (most common week in the slate)
    week_label = None
    if preds:
        weeks = [p.week for p in preds if p.week is not None]
        seasons = [p.season for p in preds if p.season is not None]
        if weeks and seasons:
            most_common_week = max(set(weeks), key=weeks.count)
            most_common_season = max(set(seasons), key=seasons.count)
            week_label = f"{most_common_season} Week {most_common_week}"

    actual_wp = wp_version or settings.active_model_version
    actual_blend = (blend_elo_weight if blend_elo_weight is not None
                    else settings.default_blend_elo)

    md = render_full_newsletter(
        preds,
        title_date=today,
        wp_version=actual_wp,
        margin_version=margin_version,
        blend_elo_weight=actual_blend,
        week_label=week_label,
    )

    blog = render_blog(preds, date_label=today, week_label=week_label)

    written = {}
    if "md" in formats:
        md_path = out_dir / "newsletter.md"
        md_path.write_text(md, encoding="utf-8")
        written["md"] = md_path
        log.info("Wrote markdown -> %s", md_path)

        if blog:
            blog_md = out_dir / "blog.md"
            blog_md.write_text(blog, encoding="utf-8")
            written["blog_md"] = blog_md
            log.info("Wrote blog markdown -> %s", blog_md)

    if "html" in formats:
        html_path = out_dir / "newsletter.html"
        html_path.write_text(render_html_newsletter(md), encoding="utf-8")
        written["html"] = html_path
        log.info("Wrote HTML -> %s", html_path)

        if blog:
            blog_html = out_dir / "blog.html"
            blog_html.write_text(render_html_newsletter(blog), encoding="utf-8")
            written["blog_html"] = blog_html
            log.info("Wrote blog HTML -> %s", blog_html)

    # Always dump JSON for downstream automation — this file is the ledger's
    # input, so it carries the run-level context each row needs: which model
    # produced it, at what blend, and the kickoff timestamp the pre-registration
    # invariant is checked against.
    json_path = out_dir / "predictions.json"
    payload = []
    for p in preds:
        d = asdict(p)
        d["wp_version"] = actual_wp
        d["margin_version"] = margin_version
        d["blend_elo_weight"] = actual_blend
        # event_start_ts / game_id already live on the record (set at predict
        # time from the schedule row), so no re-join is needed here.
        payload.append(d)
    json_path.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")
    written["json"] = json_path
    log.info("Wrote JSON -> %s", json_path)

    return written


def _kickoff_iso(row, game_date) -> str:
    """
    Best available kickoff instant as an ISO-8601 UTC string.

    nflverse schedules carry `gametime` as an ET wall-clock string ("13:00").
    Combining it with `game_date` gives the real kickoff; without it we fall
    back to 17:00 UTC (noon ET), which is early enough to be conservative —
    the ledger will refuse a borderline row rather than accept one it shouldn't.
    """
    import pandas as _pd
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    gametime = getattr(row, "gametime", None)
    if gametime is not None and _pd.notna(gametime) and str(gametime).strip():
        try:
            hh, mm = str(gametime).strip().split(":")[:2]
            # ET -> UTC. NFL regular season runs through the DST switch in
            # early November, so the offset is not constant: EDT is UTC-4,
            # EST is UTC-5. Approximate by date rather than assuming one.
            month = game_date.month
            offset = 4 if 3 <= month <= 10 else 5
            naive = _dt(game_date.year, game_date.month, game_date.day,
                        int(hh), int(mm))
            return (naive + _td(hours=offset)).replace(tzinfo=_tz.utc).isoformat()
        except (ValueError, TypeError):
            pass
    return _dt(game_date.year, game_date.month, game_date.day, 17, 0,
               tzinfo=_tz.utc).isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a weekly NFL newsletter from V3.1 + V5 outputs."
    )
    parser.add_argument(
        "--wp-version", default=None,
        help="Win-probability classifier version (default: from NFL_MODEL_VERSION env "
             "or v3.1 if set as active).",
    )
    parser.add_argument(
        "--margin-version", default="v5",
        help="Margin regressor version (default: v5). Pass empty string or use "
             "--no-margins to skip.",
    )
    parser.add_argument(
        "--no-margins", action="store_true",
        help="Skip margin predictions entirely — win probabilities only.",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=10,
        help="Schedule horizon — predict games in the next N days (default: 10).",
    )
    parser.add_argument(
        "--blend-elo", type=float, default=None,
        help="ELO blend weight (default: NFL_BLEND_ELO_DEFAULT env, currently %.2f)."
             % settings.default_blend_elo,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["md", "html"],
        choices=["md", "html"],
        help="Output formats to generate (default: both).",
    )
    args = parser.parse_args(argv)

    margin_version = None if args.no_margins else (args.margin_version or None)

    log.info("=== BallKnower Gridiron: Weekly Pipeline ===")
    log.info("Disclaimer: outputs are for entertainment & educational use only.")
    written = run_weekly_pipeline(
        wp_version=args.wp_version,
        margin_version=margin_version,
        days_ahead=args.days_ahead,
        blend_elo_weight=args.blend_elo,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
    )
    for fmt, path in written.items():
        log.info("  [%s] %s", fmt, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
