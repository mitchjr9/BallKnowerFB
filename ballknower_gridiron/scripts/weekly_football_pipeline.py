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
from ballknower_gridiron.data.season_calendar import current_nfl_season
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


# The configuration the calibration work actually validated. Anything else runs,
# but says so loudly — a slate published under a fallback config is a slate whose
# probabilities were never checked.
PRODUCTION_WP_VERSION = "v3.1"
PRODUCTION_BLEND_ELO = 0.00


def _week_label(preds: List[GamePrediction]) -> Optional[str]:
    """
    Human label for the slate, honest about multi-week spans.

    One week -> "2026 Week 1". Two or more -> "2026 Weeks 1-2", so a reader who
    sees a team twice can tell immediately that they are looking at two weeks of
    football rather than a broken model.
    """
    pairs = sorted({(p.season, p.week) for p in preds
                    if p.season is not None and p.week is not None})
    if not pairs:
        return None
    season = pairs[0][0]
    weeks = sorted({w for s, w in pairs if s == season})
    if len(pairs) != len(weeks):          # more than one season in the slate
        return f"{pairs[0][0]}-{pairs[-1][0]}"
    if len(weeks) == 1:
        return f"{season} Week {weeks[0]}"
    return f"{season} Weeks {weeks[0]}-{weeks[-1]}"


def _print_config_banner(wp_version: Optional[str],
                         margin_version: Optional[str],
                         blend_elo_weight: Optional[float]) -> None:
    """
    Print the resolved configuration before predicting anything, and warn when
    it is not the validated production config.

    This exists because shell exports do not survive a new terminal session. A
    run that silently fell back to the settings defaults produced a full public
    slate under an unvalidated model, and nothing in the output made that
    obvious until someone read the newsletter header. Resolved config is now the
    first thing printed.
    """
    wp = wp_version or settings.active_model_version
    blend = (blend_elo_weight if blend_elo_weight is not None
             else settings.default_blend_elo)
    log.info("─" * 62)
    log.info("  RESOLVED CONFIG")
    log.info("    win probability : %s", wp)
    log.info("    margin          : %s", margin_version or "(none)")
    log.info("    ELO blend       : %.2f", blend)
    log.info("─" * 62)

    problems = []
    if wp.replace(".", "_") != PRODUCTION_WP_VERSION.replace(".", "_"):
        problems.append(
            f"win-probability model is {wp}, not the validated "
            f"{PRODUCTION_WP_VERSION}")
    if abs(blend - PRODUCTION_BLEND_ELO) > 1e-9:
        problems.append(
            f"ELO blend is {blend:.2f}, not the measured "
            f"{PRODUCTION_BLEND_ELO:.2f}")
    if problems:
        log.warning("  ⚠ NOT the production configuration:")
        for pr in problems:
            log.warning("      - %s", pr)
        log.warning("    Env vars do not persist across terminal sessions — put")
        log.warning("    NFL_MODEL_VERSION and NFL_BLEND_ELO_DEFAULT in .env so")
        log.warning("    every run picks them up. Continuing anyway.")
        log.warning("─" * 62)


class StaleStateError(RuntimeError):
    """The model state has not absorbed games that have already finished."""


def _state_context(wp_version: str, margin_version: Optional[str],
                   slate_season: int, allow_stale: bool
                   ) -> Tuple[str, Dict[str, int], Dict]:
    """
    Return (asof_ts, {team: games absorbed this season}, provenance).

    Both numbers come from the model STATE, not from a separate read of the
    schedule. That coupling is the point. `data_depth` exists to say how much
    this season the ratings reflect — so it must count games the ratings have
    actually absorbed, not games that happen to have been played. Counting from
    the schedule while serving frozen ratings would lift the tier cap to
    "Strong" on ratings that know nothing about the season. `asof_ts` is the
    moment the state was built, which is exactly the information set the
    forecast was made from.

    Refuses to proceed if a finished game hasn't been absorbed, because every
    forecast built on that state would silently ignore it.
    """
    from ballknower_gridiron.data.football_loader import load_nfl_games
    from ballknower_gridiron.scripts.refresh_state import (
        read_state_meta, unabsorbed_games,
    )

    completed = load_nfl_games(seasons_back=1, completed_only=True)
    completed = completed[completed["season"].astype(int) == int(slate_season)]

    stale = {}
    for v in filter(None, [wp_version, margin_version]):
        behind = unabsorbed_games(v, completed)
        if not behind.empty:
            stale[v] = behind

    if stale:
        lines = [f"{v}: {len(b)} finished game(s) not in its ratings "
                 f"(latest {pd.to_datetime(b['game_date']).max().date()})"
                 for v, b in stale.items()]
        msg = ("Model state is behind the results:\n      "
               + "\n      ".join(lines)
               + "\n    Run first:  python -m ballknower_gridiron.scripts.refresh_state")
        if not allow_stale:
            raise StaleStateError(msg)
        log.warning("⚠ %s", msg)
        log.warning("  Continuing because --allow-stale-state was passed. These "
                    "forecasts will ignore those games.")

    meta = read_state_meta(wp_version)
    if meta:
        asof_ts = meta["refreshed_at"]
        games_played = {k: int(v) for k, v in meta.get("games_played", {}).items()}
        provenance = {"state_refreshed_at": meta["refreshed_at"],
                      "state_last_game": meta.get("last_game_absorbed"),
                      "weights_sha256": meta.get("weights_sha256", {})}
    else:
        # Never refreshed: the state is the training state. Its information set
        # ends at the last training game, and it has absorbed zero games of any
        # season after that — which keeps data_depth honest at "none".
        from ballknower_gridiron.scripts.refresh_state import (
            load_bundle, training_cutoff,
        )
        cutoff = training_cutoff(load_bundle(wp_version))
        asof_ts = (cutoff.tz_localize("UTC").isoformat() if cutoff is not None
                   else now_utc_iso())
        games_played = {}
        provenance = {"state_refreshed_at": None,
                      "state_last_game": str(cutoff.date()) if cutoff is not None else None,
                      "weights_sha256": {}}
    return asof_ts, games_played, provenance


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
    single_week: bool = True,
    week: Optional[int] = None,
    allow_stale_state: bool = False,
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
        Defaults to settings.default_blend_elo.
    single_week
        Restrict the slate to one NFL week (default). A day-count window can
        straddle a week boundary and pull in next week's Thursday game.
    week
        Force a specific week number instead of the earliest in the window.
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

    # ---- scope to ONE NFL week ------------------------------------------
    # A day-count window does not respect week boundaries. With days_ahead
    # large enough to reach the following Thursday, next week's kickoff game
    # lands in a slate headed "Week 1" — and because that game involves teams
    # who also played in Week 1, a reader sees the same team listed twice
    # against different opponents and reasonably concludes the model is broken.
    # The games are real; the SCOPE was wrong. Default to the earliest week in
    # the window, which is the one about to be played.
    if single_week and "week" in schedule.columns:
        weeks_in_window = sorted({int(w) for w in schedule["week"].dropna().unique()})
        seasons_in_window = sorted({int(s) for s in schedule["season"].dropna().unique()})
        target_week = int(week) if week is not None else weeks_in_window[0]
        target_season = seasons_in_window[0]
        before = len(schedule)
        schedule = schedule[
            (schedule["week"].astype(int) == target_week)
            & (schedule["season"].astype(int) == target_season)
        ].reset_index(drop=True)
        if len(schedule) != before:
            dropped = before - len(schedule)
            log.info(
                "Scoped slate to %d Week %d: kept %d game(s), dropped %d "
                "belonging to other week(s) %s. Pass --all-weeks to keep them.",
                target_season, target_week, len(schedule), dropped,
                [w for w in weeks_in_window if w != target_week])
        if schedule.empty:
            log.warning("No games left after scoping to week %s.", target_week)
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
    asof_ts, games_played, state_prov = _state_context(
        wp_version, margin_version,
        slate_seasons[0] if slate_seasons else current_nfl_season(),
        allow_stale_state)
    log.info("  model state as of : %s (last game absorbed: %s)",
             state_prov["state_refreshed_at"] or "training — never refreshed",
             state_prov["state_last_game"])

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
        pred.state_prov = state_prov
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
    single_week: bool = True,
    week: Optional[int] = None,
    output_dir: Optional[Path] = None,
    formats: Tuple[str, ...] = ("md", "html"),
    allow_stale_state: bool = False,
) -> dict:
    """
    Run the full weekly pipeline and write output files. Returns paths
    of written files keyed by format.
    """
    _print_config_banner(wp_version, margin_version, blend_elo_weight)

    preds = predict_upcoming_slate(
        wp_version=wp_version,
        margin_version=margin_version,
        days_ahead=days_ahead,
        blend_elo_weight=blend_elo_weight,
        single_week=single_week,
        week=week,
        allow_stale_state=allow_stale_state,
    )

    today = date_cls.today().isoformat()
    out_dir = Path(output_dir) if output_dir else (
        settings.project_root / "content" / "football" / today
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing newsletter outputs to %s", out_dir)

    # Determine week label from predictions (most common week in the slate)
    # Label from the FULL set of weeks present, not the modal one. Taking the
    # most common week silently mislabels a slate that spans a boundary — the
    # exact failure that put a Week 2 game under a "Week 1" heading.
    week_label = _week_label(preds)

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
        # Which state produced this forecast. Weights alone don't identify a
        # forecast any more — the same validated weights serve a different
        # state every week — so the ledger needs both.
        d["state_refreshed_at"] = p.state_prov.get("state_refreshed_at")
        d["state_last_game"] = p.state_prov.get("state_last_game")
        d["weights_sha256"] = p.state_prov.get("weights_sha256")
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
        "--all-weeks", action="store_true",
        help="Keep every week the date window touches. Default is to scope the "
             "slate to a single NFL week so a Week 2 game never lands in a "
             "Week 1 newsletter.",
    )
    parser.add_argument(
        "--allow-stale-state", action="store_true",
        help="Generate even if finished games haven't been folded into the "
             "ratings. The default is to refuse, because every forecast would "
             "silently ignore those games. Run refresh_state instead.",
    )
    parser.add_argument(
        "--week", type=int, default=None,
        help="Force a specific week number instead of the earliest in the window.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["md", "html"],
        choices=["md", "html"],
        help="Output formats to generate (default: both).",
    )
    args = parser.parse_args(argv)

    margin_version = None if args.no_margins else (args.margin_version or None)

    log.info("=== BallKnower Gridiron: Weekly Pipeline ===")
    try:
        return _main_run(args, margin_version)
    except StaleStateError as exc:
        print(f"\n  ✗ REFUSED — {exc}\n")
        return 1


def _main_run(args, margin_version) -> int:
    log.info("Disclaimer: outputs are for entertainment & educational use only.")
    written = run_weekly_pipeline(
        wp_version=args.wp_version,
        margin_version=margin_version,
        days_ahead=args.days_ahead,
        blend_elo_weight=args.blend_elo,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        single_week=not args.all_weeks,
        week=args.week,
        allow_stale_state=args.allow_stale_state,
    )
    for fmt, path in written.items():
        log.info("  [%s] %s", fmt, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
