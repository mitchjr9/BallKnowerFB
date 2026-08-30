"""
predict_nfl.py
==============

Quick command-line NFL game predictor. Loads trained ballknower_gridiron
models (V3.1 for win probability, V5 for margin) and prints calibrated
probabilities + predicted margin + ATS context for a single matchup.

Usage
-----
    # Basic — uses active win-probability model + V5 margins
    python predict_nfl.py <HOME> <AWAY>

    # Specify versions explicitly
    python predict_nfl.py KC BUF --wp-version v3.1 --margin-version v5

    # Playoff game (affects feature input)
    python predict_nfl.py KC BUF --playoff

    # Skip margin prediction (faster, no V5 needed)
    python predict_nfl.py KC BUF --no-margins

    # Compare against a specific Vegas spread (nflverse convention: +7 = home favored by 7)
    python predict_nfl.py KC BUF --spread +3.5

    # Set blend weight for ELO baseline
    python predict_nfl.py KC BUF --blend-elo 0.5

Team codes are standard NFL abbreviations (KC, BUF, BAL, SF, etc.) matching
those in your trained model's ELO state.

DISCLAIMER: For entertainment and educational purposes only — not financial
or betting advice.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as date_cls
from typing import Optional

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model


def _elo_baseline_probability(elo_home: float, elo_away: float, hca: float) -> float:
    diff = (elo_home + hca) - elo_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _tier(confidence: float) -> str:
    """Lock / Strong / Lean / Pass — same as the newsletter."""
    if confidence >= 0.50:
        return "🔒 Lock"
    if confidence >= 0.30:
        return "💪 Strong"
    if confidence >= 0.15:
        return "🎯 Lean"
    return "🪙 Pass"


def _pct(p: float) -> str:
    return f"{p * 100:5.1f}%"


def _safe_load_margin_model(version: str):
    try:
        from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
        return NFLSpreadModelV5.load(settings.models_dir_for(version))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict an NFL game using trained ballknower_gridiron models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("home", help="Home team abbreviation (e.g. KC)")
    parser.add_argument("away", help="Away team abbreviation (e.g. BUF)")
    parser.add_argument(
        "--wp-version", default=None,
        help="Win-probability model version (default: from NFL_MODEL_VERSION).",
    )
    parser.add_argument(
        "--margin-version", default="v5",
        help="Margin model version (default: v5). Use --no-margins to skip.",
    )
    parser.add_argument(
        "--no-margins", action="store_true",
        help="Skip margin prediction (only show win probabilities).",
    )
    parser.add_argument(
        "--playoff", action="store_true",
        help="Treat as a playoff game.",
    )
    parser.add_argument(
        "--divisional", action="store_true",
        help="Treat as a divisional matchup.",
    )
    parser.add_argument(
        "--international", action="store_true",
        help="Treat as an international game (London, Mexico City, etc.).",
    )
    parser.add_argument(
        "--blend-elo", type=float, default=None,
        help=f"ELO blend weight (default: {settings.default_blend_elo:.2f}).",
    )
    parser.add_argument(
        "--spread", type=float, default=None,
        help="Vegas spread for the home team (positive = home favored). "
             "If provided, shows ATS pick alongside margin prediction.",
    )
    parser.add_argument(
        "--home-rest", type=float, default=7.0,
        help="Home team days of rest (default: 7).",
    )
    parser.add_argument(
        "--away-rest", type=float, default=7.0,
        help="Away team days of rest (default: 7).",
    )
    args = parser.parse_args(argv)

    home = args.home.upper()
    away = args.away.upper()
    wp_version = args.wp_version or settings.active_model_version
    blend_elo_weight = (args.blend_elo if args.blend_elo is not None
                        else settings.default_blend_elo)

    try:
        wp_model = load_active_nfl_model(version=wp_version)
    except FileNotFoundError:
        print(f"\n✗ No trained {wp_version} model found at "
              f"{settings.models_dir_for(wp_version)}.")
        print(f"  Train it first:")
        print(f"    python -m ballknower_gridiron.scripts.train_football_model "
              f"--version {wp_version}\n")
        return 2
    except Exception as exc:
        print(f"\n✗ Could not load {wp_version}: {exc}\n")
        return 1

    margin_model = None
    if not args.no_margins:
        margin_model = _safe_load_margin_model(args.margin_version)

    try:
        p_model, _ = wp_model.predict_proba(
            home_team=home, away_team=away,
            game_date=date_cls.today(),
            home_rest=args.home_rest, away_rest=args.away_rest,
            is_playoff=args.playoff,
            is_international=args.international,
            is_div_game=args.divisional,
        )
    except Exception as exc:
        print(f"\n✗ Win-probability prediction failed: {exc}\n")
        return 1

    elo_home = wp_model.elo.get_rating(home)
    elo_away = wp_model.elo.get_rating(away)
    p_elo = _elo_baseline_probability(elo_home, elo_away, settings.elo_hca)
    p_blend = (1.0 - blend_elo_weight) * p_model + blend_elo_weight * p_elo

    fav = home if p_blend >= 0.5 else away
    fav_prob = max(p_blend, 1 - p_blend)
    confidence = abs(p_blend - 0.5) * 2.0
    tier = _tier(confidence)

    margin = None
    if margin_model is not None:
        try:
            margin = margin_model.predict_margin(
                home_team=home, away_team=away,
                game_date=date_cls.today(),
                home_rest=args.home_rest, away_rest=args.away_rest,
                is_playoff=args.playoff,
                is_international=args.international,
                is_div_game=args.divisional,
            )
        except Exception as exc:
            print(f"  (margin prediction unavailable: {exc})")

    playoff_tag = " [PLAYOFF]" if args.playoff else ""
    div_tag = " [DIV]" if args.divisional else ""
    intl_tag = " [INTL]" if args.international else ""
    tags = playoff_tag + div_tag + intl_tag

    print()
    print(f"  {away} @ {home}{tags}")
    print(f"  " + "─" * 54)
    print(f"  P({home}) = {_pct(p_blend)}      ELO {home}: {elo_home:7.1f}")
    print(f"  P({away}) = {_pct(1 - p_blend)}      ELO {away}: {elo_away:7.1f}")
    print()
    print(f"  Blend:   {(1-blend_elo_weight)*100:.0f}% model · "
          f"{blend_elo_weight*100:.0f}% ELO   "
          f"(raw model: {_pct(p_model)}, ELO: {_pct(p_elo)})")
    print()
    print(f"  Pick:        {fav} ({_pct(fav_prob)})")
    print(f"  Tier:        {tier}")
    print(f"  Confidence:  {confidence:.3f}")

    if margin is not None:
        print()
        signed_margin_from_fav = margin if fav == home else -margin
        print(f"  Predicted margin: {fav} by {abs(signed_margin_from_fav):.1f} "
              f"(raw: home {margin:+.1f})")
        spread = args.spread
        if spread is not None:
            ats_gap = margin - spread
            ats_pick = home if ats_gap > 0 else away
            sl_desc = (f"home favored by {spread:.1f}" if spread > 0
                       else f"away favored by {abs(spread):.1f}" if spread < 0
                       else "pick'em")
            print(f"  Vegas spread:     {sl_desc}")
            print(f"  ATS pick:         {ats_pick} (gap: {ats_gap:+.1f} pts)")

    if hasattr(wp_model, "get_qb_rating"):
        try:
            qb_h = wp_model.get_qb_rating(home)
            qb_a = wp_model.get_qb_rating(away)
            print()
            print(f"  QB rating:   {home}: {qb_h:+.3f}    {away}: {qb_a:+.3f}    "
                  f"diff: {qb_h - qb_a:+.3f}")
        except Exception:
            pass

    print()
    print(f"  _For entertainment & educational purposes only — not betting advice._\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
