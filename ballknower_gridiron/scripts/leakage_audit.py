"""
ballknower_gridiron.scripts.leakage_audit
=========================================

Forensic diagnostic for the v5 spread model when its ATS hit rate
looks too good to be true.

The v5 first-training results showed MAE = 10.32 (matching Vegas) but
ATS hit rate = 77.8% (impossible with that MAE — would require a
huge informational edge over Vegas that we don't have). This script
checks every suspect in the ATS evaluation pipeline so we can pinpoint
the leakage source.

Checks
------
  1. **Sanity-check `spread_line` from the schedule itself**:
     - Mean, std, range — does it look like a Vegas spread (-25 to +25)?
     - MAE between `-spread_line` and `point_diff` across all training
       games — should be ~10.0-10.5 (Vegas's known accuracy). If much
       lower (like < 5), `spread_line` is post-hoc / contains the result.
     - Correlation between `spread_line` and `point_diff` — should be
       around -0.4 to -0.5 (negative because favored home -> negative
       spread -> positive point_diff). If correlation is near -1.0,
       spread_line IS the result.

  2. **Check feat_df ↔ games alignment** explicitly:
     - Re-run build_training_frame, then verify that game_date AND
       home_team/away_team match game-by-game between feat_df and
       games_sorted. Misalignment would show up as mismatched team
       names at the same row index.

  3. **V5 prediction sanity checks** on the holdout:
     - Correlation between predicted_margin and actual point_diff
       (should be ~0.4 — what R²=0.15 implies; near 0 = weak model;
       near 1 = direct target leakage)
     - Correlation between predicted_margin and spread_line (should be
       positive since both estimate home expected margin)
     - corr(prediction error, spread surprise) — NOTE: this correlation
       is naturally strongly NEGATIVE (around -0.7 to -0.9) for any
       honest model with reasonable predictive power. That's because
       both quantities contain `actual_margin` with opposite signs:
       game-day variance contributes to both prediction error AND
       spread surprise, in opposite directions. A value near 0 would
       paradoxically indicate target leakage (model knows actual
       outcome); a strongly negative value is the EXPECTED honest
       signature. Print it for completeness; do not interpret as a
       smoking gun for leakage.

  4. **ATS formula sanity check** with known games:
     - Pick a few games from the holdout, print their pm/sl/am, walk
       through the ATS pick decision manually.

  5. **Per-game audit dump** — write the holdout rows with predicted
     margin, spread, actual, ATS pick, and outcome to a CSV so you
     can inspect individual games.

Usage
-----
    python -m ballknower_gridiron.scripts.leakage_audit
    python -m ballknower_gridiron.scripts.leakage_audit --dump-csv audit.csv

DISCLAIMER: For entertainment and educational purposes only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def check_schedule_data() -> pd.DataFrame:
    """Section 1: is `spread_line` actually a pre-game Vegas spread?"""
    from ballknower_gridiron.data.football_loader import load_nfl_games

    _section("SECTION 1 — Schedule-level sanity check on spread_line")
    games = load_nfl_games(seasons_back=settings.nfl_seasons_back)
    print(f"  Loaded {len(games):,} games across {games['season'].nunique()} seasons.")

    if "spread_line" not in games.columns:
        print("  ✗ spread_line column NOT present in schedule. Audit cannot continue.")
        return games

    sl = pd.to_numeric(games["spread_line"], errors="coerce")
    pd_ = pd.to_numeric(games["point_diff"], errors="coerce")
    mask = sl.notna() & pd_.notna()
    sl, pd_ = sl[mask], pd_[mask]
    n_valid = int(mask.sum())
    print(f"  spread_line present and numeric for {n_valid:,} / {len(games):,} games.")
    print()
    print(f"  spread_line stats:")
    print(f"     mean  = {sl.mean():+.2f}    (should be near 0; slight neg")
    print(f"                                    means home is slightly favored on avg)")
    print(f"     std   = {sl.std():.2f}    (typical spread distribution is 5-7)")
    print(f"     range = [{sl.min():.1f}, {sl.max():.1f}]")
    print()
    print(f"  point_diff stats:")
    print(f"     mean  = {pd_.mean():+.2f}    (NFL home-field advantage is ~+2 to +3)")
    print(f"     std   = {pd_.std():.2f}    (typical NFL game margin spread is ~13-14)")
    print(f"     range = [{pd_.min():.1f}, {pd_.max():.1f}]")
    print()

    # The killer check: how close is spread_line to point_diff?
    # Under nflverse convention (spread_line = home expected margin, positive
    # when home favored), spread_line should predict point_diff DIRECTLY,
    # not its negative.
    mae = float((pd_ - sl).abs().mean())
    rmse = float(np.sqrt(((pd_ - sl) ** 2).mean()))
    corr = float(np.corrcoef(pd_, sl)[0, 1])
    print(f"  ► CRITICAL CHECK — does spread_line predict point_diff?")
    print(f"     (nflverse convention: spread_line POSITIVE when home favored)")
    print(f"     MAE  of spread_line vs point_diff: {mae:.3f} pts")
    print(f"     RMSE of spread_line vs point_diff: {rmse:.3f} pts")
    print(f"     Correlation(spread_line, point_diff): {corr:+.4f}")
    print()
    if mae < 5.0:
        print("  ✗✗✗  WARNING: MAE far below Vegas's ~10.5 pts. spread_line probably")
        print("       contains post-hoc information (it knows the result). This is the")
        print("       source of the leakage if so.")
    elif mae > 13.0:
        print("  ✗ MAE much higher than Vegas's 10.5. Either spread_line is something")
        print("    other than the closing line, or the data has many bad spreads.")
    elif corr < 0:
        print("  ⚠ Correlation is NEGATIVE — spread_line might be in the Vegas/oddsmaker")
        print("    convention (negative when home favored) rather than the nflverse")
        print("    convention. If so, our ATS formula needs to be flipped.")
    else:
        print("  ✓ MAE looks Vegas-shaped (8.5 - 12.5) and correlation is positive.")
        print("    spread_line is the legitimate pre-game closing line in nflverse")
        print("    convention (positive when home favored).")
    print()

    # Sample of recent games
    print("  Sample (10 most recent games with spread_line):")
    sample = games[games["spread_line"].notna()].sort_values("game_date").tail(10)
    cols = ["game_date", "home_team", "away_team", "spread_line",
            "home_score", "away_score", "point_diff"]
    cols = [c for c in cols if c in sample.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sample[cols].to_string(index=False))
    print()
    return games


def check_alignment(games: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Section 2: confirm feat_df and games_sorted line up row-by-row."""
    _section("SECTION 2 — feat_df ↔ games_sorted alignment check")

    from ballknower_gridiron.models.football_model_v3 import NFLFeatureBuilderV3

    games = games.sort_values("game_date").reset_index(drop=True)
    if "point_diff" not in games.columns:
        games["point_diff"] = games["home_score"].astype(float) - games["away_score"].astype(float)

    log.info("Building features (this can take ~30s) …")
    fb = NFLFeatureBuilderV3()
    seasons = sorted(games["season"].astype(int).unique().tolist())
    fb.load_rolling_qb_ratings(seasons, games)
    fb.load_rolling_team_metrics(seasons, games)
    feat_df = fb.build_training_frame(games)

    feat_df_sorted = feat_df.sort_values("game_date").reset_index(drop=True)
    games_sorted = games.sort_values("game_date").reset_index(drop=True)

    print(f"  Built feat_df with {len(feat_df_sorted):,} rows.")
    print(f"  Comparing feat_df_sorted to games_sorted row-by-row …")
    if len(feat_df_sorted) != len(games_sorted):
        print(f"  ✗ Row count mismatch: {len(feat_df_sorted)} vs {len(games_sorted)}.")
        return None

    # The alignment we're assuming
    # date column comparison
    feat_dates = pd.to_datetime(feat_df_sorted["game_date"]).reset_index(drop=True)
    game_dates = pd.to_datetime(games_sorted["game_date"]).reset_index(drop=True)
    n_date_mismatch = int((feat_dates != game_dates).sum())
    print(f"  game_date mismatch count: {n_date_mismatch}")

    # Team comparison — feat_df doesn't normally carry team names, so we
    # can only check this if the builder happens to emit them. If not,
    # rely on the date check.
    if "home_team" in feat_df_sorted.columns and "away_team" in feat_df_sorted.columns:
        n_home_mismatch = int((feat_df_sorted["home_team"].values
                              != games_sorted["home_team"].values).sum())
        n_away_mismatch = int((feat_df_sorted["away_team"].values
                              != games_sorted["away_team"].values).sum())
        print(f"  home_team mismatch count: {n_home_mismatch}")
        print(f"  away_team mismatch count: {n_away_mismatch}")
    else:
        print(f"  (feat_df does NOT carry home_team/away_team — can't verify team alignment.)")
        print(f"  Will infer from cross-check between feat_df['home_won'] and game outcomes.")
        # Cross-check: home_won in feat_df vs games
        if "home_won" in feat_df_sorted.columns:
            n_won_mismatch = int((feat_df_sorted["home_won"].astype(int).values
                                 != games_sorted["home_won"].astype(int).values).sum())
            print(f"  home_won mismatch count: {n_won_mismatch}")
            if n_won_mismatch > 0:
                print(f"  ✗ MISALIGNMENT DETECTED — feat_df and games disagree on")
                print(f"    home_won in {n_won_mismatch} rows.")
                return feat_df_sorted

    if n_date_mismatch == 0:
        print(f"  ✓ feat_df and games_sorted are aligned row-by-row.")
    else:
        print(f"  ✗ MISALIGNMENT — game_date differs in {n_date_mismatch} rows.")
        print(f"    This is a SUSPECT cause of the ATS leakage.")
        # Show first few mismatches
        mismatches = feat_df_sorted[feat_dates != game_dates].head(5)
        print(f"    First 5 mismatched rows (by index):")
        print(mismatches[["game_date"]].head().to_string())
    return feat_df_sorted


def check_v5_predictions(feat_df: pd.DataFrame, games: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Section 3: do V5 predictions look like an honest Vegas-quality model?"""
    _section("SECTION 3 — v5 prediction correlations on holdout")

    try:
        from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
        model = NFLSpreadModelV5.load(settings.models_dir_v5)
    except FileNotFoundError:
        print("  ✗ No trained v5 model found — train it before running this audit.")
        return None

    # Align feat_df with the spread_line / point_diff from games.
    feat_df = feat_df.sort_values("game_date").reset_index(drop=True)
    games_sorted = games.sort_values("game_date").reset_index(drop=True)

    if len(feat_df) != len(games_sorted):
        print("  ✗ Row count mismatch — cannot proceed.")
        return None

    feat_df = feat_df.copy()
    feat_df["point_diff"] = games_sorted["point_diff"].values
    feat_df["spread_line"] = games_sorted["spread_line"].values
    feat_df["home_team"] = games_sorted["home_team"].values
    feat_df["away_team"] = games_sorted["away_team"].values

    split_idx = int(len(feat_df) * (1.0 - settings.test_fraction))
    test_df = feat_df.iloc[split_idx:].copy().reset_index(drop=True)

    X_test = test_df[model.feature_columns].to_numpy(dtype=float)
    X_test_s = model.scaler.transform(X_test)
    test_df["predicted_margin"] = model.regressor.predict(X_test_s)

    pm = test_df["predicted_margin"].to_numpy()
    am = test_df["point_diff"].to_numpy(dtype=float)
    sl = test_df["spread_line"].to_numpy(dtype=float)

    valid = ~np.isnan(pm) & ~np.isnan(am) & ~np.isnan(sl)
    pm, am, sl = pm[valid], am[valid], sl[valid]

    print(f"  Holdout n = {len(pm)}")
    print()

    def safe_corr(x, y):
        if len(x) < 2 or x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    c_pm_am = safe_corr(pm, am)
    c_pm_sl = safe_corr(pm, sl)
    c_sl_am = safe_corr(sl, am)
    print(f"  Correlation(predicted_margin, actual point_diff): {c_pm_am:+.4f}")
    print(f"     Honest range: +0.2 to +0.5. Near 1.0 = direct target leakage.")
    print(f"     Near 0 = model has little predictive power but not leaking.")
    print()
    print(f"  Correlation(predicted_margin, spread_line):       {c_pm_sl:+.4f}")
    print(f"     Honest range: ~+0.4-0.7 (both estimate home expected margin;")
    print(f"     positive correlation because nflverse spread_line is POSITIVE")
    print(f"     when home is favored).")
    print()
    print(f"  Correlation(spread_line, actual point_diff):      {c_sl_am:+.4f}")
    print(f"     Honest range: ~+0.45 — Vegas's own correlation with actuals.")
    print()

    # Diagnostic: model vs. Vegas accuracy comparison
    # Under correct convention (spread_line = home expected margin):
    # spread_surprise = am - sl  (positive = home over-performed vs Vegas)
    pred_err = pm - am
    spread_surprise = am - sl
    c_err_surprise = safe_corr(pred_err, spread_surprise)
    print(f"  Correlation(prediction error, spread surprise): {c_err_surprise:+.4f}")
    print(f"     NOTE: this is naturally strongly NEGATIVE (~-0.7 to -0.9) for any")
    print(f"     honest model — game-day variance appears in both terms with")
    print(f"     opposite signs. A near-zero value would paradoxically suggest")
    print(f"     leakage (model knew actual outcome). Treat strongly negative")
    print(f"     values here as a HEALTH SIGN, not a leakage flag.")
    print()

    # Also check: does predicted_margin look like a copy of actual_margin?
    diff = pm - am
    mae_pm_am = float(np.abs(diff).mean())
    pred_minus_spread = pm - sl   # how far model thinks home will beat the spread
    mae_pm_vegas = float(np.abs(pred_minus_spread).mean())
    print(f"  MAE(predicted_margin, point_diff):  {mae_pm_am:.3f}")
    print(f"  MAE(predicted_margin, spread_line): {mae_pm_vegas:.3f}")
    print(f"     The first should be ~10.5 (matching Vegas-grade prediction).")
    print(f"     If the first is < 5, that's direct target leakage.")
    print()

    if c_pm_am > 0.85:
        print("  ✗✗✗ predicted_margin is implausibly close to actual point_diff.")
        print("       Direct target leakage almost certain.")
    elif mae_pm_am < 5.0:
        print("  ✗✗ Predicted margin is too accurate to be honest.")
        print("      Likely target leakage in the feature pipeline.")
    elif c_pm_sl < 0:
        print("  ⚠ Correlation(predicted, spread_line) is NEGATIVE. That suggests")
        print("    a sign convention mismatch — verify how the feature builder")
        print("    treats home vs. away orientation.")
    else:
        print("  ✓ Correlations look as expected for an honest Vegas-quality model.")
        print("    If ATS hit rate still looks unrealistic, the bug is in the ATS")
        print("    evaluation formula, not the prediction model itself.")
    print()
    return test_df


def manual_ats_check(test_df: pd.DataFrame) -> None:
    """Section 4: walk through the ATS formula on a handful of games by hand."""
    _section("SECTION 4 — Manual ATS computation on 10 sample games")
    print("  Convention reminder: nflverse spread_line is POSITIVE when home is")
    print("  favored (it represents the home team's expected margin).")
    print("  ATS pick = home if (predicted_margin − spread_line) > 0, else away.")
    print("  ATS won  = home if (actual_margin   − spread_line) > 0, else away.")
    print("  Hit if both signs match.")
    print()

    if test_df is None or test_df.empty:
        print("  (no test_df available)")
        return

    sample = test_df.dropna(subset=["spread_line", "predicted_margin", "point_diff"]).head(10)
    print(f"  {'Date':<11} {'Matchup':<14} {'PM':>6} {'SL':>6} {'AM':>6} "
          f"{'pred_sig':>8} {'act_sig':>8} {'Pick':>5} {'Hit?':>5}")
    print("  " + "-" * 88)
    for r in sample.itertuples(index=False):
        gd = pd.to_datetime(r.game_date).strftime("%Y-%m-%d")
        matchup = f"{r.away_team}@{r.home_team}" if hasattr(r, "home_team") else "       "
        pm = float(r.predicted_margin)
        sl = float(r.spread_line)
        am = float(r.point_diff)
        ps = pm - sl
        as_ = am - sl
        pick = "home" if ps > 0 else ("away" if ps < 0 else "push")
        hit = "✓" if (ps > 0) == (as_ > 0) and as_ != 0 else ("push" if as_ == 0 else "✗")
        print(f"  {gd:<11} {matchup:<14} {pm:>+6.2f} {sl:>+6.2f} {am:>+6.2f} "
              f"{ps:>+8.2f} {as_:>+8.2f} {pick:>5} {hit:>5}")
    print()


def dump_full_audit_csv(test_df: pd.DataFrame, out_path: Path) -> None:
    """Section 5: write all holdout rows to CSV for manual inspection."""
    _section(f"SECTION 5 — Full holdout audit dump")
    if test_df is None or test_df.empty:
        print("  (no test_df available)")
        return
    cols = ["game_date", "home_team", "away_team", "predicted_margin",
            "spread_line", "point_diff"]
    cols = [c for c in cols if c in test_df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_df[cols].to_csv(out_path, index=False)
    print(f"  Wrote {len(test_df)} rows to {out_path}")
    print(f"  Suggested next steps with this CSV:")
    print(f"   - Sort by abs(predicted_margin - spread_line) descending.")
    print(f"   - For top 'Strong tier' picks, verify each one manually against")
    print(f"     a third-party source (e.g., a sports almanac or Pro Football Reference).")
    print(f"   - If even ONE 'Strong tier' pick looks wrong relative to actual")
    print(f"     pre-game spread, the data has post-hoc spread injection.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit v5 spread model for data leakage."
    )
    parser.add_argument(
        "--dump-csv", type=str, default=None,
        help="If set, write full holdout audit data to this CSV path.",
    )
    args = parser.parse_args()

    games = check_schedule_data()
    if games.empty:
        return 1
    feat_df = check_alignment(games)
    if feat_df is None:
        return 1
    test_df = check_v5_predictions(feat_df, games)
    manual_ats_check(test_df)

    out_path = Path(args.dump_csv) if args.dump_csv else (settings.logs_dir / "v5_audit.csv")
    dump_full_audit_csv(test_df, out_path)

    _section("AUDIT COMPLETE")
    print("  See sections above. Common diagnostic patterns:")
    print()
    print("   • Section 1: MAE between -spread and point_diff < 5     ")
    print("     -> spread_line column is post-hoc data, not a real pre-game spread")
    print("   • Section 1: corr(spread_line, point_diff) is POSITIVE   ")
    print("     -> spread_line is the home team's EXPECTED margin (nflverse convention)")
    print("   • Section 1: corr(spread_line, point_diff) is NEGATIVE   ")
    print("     -> spread_line is the betting line (negative = home favored)")
    print("   • Section 2: alignment mismatches > 0                    ")
    print("     -> feat_df ↔ games row misalignment")
    print("   • Section 3: corr(predicted, actual) > 0.85              ")
    print("     -> direct target leakage in features")
    print("   • Section 3: MAE(predicted, actual) < 5                  ")
    print("     -> model is unrealistically accurate; investigate features")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
