"""
football_app.py
===============

Streamlit dashboard for BallKnower Gridiron. Mirrors the basketball
dashboard pattern with seven pages adapted to the NFL's two-model setup:

  1. Home — overview of trained models (V1, V2, V3, V3.1, V4, V4.1, V5)
  2. Train Model — pick version, configure, kick off training
  3. Single Prediction — enter a matchup, get win prob + margin + ATS
  4. Weekly Schedule — pull upcoming slate, predict every game
  5. Backtest — calibration comparison + ATS diagnostics
  6. Newsletter — generate markdown/HTML for Substack
  7. Settings — ELO tuning, blend weight, model defaults

Launch:
    streamlit run football_app.py

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Make sure ballknower_gridiron is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ballknower_gridiron.config.settings import settings  # noqa: E402

st.set_page_config(
    page_title="BallKnower Gridiron",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Shared helpers
# ===========================================================================
ALL_VERSIONS = ["v1", "v2", "v3", "v3.1", "v4", "v4.1", "v5"]
WP_VERSIONS = ["v1", "v2", "v3", "v3.1", "v4", "v4.1"]   # classifiers
MARGIN_VERSIONS = ["v5"]                                   # regressors


@st.cache_resource(show_spinner=False)
def _load_wp_model_cached(version: str, cache_bust: int = 0):
    """Cache loaded win-probability classifiers."""
    from ballknower_gridiron.models.football_model_v2 import load_active_nfl_model
    return load_active_nfl_model(version=version)


@st.cache_resource(show_spinner=False)
def _load_margin_model_cached(version: str, cache_bust: int = 0):
    """Cache loaded margin regressors (V5+)."""
    from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
    return NFLSpreadModelV5.load(settings.models_dir_for(version))


def _model_status(version: str) -> dict:
    """Check if a model is trained and return its metadata."""
    path = settings.models_dir_for(version)
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return {"trained": False, "path": str(path)}
    try:
        meta = json.loads(meta_path.read_text())
        return {
            "trained": True,
            "path": str(path),
            "feature_count": len(meta.get("feature_columns", [])),
            "trained_at": meta.get("trained_at", "unknown"),
            "metrics": meta.get("metrics", {}),
            "version": meta.get("version", version),
            "kind": meta.get("kind", "classification_winprob"),
        }
    except Exception:
        return {"trained": False, "path": str(path)}


def _elo_baseline_probability(elo_home: float, elo_away: float, hca: float) -> float:
    diff = (elo_home + hca) - elo_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _tier_label(confidence: float) -> str:
    if confidence >= 0.50:
        return "🔒 Lock"
    if confidence >= 0.30:
        return "💪 Strong"
    if confidence >= 0.15:
        return "🎯 Lean"
    return "🪙 Pass"


def _format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _available_versions() -> List[str]:
    return [v for v in ALL_VERSIONS if _model_status(v)["trained"]]


def _available_wp_versions() -> List[str]:
    return [v for v in WP_VERSIONS if _model_status(v)["trained"]]


def _available_margin_versions() -> List[str]:
    return [v for v in MARGIN_VERSIONS if _model_status(v)["trained"]]


# ===========================================================================
# Sidebar
# ===========================================================================
st.sidebar.title("🏈 BallKnower Gridiron")
st.sidebar.caption("_Entertainment & educational use only — not betting advice._")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎓 Train Model", "🎯 Single Prediction",
     "📅 Weekly Schedule", "📊 Backtest", "📰 Newsletter", "⚙️ Settings"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
available = _available_versions()
if available:
    st.sidebar.success(f"✅ Trained: {', '.join(available)}")
else:
    st.sidebar.warning("No trained models yet. Go to Train Model.")


# ===========================================================================
# Page 1 — Home
# ===========================================================================
if page == "🏠 Home":
    st.title("BallKnower Gridiron Dashboard")
    st.caption("AI-powered NFL prediction — for entertainment & educational use only.")

    st.subheader("Model status")
    status_rows = []
    for v in ALL_VERSIONS:
        s = _model_status(v)
        if s["trained"]:
            metrics = s.get("metrics", {})
            kind = s.get("kind", "classification_winprob")
            if kind == "regression_spread":
                acc_or_mae = f"{metrics.get('mae', 0):.2f} MAE"
                second = f"{metrics.get('r2', 0):.3f} R²"
                third = f"{metrics.get('implied_accuracy', 0):.4f} acc"
            else:
                acc_or_mae = f"{metrics.get('accuracy', 0):.4f}" if metrics else "—"
                second = f"{metrics.get('roc_auc', 0):.4f}" if metrics else "—"
                third = f"{metrics.get('brier', 0):.4f}" if metrics else "—"
            status_rows.append({
                "Version": v.upper(),
                "Kind": "margin" if kind == "regression_spread" else "win prob",
                "Features": s["feature_count"],
                "Primary": acc_or_mae,
                "Secondary": second,
                "Tertiary": third,
                "Trained": s["trained_at"][:10] if s["trained_at"] != "unknown" else "—",
            })
        else:
            status_rows.append({"Version": v.upper(), "Kind": "—", "Features": "—",
                                "Primary": "Not trained", "Secondary": "—",
                                "Tertiary": "—", "Trained": "—"})
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    st.caption("Win-probability models: Primary = Accuracy · Secondary = ROC AUC · "
               "Tertiary = Brier. Margin models: Primary = MAE · Secondary = R² · "
               "Tertiary = implied accuracy.")

    st.subheader("Quick start")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**1. Train a model** in the Train Model page. Each version adds features:\n\n"
            "- **V1**: ELO + rest + form + bye + HFA + intl + div\n"
            "- **V2**: + QB rolling rating\n"
            "- **V3**: + rolling team EPA + Net Rating\n"
            "- **V3.1**: pruned V3 (best calibration — recommended for production)\n"
            "- **V4**: + weather + QB-change features\n"
            "- **V4.1**: pruned V4\n"
            "- **V5**: spread/margin regressor (Vegas-grade MAE ≈ 10.3)"
        )
    with col2:
        st.markdown(
            "**2. Generate weekly content** in the Newsletter page once you have at least "
            "one win-probability model + V5 trained. The pipeline combines them into a "
            "complete newsletter with:\n\n"
            "- 🔒 Lock of the Week\n"
            "- 🔥 Must-Watch Games\n"
            "- 📊 Picks by Confidence Tier\n"
            "- ⚠️ Where We Disagree with Vegas Most\n"
            "- ⚡ Quick Insights"
        )


# ===========================================================================
# Page 2 — Train Model
# ===========================================================================
elif page == "🎓 Train Model":
    st.title("Train a Model")
    st.caption("Train any version on your local data and cached PBP / QB stats.")

    with st.form("train_form"):
        version = st.selectbox(
            "Model version",
            ALL_VERSIONS,
            index=ALL_VERSIONS.index("v3.1") if "v3.1" in ALL_VERSIONS else 0,
            help="V3.1 is the production recommendation for win probabilities · "
                 "V5 is the spread/margin model"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            seasons = st.number_input(
                "Seasons of history",
                min_value=4, max_value=20,
                value=settings.nfl_seasons_back,
                help="More seasons = more training data but slower."
            )
        with col_b:
            force_refresh = st.checkbox(
                "Force-refresh caches",
                help="Re-download PBP/QB stats from nflverse instead of using local cache."
            )
        submitted = st.form_submit_button("🚀 Start training", type="primary")

    if submitted:
        with st.spinner(f"Training {version} … this may take 2-5 minutes."):
            from ballknower_gridiron.data.football_loader import load_nfl_games
            games = load_nfl_games(seasons_back=int(seasons),
                                   force_refresh=force_refresh)
            st.info(f"Loaded {len(games):,} games across "
                    f"{games['season'].nunique()} seasons.")

            normalized = version.replace(".", "_")
            try:
                if normalized == "v1":
                    from ballknower_gridiron.models.football_model import train_nfl_model
                    model = train_nfl_model(games)
                elif normalized == "v2":
                    from ballknower_gridiron.models.football_model_v2 import train_nfl_model_v2
                    model = train_nfl_model_v2(games, force_refresh_qb_stats=force_refresh)
                elif normalized == "v3":
                    from ballknower_gridiron.models.football_model_v3 import train_nfl_model_v3
                    model = train_nfl_model_v3(
                        games, force_refresh_qb_stats=force_refresh,
                        force_refresh_team_metrics=force_refresh,
                    )
                elif normalized == "v3_1":
                    from ballknower_gridiron.models.football_model_v3_1 import train_nfl_model_v3_1
                    model = train_nfl_model_v3_1(
                        games, force_refresh_qb_stats=force_refresh,
                        force_refresh_team_metrics=force_refresh,
                    )
                elif normalized == "v4":
                    from ballknower_gridiron.models.football_model_v4 import train_nfl_model_v4
                    model = train_nfl_model_v4(
                        games, force_refresh_qb_stats=force_refresh,
                        force_refresh_team_metrics=force_refresh,
                    )
                elif normalized == "v4_1":
                    from ballknower_gridiron.models.football_model_v4_1 import train_nfl_model_v4_1
                    model = train_nfl_model_v4_1(
                        games, force_refresh_qb_stats=force_refresh,
                        force_refresh_team_metrics=force_refresh,
                    )
                elif normalized == "v5":
                    from ballknower_gridiron.models.football_model_v5 import train_nfl_spread_v5
                    model = train_nfl_spread_v5(
                        games, force_refresh_qb_stats=force_refresh,
                        force_refresh_team_metrics=force_refresh,
                    )
                else:
                    st.error(f"Unknown version: {version}")
                    st.stop()

                out_dir = settings.models_dir_for(version)
                model.save(out_dir)
                st.success(f"✅ Trained {version} → saved to `{out_dir}`")
                st.json(model.metrics)

                # Bust the cache so Single Prediction reloads the new model
                st.session_state["cache_bust"] = st.session_state.get("cache_bust", 0) + 1
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.exception(exc)


# ===========================================================================
# Page 3 — Single Prediction
# ===========================================================================
elif page == "🎯 Single Prediction":
    st.title("Single Prediction")
    st.caption("Enter a matchup, get win probability + predicted margin + ATS context.")

    wp_avail = _available_wp_versions()
    margin_avail = _available_margin_versions()

    if not wp_avail:
        st.warning("No win-probability model trained yet. Go to Train Model.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        wp_version = st.selectbox(
            "Win-probability model",
            wp_avail,
            index=wp_avail.index("v3.1") if "v3.1" in wp_avail else len(wp_avail) - 1,
            help="V3.1 is recommended for production."
        )
    with col2:
        margin_options = ["(skip)"] + margin_avail
        margin_choice = st.selectbox(
            "Margin model (optional)",
            margin_options,
            index=1 if len(margin_options) > 1 else 0,
            help="V5 produces a point-margin prediction alongside the win probability."
        )
    margin_version = None if margin_choice == "(skip)" else margin_choice

    col_h, col_a = st.columns(2)
    with col_h:
        home = st.text_input("Home team (abbreviation)", value="KC").upper()
    with col_a:
        away = st.text_input("Away team (abbreviation)", value="BUF").upper()

    col_f, col_g, col_h2 = st.columns(3)
    with col_f:
        is_playoff = st.checkbox("Playoff game")
    with col_g:
        is_div = st.checkbox("Divisional matchup")
    with col_h2:
        is_intl = st.checkbox("International venue")

    col_r1, col_r2, col_b = st.columns(3)
    with col_r1:
        home_rest = st.number_input("Home rest (days)", min_value=3.0, max_value=14.0,
                                    value=7.0, step=0.5)
    with col_r2:
        away_rest = st.number_input("Away rest (days)", min_value=3.0, max_value=14.0,
                                    value=7.0, step=0.5)
    with col_b:
        blend_elo_weight = st.slider("ELO blend weight", 0.0, 1.0,
                                     settings.default_blend_elo, 0.05)

    vegas_spread = st.number_input(
        "Vegas spread (optional, positive = home favored)",
        min_value=-25.0, max_value=25.0, value=0.0, step=0.5,
        help="Used to compute the ATS pick alongside V5's margin prediction. "
             "Leave at 0 to skip."
    )

    if st.button("🎯 Predict", type="primary"):
        cache_bust = st.session_state.get("cache_bust", 0)
        try:
            wp_model = _load_wp_model_cached(wp_version, cache_bust)
        except Exception as exc:
            st.error(f"Could not load {wp_version}: {exc}")
            st.stop()

        try:
            p_model, _ = wp_model.predict_proba(
                home_team=home, away_team=away,
                game_date=date.today(),
                home_rest=home_rest, away_rest=away_rest,
                is_playoff=is_playoff,
                is_international=is_intl,
                is_div_game=is_div,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        elo_home = wp_model.elo.get_rating(home)
        elo_away = wp_model.elo.get_rating(away)
        p_elo = _elo_baseline_probability(elo_home, elo_away, settings.elo_hca)
        p_blend = (1.0 - blend_elo_weight) * p_model + blend_elo_weight * p_elo

        fav = home if p_blend >= 0.5 else away
        fav_prob = max(p_blend, 1 - p_blend)
        confidence = abs(p_blend - 0.5) * 2.0
        tier = _tier_label(confidence)

        st.subheader(f"{away} @ {home}" + (" — Playoff" if is_playoff else ""))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"P({home})", _format_pct(p_blend),
                      delta=f"raw {_format_pct(p_model)}")
        with m2:
            st.metric(f"P({away})", _format_pct(1 - p_blend),
                      delta=f"ELO {_format_pct(p_elo)}")
        with m3:
            st.metric("Confidence", f"{confidence:.3f}", delta=tier)

        st.success(f"**Pick:** {fav} ({_format_pct(fav_prob)}) · {tier}")

        # Margin prediction
        margin = None
        if margin_version is not None:
            try:
                margin_model = _load_margin_model_cached(margin_version, cache_bust)
                margin = margin_model.predict_margin(
                    home_team=home, away_team=away,
                    game_date=date.today(),
                    home_rest=home_rest, away_rest=away_rest,
                    is_playoff=is_playoff,
                    is_international=is_intl,
                    is_div_game=is_div,
                )
            except FileNotFoundError:
                st.info(f"No {margin_version} margin model trained yet.")
            except Exception as exc:
                st.warning(f"Margin prediction unavailable: {exc}")

        if margin is not None:
            st.subheader("Margin prediction")
            margin_from_fav = margin if fav == home else -margin
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(f"Predicted: {fav} by",
                          f"{abs(margin_from_fav):.1f} pts",
                          delta=f"raw: home {margin:+.1f}")
            if vegas_spread != 0.0:
                ats_gap = margin - vegas_spread
                ats_pick = home if ats_gap > 0 else away
                with mc2:
                    st.metric(f"ATS pick: {ats_pick}",
                              f"gap {ats_gap:+.1f} pts",
                              delta=f"Vegas: {vegas_spread:+.1f}")
                st.caption("ATS pick is editorial — V5's baseline ATS hit rate on the "
                           "holdout is ~51% (no edge over Vegas). Not betting advice.")

        # Team metadata
        with st.expander("Team metadata"):
            rows = []
            rows.append({"Metric": "ELO rating",
                         home: f"{elo_home:.1f}", away: f"{elo_away:.1f}",
                         "Diff (home − away)": f"{elo_home - elo_away:+.1f}"})
            if hasattr(wp_model, "get_qb_rating"):
                try:
                    qb_h = wp_model.get_qb_rating(home)
                    qb_a = wp_model.get_qb_rating(away)
                    rows.append({"Metric": "QB rating",
                                 home: f"{qb_h:+.3f}", away: f"{qb_a:+.3f}",
                                 "Diff (home − away)": f"{qb_h - qb_a:+.3f}"})
                except Exception:
                    pass
            if hasattr(wp_model, "get_team_metrics"):
                try:
                    tm_h = wp_model.get_team_metrics(home)
                    tm_a = wp_model.get_team_metrics(away)
                    for key, label in [("net_epa_per_play", "Net EPA/play"),
                                       ("pts_for_pg", "Points for/game"),
                                       ("pts_against_pg", "Points against/game")]:
                        vh = tm_h.get(key, 0.0)
                        va = tm_a.get(key, 0.0)
                        rows.append({"Metric": label,
                                     home: f"{vh:+.3f}" if "EPA" in label else f"{vh:.1f}",
                                     away: f"{va:+.3f}" if "EPA" in label else f"{va:.1f}",
                                     "Diff (home − away)": f"{vh - va:+.3f}"})
                except Exception:
                    pass
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ===========================================================================
# Page 4 — Weekly Schedule
# ===========================================================================
elif page == "📅 Weekly Schedule":
    st.title("Weekly Schedule")
    st.caption("Pull upcoming NFL games and predict the whole slate.")

    wp_avail = _available_wp_versions()
    margin_avail = _available_margin_versions()
    if not wp_avail:
        st.warning("No win-probability model trained yet. Go to Train Model.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        wp_version = st.selectbox(
            "Win-probability model", wp_avail,
            index=wp_avail.index("v3.1") if "v3.1" in wp_avail else len(wp_avail) - 1,
        )
    with col2:
        margin_options = ["(skip)"] + margin_avail
        margin_choice = st.selectbox("Margin model", margin_options,
                                     index=1 if len(margin_options) > 1 else 0)
    with col3:
        days_ahead = st.number_input("Days ahead", min_value=1, max_value=21, value=10)

    blend_elo_weight = st.slider("ELO blend weight", 0.0, 1.0,
                                 settings.default_blend_elo, 0.05)

    if st.button("📅 Predict the slate", type="primary"):
        from ballknower_gridiron.scripts.weekly_football_pipeline import (
            predict_upcoming_slate,
        )
        margin_version = None if margin_choice == "(skip)" else margin_choice
        with st.spinner("Pulling schedule and running predictions …"):
            try:
                preds = predict_upcoming_slate(
                    wp_version=wp_version,
                    margin_version=margin_version,
                    days_ahead=int(days_ahead),
                    blend_elo_weight=blend_elo_weight,
                )
            except Exception as exc:
                st.error(f"Prediction pipeline failed: {exc}")
                st.exception(exc)
                st.stop()

        if not preds:
            st.warning("No upcoming games found in that window.")
            st.stop()

        rows = []
        for g in preds:
            row = {
                "Date": g.game_date,
                "Matchup": f"{g.away_team} @ {g.home_team}",
                "Pick": g.favorite,
                "P(pick)": _format_pct(g.fav_prob),
                "Tier": g.tier,
                "P(home)": _format_pct(g.p_blended),
                "ELO": f"{g.elo_home:.0f} / {g.elo_away:.0f}",
            }
            if g.predicted_margin is not None:
                row["Predicted margin"] = f"{g.predicted_margin:+.1f}"
            if g.spread_line is not None:
                row["Vegas"] = f"{g.spread_line:+.1f}"
            if g.ats_pick is not None:
                row["ATS"] = f"{g.ats_pick} ({g.ats_gap:+.1f})"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # Stash predictions in session for the Newsletter page
        st.session_state["latest_preds"] = preds
        st.session_state["latest_wp_version"] = wp_version
        st.session_state["latest_margin_version"] = margin_version
        st.session_state["latest_blend"] = blend_elo_weight
        st.success(f"Predicted {len(preds)} games. "
                   "Switch to the Newsletter page to generate copy.")


# ===========================================================================
# Page 5 — Backtest / Diagnostics
# ===========================================================================
elif page == "📊 Backtest":
    st.title("Backtest & Diagnostics")
    st.caption("Compare model calibration and run ATS diagnostics.")

    available_wp = _available_wp_versions()
    if len(available_wp) < 2:
        st.info("Need at least two trained win-probability models to compare. "
                "Train v3 and v3.1 to start.")
    else:
        st.subheader("Calibration comparison")
        st.caption("Run the side-by-side calibration script on trained models.")
        chosen = st.multiselect(
            "Versions to compare",
            available_wp,
            default=available_wp[:2],
        )
        if st.button("Run calibration_compare") and len(chosen) >= 2:
            from ballknower_gridiron.scripts.calibration_compare import (
                compute_version_stats, print_summary_table, print_reliability_table_combined,
                print_confidence_volume, print_winner_interpretation,
            )
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                stats_list = []
                for v in chosen:
                    s = compute_version_stats(v, n_bins=10)
                    if s is not None:
                        stats_list.append(s)
                if len(stats_list) >= 2:
                    print_summary_table(stats_list)
                    print_reliability_table_combined(stats_list)
                    print_confidence_volume(stats_list)
                    print_winner_interpretation(stats_list)
            st.code(buf.getvalue(), language="text")

    st.markdown("---")
    st.subheader("V5 ATS diagnostics")
    if "v5" not in _available_margin_versions():
        st.info("No V5 margin model trained yet.")
    else:
        if st.button("Run ats_diagnostics"):
            from ballknower_gridiron.scripts.ats_diagnostics import (
                rebuild_v5_holdout, print_headline, print_tier_breakdown,
                print_threshold_sweep, print_side_breakdown,
            )
            from ballknower_gridiron.models.football_model_v5 import NFLSpreadModelV5
            model = NFLSpreadModelV5.load(settings.models_dir_for("v5"))
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                h = rebuild_v5_holdout(model)
                print_headline(model, h)
                print_tier_breakdown(h)
                print_threshold_sweep(h)
                print_side_breakdown(h)
            st.code(buf.getvalue(), language="text")


# ===========================================================================
# Page 6 — Newsletter
# ===========================================================================
elif page == "📰 Newsletter":
    st.title("Newsletter Generator")
    st.caption("Generate markdown + HTML for the week's slate. "
               "Run Weekly Schedule first to get fresh predictions.")

    preds = st.session_state.get("latest_preds")
    wp_version = st.session_state.get("latest_wp_version", "v3.1")
    margin_version = st.session_state.get("latest_margin_version", "v5")
    blend = st.session_state.get("latest_blend", settings.default_blend_elo)

    if not preds:
        st.warning("No predictions in session. Go to Weekly Schedule and "
                   "click 'Predict the slate' first.")
        st.stop()

    from ballknower_gridiron.utils.content_utils import (
        render_full_newsletter, render_html_newsletter,
    )

    # Determine week label
    weeks = [p.week for p in preds if p.week is not None]
    seasons = [p.season for p in preds if p.season is not None]
    week_label = None
    if weeks and seasons:
        most_common_week = max(set(weeks), key=weeks.count)
        most_common_season = max(set(seasons), key=seasons.count)
        week_label = f"{most_common_season} Week {most_common_week}"

    md = render_full_newsletter(
        preds,
        title_date=date.today().isoformat(),
        wp_version=wp_version,
        margin_version=margin_version,
        blend_elo_weight=blend,
        week_label=week_label,
    )
    html_doc = render_html_newsletter(md)

    tab_md, tab_html, tab_preview = st.tabs(["📝 Markdown", "🌐 HTML", "👁 Preview"])
    with tab_md:
        st.code(md, language="markdown")
        st.download_button("Download .md", md, file_name="newsletter.md",
                           mime="text/markdown")
    with tab_html:
        st.code(html_doc, language="html")
        st.download_button("Download .html", html_doc, file_name="newsletter.html",
                           mime="text/html")
    with tab_preview:
        st.markdown(md)

    st.markdown("---")
    if st.button("💾 Save to content/football/<today>/"):
        out_dir = settings.project_root / "content" / "football" / date.today().isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "newsletter.md").write_text(md, encoding="utf-8")
        (out_dir / "newsletter.html").write_text(html_doc, encoding="utf-8")
        from dataclasses import asdict
        (out_dir / "predictions.json").write_text(
            json.dumps([asdict(p) for p in preds], indent=2, default=str),
            encoding="utf-8",
        )
        st.success(f"✅ Saved to `{out_dir}`")


# ===========================================================================
# Page 7 — Settings
# ===========================================================================
elif page == "⚙️ Settings":
    st.title("Settings")
    st.caption("Current configuration (read-only — set via .env or environment).")

    st.subheader("Active model")
    st.code(f"NFL_MODEL_VERSION = {settings.active_model_version}")
    st.caption("Recommended: V3.1 for win probabilities; V5 separately for margins. "
               "Set with: `export NFL_MODEL_VERSION=v3.1`")

    st.subheader("Blend weights")
    st.code(f"NFL_BLEND_ELO_DEFAULT = {settings.default_blend_elo:.2f}")

    st.subheader("ELO")
    st.code(
        f"K-factor: {settings.elo_k}\n"
        f"Home-court advantage (HFA): {settings.elo_hca}\n"
        f"Off-season regression: {settings.elo_offseason_regression}\n"
        f"Playoff multiplier: {settings.elo_playoff_multiplier}\n"
        f"MOV cap: {settings.elo_mov_cap}\n"
    )

    st.subheader("XGBoost")
    st.code(
        f"n_estimators: {settings.xgb_n_estimators}\n"
        f"max_depth: {settings.xgb_max_depth}\n"
        f"learning_rate: {settings.xgb_learning_rate}\n"
    )

    st.subheader("Data")
    st.code(
        f"Seasons back (training): {settings.nfl_seasons_back}\n"
        f"Test fraction (holdout): {settings.test_fraction}\n"
        f"Form window: {settings.form_window} games\n"
        f"Data dir: {settings.data_dir}\n"
        f"Models dir: {settings.models_dir}\n"
    )

    st.markdown("---")
    if st.button("🔄 Reload all models from disk"):
        st.cache_resource.clear()
        st.session_state["cache_bust"] = st.session_state.get("cache_bust", 0) + 1
        st.success("Caches cleared. Next prediction will reload from disk.")
