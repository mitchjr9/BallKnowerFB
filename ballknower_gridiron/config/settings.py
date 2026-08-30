"""
ballknower_gridiron.config.settings
===================================

Single source of truth for paths and tuning knobs for the football
package. Reads the shared `.env` at the project root (same one tennis
and hoops use), but every NFL-specific variable is prefixed `NFL_*`
so the three packages never collide.

NFL tuning vs NBA tuning — why these defaults differ
----------------------------------------------------
NFL has only ~17 regular-season games and ~270 total games per
season, vs NBA's 82 and ~1,230. Three consequences:

  * **K-factor higher**     each game carries more signal (NFL=25 vs NBA=20)
  * **Off-season regression bigger**   rosters change more dramatically
                                       between seasons (NFL=0.33 vs NBA=0.25)
  * **HCA slightly lower**  ~2.5 pts vs NBA's ~3 pts (NFL=55 vs NBA=65)
                            (HCA has been declining post-COVID — watch this)
  * **More seasons of history needed**  12+ seasons (vs NBA 8) so the
                                        model has enough data
  * **MOV cap bigger**      3 TDs + a FG (24) since NFL margins span
                            wider than NBA (NBA=15)
  * **Playoff multiplier bigger**   playoffs are single-elimination and
                                    matter more in ratings (NFL=1.20
                                    vs NBA=1.05)

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


# Project root: ballknower_gridiron/config/settings.py -> up 2 = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DISCLAIMER_SHORT: str = (
    "For entertainment and educational purposes only — not financial or "
    "betting advice."
)
DISCLAIMER_LONG: str = (
    "BallKnower Gridiron is provided strictly for entertainment and "
    "educational purposes. Nothing here constitutes financial, investment, "
    "or betting advice. NFL predictions are inherently uncertain — injuries, "
    "weather, and small-sample noise (only 17 regular-season games) make "
    "this a harder forecasting problem than basketball. Do your own research."
)


@dataclass(frozen=True)
class Settings:
    """Immutable settings for the football package."""

    # --- Paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data_cache" / "football"
    # v1 at the original location; v2/v3/v4 nest deeper.
    models_dir: Path = PROJECT_ROOT / "models_artifacts" / "football"
    models_dir_v2: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v2"
    models_dir_v3: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v3"
    models_dir_v3_1: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v3_1"
    models_dir_v4: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v4"
    models_dir_v4_1: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v4_1"
    models_dir_v5: Path = PROJECT_ROOT / "models_artifacts" / "football" / "v5"
    logs_dir: Path = PROJECT_ROOT / "logs"
    log_file: Path = PROJECT_ROOT / "logs" / "ballknower_gridiron.log"

    # --- Model version -----------------------------------------------------
    # Active model used by CLI scripts + predict_nfl.py.
    # "v1" = baseline (ELO + rest + form + bye/short-week + HFA).
    # "v2" = adds QB rating + QB-healthy toggle. ← current best
    # "v3" = v2 + team EPA + points/play efficiency.
    # "v4" = planned: weather, OL/DL/ST, international, skill stars.
    active_model_version: str = os.getenv("NFL_MODEL_VERSION", "v2").lower()

    # --- Inference defaults ------------------------------------------------
    # Default --blend-elo weight (analogous to NBA's setting). NFL has higher
    # variance per game, so ELO blending tends to help more than in NBA at
    # least until v3+ are tuned. We default to 0.50 = "trust model and ELO
    # equally" — re-tune after backtests.
    default_blend_elo: float = float(os.getenv("NFL_BLEND_ELO_DEFAULT", "0.50"))

    # --- NFL data ----------------------------------------------------------
    # How many seasons of history to pull. 12 takes us back to ~2014, after
    # the NFL fully embraced the modern passing era. Pre-2014 the league
    # was meaningfully different (rules, pace, QB rating distributions).
    nfl_seasons_back: int = _env_int("NFL_SEASONS_BACK", 12)

    # --- Team ELO ----------------------------------------------------------
    elo_initial_rating: float = 1500.0
    # K-factor: NFL teams play ~17 games/season, so each game matters more
    # than an NBA game (82 games/season). 25 is the FiveThirtyEight NFL
    # default; NBA uses 20.
    elo_k_factor: float = _env_float("NFL_ELO_K", 25.0)
    # Home-field advantage in ELO points. ~55 ELO points ≈ 2.5 points of
    # margin advantage at home. Has been declining since COVID — watch this.
    # Historically the NFL number was ~3 pts (~65 ELO); we go slightly lower
    # to reflect the recent trend.
    elo_hca: float = _env_float("NFL_ELO_HCA", 55.0)
    # Off-season regression toward 1500. NFL teams change more dramatically
    # than NBA teams (free agency, draft, QB injuries reshape rosters).
    # 0.33 = pull 1/3 of the gap to 1500 at season start. NBA uses 0.25.
    elo_season_regression: float = _env_float("NFL_ELO_SEASON_REG", 0.33)
    # MOV cap. NFL margins go higher than NBA (3 TDs + FG = 24 points), so
    # the cap is correspondingly higher. NBA uses 15.
    mov_max_margin: int = _env_int("NFL_MOV_MAX", 24)
    # Playoff multiplier — playoff games matter much more in NFL since
    # there are only ~13 of them per season (vs ~85 in NBA). NBA uses 1.05.
    playoff_multiplier: float = _env_float("NFL_PLAYOFF_MULT", 1.20)

    # --- Rest / schedule features -----------------------------------------
    # NFL games are weekly. "Normal" rest is 7 days. We flag short weeks
    # (Thursday games = ~4 days rest after a Sunday game) and bye weeks
    # (~14 days rest). Cap is generous since rest matters a LOT in NFL.
    rest_cap_days: int = _env_int("NFL_REST_CAP_DAYS", 14)
    short_week_threshold: int = _env_int("NFL_SHORT_WEEK_DAYS", 5)  # ≤ days = short
    bye_week_threshold: int = _env_int("NFL_BYE_WEEK_DAYS", 10)     # ≥ days = bye

    # --- QB rating (v2) ----------------------------------------------------
    # Minimum games played for a QB to be "qualified" — i.e., included in
    # the z-score baseline used by the QB composite. NFL season is only 17
    # games, so the threshold is correspondingly small.
    qb_min_games: int = _env_int("NFL_QB_MIN_GAMES", 4)
    # Minimum pass attempts (per-game average) for a QB to be qualified.
    # Filters out gimmick / wildcat / scout-team appearances.
    qb_min_attempts_per_game: float = _env_float("NFL_QB_MIN_ATT_PG", 15.0)
    # Backup QB penalty applied beyond the rating swap. If we drop the
    # starter, we ALSO subtract this from the team's effective QB rating
    # to capture (a) system-fit uncertainty and (b) the broader downstream
    # effects (worse OL play-calling, less protection, etc).
    backup_qb_penalty: float = _env_float("NFL_BACKUP_QB_PENALTY", 1.0)

    # --- Team efficiency (v3) — ROLLING version --------------------------
    # The original "weeks 1-2 prior, blend, current" spec is reframed in
    # terms of GAMES PLAYED so it handles bye weeks correctly (a team on
    # bye Week 1 has 0 games played at Week 2, so Week 2 should still be
    # full-prior for that team).
    #   games_played < blend_start_games  -> 100% prior season
    #   games_played >= blend_end_games   -> 100% current season (rolling)
    #   in between                        -> linear blend
    # Defaults (1, 4) reproduce the original week-based intent:
    #   games_played=0,1 (week 1-2)     -> prior
    #   games_played=2,3 (week 3-4)     -> blend
    #   games_played=4+ (week 5+)       -> current
    blend_start_games: int = _env_int("NFL_BLEND_START_GAMES", 1)
    blend_end_games: int = _env_int("NFL_BLEND_END_GAMES", 4)
    # Legacy week-based names kept for backwards compat (unused by rolling
    # pipeline, but anything that imported them won't break).
    early_season_blend_start_week: int = _env_int("NFL_EARLY_BLEND_START_WK", 3)
    early_season_blend_end_week: int = _env_int("NFL_EARLY_BLEND_END_WK", 5)

    # --- Model training ----------------------------------------------------
    test_fraction: float = _env_float("NFL_TEST_FRACTION", 0.15)
    xgb_n_estimators: int = _env_int("NFL_XGB_N_ESTIMATORS", 400)
    xgb_max_depth: int = _env_int("NFL_XGB_MAX_DEPTH", 4)  # shallower than NBA (less data)
    xgb_learning_rate: float = _env_float("NFL_XGB_LEARNING_RATE", 0.05)
    random_state: int = _env_int("RANDOM_STATE", 42)
    high_confidence_margin: float = _env_float("NFL_HIGH_CONF_MARGIN", 0.15)

    # --- Logging -----------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir, self.models_dir, self.models_dir_v2,
            self.models_dir_v3, self.models_dir_v3_1,
            self.models_dir_v4, self.models_dir_v4_1,
            self.models_dir_v5, self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def models_dir_for(self, version: str) -> Path:
        """Resolve the artifact directory for a given model version.
        
        Accepts both 'v3.1' / 'v4.1' (the user-facing CLI form) and
        'v3_1' / 'v4_1' (filesystem-safe form). Both resolve to the
        same directory.
        """
        v = (version or "v1").lower().replace(".", "_")
        return {
            "v1":   self.models_dir,
            "v2":   self.models_dir_v2,
            "v3":   self.models_dir_v3,
            "v3_1": self.models_dir_v3_1,
            "v4":   self.models_dir_v4,
            "v4_1": self.models_dir_v4_1,
            "v5":   self.models_dir_v5,
        }.get(v, self.models_dir)


settings = Settings()
settings.ensure_dirs()
