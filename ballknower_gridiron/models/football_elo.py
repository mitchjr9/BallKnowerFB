"""
ballknower_gridiron.models.football_elo
=======================================

Team-level NFL ELO ratings. Design follows the FiveThirtyEight NFL Elo
approach (which is itself a refinement of their NBA Elo), with NFL-
tuned constants from `settings`:

  1. **Home-field advantage** baked into the expected-score calc. The
     home team gets `+HCA` (default 55 ≈ 2.5 pts) added to its rating.
     HCA has been declining in the NFL post-COVID — we use a slightly
     conservative value vs the historical ~65 (≈3 pts).

  2. **Margin-of-Victory multiplier** with a 24-point cap (configurable).
     NFL blowouts can hit 40+ but reward attenuates above 24 (3 TDs +
     a FG). The MOV multiplier is dampened on expected blowouts and
     amplified on upsets, same FiveThirtyEight formula as NBA.

  3. **Off-season regression**. NFL rosters change more dramatically
     between seasons than NBA (free agency, draft, QB injury chains)
     so we regress *33%* toward 1500 (vs NBA's 25%).

  4. **Playoff multiplier 1.20**. Single-elimination playoff games
     matter more than regular-season ones — and there are only ~13 of
     them per season vs the NBA's 85+, so they need a bigger weight.

  5. **Neutral-site games**. International games (London, Munich, etc.)
     and the Super Bowl are flagged in the schedule data as
     `is_international == 1`. In those games, NO HCA is applied —
     both teams are visitors. ELO updates still happen normally.

The class is serializable via `to_dict()` / `from_dict()` for saving
alongside a trained XGBoost model.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as date_cls
from typing import Dict, Optional

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger(__name__)


def expected_home_score(
    home_rating: float,
    away_rating: float,
    is_international: bool = False,
) -> float:
    """
    Probability the home team wins. HCA baked in unless this is a
    neutral-site (international) game.
    """
    hca = 0.0 if is_international else settings.elo_hca
    diff = (home_rating + hca) - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def mov_multiplier(margin: int, winner_rating: float, loser_rating: float) -> float:
    """
    FiveThirtyEight-style MOV multiplier with NFL-sized cap.

    margin : absolute point differential of this game (positive),
             capped at `settings.mov_max_margin` (default 24).
    winner_rating / loser_rating : pre-game ratings (NO HCA — pure
             ratings so the upset dampener works correctly regardless
             of venue).
    """
    capped = min(abs(margin), settings.mov_max_margin)
    elo_diff = winner_rating - loser_rating  # +ve when favorite won
    return math.log(capped + 1) * 2.2 / (elo_diff * 0.001 + 2.2)


@dataclass
class NFLEloSystem:
    """
    State-tracking NFL team ELO system.

    Attributes
    ----------
    ratings : dict[str, float]
        Team abbreviation (e.g. "KC") -> current rating.
    games_played : dict[str, int]
        Team -> total games processed (informational).
    last_game_date : dict[str, str]
        Team -> ISO date of most-recent game (informational).
    last_season : int | None
        Season int (e.g. 2024) of most recently processed game, used to
        detect season transitions for off-season regression.
    """

    ratings: Dict[str, float] = field(default_factory=dict)
    games_played: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_game_date: Dict[str, str] = field(default_factory=dict)
    last_season: Optional[int] = None

    # -- access ----------------------------------------------------------
    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, settings.elo_initial_rating)

    # -- season transition ----------------------------------------------
    def _regress_all_to_mean(self) -> None:
        """
        Off-season regression. Called when a new NFL season begins.

        NFL teams change more between seasons than NBA teams (free
        agency, draft, blockbuster QB trades), so the regression %
        is higher (default 0.33 vs NBA's 0.25).
        """
        if not self.ratings:
            return
        pct = settings.elo_season_regression
        for team, r in list(self.ratings.items()):
            self.ratings[team] = r + (settings.elo_initial_rating - r) * pct
        log.debug(
            "Applied NFL off-season regression (%.0f%% toward 1500) to %d teams.",
            pct * 100, len(self.ratings),
        )

    def roll_to_season(self, season: int) -> bool:
        """
        Advance the ELO state into `season`, applying off-season regression
        if we haven't already crossed that boundary.

        Extracted so the SERVING path can call the same transition the
        training walk performs. Without it there is a real train/serve skew:
        `_regress_all_to_mean` only ever fires inside `update_game`, which is
        only called while fitting. A model trained through February and asked
        to predict Week 1 in September therefore serves *un-regressed*
        end-of-last-season ratings — while every training example it learned
        from had the 33% regression applied at exactly this point in the
        calendar. Week 1 ELO gaps come out roughly 1.5x too wide, which
        inflates both the `elo_diff` feature and the ELO baseline the
        newsletter blends with.

        Idempotent: rolling to a season we're already in is a no-op, so
        calling it defensively from several entry points is safe.

        Returns True if a transition was applied.
        """
        if self.last_season is None or int(season) <= int(self.last_season):
            return False
        # Regress once per season crossed, mirroring the training walk, which
        # fires at each boundary it passes.
        for _ in range(int(season) - int(self.last_season)):
            self._regress_all_to_mean()
        self.last_season = int(season)
        log.info("Rolled NFL ELO state forward to season %d "
                 "(off-season regression applied).", int(season))
        return True

    # -- match update ----------------------------------------------------
    def update_game(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_date: date_cls,
        season: int,
        is_playoff: bool = False,
        is_international: bool = False,
    ) -> None:
        """
        Process one completed game.

        Order of operations:
          1. If this is a new season, regress all teams toward 1500.
          2. Read pre-game ratings (feature builders read these BEFORE
             calling this method to avoid leakage).
          3. Compute expected score (HCA off if international).
          4. Compute MOV multiplier.
          5. Update both teams' ratings symmetrically with playoff bump
             if applicable.
        """
        # 1. Off-season regression at season boundaries.
        if self.last_season is not None and season != self.last_season:
            self._regress_all_to_mean()
        self.last_season = season

        # 2 & 3. Expected score
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        exp_home = expected_home_score(r_home, r_away, is_international=is_international)

        # 4. MOV multiplier — use raw ratings (no HCA) so the upset
        # dampener is anchored on team strength alone.
        margin = abs(home_score - away_score)
        home_won = home_score > away_score
        if home_won:
            mov = mov_multiplier(margin, r_home, r_away)
        else:
            mov = mov_multiplier(margin, r_away, r_home)

        # NFL playoff bump is bigger than NBA's because there are far fewer
        # playoff games and they matter much more for ratings.
        playoff_mult = settings.playoff_multiplier if is_playoff else 1.0
        k = settings.elo_k_factor * mov * playoff_mult

        # 5. Symmetric update
        actual_home = 1.0 if home_won else 0.0
        delta = k * (actual_home - exp_home)
        self.ratings[home_team] = r_home + delta
        self.ratings[away_team] = r_away - delta

        # Bookkeeping
        self.games_played[home_team] += 1
        self.games_played[away_team] += 1
        iso = game_date.isoformat() if hasattr(game_date, "isoformat") else str(game_date)
        self.last_game_date[home_team] = iso
        self.last_game_date[away_team] = iso

    def fit_history(self, games) -> None:
        """Replay a chronologically-sorted games DataFrame, updating ratings."""
        log.info("Building NFL ELO from %d games …", len(games))
        for row in games.itertuples(index=False):
            game_date = row.game_date
            if hasattr(game_date, "date"):
                game_date = game_date.date()
            self.update_game(
                home_team=row.home_team,
                away_team=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                game_date=game_date,
                season=int(row.season),
                is_playoff=bool(getattr(row, "is_playoff", 0)),
                is_international=bool(getattr(row, "is_international", 0)),
            )
        log.info("ELO build complete. Tracking %d teams.", len(self.ratings))

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "ratings": self.ratings,
            "games_played": dict(self.games_played),
            "last_game_date": self.last_game_date,
            "last_season": self.last_season,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NFLEloSystem":
        obj = cls(
            ratings=dict(d.get("ratings", {})),
            last_game_date=dict(d.get("last_game_date", {})),
            last_season=d.get("last_season"),
        )
        obj.games_played = defaultdict(int, d.get("games_played", {}))
        return obj
