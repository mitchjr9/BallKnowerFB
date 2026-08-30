# BallKnower Gridiron (NFL)

NFL game-prediction package — sibling of `ballknower_hoops` (basketball)
and `ballknower_insights` (tennis). Same chronological-replay +
calibrated-XGBoost pipeline so your mental model carries across all
three sports.

> **DISCLAIMER:** Strictly for entertainment and educational purposes.
> Nothing here constitutes financial, investment, or betting advice.
> NFL games are noisier than NBA games (17 games/team vs 82; single-
> position dominance via QB; weather variability), so probabilities are
> calibrated but **never** guarantees.

---

## What you get

| Version | Features added                                                       | When to use                                |
| ------- | -------------------------------------------------------------------- | ------------------------------------------ |
| **v1**  | ELO + rest + bye/short-week + form (last 5) + HFA + intl + div flag  | Baseline. Trains in ~10s.                  |
| **v2**  | v1 + **QB rating + QB-healthy toggle**                               | Adds the single biggest NFL signal.        |
| **v3**  | v2 + team net points + EPA/play + early-season prior-season blend    | Best when you have 5+ weeks of new data.   |
| v4 plan | Weather, OL/DL ratings, skill-position stars, travel penalty         | Scaffolded — not yet trained.              |

---

## NFL vs NBA — what changed

| Component         | NBA           | NFL (this package)                                  |
| ----------------- | ------------- | --------------------------------------------------- |
| ELO K-factor      | 20            | **25** (fewer games/season → each one weighs more)  |
| Home-field adv.   | 65 ELO ≈ 3pt  | **55 ELO ≈ 2.5pt** (declining post-COVID)           |
| Off-season reg.   | 0.25          | **0.33** (NFL rosters change more)                  |
| MOV cap           | 15            | **24** (3 TDs + FG)                                 |
| Playoff K-mult    | 1.05          | **1.20** (single-elim, fewer games)                 |
| Seasons of data   | 8             | **12** (more variance per game → need more history) |
| Form window       | last-10       | **last-5** (17 games/season makes 10 too sluggish)  |
| Star player       | top-3 by min  | **starting QB** (one position dominates)            |
| Net rating        | LeagueDash    | **EPA/play** (computed from PBP)                    |
| Bench rating      | yes           | no (replaced by QB-healthy toggle)                  |
| International     | no            | **yes** — Neutral-site flag + HCA zeroed            |

---

## Data source

We use **[nflreadpy](https://github.com/nflverse/nflreadpy)** — the
official `nflverse` Python port released after `nfl_data_py` was
archived (September 2025). It uses Polars natively and exposes the
same datasets as the R `nflreadr` package (schedules, play-by-play,
weekly player stats, depth charts, rosters, advanced PFR stats).

Why nflreadpy:
- Maintained by the nflverse team (the original deprecated wrapper's home).
- Schedule data already includes `home_rest`, `away_rest`, `temp`, `wind`,
  `roof`, `surface`, `location` (Home/Neutral), `spread_line`, `total_line`,
  `div_game` — eliminates the "compute rest from game dates" math.
- PBP includes per-play EPA, which is the gold-standard team-quality
  signal in modern NFL analytics.

Loaders in `ballknower_gridiron/data/*.py` are isolated by responsibility
(schedules, QB stats, team efficiency) — if nflverse migrates again, swap
those out without touching the model layer.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Train v2 (default — the QB layer)
python -m ballknower_gridiron.scripts.train_football_model

# Or train v3 (adds team EPA + net-points)
python -m ballknower_gridiron.scripts.train_football_model --version v3

# Predict a game
python predict_nfl.py KC BUF
python predict_nfl.py KC BUF --week 14 --rest-home 4
python predict_nfl.py KC BUF --qb-out HOME           # "what if Mahomes is out?"
python predict_nfl.py KC BUF --international         # London game
python predict_nfl.py KC BUF --playoff --version v3

# Backtest against the ELO baseline + find best blend weight
python -m ballknower_gridiron.scripts.backtest_nfl --version v2
python -m ballknower_gridiron.scripts.backtest_nfl --version v3 --markdown report.md
```

---

## Configuration (env vars)

All settings live in `ballknower_gridiron/config/settings.py`. Every
knob is overridable via an `NFL_*` environment variable (loaded from
the project-root `.env` if present).

```bash
# Pick the active model version (used by predict_nfl.py default)
NFL_MODEL_VERSION=v2

# Tune the ELO system
NFL_ELO_K=25
NFL_ELO_HCA=55
NFL_ELO_SEASON_REG=0.33

# Early-season blend window (weeks where we mix prior + current season)
NFL_EARLY_BLEND_START_WK=3
NFL_EARLY_BLEND_END_WK=5

# QB qualification + backup penalty
NFL_QB_MIN_GAMES=4
NFL_QB_MIN_ATT_PG=15
NFL_BACKUP_QB_PENALTY=1.0

# Default --blend-elo weight at inference
NFL_BLEND_ELO_DEFAULT=0.50
```

---

## QB-out toggle (the feature you asked for)

When you predict a game with `--qb-out HOME` (or `AWAY` or `BOTH`):

1. The model looks up the team's *starting* QB rating from the training
   snapshot — that's the one that gets removed.
2. If a `--home-backup-qb-rating <float>` is provided, use it directly.
3. Otherwise the model checks if the team has a *qualified backup* in
   the training data (a QB who threw enough passes that season to meet
   `NFL_QB_MIN_ATT_PG`). If so, swap that rating in.
4. If neither: apply a generic backup heuristic — starter rating minus
   ~2.5 z-score units (calibrated from FiveThirtyEight's QB ELO
   research on backup performance vs starters).
5. **In all cases**, apply `NFL_BACKUP_QB_PENALTY` (default 1.0
   z-units) on top to capture system-fit uncertainty (worse OL play-
   calling, fewer audibles, etc).

This means you can quickly stress-test "what if Mahomes is out?" without
having to specify exactly who the backup is — but you can if you want
precision (Carson Wentz vs Bailey Zappe ≠ same backup-replacement-level).

---

## QB rating uses BOTH passer rating AND ESPN QBR

The v2/v3 QB composite z-scores 9 stats and sums them with the weights
defined in `models/qb_rating.py`. Two of those nine stats measure QB
efficiency at the league level, and we deliberately include both:

| Metric           | Scale       | Strengths                                                             | Weaknesses                                          | Weight |
| ---------------- | ----------- | --------------------------------------------------------------------- | --------------------------------------------------- | ------ |
| **Passer rating**| 0 – 158.3   | Universal, always available, 1973-vintage but still widely used.      | Doesn't credit rushing/sacks; not opponent-adjusted | 1.5    |
| **ESPN QBR**     | 0 – 100     | Opponent-adjusted, garbage-time-discounted, includes rushing/sacks    | Proprietary EPA model; sometimes slow to update     | 2.0    |

They're correlated (~0.7) so they don't double-count — they correct each
other on the margins. When QBR isn't available (a season's data hasn't
been scraped, or the GitHub source is unreachable), the model
gracefully falls back to passer-rating-only z-scores for that season.

**Data source for QBR**: we fetch ESPN QBR directly from
`https://raw.githubusercontent.com/nflverse/espnscrapeR-data/master/data/qbr-nfl-season.csv`
(the same source the R package `nflreadr::load_espn_qbr()` uses).
`nflreadpy` deliberately doesn't wrap this data, so we fetch the CSV
ourselves in `data/espn_qbr_loader.py`. Available since 2006.

---

## Early-season blend (v3)

The user's spec was: "weeks 1–2 use prior season, 3–4 blend, 5+ current".
That's implemented in `NFLFeatureBuilderV3._blended_team_metrics`:

| Week         | Blend                                       |
| ------------ | ------------------------------------------- |
| 1, 2         | 100% prior season net-pts / EPA             |
| 3            | ⅓ current + ⅔ prior                         |
| 4            | ⅔ current + ⅓ prior                         |
| 5+           | 100% current                                |

Tunable via `NFL_EARLY_BLEND_START_WK` and `NFL_EARLY_BLEND_END_WK`.

The blend is applied at **both** train and inference time so the model
learns to interpret the blended values the way they'll be fed at
prediction time.

---

## Project layout

```
ballknower_gridiron/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py             # all tuning knobs (NFL_* env vars)
├── data/
│   ├── __init__.py
│   ├── football_loader.py      # nflreadpy schedules → normalized DF
│   ├── qb_stats_loader.py      # QB weekly/season stats + depth charts
│   ├── espn_qbr_loader.py      # ESPN QBR from nflverse/espnscrapeR-data
│   └── team_efficiency_loader.py  # team EPA + net-points from PBP
├── models/
│   ├── __init__.py
│   ├── football_elo.py         # NFL-tuned ELO system
│   ├── football_model.py       # v1: ELO + rest + form + flags
│   ├── football_model_v2.py    # + QB rating + QB-out toggle
│   ├── football_model_v3.py    # + team EPA + early-season blend
│   └── qb_rating.py            # z-scored QB composite + starter logic
├── scripts/
│   ├── __init__.py
│   ├── train_football_model.py # CLI: train v1/v2/v3
│   └── backtest_nfl.py         # CLI: backtest + blend sweep + calibration
└── utils/
    ├── __init__.py
    └── logging_utils.py

predict_nfl.py                  # project-root single-game CLI
requirements.txt
```

---

## What's not (yet) in the package

These are v4 / future work. Each is scaffolded in the loaders or
described here but not wired into the trained model:

| Feature                 | Status                                              |
| ----------------------- | --------------------------------------------------- |
| Weather (wind/temp/dome)| Schedule data already exposes these. Need a v4 feature builder that reads `temp`, `wind`, `roof` and applies a small shrinkage toward 0.5 when conditions are extreme. |
| OL/DL ratings           | Need to add `models/line_rating.py` that computes pressure rate, sack rate, yards-before-contact from PBP. ~2 hours of work. |
| Special teams           | Not modeled. Field-position EPA from PBP would give a starting point. |
| Skill-position stars    | The QB rating module's pattern (`qb_rating.py`) extends naturally to WR/RB/TE — same z-score composite over PPG, TDs, yards. |
| International travel    | Currently only zeroes HCA. A full travel penalty would compute haversine distance from home stadium to game venue and time-zone delta. |

---

_Same disclaimer as above: entertainment and education only. Don't bet
the rent._
