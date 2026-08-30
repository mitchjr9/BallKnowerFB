"""
BallKnower Gridiron
===================

NFL prediction & newsletter content engine. Third sibling package
alongside `ballknower_insights` (tennis) and `ballknower_hoops` (NBA).
Shares the project root (venv, .env, logs) but no Python code with
the others.

Architecture mirrors `ballknower_hoops` deliberately so the codebase
stays consistent across sports:

  * Team-level ELO with HCA + MOV cap + off-season regression
  * Chronological feature replay (no leakage)
  * Calibrated XGBoost classifier
  * Layered model versions (v1 → v2 → v3 → ...) that inherit features

NFL-specific knobs that differ from the NBA package:
  * QB rating is the dominant "star" signal (v2)
  * Rest is measured weekly, not per-day; short-week/bye flags replace B2B
  * Bigger off-season regression (rosters change more dramatically)
  * Schedules already encode rest, weather, and neutral-site flags
  * Initial-season player data uses prior-season fallbacks (only 17 games)

For entertainment and educational purposes only — not financial or
betting advice.
"""

__version__ = "0.1.0"
