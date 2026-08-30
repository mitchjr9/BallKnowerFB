"""
ballknower_gridiron.ledger.emit_records
=======================================

Gridiron's **emitter** for the BallKnower forecast ledger. This does not define
the schema — that already exists (SQL DDL, Pydantic models, resolution policy,
invariant suite), and Quad emits against the same contract. This module maps
Gridiron's output onto it and enforces the invariants locally, so a malformed
row fails *here* rather than at the ledger boundary.

Design rules inherited from the schema, and why each shows up in this file:

* **One record = one resolvable proposition.** An NFL game emits *two* records,
  not one: a probabilistic moneyline record from V3.1 and a distributional
  margin record from V5. They resolve independently — you can be right about
  the winner and wrong about the number — and they're scored by different
  metrics (Brier / log-loss vs. CRPS) that must never be averaged into one
  headline.

* **Namespaced entity IDs.** ``nfl:team:kc``, ``nfl:event:2026_01_dal_phi``.

* **The leakage invariant is a rejected write, not a lint.** Two invariants,
  deliberately separate:
    - *Leakage* — ``asof_ts <= event_start_ts``. Every row, always. A forecast
      may never be built from data that only existed after the outcome was
      determinable.
    - *Pre-registration* — ``asof_ts <= committed_ts <= event_start_ts``. Only
      for ``provenance='live'``. A backfilled backtest row is legitimately
      committed today about a 2019 game; that's what ``provenance`` is for, and
      why public calibration surfaces filter to live rows only.

* **Tier is a pure function**, versioned, never hand-set. It takes probability
  *and* ``data_depth``, which is the honest thing to do for the NFL: a Week 1
  rating is last season's ELO regressed 33% toward the mean plus a prior-season
  EPA anchor, not anything that has happened this year. Thin data caps the
  tier. ``assert_tier`` rejects a mismatch.

* **Gridiron-specific: tier is computed on the BLENDED probability**, because
  that's the number the newsletter publishes. Tiering on the raw model
  probability while publishing the blend would make the ledger grade a
  different forecast than the one readers saw. The blend weight is part of
  ``config_hash`` so a change to it is visible in the record.

* **Nothing derived is stored.** No Brier, no stake, no feature values — those
  are views or config references (``config_hash``, ``code_commit``).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

SCHEMA_VERSION = "1"
TIER_CONFIG_VERSION = "gridiron-tier-v1"
RESOLUTION_POLICY_VERSION = "nfl-v1-draft"
MODEL_ID = "ballknower.gridiron"

LEAGUE = "nfl"


class LedgerError(ValueError):
    """Raised on any row that must not be written."""


# --------------------------------------------------------------------------- #
# Entity identity
# --------------------------------------------------------------------------- #
def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


def team_id(name: str) -> str:
    return f"{LEAGUE}:team:{slug(name)}"


def event_id(game_id, season, week, home: str, away: str) -> str:
    """
    Prefer nflverse's game_id (e.g. ``2026_01_DAL_PHI``) — it's stable, it's
    what every downstream nflverse table joins on, and it makes the grader's
    job trivial. Fall back to a deterministic composite.
    """
    if game_id is not None and str(game_id).strip() not in ("", "nan", "None"):
        return f"{LEAGUE}:event:{slug(game_id)}"
    return f"{LEAGUE}:event:{season}_w{week}_{slug(away)}_at_{slug(home)}"


# --------------------------------------------------------------------------- #
# Tier — pure function of (prob, data_depth), versioned
# --------------------------------------------------------------------------- #
def data_depth_for(games_played: Optional[int]) -> str:
    """
    How much *played* football this season backs the rating.

    Thresholds match Quad's deliberately: the ledger is cross-sport, and
    ``data_depth`` has to mean the same thing in an NFL row and a CFB row or
    the pooled reliability curve is comparing different things under one label.

    Load-bearing for the NFL specifically. In Week 1 a team's ELO is last
    season's rating regressed 33% toward 1500 and its EPA features come from a
    prior-season anchor — nothing that has happened this year. Publishing a
    "Lock" off that is advertising confidence the model has not earned.
    """
    if games_played is None:
        return "none"
    if games_played >= 5:
        return "rich"
    if games_played >= 2:
        return "thin"
    return "none"


_TIER_ORDER = ["Pass", "Lean", "Strong", "Lock"]
_DEPTH_CAP = {"rich": "Lock", "thin": "Strong", "none": "Lean"}

# Favorite-probability thresholds. These mirror content_utils.CONFIDENCE_TIERS,
# which are expressed in confidence space (2*|p-0.5|): 0.50 / 0.30 / 0.15 map
# to favorite probabilities of 0.75 / 0.65 / 0.575.
_TIER_THRESHOLDS = [(0.75, "Lock"), (0.65, "Strong"), (0.575, "Lean")]


def compute_tier(prob: float, data_depth: str) -> str:
    """
    Deterministic tier. Never set this by hand — hand-setting contaminates
    tier-level calibration with discretion, and the reliability curve ends up
    measuring judgement rather than the model.

    Delegates to `content_utils.confidence_tier`, which is what the newsletter
    and blog print. One definition, deliberately: if the ledger tiered
    independently, a reader could see "Lock" while the graded record carried
    "Lean" for the same forecast, and every tier-level reliability curve would
    quietly be measuring a different thing than the one published.
    """
    from ballknower_gridiron.utils.content_utils import (
        confidence_tier, DEPTH_CAP as _CU_CAP,
    )
    confidence = abs(float(prob) - 0.5) * 2.0
    # Map the depth label back to a representative game count so the shared
    # function does the capping.
    representative = {"rich": 8, "thin": 3, "none": 0}.get(data_depth, 0)
    label = confidence_tier(confidence, representative)
    return label.split()[-1]        # strip the emoji, keep the word


def assert_tier(record: "ForecastRecord") -> None:
    expected = compute_tier(
        record.prob if record.prob is not None else 0.5, record.data_depth)
    if record.tier != expected:
        raise LedgerError(
            f"tier {record.tier!r} was hand-set; {TIER_CONFIG_VERSION} computes "
            f"{expected!r} for prob={record.prob} depth={record.data_depth}")


# --------------------------------------------------------------------------- #
# Provenance / code identity
# --------------------------------------------------------------------------- #
def code_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def config_hash(blend_elo: Optional[float] = None) -> str:
    """
    Hash the settings that actually change a published prediction.

    ``blend_elo`` is included and passed explicitly rather than read from
    settings, because the pipeline can override it per run — and two rows
    produced under different blend weights are genuinely different forecasts
    even when every other input matches.
    """
    from ballknower_gridiron.config.settings import settings
    payload = {
        "model_version": settings.active_model_version,
        "blend_elo": (settings.default_blend_elo if blend_elo is None
                      else round(float(blend_elo), 4)),
        "elo": [settings.elo_k_factor, settings.elo_hca,
                settings.elo_season_regression, settings.mov_max_margin,
                settings.playoff_multiplier, settings.elo_initial_rating],
        "xgb": [settings.xgb_n_estimators, settings.xgb_max_depth,
                settings.xgb_learning_rate],
        "data": [settings.nfl_seasons_back, settings.test_fraction],
        "tiers": [list(t) for t in _TIER_THRESHOLDS],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
@dataclass
class ForecastRecord:
    # identity
    forecast_id: str
    schema_version: str
    record_family: str            # "probabilistic" | "distributional"

    # proposition (self-describing — a stranger must be able to grade it)
    market_type: str              # moneyline | spread
    subject_id: str
    opponent_id: Optional[str]
    event_id: str
    side: str
    line: Optional[float]
    proposition_text: str

    # forecast payload
    prob: Optional[float] = None          # probabilistic family
    mu: Optional[float] = None            # distributional family
    sigma: Optional[float] = None

    # tiering / depth
    tier: str = "Pass"
    tier_config_version: str = TIER_CONFIG_VERSION
    data_depth: str = "none"

    # market comparison
    market_source: Optional[str] = None
    market_prob_devig: Optional[float] = None
    market_line: Optional[float] = None
    market_asof_ts: Optional[str] = None
    devig_method: Optional[str] = None

    # time
    asof_ts: str = ""
    created_ts: str = ""
    committed_ts: str = ""
    event_start_ts: str = ""

    # model / provenance
    model_id: str = MODEL_ID
    model_version: str = "v3.1"
    config_hash: str = ""
    code_commit: str = ""
    provenance: str = "live"      # live | backfill | replay
    policy_version: str = RESOLUTION_POLICY_VERSION

    # chain
    prev_hash: Optional[str] = None
    row_hash: str = ""
    supersedes: Optional[str] = None

    # resolution (written later, by the grader — never by this package)
    resolution_status: str = "pending"

    def content_tuple(self) -> tuple:
        return (self.subject_id, self.event_id, self.market_type,
                self.line, self.side, self.model_id, self.model_version,
                self.asof_ts)

    def compute_id(self) -> str:
        raw = "|".join("" if v is None else str(v) for v in self.content_tuple())
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def compute_row_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "row_hash"}
        raw = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Every check here is a rejected write, not a warning."""
        if self.record_family not in ("probabilistic", "distributional"):
            raise LedgerError(f"bad record_family {self.record_family!r}")

        if self.record_family == "probabilistic":
            if self.prob is None or not (0.0 < self.prob < 1.0):
                raise LedgerError(f"prob must be in (0,1), got {self.prob}")
            if self.mu is not None:
                raise LedgerError("probabilistic record must not carry mu")
        else:
            if self.mu is None or self.sigma is None or self.sigma <= 0:
                raise LedgerError("distributional record needs mu and sigma>0")
            if self.prob is not None:
                raise LedgerError("distributional record must not carry prob")

        for f in ("asof_ts", "committed_ts", "event_start_ts"):
            if not getattr(self, f):
                raise LedgerError(f"missing timestamp {f}")

        a = _parse(self.asof_ts)
        c = _parse(self.committed_ts)
        e = _parse(self.event_start_ts)

        # LEAKAGE — every row, always.
        if a > e:
            raise LedgerError(
                f"leakage invariant violated: asof_ts ({self.asof_ts}) is after "
                f"event_start_ts ({self.event_start_ts}) — the model saw data "
                "from after the game")

        # PRE-REGISTRATION — live rows only.
        if self.provenance == "live":
            if c > e:
                raise LedgerError(
                    f"already kicked off — kickoff {self.event_start_ts}, "
                    f"committed {self.committed_ts}. A row committed after "
                    "kickoff is not a forecast. Emit it with "
                    "--provenance backfill, which is quarantined from the "
                    "public calibration curve.")
            if a > c:
                raise LedgerError(
                    f"asof_ts ({self.asof_ts}) is after committed_ts "
                    f"({self.committed_ts}) — the model saw data from after "
                    "the moment we claim to have published.")
        elif a > c:
            raise LedgerError(
                f"asof_ts ({self.asof_ts}) is after committed_ts "
                f"({self.committed_ts})")

        if self.provenance not in ("live", "backfill", "replay"):
            raise LedgerError(f"bad provenance {self.provenance!r}")

        if self.forecast_id != self.compute_id():
            raise LedgerError("forecast_id does not match its content tuple")

        if self.record_family == "probabilistic":
            assert_tier(self)

    def finalize(self, prev_hash: Optional[str]) -> "ForecastRecord":
        self.forecast_id = self.compute_id()
        self.prev_hash = prev_hash
        self.validate()
        self.row_hash = self.compute_row_hash()
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def _parse(ts: str) -> datetime:
    s = str(ts).replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Proposition text — generated, never hand-typed
# --------------------------------------------------------------------------- #
def moneyline_text(pick: str, opponent: str, kickoff: str) -> str:
    return f"{pick} defeats {opponent} ({kickoff[:10]})"


def spread_text(home: str, away: str, mu: float, kickoff: str) -> str:
    return (f"Final margin, {home} minus {away} "
            f"({kickoff[:10]}); model mean {mu:+.1f}")


# --------------------------------------------------------------------------- #
# Manifest (chain head + counts) — the thing that gets git-committed
# --------------------------------------------------------------------------- #
def build_manifest(records: List[ForecastRecord], batch_label: str,
                   blend_elo: Optional[float] = None) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch": batch_label,
        "emitter": MODEL_ID,
        "generated_ts": now_iso(),
        "record_count": len(records),
        "chain_head": records[-1].row_hash if records else None,
        "code_commit": code_commit(),
        "config_hash": config_hash(blend_elo),
        "provenance_counts": _counts([r.provenance for r in records]),
        "family_counts": _counts([r.record_family for r in records]),
        "tier_counts": _counts([r.tier for r in records
                                if r.record_family == "probabilistic"]),
        "depth_counts": _counts([r.data_depth for r in records
                                 if r.record_family == "probabilistic"]),
    }


def _counts(vals: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
