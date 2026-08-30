"""
ballknower_gridiron.ledger.selftest
===================================

Invariant suite for Gridiron's ledger emitter, in the same spirit as the
schema's own regression tests and Quad's: every rule that matters is asserted
here, including the ones that are supposed to *fail*.

A validator that has never rejected anything is not evidence it works.

    python -m ballknower_gridiron.ledger.selftest
"""
from __future__ import annotations

from typing import List, Tuple

from ballknower_gridiron.ledger.emit_records import (
    ForecastRecord, LedgerError, build_manifest, compute_tier, data_depth_for,
    event_id, team_id,
)

PASSED: List[str] = []
FAILED: List[Tuple[str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        PASSED.append(name)
    except AssertionError as e:
        FAILED.append((name, f"assertion: {e}"))
    except Exception as e:  # noqa: BLE001 — the suite reports whatever it caught
        FAILED.append((name, f"{type(e).__name__}: {e}"))


def expect_reject(name: str, fn, must_contain: str = "") -> None:
    try:
        fn()
    except LedgerError as e:
        if must_contain and must_contain.lower() not in str(e).lower():
            FAILED.append((name, f"rejected, but not for the expected reason: {e}"))
        else:
            PASSED.append(name)
        return
    except Exception as e:  # noqa: BLE001
        FAILED.append((name, f"wrong exception type: {type(e).__name__}: {e}"))
        return
    FAILED.append((name, "row was ACCEPTED but should have been rejected"))


def _base(**over) -> ForecastRecord:
    d = dict(
        forecast_id="", schema_version="1", record_family="probabilistic",
        market_type="moneyline",
        subject_id=team_id("KC"), opponent_id=team_id("BUF"),
        event_id=event_id("2026_01_BUF_KC", 2026, 1, "KC", "BUF"),
        side="home", line=None,
        proposition_text="KC defeats BUF (2026-09-10)",
        prob=0.663, data_depth="none",
        asof_ts="2026-09-01T00:00:00+00:00",
        created_ts="2026-09-03T10:00:00+00:00",
        committed_ts="2026-09-03T10:00:00+00:00",
        event_start_ts="2026-09-10T00:20:00+00:00",
        model_version="v3.1", config_hash="abc123", code_commit="deadbee",
        provenance="live",
    )
    d.update(over)
    # Tier is a pure function, so the fixture must derive it too — hardcoding a
    # tier here would just be testing that the fixture agrees with itself.
    if "tier" not in over and d.get("record_family") == "probabilistic":
        d["tier"] = compute_tier(d["prob"], d["data_depth"])
    elif "tier" not in over:
        d["tier"] = "Pass"
    return ForecastRecord(**d)


def run() -> int:
    # --- identity ------------------------------------------------------- #
    check("team_id is namespaced and slugged",
          lambda: _assert(team_id("KC") == "nfl:team:kc"))
    check("event_id prefers the nflverse game_id",
          lambda: _assert(event_id("2026_01_DAL_PHI", 2026, 1, "PHI", "DAL")
                          == "nfl:event:2026_01_dal_phi"))
    check("event_id falls back deterministically",
          lambda: _assert(event_id(None, 2026, 1, "KC", "BUF")
                          == "nfl:event:2026_w1_buf_at_kc"))

    # --- happy path ----------------------------------------------------- #
    check("valid live row is accepted", lambda: _base().finalize(None))
    check("forecast_id is deterministic across runs",
          lambda: _assert(_base().finalize(None).forecast_id
                          == _base().finalize(None).forecast_id))
    check("changing content changes the id",
          lambda: _assert(_base().finalize(None).forecast_id
                          != _base(line=-3.5).finalize(None).forecast_id))
    check("hash chain links rows", _chain_links)

    # --- the invariants ------------------------------------------------- #
    expect_reject("leakage: asof after event_start",
                  lambda: _base(asof_ts="2026-09-11T00:00:00+00:00",
                                committed_ts="2026-09-11T01:00:00+00:00"
                                ).finalize(None),
                  "leakage")
    expect_reject("pre-registration: committed after kickoff (live)",
                  lambda: _base(committed_ts="2026-09-11T00:00:00+00:00"
                                ).finalize(None),
                  "already kicked off")
    expect_reject("asof after committed (live) is refused",
                  lambda: _base(asof_ts="2026-09-05T00:00:00+00:00",
                                committed_ts="2026-09-03T10:00:00+00:00"
                                ).finalize(None),
                  "after the moment we claim to have published")
    check("same post-kickoff row is legal as backfill",
          lambda: _base(committed_ts="2026-09-11T00:00:00+00:00",
                        provenance="backfill").finalize(None))
    expect_reject("backfill still cannot see the future",
                  lambda: _base(asof_ts="2026-09-15T00:00:00+00:00",
                                committed_ts="2026-09-16T00:00:00+00:00",
                                provenance="backfill").finalize(None),
                  "leakage")

    # --- families ------------------------------------------------------- #
    expect_reject("probabilistic row may not carry mu",
                  lambda: _base(mu=7.5).finalize(None), "must not carry mu")
    expect_reject("distributional row may not carry prob",
                  lambda: _base(record_family="distributional", mu=7.5,
                                sigma=13.3).finalize(None), "must not carry prob")
    check("valid distributional row is accepted",
          lambda: _base(record_family="distributional", market_type="spread",
                        prob=None, mu=7.5, sigma=13.3, tier="Pass"
                        ).finalize(None))
    expect_reject("sigma must be positive",
                  lambda: _base(record_family="distributional", prob=None,
                                mu=7.5, sigma=0.0).finalize(None), "sigma")
    expect_reject("prob must be inside (0,1)",
                  lambda: _base(prob=1.0).finalize(None), "prob")

    # --- tiering -------------------------------------------------------- #
    check("depth mapping",
          lambda: _assert(data_depth_for(0) == "none"
                          and data_depth_for(3) == "thin"
                          and data_depth_for(9) == "rich"
                          and data_depth_for(None) == "none"))
    check("thin data caps the tier below Lock",
          lambda: _assert(compute_tier(0.97, "thin") != "Lock"))
    check("no data caps the tier at Lean",
          lambda: _assert(compute_tier(0.99, "none") == "Lean"))
    check("rich data allows Lock",
          lambda: _assert(compute_tier(0.80, "rich") == "Lock"))
    check("tier thresholds match the newsletter's",
          _tiers_match_newsletter)
    check("tier is symmetric about 0.5 (away favorites tier the same)",
          lambda: _assert(compute_tier(0.20, "rich")
                          == compute_tier(0.80, "rich")))
    expect_reject("hand-set tier is rejected",
                  lambda: _base(tier="Lock").finalize(None), "hand-set")

    # --- provenance / manifest ------------------------------------------ #
    expect_reject("unknown provenance is rejected",
                  lambda: _base(provenance="guess").finalize(None), "provenance")
    check("manifest carries chain head and counts", _manifest_ok)
    check("config_hash changes with the blend weight", _config_hash_blend)

    # --- report --------------------------------------------------------- #
    print("=" * 66)
    print("  GRIDIRON LEDGER EMITTER — INVARIANT SUITE")
    print("=" * 66)
    for n in PASSED:
        print(f"  ✓ {n}")
    for n, why in FAILED:
        print(f"  ✗ {n}\n      {why}")
    total = len(PASSED) + len(FAILED)
    print("-" * 66)
    print(f"  {len(PASSED)}/{total} passed")
    return 0 if not FAILED else 1


def _assert(cond: bool) -> None:
    if not cond:
        raise AssertionError("condition was false")


def _chain_links() -> None:
    a = _base().finalize(None)
    b = _base(line=-3.5).finalize(a.row_hash)
    _assert(b.prev_hash == a.row_hash and a.row_hash != b.row_hash)


def _tiers_match_newsletter() -> None:
    """
    The ledger's tier thresholds and the newsletter's must not drift apart.
    If they do, readers see one label and the graded track record carries
    another — which would quietly corrupt every tier-level reliability curve.
    """
    from ballknower_gridiron.utils.content_utils import confidence_tier
    for fav_prob in (0.52, 0.58, 0.66, 0.70, 0.76, 0.88):
        conf = abs(fav_prob - 0.5) * 2.0
        newsletter = confidence_tier(conf)
        ledger = compute_tier(fav_prob, "rich")   # rich = uncapped
        # newsletter labels carry emoji; compare the word only
        word = newsletter.split()[-1]
        _assert(word == ledger)


def _manifest_ok() -> None:
    a = _base().finalize(None)
    b = _base(record_family="distributional", market_type="spread", prob=None,
              mu=7.5, sigma=13.3, tier="Pass").finalize(a.row_hash)
    m = build_manifest([a, b], "nfl-test")
    _assert(m["record_count"] == 2)
    _assert(m["chain_head"] == b.row_hash)
    _assert(m["family_counts"] == {"distributional": 1, "probabilistic": 1})


def _config_hash_blend() -> None:
    from ballknower_gridiron.ledger.emit_records import config_hash
    _assert(config_hash(0.0) != config_hash(0.5))


if __name__ == "__main__":
    raise SystemExit(run())
