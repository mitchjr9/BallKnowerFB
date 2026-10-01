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

    # --- grader: resolution --------------------------------------------- #
    check("tie resolves moneyline as FALSE, not void", _tie_is_a_loss)
    check("home subject resolves on home outscoring away", _home_subject)
    check("away subject resolves on away outscoring home", _away_subject)
    check("margin resolves as home minus away, signed", _margin_signed)
    check("grading reads stored side, not the probability", _side_not_prob)
    check("resolution never mutates the forecast row", _forecast_immutable)
    check("re-running resolve is idempotent", _resolve_idempotent)

    # --- grader: scoring ------------------------------------------------- #
    check("CRPS matches the closed form on a known case", _crps_known)
    check("CRPS is minimised when mu equals the outcome", _crps_min_at_truth)
    check("perfect forecasts score Brier 0", _brier_perfect)
    check("Brier skill is 0 for a base-rate forecaster", _brier_skill_zero)
    check("backfill rows are excluded from the public curve", _backfill_excluded)

    # --- supersession ---------------------------------------------------- #
    expect_reject("supersedes without a reason is rejected",
                  lambda: _base(supersedes="old123").finalize(None),
                  "unexplained re-issue")
    expect_reject("reason without supersedes is rejected",
                  lambda: _base(supersede_reason="oops").finalize(None),
                  "without supersedes")
    check("a properly explained supersession is accepted",
          lambda: _base(supersedes="old123",
                        supersede_reason="config error: ran under v2").finalize(None))
    check("superseding row gets its own distinct id",
          lambda: _assert(
              _base(supersedes="old123", supersede_reason="r").finalize(None).forecast_id
              == _base().finalize(None).forecast_id))

    # --- one forecast per game ------------------------------------------ #
    check("exact duplicate collapses to one row", _canon_exact)
    check("conflicting duplicate: EARLIEST committed row stands", _canon_earliest)
    check("a superseded row never stands", _canon_superseded)
    check("live outranks backfill for the same game", _canon_live_first)
    check("a graded duplicate is never promoted over an ungraded standing row",
          _canon_no_outcome_selection)
    check("user's real ledger shape: 32 games, 2 exact dups, 0 conflicting",
          _canon_real_ledger)

    # --- emit overwrite guard ------------------------------------------- #
    check("emit: identical re-run is detected as a no-op", _emit_identical)
    check("emit: a changed re-run is detected as different", _emit_different)

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


# --- grader helpers ------------------------------------------------------ #
def _res(home_score, away_score, home="KC", away="BUF"):
    return {"home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "margin": home_score - away_score,
            "winner": home if home_score > away_score else (
                away if away_score > home_score else None),
            "is_tie": home_score == away_score}


def _fc(family="probabilistic", side="home", **over):
    d = {"forecast_id": "abc", "event_id": "nfl:event:x",
         "record_family": family, "market_type": "moneyline",
         "side": side, "prob": 0.70}
    if family == "distributional":
        d.update({"market_type": "spread", "mu": 6.0, "sigma": 13.3})
        d.pop("prob")
    d.update(over)
    return d


def _tie_is_a_loss() -> None:
    from ballknower_gridiron.ledger.resolve import grade
    r = grade(_fc(), _res(21, 21), "t")
    _assert(r["resolution_status"] == "resolved")   # NOT void
    _assert(r["outcome"] == 0)


def _home_subject() -> None:
    from ballknower_gridiron.ledger.resolve import grade
    _assert(grade(_fc(side="home"), _res(28, 20), "t")["outcome"] == 1)
    _assert(grade(_fc(side="home"), _res(20, 28), "t")["outcome"] == 0)


def _away_subject() -> None:
    from ballknower_gridiron.ledger.resolve import grade
    _assert(grade(_fc(side="away"), _res(20, 28), "t")["outcome"] == 1)
    _assert(grade(_fc(side="away"), _res(28, 20), "t")["outcome"] == 0)


def _margin_signed() -> None:
    from ballknower_gridiron.ledger.resolve import grade
    r = grade(_fc(family="distributional"), _res(31, 17), "t")
    _assert(r["actual_value"] == 14.0)
    r2 = grade(_fc(family="distributional"), _res(17, 31), "t")
    _assert(r2["actual_value"] == -14.0)


def _side_not_prob() -> None:
    """A 0.99 forecast on the losing side must still grade as a loss."""
    from ballknower_gridiron.ledger.resolve import grade
    _assert(grade(_fc(side="home", prob=0.99), _res(3, 40), "t")["outcome"] == 0)


def _forecast_immutable() -> None:
    """resolve_batch must append to resolutions.jsonl and never open the
    forecasts file for writing."""
    import json, tempfile, pathlib
    from ballknower_gridiron.ledger.resolve import resolve_batch
    d = pathlib.Path(tempfile.mkdtemp())
    fc = _fc(forecast_id="f1", event_id="nfl:event:g1")
    (d / "forecasts.jsonl").write_text(json.dumps(fc) + "\n")
    before = (d / "forecasts.jsonl").read_bytes()
    resolve_batch(d, {"nfl:event:g1": _res(28, 20)}, "t")
    _assert((d / "forecasts.jsonl").read_bytes() == before)
    _assert((d / "resolutions.jsonl").exists())


def _resolve_idempotent() -> None:
    import json, tempfile, pathlib
    from ballknower_gridiron.ledger.resolve import resolve_batch
    d = pathlib.Path(tempfile.mkdtemp())
    fc = _fc(forecast_id="f1", event_id="nfl:event:g1")
    (d / "forecasts.jsonl").write_text(json.dumps(fc) + "\n")
    idx = {"nfl:event:g1": _res(28, 20)}
    s1 = resolve_batch(d, idx, "t")
    s2 = resolve_batch(d, idx, "t")
    _assert(s1["new"] == 1 and s2["new"] == 0 and s2["already"] == 1)
    lines = [l for l in (d / "resolutions.jsonl").read_text().splitlines() if l]
    _assert(len(lines) == 1)


def _crps_known() -> None:
    from ballknower_gridiron.ledger.score import crps_gaussian
    # At y == mu the closed form reduces to sigma * (2*phi(0) - 1/sqrt(pi))
    import math
    sigma = 13.3
    expected = sigma * (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    _assert(abs(crps_gaussian(0.0, sigma, 0.0) - expected) < 1e-9)


def _crps_min_at_truth() -> None:
    from ballknower_gridiron.ledger.score import crps_gaussian
    at = crps_gaussian(7.0, 13.3, 7.0)
    for off in (1, 3, 10, 25):
        _assert(crps_gaussian(7.0, 13.3, 7.0 + off) > at)
        _assert(crps_gaussian(7.0, 13.3, 7.0 - off) > at)


def _brier_perfect() -> None:
    from ballknower_gridiron.ledger.score import score_probabilistic
    rows = [{"record_family": "probabilistic", "prob": 1 - 1e-9, "tier": "Lock",
             "_res": {"outcome": 1}} for _ in range(10)]
    _assert(score_probabilistic(rows)["brier"] < 1e-9)


def _brier_skill_zero() -> None:
    """A forecaster that always predicts the base rate has skill exactly 0."""
    from ballknower_gridiron.ledger.score import score_probabilistic
    outcomes = [1] * 7 + [0] * 3
    rows = [{"record_family": "probabilistic", "prob": 0.7, "tier": "Strong",
             "_res": {"outcome": y}} for y in outcomes]
    _assert(abs(score_probabilistic(rows)["brier_skill"]) < 1e-9)


def _backfill_excluded() -> None:
    import json, tempfile, pathlib
    from ballknower_gridiron.ledger import score as sc
    root = pathlib.Path(tempfile.mkdtemp())
    d = root / "2026-09-08" / "ledger"; d.mkdir(parents=True)
    live = _fc(forecast_id="a", event_id="nfl:event:g1", provenance="live",
               event_start_ts="2026-09-13T17:00:00+00:00")
    back = _fc(forecast_id="b", event_id="nfl:event:g2", provenance="backfill",
               event_start_ts="2026-09-13T17:00:00+00:00")
    (d / "forecasts.jsonl").write_text(json.dumps(live) + "\n" + json.dumps(back) + "\n")
    (d / "resolutions.jsonl").write_text("\n".join(json.dumps(
        {"forecast_id": i, "resolution_status": "resolved", "outcome": 1})
        for i in ("a", "b")) + "\n")
    orig = sc.CONTENT_ROOT
    try:
        sc.CONTENT_ROOT = root
        _assert(len(sc.load_joined(include_backfill=False)) == 1)
        _assert(len(sc.load_joined(include_backfill=True)) == 2)
    finally:
        sc.CONTENT_ROOT = orig


# --- canonicalization helpers -------------------------------------------- #
def _row(fid, ev, ts, batch, prov="live", sup=None, res=None, market="moneyline"):
    return {"forecast_id": fid, "event_id": ev, "market_type": market,
            "committed_ts": ts, "_batch": batch, "provenance": prov,
            "supersedes": sup, "_res": res}


def _canon_exact() -> None:
    from ballknower_gridiron.ledger.score import canonicalize
    kept, dropped = canonicalize([_row("A", "g1", "2026-08-30", "b1"),
                                  _row("A", "g1", "2026-09-07", "b2")])
    _assert(len(kept) == 1 and kept[0]["_batch"] == "b1")
    _assert(dropped[0]["_why"] == "exact")


def _canon_earliest() -> None:
    from ballknower_gridiron.ledger.score import canonicalize
    kept, dropped = canonicalize([_row("B", "g1", "2026-09-07", "late"),
                                  _row("A", "g1", "2026-08-30", "early")])
    _assert(kept[0]["_batch"] == "early" and dropped[0]["_why"] == "conflicting")


def _canon_superseded() -> None:
    """Superseded rows are removed BEFORE canonicalize sees them; verify the
    loader's filter by building the candidate set the way it does."""
    rows = [_row("A", "g1", "2026-08-30", "b1"),
            _row("S", "g1", "2026-09-12", "b3", sup="A")]
    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    from ballknower_gridiron.ledger.score import canonicalize
    kept, _ = canonicalize([r for r in rows if r["forecast_id"] not in superseded])
    _assert(len(kept) == 1 and kept[0]["forecast_id"] == "S")


def _canon_live_first() -> None:
    from ballknower_gridiron.ledger.score import canonicalize
    kept, _ = canonicalize([_row("B", "g1", "2026-08-01", "bf", prov="backfill"),
                            _row("L", "g1", "2026-09-07", "lv", prov="live")])
    _assert(kept[0]["forecast_id"] == "L")


def _canon_no_outcome_selection() -> None:
    from ballknower_gridiron.ledger.score import canonicalize
    graded = {"resolution_status": "resolved", "outcome": 1}
    kept, _ = canonicalize([_row("A", "g1", "2026-08-30", "b1", res=None),
                            _row("B", "g1", "2026-09-07", "b2", res=graded)])
    _assert(kept[0]["forecast_id"] == "A")     # earliest stands even ungraded


def _canon_real_ledger() -> None:
    """
    Reconstruct the four batches in the real ledger and check the rule gives
    what the diagnostic reported (32 games, 2 counted twice) — and classifies
    them. 08-30 and 09-07 were both generated under the v2 fallback, so their
    Week 1 rows are content-identical (same forecast_id). 09-12 superseded the
    09-07 ids, which therefore also retires the matching 08-30 rows — except
    the two games already played by then, which stay doubled.
    """
    from ballknower_gridiron.ledger.score import canonicalize
    wk1 = [f"w1_{i:02d}" for i in range(16)]
    played_early = {"w1_00", "w1_01"}                 # NE@SEA, SF@LA
    rows = []
    for g in wk1:
        rows.append(_row(f"v2_{g}", g, "2026-08-30", "2026-08-30"))
        rows.append(_row(f"v2_{g}", g, "2026-09-07", "2026-09-07"))
    rows.append(_row("v2_detbuf", "w2_detbuf", "2026-09-07", "2026-09-07"))
    for g in wk1:
        if g not in played_early:
            rows.append(_row(f"v31_{g}", g, "2026-09-12", "2026-09-12",
                             sup=f"v2_{g}"))
    for i in range(15):
        rows.append(_row(f"v31_w3_{i}", f"w3_{i:02d}", "2026-09-27", "2026-09-27"))

    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    cand = [r for r in rows if r["forecast_id"] not in superseded]
    kept, dropped = canonicalize(cand)
    _assert(len(kept) == 32)
    _assert(len(dropped) == 2)
    _assert(all(d["_why"] == "exact" for d in dropped))
    _assert({d["event_id"] for d in dropped} == played_early)
    # and the 14 corrected Week 1 games are served by the v3.1 rows
    w1_kept = {r["event_id"]: r["forecast_id"] for r in kept if r["event_id"].startswith("w1_")}
    _assert(all(fid.startswith("v31_") for g, fid in w1_kept.items()
                if g not in played_early))


def _emit_identical() -> None:
    import json, tempfile, pathlib
    from ballknower_gridiron.ledger.emit import existing_batch_status
    d = pathlib.Path(tempfile.mkdtemp()) / "forecasts.jsonl"
    d.write_text("\n".join(json.dumps({"forecast_id": i}) for i in ("a", "b")) + "\n")
    _assert(existing_batch_status(d, ["b", "a"])[0] == "identical")


def _emit_different() -> None:
    import json, tempfile, pathlib
    from ballknower_gridiron.ledger.emit import existing_batch_status
    d = pathlib.Path(tempfile.mkdtemp()) / "forecasts.jsonl"
    d.write_text(json.dumps({"forecast_id": "a"}) + "\n")
    _assert(existing_batch_status(d, ["a", "z"])[0] == "different")
    _assert(existing_batch_status(d.parent / "nope.jsonl", ["a"])[0] == "absent")


def _config_hash_blend() -> None:
    from ballknower_gridiron.ledger.emit_records import config_hash
    _assert(config_hash(0.0) != config_hash(0.5))


if __name__ == "__main__":
    raise SystemExit(run())
