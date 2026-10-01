"""
ballknower_gridiron.ledger.score
================================

The public calibration surface. Joins forecasts to resolutions and computes
scores **on demand** — nothing here is ever written back into the ledger.

    python -m ballknower_gridiron.ledger.score
    python -m ballknower_gridiron.ledger.score --season 2026
    python -m ballknower_gridiron.ledger.score --include-backfill
    python -m ballknower_gridiron.ledger.score --markdown > docs/calibration.md

Why scores are computed and not stored
--------------------------------------
A stored score is a number nobody can re-derive. Computing from the join means
every figure below is reproducible from two append-only files by anyone who
clones the repo, which is the entire point of publishing a track record rather
than asserting one.

The two families are reported **separately and never averaged**:

* **Probabilistic** (moneyline) — Brier, log loss, ECE, and a reliability
  table by tier and by probability bucket. Brier decomposes into calibration
  plus refinement, so both are shown: a model can be well calibrated and
  useless (always predict the base rate) or sharp and badly calibrated.

* **Distributional** (margin) — CRPS, MAE, bias, and interval coverage at ±1σ
  and ±2σ. Coverage is the honest check on `sigma`: if a 68% interval catches
  90% of games, the stated uncertainty is too wide and the CRPS flatters us.

**Backfill rows are excluded by default.** A row committed after kickoff is not
a forecast, and mixing it into the public curve would inflate the record with
hindsight. `--include-backfill` shows them, clearly separated.

DISCLAIMER: For entertainment and educational purposes only — not financial or
betting advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ballknower_gridiron.config.settings import settings

CONTENT_ROOT = settings.project_root / "content" / "football"
DISCLAIMER = ("For entertainment and educational purposes only — "
              "not financial or betting advice.")

TIER_ORDER = ["Lock", "Strong", "Lean", "Pass"]


# --------------------------------------------------------------------------- #
# Loading the join
# --------------------------------------------------------------------------- #
def canonicalize(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Reduce to one forecast per (event, market): the one that stands.

    The rule, fixed in advance so it can't be chosen after seeing results:

      1. A row named by a later row's `supersedes` is out — it was replaced,
         with a recorded reason, before its event started.
      2. Of what remains, a live row always outranks a backfill row.
      3. Among equals, the EARLIEST committed row stands.

    Earliest-stands, not latest, because the ledger already has an explicit
    mechanism for a legitimate update — `supersedes`, with a mandatory reason.
    A second row for the same game without one is, by the ledger's own rules,
    an unexplained re-issue. Letting the latest row win would let anyone
    re-emit until one looked good; letting the earliest win means the first
    thing you published is what you're held to.

    Resolution status plays no part in choosing. If the standing row is
    ungraded, the game is simply ungraded — a graded duplicate is never
    promoted in its place, because picking whichever row happens to be graded
    is selection on the outcome.

    Returns (kept, dropped). Each dropped row carries `_why`: `exact` when it's
    the same forecast recorded twice (same forecast_id — harmless bookkeeping)
    or `conflicting` when it's a different forecast for the same game (the case
    worth an auditor's attention).
    """
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["event_id"], r["market_type"]), []).append(r)

    kept, dropped = [], []
    for _, grp in groups.items():
        grp.sort(key=lambda r: (0 if r.get("provenance") == "live" else 1,
                                str(r.get("committed_ts", "")),
                                r["forecast_id"]))
        winner = grp[0]
        kept.append(winner)
        for r in grp[1:]:
            why = "exact" if r["forecast_id"] == winner["forecast_id"] else "conflicting"
            dropped.append({**r, "_why": why, "_kept_batch": winner["_batch"]})
    return kept, dropped


def load_joined_with_report(include_backfill: bool = False,
                            season: Optional[int] = None
                            ) -> Tuple[List[dict], List[dict]]:
    """Join the canonical forecast for each game to its resolution."""
    raw: List[dict] = []
    superseded: set = set()
    for ledger_dir in sorted(CONTENT_ROOT.glob("*/ledger")):
        f_path = ledger_dir / "forecasts.jsonl"
        if not f_path.exists():
            continue
        resolutions: Dict[str, dict] = {}
        r_path = ledger_dir / "resolutions.jsonl"
        if r_path.exists():
            for line in r_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    resolutions[r["forecast_id"]] = r
        for line in f_path.read_text().splitlines():
            if not line.strip():
                continue
            fc = json.loads(line)
            if fc.get("supersedes"):
                superseded.add(fc["supersedes"])
            raw.append({**fc, "_res": resolutions.get(fc["forecast_id"]),
                        "_batch": ledger_dir.parent.name})

    candidates = []
    for fc in raw:
        if fc["forecast_id"] in superseded:
            continue
        if not include_backfill and fc.get("provenance") != "live":
            continue
        candidates.append(fc)

    kept, dropped = canonicalize(candidates)

    rows = []
    for fc in kept:
        res = fc["_res"]
        if res is None or res.get("resolution_status") != "resolved":
            continue
        if season is not None:
            ts = str(fc.get("event_start_ts", ""))[:10]
            if ts:
                d = dt.date.fromisoformat(ts)
                if (d.year if d.month >= 9 else d.year - 1) != season:
                    continue
        rows.append(fc)
    return rows, dropped


def load_joined(include_backfill: bool = False,
                season: Optional[int] = None) -> List[dict]:
    return load_joined_with_report(include_backfill, season)[0]


# --------------------------------------------------------------------------- #
# Probabilistic scoring
# --------------------------------------------------------------------------- #
def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def log_loss_one(p: float, y: int) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def reliability(rows: List[dict], n_bins: int = 5) -> List[dict]:
    """Bucket by predicted probability; report predicted vs observed."""
    bins: List[dict] = []
    width = 0.5 / n_bins          # favorite probabilities live in [0.5, 1.0]
    for i in range(n_bins):
        lo, hi = 0.5 + i * width, 0.5 + (i + 1) * width
        sel = [r for r in rows
               if lo <= r["prob"] < hi or (i == n_bins - 1 and r["prob"] == hi)]
        if not sel:
            bins.append({"lo": lo, "hi": hi, "n": 0})
            continue
        pred = sum(r["prob"] for r in sel) / len(sel)
        obs = sum(r["_res"]["outcome"] for r in sel) / len(sel)
        bins.append({"lo": lo, "hi": hi, "n": len(sel),
                     "pred": pred, "obs": obs, "gap": obs - pred})
    return bins


def ece(bins: List[dict], total: int) -> float:
    if total == 0:
        return float("nan")
    return sum(b["n"] / total * abs(b["gap"]) for b in bins if b["n"])


def score_probabilistic(rows: List[dict]) -> dict:
    rows = [r for r in rows if r["record_family"] == "probabilistic"
            and r.get("prob") is not None]
    if not rows:
        return {"n": 0}
    n = len(rows)
    outcomes = [r["_res"]["outcome"] for r in rows]
    probs = [r["prob"] for r in rows]

    base_rate = sum(outcomes) / n
    # Reference Brier for "always predict the base rate" — a model that can't
    # beat this is adding nothing over knowing the favorite wins ~X% of the time.
    ref = sum(brier(base_rate, y) for y in outcomes) / n
    bs = sum(brier(p, y) for p, y in zip(probs, outcomes)) / n

    bins = reliability(rows)
    by_tier: Dict[str, dict] = {}
    for tier in TIER_ORDER:
        sel = [r for r in rows if r.get("tier") == tier]
        if not sel:
            continue
        by_tier[tier] = {
            "n": len(sel),
            "pred": sum(r["prob"] for r in sel) / len(sel),
            "obs": sum(r["_res"]["outcome"] for r in sel) / len(sel),
            "brier": sum(brier(r["prob"], r["_res"]["outcome"])
                         for r in sel) / len(sel),
        }

    # Break out by model version. A slate published under a fallback config
    # leaves permanently-committed rows from a different model in the ledger,
    # and pooling them into one curve would report a calibration number that no
    # single model ever produced. Superseded rows are excluded from the headline
    # for the same reason — the replacement is the forecast that stood.
    by_version: Dict[str, dict] = {}
    versions = {r.get("model_version", "?") for r in rows}
    if len(versions) > 1:
        for v in sorted(versions):
            sel = [r for r in rows if r.get("model_version") == v]
            by_version[v] = {
                "n": len(sel),
                "brier": sum(brier(r["prob"], r["_res"]["outcome"])
                             for r in sel) / len(sel),
                "hit_rate": sum(1 for r in sel
                                if (r["prob"] >= 0.5) ==
                                (r["_res"]["outcome"] == 1)) / len(sel),
            }

    return {
        "n": n,
        "by_version": by_version,
        "hit_rate": sum(1 for p, y in zip(probs, outcomes)
                        if (p >= 0.5) == (y == 1)) / n,
        "brier": bs,
        "brier_reference": ref,
        "brier_skill": (1 - bs / ref) if ref > 0 else float("nan"),
        "log_loss": sum(log_loss_one(p, y) for p, y in zip(probs, outcomes)) / n,
        "ece": ece(bins, n),
        "bins": bins,
        "by_tier": by_tier,
    }


# --------------------------------------------------------------------------- #
# Distributional scoring
# --------------------------------------------------------------------------- #
def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """
    Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery):

        CRPS = sigma * [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y − mu)/sigma

    Lower is better, and it is in the same units as the target — so a CRPS of
    5.8 means "typically about 5.8 points off, penalised for overconfidence".
    """
    if sigma <= 0:
        return float("nan")
    z = (y - mu) / sigma
    return sigma * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z)
                    - 1 / math.sqrt(math.pi))


def score_distributional(rows: List[dict]) -> dict:
    rows = [r for r in rows if r["record_family"] == "distributional"
            and r.get("mu") is not None]
    if not rows:
        return {"n": 0}
    n = len(rows)
    errs, crps_vals, in1, in2 = [], [], 0, 0
    for r in rows:
        mu, sigma = float(r["mu"]), float(r["sigma"])
        y = float(r["_res"]["actual_value"])
        errs.append(y - mu)
        crps_vals.append(crps_gaussian(mu, sigma, y))
        if abs(y - mu) <= sigma:
            in1 += 1
        if abs(y - mu) <= 2 * sigma:
            in2 += 1
    return {
        "n": n,
        "crps": sum(crps_vals) / n,
        "mae": sum(abs(e) for e in errs) / n,
        "rmse": math.sqrt(sum(e * e for e in errs) / n),
        "bias": sum(errs) / n,
        "cover_1sigma": in1 / n,
        "cover_2sigma": in2 / n,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return "  —  " if x != x else f"{x * 100:5.1f}%"


def print_report(prob: dict, dist: dict, n_rows: int, include_backfill: bool,
                 season: Optional[int]) -> None:
    scope = f"season {season}" if season else "all seasons"
    print()
    print("=" * 70)
    print(f"  BALLKNOWER GRIDIRON — CALIBRATION REPORT ({scope})")
    print("=" * 70)
    print(f"  graded rows: {n_rows}"
          + ("   [including backfill]" if include_backfill else "   [live only]"))

    if prob.get("n"):
        print()
        print("  MONEYLINE (probabilistic)")
        print("  " + "-" * 66)
        print(f"    n              {prob['n']}")
        print(f"    hit rate       {_fmt_pct(prob['hit_rate'])}")
        print(f"    Brier          {prob['brier']:.4f}   "
              f"(base-rate reference {prob['brier_reference']:.4f})")
        print(f"    Brier skill    {prob['brier_skill']:+.4f}   "
              f"(>0 beats always-predict-the-base-rate)")
        print(f"    log loss       {prob['log_loss']:.4f}")
        print(f"    ECE            {prob['ece']:.4f}")

        if prob.get("by_version"):
            print()
            print("    By model version  (rows from more than one config are")
            print("    reported separately — pooling them would describe a")
            print("    model that never existed)")
            print(f"      {'version':<10}{'n':>5}{'hit':>9}{'Brier':>10}")
            for v, s in prob["by_version"].items():
                print(f"      {v:<10}{s['n']:>5}{s['hit_rate'] * 100:>8.1f}%"
                      f"{s['brier']:>10.4f}")

        print()
        print("    Reliability — predicted vs observed")
        print(f"      {'bucket':<14}{'n':>5}{'pred':>9}{'obs':>9}{'gap':>9}")
        for b in prob["bins"]:
            if not b["n"]:
                continue
            print(f"      {b['lo']:.2f}-{b['hi']:.2f}   {b['n']:>5}"
                  f"{b['pred'] * 100:>8.1f}%{b['obs'] * 100:>8.1f}%"
                  f"{b['gap'] * 100:>+8.1f}pp")

        if prob["by_tier"]:
            print()
            print("    By tier")
            print(f"      {'tier':<10}{'n':>5}{'pred':>9}{'obs':>9}{'gap':>9}"
                  f"{'Brier':>9}")
            for tier in TIER_ORDER:
                t = prob["by_tier"].get(tier)
                if not t:
                    continue
                print(f"      {tier:<10}{t['n']:>5}{t['pred'] * 100:>8.1f}%"
                      f"{t['obs'] * 100:>8.1f}%"
                      f"{(t['obs'] - t['pred']) * 100:>+8.1f}pp"
                      f"{t['brier']:>9.4f}")

    if dist.get("n"):
        print()
        print("  MARGIN (distributional)")
        print("  " + "-" * 66)
        print(f"    n              {dist['n']}")
        print(f"    CRPS           {dist['crps']:.3f} pts   (lower is better)")
        print(f"    MAE            {dist['mae']:.3f} pts")
        print(f"    RMSE           {dist['rmse']:.3f} pts")
        print(f"    bias           {dist['bias']:+.3f} pts   "
              f"({'over' if dist['bias'] < 0 else 'under'}-predicting home margin)")
        print(f"    ±1σ coverage   {_fmt_pct(dist['cover_1sigma'])}  "
              f"(target 68.3%)")
        print(f"    ±2σ coverage   {_fmt_pct(dist['cover_2sigma'])}  "
              f"(target 95.4%)")
        c1 = dist["cover_1sigma"]
        if dist["n"] >= 30:
            if c1 > 0.78:
                print("      → intervals are too WIDE; sigma is overstated and "
                      "CRPS flatters the model.")
            elif c1 < 0.58:
                print("      → intervals are too NARROW; sigma is understated "
                      "and the model is overconfident.")

    if prob.get("n", 0) < 30 and dist.get("n", 0) < 30:
        print()
        print("  ⓘ Fewer than 30 graded rows in each family. Every number above "
              "is dominated by")
        print("    noise at this sample size — a 60% hit rate on 20 games is "
              "one lucky Sunday.")
        print("    Reliability curves need a few hundred rows before they mean "
              "anything.")

    print()
    print(f"  {DISCLAIMER}")
    print()


def print_collapse(dropped: List[dict], verbose: bool = False) -> None:
    """Say out loud what the one-row-per-game rule removed."""
    if not dropped:
        print("  One forecast per game: no duplicates found.\n")
        return
    exact = [d for d in dropped if d["_why"] == "exact"]
    conf = [d for d in dropped if d["_why"] == "conflicting"]
    print("  ONE FORECAST PER GAME — earliest committed row stands")
    print("  " + "-" * 66)
    if exact:
        print(f"    {len(exact)} exact duplicate(s): the same forecast recorded "
              f"in more than one batch.")
        print("      Harmless bookkeeping — collapsed so each game counts once.")
    if conf:
        print(f"    {len(conf)} CONFLICTING duplicate(s): a different forecast "
              f"for a game that already had one,")
        print("      with no `supersedes` link. The earlier forecast stands.")
    if verbose or conf:
        for d in sorted(dropped, key=lambda x: (x["_why"], x["event_id"])):
            print(f"      [{d['_why']:<11}] {d['event_id']:<34} {d['market_type']:<9} "
                  f"dropped from {d['_batch']}, kept {d['_kept_batch']}")
    print()


def print_markdown(prob: dict, dist: dict, n_rows: int,
                   season: Optional[int]) -> None:
    """Publishable version for docs/ or the newsletter."""
    scope = f"Season {season}" if season else "All seasons"
    print(f"# BallKnower Gridiron — Calibration ({scope})")
    print()
    print(f"_{n_rows} graded forecasts, live rows only. "
          f"Generated {dt.date.today().isoformat()}._")
    print()
    if prob.get("n"):
        print("## Moneyline")
        print()
        print("| metric | value |")
        print("|---|---|")
        print(f"| n | {prob['n']} |")
        print(f"| hit rate | {prob['hit_rate'] * 100:.1f}% |")
        print(f"| Brier | {prob['brier']:.4f} |")
        print(f"| Brier skill | {prob['brier_skill']:+.4f} |")
        print(f"| log loss | {prob['log_loss']:.4f} |")
        print(f"| ECE | {prob['ece']:.4f} |")
        print()
        if prob["by_tier"]:
            print("| tier | n | predicted | observed |")
            print("|---|---|---|---|")
            for tier in TIER_ORDER:
                t = prob["by_tier"].get(tier)
                if t:
                    print(f"| {tier} | {t['n']} | {t['pred'] * 100:.1f}% | "
                          f"{t['obs'] * 100:.1f}% |")
            print()
    if dist.get("n"):
        print("## Margin")
        print()
        print("| metric | value |")
        print("|---|---|")
        print(f"| n | {dist['n']} |")
        print(f"| CRPS | {dist['crps']:.3f} |")
        print(f"| MAE | {dist['mae']:.3f} |")
        print(f"| bias | {dist['bias']:+.3f} |")
        print(f"| ±1σ coverage | {dist['cover_1sigma'] * 100:.1f}% |")
        print(f"| ±2σ coverage | {dist['cover_2sigma'] * 100:.1f}% |")
        print()
    print(f"_{DISCLAIMER}_")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute the public calibration report from the ledger.")
    ap.add_argument("--season", type=int, default=None,
                    help="restrict to one NFL season label")
    ap.add_argument("--include-backfill", action="store_true",
                    help="include non-live rows (excluded by default)")
    ap.add_argument("--markdown", action="store_true",
                    help="emit markdown for publishing")
    ap.add_argument("--verbose", action="store_true",
                    help="list every duplicate row that was collapsed")
    args = ap.parse_args(argv)

    rows, dropped = load_joined_with_report(include_backfill=args.include_backfill,
                                            season=args.season)
    if not rows:
        raise SystemExit(
            "No graded forecasts found.\n"
            "  1. python -m ballknower_gridiron.ledger.emit\n"
            "  2. (wait for the games to finish)\n"
            "  3. python -m ballknower_gridiron.ledger.resolve")

    prob = score_probabilistic(rows)
    dist = score_distributional(rows)

    if args.markdown:
        print_markdown(prob, dist, len(rows), args.season)
    else:
        print_report(prob, dist, len(rows), args.include_backfill, args.season)
        print_collapse(dropped, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
