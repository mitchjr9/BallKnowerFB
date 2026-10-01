"""
ballknower_gridiron.ledger.emit
===============================

Turns a Gridiron slate into **forecast ledger rows**: append-only JSONL plus a
manifest carrying the chain head, ready to be git-committed for third-party
timestamp attestation.

    # after weekly_football_pipeline has written predictions.json
    python -m ballknower_gridiron.ledger.emit
    python -m ballknower_gridiron.ledger.emit --date 2026-09-02
    python -m ballknower_gridiron.ledger.emit --provenance backfill

Each game emits **two** records:

  * ``moneyline`` — probabilistic (V3.1's blended win probability), scored by
    Brier / log-loss
  * ``spread``    — distributional (``mu`` = V5's predicted margin, ``sigma`` =
    V5's holdout RMSE read from its metadata), scored by CRPS and interval
    coverage

Two rows rather than one because they resolve independently: a pick can be
right on the winner and wrong on the number. Merging them would make the
reliability curve uncomputable without unpacking, and the two families use
metrics that must never be averaged into a single headline.

Rows that violate the pre-registration invariant — anything where kickoff has
already passed — are **refused and reported**, not silently dropped. Run this
early in the week: Thursday Night Football kicks off before the Sunday slate,
so a Friday run will legitimately refuse TNF.

DISCLAIMER: For entertainment and educational purposes only — not financial or
betting advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import List, Optional, Tuple

from ballknower_gridiron.config.settings import settings
from ballknower_gridiron.ledger.emit_records import (
    SCHEMA_VERSION, ForecastRecord, LedgerError, build_manifest, code_commit,
    compute_tier, config_hash, data_depth_for, event_id, moneyline_text,
    now_iso, spread_text, team_id,
)
from ballknower_gridiron.utils.logging_utils import get_logger

log = get_logger("ledger.emit")

# Fallback margin uncertainty if V5's metadata can't be read. V5's measured
# holdout RMSE is ~13.3 points; guessing lower would make the CRPS scores
# flatter than the model deserves.
DEFAULT_MARGIN_SIGMA = 13.3


def _load_predictions(date_str: str) -> Tuple[List[dict], Path]:
    d = settings.project_root / "content" / "football" / date_str
    p = d / "predictions.json"
    if not p.exists():
        raise SystemExit(
            f"No predictions at {p}\n"
            f"Run: python -m ballknower_gridiron.scripts.weekly_football_pipeline")
    return json.loads(p.read_text()), d


def _margin_sigma(margin_version: str = "v5") -> float:
    """Read RMSE off the trained margin model so sigma isn't a guess."""
    meta = settings.models_dir_for(margin_version) / "metadata.json"
    try:
        m = json.loads(meta.read_text()).get("metrics", {})
        rmse = float(m.get("rmse", 0.0))
        if rmse > 0:
            return rmse
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Could not read %s RMSE from %s (%s) — using %.1f.",
                    margin_version, meta, exc, DEFAULT_MARGIN_SIGMA)
    return DEFAULT_MARGIN_SIGMA


def existing_batch_status(jsonl: Path, new_ids: List[str]) -> Tuple[str, List[dict]]:
    """'absent', 'identical' (same forecast_ids), or 'different'."""
    if not jsonl.exists():
        return "absent", []
    existing = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    same = sorted(r["forecast_id"] for r in existing) == sorted(new_ids)
    return ("identical" if same else "different"), existing


def records_for_game(g: dict, *, asof_ts: str, committed_ts: str,
                     provenance: str, sigma: float, wp_version: str,
                     margin_version: str, blend_elo: Optional[float],
                     ) -> List[ForecastRecord]:
    home, away = g["home_team"], g["away_team"]
    kickoff = g.get("event_start_ts") or g.get("game_date") or ""
    if len(str(kickoff)) == 10:          # date only -> NFL games are afternoon/evening ET
        kickoff = f"{kickoff}T17:00:00+00:00"

    ev = event_id(g.get("game_id"), g.get("season"), g.get("week"), home, away)
    depth = data_depth_for(g.get("games_played_min"))
    cfg = config_hash(blend_elo)

    # Tier on the BLENDED probability — that's what the newsletter publishes,
    # so that's what has to be graded.
    p_home = float(g["p_blended"])
    favorite = home if p_home >= 0.5 else away
    fav_prob = max(p_home, 1.0 - p_home)

    common = dict(
        schema_version=SCHEMA_VERSION, event_id=ev,
        data_depth=depth, asof_ts=asof_ts, created_ts=now_iso(),
        committed_ts=committed_ts, event_start_ts=str(kickoff),
        config_hash=cfg, code_commit=code_commit(), provenance=provenance,
    )

    out: List[ForecastRecord] = []

    # 1) moneyline — probabilistic
    out.append(ForecastRecord(
        forecast_id="", record_family="probabilistic", market_type="moneyline",
        subject_id=team_id(favorite),
        opponent_id=team_id(away if favorite == home else home),
        side="home" if favorite == home else "away",
        line=None,
        proposition_text=moneyline_text(
            favorite, away if favorite == home else home, str(kickoff)),
        prob=fav_prob,
        tier=compute_tier(fav_prob, depth),
        model_version=wp_version,
        **common))

    # 2) margin — distributional
    mu = g.get("predicted_margin")
    if mu is not None:
        out.append(ForecastRecord(
            forecast_id="", record_family="distributional", market_type="spread",
            subject_id=team_id(home), opponent_id=team_id(away),
            side="home", line=g.get("spread_line"),
            proposition_text=spread_text(home, away, float(mu), str(kickoff)),
            mu=float(mu), sigma=sigma,
            tier="Pass",              # tiers are a probabilistic-family concept
            market_source="nflverse" if g.get("spread_line") is not None else None,
            market_line=g.get("spread_line"),
            model_version=margin_version,
            **common))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Emit forecast-ledger records.")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="content/football/<date>/ folder to read (default: today)")
    ap.add_argument("--provenance", default="live",
                    choices=["live", "backfill", "replay"],
                    help="'live' rows are the only ones a public calibration "
                         "surface may show")
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp of the latest data the model saw "
                         "(default: read from the slate)")
    ap.add_argument("--margin-version", default="v5",
                    help="Margin model whose RMSE supplies sigma (default: v5).")
    ap.add_argument("--supersedes", default=None, metavar="BATCH_DATE",
                    help="Re-issue this slate, marking each new row as "
                         "superseding the matching row in an earlier batch "
                         "(e.g. --supersedes 2026-09-07). Both rows stay in the "
                         "ledger forever; nothing is edited or deleted. Only "
                         "rows whose event has NOT started can be superseded.")
    ap.add_argument("--supersede-reason", default=None,
                    help="Required with --supersedes. Recorded on every "
                         "superseding row so an auditor can see why.")
    args = ap.parse_args(argv)

    preds, out_dir = _load_predictions(args.date)
    if not preds:
        raise SystemExit("predictions.json is empty")

    # ---- supersession index ------------------------------------------------
    # Re-issuing is legitimate ONLY before the event starts and ONLY when fully
    # recorded: the original row is never touched, the new row points back at it
    # via `supersedes`, and the reason is stored alongside. That distinction is
    # the whole ballgame — correcting a config error before kickoff is
    # bookkeeping; swapping a forecast after a result is known is fraud, and the
    # timestamps are what let a stranger tell the two apart.
    supersede_index = {}
    if args.supersedes:
        if not args.supersede_reason:
            raise SystemExit(
                "--supersedes requires --supersede-reason. The reason is "
                "written onto every superseding row; an unexplained re-issue is "
                "indistinguishable from cherry-picking.")
        old_path = (settings.project_root / "content" / "football"
                    / args.supersedes / "ledger" / "forecasts.jsonl")
        if not old_path.exists():
            raise SystemExit(f"No batch to supersede at {old_path}")
        for line in old_path.read_text().splitlines():
            if line.strip():
                old = json.loads(line)
                key = (old["event_id"], old["market_type"])
                supersede_index[key] = old
        print(f"Superseding batch {args.supersedes}: "
              f"{len(supersede_index)} prior row(s) available to replace.\n")

    asof = args.asof or preds[0].get("asof_ts") or now_iso()
    committed = now_iso()
    sigma = _margin_sigma(args.margin_version)
    wp_version = preds[0].get("wp_version") or settings.active_model_version
    blend_elo = preds[0].get("blend_elo_weight")

    records: List[ForecastRecord] = []
    refused: List[tuple] = []
    prev_hash = None

    for g in preds:
        try:
            for rec in records_for_game(
                    g, asof_ts=asof, committed_ts=committed,
                    provenance=args.provenance, sigma=sigma,
                    wp_version=wp_version, margin_version=args.margin_version,
                    blend_elo=blend_elo):
                old = supersede_index.get((rec.event_id, rec.market_type))
                if old is not None:
                    rec.supersedes = old["forecast_id"]
                    rec.supersede_reason = args.supersede_reason
                rec.finalize(prev_hash)
                prev_hash = rec.row_hash
                records.append(rec)
        except LedgerError as e:
            refused.append((f"{g.get('away_team')} @ {g.get('home_team')}",
                            str(g.get("event_start_ts") or g.get("game_date", "")),
                            str(e)))

    ledger_dir = out_dir / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    jsonl = ledger_dir / "forecasts.jsonl"

    # ---- never overwrite a batch ------------------------------------------
    # A batch file is committed and pushed the moment it's written; its
    # manifest chain_head is the attestation. Rewriting it — even with the same
    # forecasts — replaces the committed_ts and chain head with new ones and
    # makes the git history the only evidence of what was first published.
    status, existing = existing_batch_status(jsonl, [r.forecast_id for r in records])
    if status != "absent":
        new_ids = sorted(r.forecast_id for r in records)
        old_ids = sorted(r["forecast_id"] for r in existing)
        if status == "identical":
            print("=" * 66)
            print(f"  FORECAST LEDGER — nfl-{args.date}")
            print("=" * 66)
            print(f"  Already emitted: all {len(new_ids)} forecasts in this batch "
                  f"are identical to the ones")
            print(f"  committed at {existing[0].get('committed_ts')}. Nothing "
                  f"written — the original batch stands.")
            return 0
        added = set(new_ids) - set(old_ids)
        raise SystemExit(
            f"\n  ✗ REFUSED — {jsonl} already exists and this run would change "
            f"it ({len(added)} forecast(s) differ).\n"
            "    A committed batch is never rewritten. If the forecasts need to "
            "change before kickoff,\n"
            "    re-issue them explicitly under a new date folder with\n"
            f"      --supersedes {args.date} --supersede-reason \"...\"\n"
            "    so the original stays in the record beside its replacement.")

    jsonl.write_text("\n".join(r.to_json() for r in records) + "\n")

    manifest = build_manifest(records, batch_label=f"nfl-{args.date}",
                              blend_elo=blend_elo)
    # Which state produced this batch. The validated weights are fixed, but
    # they serve a different state every week, so weights alone no longer
    # identify a forecast — the refresh timestamp and weight hashes together do.
    manifest["model_state"] = {
        "refreshed_at": preds[0].get("state_refreshed_at"),
        "last_game_absorbed": preds[0].get("state_last_game"),
        "weights_sha256": preds[0].get("weights_sha256") or {},
    }
    (ledger_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 66)
    print(f"  FORECAST LEDGER — nfl-{args.date}")
    print("=" * 66)
    print(f"  records written : {len(records)}  ({manifest['family_counts']})")
    print(f"  provenance      : {args.provenance}")
    print(f"  tiers           : {manifest['tier_counts']}")
    print(f"  data depth      : {manifest['depth_counts']}")
    n_super = sum(1 for r in records if r.supersedes)
    if n_super:
        print(f"  superseding     : {n_super} row(s) from batch {args.supersedes}")
        print(f"    reason        : {args.supersede_reason}")
    print(f"  asof_ts         : {asof}")
    ms = manifest["model_state"]
    print(f"  model state     : "
          + (f"refreshed {ms['refreshed_at']} (through {ms['last_game_absorbed']})"
             if ms["refreshed_at"] else
             f"training state, never refreshed (through {ms['last_game_absorbed']})"))
    print(f"  committed_ts    : {committed}")
    print(f"  margin sigma    : {sigma:.2f} pts ({args.margin_version} holdout RMSE)")
    if manifest["chain_head"]:
        print(f"  chain head      : {manifest['chain_head'][:16]}…")
    else:
        print("  chain head      : (none)")
    print(f"  config_hash     : {manifest['config_hash']}  "
          f"code_commit: {manifest['code_commit']}")
    print(f"\n  → {jsonl}")
    print(f"  → {ledger_dir / 'manifest.json'}")

    if refused:
        started = [r for r in refused if "already kicked off" in r[2]]
        other = [r for r in refused if r not in started]

        if started:
            print(f"\n  ⓘ {len(started)} game(s) skipped — already kicked off "
                  f"before this run:")
            for name, when, _ in started[:8]:
                print(f"      {name}  (kickoff {when})")
            if len(started) > 8:
                print(f"      … and {len(started) - 8} more")
            print("    This is the pre-registration invariant working, not a "
                  "failure — a forecast")
            print("    published after kickoff isn't a forecast. In the NFL "
                  "this will usually be")
            print("    Thursday Night Football on a Friday/Saturday run. "
                  "Generate the slate Tuesday")
            print("    or Wednesday to capture the full week.")

        if other:
            print(f"\n  ⚠ {len(other)} game(s) REFUSED for other reasons:")
            for name, when, why in other[:6]:
                print(f"      {name} ({when}): {why}")

    print("\n  Next: commit the manifest so the timestamp is third-party "
          "attested —")
    try:
        rel = ledger_dir.relative_to(settings.project_root)
    except ValueError:
        rel = ledger_dir
    print(f"    git add {rel} && \\")
    print(f"    git commit -m 'ledger: nfl {args.date}' && git push")
    print("\n  For entertainment & educational purposes only — not betting advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
