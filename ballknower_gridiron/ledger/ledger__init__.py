"""
ballknower_gridiron.ledger — forecast-ledger emitter and grader.

    emit_records.py  record model, invariants, tiering, hash chain
    emit.py          slate -> forecasts.jsonl + manifest.json
    resolve.py       forecasts + results -> resolutions.jsonl (append-only)
    score.py         the join -> calibration report (computed, never stored)
    selftest.py      invariant suite; run before trusting any of the above
"""
