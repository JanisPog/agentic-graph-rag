"""Smoke-test: verify that the Reranker only applies recency boosts
for genuine domain date fields, and ignores ingested_at.

Usage:  python scripts/check_reranker_recency.py
"""
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import ScoredChunk
from agents.reranker import _metadata_rescore

PASS = "✅"
FAIL = "❌"

all_ok = True


def check(label: str, old_score: float, new_score: float, expect_boost: bool, expect_penalty: bool):
    global all_ok
    delta = new_score - old_score
    if expect_boost and delta > 0.005:
        print(f"  {PASS} {label} (boosted: Δ{delta:+.4f})")
    elif expect_penalty and delta < -0.005:
        print(f"  {PASS} {label} (penalized: Δ{delta:+.4f})")
    elif not expect_boost and not expect_penalty and abs(delta) < 0.005:
        print(f"  {PASS} {label} (neutral: Δ{delta:+.4f})")
    else:
        print(f"  {FAIL} {label} (unexpected delta: Δ{delta:+.4f})")
        all_ok = False


print("=" * 60)
print("  RERANKER RECENCY SMOKE TEST")
print("=" * 60)

base_score = 0.5
now = datetime.now(timezone.utc)
recent_date = (now - timedelta(days=10)).isoformat()
old_date = (now - timedelta(days=2000)).isoformat()

# ── 1. Ticket with ticket_resolved_date ──────────────────────────────────────
ticket_meta = {
    "status": "open",  # neutral status to isolate recency
    "ticket_resolved_date": recent_date,
}
c1 = ScoredChunk(id="1", text="test", score=base_score, metadata=ticket_meta)
new_score_1 = _metadata_rescore(c1, "query")
check("[Ticket] Recent ticket_resolved_date", base_score, new_score_1, expect_boost=True, expect_penalty=False)


# ── 2. KB article with updated_at (old) ──────────────────────────────────────
kb_meta = {
    "updated_at": old_date,
}
c2 = ScoredChunk(id="2", text="test", score=base_score, metadata=kb_meta)
new_score_2 = _metadata_rescore(c2, "query")
check("[KB] Old updated_at", base_score, new_score_2, expect_boost=False, expect_penalty=True)


# ── 3. KB/Paperless with ONLY ingested_at ────────────────────────────────────
ingested_meta = {
    "ingested_at": recent_date,
}
c3 = ScoredChunk(id="3", text="test", score=base_score, metadata=ingested_meta)
new_score_3 = _metadata_rescore(c3, "query")
check("[Paperless] Only ingested_at present", base_score, new_score_3, expect_boost=False, expect_penalty=False)


# ── 4. Mixed: ingested_at is recent, created_at is old ───────────────────────
mixed_meta = {
    "ingested_at": recent_date,
    "created_at": old_date,
}
c4 = ScoredChunk(id="4", text="test", score=base_score, metadata=mixed_meta)
new_score_4 = _metadata_rescore(c4, "query")
check("[Mixed] ingested_at=recent, created_at=old", base_score, new_score_4, expect_boost=False, expect_penalty=True)


print("\n" + "=" * 60)
if all_ok:
    print(f"  {PASS} ALL CHECKS PASSED")
else:
    print(f"  {FAIL} SOME CHECKS FAILED")
print("=" * 60)

sys.exit(0 if all_ok else 1)
