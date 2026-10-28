"""Smoke test for the deterministic metadata-filter normaliser in the planner.

Runs the full planner(state) node — LLM call included — and asserts the
FINAL `metadata_filters` dict (post-normalisation) exactly matches the
expected output. Keys must match exactly, values are compared as sets
(order-independent), and NO extra keys are tolerated.

This is the spec for "Metadata-aware Contextual GraphRAG — Planner Fix".

Usage: python scripts/check_planner_metadata_filters.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import planner

PASS = "✅"
FAIL = "❌"


# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (query, expected_metadata_filters)
# Expected values are compared as sets — order does not matter.
TEST_CASES: list[tuple[str, dict[str, list[str]]]] = [
    (
        "Show me resolved billing tickets",
        # "resolved" expands to both Resolved AND Closed (CSV separates them).
        # "tickets" triggers doc_type. "billing" is topical → no category filter.
        {"status": ["Resolved", "Closed"], "doc_type": ["ticket"], "product": ["Billing System"]}
    ),
    (
        "Find high priority performance issues",
        # "performance issues" is topical → no category filter.
        # No explicit doc_type word → doc_type unset.
        {"priority": ["High"]},
    ),
    (
        "Search tickets for Cloud Storage with status closed",
        # Realistic product name from the CSV's product column.
        {"doc_type": ["ticket"], "status": ["Closed"], "product": ["Cloud Storage"]},
    ),
    (
        "How do I reset my password?",
        # Generic question — no filters at all. "password" is topical only.
        {},
    ),
    (
        "Find KB articles about account recovery",
        # "KB articles" maps to doc_type=document (no kb_article in our taxonomy).
        # "account recovery" is topical → no category filter.
        {"doc_type": ["document"]},
    ),
    (
        "Show me German documents",
        # "German" explicitly named → language=German (canonical CSV casing).
        # "documents" triggers doc_type=document.
        {"doc_type": ["document"], "language": ["German"]},
    ),
]


def _compare(actual: dict, expected: dict) -> tuple[bool, str]:
    """Strict comparison: same keys, same values (set-equal). No extras allowed."""
    if set(actual.keys()) != set(expected.keys()):
        extra   = set(actual.keys()) - set(expected.keys())
        missing = set(expected.keys()) - set(actual.keys())
        parts = []
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        return False, "; ".join(parts)

    for key, exp_values in expected.items():
        act_values = actual.get(key, [])
        if not isinstance(act_values, list):
            return False, f"key {key!r}: expected list, got {type(act_values).__name__}"
        if set(act_values) != set(exp_values):
            return False, (
                f"key {key!r}: expected {sorted(exp_values)}, "
                f"got {sorted(act_values)}"
            )
    return True, ""


def main() -> int:
    print("=" * 70)
    print("  PLANNER METADATA FILTER — STRICT CONTRACT TEST")
    print("=" * 70)
    print()

    all_ok = True
    for i, (query, expected) in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{len(TEST_CASES)}] Query: {query!r}")
        print(f"        Expected: {expected}")

        state = {"original_query": query, "sub_queries": [query]}
        try:
            result = planner(state)
        except Exception as exc:
            print(f"        {FAIL} planner() raised: {exc!r}")
            all_ok = False
            print()
            continue

        actual = result.get("metadata_filters", {})
        print(f"        Actual:   {actual}")

        ok, reason = _compare(actual, expected)
        if ok:
            print(f"        {PASS} PASS")
        else:
            print(f"        {FAIL} FAIL — {reason}")
            all_ok = False
        print()

    print("=" * 70)
    if all_ok:
        print(f"  {PASS} ALL {len(TEST_CASES)} CHECKS PASSED")
        return 0
    print(f"  {FAIL} SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())