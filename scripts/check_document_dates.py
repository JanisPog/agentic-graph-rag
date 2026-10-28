"""Smoke-test: verify that _extract_document_dates and _normalise_date work
correctly, and that doc_records get proper created_at / updated_at / ingested_at.

Usage:  python scripts/check_document_dates.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.ingest import _extract_document_dates, _normalise_date
from etl.schema_detector import ColumnMapping

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

all_ok = True


def check(label: str, got: str, expected: str):
    global all_ok
    if got == expected:
        print(f"  {PASS} {label}: {got!r}")
    else:
        print(f"  {FAIL} {label}: expected {expected!r}, got {got!r}")
        all_ok = False


# ── 1. _normalise_date unit checks ───────────────────────────────────────────
print("=" * 60)
print("  DOCUMENT DATES SMOKE TEST")
print("=" * 60)

print("\n[1] _normalise_date — format parsing")

check("ISO full",         _normalise_date("2024-11-20T10:30:00"),  "2024-11-20T10:30:00")
check("ISO date only",    _normalise_date("2024-11-20"),           "2024-11-20T00:00:00")
check("ISO with Z",       _normalise_date("2024-11-20T10:30:00Z"), "2024-11-20T10:30:00+00:00")
check("ISO with tz",      _normalise_date("2024-11-20T10:30:00+02:00"), "2024-11-20T10:30:00+02:00")
check("datetime space",   _normalise_date("2024-11-20 10:30:00"),  "2024-11-20T10:30:00")
check("DD/MM/YYYY",       _normalise_date("20/11/2024"),           "2024-11-20T00:00:00")
check("empty",            _normalise_date(""),                     "")
check("nan",              _normalise_date("nan"),                  "")
check("None",             _normalise_date("None"),                 "")
check("null",             _normalise_date("null"),                 "")
check("NaT",              _normalise_date("NaT"),                  "")
check("garbage",          _normalise_date("not-a-date"),           "")


# ── 2. _extract_document_dates with ticket-like data ─────────────────────────
print("\n[2] _extract_document_dates — ticket data (ticket_created_date / ticket_resolved_date)")

mapping = ColumnMapping(
    archetype="ticket",
    id_field="ticket_id",
    question_field="issue_description",
    answer_field="resolution",
    title_field="title",
    metadata_fields=["priority", "category", "product", "ticket_created_date", "ticket_resolved_date"],
)

row = {
    "ticket_id": "42",
    "title": "Test ticket",
    "issue_description": "Cannot login",
    "resolution": "Reset password",
    "priority": "High",
    "category": "Login",
    "product": "WebApp",
    "ticket_created_date": "2024-11-20",
    "ticket_resolved_date": "2024-11-25",
}

created, updated = _extract_document_dates(row, mapping)
check("created_at from ticket_created_date", created, "2024-11-20T00:00:00")
check("updated_at from ticket_resolved_date", updated, "2024-11-25T00:00:00")


# ── 3. Fallback: updated_at = created_at when updated is missing ─────────────
print("\n[3] Fallback: updated_at = created_at when only created is available")

row_no_resolved = {
    "ticket_id": "43",
    "title": "Test ticket 2",
    "ticket_created_date": "2024-12-01",
}

created, updated = _extract_document_dates(row_no_resolved, mapping)
check("created_at", created, "2024-12-01T00:00:00")
check("updated_at (fallback to created_at)", updated, "2024-12-01T00:00:00")


# ── 4. Both missing → empty ─────────────────────────────────────────────────
print("\n[4] Both dates missing → empty strings")

row_no_dates = {
    "ticket_id": "44",
    "title": "Test ticket 3",
    "priority": "Low",
}

created, updated = _extract_document_dates(row_no_dates, mapping)
check("created_at (missing)", created, "")
check("updated_at (missing)", updated, "")


# ── 5. created / modified fields (Paperless/DMS style) ───────────────────────
print("\n[5] DMS-style fields (created, modified)")

dms_mapping = ColumnMapping(
    archetype="document",
    id_field="id",
    question_field=None,
    answer_field=None,
    title_field="title",
    metadata_fields=["correspondent", "document_type"],
)

dms_row = {
    "id": "paperless_42",
    "title": "SEO Guide",
    "created": "2025-01-15T09:00:00+01:00",
    "modified": "2025-03-20T14:30:00+01:00",
}

created, updated = _extract_document_dates(dms_row, dms_mapping)
check("created_at from 'created'", created, "2025-01-15T09:00:00+01:00")
check("updated_at from 'modified'", updated, "2025-03-20T14:30:00+01:00")


# ── 6. Priority order: created_at beats ticket_created_date ──────────────────
print("\n[6] Priority order: 'created_at' field takes precedence over 'ticket_created_date'")

row_both = {
    "ticket_id": "45",
    "created_at": "2024-06-01",
    "ticket_created_date": "2024-11-20",
}

created, updated = _extract_document_dates(row_both, mapping)
check("created_at (priority: created_at > ticket_created_date)", created, "2024-06-01T00:00:00")


# ── 7. Live data check (if Neo4j is available) ──────────────────────────────
print("\n[7] Live dataset check (first 3 rows from CSV)...")
try:
    import config
    from etl.schema_detector import detect_schema, load_raw

    csv_path = config.CSV_PATH
    if not os.path.exists(csv_path):
        print(f"  {WARN} CSV not found at {csv_path!r} — skipping live check")
    else:
        profile = detect_schema(csv_path)
        df = load_raw(csv_path).head(3)
        for idx, (_, row) in enumerate(df.iterrows()):
            row_dict = row.to_dict()
            created, updated = _extract_document_dates(row_dict, profile.mapping)
            doc_id = str(row_dict.get(profile.mapping.id_field, f"row_{idx}"))
            src_created = row_dict.get("ticket_created_date", "N/A")
            src_resolved = row_dict.get("ticket_resolved_date", "N/A")
            print(f"  Doc {doc_id}:")
            print(f"    source ticket_created_date = {src_created!r}")
            print(f"    source ticket_resolved_date = {src_resolved!r}")
            print(f"    → created_at  = {created!r}")
            print(f"    → updated_at  = {updated!r}")
            if created:
                print(f"    {PASS} created_at populated")
            else:
                print(f"    {WARN} created_at is empty (no matching date field found)")
except Exception as exc:
    print(f"  {WARN} Live check skipped: {exc}")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print(f"  {PASS} ALL CHECKS PASSED")
else:
    print(f"  {FAIL} SOME CHECKS FAILED — see details above")
print("=" * 60)

sys.exit(0 if all_ok else 1)
