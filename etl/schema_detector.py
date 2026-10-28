"""Schema detector — uses the LLM to map arbitrary dataset columns to canonical roles.

Supports CSV, JSON, JSONL, and plain-text Markdown files. Returns a ColumnMapping
dataclass that all downstream ETL steps consume instead of hardcoded column names.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm import chat_json

log = logging.getLogger(__name__)

# ── Canonical archetypes ───────────────────────────────────────────────────────

ARCHETYPES = ("ticket", "qa", "kb_article")

# ── ColumnMapping dataclass ────────────────────────────────────────────────────

@dataclass
class ColumnMapping:
    archetype:        str                  # ticket | qa | kb_article
    id_field:         str | None           # unique row identifier
    question_field:   str | None           # problem / question text
    answer_field:     str | None           # resolution / answer text
    title_field:      str | None           # short heading / subject
    body_fields:      list[str] = field(default_factory=list)  # extra text columns
    doc_fields:       list[str] = field(default_factory=list)  # URLs / attachment refs
    metadata_fields:  list[str] = field(default_factory=list)  # structured attributes
    status_field:     str | None = None    # lifecycle field used for filtering
    status_keep:      list[str] = field(default_factory=list)  # values to retain
    ref_field:        str | None = None    # column containing foreign-key IDs to a paired dataset


@dataclass
class DatasetProfile:
    file_path:   str
    file_type:   str           # csv | json | jsonl | markdown
    row_count:   int
    columns:     list[str]
    mapping:     ColumnMapping


# ── Loaders ────────────────────────────────────────────────────────────────────

def _detect_file_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {"csv": "csv", ".csv": "csv",
            ".json": "json", ".jsonl": "jsonl",
            ".md": "markdown", ".txt": "markdown"}.get(suffix, "csv")


def load_raw(path: str) -> pd.DataFrame:
    """Load any supported file into a DataFrame."""
    ftype = _detect_file_type(path)
    if ftype == "csv":
        return pd.read_csv(path, dtype=str)
    elif ftype == "json":
        return pd.read_json(path, dtype=str)
    elif ftype == "jsonl":
        return pd.read_json(path, lines=True, dtype=str)
    elif ftype == "markdown":
        # Each H2 heading becomes one row with title + body columns
        text = Path(path).read_text(encoding="utf-8")
        sections = re.split(r"\n##\s+", text)
        rows = []
        for sec in sections[1:]:
            lines = sec.strip().split("\n", 1)
            rows.append({"title": lines[0].strip(), "body": lines[1].strip() if len(lines) > 1 else ""})
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file type: {path}")


# ── LLM-based schema detection ─────────────────────────────────────────────────

_DETECT_PROMPT = """\
You are a data schema analyst. Given column names and a few sample rows from a dataset,
identify which column serves each canonical role. Return ONLY a JSON object.

Rules:
- Use null if a role has no matching column.
- metadata_fields: list all columns that are structured attributes (category, priority, status, product, tags, version, channel, department, article_type, etc.)
- body_fields: additional free-text columns beyond question/answer/title.
- doc_fields: columns containing URLs, file paths, or attachment references.
- archetype must be one of: "ticket", "qa", "kb_article"
- status_keep: if a status_field exists, list values that represent "resolved/useful" records (e.g. ["Resolved","Closed","answered","published"]). Empty list means keep all.
- ref_field: if there is a column that contains IDs or lists of IDs that reference rows in another dataset (e.g. "article_ids", "doc_ids", "kb_refs"), set this field. Otherwise null.{paired_hint}

JSON schema:
{{
  "archetype":       "ticket" | "qa" | "kb_article",
  "id_field":        "<col>" | null,
  "question_field":  "<col>" | null,
  "answer_field":    "<col>" | null,
  "title_field":     "<col>" | null,
  "body_fields":     ["<col>", ...],
  "doc_fields":      ["<col>", ...],
  "metadata_fields": ["<col>", ...],
  "status_field":    "<col>" | null,
  "status_keep":     ["<val>", ...],
  "ref_field":       "<col>" | null
}}

Dataset filename: {filename}
Columns: {columns}
Sample rows (up to 3):
{sample}
"""


def _llm_detect(df: pd.DataFrame, filename: str, paired_hint: str = "") -> dict:
    columns = list(df.columns)
    sample_rows = df.head(3).fillna("").to_dict(orient="records")
    sample_str = json.dumps(sample_rows, indent=2, default=str)

    prompt = _DETECT_PROMPT.format(
        filename=filename,
        columns=json.dumps(columns),
        sample=sample_str,
        paired_hint=paired_hint,
    )
    try:
        result = chat_json(prompt)
        if not isinstance(result, dict):
            raise ValueError("Expected dict")
        return result
    except Exception as exc:
        log.warning("LLM schema detection failed (%s); falling back to heuristics", exc)
        return {}


# ── Heuristic fallback ─────────────────────────────────────────────────────────

_QUESTION_HINTS  = {"question", "issue", "issue_description", "problem", "query", "subject", "title", "q"}
_ANSWER_HINTS    = {"answer", "resolution", "resolution_notes", "response", "solution", "body", "a", "reply", "contents"}
_ID_HINTS        = {"id", "ticket_id", "question_id", "post_id", "doc_id", "uid", "article_id"}
_STATUS_HINTS    = {"status", "state", "resolution_status"}
_META_HINTS      = {"priority", "category", "product", "channel", "department", "tags",
                    "severity", "type", "version", "platform", "environment", "article_type"}
_REF_HINTS       = {"article_ids", "article_id", "doc_ids", "doc_id_refs", "kb_ids", "kb_refs",
                    "source_ids", "reference_ids", "linked_articles"}


def _heuristic_detect(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    get = lambda hints: next((cols[h] for h in hints if h in cols), None)

    question = get(_QUESTION_HINTS)
    answer   = get(_ANSWER_HINTS)
    id_col   = get(_ID_HINTS)
    status   = get(_STATUS_HINTS)
    ref      = get(_REF_HINTS)
    meta     = [cols[c] for c in cols if c in _META_HINTS]

    archetype = "ticket"
    if not question and not answer:
        # Single-field CSV (title+body) → kb_article *chunking strategy*.
        # The query-time doc_type is normalised to "document" at write time
        # (see _doc_type_for in etl/ingest.py); archetype stays kb_article so
        # the correct 1-chunk chunker runs. Section-based markdown/PDF docs use
        # the "document" archetype via paperless_ingest, not this heuristic.
        archetype = "kb_article"
    elif "question" in cols or "answer" in cols:
        archetype = "qa"

    return {
        "archetype":       archetype,
        "id_field":        id_col,
        "question_field":  question,
        "answer_field":    answer,
        "title_field":     None,
        "body_fields":     [],
        "doc_fields":      [],
        "metadata_fields": meta,
        "status_field":    status,
        "status_keep":     ["Resolved", "Closed"] if status else [],
        "ref_field":       ref,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_schema(
    path: str,
    archetype_override: str | None = None,
    paired_hint: str = "",
) -> DatasetProfile:
    """Detect the schema of a dataset file and return a DatasetProfile."""
    df = load_raw(path)
    filename = Path(path).name
    ftype    = _detect_file_type(path)

    log.info("Detecting schema for '%s'  (%d rows, %d cols)", filename, len(df), len(df.columns))

    llm_result  = _llm_detect(df, filename, paired_hint=paired_hint)
    heuristic   = _heuristic_detect(df)

    # Merge: LLM result takes precedence where it returned non-null values
    def pick(key: str, default: Any = None) -> Any:
        v = llm_result.get(key)
        if v is not None and v != [] and v != "":
            return v
        return heuristic.get(key, default)

    archetype = archetype_override or pick("archetype", "ticket")
    if archetype not in ARCHETYPES:
        log.warning("Unknown archetype %r — defaulting to 'ticket'", archetype)
        archetype = "ticket"

    mapping = ColumnMapping(
        archetype       = archetype,
        id_field        = pick("id_field"),
        question_field  = pick("question_field"),
        answer_field    = pick("answer_field"),
        title_field     = pick("title_field"),
        body_fields     = pick("body_fields", []) or [],
        doc_fields      = pick("doc_fields",  []) or [],
        metadata_fields = pick("metadata_fields", []) or [],
        status_field    = pick("status_field"),
        status_keep     = pick("status_keep", []) or [],
        ref_field       = pick("ref_field"),
    )

    # Validate: at least one text field must be present
    text_cols = [f for f in [mapping.question_field, mapping.answer_field,
                              mapping.title_field] + mapping.body_fields if f]
    if not text_cols:
        raise ValueError(
            f"Schema detection found no usable text columns in '{filename}'. "
            "Use --archetype to specify the dataset type manually."
        )

    log.info(
        "  archetype=%s  question=%s  answer=%s  metadata=%s  ref=%s",
        mapping.archetype, mapping.question_field, mapping.answer_field,
        mapping.metadata_fields, mapping.ref_field,
    )

    return DatasetProfile(
        file_path  = path,
        file_type  = ftype,
        row_count  = len(df),
        columns    = list(df.columns),
        mapping    = mapping,
    )


def detect_paired_schemas(
    primary_path: str,
    secondary_path: str,
    primary_archetype: str | None = None,
    secondary_archetype: str | None = None,
) -> tuple[DatasetProfile, DatasetProfile]:
    """Detect schemas for two related datasets, making the LLM aware of the pairing.

    The primary dataset is expected to contain a ref_field pointing to IDs in the
    secondary dataset (e.g. QA questions referencing KB article IDs).
    Both profiles are returned; primary.mapping.ref_field names the join column.
    """
    secondary_filename = Path(secondary_path).name
    primary_filename   = Path(primary_path).name

    # Hint injected into the primary detection prompt so the LLM knows to look
    # for a column that references the secondary dataset's IDs.
    primary_hint = (
        f"\n- This dataset is paired with '{secondary_filename}'. "
        f"Look for a column that contains IDs or lists of IDs matching rows in that file "
        f"and set it as ref_field."
    )

    log.info("Paired schema detection: primary='%s'  secondary='%s'",
             primary_filename, secondary_filename)

    primary_profile   = detect_schema(primary_path,   primary_archetype,   paired_hint=primary_hint)
    secondary_profile = detect_schema(secondary_path, secondary_archetype)

    if primary_profile.mapping.ref_field:
        log.info("  Join key: %s.%s → %s.%s",
                 primary_filename, primary_profile.mapping.ref_field,
                 secondary_filename, secondary_profile.mapping.id_field)
    else:
        log.warning("  No ref_field detected in primary dataset — cross-document edges will be skipped.")

    return primary_profile, secondary_profile
