"""Flexible chunker v2 — produces ChunkData objects for any dataset archetype.

Changes from v1:
- Works with any ColumnMapping (not hardcoded ticket columns)
- Adds context_prefix to each chunk for Contextual Retrieval
- Supports 1-chunk (kb_article), 2-chunk (ticket/qa), and N-chunk (long body) modes
- Embedding is computed over context_prefix + text (handled in embedder_v2)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from etl.schema_detector import ColumnMapping

# ── ChunkData ──────────────────────────────────────────────────────────────────

@dataclass
class ChunkData:
    id:             str
    text:           str
    embed_text:     str               # context_prefix + "\n\n" + text (used for embedding)
    chunk_type:     str               # problem | resolution | question | answer | body | section
    chunk_index:    int
    ticket_id:      str               # parent document id
    context_prefix: str = ""
    token_count:    int = 0
    lineage:        str = ""          # topological lineage string (document pipeline only)
    embedding:      list[float] | None = field(default=None, repr=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

_MAX_SECTION_CHARS = 1500   # split body chunks longer than this on paragraph boundaries


def _split_long_text(text: str, max_chars: int = _MAX_SECTION_CHARS) -> list[str]:
    """Split text at paragraph boundaries, keeping chunks under max_chars."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _build_prefix(row: dict[str, Any], mapping: ColumnMapping, chunk_type: str, doc_id: str) -> str:
    """Build a short context string that is prepended to the chunk text for embedding."""
    parts = [f"[{mapping.archetype.upper()} {doc_id}]"]

    # Add any metadata values we know about
    for mf in mapping.metadata_fields[:4]:   # cap at 4 to keep prefix short
        val = str(row.get(mf, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(f"{mf}:{val}")

    parts.append(f"type:{chunk_type}")
    return "  ".join(parts)


def _build_prefix_document(row: dict[str, Any], mapping: ColumnMapping,
                           section, doc_id: str) -> str:
    """Build context prefix for a document chunk with heading path."""
    parts = [f"[DOCUMENT {doc_id}]"]

    # Heading path (hierarchical breadcrumb)
    if section.heading_path:
        parts.append(f"path:{section.heading_path}")
    elif section.heading and section.heading != "main":
        parts.append(f"section:{section.heading}")

    # Paperless metadata
    for key in ["correspondent", "document_type"]:
        val = str(row.get(key, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(f"{key}:{val}")
    for tag in row.get("tags", [])[:3]:
        parts.append(f"tag:{tag}")

    parts.append("type:section")
    return "  ".join(parts)


def _safe_id(val: Any) -> str:
    return str(int(float(str(val)))) if str(val).replace(".", "").isdigit() else str(val).strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def make_chunks(row: dict[str, Any], mapping: ColumnMapping) -> list[ChunkData]:
    """Return one or more ChunkData objects for a single dataset row."""
    doc_id = _safe_id(row.get(mapping.id_field, "")) if mapping.id_field else str(hash(str(row)))
    chunks: list[ChunkData] = []
    idx = 0

    def _add(text: str, chunk_type: str) -> None:
        nonlocal idx
        text = str(text).strip()
        if not text or text.lower() in ("nan", "none"):
            return
        for part in _split_long_text(text):
            if not part.strip():
                continue
            prefix = _build_prefix(row, mapping, chunk_type, doc_id)
            chunks.append(ChunkData(
                id           = f"{doc_id}_{idx}",
                text         = part,
                embed_text   = prefix + "\n\n" + part,
                chunk_type   = chunk_type,
                chunk_index  = idx,
                ticket_id    = doc_id,
                context_prefix = prefix,
                token_count  = len(part.split()),
            ))
            idx += 1

    if mapping.archetype == "ticket":
        _add(row.get(mapping.question_field, ""), "problem")
        _add(row.get(mapping.answer_field,   ""), "resolution")

    elif mapping.archetype == "qa":
        q_field = mapping.question_field or mapping.title_field
        a_field = mapping.answer_field
        _add(row.get(q_field, ""), "question")
        _add(row.get(a_field, ""), "answer")

    elif mapping.archetype == "kb_article":
        if mapping.title_field:
            _add(row.get(mapping.title_field, ""), "title")
        if mapping.answer_field:
            _add(row.get(mapping.answer_field, ""), "body")
        elif mapping.question_field:
            _add(row.get(mapping.question_field, ""), "body")

    elif mapping.archetype == "document":
        # Sections and chunks are pre-built by pdf_parser
        from etl.doc_enrichment import build_chunk_lineage  # noqa: PLC0415
        for section in row.get("_parsed_sections", []):
            section_text = "\n\n".join(
                b.text for b in section.blocks
                if b.block_type in ("paragraph", "list_item")
            )
            if not section_text.strip():
                continue
            for part in _split_long_text(section_text):
                if not part.strip():
                    continue
                prefix = _build_prefix_document(row, mapping, section, doc_id)
                lineage = build_chunk_lineage(doc_id, section.heading, idx)
                chunks.append(ChunkData(
                    id             = f"{doc_id}_{idx}",
                    text           = part,
                    embed_text     = prefix + "\n\n" + part,
                    chunk_type     = "section",
                    chunk_index    = idx,
                    ticket_id      = doc_id,
                    context_prefix = prefix,
                    token_count    = len(part.split()),
                    lineage        = lineage,
                ))
                idx += 1

    # Any additional body fields
    for bf in mapping.body_fields:
        _add(row.get(bf, ""), "body")

    return chunks
