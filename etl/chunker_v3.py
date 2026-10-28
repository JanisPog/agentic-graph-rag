"""Chunker v3 — produces ChunkData objects with Metadata-as-Text prefixes for any dataset archetype."""

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
    embed_text:     str               # context_prefix + "\n\n" + text
    chunk_type:     str               # problem | resolution | question | answer | body | section | ...
    chunk_index:    int
    ticket_id:      str               # parent document id
    context_prefix: str = ""
    token_count:    int = 0
    lineage:        str = ""          # topological lineage string for document pipeline
    embedding:      list[float] | None = field(default=None, repr=False)


# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_SECTION_CHARS = 1500
_MAX_TAGS_IN_PREFIX = 3

_EMPTY_TOKENS = {"", "nan", "none", "null"}


# ── Generic helpers ────────────────────────────────────────────────────────────

def _split_long_text(text: str, max_chars: int = _MAX_SECTION_CHARS) -> list[str]:
    """Split text at paragraph boundaries, keeping chunks under max_chars."""
    text = str(text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append(current.strip())

    return chunks or ([text] if text else [])


def _clean(value: Any) -> str:
    """Return a stripped string, or empty string if value is empty / NaN / None."""
    if value is None:
        return ""

    s = str(value).strip()
    if s.lower() in _EMPTY_TOKENS:
        return ""

    return s


def _safe_id(value: Any) -> str:
    """Normalize IDs like 123.0 to 123, while keeping non-numeric IDs intact."""
    s = _clean(value)
    if not s:
        return ""

    try:
        if re.fullmatch(r"\d+(\.0+)?", s):
            return str(int(float(s)))
    except ValueError:
        pass

    return s


def _display_source_system(source_system: Any) -> str:
    """Human-readable display name for known source systems."""
    s = _clean(source_system)
    if not s:
        return ""

    known = {
        "wixqa": "WixQA",
        "expertqa": "ExpertQA",
        "expert_qa": "ExpertQA",
        "wixqa_expert": "ExpertQA",
        "customer_support": "Customer Support",
        "ticket_system": "Ticket System",
        "paperless": "Paperless",
    }

    return known.get(s.lower(), s)

def _id_header(doc_type: str, doc_id: str | None) -> str:
    """Build compact traceability header like [TICKET 123]."""
    safe_doc_id = _clean(doc_id)
    if not safe_doc_id:
        return ""

    if doc_type == "kb_article":
        doc_type = "document"

    labels = {
        "ticket": "TICKET",
        "document": "DOCUMENT",
        "qa": "QA",
    }

    label = labels.get(doc_type, doc_type.upper())
    return f"[{label} {safe_doc_id}]"

def _tags_value(row: dict[str, Any]) -> str:
    """Format tags as a short comma-separated string."""
    raw_tags = row.get("tags") or []
    if not isinstance(raw_tags, (list, tuple)):
        return ""

    cleaned = [_clean(tag) for tag in raw_tags[:_MAX_TAGS_IN_PREFIX]]
    cleaned = [tag for tag in cleaned if tag]

    return ", ".join(cleaned)


def _add_metadata_sentence(parts: list[str], pairs: list[tuple[str, str]]) -> None:
    """Append one compact metadata sentence like: Product: X; Channel: Y."""
    items = [f"{label}: {value}" for label, value in pairs if value]
    if items:
        parts.append("; ".join(items) + ".")


def _normalise_for_compare(value: str) -> str:
    """Normalize text for loose title/path comparison."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _compact_section_label(section: Any, title: str = "") -> str:
    """Return a compact section label without repeating the document title.

    Example:
        title = "Site Members: Approving a New Member"
        heading_path = "Site Members: Approving a New Member > FAQs"

    Returns:
        "FAQs"
    """
    if not section:
        return ""

    path = _clean(getattr(section, "heading_path", ""))
    heading = _clean(getattr(section, "heading", ""))

    raw = path or heading
    if not raw or raw.lower() == "main":
        return ""

    path_parts = [part.strip() for part in raw.split(">") if part.strip()]
    if not path_parts:
        return ""

    if title and _normalise_for_compare(path_parts[0]) == _normalise_for_compare(title):
        path_parts = path_parts[1:]

    if len(path_parts) > 1 and path_parts[0].lower() == "main":
        path_parts = path_parts[1:]

    if not path_parts:
        return ""

    return " > ".join(path_parts[-2:])


def _chunk_sentence(doc_type: str, chunk_type: str) -> str:
    """Return a short sentence describing what the chunk contains."""
    if chunk_type == "problem":
        return "This chunk contains the customer problem."

    if chunk_type == "resolution":
        return "This chunk contains the final resolution."

    if chunk_type == "question":
        return "This chunk contains the question."

    if chunk_type == "answer":
        return "This chunk contains the answer."

    if chunk_type == "section":
        return "This chunk contains a document section."

    if chunk_type == "section_summary":
        return "This chunk summarizes a section."

    if chunk_type == "document_summary":
        return "This chunk summarizes the full document."

    if chunk_type == "title":
        return "This chunk contains the document title."

    if chunk_type == "body":
        if doc_type == "ticket":
            return "This chunk contains supporting ticket details."
        if doc_type == "qa":
            return "This chunk contains supporting question-answer context."
        return "This chunk contains supporting document content."

    return f"This chunk contains {chunk_type} content."


def _first_available(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty value from a list of possible metadata keys."""
    for key in keys:
        value = _clean(row.get(key))
        if value:
            return value
    return ""


# ── Prefix builder v3 ──────────────────────────────────────────────────────────

def build_context_prefix_v3(
    row: dict[str, Any],
    doc_type: str,
    chunk_type: str,
    doc_id: str | None = None,
    section: Any = None,
) -> str:
    parts: list[str] = []

    if doc_type == "kb_article":
        doc_type = "document"

    header = _id_header(doc_type, doc_id)
    if header:
        parts.append(header)

    source_system = _display_source_system(row.get("source_system"))
    title = _clean(row.get("title"))

    # ── Ticket ────────────────────────────────────────────────────────────────
    if doc_type == "ticket":
        status = _clean(row.get("status"))
        priority = _clean(row.get("priority"))
        category = _clean(row.get("category"))

        description_parts: list[str] = []

        if status:
            description_parts.append(status.capitalize())

        if priority:
            description_parts.append(f"{priority.lower()}-priority")

        if category:
            category_lower = category.lower()
            description_parts.append(category_lower)
        else:
            category_lower = ""

        ticket_label = "ticket" if "support" in category_lower else "support ticket"
        description_parts.append(ticket_label)

        parts.append(" ".join(description_parts) + ".")

        product = _clean(row.get("product"))
        severity = _clean(row.get("severity"))
        channel = _clean(row.get("channel"))

        _add_metadata_sentence(
            parts,
            [
                ("Product", product),
                ("Severity", severity),
                ("Channel", channel),
            ],
        )

    # ── QA ────────────────────────────────────────────────────────────────────
    elif doc_type == "qa":
        if source_system:
            parts.append(f"Question-answer entry from {source_system}.")
        else:
            parts.append("Expert-written question-answer entry.")

        if title:
            parts.append(f'Topic: "{title}".')

        product = _clean(row.get("product"))
        category = _clean(row.get("category"))

        _add_metadata_sentence(
            parts,
            [
                ("Product", product),
                ("Category", category),
            ],
        )

    # ── Document / Paperless / KB ─────────────────────────────────────────────
    else:
        if source_system and title:
            parts.append(f'{source_system} document titled "{title}".')
        elif source_system:
            parts.append(f"{source_system} document.")
        elif title:
            parts.append(f'Document titled "{title}".')
        else:
            parts.append("Document.")

        document_type = _first_available(row, ("document_type", "doc_type_detail"))
        article_type = _clean(row.get("article_type"))
        tags = _tags_value(row)

        _add_metadata_sentence(
            parts,
            [
                ("Type", document_type),
                ("Article type", article_type),
                ("Tags", tags),
            ],
        )

        section_label = _compact_section_label(section, title)
        if section_label:
            parts.append(f"Section: {section_label}.")

    parts.append(_chunk_sentence(doc_type, chunk_type))

    return " ".join(parts)


build_context_prefix_v2 = build_context_prefix_v3


def _build_prefix(
    row: dict[str, Any],
    mapping: ColumnMapping,
    chunk_type: str,
    doc_id: str,
) -> str:
    return build_context_prefix_v3(
        row=row,
        doc_type=mapping.archetype,
        chunk_type=chunk_type,
        doc_id=doc_id,
    )


def _build_prefix_document(
    row: dict[str, Any],
    mapping: ColumnMapping,
    section: Any,
    doc_id: str,
) -> str:
    return build_context_prefix_v3(
        row=row,
        doc_type="document",
        chunk_type="section",
        doc_id=doc_id,
        section=section,
    )


def build_document_summary_prefix(
    row: dict[str, Any],
    doc_id: str | None = None,
) -> str:
    return build_context_prefix_v3(
        row=row,
        doc_type="document",
        chunk_type="document_summary",
        doc_id=doc_id,
    )


def build_section_summary_prefix(
    row: dict[str, Any],
    section: Any,
    doc_id: str | None = None,
) -> str:
    return build_context_prefix_v3(
        row=row,
        doc_type="document",
        chunk_type="section_summary",
        doc_id=doc_id,
        section=section,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def make_chunks(row: dict[str, Any], mapping: ColumnMapping) -> list[ChunkData]:
    """Return one or more ChunkData objects for a single dataset row."""
    if mapping.id_field:
        doc_id = _safe_id(row.get(mapping.id_field, ""))
    else:
        doc_id = str(abs(hash(str(row))))

    chunks: list[ChunkData] = []
    idx = 0

    def _add(text: Any, chunk_type: str) -> None:
        nonlocal idx

        text = str(text).strip()
        if not text or text.lower() in _EMPTY_TOKENS:
            return

        for part in _split_long_text(text):
            part = part.strip()
            if not part:
                continue

            prefix = build_context_prefix_v3(
                row=row,
                doc_type=mapping.archetype,
                chunk_type=chunk_type,
                doc_id=doc_id,
            )

            chunks.append(
                ChunkData(
                    id=f"{doc_id}_{idx}",
                    text=part,
                    embed_text=prefix + "\n\n" + part,
                    chunk_type=chunk_type,
                    chunk_index=idx,
                    ticket_id=doc_id,
                    context_prefix=prefix,
                    token_count=len(part.split()),
                )
            )

            idx += 1

    # ── Ticket ────────────────────────────────────────────────────────────────
    if mapping.archetype == "ticket":
        _add(row.get(mapping.question_field, ""), "problem")
        _add(row.get(mapping.answer_field, ""), "resolution")

    # ── QA ────────────────────────────────────────────────────────────────────
    elif mapping.archetype == "qa":
        q_field = mapping.question_field or mapping.title_field
        a_field = mapping.answer_field

        if q_field:
            _add(row.get(q_field, ""), "question")

        if a_field:
            _add(row.get(a_field, ""), "answer")

    # ── Legacy KB article ─────────────────────────────────────────────────────
    elif mapping.archetype == "kb_article":
        if mapping.title_field:
            _add(row.get(mapping.title_field, ""), "title")

        if mapping.answer_field:
            _add(row.get(mapping.answer_field, ""), "body")
        elif mapping.question_field:
            _add(row.get(mapping.question_field, ""), "body")

    # ── Parsed document ───────────────────────────────────────────────────────
    elif mapping.archetype == "document":
        from etl.doc_enrichment import build_chunk_lineage  # noqa: PLC0415

        for section in row.get("_parsed_sections", []):
            section_text = "\n\n".join(
                block.text
                for block in section.blocks
                if block.block_type in ("paragraph", "list_item")
            )

            if not section_text.strip():
                continue

            for part in _split_long_text(section_text):
                part = part.strip()
                if not part:
                    continue

                prefix = build_context_prefix_v3(
                    row=row,
                    doc_type="document",
                    chunk_type="section",
                    doc_id=doc_id,
                    section=section,
                )

                lineage = build_chunk_lineage(doc_id, section.heading, idx)

                chunks.append(
                    ChunkData(
                        id=f"{doc_id}_{idx}",
                        text=part,
                        embed_text=prefix + "\n\n" + part,
                        chunk_type="section",
                        chunk_index=idx,
                        ticket_id=doc_id,
                        context_prefix=prefix,
                        token_count=len(part.split()),
                        lineage=lineage,
                    )
                )

                idx += 1

    # ── Additional body fields ────────────────────────────────────────────────
    # Preserve previous behavior. If schema_detector already maps the same field
    # as question/answer/title, it should ideally not also appear in body_fields.
    for body_field in mapping.body_fields:
        _add(row.get(body_field, ""), "body")

    return chunks
