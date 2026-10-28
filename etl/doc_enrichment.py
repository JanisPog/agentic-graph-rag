"""Document enrichment — heading paths, section/document summaries, topological lineage."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

# Max chars of section text sent to LLM for summary
MAX_SECTION_INPUT = int(os.getenv("DOC_ENRICH_MAX_SECTION_INPUT", "3000"))
# Minimum section length to generate a summary
MIN_SECTION_LENGTH = 200


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SectionSummaryRecord:
    id:             str
    section_id:     str
    text:           str
    context_prefix: str
    lineage:        str
    token_count:    int = 0
    embedding:      list[float] | None = field(default=None, repr=False)


@dataclass
class DocumentSummaryRecord:
    id:             str
    doc_id:         str
    text:           str
    context_prefix: str
    lineage:        str
    token_count:    int = 0
    embedding:      list[float] | None = field(default=None, repr=False)


@dataclass
class EnrichmentResult:
    heading_paths:      list[str]
    section_summaries:  list[SectionSummaryRecord]
    document_summary:   DocumentSummaryRecord | None
    lineage_map:        dict[str, str]   # chunk_id → lineage string


# ── Lineage builder ───────────────────────────────────────────────────────────

def build_chunk_lineage(doc_id: str, section_heading: str, chunk_index: int) -> str:
    return f"Document:{doc_id} > Section:{section_heading} > Chunk:{chunk_index}"


# ── Section summary generation ────────────────────────────────────────────────

_SECTION_SUMMARY_PROMPT = """\
Summarize the following section in 2-3 sentences. Be concise and factual.
Section path: {heading_path}
Content:
{section_text}
Summary:"""


def generate_section_summary(section_text: str, heading_path: str, llm_fn: Callable) -> str:
    truncated = section_text[:MAX_SECTION_INPUT]
    prompt = _SECTION_SUMMARY_PROMPT.format(
        heading_path=heading_path,
        section_text=truncated,
    )
    try:
        return llm_fn(prompt).strip()
    except Exception as exc:
        log.warning("Section summary generation failed: %s", exc)
        return ""


# ── Document summary generation ──────────────────────────────────────────────

_DOCUMENT_SUMMARY_PROMPT = """\
You are summarizing a document titled '{title}'.
Document type: {doc_type}

Section summaries:
{numbered_sections}

Write a 3-5 sentence overview of the entire document.
Overview:"""


def generate_document_summary(
    title: str,
    section_summaries: list[str],
    metadata: dict,
    llm_fn: Callable,
) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(section_summaries) if s)
    if not numbered:
        numbered = "(no section summaries available)"

    prompt = _DOCUMENT_SUMMARY_PROMPT.format(
        title=title,
        doc_type=metadata.get("document_type", "unknown"),
        numbered_sections=numbered,
    )
    try:
        return llm_fn(prompt).strip()
    except Exception as exc:
        log.warning("Document summary generation failed: %s", exc)
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_document(
    doc_id: str,
    title: str,
    sections: list,   # list[ParsedSection] — avoid circular import
    metadata: dict,
    llm_fn: Callable,
    *,
    skip_summaries: bool = False,
) -> EnrichmentResult:
    from etl.pdf_parser import build_heading_paths  # noqa: PLC0415

    heading_paths = build_heading_paths(sections)
    for section, path in zip(sections, heading_paths):
        section.heading_path = path
    lineage_map: dict[str, str] = {}
    section_summaries: list[SectionSummaryRecord] = []

    if not skip_summaries:
        section_summary_texts: list[str] = []
        for section in sections:
            section_text = "\n\n".join(
                b.text for b in section.blocks
                if b.block_type in ("paragraph", "list_item")
            )
            if len(section_text) < MIN_SECTION_LENGTH:
                section_summary_texts.append("")
                continue

            summary_text = generate_section_summary(
                section_text, section.heading_path, llm_fn
            )
            section_summary_texts.append(summary_text)

            if summary_text:
                section_id = f"{doc_id}_sec_{section.position}"
                summary_id = f"{section_id}_summary"
                from etl.chunker_v3 import build_context_prefix_v3
                prefix_row = {"title": title}
                if metadata:
                    prefix_row.update(metadata)
                prefix = build_context_prefix_v3(prefix_row, "document", "section_summary", doc_id, section)
                section_summaries.append(SectionSummaryRecord(
                    id=summary_id,
                    section_id=section_id,
                    text=summary_text,
                    context_prefix=prefix,
                    lineage=f"Document:{doc_id} > Section:{section.heading}",
                    token_count=len(summary_text.split()),
                ))

        doc_summary_record = None
        active_summaries = [s for s in section_summary_texts if s]
        if active_summaries:
            doc_summary_text = generate_document_summary(
                title, active_summaries, metadata, llm_fn
            )
            if doc_summary_text:
                from etl.chunker_v3 import build_context_prefix_v3
                prefix_row = {"title": title}
                if metadata:
                    prefix_row.update(metadata)
                prefix = build_context_prefix_v3(prefix_row, "document", "document_summary", doc_id)
                doc_summary_record = DocumentSummaryRecord(
                    id=f"{doc_id}_summary",
                    doc_id=doc_id,
                    text=doc_summary_text,
                    context_prefix=prefix,
                    lineage=f"Document:{doc_id}",
                    token_count=len(doc_summary_text.split()),
                )
    else:
        doc_summary_record = None

    return EnrichmentResult(
        heading_paths=heading_paths,
        section_summaries=section_summaries,
        document_summary=doc_summary_record,
        lineage_map=lineage_map,
    )
