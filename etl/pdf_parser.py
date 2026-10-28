"""PDF parser — structure-aware extraction using pymupdf.

Extracts text blocks with font metadata, classifies them via a relative
font-size heuristic (H1/H2/H3/paragraph/caption), groups into sections,
and builds hierarchical heading paths. Includes header/footer detection.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    text:        str
    block_type:  str          # h1 | h2 | h3 | paragraph | list_item | caption | metadata
    font_size:   float
    is_bold:     bool
    page_number: int
    position:    int          # sequential index across entire document


@dataclass
class ParsedSection:
    heading:      str
    level:        int            # 1, 2, or 3
    position:     int            # section index within document
    blocks:       list[ParsedBlock]
    heading_path: str = ""       # set by build_heading_paths()


# ── Font-size heuristic thresholds ─────────────────────────────────────────────

H1_RATIO      = float(os.getenv("PDF_HEADING_H1_RATIO", "1.5"))
H2_RATIO      = float(os.getenv("PDF_HEADING_H2_RATIO", "1.15"))
CAPTION_RATIO  = float(os.getenv("PDF_CAPTION_RATIO", "0.85"))


# ── Header/footer detection ───────────────────────────────────────────────────

def _detect_repeating_elements(pages_blocks: list[list[dict]], tolerance: float = 5.0) -> set[str]:
    """Detect text that appears at similar y-positions across multiple pages.

    Returns a set of normalised text strings that are likely headers/footers.
    """
    if len(pages_blocks) < 3:
        return set()

    # Collect (normalised_text, y_position) per page
    top_texts: Counter[str] = Counter()
    bottom_texts: Counter[str] = Counter()

    for page_blocks in pages_blocks:
        if not page_blocks:
            continue
        sorted_by_y = sorted(page_blocks, key=lambda b: b["y"])
        # Top 2 and bottom 2 blocks on each page
        for block in sorted_by_y[:2]:
            txt = block["text"].strip()
            if txt and len(txt) < 100:
                top_texts[txt] += 1
        for block in sorted_by_y[-2:]:
            txt = block["text"].strip()
            if txt and len(txt) < 100:
                bottom_texts[txt] += 1

    n_pages = len(pages_blocks)
    threshold = max(3, n_pages * 0.5)  # must appear on ≥50% of pages

    repeating = set()
    for txt, count in top_texts.items():
        if count >= threshold:
            repeating.add(txt)
    for txt, count in bottom_texts.items():
        if count >= threshold:
            repeating.add(txt)

    if repeating:
        log.debug("Detected %d repeating header/footer elements", len(repeating))
    return repeating


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_raw_blocks(pdf_bytes: bytes) -> tuple[list[list[dict]], list[dict]]:
    """Extract text blocks with font metadata from PDF bytes using pymupdf.

    Returns:
        (pages_blocks, all_blocks) where each block is a dict with
        text, font_size, is_bold, page_number, y, x keys.
    """
    import pymupdf  # noqa: PLC0415

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_blocks: list[list[dict]] = []
    all_blocks: list[dict] = []

    for page_num, page in enumerate(doc):
        page_data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        page_blocks: list[dict] = []

        for block in page_data.get("blocks", []):
            if block.get("type") != 0:  # skip image blocks
                continue

            block_text_parts: list[str] = []
            block_font_sizes: list[float] = []
            block_bold_count = 0
            block_total_spans = 0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    block_text_parts.append(text)
                    block_font_sizes.append(span.get("size", 12.0))
                    block_total_spans += 1
                    # Check bold via font flags (bit 4 = bold in pymupdf)
                    flags = span.get("flags", 0)
                    if flags & (1 << 4):
                        block_bold_count += 1

            full_text = " ".join(block_text_parts).strip()
            if not full_text:
                continue

            avg_font_size = sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else 12.0
            is_bold = block_bold_count > block_total_spans * 0.5

            bbox = block.get("bbox", (0, 0, 0, 0))
            page_blocks.append({
                "text": full_text,
                "font_size": round(avg_font_size, 2),
                "is_bold": is_bold,
                "page_number": page_num,
                "y": bbox[1],
                "x": bbox[0],
            })

        # Sort by (y, x) to handle multi-column layouts
        page_blocks.sort(key=lambda b: (b["y"], b["x"]))
        pages_blocks.append(page_blocks)
        all_blocks.extend(page_blocks)

    doc.close()
    return pages_blocks, all_blocks


def _compute_body_font_size(blocks: list[dict]) -> float:
    """Compute the body font size as the statistical mode of all block font sizes."""
    if not blocks:
        return 12.0
    sizes = [round(b["font_size"], 1) for b in blocks]
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def _classify_blocks(blocks: list[dict], body_size: float,
                     repeating: set[str]) -> list[ParsedBlock]:
    """Classify each raw block into a ParsedBlock with block_type."""
    parsed: list[ParsedBlock] = []
    h1_found = False

    for idx, block in enumerate(blocks):
        text = block["text"]

        # Skip repeating headers/footers
        if text.strip() in repeating:
            continue

        # Skip very short non-informative text (page numbers etc.)
        if len(text.strip()) < 3 and text.strip().isdigit():
            continue

        fs = block["font_size"]
        bold = block["is_bold"]
        ratio = fs / body_size if body_size > 0 else 1.0

        # Classification
        if ratio >= H1_RATIO and not h1_found:
            block_type = "h1"
            h1_found = True
        elif ratio >= H2_RATIO and bold:
            block_type = "h2"
        elif bold and ratio >= 0.95 and len(text) < 120:
            block_type = "h3"
        elif ratio < CAPTION_RATIO:
            block_type = "caption"
        elif text.strip().startswith(("- ", "• ", "* ", "– ")):
            block_type = "list_item"
        else:
            block_type = "paragraph"

        parsed.append(ParsedBlock(
            text=text,
            block_type=block_type,
            font_size=fs,
            is_bold=bold,
            page_number=block["page_number"],
            position=idx,
        ))

    return parsed


# ── Section builder ───────────────────────────────────────────────────────────

def _group_into_sections(blocks: list[ParsedBlock]) -> list[ParsedSection]:
    """Group consecutive blocks into sections based on heading boundaries.

    H1/H2 starts a new section. H3 stays within its parent section.
    If no headings detected → single section "main".
    """
    if not blocks:
        return []

    has_headings = any(b.block_type in ("h1", "h2") for b in blocks)
    if not has_headings:
        return [ParsedSection(
            heading="main", level=1, position=0, blocks=blocks,
        )]

    sections: list[ParsedSection] = []
    current_blocks: list[ParsedBlock] = []
    current_heading = "main"
    current_level = 1

    for block in blocks:
        if block.block_type in ("h1", "h2"):
            # Save previous section if it has content
            if current_blocks:
                sections.append(ParsedSection(
                    heading=current_heading,
                    level=current_level,
                    position=len(sections),
                    blocks=current_blocks,
                ))
            current_heading = block.text.strip()
            current_level = 1 if block.block_type == "h1" else 2
            current_blocks = [block]
        else:
            current_blocks.append(block)

    # Don't forget the last section
    if current_blocks:
        sections.append(ParsedSection(
            heading=current_heading,
            level=current_level,
            position=len(sections),
            blocks=current_blocks,
        ))

    return sections


# ── Heading path builder ──────────────────────────────────────────────────────

def build_heading_paths(sections: list[ParsedSection]) -> list[str]:
    """Build hierarchical heading path for each section.

    Example: ["Index of Terms", "Index of Terms > General SEO terms"]
    """
    paths: list[str] = []
    stack: list[tuple[int, str]] = []  # [(level, heading)]
    for section in sections:
        while stack and stack[-1][0] >= section.level:
            stack.pop()
        stack.append((section.level, section.heading))
        paths.append(" > ".join(h for _, h in stack))
    return paths


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_bytes: bytes) -> tuple[str, list[ParsedSection]]:
    """Parse PDF binary → (detected_title, sections with heading_paths).

    Uses pymupdf to extract text blocks with font metadata.
    Applies relative font-size heuristic to classify blocks.
    Groups blocks into sections based on heading boundaries.
    Detects and strips repeating headers/footers.
    Calls build_heading_paths() to set heading_path on each section.
    """
    if not pdf_bytes:
        log.warning("Empty PDF bytes received")
        return "", []

    pages_blocks, all_blocks = _extract_raw_blocks(pdf_bytes)

    if not all_blocks:
        log.warning("No text blocks extracted from PDF")
        return "", []

    body_size = _compute_body_font_size(all_blocks)
    log.debug("Body font size: %.1f", body_size)

    repeating = _detect_repeating_elements(pages_blocks)
    parsed_blocks = _classify_blocks(all_blocks, body_size, repeating)

    if not parsed_blocks:
        log.warning("No blocks after classification")
        return "", []

    # Detect title from first H1 block
    title = ""
    for b in parsed_blocks:
        if b.block_type == "h1":
            title = b.text.strip()
            break

    sections = _group_into_sections(parsed_blocks)
    heading_paths = build_heading_paths(sections)
    for section, path in zip(sections, heading_paths):
        section.heading_path = path

    log.info("Parsed PDF: title=%r, %d sections, %d blocks",
             title[:60] if title else "(none)", len(sections), len(parsed_blocks))

    return title, sections
