"""
render_html.py – Render a WixQA article as a clean, print-ready HTML document.
"""

from __future__ import annotations

import html as html_lib
import re


_CSS = """
    * {
        box-sizing: border-box;
    }

    html {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 11pt;
        color: #111827;
        background: #ffffff;
    }

    body {
        margin: 0;
        line-height: 1.6;
        hyphens: none;
        overflow-wrap: break-word;
        word-break: normal;
    }

    h1.article-title {
        font-size: 20pt;
        font-weight: 700;
        color: #0d2b4e;
        margin: 0 0 8mm 0;
        border-bottom: 2px solid #0d2b4e;
        padding-bottom: 3mm;
        line-height: 1.25;
    }

    .metadata {
        background: #f4f7fb;
        border-left: 4px solid #2563eb;
        border-radius: 4px;
        padding: 4mm 5mm;
        margin-bottom: 8mm;
        font-size: 9pt;
        color: #374151;
    }

    .metadata p {
        margin: 0 0 1.5mm 0;
    }

    .metadata .label {
        font-weight: 700;
        color: #1e3a5f;
    }

    .article-body {
        font-size: 11pt;
        line-height: 1.7;
    }

    .article-body h2 {
        font-size: 15pt;
        color: #0d2b4e;
        margin: 8mm 0 3mm 0;
        line-height: 1.3;
    }

    .article-body p {
        margin: 0 0 4mm 0;
    }

    .article-body ul,
    .article-body ol {
        margin: 0 0 5mm 7mm;
        padding: 0;
    }

    .article-body li {
        margin-bottom: 2mm;
    }

    .callout {
        border-left: 4px solid #64748b;
        background: #f8fafc;
        padding: 3mm 4mm;
        margin: 0 0 5mm 0;
    }

    .callout strong {
        color: #0f172a;
    }

    img {
        max-width: 100%;
        height: auto;
    }

    img[src$=".gif"], img[src$=".GIF"] {
        display: none;
    }

    @page {
        size: A4;
        margin: 22mm 24mm;
    }
"""


CALLOUT_RE = re.compile(r"\b(Important|Note|Tip|Tips|Notes):\s*", re.IGNORECASE)

KNOWN_LIST_ITEMS = [
    "Point of sale (POS) systems",
    "Delivery management systems",
    "Loyalty programs",
]

STEP_START_RE = re.compile(
    r"\b("
    r"Go to|Click|Select|Enter|Choose|Open|Scroll|Under|From|Enable|Disable"
    r")\b"
)


def _escape(text: str) -> str:
    return html_lib.escape(text, quote=True)


def _normalize_wix_contents(text: str) -> str:
    """Repair common WixQA plain-text artifacts before HTML rendering."""
    if text is None:
        return ""

    text = str(text)
    text = html_lib.unescape(text)

    # Normalize whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines but preserve intentional paragraph breaks.
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Fix very common sentence joins:
    # "app.Webhooks" -> "app. Webhooks"
    # "URL.Note:" -> "URL. Note:"
    text = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", text)

    # Put callout labels on their own block.
    text = re.sub(r"(?<!^)\b(Important|Note|Tip):", r"\n\n\1:", text, flags=re.IGNORECASE)

    # Separate "Click to learn..." style section markers.
    text = re.sub(
        r"(Click to learn[^.\n]*\.)",
        r"\n\n\1\n\n",
        text,
        flags=re.IGNORECASE,
    )

    # Fix "such as:Point of sale..." style list joins.
    text = re.sub(r"such as:\s*", "such as:\n", text, flags=re.IGNORECASE)

    for item in KNOWN_LIST_ITEMS:
        text = text.replace(item, f"\n- {item}")

    # Common WixQA issue:
    # "systemsDelivery" -> "systems\n- Delivery"
    text = re.sub(
        r"(?<=[a-z)])(?=(Delivery management systems|Loyalty programs))",
        "\n- ",
        text,
    )

    # Make dashboard instruction sequences easier to split.
    text = re.sub(
        r"(?<=[.!?])\s+(?=(Go to|Click|Select|Enter|Choose|Open|Under|From)\b)",
        "\n",
        text,
    )

    # Clean up blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _looks_like_heading(block: str) -> bool:
    block = block.strip()

    if len(block) > 120:
        return False

    if block.endswith("."):
        return False

    heading_starts = (
        "Click to learn",
        "To ",
        "About ",
        "Adding ",
        "Setting ",
        "Managing ",
        "Using ",
    )

    return block.startswith(heading_starts)


def _render_list(lines: list[str], ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    items = []

    for line in lines:
        clean = re.sub(r"^\s*[-*]\s+", "", line).strip()
        clean = re.sub(r"^\s*\d+[.)]\s+", "", clean).strip()

        if clean:
            items.append(f"<li>{_escape(clean)}</li>")

    if not items:
        return ""

    return f"<{tag}>\n" + "\n".join(items) + f"\n</{tag}>"


def _render_paragraph(block: str) -> str:
    block = block.strip()
    return f"<p>{_escape(block)}</p>" if block else ""


def _render_callout(label: str, text: str) -> str:
    label = label.strip().capitalize()
    text = text.strip()

    if not text:
        return f'<div class="callout"><p><strong>{_escape(label)}:</strong></p></div>'

    return (
        '<div class="callout">'
        f"<p><strong>{_escape(label)}:</strong> {_escape(text)}</p>"
        "</div>"
    )


def _contents_to_html(contents: str, title: str | None = None) -> str:
    """Convert WixQA plain text into more semantic HTML."""
    text = _normalize_wix_contents(contents)

    if not text:
        return ""

    # Avoid duplicated title if the content starts with the same title.
    if title:
        normalized_title = re.sub(r"\s+", " ", title).strip().lower()
        normalized_text_start = re.sub(r"\s+", " ", text[: len(title) + 20]).strip().lower()

        if normalized_text_start.startswith(normalized_title):
            text = text[len(title):].lstrip(" \n:-")

    blocks = re.split(r"\n{2,}", text)
    html_parts: list[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        callout_match = CALLOUT_RE.match(block)
        if callout_match:
            label = callout_match.group(1)
            rest = block[callout_match.end():].strip()
            html_parts.append(_render_callout(label, rest))
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if lines and all(line.startswith("- ") for line in lines):
            html_parts.append(_render_list(lines, ordered=False))
            continue

        if lines and len(lines) >= 2 and all(STEP_START_RE.match(line) for line in lines):
            html_parts.append(_render_list(lines, ordered=True))
            continue

        if len(lines) == 1 and _looks_like_heading(lines[0]):
            html_parts.append(f"<h2>{_escape(lines[0])}</h2>")
            continue

        # If the block contains multiple instruction-like lines, render as steps.
        instruction_lines = [line for line in lines if STEP_START_RE.match(line)]
        if len(instruction_lines) >= 2 and len(instruction_lines) == len(lines):
            html_parts.append(_render_list(lines, ordered=True))
            continue

        # Fallback: one paragraph.
        html_parts.append(_render_paragraph(" ".join(lines)))

    return "\n".join(html_parts)


def render_article_html(title: str, html_body: str, metadata: dict) -> str:
    """Render a complete HTML document for one WixQA article using provided HTML body."""
    escaped_title = _escape(title)

    # Prefer simple paragraphs over tables for better PDF text extraction.
    metadata_html = "\n".join(
        f'<p><span class="label">{_escape(label)}:</span> {_escape(str(value))}</p>'
        for label, value in [
            ("Source ID", metadata.get("source_id", "")),
            ("Source URL", metadata.get("source_url", "")),
            ("Article Type", metadata.get("article_type", "")),
            ("Content Hash", metadata.get("content_hash", "")),
        ]
        if value
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escaped_title}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <h1 class="article-title">{escaped_title}</h1>

  <section class="metadata" aria-label="Document metadata">
    {metadata_html}
  </section>

  <main class="article-body">
    {html_body}
  </main>
</body>
</html>
"""