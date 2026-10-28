"""
build_metadata.py – Deterministic metadata extraction for WixQA articles.
"""

import hashlib
import re
from urllib.parse import urlparse

import pandas as pd


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_to_slug(url: str | None) -> str:
    """Derive a readable slug from a URL's path component.

    Example: ``https://support.wix.com/en/article/my-topic`` → ``my-topic``
    """
    if not url:
        return ""
    try:
        path = urlparse(url).path.rstrip("/")
        slug = path.split("/")[-1] if path else ""
        # Replace hyphens/underscores with spaces for readability
        return slug.replace("-", " ").replace("_", " ").strip()
    except Exception:
        return ""


def _count_html_sections(html_content: str) -> int:
    """Count the number of semantic sections based on H2 and H3 tags.
    
    This matches the chunking logic of the Paperless ETL pipeline which
    splits documents at H2 and H3 boundaries.
    """
    if not html_content:
        return 1
    
    matches = re.findall(r"<h[23][^>]*>", html_content, re.IGNORECASE)
    # A document is always at least 1 section even without headings
    return max(1, len(matches))


def _has_html_steps(html_content: str) -> bool:
    """Return True if the HTML contains an ordered list (<ol>).
    
    This is much more robust than parsing plain text, as WixQA
    explicitly uses <ol> tags for step-by-step instructions.
    """
    if not html_content:
        return False
        
    return bool(re.search(r"<ol\b[^>]*>", html_content, re.IGNORECASE))


def build_metadata(row: pd.Series) -> dict:
    """Build a deterministic metadata dictionary for one corpus row.

    Args:
        row: A pandas Series with at least: id, url, contents, article_type.

    Returns:
        A dict with computed metadata fields matching the Paperless-ngx
        custom field definitions:
          article_type, char_count, content_hash, has_steps, language,
          section_count, source_id, source_system, source_url, title,
          url_slug, word_count.
    """
    contents: str = str(row.get("contents", ""))
    html_content: str = str(row.get("html_content", ""))
    url: str | None = row.get("url") if pd.notna(row.get("url")) else None
    
    # Get title directly from CSV, fallback to slug or default
    raw_title = str(row.get("title", "")).strip()
    if not raw_title or raw_title.lower() == "nan":
        slug = _url_to_slug(url)
        title = slug[:120].title() if slug else "Untitled Wix KB Article"
    else:
        title = raw_title[:120]

    return {
        # --- From source CSV (direct) ---
        "source_id": str(row.get("id", "")),
        "source_url": url or "",
        "article_type": str(row.get("article_type", "")),
        # --- Hardcoded constants ---
        "source_system": "wixqa",
        "language": "en",
        # --- Derived from URL ---
        "url_slug": _url_to_slug(url),
        # --- Derived from content (deterministic) ---
        "title": title,
        "content_hash": _sha256(contents),
        "char_count": len(contents),
        "word_count": len(contents.split()),
        "section_count": _count_html_sections(html_content),
        "has_steps": _has_html_steps(html_content),  # Paperless expects boolean
    }
