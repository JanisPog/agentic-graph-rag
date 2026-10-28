#!/usr/bin/env python3
"""
scripts/01_generate_pdfs.py
============================
Generate HTML and PDF files from the WixQA knowledge-base corpus CSV.

Run from the project root::

    python scripts/01_generate_pdfs.py

Output:
  - data/generated/html/<id>.html
  - data/generated/pdf/<id>.pdf
"""

import sys
from pathlib import Path

# Make the project root importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.build_metadata import build_metadata  # noqa: E402
from src.generate_pdf import generate_pdf_from_html  # noqa: E402
from src.load_wixqa import load_wixqa_corpus  # noqa: E402
from src.render_html import render_article_html  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("WixQA → PDF Generator")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load corpus
    # ------------------------------------------------------------------ #
    print(f"\n[01] Loading corpus from: {config.RAW_CSV_PATH}")
    df = load_wixqa_corpus(config.RAW_CSV_PATH)
    total_articles = len(df)
    print(f"[01] Loaded {total_articles} article(s).")

    # ------------------------------------------------------------------ #
    # 2. Ensure output directories exist
    # ------------------------------------------------------------------ #
    config.HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 3. Generate HTML + PDF for each article
    # ------------------------------------------------------------------ #
    generated = 0
    errors = 0

    for idx, row in df.iterrows():
        source_id = str(row["id"])
        try:
            # Build metadata
            meta = build_metadata(row)

            # Use the provided title and html_content directly
            title = str(row.get("title", ""))
            html_body = str(row.get("html_content", ""))

            # Render HTML
            html = render_article_html(
                title=title,
                html_body=html_body,
                metadata=meta,
            )

            # Save HTML
            html_path = config.HTML_OUTPUT_DIR / f"{source_id}.html"
            html_path.write_text(html, encoding="utf-8")

            # Generate PDF
            pdf_path = config.PDF_OUTPUT_DIR / f"{source_id}.pdf"
            generate_pdf_from_html(html, pdf_path)

            generated += 1
            if generated % 50 == 0 or generated == 1:
                print(f"  [{generated}/{total_articles}] Generated: {source_id}")

        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Skipping article '{source_id}': {exc}")
            errors += 1

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Articles loaded : {total_articles}")
    print(f"  PDFs generated  : {generated}")
    print(f"  Errors          : {errors}")
    print(f"  HTML output     : {config.HTML_OUTPUT_DIR}")
    print(f"  PDF output      : {config.PDF_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
