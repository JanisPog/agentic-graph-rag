"""
config.py – Centralised configuration for wixqa-dms-ingestion.

All paths are resolved relative to the project root so that scripts can be
executed from any working directory as long as the project root is on
sys.path (the scripts handle this themselves).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root (one level above this file: wixqa-dms-ingestion/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from project root (silently ignored if the file is absent)
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_CSV_PATH: Path = PROJECT_ROOT / "data" / "raw" / "wix_kb_corpus.csv"
HTML_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "generated" / "html"
PDF_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "generated" / "pdf"
MAPPING_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "mappings" / "paperless_document_mapping.jsonl"

# ---------------------------------------------------------------------------
# Paperless-ngx configuration (read from environment / .env)
# ---------------------------------------------------------------------------
PAPERLESS_BASE_URL: str = os.getenv("PAPERLESS_BASE_URL", "http://localhost:8000/api")
PAPERLESS_TOKEN: str = os.getenv("PAPERLESS_TOKEN", "")
