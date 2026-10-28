#!/usr/bin/env python3
"""
scripts/02_setup_paperless.py
==============================
One-time bootstrap script to setup Paperless-ngx custom fields.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.paperless_client import PaperlessClient

def main():
    print("Paperless-ngx Setup")
    print("=" * 60)
    
    if not config.PAPERLESS_TOKEN:
        print("[ERROR] PAPERLESS_TOKEN is not set in .env")
        print("Please log into Paperless UI, generate an API token, and add it to .env.")
        sys.exit(1)

    client = PaperlessClient(base_url=config.PAPERLESS_BASE_URL, token=config.PAPERLESS_TOKEN)
    
    print(f"Connecting to {config.PAPERLESS_BASE_URL}...")
    
    field_specs = [
        ("article_type", "string"),
        ("char_count", "integer"),
        ("content_hash", "string"),
        ("has_steps", "boolean"),
        ("language", "string"),
        ("section_count", "integer"),
        ("source_id", "string"),
        ("source_system", "string"),
        ("source_url", "url"),
        ("title", "string"),
        ("url_slug", "string"),
        ("word_count", "integer"),
    ]
    
    print("Ensuring custom fields exist...")
    field_map = client.ensure_custom_fields(field_specs)
    
    print("\nCustom Fields Mapping:")
    for name, fid in field_map.items():
        print(f"  {name}: {fid}")
        
    print("\n✅ Setup complete. You can now run the ingestion script.")

if __name__ == "__main__":
    main()
