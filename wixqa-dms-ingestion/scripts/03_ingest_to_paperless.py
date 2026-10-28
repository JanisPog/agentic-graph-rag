#!/usr/bin/env python3
"""
scripts/03_ingest_to_paperless.py
==================================
Bulk ingest PDFs into Paperless-ngx with custom metadata.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.build_metadata import build_metadata
from src.load_wixqa import load_wixqa_corpus
from src.paperless_client import PaperlessClient
import logging
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs to Paperless-ngx")
    parser.add_argument("--limit", type=int, help="Process only the first N PDFs")
    parser.add_argument("--id", metavar="SOURCE_ID", help="Process only a specific PDF by source_id")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without modifying Paperless")
    args = parser.parse_args()

    print("Paperless-ngx Ingestion")
    print("=" * 60)
    
    if not config.PAPERLESS_TOKEN:
        print("[ERROR] PAPERLESS_TOKEN is not set in .env")
        sys.exit(1)

    print("[1] Loading corpus...")
    df = load_wixqa_corpus(config.RAW_CSV_PATH)
    print(f"Loaded {len(df)} articles.")
    
    metadata_by_source_id = {}
    for _, row in df.iterrows():
        meta = build_metadata(row)
        metadata_by_source_id[str(row["id"])] = meta

    print("\n[2] Finding PDFs...")
    pdf_files = sorted(config.PDF_OUTPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {config.PDF_OUTPUT_DIR}")
        sys.exit(1)
        
    if args.id:
        pdf_files = [p for p in pdf_files if p.stem == args.id]
        if not pdf_files:
            print(f"[ERROR] PDF for source_id {args.id} not found.")
            sys.exit(1)
            
    if args.limit:
        pdf_files = pdf_files[:args.limit]
        
    print(f"Found {len(pdf_files)} PDF(s) to process.")

    client = PaperlessClient(base_url=config.PAPERLESS_BASE_URL, token=config.PAPERLESS_TOKEN)

    print("\n[3] Ensuring custom fields exist...")
    field_specs = [
        ("article_type", "string"), ("char_count", "integer"), ("content_hash", "string"),
        ("has_steps", "boolean"), ("language", "string"), ("section_count", "integer"),
        ("source_id", "string"), ("source_system", "string"), ("source_url", "url"),
        ("title", "string"), ("url_slug", "string"), ("word_count", "integer")
    ]
    if not args.dry_run:
        field_id_map = client.ensure_custom_fields(field_specs)
    else:
        field_id_map = {name: i+1 for i, (name, _) in enumerate(field_specs)}
        print("[DRY RUN] Skipped creating custom fields.")

    print("\n[4] Ingesting documents...")
    
    succeeded, skipped, failed = 0, 0, 0
    
    config.MAPPING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config.MAPPING_OUTPUT_PATH, "a" if not args.dry_run else "w", encoding="utf-8") as map_file:
        for pdf_path in pdf_files:
            source_id = pdf_path.stem
            meta = metadata_by_source_id.get(source_id, {})
            title = f"Wix KB Article - {source_id}"
            
            print(f"\nProcessing '{source_id}'...")
            
            try:
                if not args.dry_run:
                    existing_id = client.find_document_by_source_id(source_id)
                    if existing_id:
                        print(f"  -> Skipped: Already ingested as Document ID {existing_id}")
                        skipped += 1
                        continue
                        
                    print(f"  -> Uploading...")
                    task_uuid = client.upload_document(pdf_path, title)
                    
                    print(f"  -> Waiting for consumption (task {task_uuid})...")
                    doc_id = client.wait_for_consumption(task_uuid)
                    
                    print(f"  -> Setting custom fields on Document ID {doc_id}...")
                    client.set_custom_fields(doc_id, meta, field_id_map)
                    
                    print(f"  -> Successfully ingested '{source_id}' -> Document ID {doc_id}")
                    
                    record = {
                        "source_id": source_id,
                        "paperless_document_id": doc_id,
                        "status": "success"
                    }
                    map_file.write(json.dumps(record) + "\n")
                    map_file.flush()
                    succeeded += 1
                else:
                    print(f"  -> [DRY RUN] Would ingest and attach metadata.")
                    succeeded += 1
            except Exception as e:
                print(f"  -> ERROR: {e}")
                if not args.dry_run:
                    record = {
                        "source_id": source_id,
                        "status": "error",
                        "error": str(e)
                    }
                    map_file.write(json.dumps(record) + "\n")
                    map_file.flush()
                failed += 1

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total:     {len(pdf_files)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    if not args.dry_run:
        print(f"  Mapping written to: {config.MAPPING_OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
