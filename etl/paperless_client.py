"""Paperless-ngx REST API client — fetches documents + metadata for ETL.

Handles paginated listing, metadata resolution, and PDF download.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)


@dataclass
class PaperlessDocument:
    paperless_id:   int
    title:          str
    correspondent:  str | None
    document_type:  str | None
    tags:           list[str]
    custom_fields:  dict[str, Any]
    created:        str
    modified:       str
    added:          str
    content:        str
    pdf_bytes:      bytes = field(default=b"", repr=False)


class PaperlessClient:
    """Thin REST client for Paperless-ngx API."""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self._base_url = (base_url or config.PAPERLESS_BASE_URL).rstrip("/")
        self._token = token or config.PAPERLESS_TOKEN
        self._headers = {
            "Authorization": f"Token {self._token}",
            "Accept": "application/json; version=2",
        }
        self._correspondents: dict[int, str] | None = None
        self._document_types: dict[int, str] | None = None
        self._tags_cache: dict[int, str] | None = None
        self._custom_field_defs: dict[int, str] | None = None

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Paperless GET {url} failed: {exc}") from exc

    def _get_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Paperless GET {url} failed: {exc}") from exc

    def _load_all_pages(self, endpoint: str) -> list[dict]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}?page_size=100"
        results: list[dict] = []
        while url:
            data = self._get_json(url)
            results.extend(data.get("results", []))
            url = data.get("next")
        return results

    def _ensure_correspondents(self) -> dict[int, str]:
        if self._correspondents is None:
            items = self._load_all_pages("correspondents/")
            self._correspondents = {i["id"]: i["name"] for i in items}
        return self._correspondents

    def _ensure_document_types(self) -> dict[int, str]:
        if self._document_types is None:
            items = self._load_all_pages("document_types/")
            self._document_types = {i["id"]: i["name"] for i in items}
        return self._document_types

    def _ensure_tags(self) -> dict[int, str]:
        if self._tags_cache is None:
            items = self._load_all_pages("tags/")
            self._tags_cache = {i["id"]: i["name"] for i in items}
        return self._tags_cache

    def _ensure_custom_fields(self) -> dict[int, str]:
        if self._custom_field_defs is None:
            items = self._load_all_pages("custom_fields/")
            self._custom_field_defs = {i["id"]: i["name"] for i in items}
        return self._custom_field_defs

    def list_documents(self, *, page_size: int = 100, ordering: str = "-created",
                       tag_filter: str | None = None, doc_type_filter: str | None = None,
                       limit: int | None = None) -> list[dict]:
        params = f"page_size={page_size}&ordering={ordering}"
        if tag_filter:
            tags = self._ensure_tags()
            tag_id = next((t for t, n in tags.items() if n.lower() == tag_filter.lower()), None)
            if tag_id:
                params += f"&tags__id={tag_id}"
        if doc_type_filter:
            dtypes = self._ensure_document_types()
            dt_id = next((d for d, n in dtypes.items() if n.lower() == doc_type_filter.lower()), None)
            if dt_id:
                params += f"&document_type__id={dt_id}"

        url = f"{self._base_url}/documents/?{params}"
        results: list[dict] = []
        while url:
            data = self._get_json(url)
            results.extend(data.get("results", []))
            if limit and len(results) >= limit:
                results = results[:limit]
                break
            url = data.get("next")
        log.info("Fetched %d documents from Paperless", len(results))
        return results

    def get_document(self, doc_id: int) -> dict:
        return self._get_json(f"{self._base_url}/documents/{doc_id}/")

    def download_pdf(self, doc_id: int) -> bytes:
        return self._get_bytes(f"{self._base_url}/documents/{doc_id}/download/")

    def build_paperless_document(self, doc_meta: dict, *, download: bool = True) -> PaperlessDocument:
        correspondents = self._ensure_correspondents()
        doc_types = self._ensure_document_types()
        tags = self._ensure_tags()
        cf_defs = self._ensure_custom_fields()

        corr_name = correspondents.get(doc_meta.get("correspondent")) if doc_meta.get("correspondent") else None
        dt_name = doc_types.get(doc_meta.get("document_type")) if doc_meta.get("document_type") else None
        tag_names = [tags.get(tid, f"tag_{tid}") for tid in doc_meta.get("tags", [])]

        custom_fields: dict[str, Any] = {}
        for cf in doc_meta.get("custom_fields", []):
            field_id = cf.get("field")
            field_name = cf_defs.get(field_id, f"field_{field_id}")
            custom_fields[field_name] = cf.get("value")

        pdf_bytes = b""
        if download:
            try:
                pdf_bytes = self.download_pdf(doc_meta["id"])
            except Exception as exc:
                log.warning("Failed to download PDF for doc %d: %s", doc_meta["id"], exc)

        return PaperlessDocument(
            paperless_id=doc_meta["id"],
            title=doc_meta.get("title", f"document_{doc_meta['id']}"),
            correspondent=corr_name, document_type=dt_name,
            tags=tag_names, custom_fields=custom_fields,
            created=doc_meta.get("created", ""),
            modified=doc_meta.get("modified", ""),
            added=doc_meta.get("added", ""),
            content=doc_meta.get("content", ""),
            pdf_bytes=pdf_bytes,
        )

    def get_ingested_ids(self) -> set[int]:
        """Check Neo4j for already-ingested Paperless document IDs."""
        try:
            from graph.neo4j_client import Neo4jClient
            with Neo4jClient() as client:
                rows = client.run(
                    "MATCH (d:Document) WHERE d.id STARTS WITH 'paperless_' RETURN d.id AS id"
                )
                return {int(r["id"].replace("paperless_", "")) for r in rows
                        if r["id"].replace("paperless_", "").isdigit()}
        except Exception as exc:
            log.warning("Could not check existing docs in Neo4j: %s", exc)
            return set()
