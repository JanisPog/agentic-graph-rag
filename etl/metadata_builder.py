"""Metadata builder — produces Metadata and Concept node dicts for Neo4j.

Metadata nodes form the structured layer beneath each Document:
  (Document)-[:HAS_METADATA]->(Metadata {key, value, value_type})

Concept nodes group related entities by abstract label:
  (Entity)-[:BELONGS_TO]->(Concept {label})

Both are lightweight — no LLM calls needed here.
"""
from __future__ import annotations

import re
from typing import Any

from etl.schema_detector import ColumnMapping
from etl.ner_v2 import ExtractedEntity

# ── Value type inference ───────────────────────────────────────────────────────

_DATE_RE    = re.compile(r"^\d{4}-\d{2}-\d{2}")
_INT_RE     = re.compile(r"^\d+$")
_FLOAT_RE   = re.compile(r"^\d+\.\d+$")
_ENUM_HINTS = {"priority", "status", "severity", "state", "channel", "type", "category"}


def _value_type(key: str, value: str) -> str:
    if _DATE_RE.match(value):
        return "date"
    if _INT_RE.match(value):
        return "int"
    if _FLOAT_RE.match(value):
        return "float"
    if key.lower() in _ENUM_HINTS:
        return "enum"
    return "str"


# ── Metadata node builder ──────────────────────────────────────────────────────

def build_metadata_nodes(
    doc_id: str,
    row: dict[str, Any],
    mapping: ColumnMapping,
) -> list[dict]:
    """Return a list of Metadata node dicts for a single document row."""
    nodes = []
    for field_name in mapping.metadata_fields:
        val = str(row.get(field_name, "")).strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        vtype = _value_type(field_name, val)
        meta_id = f"meta_{field_name}_{val.lower().replace(' ', '_')}"
        nodes.append({
            "id":         meta_id,
            "doc_id":     doc_id,
            "key":        field_name,
            "value":      val,
            "value_type": vtype,
        })
    return nodes


# ── Concept node builder ───────────────────────────────────────────────────────

# Maps entity types to a Concept label
_TYPE_TO_CONCEPT = {
    "product":      "Product",
    "category":     "Category",
    "system":       "System",
    "error_code":   "ErrorCode",
    "version":      "Version",
    "procedure":    "Procedure",
    "component":    "Component",
    "organisation": "Organisation",
}


def build_concept_nodes(entities: list[ExtractedEntity]) -> list[dict]:
    """Return deduplicated Concept node dicts derived from extracted entities."""
    seen: set[str] = set()
    concepts = []
    for e in entities:
        label = _TYPE_TO_CONCEPT.get(e.type)
        if not label:
            continue
        concept_id = f"concept_{label.lower()}"
        if concept_id not in seen:
            seen.add(concept_id)
            concepts.append({"id": concept_id, "label": label, "description": label})
    return concepts


def build_belongs_to_edges(entities: list[ExtractedEntity]) -> list[dict]:
    """Return BELONGS_TO edge dicts: {entity_id, concept_id, confidence}."""
    edges = []
    for e in entities:
        label = _TYPE_TO_CONCEPT.get(e.type)
        if not label:
            continue
        edges.append({
            "entity_id":  e.id,
            "concept_id": f"concept_{label.lower()}",
            "confidence": e.confidence,
        })
    return edges
