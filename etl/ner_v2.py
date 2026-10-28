"""Entity extractor v2 — regex + LLM extraction; ENTITY_EXTRACT_MODE=fast skips LLM."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from etl.schema_detector import ColumnMapping

log = logging.getLogger(__name__)

def _entity_extract_mode() -> str:
    return os.getenv("ENTITY_EXTRACT_MODE", "full")

# ── ExtractedEntity ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    id:         str
    name:       str
    type:       str    # product|category|system|error_code|version|procedure|component|organisation
    source:     str    # "metadata" | "regex" | "llm"
    confidence: float  # 0.0–1.0


# ── Regex patterns ─────────────────────────────────────────────────────────────

_RE_ERROR_CODE = re.compile(
    r"\b(?:0x[0-9A-Fa-f]{4,8}|error\s*\d{3,6}|http\s*[45]\d{2}|e\d{4,6})\b", re.IGNORECASE
)
_RE_VERSION = re.compile(
    r"\b(?:"
    r"v\d+(?:\.\d+)*"
    r"|(?:version|ver|release|build|update|patch)\s*\d+(?:\.\d+)*"
    r"|(?:ios|android|windows|macos|ubuntu|debian)\s*\d+(?:\.\d+)*"
    r")\b",
    re.IGNORECASE,
)
_RE_URL = re.compile(r"https?://\S+")


def _safe_entity_id(name: str, etype: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", name.strip().lower())
    return f"{etype}_{cleaned}"


def _extract_regex(text: str) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for m in _RE_ERROR_CODE.finditer(text):
        name = m.group(0).strip()
        entities.append(ExtractedEntity(
            id=_safe_entity_id(name, "error_code"), name=name,
            type="error_code", source="regex", confidence=0.95,
        ))
    for m in _RE_VERSION.finditer(text):
        name = m.group(0).strip()
        entities.append(ExtractedEntity(
            id=_safe_entity_id(name, "version"), name=name,
            type="version", source="regex", confidence=0.90,
        ))
    return entities


def _extract_metadata_entities(row: dict[str, Any], mapping: ColumnMapping) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for field_name in mapping.metadata_fields:
        val = str(row.get(field_name, "")).strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        etype = _infer_meta_type(field_name)
        entities.append(ExtractedEntity(
            id=_safe_entity_id(val, etype), name=val,
            type=etype, source="metadata", confidence=1.0,
        ))
    return entities


_META_TYPE_MAP = {
    "product": "product", "category": "category", "type": "category",
    "priority": "category", "channel": "category", "department": "organisation",
    "platform": "system", "environment": "system", "tags": "category",
    "severity": "category", "version": "version",
}

def _infer_meta_type(field_name: str) -> str:
    return _META_TYPE_MAP.get(field_name.lower(), "category")


# ── LLM extraction ─────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
Extract named entities from the following support text. Return ONLY a JSON array.
Each element: {{"name": "<text>", "type": "<type>"}}
Types: product, system, procedure, component, organisation
Ignore generic words. Return [] if nothing notable found.

Text:
{text}
"""


def _extract_llm(text: str) -> list[ExtractedEntity]:
    if not text or len(text.strip()) < 20:
        return []
    try:
        from agents.llm import chat_json  # noqa: PLC0415
        raw = chat_json(_LLM_PROMPT.format(text=text[:800]))
        if not isinstance(raw, list):
            return []
        entities = []
        for item in raw[:10]:   # cap at 10 per chunk
            if not isinstance(item, dict):
                continue
            name  = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip().lower()
            if not name or etype not in ("product", "system", "procedure", "component", "organisation"):
                continue
            entities.append(ExtractedEntity(
                id=_safe_entity_id(name, etype), name=name,
                type=etype, source="llm", confidence=0.80,
            ))
        return entities
    except Exception as exc:
        log.debug("LLM entity extraction failed: %s", exc)
        return []


def _dedup(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen: dict[str, ExtractedEntity] = {}
    for e in entities:
        if e.id not in seen or e.confidence > seen[e.id].confidence:
            seen[e.id] = e
    return list(seen.values())


def generate_surface_forms(entity: ExtractedEntity) -> list[str]:
    canonical = entity.name.strip()
    forms: set[str] = set()
    forms.add(canonical.lower())
    stripped = re.sub(r"[^a-z0-9\s]", "", canonical.lower()).strip()
    if stripped:
        forms.add(stripped)
    for token in canonical.lower().split():
        token_clean = re.sub(r"[^a-z0-9]", "", token)
        if len(token_clean) >= 4:
            forms.add(token_clean)

    return [f for f in forms if f]


_SYNONYM_PROMPT = """\
List up to 6 common synonyms, abbreviations, or informal names that a customer \
might use when referring to "{name}" (type: {etype}) in a support context.
Return ONLY a JSON array of strings. Example: ["payment method", "card", "billing info"]
Return [] if there are no common variants.
"""


def generate_synonyms_llm(entities: list[ExtractedEntity]) -> dict[str, list[str]]:
    """Return entity_id → synonym list via one LLM call per entity (ingest-time only)."""
    from agents.llm import chat_json  # noqa: PLC0415

    result: dict[str, list[str]] = {}
    for entity in entities:
        try:
            raw = chat_json(_SYNONYM_PROMPT.format(name=entity.name, etype=entity.type))
            if not isinstance(raw, list):
                result[entity.id] = []
                continue
            synonyms = [str(s).strip().lower() for s in raw if isinstance(s, str) and str(s).strip()]
            result[entity.id] = synonyms[:6]
            log.debug("Synonyms for %r (%s): %s", entity.name, entity.type, synonyms)
        except Exception as exc:
            log.debug("Synonym generation failed for %r: %s", entity.name, exc)
            result[entity.id] = []
    return result


def extract_entities(row: dict[str, Any], mapping: ColumnMapping) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    entities.extend(_extract_metadata_entities(row, mapping))
    text_fields = [mapping.question_field, mapping.answer_field] + mapping.body_fields
    combined_text = " ".join(
        str(row.get(f, "")) for f in text_fields if f and row.get(f)
    )
    entities.extend(_extract_regex(combined_text))
    if _entity_extract_mode() == "full" and combined_text.strip():
        entities.extend(_extract_llm(combined_text))
    return _dedup(entities)
