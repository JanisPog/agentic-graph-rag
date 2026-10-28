"""Metadata filter normalizer — validates and derives metadata_filters from LLM output + query keywords."""
from __future__ import annotations

import logging
import re
from typing import Iterable

import config
from agents.metadata_vocab import CLOSED_VOCAB, DOC_TYPE, LANGUAGE

log = logging.getLogger(__name__)


_DOC_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:kb|knowledge\s+base)\s+articles?\b", re.IGNORECASE),  "document"),
    (re.compile(r"\bknowledge\s+base\b",                    re.IGNORECASE),  "document"),
    (re.compile(r"\bpaperless\s+documents?\b",              re.IGNORECASE),  "document"),
    (re.compile(r"\b(?:q\s*&\s*a|qa)\b",                    re.IGNORECASE),  "qa"),
    (re.compile(r"\b(?:support\s+)?tickets?\b",             re.IGNORECASE),  "ticket"),
    (re.compile(r"\bdocuments?\b",                          re.IGNORECASE),  "document"),
]


def _derive_doc_type(query: str) -> list[str] | None:
    """Return [doc_type] if query explicitly names one, else None."""
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if pattern.search(query):
            return [doc_type]
    return None


_LANGUAGE_SYNONYMS: dict[str, str] = {
    "english":  "English", "german":   "German", "french":   "French",
    "spanish":  "Spanish", "chinese":  "Chinese", "japanese": "Japanese",
    "deutsch":      "German", "englisch":     "English", "französisch":  "French",
    "spanisch":     "Spanish", "chinesisch":   "Chinese", "japanisch":    "Japanese",
    "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "zh": "Chinese", "ja": "Japanese",
}


def _derive_language(query: str) -> list[str] | None:
    """Return [Language] if query explicitly mentions a language word."""
    query_lower = query.lower()
    matched: list[str] = []
    for synonym, canonical in _LANGUAGE_SYNONYMS.items():
        # Word-boundary match so "en" doesn't match "tenant"
        if re.search(rf"\b{re.escape(synonym)}\b", query_lower):
            if canonical not in matched:
                matched.append(canonical)
    return matched or None


def _match_against_vocab(values: Iterable[str], vocab: frozenset[str]) -> list[str]:
    """Return canonical-cased vocab matches for the given values."""
    lookup = {v.lower(): v for v in vocab}
    out: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        canonical = lookup.get(raw.strip().lower())
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def _coerce_to_list(value) -> list[str]:
    """Coerce planner output to list[str] — handles string, list, or None."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if isinstance(v, str)]
    return []


def normalise_metadata_filters(filters: dict, query: str) -> dict:
    if not isinstance(filters, dict):
        return {}

    out: dict[str, list[str]] = {}
    query_lower = query.lower()
    allowed_keys = set(config.METADATA_FILTER_KEYS)

    dropped_unknown = [k for k in filters if k not in allowed_keys]
    if dropped_unknown:
        log.debug("normalise_metadata_filters: dropped unknown keys %s", dropped_unknown)

    if "status" in allowed_keys:
        status_values: list[str] = []
        if "status" in filters:
            raw_values = _coerce_to_list(filters["status"])
            status_values = _match_against_vocab(raw_values, CLOSED_VOCAB["status"])
        if re.search(r"\bresolved\b", query_lower):
            for v in ("Resolved", "Closed"):
                if v not in status_values:
                    status_values.append(v)
        if status_values:
            out["status"] = status_values

    for key in ("priority", "channel"):
        if key in filters and key in allowed_keys:
            raw_values = _coerce_to_list(filters[key])
            matched = _match_against_vocab(raw_values, CLOSED_VOCAB[key])
            if matched:
                out[key] = matched

    derived_lang = _derive_language(query)
    if derived_lang and "language" in allowed_keys:
        out["language"] = derived_lang

    derived_dt = _derive_doc_type(query)
    if derived_dt and "doc_type" in allowed_keys:
        out["doc_type"] = derived_dt

    if "category" in filters and "category" in allowed_keys:
        if re.search(r"\bcategory\b", query_lower):
            raw_values = _coerce_to_list(filters["category"])
            cleaned = [v.strip() for v in raw_values if isinstance(v, str) and v.strip()]
            if cleaned:
                out["category"] = cleaned

    for key in ("product", "source_system", "severity"):
        if key in filters and key in allowed_keys:
            raw_values = _coerce_to_list(filters[key])
            cleaned = [v.strip() for v in raw_values if isinstance(v, str) and v.strip()]
            if cleaned:
                out[key] = cleaned

    return out