"""Canonical metadata vocabulary — closed-vocab enums and live product list for planner/normalizer."""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


STATUS: frozenset[str] = frozenset({
    "Open", "Closed", "Pending Customer", "Resolved", "In Progress",
})

PRIORITY: frozenset[str] = frozenset({
    "Low", "Medium", "High", "Urgent",
})

LANGUAGE: frozenset[str] = frozenset({
    "English", "German", "French", "Spanish", "Chinese", "Japanese",
})

CHANNEL: frozenset[str] = frozenset({
    "Web Form", "Email", "Social Media", "Phone", "Chat",
})

DOC_TYPE: frozenset[str] = frozenset({
    "ticket", "document", "qa",
})

CLOSED_VOCAB: dict[str, frozenset[str]] = {
    "status":   STATUS,
    "priority": PRIORITY,
    "language": LANGUAGE,
    "channel":  CHANNEL,
    "doc_type": DOC_TYPE,
}

KNOWN_PRODUCTS_FALLBACK: tuple[str, ...] = (
    "E-commerce Store",
    "Subscription Service",
    "Web Portal",
    "Cloud Storage",
    "Payment Gateway",
    "Analytics Dashboard",
    "API Service",
    "CRM Platform",
    "Mobile App",
    "Billing System",
)

_KNOWN_PRODUCTS_TTL_S: float = 300.0
_cache_value: tuple[str, ...] | None = None
_cache_expires_at: float = 0.0
_cache_lock = threading.Lock()


def _fetch_products_from_graph() -> tuple[str, ...]:
    from etl.neo4j_client import Neo4jClient

    with Neo4jClient() as client:
        rows = client.run(
            "MATCH (m:Metadata {key: 'product'}) "
            "RETURN DISTINCT m.value AS value "
            "ORDER BY value"
        )
    values = tuple(
        row["value"] for row in rows
        if isinstance(row.get("value"), str) and row["value"].strip()
    )
    return values


def get_known_products(force_refresh: bool = False) -> tuple[str, ...]:
    """Return known product names, cached from Neo4j with TTL fallback."""
    global _cache_value, _cache_expires_at

    now = time.monotonic()
    if not force_refresh and _cache_value is not None and now < _cache_expires_at:
        return _cache_value

    with _cache_lock:
        now = time.monotonic()
        if not force_refresh and _cache_value is not None and now < _cache_expires_at:
            return _cache_value

        try:
            values = _fetch_products_from_graph()
        except Exception as exc:
            log.debug(
                "get_known_products: live fetch failed (%s); using fallback list",
                exc,
            )
            values = ()

        if not values:
            values = KNOWN_PRODUCTS_FALLBACK

        _cache_value = values
        _cache_expires_at = now + _KNOWN_PRODUCTS_TTL_S
        return values


def invalidate_known_products_cache() -> None:
    global _cache_value, _cache_expires_at
    with _cache_lock:
        _cache_value = None
        _cache_expires_at = 0.0