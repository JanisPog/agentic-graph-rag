"""KB context loader — reads KnowledgeBaseContext from Neo4j, cached after first call."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

_CACHE: str | None = None


def load_kb_context() -> str:
    from graph.neo4j_client import Neo4jClient  # noqa: PLC0415

    with Neo4jClient() as client:
        rows = client.run(
            "MATCH (k:KnowledgeBaseContext) RETURN k LIMIT 1"
        )
    if not rows:
        return ""

    k = rows[0]["k"]
    name   = k.get("organisation_name", "")
    domain = k.get("domain_description", "")

    parts = []
    if name:
        parts.append(f"Knowledge base organisation: {name}")
    if domain:
        parts.append(f"Domain: {domain}")

    return "\n".join(parts)


def get_kb_context() -> str:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = load_kb_context()
            if _CACHE:
                log.info("KB context loaded (%d chars)", len(_CACHE))
            else:
                log.info("No KnowledgeBaseContext node found — prompts will run without org context")
        except Exception as exc:
            log.warning("Could not load KB context: %s", exc)
            _CACHE = ""
    return _CACHE
