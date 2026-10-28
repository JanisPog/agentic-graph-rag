"""Neo4j driver wrapper — connection management, query helpers, schema setup."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from neo4j import GraphDatabase

log = logging.getLogger(__name__)

import config as _config

def _embed_dim() -> int:
    return _config.EMBED_DIM


def _build_schema_statements() -> list[str]:
    dim = _embed_dim()
    return [
        "CREATE CONSTRAINT doc_id_unique      IF NOT EXISTS FOR (d:Document)   REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique    IF NOT EXISTS FOR (c:Chunk)      REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id_unique   IF NOT EXISTS FOR (e:Entity)     REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT concept_id_unique  IF NOT EXISTS FOR (c:Concept)    REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT metadata_id_unique IF NOT EXISTS FOR (m:Metadata)   REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT section_id_unique  IF NOT EXISTS FOR (s:Section)    REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT kb_context_unique  IF NOT EXISTS FOR (k:KnowledgeBaseContext) REQUIRE k.id IS UNIQUE",
        "CREATE CONSTRAINT surface_form_unique IF NOT EXISTS FOR (s:SurfaceForm) REQUIRE s.id IS UNIQUE",
        (
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{ indexConfig: {{ "
            f"`vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: 'cosine' "
            f"}}}}"
        ),
        "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.context_prefix]",
        "CREATE INDEX chunk_hop_index    IF NOT EXISTS FOR (c:Chunk)    ON (c.id, c.chunk_index)",
        "CREATE INDEX metadata_lookup    IF NOT EXISTS FOR (m:Metadata) ON (m.key, m.value)",
        "CREATE INDEX entity_name_index  IF NOT EXISTS FOR (e:Entity)   ON (e.name)",
        "CREATE INDEX surface_form_index IF NOT EXISTS FOR (s:SurfaceForm) ON (s.form)",
        "CREATE INDEX doc_type_index     IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
        "CREATE CONSTRAINT known_issue_id IF NOT EXISTS FOR (k:KnownIssue) REQUIRE k.id IS UNIQUE",
        (
            "CREATE VECTOR INDEX known_issue_embedding IF NOT EXISTS "
            "FOR (k:KnownIssue) ON (k.problem_embedding) "
            f"OPTIONS {{ indexConfig: {{ "
            f"`vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: 'cosine' "
            f"}}}}"
        ),
        "CREATE INDEX known_issue_ticket_count IF NOT EXISTS FOR (k:KnownIssue) ON (k.ticket_count)",
    ]


class Neo4jClient:
    """Thin wrapper around the Neo4j driver for use as a context manager."""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self._db = config.NEO4J_DATABASE

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._driver.close()

    # ── Connectivity ──────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        try:
            self._driver.verify_connectivity()
            log.info("Neo4j connection OK  (%s)", config.NEO4J_URI)
            return True
        except Exception as exc:
            log.error("Neo4j connection failed: %s", exc)
            return False

    # ── Query helpers ─────────────────────────────────────────────────────────

    def run(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Execute a single Cypher statement and return rows as dicts."""
        with self._driver.session(database=self._db) as session:
            result = session.run(query, parameters or {})
            return result.data()

    # ── Schema ────────────────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """Apply all constraint and index DDL statements idempotently."""
        for stmt in _build_schema_statements():
            try:
                self.run(stmt)
            except Exception as exc:
                log.warning("Schema statement warning: %s", exc)
        log.info("Schema setup complete")

    def reset_vector_index(self) -> None:
        """Drop and recreate chunk_embedding at current EMBED_DIM — use when switching embedding models."""
        dim = _embed_dim()
        log.info("Dropping chunk_embedding index (will recreate at %d dims)", dim)
        try:
            self.run("DROP INDEX chunk_embedding IF EXISTS")
        except Exception as exc:
            log.warning("Could not drop chunk_embedding: %s", exc)
        create = (
            "CREATE VECTOR INDEX chunk_embedding "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{ indexConfig: {{ "
            f"`vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: 'cosine' "
            f"}}}}"
        )
        self.run(create)
        log.info("chunk_embedding recreated at %d dims", dim)
