"""Single-ticket ETL ingestion into Neo4j.

Takes a ticket dict from either the real ticket system (MCP) or the dummy
ticket system (REST) and runs it through the same ETL pipeline used for
bulk ingestion, writing a Document + Chunks + Entities into the graph.

Field mapping
-------------
Real ticket system (MCP get_ticket):
  id              → doc id
  title           → title_field
  description     → question_field  (problem chunk)
  solution        → answer_field    (resolution chunk)
  ticket_type     → metadata
  impact          → metadata
  urgency         → metadata
  routing_primary → metadata

Dummy ticket system (REST api_get_ticket):
  ticket_id           → doc id
  title               → title_field
  problem_description → question_field  (problem chunk)
  summary             → answer_field    (resolution chunk)
  typ                 → metadata
  prioritaet          → metadata
  kategorie           → metadata
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure Prototype_v1.0 root is importable when this module is loaded from a2a/
_PROTO_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))


def _normalise(raw: dict) -> dict:
    """Produce a uniform row dict the ETL column mapping can consume."""
    row: dict = {}

    # ID — real system uses 'id' (int or str), dummy uses 'ticket_id'
    row["id"] = str(raw.get("id") or raw.get("ticket_id") or "")
    if not row["id"]:
        raise ValueError("ticket has neither 'id' nor 'ticket_id'")

    # Title
    row["title"] = str(raw.get("title") or "").strip()

    # Problem text — real system: 'description', dummy: 'problem_description'
    row["problem"] = str(
        raw.get("description") or raw.get("problem_description") or ""
    ).strip()

    # Resolution text — real system: 'solution' (may be HTML), dummy: 'summary'
    solution = str(raw.get("solution") or raw.get("summary") or "").strip()
    row["resolution"] = re.sub(r"<[^>]+>", "", solution).strip()

    # Metadata fields — accept both naming conventions
    row["ticket_type"]     = str(raw.get("ticket_type")     or raw.get("typ")       or "")
    row["priority"]        = str(raw.get("impact")          or raw.get("prioritaet") or raw.get("urgency") or "")
    row["routing_primary"] = str(raw.get("routing_primary") or raw.get("kategorie")  or "")

    return row


def _build_mapping():
    from etl.schema_detector import ColumnMapping  # noqa: PLC0415
    return ColumnMapping(
        archetype="ticket",
        id_field="id",
        question_field="problem",
        answer_field="resolution",
        title_field="title",
        body_fields=[],
        metadata_fields=["ticket_type", "priority", "routing_primary"],
        status_field=None,
        status_keep=[],
        ref_field=None,
    )


def ingest_ticket(raw: dict, *, skip_embed: bool = False) -> None:
    """Run a single ticket through the ETL pipeline into Neo4j.

    Args:
        raw:        Ticket dict from the ticket system (either field naming convention).
        skip_embed: Skip vector embedding (BM25 index only). Useful when the
                    embedding backend is unavailable.
    """
    from etl.chunker_v2 import make_chunks  # noqa: PLC0415
    from etl.embedder import embed_texts  # noqa: PLC0415
    from etl.metadata_builder import (  # noqa: PLC0415
        build_belongs_to_edges,
        build_concept_nodes,
        build_metadata_nodes,
    )
    from etl.ner_v2 import extract_entities, generate_surface_forms  # noqa: PLC0415
    from graph.neo4j_client import Neo4jClient  # noqa: PLC0415
    import config  # noqa: PLC0415

    row = _normalise(raw)
    ticket_id = row["id"]
    logger.info("ETL ingestion starting: ticket_id=%s", ticket_id)

    mapping = _build_mapping()
    datasource_id = "source_ticket_system"

    # ── In-memory structures ───────────────────────────────────────────────────
    chunks = make_chunks(row, mapping)
    if not chunks:
        logger.warning("ticket_id=%s produced no chunks — nothing to ingest", ticket_id)
        return

    doc_record = {
        "id":         ticket_id,
        "source_id":  datasource_id,
        "title":      row["title"] or ticket_id,
        "doc_type":   "ticket",
        "status":     "",
        "created_at": "",
        "updated_at": "",
    }

    meta_nodes   = build_metadata_nodes(ticket_id, row, mapping)
    entities     = extract_entities(row, mapping)
    entity_map   = {
        e.id: {"id": e.id, "name": e.name, "type": e.type,
               "source": e.source, "confidence": e.confidence}
        for e in entities
    }
    mentions = [
        {"chunk_id": c.id, "entity_id": e.id,
         "confidence": e.confidence, "method": e.source}
        for e in entities
        for c in chunks
    ]

    seen_sf: set[str] = set()
    surface_forms: list[dict] = []
    for entity in entities:
        for form in generate_surface_forms(entity):
            sf_id = f"sf_{entity.id}_{re.sub(r'[^a-z0-9]', '_', form)}"
            if sf_id not in seen_sf:
                seen_sf.add(sf_id)
                surface_forms.append({"id": sf_id, "form": form, "entity_id": entity.id})

    concept_map   = {c["id"]: c for c in build_concept_nodes(entities)}
    belongs_edges = build_belongs_to_edges(entities)

    # ── Embed ──────────────────────────────────────────────────────────────────
    if not skip_embed:
        vectors = embed_texts([c.embed_text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec

    # ── Write to Neo4j ─────────────────────────────────────────────────────────
    def _batches(lst, n=config.NEO4J_BATCH_SIZE):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    chunk_dicts = [
        {
            "id":             c.id,
            "text":           c.text,
            "chunk_type":     c.chunk_type,
            "chunk_index":    c.chunk_index,
            "ticket_id":      c.ticket_id,
            "context_prefix": c.context_prefix,
            "token_count":    c.token_count,
            "embedding":      c.embedding,
        }
        for c in chunks
    ]

    with Neo4jClient() as client:
        if not client.test_connection():
            logger.error("Neo4j not reachable — ticket_id=%s not ingested", ticket_id)
            return

        client.run(
            """
            MERGE (s:DataSource {id: $id})
            SET s.name = $name, s.type = 'ticket', s.schema_detected = true
            """,
            {"id": datasource_id, "name": "ticket-system"},
        )

        client.run(
            """
            MERGE (d:Document {id: $id})
            SET d.source_id  = $source_id,
                d.title      = $title,
                d.doc_type   = $doc_type,
                d.status     = $status,
                d.created_at = $created_at,
                d.updated_at = $updated_at
            WITH d
            MATCH (s:DataSource {id: $source_id})
            MERGE (d)-[:FROM_SOURCE]->(s)
            """,
            doc_record,
        )

        for batch in _batches(meta_nodes):
            client.run(
                """
                UNWIND $nodes AS m
                MERGE (meta:Metadata {id: m.id})
                SET meta.key = m.key, meta.value = m.value, meta.value_type = m.value_type
                WITH meta, m
                MATCH (d:Document {id: m.doc_id})
                MERGE (d)-[:HAS_METADATA]->(meta)
                """,
                {"nodes": batch},
            )

        for batch in _batches(chunk_dicts):
            client.run(
                """
                UNWIND $chunks AS c
                MERGE (chunk:Chunk {id: c.id})
                SET chunk.text           = c.text,
                    chunk.chunk_type     = c.chunk_type,
                    chunk.chunk_index    = c.chunk_index,
                    chunk.context_prefix = c.context_prefix,
                    chunk.token_count    = c.token_count,
                    chunk.embedding      = c.embedding
                WITH chunk, c
                MATCH (d:Document {id: c.ticket_id})
                MERGE (chunk)-[:PART_OF]->(d)
                """,
                {"chunks": batch},
            )

        # NEXT_CHUNK edges (problem → resolution)
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
        pairs = [
            {
                "src": a.id,
                "dst": b.id,
                "relation_type": "problem_resolution"
                if a.chunk_type == "problem" and b.chunk_type == "resolution"
                else "sequential",
            }
            for a, b in zip(sorted_chunks, sorted_chunks[1:])
        ]
        if pairs:
            client.run(
                """
                UNWIND $pairs AS p
                MATCH (a:Chunk {id: p.src})
                MATCH (b:Chunk {id: p.dst})
                MERGE (a)-[:NEXT_CHUNK {relation_type: p.relation_type}]->(b)
                """,
                {"pairs": pairs},
            )

        if entity_map:
            for batch in _batches(list(entity_map.values())):
                client.run(
                    """
                    UNWIND $entities AS e
                    MERGE (entity:Entity {id: e.id})
                    SET entity.name = e.name, entity.type = e.type,
                        entity.source = e.source, entity.confidence = e.confidence
                    """,
                    {"entities": batch},
                )
            for batch in _batches(mentions):
                client.run(
                    """
                    UNWIND $mentions AS m
                    MATCH (c:Chunk  {id: m.chunk_id})
                    MATCH (e:Entity {id: m.entity_id})
                    MERGE (c)-[:MENTIONS {confidence: m.confidence, method: m.method}]->(e)
                    """,
                    {"mentions": batch},
                )

        if surface_forms:
            for batch in _batches(surface_forms):
                client.run(
                    """
                    UNWIND $forms AS f
                    MERGE (s:SurfaceForm {id: f.id})
                    SET s.form = f.form
                    WITH s, f
                    MATCH (e:Entity {id: f.entity_id})
                    MERGE (s)-[:ALIASES]->(e)
                    """,
                    {"forms": batch},
                )

        if concept_map:
            for batch in _batches(list(concept_map.values())):
                client.run(
                    """
                    UNWIND $concepts AS c
                    MERGE (concept:Concept {id: c.id})
                    SET concept.label = c.label, concept.description = c.description
                    """,
                    {"concepts": batch},
                )
            for batch in _batches(belongs_edges):
                client.run(
                    """
                    UNWIND $edges AS e
                    MATCH (ent:Entity  {id: e.entity_id})
                    MATCH (con:Concept {id: e.concept_id})
                    MERGE (ent)-[:BELONGS_TO {confidence: e.confidence}]->(con)
                    """,
                    {"edges": batch},
                )

    logger.info(
        "ETL ingestion complete: ticket_id=%s  chunks=%d  entities=%d",
        ticket_id, len(chunks), len(entity_map),
    )
