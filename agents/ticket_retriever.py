"""Ticket-level retriever — finds similar customer support tickets via vector search.

Searches question-chunks of customer_support documents and deduplicates to
ticket level (one result per Document, keeping the highest-scoring chunk).
Used by ticket_retrieval_node as the fourth parallel retriever arm.
"""
from __future__ import annotations
import logging

import config
from agents.state import AgentState, ScoredChunk
from graph.neo4j_client import Neo4jClient
from etl.embedder import embed_texts

log = logging.getLogger(__name__)

TICKET_RETRIEVER_TOP_K        = 10
TICKET_RETRIEVER_SCORE_THRESH = 0.60

_CYPHER_TICKET_ANN = """\
CALL db.index.vector.queryNodes('chunk_embedding', $top_k_oversample, $embedding)
YIELD node AS c, score
MATCH (c)-[:PART_OF]->(d:Document)
WHERE c.chunk_type    = 'problem'
  AND d.source_system = 'customer_support'
OPTIONAL MATCH (d)-[:HAS_METADATA]->(m:Metadata)
WITH c, d, score,
     collect({key: m.key, value: m.value}) AS node_metadata,
     properties(d) AS doc_props
RETURN
    c.id            AS chunk_id,
    c.text          AS text,
    d.id            AS doc_id,
    d.source_id     AS source_id,
    d.title         AS title,
    d.status        AS status,
    d.source_system AS source_system,
    score,
    node_metadata,
    doc_props
ORDER BY score DESC
"""


def search_similar_tickets(
    query: str,
    top_k: int = TICKET_RETRIEVER_TOP_K,
    threshold: float = TICKET_RETRIEVER_SCORE_THRESH,
    user_id: str | None = None,
) -> list[dict]:
    """ANN search over customer_support problem chunks, deduplicated to ticket level."""
    if not query or not query.strip():
        log.warning("Ticket retriever: empty query — returning []")
        return []

    try:
        vectors = embed_texts([query])
    except Exception:
        log.exception("Ticket retriever: embed_texts failed — returning []")
        return []

    if not vectors or not vectors[0]:
        log.warning("Ticket retriever: embedding returned empty vector — returning []")
        return []

    embedding = vectors[0]
    oversample = top_k * 5

    try:
        with Neo4jClient() as client:
            rows = client.run(
                _CYPHER_TICKET_ANN,
                {"embedding": embedding, "top_k_oversample": oversample},
            )
    except Exception:
        log.exception("Ticket retriever: Neo4j query failed — returning []")
        return []

    best: dict[str, dict] = {}
    for r in rows:
        score = float(r["score"])
        if score < threshold:
            continue
        doc_id = r["doc_id"]
        if doc_id not in best or score > best[doc_id]["score"]:
            best[doc_id] = {
                "ticket_id":     doc_id,
                "source_id":     r.get("source_id", ""),
                "title":         r.get("title", ""),
                "status":        r.get("status", ""),
                "source_system": r.get("source_system", ""),
                "score":         score,
            }

    result = sorted(best.values(), key=lambda t: t["score"], reverse=True)[:top_k]

    if result:
        log.info(
            "Ticket retriever → %d tickets  (top score: %.4f  ticket_id: %s)",
            len(result), result[0]["score"], result[0]["ticket_id"],
        )
    else:
        log.info(
            "Ticket retriever → 0 tickets  "
            "(no customer_support docs yet or all below threshold %.2f)",
            threshold,
        )

    return result


def ticket_retrieval_node(state: AgentState) -> dict:
    query = state.get("original_query", "")
    user_id = state.get("user_id")

    similar_tickets = search_similar_tickets(query, user_id=user_id)
    return {"similar_tickets": similar_tickets}
