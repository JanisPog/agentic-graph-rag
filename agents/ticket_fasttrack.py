"""Ticket fast-track node — seeds a matching resolution chunk into RRF fusion before full retrieval."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState
from agents.crag_grader import _judge          # reuse the existing LLM judge
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)

_CYPHER_ANN_PROBLEM = """\
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
YIELD node AS chunk, score
WHERE chunk.chunk_type IN ['problem', 'question']
MATCH (chunk)-[:PART_OF]->(doc:Document)
RETURN chunk.id AS id, chunk.text AS text,
       chunk.chunk_type AS chunk_type, doc.id AS ticket_id, score
ORDER BY score DESC
LIMIT 1"""

_CYPHER_RESOLUTION = """\
MATCH (problem:Chunk {id: $chunk_id})-[:NEXT_CHUNK]->(resolution:Chunk)
RETURN resolution.id AS id, resolution.text AS text,
       resolution.chunk_type AS chunk_type"""


def _fetch_resolution(client: Neo4jClient, problem_id: str) -> dict | None:
    rows = client.run(_CYPHER_RESOLUTION, {"chunk_id": problem_id})
    if not rows:
        return None
    r = rows[0]
    return {"id": r["id"], "text": r["text"], "chunk_type": r.get("chunk_type", "")}


def ticket_fasttrack(state: AgentState) -> dict:
    query = state.get("original_query", "")
    if not query:
        return {"fasttrack_hit": None, "fasttrack_score": 0.0}

    from etl.embedder import embed_texts  # noqa: PLC0415
    vec = embed_texts([query])[0]

    with Neo4jClient() as client:
        rows = client.run(_CYPHER_ANN_PROBLEM, {"embedding": vec, "top_k": 5})

    if not rows:
        log.info("Ticket fast-track → no problem/question chunks found")
        return {"fasttrack_hit": None, "fasttrack_score": 0.0}

    top = rows[0]
    similarity = float(top["score"])

    if similarity < config.FASTTRACK_SIMILARITY_THRESH:
        log.info("Ticket fast-track → top similarity %.4f below threshold %.2f — skipping",
                 similarity, config.FASTTRACK_SIMILARITY_THRESH)
        return {"fasttrack_hit": None, "fasttrack_score": similarity}

    with Neo4jClient() as client:
        resolution = _fetch_resolution(client, top["id"])

    if not resolution:
        log.info("Ticket fast-track → problem chunk %s has no paired resolution", top["id"])
        return {"fasttrack_hit": None, "fasttrack_score": similarity}

    recall, relevant_ids, missing_hint = _judge(query, f"[1] {resolution['text']}")
    precision = 1.0 if (relevant_ids and 1 in relevant_ids) else 0.0
    combined = max(recall, 0.7 * recall + 0.3 * precision)

    log.info(
        "Ticket fast-track → similarity=%.4f  recall=%.2f  precision=%.2f  combined=%.2f  "
        "problem_id=%s  resolution_id=%s",
        similarity, recall, precision, combined, top["id"], resolution["id"],
    )

    if combined >= config.FASTTRACK_SEED_THRESH:
        log.info("Ticket fast-track → SEED into fusion (combined=%.2f)", combined)
        return {
            "fasttrack_hit":   resolution,
            "fasttrack_score": similarity,
            # context_quality NOT set here — grader will re-evaluate after full retrieval.
        }

    log.info("Ticket fast-track → low quality (combined=%.2f < %.2f) — discarding hit",
             combined, config.FASTTRACK_SEED_THRESH)
    return {"fasttrack_hit": None, "fasttrack_score": similarity}
