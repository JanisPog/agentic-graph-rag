"""Ticket Reuse Gate — KnownIssue fast-path and similar-ticket boost before the RAG pipeline."""
from __future__ import annotations

import logging
from typing import Any

import config
from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk
from agents.crag_grader import _judge as _crag_judge
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)

_CYPHER_RESOLUTION_BATCH = """\
MATCH (c:Chunk)-[:PART_OF]->(d:Document)
WHERE d.id IN $ticket_ids
  AND c.chunk_type    = 'resolution'
  AND d.source_system = 'customer_support'
RETURN c.text      AS resolution_text,
       d.id        AS ticket_id,
       d.status    AS status
"""

_CYPHER_KNOWN_ISSUE_ANN = """\
CALL db.index.vector.queryNodes('known_issue_embedding', 5, $query_vector)
YIELD node AS k, score
WHERE score >= $threshold
RETURN k.resolution_notes AS resolution_text,
       k.id               AS ticket_id,
       score
"""


def _crag_score(query: str, resolution_text: str) -> float:
    """CRAG combined score: max(recall, 0.7·recall + 0.3·precision)."""
    if not resolution_text or not resolution_text.strip():
        return 0.0
    try:
        recall, relevant_ids, _ = _crag_judge(query, resolution_text)
        precision = 1.0 if (relevant_ids and 1 in relevant_ids) else 0.0
        return max(recall, 0.7 * recall + 0.3 * precision)
    except Exception:
        log.warning("ticket_reuse_gate: CRAG judge failed — scoring 0.0")
        return 0.0


def _fetch_resolutions(ticket_ids: list[str]) -> dict[str, str]:
    """Fetch answer chunks keyed by ticket_id.  Returns {} on any DB error."""
    if not ticket_ids:
        return {}
    try:
        with Neo4jClient() as client:
            rows = client.run(_CYPHER_RESOLUTION_BATCH, {"ticket_ids": ticket_ids})
        return {
            r["ticket_id"]: r["resolution_text"]
            for r in rows
            if r.get("ticket_id") and r.get("resolution_text")
        }
    except Exception:
        log.warning("ticket_reuse_gate: resolution fetch failed — skipping batch")
        return {}


def _fetch_known_issue_candidates(query_vector: list[float]) -> list[dict[str, Any]]:
    """ANN over KnownIssue centroids.  Returns [] on any DB error."""
    try:
        with Neo4jClient() as client:
            rows = client.run(
                _CYPHER_KNOWN_ISSUE_ANN,
                {
                    "query_vector": query_vector,
                    "threshold":    config.TICKET_REUSE_THRESHOLD_HIGH,
                },
            )
        return [
            {
                "ticket_id":       r["ticket_id"],
                "resolution_text": r.get("resolution_text") or "",
                "score":           float(r["score"]),
                "source":          "known_issue",
            }
            for r in rows
            if r.get("ticket_id") and float(r.get("score", 0)) >= config.TICKET_REUSE_THRESHOLD_HIGH
        ]
    except Exception:
        log.warning("ticket_reuse_gate: KnownIssue ANN failed — skipping centroids")
        return []


def ticket_reuse_gate(state: AgentState) -> dict:
    """Fast-path gate — checks known tickets/issues before full RAG."""
    if not enabled(state, "reuse_gate"):
        log.info("Ticket reuse gate ablated — normal pipeline path")
        return {"reuse_answer": None, "reuse_ticket_id": None, "reuse_candidates": []}

    query: str = state.get("original_query") or ""

    raw_similar: list[dict] = state.get("similar_tickets") or []
    high_similar = [
        t for t in raw_similar
        if float(t.get("score", 0)) >= config.TICKET_REUSE_THRESHOLD_HIGH
    ]

    ref_ticket_ids: list[str] = []
    for mem in (state.get("ticket_ref_memories") or []):
        try:
            tid = str(mem.get("metadata", {}).get("ticket_id") or "").strip()
            if tid:
                ref_ticket_ids.append(tid)
        except Exception:
            log.warning("ticket_reuse_gate: malformed ticket_ref_memory entry — skipping")

    ref_ticket_ids = list(dict.fromkeys(ref_ticket_ids))  # deduplicate, preserve order

    similar_ids = [t["ticket_id"] for t in high_similar]
    all_ticket_ids = list(dict.fromkeys(similar_ids + ref_ticket_ids))

    resolution_map = _fetch_resolutions(all_ticket_ids)

    ticket_candidates: list[dict[str, Any]] = []
    for ticket in high_similar:
        tid = ticket["ticket_id"]
        text = resolution_map.get(tid, "")
        if text:
            ticket_candidates.append({
                "ticket_id":       tid,
                "resolution_text": text,
                "score":           float(ticket["score"]),
                "source":          "similar_ticket",
            })

    for tid in ref_ticket_ids:
        if tid not in similar_ids:  # avoid duplicates
            text = resolution_map.get(tid, "")
            if text:
                ticket_candidates.append({
                    "ticket_id":       tid,
                    "resolution_text": text,
                    "score":           config.TICKET_REUSE_THRESHOLD_HIGH,
                    "source":          "ticket_ref_memory",
                })

    ki_candidates: list[dict[str, Any]] = []
    if query.strip():
        try:
            from etl.embedder import embed_texts  # noqa: PLC0415
            vecs = embed_texts([query])
            if vecs and vecs[0]:
                ki_candidates = _fetch_known_issue_candidates(vecs[0])
        except Exception:
            log.warning("ticket_reuse_gate: query embedding failed — skipping KI centroids")

    all_candidates = ki_candidates + ticket_candidates

    if not all_candidates:
        log.info("ticket_reuse_gate: no candidates above threshold — normal pipeline")
        return {"reuse_answer": None, "reuse_ticket_id": None, "reuse_candidates": []}

    log.info(
        "ticket_reuse_gate: %d candidates found  (ki=%d  tickets=%d)",
        len(all_candidates), len(ki_candidates), len(ticket_candidates),
    )

    fast_path_chunks: list[tuple[float, ScoredChunk]] = []  # (crag, chunk)
    reuse_candidates: list[dict] = []

    for candidate in all_candidates:
        resolution_text: str = candidate["resolution_text"]
        sim: float = float(candidate["score"])
        if not resolution_text.strip():
            continue

        if sim >= config.KI_FAST_PATH_SIM_THRESH:
            crag = _crag_score(query, resolution_text)
            log.debug(
                "ticket_reuse_gate: candidate=%s  source=%s  sim=%.3f  crag=%.3f",
                candidate["ticket_id"], candidate["source"], sim, crag,
            )
            if crag >= config.CRAG_SUFFICIENT_THRESH:
                fast_path_chunks.append((crag, ScoredChunk(
                    id=candidate["ticket_id"],
                    text=resolution_text,
                    score=sim,
                    chunk_type="resolution",
                    ticket_id=candidate["ticket_id"],
                    metadata={"source": candidate["source"]},
                )))
            elif sim >= config.TICKET_REUSE_THRESHOLD_MID:
                reuse_candidates.append({
                    "ticket_id":       candidate["ticket_id"],
                    "resolution_text": resolution_text,
                    "score":           crag,
                })
        elif sim >= config.TICKET_REUSE_THRESHOLD_MID:
            reuse_candidates.append({
                "ticket_id":       candidate["ticket_id"],
                "resolution_text": resolution_text,
                "score":           sim,
            })

    if fast_path_chunks:
        fast_path_chunks.sort(key=lambda x: x[0], reverse=True)
        chunks = [c for _, c in fast_path_chunks]
        best_crag = fast_path_chunks[0][0]
        log.info(
            "ticket_reuse_gate: KI-FAST-PATH  %d chunk(s)  best_crag=%.3f  ids=%s → inspect_node",
            len(chunks), best_crag, [c.id for c in chunks],
        )
        return {
            "reranked_hits":     chunks,
            "reuse_ticket_id":   chunks[0].id,
            "reuse_candidates":  [],
            "ki_fast_path":      True,
            "context_quality":   best_crag,
            "crag_verdict":      "sufficient",
            "crag_missing_hint": "",
        }

    if reuse_candidates:
        log.info(
            "ticket_reuse_gate: %d boost candidates — normal pipeline",
            len(reuse_candidates),
        )
    else:
        log.info("ticket_reuse_gate: no candidates above fast-path threshold — normal pipeline")

    return {
        "reuse_answer":     None,
        "reuse_ticket_id":  None,
        "reuse_candidates": reuse_candidates,
        "ki_fast_path":     False,
    }
