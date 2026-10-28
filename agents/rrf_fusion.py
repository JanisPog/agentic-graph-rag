"""RRF fusion node — merges semantic, BM25, and graph ranked lists."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState, ScoredChunk

log = logging.getLogger(__name__)


def _rrf(ranked_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            # Preserve richer metadata when a chunk appears in multiple lists
            if chunk.id in chunks and chunks[chunk.id].metadata and not chunk.metadata:
                chunk.metadata = chunks[chunk.id].metadata
            chunks[chunk.id] = chunk
    fused = sorted(chunks.values(), key=lambda c: scores[c.id], reverse=True)
    # Attach fused score back onto each chunk
    for c in fused:
        c.score = scores[c.id]
    return fused


def rrf_fusion(state: AgentState) -> dict:
    semantic = state.get("semantic_hits") or []
    bm25     = state.get("bm25_hits") or []
    graph    = state.get("graph_hits") or []
    web      = state.get("web_hits") or []
    lists = [semantic, bm25, graph, web]

    fasttrack_hit = state.get("fasttrack_hit")
    if fasttrack_hit:
        seed_chunk = ScoredChunk(
            id=fasttrack_hit["id"],
            text=fasttrack_hit["text"],
            score=1.0,
            chunk_type=fasttrack_hit.get("chunk_type", ""),
        )
        lists.append([seed_chunk])
        log.info("RRF fusion → seeding fasttrack hit %s", fasttrack_hit["id"])

    fused = _rrf(lists, k=config.RRF_K)[:config.FUSION_POOL_SIZE]
    log.info("RRF fusion → %d unique chunks  (from semantic=%d  bm25=%d  graph=%d  web=%d)",
             len(fused), len(semantic), len(bm25), len(graph), len(web))
    return {
        "fused_hits":    fused,
        "semantic_hits": [],
        "bm25_hits":     [],
        "graph_hits":    [],
        "web_hits":      [],
    }
