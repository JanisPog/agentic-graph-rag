"""BM25 keyword retrieval node — uses Neo4j full-text (Lucene) index."""
from __future__ import annotations
import logging

import config
from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk
from agents.retrieval_utils import (
    passes_metadata_filter,
    run_bm25,
    split_hard_soft,
    soft_metadata_boost,
)
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)


def _normalise(hits: list[ScoredChunk]) -> list[ScoredChunk]:
    """Min-max normalise scores to [0, 1]."""
    if not hits:
        return hits
    lo = min(h.score for h in hits)
    hi = max(h.score for h in hits)
    if hi == lo:
        for h in hits:
            h.score = 1.0
    else:
        for h in hits:
            h.score = (h.score - lo) / (hi - lo)
    return hits


def bm25_retriever(state: AgentState) -> dict:
    queries = state.get("expanded_queries") or [state["original_query"]]
    meta_filter: dict = (
        (state.get("metadata_filters") or {})
        if enabled(state, "metadata_filter") else {}
    )
    
    hard_filter, soft_filter = split_hard_soft(meta_filter)

    best: dict[str, ScoredChunk] = {}
    fallback: dict[str, ScoredChunk] = {}  # safety net if hard filter removes everything
    raw_hit_count = 0

    fetch_limit = config.RETRIEVAL_TOP_K * 5 if meta_filter else config.RETRIEVAL_TOP_K

    with Neo4jClient() as client:
        for q in queries:
            try:
                hits = run_bm25(client, q, fetch_limit)
            except Exception as exc:
                log.warning("BM25 query failed for %r: %s", q, exc)
                continue
            for sc in hits:
                raw_hit_count += 1
                if hard_filter and (sc.id not in fallback or sc.score > fallback[sc.id].score):
                    fallback[sc.id] = sc
                if hard_filter and not passes_metadata_filter(sc.metadata, hard_filter):
                    continue
                if soft_filter:
                    sc.score *= soft_metadata_boost(sc.metadata, soft_filter)
                if sc.id not in best or sc.score > best[sc.id].score:
                    best[sc.id] = sc

    pool = best
    used_fallback = False
    if hard_filter and not best and fallback:
        pool = fallback
        used_fallback = True
        log.warning(
            "BM25 retriever: hard filter %s removed all %d raw hits — "
            "falling back to unfiltered results",
            hard_filter, raw_hit_count,
        )

    hits = _normalise(sorted(pool.values(), key=lambda c: c.score, reverse=True)[:config.RETRIEVAL_TOP_K])

    if meta_filter:
        log.info(
            "BM25 retriever filter stats: %d raw hits → %d hits "
            "(hard=%s soft=%s fallback=%s)",
            raw_hit_count, len(hits), hard_filter or {}, soft_filter or {}, used_fallback,
        )
            
    if hits:
        log.info("BM25 retriever → %d hits  (top score: %.4f  id: %s)",
                 len(hits), hits[0].score, hits[0].id)
        sample_meta = hits[0].metadata
        log.info("[BM25] Top hit metadata keys: %s", list(sample_meta.keys()) if sample_meta else [])
    else:
        log.info("BM25 retriever → 0 hits")
    return {"bm25_hits": hits}
