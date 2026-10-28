"""Semantic retrieval node — vector similarity search via Neo4j."""
from __future__ import annotations
import logging

import config
from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk
from agents.retrieval_utils import run_ann, _merge_metadata
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)


def semantic_retriever(state: AgentState) -> dict:
    meta_filter: dict = (
        (state.get("metadata_filters") or {})
        if enabled(state, "metadata_filter") else {}
    )

    # On retry use only the missing_hint — re-running the original queries returns the same hits.
    hint = state.get("crag_missing_hint", "")
    if hint:
        queries = [hint]
    else:
        queries = state.get("expanded_queries") or [state["original_query"]]

    from etl.embedder import embed_texts  # noqa: PLC0415
    vectors = embed_texts(queries)

    best: dict[str, ScoredChunk] = {}
    with Neo4jClient() as client:
        for vec in vectors:
            hits = run_ann(client, vec, config.RETRIEVAL_TOP_K, meta_filter or None)
            if meta_filter:
                log.debug("Semantic retriever (filtered by %s) → %d raw rows", meta_filter, len(hits))
            for sc in hits:
                if sc.metadata:
                    log.info(f"[Semantic] Chunk {sc.id[:12]}... has metadata: {list(sc.metadata.keys())}")
                if sc.id not in best or sc.score > best[sc.id].score:
                    best[sc.id] = sc

    result = sorted(best.values(), key=lambda c: c.score, reverse=True)[:config.RETRIEVAL_TOP_K]
    if result:
        log.info("Semantic retriever → %d hits  (top score: %.4f  id: %s)%s",
                 len(result), result[0].score, result[0].id,
                 f"  [filtered: {meta_filter}]" if meta_filter else "")
    else:
        log.info("Semantic retriever → 0 hits  (embeddings may be NULL — run ingest without --no-embed)")
    return {"semantic_hits": result}
