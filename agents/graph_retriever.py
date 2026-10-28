"""Graph retrieval node — Personalized PageRank over a KNN-seeded local subgraph."""
from __future__ import annotations
import logging
import re
from collections import defaultdict

import numpy as np

import config
from agents.state import AgentState, ScoredChunk
from etl.ner_v2 import _extract_regex
from agents.retrieval_utils import (
    _merge_metadata,
    passes_metadata_filter,
    split_hard_soft,
    soft_metadata_boost,
)
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)

_ENTITY_SEED_SCORE = 0.60  # fixed score for entity seeds lacking a KNN cosine value

_CYPHER_ENTITY_SEEDS = """\
UNWIND $forms AS form
MATCH (s:SurfaceForm {form: form})-[:ALIASES]->(e:Entity)
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (c)-[:PART_OF]->(doc:Document)
RETURN DISTINCT
    c.id         AS chunk_id,
    c.text       AS chunk_text,
    c.chunk_type AS chunk_type,
    doc.id       AS ticket_id,
    e.name       AS entity_name,
    s.form       AS form
"""

_FORM_STOPWORDS = {
    "with", "from", "your", "this", "that", "have", "what", "when", "where",
    "which", "while", "about", "into", "their", "there", "would", "could",
    "should", "wix", "google", "site", "page", "https", "http", "www", "com",
    "click", "select", "settings", "want", "need", "help", "using", "able",
}


def _extract_query_forms(queries: list[str]) -> list[str]:
    """Extract regex entities, length-gated unigrams, and bigrams for SurfaceForm lookup."""
    combined = " ".join(queries)
    forms: set[str] = set()

    for entity in _extract_regex(combined):
        forms.add(entity.name.lower())

    raw_tokens = [t for t in re.split(r"[^a-z0-9]+", combined.lower()) if t]

    for t in raw_tokens:
        if len(t) >= config.GRAPH_MIN_UNIGRAM_LEN and t not in _FORM_STOPWORDS:
            forms.add(t)

    for a, b in zip(raw_tokens, raw_tokens[1:]):
        forms.add(f"{a} {b}")

    log.info("Query surface forms → %d candidates for SurfaceForm lookup", len(forms))
    return list(forms)


_EDGE_WEIGHT = {
    "NEXT_CHUNK": 0.9,
    "ENTITY":     0.5,
}
_RELATED_TO_FALLBACK = 0.7  # fallback when rel.score is NULL
_PPR_DAMPING    = 0.75
_PPR_ITERATIONS = 20

_CYPHER_SUBGRAPH = """\
MATCH (seed:Chunk) WHERE seed.id IN $seed_ids

// Collect all reachable neighbours within 2 hops across all edge types
CALL (seed) {
    MATCH (seed)-[r:NEXT_CHUNK|RELATED_TO*1..2]-(nbr:Chunk)
    RETURN nbr, r, 'structural' AS path_type
    UNION
    MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(nbr:Chunk)
    WHERE nbr.id <> seed.id
    RETURN nbr, null AS r, 'entity' AS path_type
}

MATCH (nbr)-[:PART_OF]->(doc:Document)
WITH seed, nbr, r, path_type, doc

// For variable-length paths, r is a list — take the last relationship
WITH seed, nbr, doc, path_type,
     CASE
       WHEN path_type = 'entity'      THEN 'ENTITY'
       WHEN type(last(r)) = 'RELATED_TO' THEN 'RELATED_TO'
       ELSE 'NEXT_CHUNK'
     END AS edge_type,
     CASE
       WHEN path_type = 'entity'      THEN $entity_weight
       WHEN type(last(r)) = 'RELATED_TO' THEN coalesce(last(r).score, $related_fallback)
       ELSE $next_chunk_weight
     END AS edge_weight

RETURN seed.id  AS src,
       nbr.id   AS dst,
       nbr.text AS dst_text,
       nbr.chunk_type AS dst_chunk_type,
       doc.id   AS dst_ticket_id,
       edge_type,
       edge_weight
"""

_CYPHER_CHUNK_METADATA = """\
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {id: cid})-[:PART_OF]->(d:Document)
OPTIONAL MATCH (d)-[:HAS_METADATA]->(m:Metadata)
WITH c, d, collect({key: m.key, value: m.value}) AS node_metadata,
     properties(d) AS doc_properties
RETURN c.id AS chunk_id, node_metadata, doc_properties"""


def _run_ppr(
    seed_ids: list[str],
    seed_scores: dict[str, float],
    edges: list[dict],
    top_k: int,
    damping: float,
    iterations: int,
) -> list[tuple[str, float, str]]:
    """Return [(node_id, ppr_score, best_edge_type), ...] for non-seed nodes."""
    all_nodes: set[str] = set(seed_ids)
    for e in edges:
        all_nodes.add(e["src"])
        all_nodes.add(e["dst"])
    node_list = list(all_nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    if n == 0:
        return []

    A = np.zeros((n, n), dtype=np.float32)
    best_edge: dict[tuple[int, int], str] = {}
    for e in edges:
        i, j = idx[e["src"]], idx[e["dst"]]
        w = float(e["edge_weight"])
        A[j, i] = max(A[j, i], w)
        A[i, j] = max(A[i, j], w)  # undirected
        best_edge[(j, i)] = e["edge_type"]
        best_edge[(i, j)] = e["edge_type"]

    col_sums = A.sum(axis=0)
    dangling = col_sums == 0
    col_sums[dangling] = 1.0
    A = A / col_sums

    seed_total = sum(seed_scores.get(s, 0.0) for s in seed_ids)
    personalization = np.zeros(n, dtype=np.float32)
    for s in seed_ids:
        personalization[idx[s]] = seed_scores.get(s, 0.0)
    if seed_total > 0:
        personalization /= seed_total
    else:
        for s in seed_ids:
            personalization[idx[s]] = 1.0 / len(seed_ids)

    pr = personalization.copy()
    for _ in range(iterations):
        pr = damping * A @ pr + (1.0 - damping) * personalization

    seed_set = set(seed_ids)
    results = []
    for node_id, i in idx.items():
        if node_id in seed_set:
            continue
        # Determine best contributing edge type for logging
        best_et = "unknown"
        best_w = -1.0
        for s in seed_ids:
            si = idx.get(s)
            if si is None:
                continue
            key = (i, si)
            if key in best_edge and A[i, si] > best_w:
                best_w = A[i, si]
                best_et = best_edge[key]
        results.append((node_id, float(pr[i]), best_et))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def graph_retriever(state: AgentState) -> dict:
    semantic_seeds = (state.get("semantic_hits") or [])[:config.GRAPH_SEED_K]
    base_queries: list[str] = state.get("sub_queries") or [state["original_query"]]
    query_forms = _extract_query_forms(base_queries)

    if not semantic_seeds and not query_forms:
        log.info("Graph retriever → skipped (no semantic seeds, no query surface forms)")
        return {"graph_hits": []}

    seed_ids: list[str] = [s.id for s in semantic_seeds]
    seed_scores: dict[str, float] = {s.id: s.score for s in semantic_seeds}

    with Neo4jClient() as client:
        if query_forms:
            entity_rows = client.run(_CYPHER_ENTITY_SEEDS, {"forms": query_forms})

            matched_entities: dict[str, str] = {}  # entity_name → matched form
            for row in entity_rows:
                matched_entities[row["entity_name"]] = row.get("form", "?")

            if matched_entities:
                log.info(
                    "SurfaceForm matches → %d entities resolved: %s",
                    len(matched_entities),
                    ", ".join(f"{e!r} (via {f!r})" for e, f in list(matched_entities.items())[:8]),
                )
            else:
                log.info(
                    "SurfaceForm matches → 0 entities resolved from %d candidate forms",
                    len(query_forms),
                )

            entity_budget = max(0, config.GRAPH_MAX_SEEDS - len(seed_ids))
            n_entity_seeds_added = 0
            n_dropped = 0
            for row in entity_rows:
                cid = row["chunk_id"]
                if cid in seed_scores:
                    continue
                if n_entity_seeds_added >= entity_budget:
                    n_dropped += 1
                    continue
                seed_ids.append(cid)
                seed_scores[cid] = _ENTITY_SEED_SCORE
                n_entity_seeds_added += 1

            if n_entity_seeds_added or n_dropped:
                log.info(
                    "Graph retriever → %d entity-anchored seeds added "
                    "(%d semantic seeds present, %d over GRAPH_MAX_SEEDS=%d cap dropped)",
                    n_entity_seeds_added, len(semantic_seeds), n_dropped, config.GRAPH_MAX_SEEDS,
                )

        if not seed_ids:
            log.info("Graph retriever → skipped (no seeds after entity lookup)")
            return {"graph_hits": []}

        rows = client.run(_CYPHER_SUBGRAPH, {
            "seed_ids":           seed_ids,
            "next_chunk_weight":  _EDGE_WEIGHT["NEXT_CHUNK"],
            "entity_weight":      _EDGE_WEIGHT["ENTITY"],
            "related_fallback":   _RELATED_TO_FALLBACK,
        })

    if not rows:
        log.info("Graph retriever → 0 subgraph edges from %d seeds", len(seed_ids))
        return {"graph_hits": []}

    node_meta: dict[str, dict] = {}
    for r in rows:
        node_meta[r["dst"]] = {
            "text":       r["dst_text"],
            "chunk_type": r.get("dst_chunk_type", ""),
            "ticket_id":  r.get("dst_ticket_id", ""),
        }

    ppr_results = _run_ppr(
        seed_ids=seed_ids,
        seed_scores=seed_scores,
        edges=rows,
        top_k=config.RETRIEVAL_TOP_K,
        damping=_PPR_DAMPING,
        iterations=_PPR_ITERATIONS,
    )

    entity_seed_set = set(seed_ids[len(semantic_seeds):])

    hits = []
    by_method: dict[str, int] = defaultdict(int)
    entity_seeded_hits = 0
    for node_id, ppr_score, edge_type in ppr_results:
        meta = node_meta.get(node_id)
        if meta is None:
            continue
        if edge_type == "ENTITY" or any(
            s in entity_seed_set for s in seed_ids
            if node_meta.get(node_id) is not None
        ):
            entity_seeded_hits += 1
        hits.append(ScoredChunk(
            id=node_id,
            text=meta["text"],
            score=ppr_score,
            chunk_type=meta["chunk_type"],
            ticket_id=meta["ticket_id"],
        ))
        by_method[edge_type] += 1

    if hits:
        hit_ids = [h.id for h in hits]
        with Neo4jClient() as meta_client:
            meta_rows = meta_client.run(_CYPHER_CHUNK_METADATA, {"chunk_ids": hit_ids})
        meta_by_id = {
            r["chunk_id"]: _merge_metadata(
                r.get("node_metadata", []),
                r.get("doc_properties", {}),
            )
            for r in meta_rows
        }
        meta_filter = state.get("metadata_filters") or {}
        hard_filter, soft_filter = split_hard_soft(meta_filter)

        filtered_hits = []
        fallback_hits = []
        for h in hits:
            h.metadata = meta_by_id.get(h.id, {})
            if hard_filter:
                fallback_hits.append(h)
                if not passes_metadata_filter(h.metadata, hard_filter):
                    continue
            if soft_filter:
                h.score *= soft_metadata_boost(h.metadata, soft_filter)
            filtered_hits.append(h)

        used_fallback = False
        if hard_filter and not filtered_hits and fallback_hits:
            filtered_hits = fallback_hits
            used_fallback = True
            log.warning(
                "Graph retriever: hard filter %s removed all %d raw hits — "
                "falling back to unfiltered results", hard_filter, len(hits),
            )

        if meta_filter:
            log.info(
                "Graph retriever filter stats: %d raw hits → %d hits "
                "(hard=%s soft=%s fallback=%s)",
                len(hits), len(filtered_hits), hard_filter or {}, soft_filter or {}, used_fallback,
            )

        filtered_hits.sort(key=lambda c: c.score, reverse=True)
        hits = filtered_hits

        if hits:
            sample_meta = hits[0].metadata
            log.info("[Graph] Top hit metadata keys: %s", list(sample_meta.keys()) if sample_meta else [])

    log.info(
        "Graph retriever (PPR) → %d hits from %d seeds (%d semantic + %d entity)  "
        "edges: %s  entity-influenced hits: %d  (damping=%.2f  iters=%d)",
        len(hits), len(seed_ids),
        len(semantic_seeds), len(seed_ids) - len(semantic_seeds),
        dict(by_method), entity_seeded_hits, _PPR_DAMPING, _PPR_ITERATIONS,
    )
    return {
        "graph_hits": hits[:config.RETRIEVAL_TOP_K],
        "graph_seeds_total":     len(seed_ids),
        "graph_entity_seeds":    len(seed_ids) - len(semantic_seeds),
        "graph_nodes_returned":  len(hits),
        "graph_entity_hits":     entity_seeded_hits,
    }
