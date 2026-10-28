"""Shared retrieval helpers — ANN and BM25 queries used by multiple nodes."""
from __future__ import annotations

import config
from agents.state import ScoredChunk
from graph.neo4j_client import Neo4jClient

_CYPHER_ANN = """\
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
YIELD node AS c, score
MATCH (c)-[:PART_OF]->(d:Document)
OPTIONAL MATCH (d)-[:HAS_METADATA]->(m:Metadata)
WITH c, d, score, 
     collect({key: m.key, value: m.value}) as node_metadata,
     properties(d) as doc_properties
RETURN 
    c.id AS id,
    c.text AS text,
    c.chunk_type AS chunk_type,
    d.id AS ticket_id,
    score,
    node_metadata,
    doc_properties
ORDER BY score DESC
LIMIT $top_k"""

def run_ann(
    client: Neo4jClient,
    embedding: list[float],
    top_k: int,
    meta_filter: dict[str, str] | None = None,
) -> list[ScoredChunk]:
    fetch_limit = top_k * 5 if meta_filter else top_k
    rows = client.run(_CYPHER_ANN, {"embedding": embedding, "top_k": fetch_limit})

    hard_filter, soft_filter = split_hard_soft(meta_filter or {})

    kept: list[ScoredChunk] = []
    fallback: list[ScoredChunk] = []

    for r in rows:
        metadata = _merge_metadata(r.get("node_metadata", []), r.get("doc_properties", {}))
        sc = ScoredChunk(
            id=r["id"],
            text=r["text"],
            score=float(r["score"]),
            chunk_type=r.get("chunk_type", ""),
            ticket_id=r.get("ticket_id", ""),
            metadata=metadata,
        )
        if hard_filter:
            fallback.append(sc)
            if not passes_metadata_filter(metadata, hard_filter):
                continue
        if soft_filter:
            sc.score *= soft_metadata_boost(metadata, soft_filter)
        kept.append(sc)

    pool = kept
    if hard_filter and not kept and fallback:
        pool = fallback

    pool.sort(key=lambda c: c.score, reverse=True)
    return pool[:top_k]


_CYPHER_BM25 = """\
CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
YIELD node AS chunk, score
MATCH (chunk)-[:PART_OF]->(doc:Document)
OPTIONAL MATCH (doc)-[:HAS_METADATA]->(m:Metadata)
WITH chunk, doc, score,
     collect({key: m.key, value: m.value}) AS node_metadata,
     properties(doc) AS doc_properties
RETURN chunk.id AS id, chunk.text AS text,
       chunk.chunk_type AS chunk_type, doc.id AS ticket_id, score,
       node_metadata, doc_properties
ORDER BY score DESC"""


def run_bm25(client: Neo4jClient, query: str, top_k: int) -> list[ScoredChunk]:
    safe_q = query.replace('"', '\\"')
    rows = client.run(_CYPHER_BM25, {"query": safe_q, "top_k": top_k})
    return [
        ScoredChunk(
            id=r["id"],
            text=r["text"],
            score=float(r["score"]),
            chunk_type=r.get("chunk_type", ""),
            ticket_id=r.get("ticket_id", ""),
            metadata=_merge_metadata(r.get("node_metadata", []), r.get("doc_properties", {})),
        )
        for r in rows
    ]


def _merge_metadata(node_metadata: list, doc_properties: dict) -> dict:
    """Merge Metadata nodes and Document properties; node_metadata wins on conflict."""
    PROPERTY_FIELDS = [
        'title', 'created', 'modified', 'word_count', 'char_count',
        'section_count', 'has_steps', 'content_hash', 'url_slug',
        'source_id', 'source_url', 'doc_type', 'source_system',
        'created_at', 'updated_at', 'status'
    ]
    
    metadata = {}
    for key in PROPERTY_FIELDS:
        if key in doc_properties and doc_properties[key] is not None:
            metadata[key] = doc_properties[key]
    for item in node_metadata:
        if item and 'key' in item and 'value' in item:
            k = item['key']
            v = item['value']
            if k is not None and v is not None and str(k).strip() != "":
                metadata[k] = v
    
    return metadata


_SOFT_FILTER_KEYS = frozenset({"status", "priority", "channel", "language", "severity"})


def _key_matches(metadata: dict, key: str, allowed_values) -> bool | None:
    """None = field absent, True = matched, False = field present but no match."""
    if not isinstance(allowed_values, list):
        allowed_values = [allowed_values]
    allowed_lower = {str(v).strip().lower() for v in allowed_values}

    actual_raw = metadata.get(key)
    if actual_raw is None or actual_raw == "":
        return None

    if isinstance(actual_raw, list):
        actual_lower = {str(v).strip().lower() for v in actual_raw if v}
        return bool(actual_lower & allowed_lower)
    return str(actual_raw).strip().lower() in allowed_lower


def passes_metadata_filter(metadata: dict, filters: dict) -> bool:
    """AND across filter keys, OR across values per key; absent soft keys pass, absent hard keys fail."""
    if not filters:
        return True

    for key, allowed_values in filters.items():
        res = _key_matches(metadata, key, allowed_values)
        if res is None:
            if key in _SOFT_FILTER_KEYS:
                continue
            return False
        if not res:
            return False

    return True


def split_hard_soft(filters: dict) -> tuple[dict, dict]:
    """Partition filters into hard (gate retrieval) and soft (boost ranking) by config.METADATA_HARD_KEYS."""
    hard = {k: v for k, v in filters.items() if k in config.METADATA_HARD_KEYS}
    soft = {k: v for k, v in filters.items() if k not in config.METADATA_HARD_KEYS}
    return hard, soft


def soft_metadata_boost(metadata: dict, soft_filters: dict) -> float:
    """Multiplicative boost scaled by fraction of matched soft keys; 1.0 when nothing matches."""
    if not soft_filters:
        return 1.0
    matched = sum(
        1 for k, v in soft_filters.items() if _key_matches(metadata, k, v) is True
    )
    frac = matched / len(soft_filters)
    return 1.0 + config.METADATA_BOOST_WEIGHT * frac
