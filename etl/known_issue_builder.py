"""KnownIssue aggregation — clusters similar customer_support tickets into (:KnownIssue) nodes."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import config
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)

CLUSTER_THRESHOLD: float = 0.75
MIN_CLUSTER_SIZE:  int   = 3
_RECOMPUTE_EVERY: int = 5

_CYPHER_FIND_CLUSTER = """\
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
YIELD node AS c, score
MATCH (c)-[:PART_OF]->(d:Document)
WHERE c.chunk_type    = 'problem'
  AND d.source_system = 'customer_support'
  AND d.id            <> $current_doc_id
  AND score           >= $threshold
WITH d, max(score) AS best_score
OPTIONAL MATCH (d)-[:HAS_METADATA]->(m:Metadata {key: 'product'})
WITH d, best_score, m.value AS doc_product
WHERE $product IS NULL OR doc_product = $product
RETURN d.id AS ticket_id, best_score
ORDER BY best_score DESC
LIMIT $top_k
"""

_CYPHER_FIND_EXISTING_KI = """\
MATCH (d:Document)-[:SAME_ISSUE]->(k:KnownIssue)
WHERE d.id IN $ticket_ids
RETURN k.id AS ki_id, k.ticket_count AS ticket_count
LIMIT 1
"""

_CYPHER_GET_CLUSTER_TEXTS = """\
MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $ticket_id})
WHERE c.chunk_type = 'problem'
RETURN c.text AS text
LIMIT 1
"""

_CYPHER_GET_CLUSTER_EMBEDDINGS = """\
MATCH (c:Chunk)-[:PART_OF]->(d:Document)
WHERE d.id IN $ticket_ids
  AND c.chunk_type = 'problem'
RETURN d.id AS ticket_id, c.embedding AS embedding
"""

_CYPHER_CREATE_KI = """\
CREATE (k:KnownIssue {
    id:                $id,
    summary:           $summary,
    problem_embedding: $embedding,
    ticket_count:      $ticket_count,
    resolution_notes:  $resolution_notes,
    created_at:        $now,
    updated_at:        $now
})
"""

_CYPHER_UPDATE_KI = """\
MATCH (k:KnownIssue {id: $ki_id})
SET k.ticket_count = k.ticket_count + 1,
    k.updated_at   = $now
"""

_CYPHER_UPDATE_KI_FULL = """\
MATCH (k:KnownIssue {id: $ki_id})
SET k.ticket_count       = k.ticket_count + 1,
    k.summary            = $summary,
    k.problem_embedding  = $embedding,
    k.updated_at         = $now
"""

_CYPHER_ADD_SAME_ISSUE = """\
MATCH (d:Document  {id: $doc_id})
MATCH (k:KnownIssue {id: $ki_id})
MERGE (d)-[:SAME_ISSUE {similarity_score: $score}]->(k)
"""

_CYPHER_ADD_SAME_ISSUE_BATCH = """\
UNWIND $rows AS row
MATCH (d:Document  {id: row.doc_id})
MATCH (k:KnownIssue {id: $ki_id})
MERGE (d)-[:SAME_ISSUE {similarity_score: row.score}]->(k)
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    n = len(embeddings)
    dim = len(embeddings[0])
    result = [0.0] * dim
    for vec in embeddings:
        for i, v in enumerate(vec):
            result[i] += v
    return [x / n for x in result]


def _generate_summary(ticket_ids: list[str]) -> str:
    """LLM summary of the common issue across up to 5 cluster tickets."""
    texts: list[str] = []
    try:
        with Neo4jClient() as client:
            for tid in ticket_ids[:5]:
                rows = client.run(_CYPHER_GET_CLUSTER_TEXTS, {"ticket_id": tid})
                if rows and rows[0].get("text"):
                    texts.append(rows[0]["text"])
    except Exception:
        log.exception("KnownIssue summary: failed to fetch problem texts")

    if not texts:
        return "Multiple similar customer support issues (summary unavailable)."

    from agents.llm import chat  # noqa: PLC0415 — deferred to avoid circular import

    sample = "\n".join(f"- {t[:300]}" for t in texts)
    prompt = (
        "You are summarising a cluster of similar customer support tickets "
        "to create a reusable 'Known Issue' entry.\n\n"
        "The following are representative problem descriptions from the cluster:\n"
        f"{sample}\n\n"
        "Write a concise one-paragraph summary (2–4 sentences) that captures "
        "the common issue, who is affected, and what they cannot do. "
        "Do not include solutions. Plain text, no markdown."
    )
    try:
        return chat(prompt).strip()
    except Exception:
        log.exception("KnownIssue summary: LLM call failed — using fallback")
        return texts[0][:500] if texts else "Summary unavailable."


# ── Public API ─────────────────────────────────────────────────────────────────

def find_similar_ticket_cluster(
    doc_id: str,
    embedding: list[float],
    threshold: float = CLUSTER_THRESHOLD,
    min_cluster: int = MIN_CLUSTER_SIZE,
    product: str | None = None,
) -> list[tuple[str, float]]:
    """Return (ticket_id, score) pairs above threshold; [] if fewer than min_cluster found."""
    if not embedding:
        log.warning("KnownIssue cluster search: empty embedding for doc_id=%s", doc_id)
        return []

    top_k_search = max(50, min_cluster * 10)

    try:
        with Neo4jClient() as client:
            rows = client.run(
                _CYPHER_FIND_CLUSTER,
                {
                    "embedding":      embedding,
                    "current_doc_id": doc_id,
                    "threshold":      threshold,
                    "top_k":          top_k_search,
                    "product":        product,
                },
            )
    except Exception:
        log.exception("KnownIssue cluster search: Neo4j query failed for doc_id=%s", doc_id)
        return []

    results = [(r["ticket_id"], float(r["best_score"])) for r in rows if r.get("ticket_id")]

    if len(results) < min_cluster:
        log.debug(
            "KnownIssue cluster search: %d matches below min_cluster=%d for doc_id=%s",
            len(results), min_cluster, doc_id,
        )
        return []

    log.info(
        "KnownIssue cluster: %d similar tickets found for doc_id=%s  (top score: %.4f)",
        len(results), doc_id, results[0][1] if results else 0.0,
    )
    return results


def upsert_known_issue(
    cluster_ticket_ids: list[tuple[str, float]],
    new_ticket_id: str,
    resolution_notes: str = "",
) -> None:
    """Create or update a KnownIssue node; regenerates summary+centroid every _RECOMPUTE_EVERY attachments.
    """
    if not cluster_ticket_ids:
        return

    all_ids = [tid for tid, _ in cluster_ticket_ids]
    score_map = {tid: score for tid, score in cluster_ticket_ids}
    now = _now_iso()

    try:
        with Neo4jClient() as client:
            ki_rows = client.run(
                _CYPHER_FIND_EXISTING_KI,
                {"ticket_ids": all_ids},
            )

            if ki_rows:
                ki_id       = ki_rows[0]["ki_id"]
                ticket_count = int(ki_rows[0]["ticket_count"])
                new_count   = ticket_count + 1

                if new_count % _RECOMPUTE_EVERY == 0:
                    emb_rows = client.run(
                        _CYPHER_GET_CLUSTER_EMBEDDINGS,
                        {"ticket_ids": all_ids + [new_ticket_id]},
                    )
                    embeddings = [
                        r["embedding"] for r in emb_rows
                        if r.get("embedding")
                    ]
                    centroid_vec = _centroid(embeddings)
                    summary = _generate_summary(all_ids + [new_ticket_id])

                    client.run(
                        _CYPHER_UPDATE_KI_FULL,
                        {
                            "ki_id":     ki_id,
                            "summary":   summary,
                            "embedding": centroid_vec,
                            "now":       now,
                        },
                    )
                    log.info(
                        "KnownIssue %s: recomputed summary+centroid at ticket_count=%d",
                        ki_id, new_count,
                    )
                else:
                    client.run(_CYPHER_UPDATE_KI, {"ki_id": ki_id, "now": now})

                client.run(
                    _CYPHER_ADD_SAME_ISSUE,
                    {
                        "doc_id": new_ticket_id,
                        "ki_id":  ki_id,
                        "score":  1.0,  # exact match by definition — new ticket triggers the cluster
                    },
                )
                log.info(
                    "KnownIssue %s updated — ticket_count now %d, new ticket %s attached",
                    ki_id, new_count, new_ticket_id,
                )

            else:
                ki_id = str(uuid.uuid4())
                emb_rows = client.run(
                    _CYPHER_GET_CLUSTER_EMBEDDINGS,
                    {"ticket_ids": all_ids + [new_ticket_id]},
                )
                embeddings = [r["embedding"] for r in emb_rows if r.get("embedding")]
                centroid_vec = _centroid(embeddings)
                summary = _generate_summary(all_ids + [new_ticket_id])
                client.run(
                    _CYPHER_CREATE_KI,
                    {
                        "id":               ki_id,
                        "summary":          summary,
                        "embedding":        centroid_vec,
                        "ticket_count":     len(all_ids) + 1,
                        "resolution_notes": resolution_notes,
                        "now":              now,
                    },
                )

                edge_rows = [
                    {"doc_id": tid, "score": score_map.get(tid, CLUSTER_THRESHOLD)}
                    for tid in all_ids
                ]
                client.run(
                    _CYPHER_ADD_SAME_ISSUE_BATCH,
                    {"rows": edge_rows, "ki_id": ki_id},
                )
                client.run(
                    _CYPHER_ADD_SAME_ISSUE,
                    {"doc_id": new_ticket_id, "ki_id": ki_id, "score": 1.0},
                )

                log.info(
                    "KnownIssue %s created — %d tickets in cluster, summary: %r…",
                    ki_id, len(all_ids) + 1, summary[:80],
                )

    except Exception:
        log.exception(
            "upsert_known_issue failed for new_ticket_id=%s — skipping, ingest continues",
            new_ticket_id,
        )
