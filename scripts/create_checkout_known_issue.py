"""Create a dedicated KnownIssue for the 3 checkout test tickets (T001/T002/T003).

Run AFTER deleting their wrong SAME_ISSUE edges:
    MATCH (d:Document)-[r:SAME_ISSUE]->(k:KnownIssue)
    WHERE d.id IN ['T001','T002','T003']
    DELETE r

Usage:
    .venv/bin/python scripts/create_checkout_known_issue.py
"""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
log = logging.getLogger(__name__)

from datetime import datetime, timezone
from graph.neo4j_client import Neo4jClient
from etl.embedder import embed_texts

TICKET_IDS = ["T001", "T002", "T003"]

_GET_CHUNKS = """\
MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id})
WHERE c.chunk_type = 'problem'
RETURN c.text AS text, c.embedding AS embedding
LIMIT 1
"""

_GET_RESOLUTION = """\
MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id})
WHERE c.chunk_type = 'resolution'
RETURN c.text AS text LIMIT 1
"""

_CREATE_KI = """\
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

_LINK = """\
MATCH (d:Document {id: $doc_id})
MATCH (k:KnownIssue {id: $ki_id})
MERGE (d)-[:SAME_ISSUE {similarity_score: 1.0}]->(k)
"""

def main():
    with Neo4jClient() as client:
        texts, embeddings, resolutions = [], [], []
        for tid in TICKET_IDS:
            rows = client.run(_GET_CHUNKS, {"doc_id": tid})
            if not rows:
                log.warning("No problem chunk for %s — skipping", tid)
                continue
            texts.append(rows[0]["text"])
            embeddings.append(rows[0]["embedding"])
            res = client.run(_GET_RESOLUTION, {"doc_id": tid})
            resolutions.append(res[0]["text"] if res else "")

    if not embeddings:
        log.error("No embeddings found — did you ingest the test tickets?")
        return

    # Centroid = mean of embeddings
    n = len(embeddings)
    centroid = [sum(e[i] for e in embeddings) / n for i in range(len(embeddings[0]))]

    summary = (
        "Customers are unable to complete checkout in Wix online stores. "
        "After entering payment details the transaction freezes with an endless loading spinner "
        "and orders never complete. Root cause is a conflict between a recently installed "
        "third-party app and the Wix Payments gateway."
    )
    resolution = resolutions[0] if resolutions else ""
    ki_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with Neo4jClient() as client:
        client.run(_CREATE_KI, {
            "id":               ki_id,
            "summary":          summary,
            "embedding":        centroid,
            "ticket_count":     n,
            "resolution_notes": resolution,
            "now":              now,
        })
        for tid in TICKET_IDS:
            client.run(_LINK, {"doc_id": tid, "ki_id": ki_id})

    log.info("KnownIssue created: id=%s  tickets=%d", ki_id, n)
    log.info("Summary: %s", summary)

if __name__ == "__main__":
    main()
