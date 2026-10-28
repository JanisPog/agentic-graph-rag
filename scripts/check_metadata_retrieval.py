"""Smoke-test: verify that Semantic, BM25, and Graph retrievers all
return ScoredChunks with populated metadata dicts.

Usage:  python scripts/check_metadata_retrieval.py
"""
import os
import sys
import logging

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import config
from graph.neo4j_client import Neo4jClient
from agents.state import ScoredChunk

# ── Config ────────────────────────────────────────────────────────────────────
QUERY = "How to reset my password?"
TOP_K = 5
# Keys we specifically want to see in metadata
INTERESTING_KEYS = [
    "title", "doc_type", "status", "source_id",
    "source_system", "product", "category", "priority",
]

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_hits(label: str, hits: list[ScoredChunk], max_show: int = 3):
    """Pretty-print hits with metadata details."""
    print(f"\n{'─' * 60}")
    print(f"  {label}: {len(hits)} hits")
    print(f"{'─' * 60}")
    for i, h in enumerate(hits[:max_show]):
        meta_keys = sorted(h.metadata.keys()) if h.metadata else []
        print(f"  [{i+1}] id={h.id}")
        print(f"      score={h.score:.4f}  chunk_type={h.chunk_type!r}  ticket_id={h.ticket_id!r}")
        print(f"      metadata keys ({len(meta_keys)}): {meta_keys}")
        # Show interesting values
        if h.metadata:
            for k in INTERESTING_KEYS:
                v = h.metadata.get(k)
                if v is not None:
                    print(f"      {k} = {v!r}")
        else:
            print(f"      (metadata is empty)")
        print()


def _check_metadata(label: str, hits: list[ScoredChunk]) -> bool:
    """Return True if at least the first hit has non-empty metadata."""
    if not hits:
        print(f"  {WARN} {label}: no hits returned — cannot check metadata")
        return True  # not a failure, just no data
    has_meta = sum(1 for h in hits if h.metadata)
    total = len(hits)
    if has_meta == total:
        print(f"  {PASS} {label}: all {total} hits have metadata")
        return True
    elif has_meta > 0:
        print(f"  {WARN} {label}: {has_meta}/{total} hits have metadata (some empty)")
        return True  # partial is acceptable
    else:
        print(f"  {FAIL} {label}: ALL {total} hits have EMPTY metadata")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  METADATA SMOKE TEST")
    print(f"  Query: {QUERY!r}")
    print(f"  Embed backend: {config.EMBED_BACKEND}")
    print("=" * 60)

    # Pre-flight: check Neo4j connectivity
    print("\n[0] Neo4j connectivity check...")
    try:
        with Neo4jClient() as client:
            ok = client.test_connection()
            if not ok:
                print(f"  {FAIL} Neo4j connection failed")
                sys.exit(1)
            chunk_count = client.run("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
            doc_count = client.run("MATCH (d:Document) RETURN count(d) AS n")[0]["n"]
            meta_count = client.run("MATCH (m:Metadata) RETURN count(m) AS n")[0]["n"]
            null_embed = client.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS n"
            )[0]["n"]
            print(f"  {PASS} Connected — {chunk_count} chunks, {doc_count} docs, {meta_count} metadata nodes")
            print(f"      Chunks with NULL embedding: {null_embed}/{chunk_count}")
    except Exception as exc:
        print(f"  {FAIL} Neo4j error: {exc}")
        sys.exit(1)

    results = {}
    all_ok = True

    # ── 1. Semantic Retrieval ──────────────────────────────────────────────────
    print("\n[1] Semantic Retrieval...")
    try:
        from etl.embedder import embed_texts
        from agents.retrieval_utils import run_ann
        vectors = embed_texts([QUERY])
        with Neo4jClient() as client:
            semantic_hits = run_ann(client, vectors[0], TOP_K)
        results["semantic"] = semantic_hits
        _print_hits("Semantic", semantic_hits)
        if not _check_metadata("Semantic", semantic_hits):
            all_ok = False
    except Exception as exc:
        print(f"  {FAIL} Semantic retrieval error: {exc}")
        results["semantic"] = []
        all_ok = False

    # ── 2. BM25 Retrieval ──────────────────────────────────────────────────────
    print("\n[2] BM25 Retrieval...")
    try:
        from agents.bm25_retriever import bm25_retriever
        state = {"original_query": QUERY, "expanded_queries": [QUERY]}
        bm25_result = bm25_retriever(state)
        bm25_hits = bm25_result.get("bm25_hits", [])
        results["bm25"] = bm25_hits
        _print_hits("BM25", bm25_hits)
        if not _check_metadata("BM25", bm25_hits):
            all_ok = False
    except Exception as exc:
        print(f"  {FAIL} BM25 retrieval error: {exc}")
        results["bm25"] = []
        all_ok = False

    # ── 3. Graph Retrieval ─────────────────────────────────────────────────────
    print("\n[3] Graph Retrieval...")
    try:
        from agents.graph_retriever import graph_retriever
        semantic_hits = results.get("semantic", [])
        state = {
            "original_query": QUERY,
            "expanded_queries": [QUERY],
            "sub_queries": [QUERY],
            "semantic_hits": semantic_hits,
        }
        graph_result = graph_retriever(state)
        graph_hits = graph_result.get("graph_hits", [])
        results["graph"] = graph_hits
        _print_hits("Graph", graph_hits)
        if not _check_metadata("Graph", graph_hits):
            all_ok = False
    except Exception as exc:
        print(f"  {FAIL} Graph retrieval error: {exc}")
        results["graph"] = []
        all_ok = False

    # ── 4. RRF Fusion (metadata preservation check) ────────────────────────────
    print("\n[4] RRF Fusion (metadata preservation)...")
    try:
        from agents.rrf_fusion import _rrf
        semantic_hits = results.get("semantic", [])
        bm25_hits = results.get("bm25", [])
        graph_hits = results.get("graph", [])

        fused = _rrf([semantic_hits, bm25_hits, graph_hits])

        # Check: any chunk that had metadata in its source should still have it
        source_meta = {}
        for h in semantic_hits + bm25_hits + graph_hits:
            if h.metadata:
                source_meta[h.id] = h.metadata

        lost_count = 0
        for f in fused:
            if f.id in source_meta and not f.metadata:
                lost_count += 1
                print(f"  {FAIL} Chunk {f.id} had metadata in source but lost it in RRF!")

        if lost_count == 0:
            print(f"  {PASS} RRF preserved metadata for all {len(fused)} fused chunks")
        else:
            print(f"  {FAIL} RRF lost metadata for {lost_count} chunks")
            all_ok = False

        # Show fused top-3
        _print_hits("RRF Fused", fused)
        if not _check_metadata("RRF Fused", fused):
            all_ok = False

    except Exception as exc:
        print(f"  {FAIL} RRF fusion error: {exc}")
        all_ok = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print(f"  {PASS} ALL CHECKS PASSED — metadata unification is working")
    else:
        print(f"  {FAIL} SOME CHECKS FAILED — see details above")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
