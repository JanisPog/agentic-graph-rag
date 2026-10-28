"""Quick diagnostic — checks Neo4j state and LLM connectivity.

Usage:  py diagnose.py
"""
import json
import sys
import logging
logging.basicConfig(level=logging.WARNING)  # suppress noise

sys.path.insert(0, ".")
import config
from graph.neo4j_client import Neo4jClient

print("=" * 60)
print("DIAGNOSE — Prototype v1.0")
print("=" * 60)

# ── 1. Neo4j connectivity ──────────────────────────────────────────────────────
print("\n[1] Neo4j connectivity")
try:
    with Neo4jClient() as client:
        ok = client.test_connection()
        print(f"    Connected: {ok}")

        # Node counts
        rows = client.run("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt ORDER BY lbl")
        for r in rows:
            print(f"    {r['lbl']:15s}  {r['cnt']:,} nodes")

        # Check embeddings
        null_embed = client.run(
            "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS n"
        )[0]["n"]
        total_chunks = client.run("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
        print(f"\n    Chunks with NULL embedding: {null_embed:,} / {total_chunks:,}")
        if null_embed == total_chunks:
            print("    ⚠  ALL embeddings are NULL — semantic retrieval will return 0 hits.")
            print("       Fix: py etl/ingest.py --clear --limit 1000")
        elif null_embed > 0:
            print(f"    ⚠  {null_embed:,} chunks missing embeddings.")

        # Sample a chunk
        sample = client.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text LIMIT 1")
        if sample:
            print(f"\n    Sample chunk id : {sample[0]['id']}")
            print(f"    Sample chunk text: {sample[0]['text'][:120]!r}")
except Exception as exc:
    print(f"    ERROR: {exc}")

# ── 2. Ollama embed ────────────────────────────────────────────────────────────
print(f"\n[2] Ollama embed  model={config.OLLAMA_EMBED_MODEL}")
try:
    from etl.embedder import embed_texts
    vec = embed_texts(["test login issue"])
    print(f"    Embedding dim : {len(vec[0])}")
    print(f"    First 5 values: {vec[0][:5]}")
    if len(vec[0]) != config.EMBED_DIM:
        print(f"    ⚠  Expected {config.EMBED_DIM} dims, got {len(vec[0])}.")
        print(f"       EMBED_DIM={config.EMBED_DIM} but model returned {len(vec[0])} dims.")
        print("       Either update EMBED_DIM in .env or reset the vector index.")
except Exception as exc:
    print(f"    ERROR: {exc}")

# ── 3. Ollama chat ─────────────────────────────────────────────────────────────
print(f"\n[3] Ollama chat   model={config.OLLAMA_CHAT_MODEL}")
try:
    from agents.llm import chat_json
    result = chat_json('Return JSON only: {"status": "ok", "model": "working"}')
    print(f"    Response: {result}")
except Exception as exc:
    print(f"    ERROR: {exc}")

# ── 4. BM25 spot-check ────────────────────────────────────────────────────────
print("\n[4] BM25 spot-check (query: 'login')")
try:
    with Neo4jClient() as client:
        rows = client.run(
            "CALL db.index.fulltext.queryNodes('chunk_fulltext', 'login', {limit: 3}) "
            "YIELD node, score RETURN node.id AS id, score, node.text AS text"
        )
        if rows:
            for r in rows:
                print(f"    id={r['id']}  score={r['score']:.3f}  text={r['text'][:80]!r}")
        else:
            print("    No results — check chunk_fulltext index exists.")
except Exception as exc:
    print(f"    ERROR: {exc}")

print("\n" + "=" * 60)
