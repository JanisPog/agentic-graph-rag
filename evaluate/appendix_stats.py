"""Appendix statistics generator for the paper.

Reads the live Neo4j graph and the WixQA expert-written eval set, and produces
the appendix-worthy retrieval-side statistics. No DB schema changes required,
no chat LLM calls — only the embedder is needed to vectorise queries.

Sections produced
-----------------
  1. Corpus descriptive statistics       (Cypher only)
       chunk count, chunk-length distribution, embedding-norm distribution,
       SAGE RELATED_TO degree, MENTIONS degree, SurfaceForm count per entity.

  2. Query–gold geometry                 (embedder + Cypher)
       For each eval query:
         cosine(query, every gold chunk)  → mean / max / median
         rank of the best gold chunk in a top-100 ANN sweep
         query token length, gold chunk count.

  3. Per-arm Recall@K                    (Cypher only)
       Semantic (run_ann) and BM25 (run_bm25) on the raw query.
       Hit-at-K for K in {1, 5, 10, 20, 50, 100}.

  4. Figures                             (matplotlib)
       Histograms, scatter plots, Recall@K curves saved as PNG.

  5. Summary markdown                    (table for the appendix)

Outputs
-------
  evaluate/results/appendix/
    corpus_stats.json
    per_query.jsonl
    per_query.csv
    figures/*.png
    summary.md

Usage
-----
    py evaluate/appendix_stats.py
    py evaluate/appendix_stats.py --limit 50           # quick smoke run
    py evaluate/appendix_stats.py --skip-arms          # cosine + corpus only
    py evaluate/appendix_stats.py --skip-figures       # data only
    py evaluate/appendix_stats.py --skip-corpus        # eval queries only
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from graph.neo4j_client import Neo4jClient  # noqa: E402
from agents.retrieval_utils import run_ann, run_bm25  # noqa: E402

log = logging.getLogger("appendix_stats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# Paths and constants
# ══════════════════════════════════════════════════════════════════════════════

# All three WixQA datasets share the same schema (question, answer, article_ids).
# The script reads any subset that exists on disk; figures and summary tables
# are stratified by dataset so the appendix can compare query distributions.
EVAL_CSVS: dict[str, Path] = {
    "expert":     Path("data/wixqa_expertwritten.csv"),
    "simulated":  Path("data/wixqa_simulated.csv"),
    "synthetic":  Path("data/wixqa_synthetic.csv"),
}
OUT_DIR     = Path("evaluate/results/appendix")
FIG_DIR     = OUT_DIR / "figures"

RECALL_K_LIST = [1, 5, 10, 20, 50, 100]
ANN_SWEEP_K   = 100   # top-K for the rank-of-gold sweep


# ══════════════════════════════════════════════════════════════════════════════
# Eval loader
# ══════════════════════════════════════════════════════════════════════════════

def load_eval_csv(
    csv_path: Path,
    dataset_label: str,
    limit: int | None = None,
) -> list[dict]:
    """Return list of {qid, dataset, question, gold_article_ids} for one CSV.

    All three WixQA datasets share the schema (`question, answer, article_ids`).
    `article_ids` is a Python-list literal string. Rows with malformed or empty
    refs are skipped.
    """
    import pandas as pd  # noqa: PLC0415
    if not csv_path.exists():
        log.warning("Eval set %s not found — skipping.", csv_path)
        return []
    df = pd.read_csv(csv_path)
    rows: list[dict] = []
    for i, r in df.iterrows():
        try:
            ids = ast.literal_eval(r["article_ids"])
            if not isinstance(ids, (list, tuple)):
                ids = [ids]
            ids = [str(x).strip() for x in ids if x]
        except (ValueError, SyntaxError):
            ids = [s.strip().strip("'\"") for s in str(r["article_ids"]).strip("[]").split(",")]
            ids = [x for x in ids if x]
        if not ids:
            continue
        rows.append({
            "qid":              f"{dataset_label}_q{i:05d}",
            "dataset":          dataset_label,
            "question":         str(r["question"]).strip(),
            "gold_article_ids": ids,
        })
        if limit and len(rows) >= limit:
            break
    log.info("Loaded %d eval queries from %s (%s)", len(rows), csv_path, dataset_label)
    return rows


def load_eval_sets(
    datasets: list[str] | None = None,
    limit_per_dataset: int | None = None,
) -> list[dict]:
    """Load one or more WixQA datasets (expert / simulated / synthetic).

    `datasets` is the list of labels to include; defaults to all three.
    `limit_per_dataset` caps each dataset independently.
    """
    keys = datasets or list(EVAL_CSVS.keys())
    rows: list[dict] = []
    for k in keys:
        if k not in EVAL_CSVS:
            log.warning("Unknown dataset label %r — skipping.", k)
            continue
        rows.extend(load_eval_csv(EVAL_CSVS[k], k, limit_per_dataset))
    log.info("Combined eval queries across %d dataset(s): %d total", len(keys), len(rows))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Cypher helpers — resolved at runtime against the live graph
# ══════════════════════════════════════════════════════════════════════════════

_CYPHER_GOLD_CHUNKS = """\
// Accept either d.id == article_id (hashed/normalised) or d.source_id == article_id
// (preserved original) so the mapping is robust across ingest variants.
MATCH (d:Document)
WHERE d.id IN $aids OR d.source_id IN $aids
MATCH (c:Chunk)-[:PART_OF]->(d)
WHERE c.embedding IS NOT NULL
RETURN c.id          AS chunk_id,
       c.text        AS text,
       c.embedding   AS embedding,
       c.token_count AS token_count,
       d.id          AS doc_id
"""

_CYPHER_DOC_LOOKUP = """\
MATCH (d:Document)
WHERE d.id IN $aids OR d.source_id IN $aids
RETURN DISTINCT d.id AS doc_id
"""

_CYPHER_CORPUS_COUNTS = """\
MATCH (c:Chunk) WITH count(c) AS chunk_count
MATCH (d:Document) WITH chunk_count, count(d) AS doc_count
MATCH (e:Entity) WITH chunk_count, doc_count, count(e) AS entity_count
OPTIONAL MATCH (s:SurfaceForm) WITH chunk_count, doc_count, entity_count, count(s) AS surface_form_count
OPTIONAL MATCH ()-[r:RELATED_TO]->() WITH chunk_count, doc_count, entity_count, surface_form_count, count(r) AS related_edge_count
OPTIONAL MATCH ()-[r:MENTIONS]->() WITH chunk_count, doc_count, entity_count, surface_form_count, related_edge_count, count(r) AS mentions_edge_count
OPTIONAL MATCH ()-[r:NEXT_CHUNK]->() WITH chunk_count, doc_count, entity_count, surface_form_count, related_edge_count, mentions_edge_count, count(r) AS next_edge_count
RETURN chunk_count, doc_count, entity_count, surface_form_count,
       related_edge_count, mentions_edge_count, next_edge_count
"""

_CYPHER_CHUNK_TOKEN_SAMPLE = """\
MATCH (c:Chunk) WHERE c.token_count IS NOT NULL
WITH c LIMIT $sample
RETURN c.token_count AS tc
"""

_CYPHER_EMBED_NORM_SAMPLE = """\
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
WITH c LIMIT $sample
RETURN c.embedding AS embedding
"""

_CYPHER_SAGE_DEGREE = """\
MATCH (c:Chunk)
OPTIONAL MATCH (c)-[r:RELATED_TO]-()
WITH c, count(r) AS deg
RETURN deg
"""

_CYPHER_MENTIONS_DEGREE = """\
MATCH (e:Entity)
OPTIONAL MATCH (c:Chunk)-[r:MENTIONS]->(e)
WITH e, count(r) AS deg
RETURN deg
"""

_CYPHER_SURFACE_FORMS_PER_ENTITY = """\
MATCH (e:Entity)
OPTIONAL MATCH (s:SurfaceForm)-[:ALIASES]->(e)
WITH e, count(s) AS n
RETURN n
"""


# ══════════════════════════════════════════════════════════════════════════════
# Numeric helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine(a, b) -> float:
    import numpy as np  # noqa: PLC0415
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_stats(values: list[float]) -> dict:
    import numpy as np  # noqa: PLC0415
    if not values:
        return {"mean": None, "median": None, "std": None,
                "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype="float64")
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      int(arr.size),
    }


def token_count(text: str) -> int:
    # Cheap whitespace tokenisation — adequate for query-length analysis.
    return len(text.split())


def expand_article_ids(ids: list[str]) -> list[str]:
    """Split WixQA compound gold IDs into 64-char component hashes.

    WixQA's expert split represents multi-article gold sets by concatenating
    the SHA-256 hashes of each article (128 chars = 2 articles, 192 = 3, …).
    The ingest stores each article under its own single 64-char hash, so the
    naive `d.id IN $aids` lookup misses every multi-article query. This
    helper expands each compound ID into its component hashes; single 64-char
    IDs pass through unchanged.
    """
    out: list[str] = []
    for aid in ids:
        s = str(aid).strip()
        if not s:
            continue
        out.append(s)  # always try the original form first (for source_id matches)
        if len(s) > 64 and len(s) % 64 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
            for i in range(0, len(s), 64):
                out.append(s[i:i + 64])
    # De-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ══════════════════════════════════════════════════════════════════════════════
# 1. Corpus descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════

def collect_corpus_stats(client: Neo4jClient, sample: int = 5000) -> dict:
    log.info("Collecting corpus descriptive stats (sample=%d) ...", sample)
    out: dict = {}

    counts = client.run(_CYPHER_CORPUS_COUNTS, {})
    if counts:
        out["counts"] = {k: int(v or 0) for k, v in counts[0].items()}
    else:
        out["counts"] = {}

    # Chunk token-count distribution
    rows = client.run(_CYPHER_CHUNK_TOKEN_SAMPLE, {"sample": sample})
    tcs = [int(r["tc"]) for r in rows if r["tc"] is not None]
    out["chunk_token_count"] = safe_stats(tcs)
    out["_chunk_token_sample"] = tcs

    # Embedding norm distribution
    import numpy as np  # noqa: PLC0415
    rows = client.run(_CYPHER_EMBED_NORM_SAMPLE, {"sample": sample})
    norms = []
    for r in rows:
        v = r.get("embedding")
        if v is None:
            continue
        norms.append(float(np.linalg.norm(np.asarray(v, dtype="float32"))))
    out["embedding_norm"] = safe_stats(norms)
    out["_embedding_norm_sample"] = norms

    # SAGE RELATED_TO degree per chunk
    rows = client.run(_CYPHER_SAGE_DEGREE, {})
    sage_deg = [int(r["deg"]) for r in rows]
    out["sage_degree_per_chunk"] = safe_stats(sage_deg)
    out["_sage_degree_sample"] = sage_deg

    # MENTIONS in-degree per entity
    rows = client.run(_CYPHER_MENTIONS_DEGREE, {})
    mentions_deg = [int(r["deg"]) for r in rows]
    out["mentions_degree_per_entity"] = safe_stats(mentions_deg)
    out["_mentions_degree_sample"] = mentions_deg

    # SurfaceForm count per entity (alias breadth)
    rows = client.run(_CYPHER_SURFACE_FORMS_PER_ENTITY, {})
    sf_per_e = [int(r["n"]) for r in rows]
    out["surface_forms_per_entity"] = safe_stats(sf_per_e)
    out["_surface_forms_sample"] = sf_per_e

    log.info("  counts: %s", out["counts"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-query: cosine geometry + ANN rank + Recall@K
# ══════════════════════════════════════════════════════════════════════════════

def fetch_gold_chunks(client: Neo4jClient, article_ids: list[str]) -> list[dict]:
    aids = expand_article_ids(article_ids)
    rows = client.run(_CYPHER_GOLD_CHUNKS, {"aids": aids})
    out = []
    for r in rows:
        emb = r.get("embedding")
        if emb is None:
            continue
        out.append({
            "chunk_id": r["chunk_id"],
            "text":     r["text"],
            "embedding": emb,
            "token_count": r.get("token_count"),
            "doc_id":   r["doc_id"],
        })
    return out


def fetch_resolved_doc_ids(client: Neo4jClient, article_ids: list[str]) -> list[str]:
    aids = expand_article_ids(article_ids)
    rows = client.run(_CYPHER_DOC_LOOKUP, {"aids": aids})
    return [r["doc_id"] for r in rows]


def hit_at_k_lists(retrieved_ids: list[str], gold_ids: set[str], ks: list[int]) -> dict:
    """For each K, did the top-K retrieved set contain any gold chunk?"""
    out: dict = {}
    gold_ids_lower = {str(g) for g in gold_ids}
    for k in ks:
        topk = retrieved_ids[:k]
        out[str(k)] = any(str(rid) in gold_ids_lower for rid in topk)
    return out


def rank_of_best_gold(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    gold_ids = {str(g) for g in gold_ids}
    for i, rid in enumerate(retrieved_ids):
        if str(rid) in gold_ids:
            return i + 1   # 1-based
    return None


def collect_per_query(
    client: Neo4jClient,
    eval_rows: list[dict],
    skip_arms: bool = False,
) -> list[dict]:
    """For every eval query: embed once, fetch gold chunks, compute cosine
    + per-arm Recall@K. Returns a list of dict rows (one per query)."""
    from etl.embedder import embed_texts  # noqa: PLC0415

    out: list[dict] = []
    queries = [r["question"] for r in eval_rows]

    log.info("Embedding %d queries ...", len(queries))
    q_vecs = embed_texts(queries)

    n_missing_doc = 0
    n_missing_chunk = 0
    for i, (row, qvec) in enumerate(zip(eval_rows, q_vecs)):
        gold_chunks = fetch_gold_chunks(client, row["gold_article_ids"])
        resolved_docs = fetch_resolved_doc_ids(client, row["gold_article_ids"])
        if not resolved_docs:
            n_missing_doc += 1
        if not gold_chunks:
            n_missing_chunk += 1

        # Cosine vs each gold chunk
        cosines = [cosine(qvec, gc["embedding"]) for gc in gold_chunks]
        gold_ids = {gc["chunk_id"] for gc in gold_chunks}

        record: dict = {
            "qid": row["qid"],
            "dataset": row.get("dataset", "unknown"),
            "question": row["question"],
            "query_token_count": token_count(row["question"]),
            "gold_article_ids": row["gold_article_ids"],
            "resolved_doc_count": len(resolved_docs),
            "gold_chunk_count": len(gold_chunks),
            "cosine_to_gold": safe_stats(cosines),
        }

        if not skip_arms:
            # Semantic ANN sweep — top ANN_SWEEP_K
            try:
                ann_hits = run_ann(client, qvec, ANN_SWEEP_K, None)
                ann_ids = [h.id for h in ann_hits]
            except Exception as exc:
                log.warning("  [%s] ANN failed: %s", row["qid"], exc)
                ann_ids = []

            try:
                bm25_hits = run_bm25(client, row["question"], ANN_SWEEP_K)
                bm25_ids = [h.id for h in bm25_hits]
            except Exception as exc:
                log.warning("  [%s] BM25 failed: %s", row["qid"], exc)
                bm25_ids = []

            record["best_gold_rank_semantic"] = rank_of_best_gold(ann_ids, gold_ids)
            record["best_gold_rank_bm25"]     = rank_of_best_gold(bm25_ids, gold_ids)
            record["hit_at_k_semantic"]       = hit_at_k_lists(ann_ids, gold_ids, RECALL_K_LIST)
            record["hit_at_k_bm25"]           = hit_at_k_lists(bm25_ids, gold_ids, RECALL_K_LIST)

        out.append(record)

        if (i + 1) % 25 == 0:
            log.info("  processed %d / %d queries", i + 1, len(eval_rows))

    log.info("Per-query pass complete. %d/%d had no resolved document, "
             "%d/%d had no embedded gold chunk.",
             n_missing_doc, len(eval_rows), n_missing_chunk, len(eval_rows))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pure-stdlib Pearson r; returns None for degenerate inputs."""
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs[:n])
    syy = sum((y - my) ** 2 for y in ys[:n])
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = (sxx * syy) ** 0.5
    return sxy / den if den > 0 else None


def _aggregate_slice(rows: list[dict]) -> dict:
    """Headline stats for one slice of per-query rows (used for global + per-dataset)."""
    mean_gold_cos = [r["cosine_to_gold"]["mean"] for r in rows if r["cosine_to_gold"]["n"]]
    max_gold_cos  = [r["cosine_to_gold"]["max"]  for r in rows if r["cosine_to_gold"]["n"]]

    # Corpus-coverage diagnostics: how many of the gold article_ids actually
    # resolve to a Document / Chunk in the live graph. A query whose gold IDs
    # never landed in the corpus is unrecoverable by any retriever — this is
    # the ceiling stat the paper needs when claiming end-to-end recall.
    n_zero_doc      = sum(1 for r in rows if r.get("resolved_doc_count", 0) == 0)
    n_zero_chunk    = sum(1 for r in rows if r.get("gold_chunk_count", 0) == 0)
    n_full_doc_res  = sum(1 for r in rows
                          if r.get("resolved_doc_count", 0) == len(r.get("gold_article_ids") or []))
    # Per-query doc-resolution rate (#resolved / #requested)
    doc_res_rates = [
        (r.get("resolved_doc_count", 0) / max(1, len(r.get("gold_article_ids") or [])))
        for r in rows
    ]

    summary: dict = {
        "n_queries":            len(rows),
        "n_with_gold_chunks":   sum(1 for r in rows if r["gold_chunk_count"] > 0),
        "n_zero_resolved_docs": n_zero_doc,
        "n_zero_gold_chunks":   n_zero_chunk,
        "pct_fully_resolved_docs": n_full_doc_res / max(1, len(rows)),
        "doc_resolution_rate":  safe_stats(doc_res_rates),
        "mean_gold_cosine":     safe_stats(mean_gold_cos),
        "max_gold_cosine":      safe_stats(max_gold_cos),
        "query_token_count":    safe_stats([r["query_token_count"] for r in rows]),
        "gold_chunk_count":     safe_stats([r["gold_chunk_count"]   for r in rows]),
    }

    if any("hit_at_k_semantic" in r for r in rows):
        for arm in ("semantic", "bm25"):
            recall: dict = {}
            for k in RECALL_K_LIST:
                hits = [r.get(f"hit_at_k_{arm}", {}).get(str(k), False) for r in rows]
                recall[str(k)] = sum(1 for h in hits if h) / max(1, len(hits))
            summary[f"recall_at_k_{arm}"] = recall

        # Fusion ceiling: did EITHER arm find gold in top-K? Any reranker /
        # RRF / planner that consumes both arms is upper-bounded by this curve.
        # Reporting it lets us frame PIVOT as approaching the ceiling
        # efficiently rather than lifting it.
        union_recall: dict = {}
        for k in RECALL_K_LIST:
            n_hit = sum(
                1 for r in rows
                if r.get("hit_at_k_semantic", {}).get(str(k), False)
                or r.get("hit_at_k_bm25",     {}).get(str(k), False)
            )
            union_recall[str(k)] = n_hit / max(1, len(rows))
        summary["recall_at_k_union"] = union_recall

        # Corpus-bound ceiling: max recall achievable given gold chunks that
        # actually exist in the graph. R@K_union cannot exceed this.
        summary["recall_ceiling_in_corpus"] = (
            sum(1 for r in rows if r.get("gold_chunk_count", 0) > 0) / max(1, len(rows))
        )

        ranks_sem  = [r["best_gold_rank_semantic"] for r in rows
                      if r.get("best_gold_rank_semantic") is not None]
        ranks_bm25 = [r["best_gold_rank_bm25"] for r in rows
                      if r.get("best_gold_rank_bm25") is not None]
        summary["best_gold_rank_semantic"] = safe_stats(ranks_sem)
        summary["best_gold_rank_bm25"]     = safe_stats(ranks_bm25)
        summary["pct_with_gold_in_top100_semantic"] = len(ranks_sem)  / max(1, len(rows))
        summary["pct_with_gold_in_top100_bm25"]    = len(ranks_bm25) / max(1, len(rows))

        # Length-vs-retrievability correlations. Justifies the scatterplot in
        # the appendix: does longer / shorter querying systematically affect
        # dense retrievability and top-10 success?
        with_gold = [r for r in rows if r.get("cosine_to_gold", {}).get("n")]
        if len(with_gold) >= 3:
            xs = [r["query_token_count"]      for r in with_gold]
            ys = [r["cosine_to_gold"]["mean"] for r in with_gold]
            summary["pearson_qlen_vs_cosine"] = _pearson(xs, ys)
        if len(rows) >= 3:
            xs = [r["query_token_count"] for r in rows]
            ys = [1.0 if r.get("hit_at_k_semantic", {}).get("10") else 0.0 for r in rows]
            summary["pearson_qlen_vs_hit10_semantic"] = _pearson(xs, ys)
            ys = [1.0 if r.get("hit_at_k_bm25", {}).get("10") else 0.0 for r in rows]
            summary["pearson_qlen_vs_hit10_bm25"] = _pearson(xs, ys)
    return summary


def group_by_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("dataset", "unknown"), []).append(r)
    return out


def aggregate_per_query(rows: list[dict]) -> dict:
    """Headline numbers — combined across datasets and stratified per dataset."""
    summary = _aggregate_slice(rows)
    per_ds = {label: _aggregate_slice(sub) for label, sub in group_by_dataset(rows).items()}
    summary["per_dataset"] = per_ds
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 4. Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(per_query: list[dict], corpus: dict, out_dir: Path) -> None:
    try:
        import matplotlib  # noqa: PLC0415
    except ModuleNotFoundError:
        log.warning("matplotlib not installed — skipping figures. "
                    "Install with `pip install matplotlib` or rerun with --skip-figures.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=130)
        plt.close()
        log.info("  fig → %s", out_dir / name)

    # Group once: every query-side figure overlays one trace per dataset.
    by_ds = group_by_dataset(per_query)
    # Stable label order — preferred datasets first, then any extras alphabetically.
    preferred = ["expert", "simulated", "synthetic"]
    ds_labels = [d for d in preferred if d in by_ds] + \
                sorted(d for d in by_ds if d not in preferred)

    # 1. Cosine(query, gold) distribution — mean per query, one curve per dataset
    if any(r["cosine_to_gold"]["n"] for r in per_query):
        plt.figure(figsize=(9, 4.5))
        for ds in ds_labels:
            vals = [r["cosine_to_gold"]["mean"]
                    for r in by_ds[ds] if r["cosine_to_gold"]["n"]]
            if vals:
                plt.hist(vals, bins=40, alpha=0.5, label=f"{ds} (n={len(vals)})")
        plt.xlabel("mean cosine similarity to gold chunk(s)")
        plt.ylabel("queries")
        plt.title("Query–gold cosine geometry per dataset")
        plt.legend()
        _save("01_cosine_to_gold_dist.png")

    # 2. Cosine vs query length scatter, colour-coded by dataset
    if any(r["cosine_to_gold"]["n"] for r in per_query):
        plt.figure(figsize=(9, 5))
        for ds in ds_labels:
            xs = [r["query_token_count"] for r in by_ds[ds] if r["cosine_to_gold"]["n"]]
            ys = [r["cosine_to_gold"]["mean"] for r in by_ds[ds] if r["cosine_to_gold"]["n"]]
            if xs:
                plt.scatter(xs, ys, alpha=0.35, s=12, label=f"{ds} (n={len(xs)})")
        plt.xlabel("query length (tokens)")
        plt.ylabel("mean cosine to gold chunks")
        plt.title("Query length vs. dense retrievability of gold (per dataset)")
        plt.legend()
        _save("02_cosine_vs_query_length.png")

    # 2b. Three-panel scatter — same data as fig 02 but split per dataset so each
    #     subplot has its own regression-friendly view. This is the figure the
    #     appendix references when discussing query-length effects.
    if any(r["cosine_to_gold"]["n"] for r in per_query):
        n_panels = len(ds_labels)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]
        for ax, ds in zip(axes, ds_labels):
            xs = [r["query_token_count"] for r in by_ds[ds] if r["cosine_to_gold"]["n"]]
            ys = [r["cosine_to_gold"]["mean"] for r in by_ds[ds] if r["cosine_to_gold"]["n"]]
            ax.scatter(xs, ys, alpha=0.45, s=14)
            ax.set_title(f"{ds} (n={len(xs)})")
            ax.set_xlabel("query length (tokens)")
            ax.grid(alpha=0.3)
            if xs and len(xs) >= 3:
                # OLS line for visual orientation
                m = np.polyfit(xs, ys, 1)
                xr = np.linspace(min(xs), max(xs), 50)
                ax.plot(xr, np.polyval(m, xr), linestyle="--", linewidth=1.2)
        axes[0].set_ylabel("mean cosine to gold chunks")
        fig.suptitle("Query length vs. dense retrievability — per dataset")
        _save("02b_cosine_vs_qlen_panel.png")

    # 3. Best-gold-rank in ANN top-K (semantic) — one histogram per dataset, semantic only
    if any("best_gold_rank_semantic" in r for r in per_query):
        plt.figure(figsize=(9, 4.5))
        bins = list(range(1, ANN_SWEEP_K + 2, 2))
        for ds in ds_labels:
            ranks = [r["best_gold_rank_semantic"] for r in by_ds[ds]
                     if r.get("best_gold_rank_semantic") is not None]
            if ranks:
                plt.hist(ranks, bins=bins, alpha=0.45, label=f"{ds} (n={len(ranks)})")
        plt.xlabel(f"rank of best gold chunk in semantic top-{ANN_SWEEP_K}")
        plt.ylabel("queries")
        plt.title("Where does the gold chunk land in dense retrieval? (per dataset)")
        plt.legend()
        _save("03_best_gold_rank.png")

    # 4. Recall@K curves — semantic + BM25, dashed/solid per arm, colour per dataset
    if any("hit_at_k_semantic" in r for r in per_query):
        plt.figure(figsize=(9, 5))
        for ds in ds_labels:
            sub = by_ds[ds]
            if not sub:
                continue
            for arm, marker, ls in (("semantic", "o", "-"), ("bm25", "s", "--")):
                ys = []
                for k in RECALL_K_LIST:
                    hits = [r.get(f"hit_at_k_{arm}", {}).get(str(k), False)
                            for r in sub]
                    ys.append(sum(1 for h in hits if h) / max(1, len(hits)))
                plt.plot(RECALL_K_LIST, ys, marker=marker, linestyle=ls,
                         label=f"{ds} · {arm}")
        plt.xscale("log")
        plt.xlabel("K  (top-K)")
        plt.ylabel("Recall@K")
        plt.title("Per-arm Recall@K against gold chunk set (per dataset)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        _save("04_recall_at_k.png")

    # 5. Chunk token-count distribution
    tcs = corpus.get("_chunk_token_sample") or []
    if tcs:
        plt.figure(figsize=(8, 4))
        plt.hist(tcs, bins=50)
        plt.xlabel("chunk token count")
        plt.ylabel("chunks")
        plt.title("Chunk-length distribution (sampled)")
        _save("05_chunk_length_dist.png")

    # 6. Embedding norm
    norms = corpus.get("_embedding_norm_sample") or []
    if norms:
        plt.figure(figsize=(8, 4))
        plt.hist(norms, bins=50)
        plt.xlabel("L2 norm of chunk embedding")
        plt.ylabel("chunks")
        plt.title("Embedding-norm distribution (sampled)")
        _save("06_embedding_norm_dist.png")

    # 7. SAGE degree
    sd = corpus.get("_sage_degree_sample") or []
    if sd:
        max_d = max(sd) if sd else 0
        bins = list(range(0, min(max_d, 40) + 2))
        plt.figure(figsize=(8, 4))
        plt.hist(sd, bins=bins)
        plt.xlabel("RELATED_TO degree per chunk (SAGE)")
        plt.ylabel("chunks")
        plt.title("Graph mass available for PPR via SAGE")
        _save("07_sage_degree_dist.png")

    # 8. MENTIONS degree
    md = corpus.get("_mentions_degree_sample") or []
    if md:
        # Tail is long — log scale
        plt.figure(figsize=(8, 4))
        plt.hist([d for d in md if d > 0], bins=50)
        plt.xlabel("MENTIONS in-degree per entity")
        plt.ylabel("entities")
        plt.yscale("log")
        plt.title("Entity hub distribution (log-scale)")
        _save("08_mentions_degree_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Markdown summary
# ══════════════════════════════════════════════════════════════════════════════

def write_summary_md(corpus: dict, query_agg: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Appendix — Retrieval Statistics")
    lines.append("")
    lines.append("Generated by `evaluate/appendix_stats.py`. All figures live in `figures/`.")
    lines.append("")

    # A. Corpus
    c = corpus.get("counts") or {}
    lines.append("## A. Corpus")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(f"| Documents | {c.get('doc_count', 'n/a'):,} |")
    lines.append(f"| Chunks | {c.get('chunk_count', 'n/a'):,} |")
    lines.append(f"| Entities | {c.get('entity_count', 'n/a'):,} |")
    lines.append(f"| SurfaceForm nodes | {c.get('surface_form_count', 'n/a'):,} |")
    lines.append(f"| RELATED_TO edges (SAGE) | {c.get('related_edge_count', 'n/a'):,} |")
    lines.append(f"| MENTIONS edges | {c.get('mentions_edge_count', 'n/a'):,} |")
    lines.append(f"| NEXT_CHUNK edges | {c.get('next_edge_count', 'n/a'):,} |")
    lines.append("")

    def _fmt_stats(d: dict | None) -> str:
        if not d or d.get("n") in (0, None):
            return "—"
        return (f"mean={d['mean']:.3f}  median={d['median']:.3f}  "
                f"std={d['std']:.3f}  min={d['min']:.3f}  max={d['max']:.3f}  n={d['n']}")

    lines.append("| Distribution | Stats |")
    lines.append("|---|---|")
    lines.append(f"| Chunk token count | {_fmt_stats(corpus.get('chunk_token_count'))} |")
    lines.append(f"| Embedding L2 norm | {_fmt_stats(corpus.get('embedding_norm'))} |")
    lines.append(f"| SAGE degree / chunk | {_fmt_stats(corpus.get('sage_degree_per_chunk'))} |")
    lines.append(f"| MENTIONS in-degree / entity | {_fmt_stats(corpus.get('mentions_degree_per_entity'))} |")
    lines.append(f"| SurfaceForms / entity | {_fmt_stats(corpus.get('surface_forms_per_entity'))} |")
    lines.append("")

    # B. Query–gold geometry (combined + per dataset)
    lines.append("## B. Query–gold geometry")
    lines.append("")
    lines.append(f"- Queries evaluated: **{query_agg.get('n_queries', 0)}**")
    lines.append(f"- Queries with at least one embedded gold chunk: "
                 f"**{query_agg.get('n_with_gold_chunks', 0)}**")
    lines.append(f"- Queries with zero resolved gold documents (gold IDs not in corpus): "
                 f"**{query_agg.get('n_zero_resolved_docs', 0)}**")
    fr = query_agg.get("pct_fully_resolved_docs")
    if fr is not None:
        lines.append(f"- Queries with all gold article_ids resolved: **{fr:.1%}**")
    rc = query_agg.get("recall_ceiling_in_corpus")
    if rc is not None:
        lines.append(f"- Hard recall ceiling (gold present in corpus): **{rc:.1%}** "
                     f"— no retriever can exceed this on the current ingest.")
    lines.append("")
    lines.append("### Combined across datasets")
    lines.append("")
    lines.append("| Per-query distribution | Stats |")
    lines.append("|---|---|")
    lines.append(f"| Mean cosine(query, gold) | {_fmt_stats(query_agg.get('mean_gold_cosine'))} |")
    lines.append(f"| Max  cosine(query, gold) | {_fmt_stats(query_agg.get('max_gold_cosine'))} |")
    lines.append(f"| Query token count | {_fmt_stats(query_agg.get('query_token_count'))} |")
    lines.append(f"| Gold chunks / query | {_fmt_stats(query_agg.get('gold_chunk_count'))} |")
    lines.append("")

    per_ds = query_agg.get("per_dataset") or {}
    if per_ds:
        lines.append("### Per dataset")
        lines.append("")
        lines.append("| Dataset | n | n with gold | mean cos (mean) | max cos (mean) | "
                     "query len (mean) |")
        lines.append("|---|---|---|---|---|---|")
        for ds, ds_agg in per_ds.items():
            mc = ds_agg.get("mean_gold_cosine") or {}
            xc = ds_agg.get("max_gold_cosine")  or {}
            ql = ds_agg.get("query_token_count") or {}
            lines.append(
                f"| {ds} | {ds_agg.get('n_queries', 0)} | "
                f"{ds_agg.get('n_with_gold_chunks', 0)} | "
                f"{mc.get('mean'):.3f} | {xc.get('mean'):.3f} | "
                f"{ql.get('mean'):.1f} |"
                if mc.get("n") else
                f"| {ds} | {ds_agg.get('n_queries', 0)} | "
                f"{ds_agg.get('n_with_gold_chunks', 0)} | — | — | — |"
            )
        lines.append("")

        # B.2 Corpus-coverage table — the smoking gun for the recall ceiling.
        lines.append("### Corpus coverage (gold-ID resolution)")
        lines.append("")
        lines.append("| Dataset | n | zero docs resolved | zero gold chunks | "
                     "fully resolved | hard ceiling |")
        lines.append("|---|---|---|---|---|---|")
        for ds, ds_agg in per_ds.items():
            n = ds_agg.get("n_queries", 0)
            zd = ds_agg.get("n_zero_resolved_docs", 0)
            zc = ds_agg.get("n_zero_gold_chunks", 0)
            fr = ds_agg.get("pct_fully_resolved_docs")
            rc = ds_agg.get("recall_ceiling_in_corpus")
            fr_s = f"{fr:.1%}" if fr is not None else "—"
            rc_s = f"{rc:.1%}" if rc is not None else "—"
            lines.append(f"| {ds} | {n} | {zd} | {zc} | {fr_s} | {rc_s} |")
        lines.append("")
        lines.append("> *Hard ceiling* = fraction of queries with at least one gold chunk "
                     "actually present in the corpus. End-to-end Recall@K cannot exceed this.")
        lines.append("")

        # B.3 Length correlations — justifies the scatterplot.
        lines.append("### Query length × dense retrievability (Pearson r)")
        lines.append("")
        lines.append("| Dataset | r(qlen, cos̄) | r(qlen, hit@10 sem) | r(qlen, hit@10 bm25) |")
        lines.append("|---|---|---|---|")
        def _fr(v):
            return "—" if v is None else f"{v:+.2f}"
        # Combined first
        lines.append(
            f"| **combined** | {_fr(query_agg.get('pearson_qlen_vs_cosine'))} | "
            f"{_fr(query_agg.get('pearson_qlen_vs_hit10_semantic'))} | "
            f"{_fr(query_agg.get('pearson_qlen_vs_hit10_bm25'))} |"
        )
        for ds, ds_agg in per_ds.items():
            lines.append(
                f"| {ds} | {_fr(ds_agg.get('pearson_qlen_vs_cosine'))} | "
                f"{_fr(ds_agg.get('pearson_qlen_vs_hit10_semantic'))} | "
                f"{_fr(ds_agg.get('pearson_qlen_vs_hit10_bm25'))} |"
            )
        lines.append("")

    # C. Per-arm Recall@K — combined + per dataset
    if "recall_at_k_semantic" in query_agg:
        lines.append("## C. Per-arm Recall@K (raw query, no fusion, no rerank)")
        lines.append("")
        lines.append("### Combined")
        lines.append("")
        lines.append("| K | Semantic ANN | BM25 | Union (fusion ceiling) |")
        lines.append("|---|---|---|---|")
        sem = query_agg["recall_at_k_semantic"]
        bm  = query_agg["recall_at_k_bm25"]
        uni = query_agg.get("recall_at_k_union") or {}
        for k in RECALL_K_LIST:
            u = uni.get(str(k))
            u_s = f"{u:.3f}" if u is not None else "—"
            lines.append(f"| {k} | {sem[str(k)]:.3f} | {bm[str(k)]:.3f} | {u_s} |")
        lines.append("")
        lines.append("> *Union* = the fraction of queries with at least one gold chunk "
                     "in the top-K of EITHER arm. This is the upper bound for any "
                     "reranker, RRF fusion, or planner that consumes both arms.")
        lines.append("")
        lines.append(f"- Queries with any gold chunk in semantic top-{ANN_SWEEP_K}: "
                     f"**{query_agg['pct_with_gold_in_top100_semantic']:.1%}**")
        lines.append(f"- Queries with any gold chunk in BM25 top-{ANN_SWEEP_K}: "
                     f"**{query_agg['pct_with_gold_in_top100_bm25']:.1%}**")
        lines.append("")

        if per_ds:
            lines.append("### Per dataset (Recall@K)")
            lines.append("")
            header = "| Dataset | arm | " + " | ".join(f"R@{k}" for k in RECALL_K_LIST) + " |"
            sep    = "|---|---|" + "|".join(["---"] * len(RECALL_K_LIST)) + "|"
            lines.append(header)
            lines.append(sep)
            for ds, ds_agg in per_ds.items():
                for arm in ("semantic", "bm25", "union"):
                    rec = ds_agg.get(f"recall_at_k_{arm}")
                    if not rec:
                        continue
                    cells = " | ".join(f"{rec[str(k)]:.3f}" for k in RECALL_K_LIST)
                    lines.append(f"| {ds} | {arm} | {cells} |")
            lines.append("")

    lines.append("## D. Figures")
    lines.append("")
    for f in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- ![{f.stem}](figures/{f.name})")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote summary → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. JSONL / CSV serialisation
# ══════════════════════════════════════════════════════════════════════════════

def write_per_query(rows: list[dict], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote per-query JSONL → %s", jsonl_path)

    # Flatten for CSV — keep only scalar fields suitable for pandas
    import pandas as pd  # noqa: PLC0415
    flat = []
    for r in rows:
        cos = r.get("cosine_to_gold") or {}
        d = {
            "qid": r["qid"],
            "dataset": r.get("dataset", "unknown"),
            "query_token_count": r.get("query_token_count"),
            "gold_chunk_count":  r.get("gold_chunk_count"),
            "resolved_doc_count": r.get("resolved_doc_count"),
            "cosine_mean":       cos.get("mean"),
            "cosine_max":        cos.get("max"),
            "cosine_median":     cos.get("median"),
            "best_gold_rank_semantic": r.get("best_gold_rank_semantic"),
            "best_gold_rank_bm25":     r.get("best_gold_rank_bm25"),
        }
        for arm in ("semantic", "bm25"):
            for k in RECALL_K_LIST:
                d[f"hit_{arm}@{k}"] = r.get(f"hit_at_k_{arm}", {}).get(str(k))
        flat.append(d)
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    log.info("Wrote per-query CSV → %s", csv_path)


def write_corpus_stats(corpus: dict, path: Path) -> None:
    # Strip internal sample arrays before serialising — the JSON would be huge.
    public = {k: v for k, v in corpus.items() if not k.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, ensure_ascii=False)
    log.info("Wrote corpus_stats → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(EVAL_CSVS.keys()),
                    choices=list(EVAL_CSVS.keys()),
                    help="WixQA datasets to include (default: all three).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap per dataset on number of eval queries (default: all).")
    ap.add_argument("--corpus-sample", type=int, default=5000,
                    help="Sample size for chunk/embedding-norm stats.")
    ap.add_argument("--skip-arms", action="store_true",
                    help="Skip per-arm Recall@K (cosine + corpus only).")
    ap.add_argument("--skip-corpus", action="store_true",
                    help="Skip corpus descriptive stats.")
    ap.add_argument("--skip-figures", action="store_true",
                    help="Skip matplotlib figure rendering.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="Output directory (default: evaluate/results/appendix).")
    args = ap.parse_args()

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_sets(datasets=args.datasets, limit_per_dataset=args.limit)
    if not eval_rows:
        log.error("No eval queries loaded — aborting.")
        sys.exit(1)

    with Neo4jClient() as client:
        corpus_stats = {}
        if not args.skip_corpus:
            corpus_stats = collect_corpus_stats(client, sample=args.corpus_sample)
            write_corpus_stats(corpus_stats, out_dir / "corpus_stats.json")

        per_query = collect_per_query(client, eval_rows, skip_arms=args.skip_arms)
        query_agg = aggregate_per_query(per_query)

    write_per_query(per_query, out_dir / "per_query.jsonl", out_dir / "per_query.csv")

    if not args.skip_figures:
        make_figures(per_query, corpus_stats, fig_dir)

    write_summary_md(corpus_stats, query_agg, out_dir / "summary.md")

    # Also dump the aggregate as JSON for further processing
    with (out_dir / "query_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(query_agg, f, indent=2, ensure_ascii=False)
    log.info("Wrote aggregate → %s", out_dir / "query_aggregate.json")

    log.info("Done. See %s/summary.md for the appendix.", out_dir)


if __name__ == "__main__":
    main()
