"""
evaluate/eval.py — Three-layer evaluation CLI for the Graph-RAG prototype.

Layers:
  1  retrieval   Precision@k, Recall@k, MRR, NDCG@k  (no LLM cost)
  2  ragas       Faithfulness, Answer Correctness, Context Recall  (LLM cost)
  3  routing     CRAG verdict distribution + deferral rate
  4  sweep       Context-engineering ablation: retrieval metrics + cost
                 (latency, LLM calls, est. tokens) per signal configuration

Usage:
  python -m evaluate.eval --layer retrieval
  python -m evaluate.eval --layer ragas
  python -m evaluate.eval --layer routing
  python -m evaluate.eval --layer sweep        # baseline / full / leave-one-out
  python -m evaluate.eval --layer all          # retrieval + ragas + routing

Options:
  --dataset PATH     path to wixqa_expertwritten.csv  (default: data/wixqa_expertwritten.csv)
  --k INT            top-k cutoff for retrieval metrics  (default: 5)
  --out PATH         write results CSV to this path  (default: evaluate/results_<layer>.csv)
  --limit INT        only evaluate first N rows (useful for smoke-testing)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── make sure the project root is on sys.path when run as a module ────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _run_pipeline(question: str) -> dict[str, Any]:
    """Run the full LangGraph pipeline and return the final AgentState."""
    from graph_builder import build_graph
    graph = build_graph()
    state = graph.invoke({"original_query": question})
    return state


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        print("  [warn] no rows to write")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved {len(rows)} rows to {path}")


def _progress(i: int, total: int, question: str) -> None:
    print(f"  [{i}/{total}] {question[:70]}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    rels = [1 if cid in relevant_ids else 0 for cid in retrieved_ids[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _doc_id(chunk_id: str) -> str:
    """Map a chunk id ('{doc_id}_{index}') back to its parent document id.

    The benchmark grounds on document/article ids, but retrievers return chunk
    ids carrying a trailing _<index>. Only a trailing all-digit segment is
    stripped, so bare document ids (64-char hex hashes, numeric ids) are
    unaffected.
    """
    head, sep, tail = chunk_id.rpartition("_")
    return head if sep and tail.isdigit() else chunk_id


def _retrieved_doc_ids(reranked: list) -> list[str]:
    """Ordered, de-duplicated parent-document ids for a list of ScoredChunks."""
    out: list[str] = []
    seen: set[str] = set()
    for c in reranked:
        d = _doc_id(c.id)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


import re as _re

# A token made only of 64-char hex ids plus separators (comma / whitespace / quotes).
_HEX_TOKEN_RE = _re.compile(r"^[0-9a-fA-F,\s'\"]+$")
_HEX_ID_RE = _re.compile(r"[0-9a-fA-F]{64}")


def _split_concat_ids(token: str) -> list[str]:
    """Split WixQA multi-article ids packed into one token into individual ids.

    Multi-article ExpertWritten answers encode several article ids in a single
    field value — either concatenated ('hash1hash2') or comma-joined
    ('hash1, hash2') — while the KB grounds on individual 64-char SHA-256 ids.
    This extracts every 64-char hex run. Guarded so it never touches non-WixQA
    data: it only fires on tokens longer than 64 chars composed solely of hex
    digits and separators; numeric ticket ids and ordinary slugs pass through.
    """
    t = token.strip()
    if len(t) > 64 and _HEX_TOKEN_RE.match(t):
        ids = _HEX_ID_RE.findall(t)
        if ids:
            return ids
    return [t]


def _parse_article_ids(raw: str) -> set[str]:
    """Parse the article_ids column — stored as a Python list literal string.

    Concatenated multi-article ids are split into their 64-char components so
    multi-article grounding is counted correctly (see _split_concat_ids).
    """
    raw = raw.strip()
    if not raw or raw in ("[]", ""):
        return set()
    try:
        parsed = json.loads(raw.replace("'", '"'))
        items = parsed if isinstance(parsed, list) else []
    except Exception:
        # fallback: strip brackets and split by comma
        inner = raw.strip("[]").replace("'", "").replace('"', "")
        items = [s.strip() for s in inner.split(",") if s.strip()]
    return {part for item in items for part in _split_concat_ids(str(item))}


def run_retrieval(dataset: list[dict], k: int, out: str) -> None:
    print(f"\n=== Layer 1: Retrieval  (k={k}) ===")

    from graph_builder import build_graph
    graph = build_graph()

    rows, p_sum, r_sum, mrr_sum, ndcg_sum = [], 0.0, 0.0, 0.0, 0.0
    total = len(dataset)

    for i, row in enumerate(dataset, start=1):
        question = row["question"]
        relevant_ids = _parse_article_ids(row.get("article_ids", ""))
        _progress(i, total, question)

        state = graph.invoke({"original_query": question})
        reranked = state.get("reranked_hits", [])
        retrieved_ids = _retrieved_doc_ids(reranked)

        topk = retrieved_ids[:k]
        hits = len(set(topk) & relevant_ids)
        precision = hits / k if k else 0.0
        recall = hits / len(relevant_ids) if relevant_ids else 0.0
        mrr = _mrr(retrieved_ids, relevant_ids)
        ndcg = _ndcg(retrieved_ids, relevant_ids, k)

        p_sum += precision
        r_sum += recall
        mrr_sum += mrr
        ndcg_sum += ndcg

        rows.append({
            "question": question,
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            "mrr": round(mrr, 4),
            f"ndcg@{k}": round(ndcg, 4),
            "relevant_ids": "|".join(relevant_ids),
            "retrieved_top5": "|".join(topk),
        })

    n = len(rows)
    print(f"\n  Precision@{k}  : {p_sum/n:.4f}")
    print(f"  Recall@{k}     : {r_sum/n:.4f}")
    print(f"  MRR           : {mrr_sum/n:.4f}")
    print(f"  NDCG@{k}       : {ndcg_sum/n:.4f}")
    _write_csv(out, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — RAGAS end-to-end metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas(dataset: list[dict], out: str) -> None:
    print("\n=== Layer 2: RAGAS answer quality ===")

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_correctness, context_recall
        from datasets import Dataset as HFDataset
    except ImportError:
        print("  [error] ragas or datasets not installed. Run: pip install ragas datasets")
        return

    from graph_builder import build_graph
    graph = build_graph()

    records = []
    total = len(dataset)

    for i, row in enumerate(dataset, start=1):
        question = row["question"]
        reference = row["answer"]
        _progress(i, total, question)

        state = graph.invoke({"original_query": question})
        answer = state.get("final_answer", "")
        contexts = [c.text for c in state.get("reranked_hits", [])]

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": reference,
        })

    hf_dataset = HFDataset.from_list(records)
    print("\n  Running RAGAS scoring (this calls the LLM for each row)…")

    result = ragas_evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_correctness, context_recall],
    )

    df = result.to_pandas()
    df["question"] = [r["question"] for r in records]
    df["ground_truth"] = [r["ground_truth"] for r in records]

    means = df[["faithfulness", "answer_correctness", "context_recall"]].mean()
    print(f"\n  Faithfulness       : {means['faithfulness']:.4f}")
    print(f"  Answer Correctness : {means['answer_correctness']:.4f}")
    print(f"  Context Recall     : {means['context_recall']:.4f}")

    df.to_csv(out, index=False)
    print(f"  → saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — CRAG routing analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_routing(dataset: list[dict], out: str) -> None:
    print("\n=== Layer 3: CRAG routing analysis ===")

    from graph_builder import build_graph
    graph = build_graph()

    rows = []
    verdict_counts: dict[str, int] = {"sufficient": 0, "retry": 0, "defer": 0}
    score_sum = 0.0
    total = len(dataset)

    for i, row in enumerate(dataset, start=1):
        question = row["question"]
        _progress(i, total, question)

        state = graph.invoke({"original_query": question})
        verdict = state.get("crag_verdict", "unknown")
        score = state.get("context_quality", 0.0)
        retries = state.get("crag_retries", 0)
        react_iters = state.get("react_iterations", 0)

        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        score_sum += score

        rows.append({
            "question": question,
            "crag_score": round(score, 4),
            "crag_verdict": verdict,
            "crag_retries": retries,
            "react_iterations": react_iters,
            "deferred": verdict == "defer",
        })

    n = len(rows)
    print(f"\n  Avg CRAG score   : {score_sum/n:.4f}")
    print(f"  Verdict counts   : {verdict_counts}")
    print(f"  Deferral rate    : {verdict_counts.get('defer', 0)/n:.1%}")
    print(f"  Retry rate       : {verdict_counts.get('retry', 0)/n:.1%}")
    _write_csv(out, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Context-engineering ablation sweep
# ─────────────────────────────────────────────────────────────────────────────

# Signals toggled via state["ablation"] (see agents/ablation.py).
_SWEEP_SIGNALS = [
    "metadata_filter", "rerank_boost", "memory",
    "reuse_gate", "outcome_modulator", "metadata_in_generator",
]


def _sweep_configs() -> list[tuple[str, dict]]:
    """baseline (all off), full (all on), and leave-one-out from full."""
    configs: list[tuple[str, dict]] = [
        ("baseline", {s: False for s in _SWEEP_SIGNALS}),
        ("full",     {}),  # empty → all config defaults (on)
    ]
    for s in _SWEEP_SIGNALS:
        configs.append((f"-{s}", {s: False}))  # full minus one signal
    return configs


def run_sweep(dataset: list[dict], k: int, out: str) -> None:
    print(f"\n=== Layer 4: Context-engineering ablation sweep  (k={k}) ===")

    from graph_builder import build_graph
    from agents import llm

    graph = build_graph()  # build once; Option B toggles signals via state

    # Warm up once so model-load cost does not pollute the first config's latency.
    if dataset:
        print("  warming up (1 throwaway query)…")
        try:
            graph.invoke({"original_query": dataset[0]["question"]})
        except Exception as exc:
            print(f"  [warn] warmup failed: {exc}")

    summary_rows: list[dict] = []
    configs = _sweep_configs()

    for name, ablation in configs:
        print(f"\n  ── config: {name}  ablation={ablation or '(full defaults)'}")
        p_sum = r_sum = mrr_sum = ndcg_sum = 0.0
        lat_sum = calls_sum = ptok_sum = ctok_sum = 0.0
        n = 0

        for i, row in enumerate(dataset, start=1):
            question = row["question"]
            relevant_ids = _parse_article_ids(row.get("article_ids", ""))
            _progress(i, len(dataset), question)

            llm.reset_metrics()
            t0 = time.perf_counter()
            state = graph.invoke({"original_query": question, "ablation": ablation})
            latency = time.perf_counter() - t0
            m = llm.get_metrics()

            retrieved_ids = _retrieved_doc_ids(state.get("reranked_hits", []))
            topk = retrieved_ids[:k]
            hits = len(set(topk) & relevant_ids)
            p_sum += hits / k if k else 0.0
            r_sum += hits / len(relevant_ids) if relevant_ids else 0.0
            mrr_sum += _mrr(retrieved_ids, relevant_ids)
            ndcg_sum += _ndcg(retrieved_ids, relevant_ids, k)
            lat_sum += latency
            calls_sum += m["llm_calls"]
            ptok_sum += m["prompt_tokens_est"]
            ctok_sum += m["completion_tokens_est"]
            n += 1

        n = max(n, 1)
        rowd = {
            "config":               name,
            f"precision@{k}":       round(p_sum / n, 4),
            f"recall@{k}":          round(r_sum / n, 4),
            "mrr":                  round(mrr_sum / n, 4),
            f"ndcg@{k}":            round(ndcg_sum / n, 4),
            "latency_s":            round(lat_sum / n, 3),
            "llm_calls":            round(calls_sum / n, 2),
            "prompt_tokens_est":    round(ptok_sum / n),
            "completion_tokens_est": round(ctok_sum / n),
        }
        summary_rows.append(rowd)
        print(f"    → P@{k}={rowd[f'precision@{k}']}  R@{k}={rowd[f'recall@{k}']}  "
              f"MRR={rowd['mrr']}  NDCG@{k}={rowd[f'ndcg@{k}']}  "
              f"lat={rowd['latency_s']}s  calls={rowd['llm_calls']}")

    print("\n  Ablation summary (mean per query):")
    for rowd in summary_rows:
        print(f"    {rowd['config']:<24} P@{k}={rowd[f'precision@{k}']:<7} "
              f"NDCG@{k}={rowd[f'ndcg@{k}']:<7} lat={rowd['latency_s']:<7} "
              f"calls={rowd['llm_calls']}")
    _write_csv(out, summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _default_out(layer: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = "csv"
    return str(ROOT / "evaluate" / f"results_{layer}_{ts}.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Graph-RAG prototype across three layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--layer",
        choices=["retrieval", "ragas", "routing", "sweep", "all"],
        required=True,
        help="Which evaluation layer to run. 'sweep' = context-engineering ablation.",
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "wixqa_expertwritten.csv"),
        help="Path to the ground-truth CSV.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k cutoff for retrieval metrics (layer 1).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to evaluate/results_<layer>_<timestamp>.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N rows (smoke-test mode).",
    )
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset, args.limit)
    print(f"Loaded {len(dataset)} rows from {args.dataset}")

    layers = ["retrieval", "ragas", "routing"] if args.layer == "all" else [args.layer]

    for layer in layers:
        out = args.out if args.out and len(layers) == 1 else _default_out(layer)
        if layer == "retrieval":
            run_retrieval(dataset, args.k, out)
        elif layer == "ragas":
            run_ragas(dataset, out)
        elif layer == "routing":
            run_routing(dataset, out)
        elif layer == "sweep":
            run_sweep(dataset, args.k, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
