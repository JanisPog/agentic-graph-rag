"""WixQA benchmark runner — one pass over a WixQA subset with detailed logging.

Runs every question through the full LangGraph pipeline a single time and, from
that one run, reports four families of metrics together:

  • Retrieval   Precision@k, Recall@k, MRR, NDCG@k   (needs article_ids grounding)
  • Answer      confidence, answer length, citation count
  • Routing     CRAG verdict + score, retries, ReAct iterations, deferral rate
  • Cost        latency, LLM calls, estimated prompt/completion tokens

With --paper-metrics, the WixQA paper suite (§4.3) is added:
  • F1, BLEU, ROUGE-1, ROUGE-2  (lexical, generated vs. gold)
  • Factuality      LLM judge (Appendix C) — answer vs. gold, 0–1
  • Context Recall  LLM judge (Appendix D) — retrieved context vs. gold, 0–1

Every query is logged in detail; a full aggregate summary is logged at the end.
Per-query rows are written to a CSV and the aggregate to a JSON sidecar.

Which dataset is used is configurable (in priority order):
    1. --dataset PATH
    2. $WIXQA_DATASET
    3. data/wixqa_expertwritten.csv   (default)

Usage::

    py -m evaluate.wixqa
    py -m evaluate.wixqa --dataset data/wixqa_synthetic.csv --k 5
    WIXQA_DATASET=data/wixqa_synthetic.csv py -m evaluate.wixqa --limit 20
    py -m evaluate.wixqa --ragas                 # add LLM-judged answer quality
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers so the benchmark output stays readable.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers",
               "filelock", "neo4j", "neo4j.pool", "neo4j.io"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("wixqa")

# Reuse the existing metric helpers — do NOT duplicate or modify them.
from evaluate.eval import _parse_article_ids, _mrr, _ndcg  # noqa: E402


def _resolve_dataset(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env = os.getenv("WIXQA_DATASET")
    if env:
        return env
    return str(ROOT / "data" / "wixqa_expertwritten.csv")


# Fraction of rows (from the top) ingested as the "ticket" layer; the remainder
# is the held-out test set. Must match the ingest-side --limit used to load the
# train split, so no test question's QA pair is present in the graph.
TRAIN_FRACTION = 0.8


def _load_dataset(path: str, limit: int | None, split: str = "test") -> list[dict]:
    """Load WixQA rows, optionally restricted to a positional train/test split.

    The split is deterministic by row order (no shuffling): the first
    TRAIN_FRACTION of rows are the 'train' rows ingested as tickets, the rest are
    the held-out 'test' queries. `--limit` is applied AFTER slicing so the split
    boundary is stable regardless of limit.
    """
    all_rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    n_train = int(len(all_rows) * TRAIN_FRACTION)
    if split == "train":
        rows = all_rows[:n_train]
    elif split == "test":
        rows = all_rows[n_train:]
    else:  # "all"
        rows = all_rows

    if limit:
        rows = rows[:limit]
    return rows


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _doc_id(chunk_id: str) -> str:
    """Map a chunk id ('{doc_id}_{index}') back to its parent document id.

    The benchmark grounds on document/article ids, but retrievers return chunk
    ids carrying a trailing _<index>. Only a trailing all-digit segment is
    stripped, so bare document ids (64-char hex hashes, numeric ids) are
    unaffected.
    """
    head, sep, tail = chunk_id.rpartition("_")
    return head if sep and tail.isdigit() else chunk_id


def run(dataset_path: str, k: int, limit: int | None, out_csv: str,
        do_ragas: bool, do_paper: bool, split: str) -> None:
    rows = _load_dataset(dataset_path, limit, split)
    if not rows:
        log.error("No rows loaded from %s (split=%s)", dataset_path, split)
        return

    has_grounding = any(_parse_article_ids(r.get("article_ids", "")) for r in rows)

    log.info("═" * 70)
    log.info("WixQA benchmark")
    log.info("  dataset       : %s", dataset_path)
    log.info("  split         : %s  (train fraction=%.0f%%, held-out test = last %.0f%%)",
             split, TRAIN_FRACTION * 100, (1 - TRAIN_FRACTION) * 100)
    log.info("  rows          : %d%s", len(rows), f" (limited from CLI)" if limit else "")
    log.info("  top-k         : %d", k)
    log.info("  grounding     : %s", "present → retrieval metrics enabled"
             if has_grounding else "ABSENT → retrieval metrics will be 0.0")
    log.info("  ragas         : %s", "on" if do_ragas else "off")
    log.info("  paper metrics : %s", "on (F1/BLEU/ROUGE + Factuality + Context Recall judges)"
             if do_paper else "off")
    if do_paper:
        import config as _cfg
        _judge = (_cfg.JUDGE_MODEL or "").strip() or f"{_cfg.LITELLM_CHAT_MODEL} (== generator; self-judged)"
        log.info("  judge model   : %s", _judge)
    log.info("═" * 70)

    if do_paper:
        from evaluate import paper_metrics

    from graph_builder import build_graph
    from agents import llm
    graph = build_graph()

    # Accumulators
    agg = {
        "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
        "confidence": 0.0, "answer_len": 0.0, "crag_score": 0.0,
        "latency": 0.0, "llm_calls": 0.0, "ptok": 0.0, "ctok": 0.0,
    }
    verdict_counts: dict[str, int] = {}
    # Retrieval means are computed only over queries where retrieval actually ran
    # AND gold grounding exists. Fast-track HARD SKIP bypasses retrieval, so those
    # queries are excluded from Precision/Recall/MRR/NDCG (counted separately).
    n_retrieval_scored = 0
    n_fasttrack_skip = 0
    per_query: list[dict] = []
    ragas_records: list[dict] = []

    # Paper-metric accumulators (sum + count; judges may return None on failure)
    paper_agg = {"f1": 0.0, "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0}
    fact_sum, fact_n = 0.0, 0
    cr_sum, cr_n = 0.0, 0

    total = len(rows)
    t_start = time.perf_counter()

    for i, row in enumerate(rows, start=1):
        question = row.get("question", "")
        reference = row.get("answer", "")
        relevant_ids = _parse_article_ids(row.get("article_ids", ""))

        llm.reset_metrics()
        t0 = time.perf_counter()
        try:
            state = graph.invoke({"original_query": question})
        except Exception as exc:  # noqa: BLE001
            log.exception("  [%d/%d] pipeline error: %s", i, total, exc)
            continue
        latency = time.perf_counter() - t0
        m = llm.get_metrics()

        reranked = state.get("reranked_hits", []) or []
        # Map chunk ids → parent document ids and dedupe to document level: the
        # benchmark grounds on article/document ids and the paper's top-k is over
        # documents, while retrievers return chunk ids ({doc_id}_{index}).
        retrieved_ids: list[str] = []
        _seen_docs: set[str] = set()
        for c in reranked:
            d = _doc_id(c.id)
            if d not in _seen_docs:
                _seen_docs.add(d)
                retrieved_ids.append(d)
        topk = retrieved_ids[:k]
        hits = set(topk) & relevant_ids

        precision = len(hits) / k if k else 0.0
        recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
        mrr = _mrr(retrieved_ids, relevant_ids) if relevant_ids else 0.0
        ndcg = _ndcg(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0

        answer = state.get("final_answer", "") or ""
        confidence = float(state.get("confidence", 0.0) or 0.0)
        verdict = state.get("crag_verdict", "unknown") or "unknown"
        crag_score = float(state.get("context_quality", 0.0) or 0.0)
        retries = int(state.get("crag_retries", 0) or 0)
        react_iters = int(state.get("react_iterations", 0) or 0)
        deferred = bool(state.get("deferral_reason")) or verdict == "defer"
        # Fast-track HARD SKIP bypasses the retrieval pipeline, so reranked_hits is
        # the matched ticket resolution, not retrieved KB docs. Retrieval metrics
        # are meaningless for these and would unfairly drag down Recall/MRR.
        fasttrack_skip = bool(state.get("fasttrack_hard_skip"))
        retrieval_scored = bool(relevant_ids) and not fasttrack_skip

        # ── Report instrumentation (Q1–4): which arms ran, EVOLVE strategy, graph use ──
        plan = state.get("plan") or {}
        arms = [name for flag, name in (
            ("use_semantic", "semantic"), ("use_bm25", "bm25"),
            ("use_graph", "graph"), ("use_surface_form", "surface_form"),
            ("use_web_search", "web"),
        ) if plan.get(flag)]
        arms_used = "+".join(arms) if arms else "(none)"
        plan_intent = plan.get("intent", "")
        evolve_strategy = state.get("evolve_strategy", "") or ""
        tried_strategies = "+".join(state.get("tried_strategies") or [])
        graph_seeds = int(state.get("graph_seeds_total", 0) or 0)
        graph_entity_seeds = int(state.get("graph_entity_seeds", 0) or 0)  # surface-form seeds
        graph_nodes = int(state.get("graph_nodes_returned", 0) or 0)

        # ── RQ1/RQ2 instrumentation: planner decisions + INSPECT routing explanation ──
        # inspect_reason explains WHY INSPECT skipped the retry loop (monotonic rule).
        inspect_reason = state.get("inspect_reason", "") or ""
        # Plan-level fields that drove the retrieval arm selection and loop budget.
        plan_skip_reflact   = bool(plan.get("skip_reflact", False))
        plan_react_budget   = int(plan.get("react_budget", 0) or 0)
        plan_conf_target    = float(plan.get("confidence_target", 0.0) or 0.0)
        # Arms the planner explicitly disabled (arms present in ALL_ARMS but absent from plan).
        _all_arm_flags = {"use_semantic", "use_bm25", "use_graph", "use_surface_form", "use_web_search"}
        _arm_label = {"use_semantic": "semantic", "use_bm25": "bm25", "use_graph": "graph",
                      "use_surface_form": "surface_form", "use_web_search": "web"}
        arms_skipped = "+".join(
            _arm_label[f] for f in sorted(_all_arm_flags) if not plan.get(f)
        ) or "(none)"

        # Accumulate
        agg["confidence"] += confidence
        agg["answer_len"] += len(answer)
        agg["crag_score"] += crag_score
        agg["latency"] += latency
        agg["llm_calls"] += m["llm_calls"]
        agg["ptok"] += m["prompt_tokens_est"]
        agg["ctok"] += m["completion_tokens_est"]
        if fasttrack_skip:
            n_fasttrack_skip += 1
        if retrieval_scored:
            agg["precision"] += precision
            agg["recall"] += recall
            agg["mrr"] += mrr
            agg["ndcg"] += ndcg
            n_retrieval_scored += 1
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Paper metrics (computed AFTER the cost snapshot so judge LLM calls do
        # not inflate the pipeline cost numbers for this query).
        lex = None
        fact = cr = {"score": None}
        if do_paper:
            lex = paper_metrics.lexical_metrics(answer, reference)
            for key in paper_agg:
                paper_agg[key] += lex[key]
            fact = paper_metrics.judge_factuality(question, answer, reference)
            context_text = "\n\n".join(c.text for c in reranked)
            cr = paper_metrics.judge_context_recall(question, context_text, reference)
            if fact["score"] is not None:
                fact_sum += fact["score"]; fact_n += 1
            if cr["score"] is not None:
                cr_sum += cr["score"]; cr_n += 1

        # Detailed per-query log
        log.info("─" * 70)
        log.info("[%d/%d] Q: %s", i, total, _truncate(question, 96))
        if fasttrack_skip:
            log.info("       retrieval  SKIPPED (fast-track hard skip — excluded from retrieval means)")
        elif relevant_ids:
            log.info("       retrieval  P@%d=%.3f  R@%d=%.3f  MRR=%.3f  NDCG@%d=%.3f  (%d/%d gold hit)",
                     k, precision, k, recall, mrr, k, ndcg, len(hits), len(relevant_ids))
            missed = relevant_ids - hits
            if missed:
                log.info("       missed gold: %s", ", ".join(_truncate(x, 16) for x in list(missed)[:5]))
        else:
            log.info("       retrieval  (no gold article_ids for this row)")
        log.info("       routing    verdict=%s  crag=%.3f  retries=%d  react_iters=%d  deferred=%s",
                 verdict, crag_score, retries, react_iters, deferred)
        log.info("       answer     conf=%.2f  len=%d  cited=%d",
                 confidence, len(answer), len(state.get("cited_chunk_ids", []) or []))
        log.info("       cost       latency=%.2fs  llm_calls=%d  ptok≈%d  ctok≈%d",
                 latency, m["llm_calls"], m["prompt_tokens_est"], m["completion_tokens_est"])
        if do_paper and lex is not None:
            _fact = "n/a" if fact["score"] is None else f"{fact['score']:.2f}"
            _cr = "n/a" if cr["score"] is None else f"{cr['score']:.2f}"
            log.info("       paper      F1=%.3f BLEU=%.3f R1=%.3f R2=%.3f | Factuality=%s ContextRecall=%s",
                     lex["f1"], lex["bleu"], lex["rouge1"], lex["rouge2"], _fact, _cr)
            if fact["explanation"]:
                log.info("       factuality: %s", _truncate(fact["explanation"], 160))
        log.info("       A: %s", _truncate(answer, 200))

        row_out = {
            "idx": i,
            "question": question,
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            "mrr": round(mrr, 4),
            f"ndcg@{k}": round(ndcg, 4),
            "n_relevant": len(relevant_ids),
            "n_hit": len(hits),
            "crag_verdict": verdict,
            "crag_score": round(crag_score, 4),
            "crag_retries": retries,
            "react_iterations": react_iters,
            "deferred": deferred,
            "fasttrack_skip": fasttrack_skip,
            "retrieval_scored": retrieval_scored,
            # Report instrumentation (Q1–4)
            "plan_intent": plan_intent,
            "arms_used": arms_used,
            "evolve_strategy": evolve_strategy,
            "tried_strategies": tried_strategies,
            "graph_seeds": graph_seeds,
            "graph_entity_seeds": graph_entity_seeds,
            "graph_nodes": graph_nodes,
            "confidence": round(confidence, 4),
            "answer_len": len(answer),
            "latency_s": round(latency, 3),
            "llm_calls": m["llm_calls"],
            "prompt_tokens_est": m["prompt_tokens_est"],
            "completion_tokens_est": m["completion_tokens_est"],
            # RQ1/RQ2: planner decisions + INSPECT routing explanation
            "inspect_reason":     inspect_reason,
            "plan_skip_reflact":  plan_skip_reflact,
            "plan_react_budget":  plan_react_budget,
            "plan_conf_target":   round(plan_conf_target, 4),
            "arms_skipped":       arms_skipped,
            "relevant_ids": "|".join(relevant_ids),
            "retrieved_topk": "|".join(topk),
            # Full generated answer text — kept so a result CSV is self-diagnosing
            # (lets you see WHAT the system said for a low-Factuality row, not just
            # the score). Newlines flattened so the CSV cell stays single-line.
            "answer": (answer or "").replace("\r", " ").replace("\n", " ").strip(),
        }
        if do_paper and lex is not None:
            row_out.update({
                "paper_f1":             round(lex["f1"], 4),
                "paper_bleu":           round(lex["bleu"], 4),
                "paper_rouge1":         round(lex["rouge1"], 4),
                "paper_rouge2":         round(lex["rouge2"], 4),
                "paper_factuality":     "" if fact["score"] is None else round(fact["score"], 4),
                "paper_context_recall": "" if cr["score"] is None else round(cr["score"], 4),
                # Judge rationales — the WHY behind each score. Without these a low
                # Factuality row is undiagnosable from the CSV alone.
                "paper_factuality_explanation":     (fact.get("explanation") or "").replace("\n", " ").strip(),
                "paper_context_recall_explanation": (cr.get("explanation") or "").replace("\n", " ").strip(),
            })
        per_query.append(row_out)

        if do_ragas:
            ragas_records.append({
                "question": question,
                "answer": answer,
                "contexts": [c.text for c in reranked],
                "ground_truth": reference,
            })

    n = len(per_query)
    if n == 0:
        log.error("No queries completed successfully.")
        return
    rs = max(n_retrieval_scored, 1)   # retrieval-metric denominator
    wall = time.perf_counter() - t_start

    # ── Optional RAGAS layer ──────────────────────────────────────────────────
    ragas_means: dict[str, float] = {}
    if do_ragas and ragas_records:
        ragas_means = _run_ragas(ragas_records)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary = {
        "dataset": dataset_path,
        "rows_evaluated": n,
        "retrieval_scored": n_retrieval_scored,
        "fasttrack_skips": n_fasttrack_skip,
        "fasttrack_hit_rate": round(n_fasttrack_skip / n, 4),
        "k": k,
        "wall_time_s": round(wall, 1),
        f"precision@{k}": round(agg["precision"] / rs, 4),
        f"recall@{k}": round(agg["recall"] / rs, 4),
        "mrr": round(agg["mrr"] / rs, 4),
        f"ndcg@{k}": round(agg["ndcg"] / rs, 4),
        "avg_confidence": round(agg["confidence"] / n, 4),
        "avg_answer_len": round(agg["answer_len"] / n, 1),
        "avg_crag_score": round(agg["crag_score"] / n, 4),
        "deferral_rate": round(verdict_counts.get("defer", 0) / n, 4),
        "verdict_distribution": verdict_counts,
        "avg_latency_s": round(agg["latency"] / n, 3),
        "avg_llm_calls": round(agg["llm_calls"] / n, 2),
        "avg_prompt_tokens_est": round(agg["ptok"] / n),
        "avg_completion_tokens_est": round(agg["ctok"] / n),
    }
    if ragas_means:
        summary.update({f"ragas_{k2}": round(v, 4) for k2, v in ragas_means.items()})
    if do_paper:
        summary.update({
            "paper_f1":      round(paper_agg["f1"] / n, 4),
            "paper_bleu":    round(paper_agg["bleu"] / n, 4),
            "paper_rouge1":  round(paper_agg["rouge1"] / n, 4),
            "paper_rouge2":  round(paper_agg["rouge2"] / n, 4),
            "paper_factuality":     round(fact_sum / fact_n, 4) if fact_n else None,
            "paper_context_recall": round(cr_sum / cr_n, 4) if cr_n else None,
            "paper_factuality_scored":     fact_n,
            "paper_context_recall_scored": cr_n,
        })

    log.info("═" * 70)
    log.info("WixQA RESULTS — %s", os.path.basename(dataset_path))
    log.info("  queries evaluated     : %d", n)
    log.info("  wall time             : %.1fs", wall)
    log.info("  ── Retrieval (means over %d retrieval-scored queries) ─────", n_retrieval_scored)
    log.info("  Precision@%d           : %.4f", k, summary[f"precision@{k}"])
    log.info("  Recall@%d              : %.4f", k, summary[f"recall@{k}"])
    log.info("  MRR                   : %.4f", summary["mrr"])
    log.info("  NDCG@%d               : %.4f", k, summary[f"ndcg@{k}"])
    log.info("  Fast-track skips      : %d/%d (%.1f%%) — excluded from retrieval means",
             n_fasttrack_skip, n, summary["fasttrack_hit_rate"] * 100)
    log.info("  ── Answer ────────────────────────────────────────")
    log.info("  Avg confidence        : %.4f", summary["avg_confidence"])
    log.info("  Avg answer length     : %.1f chars", summary["avg_answer_len"])
    if do_paper:
        log.info("  ── Paper metrics (WixQA §4.3) ────────────────────")
        log.info("  F1                    : %.4f", summary["paper_f1"])
        log.info("  BLEU                  : %.4f", summary["paper_bleu"])
        log.info("  ROUGE-1               : %.4f", summary["paper_rouge1"])
        log.info("  ROUGE-2               : %.4f", summary["paper_rouge2"])
        _f = summary["paper_factuality"]
        _c = summary["paper_context_recall"]
        log.info("  Factuality (LLM judge): %s  (%d/%d scored)",
                 "n/a" if _f is None else f"{_f:.4f}", fact_n, n)
        log.info("  Context Recall (judge): %s  (%d/%d scored)",
                 "n/a" if _c is None else f"{_c:.4f}", cr_n, n)
    if ragas_means:
        log.info("  ── RAGAS (LLM-judged) ────────────────────────────")
        for k2, v in ragas_means.items():
            log.info("  %-21s : %.4f", k2, v)
    log.info("  ── Routing ───────────────────────────────────────")
    log.info("  Avg CRAG score        : %.4f", summary["avg_crag_score"])
    log.info("  Verdict distribution  : %s", verdict_counts)
    log.info("  Deferral rate         : %.1f%%", summary["deferral_rate"] * 100)
    log.info("  ── Cost (means) ──────────────────────────────────")
    log.info("  Latency / query       : %.3fs", summary["avg_latency_s"])
    log.info("  LLM calls / query      : %.2f", summary["avg_llm_calls"])
    log.info("  Tokens / query (p/c)  : ≈%d / ≈%d",
             summary["avg_prompt_tokens_est"], summary["avg_completion_tokens_est"])
    log.info("═" * 70)

    # ── Write outputs ─────────────────────────────────────────────────────────
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        w.writeheader()
        w.writerows(per_query)
    summary_path = str(Path(out_csv).with_suffix(".summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Per-query results → %s", out_csv)
    log.info("Summary           → %s", summary_path)


def _run_ragas(records: list[dict]) -> dict[str, float]:
    """Score answer quality with RAGAS (faithfulness / correctness / context recall)."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_correctness, context_recall
        from datasets import Dataset as HFDataset
    except ImportError:
        log.warning("ragas/datasets not installed — skipping RAGAS layer "
                    "(pip install ragas datasets)")
        return {}

    log.info("Running RAGAS scoring over %d records (LLM calls per row)…", len(records))
    hf = HFDataset.from_list(records)
    result = ragas_evaluate(hf, metrics=[faithfulness, answer_correctness, context_recall])
    df = result.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_correctness": float(df["answer_correctness"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the WixQA benchmark with detailed logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=None,
                        help="CSV path. Overrides $WIXQA_DATASET. "
                             "Default: data/wixqa_expertwritten.csv")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test",
                        help="Row partition to evaluate. 'test' (default) = held-out last "
                             f"{int((1 - TRAIN_FRACTION) * 100)}%% not ingested as tickets; "
                             "'train' = first 80%%; 'all' = whole file (legacy, contaminated).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N rows of the chosen split.")
    parser.add_argument("--out", default=None, help="Per-query CSV output path.")
    parser.add_argument("--ragas", action="store_true",
                        help="Also compute LLM-judged RAGAS answer quality.")
    parser.add_argument("--paper-metrics", action="store_true",
                        help="Also compute the WixQA paper metrics (§4.3): F1, BLEU, "
                             "ROUGE-1/2 + Factuality and Context Recall LLM judges "
                             "(2 extra LLM calls per query).")
    args = parser.parse_args()

    dataset_path = _resolve_dataset(args.dataset)
    if not Path(dataset_path).exists():
        log.error("Dataset not found: %s", dataset_path)
        log.error("Download it first, e.g.: py etl/download_wixqa.py --subset synthetic")
        sys.exit(1)

    if args.out:
        out_csv = args.out
    else:
        stem = Path(dataset_path).stem
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = str(ROOT / "evaluate" / f"results_{stem}_{ts}.csv")

    run(dataset_path, args.k, args.limit, out_csv, args.ragas, args.paper_metrics, args.split)


if __name__ == "__main__":
    main()
