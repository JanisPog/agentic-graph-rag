"""Single-shot RAG baseline — the ablation arm for the agentic pipeline.

This is the control for the scientific claim "the agentic Graph-RAG pipeline is
what delivers the performance." It runs the **same Qwen model** against the
**same Neo4j KB** with the **same embedding index** and is scored by the **same
judge** (config.JUDGE_MODEL) as evaluate/wixqa.py — only the *architecture*
differs:

    full pipeline  : decompose → expand/HyDE → semantic+BM25+graph+web fan-out →
                     RRF fusion → rerank → outcome-modulate → CRAG → PIVOT
                     (INSPECT/EVOLVE/ReAct/VERIFY) → cited generation
    single-shot    : ONE semantic ANN pass → stuff top-k chunks → ONE generation

So everything between retrieval and generation — the agentic orchestration layer
(query understanding, multi-arm fusion, reranking, CRAG quality-gating, the PIVOT
reasoning loop) — is ablated. If single-shot scores far lower on the same split
with the same judge, that delta is the orchestration layer's contribution.

This file imports existing helpers (dataset loading, grounding parsing, retrieval
metrics, the ANN query, the paper-metric judges) but MODIFIES NOTHING. The full
pipeline run path (evaluate/wixqa.py, graph_builder, agents/*) is untouched.

The CLI mirrors evaluate/wixqa.py (--dataset/--k/--split/--limit/--paper-metrics/
--ragas) and writes the same per-query CSV + .summary.json sidecar, so a
single-shot summary diffs directly against a pipeline summary.

Usage::

    py -m evaluate.single_shot --paper-metrics
    py -m evaluate.single_shot --dataset data/wixqa_synthetic.csv --k 5 --paper-metrics
    py -m evaluate.single_shot --limit 20 --ragas
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers",
               "filelock", "neo4j", "neo4j.pool", "neo4j.io"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("single_shot")

# Reuse the existing helpers verbatim — do NOT duplicate or modify them. The
# dataset loader, split logic, truncation, doc-id mapping and retrieval metrics
# are shared with the pipeline runner so both arms see identical inputs/scoring.
from evaluate.eval import _parse_article_ids, _mrr, _ndcg  # noqa: E402
from evaluate.wixqa import (  # noqa: E402
    _load_dataset, _truncate, _doc_id, _resolve_dataset, TRAIN_FRACTION,
)


# Minimal single-shot generation prompt. Deliberately NOT the pipeline's
# generator prompt: the baseline must be a plain "answer from this context" call
# with no pipeline-specific scaffolding (no partial-answer directives, no
# reasoning trace, no reflection summary). Citation format mirrors the pipeline's
# so the generated answers are comparable in shape, but nothing here adds agentic
# behaviour.
_PROMPT = """\
You are a technical support specialist. Answer the user's query using ONLY the
provided context passages. Do not introduce external knowledge. If the context
gives a step-by-step procedure, reproduce all of its steps in order with their
specific UI labels, field names, and links. Stay scoped to the query — do not add
background, caveats, or extra procedures the user did not ask for.

For each factual claim, cite the passage it came from using [P<n>] format.

USER QUERY:
{question}

CONTEXT PASSAGES:
{passages}

Respond in this JSON structure:
{{
  "answer": "<full answer text with inline citations [P1], [P2], ...>",
  "cited_passage_ids": ["<chunk_id>"],
  "confidence": <float 0.0-1.0>
}}"""


def _fmt_passages(chunks: list) -> str:
    """Plain passage formatting — id header + text, no metadata injection.

    Metadata injection is one of the pipeline's studied features; the single-shot
    baseline deliberately does not use it.
    """
    if not chunks:
        return "(no context passages retrieved)"
    return "\n\n---\n\n".join(
        f"[P{i + 1}] (id={c.id})\n{c.text}" for i, c in enumerate(chunks)
    )


def _single_shot(question: str, k: int):
    """One ANN retrieval pass + one generation call.

    Returns (answer, cited_ids, confidence, retrieved_chunks). Retrieval reuses
    the shared run_ann helper against the same chunk_embedding index the pipeline
    uses — no decomposition, expansion, BM25/graph/web arms, fusion, rerank, CRAG
    or PIVOT. The top-k ANN hits are stuffed straight into the generation prompt.
    """
    import config  # noqa: PLC0415
    from agents import llm  # noqa: PLC0415
    from agents.retrieval_utils import run_ann  # noqa: PLC0415
    from graph.neo4j_client import Neo4jClient  # noqa: PLC0415
    from agents.llm import chat_json  # noqa: PLC0415

    # ── Retrieval: single semantic ANN pass on the raw query ──────────────────
    vec = llm.embed_query(question)
    with Neo4jClient() as client:
        chunks = run_ann(client, vec, k)

    passages = _fmt_passages(chunks)
    prompt = _PROMPT.format(question=question, passages=passages)

    # ── Generation: single call, same backend/temperature as the generator ────
    try:
        result = chat_json(prompt, temperature=config.TEMPERATURE_GENERATE)
        answer = result.get("answer", "") or ""
        cited_ids = result.get("cited_passage_ids", []) or []
        confidence = float(result.get("confidence", 0.5))
    except Exception as exc:  # noqa: BLE001
        # Salvage path mirrors the generator's: peel an "answer" field out of a
        # malformed/truncated response rather than dumping raw context.
        import re  # noqa: PLC0415
        log.warning("  single-shot generation failed (%s); using salvage path", exc)
        raw = str(exc)
        m = re.search(r'"answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', raw)
        if m:
            answer = m.group(1).encode().decode("unicode_escape", errors="ignore").strip()
        else:
            answer = ("I'm sorry — I could not compose a structured answer for this "
                      "query. Please try rephrasing or contact a human support agent.")
        cited_ids = [c.id for c in chunks[:5]]
        confidence = 0.3

    return answer, cited_ids, confidence, chunks


def run(dataset_path: str, k: int, limit: int | None, out_csv: str,
        do_ragas: bool, do_paper: bool, split: str) -> None:
    rows = _load_dataset(dataset_path, limit, split)
    if not rows:
        log.error("No rows loaded from %s (split=%s)", dataset_path, split)
        return

    has_grounding = any(_parse_article_ids(r.get("article_ids", "")) for r in rows)

    log.info("═" * 70)
    log.info("SINGLE-SHOT RAG baseline  (ablation arm — no agentic pipeline)")
    log.info("  dataset       : %s", dataset_path)
    log.info("  split         : %s  (train fraction=%.0f%%, held-out test = last %.0f%%)",
             split, TRAIN_FRACTION * 100, (1 - TRAIN_FRACTION) * 100)
    log.info("  rows          : %d%s", len(rows), " (limited from CLI)" if limit else "")
    log.info("  top-k         : %d  (chunks stuffed into one generation call)", k)
    log.info("  grounding     : %s", "present → retrieval metrics enabled"
             if has_grounding else "ABSENT → retrieval metrics will be 0.0")
    log.info("  ragas         : %s", "on" if do_ragas else "off")
    log.info("  paper metrics : %s", "on (F1/BLEU/ROUGE + Factuality + Context Recall judges)"
             if do_paper else "off")
    if do_paper:
        import config as _cfg  # noqa: PLC0415
        _judge = (_cfg.JUDGE_MODEL or "").strip() or f"{_cfg.LITELLM_CHAT_MODEL} (== generator; self-judged)"
        log.info("  judge model   : %s", _judge)
    log.info("═" * 70)

    if do_paper:
        from evaluate import paper_metrics  # noqa: PLC0415
    from agents import llm  # noqa: PLC0415

    # Accumulators — same shape as the pipeline runner so summaries align.
    agg = {
        "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
        "confidence": 0.0, "answer_len": 0.0,
        "latency": 0.0, "llm_calls": 0.0, "ptok": 0.0, "ctok": 0.0,
    }
    n_retrieval_scored = 0
    per_query: list[dict] = []
    ragas_records: list[dict] = []

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
            answer, cited_ids, confidence, chunks = _single_shot(question, k)
        except Exception as exc:  # noqa: BLE001
            log.exception("  [%d/%d] single-shot error: %s", i, total, exc)
            continue
        latency = time.perf_counter() - t0
        m = llm.get_metrics()

        # Map chunk ids → parent document ids and dedupe to document level (same
        # as the pipeline runner: grounding + paper top-k are over documents).
        retrieved_ids: list[str] = []
        _seen_docs: set[str] = set()
        for c in chunks:
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
        retrieval_scored = bool(relevant_ids)

        agg["confidence"] += confidence
        agg["answer_len"] += len(answer)
        agg["latency"] += latency
        agg["llm_calls"] += m["llm_calls"]
        agg["ptok"] += m["prompt_tokens_est"]
        agg["ctok"] += m["completion_tokens_est"]
        if retrieval_scored:
            agg["precision"] += precision
            agg["recall"] += recall
            agg["mrr"] += mrr
            agg["ndcg"] += ndcg
            n_retrieval_scored += 1

        # Paper metrics — AFTER the cost snapshot so judge calls don't inflate
        # the baseline's reported cost. Same judge functions as the pipeline run.
        lex = None
        fact = cr = {"score": None, "explanation": ""}
        if do_paper:
            lex = paper_metrics.lexical_metrics(answer, reference)
            for key in paper_agg:
                paper_agg[key] += lex[key]
            fact = paper_metrics.judge_factuality(question, answer, reference)
            context_text = "\n\n".join(c.text for c in chunks)
            cr = paper_metrics.judge_context_recall(question, context_text, reference)
            if fact["score"] is not None:
                fact_sum += fact["score"]; fact_n += 1
            if cr["score"] is not None:
                cr_sum += cr["score"]; cr_n += 1

        log.info("─" * 70)
        log.info("[%d/%d] Q: %s", i, total, _truncate(question, 96))
        if relevant_ids:
            log.info("       retrieval  P@%d=%.3f  R@%d=%.3f  MRR=%.3f  NDCG@%d=%.3f  (%d/%d gold hit)",
                     k, precision, k, recall, mrr, k, ndcg, len(hits), len(relevant_ids))
            missed = relevant_ids - hits
            if missed:
                log.info("       missed gold: %s", ", ".join(_truncate(x, 16) for x in list(missed)[:5]))
        else:
            log.info("       retrieval  (no gold article_ids for this row)")
        log.info("       answer     conf=%.2f  len=%d  cited=%d",
                 confidence, len(answer), len(cited_ids))
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
            "retrieval_scored": retrieval_scored,
            "confidence": round(confidence, 4),
            "answer_len": len(answer),
            "latency_s": round(latency, 3),
            "llm_calls": m["llm_calls"],
            "prompt_tokens_est": m["prompt_tokens_est"],
            "completion_tokens_est": m["completion_tokens_est"],
            "relevant_ids": "|".join(relevant_ids),
            "retrieved_topk": "|".join(topk),
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
                "paper_factuality_explanation":     (fact.get("explanation") or "").replace("\n", " ").strip(),
                "paper_context_recall_explanation": (cr.get("explanation") or "").replace("\n", " ").strip(),
            })
        per_query.append(row_out)

        if do_ragas:
            ragas_records.append({
                "question": question,
                "answer": answer,
                "contexts": [c.text for c in chunks],
                "ground_truth": reference,
            })

    n = len(per_query)
    if n == 0:
        log.error("No queries completed successfully.")
        return
    rs = max(n_retrieval_scored, 1)
    wall = time.perf_counter() - t_start

    ragas_means: dict[str, float] = {}
    if do_ragas and ragas_records:
        ragas_means = _run_ragas(ragas_records)

    summary = {
        "arm": "single_shot",
        "dataset": dataset_path,
        "rows_evaluated": n,
        "retrieval_scored": n_retrieval_scored,
        "k": k,
        "wall_time_s": round(wall, 1),
        f"precision@{k}": round(agg["precision"] / rs, 4),
        f"recall@{k}": round(agg["recall"] / rs, 4),
        "mrr": round(agg["mrr"] / rs, 4),
        f"ndcg@{k}": round(agg["ndcg"] / rs, 4),
        "avg_confidence": round(agg["confidence"] / n, 4),
        "avg_answer_len": round(agg["answer_len"] / n, 1),
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
    log.info("SINGLE-SHOT RESULTS — %s", os.path.basename(dataset_path))
    log.info("  queries evaluated     : %d", n)
    log.info("  wall time             : %.1fs", wall)
    log.info("  ── Retrieval (means over %d retrieval-scored queries) ─────", n_retrieval_scored)
    log.info("  Precision@%d           : %.4f", k, summary[f"precision@{k}"])
    log.info("  Recall@%d              : %.4f", k, summary[f"recall@{k}"])
    log.info("  MRR                   : %.4f", summary["mrr"])
    log.info("  NDCG@%d               : %.4f", k, summary[f"ndcg@{k}"])
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
    log.info("  ── Cost (means) ──────────────────────────────────")
    log.info("  Latency / query       : %.3fs", summary["avg_latency_s"])
    log.info("  LLM calls / query      : %.2f", summary["avg_llm_calls"])
    log.info("  Tokens / query (p/c)  : ≈%d / ≈%d",
             summary["avg_prompt_tokens_est"], summary["avg_completion_tokens_est"])
    log.info("═" * 70)

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
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import faithfulness, answer_correctness, context_recall  # noqa: PLC0415
        from datasets import Dataset as HFDataset  # noqa: PLC0415
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
        description="Run the single-shot RAG baseline (ablation arm) with detailed logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=None,
                        help="CSV path. Overrides $WIXQA_DATASET. "
                             "Default: data/wixqa_expertwritten.csv")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k chunks retrieved AND stuffed into the generation call.")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test",
                        help="Row partition to evaluate. 'test' (default) = held-out last "
                             f"{int((1 - TRAIN_FRACTION) * 100)}%% not ingested as tickets; "
                             "'train' = first 80%%; 'all' = whole file. Match the pipeline run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N rows of the chosen split.")
    parser.add_argument("--out", default=None, help="Per-query CSV output path.")
    parser.add_argument("--ragas", action="store_true",
                        help="Also compute LLM-judged RAGAS answer quality.")
    parser.add_argument("--paper-metrics", action="store_true",
                        help="Also compute the WixQA paper metrics (§4.3): F1, BLEU, "
                             "ROUGE-1/2 + Factuality and Context Recall LLM judges "
                             "(same judge as the pipeline run).")
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
        out_csv = str(ROOT / "evaluate" / f"results_singleshot_{stem}_{ts}.csv")

    run(dataset_path, args.k, args.limit, out_csv, args.ragas, args.paper_metrics, args.split)


if __name__ == "__main__":
    main()
