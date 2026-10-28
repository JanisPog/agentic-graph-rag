# Appendix A3 — Reproduction

## A3.1 — Datasets & split

| Subset | Source config (HF `Wix/WixQA`) | Size | Grounding |
|---|---|---|---|
| ExpertWritten | `wixqa_expertwritten` | 200 | 1–3 articles (27% multi-article) |
| Simulated | `wixqa_simulated` | 200 | 1–2 articles (14% multi-article) |
| Synthetic | `wixqa_synthetic` | 6,221 | exactly 1 article |
| KB corpus | `wix_kb_corpus` | 6,221 articles | — |

**Train/test split.** Deterministic by row order (no shuffling). The first **80%** of each
subset's rows are the *train* rows, ingested into the graph as tickets (problem/resolution
pairs); the last **20%** are the held-out *test* queries. The split is applied at **read time**
— the dataset CSV files are never modified, for reproducibility. `article_ids` reference KB
article ids (64-char SHA-256) which become `Document.id` at ingest, so grounding aligns with
retrieved chunks' parent documents. Multi-article ids (concatenated/comma-joined) are split
into individual 64-char ids at read time by both the eval grounding parser and the ingest ref
parser — again leaving the files unchanged.

Simulated was additionally run on the **full 200 rows** (`--split all`); this is
contamination-free because no Simulated QA pairs were ingested as tickets or documents.

## A3.2 — Ingest (KB + train-split tickets)

```bash
# Full KB (6,221 articles) — required for Synthetic, canonical for all subsets
docker compose exec graph-rag-agent python etl/ingest.py \
  --primary  data/wixqa_expertwritten.csv \
  --secondary data/wix_kb_corpus.csv --clear
```

## A3.3 — Evaluation (per subset)

```bash
# ExpertWritten, held-out test split, full paper metrics
docker compose exec graph-rag-agent python evaluate/wixqa.py \
  --dataset data/wixqa_expertwritten.csv --split test --paper-metrics

# Synthetic test split (use --limit N for a sample)
docker compose exec graph-rag-agent python evaluate/wixqa.py \
  --dataset data/wixqa_synthetic.csv --split test --limit 500 --paper-metrics

# Simulated, full set (contamination-free: no Simulated tickets ingested)
docker compose exec graph-rag-agent python evaluate/wixqa.py \
  --dataset data/wixqa_simulated.csv --split all --paper-metrics
```

Outputs a per-query CSV (`evaluate/results_<subset>_<ts>.csv`, includes full answer text and
both judge rationales) and an aggregate `*.summary.json`. Behavioural analysis:

```bash
docker compose exec graph-rag-agent python evaluate/analyze_report.py <result.csv>
```

## A3.4 — Model / judge configuration (environment)

| Variable | Value used | Purpose |
|---|---|---|
| `CHAT_BACKEND` | `vllm` / `litellm` | generator + internal LLM calls (CRAG/planner/INSPECT/VERIFY) |
| generator model | `Qwen3.6-35B-A3B` (also 9B baseline) | answer generation |
| embedding model | `Qwen3-embedding-8b` (4096-dim) | dense retrieval; index built with this — must not change |
| reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | rerank; non-LLM |
| `JUDGE_MODEL` | empty (self) or `fhms/gemma-4-31b-it` | paper-metric judges, decoupled from generator |
| top-k | 5 | retrieval depth scored (matches paper) |

> **Backend reproducibility caveat.** The same Qwen3.6-35B-A3B weights served via the LiteLLM
> gateway vs. local vLLM (FP8) produced different Factuality (≈0.88 vs ≈0.71). The *routing /
> tool-use behaviour* is stable across backends; the *absolute quality numbers* are
> serving-sensitive. Headline Factuality figures are from the LiteLLM gateway.
