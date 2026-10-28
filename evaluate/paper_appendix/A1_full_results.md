# Appendix A1 — Full Results Grid

All runs use the held-out **test split** (last 20% of rows, never ingested as tickets) over
the full **6,221-article Wix KB corpus**, with the WixQA paper evaluation suite (§4.3) and
`top-k = 5`. Generator/judge: Qwen-class chat model unless noted; embeddings: Qwen3-embedding-8b.

## A1.1 — This system, all configurations

| Subset | N | Model | Judge | Fast-path | Recall@5 | MRR | NDCG@5 | Factuality | Ctx Recall | F1 | BLEU | ROUGE-1 | ROUGE-2 | Defer% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ExpertWritten | 40 | Qwen3.5-9B | self | on | 0.630 | 0.464 | 0.513 | 0.625 | 0.766 | 0.398 | 0.139 | 0.418 | 0.228 | 2.5 |
| ExpertWritten | 40 | Qwen3.5-9B | self | off | 0.692 | 0.473 | 0.539 | 0.758 | 0.873 | 0.449 | 0.196 | 0.468 | 0.272 | 2.5 |
| ExpertWritten | 40 | Qwen3.6-35B-A3B | self | off | 0.704 | 0.538 | 0.592 | **0.879** | **0.975** | 0.458 | 0.208 | 0.475 | 0.296 | 0.0 |
| ExpertWritten | 40 | Qwen3.6-35B-A3B | Gemma-4-31B | off | 0.717 | 0.478 | 0.543 | 0.869 | 0.921 | 0.462 | 0.224 | 0.480 | 0.306 | 2.5 |
| ExpertWritten (single-shot) | 40 | Qwen3.6-35B-A3B | self | — | 0.454 | 0.350 | 0.398 | 0.709 | 0.721 | 0.400 | 0.176 | 0.421 | 0.244 | — |
| Simulated | 200 | Qwen3.5-9B | self | off | 0.620 | 0.450 | 0.491 | 0.603 | 0.727 | 0.299 | 0.086 | 0.315 | 0.151 | 11.5 |
| Simulated | 40 | Qwen3.6-35B-A3B | self | off | 0.625 | 0.408 | 0.454 | 0.805 | 0.810 | 0.292 | 0.086 | 0.309 | 0.155 | 7.5 |
| Simulated | 40 | Qwen3.6-35B-A3B | Gemma-4-31B | off | 0.625 | 0.450 | 0.486 | 0.823 | 0.894 | 0.278 | 0.068 | 0.297 | 0.139 | 7.5 |
| Simulated | 200 | Qwen3.6-35B-A3B | self | off | 0.620 | 0.450 | 0.491 | — | — | — | — | — | — | 9.0 |
| Synthetic | 500 | Qwen3.5-9B | self | off | 0.979 | 0.850 | 0.882 | 0.804 | 0.886 | 0.546 | 0.324 | 0.557 | 0.426 | 2.4 |
| Synthetic | 500 | Qwen3.6-35B-A3B | self | off | 0.970 | 0.779 | 0.826 | **0.982** | **0.997** | 0.624 | 0.415 | 0.634 | 0.526 | 0.4 |

> "self" judge = the Qwen generator model also scores Factuality/Context Recall.
> "Gemma-4-31B" = a neutral third-family judge used to test for self-preference bias.
> "single-shot" = one LLM call, retrieve-and-stuff, no PIVOT pipeline (RQ baseline).
> Recall means are over retrieval-scored queries only (fast-track hard-skips excluded; with
> hard-skip removed, all queries are scored).

## A1.2 — WixQA paper baselines (for comparison)

Single-shot RAG (FlashRAG; BM25 or E5 dense; `top-k=5`), from the paper's Tables 3 & 5.

**ExpertWritten**

| Retriever | Model | F1 | BLEU | ROUGE-1 | ROUGE-2 | Factuality | Ctx Recall |
|---|---|---|---|---|---|---|---|
| E5 | GPT-4o | 0.37 | 0.08 | 0.30 | 0.12 | **0.85** | 0.81 |
| E5 | Claude 3.7 | 0.39 | 0.10 | 0.32 | 0.13 | 0.82 | 0.81 |
| E5 | Gemini 2.0 Flash | 0.43 | 0.14 | 0.35 | 0.17 | 0.76 | 0.81 |
| E5 | GPT-4o Mini | 0.39 | 0.09 | 0.31 | 0.12 | 0.79 | 0.81 |
| BM25 | GPT-4o | 0.36 | 0.08 | 0.29 | 0.12 | 0.83 | 0.73 |

**Synthetic**

| Retriever | Model | F1 | BLEU | ROUGE-1 | ROUGE-2 | Factuality | Ctx Recall |
|---|---|---|---|---|---|---|---|
| E5 | GPT-4o | 0.48 | 0.21 | 0.44 | 0.26 | **0.87** | 0.96 |
| E5 | Gemini 2.0 Flash | 0.58 | 0.33 | 0.53 | 0.36 | 0.84 | 0.97 |
| E5 | Claude 3.7 | 0.43 | 0.18 | 0.38 | 0.22 | 0.83 | 0.97 |
| E5 | GPT-4o Mini | 0.44 | 0.18 | 0.39 | 0.20 | 0.86 | 0.97 |

**Simulated** — the paper reports Factuality ~0.77 (GPT-4o / Claude 3.7, E5); per-model
lexical/Context-Recall values not all reproduced.

## A1.3 — External comparison: AgenticRAG

[arXiv 2605.05538] — agentic system on GPT-5-mini, reports Factuality only.

| System | Base model | ExpertWritten Fact. | Simulated Fact. |
|---|---|---|---|
| AgenticRAG | GPT-5-mini | 0.96 | 0.94 |
| This system | Qwen3.6-35B-A3B | 0.879 | 0.805 |

> AgenticRAG's figures are self-reported under its own GPT-5-mini judge and were not
> re-scored here. Each system in this comparison is judged by its own generation family
> (GPT-4o for the paper, GPT-5-mini for AgenticRAG, Qwen/Gemma for this work) — the
> consistent convention, since the other judge models are unavailable for re-judging.
> Judge-independent lexical metrics (F1/BLEU/ROUGE) corroborate the Factuality ranking.
