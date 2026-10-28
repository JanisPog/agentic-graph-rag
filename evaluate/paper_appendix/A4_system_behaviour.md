# Appendix A4 — System Behaviour (tool use, routing, cost, calibration)

Per-query instrumentation from the **Simulated** test set (N=200, Qwen3.6-35B-A3B) — the
hardest subset, which exercises the adaptive-control machinery most. Behavioural figures are
robust to the absolute Factuality number (which is serving-sensitive).

## A4.1 — Method-invocation profile

| Method | Dispatched by | Invoked | Rate | Trigger |
|---|---|---|---|---|
| semantic retrieval (dense ANN) | planner | 200/200 | 100% | always on |
| BM25 retrieval (lexical) | planner | 200/200 | 100% | always on |
| reranker (cross-encoder, non-LLM) | pipeline | 200/200 | 100% | always on |
| RRF fusion (non-LLM) | pipeline | 200/200 | 100% | always on |
| CRAG grader (LLM) | pipeline | 200/200 | 100% | always on |
| graph traversal (PPR) | planner | 18/200 | 9.0% | intent multi_hop / entity-linked |
| surface-form entity seeding | graph arm | 13/200 | 6.5% | fires with graph path |
| ReAct budget *allocated* | planner | 16/200 | 8.0% | multi-sub-goal / multi_hop |
| ReAct loop *run* | INSPECT | 1/200 | 0.5% | only if CRAG < skip-react thresh |
| INSPECT (deep gap analysis) | router | 27/200 | 13.5% | skipped when CRAG ≥ 0.85 |
| EVOLVE (adaptive re-query) | INSPECT | 25/200 | 12.5% | recoverable gap |
| → `semantic_requery` | EVOLVE | 20/25 | 80% | dense re-query |
| → `bm25_retry` | EVOLVE | 3/25 | 12% | vocabulary gap |
| → `surface_form` | EVOLVE | 1/25 | 4% | entity-linkage gap |
| → `graph_hop` | EVOLVE | 1/25 | 4% | multi-hop gap |
| web search | planner / EVOLVE | 0/200 | 0% | gated off (`WEB_SEARCH_ENABLED=false`) |
| ticket fast-track (hard-skip) | pre-pipeline | 0/200 | 0% | demoted to seed-only |
| deferral | router | 18/200 | 9.0% | CRAG below floor, no viable retry |

Intent → arm mapping: factual (91%) → semantic+bm25; multi_hop (6.5%) → graph 100%;
out_of_domain (2.5%) → semantic+bm25 only (web disabled).

## A4.2 — Per-node LLM-call breakdown (PIVOT vs always-on baseline)

| Node | PIVOT | Baseline |
|---|---|---|
| Query Decomposer | 1 | 1 |
| Query Expander (HyDE gated) | 1–2 | 4–6 |
| Planner / Intent Classifier | 1 | 1 |
| BM25 / Semantic / Graph | 0 | 0 |
| Reranker (cross-encoder) | 0 | 0 |
| CRAG Grader | 1 | 1 |
| Inspect Node | 0–1 | 1 |
| Loops (ReAct / reflact) | 0–2 | 2–6 |
| Verify Node | 1 | — |
| Generator | 1 | 1–2 |
| **Happy path** | **≈7–9** | ≈13–17 |
| **One Evolve retry** | **≈13–15** | ≈20–25 |
| **Out-of-domain** | **≈4** | ≈12+ |

> Totals are measured (happy-path ≈7.8 calls/query, one-retry ≈14.5, CRAG-always-on=1); the
> per-node split is reconstructed to match the measured totals — the system logs total calls
> per query, not per-node counts.

Cost by routing path (measured): skip-react ≈7.99 calls (n=173); retry / max-retries 11–12.

## A4.3 — Routing distribution (CRAG score bands)

Thresholds: `CRAG_SUFFICIENT_THRESH=0.80`, `CRAG_SKIP_REACT_THRESH=0.92`,
`CRAG_RETRY_THRESH=0.30`. (Band boundaries below use the analysis tool's coarser cuts.)

| CRAG score band | Queries | Behaviour | Avg Factuality |
|---|---|---|---|
| ≥ 0.85 (skip-react) | 174 (87.0%) | straight to VERIFY — INSPECT call saved | 0.723 |
| 0.70 – 0.85 (sufficient, budget) | 8 (4.0%) | reflact/react if budget | 0.494 |
| 0.30 – 0.70 (retry zone) | 4 (2.0%) | EVOLVE re-query | 0.000 |
| < 0.30 (hard defer) | 14 (7.0%) | deferral | 0.000 |

Verdicts `{sufficient: 182, defer: 14, retry: 4}`, deferral 9.0%. All 18 queries scoring
CRAG < 0.70 produced Factuality 0.000 — the grader identified unanswerable context *before*
generation. INSPECT reasons: `score_above_skip_react_thresh` 173 (86.5%, INSPECT call skipped);
`max_retries_reached_below_floor` 10 (5%, Fact 0.0); `no_viable_strategy` 7 (3.5%, Fact 0.0).

> *Selection-bias note.* Graph/surface-form queries scored lower Factuality (0.56–0.60) than
> plain semantic+bm25 (0.66) — because those arms fire on intrinsically harder multi-hop /
> entity-heavy queries, not because the arms hurt. Causal per-arm contribution needs an
> ablation (arm-off re-run), not this observational split.

## A4.4 — Confidence calibration ("Knowing When")

Generator self-reported confidence vs. gold-judged Factuality (Simulated, N=200):

| Confidence | Queries | Avg Factuality |
|---|---|---|
| 0.0 – 0.50 | 25 | 0.000 |
| 0.5 – 0.70 | 1 | 0.200 |
| 0.7 – 0.90 | 18 | 0.431 |
| 0.9 – 1.00 | 156 | 0.781 |

Strictly monotonic. The same pattern holds on Synthetic (N=500: 0.03 → 0.57 → 0.70 → 0.85)
and ExpertWritten — confidence reliably separates correct answers from failures, the
foundation for safe deferral/escalation.

## A4.5 — Cost profile (ExpertWritten, fast-track on vs off)

| Metric | no-fastpass (full pipeline) | fast-path enabled |
|---|---|---|
| latency / query | ~13 s | 13.4 s |
| LLM calls / query | ~7.8 | 5.87 |
| prompt tokens / query | ≈21,000 | ≈9,685 |
| completion tokens / query | ≈1,100 | ≈686 |

Fast-path cuts ~25% of calls and ~half the prompt tokens, but served wrong-ticket answers
(Factuality 0.48 vs 0.87) — see Appendix A5 negative results.
