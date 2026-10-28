# Appendix A5 — Hyperparameters & Negative Results

## A5.1 — Key hyperparameters

All hand-tuned (not learned); env-overridable. Values used for the reported runs.

**Retrieval / fusion**

| Param | Value | Role |
|---|---|---|
| `RETRIEVAL_TOP_K` | 20 | candidates per arm into RRF fusion |
| `RERANKER_TOP_K` | 10 | chunks kept after cross-encoder |
| `RRF_K` | 60 | RRF rank-smoothing constant |
| `FUSION_POOL_SIZE` | 30 | fused pool size passed to reranker |
| top-k (scored) | 5 | retrieval depth evaluated (matches paper) |

**CRAG / routing**

| Param | Value | Role |
|---|---|---|
| `CRAG_SUFFICIENT_THRESH` | 0.80 | score ≥ → sufficient |
| `CRAG_SKIP_REACT_THRESH` | 0.92 | score ≥ → skip INSPECT, straight to VERIFY |
| `CRAG_RETRY_THRESH` | 0.30 | score < → hard defer floor |
| `CRAG_MAX_RETRIES` | 3 | max EVOLVE retries |
| `CRAG_PRECISION_TOP_N` | 5 | precision judged over top-N passages |
| `VERIFY_MIN_COVERAGE` | 0.5 | min sub-goal coverage to generate vs defer |
| `INSPECT_REQUIRE_IMPROVEMENT` | true | monotonic-acceptance rule |

**Graph (PPR)**

| Param | Value | Role |
|---|---|---|
| `GRAPH_SEED_K` | 3 | top semantic hits used as PPR seeds |
| `GRAPH_HOP_DEPTH` | 2 | subgraph hop depth |
| `GRAPH_MAX_SEEDS` | 40 | hard cap on PPR seeds (prevents stall) |
| `GRAPH_MIN_UNIGRAM_LEN` | 5 | min token length for SurfaceForm unigram lookup |

**Fast-track (demoted to seed-only)**

| Param | Value | Role |
|---|---|---|
| `FASTTRACK_SIMILARITY_THRESH` | 0.82 | min similarity to attempt fast-track |
| `FASTTRACK_SEED_THRESH` | 0.70 | min CRAG to seed hit into fusion |
| hard-skip | **removed** | no longer bypasses the pipeline |

**Temperatures:** structured nodes 0.0; generator 0.1; reasoning loop 0.3; expansion 0.5.

## A5.2 — Negative / cautionary results

These delimit *when* the architecture's mechanisms help, and document rigour.

**N1 — Fast-track hard-skip regression (reverted → seed-only).**
A strong ticket match originally bypassed the entire retrieval+generation stack and served the
reused answer. Measured: hard-skipped queries scored Factuality **0.483 vs 0.865** for full
generation. Cause: high embedding similarity matched topically-*near-but-wrong* tickets, which
the CRAG judge could not detect (it returned crag=1.0 for both correct and wrong tickets;
Context-Recall-vs-gold was 0.82 for good fast-tracks but **0.17** for bad — a signal
unavailable at routing time). **Lesson:** a call-saving shortcut is only safe when its
sufficiency signal reliably predicts answer quality. Fix: demote hard-skip to seed-only
(fast-track only seeds RRF; the query always runs the full pipeline). Effect on ExpertWritten:
Factuality 0.625 → 0.758.

**N2 — Deeper retrieval pool did not help (reverted).**
Raising `RETRIEVAL_TOP_K` 20→60 (with `FUSION_POOL_SIZE` 30→80) to give the reranker more
candidates *lowered* retrieval: Recall@5 0.704 → 0.654, MRR 0.538 → 0.477, Factuality
unchanged. The deeper pool fed the cross-encoder more distractors it sometimes ranked above
the true gold. **Lesson:** the recall ceiling (~0.62–0.70 on natural-language queries) is a
first-stage recall / reranker-discrimination limit, not a pool-depth limit. Reverted to 20/30.

**N3 — Forcing EVOLVE to prefer semantic re-query hurt quality (reverted).**
Biasing INSPECT to prefer `semantic_requery` when CRAG was low dropped Factuality 0.879 →
0.705. It overrode the LLM's per-gap strategy choice (semantic re-fetched the same failing
results on vocabulary/multi-hop gaps). **Lesson:** per-gap-type strategy selection outperforms
a fixed retry preference. Reverted.

**N4 — Graph PPR is entity-seeded only (known imperfection).**
Because the semantic arm runs in a parallel branch, `semantic_hits` is empty when the graph
node reads it, so PPR is seeded entirely by surface-form entities (measured: 0 semantic seeds
on every graph query). The graph arm as evaluated therefore runs on half its intended signal.
Not fixed for the reported runs; disclosed.

**N5 — Single-turn benchmark penalizes safe deferral.**
WixQA has no A2A clarification loop, so a *correct* deferral ("I can't answer this yet — please
clarify") becomes the final answer and scores Factuality 0.0, identical to a confident
hallucination. The ~9% deferral rate on the hard subset therefore *understates* the system;
reported Factuality is a lower bound. (`ALLOW_DEFERRAL=false` forces best-effort answers
instead, to quantify the gap.)
