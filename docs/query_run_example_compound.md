# Query Run Example — Compound Multi-Topic Query

> **Query:** "I want to delete some payment methods and i want to set a currency for my course"
> **Date:** 2026-04-29
> **Backend:** LiteLLM chat · LiteLLM embed (`fhms/qwen3-embedding-8b`, 4096-dim)
> **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## Why this query is interesting

This is a **compound query** — the user is asking two independent questions in one message:

1. How to delete payment methods
2. How to set a currency for a course

Both questions are answered by different parts of the knowledge base. A single embedding vector for the full query would land between both topics and retrieve neither well. The `query_decomposer` node was added specifically to handle this case.

---

## Graph Flow

```
USER QUERY
"I want to delete some payment methods and i want to set a currency for my course"
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  query_decomposer                                 17:56:11      │
│                                                                 │
│  LLM detects two independent topics and splits the query:       │
│    [0] "I want to delete some payment methods"                  │
│    [1] "I want to set a currency for my course"                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: sub_queries (2)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  query_expander                                   17:56:57      │
│                                                                 │
│  3 reformulations generated per sub-query → 8 total queries:    │
│                                                                 │
│  Sub-query 1 — "I want to delete some payment methods"          │
│    [0] I want to delete some payment methods  (original)        │
│    [1] How do I remove payment methods                          │
│    [2] Steps to delete billing information                      │
│    [3] Clear payment options                                    │
│                                                                 │
│  Sub-query 2 — "I want to set a currency for my course"         │
│    [4] I want to set a currency for my course (original)        │
│    [5] How to configure the currency for my course              │
│    [6] Where do I find the currency setting for my course       │
│    [7] Set the currency type for my course                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: expanded_queries (8)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  intent_classifier                                17:57:20      │
│                                                                 │
│  Intent  → factual                                              │
│  Filters → {} (none extracted)                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  semantic_retriever                               17:57:21      │
│                                                                 │
│  Embeds all 8 expanded queries in one batch (LiteLLM).          │
│  Each vector runs an independent ANN pass against the           │
│  chunk_embedding HNSW index (4096-dim cosine).                  │
│  Results deduplicated by max score across all 8 passes.         │
│                                                                 │
│  → 20 hits  (top score: 0.9647, chunk -3613217582609472337_0)   │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: semantic_hits (20)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  bm25_retriever                                   17:57:21      │
│                                                                 │
│  All 8 expanded queries run as independent Lucene fulltext       │
│  searches against chunk_fulltext index.                         │
│  Results deduplicated by max score; scores min-max normalised.  │
│                                                                 │
│  → 20 hits  (top score: 1.0000, chunk -3613217582609472337_0)   │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: bm25_hits (20)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  graph_retriever (PPR)                            17:57:21      │
│                                                                 │
│  Top 3 semantic hits used as seed nodes.                        │
│  2-hop subgraph traversal via NEXT_CHUNK + RELATED_TO edges     │
│  (entity co-occurrence path also attempted).                    │
│  Personalized PageRank run over subgraph                        │
│  (damping=0.75, 20 iterations).                                 │
│                                                                 │
│  Edge types found: NEXT_CHUNK × 2, ENTITY × 18                 │
│  → 20 hits from 3 seeds                                         │
│                                                                 │
│  Note: Neo4j deprecation warning for CALL subquery without      │
│  variable scope clause — does not affect results.               │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: graph_hits (20)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  rrf_fusion                                       17:57:21      │
│                                                                 │
│  Reciprocal Rank Fusion (k=60) across three ranked lists:       │
│    semantic : 20 chunks                                         │
│    bm25     : 20 chunks                                         │
│    graph    : 20 chunks                                         │
│                                                                 │
│  → 30 unique chunks                                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: fused_hits (30)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  reranker                                         17:58:10      │
│                                                                 │
│  Cross-encoder ms-marco-MiniLM-L-6-v2 loaded (first run).      │
│  All 30 fused chunks scored against the original query.         │
│                                                                 │
│  → kept 10 / 30  (top score: 2.9275, chunk -3470292782941462354_0)│
└──────────────────────────────┬──────────────────────────────────┘
                               │  State: reranked_hits (10)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  crag_grader                                      17:58:25      │
│                                                                 │
│  LLM scores context quality against the original query.         │
│                                                                 │
│  Score   → 0.90                                                 │
│  Verdict → sufficient  (threshold: 0.70)                        │
│  Missing → '' (nothing identified as missing)                   │
│  Retries → 0                                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  verdict=sufficient → ReAct
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ReAct LOOP                                                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Iteration 1                              17:58:28        │  │
│  │                                                           │  │
│  │  Thought: Context covers both parts — passages [3–6]      │  │
│  │  detail the Language & Region currency process;           │  │
│  │  passages [8–10] show disconnecting payment providers.    │  │
│  │  Need more specific info on managing multiple payment     │  │
│  │  methods selectively — how to remove non-Square           │  │
│  │  providers while keeping Square active.                   │  │
│  │                                                           │  │
│  │  Action: semantic_search                                  │  │
│  │  Sub-query: "Wix delete remove payment methods keep       │  │
│  │             only Square payment processor"                │  │
│  │  → 1 chunk embedded and searched                          │  │
│  │                                                           │  │
│  │  Observe: goal_achieved=False  pivot=False                │  │
│  │           context_size=15                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Iteration 2                              17:58:34        │  │
│  │                                                           │  │
│  │  Thought: Still missing specifics on selective provider   │  │
│  │  management — whether Wix allows keeping one provider     │  │
│  │  while disconnecting others, and any constraints on       │  │
│  │  this workflow.                                           │  │
│  │                                                           │  │
│  │  Action: semantic_search                                  │  │
│  │  Sub-query: "Wix keep only Square payment provider        │  │
│  │             disconnect other providers"                   │  │
│  │  → 1 chunk embedded and searched                          │  │
│  │                                                           │  │
│  │  Observe: goal_achieved=True   pivot=False                │  │
│  │           context_size=15                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  goal_achieved=True → generator
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  generator                                        17:59:17      │
│                                                                 │
│  LLM synthesises answer from 15 context chunks.                 │
│  answer_type → direct                                           │
│  confidence  → 0.90                                             │
│  cited_chunks → 4  (P3, P4, P9, P10)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                            [END]
```

---

## Answer

> To delete a payment method, go to **Accept Payments** on your site's dashboard, click **Manage** beside the relevant payment provider, and click **Disconnect** [P9]. If you don't want to fully disconnect the payment provider but remove it temporarily from your checkout page, you can disable the **Accepting payments** toggle instead [P10]. To set a currency for your course, go to the **Language & Region** section in your site's dashboard, scroll to **Site Currency**, select your preferred currency, and click **Save** [P3, P4].

**Confidence:** 0.90
**Cited chunks:** P3 · P4 · P9 · P10

---

## Step Summary

| # | Node | Output | Time |
|---|------|--------|------|
| 1 | `query_decomposer` | 2 sub-queries detected and split | 17:56:11 |
| 2 | `query_expander` | 8 expanded queries (4 per sub-query) | 17:56:57 |
| 3 | `intent_classifier` | `factual`, no metadata filters | 17:57:20 |
| 4 | `semantic_retriever` | 20 hits, top score 0.9647 | 17:57:21 |
| 5 | `bm25_retriever` | 20 hits, top score 1.0000 | 17:57:21 |
| 6 | `graph_retriever` | 20 hits via PPR (NEXT_CHUNK × 2, ENTITY × 18) | 17:57:21 |
| 7 | `rrf_fusion` | 30 unique chunks | 17:57:21 |
| 8 | `reranker` | 10 kept, top score 2.9275 | 17:58:17 |
| 9 | `crag_grader` | score=0.90, verdict=sufficient | 17:58:25 |
| 10 | `react_thinker` iter 1 | semantic_search for selective provider management | 17:58:28 |
| 11 | `react_thinker` iter 2 | semantic_search for keeping Square, disconnecting others | 17:58:34 |
| 12 | `generator` | direct answer, confidence=0.90, 4 citations | 17:59:17 |

**Total wall time:** ~3 minutes (dominated by LLM calls and cross-encoder cold start)

---

## Key Observations

### Query decomposition worked correctly
The compound query was split into two precise, self-contained sub-queries at `17:56:11`. Each sub-query then received its own set of 4 expansions, giving the semantic and BM25 retrievers 8 independent search vectors instead of 1 blended one. Both topics were covered in the final context.

### Both sub-topics found in retrieval
The top-scoring chunk (`-3613217582609472337_0`, score 0.9647 semantic / 1.0000 BM25) ranked first in both dense and sparse retrieval — a strong signal that the query decomposition correctly anchored both sub-queries. The final answer cites chunks from both the currency (P3, P4) and payment (P9, P10) domains.

### Graph retrieval contributed strongly
20 chunks were returned via PPR from 3 seeds, with ENTITY edges (×18) dominating over structural NEXT_CHUNK edges (×2). This is a marked improvement over the previous run — entity co-occurrence paths connected payment and currency topics effectively through shared entities in the knowledge graph.

### ReAct loop focused on selective provider management
CRAG already scored context as sufficient (0.90) before the ReAct loop ran. The two ReAct iterations targeted a specific gap: how to selectively keep Square while disconnecting other providers. The context size remained at 15 after both iterations (no net-new chunks added), suggesting the needed information was already present but required targeted retrieval to confirm. Iteration 2 confirmed goal achieved.

### Confidence was moderate (0.90)
Both questions had answerable chunks in the KB. The generator cited 4 specific passages. The slightly lower confidence compared to prior runs reflects the nuance around selective provider management, where the KB provides partial but not exhaustive coverage of the exact workflow.
