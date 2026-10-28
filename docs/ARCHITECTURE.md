# Architecture — Agentic Graph-RAG Support System
## Prototype v1.0

---

## 1. Overview

A stateful, agentic Retrieval-Augmented Generation system that resolves support queries against a Neo4j knowledge graph. The pipeline combines three retrieval strategies plus optional web search, fuses and reranks results with outcome-signal modulation, applies a context quality gate (CRAG), and runs an iterative reasoning loop (PIVOT) before generating a cited answer.

**Stack:** Neo4j 5.x · LangGraph · LiteLLM / vLLM / Gemini / Ollama · sentence-transformers · Brave/Tavily web search

---

## 2. Repository Layout

```
Prototype_v1.0/
├── data/
│   ├── customer_support_tickets_200k.csv
│   ├── wixqa_expertwritten.csv        # 200 expert Q&A pairs with article_ids refs
│   ├── wixqa_synthetic.csv            # 6 221 LLM-generated Q&A pairs (1 article each) — downloaded
│   └── wix_kb_corpus.csv              # 6 221 Wix KB articles (id, title, contents, article_type)
├── etl/
│   ├── ingest.py          # ETL orchestrator (CLI) — single-file and paired two-file mode
│   ├── download_wixqa.py  # Downloads WixQA subsets + KB corpus from Hugging Face → data/
│   ├── chunker_v2.py      # ChunkData dataclass + make_chunks()
│   ├── embedder.py        # LiteLLM / Gemini / Ollama / vLLM embedding dispatcher
│   ├── ner_v2.py          # LLM + regex entity extraction + synonym expansion
│   ├── schema_detector.py # LLM schema detection + detect_paired_schemas()
│   ├── metadata_builder.py
│   ├── sage_builder.py    # SAGE RELATED_TO edge builder
│   ├── doc_enrichment.py  # LLM-generated context_prefix per chunk at ingest time
│   ├── pdf_parser.py      # PDF → structured sections
│   ├── paperless_ingest.py
│   └── ticket_ingest.py   # Single-ticket live ETL (used by A2A executor)
├── graph/
│   ├── neo4j_client.py    # Driver wrapper + schema DDL
│   └── schema.cypher      # Index / constraint definitions (reference only)
├── agents/
│   ├── state.py              # AgentState TypedDict + ScoredChunk + reducers
│   ├── llm.py                # LLM + embedding dispatcher (LiteLLM / Gemini / Ollama / vLLM)
│   ├── retrieval_utils.py    # Shared ANN + BM25 Cypher helpers + metadata filter
│   ├── ablation.py           # Per-query ablation overrides (state.ablation dict)
│   ├── kb_context.py         # Loads KnowledgeBaseContext from Neo4j; cached prompt injection
│   ├── pivot_log.py          # Unified [PIVOT] decision logging
│   ├── metadata_vocab.py     # Canonical vocabularies (STATUS, PRIORITY, CHANNEL, …)
│   ├── query_decomposer.py   # Splits compound queries into ≤4 sub-queries
│   ├── query_expander.py     # Paraphrase + HyDE expansion per sub-query
│   ├── memory_context_builder.py   # Categorises raw mem0 memories by type
│   ├── memory_query_rewriter.py    # Rewrites sub_queries using user memory context
│   ├── ticket_fasttrack.py         # Pre-planner fast-track: ANN on problem/question chunks
│   ├── ticket_retriever.py         # Parallel retriever: similar tickets from support history
│   ├── ticket_reuse_gate.py        # KnownIssue + similar-ticket fast-path gate
│   ├── planner.py                  # PIVOT PLAN — structured plan dict
│   ├── intent_classifier.py        # Legacy — only used when PIVOT_ENABLED=false
│   ├── semantic_retriever.py
│   ├── bm25_retriever.py
│   ├── graph_retriever.py
│   ├── web_search_retriever.py     # Brave / Tavily / Parallel web search arm
│   ├── rrf_fusion.py
│   ├── failed_attempt_extractor.py # Extracts per_user_failed / resolved from memory
│   ├── reranker.py                 # Cross-encoder + metadata boost (_metadata_rescore)
│   ├── outcome_score_modulator.py  # Adjusts scores using failed/resolved/reuse signals
│   ├── crag_grader.py              # Pure grader — recall + precision → combined score
│   ├── inspect_node.py             # PIVOT INSPECT — routing, monotonic rule
│   ├── evolve_node.py              # PIVOT EVOLVE — adaptive retry, resets accumulators
│   ├── verify_node.py              # PIvVOT VERIFY — final constraint check, partial-answer path
│   ├── react_loop.py               # react_thinker · react_actor · react_observer
│   ├── reflact_loop.py             # reflact_thinker (state-reflective variant)
│   ├── generator.py                # Cited answer synthesis; reads partial_answer_note
│   └── deferral_handler.py         # Emits structured A2A gap payload
├── a2a/
│   └── executor.py        # A2A AgentExecutor bridging A2A requests → LangGraph
├── evaluate/
│   ├── eval.py            # 4-layer eval CLI (retrieval / ragas / routing / sweep)
│   ├── wixqa.py           # WixQA benchmark runner — detailed per-query + aggregate logging
│   ├── paper_metrics.py   # WixQA paper §4.3 metrics: F1/BLEU/ROUGE + Factuality/Context-Recall judges
│   ├── retrieval_metrics.py
│   └── ragas_runner.py
├── graph_builder.py       # LangGraph StateGraph compile() entrypoint
├── run.py                 # CLI query runner
├── diagnose.py            # Connectivity + embedding health check
├── config.py              # All constants, loaded from .env
└── .env                   # API keys + backend selection (not committed)
```

---

## 3. Data Model — Neo4j Graph Schema

### Node Types

| Label | Key Properties | Layer |
|---|---|---|
| `DataSource` | `id`, `name`, `type` (`ticket_system`\|`qa_forum`\|`kb_article`\|`document`), `schema_detected` | Source |
| `Document` | `id`, `source_id`, `title`, `doc_type`, `status`, `created_at`, `updated_at` | Document |
| `Metadata` | `id`, `doc_id`, `key`, `value`, `value_type` (`str`\|`int`\|`date`\|`enum`) | Metadata |
| `Section` | `id`, `doc_id`, `heading`, `level`, `position` | Structure |
| `Chunk` | `id`, `text`, `chunk_type`, `chunk_index`, `embedding` (float[`EMBED_DIM`]), `context_prefix`, `token_count` | Chunk |
| `Entity` | `id`, `name`, `type`, `source`, `confidence` | Entity |
| `SurfaceForm` | `id`, `form` (lowercase string) | Entity |
| `Concept` | `id`, `label`, `description` | Concept |
| `KnowledgeBaseContext` | `id` (`"kb_context"`), `organisation_name`, `domain_description` | Context |
| `KnownIssue` | `id`, `resolution_notes`, `embedding` (float[`EMBED_DIM`]) | Reuse |

> **`KnowledgeBaseContext`** — a single singleton node written once at ETL time by `_detect_kb_context()`. Injected into CRAG, INSPECT, ReAct, and generator prompts so they understand KB scope (e.g. "this is a Wix KB") and do not penalise recall for vocabulary synonyms.

> **`KnownIssue`** — centroid nodes representing recurring issues with known resolutions. Used by `ticket_reuse_gate` for ANN fast-path routing before the full retrieval stack.

### Relationship Types

| Relationship | Direction | Properties | Purpose |
|---|---|---|---|
| `PART_OF` | Chunk → Document | — | Chunk belongs to document |
| `PART_OF` | Section → Document | `position` | Section belongs to document |
| `PART_OF` | Chunk → Section | `position` | Chunk inside section |
| `HAS_METADATA` | Document → Metadata | — | Structured metadata layer |
| `NEXT_CHUNK` | Chunk → Chunk | `relation_type` (`problem_resolution`\|`sequential`\|`qa`) | Sequential / logical flow |
| `MENTIONS` | Chunk → Entity | `confidence`, `method` | NER link |
| `RELATED_TO` | Chunk ↔ Chunk | `score`, `method` (`sage_semantic`) | SAGE cross-chunk edges |
| `ALIASES` | SurfaceForm → Entity | — | Surface form → canonical entity |
| `BELONGS_TO` | Entity → Concept | `confidence` | Entity → abstract concept |
| `FROM_SOURCE` | Document → DataSource | — | Provenance |
| `REFERENCES` | Document → Document | — | Cross-dataset link (QA answer → KB article) |

### Indexes

| Index | Type | Purpose |
|---|---|---|
| `chunk_embedding` | HNSW vector, cosine, `EMBED_DIM` | Semantic retrieval |
| `chunk_fulltext` | Lucene full-text | BM25 keyword retrieval (indexes `context_prefix` + `text`) |
| `chunk_hop_index` | Composite `(id, chunk_index)` | Fast graph traversal |
| `metadata_lookup` | Composite `(key, value)` | Metadata pre-filtering |
| `entity_name_index` | B-tree `name` | Entity lookup |
| `surface_form_index` | B-tree `form` | Surface-form → entity resolution at query time |
| `doc_type_index` | B-tree `doc_type` | Cross-dataset REFERENCES traversal |
| `known_issue_embedding` | HNSW vector, cosine, `EMBED_DIM` | KnownIssue ANN for reuse gate |
| `doc_id_unique` | Uniqueness constraint | Document deduplication |
| `chunk_id_unique` | Uniqueness constraint | Chunk deduplication |
| `entity_id_unique` | Uniqueness constraint | Entity deduplication |
| `concept_id_unique` | Uniqueness constraint | Concept deduplication |

---

## 4. ETL Pipeline

```
Input Dataset (CSV / JSON / Markdown / JSONL)
      │
      ▼
SchemaDetectorAgent       LLM inspects column names + sample rows
                          → ColumnMapping (question/answer/metadata fields)
                          → DatasetProfile (archetype: ticket | qa | kb_article)
      │
      ▼
_load_and_filter()        Generic loader using ColumnMapping
                          supports CSV, JSON, JSONL, Markdown
                          dedup on id_field; filter by status if present
      │
      ▼
make_chunks()             Flexible chunker (chunker_v2.py):
                            1 chunk  — kb_article body only
                            2 chunks — question + answer (ticket/qa)
                            N chunks — long bodies split by section
                          ┌─────────────────────────────────────────┐
                          │  Contextual Chunk Enrichment             │
                          │  Each chunk gets an LLM-generated        │
                          │  context_prefix at ingest time:          │
                          │  "This chunk is from a {archetype}       │
                          │   about {product} / {category}..."       │
                          │  Embedding = embed(prefix + chunk.text)  │
                          └─────────────────────────────────────────┘
      │
      ▼
EntityExtractorAgent      LLM + regex entity extraction (ner_v2.py)
                          products, systems, error codes, versions,
                          procedures, components, organisations
                          ┌─────────────────────────────────────────┐
                          │  SurfaceForm compilation                 │
                          │  Per unique entity, at ingest time:      │
                          │  1. Deterministic forms (lowercased,     │
                          │     stripped, token splits)              │
                          │  2. LLM synonym expansion (entity-mode   │
                          │     full only): up to 6 aliases/abbrevs │
                          │  Stored as SurfaceForm -[:ALIASES]→ Ent │
                          └─────────────────────────────────────────┘
      │
      ▼
MetadataBuilder           All structured fields → Metadata nodes
                          ConceptMapper: entity → Concept nodes
      │
      ▼
embed_chunks()            vLLM / Gemini / Ollama
(embedder.py)             batch size 50; embeds context_prefix + text
      │
      ▼
SAGEGraphBuilder          Offline RELATED_TO edge construction:
(sage_builder.py)           cosine similarity over chunk embeddings
                            ANN top-K per chunk (FAISS / numpy fallback)
                            SAGE_PERCENTILE threshold + SAGE_MIN_SCORE floor
                            → RELATED_TO {method:"sage_semantic"}
                          ⚠ O(n²) exact fallback; use FAISS for >10k chunks
      │
      ▼
[Optional] REFERENCES     If --secondary provided:
edge construction           primary.ref_field → secondary.id_field
                            → REFERENCES edges (Document → Document)
      │
      ▼
Neo4j batch writes        Order: DataSource → Document → Metadata
(UNWIND, 500/batch)             → Section → Chunk → Entity → SurfaceForm
                                → Concept → PART_OF → HAS_METADATA
                                → NEXT_CHUNK → MENTIONS → ALIASES
                                → BELONGS_TO → RELATED_TO → REFERENCES
```

**CLI (run inside the Docker container):**
```bash
# On the Linux host where Docker is running:
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary  data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv \
    --articles --clear --limit 5000 --entity-mode fast

# --articles: only ingest KB articles referenced by primary dataset's ref_field
# --clear:    wipe Neo4j before ingesting (omit to append/update)
# --entity-mode fast: skip LLM synonym expansion (regex only, faster)

# Detect schemas without writing:
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv --detect-only
```

---

## 5. Agent Architecture — LangGraph State Machine

The pipeline runs the **PIVOT** loop (Plan-Inspect-eVOlve-Verify) by default (`PIVOT_ENABLED=true`). A legacy intent-classifier path is preserved behind `PIVOT_ENABLED=false`.

### Shared State (`AgentState`)

```
# Input
original_query          str
ablation                dict               # per-query signal ablation overrides

# Memory / user context
user_id                 str | None
ticket_id               str | None
user_memories           list[dict]         # raw mem0 results
similar_tickets         list[dict]         # output of ticket_retriever
profile_memories        list[dict]         # category == 'profile'
preference_memories     list[dict]         # category == 'preference'
interaction_memories    list[dict]         # category == 'interaction'
ticket_ref_memories     list[dict]         # entries with metadata.ticket_id set

# Ticket Reuse Gate
reuse_answer            str | None         # answer text when fast-path fires
reuse_ticket_id         str | None         # source KI/ticket of the reused answer
reuse_candidates        list[dict]         # mid-score candidates for RRF boost
ki_fast_path            bool               # True when routed via KnownIssue → crag_grader

# Failed Attempt Extractor
per_user_failed         list[str]          # solutions this user tried without success
global_failed           list[str]          # solutions that fail broadly (KnownIssue ANN)
resolved                list[str]          # solutions that helped this user

# Query understanding
sub_queries             list[str]          # set by query_decomposer
expanded_queries        list[str]          # paraphrase variants + HyDE per sub-query
hyde_query              str                # raw HyDE answer text (first sub-query)
intent                  str                # 'factual'|'multi_hop'|'global'|'metadata_filter'|'out_of_domain'
plan                    dict               # structured RetrievalPlan from planner
metadata_filters        dict[str, str]

# PIVOT state
sub_goals               list[str]          # ALL must be satisfied
evidence_gaps           list[str]          # unresolved sub-goals
evidence_found          list[str]          # resolved sub-goals
inspect_verdict         str                # 'skip_evolve'|'reflact'|'react'|'evolve'|'defer'
inspect_reason          str
evolve_strategy         str                # 'bm25_retry'|'graph_hop'|'surface_form'|'semantic_requery'|'web_search'|'defer'
tried_strategies        list[str]
verify_result           dict
partial_answer_note     str                # injected into generator when sub-goals are partially met
deferral_reason         dict               # structured A2A payload for the voice agent
deferral_context        dict               # voice agent retry context (gaps + user clarification)

# Ticket fast-track
fasttrack_hit           dict | None        # resolution chunk if similarity >= FASTTRACK_SIMILARITY_THRESH
fasttrack_score         float

# Retrieval — _add_or_reset reducer; None = reset on EVOLVE retry
semantic_hits           Annotated[list[ScoredChunk], _add_or_reset]
bm25_hits               Annotated[list[ScoredChunk], _add_or_reset]
graph_hits              Annotated[list[ScoredChunk], _add_or_reset]
web_hits                Annotated[list[ScoredChunk], _add_or_reset]
fused_hits              list[ScoredChunk]
reranked_hits           list[ScoredChunk]
outcome_modulated_hits  list[ScoredChunk]  # reranked_hits after outcome signal adjustment
reranker_score_cache    dict               # chunk_id → cross-encoder score; reused on EVOLVE retries

# ReAct / ReflAct loop
react_iterations        int
react_scratchpad        list[ReActStep]    {thought, action, sub_query, observation}
reflact_scratchpad      list[ReflActStep]  {reflection, action, sub_query, observation}
reflection_summary      str
goal_achieved           bool
pivot_required          bool
prev_sub_queries        list[str]

# CRAG
context_quality         float 0.0–1.0      # 0.7×recall + 0.3×precision
crag_verdict            str                # 'sufficient'|'retry'|'defer' (hint only — INSPECT routes)
crag_retries            int
crag_missing_hint       str

# Generation
final_answer            str
cited_chunk_ids         list[str]
confidence              float
```

### Graph Execution Flow (PIVOT)

```
 ┌──────────────────────────────────────────────────────────┐
 │                    USER QUERY                            │
 └──────────────────────┬───────────────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   memory_context_builder     │  Splits raw mem0 → typed lists
         │   (no-op if memories=[])     │  profile / preference / interaction
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   query_decomposer           │  LLM → splits compound queries
         └──────────────┬───────────────┘  State: sub_queries
                        ▼
         ┌──────────────────────────────────────────────────┐
         │  ticket_fasttrack                                │
         │  ANN on problem/question chunks only             │
         │  similarity >= FASTTRACK_SIMILARITY_THRESH(0.82) │
         │  → follows [:NEXT_CHUNK] to resolution chunk    │
         │  → CRAG grade (single LLM call)                 │
         └──────┬──────────────────────┬────────────────────┘
                │ combined < 0.85       │ combined >= 0.85
                ▼                      └──────────────────► verify_node
         ┌──────────────────────────────────────────────────┐
         │  ticket_reuse_gate                               │
         │  KnownIssue ANN (sim >= KI_FAST_PATH_SIM_THRESH) │
         │  → CRAG grade on KI resolution                  │
         │  ki_fast_path=True → inspect_node directly      │
         │  mid-score (>= TICKET_REUSE_THRESHOLD_MID)      │
         │  → add to reuse_candidates for RRF boost        │
         └──────┬──────────────────────┬────────────────────┘
                │ ki_fast_path=False    │ ki_fast_path=True
                ▼                      └──────────────────► inspect_node
         ┌──────────────────────────────────────────────────┐
         │  memory_query_rewriter                           │
         │  Enriches sub_queries with user context          │
         │  (no LLM call if all memory lists empty)         │
         └──────────────┬───────────────────────────────────┘
                        ▼
         ┌──────────────────────────────────────────────────┐
         │  planner  (PIVOT PLAN)                           │
         │  LLM → structured plan dict:                     │
         │    • intent                                      │
         │    • sub_goals                                   │
         │    • use_semantic / use_bm25 / use_graph         │
         │      / use_web_search / use_surface_form         │
         │    • skip_reflact, react_budget                  │
         │    • confidence_target                           │
         │    • failure_modes / fallback_strategy           │
         │  [PIVOT] PLAN log emitted                        │
         └──────────────┬───────────────────────────────────┘
                        ▼
         ┌──────────────────────────────────────────────────┐
         │  query_expander  (paraphrase + HyDE)             │
         │  3 paraphrase variants per sub-query             │
         │  HyDE gated on plan.use_hyde                     │
         └────────────────────────┬─────────────────────────┘
                                  │
         ┌──── plan-conditioned parallel fan-out ────────────┐
         │ (intent=out_of_domain → deferral_handler)         │
         ▼         ▼         ▼         ▼                     │
  ┌──────────┐ ┌──────┐ ┌───────┐ ┌──────────┐              │
  │ semantic_│ │ bm25_│ │graph_ │ │web_search│              │
  │ retriever│ │retr. │ │retr.  │ │retriever │              │
  └────┬─────┘ └──┬───┘ └───┬───┘ └────┬─────┘              │
       └──────────┴──────────┴──────────┘                    │
                        ▼                                    │
              ┌───────────────────────┐                      │
              │  rrf_fusion  (k=60)   │                      │
              │  injects fasttrack_   │                      │
              │  hit as rank-1 if set │                      │
              └──────────┬────────────┘                      │
                         ▼                                   │
              ┌───────────────────────┐                      │
              │ failed_attempt_       │                      │
              │ extractor             │                      │
              │ per_user_failed /     │                      │
              │ global_failed /       │                      │
              │ resolved from memory  │                      │
              └──────────┬────────────┘                      │
                         ▼                                   │
              ┌───────────────────────┐                      │
              │  reranker             │  cross-encoder       │
              │  + _metadata_rescore  │  recency/type/source │
              └──────────┬────────────┘                      │
                         ▼                                   │
              ┌───────────────────────┐                      │
              │ outcome_score_        │                      │
              │ modulator             │                      │
              │ penalises failed,     │                      │
              │ boosts resolved &     │                      │
              │ reuse_candidates      │                      │
              └──────────┬────────────┘                      │
                         ▼                                   │
              ┌───────────────────────────────────────────┐  │
              │  crag_grader  (pure scorer)                │  │
              │  recall + precision → context_quality      │  │
              │  No routing — INSPECT owns the decision    │  │
              └──────────┬────────────────────────────────┘  │
                         ▼                                   │
         ┌───────────────────────────────────────────────┐   │
         │  inspect_node  (PIVOT INSPECT)                 │   │
         │  3-question reflection (RESULT/PLAN/REVISION)  │   │
         │  • skip_reflact gate (CRAG score + plan hint)  │   │
         │  • monotonic rule (untried strategy +          │   │
         │    expected_improvement=true)                  │   │
         │  • updates evidence_gaps / evidence_found      │   │
         │  [PIVOT] INSPECT log emitted                   │   │
         └──┬──────┬─────────────┬────────────────┬──────┘   │
            │      │             │                │           │
     skip_evolve reflact|react evolve           defer         │
            │      │             │                │           │
            │      ▼             ▼                ▼           │
            │ ┌─────────┐  ┌──────────┐  ┌──────────────┐   │
            │ │reflact_ │  │ evolve_  │  │  deferral_   │───┼─► END
            │ │thinker /│  │ node     │  │  handler     │   │
            │ │react_   │  │ (resets  │  └──────────────┘   │
            │ │thinker  │  │ hit lists│                      │
            │ └──┬──────┘  └────┬─────┘                      │
            │    ▼              │                             │
            │ ┌─────────┐       └──► [retriever arm]          │
            │ │react_   │           → rrf_fusion              │
            │ │actor    │           → failed_attempt_extr.    │
            │ └──┬──────┘           → reranker               │
            │    ▼                  → outcome_modulator       │
            │ ┌─────────┐           → crag_grader             │
            │ │react_   │           → inspect_node            │
            │ │observer │                                     │
            │ └──┬──────┘                                     │
            │    │ goal? or max_iter?                         │
            └────┴──────────────────► verify_node             │
                                           │                  │
                      ┌────────────────────┴──────┐           │
                      ▼                           ▼           │
         ┌───────────────────────────┐  ┌──────────────┐     │
         │  verify_node (PIVOT VERIFY)│  │ deferral_    │─────┘
         │  Final constraint check   │  │ handler      │
         │  sub_goal: satisfied /    │  └──────────────┘
         │  partial / missing        │
         │  TRUSTS CRAG: if score ≥  │
         │  CRAG_SUFFICIENT_THRESH   │
         │  never defers, only adds  │
         │  partial_note             │
         │  [PIVOT] VERIFY log       │
         └──────────────┬────────────┘
                        ▼
             ┌──────────────────────┐
             │     generator        │  LLM: cited JSON answer
             │  Reads partial_note  │  Idempotency guard
             └──────────────────────┘
                        ▼
                       END
```

EVOLVE routes back to a specific retriever via `_after_evolve`:
- `bm25_retry` / `surface_form` → `bm25_retriever`
- `graph_hop` → `graph_retriever`
- `semantic_requery` → `semantic_retriever`
- `web_search` → `web_search_retriever`
- `defer` → `deferral_handler`

---

## 6. Node Catalogue

| Node | File | Responsibility |
|---|---|---|
| `memory_context_builder` | agents/memory_context_builder.py | Splits raw mem0 results into typed views: profile, preference, interaction, ticket_ref. No-op when memories list is empty |
| `query_decomposer` | agents/query_decomposer.py | LLM detects compound queries and splits into ≤4 independent sub-queries; atomic queries pass through unchanged |
| `ticket_fasttrack` | agents/ticket_fasttrack.py | ANN restricted to `chunk_type IN [problem, question]`; if top similarity ≥ `FASTTRACK_SIMILARITY_THRESH` follows `[:NEXT_CHUNK]` to resolution chunk and runs CRAG judge. **Hard skip** (combined ≥ 0.85): routes directly to `verify_node`. **Seed** (0.70–0.85): injects hit as boosted rank-1 in RRF. **Discard** (< 0.70): clears fasttrack_hit |
| `ticket_reuse_gate` | agents/ticket_reuse_gate.py | ANN over KnownIssue centroids. If similarity ≥ `KI_FAST_PATH_SIM_THRESH`: CRAG grades the KI resolution; combined ≥ `TICKET_REUSE_THRESHOLD_HIGH` → `ki_fast_path=True` → routes directly to `inspect_node`. Mid-score (≥ `TICKET_REUSE_THRESHOLD_MID`) candidates added to `reuse_candidates` for RRF boost |
| `memory_query_rewriter` | agents/memory_query_rewriter.py | Enriches sub_queries with user profile/preference context. No LLM call when all memory lists are empty |
| `planner` **(PIVOT PLAN)** | agents/planner.py | One LLM call → structured plan dict: intent, sub_goals, retrieval-arm flags (use_semantic/bm25/graph/web_search), skip_reflact hint, react_budget, confidence_target, failure_modes, fallback_strategy. Emits `[PIVOT] PLAN` log |
| `intent_classifier` | agents/intent_classifier.py | Legacy — only wired when `PIVOT_ENABLED=false` |
| `query_expander` | agents/query_expander.py | 3 paraphrase variants per sub-query + HyDE hypothetical answer embedding (gated on `plan.use_hyde`); deduplicated into `expanded_queries` + `hyde_query` |
| `semantic_retriever` | agents/semantic_retriever.py | Dense HNSW vector search; metadata pre-filter; runs when `plan.use_semantic`. On retry uses `crag_missing_hint` as leading query |
| `bm25_retriever` | agents/bm25_retriever.py | Neo4j Lucene full-text (indexes `context_prefix + text`); min-max normalised; runs when `plan.use_bm25` |
| `graph_retriever` | agents/graph_retriever.py | Surface-form extraction (regex + SurfaceForm index lookup, no LLM) → entity seeds; KNN-guided PPR over NEXT_CHUNK + RELATED_TO + MENTIONS edges (damping=0.75, iterations=20); runs when `plan.use_graph` |
| `web_search_retriever` | agents/web_search_retriever.py | Brave / Tavily / Parallel web search; results returned as `chunk_type="web_result"` with `WEB_SEARCH_TRUST_PENALTY` applied so KB chunks rank higher; runs when `plan.use_web_search` or when EVOLVE picks `web_search` strategy |
| `ticket_retriever` | agents/ticket_retriever.py | ANN over problem/question chunks from the ticket corpus; returns similar tickets into `similar_tickets`; threshold 0.60, top_k 10 |
| `rrf_fusion` | agents/rrf_fusion.py | Late fusion — RRF k=60 merges semantic + BM25 + graph + web hit lists. Injects `fasttrack_hit` and `reuse_candidates` as boosted candidates when present |
| `failed_attempt_extractor` | agents/failed_attempt_extractor.py | LLM extracts `per_user_failed` and `resolved` from `interaction_memories`; `global_failed` from KnownIssue ANN (threshold: `OUTCOME_KNOWN_ISSUE_THRESH`) |
| `reranker` | agents/reranker.py | `BAAI/bge-reranker-v2-m3` cross-encoder (vLLM) or `ms-marco-MiniLM-L-6-v2` (local). Keeps top-`RERANKER_TOP_K`. Optional `_metadata_rescore` applies additive boosts: recency, doc_type, source authority, status, priority, chunk type. Gated by `RERANK_BOOST_ENABLED`. Reuses `reranker_score_cache` on EVOLVE retries |
| `outcome_score_modulator` | agents/outcome_score_modulator.py | Post-reranking signal injection: penalty for `per_user_failed` (×0.3) and `global_failed` (×0.6), boost for `resolved` (×1.4) and `reuse_candidates` (×1.2). Soft text-overlap matching. Score clamped to [`OUTCOME_SCORE_MIN`, `OUTCOME_SCORE_MAX`]. Gated by `OUTCOME_MODULATOR_ENABLED` |
| `crag_grader` | agents/crag_grader.py | **Pure scorer.** Single LLM call → recall + precision → combined = 0.7×recall + 0.3×precision. Writes `context_quality`, `crag_verdict` (hint only), `crag_missing_hint`. Routing owned by `inspect_node` |
| `inspect_node` **(PIVOT INSPECT)** | agents/inspect_node.py | 3-question PIVOT reflection. Enforces **skip_reflact gate** and **monotonic rule** (only EVOLVE when strategy is untried AND `expected_improvement=true`). Gap reasons: `wrong_terms`, `wrong_tool`, `kb_insufficient`, `irrecoverable`. Emits `[PIVOT] INSPECT` log |
| `evolve_node` **(PIVOT EVOLVE)** | agents/evolve_node.py | Records strategy in `tried_strategies`, increments `crag_retries`, **resets** all hit lists to `[]` via `_add_or_reset(None)` so the retry starts clean. Emits `[PIVOT] EVOLVE` log |
| `verify_node` **(PIVOT VERIFY)** | agents/verify_node.py | Final constraint check. Marks each sub_goal satisfied/partial/missing. **Trusts CRAG**: if `context_quality ≥ CRAG_SUFFICIENT_THRESH` never defers — only sets `partial_answer_note`. Routes `generate` / `generate_with_partial_note` / `defer`. Emits `[PIVOT] VERIFY` log |
| `react_thinker` | agents/react_loop.py | LLM thought step — identifies missing evidence, selects next tool |
| `reflact_thinker` | agents/reflact_loop.py | State-reflective variant — reflection + running `reflection_summary` + tool choice; used when `USE_REFLACT=true` |
| `react_actor` | agents/react_loop.py | Dispatches chosen retrieval tool, merges results into `reranked_hits` |
| `react_observer` | agents/react_loop.py | Deduplicates hits, optional goal check, topological pivot detection; routes to `verify_node` |
| `generator` | agents/generator.py | Final LLM call — multi-hop synthesis with inline citations. **Idempotency guard**: returns `{}` if `final_answer` already set. Reads `partial_answer_note` and tells the user which sub-goals are unresolved |
| `deferral_handler` | agents/deferral_handler.py | Emits structured A2A payload (`deferral_reason` dict: `type`, `original_query`, `gaps`, `covered`, `suggested_clarification`). Types: `out_of_domain`, `irrecoverable_gap`, `irrecoverable_gap_after_retry` |
| `pivot_log` *(helper)* | agents/pivot_log.py | Unified `[PIVOT]` decision logger — every decision is greppable: `PLAN → INSPECT → EVOLVE/REACT → VERIFY → GENERATE/defer` |

---

## 7. Conditional Routing (PIVOT mode)

### After `ticket_fasttrack` (`_after_fasttrack`)

| Condition | Next node |
|---|---|
| combined ≥ 0.85 AND `fasttrack_hit` set | `verify_node` — hard skip, bypasses full pipeline |
| combined 0.70–0.85 OR no hit | `ticket_reuse_gate` — normal flow; `fasttrack_hit` may seed RRF |

### After `ticket_reuse_gate` (`_after_reuse_gate`)

| Condition | Next node |
|---|---|
| `ki_fast_path == True` | `inspect_node` — KI resolution already CRAG-graded |
| otherwise | `memory_query_rewriter` → planner → normal flow |

### After `planner` (`_after_planner`)

| Condition | Targets |
|---|---|
| `plan.intent == "out_of_domain"` | `deferral_handler` |
| otherwise | `[semantic_retriever, bm25_retriever, graph_retriever, web_search_retriever]` filtered by `plan.use_*` flags (parallel fan-out) |

### After `inspect_node` (`_after_inspect`)

| Condition | Next node | Reason |
|---|---|---|
| `score ≥ CRAG_SKIP_REACT_THRESH (0.85)` | `verify_node` | Very high score |
| `score ≥ conf_target` AND (`skip_reflact` OR `budget == 0`) | `verify_node` | Sufficient + no loop needed |
| `score ≥ conf_target` AND `budget > 0` | `reflact_thinker` / `react_thinker` | Sufficient but loop may add evidence |
| `retries ≥ CRAG_MAX_RETRIES` AND `score < CRAG_RETRY_THRESH` | `deferral_handler` | Retries exhausted below floor |
| `retries ≥ CRAG_MAX_RETRIES` | `deferral_handler` | Retries exhausted |
| `score < CRAG_RETRY_THRESH` AND gap is `irrecoverable` | `deferral_handler` | Hard irrecoverable |
| `score < CRAG_RETRY_THRESH` AND gap is recoverable AND retries remain | falls through to monotonic rule | Let EVOLVE try |
| Strategy untried AND `expected_improvement=true` | `evolve_node` | Monotonic rule satisfied |
| otherwise | `deferral_handler` | No viable improvement |

### After `evolve_node` (`_after_evolve`)

| `evolve_strategy` | Next node |
|---|---|
| `bm25_retry` / `surface_form` | `bm25_retriever` |
| `graph_hop` | `graph_retriever` |
| `semantic_requery` | `semantic_retriever` |
| `web_search` | `web_search_retriever` |
| `defer` | `deferral_handler` |

### After `react_observer` (`_after_observer_pivot`)

| Condition | Next node |
|---|---|
| `goal_achieved == True` OR `react_iterations >= REACT_MAX_ITERATIONS` | `verify_node` |
| otherwise | `reflact_thinker` / `react_thinker` |

### After `verify_node` (`_after_verify`)

| Condition | Next node |
|---|---|
| CRAG sufficient AND any sub_goal missing/partial | `generator` with `partial_answer_note` |
| CRAG sufficient AND all sub_goals satisfied | `generator` (clean) |
| `coverage_ratio ≥ VERIFY_MIN_COVERAGE` OR any satisfied | `generator` (possibly with partial note) |
| otherwise | `deferral_handler` |

---

## 8. LLM + Embedding Backends

```
EMBED_BACKEND=vllm    # or gemini | ollama | litellm
CHAT_BACKEND=vllm     # or gemini | ollama | litellm
RERANKER_BACKEND=vllm # or local (sentence-transformers)
```

| Backend | Embed model | Chat model | Notes |
|---|---|---|---|
| **vLLM** (GPU server) | `Qwen/Qwen3-Embedding-8B` (4096-dim, localhost:8011) | `Qwen/Qwen3.5-9B` (localhost:8010) | Reranker: `BAAI/bge-reranker-v2-m3` (localhost:8012) |
| **LiteLLM** | `fhms/qwen3-embedding-8b` (4096-dim) | `gwdg/qwen3.5-35b-a3b` | Proxied API |
| **Gemini** | `gemini-embedding-001` (768-dim) | `gemini-2.5-flash` | API key required; re-ingest required if switching backends |
| **Ollama** | `nomic-embed-text` (768-dim) | `qwen2.5:7b` | Local, no rate limits |

---

## 9. Key Configuration Parameters

| Parameter | Default | Effect |
|---|---|---|
| **Ticket Fast-Track** | | |
| `FASTTRACK_SIMILARITY_THRESH` | 0.82 | Min cosine similarity on a problem/question chunk to attempt fast-track |
| `FASTTRACK_HARD_SKIP_THRESH` | 0.85 | CRAG combined score to bypass full pipeline → verify_node directly |
| `FASTTRACK_SEED_THRESH` | 0.70 | CRAG combined score to seed into RRF fusion |
| **Ticket Reuse Gate** | | |
| `KI_FAST_PATH_SIM_THRESH` | 0.85 | KnownIssue ANN similarity threshold to attempt CRAG grade |
| `TICKET_REUSE_THRESHOLD_HIGH` | 0.70 | CRAG combined floor to trigger ki_fast_path route |
| `TICKET_REUSE_THRESHOLD_MID` | 0.40 | Floor to add candidate to reuse_candidates for RRF boost |
| **CRAG** | | |
| `CRAG_SUFFICIENT_THRESH` | 0.70 | Context is "good enough" — INSPECT routes to VERIFY / loop |
| `CRAG_SKIP_REACT_THRESH` | 0.85 | Skip reasoning loop entirely; straight to VERIFY |
| `CRAG_RETRY_THRESH` | 0.30 | Below this → hard defer (unless retries remain and gap is recoverable) |
| `CRAG_MAX_RETRIES` | 3 | Maximum EVOLVE cycles before forced deferral |
| `CRAG_DEFERRAL_PARTIAL_THRESH` | 0.35 | Legacy partial-answer threshold |
| **PIVOT** | | |
| `PIVOT_ENABLED` | true | Master switch — false falls back to legacy intent_classifier |
| `INSPECT_REQUIRE_IMPROVEMENT` | true | Monotonic rule — only EVOLVE when expected_improvement=true |
| `VERIFY_MIN_COVERAGE` | 0.5 | Sub-goal coverage ratio to proceed to generator (non-CRAG-sufficient path) |
| `PLAN_DEFAULT_REACT_BUDGET_FACTUAL` | 0 | Default ReAct budget for factual queries |
| `PLAN_DEFAULT_REACT_BUDGET_MULTI_HOP` | 2 | Default ReAct budget for multi-hop queries |
| `PLAN_DEFAULT_CONFIDENCE_FACTUAL` | 0.75 | Per-query CRAG target for factual queries |
| `PLAN_DEFAULT_CONFIDENCE_MULTI_HOP` | 0.65 | Per-query CRAG target for multi-hop queries |
| `USE_REFLACT` | true | Use reflact_thinker instead of plain react_thinker |
| `REACT_MAX_ITERATIONS` | 2 | Hard cap on ReAct / ReflAct loop cycles |
| **Web Search** | | |
| `WEB_SEARCH_ENABLED` | true | Enable web search retrieval arm |
| `WEB_SEARCH_PROVIDER` | tavily | `brave` \| `tavily` \| `parallel` |
| `BRAVE_SEARCH_API_KEY` | — | API key for Brave Search |
| `TAVILY_API_KEY` | — | API key for Tavily |
| `WEB_SEARCH_MAX_RESULTS` | 5 | Max results per web search call |
| `WEB_SEARCH_TIMEOUT_S` | 10 | Web search request timeout |
| `WEB_SEARCH_TRUST_PENALTY` | 0.85 | Multiplier on web chunk scores (keeps KB chunks preferred) |
| **Outcome Score Modulator** | | |
| `OUTCOME_PENALTY_PER_USER` | 0.30 | Score multiplier for per_user_failed solutions |
| `OUTCOME_PENALTY_GLOBAL` | 0.60 | Score multiplier for globally failed solutions |
| `OUTCOME_BOOST_RESOLVED` | 1.40 | Score multiplier for resolved solutions |
| `OUTCOME_BOOST_REUSE` | 1.20 | Score multiplier for reuse_candidates |
| `OUTCOME_SCORE_MIN` | 0.05 | Minimum score floor after modulation |
| `OUTCOME_SCORE_MAX` | 1.00 | Maximum score cap after modulation |
| `OUTCOME_KNOWN_ISSUE_THRESH` | 0.75 | KnownIssue ANN threshold for global_failed lookup |
| **Ablation Switches** (all default off; enable per run in `.env` or via `state["ablation"]`) | | |
| `METADATA_IN_GENERATOR_ENABLED` | false | Inject metadata headers into generator prompt |
| `METADATA_FILTER_ENABLED` | false | Retrieval-time metadata pre-filtering |
| `RERANK_BOOST_ENABLED` | false | Reranking-time metadata boost (_metadata_rescore) |
| `MEMORY_ENABLED` | false | Memory-conditioned query rewriting |
| `REUSE_GATE_ENABLED` | false | KnownIssue / similar-ticket fast-path gate |
| `OUTCOME_MODULATOR_ENABLED` | false | Outcome-signal score modulation |
| **Retrieval** | | |
| `RETRIEVAL_TOP_K` | 20 | Results per retriever per query |
| `RERANKER_TOP_K` | 10 | Chunks kept after cross-encoder |
| `RRF_K` | 60 | RRF rank-smoothing constant |
| `GRAPH_SEED_K` | 3 | Semantic seeds passed to graph_retriever |
| `GRAPH_HOP_DEPTH` | 2 | Max NEXT_CHUNK / RELATED_TO hops from seed |
| **SAGE** | | |
| `SAGE_ENABLED` | true | Build RELATED_TO edges at ETL time |
| `SAGE_TOP_K_PER_CHUNK` | 5 | SAGE neighbours per chunk (production: 50) |
| `SAGE_MIN_SCORE` | 0.88 | Hard floor for SAGE edges |
| `SAGE_PERCENTILE` | 95 | Percentile threshold within candidates |
| `SAGE_MAX_EDGES` | 5000 | Absolute cap for prototype safety |
| **ETL** | | |
| `NEO4J_BATCH_SIZE` | 500 | Nodes per UNWIND batch |
| `EMBED_BATCH_SIZE` | 50 | Texts per embedding API call |

---

## 10. A2A Deferral Protocol

The `deferral_handler` emits a structured JSON payload consumed by the voice agent. The A2A executor (`a2a/executor.py`) handles two deferral lifecycles:

**First deferral** (`irrecoverable_gap`):
- TaskState → `INPUT_REQUIRED`
- Voice agent speaks the `clarification_prompt` and prompts the user
- On reply, voice agent sends `{ user_clarification: "..." }` to the same task context

**Retry** (orchestrator reads previous deferral from `InMemoryTaskStore`):
- Merges original query + user clarification into enriched retry query
- Pre-seeds pipeline with `evidence_gaps` and `crag_missing_hint` from previous deferral
- `crag_retries` starts at 1

**Final failure** (`irrecoverable_gap_after_retry`):
- TaskState → `FAILED` (`updater.failed()`)
- Voice agent reads the escalation message and hands off to a human support agent

**Out-of-domain**:
- TaskState → `INPUT_REQUIRED` (or immediate `FAILED` depending on severity)
- Emitted by `deferral_handler` when planner classifies `intent=out_of_domain`

---

## 11. Running the System

### Start services

```bash
sudo docker compose up -d
sudo docker compose logs -f graph-rag-agent
```

### Ingest data

```bash
# Full paired ingest (QA + referenced KB articles only) — recommended
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary  data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv \
    --articles --clear --limit 5000 --entity-mode fast

# Dev run with hard limits on both datasets
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary  data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv \
    --clear --limit 200 --secondary-limit 200 --entity-mode fast

# Inspect detected schemas without writing
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv --detect-only
```

### Health check

```bash
sudo docker compose exec graph-rag-agent python diagnose.py
```

### Run a query

```bash
sudo docker compose exec graph-rag-agent python run.py "My login keeps failing"
sudo docker compose exec graph-rag-agent python run.py --query "How do I reset my password?" --json
```

### Rebuild after code changes

```bash
sudo bash ./redeploy.sh
```

---

## 12. WixQA Benchmark Evaluation

The system is benchmarked against **WixQA** — an enterprise RAG benchmark grounded in a 6 221-article snapshot of the Wix Help Center. Three QA subsets are supported, all normalised to the same `question, answer, article_ids` schema:

| Subset | Size | Grounding | Notes |
|---|---|---|---|
| `wixqa_expertwritten` | 200 | 1–3 articles (27 % multi-article) | Expert-authored step-by-step answers |
| `wixqa_simulated` | 200 | 1–2 articles (14 % multi-article) | Distilled from user–chatbot dialogues |
| `wixqa_synthetic` | 6 221 | exactly 1 article | LLM-generated, one pair per KB article |

`article_ids` reference KB article ids (64-char SHA-256), which become `Document.id` at ingest — so grounding aligns directly with retrieved chunks' parent documents. Multi-article ExpertWritten rows pack several ids into one field value (concatenated or comma-joined); both the eval grounding parser (`_parse_article_ids`) and the ingest ref parser (`_parse_ref_ids`) split these into individual 64-char ids at **read time**, so the dataset files are never modified.

### Step 1 — Download the datasets

```bash
# Synthetic subset + KB corpus (ExpertWritten + wix_kb_corpus already in data/)
python etl/download_wixqa.py --subset synthetic --kb

# Other subsets / options
python etl/download_wixqa.py --subset simulated
python etl/download_wixqa.py --subset expertwritten --force   # re-fetch, overwrite
```

Writes `data/wixqa_<subset>.csv`. The downloader auto-detects Hugging Face config names, normalises columns, and maps article URLs → KB ids where needed.

### Step 2 — Ingest the KB for evaluation

Grounding requires the KB articles in Neo4j. One ingest covers all three subsets (each grounds to KB ids).

```bash
# Full KB (6 221 articles) — required for Synthetic; canonical for all subsets
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary  data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv --clear

# Reduced but VALID KB for fast ExpertWritten iteration:
# all gold articles + N random distractors (keeps a realistic haystack)
sudo docker compose exec graph-rag-agent \
  python etl/ingest.py \
    --primary  data/wixqa_expertwritten.csv \
    --secondary data/wix_kb_corpus.csv \
    --articles --distractor-sample 2000 --distractor-seed 42 --clear
```

> `--articles` alone ingests **only** gold articles, which shrinks the haystack and inflates retrieval scores — use `--distractor-sample N` to keep distractors. Scores on a reduced KB are **not comparable** to full-KB runs or the WixQA paper baselines; always report the corpus size. `--articles` gives no reduction for Synthetic (it references 100 % of the KB).

### Step 3 — Run the benchmark

The dataset is configurable (priority: `--dataset` → `$WIXQA_DATASET` → default `data/wixqa_expertwritten.csv`).

```bash
# ExpertWritten (default), top-k = 5
sudo docker compose exec graph-rag-agent python -m evaluate.wixqa --k 5

# Switch subset
sudo docker compose exec graph-rag-agent \
  python -m evaluate.wixqa --dataset data/wixqa_synthetic.csv --k 5

# Smoke-test on first N rows; add LLM-judged answer quality
sudo docker compose exec graph-rag-agent \
  python -m evaluate.wixqa --limit 20 --ragas

# Reproduce the WixQA paper metrics (§4.3), incl. Factuality
sudo docker compose exec graph-rag-agent \
  python -m evaluate.wixqa --dataset data/wixqa_expertwritten.csv --paper-metrics
```

A single pipeline pass per question yields four metric families, **logged in detail per query and as an aggregate summary**:

| Family | Metrics |
|---|---|
| **Retrieval** | Precision@k, Recall@k, MRR, NDCG@k (vs. `article_ids` grounding) |
| **Answer** | confidence, answer length, citation count |
| **Routing** | CRAG verdict + score, retries, ReAct iterations, deferral rate |
| **Cost** | latency, LLM calls, estimated prompt/completion tokens |

**`--paper-metrics`** adds the WixQA paper suite (§4.3) for direct comparison with the published baselines (`evaluate/paper_metrics.py`):

| Metric | Type | Compares |
|---|---|---|
| F1 / BLEU / ROUGE-1 / ROUGE-2 | lexical (no LLM) | generated answer vs. gold answer |
| **Factuality** | LLM judge (Appendix C prompt, 0–1) | generated answer vs. gold answer |
| **Context Recall** | LLM judge (Appendix D prompt, 0–1) | retrieved context vs. gold answer |

> The paper used GPT-4o as the judge; here the judge is whatever `CHAT_BACKEND` points to — use a strong model for fidelity. Lexical metrics are standard pure-Python implementations (no external deps) and may differ marginally from the paper's FlashRAG numbers. The judges run *after* each query's cost snapshot, so they don't inflate the reported pipeline cost.

`--ragas` adds RAGAS Faithfulness, Answer Correctness, and Context Recall (note: RAGAS Faithfulness measures grounding in *retrieved context*, which is **not** the same as the paper's Factuality, which compares to the *gold answer*). Outputs: a per-query CSV (`evaluate/results_<subset>_<timestamp>.csv`) and an aggregate `*.summary.json`. The CSV stores the full generated `answer` plus the Factuality / Context-Recall judge rationales, so a low-scoring row is diagnosable from the CSV alone.

### Known limitations & noise floor

Two properties of the benchmark to keep in mind when reading small-split numbers:

- **Judge variance / noise floor.** The LLM judges (CRAG grader and the paper Factuality/Context-Recall judges) run at temperature 0 but still vary run-to-run on *borderline* queries. Observed: the same ExpertWritten query (general "Google Ads pricing" vs. the KB's "Google Ads **with Wix**" content) flipped CRAG between `sufficient` and `defer` across identical runs, swinging that row's Factuality 0.0 ↔ 0.9. On a 40-row split this means aggregate deltas smaller than ≈0.02–0.03 are **noise, not signal** — do not tune on them. Larger runs (Synthetic, 6 221 rows) average this out.

- **Gold-label ambiguity.** Some gold answers resolve a genuinely ambiguous question in one particular direction; the system's alternative reading can be reasonable yet score 0. Documented example: *"How users can check my calendar on the website?"* — the phrasing reads as visitor-facing, but the gold answer is the **owner-side** procedure for adding a calendar app. The generator (correctly cautious) declines to invent visitor-viewing steps and instead notes the gap, scoring Factuality 0.0 despite retrieving the right passages. This is a benchmark-labeling artifact, not a retrieval or routing failure, and is intentionally **not** patched out — forcing the generator to answer through such ambiguity would degrade genuine owner-vs-visitor queries. The CRAG grader *is* hardened against over-parsing colloquial actor/perspective phrasings (so it no longer false-defers these), but the final answer remains the model's judgment call.

### Related: 4-layer eval CLI

`evaluate/eval.py` provides complementary layered runs over the same datasets:

```bash
python -m evaluate.eval --layer retrieval   # P@k / R@k / MRR / NDCG (no LLM cost)
python -m evaluate.eval --layer ragas       # Faithfulness / Correctness / Context Recall
python -m evaluate.eval --layer routing     # CRAG verdict distribution + deferral rate
python -m evaluate.eval --layer sweep       # context-engineering ablation matrix + cost
```

---

## 13. PIVOT Reference

For the full PIVOT (Plan-Inspect-eVOlve-Verify) design, including the failure
archetypes, the monotonic-acceptance rule, the skip-reflact gate and the A2A
deferral payload, see the accompanying paper (Sections 4 and 5).

The unified `[PIVOT]` log lines emitted by `agents/pivot_log.py` make every decision greppable — a single query trace shows:
```
PLAN → INSPECT → (EVOLVE | REACT | REFLACT) → VERIFY → (GENERATE | defer)
```
with the reason for each routing decision.
