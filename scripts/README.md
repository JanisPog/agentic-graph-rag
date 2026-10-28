# `scripts/`

Standalone smoke tests and one-off utilities. These are **not** part of the
request path — they exist to verify behaviour that is awkward to unit-test
because it depends on a populated Neo4j instance or a live LLM endpoint.

Run them from the repository root with a configured `.env`:

```bash
python scripts/check_metadata_retrieval.py
```

| Script | Verifies |
|---|---|
| `check_context_prefix_v3.py` | `build_context_prefix_v3()` output across chunk scenarios |
| `check_document_dates.py` | Date extraction/normalisation and `created_at` / `updated_at` / `ingested_at` on doc records |
| `check_metadata_filters.py` | BM25 and graph metadata filter logic |
| `check_metadata_retrieval.py` | Semantic, BM25 and graph retrievers all return `ScoredChunk`s with populated metadata |
| `check_paperless_graph_contract.py` | Paperless ingestion fulfils the graph contract |
| `check_planner_metadata_filters.py` | Planner's deterministic metadata-filter normaliser (runs a live LLM call) |
| `check_reranker_recency.py` | Reranker applies recency boosts only to genuine domain date fields, ignoring `ingested_at` |
| `create_checkout_known_issue.py` | Utility: seeds a sample `KnownIssue` node for manual testing |

Each prints its findings and exits non-zero on failure.
