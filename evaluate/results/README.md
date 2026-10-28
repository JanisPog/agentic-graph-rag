# `evaluate/results/`

Committed evaluation output that backs the numbers reported in the paper.
Everything here is regenerable — it is committed so the reported figures can be
inspected without access to a populated Neo4j instance.

## `appendix/`

Retrieval-side statistics over the three WixQA splits (`expert`, `simulated`,
`synthetic`), 200 queries each (n = 600). Produced by:

```bash
python evaluate/appendix_stats.py
```

| File | Contents |
|---|---|
| `summary.md` | Human-readable tables: corpus size, per-split retrieval geometry |
| `query_aggregate.json` | Aggregated metrics per split |
| `per_query.csv` | Flat per-query metrics (cosine stats, gold ranks, hit@k) |
| `per_query.jsonl` | Same, plus question text and gold article IDs |

The narrative discussion of these numbers, together with the figures in
`docs/figures/`, lives in [`docs/appendix_retrieval_stats.md`](../../docs/appendix_retrieval_stats.md).

Superseded runs (an earlier 150-query run, and a pre-bugfix run of the same 600
queries) have been removed; only the final run is kept.
