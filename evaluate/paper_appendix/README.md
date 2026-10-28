# Paper Appendix (≤ 4 pages)

Consolidated supplementary material for the WixQA evaluation. Each file is a self-contained,
paper-ready section (tables / prompts / commands / configs — not narrative).

| File | Content | Est. pages |
|---|---|---|
| `A1_full_results.md` | Full results grid (all subsets × metrics × model/judge/fast-path variants) + paper baselines + AgenticRAG comparison | ~1.25 |
| `A2_judge_prompts.md` | Factuality & Context-Recall judge prompts (verbatim) + lexical-metric note | ~0.75 |
| `A3_reproduction.md` | Datasets, train/test split, ingest + eval commands, model/judge config, backend caveat | ~0.5 |
| `A4_system_behaviour.md` | Method-invocation profile, per-node LLM-call breakdown, routing distribution, calibration, cost profile | ~1.0 |
| `A5_hyperparams_and_negative_results.md` | All hyperparameters + 5 negative/cautionary results (fast-track, top-k, EVOLVE bias, graph seeding, deferral) | ~0.75 |

**Total ≈ 4 pages.** Trim order if over budget: drop the 8B/older-config rows from A1 (keep
35B + baselines), then condense A4.2's per-node table.

## Caveats carried throughout (state once in the paper)
- **Judge:** absolute Factuality is judged per-system by its own generation family (paper =
  GPT-4o, AgenticRAG = GPT-5-mini, this work = Qwen/Gemma); lexical metrics are
  judge-independent and corroborate the ranking. Self-judged result validated against a neutral
  Gemma judge (≈0.01 difference).
- **Backend:** absolute quality is serving-sensitive (gateway vs FP8 vLLM, ≈0.88 vs ≈0.71);
  routing/tool-use behaviour is stable.
- **Sample size:** ExpertWritten N=40 (±0.03 noise); Simulated N=200, Synthetic N=500 are solid.
- **Per-node costs** are reconstructed to match measured totals, not directly measured.
