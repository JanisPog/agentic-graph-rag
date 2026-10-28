"""Behavioural analysis of a WixQA result CSV — for the technical report.

Reads a results_wixqa_*.csv produced by evaluate/wixqa.py (with the report
instrumentation columns) and prints, in one pass:

  Q1  Retries / iterations     — avg CRAG retries, avg ReAct iterations
  Q2  Retrieval method used     — which arms fired, plan intent distribution,
                                  EVOLVE strategy usage
  Q3  Surface-form contribution — Factuality with vs without surface-form seeds
  Q4  Graph traversal extent    — how often the graph fired, avg seeds / nodes
  +   Routing                   — CRAG verdicts, deferral rate
  +   Quality by arm            — Factuality grouped by which arms ran

Pure standard library (no pandas). Works on any result CSV; columns added later
are picked up automatically, missing ones are skipped with a note.

Usage::

    python evaluate/analyze_report.py evaluate/results_wixqa_expertwritten_<ts>.csv
    # or let it pick the most recent:
    python evaluate/analyze_report.py
"""
from __future__ import annotations

import csv
import glob
import sys
from collections import Counter
from statistics import mean


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in ("", "None", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _i(row: dict, key: str, default: int = 0) -> int:
    return int(_f(row, key, default))


def _has(rows: list[dict], col: str) -> bool:
    return bool(rows) and col in rows[0]


def _pct(n: int, total: int) -> str:
    return f"{100*n/total:.1f}%" if total else "—"


def _mean_or_na(vals: list[float]) -> str:
    return f"{mean(vals):.3f}" if vals else "n/a"


def analyze(path: str) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if not n:
        print("No rows in", path)
        return

    print("=" * 72)
    print("WixQA behavioural analysis —", path)
    print(f"  queries: {n}")
    print("=" * 72)

    # ── Q1: retries / iterations ──────────────────────────────────────────────
    print("\n── Q1  Retries & iterations ──────────────────────────────")
    if _has(rows, "crag_retries"):
        retries = [_i(r, "crag_retries") for r in rows]
        react = [_i(r, "react_iterations") for r in rows]
        print(f"  avg CRAG retries        : {mean(retries):.2f}  (max {max(retries)})")
        print(f"  avg ReAct iterations    : {mean(react):.2f}  (max {max(react)})")
        print(f"  queries with >=1 retry  : {sum(1 for x in retries if x):d} ({_pct(sum(1 for x in retries if x), n)})")
    else:
        print("  (crag_retries column absent — re-run with current code)")

    # ── Q2: retrieval method selection ────────────────────────────────────────
    print("\n── Q2  Retrieval method selected ─────────────────────────")
    if _has(rows, "arms_used"):
        arms = Counter(r.get("arms_used", "") for r in rows)
        print("  arms fired (per query):")
        for combo, c in arms.most_common():
            print(f"    {combo or '(none)':28} {c:4d}  ({_pct(c, n)})")
        # individual arm participation
        indiv = Counter()
        for r in rows:
            for a in (r.get("arms_used") or "").split("+"):
                if a and a != "(none)":
                    indiv[a] += 1
        print("  individual arm participation:")
        for a, c in indiv.most_common():
            print(f"    {a:28} {c:4d}  ({_pct(c, n)})")
    else:
        print("  (arms_used column absent — re-run with current code)")
    if _has(rows, "plan_intent"):
        intents = Counter(r.get("plan_intent", "") for r in rows)
        print("  plan intent:")
        for it, c in intents.most_common():
            print(f"    {it or '(none)':28} {c:4d}  ({_pct(c, n)})")
    if _has(rows, "evolve_strategy"):
        evolved = [r for r in rows if (r.get("evolve_strategy") or "").strip()]
        print(f"  EVOLVE fired            : {len(evolved):d} ({_pct(len(evolved), n)})")
        if evolved:
            for s, c in Counter(r["evolve_strategy"] for r in evolved).most_common():
                print(f"    strategy {s:20} {c:4d}")

    # ── Q3: surface-form contribution ─────────────────────────────────────────
    print("\n── Q3  Surface-form contribution ─────────────────────────")
    if _has(rows, "graph_entity_seeds") and _has(rows, "paper_factuality"):
        with_sf = [_f(r, "paper_factuality") for r in rows
                   if _i(r, "graph_entity_seeds") > 0 and r.get("paper_factuality") not in ("", "None")]
        without_sf = [_f(r, "paper_factuality") for r in rows
                      if _i(r, "graph_entity_seeds") == 0 and r.get("paper_factuality") not in ("", "None")]
        print(f"  queries WITH surface-form seeds : {len(with_sf):d}  → Factuality {_mean_or_na(with_sf)}")
        print(f"  queries WITHOUT                 : {len(without_sf):d}  → Factuality {_mean_or_na(without_sf)}")
        print("  (correlation only — causal proof needs a surface-form-off ablation)")
    else:
        print("  (graph_entity_seeds / paper_factuality absent)")

    # ── Q4: graph traversal extent ────────────────────────────────────────────
    print("\n── Q4  Graph traversal extent ────────────────────────────")
    if _has(rows, "graph_seeds"):
        fired = [r for r in rows if _i(r, "graph_seeds") > 0]
        print(f"  graph arm fired         : {len(fired):d} ({_pct(len(fired), n)})")
        if fired:
            print(f"  avg PPR seeds (when fired)  : {mean(_i(r,'graph_seeds') for r in fired):.1f}")
            print(f"  avg entity seeds (surf.form): {mean(_i(r,'graph_entity_seeds') for r in fired):.1f}")
            print(f"  avg nodes returned          : {mean(_i(r,'graph_nodes') for r in fired):.1f}")
            print(f"  max PPR seeds               : {max(_i(r,'graph_seeds') for r in fired)}")
    else:
        print("  (graph_seeds column absent — re-run with current code)")

    # ── Routing summary ───────────────────────────────────────────────────────
    print("\n── Routing ───────────────────────────────────────────────")
    if _has(rows, "crag_verdict"):
        verdicts = Counter(r.get("crag_verdict", "") for r in rows)
        print("  CRAG verdicts:", dict(verdicts))
        deferred = sum(1 for r in rows if str(r.get("deferred")).lower() == "true")
        print(f"  deferral rate           : {_pct(deferred, n)}")
    if _has(rows, "fasttrack_skip"):
        ft = sum(1 for r in rows if str(r.get("fasttrack_skip")).lower() == "true")
        print(f"  fast-track hard-skips   : {ft:d} ({_pct(ft, n)})")

    # ── Quality by arm combination ────────────────────────────────────────────
    if _has(rows, "arms_used") and _has(rows, "paper_factuality"):
        print("\n── Factuality by arm combination ─────────────────────────")
        by_arm: dict[str, list[float]] = {}
        for r in rows:
            if r.get("paper_factuality") in ("", "None"):
                continue
            by_arm.setdefault(r.get("arms_used", ""), []).append(_f(r, "paper_factuality"))
        for combo, vals in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
            print(f"    {combo or '(none)':28} n={len(vals):3d}  Factuality {_mean_or_na(vals)}")

    # ── RQ1: INSPECT routing decisions — WHY retries were skipped ─────────────
    # Answers: did the monotonic acceptance rule fire, or was score high enough
    # to short-circuit INSPECT entirely?  Maps each inspect_reason to its count
    # and, where factuality is available, its average quality.
    if _has(rows, "inspect_reason"):
        print("\n── RQ1  INSPECT routing reasons (monotonic acceptance) ────")
        reason_counts: dict[str, int] = {}
        reason_fact:   dict[str, list[float]] = {}
        for r in rows:
            reason = r.get("inspect_reason") or "(unknown)"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if r.get("paper_factuality") not in ("", "None"):
                reason_fact.setdefault(reason, []).append(_f(r, "paper_factuality"))
        for reason, cnt in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            fact_str = f"  Factuality {_mean_or_na(reason_fact.get(reason, []))}" \
                       if reason_fact.get(reason) else ""
            print(f"    {reason:42} {cnt:4d}  ({_pct(cnt, n)}){fact_str}")
        print("  (score_above_skip_react_thresh → INSPECT LLM call saved entirely)")
        print("  (inspect_confirms_full_coverage → sub-goals all met, no retry needed)")
    else:
        print("\n── RQ1  INSPECT routing reasons ──────────────────────────")
        print("  (inspect_reason column absent — re-run with current code)")

    # ── RQ1: LLM call efficiency — cost of skipping vs. retrying ─────────────
    if _has(rows, "llm_calls") and _has(rows, "inspect_reason"):
        print("\n── RQ1  LLM call cost by routing path ─────────────────────")
        by_reason: dict[str, list[int]] = {}
        for r in rows:
            reason = r.get("inspect_reason") or "(unknown)"
            by_reason.setdefault(reason, []).append(_i(r, "llm_calls"))
        for reason, calls in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"    {reason:42} avg_llm_calls={mean(calls):.2f}  n={len(calls)}")

    # ── RQ2: Planner arm-selection explainability ─────────────────────────────
    # Shows which arms the planner deliberately skipped and how often, so we can
    # verify the planner only activates arms that benefit the query type.
    if _has(rows, "arms_skipped"):
        print("\n── RQ2  Planner arm-skip decisions ────────────────────────")
        skip_counts: dict[str, int] = {}
        skip_fact:   dict[str, list[float]] = {}
        for r in rows:
            skipped = r.get("arms_skipped") or "(none)"
            skip_counts[skipped] = skip_counts.get(skipped, 0) + 1
            if r.get("paper_factuality") not in ("", "None"):
                skip_fact.setdefault(skipped, []).append(_f(r, "paper_factuality"))
        for skipped, cnt in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            fact_str = f"  Factuality {_mean_or_na(skip_fact.get(skipped, []))}" \
                       if skip_fact.get(skipped) else ""
            print(f"    skipped={skipped:36} {cnt:4d}  ({_pct(cnt, n)}){fact_str}")
    else:
        print("\n── RQ2  Planner arm-skip decisions ────────────────────────")
        print("  (arms_skipped column absent — re-run with current code)")

    # ── RQ2: Planner intent → arm selection cross-tab ─────────────────────────
    # Shows whether the planner consistently maps each intent to the right arms.
    if _has(rows, "plan_intent") and _has(rows, "arms_used"):
        print("\n── RQ2  Plan intent × arm combination ─────────────────────")
        intent_arm: dict[str, dict[str, int]] = {}
        for r in rows:
            intent = r.get("plan_intent") or "(none)"
            arms   = r.get("arms_used") or "(none)"
            intent_arm.setdefault(intent, {})
            intent_arm[intent][arms] = intent_arm[intent].get(arms, 0) + 1
        for intent, arm_counts in sorted(intent_arm.items()):
            total_intent = sum(arm_counts.values())
            print(f"  {intent}  (n={total_intent})")
            for arms, cnt in sorted(arm_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {arms:44} {cnt:4d}  ({_pct(cnt, total_intent)})")

    # ── RQ2: ReAct budget — was multi-hop reasoning ever triggered? ───────────
    if _has(rows, "plan_react_budget"):
        print("\n── RQ2  Planner ReAct budget allocation ───────────────────")
        budget_counts = Counter(_i(r, "plan_react_budget") for r in rows)
        for budget, cnt in sorted(budget_counts.items()):
            print(f"    react_budget={budget}  {cnt:4d} queries  ({_pct(cnt, n)})")
        if _has(rows, "react_iterations"):
            used = [r for r in rows if _i(r, "react_iterations") > 0]
            budgeted = [r for r in rows if _i(r, "plan_react_budget") > 0]
            print(f"  queries with budget>0    : {len(budgeted)} ({_pct(len(budgeted), n)})")
            print(f"  queries that actually ran ReAct: {len(used)} ({_pct(len(used), n)})")
    else:
        print("\n── RQ2  Planner ReAct budget ──────────────────────────────")
        print("  (plan_react_budget column absent — re-run with current code)")

    # ── RQ1+RQ2: CRAG score distribution — threshold coverage ─────────────────
    # Shows how scores distribute around the two routing thresholds
    # (CRAG_SUFFICIENT_THRESH=0.70, CRAG_SKIP_REACT_THRESH=0.85) to explain
    # why almost no retries fired.
    if _has(rows, "crag_score"):
        print("\n── RQ1+RQ2  CRAG score distribution (threshold context) ───")
        scores = [_f(r, "crag_score") for r in rows]
        def _band(s: float) -> int:
            if s < 0.30: return 0
            if s < 0.70: return 1
            if s < 0.85: return 2
            return 3
        band_labels = [
            "score < 0.30  (hard-defer floor)",
            "0.30 ≤ score < 0.70  (retry zone)",
            "0.70 ≤ score < 0.85  (sufficient, reflact/react if budget)",
            "score ≥ 0.85  (skip_react_thresh → INSPECT LLM skipped)",
        ]
        band_counts  = [0] * 4
        band_facts:  list[list[float]] = [[] for _ in range(4)]
        for r in rows:
            b = _band(_f(r, "crag_score"))
            band_counts[b] += 1
            if _has(rows, "paper_factuality") and r.get("paper_factuality") not in ("", "None"):
                band_facts[b].append(_f(r, "paper_factuality"))
        for b, label in enumerate(band_labels):
            fact_str = f"  Factuality {_mean_or_na(band_facts[b])}" if band_facts[b] else ""
            print(f"    {label:52} n={band_counts[b]:3d}  ({_pct(band_counts[b], n)}){fact_str}")

    print("\n" + "=" * 72)


def _is_instrumented(path: str) -> bool:
    """True if the CSV has the report-instrumentation columns (post-update runs)."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        return "arms_used" in header and "graph_seeds" in header
    except OSError:
        return False


def main() -> None:
    import os
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        candidates = glob.glob("evaluate/results_wixqa_*.csv") or \
                     glob.glob("/app/evaluate/results_wixqa_*.csv")
        if not candidates:
            print("No result CSV found. Pass a path: python evaluate/analyze_report.py <csv>")
            sys.exit(1)
        # Sort by ACTUAL modification time (newest first) — NOT alphabetical, which
        # wrongly preferred 'synthetic' > 'expertwritten' regardless of run time.
        candidates.sort(key=os.path.getmtime, reverse=True)
        # Prefer the newest run that actually has the instrumentation columns; fall
        # back to the newest overall if none are instrumented (with a warning).
        instrumented = [c for c in candidates if _is_instrumented(c)]
        if instrumented:
            path = instrumented[0]
            print(f"(no path given — newest instrumented run: {path})\n")
        else:
            path = candidates[0]
            print(f"(no path given — newest run: {path})")
            print("  WARNING: no instrumented CSV found; Q2/Q3/Q4 will be blank. "
                  "Re-run evaluate/wixqa.py with current code.\n")
    analyze(path)


if __name__ == "__main__":
    main()
