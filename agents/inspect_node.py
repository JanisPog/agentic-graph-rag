"""PIVOT INSPECT node — routing decision after crag_grader.

Verdicts: skip_evolve | reflact | react | evolve | defer
Enforces the monotonic retry rule: EVOLVE only fires when the proposed
strategy is expected to improve and has not already been tried.
"""
from __future__ import annotations
import logging

import config
from agents.state import AgentState, ScoredChunk
from agents.llm import chat_json
from agents.kb_context import get_kb_context
from agents.pivot_log import pivot_log

log = logging.getLogger(__name__)

_INSPECT_PROMPT = """\
You are the INSPECT stage of a retrieval pipeline (PIVOT framework).

{kb_context_block}\
Original query: {query}
Plan intent: {intent}
Sub-goals (each must be satisfied): {sub_goals}
Evidence already found: {evidence_found}
Strategies already tried this query: {tried_strategies}

Current retrieval state:
  CRAG combined score: {score:.2f}
  CRAG verdict:        {verdict}
  Retries so far:      {retries}/{max_retries}

Retrieved passages (top 5 of {n_chunks}):
{context}

Answer the 3 INSPECT questions, then give a routing decision.

1. RESULT CHECK: For each sub-goal, is it covered by the current passages?
2. PLAN STATUS: For uncovered sub-goals, classify the gap:
   - "wrong_terms"   — KB has the info but uses different wording
   - "wrong_tool"    — a different retriever (bm25/graph) would find it
   - "kb_insufficient"  — content genuinely absent from the KB but may exist on the web
   - "irrecoverable"    — content cannot be found anywhere (unanswerable)
   - "none"          — nothing missing
3. REVISION: If a gap exists, what untried strategy should EVOLVE try?
   Will it score strictly higher than the current {score:.2f}?

Domain assumption rule: unless the user explicitly named a DIFFERENT product,
assume the query is about the documented KB system. Do NOT mark a sub-goal
missing just because passages don't explicitly name the KB's system.

Terminology rule: vocabulary differences between the user and the KB (e.g.
"delete" vs "disconnect") are NOT gaps — count those sub-goals as covered.

Return JSON only:
{{
  "covered_sub_goals":    ["<sub_goal text>", ...],
  "missing_sub_goals":    ["<sub_goal text>", ...],
  "gap_reason":          "wrong_terms|wrong_tool|kb_insufficient|irrecoverable|none",
  "revision_strategy":   "bm25_retry|graph_hop|surface_form|semantic_requery|web_search|none",
  "revised_query":        "<reformulated query string, or empty>",
  "expected_improvement": true,
  "irrecoverable_gaps":   []
}}
"""

_KB_HEADER = "Knowledge base context: {kb_context}\n\n"


def _fmt_chunks(chunks: list[ScoredChunk], limit: int = 5) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[{i+1}] {c.text[:400]}" for i, c in enumerate(chunks[:limit]))


def _safe_inspect_call(query: str, plan: dict, score: float, verdict: str,
                       chunks: list[ScoredChunk], retries: int,
                       evidence_found: list[str], tried: list[str]) -> dict:
    kb_ctx = get_kb_context()
    kb_block = _KB_HEADER.format(kb_context=kb_ctx) if kb_ctx else ""
    try:
        result = chat_json(_INSPECT_PROMPT.format(
            kb_context_block=kb_block,
            query=query,
            intent=plan.get("intent", "factual"),
            sub_goals=plan.get("sub_goals") or [query],
            evidence_found=evidence_found or "(none yet)",
            tried_strategies=tried or "(none yet)",
            score=score,
            verdict=verdict,
            retries=retries,
            max_retries=config.CRAG_MAX_RETRIES,
            n_chunks=len(chunks),
            context=_fmt_chunks(chunks),
        ))
        if not isinstance(result, dict):
            raise ValueError("INSPECT did not return a JSON object")
        return result
    except Exception as exc:
        log.warning("INSPECT LLM call failed (%s) — defaulting to passive verdict", exc)
        return {
            "covered_sub_goals": list(plan.get("sub_goals") or []),
            "missing_sub_goals": [],
            "gap_reason": "none",
            "revision_strategy": "none",
            "revised_query": "",
            "expected_improvement": False,
            "irrecoverable_gaps": [],
        }


def _decide_route(score: float, plan: dict, inspect_result: dict,
                  retries: int, tried: list[str]) -> tuple[str, str]:
    """Thin wrapper that suppresses deferral when ALLOW_DEFERRAL is False."""
    verdict, reason = _decide_route_core(score, plan, inspect_result, retries, tried)
    if config.ALLOW_DEFERRAL or verdict != "defer":
        return verdict, reason

    strategy = inspect_result.get("revision_strategy", "none")
    retries_left = retries < config.CRAG_MAX_RETRIES
    if retries_left and strategy not in ("none", "defer", None) and strategy not in tried:
        return "evolve", f"deferral_suppressed_evolve:{strategy}"
    # No viable retry — fall through to generation rather than defer.
    return "skip_evolve", f"deferral_suppressed_generate:{reason}"


def _decide_route_core(score: float, plan: dict, inspect_result: dict,
                       retries: int, tried: list[str]) -> tuple[str, str]:
    """Pure routing decision → (verdict, reason)."""
    skip_hint = bool(plan.get("skip_reflact", False))
    budget    = int(plan.get("react_budget", 0))
    conf_target = float(plan.get("confidence_target", config.CRAG_SUFFICIENT_THRESH))

    if score >= config.CRAG_SKIP_REACT_THRESH:
        return "skip_evolve", "score_above_skip_react_thresh"

    if score >= conf_target:
        if skip_hint or budget <= 0:
            return "skip_evolve", "score_sufficient_and_skip_hint_or_no_budget"
        missing = inspect_result.get("missing_sub_goals") or []
        gap_reason = inspect_result.get("gap_reason", "none")
        if not missing and gap_reason in ("none", None, ""):
            return "skip_evolve", "inspect_confirms_full_coverage"
        return ("reflact" if config.USE_REFLACT else "react",
                "score_sufficient_budget_remaining")

    if retries >= config.CRAG_MAX_RETRIES:
        if score < config.CRAG_RETRY_THRESH:
            return "defer", "max_retries_reached_below_floor"
        return "defer", "max_retries_reached"

    if score < config.CRAG_RETRY_THRESH:
        gap_reason_early = inspect_result.get("gap_reason", "none")
        if gap_reason_early == "irrecoverable":
            return "defer", "score_below_retry_thresh_irrecoverable"

    gap_reason = inspect_result.get("gap_reason", "none")
    if gap_reason == "irrecoverable":
        return "defer", "irrecoverable_gap"

    strategy = inspect_result.get("revision_strategy", "none")
    expected = bool(inspect_result.get("expected_improvement", False))

    if strategy == "none" or strategy == "defer":
        return "defer", "no_viable_strategy"

    if config.INSPECT_REQUIRE_IMPROVEMENT:
        if not expected:
            return "defer", "no_expected_improvement"
        if strategy in tried:
            return "defer", f"strategy_already_tried:{strategy}"

    return "evolve", f"monotonic_ok:{strategy}"


def inspect_node(state: AgentState) -> dict:
    plan      = state.get("plan") or {}
    score     = float(state.get("context_quality", 0.0))
    verdict   = state.get("crag_verdict", "sufficient")
    chunks    = state.get("reranked_hits") or []
    retries   = int(state.get("crag_retries", 0))
    tried     = list(state.get("tried_strategies") or [])
    sub_goals = plan.get("sub_goals") or [state["original_query"]]
    evidence_found = list(state.get("evidence_found") or [])
    query     = state["original_query"]

    if score >= config.CRAG_SKIP_REACT_THRESH and not state.get("evidence_gaps"):
        route, reason = "skip_evolve", "score_above_skip_react_thresh"
        inspect_result = {
            "covered_sub_goals": list(sub_goals),
            "missing_sub_goals": [],
            "gap_reason": "none",
            "revision_strategy": "none",
            "revised_query": "",
            "expected_improvement": False,
            "irrecoverable_gaps": [],
        }
    else:
        inspect_result = _safe_inspect_call(
            query, plan, score, verdict, chunks, retries, evidence_found, tried,
        )
        route, reason = _decide_route(score, plan, inspect_result, retries, tried)

    covered = inspect_result.get("covered_sub_goals") or []
    missing = inspect_result.get("missing_sub_goals") or []
    new_found = list(dict.fromkeys(evidence_found + list(covered)))
    new_gaps  = [g for g in (state.get("evidence_gaps") or sub_goals) if g not in new_found]
    if missing:
        for m in missing:
            if m not in new_gaps:
                new_gaps.append(m)

    pivot_log(
        "INSPECT",
        decision=route,
        reason=reason,
        score=f"{score:.2f}",
        retry=f"{retries}/{config.CRAG_MAX_RETRIES}",
        covered=f"{len(new_found)}/{len(sub_goals)}",
        gap_reason=inspect_result.get("gap_reason", "none"),
        next_strategy=inspect_result.get("revision_strategy", "none"),
    )

    updates: dict = {
        "inspect_verdict": route,
        "inspect_reason":  reason,
        "evidence_found":  new_found,
        "evidence_gaps":   new_gaps,
    }
    if route == "evolve":
        strategy = inspect_result.get("revision_strategy", "semantic_requery")
        if strategy == "web_search" and not config.WEB_SEARCH_ENABLED:
            log.info("INSPECT chose web_search but it is disabled — using semantic_requery")
            strategy = "semantic_requery"
        updates["evolve_strategy"] = strategy
        revised = (inspect_result.get("revised_query") or "").strip()
        if revised:
            updates["crag_missing_hint"] = revised
    return updates
