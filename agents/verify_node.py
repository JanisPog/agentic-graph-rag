"""PIVOT VERIFY node — final sub-goal coverage check before generation."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState, ScoredChunk
from agents.llm import chat_json
from agents.kb_context import get_kb_context
from agents.pivot_log import pivot_log

log = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
You are the VERIFY stage of a retrieval pipeline (PIVOT framework).
Your task is the final constraint check before the answer is written.

{kb_context_block}\
Original query: {original_query}

Sub-goals (each must be satisfied by the retrieved evidence):
{sub_goals}

Retrieved evidence (top {n} passages):
{evidence}

Instructions:
1. Re-read the original query carefully.
2. For each sub-goal, decide if the evidence supports a complete answer.
   - "satisfied": evidence fully covers this sub-goal.
   - "partial":   evidence covers some but not all.
   - "missing":   evidence does not cover this sub-goal.
3. Do NOT discard your work — if some sub-goals are satisfied and others are
   not, prefer a partial answer over deferral.
4. Only recommend "defer" when fewer than {min_coverage:.0%} of sub-goals
   are even partially satisfied AND there is no useful partial answer.

KB terminology rule: if the KB uses a different but equivalent term for what
the user asked (e.g. "disconnect" instead of "delete"), count the sub-goal
as satisfied — vocabulary differences are NOT gaps.

Domain assumption rule: this agent serves a single knowledge base whose scope
is described in the KB context above. Unless the user explicitly named a
DIFFERENT product (e.g. "in Shopify", "for my Gmail account"), assume the
query is about the documented system. Do NOT mark a sub-goal as missing or
partial just because the retrieved passages don't explicitly name the KB's
system — that context is implicit and the answer is still valid.

Return JSON only:
{{
  "sub_goal_status": [
    {{"goal": "<text>", "status": "satisfied|partial|missing", "note": "<short>"}},
    ...
  ],
  "all_satisfied":      true,
  "irrecoverable_gaps": ["<gap description>"],
  "partial_note":       "<short user-facing note if some goals only partial/missing>",
  "proceed":            "generate|generate_with_partial_note|defer"
}}
"""

_KB_HEADER = "Knowledge base context: {kb_context}\n\n"


def _fmt_evidence(chunks: list[ScoredChunk], limit: int = 8) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[{i+1}] {c.text[:400]}" for i, c in enumerate(chunks[:limit]))


def verify_node(state: AgentState) -> dict:
    plan      = state.get("plan") or {}
    sub_goals = plan.get("sub_goals") or [state["original_query"]]
    chunks    = state.get("reranked_hits") or []
    kb_ctx    = get_kb_context()
    kb_block  = _KB_HEADER.format(kb_context=kb_ctx) if kb_ctx else ""

    try:
        result = chat_json(_VERIFY_PROMPT.format(
            kb_context_block=kb_block,
            original_query=state["original_query"],
            sub_goals="\n".join(f"  - {g}" for g in sub_goals),
            evidence=_fmt_evidence(chunks),
            n=len(chunks),
            min_coverage=config.VERIFY_MIN_COVERAGE,
        ))
        if not isinstance(result, dict):
            raise ValueError("VERIFY did not return a JSON object")
    except Exception as exc:
        log.warning("VERIFY LLM call failed (%s) — defaulting to generate", exc)
        result = {
            "sub_goal_status": [{"goal": g, "status": "satisfied", "note": ""} for g in sub_goals],
            "all_satisfied": True,
            "irrecoverable_gaps": [],
            "partial_note": "",
            "proceed": "generate",
        }

    statuses = result.get("sub_goal_status") or []
    satisfied = [s.get("goal", "") for s in statuses if s.get("status") == "satisfied"]
    partial   = [s.get("goal", "") for s in statuses if s.get("status") == "partial"]
    missing   = [s.get("goal", "") for s in statuses if s.get("status") == "missing"]
    covered   = satisfied + partial

    n_goals = max(1, len(sub_goals))
    coverage_ratio = (len(satisfied) + 0.5 * len(partial)) / n_goals

    proceed_hint = result.get("proceed", "generate")
    partial_note = (result.get("partial_note") or "").strip()

    if (partial or missing) and not partial_note:
        gap_list = ", ".join((missing or partial)[:3])
        partial_note = f"Note: the following could not be fully answered from the knowledge base: {gap_list}."

    # CRAG score gates VERIFY: if CRAG says sufficient, VERIFY cannot defer.
    score = float(state.get("context_quality", 0.0))
    crag_sufficient = score >= config.CRAG_SUFFICIENT_THRESH

    if crag_sufficient:
        route = "generator"
        if missing or partial:
            decision = "generate_with_partial_note"
        else:
            decision = "generate"
            partial_note = ""
    elif coverage_ratio >= config.VERIFY_MIN_COVERAGE or satisfied:
        # CRAG was borderline but VERIFY found enough coverage for a useful answer.
        route = "generator"
        decision = "generate_with_partial_note" if (missing or partial) else "generate"
        if decision == "generate":
            partial_note = ""
    elif not config.ALLOW_DEFERRAL:
        route = "generator"
        decision = "generate_with_partial_note"
    else:
        route = "deferral_handler"
        decision = "defer"

    reason = (
        "crag_sufficient_overrides_verify" if (crag_sufficient and (missing or partial))
        else f"coverage={coverage_ratio:.2f}_min={config.VERIFY_MIN_COVERAGE}"
    )
    pivot_log(
        "VERIFY",
        decision=decision,
        reason=reason,
        crag=f"{score:.2f}",
        coverage=f"{len(satisfied)}+{len(partial)}p/{n_goals}",
        partial=bool(partial or missing),
        proceed_hint=proceed_hint,
    )

    return {
        "verify_result":       result,
        "evidence_found":      covered,
        "evidence_gaps":       missing,
        "goal_achieved":       not (missing or partial),
        "partial_answer_note": partial_note,
        "inspect_verdict":     decision,
    }
