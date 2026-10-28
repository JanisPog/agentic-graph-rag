"""ReflAct reasoning loop — thinker node."""
from __future__ import annotations
import logging
import unicodedata

import config
from agents.state import AgentState, ScoredChunk, ReflActStep
from agents.llm import chat_json
from agents.kb_context import get_kb_context

log = logging.getLogger(__name__)

_REFLACT_PROMPT = """\
You are a retrieval agent solving a support query.

{kb_context_block}\
ORIGINAL GOAL:
{original_query}

ACCUMULATED FINDINGS:
{reflection_summary}

STEP HISTORY:
{scratchpad}

CURRENT EVIDENCE PASSAGES:
{context}

AVAILABLE TOOLS:
semantic_search | bm25_search | graph_hop | no_action
{pivot_hint}
First reflect on the current evidence state in relation to the original goal.
Your reflection must explicitly mention:
- what is already known,
- what is still missing,
- whether the current evidence is sufficient to answer the goal.

Then choose exactly one next action.

Set goal_achieved=true ONLY if the current evidence is sufficient AND no further retrieval is needed. In that case, you MUST set action='no_action'.

Return JSON only:
{{
  "reflection": "<state/evidence reflection in relation to the goal>",
  "summary_update": "<1 short sentence capturing only the key new fact or insight>",
  "action": "semantic_search | bm25_search | graph_hop | no_action",
  "sub_query": "<precise sub-query string>",
  "expected_evidence": "<what evidence would close the gap>",
  "goal_achieved": true | false
}}"""

def _fmt_reflact_scratchpad(steps: list[ReflActStep]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(
        f"Step {i+1}:\n  Reflection: {s.reflection}\n  Action: {s.action}({s.sub_query})\n  Observation: {s.observation}"
        for i, s in enumerate(steps)
    )

def _fmt_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks[:10]))

def reflact_thinker(state: AgentState) -> dict:
    pivot = state.get("pivot_required", False)
    pivot_hint = (
        "\n⚠️  PIVOT REQUIRED: use a different tool and different query than the last step.\n"
        if pivot else ""
    )
    context = _fmt_chunks(state.get("reranked_hits") or [])
    scratchpad_list = list(state.get("reflact_scratchpad") or [])
    scratchpad = _fmt_reflact_scratchpad(scratchpad_list)
    reflection_summary = state.get("reflection_summary") or "(none yet)"

    kb_ctx = get_kb_context()
    kb_context_block = f"Knowledge base context: {kb_ctx}\n\n" if kb_ctx else ""

    try:
        result = chat_json(_REFLACT_PROMPT.format(
            original_query=state["original_query"],
            reflection_summary=reflection_summary,
            scratchpad=scratchpad,
            context=context,
            pivot_hint=pivot_hint,
            kb_context_block=kb_context_block,
        ), temperature=config.TEMPERATURE_REASON)
        reflection = result.get("reflection", "")
        summary_update = result.get("summary_update", "")
        action = result.get("action", "no_action")
        sub_query = result.get("sub_query") or state["original_query"]
        expected_evidence = result.get("expected_evidence", "")
        goal_achieved = bool(result.get("goal_achieved", False))
        
        if goal_achieved:
            action = "no_action"
        if action != "no_action":
            goal_achieved = False
    except Exception as exc:
        log.warning("reflact_thinker failed (%s); choosing no_action", exc)
        reflection = f"Error during reflection: {exc}"
        summary_update = ""
        action = "no_action"
        sub_query = state["original_query"]
        expected_evidence = ""
        goal_achieved = False

    iteration = state.get("react_iterations", 0) + 1
    log.info("ReflAct [iter %d] reflection: %s", iteration, reflection)
    log.info("ReflAct [iter %d] action:  %s(%r)", iteration, action, sub_query)

    step = ReflActStep(
        reflection=reflection,
        action=action,
        sub_query=sub_query,
        expected_evidence=expected_evidence
    )
    scratchpad_list.append(step)

    if summary_update:
        base = "" if reflection_summary == "(none yet)" else reflection_summary.rstrip()
        updated_summary = (base + f"\n- [Iter {iteration}] {summary_update}").strip()
    else:
        updated_summary = reflection_summary

    return {
        "reflact_scratchpad": scratchpad_list,
        "react_iterations": iteration,
        "reflection_summary": updated_summary,
        "goal_achieved": goal_achieved,
    }
