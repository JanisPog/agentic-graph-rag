"""CRAG context quality gate — single LLM call scoring recall + precision.

score = max(recall, 0.7*recall + 0.3*precision)
Routing: sufficient / retry / defer based on CRAG_*_THRESH config values.
missing_information drives the focused re-query hint on retry.
"""
from __future__ import annotations
import logging

import config
from agents.state import AgentState, ScoredChunk
from agents.llm import chat_json
from agents.kb_context import get_kb_context

log = logging.getLogger(__name__)

_PROMPT = """\
You are a retrieval auditor. Evaluate the retrieved context passages against the user query.

{kb_context_block}\
{deferral_block}\
User query: {query}

Retrieved context (each passage is numbered):
{context}

Perform two evaluations in a single pass:

1. RECALL — does the context contain the key facts needed to fully answer the query?
   Important scoring rules:
   - If the KB uses a different but equivalent term for what the user asks (e.g. the user
     says "delete" but the KB says "disconnect" or "remove"), treat that as FULL coverage —
     do NOT penalise recall for vocabulary differences when the procedure is the same.
   - Only mark recall as incomplete when a genuinely different action or fact is missing.
   - Queries outside the KB's domain (e.g. a question about Gmail when the KB covers {org_name})
     should score recall=0.0 and note the domain mismatch clearly.
   - Do NOT invent requirements that the user did not ask for. Evaluate only what the user
     explicitly requested — do not penalise recall for more specific sub-cases or edge cases
     that were not mentioned in the query. If the user asks "how do I X" and the context
     explains how to X (even under a slightly different label), recall is satisfied.
   - If the user asks about a general feature (e.g. "payment method", "currency") and the
     context covers that feature at the site/account level, treat that as sufficient unless
     the user specifically asked about a narrower context (e.g. "inside the Courses editor").
   - Do NOT over-parse the ACTOR or PERSPECTIVE in a colloquial query. Phrasings like
     "how users can check my calendar" or "how people see my page" usually just mean "how
     do I set up / enable this feature", not a demand for separate visitor-side instructions.
     If the context provides the procedure for the feature, recall is satisfied — do NOT
     split the query into owner-vs-visitor (or admin-vs-end-user) actors and penalise recall
     for a perspective the user did not explicitly emphasise.

2. PRECISION — for each passage, decide if it contains information relevant to
   answering the query (relevant) or is noise/off-topic (not relevant).

Return JSON only:
{{
  "recall_score": <float 0.0 to 1.0>,
  "missing_information": "<concise description of what is absent, or empty string if complete>",
  "relevant_ids": [<list of 1-based passage numbers that are relevant>],
  "precision_score": <float: relevant_count / total_count>
}}"""

_KB_CONTEXT_HEADER = """\
Knowledge base context (use this to interpret the query correctly):
{kb_context}

"""

_DEFERRAL_HEADER = """\
Previous attempt context: A prior retrieval pass for this conversation was \
deferred because the following information could not be found: {gaps}. \
The user has now provided clarification: "{clarification}". \
If the retrieved passages still do not cover these gaps, mark recall lower \
accordingly — do not give credit for gaps that remain unresolved.

"""


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))


def _build_deferral_block(deferral_context: dict | None) -> str:
    if not deferral_context:
        return ""
    gaps = "; ".join(deferral_context.get("gaps") or []) or "unspecified"
    clarification = deferral_context.get("user_clarification") or "(none)"
    return _DEFERRAL_HEADER.format(gaps=gaps, clarification=clarification)


def _judge(query: str, context: str,
           deferral_context: dict | None = None) -> tuple[float, list[int] | None, str]:
    """Single LLM call → (recall, relevant_ids, missing_hint). relevant_ids is None on failure."""
    try:
        kb_ctx = get_kb_context()
        kb_context_block = _KB_CONTEXT_HEADER.format(kb_context=kb_ctx) if kb_ctx else ""
        org_name = kb_ctx.split("\n")[0].replace("Knowledge base organisation: ", "") if kb_ctx else "this KB"
        result = chat_json(_PROMPT.format(
            query=query,
            context=context,
            kb_context_block=kb_context_block,
            deferral_block=_build_deferral_block(deferral_context),
            org_name=org_name,
        ))
        recall = max(0.0, min(1.0, float(result.get("recall_score", 0.5))))
        missing = str(result.get("missing_information", "")).strip()
        rel_ids: list[int] = []
        for r in result.get("relevant_ids", []) or []:
            try:
                rel_ids.append(int(r))
            except (ValueError, TypeError):
                continue
        return recall, rel_ids, missing
    except Exception as exc:
        log.warning("CRAG judge failed (%s); defaulting recall=0.5 precision=0.5", exc)
        return 0.5, None, ""


def crag_grader(state: AgentState) -> dict:
    chunks = state.get("outcome_modulated_hits") or state.get("reranked_hits") or []
    sub_queries = state.get("sub_queries") or []
    query = " | ".join(sub_queries) if sub_queries else state["original_query"]

    plan = state.get("plan") or {}
    if plan.get("intent") == "out_of_domain" and chunks and all(
        c.id.startswith("web_") for c in chunks
    ):
        log.info("CRAG → out_of_domain web-only — bypassing KB eval (sufficient)")
        return {
            "context_quality": 1.0,
            "crag_verdict": "sufficient",
            "crag_missing_hint": "",
        }

    if not chunks:
        log.info("CRAG → no context retrieved (combined=0.00)")
        return {
            "context_quality": 0.0,
            "crag_verdict": "defer",
            "crag_missing_hint": "No context retrieved",
        }

    context = _format_context(chunks)
    deferral_context = state.get("deferral_context")
    recall, relevant_ids, missing_hint = _judge(query, context, deferral_context)

    top_n = max(1, config.CRAG_PRECISION_TOP_N)
    n_top = min(top_n, len(chunks))
    if relevant_ids is None:        # judge failed → neutral precision
        precision = 0.5
    else:
        rel_in_top = sum(1 for r in relevant_ids if 1 <= r <= n_top)
        precision = rel_in_top / n_top if n_top else 0.0

    combined = 0.7 * recall + 0.3 * precision
    score = max(recall, combined)

    if score >= config.CRAG_SUFFICIENT_THRESH:
        verdict = "sufficient"
    elif score >= config.CRAG_RETRY_THRESH:
        verdict = "retry"
    else:
        verdict = "defer"

    log.info(
        "CRAG → recall=%.2f  precision=%.2f(top%d)  combined=%.2f  score=%.2f  verdict=%s  missing=%r",
        recall, precision, top_n, combined, score, verdict, missing_hint or "",
    )
    return {
        "context_quality": score,
        "crag_verdict": verdict,
        "crag_missing_hint": missing_hint,
    }
