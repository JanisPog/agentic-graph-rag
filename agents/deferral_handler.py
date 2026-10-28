"""Deferral handler — emits a structured deferral notice when retrieval fails."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState

log = logging.getLogger(__name__)


def deferral_handler(state: AgentState) -> dict:
    missing          = state.get("crag_missing_hint", "")
    quality          = state.get("context_quality", 0.0)
    query            = state["original_query"]
    plan             = state.get("plan") or {}
    intent           = plan.get("intent", "")
    gaps             = state.get("evidence_gaps") or []
    found            = state.get("evidence_found") or []
    retries          = state.get("crag_retries", 0)
    is_retry         = bool(state.get("deferral_context"))

    gap_text = ", ".join(gaps[:5]) if gaps else (missing or "unknown")

    if intent == "out_of_domain":
        notice = (
            "DEFERRAL (out_of_domain): The query is outside the scope of this "
            "knowledge base.\n\n"
            f"Query: {query}\n\n"
            "Suggested clarification for the user: ask whether they meant to "
            "consult a different system, or refine the question to topics "
            "covered by this knowledge base."
        )
        log.warning("Deferring query: %r (intent=out_of_domain)", query)
        return {
            "final_answer":     notice,
            "cited_chunk_ids":  [],
            "confidence":       0.0,
            "deferral_reason": {
                "type":                    "out_of_domain",
                "original_query":          query,
                "gaps":                    [],
                "covered":                 [],
                "suggested_clarification": "Is your question about this system, or did you mean another product?",
            },
        }

    if is_retry:
        notice = (
            f"DEFERRAL (irrecoverable): Even with the additional context provided, "
            f"the system could not find sufficient information after {retries} "
            f"retrieval attempt(s).\n\n"
            f"Query: {query}\n"
            f"Context quality score: {quality:.2f}\n"
            f"Resolved so far: {', '.join(found) if found else '(none)'}\n"
            f"Still missing: {gap_text}\n\n"
            "This topic does not appear to be covered in the knowledge base. "
            "Please contact a human support agent for further assistance."
        )
        log.warning(
            "Deferring query (retry exhausted): %r (quality=%.2f) gaps=%s",
            query, quality, gaps or [missing],
        )
        return {
            "final_answer":    notice,
            "cited_chunk_ids": [],
            "confidence":      0.0,
            "deferral_reason": {
                "type":                    "irrecoverable_gap_after_retry",
                "original_query":          query,
                "gaps":                    gaps or ([missing] if missing else []),
                "covered":                 found,
                "suggested_clarification": None,
            },
        }

    notice = (
        f"DEFERRAL: The system could not find sufficient context after "
        f"{retries} retrieval attempt(s).\n\n"
        f"Query: {query}\n"
        f"Context quality score: {quality:.2f}\n"
        f"Resolved so far: {', '.join(found) if found else '(none)'}\n"
        f"Missing information: {gap_text}\n\n"
        "Please contact a human support agent for further assistance."
    )
    log.warning(
        "Deferring query: %r (quality=%.2f) gaps=%s",
        query, quality, gaps or [missing],
    )
    return {
        "final_answer":    notice,
        "cited_chunk_ids": [],
        "confidence":      0.0,
        "deferral_reason": {
            "type":                    "irrecoverable_gap",
            "original_query":          query,
            "gaps":                    gaps or ([missing] if missing else []),
            "covered":                 found,
            "suggested_clarification": (
                f"To complete this answer, the following is needed: {gap_text}."
            ),
        },
    }
