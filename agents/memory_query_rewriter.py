"""Memory query rewriter — enriches sub_queries with profile/interaction/ticket_ref context."""
from __future__ import annotations

import logging
import re

from agents.ablation import enabled
from agents.state import AgentState
from agents.llm import chat

log = logging.getLogger(__name__)

_MAX_ENRICHED_LEN = 500

_SYSTEM_PROMPT = (
    "You are a search query optimizer for a customer support system. "
    "Given a user query and context about the user, reformulate the query "
    "to be more specific and targeted. "
    "Focus on what information is actually needed to solve the problem. "
    "Return only the reformulated query, nothing else. "
    "Keep it concise — one to two sentences maximum."
)

_USER_TEMPLATE = """\
Original query: {query}
{profile_block}\
{interaction_block}\
{ticket_block}\
Reformulate the query considering this context.
Identify what intermediate facts are missing to answer this query."""


def _build_context_blocks(
    profile_memories: list[dict],
    interaction_memories: list[dict],
    ticket_ref_memories: list[dict],
) -> tuple[str, str, str]:
    profile_lines = [
        (m.get("memory") or "").strip()
        for m in profile_memories
        if (m.get("memory") or "").strip()
    ]
    profile_block = (
        "\nUser context:\n" + "\n".join(f"- {l}" for l in profile_lines) + "\n"
        if profile_lines else ""
    )

    interaction_lines = [
        (m.get("memory") or "").strip()
        for m in interaction_memories[:3]
        if (m.get("memory") or "").strip()
    ]
    interaction_block = (
        "\nInteraction history:\n" + "\n".join(f"- {l}" for l in interaction_lines) + "\n"
        if interaction_lines else ""
    )

    ticket_ids = []
    for m in ticket_ref_memories:
        try:
            tid = str(m.get("metadata", {}).get("ticket_id") or "").strip()
            if tid:
                ticket_ids.append(tid)
        except Exception:
            pass
    ticket_block = (
        f"\nReferenced tickets: {', '.join(ticket_ids)}\n"
        if ticket_ids else ""
    )

    return profile_block, interaction_block, ticket_block


def _enrich_one(
    query: str,
    profile_block: str,
    interaction_block: str,
    ticket_block: str,
) -> str:
    """Enrich a single query string. Returns original on any failure."""
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        + _USER_TEMPLATE.format(
            query=query,
            profile_block=profile_block,
            interaction_block=interaction_block,
            ticket_block=ticket_block,
        )
    )
    try:
        raw = chat(prompt).strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if raw and len(raw) <= _MAX_ENRICHED_LEN:
            return raw
        log.info("Query rewriter: response invalid for %r — keeping original", query[:50])
    except Exception as exc:
        log.warning("Query rewriter: LLM failed for %r (%s) — keeping original", query[:50], exc)
    return query


def memory_query_rewriter(state: AgentState) -> dict:
    """Enrich each sub_query with memory context; write back into sub_queries."""
    if not enabled(state, "memory"):
        log.info("Query rewriter: memory ablated — skipping enrichment")
        return {}

    profile_memories:     list[dict] = state.get("profile_memories")     or []
    interaction_memories: list[dict] = state.get("interaction_memories") or []
    ticket_ref_memories:  list[dict] = state.get("ticket_ref_memories")  or []

    if not profile_memories and not interaction_memories and not ticket_ref_memories:
        log.info("Query rewriter: no memories — skipping enrichment")
        return {}

    sub_queries: list[str] = state.get("sub_queries") or [state.get("original_query", "")]
    sub_queries = [q for q in sub_queries if q.strip()]
    if not sub_queries:
        return {}

    profile_block, interaction_block, ticket_block = _build_context_blocks(
        profile_memories, interaction_memories, ticket_ref_memories
    )

    enriched: list[str] = []
    for q in sub_queries:
        eq = _enrich_one(q, profile_block, interaction_block, ticket_block)
        enriched.append(eq)
        log.info("Query rewriter: %r → %r", q[:60], eq[:60])

    log.info("Query rewriter: enriched %d sub_queries", len(enriched))
    return {"sub_queries": enriched}
