"""Query expansion node — paraphrase + HyDE per sub-query."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState
from agents.llm import chat, chat_json
from agents.kb_context import get_kb_context

log = logging.getLogger(__name__)

_PROMPT_EXPAND = """\
You are a query expansion engine for a technical support knowledge base.
Given the user query below, generate exactly 3 alternative phrasings that:
  - Cover different vocabulary (synonyms, error codes, formal/informal)
  - Vary specificity (broad / narrow)
  - Do NOT add new factual assumptions

Return a JSON list of 3 strings. No preamble.

User query: {query}"""

_PROMPT_HYDE = """\
You are a technical support assistant. Write a concise, plausible answer (2-4
sentences) to the question below, as if it appeared in a knowledge base article.
Do not hedge or say you don't know — generate a realistic resolution.
{kb_context_block}Reply with the answer text only. No preamble, no JSON, no labels.

User query: {query}"""


def _expand_one(query: str) -> list[str]:
    try:
        result = chat_json(_PROMPT_EXPAND.format(query=query), temperature=config.TEMPERATURE_EXPAND)
        if isinstance(result, list) and result:
            return [str(x) for x in result[:3]]
    except Exception as exc:
        log.warning("Paraphrase expansion failed for %r: %s", query, exc)
    return []


def _hyde(query: str) -> str | None:
    try:
        kb_ctx = get_kb_context()
        kb_context_block = (
            f"{kb_ctx}\nIf not stated otherwise, assume the query concerns this "
            f"product/service.\n" if kb_ctx else ""
        )
        result = chat(
            _PROMPT_HYDE.format(query=query, kb_context_block=kb_context_block),
            temperature=config.TEMPERATURE_EXPAND,
        ).strip()
        import re
        result = re.sub(r"<think>[\s\S]*?</think>", "", result).strip()
        return result or None
    except Exception as exc:
        log.warning("HyDE generation failed for %r: %s", query, exc)
    return None


def query_expander(state: AgentState) -> dict:
    sub_queries = state.get("sub_queries") or [state["original_query"]]
    plan = state.get("plan") or {}
    use_hyde = bool(plan.get("use_hyde", True))

    expanded: list[str] = []
    seen: set[str] = set()
    hyde_texts: list[str] = []

    def _add(q: str) -> None:
        if q and q not in seen:
            seen.add(q)
            expanded.append(q)

    for sq in sub_queries:
        _add(sq)
        for q in _expand_one(sq):
            _add(q)
        if use_hyde:
            if hy := _hyde(sq):
                hyde_texts.append(hy)
                _add(hy)

    hyde_query = hyde_texts[0] if hyde_texts else ""

    log.info("Query expansion → %d queries across %d sub-queries (paraphrase%s):",
             len(expanded), len(sub_queries),
             " + HyDE" if use_hyde else "; HyDE skipped per plan")
    for i, q in enumerate(expanded):
        log.info("  [%d] %s", i, q)
    if hyde_query:
        log.info("  HyDE answer: %s", hyde_query[:120])

    return {"expanded_queries": expanded, "hyde_query": hyde_query}
