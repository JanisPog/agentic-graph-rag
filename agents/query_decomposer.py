"""Query decomposer — splits compound queries into independent sub_queries for parallel retrieval."""
from __future__ import annotations
import logging

from agents.state import AgentState
from agents.llm import chat_json

log = logging.getLogger(__name__)

_PROMPT = """\
You are a query analysis engine for a technical support knowledge base.

Decide whether the user query contains two or more INDEPENDENT questions that
would be answered by different parts of the knowledge base.

Rules:
- Split only when the sub-topics are clearly independent (different features,
  settings, or concepts). Do NOT split a single question with multiple clauses
  that all belong to the same topic.
- Each sub-query must be self-contained and answerable on its own.
- Maximum 4 sub-queries.
- If the query is atomic (one topic), return a list with just the original query.

Return a JSON object with one key "sub_queries": a list of strings.
No preamble, no explanation.

User query: {query}"""


def query_decomposer(state: AgentState) -> dict:
    query = state["original_query"]
    try:
        result = chat_json(_PROMPT.format(query=query))
        sub_queries = result.get("sub_queries") if isinstance(result, dict) else None
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [query]
        else:
            sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]
            if not sub_queries:
                sub_queries = [query]
    except Exception as exc:
        log.warning("Query decomposition failed (%s); treating as single query", exc)
        sub_queries = [query]

    if len(sub_queries) > 1:
        log.info("Query decomposed into %d sub-queries:", len(sub_queries))
        for i, q in enumerate(sub_queries):
            log.info("  [%d] %s", i, q)
    else:
        log.info("Query is atomic — no decomposition")

    return {"sub_queries": sub_queries}
