"""Intent classification node — routes query to the right retrieval strategy."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState
from agents.llm import chat_json

log = logging.getLogger(__name__)

_PROMPT = """\
Classify the following support query into exactly one intent category, and extract
any explicit metadata filters present in the query.

Intent categories:
  - factual         : single-step lookup, one clear answer expected
  - multi_hop       : answer requires chaining facts across multiple documents
  - global          : question about trends, summaries, or aggregate patterns
  - metadata_filter : query explicitly scopes by a structured attribute

Known metadata fields you may extract values for:
  {metadata_keys}

Return JSON only:
{{
  "intent": "factual" | "multi_hop" | "global" | "metadata_filter",
  "metadata_filters": {{}}  // empty unless intent is metadata_filter
}}

Examples of metadata_filter queries and their extractions:
  "Show High priority billing tickets" → {{"priority": "High", "category": "Billing"}}
  "All open TechCorp Pro issues"       → {{"status": "Open", "product": "TechCorp Pro"}}

Query: {query}"""


def intent_classifier(state: AgentState) -> dict:
    query = state["original_query"]
    metadata_keys = ", ".join(config.METADATA_FILTER_KEYS)
    try:
        result = chat_json(_PROMPT.format(query=query, metadata_keys=metadata_keys))
        intent = result.get("intent", "factual")
        if intent not in ("factual", "multi_hop", "global", "metadata_filter"):
            intent = "factual"
        metadata_filters = result.get("metadata_filters") or {}
        if not isinstance(metadata_filters, dict):
            metadata_filters = {}
    except Exception as exc:
        log.warning("Intent classification failed (%s); defaulting to 'factual'", exc)
        intent = "factual"
        metadata_filters = {}

    log.info("Intent classified → %s  filters=%s", intent, metadata_filters or "{}")
    return {"intent": intent, "metadata_filters": metadata_filters}
