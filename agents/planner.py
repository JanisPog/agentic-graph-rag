"""Planner node — produces a structured PIVOT plan dict consumed by all downstream nodes."""
from __future__ import annotations
import logging

import config
from agents.state import AgentState
from agents.llm import chat_json
from agents.kb_context import get_kb_context
from agents.pivot_log import pivot_log
from agents.metadata_vocab import (
    STATUS, PRIORITY, LANGUAGE, CHANNEL, DOC_TYPE, get_known_products,
)
from agents.metadata_normalization import normalise_metadata_filters

log = logging.getLogger(__name__)

_PROMPT = """\
PLAN BEFORE ACT.

You are the planning stage of a customer-support search agent.
Before any retrieval tool runs, produce an explicit plan.

Knowledge base context: {kb_context}
Available metadata filter keys: {metadata_keys}

Canonical metadata vocabulary (use these EXACT values when extracting filters):
  status:   {status_vocab}
  priority: {priority_vocab}
  language: {language_vocab}
  channel:  {channel_vocab}
  doc_type: {doc_type_vocab}

Known product names — extract VERBATIM into `product` when the user mentions one
(this list is a hint, not exhaustive — user may name products not listed here):
  {known_products}

Available retrieval tools:
  - semantic_search    (dense vector ANN over chunks)
  - bm25_search        (Lucene fulltext, exact-term match — also covers surface forms)
  - graph_traversal    (PPR over NEXT_CHUNK + RELATED_TO + MENTIONS)
  - surface_form_index (compiled aliases — covered by bm25 fulltext at retrieval time)
  - web_search        (internet search — use ONLY when query is clearly outside the KB
                       domain or explicitly asks for real-time / external information)

User query: {query}
Sub-queries identified by decomposer: {sub_queries}
{memory_block}\
{retry_block}
Produce JSON only, with this exact structure:
{{
  "intent": "factual|multi_hop|global|metadata_filter|out_of_domain",
  "goal_one_sentence": "<restate goal in one sentence>",
  "sub_goals": ["<constraint 1>", "<constraint 2>"],
  "use_semantic": true,
  "use_bm25": true,
  "use_graph": true,
  "use_web_search": false,
  "use_surface_form": false,
  "use_hyde": true,
  "metadata_filters": {{}},
  "skip_reflact": false,
  "react_budget": 1,
  "confidence_target": 0.70,
  "failure_modes": ["<potential issue>"],
  "fallback_strategy": "bm25_retry|graph_hop|web_search|defer|partial_answer",
  "future_tools": []
}}

Rules:
- out_of_domain AND the query is a general knowledge question that can be answered
  via the internet:
    use_semantic=false, use_bm25=false, use_graph=false, use_web_search=true, use_hyde=false,
    skip_reflact=true, react_budget=0
- out_of_domain AND the query is unanswerable / nonsensical / a system command:
    use_semantic=false, use_bm25=false, use_graph=false, use_web_search=false, use_hyde=false,
    skip_reflact=true, react_budget=0
- factual with one sub_goal: skip_reflact=true, react_budget=0, use_graph=false (saves PPR)
- multi_hop OR more than one sub_goal: skip_reflact=false, react_budget=1 or 2, use_graph=true
- If user term may differ from KB term (e.g. "delete" vs "disconnect"): use_surface_form=true
- use_hyde: HyDE generates a hypothetical-answer LLM call per sub-query. It is most
  useful for conceptual / multi-hop queries where the user phrasing is short and
  the KB phrasing is long. Set use_hyde=false for exact-term queries (error codes,
  specific setting names) — BM25 already handles those well and the extra LLM call
  is wasted. Default to true for multi_hop/global; default to false for factual
  queries that lean on exact terms.
- use_web_search=false for all in-domain queries (web comes via EVOLVE fallback if needed)
- DEFERRAL RETRY: if "Previous attempt was deferred" appears above, this is a
  second attempt with user-supplied clarification. You MUST:
    1. Enable ALL retrieval arms (use_semantic=true, use_bm25=true, use_graph=true).
    2. Set react_budget=2 and skip_reflact=false to maximise recall.
    3. Do NOT mark intent as out_of_domain — the user has provided new context.
- Never set react_budget > 2.
- IMPORTANT: sub_goals must match the decomposer output. If the decomposer
  returned a single sub-query, sub_goals MUST be a single entry containing
  that query verbatim. Do NOT invent additional constraints the user did not
  state. Inventing sub-goals causes spurious "missing evidence" downstream.
- DOMAIN ASSUMPTION: unless the user explicitly mentions a different product,
  assume the query is about the KB's documented system. Do NOT add sub_goals
  like "confirm this is about <KB system>" — that context is implicit and
  enforcing it as a constraint causes false coverage gaps.
- METADATA FILTERS: Map user constraints to `metadata_filters` ONLY when the
  user explicitly mentions them. Use the canonical vocabulary above for
  values — do NOT invent your own casing or synonyms. Values can be a single
  string or list of strings.
  - status / priority / channel: extract if mentioned, use canonical casing.
  - language: ONLY if user EXPLICITLY asks for a language (e.g. "German documents").
              Never default to English. Never assume from the query language.
  - doc_type: ONLY if user EXPLICITLY differentiates content types ("tickets",
              "KB articles", "documents"). Do NOT infer from problem keywords
              or questions. If unsure, leave unset.
  - category: ONLY if the user literally says the word "category" (e.g.
              "tickets in category Billing"). Topical phrases like
              "performance issues" are search terms, NOT filters.
  - product: extract if user names a specific product.
  Do NOT guess any filter. When in doubt, leave `metadata_filters` empty —
  semantic and BM25 search will handle topical relevance.
"""


def _fallback_plan(sub_queries: list[str]) -> dict:
    """Fallback plan when the LLM call fails — enables all retrieval arms."""
    return {
        "intent": "factual",
        "goal_one_sentence": sub_queries[0] if sub_queries else "",
        "sub_goals": list(sub_queries),
        "use_semantic": True,
        "use_bm25": True,
        "use_graph": True,
        "use_web_search": False,
        "use_surface_form": False,
        "use_hyde": True,
        "metadata_filters": {},
        "skip_reflact": False,
        "react_budget": 1,
        "confidence_target": config.CRAG_SUFFICIENT_THRESH,
        "failure_modes": ["planner_fallback"],
        "fallback_strategy": "partial_answer",
        "future_tools": [],
    }


def _validate_plan(plan: dict, sub_queries: list[str], original_query: str) -> dict:
    """Normalise and validate the planner output, guaranteeing the downstream contract."""
    valid_intents = ("factual", "multi_hop", "global", "metadata_filter", "out_of_domain")
    if plan.get("intent") not in valid_intents:
        plan["intent"] = "factual"

    _bool_defaults = {
        "use_semantic":     True,
        "use_bm25":         True,
        "use_graph":        True,
        "use_web_search":   False,
        "use_surface_form": False,
        "use_hyde":         True,
        "skip_reflact":     False,
    }
    for k, default in _bool_defaults.items():
        plan[k] = bool(plan.get(k, default))

    try:
        budget = int(plan.get("react_budget", 1))
    except (TypeError, ValueError):
        budget = 1
    plan["react_budget"] = max(0, min(2, budget))

    if plan["react_budget"] == 0:
        plan["skip_reflact"] = True

    try:
        ct = float(plan.get("confidence_target", config.CRAG_SUFFICIENT_THRESH))
    except (TypeError, ValueError):
        ct = config.CRAG_SUFFICIENT_THRESH
    plan["confidence_target"] = max(0.3, min(0.95, ct))

    sgs = plan.get("sub_goals")
    if not isinstance(sgs, list) or not sgs:
        sgs = []
    if len(sub_queries) <= 1:
        plan["sub_goals"] = list(sub_queries) if sub_queries else [""]
    else:
        plan["sub_goals"] = (sgs or list(sub_queries))[: len(sub_queries)]

    raw_filters = plan.get("metadata_filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    plan["metadata_filters"] = normalise_metadata_filters(raw_filters, original_query)

    for k, default in (("failure_modes", []), ("future_tools", []),
                       ("fallback_strategy", "partial_answer"),
                       ("goal_one_sentence", "")):
        plan.setdefault(k, default)

    # out_of_domain: still run KB retrieval arms — a misclassification must not be terminal.
    # Only trim cost-saver LLM calls (HyDE / reflact) that add nothing for off-topic queries.
    if plan["intent"] == "out_of_domain":
        plan["use_semantic"] = True
        plan["use_bm25"] = True
        plan["use_hyde"] = False
        plan["skip_reflact"] = True
        plan["react_budget"] = 0

    if not config.WEB_SEARCH_ENABLED:
        plan["use_web_search"] = False

    return plan


def _build_memory_block(state: AgentState) -> str:
    profile     = state.get("profile_memories")     or []
    interaction = state.get("interaction_memories") or []
    ticket_refs = state.get("ticket_ref_memories")  or []

    if not profile and not interaction and not ticket_refs:
        return ""

    lines = ["\nUser memory context:"]

    if profile:
        profile_texts = [(m.get("memory") or "").strip() for m in profile if (m.get("memory") or "").strip()]
        if profile_texts:
            lines.append("  Profile: " + "; ".join(profile_texts[:3]))

    if interaction:
        interaction_texts = [(m.get("memory") or "").strip() for m in interaction[:3] if (m.get("memory") or "").strip()]
        if interaction_texts:
            lines.append("  Recent interactions: " + "; ".join(interaction_texts))

    if ticket_refs:
        ticket_ids = []
        for m in ticket_refs:
            try:
                tid = str(m.get("metadata", {}).get("ticket_id") or "").strip()
                if tid:
                    ticket_ids.append(tid)
            except Exception:
                pass
        if ticket_ids:
            lines.append(f"  Referenced tickets: {', '.join(ticket_ids)}")

    return "\n".join(lines) + "\n\n"


def _build_retry_block(deferral_context: dict) -> str:
    gaps    = deferral_context.get("gaps") or []
    orig_q  = deferral_context.get("original_query", "")
    clarify = deferral_context.get("user_clarification", "")
    lines = ["Previous attempt was deferred (insufficient context)."]
    if orig_q:
        lines.append(f"  Original query:            {orig_q}")
    if clarify:
        lines.append(f"  User clarification:        {clarify}")
    if gaps:
        lines.append(f"  Information still missing: {'; '.join(gaps)}")
    lines.append("Your plan must use the clarification to find what was missing.")
    return "\n".join(lines) + "\n"


def planner(state: AgentState) -> dict:
    query = state["original_query"]
    sub_queries = state.get("sub_queries") or [query]
    kb_ctx = get_kb_context()
    metadata_keys = ", ".join(config.METADATA_FILTER_KEYS)
    deferral_context = state.get("deferral_context")
    retry_block = _build_retry_block(deferral_context) if deferral_context else ""

    status_vocab   = ", ".join(sorted(STATUS))
    priority_vocab = ", ".join(sorted(PRIORITY))
    language_vocab = ", ".join(sorted(LANGUAGE))
    channel_vocab  = ", ".join(sorted(CHANNEL))
    doc_type_vocab = ", ".join(sorted(DOC_TYPE))
    known_products = ", ".join(get_known_products())
    memory_block   = _build_memory_block(state)

    try:
        raw = chat_json(_PROMPT.format(
            query=query,
            sub_queries=sub_queries,
            kb_context=kb_ctx or "(none)",
            metadata_keys=metadata_keys,
            status_vocab=status_vocab,
            priority_vocab=priority_vocab,
            language_vocab=language_vocab,
            channel_vocab=channel_vocab,
            doc_type_vocab=doc_type_vocab,
            known_products=known_products,
            memory_block=memory_block,
            retry_block=retry_block,
        ))
        plan = _validate_plan(raw if isinstance(raw, dict) else {}, sub_queries, query)
    except Exception as exc:
        log.warning("Planner LLM failed (%s); using fallback plan", exc)
        plan = _fallback_plan(sub_queries)
        plan["metadata_filters"] = normalise_metadata_filters(
            plan["metadata_filters"], query
        )

    arms = [name for name, flag in (
        ("semantic", plan["use_semantic"]),
        ("bm25",     plan["use_bm25"]),
        ("graph",    plan["use_graph"]),
        ("web",      plan["use_web_search"]),
    ) if flag]

    pivot_log(
        "PLAN",
        decision=plan["intent"],
        reason=f"{len(plan['sub_goals'])}_subgoals",
        arms=",".join(arms) or "none",
        budget=plan["react_budget"],
        conf_target=f"{plan['confidence_target']:.2f}",
        skip_reflact_hint=plan["skip_reflact"],
        surface_form=plan["use_surface_form"],
        hyde=plan["use_hyde"],
    )

    return {
        "plan": plan,
        "intent": plan["intent"],
        "metadata_filters": plan["metadata_filters"],
        "sub_goals": list(plan["sub_goals"]),
        "evidence_gaps": list(plan["sub_goals"]),
        "evidence_found": [],
        "tried_strategies": [],
    }