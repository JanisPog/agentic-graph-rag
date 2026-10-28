"""PIVOT EVOLVE node — records retry strategy and resets retrieval accumulators."""
from __future__ import annotations
import logging

from agents.state import AgentState
from agents.pivot_log import pivot_log

log = logging.getLogger(__name__)


def evolve_node(state: AgentState) -> dict:
    plan      = state.get("plan") or {}
    strategy  = state.get("evolve_strategy") or "semantic_requery"
    retries   = int(state.get("crag_retries", 0))
    tried     = list(state.get("tried_strategies") or [])
    revised   = state.get("crag_missing_hint", "")
    missing   = state.get("evidence_gaps") or []

    if strategy not in tried:
        tried.append(strategy)

    pivot_log(
        "EVOLVE",
        decision=strategy,
        reason=state.get("inspect_reason", "inspect_routed"),
        retry=f"{retries + 1}/{plan.get('confidence_target', 0.0)}",
        revised_query=(revised[:60] + "...") if len(revised) > 60 else revised,
        gaps=len(missing),
    )

    return {
        "tried_strategies": tried,
        "crag_retries":     retries + 1,
        "semantic_hits":    None,
        "bm25_hits":        None,
        "graph_hits":       None,
        "web_hits":         None,
    }
