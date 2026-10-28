"""LangGraph StateGraph — assembles all agent nodes and compiles the pipeline.

Two pipelines are supported:

  PIVOT_ENABLED = True  (default)
    query_decomposer → ticket_fasttrack
        ├─ hard skip (crag>=0.85) → verify_node → generator
        └─ continue → ticket_reuse_gate → memory_query_rewriter → planner → query_expander
        → [plan-conditioned fan-out: semantic / bm25 / graph / web / ticket]
        → rrf_fusion → reranker → crag_grader → inspect_node
            ├─ skip_evolve → verify_node
            ├─ reflact / react → loop → verify_node
            ├─ evolve → evolve_node → (semantic_retriever | bm25_retriever | graph_retriever)
            └─ defer → deferral_handler
        verify_node → (generator | deferral_handler)

  PIVOT_ENABLED = False  (legacy)
    Original intent_classifier-driven pipeline. Kept for A/B comparison.
"""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.query_decomposer import query_decomposer
from agents.query_expander import query_expander
from agents.intent_classifier import intent_classifier
from agents.planner import planner
from agents.semantic_retriever import semantic_retriever
from agents.bm25_retriever import bm25_retriever
from agents.graph_retriever import graph_retriever
from agents.rrf_fusion import rrf_fusion
from agents.reranker import reranker
from agents.crag_grader import crag_grader
from agents.inspect_node import inspect_node
from agents.evolve_node import evolve_node
from agents.verify_node import verify_node
from agents.generator import generator
from agents.deferral_handler import deferral_handler
from agents.web_search_retriever import web_search_retriever
from agents.ticket_retriever import ticket_retrieval_node
from agents.memory_context_builder import memory_context_builder
from agents.ticket_reuse_gate import ticket_reuse_gate
from agents.memory_query_rewriter import memory_query_rewriter
from agents.failed_attempt_extractor import failed_attempt_extractor
from agents.outcome_score_modulator import outcome_score_modulator
from agents.ticket_fasttrack import ticket_fasttrack

import config

from agents.react_loop import react_thinker, react_actor, react_observer

if config.USE_REFLACT:
    from agents.reflact_loop import reflact_thinker

log = logging.getLogger(__name__)


# ── PIVOT routing functions ───────────────────────────────────────────────────

def _after_fasttrack(state: AgentState) -> str:
    """Route after ticket_fasttrack.

    Always continue to ticket_reuse_gate, then the full retrieval pipeline.
    The fast-track HARD-SKIP path (bypass straight to verify_node) was removed:
    it served topically-near-but-wrong ticket answers the CRAG judge could not
    catch (WixQA-synthetic: hard-skip Factuality ~0.48 vs ~0.87 for full
    generation). A strong fast-track match now only SEEDS rrf_fusion via
    `fasttrack_hit`; it never bypasses the generator. See agents/ticket_fasttrack.py.
    """
    return "ticket_reuse_gate"


def _after_expander(state: AgentState) -> list[str]:
    """Plan-conditioned fan-out (fires after query_expander).

    The pipeline order is now: decomposer → planner → expander → retrievers,
    so the expander can read `plan.use_hyde` and skip the HyDE LLM call when
    the planner judged it unnecessary.

    Activates only the retrieval arms the plan requested. All arms are fanned
    out in parallel from here — graph_retriever no longer waits for semantic.
    This is critical: when graph_retriever fired AFTER semantic via a separate
    edge, rrf_fusion was scheduled twice (once when bm25 finished, again when
    graph finished), causing the entire downstream chain (rerank → CRAG →
    INSPECT → reflact → VERIFY → generator) to run twice. Fanning all retrievers
    in parallel from one source makes rrf_fusion fire exactly once.

    web_search_retriever is activated either:
      - by the planner for out_of_domain queries (Gate 1), or
      - by EVOLVE for kb_insufficient gaps (Gate 2, via _after_evolve).
    """
    plan = state.get("plan") or {}
    # NOTE: out_of_domain no longer short-circuits to deferral here. The planner's
    # domain guess is ungrounded (it never sees the corpus), so we let out_of_domain
    # queries run the KB retrieval arms and defer ONLY if CRAG finds nothing — a
    # misclassification can no longer zero out recall. (Planner keeps use_semantic/
    # use_bm25 on for out_of_domain; see agents/planner.py.)
    arms = []
    if plan.get("use_semantic", True):
        arms.append("semantic_retriever")
    if plan.get("use_bm25", True):
        arms.append("bm25_retriever")
    if plan.get("use_graph", False) or plan.get("use_surface_form", False):
        arms.append("graph_retriever")
    if plan.get("use_web_search", False):
        arms.append("web_search_retriever")
    if plan.get("use_ticket_retriever", True):
        arms.append("ticket_retrieval_node")
    if not arms:
        arms = ["semantic_retriever"]
    return arms


def _after_reuse_gate(state: AgentState) -> str:
    """KI match found → inspect_node (CRAG already scored in gate). No match → normal pipeline."""
    if state.get("ki_fast_path"):
        return "inspect_node"
    return "memory_query_rewriter"


def _after_inspect(state: AgentState) -> str:
    verdict = state.get("inspect_verdict", "skip_evolve")
    if verdict == "skip_evolve":
        return "verify_node"
    if verdict == "reflact":
        return "reflact_thinker"
    if verdict == "react":
        return "react_thinker"
    if verdict == "evolve":
        return "evolve_node"
    # defer: if we came via KI fast-path and haven't tried full retrieval yet,
    # fall back to memory_query_rewriter (decomposer + planner already ran).
    if state.get("ki_fast_path") and not state.get("tried_strategies"):
        log.info("inspect_node: KI resolution insufficient — falling back to full retrieval")
        return "memory_query_rewriter"
    return "deferral_handler"


def _after_evolve(state: AgentState) -> str:
    """Map EVOLVE strategy → specific retriever arm. The retriever then flows
    forward through rrf_fusion → reranker → crag_grader → inspect_node again."""
    strategy = state.get("evolve_strategy", "semantic_requery")
    return {
        "bm25_retry":       "bm25_retriever",
        "graph_hop":        "graph_retriever",
        "surface_form":     "bm25_retriever",  # surface forms live in fulltext index
        "semantic_requery": "semantic_retriever",
        "web_search":       "web_search_retriever",
        "defer":            "deferral_handler",
    }.get(strategy, "semantic_retriever")


def _after_observer_pivot(state: AgentState) -> str:
    """ReAct/ReflAct observer → verify_node (VERIFY is the single goal authority).

    The legacy goal_achieved short-circuit is preserved as a soft hint —
    if the loop says we're done, we still pass through VERIFY to confirm.
    """
    if state.get("goal_achieved"):
        return "verify_node"
    if state.get("react_iterations", 0) >= config.REACT_MAX_ITERATIONS:
        return "verify_node"
    return "reflact_thinker" if config.USE_REFLACT else "react_thinker"


def _after_verify(state: AgentState) -> str:
    verdict = state.get("inspect_verdict", "generate")
    if verdict == "defer":
        return "deferral_handler"
    return "generator"


# ── Legacy routing functions ──────────────────────────────────────────────────

def _after_crag_legacy(state: AgentState) -> str:
    verdict = state.get("crag_verdict", "sufficient")
    if verdict == "sufficient":
        return "reflact_thinker" if config.USE_REFLACT else "react_thinker"
    elif verdict == "retry":
        return "semantic_retriever"
    else:
        return "deferral_handler"


def _after_observer_legacy(state: AgentState) -> str:
    if state.get("goal_achieved"):
        return "generator"
    if state.get("react_iterations", 0) >= config.REACT_MAX_ITERATIONS:
        return "generator"
    return "reflact_thinker" if config.USE_REFLACT else "react_thinker"


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_pivot_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("memory_context_builder",  memory_context_builder)
    g.add_node("ticket_reuse_gate",       ticket_reuse_gate)
    g.add_node("query_decomposer",        query_decomposer)
    g.add_node("ticket_fasttrack",   ticket_fasttrack)
    g.add_node("planner",                 planner)
    g.add_node("memory_query_rewriter",   memory_query_rewriter)
    g.add_node("query_expander",          query_expander)

    g.add_node("semantic_retriever", semantic_retriever)
    g.add_node("bm25_retriever",     bm25_retriever)
    g.add_node("graph_retriever",    graph_retriever)
    g.add_node("web_search_retriever", web_search_retriever)
    g.add_node("ticket_retrieval_node", ticket_retrieval_node)
    g.add_node("rrf_fusion",               rrf_fusion)
    g.add_node("failed_attempt_extractor",  failed_attempt_extractor)
    g.add_node("reranker",                  reranker)
    g.add_node("outcome_score_modulator",   outcome_score_modulator)
    g.add_node("crag_grader",        crag_grader)
    g.add_node("inspect_node",       inspect_node)
    g.add_node("evolve_node",        evolve_node)
    g.add_node("verify_node",        verify_node)
    g.add_node("react_thinker",      react_thinker)
    g.add_node("react_actor",        react_actor)
    g.add_node("react_observer",     react_observer)
    g.add_node("generator",          generator)
    g.add_node("deferral_handler", deferral_handler)
    if config.USE_REFLACT:
        g.add_node("reflact_thinker", reflact_thinker)


    # Memory categorisation → decomposer → planner (memory-aware orchestrator).
    # Planner decides use_ki_gate; if true → ticket_reuse_gate before retrieval.
    # KI match → crag_grader; no match or use_ki_gate=False → memory_query_rewriter.
    g.set_entry_point("memory_context_builder")
    g.add_edge("memory_context_builder", "query_decomposer")
    g.add_edge("query_decomposer", "ticket_fasttrack")
    g.add_conditional_edges("ticket_fasttrack", _after_fasttrack, {
        "verify_node": "verify_node",
        "ticket_reuse_gate":     "ticket_reuse_gate",
    })
    g.add_conditional_edges("ticket_reuse_gate", _after_reuse_gate, {
        "inspect_node":          "inspect_node",
        "memory_query_rewriter": "memory_query_rewriter",
    })
    g.add_edge("memory_query_rewriter", "planner")
    g.add_edge("planner",                "query_expander")


    # Plan-conditioned fan-out — all retrievers in parallel from one source.
    # rrf_fusion has exactly N static incoming edges (one per retriever) and
    # fires once when all activated arms have completed. Out-of-domain queries
    # short-circuit to deferral_handler with zero retrieval calls.
    g.add_conditional_edges("query_expander", _after_expander, {
        "semantic_retriever":    "semantic_retriever",
        "bm25_retriever":        "bm25_retriever",
        "graph_retriever":       "graph_retriever",
        "web_search_retriever":  "web_search_retriever",
        "ticket_retrieval_node": "ticket_retrieval_node",
        "deferral_handler":    "deferral_handler",
    })

    g.add_edge("semantic_retriever",     "rrf_fusion")
    g.add_edge("bm25_retriever",         "rrf_fusion")
    g.add_edge("graph_retriever",        "rrf_fusion")
    g.add_edge("web_search_retriever",   "rrf_fusion")
    g.add_edge("ticket_retrieval_node",  "rrf_fusion")
    g.add_edge("rrf_fusion",               "failed_attempt_extractor")
    g.add_edge("failed_attempt_extractor",  "reranker")
    g.add_edge("reranker",                  "outcome_score_modulator")
    g.add_edge("outcome_score_modulator",   "crag_grader")
    g.add_edge("crag_grader",        "inspect_node")

    # INSPECT routing (memory_query_rewriter = KI-fast-path fallback; planner already ran)
    inspect_targets = {
        "verify_node":           "verify_node",
        "react_thinker":         "react_thinker",
        "evolve_node":           "evolve_node",
        "deferral_handler":    "deferral_handler",
        "memory_query_rewriter": "memory_query_rewriter",
    }
    if config.USE_REFLACT:
        inspect_targets["reflact_thinker"] = "reflact_thinker"
    g.add_conditional_edges("inspect_node", _after_inspect, inspect_targets)

    # EVOLVE → retriever (and back through rrf_fusion → ... → inspect_node)
    g.add_conditional_edges("evolve_node", _after_evolve, {
        "semantic_retriever":    "semantic_retriever",
        "bm25_retriever":        "bm25_retriever",
        "graph_retriever":       "graph_retriever",
        "web_search_retriever":  "web_search_retriever",
        "deferral_handler":    "deferral_handler",
    })

    # Reasoning loop (ReAct or ReflAct) → react_observer → verify_node
    if config.USE_REFLACT:
        g.add_edge("reflact_thinker", "react_actor")
    g.add_edge("react_thinker", "react_actor")
    g.add_edge("react_actor",   "react_observer")
    observer_targets = {
        "verify_node":     "verify_node",
        "react_thinker":   "react_thinker",
    }
    if config.USE_REFLACT:
        observer_targets["reflact_thinker"] = "reflact_thinker"
    g.add_conditional_edges("react_observer", _after_observer_pivot, observer_targets)

    # VERIFY → generator | deferral_handler
    g.add_conditional_edges("verify_node", _after_verify, {
        "generator":          "generator",
        "deferral_handler": "deferral_handler",
    })

    # Terminals
    g.add_edge("generator",          END)
    g.add_edge("deferral_handler", END)
    return g


def _build_legacy_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("memory_context_builder",  memory_context_builder)
    g.add_node("ticket_reuse_gate",       ticket_reuse_gate)
    g.add_node("query_decomposer",        query_decomposer)
    g.add_node("memory_query_rewriter",   memory_query_rewriter)
    g.add_node("query_expander",          query_expander)
    g.add_node("intent_classifier",       intent_classifier)
    g.add_node("semantic_retriever", semantic_retriever)
    g.add_node("bm25_retriever",     bm25_retriever)
    g.add_node("graph_retriever",    graph_retriever)
    g.add_node("rrf_fusion",         rrf_fusion)
    g.add_node("reranker",           reranker)
    g.add_node("crag_grader",        crag_grader)
    g.add_node("react_thinker",      react_thinker)
    g.add_node("react_actor",        react_actor)
    g.add_node("react_observer",     react_observer)
    g.add_node("generator",          generator)
    g.add_node("deferral_handler", deferral_handler)

    g.set_entry_point("memory_context_builder")
    g.add_edge("memory_context_builder",  "query_decomposer")
    g.add_edge("query_decomposer",        "memory_query_rewriter")
    g.add_edge("memory_query_rewriter",   "query_expander")
    g.add_edge("query_expander",          "intent_classifier")

    g.add_edge("intent_classifier",  "semantic_retriever")
    g.add_edge("intent_classifier",  "bm25_retriever")
    g.add_edge("semantic_retriever", "graph_retriever")
    g.add_edge("bm25_retriever",     "rrf_fusion")
    g.add_edge("graph_retriever",    "rrf_fusion")
    g.add_edge("rrf_fusion",         "reranker")
    g.add_edge("reranker",           "crag_grader")

    g.add_conditional_edges("crag_grader", _after_crag_legacy, {
        "react_thinker":      "react_thinker",
        "reflact_thinker":    "reflact_thinker",
        "semantic_retriever": "semantic_retriever",
        "deferral_handler": "deferral_handler",
    })

    if config.USE_REFLACT:
        g.add_node("reflact_thinker", reflact_thinker)
        g.add_edge("reflact_thinker", "react_actor")
    g.add_edge("react_thinker", "react_actor")
    g.add_edge("react_actor",   "react_observer")
    observer_targets = {"generator": "generator"}
    if config.USE_REFLACT:
        observer_targets["reflact_thinker"] = "reflact_thinker"
    else:
        observer_targets["react_thinker"] = "react_thinker"
    g.add_conditional_edges("react_observer", _after_observer_legacy, observer_targets)

    g.add_edge("generator",          END)
    g.add_edge("deferral_handler", END)
    return g


def build_graph():
    g = _build_pivot_graph() if config.PIVOT_ENABLED else _build_legacy_graph()
    return g.compile()


# Compiled graph — importable by the CLI
graph = build_graph()
