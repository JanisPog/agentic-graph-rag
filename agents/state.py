"""Shared state schema for the LangGraph agent pipeline."""
from __future__ import annotations
import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict


@dataclass
class ScoredChunk:
    id: str
    text: str
    score: float
    chunk_type: str = ""
    ticket_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ReActStep:
    thought: str
    action: str
    sub_query: str
    observation: str = ""


@dataclass
class ReflActStep:
    reflection: str
    action: str
    sub_query: str
    expected_evidence: str = ""
    observation: str = ""

def _last(a, b):
    """Keep the most recent write — required reducer for parallel fan-in keys."""
    return b


def _add_or_reset(a, b):
    """Concat lists; `None` resets to [] so EVOLVE retries start with a clean slate."""
    if a is None:
        a = []
    if b is None:
        return []
    return list(a) + list(b)


class AgentState(TypedDict, total=False):
    # Input
    original_query: str

    # Maps signal key → bool; unset keys fall back to config defaults. See agents/ablation.py.
    ablation: dict

    # Memory / user context — populated by executor, consumed by memory nodes
    user_id:         str | None          # identifies the requesting user
    ticket_id:       str | None          # active support ticket for this session
    user_memories:   list[dict]          # raw mem0 results (category, memory, score, …)
    similar_tickets: Annotated[list[dict], _last]  # output of ticket_retrieval_node

    # Categorised memory views — written once by memory_context_builder
    profile_memories:     Annotated[list[dict], _last]
    preference_memories:  Annotated[list[dict], _last]
    interaction_memories: Annotated[list[dict], _last]
    ticket_ref_memories:  Annotated[list[dict], _last]

    # Ticket Reuse Gate output
    reuse_answer:     Annotated[str | None, _last]
    reuse_ticket_id:  Annotated[str | None, _last]
    reuse_candidates: Annotated[list[dict], _last]
    ki_fast_path:     Annotated[bool, _last]

    # Failed Attempt Extractor output
    per_user_failed: Annotated[list[str], _last]
    global_failed:   Annotated[list[str], _last]
    resolved:        Annotated[list[str], _last]

    # Query understanding
    sub_queries: list[str]           # set by query_decomposer
    expanded_queries: list[str]
    hyde_query: str
    # 'factual' | 'multi_hop' | 'global' | 'metadata_filter' | 'out_of_domain'
    intent: Annotated[str, _last]
    # PIVOT plan dict produced by the planner. See agents/planner.py for schema.
    plan: Annotated[dict, _last]
    metadata_filters: dict[str, str]

    # PIVOT state (planner / inspect / evolve / verify)
    sub_goals:           Annotated[list[str], _last]
    evidence_gaps:       Annotated[list[str], _last]
    evidence_found:      Annotated[list[str], _last]
    inspect_verdict:     Annotated[str, _last]
    inspect_reason:      Annotated[str, _last]
    evolve_strategy:     Annotated[str, _last]
    tried_strategies:    Annotated[list[str], _last]
    verify_result:       Annotated[dict, _last]
    partial_answer_note: Annotated[str, _last]
    deferral_reason:     Annotated[dict, _last]
    deferral_context:    Annotated[dict, _last]

    # Retrieval results — None write triggers _add_or_reset reset branch (EVOLVE retry).
    semantic_hits: Annotated[list[ScoredChunk], _add_or_reset]
    bm25_hits:     Annotated[list[ScoredChunk], _add_or_reset]
    graph_hits:    Annotated[list[ScoredChunk], _add_or_reset]
    web_hits:      Annotated[list[ScoredChunk], _add_or_reset]

    graph_seeds_total:    Annotated[int, _last]
    graph_entity_seeds:   Annotated[int, _last]
    graph_nodes_returned: Annotated[int, _last]
    graph_entity_hits:    Annotated[int, _last]

    fused_hits:    Annotated[list[ScoredChunk], _last]
    reranked_hits: Annotated[list[ScoredChunk], _last]

    outcome_modulated_hits: Annotated[list[ScoredChunk], _last]
    reranker_score_cache: Annotated[dict, _last]

    react_iterations: Annotated[int, _last]
    react_scratchpad: Annotated[list[ReActStep], _last]
    goal_achieved:    Annotated[bool, _last]
    pivot_required:   Annotated[bool, _last]
    prev_sub_queries: Annotated[list[str], _last]

    reflact_scratchpad: Annotated[list[ReflActStep], _last]
    reflection_summary: Annotated[str, _last]
    _last_new_hits:     Annotated[list[ScoredChunk], _last]

    fasttrack_hit:       Annotated[dict | None, _last]
    fasttrack_score:     Annotated[float, _last]
    fasttrack_hard_skip: Annotated[bool, _last]

    context_quality:  Annotated[float, _last]
    crag_verdict:     Annotated[str, _last]
    crag_retries:     Annotated[int, _last]
    crag_missing_hint: Annotated[str, _last]

    final_answer:    Annotated[str, _last]
    cited_chunk_ids: Annotated[list[str], _last]
    confidence:      Annotated[float, _last]
