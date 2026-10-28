"""ReAct reasoning loop — thinker, actor, observer nodes."""
from __future__ import annotations
import logging
import unicodedata

import config
from agents.state import AgentState, ScoredChunk, ReActStep
from agents.llm import chat_json, embed_query
from agents.retrieval_utils import run_ann, run_bm25
from agents.kb_context import get_kb_context
from graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)

_THOUGHT_PROMPT = """\
You are a reasoning agent solving a support query step by step.

{kb_context_block}\
ORIGINAL GOAL (do not drift from this):
{original_query}

ACCUMULATED SCRATCHPAD:
{scratchpad}

CURRENT CONTEXT PASSAGES:
{context}

AVAILABLE TOOLS: semantic_search | bm25_search | graph_hop | no_action
{pivot_hint}
Think about what key fact is still missing to answer the goal.
Important: if the KB uses a different but equivalent term for what the user asked
(e.g. "disconnect" instead of "delete"), that terminology difference is NOT a missing fact.
Return JSON only:
{{
  "thought": "<reasoning about missing fact>",
  "action": "semantic_search | bm25_search | graph_hop | no_action",
  "sub_query": "<precise sub-query string>",
  "expected_evidence": "<what you expect to find>"
}}"""

_GOAL_CHECK_PROMPT = """\
Given the original query and accumulated evidence below, is there now sufficient
information to write a complete, factual answer?

{kb_context_block}\
Original query: {original_query}

Evidence:
{evidence}

Important: if the KB uses a different but equivalent term for what the user asked
(e.g. "disconnect" instead of "delete", "unlink" instead of "remove"), treat that
as sufficient — vocabulary differences between user and KB are NOT gaps.

Return JSON only: {{"sufficient": true | false}}"""


def _fmt_react_scratchpad(steps: list[ReActStep]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(
        f"Step {i+1}:\n  Thought: {s.thought}\n  Action: {s.action}({s.sub_query})\n  Observation: {s.observation}"
        for i, s in enumerate(steps)
    )


def _fmt_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks[:10]))


def _normalise_query(q: str) -> str:
    return unicodedata.normalize("NFKC", q).lower().strip()


def _semantic_search(sub_query: str, top_k: int = 10) -> list[ScoredChunk]:
    from etl.embedder import embed_texts  # noqa: PLC0415
    vec = embed_texts([sub_query])[0]
    with Neo4jClient() as client:
        return run_ann(client, vec, top_k)


def _bm25_search(sub_query: str, top_k: int = 10) -> list[ScoredChunk]:
    with Neo4jClient() as client:
        return run_bm25(client, sub_query, top_k)


def _graph_hop(seed_ids: list[str], top_k: int = 10) -> list[ScoredChunk]:
    _CYPHER = (
        "MATCH (seed:Chunk) WHERE seed.id IN $seed_ids "
        "OPTIONAL MATCH (seed)-[:NEXT_CHUNK*1..2]-(nbr1:Chunk) "
        "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(nbr2:Chunk) "
        "WITH $seed_ids AS seed_ids, collect(DISTINCT nbr1) + collect(DISTINCT nbr2) AS neighbours "
        "UNWIND neighbours AS c "
        "WITH c, seed_ids WHERE c IS NOT NULL AND NOT c.id IN seed_ids "
        "MATCH (c)-[:PART_OF]->(doc:Document) "
        "RETURN DISTINCT c.id AS id, c.text AS text, "
        "c.chunk_type AS chunk_type, doc.id AS ticket_id, 0.5 AS score "
        "LIMIT $top_k"
    )
    with Neo4jClient() as client:
        rows = client.run(_CYPHER, {"seed_ids": seed_ids, "top_k": top_k})
    return [ScoredChunk(id=r["id"], text=r["text"], score=float(r["score"]),
                        chunk_type=r.get("chunk_type",""), ticket_id=r.get("ticket_id",""))
            for r in rows]


def react_thinker(state: AgentState) -> dict:
    pivot = state.get("pivot_required", False)
    pivot_hint = (
        "\n⚠️  PIVOT REQUIRED: use a different tool and different query than the last step.\n"
        if pivot else ""
    )
    context = _fmt_chunks(state.get("reranked_hits") or [])
    scratchpad = _fmt_react_scratchpad(state.get("react_scratchpad") or [])

    kb_ctx = get_kb_context()
    kb_context_block = f"Knowledge base context: {kb_ctx}\n\n" if kb_ctx else ""

    try:
        result = chat_json(_THOUGHT_PROMPT.format(
            original_query=state["original_query"],
            scratchpad=scratchpad,
            context=context,
            pivot_hint=pivot_hint,
            kb_context_block=kb_context_block,
        ), temperature=config.TEMPERATURE_REASON)
        thought = result.get("thought", "")
        action = result.get("action", "no_action")
        sub_query = result.get("sub_query") or state["original_query"]
    except Exception as exc:
        log.warning("react_thinker failed (%s); choosing no_action", exc)
        thought = f"Error during thinking: {exc}"
        action = "no_action"
        sub_query = state["original_query"]

    iteration = state.get("react_iterations", 0) + 1
    log.info("ReAct [iter %d] thought: %s", iteration, thought)
    log.info("ReAct [iter %d] action:  %s(%r)", iteration, action, sub_query)

    step = ReActStep(thought=thought, action=action, sub_query=sub_query)
    scratchpad_list = list(state.get("react_scratchpad") or [])
    scratchpad_list.append(step)

    return {
        "react_scratchpad": scratchpad_list,
        "react_iterations": iteration,
    }


def react_actor(state: AgentState) -> dict:
    is_reflact = getattr(config, "USE_REFLACT", False)
    scratchpad_key = "reflact_scratchpad" if is_reflact else "react_scratchpad"
    scratchpad = list(state.get(scratchpad_key) or [])
    if not scratchpad:
        return {}

    step = scratchpad[-1]
    action = step.action
    sub_query = step.sub_query
    reranked = state.get("reranked_hits") or []

    try:
        if action == "semantic_search":
            new_hits = _semantic_search(sub_query)
        elif action == "bm25_search":
            new_hits = _bm25_search(sub_query)
        elif action == "graph_hop":
            seeds = [c.id for c in reranked[:5]]
            new_hits = _graph_hop(seeds)
        else:
            new_hits = []
        observation = f"Retrieved {len(new_hits)} chunk(s)."
    except Exception as exc:
        log.warning("react_actor tool %r failed: %s", action, exc)
        new_hits = []
        observation = f"Tool failed: {exc}"

    existing = {c.id: c for c in reranked}
    for c in new_hits:
        if c.id not in existing or c.score > existing[c.id].score:
            existing[c.id] = c

    step.observation = observation
    scratchpad[-1] = step

    return {
        scratchpad_key: scratchpad,
        "reranked_hits": list(existing.values()),
    }


def react_observer(state: AgentState) -> dict:
    is_reflact = getattr(config, "USE_REFLACT", False)
    scratchpad_key = "reflact_scratchpad" if is_reflact else "react_scratchpad"
    scratchpad = list(state.get(scratchpad_key) or [])
    prev_queries = list(state.get("prev_sub_queries") or [])

    current_step = scratchpad[-1] if scratchpad else None
    pivot_required = False
    if current_step:
        norm_q = _normalise_query(current_step.sub_query)
        if norm_q in [_normalise_query(q) for q in prev_queries]:
            pivot_required = True
            log.info("Topological pivot triggered for query: %r", current_step.sub_query)
        prev_queries.append(current_step.sub_query)

    iter_n = state.get("react_iterations", 0)
    is_last_iter = iter_n >= config.REACT_MAX_ITERATIONS
    goal_achieved = state.get("goal_achieved", False)

    if not is_reflact and is_last_iter:
        evidence = _fmt_chunks(state.get("reranked_hits") or [])
        kb_ctx = get_kb_context()
        kb_context_block = f"Knowledge base context: {kb_ctx}\n\n" if kb_ctx else ""
        try:
            result = chat_json(_GOAL_CHECK_PROMPT.format(
                original_query=state["original_query"],
                evidence=evidence,
                kb_context_block=kb_context_block,
            ), temperature=config.TEMPERATURE_REASON)
            goal_achieved = bool(result.get("sufficient", False))
        except Exception as exc:
            log.warning("Goal check failed (%s)", exc)
            
    mode_name = "ReflAct" if is_reflact else "ReAct"
    log.info("%s [iter %d] observe: goal_achieved=%s  pivot=%s  context_size=%d",
             mode_name, iter_n, goal_achieved, pivot_required,
             len(state.get("reranked_hits") or []))
    return {
        "goal_achieved": goal_achieved,
        "pivot_required": pivot_required,
        "prev_sub_queries": prev_queries,
    }
