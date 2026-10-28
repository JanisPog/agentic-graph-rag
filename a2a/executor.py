"""A2A AgentExecutor — bridges incoming A2A requests to the LangGraph pipeline."""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make Prototype_v1.0 root importable when running from inside a2a/
_PROTO_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from graph_builder import graph

logger = logging.getLogger(__name__)

# Node names that trigger a status broadcast to listening agents
_RETRIEVAL_DONE_NODE  = "crag_grader"       # all three retrievers + fusion + reranker done
_REACT_START_NODE     = "react_thinker"     # first entry into the ReAct loop
_REACT_ITER_NODE      = "react_observer"    # each completed ReAct cycle
_FINAL_NODES          = {"generator", "deferral_handler"}


def _previous_deferral(context: RequestContext) -> dict | None:
    """Return deferral_reason from the last artifact if the prior run deferred, else None."""
    task = context.current_task
    if not task or not task.artifacts:
        return None
    last_artifact = task.artifacts[-1]
    for part in last_artifact.parts:
        if part.HasField("text"):
            try:
                payload = json.loads(part.text)
                if payload.get("status") in ("deferred", "deferred_final"):
                    return payload.get("deferral_reason")
            except (json.JSONDecodeError, AttributeError):
                pass
    return None


def _make_initial_state(
    query: str,
    user_id: str | None = None,
    ticket_id: str | None = None,
    user_memories: list | None = None,
    deferral_context: dict | None = None,
) -> dict:
    state: dict = {
        "original_query":       query,
        "user_id":              user_id,
        "ticket_id":            ticket_id,
        "user_memories":        user_memories or [],
        "similar_tickets":      [],
        "profile_memories":     [],
        "preference_memories":  [],
        "interaction_memories": [],
        "ticket_ref_memories":  [],
        "reuse_answer":         None,
        "reuse_ticket_id":      None,
        "reuse_candidates":     [],
        "per_user_failed":      [],
        "global_failed":        [],
        "resolved":             [],
        "expanded_queries":     [],
        "intent":               "factual",
        "semantic_hits":        [],
        "bm25_hits":            [],
        "graph_hits":           [],
        "fused_hits":           [],
        "reranked_hits":        [],
        "react_iterations":     0,
        "react_scratchpad":     [],
        "goal_achieved":        False,
        "pivot_required":       False,
        "prev_sub_queries":     [],
        "context_quality":      0.0,
        "crag_verdict":         "",
        "crag_retries":         0,
        "crag_missing_hint":    "",
        "final_answer":         "",
        "cited_chunk_ids":      [],
        "confidence":           0.0,
    }
    if deferral_context:
        state["deferral_context"]  = deferral_context
        state["evidence_gaps"]     = deferral_context.get("gaps") or []
        state["crag_retries"]      = 1
        gaps = deferral_context.get("gaps") or []
        if gaps:
            state["crag_missing_hint"] = "; ".join(gaps[:3])
    return state


def _serialise(obj):
    """JSON fallback for LangGraph dataclass nodes."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return str(obj)


async def _status(updater: TaskUpdater, text: str) -> None:
    """Send a WORKING status update with a human-readable message."""
    await updater.update_status(
        TaskState.TASK_STATE_WORKING,
        message=updater.new_agent_message([Part(text=text)]),
    )


def _stream_graph(initial_state: dict) -> dict:
    final_state: dict = {}
    for chunk in graph.stream(initial_state):
        final_state.update(next(iter(chunk.values())))
        final_state["__last_node__"] = next(iter(chunk.keys()))
        yield chunk
    return final_state


async def _run_pipeline_with_updates(
    updater: TaskUpdater,
    initial_state: dict,
) -> dict:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    final_state: dict = {}
    react_started = False

    def _producer():
        try:
            for chunk in graph.stream(initial_state):
                node_name = next(iter(chunk.keys()))
                partial   = next(iter(chunk.values()))
                loop.call_soon_threadsafe(queue.put_nowait, (node_name, partial, None))
            loop.call_soon_threadsafe(queue.put_nowait, (None, None, None))  # sentinel
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, (None, None, exc))

    thread = asyncio.get_event_loop().run_in_executor(None, _producer)

    while True:
        node_name, partial, exc = await queue.get()

        if exc is not None:
            raise exc
        if node_name is None:
            break

        # Merge partial state into running final state
        if partial:
            final_state.update(partial)

        if node_name == _RETRIEVAL_DONE_NODE:
            quality  = final_state.get("context_quality", 0.0)
            n_chunks = len(final_state.get("reranked_hits", []))
            await _status(
                updater,
                f"Retrieval & grading complete — {n_chunks} chunks reranked, "
                f"context quality: {quality:.2f}",
            )

        elif node_name == _REACT_START_NODE and not react_started:
            react_started = True
            await _status(updater, "Starting ReAct reasoning loop…")

        elif node_name == _REACT_ITER_NODE:
            iteration = final_state.get("react_iterations", 0)
            goal      = final_state.get("goal_achieved", False)
            await _status(
                updater,
                f"ReFlAct iteration {iteration} complete"
                + (" — goal achieved" if goal else ""),
            )

    await thread  # ensure thread is done
    return final_state


def _build_clarification_message(deferral: dict) -> str | None:
    """Build a clarification message for the voice agent; None for irrecoverable_gap_after_retry."""
    deferral_type = deferral.get("type", "irrecoverable_gap")
    gaps          = deferral.get("gaps") or []
    clarification = deferral.get("suggested_clarification") or ""

    if deferral_type == "irrecoverable_gap_after_retry":
        gap_list = "; ".join(gaps[:3]) if gaps else "the requested information"
        return (
            f"I'm sorry, but even with the additional details you provided, "
            f"I was unable to find information about: {gap_list}. "
            "This topic does not appear to be covered in our knowledge base. "
            "Please contact a human support agent for further assistance."
        )

    if deferral_type == "out_of_domain":
        return (
            "I'm sorry, but that question is outside the scope of my knowledge base. "
            + (clarification or "Could you clarify what system or topic you're asking about?")
        )

    if gaps:
        gap_list = "; ".join(gaps[:3])
        return (
            f"I wasn't able to find enough information to fully answer your question. "
            f"Specifically, I'm missing: {gap_list}. "
            + (clarification or "Could you provide more details so I can help you better?")
        )

    return (
        "I wasn't able to find sufficient information to answer your question. "
        + (clarification or "Could you provide more context or rephrase your question?")
    )


async def _emit_deferral(
    updater: TaskUpdater,
    result: dict,
    deferral: dict,
    user_id,
    ticket_id,
) -> None:
    deferral_type    = deferral.get("type", "irrecoverable_gap")
    is_final_failure = deferral_type == "irrecoverable_gap_after_retry"

    enriched_deferral = {
        **deferral,
        "suggested_clarification": _build_clarification_message(deferral),
    }

    payload = {
        "status":          "deferred_final" if is_final_failure else "deferred",
        "answer":          result.get("final_answer", ""),
        "deferral_reason": enriched_deferral,
        "confidence":      0.0,
        "cited_chunk_ids": [],
        "user_id":         user_id,
        "ticket_id":       ticket_id,
    }

    logger.warning(
        "Query deferred (type=%s, gaps=%s) ticket=%s",
        deferral_type, deferral.get("gaps"), ticket_id,
    )

    await updater.add_artifact(
        parts=[Part(text=json.dumps(payload, default=_serialise, ensure_ascii=False))],
        name="graph_rag_result",
    )

    if is_final_failure:
        await updater.failed(
            message=updater.new_agent_message([Part(text="Deferring query (irrecoverable gap)")]),
        )
    else:
        await updater.update_status(
            TaskState.TASK_STATE_INPUT_REQUIRED,
            message=updater.new_agent_message([Part(text="Deferring query — input required")]),
        )


async def _emit_answer(
    updater: TaskUpdater,
    result: dict,
    verdict: str,
    confidence: float,
    user_id,
    ticket_id,
) -> None:
    payload = {
        "status":          "answered",
        "answer":          result.get("final_answer", ""),
        "confidence":      confidence,
        "crag_verdict":    verdict,
        "cited_chunk_ids": result.get("cited_chunk_ids", []),
        "user_id":         user_id,
        "ticket_id":       ticket_id,
    }

    await updater.add_artifact(
        parts=[Part(text=json.dumps(payload, default=_serialise, ensure_ascii=False))],
        name="graph_rag_result",
    )
    await updater.complete(
        message=updater.new_agent_message(
            [Part(text=f"Answer ready (confidence: {confidence:.2f})")]
        )
    )


class GraphRagExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id    = context.task_id
        context_id = context.context_id
        updater    = TaskUpdater(event_queue, task_id, context_id)

        raw_input: str = context.get_user_input()

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        if not raw_input or not raw_input.strip():
            await updater.add_artifact(
                parts=[Part(text=json.dumps({"error": "Empty query received"}))],
                name="graph_rag_result",
            )
            await updater.complete()
            return

        try:
            payload_in      = json.loads(raw_input)
            query           = payload_in.get("query", "").strip()
            user_id         = payload_in.get("user_id")
            ticket_id       = payload_in.get("ticket_id")
            user_memories   = payload_in.get("user_memories", [])
            interaction_mem = payload_in.get("interaction_memory_hits", [])
            user_clarification = payload_in.get("user_clarification", "").strip()
        except (json.JSONDecodeError, AttributeError):
            query              = raw_input.strip()
            user_id            = None
            ticket_id          = None
            user_memories      = []
            interaction_mem    = []
            user_clarification = ""

        prev_deferral = _previous_deferral(context)
        if prev_deferral:
            original_q    = prev_deferral.get("original_query", "").strip()
            gaps          = prev_deferral.get("gaps") or []
            clarification = user_clarification or query

            logger.info(
                "RETRY detected — context_id=%s task_id=%s | "
                "original_query=%r | gaps=%s | user_clarification=%r",
                context_id, task_id, original_q, gaps, clarification,
            )

            if original_q and clarification and clarification != original_q:
                query = f"{original_q} — additional context: {clarification}"
                logger.info("RETRY merged query: %r", query)
            elif original_q:
                query = original_q
                logger.info("RETRY no clarification supplied — retrying original query: %r", query)

            deferral_context = {
                "original_query":     original_q,
                "user_clarification": clarification,
                "gaps":               gaps,
            }
        else:
            deferral_context = None

        if not query:
            await updater.add_artifact(
                parts=[Part(text=json.dumps({"error": "Empty query received"}))],
                name="graph_rag_result",
            )
            await updater.complete()
            return

        logger.info(
            "Graph-RAG query: %r  (user_id=%s, ticket_id=%s, memories=%d, interactions=%d, retry=%s)",
            query, user_id, ticket_id, len(user_memories), len(interaction_mem),
            bool(prev_deferral),
        )

        status_prefix = "[RETRY] " if prev_deferral else ""
        await _status(updater, f"{status_prefix}Received query: {query!r}" + (f" [ticket={ticket_id}]" if ticket_id else ""))
        await _status(updater, "Running retrieval pipeline (semantic + BM25 + graph)…")

        try:
            result = await _run_pipeline_with_updates(
                updater,
                _make_initial_state(
                    query,
                    user_id=user_id,
                    ticket_id=ticket_id,
                    user_memories=user_memories,
                    deferral_context=deferral_context,
                ),
            )
        except Exception as exc:
            logger.exception("Pipeline error for query %r", query)
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=updater.new_agent_message(
                    [Part(text=f"Pipeline error: {exc}")]
                ),
            )
            return

        verdict         = result.get("crag_verdict", "")
        confidence      = result.get("confidence", 0.0)
        deferral        = result.get("deferral_reason")
        is_deferred     = deferral is not None

        if is_deferred:
            deferral_type = deferral.get("type", "unknown")
            gaps_out      = deferral.get("gaps") or []
            covered_out   = deferral.get("covered") or []
            if prev_deferral:
                logger.warning(
                    "RETRY outcome — DEFERRED AGAIN (type=%s) "
                    "context_id=%s task_id=%s | still_missing=%s | resolved=%s",
                    deferral_type, context_id, task_id, gaps_out, covered_out,
                )
            else:
                logger.warning(
                    "FIRST DEFERRAL (type=%s) context_id=%s task_id=%s "
                    "| missing=%s | resolved=%s | crag=%.2f",
                    deferral_type, context_id, task_id,
                    gaps_out, covered_out, result.get("context_quality", 0.0),
                )
        else:
            cited = result.get("cited_chunk_ids") or []
            if prev_deferral:
                logger.info(
                    "RETRY outcome — ANSWERED after deferral "
                    "context_id=%s task_id=%s | confidence=%.2f crag=%s cited=%d",
                    context_id, task_id, confidence, verdict, len(cited),
                )
            else:
                logger.info(
                    "ANSWERED context_id=%s task_id=%s | confidence=%.2f crag=%s cited=%d",
                    context_id, task_id, confidence, verdict, len(cited),
                )

        await _status(
            updater,
            f"Pipeline complete — CRAG verdict: {verdict}, confidence: {confidence:.2f}"
            + (" [DEFERRED]" if is_deferred else ""),
        )

        if is_deferred:
            await _emit_deferral(updater, result, deferral, user_id, ticket_id)
        else:
            await _emit_answer(updater, result, verdict, confidence, user_id, ticket_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
