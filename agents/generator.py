"""Final generation node — Multi-Hop Synthesis with cited answer."""
from __future__ import annotations
import logging

from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk, ReActStep
import config
from agents.llm import chat_json
from agents.kb_context import get_kb_context

log = logging.getLogger(__name__)

_CITATION_PREFERENCE = """
When citing sources, prefer passages with:
- Status "Closed" or "Resolved" (proven solutions)
- Recent dates (current information)
- High customer satisfaction ratings (verified quality)
"""

_PROMPT = """\
You are a technical support specialist. Answer the user's query using ONLY
the provided context passages and your reasoning trace. Do not introduce
external knowledge.
{citation_preference_block}
{kb_context_block}\
Include every essential step the context provides for the query: when the
context gives a numbered or step-by-step procedure, reproduce all of its steps,
in order, with their specific UI labels, field names, and links — do not merge or
summarise steps away. But stay scoped to the query: do not add background,
explanation, caveats, or extra procedures the user did not ask for. Match the
directness of a KB answer — complete on the steps that matter, nothing beyond them.

Write the answer as a direct solution to the user. Do NOT comment on the context
itself or on this task — no remarks about whether the passages are complete,
sufficient, missing, or what the "provided steps" do or don't contain. If a needed
detail is genuinely absent, simply give the steps you can and stop; do not narrate
the gap unless a PARTIAL-ANSWER DIRECTIVE below explicitly tells you to.

Important: if the KB uses a different but equivalent term for what the user asked
(e.g. "disconnect" instead of "delete", "unlink" instead of "remove"), provide the
KB's procedure and note the terminology difference — do not withhold the answer.

Important: do NOT over-parse the actor or perspective in a colloquial query.
Phrasings like "how users can check my calendar" or "how people see my page"
usually just mean "how do I set up / enable this feature" — they are NOT a demand
for separate visitor-side or end-user-side instructions. When the context gives the
procedure for the feature, ANSWER WITH THAT PROCEDURE. Do not refuse, hedge, or say
the context only covers the owner/admin side and not the visitor side, unless the
user EXPLICITLY emphasised a distinct perspective (e.g. "what will my visitors see").

For each factual claim in your answer, cite the passage:
- KB passages: use [P<n>] format
- Web sources: use [Web<n>: <url>] format (the URL is shown in the passage header)
{partial_answer_block}\
ORIGINAL QUERY:
{original_query}

CONTEXT PASSAGES:
{formatted_passages}

REASONING TRACE:
{scratchpad}

REFLECTION SUMMARY (key findings from retrieval loop):
{reflection_summary}

Respond in this JSON structure:
{{
  "answer": "<full answer text with inline citations [P1], [P2], ...>",
  "cited_passage_ids": ["chunk_id_1"],
  "confidence": <float 0.0-1.0>,
  "answer_type": "direct" | "multi_hop" | "partial" | "unknown"
}}"""



def _metadata_header(c, n: int) -> list[str]:
    """Build per-passage metadata header lines. Used when METADATA_IN_GENERATOR_ENABLED."""
    lines = [f"[P{n}] (id={c.id})"]
    meta = c.metadata

    if 'title' in meta:
        lines.append(f"Title: {meta['title']}")

    if 'document_type' in meta:
        lines.append(f"Type: {meta['document_type']}")
    elif 'article_type' in meta:
        lines.append(f"Type: {meta['article_type']}")
    elif 'category' in meta:
        lines.append(f"Category: {meta['category']}")

    if 'correspondent' in meta:
        lines.append(f"Source: {meta['correspondent']}")
    elif 'source_system' in meta:
        lines.append(f"Source: {meta['source_system']}")
    elif 'channel' in meta:
        lines.append(f"Channel: {meta['channel']}")

    if 'ticket_created_date' in meta:
        lines.append(f"Date: {str(meta['ticket_created_date']).split('T')[0]}")
    elif 'created' in meta:
        lines.append(f"Date: {str(meta['created']).split('T')[0]}")
    elif 'modified' in meta:
        lines.append(f"Last Modified: {str(meta['modified']).split('T')[0]}")

    if 'status' in meta:
        lines.append(f"Status: {meta['status']}")

    if c.chunk_type == "section":
        if 'heading_path' in meta:
            lines.append(f"Section: {meta['heading_path']}")
        tags = meta.get('tags')
        if isinstance(tags, list) and tags:
            lines.append(f"Tags: {', '.join(tags[:3])}")
    elif c.chunk_type in ('problem', 'resolution', 'question', 'answer'):
        if meta.get('priority', '').lower() in ('high', 'urgent', 'critical'):
            lines.append(f"Priority: {meta['priority']}")
        csat = meta.get('customer_satisfaction_score')
        if csat is not None:
            try:
                csat_float = float(csat)
                if csat_float >= 4.0:
                    lines.append(f"Customer Rating: {csat_float}/5.0")
            except (ValueError, TypeError):
                pass

    return lines


def _fmt_passages(chunks: list, include_metadata: bool) -> str:
    parts = []
    for i, c in enumerate(chunks):
        if c.chunk_type == "web_result":
            lines = c.text.strip().split('\n')
            url = lines[-1] if lines else "unknown"
            content = '\n'.join(lines[:-1]) if len(lines) > 1 else c.text
            parts.append(f"[Web{i + 1}: {url}]\n{content}")
        elif include_metadata:
            header = "\n".join(_metadata_header(c, i + 1))
            parts.append(f"{header}\n\n{c.text}")
        else:
            parts.append(f"[P{i + 1}] (id={c.id})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _fmt_scratchpad(steps: list[ReActStep]) -> str:
    if not steps:
        return "(no reasoning trace)"
    return "\n".join(
        f"Step {i+1}: thought={s.thought!r} action={s.action}({s.sub_query!r}) observation={s.observation!r}"
        for i, s in enumerate(steps)
    )




def generator(state: AgentState) -> dict:
    # Guard against duplicate invocations from parallel edge resolution.
    existing = (state.get("final_answer") or "").strip()
    if existing:
        log.info("Generator → skipping (final_answer already set, %d chars)", len(existing))
        return {}

    reuse_answer = (state.get("reuse_answer") or "").strip()
    if reuse_answer:
        answer = f"Based on a similar case:\n\n{reuse_answer}"
        log.info(
            "Generator → reuse fast-path  ticket=%s",
            state.get("reuse_ticket_id"),
        )
        return {
            "final_answer":    answer,
            "cited_chunk_ids": [state.get("reuse_ticket_id") or ""],
            "confidence":      0.85,
        }

    chunks = state.get("reranked_hits") or []
    scratchpad = state.get("react_scratchpad") or []
    query = state["original_query"]

    meta_in_gen = enabled(state, "metadata_in_generator")
    passages = _fmt_passages(chunks, include_metadata=meta_in_gen)

    for i, chunk in enumerate(chunks):
        log.info(f"[Generator] P{i+1}: id={chunk.id}, type={chunk.chunk_type}, score={chunk.score:.4f}")

    trace = _fmt_scratchpad(scratchpad)
    reflection_summary = state.get("reflection_summary") or "(none)"

    kb_ctx = get_kb_context()
    kb_context_block = f"Knowledge base context: {kb_ctx}\n\n" if kb_ctx else ""

    partial_note = (state.get("partial_answer_note") or "").strip()
    partial_answer_block = (
        f"\nPARTIAL-ANSWER DIRECTIVE: Some sub-goals could not be fully answered "
        f"from the available evidence. Acknowledge this explicitly in your reply:\n"
        f"  {partial_note}\n"
        f"Answer what you can with citations, then state which sub-goal(s) are "
        f"unresolved and why.\n"
        if partial_note else ""
    )

    citation_preference_block = _CITATION_PREFERENCE if meta_in_gen else ""

    prompt = _PROMPT.format(
        original_query=query,
        formatted_passages=passages,
        scratchpad=trace,
        reflection_summary=reflection_summary,
        kb_context_block=kb_context_block,
        partial_answer_block=partial_answer_block,
        citation_preference_block=citation_preference_block,
    )
    log.info(f"[Generator] Final prompt preview (first 1000 chars):\n{prompt[:1000]}")

    answer_type = "unknown"
    try:
        result = chat_json(prompt, temperature=config.TEMPERATURE_GENERATE)
        answer = result.get("answer", "")
        cited_ids = result.get("cited_passage_ids", [])
        confidence = float(result.get("confidence", 0.5))
        answer_type = result.get("answer_type", "unknown")
    except Exception as exc:
        log.warning("Generator failed (%s); using salvage path", exc)
        raw = str(exc)
        salvaged = ""
        import re, json as _json
        m = re.search(r'"answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', raw)
        if m:
            salvaged = m.group(1).encode().decode("unicode_escape", errors="ignore").strip()
        if not salvaged:
            salvaged = (
                "I'm sorry — I could not compose a structured answer for this query. "
                "Please try rephrasing or contact a human support agent."
            )
        answer = salvaged
        cited_ids = [c.id for c in chunks[:5]]
        confidence = 0.3

    log.info("Generator → answer_type=%s  confidence=%.2f  cited_chunks=%d",
             answer_type, confidence, len(cited_ids))
    return {
        "final_answer": answer,
        "cited_chunk_ids": cited_ids,
        "confidence": confidence,
    }
