"""Failed attempt extractor — classifies interaction memories into failed/resolved solution lists."""
from __future__ import annotations

import logging
from agents.state import AgentState
from agents.llm import chat_json

log = logging.getLogger(__name__)

_PROMPT = """\
You are analysing a customer support interaction history.
Below are memory entries recording things this user has tried or experienced.

For each entry, decide:
- "failed"   — the user tried a specific solution/step and it did NOT resolve their problem
- "resolved" — the user applied a solution and it DID resolve their problem
- "neutral"  — the entry does not describe a concrete attempted solution

Interaction memories:
{memories}

Return JSON only — no preamble:
{{
  "failed":   ["<concise description of failed solution>", ...],
  "resolved": ["<concise description of resolved solution>", ...]
}}

Rules:
- Keep each description concise (one short phrase, ≤15 words)
- Omit neutral entries entirely
- If no entries qualify, return empty lists
"""


def failed_attempt_extractor(state: AgentState) -> dict:
    interaction_memories: list[dict] = state.get("interaction_memories") or []

    if not interaction_memories:
        log.info("Failed-attempt extractor: no interaction memories — skipping")
        return {"per_user_failed": [], "global_failed": [], "resolved": []}

    memory_lines = []
    for m in interaction_memories:
        text = (m.get("memory") or "").strip()
        if text:
            memory_lines.append(f"- {text}")

    if not memory_lines:
        return {"per_user_failed": [], "global_failed": [], "resolved": []}

    per_user_failed: list[str] = []
    resolved: list[str] = []

    try:
        result = chat_json(_PROMPT.format(memories="\n".join(memory_lines)))
        raw_failed   = result.get("failed")   or []
        raw_resolved = result.get("resolved") or []

        if isinstance(raw_failed, list):
            per_user_failed = [str(x).strip() for x in raw_failed if x]
        if isinstance(raw_resolved, list):
            resolved = [str(x).strip() for x in raw_resolved if x]

    except Exception as exc:
        log.warning("Failed-attempt extractor: LLM call failed (%s) — empty lists", exc)

    log.info(
        "Failed-attempt extractor: per_user_failed=%d  resolved=%d  global_failed=0",
        len(per_user_failed), len(resolved),
    )

    return {
        "per_user_failed": per_user_failed,
        "global_failed":   [],
        "resolved":        resolved,
    }
