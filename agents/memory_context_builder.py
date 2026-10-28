"""Memory context builder — splits raw user_memories into profile/preference/interaction/ticket_ref lists."""
from __future__ import annotations

import logging

from agents.state import AgentState

log = logging.getLogger(__name__)


def memory_context_builder(state: AgentState) -> dict:
    raw: list[dict] = state.get("user_memories") or []

    profile:     list[dict] = []
    preference:  list[dict] = []
    interaction: list[dict] = []
    ticket_refs: list[dict] = []

    for entry in raw:
        try:
            meta: dict = entry.get("metadata") or {}
            category: str = str(meta.get("category") or "").lower().strip()

            if category == "profile":
                profile.append(entry)
            elif category == "preference":
                preference.append(entry)
            elif category == "interaction":
                interaction.append(entry)
            tid = meta.get("ticket_id")
            if tid is not None and str(tid).strip():
                ticket_refs.append(entry)

        except Exception:
            log.warning(
                "memory_context_builder: malformed entry skipped — %r",
                str(entry)[:120],
            )

    log.info(
        "Memory context: profile=%d  preference=%d  interaction=%d  ticket_refs=%d",
        len(profile), len(preference), len(interaction), len(ticket_refs),
    )

    return {
        "profile_memories":     profile,
        "preference_memories":  preference,
        "interaction_memories": interaction,
        "ticket_ref_memories":  ticket_refs,
    }
