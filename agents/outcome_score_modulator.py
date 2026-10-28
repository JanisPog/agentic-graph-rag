"""Outcome score modulator — adjusts reranker scores using failed/resolved outcome signals."""
from __future__ import annotations

import logging
from typing import Literal

import config
from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk
from graph.neo4j_client import Neo4jClient
from etl.embedder import embed_texts

log = logging.getLogger(__name__)

MatchType = Literal["per_user_failed", "global_failed", "resolved", "reuse_candidate"] | None

_CYPHER_KNOWN_ISSUES = """
CALL db.index.vector.queryNodes('known_issue_embedding', 3, $query_vector)
YIELD node AS k, score
WHERE score >= $threshold
RETURN k.resolution_notes AS resolution_text, score
"""


def _load_global_failed(expanded_queries: list[str]) -> list[str]:
    """Query KnownIssue centroids for globally-failing resolution notes."""
    if not expanded_queries:
        return []
    try:
        vecs = embed_texts([expanded_queries[0]])
        if not vecs or not vecs[0]:
            return []
        query_vector = vecs[0]

        with Neo4jClient() as client:
            rows = client.run(
                _CYPHER_KNOWN_ISSUES,
                {"query_vector": query_vector, "threshold": config.OUTCOME_KNOWN_ISSUE_THRESH},
            )

        notes = []
        for row in rows:
            text = (row.get("resolution_text") or "").strip()
            if text:
                notes.append(text)
        return notes

    except Exception as exc:
        log.warning("Outcome modulator: KnownIssue lookup failed (%s) — global_failed stays empty", exc)
        return []


def _significant_words(text: str) -> set[str]:
    """Words longer than 4 chars — filters most stop words."""
    return {w.lower() for w in text.split() if len(w) > 4}


def _match_type(
    hit: ScoredChunk,
    per_user_failed: list[str],
    global_failed: list[str],
    resolved: list[str],
    reuse_candidate_ids: set[str],
) -> tuple[MatchType, bool]:
    """Return (match_type, is_exact). Priority: per_user_failed > global_failed > resolved > reuse."""
    hit_ticket_id = (hit.ticket_id or "").strip()
    hit_words = _significant_words(hit.text)

    def _signal_matches(signal: str, use_ticket: bool) -> bool:
        if use_ticket and hit_ticket_id and hit_ticket_id == signal.strip():
            return True
        sig_words = _significant_words(signal)
        return len(hit_words & sig_words) >= 2

    if hit.id in reuse_candidate_ids:
        return "reuse_candidate", True

    for signal in per_user_failed:
        if hit_ticket_id and hit_ticket_id == signal.strip():
            return "per_user_failed", True
    for signal in per_user_failed:
        if _signal_matches(signal, use_ticket=False):
            return "per_user_failed", False

    for signal in global_failed:
        if hit_ticket_id and hit_ticket_id == signal.strip():
            return "global_failed", True
    for signal in global_failed:
        if _signal_matches(signal, use_ticket=False):
            return "global_failed", False

    for signal in resolved:
        if hit_ticket_id and hit_ticket_id == signal.strip():
            return "resolved", True
    for signal in resolved:
        if _signal_matches(signal, use_ticket=False):
            return "resolved", False

    return None, False


def _adjusted_score(original: float, match_type: MatchType, is_exact: bool) -> float:
    """Apply outcome factor; halves the delta for soft (text-overlap) matches."""
    factor_map: dict[str, float] = {
        "per_user_failed":  config.OUTCOME_PENALTY_PER_USER,
        "global_failed":    config.OUTCOME_PENALTY_GLOBAL,
        "resolved":         config.OUTCOME_BOOST_RESOLVED,
        "reuse_candidate":  config.OUTCOME_BOOST_REUSE,
    }
    if match_type is None:
        return original

    factor = factor_map[match_type]

    if not is_exact:
        factor = 1.0 + (factor - 1.0) * 0.5

    new_score = original * factor
    return max(config.OUTCOME_SCORE_MIN, min(config.OUTCOME_SCORE_MAX, new_score))


def outcome_score_modulator(state: AgentState) -> dict:
    reranked_hits: list[ScoredChunk] = state.get("reranked_hits") or []

    if not enabled(state, "outcome_modulator"):
        log.info("Outcome modulator ablated — passthrough")
        return {"outcome_modulated_hits": reranked_hits}

    per_user_failed: list[str] = state.get("per_user_failed") or []
    resolved: list[str] = state.get("resolved") or []
    reuse_candidates: list[dict] = state.get("reuse_candidates") or []

    global_failed: list[str] = state.get("global_failed") or []
    if not global_failed:
        expanded_queries: list[str] = state.get("expanded_queries") or []
        global_failed = _load_global_failed(expanded_queries)
        if global_failed:
            log.info("Outcome modulator: loaded %d global_failed entries from KnownIssue", len(global_failed))

    reuse_candidate_ids: set[str] = {
        str(c.get("source_id") or c.get("ticket_id") or "")
        for c in reuse_candidates
        if c
    } - {""}

    if not per_user_failed and not global_failed and not resolved and not reuse_candidate_ids:
        log.info("Outcome modulator: no signals — passthrough")
        return {"outcome_modulated_hits": reranked_hits, "global_failed": global_failed}

    adjusted_hits: list[ScoredChunk] = []
    counts: dict[str, int] = {
        "per_user_failed": 0,
        "global_failed": 0,
        "resolved": 0,
        "reuse_candidate": 0,
    }
    adjusted_count = 0

    for hit in reranked_hits:
        match_type, is_exact = _match_type(
            hit, per_user_failed, global_failed, resolved, reuse_candidate_ids
        )
        new_score = _adjusted_score(hit.score, match_type, is_exact)

        if match_type is not None:
            log.debug(
                "Score modulator: chunk=%s match=%s%s score %.3f → %.3f",
                hit.id, match_type, " (exact)" if is_exact else " (fuzzy)", hit.score, new_score,
            )
            counts[match_type] += 1
            adjusted_count += 1

        adjusted_hits.append(ScoredChunk(
            id=hit.id,
            text=hit.text,
            score=new_score,
            chunk_type=hit.chunk_type,
            ticket_id=hit.ticket_id,
            metadata=hit.metadata,
        ))

    adjusted_hits.sort(key=lambda c: c.score, reverse=True)

    log.info(
        "Score modulator: %d hits adjusted "
        "(per_user=%d global=%d resolved=%d reuse=%d)",
        adjusted_count,
        counts["per_user_failed"],
        counts["global_failed"],
        counts["resolved"],
        counts["reuse_candidate"],
    )

    return {"outcome_modulated_hits": adjusted_hits, "global_failed": global_failed}
