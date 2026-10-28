"""Cross-encoder reranking node.

RERANKER_BACKEND="local"  → sentence-transformers cross-encoder (ms-marco-MiniLM-L-6-v2)
RERANKER_BACKEND="vllm"   → BGE-reranker-v2-m3 on GPU server via SSH tunnel (port 8002)
"""
from __future__ import annotations
import json
import logging
import urllib.error
import urllib.request

import config
from agents.ablation import enabled
from agents.state import AgentState, ScoredChunk

log = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None   # lazy-loaded


def _get_local_model():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        log.info("Loading cross-encoder model %s ...", _MODEL_NAME)
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def _rerank_local(query: str, candidates: list[ScoredChunk]) -> list[tuple[ScoredChunk, float]]:
    model = _get_local_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    return list(zip(candidates, scores))


def _rerank_vllm(query: str, candidates: list[ScoredChunk]) -> list[tuple[ScoredChunk, float]]:
    """Call vLLM's /v1/score endpoint (BGE reranker) via SSH tunnel on port 8002."""
    url = f"{config.VLLM_RERANKER_BASE_URL.rstrip('/')}/v1/score"
    headers = {"Content-Type": "application/json"}
    if config.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.VLLM_API_KEY}"
    payload = json.dumps({
        "model": config.VLLM_RERANKER_MODEL,
        "text_1": query,
        "text_2": [c.text for c in candidates],
        "truncate_prompt_tokens": 512,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"vLLM reranker returned HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"vLLM reranker request to {url} failed: {exc}. "
            "Is the SSH tunnel active? (ssh gpu-server)"
        ) from exc

    scores = [item["score"] for item in sorted(data["data"], key=lambda x: x["index"])]
    return list(zip(candidates, scores))


def _metadata_rescore(chunk) -> float:
    """Adjust cross-encoder score based on metadata quality signals (recency, doc type, status)."""
    from datetime import datetime
    
    score = chunk.score
    meta = chunk.metadata
    
    date_field = None
    for k in ["updated_at", "ticket_resolved_date", "modified", "last_modified", 
              "created_at", "ticket_created_date", "created"]:
        if meta.get(k):
            date_field = meta[k]
            break

    if date_field:
        try:
            from datetime import timezone
            if isinstance(date_field, str):
                date_str = str(date_field).strip()
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"
                if len(date_str) == 10:  # YYYY-MM-DD
                    date_str += "T00:00:00+00:00"
                doc_date = datetime.fromisoformat(date_str)
            else:
                doc_date = date_field

            if doc_date.tzinfo is None:
                doc_date = doc_date.replace(tzinfo=timezone.utc)
            
            age_days = (datetime.now(timezone.utc) - doc_date).days
            
            if age_days < 180:       # < 6 months
                score += 0.03
            elif age_days < 365:     # < 1 year
                score += 0.01
            elif age_days > 1825:    # > 5 years
                score -= 0.05
        except (ValueError, TypeError, AttributeError):
            pass

    doc_type = meta.get('doc_type', '')
    _DOC_TYPE_BOOSTS = {"qa": 0.03, "document": 0.01, "kb_article": 0.01, "ticket": 0.0}
    score += _DOC_TYPE_BOOSTS.get(doc_type, 0.0)
    
    correspondent = meta.get('correspondent', '').lower()
    source_system = meta.get('source_system', '').lower()
    channel = meta.get('channel', '').lower()
    combined_source = f"{correspondent} {source_system} {channel}".strip()
    
    if any(x in combined_source for x in ['wix', 'support', 'official', 'internal']):
        score += 0.02
    
    status = meta.get('status', '').lower()
    if status in ['resolved', 'closed', 'completed']:
        score += 0.01
    
    priority = meta.get('priority', '').lower()
    if priority in ['high', 'urgent', 'critical']:
        score += 0.005

    csat = meta.get('customer_satisfaction_score')
    if csat is not None:
        try:
            csat_float = float(csat)
            if csat_float >= 4.5:
                score += 0.01
        except (ValueError, TypeError):
            pass

    _CHUNK_TYPE_BOOSTS = {"resolution": 0.03, "answer": 0.03, "problem": -0.01, "question": -0.01}
    score += _CHUNK_TYPE_BOOSTS.get(chunk.chunk_type, 0.0)
    
    return score


def reranker(state: AgentState) -> dict:
    candidates = state.get("fused_hits") or []
    if not candidates:
        return {"reranked_hits": []}

    sub_queries = state.get("sub_queries") or []
    query = " | ".join(sub_queries) if sub_queries else state["original_query"]
    backend = config.RERANKER_BACKEND.lower()

    # chunk_id is a safe cache key within a single run; EVOLVE retries reuse cached scores.
    cache: dict = dict(state.get("reranker_score_cache") or {})
    cached_pairs:  list[tuple[ScoredChunk, float]] = []
    missing:       list[ScoredChunk] = []
    for c in candidates:
        if c.id in cache:
            cached_pairs.append((c, cache[c.id]))
        else:
            missing.append(c)

    if missing:
        if backend == "vllm":
            new_pairs = _rerank_vllm(query, missing)
        else:
            new_pairs = _rerank_local(query, missing)
        for c, s in new_pairs:
            cache[c.id] = float(s)
    else:
        new_pairs = []

    all_pairs = cached_pairs + list(new_pairs)

    boost_on = enabled(state, "rerank_boost")
    score_changes = []
    rescored_pairs = []
    for chunk, score in all_pairs:
        chunk.score = float(score)
        old_score = chunk.score

        if boost_on:
            chunk.score = _metadata_rescore(chunk)
        rescored_pairs.append((chunk, chunk.score))
        
        if abs(chunk.score - old_score) > 0.001:
            score_changes.append({
                'id': chunk.id[:12],
                'old': old_score,
                'new': chunk.score,
                'delta': chunk.score - old_score,
                'status': chunk.metadata.get('status', 'N/A'),
                'priority': chunk.metadata.get('priority', 'N/A')
            })

    ranked = sorted(rescored_pairs, key=lambda x: x[1], reverse=True)
    top = [c for c, _ in ranked[:config.RERANKER_TOP_K]]

    log.info("Reranker [%s] → kept %d / %d  (top: %.4f  id: %s)  cache: %d hits, %d new",
             backend, len(top), len(candidates),
             top[0].score if top else 0.0,
             top[0].id if top else "—",
             len(cached_pairs), len(missing))

    if score_changes:
        log.info("[Reranker] Metadata rescoring changed %d scores:", len(score_changes))
        for change in score_changes[:5]:
            log.info("  %s: %.4f → %.4f (Δ%+.4f, status=%s, prio=%s)",
                     change['id'], change['old'], change['new'],
                     change['delta'], change['status'], change['priority'])
    else:
        log.info("[Reranker] No significant score changes from metadata")
    
    if top:
        log.info("[Reranker] Top chunk after rescore: id=%s, score=%.4f", top[0].id, top[0].score)

    return {"reranked_hits": top, "reranker_score_cache": cache}
