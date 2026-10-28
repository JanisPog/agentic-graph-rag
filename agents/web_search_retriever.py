"""Web search retriever — Brave Search / Tavily / Parallel backends."""
from __future__ import annotations
import hashlib
import logging
import time

import config
from agents.state import AgentState, ScoredChunk
from agents.pivot_log import pivot_log

log = logging.getLogger(__name__)


def _search_brave(query: str, max_results: int, timeout: int) -> list[dict]:
    """Brave Search API — requires BRAVE_SEARCH_API_KEY."""
    import requests  # noqa: PLC0415

    api_key = getattr(config, "BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        log.error("BRAVE_SEARCH_API_KEY not set — cannot use Brave backend")
        return []

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params={"q": query, "count": max_results},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("description", ""),
            }
            for r in data.get("web", {}).get("results", [])
        ]
    except Exception as exc:
        log.warning("Brave search failed: %s", exc)
        return []


def _search_tavily(query: str, max_results: int, timeout: int) -> list[dict]:
    """Tavily search — requires TAVILY_API_KEY."""
    try:
        from tavily import TavilyClient  # noqa: PLC0415
    except ImportError:
        log.error("tavily-python not installed. Run: pip install tavily-python")
        return []

    api_key = getattr(config, "TAVILY_API_KEY", "")
    if not api_key:
        log.error("TAVILY_API_KEY not set — cannot use Tavily backend")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception as exc:
        log.warning("Tavily search failed: %s", exc)
        return []


def _search_parallel(query: str, max_results: int, timeout: int) -> list[dict]:
    """Parallel Web Systems search — requires PARALLEL_API_KEY."""
    import requests  # noqa: PLC0415

    api_key = getattr(config, "PARALLEL_API_KEY", "")
    if not api_key:
        log.error("PARALLEL_API_KEY not set — cannot use Parallel backend")
        return []

    try:
        resp = requests.post(
            "https://api.parallel.ai/v1/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "objective": query,
                "search_queries": [query],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            excerpts = r.get("excerpts", [])
            content = "\n\n".join(excerpts) if excerpts else ""
            if content:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": content,
                })
        return results
    except Exception as exc:
        log.warning("Parallel search failed: %s", exc)
        return []


_BACKENDS = {
    "brave": _search_brave,
    "tavily": _search_tavily,
    "parallel": _search_parallel,
}


def _to_scored_chunks(results: list[dict], trust_penalty: float) -> list[ScoredChunk]:
    chunks = []
    for rank, r in enumerate(results, start=1):
        url = r.get("url", "")
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        if not content:
            continue

        chunk_id = f"web_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        text = f"{title}\n\n{content}\n\nSource: {url}"
        raw_score = 1.0 / rank
        score = raw_score * trust_penalty

        chunks.append(ScoredChunk(
            id=chunk_id,
            text=text,
            score=score,
            chunk_type="web_result",
            ticket_id="",  # no KB ticket association
        ))
    return chunks


def web_search_retriever(state: AgentState) -> dict:
    """LangGraph node — performs a web search and returns web_hits."""
    if not getattr(config, "WEB_SEARCH_ENABLED", False):
        log.info("Web search disabled (WEB_SEARCH_ENABLED=false)")
        pivot_log("WEB_SEARCH", decision="disabled", reason="config")
        return {"web_hits": []}

    query = (state.get("crag_missing_hint") or "").strip()
    if not query:
        query = state["original_query"]

    provider = getattr(config, "WEB_SEARCH_PROVIDER", "duckduckgo")
    max_results = getattr(config, "WEB_SEARCH_MAX_RESULTS", 5)
    timeout = getattr(config, "WEB_SEARCH_TIMEOUT_S", 10)
    trust_penalty = getattr(config, "WEB_SEARCH_TRUST_PENALTY", 0.85)

    search_fn = _BACKENDS.get(provider)
    if not search_fn:
        log.error("Unknown web search provider: %r (available: %s)",
                  provider, ", ".join(_BACKENDS))
        return {"web_hits": []}

    t0 = time.perf_counter()
    raw_results = search_fn(query, max_results, timeout)
    elapsed = time.perf_counter() - t0

    chunks = _to_scored_chunks(raw_results, trust_penalty)

    pivot_log(
        "WEB_SEARCH",
        decision=f"{len(chunks)}_results",
        reason=f"provider={provider}",
        query=query[:80],
        elapsed_ms=f"{elapsed * 1000:.0f}",
        trust_penalty=f"{trust_penalty:.2f}",
    )

    log.info("Web search [%s] → %d chunks in %.1fs  query=%r",
             provider, len(chunks), elapsed, query[:60])

    return {"web_hits": chunks}
