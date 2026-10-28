"""Embedding backend dispatcher — Gemini / Ollama / LiteLLM / vLLM via config.EMBED_BACKEND."""
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)

# ── Gemini rate-limit constants ───────────────────────────────────────────────
MIN_REQUEST_INTERVAL = 0.7   # seconds between calls (~85 RPM)
MAX_RETRIES          = 6
RETRY_BASE_WAIT      = 60    # seconds; attempt n waits RETRY_BASE_WAIT * 2^(n-1)


# ══════════════════════════════════════════════════════════════════════════════
# Gemini backend
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_client():
    from google import genai  # noqa: PLC0415
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to .env or switch EMBED_BACKEND=ollama."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE_LIMIT" in msg


def _gemini_embed_batch(client, batch: list[str]) -> list[list[float]]:
    from google.genai import types  # noqa: PLC0415
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=config.EMBED_DIMENSIONS,
                ),
            )
            return [e.values for e in response.embeddings]
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                wait = RETRY_BASE_WAIT * (2 ** (attempt - 1))
                log.warning(
                    "Rate limit hit (attempt %d/%d) — waiting %ds before retry",
                    attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini embedding failed after {MAX_RETRIES} retries")


def _embed_texts_gemini(texts: list[str]) -> list[list[float]]:
    client = _gemini_client()
    results: list[list[float]] = []
    total = len(texts)
    last_request_at = 0.0

    for start in range(0, total, config.EMBED_BATCH_SIZE):
        elapsed = time.monotonic() - last_request_at
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        results.extend(_gemini_embed_batch(client, batch))
        last_request_at = time.monotonic()

        done = min(start + config.EMBED_BATCH_SIZE, total)
        log.info("  [gemini] Embedded %d / %d chunks", done, total)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Ollama backend
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_post(endpoint: str, payload: dict) -> dict:
    """POST JSON to the Ollama REST API; return parsed response dict."""
    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request to {url} failed: {exc}. "
            "Is Ollama running? Try: ollama serve"
        ) from exc


def _embed_texts_ollama(texts: list[str]) -> list[list[float]]:
    results: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        resp = _ollama_post("api/embed", {
            "model": config.OLLAMA_EMBED_MODEL,
            "input": batch,
        })
        embeddings = resp.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(embeddings or [])} embeddings for "
                f"batch of {len(batch)}: {resp}"
            )
        results.extend(embeddings)

        done = min(start + config.EMBED_BATCH_SIZE, total)
        log.info("  [ollama] Embedded %d / %d chunks", done, total)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# LiteLLM backend
# ══════════════════════════════════════════════════════════════════════════════

def _embed_texts_litellm(texts: list[str]) -> list[list[float]]:
    if not config.LITELLM_API_KEY:
        raise ValueError(
            "LITELLM_API_KEY is not set. Add it to .env or switch EMBED_BACKEND."
        )
    base_url = config.LITELLM_BASE_URL.rstrip("/")
    url = f"{base_url}/embeddings"
    results: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        payload = json.dumps({"model": config.LITELLM_EMBED_MODEL, "input": batch}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LITELLM_API_KEY}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LiteLLM embed request to {url} failed: {exc}") from exc

        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"LiteLLM returned {len(embeddings)} embeddings for batch of {len(batch)}"
            )
        results.extend(embeddings)

        done = min(start + config.EMBED_BATCH_SIZE, total)
        log.info("  [litellm] Embedded %d / %d chunks", done, total)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# vLLM backend
# ══════════════════════════════════════════════════════════════════════════════

# 3800 char ceiling ≈ 4096 token limit minus tokenizer overhead.
_VLLM_MAX_CHARS = 3_800


def _truncate_for_vllm(texts: list[str]) -> list[str]:
    truncated = []
    for t in texts:
        if len(t) > _VLLM_MAX_CHARS:
            log.warning(
                "  [vllm] Truncating chunk from %d → %d chars (exceeds ~4096-token limit)",
                len(t), _VLLM_MAX_CHARS,
            )
            truncated.append(t[:_VLLM_MAX_CHARS])
        else:
            truncated.append(t)
    return truncated


def _vllm_embed_batch(url: str, headers: dict, batch: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": config.VLLM_EMBED_MODEL, "input": batch}).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"vLLM embed HTTP {exc.code} for batch of {len(batch)}: {body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"vLLM embed request to {url} failed: {exc}. "
            "Is the SSH tunnel active? (ssh gpu-server)"
        ) from exc

    embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    if len(embeddings) != len(batch):
        raise RuntimeError(
            f"vLLM returned {len(embeddings)} embeddings for batch of {len(batch)}"
        )
    return embeddings


def _embed_texts_vllm(texts: list[str]) -> list[list[float]]:
    base_url = config.VLLM_EMBED_BASE_URL.rstrip("/")
    url = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if config.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.VLLM_API_KEY}"

    texts = _truncate_for_vllm(texts)
    results: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        results.extend(_vllm_embed_batch(url, headers, batch))
        done = min(start + config.EMBED_BATCH_SIZE, total)
        log.info("  [vllm] Embedded %d / %d chunks", done, total)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str]) -> list[list[float]]:
    backend = config.EMBED_BACKEND.lower()
    if backend == "gemini":
        return _embed_texts_gemini(texts)
    elif backend == "ollama":
        return _embed_texts_ollama(texts)
    elif backend == "litellm":
        return _embed_texts_litellm(texts)
    elif backend == "vllm":
        return _embed_texts_vllm(texts)
    else:
        raise ValueError(
            f"Unknown EMBED_BACKEND={config.EMBED_BACKEND!r}. "
            "Valid options: 'gemini', 'ollama', 'litellm', 'vllm'."
        )


def embed_chunks(chunks: list) -> None:
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
