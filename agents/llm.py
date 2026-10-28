"""LLM + embedding helpers — dispatches to LiteLLM, Gemini, Ollama, or vLLM based on config."""
from __future__ import annotations
import json
import logging
import urllib.error
import urllib.request

import config
from etl.embedder import embed_texts

log = logging.getLogger(__name__)


# Backend-agnostic counters; token counts are estimates (chars/4). Not thread-safe.
_metrics = {"llm_calls": 0, "prompt_tokens_est": 0, "completion_tokens_est": 0}


def reset_metrics() -> None:
    _metrics.update(llm_calls=0, prompt_tokens_est=0, completion_tokens_est=0)


def get_metrics() -> dict:
    return dict(_metrics)


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


# ── Chat backends ─────────────────────────────────────────────────────────────

def _chat_litellm(prompt: str, temperature: float, model: str | None = None) -> str:
    url = f"{config.LITELLM_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model or config.LITELLM_CHAT_MODEL,
        # /no_think disables Qwen3 chain-of-thought when model is a Qwen3 variant.
        "messages": [{"role": "user", "content": prompt + " /no_think"}],
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.LITELLM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"LiteLLM chat request failed: {exc}. Check LITELLM_BASE_URL and LITELLM_API_KEY."
        ) from exc


def _chat_gemini(prompt: str, temperature: float) -> str:
    from google import genai  # noqa: PLC0415
    from google.genai import types as genai_types  # noqa: PLC0415
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_CHAT_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def _chat_ollama(prompt: str, temperature: float) -> str:
    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": config.OLLAMA_CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama chat request failed: {exc}. Is Ollama running?"
        ) from exc


def _chat_vllm(prompt: str, temperature: float) -> str:
    url = f"{config.VLLM_CHAT_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.VLLM_API_KEY}"
    payload = json.dumps({
        "model": config.VLLM_CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # Disable Qwen3 chain-of-thought — we need structured JSON output, not reasoning traces.
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"vLLM chat request failed: {exc}. Is the SSH tunnel active? (ssh gpu-server)"
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def chat(prompt: str, temperature: float = config.TEMPERATURE_REASON) -> str:
    """Send a prompt to the configured chat backend and return raw text."""
    backend = config.CHAT_BACKEND.lower()
    if backend == "litellm":
        result = _chat_litellm(prompt, temperature)
    elif backend == "vllm":
        result = _chat_vllm(prompt, temperature)
    elif backend == "gemini":
        result = _chat_gemini(prompt, temperature)
    elif backend == "ollama":
        result = _chat_ollama(prompt, temperature)
    else:
        raise ValueError(f"Unknown CHAT_BACKEND={config.CHAT_BACKEND!r}")

    _metrics["llm_calls"] += 1
    _metrics["prompt_tokens_est"] += _est_tokens(prompt)
    _metrics["completion_tokens_est"] += _est_tokens(result)
    return result


def _parse_json_response(raw: str) -> dict | list:
    """Extract the first JSON object/array from a raw model response.

    Shared by chat_json and judge_chat_json so both get identical robust parsing
    (bare parse → fenced block → last {...}/[...] block, after stripping Qwen3
    <think>…</think> traces).
    """
    import re
    raw = (raw or "").strip()

    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        matches = list(re.finditer(pattern, raw))
        if matches:
            try:
                return json.loads(matches[-1].group(0))
            except json.JSONDecodeError:
                pass

    raise ValueError(f"No valid JSON found in model response: {raw[:200]!r}")


def chat_json(prompt: str, temperature: float = config.TEMPERATURE_STRUCTURED) -> dict | list:
    """Send a prompt and parse the first JSON object/array from the response."""
    return _parse_json_response(chat(prompt, temperature))


# ── Evaluation judge (separate model, decoupled from the generator) ───────────

def judge_chat(prompt: str, temperature: float = 0.0) -> str:
    """Run a prompt through the judge model (JUDGE_MODEL config) or fall back to chat()."""
    model = (config.JUDGE_MODEL or "").strip()
    if not model:
        return chat(prompt, temperature)
    return _chat_litellm(prompt, temperature, model=model)


def judge_chat_json(prompt: str, temperature: float = 0.0) -> dict | list:
    """judge_chat + robust JSON parsing (same extraction as chat_json)."""
    return _parse_json_response(judge_chat(prompt, temperature))


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    """Embed a single query string using the configured embedding backend."""
    return embed_texts([text])[0]
