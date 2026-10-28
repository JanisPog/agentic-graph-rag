"""A2A test client for the Graph-RAG agent.

Usage:
    .venv/bin/python a2a/test_client.py "Wie kann ich mein Wix Passwort zurücksetzen?"
    .venv/bin/python a2a/test_client.py "..." --user test1
    .venv/bin/python a2a/test_client.py "..." --user test1 --memory-url https://memory-host:8000
"""
import argparse
import asyncio
import os
import json
import ssl
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Override via AGENT_URL / MEMORY_SERVICE_URL (or --memory-url) to point at a
# non-local deployment.
BASE_URL = os.getenv("AGENT_URL", "http://localhost:8015")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8000")
DEFAULT_QUERY = "My login keeps failing after the recent update."


async def fetch_memories(user_id: str, query: str, memory_url: str) -> list[dict]:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        resp = await client.post(
            f"{memory_url}/memory-service/memory/search",
            json={"user_id": user_id, "query": query, "limit": 10, "threshold": 0.0,
                  "categories": ["interaction", "profile", "preference"]},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        print(f"Memories fetched: {len(results)}")
        for m in results:
            print(f"  [{m.get('metadata', {}).get('category', '?')}] {m.get('memory', '')[:80]}")
        return results


async def main(query: str, user_id: str | None, memory_url: str) -> None:
    print(f"Query: {query}")
    if user_id:
        print(f"User:  {user_id}")
    print("─" * 60)

    memories: list[dict] = []
    if user_id:
        try:
            memories = await fetch_memories(user_id, query, memory_url)
        except Exception as e:
            print(f"Warning: could not fetch memories ({e})")

    payload_text = json.dumps({
        "query": query,
        "user_id": user_id,
        "user_memories": memories,
    })

    async with httpx.AsyncClient(
        timeout=300.0,
        headers={"A2A-Version": "1.0"},
    ) as httpx_client:
        resp = await httpx_client.post(
            f"{BASE_URL}/",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": payload_text}],
                        "messageId": str(uuid.uuid4()),
                    }
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        print(f"RPC Error: {data['error']}")
        return

    task = data.get("result", {}).get("task", {})
    state = task.get("status", {}).get("state", "")
    print(f"State: {state}\n")

    artifacts = task.get("artifacts", [])
    for artifact in artifacts:
        for part in (artifact.get("parts") or []):
            text = part.get("text", "")
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                print("Response:", text)
                continue

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            status = result.get("status", "answered")

            if status in ("deferred", "deferred_final"):
                reason = result.get("deferral_reason") or {}
                if status == "deferred_final":
                    print("✗  FINAL DEFERRAL — could not answer even after clarification")
                else:
                    print("⚠  DEFERRED — insufficient context, clarification requested")
                print(f"\nMessage (for voice agent):\n  {reason.get('suggested_clarification', '')}")
                gaps = reason.get("gaps") or []
                if gaps:
                    print(f"\nMissing information:")
                    for g in gaps:
                        print(f"  • {g}")
                covered = reason.get("covered") or []
                if covered:
                    print(f"\nResolved so far:")
                    for c in covered:
                        print(f"  ✓ {c}")
                print(f"\nPartial answer (if any):\n{result.get('answer', '(none)')}")
            else:
                print(f"Answer:\n{result.get('answer', '(no answer)')}")
                print(f"\nConfidence:   {result.get('confidence', 0):.2f}")
                print(f"CRAG verdict: {result.get('crag_verdict', 'n/a')}")
                cited = result.get("cited_chunk_ids", [])
                if cited:
                    print(f"Cited chunks ({len(cited)}):")
                    for cid in cited:
                        print(f"  • {cid}")

    if not artifacts:
        msg = task.get("status", {}).get("message", {})
        for part in (msg.get("parts") or []):
            print("Status message:", part.get("text", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--user", default=None, help="user_id for memory lookup")
    parser.add_argument("--memory-url", default=MEMORY_SERVICE_URL)
    args = parser.parse_args()
    asyncio.run(main(args.query, args.user, args.memory_url))
