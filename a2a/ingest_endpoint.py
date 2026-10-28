"""POST /ingest-ticket — receives a ticket_id, fetches the ticket via MCP get_ticket,
and runs it through the ETL pipeline into Neo4j.

Environment variables (added to Prototype_v1.0/.env):
  TICKET_MCP_URL    URL of the real ticket system MCP server
                    e.g. http://host.docker.internal:8040/mcp/
  TICKET_MCP_TOKEN  Bearer token for the MCP server
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

# Ensure Prototype_v1.0 root is importable
_PROTO_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from etl.ticket_ingest import ingest_ticket  # noqa: E402

logger = logging.getLogger("graph-rag-agent.ingest")


async def _fetch_ticket_via_mcp(ticket_id: str) -> dict:
    """Call get_ticket on the real ticket system MCP server.

    Returns the raw ticket dict from the MCP tool response.
    Raises RuntimeError when the MCP call fails.
    """
    from mcp.client.session import ClientSession  # noqa: PLC0415
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

    mcp_url   = os.environ["TICKET_MCP_URL"]
    mcp_token = os.environ.get("TICKET_MCP_TOKEN", "")

    headers = {"Authorization": f"Bearer {mcp_token}"} if mcp_token else {}

    # The real ticket system expects an integer ID.
    # If the canonical ID is a UUID (fallback when real system was unavailable),
    # the MCP call will fail — log clearly rather than crash cryptically.
    try:
        int_id = int(ticket_id)
    except ValueError:
        raise RuntimeError(
            f"ticket_id={ticket_id!r} is not an integer — "
            "this ID was created without the real ticket system and cannot be fetched via MCP"
        )

    logger.debug("MCP get_ticket: url=%s ticket_id=%s", mcp_url, int_id)

    async with streamablehttp_client(url=mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_ticket", {"ticket_id": int_id})
            if result.isError:
                raise RuntimeError(f"MCP get_ticket returned error: {result.content}")
            return json.loads(result.content[0].text)


async def ingest_ticket_endpoint(request: Request) -> JSONResponse:
    """POST /ingest-ticket

    Body: {"ticket_id": "<canonical_ticket_id>"}

    Fetches the ticket from the real ticket system via MCP get_ticket, then
    runs it through the ETL pipeline to add it to the Neo4j knowledge graph.
    Returns 202 immediately — ETL runs in a thread executor.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    ticket_id = str(body.get("ticket_id", "")).strip()
    if not ticket_id:
        return JSONResponse({"error": "missing 'ticket_id'"}, status_code=422)

    if not os.environ.get("TICKET_MCP_URL"):
        logger.error("TICKET_MCP_URL not set — cannot fetch ticket %s", ticket_id)
        return JSONResponse(
            {"error": "TICKET_MCP_URL not configured on this agent"},
            status_code=503,
        )

    logger.info("ingest-ticket: fetching ticket_id=%s via MCP", ticket_id)

    async def _run() -> None:
        try:
            raw = await _fetch_ticket_via_mcp(ticket_id)
            # Ensure the id field is present with the canonical id
            raw.setdefault("id", ticket_id)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ingest_ticket, raw)
        except Exception as exc:
            logger.error(
                "ingest-ticket failed for ticket_id=%s: %s\n%s",
                ticket_id, exc, traceback.format_exc(),
            )

    asyncio.create_task(_run())

    return JSONResponse({"status": "accepted", "ticket_id": ticket_id}, status_code=202)
