"""A2A server entrypoint for the Graph-RAG pipeline.

Serves the LangGraph pipeline as an A2A-compliant JSON-RPC 2.0 agent.

Usage:
    cd Prototype_v1.0
    py a2a/main.py

The SDK automatically exposes:
    POST /                            – A2A JSON-RPC task ingress
    GET  /.well-known/agent-card.json – Agent discovery
    GET  /health                      – Health check (custom route below)
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure Prototype_v1.0 root is on sys.path so graph_builder etc. are importable
_PROTO_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from executor import GraphRagExecutor
from ingest_endpoint import ingest_ticket_endpoint

# Load .env from Prototype_v1.0 root (Neo4j / Ollama / Gemini credentials)
load_dotenv(_PROTO_ROOT / ".env")

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("graph-rag-agent")

_HOST = os.getenv("A2A_HOST", "0.0.0.0")
_PORT = int(os.getenv("A2A_PORT", "8015"))

_CARD_PATH = Path(__file__).parent / ".well-known" / "agent-card.json"


def load_agent_card() -> AgentCard:
    data = json.loads(_CARD_PATH.read_text(encoding="utf-8"))
    return AgentCard(**data)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "1.0.0"})


def build_app() -> Starlette:
    agent_card = load_agent_card()

    handler = DefaultRequestHandler(
        agent_executor=GraphRagExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = (
        create_agent_card_routes(agent_card=agent_card)
        + create_jsonrpc_routes(request_handler=handler, rpc_url="/")
        + [
            Route("/health", endpoint=health),
            Route("/ingest-ticket", endpoint=ingest_ticket_endpoint, methods=["POST"]),
        ]
    )

    return Starlette(routes=routes)


if __name__ == "__main__":
    logger.info("Graph-RAG A2A agent starting on %s:%d …", _HOST, _PORT)
    uvicorn.run(build_app(), host=_HOST, port=_PORT)
