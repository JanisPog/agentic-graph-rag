#!/usr/bin/env bash
# Rebuild and redeploy the graph-rag-agent container.
# Usage: sudo bash ./redeploy.sh

set -euo pipefail

echo "==> Building graph-rag-agent..."
sudo docker compose build graph-rag-agent

echo "==> Restarting graph-rag-agent..."
sudo docker compose up -d graph-rag-agent

echo "==> Done. Tailing logs (Ctrl-C to stop)..."
sudo docker compose logs -f graph-rag-agent
