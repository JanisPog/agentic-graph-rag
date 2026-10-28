#!/usr/bin/env bash
# Send ticket_id=6 to the /ingest-ticket endpoint and print the response.
# Usage: ./test_ingest_ticket.sh [host] [port]
#   host defaults to localhost
#   port defaults to 8015

HOST=${1:-localhost}
PORT=${2:-8015}
URL="http://${HOST}:${PORT}/ingest-ticket"

echo "POST ${URL}"
echo '{"ticket_id": "6"}'
echo "---"

curl -s -w "\nHTTP %{http_code}\n" \
  -X POST "${URL}" \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "6"}'
