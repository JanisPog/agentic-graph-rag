#!/usr/bin/env bash
# Send a single query to a running A2A agent and print the answer.
#
# Usage:
#   ./query.sh "How do I delete a payment method and set the currency for a course?"
#
# Point AGENT_URL at your deployment if it is not running locally:
#   AGENT_URL=http://my-host:8015 ./query.sh "..."

QUERY="${*:-How do I delete a payment method and set the currency for a course?}"
URL="${AGENT_URL:-http://localhost:8015}"

python3 - "$URL" "$QUERY" <<'EOF'
import sys, json, uuid, urllib.request, urllib.error

url, query = sys.argv[1], sys.argv[2]

payload = json.dumps({
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
        "message": {
            "role": 1,
            "parts": [{"text": query}],
            "message_id": str(uuid.uuid4()),
        }
    }
}).encode()

req = urllib.request.Request(
    url + "/",
    data=payload,
    headers={"Content-Type": "application/json", "A2A-Version": "1.0"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print("HTTP error:", e.code, e.read().decode())
    sys.exit(1)
except Exception as e:
    print("Request failed:", e)
    sys.exit(1)

# Check for JSON-RPC error
if "error" in data:
    print("Server error:", data["error"])
    sys.exit(1)

# The A2A SDK wraps the result in a Task. Navigate:
# data.result.task.artifacts[].parts[].text
# or
# data.result.artifacts[].parts[].text  (if message is returned directly)
def find_artifact_text(obj):
    """Walk the result tree to find the first artifact text part."""
    if isinstance(obj, dict):
        # Try task wrapper first, then direct artifacts
        for key in ("task", "message"):
            if key in obj:
                result = find_artifact_text(obj[key])
                if result:
                    return result
        if "artifacts" in obj:
            for art in obj["artifacts"]:
                for part in art.get("parts", []):
                    if "text" in part:
                        return part["text"]
    return None

result = data.get("result", {})
raw = find_artifact_text(result)

if raw is None:
    print("Unexpected response shape — full response:")
    print(json.dumps(data, indent=2))
    sys.exit(1)

try:
    d = json.loads(raw)
except json.JSONDecodeError:
    print(raw)
    sys.exit(0)

if "error" in d:
    print("Pipeline error:", d["error"])
    sys.exit(1)

print()
print("Answer:")
print(d.get("answer", "(no answer)"))
print()
print(f"Confidence:   {d.get('confidence', 0):.2f}")
print(f"CRAG verdict: {d.get('crag_verdict', 'n/a')}")
cited = d.get("cited_chunk_ids", [])
if cited:
    print(f"Cited chunks ({len(cited)}):")
    for c in cited:
        print(f"  • {c}")
EOF
