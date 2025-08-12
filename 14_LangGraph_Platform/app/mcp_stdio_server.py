from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def handle_request(request: dict) -> dict:
    method = request.get("method")
    req_id = request.get("id")
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the input payload",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "time",
                        "description": "Return current UTC time ISO8601",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }
    if method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "echo":
            text = arguments.get("text", "")
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": str(text)}}
        if name == "time":
            now = datetime.now(timezone.utc).isoformat()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": now}}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Unknown tool"}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": "pong"}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}


def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            request = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        response = handle_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()


