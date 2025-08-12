from __future__ import annotations

import json
import subprocess
import threading
import sys
from typing import Any, Dict


class MCPServerProcess:
    """Manage a single long-lived stdio MCP server subprocess.

    Requests are serialized under a lock to keep things simple and safe.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "app.mcp_stdio_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def call(self, name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        arguments = arguments or {}
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            req_id = self._next_id
            self._next_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server returned no data")
            return json.loads(line.strip())


_singleton: MCPServerProcess | None = None
_singleton_lock = threading.Lock()


def get_mcp_server() -> MCPServerProcess:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = MCPServerProcess()
    return _singleton


