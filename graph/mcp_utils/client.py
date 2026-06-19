"""
Small stdio JSON-RPC client for the in-repo custom brief MCP server.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class CustomBriefMCPError(RuntimeError):
    """Raised when custom brief MCP interactions fail."""


class CustomBriefMCPClient:
    """Tiny JSON-RPC client for the custom brief MCP server."""

    def __init__(self, command: Optional[List[str]] = None, cwd: Optional[str] = None):
        self.command = command or [sys.executable, "-m", "graph.mcp_utils.server"]
        self.cwd = cwd or str(Path(__file__).resolve().parents[2])
        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 1

    def __enter__(self) -> "CustomBriefMCPClient":
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=self.cwd,
            bufsize=1,
        )
        self._initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise CustomBriefMCPError("MCP process stdin is not available")
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise CustomBriefMCPError("MCP process stdout is not available")
        line = self.proc.stdout.readline()
        if not line:
            stderr_text = ""
            if self.proc.stderr:
                try:
                    stderr_text = self.proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise CustomBriefMCPError(
                f"MCP server returned no response. Stderr: {stderr_text[:1000]}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise CustomBriefMCPError(f"Invalid MCP JSON response: {line[:500]}") from exc

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )
        while True:
            msg = self._recv()
            if "id" not in msg or msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise CustomBriefMCPError(f"MCP error from {method}: {msg['error']}")
            return msg.get("result")

    def _notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "brief-tutor-custom-mcp-client", "version": "0.1.0"},
            },
        )
        self._notification("notifications/initialized")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self._request("tools/call", {"name": name, "arguments": arguments})

