"""
MCP-based Excel sampling utilities.

This module provides an optional parser path that reads only targeted XLSX ranges
through an MCP Excel server (instead of loading whole sheets with pandas).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from graph.models import CampaignBrief
from graph.tools import parse_campaign_sheet


class MCPExcelError(RuntimeError):
    """Raised when MCP Excel interaction fails."""


class MCPProgressReporter:
    """
    Tracks and prints MCP parsing progress.

    Designed so the same updates can later be forwarded to a frontend callback.
    """

    def __init__(self) -> None:
        self._last_percent = -1

    def update(self, percent: int, step: str) -> None:
        clamped = max(0, min(100, int(percent)))
        # Avoid duplicate prints if a stage is reported multiple times.
        if clamped == self._last_percent:
            return
        self._last_percent = clamped
        print(f"[MCP Sampling] {clamped:>3}% - {step}")


class MCPExcelClient:
    """Tiny stdio JSON-RPC client for MCP servers."""

    def __init__(self, command: List[str], cwd: Optional[str] = None):
        self.command = command
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 1

    def __enter__(self) -> "MCPExcelClient":
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
            raise MCPExcelError("MCP process stdin is not available")
        line = json.dumps(payload, ensure_ascii=True)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise MCPExcelError("MCP process stdout is not available")
        line = self.proc.stdout.readline()
        if not line:
            stderr_text = ""
            if self.proc.stderr:
                try:
                    stderr_text = self.proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise MCPExcelError(
                f"MCP server returned no response. Stderr: {stderr_text[:1000]}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPExcelError(f"Invalid MCP JSON response: {line[:500]}") from exc

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
            # Ignore notifications and unrelated responses.
            if "id" not in msg or msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise MCPExcelError(f"MCP error from {method}: {msg['error']}")
            return msg.get("result")

    def _notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        )

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "brief-tutor-mcp-sampler", "version": "0.1.0"},
            },
        )
        self._notification("notifications/initialized")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )

    def list_tools(self) -> List[str]:
        result = self._request("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        names: List[str] = []
        for item in tools:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return names


def _default_mcp_command() -> List[str]:
    # Windows-compatible default from excel-mcp-server docs.
    command = os.getenv("EXCEL_MCP_COMMAND")
    if command:
        return command.split(" ")
    return ["cmd", "/c", "npx", "--yes", "@negokaz/excel-mcp-server"]


def _extract_payload(result: Any) -> Any:
    if isinstance(result, dict):
        if "structuredContent" in result:
            return result["structuredContent"]
        if "content" in result and isinstance(result["content"], list):
            for item in result["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        text = item["text"].strip()
                        try:
                            return json.loads(text)
                        except Exception:
                            return text
                    if item.get("type") == "json":
                        return item.get("json")
        return result
    return result


def _normalize_to_table(payload: Any) -> List[List[Any]]:
    # Common shape: {"values": [[...], [...]]}
    if isinstance(payload, dict):
        values = payload.get("values")
        if isinstance(values, list):
            return values
        # Alternate shapes: {"rows": [...]}
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return payload
        if payload and isinstance(payload[0], dict):
            # Convert list-of-dicts into row-ordered values.
            keys = sorted({k for row in payload if isinstance(row, dict) for k in row.keys()})
            out = [keys]
            for row in payload:
                out.append([row.get(k) for k in keys])
            return out

    if isinstance(payload, str):
        html_match = re.search(r"<table[\s\S]*?</table>", payload, flags=re.IGNORECASE)
        if html_match:
            try:
                html_table = html_match.group(0)
                parsed_tables = pd.read_html(StringIO(html_table))
                if parsed_tables:
                    df = parsed_tables[0]
                    # Drop leading row-number/index column often present in HTML output.
                    if df.shape[1] > 1:
                        first_col = df.iloc[:, 0]
                        numeric_ratio = pd.to_numeric(first_col, errors="coerce").notna().mean()
                        if numeric_ratio >= 0.8:
                            df = df.iloc[:, 1:]
                    return df.where(pd.notna(df), "").values.tolist()
            except Exception:
                # Fallback to text parsing below.
                pass

        lines = [ln for ln in payload.splitlines() if ln.strip()]
        return [ln.split("\t") for ln in lines]

    return []


def _read_sheet(
    client: MCPExcelClient,
    file_absolute_path: str,
    sheet_name: str,
) -> tuple[pd.DataFrame, Any]:
    result = client.call_tool(
        "excel_read_sheet",
        {
            "fileAbsolutePath": file_absolute_path,
            "sheetName": sheet_name,
        },
    )
    payload = _extract_payload(result)
    table = _normalize_to_table(payload)
    if not table:
        raise MCPExcelError(
            f"excel_read_sheet returned no tabular data for sheet '{sheet_name}'"
        )
    return pd.DataFrame(table), payload


def _sheet_payload_to_json(payload: Any) -> Dict[str, Any]:
    """
    Normalize sheet payload into a stable JSON shape for downstream consumers.
    """
    normalized_rows = _normalize_to_table(payload)
    return {
        "rows": normalized_rows,
        "rowCount": len(normalized_rows),
        "columnCount": max((len(r) for r in normalized_rows), default=0),
    }


def _extract_sheet_names(describe_payload: Any) -> List[str]:
    """
    Extract all sheet names from excel_describe_sheets payload.
    """
    names: List[str] = []

    if isinstance(describe_payload, dict):
        for key in ("sheets", "sheetInfos", "sheet_info", "sheetInfo"):
            value = describe_payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        names.append(item["name"])
                    elif isinstance(item, str):
                        names.append(item)
        if not names:
            for value in describe_payload.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                            names.append(item["name"])
    elif isinstance(describe_payload, list):
        for item in describe_payload:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
            elif isinstance(item, str):
                names.append(item)

    # De-duplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def parse_spreadsheet_via_mcp_sampling(spreadsheet_path: str) -> Dict[str, Any]:
    """
    Parse brief spreadsheet through MCP:
    1) describe all workbook sheets
    2) read first sheet completely
    3) return structured JSON for CampaignBrief/Campaigns creation
    """
    progress = MCPProgressReporter()
    progress.update(0, "Starting MCP sampling workflow")

    file_path = str(Path(spreadsheet_path).resolve())
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

    progress.update(8, "Launching MCP Excel server")
    with MCPExcelClient(command=_default_mcp_command(), cwd=str(Path(file_path).parent)) as client:
        progress.update(12, "Validating required MCP Excel tools")
        available_tools = set(client.list_tools())
        required_tools = {"excel_describe_sheets", "excel_read_sheet"}
        missing_tools = sorted(required_tools - available_tools)
        if missing_tools:
            raise MCPExcelError(
                "Required MCP tools are missing: "
                + ", ".join(missing_tools)
                + f". Available tools: {sorted(available_tools)}"
            )

        # Validate sheet availability first.
        progress.update(15, "Describing workbook sheets")
        describe_result = client.call_tool(
            "excel_describe_sheets", {"fileAbsolutePath": file_path}
        )
        describe_payload = _extract_payload(describe_result)
        sheet_names = _extract_sheet_names(describe_payload)
        if not sheet_names:
            raise MCPExcelError("No sheets were returned by excel_describe_sheets")

        first_sheet_name = sheet_names[0]
        progress.update(30, f"Reading first sheet completely: {first_sheet_name}")
        df1, sheet1_payload = _read_sheet(
            client, file_path, first_sheet_name
        )
        progress.update(45, f"Parsing first sheet data: {first_sheet_name}")
        meta_results, campaigns_results = parse_campaign_sheet(df1, first_sheet_name)

        progress.update(62, "Building campaign brief model")
        campaign_brief = CampaignBrief(
            spreadsheet_path=spreadsheet_path,
            task_type=meta_results.get("task_type") or "",
            asset_summary=meta_results.get("asset_summary"),
            dealership_name=meta_results.get("dealership_name"),
            content_11_20=meta_results.get("content_11_20", False),
            campaigns=campaigns_results,
        )

        sheet2_payload: Any = None
        if campaign_brief.content_11_20 and "CampaignContent_11_20" in sheet_names:
            progress.update(75, "Reading CampaignContent_11_20 via excel_read_sheet")
            df2, sheet2_payload = _read_sheet(
                client, file_path, "CampaignContent_11_20"
            )
            progress.update(88, "Parsing CampaignContent_11_20 data")
            _, campaigns2 = parse_campaign_sheet(df2, "CampaignContent_11_20")
            campaign_brief.campaigns.extend(campaigns2)

    progress.update(95, "Serializing campaign brief result")

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="python")
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    result = _serialize(campaign_brief)
    result["mcp_spreadsheet_info"] = {
        "fileAbsolutePath": file_path,
        "sheetInfo": describe_payload,  # Full excel_describe_sheets response
        "sheetNames": sheet_names,  # All workbook sheets
        "firstSheetName": first_sheet_name,  # Sheet read first and completely
        "sheetDataJson": {
            first_sheet_name: _sheet_payload_to_json(sheet1_payload),
            "CampaignContent_11_20": _sheet_payload_to_json(sheet2_payload),
        },
    }
    # Explicit structure for downstream creation of CampaignBrief/Campaigns.
    result["campaign_brief_input_json"] = {
        "spreadsheet_path": result.get("spreadsheet_path"),
        "task_type": result.get("task_type"),
        "asset_summary": result.get("asset_summary"),
        "dealership_name": result.get("dealership_name"),
        "content_11_20": result.get("content_11_20"),
        "campaigns": result.get("campaigns", []),
    }
    progress.update(100, "MCP sampling workflow complete")
    return result
