"""
In-repo custom MCP server for spreadsheet brief ingestion.

Tools:
- brief_read_workbook
- brief_extract_values
- brief_curate_campaign_brief
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from graph.mcp_utils.contracts import normalize_sheet_rows, validate_campaign_brief_payload


def _as_json_line(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
    sys.stdout.flush()


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _json_safe(obj: Any) -> Any:
    """Recursively normalize objects so MCP responses are JSON serializable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        # numpy scalar support (np.int64, np.float64, etc.)
        try:
            return obj.item()
        except Exception:
            pass
    if pd.isna(obj):
        return ""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _rows_to_dataframe(rows: Any) -> pd.DataFrame:
    return pd.DataFrame(normalize_sheet_rows(rows))


def brief_read_workbook(spreadsheet_path: str) -> Dict[str, Any]:
    file_path = str(Path(spreadsheet_path).resolve())
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

    xls = pd.ExcelFile(file_path)
    workbook: Dict[str, Any] = {
        "spreadsheet_path": spreadsheet_path,
        "resolved_path": file_path,
        "sheet_names": list(xls.sheet_names),
        "sheets": {},
    }
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name, header=None)
        rows = _json_safe(df.where(pd.notna(df), "").values.tolist())
        workbook["sheets"][sheet_name] = {
            "rows": rows,
            "rowCount": len(rows),
            "columnCount": max((len(r) for r in rows), default=0),
        }
    return workbook


def brief_extract_values(workbook_payload: Dict[str, Any]) -> Dict[str, Any]:
    from graph.tools import parse_campaign_sheet  # Local import to avoid startup cycles.

    sheets = workbook_payload.get("sheets", {}) if isinstance(workbook_payload, dict) else {}
    if not isinstance(sheets, dict) or not sheets:
        raise ValueError("workbook_payload.sheets is empty or invalid")

    first_sheet_name = "CampaignContent_1_10"
    if first_sheet_name not in sheets:
        first_sheet_name = next(iter(sheets.keys()))

    first_rows = sheets.get(first_sheet_name, {}).get("rows", [])
    df1 = _rows_to_dataframe(first_rows)
    meta_results, campaigns_results = parse_campaign_sheet(df1, first_sheet_name)

    content_11_20 = bool(meta_results.get("content_11_20", False))
    if content_11_20 and "CampaignContent_11_20" in sheets:
        second_rows = sheets["CampaignContent_11_20"].get("rows", [])
        df2 = _rows_to_dataframe(second_rows)
        _, campaigns2 = parse_campaign_sheet(df2, "CampaignContent_11_20")
        campaigns_results.extend(campaigns2)

    return {
        "spreadsheet_path": workbook_payload.get("spreadsheet_path", ""),
        "task_type": meta_results.get("task_type") or "",
        "asset_summary": meta_results.get("asset_summary"),
        "dealership_name": meta_results.get("dealership_name"),
        "content_11_20": content_11_20,
        "campaigns": _serialize(campaigns_results),
        "mcp_debug": {
            "sheet_names": workbook_payload.get("sheet_names", []),
            "first_sheet_name": first_sheet_name,
        },
    }


def brief_curate_campaign_brief(extracted_values: Dict[str, Any]) -> Dict[str, Any]:
    from graph.models import CampaignBrief

    campaign_brief = CampaignBrief(
        spreadsheet_path=extracted_values.get("spreadsheet_path", ""),
        task_type=extracted_values.get("task_type") or "",
        asset_summary=extracted_values.get("asset_summary"),
        dealership_name=extracted_values.get("dealership_name"),
        content_11_20=bool(extracted_values.get("content_11_20", False)),
        campaigns=extracted_values.get("campaigns", []),
    )
    curated = _serialize(campaign_brief)
    if extracted_values.get("mcp_debug"):
        curated["mcp_debug"] = extracted_values["mcp_debug"]
    return validate_campaign_brief_payload(curated)


TOOLS = {
    "brief_read_workbook": brief_read_workbook,
    "brief_extract_values": brief_extract_values,
    "brief_curate_campaign_brief": brief_curate_campaign_brief,
}


def _handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "brief-tutor-custom-mcp", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        tools = [
            {
                "name": "brief_read_workbook",
                "description": "Read all workbook sheets and return normalized JSON rows.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"spreadsheet_path": {"type": "string"}},
                    "required": ["spreadsheet_path"],
                },
            },
            {
                "name": "brief_extract_values",
                "description": "Extract campaign brief values and campaigns from workbook payload.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"workbook_payload": {"type": "object"}},
                    "required": ["workbook_payload"],
                },
            },
            {
                "name": "brief_curate_campaign_brief",
                "description": "Curate extracted values into CampaignBrief JSON contract.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"extracted_values": {"type": "object"}},
                    "required": ["extracted_values"],
                },
            },
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name not in TOOLS:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")

        if name == "brief_read_workbook":
            result = TOOLS[name](spreadsheet_path=arguments.get("spreadsheet_path", ""))
        elif name == "brief_extract_values":
            result = TOOLS[name](workbook_payload=arguments.get("workbook_payload", {}))
        else:
            result = TOOLS[name](extracted_values=arguments.get("extracted_values", {}))

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"structuredContent": _json_safe(result)},
        }

    # Notification-like methods get empty success if id is present.
    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if "id" not in request:
                # Notification: no response expected.
                continue
            response = _handle_request(request)
        except Exception as exc:
            request_id = None
            try:
                request_id = request.get("id")  # type: ignore[name-defined]
            except Exception:
                request_id = None
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }
        _as_json_line(response)


if __name__ == "__main__":
    main()
