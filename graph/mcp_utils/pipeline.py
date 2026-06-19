"""
Custom MCP orchestration pipeline for spreadsheet parsing.
"""
from __future__ import annotations

import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph.mcp_utils.client import CustomBriefMCPClient, CustomBriefMCPError
from graph.mcp_utils.contracts import validate_campaign_brief_payload


def _extract_payload(result: Any) -> Any:
    if isinstance(result, dict) and "structuredContent" in result:
        return result["structuredContent"]
    return result


def _custom_mcp_command() -> Optional[List[str]]:
    command = os.getenv("CUSTOM_BRIEF_MCP_COMMAND", "").strip()
    if not command:
        return None
    return shlex.split(command, posix=False)


def parse_spreadsheet_via_custom_mcp(spreadsheet_path: str) -> Dict[str, Any]:
    """
    3-step custom MCP flow:
    1) retrieve workbook data
    2) extract important values
    3) curate into CampaignBrief JSON
    """
    file_path = str(Path(spreadsheet_path).resolve())
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

    started_at = time.time()
    with CustomBriefMCPClient(command=_custom_mcp_command()) as client:
        workbook_result = client.call_tool(
            "brief_read_workbook", {"spreadsheet_path": file_path}
        )
        workbook_payload = _extract_payload(workbook_result)

        extracted_result = client.call_tool(
            "brief_extract_values", {"workbook_payload": workbook_payload}
        )
        extracted_payload = _extract_payload(extracted_result)

        curated_result = client.call_tool(
            "brief_curate_campaign_brief", {"extracted_values": extracted_payload}
        )
        curated_payload = _extract_payload(curated_result)

    payload = validate_campaign_brief_payload(curated_payload)
    debug_block = payload.get("mcp_debug") if isinstance(payload, dict) else None
    if not isinstance(debug_block, dict):
        debug_block = {}
    debug_block["custom_mcp_elapsed_ms"] = int((time.time() - started_at) * 1000)
    payload["mcp_debug"] = debug_block
    payload["campaign_brief_input_json"] = {
        "spreadsheet_path": payload.get("spreadsheet_path"),
        "task_type": payload.get("task_type"),
        "asset_summary": payload.get("asset_summary"),
        "dealership_name": payload.get("dealership_name"),
        "content_11_20": payload.get("content_11_20"),
        "campaigns": payload.get("campaigns", []),
    }
    return payload


__all__ = ["parse_spreadsheet_via_custom_mcp", "CustomBriefMCPError"]
