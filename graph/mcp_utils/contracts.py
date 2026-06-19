"""
Contracts for custom brief MCP payloads.
"""
from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_BRIEF_KEYS = [
    "spreadsheet_path",
    "task_type",
    "asset_summary",
    "dealership_name",
    "content_11_20",
    "campaigns",
]


def validate_campaign_brief_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the curated payload shape expected by `CampaignBrief`.
    Raises ValueError on invalid payload.
    """
    if not isinstance(payload, dict):
        raise ValueError("Curated brief payload must be a JSON object")

    missing = [key for key in REQUIRED_BRIEF_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Curated brief payload missing required keys: {missing}")

    if not isinstance(payload.get("campaigns"), list):
        raise ValueError("Curated brief payload field 'campaigns' must be a list")

    return payload


def normalize_sheet_rows(rows: Any) -> List[List[Any]]:
    """Ensure rows are represented as a list-of-lists."""
    if not isinstance(rows, list):
        return []
    normalized: List[List[Any]] = []
    for row in rows:
        if isinstance(row, list):
            normalized.append(row)
        elif isinstance(row, tuple):
            normalized.append(list(row))
        else:
            normalized.append([row])
    return normalized
