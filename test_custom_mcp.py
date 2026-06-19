"""
Quick smoke test for the custom brief MCP pipeline.

Usage:
  python test_custom_mcp.py "C:/path/to/file.xlsx"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from graph.mcp_utils.pipeline import parse_spreadsheet_via_custom_mcp


def _campaign_preview(campaign: dict) -> dict:
    offer = campaign.get("offer_details", {}) if isinstance(campaign, dict) else {}
    assets = campaign.get("assets", {}) if isinstance(campaign, dict) else {}
    return {
        "campaign_id": campaign.get("campaign_id", ""),
        "headline": offer.get("headline", ""),
        "offer": offer.get("offer", ""),
        "cta": offer.get("cta", ""),
        "asset_flags": {
            "sl_bn_srp_da": bool(str(assets.get("sl_bn_srp_da", "")).strip()),
            "facebook_assets": bool(str(assets.get("facebook_assets", "")).strip()),
            "instagram_assets": bool(str(assets.get("instagram_assets", "")).strip()),
            "google_assets": bool(str(assets.get("google_assets", "")).strip()),
        },
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python test_custom_mcp.py \"C:/path/to/file.xlsx\"")
        return 1

    spreadsheet_path = str(Path(sys.argv[1]).resolve())
    if not Path(spreadsheet_path).exists():
        print(f"File not found: {spreadsheet_path}")
        return 1

    result = parse_spreadsheet_via_custom_mcp(spreadsheet_path)
    campaigns = result.get("campaigns", [])
    summary = {
        "spreadsheet_path": result.get("spreadsheet_path"),
        "task_type": result.get("task_type"),
        "dealership_name": result.get("dealership_name"),
        "campaign_count": len(campaigns),
        "campaigns_preview": [
            _campaign_preview(c) for c in campaigns[:20] if isinstance(c, dict)
        ],
        "mcp_debug": result.get("mcp_debug", {}),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
