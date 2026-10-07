"""Load and persist campaign briefs through Supabase."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from graph.models import Assets, Campaign, CampaignBrief, OfferDetails, StyleDescriptions
from graph.supabase import is_supabase_configured
from graph.supabase.repository import SupabaseWorkflowRepository, parse_brief_identifiers


class SupabaseUnavailable(RuntimeError):
    """Supabase is configured, but the workflow could not consult it."""


def file_sha256(path: str) -> Optional[str]:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_repository() -> Optional[SupabaseWorkflowRepository]:
    """Return a repository, or None when Supabase is not configured.

    A configured integration that cannot be reached raises SupabaseUnavailable.
    """
    if not is_supabase_configured(use_service_role=True):
        return None
    try:
        import supabase  # noqa: F401
    except Exception as exc:
        raise SupabaseUnavailable(
            "Supabase is configured but the client could not be loaded. "
            f"{exc}"
        ) from exc
    try:
        return SupabaseWorkflowRepository(use_service_role=True)
    except Exception as exc:
        raise SupabaseUnavailable(f"Supabase could not be consulted: {exc}") from exc


def _brief_id_for(path: str) -> str:
    ids = parse_brief_identifiers(path)
    return str(ids.get("brief_id") or ids.get("instance_id") or Path(path).stem)


def _campaign_from_row(row: Dict[str, Any]) -> Campaign:
    style = row.get("style_descriptions") if isinstance(row.get("style_descriptions"), dict) else {}
    assets = row.get("assets") if isinstance(row.get("assets"), dict) else {}
    return Campaign(
        campaign_id=str(row.get("campaign_external_id") or ""),
        offer_details=OfferDetails(
            headline=str(row.get("headline") or ""),
            offer=str(row.get("offer") or ""),
            body=str(row.get("body") or ""),
            cta=str(row.get("cta") or ""),
            disclaimer=str(row.get("disclaimer") or ""),
        ),
        style_descriptions=StyleDescriptions(
            asset_style_direction=str(style.get("asset_style_direction") or ""),
            additional_style_information=str(style.get("additional_style_information") or ""),
            vehicle_photography=str(style.get("vehicle_photography") or ""),
            logos=str(style.get("logos") or ""),
        ),
        assets=Assets(
            sl_bn_srp_da=str(assets.get("sl_bn_srp_da") or ""),
            sl_m_bn_m=str(assets.get("sl_m_bn_m") or ""),
            facebook_assets=str(assets.get("facebook_assets") or ""),
            instagram_assets=str(assets.get("instagram_assets") or ""),
            google_assets=str(assets.get("google_assets") or ""),
            ot_1=str(assets.get("ot_1") or ""),
            ot_2=str(assets.get("ot_2") or ""),
            ot_3=str(assets.get("ot_3") or ""),
            ot_4=str(assets.get("ot_4") or ""),
            ot_5=str(assets.get("ot_5") or ""),
            ot_6=str(assets.get("ot_6") or ""),
        ),
    )


def load_campaign_brief(spreadsheet_path: str) -> Optional[CampaignBrief]:
    """Rebuild a brief from Supabase when the stored file hash still matches."""
    repo = open_repository()
    if repo is None:
        return None
    try:
        row = repo.get_brief_by_brief_id(_brief_id_for(spreadsheet_path))
    except Exception as exc:
        raise SupabaseUnavailable(f"Supabase could not be consulted: {exc}") from exc
    if not isinstance(row, dict):
        return None
    current_hash = file_sha256(spreadsheet_path)
    saved_hash = str(row.get("source_file_hash") or "")
    if not current_hash or not saved_hash or saved_hash != current_hash:
        return None
    brief_fk = row.get("id")
    if not isinstance(brief_fk, int):
        return None
    try:
        campaign_rows = repo.list_campaigns_by_brief_fk(brief_fk)
    except Exception as exc:
        raise SupabaseUnavailable(f"Supabase could not be consulted: {exc}") from exc
    if not campaign_rows:
        return None
    return CampaignBrief(
        spreadsheet_path=str(row.get("source_file_path") or spreadsheet_path),
        task_type=str(row.get("task_type") or ""),
        asset_summary=None,
        dealership_name=row.get("dealership_name"),
        content_11_20=False,
        campaigns=[_campaign_from_row(item) for item in campaign_rows if isinstance(item, dict)],
    )
