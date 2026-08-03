"""
Supabase repository layer for workflow persistence.

This module keeps table access logic in one place so workflow nodes can call
high-level methods instead of building ad-hoc DB payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from graph.models import CampaignBrief, Campaign
from graph.brief_naming import parse_brief_filename
from graph.supabase.client import get_supabase_client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_similarity_score(score: float) -> float:
    """
    Accept score in [0,1] or [0,100], return [0,100] rounded to 2 decimals.
    """
    if score <= 1.0:
        return round(float(score) * 100.0, 2)
    return round(float(score), 2)


def _grade_from_score(score_0_100: float, strong_threshold: float, review_threshold: float) -> str:
    if score_0_100 >= strong_threshold:
        return "strong"
    if score_0_100 >= review_threshold:
        return "human_check"
    return "weak"


def parse_brief_identifiers(spreadsheet_path: str) -> Dict[str, Optional[str]]:
    """
    Parse common file naming pattern:
    YYYY-MM-<family_slug>-A-<numeric_id>.xlsx
    """
    file_name = Path(spreadsheet_path or "").name
    stem = Path(file_name).stem if file_name else ""
    parsed = parse_brief_filename(file_name)
    if not parsed:
        return {
            "brief_id": stem or None,
            "dealership_family_id": None,
            "instance_id": None,
            "source_file_name": file_name or None,
        }
    return {
        "brief_id": parsed.get("brief_id"),
        "dealership_family_id": parsed.get("account_id"),
        "instance_id": parsed.get("campaign_token"),
        "source_file_name": parsed.get("source_file_name") or file_name,
    }


@dataclass
class SupabaseWorkflowRepository:
    """
    Repository for persisting campaign workflow entities into Supabase tables.
    """

    use_service_role: bool = True

    def __post_init__(self) -> None:
        self.client = get_supabase_client(use_service_role=self.use_service_role)

    # ---------- briefs ----------
    def upsert_brief(
        self,
        campaign_brief: CampaignBrief,
        *,
        source_file_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        ids = parse_brief_identifiers(campaign_brief.spreadsheet_path)
        payload = {
            "brief_id": ids.get("brief_id") or Path(campaign_brief.spreadsheet_path).stem,
            "dealership_family_id": ids.get("dealership_family_id"),
            "dealership_name": campaign_brief.dealership_name,
            "task_type": campaign_brief.task_type,
            "source_file_name": ids.get("source_file_name") or Path(campaign_brief.spreadsheet_path).name,
            "source_file_path": campaign_brief.spreadsheet_path,
            "source_file_hash": source_file_hash,
            "updated_at": _now_iso(),
        }
        response = (
            self.client.table("briefs")
            .upsert(payload, on_conflict="brief_id")
            .execute()
        )
        rows = getattr(response, "data", None) or []
        if not rows:
            raise RuntimeError("Supabase upsert_brief returned no row.")
        return rows[0]

    def get_brief_by_brief_id(self, brief_id: str) -> Optional[Dict[str, Any]]:
        response = (
            self.client.table("briefs")
            .select("*")
            .eq("brief_id", brief_id)
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return rows[0] if rows else None

    # ---------- dealership metadata ----------
    def get_dealership_account_metadata(self, account_id: str) -> Optional[Dict[str, Any]]:
        normalized_account = str(account_id or "").strip().lower()
        if not normalized_account:
            return None

        account_resp = (
            self.client.table("dealership_accounts")
            .select("*")
            .eq("account_id", normalized_account)
            .limit(1)
            .execute()
        )
        account_rows = getattr(account_resp, "data", None) or []
        if not account_rows:
            return None

        account_row = dict(account_rows[0])
        account_pk = account_row.get("id")
        oem_rows: List[Dict[str, Any]] = []
        if isinstance(account_pk, int):
            oem_resp = (
                self.client.table("dealership_account_oems")
                .select("oem")
                .eq("account_fk", account_pk)
                .execute()
            )
            oem_rows = list(getattr(oem_resp, "data", None) or [])

        group_info = None
        group_fk = account_row.get("group_fk")
        if isinstance(group_fk, int):
            group_resp = (
                self.client.table("dealership_groups")
                .select("*")
                .eq("id", group_fk)
                .limit(1)
                .execute()
            )
            group_rows = getattr(group_resp, "data", None) or []
            group_info = dict(group_rows[0]) if group_rows else None

        oems = [str(item.get("oem", "")).strip() for item in oem_rows if str(item.get("oem", "")).strip()]
        return {
            "account_id": normalized_account,
            "group_fk": group_fk,
            "group": group_info,
            "oem_family": str(account_row.get("oem_family", "NA") or "NA").strip(),
            "handles_all_oems": bool(account_row.get("handles_all_oems", False)),
            "oems": oems,
        }

    @staticmethod
    def oem_compatible(
        source_account: Optional[Dict[str, Any]],
        candidate_account: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Compatibility rule for hard OEM filter.
        """
        if not source_account or not candidate_account:
            return False

        source_all = bool(source_account.get("handles_all_oems", False))
        candidate_all = bool(candidate_account.get("handles_all_oems", False))
        if source_all or candidate_all:
            return True

        source_oems = {
            str(item).strip().lower()
            for item in (source_account.get("oems", []) or [])
            if str(item).strip()
        }
        candidate_oems = {
            str(item).strip().lower()
            for item in (candidate_account.get("oems", []) or [])
            if str(item).strip()
        }
        if "all" in source_oems or "all" in candidate_oems:
            return True
        if source_oems and candidate_oems and source_oems.intersection(candidate_oems):
            return True

        source_family = str(source_account.get("oem_family", "NA") or "NA").strip().lower()
        candidate_family = str(candidate_account.get("oem_family", "NA") or "NA").strip().lower()
        if source_family != "na" and candidate_family != "na" and source_family == candidate_family:
            return True
        return False

    # ---------- campaigns ----------
    def upsert_campaigns(
        self,
        brief_fk: int,
        campaign_brief: CampaignBrief,
        *,
        dealership_family_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ids = parse_brief_identifiers(campaign_brief.spreadsheet_path)
        family_id = dealership_family_id or ids.get("dealership_family_id")
        payloads: List[Dict[str, Any]] = []
        for campaign in campaign_brief.campaigns:
            payloads.append(self._campaign_payload(brief_fk, campaign, campaign_brief.task_type, family_id))

        if not payloads:
            return []

        response = (
            self.client.table("campaigns")
            .upsert(payloads, on_conflict="brief_fk,campaign_external_id")
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def list_campaigns_by_brief_fk(self, brief_fk: int) -> List[Dict[str, Any]]:
        response = (
            self.client.table("campaigns")
            .select("*")
            .eq("brief_fk", brief_fk)
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def get_campaign_id_map(self, brief_fk: int) -> Dict[str, int]:
        rows = self.list_campaigns_by_brief_fk(brief_fk)
        id_map: Dict[str, int] = {}
        for row in rows:
            ext_id = str(row.get("campaign_external_id", "")).strip()
            row_id = row.get("id")
            if ext_id and isinstance(row_id, int):
                id_map[ext_id] = row_id
        return id_map

    # ---------- runs ----------
    def create_analysis_run(
        self,
        *,
        route_type: str,
        brief_fk: Optional[int],
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "route_type": route_type,
            "brief_fk": brief_fk,
            "status": "started",
            "config_snapshot": config_snapshot or {},
            "started_at": _now_iso(),
        }
        response = self.client.table("analysis_runs").insert(payload).execute()
        rows = getattr(response, "data", None) or []
        if not rows:
            raise RuntimeError("Supabase create_analysis_run returned no row.")
        return rows[0]

    def finish_analysis_run(
        self,
        run_fk: int,
        *,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "status": status,
            "error_message": error_message,
            "finished_at": _now_iso(),
        }
        response = (
            self.client.table("analysis_runs")
            .update(payload)
            .eq("id", run_fk)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return rows[0] if rows else None

    # ---------- similarity ----------
    def upsert_similarity_rows(
        self,
        rows: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payloads = [dict(item) for item in rows]
        if not payloads:
            return []
        response = (
            self.client.table("campaign_similarity")
            .upsert(payloads, on_conflict="run_fk,source_campaign_fk,target_campaign_fk")
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    # ---------- diagnoses ----------
    def upsert_diagnosis_rows(
        self,
        rows: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payloads = [dict(item) for item in rows]
        if not payloads:
            return []
        response = (
            self.client.table("campaign_diagnoses")
            .upsert(payloads, on_conflict="run_fk,campaign_external_id")
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def build_diagnosis_rows(
        self,
        *,
        run_fk: int,
        brief_fk: int,
        diagnoses: List[Dict[str, Any]],
        campaign_id_map: Dict[str, int],
        eval_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for diagnosis in diagnoses:
            campaign_external_id = str(diagnosis.get("campaign_id", "")).strip()
            if not campaign_external_id:
                continue
            campaign_fk = campaign_id_map.get(campaign_external_id)
            output.append(
                {
                    "run_fk": run_fk,
                    "brief_fk": brief_fk,
                    "campaign_fk": campaign_fk,
                    "campaign_external_id": campaign_external_id,
                    "status": diagnosis.get("status"),
                    "diagnosis": diagnosis.get("diagnosis"),
                    "issues": list(diagnosis.get("issues", []) or []),
                    "recommendations": list(diagnosis.get("recommendations", []) or []),
                    "grounding_evidence": list(diagnosis.get("grounding_evidence", []) or []),
                    "eval_metrics": eval_metrics or {},
                    "updated_at": _now_iso(),
                }
            )
        return output

    def build_similarity_rows(
        self,
        *,
        run_fk: int,
        target_file_name: str,
        strong_matches: List[Dict[str, Any]],
        review_matches: List[Dict[str, Any]],
        campaign_fk_lookup: Dict[Tuple[str, str], int],
        strong_threshold: float = 80.0,
        review_threshold: float = 50.0,
        include_weak: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build campaign_similarity rows from branch payload matches.

        campaign_fk_lookup key format:
          (brief_file_name, campaign_external_id) -> campaign_row_id
        """
        output: List[Dict[str, Any]] = []

        def _append_matches(matches: List[Dict[str, Any]]) -> None:
            for match in matches:
                source_file = str(target_file_name or "").strip()
                target_file = str(match.get("file_name") or "").strip()
                source_campaign_ext = str(match.get("target_campaign_id") or "").strip()
                target_campaign_ext = str(match.get("candidate_campaign_id") or "").strip()
                if not source_file or not target_file or not source_campaign_ext or not target_campaign_ext:
                    continue

                source_fk = campaign_fk_lookup.get((source_file, source_campaign_ext))
                target_fk = campaign_fk_lookup.get((target_file, target_campaign_ext))
                if not source_fk or not target_fk:
                    continue

                score_0_100 = _normalize_similarity_score(float(match.get("similarity_score", 0.0)))
                grade = _grade_from_score(score_0_100, strong_threshold, review_threshold)
                if grade == "weak" and not include_weak:
                    continue

                output.append(
                    {
                        "run_fk": run_fk,
                        "source_campaign_fk": source_fk,
                        "target_campaign_fk": target_fk,
                        "similarity_score": score_0_100,
                        "similarity_grade": grade,
                        "match_reason": str(match.get("match_reason", "")).strip() or None,
                        "evidence_points": list(match.get("evidence_points", []) or []),
                        "score_breakdown": {
                            "scoring_path": match.get("scoring_path", "content"),
                            "pair_status": match.get("pair_status", "none"),
                            "pair_basis": match.get("pair_basis", "content"),
                            "style_direction_similarity": _normalize_similarity_score(
                                float(match.get("style_direction_similarity", match.get("asset_and_style_similarity", 0.0)))
                            ),
                            "style_fields_similarity": _normalize_similarity_score(
                                float(match.get("style_fields_similarity", 0.0))
                            ),
                            "asset_structure_similarity": _normalize_similarity_score(
                                float(match.get("asset_structure_similarity", 0.0))
                            ),
                            "campaign_wording_similarity": _normalize_similarity_score(
                                float(match.get("campaign_wording_similarity", 0.0))
                            ),
                            "dealership_relationship": _normalize_similarity_score(
                                float(match.get("dealership_relationship", 0.0))
                            ),
                            "reference_strength": _normalize_similarity_score(
                                float(match.get("reference_strength", match.get("reference_id_boost", 0.0)))
                            ),
                        },
                    }
                )

        _append_matches(strong_matches)
        _append_matches(review_matches)
        return output

    @staticmethod
    def _campaign_payload(
        brief_fk: int,
        campaign: Campaign,
        task_type: str,
        dealership_family_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "brief_fk": brief_fk,
            "campaign_external_id": campaign.campaign_id,
            "headline": campaign.offer_details.headline,
            "offer": campaign.offer_details.offer,
            "body": campaign.offer_details.body,
            "cta": campaign.offer_details.cta,
            "disclaimer": campaign.offer_details.disclaimer,
            "style_descriptions": campaign.style_descriptions.model_dump(mode="python"),
            "assets": campaign.assets.model_dump(mode="python"),
            "task_type": task_type,
            "dealership_family_id": dealership_family_id,
            "updated_at": _now_iso(),
        }
