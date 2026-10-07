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
    YYYY-MM-<family_slug>-[A/D]-<numeric_id>.xlsx

    brief_id is the campaign token (A-/D-<numeric_id>), not the full filename stem.
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
    campaign_token = str(parsed.get("campaign_token") or "").strip().upper() or None
    return {
        "brief_id": campaign_token,
        "dealership_family_id": parsed.get("account_id"),
        "instance_id": campaign_token,
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
        # Prefer A-/D-<numeric_id> as the stable brief identity used for references.
        brief_id = ids.get("brief_id") or ids.get("instance_id") or Path(campaign_brief.spreadsheet_path).stem
        payload = {
            "brief_id": brief_id,
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
    @staticmethod
    def _normalize_oem_tokens(values: Iterable[Any]) -> List[str]:
        """
        Expand OEM values, including comma-separated entries in one cell.
        Examples: "Subaru", "GMC,Buick", "NA", "All"
        """
        tokens: List[str] = []
        seen: set = set()
        for value in values:
            raw = str(value or "").strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split(",")]
            for part in parts:
                if not part:
                    continue
                key = part.lower()
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(part)
        return tokens

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
            group_rows = list(getattr(group_resp, "data", None) or [])
            group_info = dict(group_rows[0]) if group_rows else None

        oems = self._normalize_oem_tokens([item.get("oem") for item in oem_rows])
        oem_tokens_lower = {token.lower() for token in oems}
        handles_all = bool(account_row.get("handles_all_oems", False)) or bool(
            oem_tokens_lower.intersection({"na", "all"})
        )
        return {
            "account_id": normalized_account,
            "account_name": account_row.get("account_name"),
            "group_fk": group_fk,
            "group": group_info,
            "oem_family": str(account_row.get("oem_family", "NA") or "NA").strip(),
            "handles_all_oems": handles_all,
            "oems": oems,
            "is_active": bool(account_row.get("is_active", True)),
        }

    @staticmethod
    def oem_compatible(
        source_account: Optional[Dict[str, Any]],
        candidate_account: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Compatibility rule for hard OEM filter / prioritized widen.
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
        if "all" in source_oems or "all" in candidate_oems or "na" in source_oems or "na" in candidate_oems:
            return True
        if source_oems and candidate_oems and source_oems.intersection(candidate_oems):
            return True

        source_family = str(source_account.get("oem_family", "NA") or "NA").strip().lower()
        candidate_family = str(candidate_account.get("oem_family", "NA") or "NA").strip().lower()
        if source_family != "na" and candidate_family != "na" and source_family == candidate_family:
            return True
        return False

    def list_compatible_account_ids(
        self,
        source_account_id: str,
        *,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return accountIDs compatible with the source account by OEM overlap,
        OEM family, or handles-all/NA behavior.

        Uses batched reads so widen prioritization stays cheap even with many accounts.
        """
        source_meta = self.get_dealership_account_metadata(source_account_id)
        if not source_meta:
            return []

        query = self.client.table("dealership_accounts").select(
            "id,account_id,account_name,group_fk,oem_family,handles_all_oems,is_active"
        )
        if not include_inactive:
            query = query.eq("is_active", True)
        account_rows = list(getattr(query.execute(), "data", None) or [])
        if not account_rows:
            return []

        account_pks = [
            int(row["id"])
            for row in account_rows
            if isinstance(row, dict) and isinstance(row.get("id"), int)
        ]
        oems_by_account_fk: Dict[int, List[str]] = {}
        if account_pks:
            oem_resp = (
                self.client.table("dealership_account_oems")
                .select("account_fk,oem")
                .in_("account_fk", account_pks)
                .execute()
            )
            for item in list(getattr(oem_resp, "data", None) or []):
                if not isinstance(item, dict):
                    continue
                account_fk = item.get("account_fk")
                if not isinstance(account_fk, int):
                    continue
                expanded = self._normalize_oem_tokens([item.get("oem")])
                if not expanded:
                    continue
                oems_by_account_fk.setdefault(account_fk, [])
                for token in expanded:
                    if token not in oems_by_account_fk[account_fk]:
                        oems_by_account_fk[account_fk].append(token)

        group_fks = sorted(
            {
                int(row["group_fk"])
                for row in account_rows
                if isinstance(row, dict) and isinstance(row.get("group_fk"), int)
            }
        )
        groups_by_id: Dict[int, Dict[str, Any]] = {}
        if group_fks:
            group_resp = (
                self.client.table("dealership_groups")
                .select("id,group_name")
                .in_("id", group_fks)
                .execute()
            )
            for item in list(getattr(group_resp, "data", None) or []):
                if isinstance(item, dict) and isinstance(item.get("id"), int):
                    groups_by_id[int(item["id"])] = dict(item)

        source_id = str(source_account_id or "").strip().lower()
        source_oems = {
            str(item).strip().lower()
            for item in (source_meta.get("oems", []) or [])
            if str(item).strip()
        }
        compatible: List[Dict[str, Any]] = []
        for row in account_rows:
            if not isinstance(row, dict):
                continue
            account_id = str(row.get("account_id") or "").strip().lower()
            if not account_id or account_id == source_id:
                continue

            account_pk = row.get("id")
            oems = list(oems_by_account_fk.get(int(account_pk), [])) if isinstance(account_pk, int) else []
            oem_tokens_lower = {token.lower() for token in oems}
            handles_all = bool(row.get("handles_all_oems", False)) or bool(
                oem_tokens_lower.intersection({"na", "all"})
            )
            group_fk = row.get("group_fk")
            group_info = groups_by_id.get(int(group_fk)) if isinstance(group_fk, int) else None
            candidate_meta = {
                "account_id": account_id,
                "account_name": row.get("account_name"),
                "group_fk": group_fk,
                "group": group_info,
                "oem_family": str(row.get("oem_family", "NA") or "NA").strip(),
                "handles_all_oems": handles_all,
                "oems": oems,
            }
            if not self.oem_compatible(source_meta, candidate_meta):
                continue

            match_basis = "oem_family"
            if source_meta.get("handles_all_oems") or handles_all:
                match_basis = "handles_all"
            elif source_oems.intersection(oem_tokens_lower):
                match_basis = "oem_overlap"

            compatible.append(
                {
                    "account_id": account_id,
                    "account_name": row.get("account_name"),
                    "group_fk": group_fk,
                    "group_name": (group_info or {}).get("group_name") if group_info else None,
                    "oem_family": candidate_meta["oem_family"],
                    "oems": oems,
                    "match_basis": match_basis,
                }
            )

        basis_rank = {"oem_overlap": 0, "oem_family": 1, "handles_all": 2}
        compatible.sort(
            key=lambda item: (
                basis_rank.get(str(item.get("match_basis") or ""), 9),
                str(item.get("account_id") or ""),
            )
        )
        return compatible

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

    def list_latest_diagnoses(self, brief_fk: int) -> List[Dict[str, Any]]:
        runs = (
            self.client.table("analysis_runs")
            .select("id")
            .eq("brief_fk", brief_fk)
            .eq("route_type", "standard_analyzer")
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        run_rows = list(getattr(runs, "data", None) or [])
        if not run_rows:
            return []
        run_fk = run_rows[0].get("id")
        if not isinstance(run_fk, int):
            return []
        response = (
            self.client.table("campaign_diagnoses")
            .select("*")
            .eq("run_fk", run_fk)
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def upsert_brief_similarity_matches(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        response = (
            self.client.table("brief_similarity_matches")
            .upsert(rows, on_conflict="target_brief_id,candidate_brief_id")
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def list_brief_similarity_matches(self, target_brief_id: str) -> List[Dict[str, Any]]:
        response = (
            self.client.table("brief_similarity_matches")
            .select("*")
            .eq("target_brief_id", str(target_brief_id or "").strip().upper())
            .order("brief_similarity_score", desc=True)
            .execute()
        )
        return list(getattr(response, "data", None) or [])

    def save_run_documents(self, run_fk: int, documents: List[Dict[str, Any]]) -> None:
        self.client.table("analysis_runs").update({"documents": documents}).eq("id", run_fk).execute()

    # ---------- similarity agent memory/cache ----------
    def get_similarity_agent_cache(
        self,
        *,
        target_brief_id: str,
        account_id: Optional[str] = None,
        candidate_fingerprint: Optional[str] = None,
        oem_family: Optional[str] = None,
        max_age_hours: float = 168.0,
        allow_account_fallback: bool = True,
        allow_oem_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch the newest usable similarity-agent cache row.

        Preference order:
          1) exact target + account + fingerprint
          2) same target + account (any fingerprint)
          3) same account recent (partial reuse)
          4) same OEM family recent (partial reuse)
        """
        target_id = str(target_brief_id or "").strip().upper()
        if not target_id:
            return None
        account = str(account_id or "").strip().lower() or None
        fingerprint = str(candidate_fingerprint or "").strip()
        oem = str(oem_family or "").strip() or None

        def _fresh(row: Dict[str, Any]) -> bool:
            updated_at = row.get("updated_at") or row.get("created_at")
            if not updated_at or max_age_hours <= 0:
                return True
            try:
                ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 3600.0
                return age_hours <= float(max_age_hours)
            except Exception:
                return True

        # 1) Exact fingerprint match for this target/account.
        query = (
            self.client.table("similarity_agent_cache")
            .select("*")
            .eq("target_brief_id", target_id)
            .order("updated_at", desc=True)
            .limit(5)
        )
        if account:
            query = query.eq("account_id", account)
        if fingerprint:
            query = query.eq("candidate_fingerprint", fingerprint)
        rows = list(getattr(query.execute(), "data", None) or [])
        for row in rows:
            if isinstance(row, dict) and _fresh(row):
                row = dict(row)
                row["_cache_hit_mode"] = "exact"
                return row

        # 2) Same target + account, ignore fingerprint.
        if fingerprint:
            query = (
                self.client.table("similarity_agent_cache")
                .select("*")
                .eq("target_brief_id", target_id)
                .order("updated_at", desc=True)
                .limit(5)
            )
            if account:
                query = query.eq("account_id", account)
            rows = list(getattr(query.execute(), "data", None) or [])
            for row in rows:
                if isinstance(row, dict) and _fresh(row):
                    row = dict(row)
                    row["_cache_hit_mode"] = "target_account"
                    return row

        # 3) Same account recent (for overlapping pair enrichment).
        if allow_account_fallback and account:
            query = (
                self.client.table("similarity_agent_cache")
                .select("*")
                .eq("account_id", account)
                .order("updated_at", desc=True)
                .limit(5)
            )
            rows = list(getattr(query.execute(), "data", None) or [])
            for row in rows:
                if isinstance(row, dict) and _fresh(row):
                    row = dict(row)
                    row["_cache_hit_mode"] = "account"
                    return row

        # 4) Same OEM family recent.
        if allow_oem_fallback and oem and oem.upper() != "NA":
            query = (
                self.client.table("similarity_agent_cache")
                .select("*")
                .eq("oem_family", oem)
                .order("updated_at", desc=True)
                .limit(5)
            )
            rows = list(getattr(query.execute(), "data", None) or [])
            for row in rows:
                if isinstance(row, dict) and _fresh(row):
                    row = dict(row)
                    row["_cache_hit_mode"] = "oem"
                    return row

        return None

    def upsert_similarity_agent_cache(
        self,
        *,
        target_brief_id: str,
        account_id: Optional[str],
        candidate_fingerprint: str,
        payload: Dict[str, Any],
        oem_family: Optional[str] = None,
        task_type: Optional[str] = None,
        pair_count: int = 0,
    ) -> Optional[Dict[str, Any]]:
        target_id = str(target_brief_id or "").strip().upper()
        if not target_id:
            return None
        row = {
            "target_brief_id": target_id,
            "account_id": str(account_id or "").strip().lower() or None,
            "oem_family": str(oem_family or "").strip() or None,
            "candidate_fingerprint": str(candidate_fingerprint or "").strip(),
            "task_type": task_type,
            "pair_count": int(pair_count or 0),
            "payload": payload or {},
            "updated_at": _now_iso(),
        }
        response = (
            self.client.table("similarity_agent_cache")
            .upsert(row, on_conflict="target_brief_id,account_id,candidate_fingerprint")
            .execute()
        )
        rows = list(getattr(response, "data", None) or [])
        return rows[0] if rows else None
