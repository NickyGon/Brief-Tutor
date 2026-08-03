"""
Standalone Supabase connectivity + table read check.

Usage:
  python check_supabase_tables.py
  python check_supabase_tables.py --service-role
  python check_supabase_tables.py --write-smoke

Requires .env values:
  SUPABASE_ENABLED=true
  SUPABASE_URL=https://....supabase.co
  SUPABASE_ANON_KEY=...
  SUPABASE_SERVICE_ROLE_KEY=...   (if using --service-role / default write path)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Tables used by the similarity/workflow repository + dealership metadata.
TABLES_TO_CHECK = [
    "dealership_groups",
    "dealership_accounts",
    "dealership_account_oems",
    "briefs",
    "campaigns",
    "analysis_runs",
    "campaign_similarity",
    "campaign_diagnoses",
]


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _summarize_rows(rows: List[Dict[str, Any]], limit: int = 3) -> None:
    if not rows:
        print("  rows: (empty)")
        return
    print(f"  rows returned: {len(rows)}")
    for idx, row in enumerate(rows[:limit], start=1):
        keys = list(row.keys())
        preview = {k: row.get(k) for k in keys[:6]}
        print(f"  sample[{idx}]: {preview}")


def check_config(*, force_enabled: bool) -> Tuple[bool, Dict[str, Any]]:
    if force_enabled and not _truthy(os.getenv("SUPABASE_ENABLED", "false")):
        os.environ["SUPABASE_ENABLED"] = "true"
        print("[config] Forced SUPABASE_ENABLED=true for this script run.")

    from graph.supabase.config import SupabaseSettings

    settings = SupabaseSettings.from_env()
    info = {
        "enabled": settings.enabled,
        "url": settings.url,
        "schema": settings.schema,
        "anon_key_set": bool(settings.anon_key),
        "service_role_key_set": bool(settings.service_role_key),
    }
    _print_header("Supabase config")
    for key, value in info.items():
        if key == "url" and value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    ok = settings.enabled and bool(settings.url) and (
        settings.anon_key or settings.service_role_key
    )
    return ok, info


def check_table_reads(*, use_service_role: bool, limit: int) -> Dict[str, Dict[str, Any]]:
    from graph.supabase.client import get_supabase_client, reset_supabase_clients

    reset_supabase_clients()
    client = get_supabase_client(use_service_role=use_service_role)
    mode = "service_role" if use_service_role else "anon"
    _print_header(f"Table reads ({mode})")

    results: Dict[str, Dict[str, Any]] = {}
    for table in TABLES_TO_CHECK:
        try:
            response = client.table(table).select("*").limit(limit).execute()
            rows = list(getattr(response, "data", None) or [])
            results[table] = {"ok": True, "count": len(rows), "error": None}
            print(f"[OK] {table}")
            _summarize_rows(rows, limit=min(limit, 2))
        except Exception as exc:
            results[table] = {"ok": False, "count": 0, "error": str(exc)}
            print(f"[FAIL] {table}: {exc}")
    return results


def write_smoke_dealership(*, use_service_role: bool = True) -> bool:
    """
    Minimal create/read/update/delete against dealership_groups.
    Uses a unique group_code so it won't collide with real seed data.
    """
    from graph.supabase.client import get_supabase_client, reset_supabase_clients

    _print_header("CRUD smoke (dealership_groups)")
    reset_supabase_clients()
    client = get_supabase_client(use_service_role=use_service_role)
    marker = "brief_tutor_smoke_check"
    group_code = f"{marker}_tmp"

    try:
        # Cleanup any leftover from a previous interrupted run.
        client.table("dealership_groups").delete().eq("group_code", group_code).execute()

        insert_payload = {
            "group_name": "Brief Tutor Smoke Check",
            "group_code": group_code,
            "notes": "temporary row from check_supabase_tables.py",
        }
        inserted = (
            client.table("dealership_groups")
            .insert(insert_payload)
            .execute()
        )
        rows = list(getattr(inserted, "data", None) or [])
        if not rows:
            print("[FAIL] insert returned no row")
            return False
        row_id = rows[0].get("id")
        print(f"[OK] CREATE id={row_id}")

        selected = (
            client.table("dealership_groups")
            .select("*")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        selected_rows = list(getattr(selected, "data", None) or [])
        if not selected_rows:
            print("[FAIL] READ after create returned no row")
            return False
        print(f"[OK] READ group_code={selected_rows[0].get('group_code')}")

        updated = (
            client.table("dealership_groups")
            .update({"notes": "updated by check_supabase_tables.py"})
            .eq("id", row_id)
            .execute()
        )
        updated_rows = list(getattr(updated, "data", None) or [])
        if not updated_rows:
            print("[FAIL] UPDATE returned no row")
            return False
        print(f"[OK] UPDATE notes={updated_rows[0].get('notes')}")

        deleted = (
            client.table("dealership_groups")
            .delete()
            .eq("id", row_id)
            .execute()
        )
        print(f"[OK] DELETE completed (response rows={len(getattr(deleted, 'data', None) or [])})")
        return True
    except Exception as exc:
        print(f"[FAIL] CRUD smoke failed: {exc}")
        try:
            client.table("dealership_groups").delete().eq("group_code", group_code).execute()
        except Exception:
            pass
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Supabase table connectivity and reads.")
    parser.add_argument(
        "--service-role",
        action="store_true",
        default=True,
        help="Use SUPABASE_SERVICE_ROLE_KEY (default: true).",
    )
    parser.add_argument(
        "--anon",
        action="store_true",
        help="Use SUPABASE_ANON_KEY instead of service role.",
    )
    parser.add_argument(
        "--force-enabled",
        action="store_true",
        help="Temporarily set SUPABASE_ENABLED=true for this run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max rows to fetch per table (default: 5).",
    )
    parser.add_argument(
        "--write-smoke",
        action="store_true",
        help="Run a temporary CREATE/READ/UPDATE/DELETE on dealership_groups.",
    )
    args = parser.parse_args()
    use_service_role = not args.anon

    config_ok, _ = check_config(force_enabled=args.force_enabled)
    if not config_ok:
        print(
            "\n[ERROR] Supabase is not configured. Set SUPABASE_ENABLED=true and "
            "SUPABASE_URL + keys in .env (or pass --force-enabled)."
        )
        return 1

    read_results = check_table_reads(use_service_role=use_service_role, limit=max(1, args.limit))
    ok_tables = [name for name, result in read_results.items() if result.get("ok")]
    fail_tables = [name for name, result in read_results.items() if not result.get("ok")]

    write_ok: Optional[bool] = None
    if args.write_smoke:
        if not use_service_role:
            print("\n[WARN] --write-smoke with --anon may fail under RLS; prefer service role.")
        write_ok = write_smoke_dealership(use_service_role=use_service_role)

    _print_header("Summary")
    print(f"  readable tables: {len(ok_tables)}/{len(TABLES_TO_CHECK)}")
    if ok_tables:
        print(f"  ok: {', '.join(ok_tables)}")
    if fail_tables:
        print(f"  fail: {', '.join(fail_tables)}")
    if write_ok is not None:
        print(f"  write smoke: {'OK' if write_ok else 'FAIL'}")

    if fail_tables:
        return 2
    if write_ok is False:
        return 3
    print("\nSupabase table access looks healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
