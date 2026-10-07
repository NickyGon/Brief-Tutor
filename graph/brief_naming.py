"""
Shared helpers for campaign brief filename parsing and path resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import re


BRIEF_FILENAME_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<account_id>.+)-(?P<campaign_token>[AD]-\d{5,12})$",
    flags=re.IGNORECASE,
)


def parse_brief_filename(path_or_name: str) -> Optional[Dict[str, str]]:
    """
    Parse campaign brief filename patterns like:
      - 2025-11-rogerbeasleyvolvovcna-A-25008537.xlsx
      - 2026-06-tonydivinousedcarsntrucks-D-94095.xlsx
    """
    file_name = Path(path_or_name or "").name
    stem = Path(file_name).stem
    match = BRIEF_FILENAME_PATTERN.match(stem)
    if not match:
        return None
    account_id = str(match.group("account_id") or "").strip().lower()
    campaign_token = str(match.group("campaign_token") or "").strip().upper()
    return {
        "year": str(match.group("year")),
        "month": str(match.group("month")),
        "account_id": account_id,
        "campaign_token": campaign_token,
        "source_file_name": file_name,
        "brief_id": stem,
    }


def resolve_spreadsheet_path(
    spreadsheet_path: str,
    *,
    project_root: Optional[Path] = None,
    strict: bool = True,
) -> Path:
    """
    Resolve relative or absolute spreadsheet path to a canonical absolute path.
    """
    input_path = Path(spreadsheet_path)
    candidates = []

    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append(Path.cwd() / input_path)
        if project_root is not None:
            candidates.append(Path(project_root) / input_path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    if strict:
        raise FileNotFoundError(f"Spreadsheet file not found: {spreadsheet_path}")

    if input_path.is_absolute():
        return input_path.resolve()

    if project_root is not None:
        return (Path(project_root) / input_path).resolve()
    return (Path.cwd() / input_path).resolve()
