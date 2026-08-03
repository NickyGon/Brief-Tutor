"""
Tools for the LangGraph agent workflow.
"""
from typing import List, Dict, Optional, Set, Any
from langchain.tools import tool
from graph.models import Campaign, CampaignBrief, OfferDetails, StyleDescriptions, Assets
from graph.brief_naming import parse_brief_filename, resolve_spreadsheet_path
from collections import Counter
from difflib import SequenceMatcher
import re
import json
import ast
import pandas as pd
import os
import sys
import tempfile
import glob
import traceback
from pathlib import Path
from dotenv import load_dotenv
try:
    from graph.llm_provider import create_embeddings, get_primary_provider
except Exception:
    # Keep spreadsheet/core tools usable even when optional provider deps are missing.
    def get_primary_provider() -> str:  # type: ignore[no-redef]
        return os.getenv("LLM_PROVIDER", "openai")

    def create_embeddings(provider: str, model: str):  # type: ignore[no-redef]
        from langchain_openai import OpenAIEmbeddings

        if provider and provider.lower() != "openai":
            raise RuntimeError(
                "Requested non-openai embeddings provider but optional provider "
                "dependencies are unavailable. Install provider extras or set "
                "EMBEDDING_PROVIDER=openai."
            )
        return OpenAIEmbeddings(model=model)

# Load environment variables
load_dotenv()

# Qdrant and RAG imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
    from langchain_qdrant import Qdrant
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    Filter = None
    FieldCondition = None
    MatchValue = None

# Import from rag_ingestion (may need path adjustment)
try:
    # Add project root to path if needed for import
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    import rag_ingestion
    # Use getattr to safely access attributes (handles cases where module is still loading)
    EMBEDDING_MODEL = getattr(rag_ingestion, 'EMBEDDING_MODEL', None)
    get_drive_service = getattr(rag_ingestion, 'get_drive_service', None)
    download_file_content = getattr(rag_ingestion, 'download_file_content', None)
except (ImportError, AttributeError):
    # Will be imported dynamically if needed
    rag_ingestion = None
    EMBEDDING_MODEL = None
    get_drive_service = None
    download_file_content = None

def parse_asset_codes(raw: Optional[str]) -> Set[str]:
    ASSET_CODES = {"SL", "BN", "SRP", "DA", "SL_M", "BN_M"}
    if not raw:
        return set()
    tokens = [t.strip().upper() for t in raw.split("|")]
    tokens = [t for t in tokens if t]
    return {t for t in tokens if t in ASSET_CODES}

def parse_asset_summary_counts(asset_summary: Optional[str]) -> Dict[str, int]:
    if not asset_summary:
        return {}
    text = asset_summary.upper()
    counts: Dict[str, int] = {}
    for key, value in re.findall(r'([A-Z_]+)\s*:\s*(\d+)', text):
        counts[key] = int(value)
    return counts


ROW_LABEL_TO_KEY = {
    "SL | BN | SRP | DA": "sl_bn_srp_da",
    "SL_M | BN_M ": "sl_m_bn_m",
    "Facebook Assets": "facebook_assets",
    "Instagram Assets": "instagram_assets",
    "Google Assets": "google_assets",
    "Asset Style Direction": "asset_style_direction",
    "Additional Style Information": "additional_style_information",
    "Vehicle Photography": "vehicle_photography",
    "Logos": "logos",
    "Headline": "headline",
    "Offer": "offer",
    "Body": "body",
    "CTA": "cta",
    "Disclaimer": "disclaimer",
}

OT_LABELS = [f"OT {i}" for i in range(1, 7)]  # "OT 1"..."OT 6"

def _build_row_index_map(df: pd.DataFrame) -> tuple[dict, dict]:
    """Return (row_map, ot_row_map) based on labels in the detected label column."""
    # Guard for empty/sparse dataframes (can happen with MCP sampling on malformed sheets)
    if df is None or df.empty or df.shape[1] == 0:
        return {}, {}

    label_col_idx = _find_label_column(df)
    if label_col_idx < 0 or label_col_idx >= df.shape[1]:
        return {}, {}

    label_col = df.iloc[:, label_col_idx]

    row_map: dict[str, int] = {}
    for label, key in ROW_LABEL_TO_KEY.items():
        matches = label_col[label_col == label].index
        if len(matches):
            row_map[key] = int(matches[0])

    ot_rows: dict[str, int] = {}
    for i, label in enumerate(OT_LABELS, start=1):
        matches = label_col[label_col == label].index
        if len(matches):
            ot_rows[f"ot_{i}"] = int(matches[0])

    return row_map, ot_rows


def _extract_sheet_meta(df: pd.DataFrame) -> dict:
    """
    Find the meta header row (Task Type / Asset Summary / Dealership Name / Content 11-20)
    and read the values from the row immediately below it.

    This makes the parser robust to extra padding rows/columns.
    """
    header_row_idx = None

    # 1) Find the row that contains "Task Type"
    for r in range(df.shape[0]):
        if (df.iloc[r] == "Task Type").any():
            header_row_idx = r
            break

    if header_row_idx is None or header_row_idx + 1 >= df.shape[0]:
        # Fallback: keep old behavior if something weird happens
        task_type_cell = df.iloc[2, 3] if df.shape[0] > 3 else None
        task_type = str(task_type_cell).strip() if pd.notna(task_type_cell) else None
        return {
            "task_type": task_type,
            "asset_summary": None,
            "dealership_name": None,
            "content_11_20": False,
        }

    value_row_idx = header_row_idx + 1
    header_row = df.iloc[header_row_idx]
    value_row = df.iloc[value_row_idx]

    # Map header labels to their column indices on that row
    col_map: dict[str, int] = {}
    for c in range(df.shape[1]):
        val = header_row[c]
        if isinstance(val, str):
            text = val.strip()
            if text in ("Task Type", "Asset Summary", "Dealership Name", "Content 11-20"):
                col_map[text] = c

    def get_str(label: str) -> Optional[str]:
        c = col_map.get(label)
        if c is None:
            return None
        v = value_row[c]
        if pd.isna(v):
            return None
        return str(v).strip()

    task_type = get_str("Task Type")
    asset_summary = get_str("Asset Summary")
    dealership_name = get_str("Dealership Name")
    content_flag = get_str("Content 11-20")

    content_11_20_flag = (
        isinstance(content_flag, str) and content_flag.strip().lower() == "yes"
    )

    return {
        "task_type": task_type,
        "asset_summary": asset_summary,
        "dealership_name": dealership_name,
        "content_11_20": content_11_20_flag,
    }

def _find_label_column(df: pd.DataFrame) -> int:
    """
    Find the column that contains row labels like 'SL | BN | SRP | DA', 'Facebook Assets', etc.
    Falls back to column 3 (D) if nothing is found.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return -1

    candidate_labels = list(ROW_LABEL_TO_KEY.keys()) + OT_LABELS

    for c in range(df.shape[1]):
        col = df.iloc[:, c]
        if any(isinstance(v, str) and v.strip() in candidate_labels for v in col):
            return c

    # Fallback to old assumption (column D) only when it exists.
    return 3 if df.shape[1] > 3 else 0


def parse_campaign_sheet(
    df: pd.DataFrame,
    sheet_tag: str,
) -> tuple[dict, list[Campaign]]:
    """
    Parse one sheet (either 'CampaignContent_1_10' or 'CampaignContent_11_20')
    into meta + campaign list.

    This version is robust to the outer "square" of empty/bordered cells because:
    - It finds the meta header row by searching for the cell 'Task Type'
    - It finds the label column by searching for known row labels (e.g. 'SL | BN | SRP | DA')
    instead of assuming fixed row/column indices.
    """
    try:
        # --- meta / high-level attributes ---
        try:
            meta = _extract_sheet_meta(df)
        except Exception as e:
            print(f"[ERROR] Failed to extract sheet metadata from '{sheet_tag}': {type(e).__name__}: {str(e)}")
            raise

        task_type = meta.get("task_type")
        asset_summary = meta.get("asset_summary")
        dealership_name = meta.get("dealership_name")
        content_11_20_flag = meta.get("content_11_20", False)

        meta = {
            "task_type": task_type,
            "asset_summary": asset_summary,
            "dealership_name": dealership_name,
            "content_11_20": content_11_20_flag,
        }

        # --- per-campaign attributes ---
        # Build a map of row labels → row index (e.g. 'SL | BN | SRP | DA', 'Headline', etc.)
        try:
            row_map, ot_rows = _build_row_index_map(df)
        except Exception as e:
            print(f"[ERROR] Failed to build row index map for '{sheet_tag}': {type(e).__name__}: {str(e)}")
            raise

        campaigns: list[Campaign] = []

        # Detect the campaign header row dynamically by finding the first row
        # containing "Content 1" (or close variant), rather than assuming fixed index.
        CAMPAIGN_HEADER_ROW = None
        for r in range(df.shape[0]):
            row_values = df.iloc[r]
            found_content_1 = any(
                isinstance(cell, str) and re.match(r"^\s*Content\s*1\s*$", cell, flags=re.IGNORECASE)
                for cell in row_values
            )
            if found_content_1:
                CAMPAIGN_HEADER_ROW = r
                break

        # Fallback to historical template position if dynamic detection fails.
        if CAMPAIGN_HEADER_ROW is None:
            CAMPAIGN_HEADER_ROW = 7

        # Detect campaign columns dynamically from header row values like "Content 1", "Content 2", etc.
        campaign_columns: List[int] = []
        if CAMPAIGN_HEADER_ROW < df.shape[0]:
            for col_idx in range(df.shape[1]):
                header_cell = df.iloc[CAMPAIGN_HEADER_ROW, col_idx]
                if isinstance(header_cell, str) and re.match(r"^\s*Content\s+\d+\s*$", header_cell, flags=re.IGNORECASE):
                    campaign_columns.append(col_idx)

        # Fallback to historical fixed range if dynamic detection finds nothing.
        if not campaign_columns:
            campaign_columns = list(range(4, df.shape[1]))

        for col in campaign_columns:
            try:
                # Check if we can access the header row
                if CAMPAIGN_HEADER_ROW >= df.shape[0]:
                    print(f"[WARN] Campaign header row {CAMPAIGN_HEADER_ROW} is out of bounds for sheet '{sheet_tag}' (max row: {df.shape[0] - 1})")
                    break

                name_cell = df.iloc[CAMPAIGN_HEADER_ROW, col]
                if not isinstance(name_cell, str) or not name_cell.strip():
                    # Skip columns without a campaign
                    continue

                campaign_id = name_cell.strip()

                # Initialize dictionaries for nested models
                assets_data: Dict[str, str] = {}
                style_descriptions_data: Dict[str, str] = {}
                offer_details_data: Dict[str, str] = {}

                # Define which keys belong to which model
                assets_keys = {"sl_bn_srp_da", "sl_m_bn_m", "facebook_assets", "instagram_assets", "google_assets"}
                style_keys = {"asset_style_direction", "additional_style_information", "vehicle_photography", "logos"}
                offer_keys = {"headline", "offer", "body", "cta", "disclaimer"}

                # Normal labeled rows - assign to appropriate nested structure
                for key, row_idx in row_map.items():
                    try:
                        # Safety guard: row_idx should be in range, but we double-check
                        if row_idx >= df.shape[0]:
                            continue

                        value = df.iloc[row_idx, col]
                        if pd.isna(value):
                            continue

                        value_str = str(value).strip()
                        
                        # For asset keys, skip empty values and "None" placeholders
                        if key in assets_keys:
                            # Skip if blank/empty
                            if not value_str:
                                continue
                            
                            # Skip if it's "None" placeholder (case-insensitive)
                            if value_str.lower() == "none":
                                continue
                            
                            assets_data[key] = value_str
                        elif key in style_keys:
                            style_descriptions_data[key] = value_str
                        elif key in offer_keys:
                            offer_details_data[key] = value_str
                    except Exception as e:
                        print(f"[WARN] Error processing row '{key}' (index {row_idx}) for campaign '{campaign_id}' in column {col}: {type(e).__name__}: {str(e)}")
                        continue

                # OT 1–6 rows - these are additional assets
                # Skip blank values and placeholder text
                placeholder_patterns = [
                    "choose from dropdown",
                    "write dimensions",
                    "choose from dropdown or write dimensions of ot",
                    "choose from dropdown or write dimensions"
                ]
                
                for key, row_idx in ot_rows.items():
                    try:
                        if row_idx >= df.shape[0]:
                            continue

                        value = df.iloc[row_idx, col]
                        if pd.isna(value):
                            continue

                        value_str = str(value).strip()
                        
                        # Skip if blank or empty
                        if not value_str:
                            continue
                        
                        # Skip if it matches placeholder patterns (case-insensitive)
                        value_lower = value_str.lower()
                        if any(pattern in value_lower for pattern in placeholder_patterns):
                            continue

                        assets_data[key] = value_str
                    except Exception as e:
                        print(f"[WARN] Error processing OT row '{key}' (index {row_idx}) for campaign '{campaign_id}' in column {col}: {type(e).__name__}: {str(e)}")
                        continue

                # Create Pydantic model instances with defaults for missing required fields
                try:
                    assets = Assets(
                        sl_bn_srp_da=assets_data.get("sl_bn_srp_da", ""),
                        sl_m_bn_m=assets_data.get("sl_m_bn_m", ""),
                        facebook_assets=assets_data.get("facebook_assets", ""),
                        instagram_assets=assets_data.get("instagram_assets", ""),
                        google_assets=assets_data.get("google_assets", ""),
                        ot_1=assets_data.get("ot_1", ""),
                        ot_2=assets_data.get("ot_2", ""),
                        ot_3=assets_data.get("ot_3", ""),
                        ot_4=assets_data.get("ot_4", ""),
                        ot_5=assets_data.get("ot_5", ""),
                        ot_6=assets_data.get("ot_6", ""),
                    )

                    style_descriptions = StyleDescriptions(
                        asset_style_direction=style_descriptions_data.get("asset_style_direction", ""),
                        additional_style_information=style_descriptions_data.get("additional_style_information", ""),
                        vehicle_photography=style_descriptions_data.get("vehicle_photography", ""),
                        logos=style_descriptions_data.get("logos", ""),
                    )

                    # Check if headline exists - campaigns without headlines are not valid campaigns
                    headline = offer_details_data.get("headline", "").strip()
                    if not headline:
                        continue

                    offer_details = OfferDetails(
                        headline=headline,
                        offer=offer_details_data.get("offer", ""),
                        body=offer_details_data.get("body", ""),
                        cta=offer_details_data.get("cta", ""),
                        disclaimer=offer_details_data.get("disclaimer", ""),
                    )

                    # Create Campaign object
                    campaign = Campaign(
                        campaign_id=campaign_id,
                        style_descriptions=style_descriptions,
                        offer_details=offer_details,
                        assets=assets,
                    )
                    campaigns.append(campaign)
                except Exception as e:
                    print(f"[ERROR] Failed to create Campaign object for '{campaign_id}' in column {col}: {type(e).__name__}: {str(e)}")
                    print(f"       Assets data: {assets_data}")
                    print(f"       Style data: {style_descriptions_data}")
                    print(f"       Offer data: {offer_details_data}")
                    continue

            except Exception as e:
                print(f"[ERROR] Failed to process column {col} in sheet '{sheet_tag}': {type(e).__name__}: {str(e)}")
                continue

        return meta, campaigns

    except Exception as e:
        print(f"[ERROR] Critical error in parse_campaign_sheet for '{sheet_tag}': {type(e).__name__}: {str(e)}")
        print(f"[ERROR] Traceback:")
        traceback.print_exc()
        # Return empty results on critical failure
        return {
            "task_type": None,
            "asset_summary": None,
            "dealership_name": None,
            "content_11_20": False,
        }, []


def _parse_spreadsheet_internal(spreadsheet_path: str) -> dict:
    """
    Internal function to parse a spreadsheet without the tool wrapper.
    This can be called directly from other Python code.
    
    Returns:
        A dict with: task_type, asset_summary, dealership_name, content_11_20, campaigns.
    """
    use_custom_brief_mcp = (os.getenv("CUSTOM_BRIEF_MCP_ENABLED", "false").strip().lower() == "true")
    if use_custom_brief_mcp:
        print(f"[Spreadsheet Parser] Loading spreadsheet via custom MCP: {spreadsheet_path}")
        try:
            from graph.mcp_utils import parse_spreadsheet_via_custom_mcp

            return parse_spreadsheet_via_custom_mcp(spreadsheet_path)
        except Exception as exc:
            print(f"[Spreadsheet Parser] Custom MCP failed, falling back to existing parsers: {exc}")

    use_mcp_sampling = (os.getenv("EXCEL_MCP_ENABLED", "false").strip().lower() == "true")
    if use_mcp_sampling:
        print(f"[Spreadsheet Parser] Loading spreadsheet via MCP sampling: {spreadsheet_path}")
        try:
            from graph.mcp_excel_sampling import parse_spreadsheet_via_mcp_sampling

            return parse_spreadsheet_via_mcp_sampling(spreadsheet_path)
        except Exception as exc:
            print(f"[Spreadsheet Parser] MCP sampling failed, falling back to pandas parser: {exc}")

    print(f"[Spreadsheet Parser] Loading spreadsheet from: {spreadsheet_path}")
    xls = pd.ExcelFile(spreadsheet_path)

    # First tab: CampaignContent_1_10
    df1 = pd.read_excel(xls, "CampaignContent_1_10", header=None)
    meta_results, campaigns_results = parse_campaign_sheet(df1, "CampaignContent_1_10")

    # Create CampaignBrief Pydantic model
    campaign_brief = CampaignBrief(
        spreadsheet_path=spreadsheet_path,
        task_type=meta_results["task_type"] or "",
        asset_summary=meta_results.get("asset_summary"),
        dealership_name=meta_results.get("dealership_name"),
        content_11_20=meta_results.get("content_11_20", False),
        campaigns=campaigns_results
    )

    # If Content 11-20 == Yes, also read the second tab and extend campaigns
    if campaign_brief.content_11_20:
        if "CampaignContent_11_20" in xls.sheet_names:
            df2 = pd.read_excel(xls, "CampaignContent_11_20", header=None)
            _, campaigns2 = parse_campaign_sheet(df2, "CampaignContent_11_20")
            campaign_brief.campaigns.extend(campaigns2)

    # Return as dict with all nested models serialized
    # Recursively convert all Pydantic models to dicts for JSON serialization
    def serialize_pydantic_model(obj):
        """Recursively serialize Pydantic models to dicts"""
        if hasattr(obj, 'model_dump'):
            # Pydantic model - convert to dict
            return obj.model_dump(mode='python')
        elif isinstance(obj, dict):
            # Dict - recursively serialize values
            return {k: serialize_pydantic_model(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # List - recursively serialize items
            return [serialize_pydantic_model(item) for item in obj]
        else:
            # Primitive type or already serialized
            return obj
    
    # Convert the entire CampaignBrief to a fully serialized dict
    result_dict = serialize_pydantic_model(campaign_brief)
    
    # Verify no Pydantic models remain
    try:
        json.dumps(result_dict)  # Test if it's JSON serializable
    except TypeError as e:
        print(f"[ERROR] Result is not JSON serializable: {str(e)}")
        # Fallback: use model_dump_json and parse back
        result_dict = json.loads(campaign_brief.model_dump_json())
    
    return result_dict


@tool
def load_and_parse_spreadsheet(spreadsheet_path: str) -> CampaignBrief:
    """
    Load the campaign content spreadsheet and normalize it into a structured schema
    for the brief creator.

    Use this tool when you need to:
    - Read an Excel spreadsheet with campaign content.
    - Get a list of campaigns and their attributes.
    - Prepare data for later asset counting.

    Input:
    - spreadsheet_path: local path to the .xlsx file.

    Output:
    - A dict with: task_type, asset_summary, dealership_name, content_11_20, campaigns.
    """

    # Use the internal function to do the actual parsing
    return _parse_spreadsheet_internal(spreadsheet_path)


@tool
def load_and_parse_spreadsheet_mcp(spreadsheet_path: str) -> CampaignBrief:
    """
    Load the campaign spreadsheet using MCP Excel tools and normalize it into a
    structured brief.

    This tool uses:
    - excel_describe_sheets(fileAbsolutePath)
    - excel_read_sheet(fileAbsolutePath, sheetName)

    It is intended for the Brief Creator agent to explicitly parse spreadsheet
    content through the MCP path.
    """
    from graph.mcp_excel_sampling import parse_spreadsheet_via_mcp_sampling

    return parse_spreadsheet_via_mcp_sampling(spreadsheet_path)


@tool
def load_and_parse_spreadsheet_custom_mcp(spreadsheet_path: str) -> CampaignBrief:
    """
    Load and parse the campaign spreadsheet through the in-repo custom MCP flow:
    1) read workbook data
    2) extract campaign brief values
    3) curate CampaignBrief-shaped JSON
    """
    use_custom_brief_mcp = (
        os.getenv("CUSTOM_BRIEF_MCP_ENABLED", "false").strip().lower() == "true"
    )
    if not use_custom_brief_mcp:
        print(
            "[Spreadsheet Parser] CUSTOM_BRIEF_MCP_ENABLED=false; "
            "falling back to internal parser path."
        )
        return _parse_spreadsheet_internal(spreadsheet_path)

    from graph.mcp_utils import parse_spreadsheet_via_custom_mcp

    return parse_spreadsheet_via_custom_mcp(spreadsheet_path)


def extract_family_slug_from_filename(spreadsheet_path: str) -> Optional[str]:
    """
    Extract accountID/family slug from filename convention:
    YYYY-MM-<accountID>-<A-|D-><numeric_id>.xlsx
    """
    parsed = parse_brief_filename(spreadsheet_path)
    if not parsed:
        return None
    return parsed.get("account_id")


def extract_campaign_instance_id(spreadsheet_path: str) -> Optional[str]:
    """Extract the A-/D- token from standard brief filenames."""
    parsed = parse_brief_filename(spreadsheet_path)
    if not parsed:
        return None
    return parsed.get("campaign_token")


def _find_campaigns_root(path: Path) -> Optional[Path]:
    for parent in [path] + list(path.parents):
        if parent.name.lower() == "campaigns":
            return parent
    return None


def classify_campaign_path_hierarchy(target_path: str) -> Dict[str, Any]:
    """
    Classify hierarchy context for target spreadsheet path.

    Returns keys:
      - target_file_path
      - target_file_name
      - account_id
      - account_folder
      - group_folder
      - campaigns_root
      - has_group_folder
    """
    project_root = Path(__file__).parent.parent
    resolved_target = resolve_spreadsheet_path(
        target_path,
        project_root=project_root,
        strict=False,
    )
    parsed = parse_brief_filename(resolved_target.name) or {}
    account_folder = resolved_target.parent
    campaigns_root = _find_campaigns_root(account_folder)
    group_folder: Optional[Path] = None

    if campaigns_root and account_folder.parent != campaigns_root:
        # file <- account folder <- group folder <- Campaigns
        if campaigns_root in account_folder.parents:
            group_candidate = account_folder.parent
            if group_candidate != campaigns_root:
                group_folder = group_candidate

    return {
        "target_file_path": str(resolved_target),
        "target_file_name": resolved_target.name,
        "account_id": parsed.get("account_id"),
        "campaign_token": parsed.get("campaign_token"),
        "account_folder": str(account_folder),
        "group_folder": str(group_folder) if group_folder else None,
        "campaigns_root": str(campaigns_root) if campaigns_root else None,
        "has_group_folder": bool(group_folder),
    }


def _iter_candidate_xlsx_files(root: Path, blocked_dirs: Set[str]) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*.xlsx"):
        if any(part.lower() in blocked_dirs for part in path.parts):
            continue
        files.append(path.resolve())
    return files


def list_same_family_local_spreadsheets(
    target_path: str,
    search_scope: str = "account",
) -> List[str]:
    """
    Return local .xlsx candidates using hierarchical search scopes:
      - account: same accountID folder only
      - group: other account folders under the same group folder
      - campaigns: siblings under Campaigns root
    """
    hierarchy = classify_campaign_path_hierarchy(target_path)
    account_id = str(hierarchy.get("account_id") or "").strip().lower()
    target_resolved = str(hierarchy.get("target_file_path") or "")
    account_folder = Path(str(hierarchy.get("account_folder") or ""))
    group_folder_raw = hierarchy.get("group_folder")
    campaigns_root_raw = hierarchy.get("campaigns_root")
    group_folder = Path(group_folder_raw) if group_folder_raw else None
    campaigns_root = Path(campaigns_root_raw) if campaigns_root_raw else None

    if not target_resolved or not account_folder:
        return []

    blocked_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    candidate_files: List[Path] = []

    scope = str(search_scope or "account").strip().lower()
    if scope == "account":
        candidate_files = _iter_candidate_xlsx_files(account_folder, blocked_dirs)
    elif scope == "group":
        if group_folder:
            candidate_files = _iter_candidate_xlsx_files(group_folder, blocked_dirs)
            candidate_files = [p for p in candidate_files if account_folder not in p.parents]
        else:
            candidate_files = []
    elif scope == "campaigns":
        if campaigns_root:
            candidate_files = _iter_candidate_xlsx_files(campaigns_root, blocked_dirs)
            candidate_files = [p for p in candidate_files if account_folder not in p.parents]
        else:
            candidate_files = []
    else:
        raise ValueError(f"Unsupported search_scope '{search_scope}'")

    matched_files: List[str] = []
    seen: Set[str] = set()
    for path in candidate_files:
        resolved = str(path.resolve())
        if resolved == target_resolved:
            continue

        parsed = parse_brief_filename(path.name)
        if not parsed:
            continue

        # Keep same-account files in account scope. Wider scopes intentionally
        # include other accountIDs for fallback search.
        if scope == "account" and account_id and parsed.get("account_id") != account_id:
            continue

        if resolved in seen:
            continue
        seen.add(resolved)
        matched_files.append(resolved)

    return sorted(matched_files)


def parse_local_spreadsheet_to_campaign_brief(spreadsheet_path: str) -> CampaignBrief:
    """
    Parse a local spreadsheet path using the same internal parser used by brief creator.
    """
    project_root = Path(__file__).parent.parent
    resolved_path = resolve_spreadsheet_path(
        spreadsheet_path,
        project_root=project_root,
        strict=False,
    )
    parsed = _parse_spreadsheet_internal(str(resolved_path))
    if isinstance(parsed, CampaignBrief):
        parsed.spreadsheet_path = str(resolved_path)
        return parsed
    if isinstance(parsed, dict):
        parsed["spreadsheet_path"] = str(resolved_path)
        return CampaignBrief(**parsed)
    raise TypeError(f"Unexpected parsed spreadsheet payload type: {type(parsed)}")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _join_non_empty_values(values: List[Any]) -> str:
    parts = [_normalize_text(v) for v in values if _normalize_text(v)]
    return " | ".join(parts)


def _extract_campaign_text_sections(campaign: Campaign) -> Dict[str, str]:
    assets_data = campaign.assets.model_dump(mode="python") if hasattr(campaign.assets, "model_dump") else {}
    style_data = (
        campaign.style_descriptions.model_dump(mode="python")
        if hasattr(campaign.style_descriptions, "model_dump")
        else {}
    )
    offer_data = (
        campaign.offer_details.model_dump(mode="python")
        if hasattr(campaign.offer_details, "model_dump")
        else {}
    )
    return {
        "assets": _join_non_empty_values([assets_data.get(k, "") for k in sorted(assets_data.keys())]),
        "style": _join_non_empty_values([style_data.get(k, "") for k in sorted(style_data.keys())]),
        "offer": _join_non_empty_values([offer_data.get(k, "") for k in sorted(offer_data.keys())]),
    }


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(a=a, b=b).ratio())


def _binary_presence_vector(values: List[Any]) -> List[int]:
    return [1 if _normalize_text(v) else 0 for v in values]


def _vector_similarity(a: List[int], b: List[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for left, right in zip(a, b) if left == right)
    return float(matches / len(a)) if a else 0.0


_CAMPAIGN_REF_ID_PATTERN = re.compile(
    r"(?i)\b([AD])\s*[-–—]?\s*(\d{5,12})\b"
)
_COPY_REFER_CUE_PATTERN = re.compile(
    r"(?i)\b("
    r"copy\s+from|copied\s+from|copying\s+from|copy\b|"
    r"refer\s+to|refers\s+to|referred\s+to|reference(?:d)?|refer\b|"
    r"based\s+on|use\s+(?:as\s+)?(?:base|template)|"
    r"from\s+(?:campaign|brief)|see\s+(?:campaign|brief)|"
    r"cu\s*:|match\s+(?:to|with)|same\s+as|duplicate(?:\s+of)?"
    r")\b"
)


def _normalize_campaign_ref_id(letter: str, digits: str) -> str:
    return f"{str(letter).strip().upper()}-{str(digits).strip()}"


def _extract_referenced_campaign_ids(*texts: Any) -> set:
    """Extract A-/D- campaign IDs referenced in free-text fields."""
    found: set = set()
    for text in texts:
        normalized = str(text or "")
        if not normalized.strip():
            continue
        for match in _CAMPAIGN_REF_ID_PATTERN.findall(normalized):
            if isinstance(match, tuple) and len(match) >= 2:
                found.add(_normalize_campaign_ref_id(match[0], match[1]))
            elif isinstance(match, str) and match:
                found.add(str(match).strip().upper())
    return found


def _extract_cued_reference_ids(*texts: Any) -> set:
    """
    Extract A-/D- IDs from fields that also contain copy/refer cue wording
    (e.g. "Copy from D-12345", "Refer to A-99999").
    """
    found: set = set()
    for text in texts:
        normalized = str(text or "")
        if not normalized.strip():
            continue
        if not _COPY_REFER_CUE_PATTERN.search(normalized):
            continue
        found |= _extract_referenced_campaign_ids(normalized)
    return found


def _campaign_reference_haystack(campaign: Campaign) -> List[str]:
    style = campaign.style_descriptions
    assets = campaign.assets.model_dump(mode="python")
    return [
        getattr(style, "asset_style_direction", "") or "",
        getattr(style, "additional_style_information", "") or "",
        getattr(style, "vehicle_photography", "") or "",
        getattr(style, "logos", "") or "",
        *[str(assets.get(key, "") or "") for key in sorted(assets.keys())],
    ]


def _briefs_share_account_or_group(target_brief: CampaignBrief, candidate_brief: CampaignBrief) -> bool:
    target_family = extract_family_slug_from_filename(target_brief.spreadsheet_path or "")
    candidate_family = extract_family_slug_from_filename(candidate_brief.spreadsheet_path or "")
    if target_family and candidate_family and target_family == candidate_family:
        return True
    try:
        target_hierarchy = classify_campaign_path_hierarchy(target_brief.spreadsheet_path or "")
        candidate_hierarchy = classify_campaign_path_hierarchy(candidate_brief.spreadsheet_path or "")
        target_group = str(target_hierarchy.get("group_folder") or "").strip().lower()
        candidate_group = str(candidate_hierarchy.get("group_folder") or "").strip().lower()
        if target_group and candidate_group and target_group == candidate_group:
            return True
    except Exception:
        pass
    return False


def _reference_path_signal(
    target_campaign: Campaign,
    candidate_campaign: Campaign,
    target_brief: CampaignBrief,
    candidate_brief: CampaignBrief,
) -> Dict[str, Any]:
    """
    Detect copy/refer ID signals that should lead similarity scoring.

    Same account/group required. Signal triggers when:
      - both sides mention the same A-/D- ID in Style Direction (or related fields), or
      - candidate references target brief ID and/or vice versa
    Cue wording ("Copy from", "Refer to", etc.) increases strength but is not required
    when the same referred ID is present on both sides.
    """
    empty = {
        "scoring_path": "content",
        "reference_strength": 0.0,
        "reference_id_boost": 0.0,
        "reference_boost_reasons": [],
        "has_copy_refer_signal": False,
    }

    target_haystack = _campaign_reference_haystack(target_campaign)
    candidate_haystack = _campaign_reference_haystack(candidate_campaign)
    target_refs = _extract_referenced_campaign_ids(*target_haystack)
    candidate_refs = _extract_referenced_campaign_ids(*candidate_haystack)
    target_cued_refs = _extract_cued_reference_ids(*target_haystack)
    candidate_cued_refs = _extract_cued_reference_ids(*candidate_haystack)

    if not target_refs and not candidate_refs:
        return empty

    same_scope = _briefs_share_account_or_group(target_brief, candidate_brief)
    if not same_scope:
        print(
            "[Family Similarity] Reference IDs found but skipped "
            "(not same account/group): "
            f"target={sorted(target_refs)[:3]} candidate={sorted(candidate_refs)[:3]}"
        )
        return empty

    target_file_id = (extract_campaign_instance_id(target_brief.spreadsheet_path or "") or "").upper()
    candidate_file_id = (extract_campaign_instance_id(candidate_brief.spreadsheet_path or "") or "").upper()

    strength = 0.0
    reasons: List[str] = []

    shared_refs = sorted(target_refs & candidate_refs)
    shared_cued_refs = sorted(target_cued_refs & candidate_cued_refs)
    if shared_cued_refs:
        strength = max(strength, 0.94)
        reasons.append(
            f"shared copy/refer ID(s) with cue wording: {', '.join(shared_cued_refs[:3])}"
        )
    elif shared_refs:
        # Same referred ID in Style Direction (etc.) is enough to indicate pairing intent.
        strength = max(strength, 0.90)
        reasons.append(f"shared referred ID(s) in style/assets: {', '.join(shared_refs[:3])}")

    candidate_points_to_target = bool(
        target_file_id and (target_file_id in candidate_cued_refs or target_file_id in candidate_refs)
    )
    target_points_to_candidate = bool(
        candidate_file_id and (candidate_file_id in target_cued_refs or candidate_file_id in target_refs)
    )
    cued_cross = bool(
        (target_file_id and target_file_id in candidate_cued_refs)
        or (candidate_file_id and candidate_file_id in target_cued_refs)
    )

    if candidate_points_to_target and target_points_to_candidate:
        strength = max(strength, 0.98 if cued_cross else 0.95)
        reasons.append(
            f"mutual brief-ID references ({target_file_id} <-> {candidate_file_id})"
        )
    elif candidate_points_to_target:
        strength = max(strength, 0.94 if cued_cross else 0.90)
        reasons.append(f"candidate references target brief ID {target_file_id}")
    elif target_points_to_candidate:
        strength = max(strength, 0.94 if cued_cross else 0.90)
        reasons.append(f"target references candidate brief ID {candidate_file_id}")

    if strength <= 0.0:
        return empty

    return {
        "scoring_path": "reference",
        "reference_strength": round(strength, 6),
        "reference_id_boost": round(strength, 6),
        "reference_boost_reasons": reasons,
        "has_copy_refer_signal": True,
    }


def _asset_structure_similarity(target_campaign: Campaign, candidate_campaign: Campaign) -> float:
    target_assets = target_campaign.assets.model_dump(mode="python")
    candidate_assets = candidate_campaign.assets.model_dump(mode="python")
    keys = sorted(set(target_assets.keys()) | set(candidate_assets.keys()))
    target_vector = _binary_presence_vector([target_assets.get(key, "") for key in keys])
    candidate_vector = _binary_presence_vector([candidate_assets.get(key, "") for key in keys])
    presence = _vector_similarity(target_vector, candidate_vector)
    text = _text_similarity(
        _join_non_empty_values([target_assets.get(key, "") for key in keys]),
        _join_non_empty_values([candidate_assets.get(key, "") for key in keys]),
    )
    if presence > 0 and text > 0:
        return (0.55 * presence) + (0.45 * text)
    return presence or text


def _style_direction_section_similarity(
    target_campaign: Campaign,
    candidate_campaign: Campaign,
) -> Dict[str, float]:
    """
    Single StyleDirection-section score combining:
      Style Direction, Additional Style, Vehicle Photography, Logos, and Assets columns.
    """
    target_style = target_campaign.style_descriptions
    candidate_style = candidate_campaign.style_descriptions
    field_pairs = [
        (
            getattr(target_style, "asset_style_direction", "") or "",
            getattr(candidate_style, "asset_style_direction", "") or "",
        ),
        (
            getattr(target_style, "additional_style_information", "") or "",
            getattr(candidate_style, "additional_style_information", "") or "",
        ),
        (
            getattr(target_style, "vehicle_photography", "") or "",
            getattr(candidate_style, "vehicle_photography", "") or "",
        ),
        (
            getattr(target_style, "logos", "") or "",
            getattr(candidate_style, "logos", "") or "",
        ),
    ]

    field_scores: List[float] = []
    for left, right in field_pairs:
        if not _normalize_text(left) and not _normalize_text(right):
            continue
        field_scores.append(_text_similarity(left, right))

    style_fields_similarity = (
        sum(field_scores) / len(field_scores) if field_scores else 0.0
    )
    asset_structure_similarity = _asset_structure_similarity(target_campaign, candidate_campaign)

    if style_fields_similarity > 0 and asset_structure_similarity > 0:
        style_direction_similarity = (0.5 * style_fields_similarity) + (0.5 * asset_structure_similarity)
    else:
        style_direction_similarity = style_fields_similarity or asset_structure_similarity

    return {
        "style_direction_similarity": style_direction_similarity,
        "style_fields_similarity": style_fields_similarity,
        "asset_structure_similarity": asset_structure_similarity,
    }


def _campaign_wording_similarity(target_campaign: Campaign, candidate_campaign: Campaign) -> float:
    """
    Campaign structure + wording: offer-field presence pattern plus offer text similarity.
    """
    target_offer = target_campaign.offer_details.model_dump(mode="python")
    candidate_offer = candidate_campaign.offer_details.model_dump(mode="python")
    offer_keys = ["headline", "offer", "body", "cta", "disclaimer"]

    structure = _vector_similarity(
        _binary_presence_vector([target_offer.get(key, "") for key in offer_keys]),
        _binary_presence_vector([candidate_offer.get(key, "") for key in offer_keys]),
    )
    wording = _text_similarity(
        _join_non_empty_values([target_offer.get(key, "") for key in offer_keys]),
        _join_non_empty_values([candidate_offer.get(key, "") for key in offer_keys]),
    )
    return (0.30 * structure) + (0.70 * wording)


def _dealership_relationship_score(target_brief: CampaignBrief, candidate_brief: CampaignBrief) -> float:
    """
    Dealership / OEM / group proximity:
      same accountID -> 1.0
      same group folder -> 0.85
      otherwise dealership-name text similarity
    """
    target_family = extract_family_slug_from_filename(target_brief.spreadsheet_path or "")
    candidate_family = extract_family_slug_from_filename(candidate_brief.spreadsheet_path or "")
    if target_family and candidate_family and target_family == candidate_family:
        return 1.0

    try:
        target_hierarchy = classify_campaign_path_hierarchy(target_brief.spreadsheet_path or "")
        candidate_hierarchy = classify_campaign_path_hierarchy(candidate_brief.spreadsheet_path or "")
        target_group = str(target_hierarchy.get("group_folder") or "").strip().lower()
        candidate_group = str(candidate_hierarchy.get("group_folder") or "").strip().lower()
        if target_group and candidate_group and target_group == candidate_group:
            return 0.85
    except Exception:
        pass

    target_name = _normalize_text(target_brief.dealership_name or "")
    candidate_name = _normalize_text(candidate_brief.dealership_name or "")
    if not target_name or not candidate_name:
        return 0.0
    return _text_similarity(target_name, candidate_name)


def compute_campaign_similarity(
    target_campaign: Campaign,
    candidate_campaign: Campaign,
    dealership_relationship: float,
    *,
    target_brief: Optional[CampaignBrief] = None,
    candidate_brief: Optional[CampaignBrief] = None,
) -> Dict[str, Any]:
    """
    Dual-path campaign similarity used to decide absolute campaign pairing:

    Always:
      10% dealership / OEM / group proximity

    Reference path (copy/refer cue + matching A-/D- IDs in same account/group):
      75% reference_strength + 15% content blend (style/assets + wording)

    Content path (no copy/refer ID signal):
      60% StyleDirection section + 30% campaign wording/structure
    """
    style_section = _style_direction_section_similarity(target_campaign, candidate_campaign)
    style_direction_similarity = float(style_section["style_direction_similarity"])
    campaign_wording_similarity = _campaign_wording_similarity(target_campaign, candidate_campaign)
    dealership_relationship = max(0.0, min(1.0, float(dealership_relationship)))
    content_blend = (0.67 * style_direction_similarity) + (0.33 * campaign_wording_similarity)

    ref_info: Dict[str, Any] = {
        "scoring_path": "content",
        "reference_strength": 0.0,
        "reference_id_boost": 0.0,
        "reference_boost_reasons": [],
        "has_copy_refer_signal": False,
    }
    if target_brief is not None and candidate_brief is not None:
        ref_info = _reference_path_signal(
            target_campaign,
            candidate_campaign,
            target_brief,
            candidate_brief,
        )

    scoring_path = str(ref_info.get("scoring_path") or "content")
    reference_strength = float(ref_info.get("reference_strength", 0.0) or 0.0)

    if scoring_path == "reference" and reference_strength > 0:
        weighted = (
            (0.10 * dealership_relationship)
            + (0.75 * reference_strength)
            + (0.15 * content_blend)
        )
        pair_basis = "reference"
    else:
        scoring_path = "content"
        weighted = (
            (0.10 * dealership_relationship)
            + (0.60 * style_direction_similarity)
            + (0.30 * campaign_wording_similarity)
        )
        if style_direction_similarity >= campaign_wording_similarity:
            pair_basis = "style_assets"
        else:
            pair_basis = "content"

    final_score = max(0.0, min(1.0, weighted))
    return {
        "similarity_score": round(final_score, 6),
        "scoring_path": scoring_path,
        "pair_basis": pair_basis,
        "style_direction_similarity": round(style_direction_similarity, 6),
        "style_fields_similarity": round(float(style_section["style_fields_similarity"]), 6),
        "asset_structure_similarity": round(float(style_section["asset_structure_similarity"]), 6),
        "campaign_wording_similarity": round(campaign_wording_similarity, 6),
        "dealership_relationship": round(dealership_relationship, 6),
        "reference_strength": round(reference_strength, 6),
        "reference_id_boost": round(float(ref_info.get("reference_id_boost", 0.0) or 0.0), 6),
        "reference_boost_reasons": list(ref_info.get("reference_boost_reasons", []) or []),
        "has_copy_refer_signal": bool(ref_info.get("has_copy_refer_signal")),
        "asset_and_style_similarity": round(style_direction_similarity, 6),
    }


def _classify_pair_status(
    match: Dict[str, Any],
    *,
    absolute_threshold: float = 0.80,
    likely_threshold: float = 0.70,
    review_threshold: float = 0.50,
) -> str:
    """
    Decide whether a scored campaign pair is an absolute pair, likely pair, review, or none.
    """
    overall = float(match.get("similarity_score", 0.0) or 0.0)
    style = float(match.get("style_direction_similarity", 0.0) or 0.0)
    wording = float(match.get("campaign_wording_similarity", 0.0) or 0.0)
    ref = float(match.get("reference_strength", match.get("reference_id_boost", 0.0)) or 0.0)
    has_ref = bool(match.get("has_copy_refer_signal"))

    # Absolute pairing: explicit copy/refer lock, or very strong style/assets agreement.
    if has_ref and ref >= 0.90 and overall >= absolute_threshold:
        return "absolute"
    if style >= 0.82 and overall >= absolute_threshold:
        return "absolute"
    if style >= 0.75 and wording >= 0.70 and overall >= absolute_threshold:
        return "absolute"

    if overall >= likely_threshold and (style >= 0.65 or has_ref or wording >= 0.70):
        return "likely"

    if overall >= review_threshold:
        return "review"
    return "none"


def _pair_priority_key(match: Dict[str, Any]) -> tuple:
    status_rank = {"absolute": 0, "likely": 1, "review": 2, "none": 3}
    basis_rank = {"reference": 0, "style_assets": 1, "content": 2, "mixed": 3}
    return (
        status_rank.get(str(match.get("pair_status") or "none"), 9),
        basis_rank.get(str(match.get("pair_basis") or "content"), 9),
        -float(match.get("similarity_score", 0.0) or 0.0),
        -float(match.get("style_direction_similarity", 0.0) or 0.0),
        -float(match.get("reference_strength", 0.0) or 0.0),
    )


def _assign_one_to_one_pairs(scored_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Greedy 1:1 assignment so each target campaign and each candidate campaign
    appears in at most one selected pair.
    """
    used_targets: set = set()
    used_candidates: set = set()
    assigned: List[Dict[str, Any]] = []
    for match in sorted(scored_pairs, key=_pair_priority_key):
        if str(match.get("pair_status") or "none") == "none":
            continue
        target_id = str(match.get("target_campaign_id") or "").strip()
        candidate_id = str(match.get("candidate_campaign_id") or "").strip()
        if not target_id or not candidate_id:
            continue
        if target_id in used_targets or candidate_id in used_candidates:
            continue
        used_targets.add(target_id)
        used_candidates.add(candidate_id)
        assigned.append(match)
    return assigned


def resolve_global_campaign_pairs(
    matches: List[Dict[str, Any]],
    *,
    target_campaign_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve absolute/likely/review pairs across all candidate files.

    Each target campaign may pair to at most one candidate campaign (file + Content ID).
    """
    usable = [dict(item) for item in matches if isinstance(item, dict)]
    for item in usable:
        if not item.get("pair_status"):
            item["pair_status"] = _classify_pair_status(item)

    used_targets: set = set()
    used_candidate_keys: set = set()
    absolute_pairs: List[Dict[str, Any]] = []
    likely_pairs: List[Dict[str, Any]] = []
    review_pairs: List[Dict[str, Any]] = []

    for match in sorted(usable, key=_pair_priority_key):
        status = str(match.get("pair_status") or "none")
        if status == "none":
            continue
        target_id = str(match.get("target_campaign_id") or "").strip()
        candidate_id = str(match.get("candidate_campaign_id") or "").strip()
        file_name = str(match.get("file_name") or "").strip()
        candidate_key = f"{file_name}|{candidate_id}"
        if not target_id or not candidate_id:
            continue
        if target_id in used_targets or candidate_key in used_candidate_keys:
            continue
        used_targets.add(target_id)
        used_candidate_keys.add(candidate_key)
        if status == "absolute":
            absolute_pairs.append(match)
        elif status == "likely":
            likely_pairs.append(match)
        elif status == "review":
            review_pairs.append(match)

    known_targets = [
        str(cid).strip()
        for cid in (target_campaign_ids or [])
        if str(cid).strip()
    ]
    if not known_targets:
        known_targets = sorted(
            {
                str(item.get("target_campaign_id") or "").strip()
                for item in usable
                if str(item.get("target_campaign_id") or "").strip()
            }
        )
    unpaired_targets = [cid for cid in known_targets if cid not in used_targets]

    # Compatibility buckets used by existing agent/report paths.
    strong_matches = absolute_pairs + likely_pairs
    review_matches = review_pairs
    return {
        "absolute_pairs": absolute_pairs,
        "likely_pairs": likely_pairs,
        "review_pairs": review_pairs,
        "unpaired_targets": unpaired_targets,
        "strong_matches": strong_matches,
        "review_matches": review_matches,
    }


def compare_briefs_and_rank(
    target_brief: CampaignBrief,
    candidate_brief: CampaignBrief,
    strong_threshold: float = 0.80,
    review_threshold: float = 0.50,
    likely_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Score target vs candidate campaigns and assign absolute 1:1 pairs inside this file.

    Pairing priority:
      1) copy/refer reference locks
      2) strong StyleDirection/assets (+ wording) agreement
      3) review-level content similarity
    """
    dealership_relationship = _dealership_relationship_score(target_brief, candidate_brief)
    scored_pairs: List[Dict[str, Any]] = []
    for target_campaign in target_brief.campaigns:
        for candidate_campaign in candidate_brief.campaigns:
            scores = compute_campaign_similarity(
                target_campaign=target_campaign,
                candidate_campaign=candidate_campaign,
                dealership_relationship=dealership_relationship,
                target_brief=target_brief,
                candidate_brief=candidate_brief,
            )
            current = {
                "target_campaign_id": target_campaign.campaign_id,
                "candidate_campaign_id": candidate_campaign.campaign_id,
                **scores,
            }
            current["pair_status"] = _classify_pair_status(
                current,
                absolute_threshold=strong_threshold,
                likely_threshold=likely_threshold,
                review_threshold=review_threshold,
            )
            scored_pairs.append(current)

    assigned_pairs = _assign_one_to_one_pairs(scored_pairs)
    absolute_pairs = [m for m in assigned_pairs if m.get("pair_status") == "absolute"]
    likely_pairs = [m for m in assigned_pairs if m.get("pair_status") == "likely"]
    review_pairs = [m for m in assigned_pairs if m.get("pair_status") == "review"]
    strong_matches = absolute_pairs + likely_pairs

    paired_targets = {str(m.get("target_campaign_id") or "") for m in assigned_pairs}
    unpaired_targets = [
        campaign.campaign_id
        for campaign in target_brief.campaigns
        if campaign.campaign_id not in paired_targets
    ]

    assigned_pairs.sort(key=_pair_priority_key)
    scored_by_score = sorted(
        scored_pairs,
        key=lambda item: float(item.get("similarity_score", 0.0) or 0.0),
        reverse=True,
    )
    # Use assigned pairs when available; otherwise report the strongest raw scores
    # so console/report averages are not zeroed just because nothing cleared thresholds.
    reporting_pool = assigned_pairs if assigned_pairs else scored_by_score[: max(1, len(target_brief.campaigns))]
    file_similarity = (
        float(reporting_pool[0]["similarity_score"]) if reporting_pool else 0.0
    )
    best_raw_pair = scored_by_score[0] if scored_by_score else None

    def _avg(key: str, rows: Optional[List[Dict[str, Any]]] = None) -> float:
        pool = rows if rows is not None else reporting_pool
        if not pool:
            return 0.0
        return sum(float(item.get(key, 0.0) or 0.0) for item in pool) / len(pool)

    return {
        "file_similarity_score": round(file_similarity, 6),
        "component_averages": {
            "style_direction_similarity": round(_avg("style_direction_similarity"), 6),
            "style_fields_similarity": round(_avg("style_fields_similarity"), 6),
            "asset_structure_similarity": round(_avg("asset_structure_similarity"), 6),
            "campaign_wording_similarity": round(_avg("campaign_wording_similarity"), 6),
            "dealership_relationship": round(_avg("dealership_relationship"), 6),
            "reference_strength": round(_avg("reference_strength"), 6),
            "reference_id_boost": round(_avg("reference_id_boost"), 6),
            "asset_and_style_similarity": round(_avg("style_direction_similarity"), 6),
        },
        "best_campaign_matches": assigned_pairs,
        "best_raw_pair": best_raw_pair,
        "absolute_pairs": absolute_pairs,
        "likely_pairs": likely_pairs,
        "review_pairs": review_pairs,
        "unpaired_targets": unpaired_targets,
        "strong_matches": strong_matches,
        "review_matches": review_pairs,
        "all_scored_pairs": scored_pairs,
    }


def write_family_similarity_outputs(
    target_brief: CampaignBrief,
    similarity_payload: Dict[str, Any],
) -> Dict[str, str]:
    """
    Write JSON and TXT outputs for the family-similarity branch.
    """
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    similarity_dir = results_dir / "similarity"
    results_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)

    target_name = Path(target_brief.spreadsheet_path).name
    target_stem = Path(target_name).stem
    json_path = similarity_dir / f"{target_stem}-family-similarity.json"
    report_path = results_dir / f"{target_stem}-family-similarity-report.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(similarity_payload, file, indent=2, ensure_ascii=False)

    absolute_pairs = similarity_payload.get("absolute_pairs") or similarity_payload.get("strong_matches", [])
    likely_pairs = similarity_payload.get("likely_pairs") or []
    review_pairs = similarity_payload.get("review_pairs") or similarity_payload.get("review_matches", [])
    unpaired_targets = similarity_payload.get("unpaired_targets") or []

    def _append_pair_lines(lines: List[str], items: List[Dict[str, Any]]) -> None:
        if not items:
            lines.append("None")
            return
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. [{item.get('pair_status', 'n/a')}] {item.get('similarity_percent', 0):.2f}% | "
                f"Target {item.get('target_campaign_id')} <-> Candidate {item.get('candidate_campaign_id')} | "
                f"File: {item.get('file_name')} | basis={item.get('pair_basis', item.get('scoring_path', 'n/a'))}"
            )
            lines.append(
                "   Breakdown: "
                f"path={item.get('scoring_path', 'content')} | "
                f"styleDirection={float(item.get('style_direction_similarity', item.get('asset_and_style_similarity', 0.0))) * 100:.1f}% "
                f"(styleFields={float(item.get('style_fields_similarity', 0.0)) * 100:.1f}%, "
                f"assets={float(item.get('asset_structure_similarity', 0.0)) * 100:.1f}%), "
                f"wording={float(item.get('campaign_wording_similarity', 0.0)) * 100:.1f}%, "
                f"dealership={float(item.get('dealership_relationship', 0.0)) * 100:.1f}%, "
                f"refStrength={float(item.get('reference_strength', item.get('reference_id_boost', 0.0))) * 100:.1f}%"
            )
            boost_reasons = item.get("reference_boost_reasons", []) or []
            if isinstance(boost_reasons, list) and boost_reasons:
                lines.append(f"   Ref signal: {'; '.join(str(r) for r in boost_reasons[:3])}")
            reason = str(item.get("match_reason", "")).strip()
            evidence_points = item.get("evidence_points", []) or []
            if reason:
                lines.append(f"   Reason: {reason}")
            if isinstance(evidence_points, list) and evidence_points:
                for evidence in evidence_points[:4]:
                    lines.append(f"   - Evidence: {str(evidence)}")

    lines: List[str] = []
    lines.append(f"File Name: {target_name}")
    lines.append(
        "Dealership Name and ID: "
        f"{similarity_payload.get('dealership_name', 'N/A')} "
        f"({similarity_payload.get('dealership_family_id', 'N/A')})"
    )
    lines.append(f"Type of Task: {similarity_payload.get('task_type', 'N/A')}")
    lines.append("")
    lines.append("Absolute campaign pairs (reference and/or strong style-assets lock):")
    _append_pair_lines(lines, absolute_pairs if isinstance(absolute_pairs, list) else [])
    lines.append("")
    lines.append("Likely campaign pairs:")
    _append_pair_lines(lines, likely_pairs if isinstance(likely_pairs, list) else [])
    lines.append("")
    lines.append("Campaign pairs for human review:")
    _append_pair_lines(lines, review_pairs if isinstance(review_pairs, list) else [])
    lines.append("")
    lines.append("Unpaired target campaigns:")
    if unpaired_targets:
        for idx, campaign_id in enumerate(unpaired_targets, start=1):
            lines.append(f"{idx}. {campaign_id}")
    else:
        lines.append("None")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return {"json_path": str(json_path), "report_path": str(report_path)}


def _get_qdrant_client():
    """
    Creates and returns a Qdrant client instance.
    
    Returns:
        QdrantClient instance or None if not available
    """
    if not QDRANT_AVAILABLE:
        return None
    
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url:
        raise RuntimeError("QDRANT_URL env var is not set")

    client = QdrantClient(url=url, api_key=api_key)
    return client


def _get_qdrant_vectorstore(collection_name: str = None):
    """
    Creates and returns a Qdrant vectorstore instance.
    
    Args:
        collection_name: Name of the Qdrant collection to use
        
    Returns:
        Qdrant vectorstore instance or None if not available
    """
    if not QDRANT_AVAILABLE:
        return None
    
    client = _get_qdrant_client()
    if not client:
        return None
    
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "my_rag_collection")
    
    try:
        # Use the same model/provider defaults as ingestion.
        embedding_model = EMBEDDING_MODEL or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", get_primary_provider())
        embeddings = create_embeddings(provider=embedding_provider, model=embedding_model)
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Warning: Could not create Qdrant vectorstore: {e}")
        return None


@tool
def retrieve_rag_information(
    query: str, 
    collection_name: Optional[str] = None, 
    top_k: int = 5,
    file_type: Optional[str] = None
) -> str:
    """
    Retrieve relevant information from documents stored in Qdrant vector database.
    This tool helps agents get guidance and context from stored documentation or campaign briefs.
    
    Use this tool when you need to:
    - Find guidelines, best practices, or examples from documentation
    - Get context about campaign creation, themes, or updates
    - Look up specific information that might be in stored documents
    - Retrieve similar campaign briefs (spreadsheets) for reference
    
    Args:
        query: The search query to find relevant documents
        collection_name: Optional name of the Qdrant collection to search (defaults to env var)
        top_k: Number of most relevant documents to retrieve (default: 5)
        file_type: Optional filter by file type. Use "campaign_brief" to retrieve only spreadsheets,
                   or "document" for regular documents. If None, retrieves all types.
        
    Returns:
        A string containing the retrieved relevant information from documents.
        For campaign briefs, includes structured information about task type, dealership, and campaigns.
    """
    if not QDRANT_AVAILABLE:
        return "Error: Qdrant dependencies are not installed. Please install qdrant-client and langchain-community."
    
    vectorstore = _get_qdrant_vectorstore(collection_name)
    
    if not vectorstore:
        return "Error: Could not connect to Qdrant vector database. Please check your QDRANT_HOST, QDRANT_PORT, and QDRANT_COLLECTION_NAME environment variables."
    
    try:
        # Get Qdrant client directly for better payload access
        client = _get_qdrant_client()
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "my_rag_collection")
        
        # Use vectorstore for embedding the query
        query_vector = vectorstore.embeddings.embed_query(query)
        
        # Search directly with Qdrant client to get full payloads
        search_filter = None
        if file_type:
            search_filter = Filter(
                must=[FieldCondition(
                    key="file_type",
                    match=MatchValue(value=file_type)
                )]
            )
        
        # Use query_points method (correct Qdrant client API)
        search_results = client.query_points(
            collection_name=collection_name_actual,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )
        
        # Extract points from the response
        if hasattr(search_results, 'points'):
            points = search_results.points
        else:
            points = []
        
        if not points:
            return f"No relevant documents found for query: '{query}'"
        
        # Format the retrieved documents
        results = []
        for i, result in enumerate(points, 1):
            payload = result.payload or {}
            
            # Extract content - it might be in different fields
            content = payload.get("page_content") or payload.get("text") or ""
            
            # Check if this is a campaign brief with structured data
            structured_brief = payload.get("structured_brief")
            file_type_actual = payload.get("file_type")
            
            # Check for new format (per-campaign documents) or old format (structured_brief)
            campaign_json = payload.get("campaign_json")
            brief_metadata = payload.get("brief_metadata")
            
            if structured_brief and isinstance(structured_brief, dict):
                # Old format: structured campaign brief
                brief_info = []
                brief_info.append(f"Campaign Brief {i}: {structured_brief.get('file_name', 'Unknown')}")
                brief_info.append(f"  Task Type: {structured_brief.get('task_type', 'Unknown')}")
                brief_info.append(f"  Dealership: {structured_brief.get('dealership_name', 'Unknown')}")
                brief_info.append(f"  Total Campaigns: {structured_brief.get('total_campaigns', len(structured_brief.get('campaigns', [])))}")
                brief_info.append(f"  Asset Summary: {structured_brief.get('asset_summary', 'N/A')}")
                
                # Include campaign summaries
                campaigns = structured_brief.get('campaigns', [])
                if campaigns:
                    brief_info.append(f"  Campaigns:")
                    for camp in campaigns[:5]:  # Limit to first 5 for brevity
                        if camp and isinstance(camp, dict):
                            brief_info.append(f"    - {camp.get('campaign_id', 'Unknown')}: {camp.get('headline', '')[:50]}...")
                    if len(campaigns) > 5:
                        brief_info.append(f"    ... and {len(campaigns) - 5} more campaigns")
                
                brief_info.append(f"\n  Full Text Content:\n{content}")
                results.append("\n".join(brief_info))
            elif campaign_json and isinstance(campaign_json, dict):
                # New format: per-campaign document
                brief_info = []
                if brief_metadata and isinstance(brief_metadata, dict):
                    brief_info.append(f"Campaign {i}: {brief_metadata.get('file_name', 'Unknown')}")
                    brief_info.append(f"  Task Type: {brief_metadata.get('task_type', 'Unknown')}")
                    brief_info.append(f"  Dealership: {brief_metadata.get('dealership_name', 'Unknown')}")
                else:
                    brief_info.append(f"Campaign {i}: {campaign_json.get('campaign_id', 'Unknown')}")
                
                # Add campaign details
                offer_details = campaign_json.get('offer_details', {})
                if offer_details and isinstance(offer_details, dict):
                    brief_info.append(f"  Headline: {offer_details.get('headline', 'N/A')}")
                    brief_info.append(f"  Offer: {offer_details.get('offer', 'N/A')}")
                
                brief_info.append(f"\n  Full Campaign JSON:\n{json.dumps(campaign_json, indent=2)}")
                brief_info.append(f"\n  Full Text Content:\n{content}")
                results.append("\n".join(brief_info))
            elif file_type_actual == "campaign" or file_type_actual == "campaign_brief":
                # Fallback: campaign-related document but no structured data
                brief_info = []
                brief_info.append(f"Campaign Document {i}:")
                brief_info.append(f"  File Type: {file_type_actual}")
                brief_info.append(f"  Content:\n{content}")
                results.append("\n".join(brief_info))
            else:
                # Regular document
                source = payload.get("source") or payload.get("file_name", "Unknown")
                results.append(f"Document {i} (Source: {source}):\n{content}\n")
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error retrieving information from Qdrant: {str(e)}"


@tool
def retrieve_campaign_briefs(
    query: str,
    collection_name: Optional[str] = None,
    top_k: int = 3,
    task_type_filter: Optional[str] = None
) -> str:
    """
    Retrieve similar campaign briefs (spreadsheets) from the Qdrant database.
    This is a specialized tool for finding similar campaign briefs based on task type, dealership, or campaign content.
    
    Use this tool when you need to:
    - Find similar campaign briefs for reference
    - Look up examples of specific task types (Theme, New Creative, Campaign Update)
    - Get context from previous campaign briefs for a specific dealership
    
    Args:
        query: Search query (e.g., "New Creative campaigns for dealership X", "Theme campaigns")
        collection_name: Optional name of the Qdrant collection to search (defaults to env var)
        top_k: Number of most relevant briefs to retrieve (default: 3)
        task_type_filter: Optional filter by task type ("Theme", "New Creative", "Campaign Update")
        
    Returns:
        A string containing structured information about similar campaign briefs
    """
    if not QDRANT_AVAILABLE:
        return "Error: Qdrant dependencies are not installed."
    
    vectorstore = _get_qdrant_vectorstore(collection_name)
    
    if not vectorstore:
        return "Error: Could not connect to Qdrant vector database."
    
    try:
        # Get Qdrant client directly for better payload access
        client = _get_qdrant_client()
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "my_rag_collection")
        
        # Use vectorstore for embedding the query
        query_vector = vectorstore.embeddings.embed_query(query)
        
        # Filter for campaign briefs only
        search_filter = Filter(
            must=[FieldCondition(
                key="file_type",
                match=MatchValue(value="campaign_brief")
            )]
        )
        
        # Add task type filter if specified
        if task_type_filter:
            search_filter.must.append(
                FieldCondition(
                    key="structured_brief.task_type",
                    match=MatchValue(value=task_type_filter)
                )
            )
        
        # Search directly with Qdrant client using query_points
        search_results = client.query_points(
            collection_name=collection_name_actual,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )
        
        # Extract points from the response
        if hasattr(search_results, 'points'):
            points = search_results.points
        else:
            points = []
        
        if not points:
            filter_msg = f" (filtered by task_type: {task_type_filter})" if task_type_filter else ""
            return f"No relevant campaign briefs found for query: '{query}'{filter_msg}"
        
        # Format the retrieved briefs
        filtered_briefs = []
        for result in points:
            payload = result.payload or {}
            structured_brief = payload.get("structured_brief")
            campaign_json = payload.get("campaign_json")
            brief_metadata = payload.get("brief_metadata")
            
            # Accept either old format (structured_brief) or new format (campaign_json + brief_metadata)
            if structured_brief and isinstance(structured_brief, dict):
                filtered_briefs.append(("old", result, structured_brief))
            elif campaign_json and isinstance(campaign_json, dict):
                filtered_briefs.append(("new", result, campaign_json, brief_metadata))
        
        if not filtered_briefs:
            filter_msg = f" (filtered by task_type: {task_type_filter})" if task_type_filter else ""
            return f"No relevant campaign briefs found for query: '{query}'{filter_msg}"
        
        # Format the retrieved briefs
        results = []
        for i, brief_data in enumerate(filtered_briefs, 1):
            brief_info = []
            brief_info.append(f"Similar Campaign Brief {i}:")
            
            if brief_data[0] == "old":
                # Old format: structured_brief
                _, result, structured_brief = brief_data
                brief_info.append(f"  File Name: {structured_brief.get('file_name', 'Unknown')}")
                brief_info.append(f"  Task Type: {structured_brief.get('task_type', 'Unknown')}")
                brief_info.append(f"  Dealership: {structured_brief.get('dealership_name', 'Unknown')}")
                brief_info.append(f"  Total Campaigns: {structured_brief.get('total_campaigns', len(structured_brief.get('campaigns', [])))}")
                brief_info.append(f"  Asset Summary: {structured_brief.get('asset_summary', 'N/A')}")
                
                # Include campaign details
                campaigns = structured_brief.get('campaigns', [])
                if campaigns:
                    brief_info.append(f"  Campaigns:")
                    for camp in campaigns:
                        if camp and isinstance(camp, dict):
                            brief_info.append(f"    - {camp.get('campaign_id', 'Unknown')}: {camp.get('headline', 'N/A')}")
            else:
                # New format: campaign_json + brief_metadata
                _, result, campaign_json, brief_metadata = brief_data
                if brief_metadata and isinstance(brief_metadata, dict):
                    brief_info.append(f"  File Name: {brief_metadata.get('file_name', 'Unknown')}")
                    brief_info.append(f"  Task Type: {brief_metadata.get('task_type', 'Unknown')}")
                    brief_info.append(f"  Dealership: {brief_metadata.get('dealership_name', 'Unknown')}")
                brief_info.append(f"  Campaign ID: {campaign_json.get('campaign_id', 'Unknown')}")
                offer_details = campaign_json.get('offer_details', {})
                if offer_details and isinstance(offer_details, dict):
                    brief_info.append(f"  Headline: {offer_details.get('headline', 'N/A')}")
            
            results.append("\n".join(brief_info))
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Error retrieving campaign briefs: {str(e)}"


@tool
def find_similar_campaigns(
    task_type: str,
    dealership_name: Optional[str] = None,
    asset_summary: Optional[str] = None,
    top_k: int = 5,
    collection_name: Optional[str] = None
) -> str:
    """
    Find similar campaign briefs based on metadata filters.
    This tool searches for campaign briefs in Qdrant using a two-step approach:
    1. First retrieves campaigns by task_type (with optional semantic search on asset_summary)
    2. Then optionally filters by dealership_name - if no dealership matches are found, returns the original task_type results
    
    Use this tool when you need to:
    - Find similar campaign briefs for evaluation or comparison
    - Look up campaigns with the same task type, optionally filtered by dealership
    - Get references to similar campaigns before fetching their full data
    
    Args:
        task_type: Task type to filter by (e.g., "Theme", "New Creative", "Campaign Update")
        dealership_name: Optional dealership name to filter by. If provided, will filter results by dealership,
                        but if no matches are found, will return the original task_type results instead.
        asset_summary: Optional asset summary text for semantic search (e.g., "SL: 5, BN: 3")
        top_k: Number of most similar briefs to return (default: 5)
        collection_name: Optional name of the Qdrant collection to search (defaults to env var)
        
    Returns:
        A string containing metadata about similar campaign briefs, including:
        - File name, task type, dealership, asset summary
        - Total campaigns and campaign IDs
        - Google Drive file_id for fetching full data
    """
    if not QDRANT_AVAILABLE:
        return "Error: Qdrant dependencies are not installed."
    
    client = _get_qdrant_client()
    if not client:
        return "Error: Could not connect to Qdrant vector database."
    
    try:
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "my_rag_collection")
        
        # Build filter for metadata-based search (task_type only first)
        # Step 1: Filter by task_type only (and file_type)
        base_filter_conditions = [
            FieldCondition(
                key="file_type",
                match=MatchValue(value="campaign_metadata")
            ),
            FieldCondition(
                key="task_type",
                match=MatchValue(value=task_type)
            )
        ]
        
        base_search_filter = Filter(must=base_filter_conditions)
        
        # Step 2: Retrieve campaigns by task_type (with semantic search if asset_summary provided)
        if asset_summary:
            # Use vectorstore for embedding the asset_summary query
            vectorstore = _get_qdrant_vectorstore(collection_name_actual)
            if vectorstore:
                query_vector = vectorstore.embeddings.embed_query(asset_summary)
                
                # Search with task_type filter only
                search_results = client.query_points(
                    collection_name=collection_name_actual,
                    query=query_vector,
                    query_filter=base_search_filter,
                    limit=top_k * 2,  # Get more results to allow for dealership filtering
                    with_payload=True
                )
            else:
                # Fallback: scroll with filter only (no semantic search)
                search_results = client.scroll(
                    collection_name=collection_name_actual,
                    scroll_filter=base_search_filter,
                    limit=top_k * 2,
                    with_payload=True
                )
        else:
            # No semantic search, just filter-based retrieval by task_type
            search_results = client.scroll(
                collection_name=collection_name_actual,
                scroll_filter=base_search_filter,
                limit=top_k * 2,
                with_payload=True
            )
        
        # Extract points from the response
        if hasattr(search_results, 'points'):
            points = search_results.points
        elif isinstance(search_results, tuple):
            points, _ = search_results
        else:
            points = []
        
        if not points:
            return f"No similar campaign briefs found matching task_type='{task_type}'"
        
        # Step 3: If dealership_name is provided, filter results by dealership
        # If no results match dealership, use the original results
        original_points = points
        if dealership_name:
            filtered_points = [
                point for point in points
                if point.payload and point.payload.get('dealership_name') == dealership_name
            ]
            
            # If dealership filter found results, use them; otherwise use original
            if filtered_points:
                points = filtered_points[:top_k]  # Limit to top_k
            else:
                # No matches for dealership, use original results
                points = original_points[:top_k]
        else:
            # No dealership filter, just limit to top_k
            points = points[:top_k]
        
        # Format the results
        results = []
        for i, point in enumerate(points, 1):
            payload = point.payload or {}
            
            brief_info = []
            brief_info.append(f"Similar Campaign Brief {i}:")
            brief_info.append(f"  File Name: {payload.get('file_name', 'Unknown')}")
            brief_info.append(f"  File ID (Google Drive): {payload.get('file_id', 'Unknown')}")
            brief_info.append(f"  Task Type: {payload.get('task_type', 'Unknown')}")
            brief_info.append(f"  Dealership: {payload.get('dealership_name', 'Unknown')}")
            brief_info.append(f"  Asset Summary: {payload.get('asset_summary', 'N/A')}")
            brief_info.append(f"  Total Campaigns: {payload.get('total_campaigns', 0)}")
            
            campaign_ids = payload.get('campaign_ids', [])
            if campaign_ids:
                brief_info.append(f"  Campaign IDs: {', '.join(campaign_ids[:10])}")
                if len(campaign_ids) > 10:
                    brief_info.append(f"    ... and {len(campaign_ids) - 10} more")
            
            brief_info.append(f"  Folder Path: {payload.get('folder_path', 'N/A')}")
            brief_info.append(f"  Modified: {payload.get('file_modified_time', 'Unknown')}")
            brief_info.append("")
            brief_info.append("  Use fetch_campaign_brief_from_drive(file_id) to get full campaign data.")
            
            results.append("\n".join(brief_info))
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error finding similar campaigns: {str(e)}"


@tool
def find_similar_diagnoses(
    task_type: str,
    status: Optional[str] = None,
    dealership_name: Optional[str] = None,
    query: Optional[str] = None,
    top_k: int = 5,
    collection_name: Optional[str] = None
) -> str:
    """
    Find similar past diagnoses based on filters and optional semantic search query.
    This tool searches for diagnosis spreadsheets in Qdrant that match the criteria.
    
    Use this tool when you need to:
    - Find similar past diagnoses for a specific task type to compare against current diagnoses
    - Look up how similar diagnoses were evaluated in the past for consistency
    - Retrieve past diagnosis examples to learn from QA decisions and patterns
    
    Args:
        task_type: Task type to filter by (required) - e.g., "Theme", "New Creative", "Campaign Update"
        status: Optional status to filter by - if provided, finds diagnoses containing this status
               (e.g., "critical", "observed", "passed")
        dealership_name: Optional dealership name to filter by - finds diagnoses for the same dealership
        query: Optional semantic search query for finding relevant diagnoses by content
               (e.g., "missing headline campaigns" or "asset compliance issues")
        top_k: Number of most similar diagnoses to return (default: 5)
        collection_name: Optional name of the Qdrant collection to search (defaults to env var)
        
    Returns:
        A string containing metadata about similar past diagnoses, including:
        - File name, task type, dealership, statuses, total diagnoses
        - Diagnosis date, QA result, campaign IDs
        - Google Drive file_id for fetching full data
    """
    if not QDRANT_AVAILABLE:
        return "Error: Qdrant dependencies are not installed."
    
    client = _get_qdrant_client()
    if not client:
        return "Error: Could not connect to Qdrant vector database."
    
    try:
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "my_rag_collection")
        
        # Build filter for diagnosis search
        # Base filter: file_type must be "campaign_diagnosis" and task_type must match
        base_filter_conditions = [
            FieldCondition(
                key="file_type",
                match=MatchValue(value="campaign_diagnosis")
            ),
            FieldCondition(
                key="task_type",
                match=MatchValue(value=task_type)
            )
        ]
        
        # Optional: filter by dealership_name
        if dealership_name:
            base_filter_conditions.append(
                FieldCondition(
                    key="dealership_name",
                    match=MatchValue(value=dealership_name)
                )
            )
        
        # Note: status filtering is complex because statuses is stored as a list in payload
        # For now, we'll do semantic search if query is provided, otherwise filter-based retrieval
        # Status filtering could be added later with array matching if needed
        
        base_search_filter = Filter(must=base_filter_conditions)
        
        # If query is provided, use semantic search; otherwise use filter-based scroll
        if query:
            # Use vectorstore for embedding the query
            vectorstore = _get_qdrant_vectorstore(collection_name_actual)
            if vectorstore:
                query_vector = vectorstore.embeddings.embed_query(query)
                
                # Search with filters
                search_results = client.query_points(
                    collection_name=collection_name_actual,
                    query=query_vector,
                    query_filter=base_search_filter,
                    limit=top_k,
                    with_payload=True
                )
            else:
                # Fallback: scroll with filter only (no semantic search)
                search_results = client.scroll(
                    collection_name=collection_name_actual,
                    scroll_filter=base_search_filter,
                    limit=top_k,
                    with_payload=True
                )
        else:
            # No semantic search, just filter-based retrieval
            search_results = client.scroll(
                collection_name=collection_name_actual,
                scroll_filter=base_search_filter,
                limit=top_k,
                with_payload=True
            )
        
        # Extract points from the response
        if hasattr(search_results, 'points'):
            points = search_results.points
        elif isinstance(search_results, tuple):
            points, _ = search_results
        else:
            points = []
        
        if not points:
            filter_msg = f" (task_type='{task_type}'"
            if dealership_name:
                filter_msg += f", dealership='{dealership_name}'"
            if status:
                filter_msg += f", status='{status}'"
            filter_msg += ")"
            return f"No similar diagnoses found{filter_msg}"
        
        # Filter by status if provided (check if status appears in statuses list)
        if status:
            filtered_points = []
            for point in points:
                payload = point.payload or {}
                statuses_list = payload.get("statuses", [])
                if status in statuses_list:
                    filtered_points.append(point)
            
            if filtered_points:
                points = filtered_points[:top_k]
            # If no matches, still return the original results (might be useful)
        
        # Format the results
        results = []
        for i, point in enumerate(points[:top_k], 1):
            payload = point.payload or {}
            
            brief_info = []
            brief_info.append(f"Similar Diagnosis Record {i}:")
            brief_info.append(f"  File Name: {payload.get('file_name', 'Unknown')}")
            brief_info.append(f"  File ID (Google Drive): {payload.get('file_id', 'Unknown')}")
            brief_info.append(f"  Task Type: {payload.get('task_type', 'Unknown')}")
            brief_info.append(f"  Dealership: {payload.get('dealership_name', 'Unknown')}")
            brief_info.append(f"  Statuses: {', '.join(payload.get('statuses', []))}")
            brief_info.append(f"  Total Diagnoses: {payload.get('total_diagnoses', 0)}")
            brief_info.append(f"  Diagnosis Date: {payload.get('diagnosis_date', 'Unknown')}")
            brief_info.append(f"  QA Result: {payload.get('qa_result', 'Unknown')}")
            
            campaign_ids = payload.get('campaign_ids', [])
            if campaign_ids:
                brief_info.append(f"  Campaign IDs: {', '.join(campaign_ids[:10])}")
                if len(campaign_ids) > 10:
                    brief_info.append(f"    ... and {len(campaign_ids) - 10} more")
            
            brief_info.append(f"  Folder Path: {payload.get('folder_path', 'N/A')}")
            brief_info.append(f"  Modified: {payload.get('file_modified_time', 'Unknown')}")
            brief_info.append("")
            brief_info.append("  Use fetch_campaign_brief_from_drive(file_id) to get full diagnosis data.")
            
            results.append("\n".join(brief_info))
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error finding similar diagnoses: {str(e)}"


@tool
def fetch_campaign_brief_from_drive(file_id: str) -> str:
    """
    Fetch full campaign brief data from Google Drive using file_id.
    This tool downloads the spreadsheet from Google Drive, parses it, and returns
    the complete CampaignBrief JSON with all campaign details.
    
    Use this tool when you need to:
    - Get full campaign data after finding similar campaigns with find_similar_campaigns()
    - Retrieve complete campaign details for evaluation or comparison
    - Access all campaign information that isn't stored in Qdrant metadata
    
    Args:
        file_id: Google Drive file ID of the campaign brief spreadsheet
        
    Returns:
        A JSON string containing the full CampaignBrief structure with all campaigns,
        including offer details, assets, style descriptions, etc.
    """
    try:
        # Import Google Drive helpers from rag_ingestion if not already imported
        if get_drive_service is None or download_file_content is None:
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            # Import dynamically if top-level import failed
            # Note: This is a fallback for when top-level import fails
            # Using __import__ to avoid import statement inside function
            rag_ingestion_module = __import__('rag_ingestion', fromlist=['get_drive_service', 'download_file_content'])
            get_drive_service = rag_ingestion_module.get_drive_service
            download_file_content = rag_ingestion_module.download_file_content
        
        # Get Google Drive service
        drive_service = get_drive_service()
        
        # Download the file
        content = download_file_content(drive_service, file_id, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # Save to temporary file and parse
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
                tmp.write(content)
                temp_file = tmp.name
            
            # Use _parse_spreadsheet_internal to get full campaign data (not the tool wrapper)
            campaign_brief = _parse_spreadsheet_internal(temp_file)
            
            # Return as JSON string
            return json.dumps(campaign_brief, indent=2)
        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
    
    except Exception as e:
        return f"Error fetching campaign brief from Google Drive (file_id={file_id}): {str(e)}"


@tool
def identify_previous_campaign_id(
    campaigns: List[Dict[str, Any]]
) -> str:
    """
    Identify the previous campaign ID from the current brief's campaigns by extracting it from
    the Asset Style Direction or Additional Style Information fields.
    
    This tool searches through all campaigns in the current brief and looks for previous campaign ID
    references (e.g., "CU: A-12345678" or "A-12345678") in the style description fields.
    
    Use this tool FIRST when evaluating Campaign Update campaigns to identify the previous campaign ID
    that the current brief references. This should be done BEFORE matching campaigns and BEFORE calling RAG.
    
    IMPORTANT: 
    - If multiple unique IDs are found across campaigns, a flag will be raised.
    - If no reference is found, a flag will be raised and evaluation should continue with a warning.
    - Only after extracting the ID and matching campaigns should RAG be called for additional rules.
    
    Args:
        campaigns: List of campaign dictionaries from the current brief. Each campaign should have
                   a "style_descriptions" field containing "asset_style_direction" and 
                   "additional_style_information" fields.
        
    Returns:
        The previous campaign ID in the format "A-XXXXXXXX" if found, or a flag/warning message
        if multiple IDs found or no ID found.
    """
    try:
        if not campaigns:
            return "FLAG: No campaigns provided to identify previous campaign ID."
        
        # Handle campaigns passed as JSON string
        if isinstance(campaigns, str):
            try:
                campaigns = json.loads(campaigns)
            except json.JSONDecodeError:
                return f"Error: Could not parse campaigns as JSON. Received: {str(campaigns)[:200]}"
        
        # Ensure campaigns is a list
        if not isinstance(campaigns, list):
            return f"Error: campaigns must be a list. Received type: {type(campaigns).__name__}"
        
        # Patterns to match campaign IDs
        # Look for "CU: A-12345678" or just "A-12345678" or similar patterns
        patterns = [
            r'CU:\s*([A-Z]-\d{7,9})',  # CU: A-12345678
            r'([A-Z]-\d{8})',  # A-12345678 (8 digits)
            r'([A-Z]-\d{7,9})',  # A-1234567 or A-123456789 (flexible)
        ]
        
        found_ids = []
        
        # Search through all campaigns
        for campaign in campaigns:
            # Handle both dict and Campaign object
            if isinstance(campaign, dict):
                style_desc = campaign.get("style_descriptions", {})
            else:
                # Campaign object
                style_desc = campaign.style_descriptions if hasattr(campaign, 'style_descriptions') else {}
            
            # Get the fields to search
            if isinstance(style_desc, dict):
                asset_style = style_desc.get("asset_style_direction", "") or ""
                additional_style = style_desc.get("additional_style_information", "") or ""
            else:
                # StyleDescriptions object
                asset_style = getattr(style_desc, "asset_style_direction", "") or ""
                additional_style = getattr(style_desc, "additional_style_information", "") or ""
            
            # Combine both fields for searching
            text_to_search = f"{asset_style} {additional_style}".strip()
            
            if not text_to_search:
                continue
            
            # Search for ID patterns
            for pattern in patterns:
                matches = re.findall(pattern, text_to_search, re.IGNORECASE)
                if matches:
                    # re.findall returns tuples for patterns with groups, strings for patterns without groups
                    for match in matches:
                        if isinstance(match, tuple):
                            # Extract the group (the ID part)
                            found_ids.extend([m for m in match if m])
                        else:
                            # Direct match (string)
                            found_ids.append(match)
        
        if not found_ids:
            return (
                "FLAG: No previous campaign ID reference found in any campaign's Asset Style Direction "
                "or Additional Style Information fields. Evaluation will continue, but the previous "
                "campaign ID could not be automatically identified. Please check the campaign fields manually."
            )
        
        # Normalize IDs (uppercase) and remove duplicates
        seen = set()
        unique_ids = []
        for id_val in found_ids:
            normalized = id_val.upper().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_ids.append(normalized)
        
        if len(unique_ids) > 1:
            return (
                f"FLAG: Multiple unique previous campaign IDs found across campaigns: {', '.join(unique_ids)}\n"
                f"This may indicate inconsistent references. Using the first ID found: {unique_ids[0]}\n\n"
                f"Previous Campaign ID: {unique_ids[0]}"
            )
        
        # Single unique ID found
        previous_campaign_id = unique_ids[0]
        return f"Previous Campaign ID identified: {previous_campaign_id}"
    
    except Exception as e:
        return f"Error identifying previous campaign ID: {str(e)}"


@tool
def find_and_load_previous_campaign_brief(
    previous_campaign_id: str,
    task_type: Optional[str] = None,
    for_main_brief: bool = True
) -> str:
    """
    Find a campaign brief file that contains the previous campaign ID in its filename,
    then load and parse it to create a campaign brief.
    
    This tool FIRST searches locally in the project directory for .xlsx files containing the 
    previous campaign ID.
    
    For the main campaign brief's previous version (for_main_brief=True):
    - Searches ONLY locally in the project directory
    - If not found locally, returns a CRITICAL ERROR immediately (does NOT search Google Drive)
    - Google Drive search is NOT allowed for the main brief's previous version
    
    For similar briefs' previous versions (for_main_brief=False):
    - Searches locally first
    - If not found locally, then searches in Google Drive folders
    
    Local search: Looks in the project root directory for .xlsx files.
    Google Drive search (only when for_main_brief=False):
    - For "Campaign Update": Searches in Campaigns/Campaign Update/Previous/
    - For other task types: Searches in Campaigns/[Task Type]/ folders
    
    Use this tool after identifying the previous campaign ID to load the original campaign brief
    that the Campaign Update campaigns are referencing.
    
    Args:
        previous_campaign_id: The previous campaign ID to search for (e.g., "A-12345678")
        task_type: Optional task type to narrow Google Drive search (e.g., "New Creative", "Theme", "Campaign Update").
                   If not provided, searches all task type folders. For "Campaign Update", automatically
                   searches in the "Previous" subfolder.
        for_main_brief: If True (default), only searches locally and returns error if not found.
                       If False, searches locally first, then Google Drive if not found locally.
        
    Returns:
        A JSON string containing the full CampaignBrief structure with all campaigns from the
        previous campaign brief file. Returns an error if the file cannot be found or parsed.
    """
    try:
        # FIRST: Search locally in the project directory
        project_root = Path(__file__).parent.parent
        local_xlsx_files = list(project_root.glob("*.xlsx"))
        
        previous_id_normalized = str(previous_campaign_id or "").strip().upper()
        # Use shared filename parser first (A-/D- aware), then keep legacy substring fallback.
        matching_local_files = []
        for local_file in local_xlsx_files:
            parsed_name = parse_brief_filename(local_file.name)
            parsed_token = str(parsed_name.get("campaign_token", "")).upper() if parsed_name else ""
            if parsed_token and parsed_token == previous_id_normalized:
                matching_local_files.append(local_file)
                continue
            if previous_id_normalized and previous_id_normalized in local_file.name.upper():
                matching_local_files.append(local_file)
        
        if matching_local_files:
            # Use the first matching local file
            local_file = matching_local_files[0]
            print(f"[Previous Campaign] Found file locally: {local_file.name}")
            
            # Parse the local file
            campaign_brief = _parse_spreadsheet_internal(str(local_file))
            
            # Return as JSON string
            return json.dumps(campaign_brief, indent=2)
        
        # If not found locally
        if for_main_brief:
            # For main brief: return CRITICAL ERROR immediately, do NOT search Google Drive
            return (
                f"CRITICAL ERROR: No matching file found locally for previous campaign ID '{previous_campaign_id}'.\n"
                f"Previous Campaign ID: {previous_campaign_id}\n"
                f"Searched locally in project directory: Not found\n\n"
                f"EVALUATION CANNOT PROCEED: The main campaign brief's previous version must be found locally "
                f"in the project directory. Google Drive search is not allowed for the main brief's previous version.\n"
                f"Please ensure the previous campaign brief file exists locally with the campaign ID '{previous_campaign_id}' in its filename."
            )
        
        # For similar briefs: if not found locally, search in Google Drive
        print(f"[Previous Campaign] Not found locally, searching Google Drive for ID: {previous_campaign_id}")
        
        # Import Google Drive helpers from rag_ingestion
        if get_drive_service is None or download_file_content is None:
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            # Import dynamically if top-level import failed
            rag_ingestion_module = __import__('rag_ingestion', fromlist=['get_drive_service', 'download_file_content', 'find_folder_by_name', 'list_files_in_folder'])
            get_drive_service_func = rag_ingestion_module.get_drive_service
            download_file_content_func = rag_ingestion_module.download_file_content
            find_folder_by_name = rag_ingestion_module.find_folder_by_name
            list_files_in_folder = rag_ingestion_module.list_files_in_folder
        else:
            get_drive_service_func = get_drive_service
            download_file_content_func = download_file_content
            # Import find_folder_by_name and list_files_in_folder
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            rag_ingestion_module = __import__('rag_ingestion', fromlist=['find_folder_by_name', 'list_files_in_folder'])
            find_folder_by_name = rag_ingestion_module.find_folder_by_name
            list_files_in_folder = rag_ingestion_module.list_files_in_folder
        
        if get_drive_service_func is None or download_file_content_func is None:
            return f"ERROR: Could not import Google Drive service functions. Previous Campaign ID: {previous_campaign_id}"
        
        # Get Google Drive service
        drive_service = get_drive_service_func()
        
        # Get main Drive folder ID from environment
        main_drive_folder_id = os.getenv("CAMPAIGNS_DRIVE_FOLDER_ID")
        if not main_drive_folder_id:
            return (
                f"ERROR: CAMPAIGNS_DRIVE_FOLDER_ID environment variable is not set.\n"
                f"Previous Campaign ID: {previous_campaign_id}"
            )
        
        # Find "Campaigns" folder
        campaigns_folder = find_folder_by_name(drive_service, main_drive_folder_id, "Campaigns")
        if not campaigns_folder:
            return (
                f"ERROR: 'Campaigns' folder not found in Google Drive.\n"
                f"Previous Campaign ID: {previous_campaign_id}"
            )
        
        campaigns_folder_id = campaigns_folder["id"]
        
        # Task type folders to search
        if task_type:
            task_type_folders = [task_type]
        else:
            task_type_folders = ["New Creative", "Theme", "Campaign Update"]
        
        xlsx_mime_type = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        matching_files = []
        
        # Search through task type folders
        for task_type_name in task_type_folders:
            task_type_folder = find_folder_by_name(drive_service, campaigns_folder_id, task_type_name)
            if not task_type_folder:
                continue
            
            task_type_folder_id = task_type_folder["id"]
            
            # For Campaign Update, search in "Previous" subfolder (where previous campaign briefs are stored)
            if task_type_name == "Campaign Update":
                previous_folder = find_folder_by_name(drive_service, task_type_folder_id, "Previous")
                if previous_folder:
                    previous_folder_id = previous_folder["id"]
                    files = list_files_in_folder(drive_service, previous_folder_id, file_types=xlsx_mime_type)
                    matching_files.extend([(f, f"{task_type_name}/Previous") for f in files])
                else:
                    # If Previous folder not found, log a warning but continue
                    print(f"[Previous Campaign] Warning: 'Previous' folder not found in 'Campaign Update', searching in main folder")
                    # Fallback: search in the task type folder itself
                    files = list_files_in_folder(drive_service, task_type_folder_id, file_types=xlsx_mime_type)
                    matching_files.extend([(f, task_type_name) for f in files])
            else:
                # For other task types, search in the task type folder itself
                files = list_files_in_folder(drive_service, task_type_folder_id, file_types=xlsx_mime_type)
                matching_files.extend([(f, task_type_name) for f in files])
        
        if not matching_files:
            return (
                f"ERROR: No campaign brief files found in Google Drive.\n"
                f"Previous Campaign ID: {previous_campaign_id}\n"
                f"Searched in: Campaigns/{', '.join(task_type_folders)}/"
            )
        
        # Find file(s) containing the previous campaign ID in the filename
        matching_files_with_id = []
        for file_meta, folder_path in matching_files:
            file_name = str(file_meta.get("name", ""))
            parsed_name = parse_brief_filename(file_name)
            parsed_token = str(parsed_name.get("campaign_token", "")).upper() if parsed_name else ""
            if parsed_token and parsed_token == previous_id_normalized:
                matching_files_with_id.append((file_meta, folder_path))
                continue
            if previous_id_normalized and previous_id_normalized in file_name.upper():
                matching_files_with_id.append((file_meta, folder_path))
        
        if not matching_files_with_id:
            # Show some example filenames for debugging
            example_files = [f["name"] for f, _ in matching_files[:5]]
            return (
                f"CRITICAL ERROR: No matching file found for previous campaign ID '{previous_campaign_id}'.\n"
                f"Previous Campaign ID: {previous_campaign_id}\n"
                f"Searched locally in project directory: Not found\n"
                f"Searched in Google Drive: Found {len(matching_files)} .xlsx file(s) but none match the campaign ID.\n"
                f"Example files from Drive: {example_files}\n\n"
                f"EVALUATION CANNOT PROCEED: Without the previous campaign brief, it is impossible to determine "
                f"which changes are from the previous version. The previous campaign brief file must be found "
                f"locally in the project directory or in Google Drive with the campaign ID '{previous_campaign_id}' in its filename.\n"
                f"Please ensure the previous campaign brief exists locally or in the correct Google Drive folder structure."
            )
        
        # Use the first matching file (or most recent if multiple)
        if len(matching_files_with_id) > 1:
            # Sort by modified time, most recent first
            matching_files_with_id.sort(
                key=lambda x: x[0].get("modifiedTime", ""), 
                reverse=True
            )
        
        target_file, folder_path = matching_files_with_id[0]
        file_id = target_file["id"]
        file_name = target_file["name"]
        
        print(f"[Previous Campaign] Found file in Google Drive: {file_name} (id={file_id}, folder={folder_path})")
        
        # Download and parse the file using fetch_campaign_brief_from_drive logic
        content = download_file_content_func(drive_service, file_id, target_file["mimeType"])
        
        # Save to temporary file and parse
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
                tmp.write(content)
                temp_file = tmp.name
            
            # Use _parse_spreadsheet_internal to get full campaign data
            campaign_brief = _parse_spreadsheet_internal(temp_file)
            
            # Return as JSON string
            return json.dumps(campaign_brief, indent=2)
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass  # Ignore cleanup errors
    
    except Exception as e:
        error_details = traceback.format_exc()
        return (
            f"Error finding and loading previous campaign brief (ID: {previous_campaign_id}): {str(e)}\n"
            f"Details: {error_details}"
        )


@tool
def match_campaigns_by_headline(
    current_campaign_brief_json: str,
    previous_campaign_brief_json: str
) -> str:
    """
    Match campaigns between the current campaign brief (A) and the previous campaign brief (B) by their headline.
    
    This tool compares campaigns from both briefs and creates a filtered set of matching campaigns based on
    their headline text (case-insensitive, trimmed). This should be used after loading the previous campaign brief
    to identify which campaigns in the current brief correspond to which campaigns in the previous brief.
    
    Args:
        current_campaign_brief_json: JSON string of the current campaign brief (brief A) being processed
        previous_campaign_brief_json: JSON string of the previous campaign brief (brief B) loaded from the campaign ID
        
    Returns:
        A JSON string containing:
        - matched_campaigns: List of matched campaign pairs with current and previous campaign data
        - unmatched_current: List of campaigns from current brief that had no match
        - unmatched_previous: List of campaigns from previous brief that had no match
        - summary: Statistics about the matching process
    """
    def _parse_brief_payload(payload: Any, field_name: str) -> Dict[str, Any]:
        """Parse a brief payload that may arrive as dict or imperfect JSON-like string."""
        if isinstance(payload, dict):
            return payload
        if not isinstance(payload, str):
            raise ValueError(f"{field_name} must be a dict or string payload")

        text = payload.strip()
        if not text:
            raise ValueError(f"{field_name} is empty")

        # Primary parse path (proper JSON).
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Retry for malformed Windows-path escapes in JSON strings.
        if "\\" in text:
            try:
                escaped = text.replace("\\", "\\\\")
                parsed = json.loads(escaped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Last-chance parse for Python-dict-like payload strings.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        raise ValueError(f"Could not parse {field_name} as a brief object")

    try:
        # Parse both briefs
        current_brief = _parse_brief_payload(current_campaign_brief_json, "current_campaign_brief_json")
        previous_brief = _parse_brief_payload(previous_campaign_brief_json, "previous_campaign_brief_json")
        
        current_campaigns = current_brief.get("campaigns", [])
        previous_campaigns = previous_brief.get("campaigns", [])
        
        # Normalize headline for comparison (case-insensitive, trimmed)
        def normalize_headline(headline: str) -> str:
            if not headline:
                return ""
            return headline.strip().lower()
        
        # Build a map of normalized headline -> list of previous campaigns (in case of duplicates)
        previous_headline_map: Dict[str, List[Dict[str, Any]]] = {}
        for prev_campaign in previous_campaigns:
            headline = prev_campaign.get("offer_details", {}).get("headline", "")
            normalized = normalize_headline(headline)
            if normalized:
                if normalized not in previous_headline_map:
                    previous_headline_map[normalized] = []
                previous_headline_map[normalized].append(prev_campaign)
        
        # Match current campaigns with previous campaigns
        matched_campaigns: List[Dict[str, Any]] = []
        matched_previous_indices = set()
        
        for curr_campaign in current_campaigns:
            curr_headline = curr_campaign.get("offer_details", {}).get("headline", "")
            normalized_curr = normalize_headline(curr_headline)
            
            if normalized_curr and normalized_curr in previous_headline_map:
                # Found a match - use the first available previous campaign with this headline
                for prev_campaign in previous_headline_map[normalized_curr]:
                    prev_campaign_id = prev_campaign.get("campaign_id", "")
                    # Check if we haven't already matched this previous campaign
                    if prev_campaign not in [m.get("previous_campaign") for m in matched_campaigns]:
                        matched_campaigns.append({
                            "current_campaign": curr_campaign,
                            "previous_campaign": prev_campaign,
                            "headline": curr_headline,  # Original headline (not normalized)
                            "match_type": "headline"
                        })
                        break
            else:
                # No match found for this current campaign
                pass
        
        # Find unmatched campaigns
        matched_previous_ids = {m["previous_campaign"].get("campaign_id", "") for m in matched_campaigns}
        unmatched_current = [
            curr for curr in current_campaigns
            if normalize_headline(curr.get("offer_details", {}).get("headline", "")) not in previous_headline_map
        ]
        unmatched_previous = [
            prev for prev in previous_campaigns
            if prev.get("campaign_id", "") not in matched_previous_ids
        ]
        
        # Create summary
        summary = {
            "total_current_campaigns": len(current_campaigns),
            "total_previous_campaigns": len(previous_campaigns),
            "matched_count": len(matched_campaigns),
            "unmatched_current_count": len(unmatched_current),
            "unmatched_previous_count": len(unmatched_previous),
            "match_rate": f"{len(matched_campaigns) / len(current_campaigns) * 100:.1f}%" if current_campaigns else "0%"
        }
        
        result = {
            "matched_campaigns": matched_campaigns,
            "unmatched_current": unmatched_current,
            "unmatched_previous": unmatched_previous,
            "summary": summary
        }
        
        print(f"[Campaign Matching] Matched {len(matched_campaigns)} campaigns by headline")
        print(f"  Current campaigns: {len(current_campaigns)}, Previous campaigns: {len(previous_campaigns)}")
        print(f"  Unmatched current: {len(unmatched_current)}, Unmatched previous: {len(unmatched_previous)}")
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        error_details = traceback.format_exc()
        return (
            f"Error matching campaigns by headline: {str(e)}\n"
            f"Details: {error_details}"
        )


@tool
def store_text_document_to_drive(
    content: str,
    file_name: str,
    task_type: str,
    drive_service: Optional[Any] = None,
    main_drive_folder_id: Optional[str] = None
) -> str:
    """
    Store a text document to Google Drive in the appropriate folder structure.
    
    Stores to Campaigns/[Task Type]/Diagnoses/ folder (same as Excel diagnoses).
    
    Args:
        content: The text content of the document to store
        file_name: The name of the file (e.g., 'spreadsheet-brief-resume.txt')
        task_type: Task type for folder structure (e.g., 'Theme', 'New Creative', 'Campaign Update')
        drive_service: Optional Google Drive service (will be created if None)
        main_drive_folder_id: Optional main Drive folder ID (defaults to env var CAMPAIGNS_DRIVE_FOLDER_ID)
    
    Returns:
        JSON string with file_id, file_name, and folder_path from Google Drive
    """
    try:
        # Import store_text_document_to_drive from rag_ingestion
        if rag_ingestion is None:
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            import rag_ingestion as ri
        else:
            ri = rag_ingestion
        
        store_func = getattr(ri, 'store_text_document_to_drive', None)
        if store_func is None:
            return json.dumps({
                "error": "store_text_document_to_drive function not found in rag_ingestion module"
            })
        
        # Call the function
        result = store_func(
            content=content,
            file_name=file_name,
            task_type=task_type,
            drive_service=drive_service,
            main_drive_folder_id=main_drive_folder_id
        )
        
        return json.dumps(result)
    
    except Exception as e:
        error_details = traceback.format_exc()
        return json.dumps({
            "error": f"Error storing text document to Drive: {str(e)}",
            "details": error_details
        })


@tool
def write_document_to_file(
    content: str,
    file_name: str,
    document_type: str
) -> str:
    """
    Write a text document to the local `results` directory in the project.
    
    This tool is used by the Document Creator Agent to save the Brief Resume and Full Diagnoses Listing documents.
    
    Args:
        content: The complete text content of the document to write
        file_name: The name of the file (e.g., '2025-07-rogerbeasleyvolvovcna-A-20340276-resume.txt')
                  Should include the full filename with extension
        document_type: Type of document being written - either "brief_resume" or "full_listing"
                      Used for logging purposes
    
    Returns:
        JSON string with success status, file_path, and file_name
    """
    try:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the file_name doesn't contain path separators (security)
        if "/" in file_name or "\\" in file_name:
            # Extract just the filename
            file_name = Path(file_name).name
        file_path = results_dir / file_name
        
        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = {
            "success": True,
            "file_path": str(file_path),
            "file_name": file_name,
            "document_type": document_type,
            "content_length": len(content),
            "message": f"Successfully wrote {document_type} document to results/{file_name}"
        }
        
        print(f"[Document Creator Tool] ✓ Wrote {document_type} document: {file_name} ({len(content)} characters)")
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        error_details = traceback.format_exc()
        error_result = {
            "success": False,
            "error": f"Error writing {document_type} document: {str(e)}",
            "details": error_details,
            "file_name": file_name
        }
        print(f"[Document Creator Tool] ⚠️  ERROR: Failed to write {document_type} document: {e}")
        return json.dumps(error_result, indent=2)


def get_available_tools(agent_name: Optional[str] = None) -> list:
    """
    Returns a list of available tools for a specific agent.
    
    Args:
        agent_name: Name of the agent to get tools for. If None, returns default tools.
                   Valid values: "brief_creator", "theme_agent", "new_creative_agent", 
                   "campaign_update_agent", or None for default.
    
    Returns:
        List of tool objects for the specified agent
    """
    # Base tool that all agents should have
    base_tools = [load_and_parse_spreadsheet]
    
    # RAG tools for agents that need document guidance + similar campaign search + similar diagnoses for self-validation
    # Note: task-specific agents should NOT have load_and_parse_spreadsheet to prevent reloading
    rag_tools = [retrieve_rag_information, find_similar_campaigns, fetch_campaign_brief_from_drive, find_similar_diagnoses]
    
    # Campaign Update agent needs additional tools to identify and load previous campaign ID + similar diagnoses for self-validation
    campaign_update_tools = [
        retrieve_rag_information,
        find_similar_campaigns,
        fetch_campaign_brief_from_drive,
        find_similar_diagnoses,
        identify_previous_campaign_id,
        find_and_load_previous_campaign_brief,
        match_campaigns_by_headline
    ]
    
    
    # QA agent tools: RAG access + similar diagnoses search (more efficient than campaign briefs)
    qa_agent_tools = [load_and_parse_spreadsheet, retrieve_rag_information, find_similar_diagnoses]
    
    # Document creator agent tools: RAG access + document writing tool
    document_creator_agent_tools = [retrieve_rag_information, write_document_to_file]
    
    # Define tool sets for each agent
    agent_tool_map = {
        "brief_creator": [load_and_parse_spreadsheet],
        "theme_agent": rag_tools,  # Has RAG access for rules + similar campaign search
        "new_creative_agent": rag_tools,  # Has RAG access for rules + similar campaign search
        "campaign_update_agent": campaign_update_tools,  # Has RAG access + previous campaign ID identification
        "qa_agent": qa_agent_tools,  # Has RAG access for QA rules + campaign brief retrieval + similar diagnoses search
        "document_creator_agent": document_creator_agent_tools,  # Has RAG access + text document storage
        "family_similarity_agent": [],  # Synthesizer; specialists are invoked directly
        "family_sim_reference_agent": [],
        "family_sim_style_agent": [],
        "family_sim_wording_agent": [],
    }
    
    # Normalize agent name
    if agent_name:
        agent_name_lower = agent_name.lower().strip()
        # Return agent-specific tools if available, otherwise base tools
        return agent_tool_map.get(agent_name_lower, base_tools)
    
    # Default: return base tools
    return base_tools

