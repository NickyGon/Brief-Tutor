"""
Tools for the LangGraph agent workflow.
"""
from typing import List, Dict, Optional, Set
from langchain.tools import tool
from graph.models import Campaign, CampaignBrief, OfferDetails, StyleDescriptions, Assets
from collections import Counter
import re
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant and RAG imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import Qdrant
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

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
    label_col_idx = _find_label_column(df)
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
    candidate_labels = list(ROW_LABEL_TO_KEY.keys()) + OT_LABELS

    for c in range(df.shape[1]):
        col = df.iloc[:, c]
        if any(isinstance(v, str) and v.strip() in candidate_labels for v in col):
            return c

    # Fallback to old assumption (column D)
    return 3


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

        # In your template, the row with "Content 1"..."Content 10"/"Content 20"
        # is at index 7 (0-based) even with the extra border / padding.
        CAMPAIGN_HEADER_ROW = 7  # row with "Content 1"..."Content 10"/"Content 20"

        # Columns E..N → indices 4..13 (we still guard with df.shape[1])
        for col in range(4, df.shape[1]):
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

                        if key in assets_keys:
                            assets_data[key] = value_str
                        elif key in style_keys:
                            style_descriptions_data[key] = value_str
                        elif key in offer_keys:
                            offer_details_data[key] = value_str
                    except Exception as e:
                        print(f"[WARN] Error processing row '{key}' (index {row_idx}) for campaign '{campaign_id}' in column {col}: {type(e).__name__}: {str(e)}")
                        continue

                # OT 1–6 rows - these are additional assets
                for key, row_idx in ot_rows.items():
                    try:
                        if row_idx >= df.shape[0]:
                            continue

                        value = df.iloc[row_idx, col]
                        if pd.isna(value):
                            continue

                        assets_data[key] = str(value).strip()
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
        import traceback
        print(f"[ERROR] Traceback:")
        traceback.print_exc()
        # Return empty results on critical failure
        return {
            "task_type": None,
            "asset_summary": None,
            "dealership_name": None,
            "content_11_20": False,
        }, []


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

    print(f"[Brief Creator] Loading spreadsheet from: {spreadsheet_path}")
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
    import json
    try:
        json.dumps(result_dict)  # Test if it's JSON serializable
    except TypeError as e:
        print(f"[ERROR] Result is not JSON serializable: {str(e)}")
        # Fallback: use model_dump_json and parse back
        result_dict = json.loads(campaign_brief.model_dump_json())
    
    return result_dict


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
    
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "campaign_documents")
    
    try:
        embeddings = OpenAIEmbeddings()
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
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "campaign_documents")
        
        # Use vectorstore for embedding the query
        query_vector = vectorstore.embeddings.embed_query(query)
        
        # Search directly with Qdrant client to get full payloads
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
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
        task_type_filter: Optional filter by task type ("Theme", "New Creative", "Campaign Update", "Rework")
        
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
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "campaign_documents")
        
        # Use vectorstore for embedding the query
        query_vector = vectorstore.embeddings.embed_query(query)
        
        # Filter for campaign briefs only
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
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
    
    # RAG tool for agents that need document guidance (rules/guidelines only)
    # Note: task-specific agents should NOT have load_and_parse_spreadsheet to prevent reloading
    rag_tools = [retrieve_rag_information]
    
    # RAG tools including campaign brief retrieval (for QA validation)
    rag_tools_with_briefs = [load_and_parse_spreadsheet, retrieve_rag_information, retrieve_campaign_briefs]
    
    # Define tool sets for each agent
    agent_tool_map = {
        "brief_creator": base_tools,
        "theme_agent": rag_tools,  # Has RAG access for rules only
        "new_creative_agent": rag_tools,  # Has RAG access for rules only
        "campaign_update_agent": rag_tools,  # Has RAG access for rules only
        "qa_agent": rag_tools_with_briefs,  # Has RAG access for QA rules + campaign brief retrieval for validation
    }
    
    # Normalize agent name
    if agent_name:
        agent_name_lower = agent_name.lower().strip()
        # Return agent-specific tools if available, otherwise base tools
        return agent_tool_map.get(agent_name_lower, base_tools)
    
    # Default: return base tools
    return base_tools

