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


def _parse_spreadsheet_internal(spreadsheet_path: str) -> dict:
    """
    Internal function to parse a spreadsheet without the tool wrapper.
    This can be called directly from other Python code.
    
    Returns:
        A dict with: task_type, asset_summary, dealership_name, content_11_20, campaigns.
    """
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
        # Use the same embedding model as ingestion (import from rag_ingestion)
        try:
            from rag_ingestion import EMBEDDING_MODEL
            embedding_model = EMBEDDING_MODEL
        except ImportError:
            # Fallback if import fails (shouldn't happen in normal operation)
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        embeddings = OpenAIEmbeddings(model=embedding_model)
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
        task_type: Task type to filter by (e.g., "Theme", "New Creative", "Campaign Update", "Rework")
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
        collection_name_actual = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "campaign_documents")
        
        # Build filter for metadata-based search (task_type only first)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
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
        # Import Google Drive helpers from rag_ingestion
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from rag_ingestion import get_drive_service, download_file_content
        import tempfile
        
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
    collection_name: Optional[str] = None,
    top_k: int = 10
) -> str:
    """
    Identify the previous campaign ID that all campaigns in a Campaign Update brief should reference.
    This tool retrieves RAG documentation about Campaign Update rules and extracts the previous campaign ID pattern.
    
    Use this tool FIRST when evaluating Campaign Update campaigns to identify the previous campaign ID
    (e.g., "A-12345678") that all campaigns should reference according to the rules.
    
    This tool is specifically for Campaign Update agents and should be called before evaluating campaigns.
    
    IMPORTANT: If this tool returns an ERROR, the agent must stop evaluation and report the error.
    The previous campaign ID is required for Campaign Update evaluation and cannot be skipped.
    
    Args:
        collection_name: Optional name of the Qdrant collection to search (defaults to env var)
        top_k: Number of documents to retrieve for searching (default: 10, increased to find the ID)
        
    Returns:
        The previous campaign ID in the format "A-XXXXXXXX" or similar pattern found in the rules.
        Returns an ERROR message if the previous campaign ID cannot be found in the RAG documentation.
    """
    if not QDRANT_AVAILABLE:
        return "Error: Qdrant dependencies are not installed."
    
    try:
        # Use retrieve_rag_information to get Campaign Update rules
        rag_result = retrieve_rag_information(
            query="Campaign Update previous campaign ID format pattern A-12345678 how to identify",
            collection_name=collection_name,
            file_type="document",
            top_k=top_k
        )
        
        if "Error" in rag_result:
            raise ValueError(f"Failed to retrieve RAG information: {rag_result}")
        
        # Search for campaign ID patterns in the retrieved text
        # Pattern: A- followed by 8 digits (e.g., A-12345678)
        import re
        
        # Look for patterns like "A-12345678" or "A-XXXXXXXX" or similar
        patterns = [
            r'A-\d{8}',  # A-12345678
            r'A-\d{7,9}',  # A-1234567 or A-123456789 (flexible)
            r'[A-Z]-\d{7,9}',  # Any letter followed by dash and digits
        ]
        
        found_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, rag_result, re.IGNORECASE)
            if matches:
                found_ids.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for id_val in found_ids:
            if id_val.upper() not in seen:
                seen.add(id_val.upper())
                unique_ids.append(id_val)
        
        if not unique_ids:
            # Return a clear error message that the agent can recognize
            return (
                f"ERROR: Previous campaign ID not found in RAG documentation.\n\n"
                f"Retrieved RAG information preview:\n{rag_result[:500]}...\n\n"
                f"CRITICAL: The previous campaign ID (format: A-12345678) could not be identified from the RAG rules.\n"
                f"This is required for Campaign Update evaluation. Please ensure:\n"
                f"1. The Campaign Update documentation contains the previous campaign ID pattern\n"
                f"2. The ID format matches patterns like 'A-12345678' or similar\n"
                f"3. The documentation is properly indexed in the RAG collection\n\n"
                f"Evaluation cannot proceed without identifying the previous campaign ID."
            )
        
        # If multiple IDs found, use the most common one or the first one
        # Typically there should be one consistent ID
        previous_campaign_id = unique_ids[0]
        
        if len(unique_ids) > 1:
            # Warn if multiple IDs found, but return the first one
            return (
                f"WARNING: Multiple campaign ID patterns found: {', '.join(unique_ids)}\n"
                f"Using: {previous_campaign_id}\n\n"
                f"Previous Campaign ID: {previous_campaign_id}"
            )
        
        return f"Previous Campaign ID identified: {previous_campaign_id}"
    
    except ValueError as e:
        # Re-raise ValueError (our custom error)
        raise e
    except Exception as e:
        return f"Error identifying previous campaign ID: {str(e)}"


@tool
def find_and_load_previous_campaign_brief(
    previous_campaign_id: str,
    search_directory: Optional[str] = None
) -> str:
    """
    Find a campaign brief file locally that contains the previous campaign ID in its filename,
    then load and parse it to create a campaign brief.
    
    This tool searches for spreadsheet files (.xlsx) in the local filesystem that have the previous campaign ID
    (e.g., "A-12345678") in their filename, and parses it into a campaign brief.
    
    Use this tool after identifying the previous campaign ID to load the original campaign brief
    that the Campaign Update campaigns are referencing.
    
    Args:
        previous_campaign_id: The previous campaign ID to search for (e.g., "A-12345678")
        search_directory: Optional local directory path to search in. If not provided, searches
                         in the current working directory.
        
    Returns:
        A JSON string containing the full CampaignBrief structure with all campaigns from the
        previous campaign brief file. Returns an error if the file cannot be found or parsed.
    """
    try:
        from pathlib import Path
        import glob
        
        # Determine search directory
        if search_directory:
            search_path = Path(search_directory)
            # Resolve relative paths to absolute paths
            if not search_path.is_absolute():
                search_path = Path.cwd() / search_path
            search_path = search_path.resolve()
            
            if not search_path.exists():
                return (
                    f"ERROR: Search directory does not exist: {search_path}\n"
                    f"Previous Campaign ID: {previous_campaign_id}"
                )
            if not search_path.is_dir():
                return (
                    f"ERROR: Search path is not a directory: {search_path}\n"
                    f"Previous Campaign ID: {previous_campaign_id}"
                )
        else:
            # Default to current working directory (where Python script is run from)
            search_path = Path.cwd().resolve()
        
        # Search for .xlsx files in the directory (recursively)
        xlsx_pattern = str(search_path / "**" / "*.xlsx")
        all_files = glob.glob(xlsx_pattern, recursive=True)
        
        if not all_files:
            return (
                f"ERROR: No .xlsx files found in search directory.\n"
                f"Directory: {search_path}\n"
                f"Previous Campaign ID: {previous_campaign_id}"
            )
        
        # Find file(s) containing the previous campaign ID in the filename
        matching_files = [
            f for f in all_files
            if previous_campaign_id.upper() in Path(f).name.upper()
        ]
        
        if not matching_files:
            # Show some example filenames for debugging
            example_files = [Path(f).name for f in all_files[:5]]
            return (
                f"ERROR: No file found containing previous campaign ID '{previous_campaign_id}' in filename.\n"
                f"Searched in directory: {search_path}\n"
                f"Found {len(all_files)} .xlsx file(s) but none match the campaign ID.\n"
                f"Example files: {example_files}"
            )
        
        # Use the first matching file (or most recent if multiple)
        if len(matching_files) > 1:
            # Sort by modification time, most recent first
            matching_files.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
        
        target_file = matching_files[0]
        file_name = Path(target_file).name
        
        print(f"[Previous Campaign] Found file: {file_name} (path={target_file})")
        
        # Parse the spreadsheet into a campaign brief
        campaign_brief = _parse_spreadsheet_internal(target_file)
        
        # Return as JSON string
        return json.dumps(campaign_brief, indent=2)
    
    except Exception as e:
        import traceback
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
    try:
        import json
        from typing import Dict, List, Any
        
        # Parse both briefs
        current_brief = json.loads(current_campaign_brief_json) if isinstance(current_campaign_brief_json, str) else current_campaign_brief_json
        previous_brief = json.loads(previous_campaign_brief_json) if isinstance(previous_campaign_brief_json, str) else previous_campaign_brief_json
        
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
        import traceback
        error_details = traceback.format_exc()
        return (
            f"Error matching campaigns by headline: {str(e)}\n"
            f"Details: {error_details}"
        )


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
    
    # RAG tools for agents that need document guidance + similar campaign search
    # Note: task-specific agents should NOT have load_and_parse_spreadsheet to prevent reloading
    rag_tools = [retrieve_rag_information, find_similar_campaigns, fetch_campaign_brief_from_drive]
    
    # Campaign Update agent needs additional tools to identify and load previous campaign ID
    campaign_update_tools = [
        retrieve_rag_information,
        find_similar_campaigns,
        fetch_campaign_brief_from_drive,
        identify_previous_campaign_id,
        find_and_load_previous_campaign_brief,
        match_campaigns_by_headline
    ]
    
    # RAG tools including campaign brief retrieval (for QA validation)
    rag_tools_with_briefs = [load_and_parse_spreadsheet, retrieve_rag_information, retrieve_campaign_briefs, find_similar_campaigns, fetch_campaign_brief_from_drive]
    
    # Define tool sets for each agent
    agent_tool_map = {
        "brief_creator": base_tools,
        "theme_agent": rag_tools,  # Has RAG access for rules + similar campaign search
        "new_creative_agent": rag_tools,  # Has RAG access for rules + similar campaign search
        "campaign_update_agent": campaign_update_tools,  # Has RAG access + previous campaign ID identification
        "qa_agent": rag_tools_with_briefs,  # Has RAG access for QA rules + campaign brief retrieval for validation
    }
    
    # Normalize agent name
    if agent_name:
        agent_name_lower = agent_name.lower().strip()
        # Return agent-specific tools if available, otherwise base tools
        return agent_tool_map.get(agent_name_lower, base_tools)
    
    # Default: return base tools
    return base_tools

