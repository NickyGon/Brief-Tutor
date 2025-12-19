"""
rag_ingestion.py

Core logic to sync documents from a Google Drive folder into a Qdrant
vector database for RAG.

You still need to plug in a real embedding function in `embed_text()`.
"""

import io
import tempfile
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from datetime import datetime

from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pypdf import PdfReader
from docx import Document as DocxDocument  # for .docx files
from dotenv import load_dotenv
from uuid import uuid5, NAMESPACE_URL

load_dotenv()

# -----------------------------
# Basic configuration constants
# -----------------------------

# You can override these via environment variables if you want.
CHUNK_SIZE_CHARS = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "200"))

# Batch size for Qdrant upserts (to avoid timeouts on large files)
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "50"))  # Process 50 chunks at a time
QDRANT_MAX_RETRIES = int(os.getenv("QDRANT_MAX_RETRIES", "3"))  # Retry up to 3 times
QDRANT_RETRY_DELAY = int(os.getenv("QDRANT_RETRY_DELAY", "5"))  # Wait 5 seconds between retries

# -----------------------------
# Embedding Model Configuration (Single Source of Truth)
# -----------------------------

def get_embedding_model() -> str:
    """Get the embedding model name from environment variable. Defaults to text-embedding-3-large."""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def get_vector_size() -> int:
    """
    Get the vector size based on embedding model.
    
    Note: RAG_VECTOR_SIZE env var can override, but ensure_qdrant_collection()
    will enforce the correct size for the embedding model to prevent mismatches.
    """
    embedding_model = get_embedding_model()
    default_size = 3072 if "large" in embedding_model.lower() else 1536
    # Allow override via env var, but validation in ensure_qdrant_collection will enforce correctness
    return int(os.getenv("RAG_VECTOR_SIZE", str(default_size)))


# Global constants (computed once at module load)
EMBEDDING_MODEL = get_embedding_model()
VECTOR_SIZE = get_vector_size()

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx"
}


# -----------------------------
# Google Drive helpers
# -----------------------------


def get_drive_service() -> Any:
    """
    Builds an authenticated Google Drive API service using a service account.

    Expects:
        - GOOGLE_SERVICE_ACCOUNT_FILE: path to your service account JSON.
    """
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not service_account_file:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_FILE env var is not set. "
            "Point it to your service account JSON file."
        )

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def list_files_in_folder(
    drive_service: Any, folder_id: str, file_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    List all supported files in a given Google Drive folder (non-trashed).

    Args:
        drive_service: Google Drive API service
        folder_id: ID of the folder to list files from
        file_types: Optional list of MIME types to filter. If None, uses all SUPPORTED_MIME_TYPES.
                   Can be used to filter only documents (pdf, docx) or only spreadsheets (xlsx).

    Returns:
        A list of dicts: {id, name, mimeType, modifiedTime}
    """
    if file_types is None:
        mime_types = list(SUPPORTED_MIME_TYPES.keys())
    else:
        mime_types = file_types
    
    query_mime = " or ".join(
        [f"mimeType='{mt}'" for mt in mime_types]
    )
    query = f"'{folder_id}' in parents and trashed=false and ({query_mime})"

    files: List[Dict[str, Any]] = []
    page_token = None

    while True:
        response = (
            drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return files


def list_folders_in_folder(
    drive_service: Any, folder_id: str
) -> List[Dict[str, Any]]:
    """
    List all folders (subdirectories) in a given Google Drive folder (non-trashed).

    Args:
        drive_service: Google Drive API service
        folder_id: ID of the folder to list subfolders from

    Returns:
        A list of dicts: {id, name, mimeType, modifiedTime}
    """
    query = f"'{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"

    folders: List[Dict[str, Any]] = []
    page_token = None

    while True:
        response = (
            drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )

        folders.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return folders


def find_folder_by_name(
    drive_service: Any, parent_folder_id: str, folder_name: str
) -> Optional[Dict[str, Any]]:
    """
    Find a subfolder by name within a parent folder.

    Args:
        drive_service: Google Drive API service
        parent_folder_id: ID of the parent folder
        folder_name: Name of the folder to find

    Returns:
        Folder dict with {id, name, mimeType, modifiedTime} or None if not found
    """
    folders = list_folders_in_folder(drive_service, parent_folder_id)
    for folder in folders:
        if folder.get("name") == folder_name:
            return folder
    return None


def download_file_content(
    drive_service: Any, file_id: str, mime_type: str
) -> bytes:
    """
    Download file content from Google Drive as raw bytes.

    For PDFs and Office formats, we can usually use .get_media() directly.
    If you need export for Google Docs/Sheets/Slides, you'd use files().export().
    """

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return fh.read()


# -----------------------------
# Text extraction helpers
# -----------------------------


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extracts text from a PDF file (bytes) using pypdf.
    """
    import io

    reader = PdfReader(io.BytesIO(content))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(content: bytes) -> str:
    """
    Extracts text from a .docx file (bytes) using python-docx.
    """
    import io

    doc = DocxDocument(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text_from_xlsx(content: bytes, file_meta: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
    """
    Extracts structured information from an Excel spreadsheet (bytes).
    Uses load_and_parse_spreadsheet from tools.py to parse the spreadsheet.
    
    Returns:
        tuple: (text_representation, structured_brief_data)
        - text_representation: A text summary of the spreadsheet for vector search
        - structured_brief_data: Parsed CampaignBrief-like structure (None if parsing fails)
    """

    
    # Add project root to path to import from graph.tools
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from graph.tools import _parse_spreadsheet_internal
    except ImportError:
        # Fallback if import fails
        import pandas as pd
        try:
            xls = pd.ExcelFile(io.BytesIO(content))
            text_parts = [f"Spreadsheet: {file_meta.get('name', 'Unknown')}"]
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name, header=None)
                text_parts.append(f"\nSheet: {sheet_name}")
                text_parts.append(df.head(20).to_string())
            return "\n".join(text_parts), None
        except Exception as fallback_error:
            return f"Error extracting text from Excel file: {str(fallback_error)}", None
    
    # Create a temporary file to save the Excel content
    temp_file = None
    try:
        # Create temporary file with .xlsx extension
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
            tmp.write(content)
            temp_file = tmp.name
        
        # Use _parse_spreadsheet_internal to parse the file (not the tool wrapper)
        # Note: _parse_spreadsheet_internal returns a dict (not a CampaignBrief object)
        campaign_brief = _parse_spreadsheet_internal(temp_file)
        
        # Extract structured data from the returned dict
        structured_data = {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "file_type": "campaign_brief",
            "spreadsheet_path": file_meta.get("name", ""),
            "task_type": campaign_brief.get("task_type"),
            "asset_summary": campaign_brief.get("asset_summary"),
            "dealership_name": campaign_brief.get("dealership_name"),
            "content_11_20": campaign_brief.get("content_11_20", False),
            "campaigns": []
        }
        
        # Extract campaign information
        # campaigns is a list of Campaign Pydantic objects
        campaigns = campaign_brief.get("campaigns", [])
        total_campaigns = len(campaigns)
        
        for campaign in campaigns:
            # campaign is a Campaign Pydantic model
            campaign_id = campaign.campaign_id
            offer_details = campaign.offer_details
            
            headline = offer_details.headline if offer_details else ""
            offer = offer_details.offer if offer_details else ""
            
            structured_data["campaigns"].append({
                "campaign_id": campaign_id,
                "headline": headline[:100] if headline else "",  # Truncate for storage
                "offer": offer[:100] if offer else "",
            })
        
        structured_data["total_campaigns"] = total_campaigns
        
        # Create text representation for vector search
        text_parts = []
        text_parts.append(f"Campaign Brief: {file_meta.get('name', 'Unknown')}")
        text_parts.append(f"Task Type: {structured_data.get('task_type', 'Unknown')}")
        text_parts.append(f"Dealership: {structured_data.get('dealership_name', 'Unknown')}")
        text_parts.append(f"Asset Summary: {structured_data.get('asset_summary', 'N/A')}")
        text_parts.append(f"Number of Campaigns: {total_campaigns}")
        
        # Add campaign summaries
        for campaign in campaigns[:10]:  # Limit to first 10 for text representation
            campaign_id = campaign.campaign_id
            offer_details = campaign.offer_details
            
            text_parts.append(f"\nCampaign {campaign_id}:")
            if offer_details:
                headline = offer_details.headline
                offer = offer_details.offer
                body = offer_details.body
                
                if headline:
                    text_parts.append(f"  Headline: {headline}")
                if offer:
                    text_parts.append(f"  Offer: {offer}")
                if body:
                    text_parts.append(f"  Body: {body[:200]}...")  # Truncate for text representation
        
        if total_campaigns > 10:
            text_parts.append(f"\n... and {total_campaigns - 10} more campaigns")
        
        text_representation = "\n".join(text_parts)
        return text_representation, structured_data
        
    except Exception as e:
        # Fallback to basic extraction if parsing fails
        import pandas as pd
        try:
            xls = pd.ExcelFile(io.BytesIO(content))
            text_parts = [f"Spreadsheet: {file_meta.get('name', 'Unknown')}"]
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name, header=None)
                text_parts.append(f"\nSheet: {sheet_name}")
                # Only include first few rows for text representation
                text_parts.append(df.head(20).to_string())
            return "\n".join(text_parts), None
        except Exception as fallback_error:
            return f"Error extracting text from Excel file: {str(e)}", None
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


def extract_campaign_metadata(
    content: bytes, 
    file_meta: Dict[str, Any]
) -> tuple[Dict[str, Any], str]:
    """
    Extract lightweight metadata from campaign brief spreadsheets.
    This function extracts only essential metadata for similarity search,
    not the full campaign data.
    
    Args:
        content: Raw bytes of the Excel file
        file_meta: File metadata dict from Google Drive with keys: id, name, modifiedTime
    
    Returns:
        tuple: (metadata_dict, compact_text_representation)
        - metadata_dict: Contains file_id, file_name, task_type, dealership_name,
                        asset_summary, total_campaigns, campaign_ids, file_modified_time
        - compact_text_representation: Text string for embedding/search
    """
    # Add project root to path to import from graph.tools
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    temp_file = None
    try:
        from graph.tools import _parse_spreadsheet_internal
    except ImportError:
        # Fallback: return minimal metadata if import fails
        return {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "task_type": None,
            "dealership_name": None,
            "asset_summary": None,
            "total_campaigns": 0,
            "campaign_ids": [],
            "file_modified_time": file_meta.get("modifiedTime"),
        }, f"Campaign Brief: {file_meta.get('name', 'Unknown')}"
    
    try:
        # Create temporary file to save the Excel content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
            tmp.write(content)
            temp_file = tmp.name
        
        # Use _parse_spreadsheet_internal to parse the file (not the tool wrapper)
        campaign_brief = _parse_spreadsheet_internal(temp_file)
        
        # Extract only essential metadata
        task_type = campaign_brief.get("task_type")
        dealership_name = campaign_brief.get("dealership_name")
        asset_summary = campaign_brief.get("asset_summary")
        campaigns = campaign_brief.get("campaigns", [])
        
        # Extract campaign IDs only (not full campaign data)
        campaign_ids = []
        for campaign in campaigns:
            if isinstance(campaign, dict):
                campaign_id = campaign.get("campaign_id")
            else:
                # Campaign might be a Pydantic model
                campaign_id = getattr(campaign, "campaign_id", None)
            if campaign_id:
                campaign_ids.append(campaign_id)
        
        metadata = {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "task_type": task_type,
            "dealership_name": dealership_name,
            "asset_summary": asset_summary,
            "total_campaigns": len(campaign_ids),
            "campaign_ids": campaign_ids,
            "file_modified_time": file_meta.get("modifiedTime"),
        }
        
        # Create compact text representation for embedding
        compact_text_parts = [
            f"Campaign Brief: {file_meta.get('name', 'Unknown')}",
            f"Task Type: {task_type or 'Unknown'}",
            f"Dealership: {dealership_name or 'Unknown'}",
            f"Asset Summary: {asset_summary or 'N/A'}",
            f"Campaigns: {', '.join(campaign_ids) if campaign_ids else 'None'}"
        ]
        compact_text = "\n".join(compact_text_parts)
        
        return metadata, compact_text
        
    except Exception as e:
        # Fallback: return minimal metadata on error
        print(f"[WARN] Failed to extract campaign metadata from {file_meta.get('name', 'Unknown')}: {e}")
        return {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "task_type": None,
            "dealership_name": None,
            "asset_summary": None,
            "total_campaigns": 0,
            "campaign_ids": [],
            "file_modified_time": file_meta.get("modifiedTime"),
        }, f"Campaign Brief: {file_meta.get('name', 'Unknown')}"
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


def extract_text_for_file(mime_type: str, content: bytes, file_meta: Optional[Dict[str, Any]] = None) -> tuple[str, Optional[Dict[str, Any]]]:
    """
    Dispatch to the appropriate extractor based on MIME type.
    
    Returns:
        tuple: (text, structured_data)
        - text: Text representation for vector search
        - structured_data: Optional structured data (for spreadsheets, contains parsed brief info)
    """
    kind = SUPPORTED_MIME_TYPES.get(mime_type)
    if kind == "pdf":
        return extract_text_from_pdf(content), None
    elif kind == "docx":
        return extract_text_from_docx(content), None
    elif kind == "xlsx":
        file_meta = file_meta or {}
        return extract_text_from_xlsx(content, file_meta)
    else:
        raise ValueError(f"Unsupported mime type: {mime_type}")


# -----------------------------
# Chunking helpers
# -----------------------------


def chunk_text(
    text: str,
    file_meta: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """
    Splits a long text into overlapping character-based chunks.

    Returns a list of dicts:
        {
            "id_suffix": int,
            "text": str,
            "payload": {...}
        }

    where payload already includes useful file-level metadata.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunk_payload = {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "file_modified_time": file_meta["modifiedTime"],
            "chunk_index": idx,
            "page_content": chunk_text,  # Store text content for LangChain compatibility
            "text": chunk_text,  # Also store as text for direct access
        }
        chunks.append(
            {
                "id_suffix": idx,
                "text": chunk_text,
                "payload": chunk_payload,
            }
        )
        idx += 1
        # Move forward with overlap
        start = end - overlap

    return chunks


# -----------------------------
# Qdrant helpers
# -----------------------------


def get_qdrant_client() -> QdrantClient:
    """
    Creates a Qdrant client using environment variables:

        QDRANT_URL
        QDRANT_API_KEY  (if using Qdrant Cloud)
    """
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url:
        raise RuntimeError("QDRANT_URL env var is not set")

    client = QdrantClient(url=url, api_key=api_key)
    return client


def ensure_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int = None
) -> None:
    """
    Ensures the Qdrant collection exists with the correct vector size.
    If it doesn't, it creates it. If it exists with wrong dimensions, deletes and recreates it.
    Also ensures required payload indexes exist.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Vector size to use (defaults to VECTOR_SIZE, which should be 3072 for text-embedding-3-large)
    """
    # CRITICAL: Always enforce correct vector size based on embedding model
    # This overrides any incorrect RAG_VECTOR_SIZE environment variable or VECTOR_SIZE constant
    embedding_model = EMBEDDING_MODEL
    
    # Calculate expected size directly from embedding model (don't use get_vector_size() which respects RAG_VECTOR_SIZE)
    if "large" in embedding_model.lower():
        expected_size = 3072
    else:
        expected_size = 1536
    
    # Override vector_size if it was passed in or from VECTOR_SIZE constant
    if vector_size is None:
        vector_size = VECTOR_SIZE
    
    # Always enforce the correct size for the embedding model
    if vector_size != expected_size:
        print(f"⚠ WARNING: Vector size {vector_size} doesn't match expected {expected_size} for {embedding_model}")
        print(f"  Overriding to correct size {expected_size}")
        vector_size = expected_size
    
    # Log the vector size being used for transparency (after validation)
    print(f"📊 Collection Configuration:")
    print(f"   Embedding Model: {embedding_model}")
    print(f"   Vector Size: {vector_size} dimensions")
    
    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"✓ Created collection '{collection_name}' with vector size {vector_size} dimensions")
    else:
        # Check if the existing collection has the correct vector size
        collection_info = client.get_collection(collection_name)
        existing_vector_size = collection_info.config.params.vectors.size
        
        if existing_vector_size != vector_size:
            print(f"⚠ Collection '{collection_name}' has vector size {existing_vector_size}, but expected {vector_size}")
            print(f"  Deleting and recreating collection with correct dimensions...")
            
            # Delete the existing collection
            client.delete_collection(collection_name)
            print(f"  ✓ Deleted old collection")
            
            # Create new collection with correct dimensions
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            print(f"  ✓ Created new collection '{collection_name}' with vector size {vector_size}")
            print(f"  ⚠ NOTE: All existing data in this collection has been deleted.")
            print(f"  ⚠ You will need to re-run the ingestion script to re-index your documents.")
        else:
            print(f"✓ Collection '{collection_name}' exists with correct vector size {vector_size}")
    
    # Ensure payload indexes exist for filtering
    # Always try to create indexes - Qdrant will return an error if they already exist
    required_indexes = {
        "file_id": qmodels.PayloadSchemaType.KEYWORD,
        "file_modified_time": qmodels.PayloadSchemaType.KEYWORD,
        "file_type": qmodels.PayloadSchemaType.KEYWORD,
        "task_type": qmodels.PayloadSchemaType.KEYWORD,  # For campaign metadata filtering
        "dealership_name": qmodels.PayloadSchemaType.KEYWORD,  # For campaign metadata filtering
    }
    
    for field_name, field_schema in required_indexes.items():
        index_created = False
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✓ Created payload index for '{field_name}'")
            index_created = True
        except Exception as e:
            # Check if it's an "already exists" error
            error_msg = str(e).lower()
            error_str = str(e)
            
            # Qdrant returns different error messages for existing indexes
            if any(phrase in error_msg for phrase in [
                "already exists", 
                "duplicate", 
                "index.*already",
                "already.*index"
            ]):
                print(f"✓ Index for '{field_name}' already exists")
                index_created = True
            else:
                # For any other error, print it and re-raise to see what's wrong
                print(f"✗ Error creating index for '{field_name}': {error_str}")
                print(f"  Error type: {type(e).__name__}")
                # Re-raise to see the full error
                raise RuntimeError(
                    f"Failed to create required payload index '{field_name}'. "
                    f"This index is required for filtering. Error: {error_str}"
                ) from e
        
        # Verify the index actually exists by trying a test query
        if index_created:
            try:
                # Try a simple scroll with filter to verify index works
                test_filter = qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key=field_name, 
                        match=qmodels.MatchValue(value="__test_verification__")
                    )]
                )
                # This should not error if index exists (even if no results)
                client.scroll(
                    collection_name=collection_name,
                    scroll_filter=test_filter,
                    limit=1,
                )
                print(f"✓ Verified index for '{field_name}' is working")
            except Exception as verify_error:
                error_str = str(verify_error)
                if "index required" in error_str.lower() or "index.*not found" in error_str.lower():
                    print(f"✗ WARNING: Index '{field_name}' was reported as created but verification failed!")
                    print(f"  This suggests the index creation didn't actually work.")
                    print(f"  Verification error: {error_str}")
                    raise RuntimeError(
                        f"Index '{field_name}' creation was reported successful but verification failed. "
                        f"This index is required. Error: {error_str}"
                    ) from verify_error
                # Other errors (like connection issues) are OK for verification
                print(f"  Note: Could not verify index (non-critical): {verify_error}")


def get_latest_indexed_modified_time(
    client: QdrantClient, collection_name: str, file_id: str
) -> Optional[str]:
    """
    Looks in Qdrant for any chunk belonging to this file_id and
    returns the latest (max) file_modified_time present in payload.

    Returns None if file_id is not indexed yet.
    """
    # Scroll with filter on file_id; just get a few points is enough
    scroll_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="file_id", match=qmodels.MatchValue(value=file_id))]
    )

    # We don't care about all points, just enough to see a payload
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=10,
    )

    if not points:
        return None

    times = []
    for p in points:
        payload = p.payload or {}
        t = payload.get("file_modified_time")
        if t:
            times.append(t)

    if not times:
        return None

    # Return the max timestamp string lexicographically (ISO 8601)
    return max(times)


def needs_update(
    client: QdrantClient,
    collection_name: str,
    file_meta: Dict[str, Any],
) -> bool:
    """
    Compares Google Drive's modifiedTime for this file against what's
    already in Qdrant. Returns True if:

        - file has never been indexed, or
        - Drive's modifiedTime is more recent.
    """
    drive_time_str = file_meta["modifiedTime"]
    existing_time_str = get_latest_indexed_modified_time(
        client, collection_name, file_meta["id"]
    )

    if existing_time_str is None:
        return True  # never indexed

    # Both are RFC3339 / ISO-ish, we can compare as datetimes
    drive_time = datetime.fromisoformat(drive_time_str.replace("Z", "+00:00"))
    existing_time = datetime.fromisoformat(existing_time_str.replace("Z", "+00:00"))

    return drive_time > existing_time


# -----------------------------
# Embeddings
# -----------------------------


def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for a list of texts using OpenAI embeddings.
    
    Uses the embedding model specified by EMBEDDING_MODEL environment variable.
    Defaults to "text-embedding-3-large" (3072 dimensions, better quality).
    Alternative: "text-embedding-3-small" (1536 dimensions, more cost-effective).
    
    Both models support up to 8,192 tokens per input, which is well above the chunk size.
    """
    embedding_model = EMBEDDING_MODEL
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Batch process texts (OpenAI API handles batching efficiently)
        resp = client.embeddings.create(
            model=embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]
    
    except ImportError:
        print("[WARN] OpenAI library not available. Using fallback hash-based embeddings.")
        print("       Install with: pip install openai")
        # Fallback to hash-based embeddings if OpenAI is not available
        import hashlib
        import math

        vectors: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # repeat / trim to VECTOR_SIZE
            raw = list(h) * ((VECTOR_SIZE // len(h)) + 1)
            raw = raw[:VECTOR_SIZE]
            # normalize to [0,1]
            vec = [x / 255.0 for x in raw]
            # l2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors
    
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        print("       Falling back to hash-based embeddings.")
        # Fallback to hash-based embeddings on error
        import hashlib
        import math

        vectors: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            raw = list(h) * ((VECTOR_SIZE // len(h)) + 1)
            raw = raw[:VECTOR_SIZE]
            vec = [x / 255.0 for x in raw]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


# -----------------------------
# Helper function for batched Qdrant upserts with retry logic
# -----------------------------

def batch_upsert_with_retry(
    qdrant_client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = QDRANT_BATCH_SIZE,
    max_retries: int = QDRANT_MAX_RETRIES,
    retry_delay: int = QDRANT_RETRY_DELAY
) -> None:
    """
    Upsert chunks to Qdrant in batches with retry logic to handle timeouts.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the collection
        chunks: List of chunk dictionaries with id_suffix and payload
        embeddings: List of embedding vectors (one per chunk)
        batch_size: Number of chunks to process per batch (default: QDRANT_BATCH_SIZE)
        max_retries: Maximum number of retry attempts (default: QDRANT_MAX_RETRIES)
        retry_delay: Delay in seconds between retries (default: QDRANT_RETRY_DELAY)
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return
    
    # Get file_id from first chunk for ID generation
    file_id = chunks[0]["payload"].get("file_id", "unknown")
    
    # Process in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]
        batch_embeddings = embeddings[batch_start:batch_end]
        
        # Prepare batch
        batch_ids = [uuid5(NAMESPACE_URL, f"{file_id}_{c['id_suffix']}") for c in batch_chunks]
        batch_payloads = [c["payload"] for c in batch_chunks]
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=qmodels.Batch(
                        ids=batch_ids,
                        vectors=batch_embeddings,
                        payloads=batch_payloads,
                    ),
                )
                # Success - move to next batch
                if batch_start == 0:
                    print(f"  [BATCH {batch_start//batch_size + 1}] Successfully upserted {len(batch_chunks)} chunks")
                break
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    print(f"  [BATCH {batch_start//batch_size + 1}] Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if is_timeout:
                        print(f"  [BATCH {batch_start//batch_size + 1}] Timeout detected. Retrying in {wait_time} seconds...")
                    else:
                        print(f"  [BATCH {batch_start//batch_size + 1}] Error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise Exception(
                        f"Failed to upsert batch {batch_start//batch_size + 1} after {max_retries} attempts. "
                        f"Last error: {e}"
                    )


# -----------------------------
# Helper function for processing spreadsheets
# -----------------------------


def _process_spreadsheet_files(
    drive_service: Any,
    qdrant_client: QdrantClient,
    collection_name: str,
    spreadsheet_files: List[Dict[str, Any]],
    task_type: str,
    folder_path: str,
) -> int:
    """
    Process a list of spreadsheet files and index lightweight metadata into Qdrant.
    Stores only ONE document per brief (not per-campaign) with metadata for similarity search.
    Full campaign data remains in Google Drive and can be fetched on-demand.
    
    Args:
        drive_service: Google Drive API service
        qdrant_client: Qdrant client
        collection_name: Name of the Qdrant collection
        spreadsheet_files: List of file metadata dicts from Google Drive
        task_type: The task type folder name (e.g., "Campaign Update", "New Creative", "Theme")
        folder_path: The folder path for logging (e.g., "Campaign Update/Actual", "New Creative")
    
    Returns:
        Number of files successfully processed
    """
    processed_count = 0
    
    for f in spreadsheet_files:
        file_id = f["id"]
        file_name = f["name"]
        mime_type = f["mimeType"]

        if not needs_update(qdrant_client, collection_name, f):
            print(f"      [SKIP] {file_name} (id={file_id}) is up to date.")
            continue

        print(f"      [UPDATE] Processing: {file_name} (id={file_id}, mime={mime_type})")

        try:
            content = download_file_content(drive_service, file_id, mime_type)
            
            # Extract lightweight metadata only (not full campaign data)
            metadata, compact_text = extract_campaign_metadata(content, f)
            
            if not compact_text.strip():
                print(f"      [WARN] No metadata extracted from {file_name}, skipping.")
                continue

            # Create a single embedding for the entire brief metadata
            # (not chunked, since it's already compact)
            embedding = embed_text([compact_text])[0]

            # Build payload with metadata and Google Drive file_id for on-demand fetching
            payload = {
                "file_type": "campaign_metadata",
                "file_id": file_id,  # Google Drive file ID for fetching full data
                "file_name": metadata["file_name"],
                "task_type": metadata["task_type"],
                "dealership_name": metadata["dealership_name"],
                "asset_summary": metadata["asset_summary"],
                "total_campaigns": metadata["total_campaigns"],
                "campaign_ids": metadata["campaign_ids"],
                "file_modified_time": metadata["file_modified_time"],
                "folder_task_type": task_type,  # From folder structure
                "folder_path": folder_path,  # From folder structure
                "text": compact_text,  # For backward compatibility
                "page_content": compact_text,  # For LangChain compatibility
            }

            # Use file_id as the unique identifier (single point per brief)
            # This ensures we only have one document per brief in Qdrant
            point_id = uuid5(NAMESPACE_URL, f"campaign_metadata_{file_id}")

            # Retry logic for single point upsert
            for attempt in range(QDRANT_MAX_RETRIES):
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=qmodels.Batch(
                            ids=[point_id],
                            vectors=[embedding],
                            payloads=[payload],
                        ),
                    )
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    is_timeout = "timeout" in error_msg or "timed out" in error_msg
                    
                    if attempt < QDRANT_MAX_RETRIES - 1:
                        wait_time = QDRANT_RETRY_DELAY * (attempt + 1)
                        print(f"      [RETRY] Attempt {attempt + 1}/{QDRANT_MAX_RETRIES} failed: {e}")
                        if is_timeout:
                            print(f"      [RETRY] Timeout detected. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Failed to upsert campaign metadata after {QDRANT_MAX_RETRIES} attempts: {e}")

            print(
                f"      [OK] Indexed metadata for {file_name} "
                f"({metadata['total_campaigns']} campaigns) "
                f"into collection '{collection_name}'"
            )
            processed_count += 1

        except Exception as e:
            print(f"      [ERROR] Failed to process {file_name} (id={file_id}): {e}")
            import traceback
            traceback.print_exc()
    
    return processed_count


# -----------------------------
# Main sync function
# -----------------------------


def sync_from_gdrive_folder(
    folder_id: str,
    collection_name: str,
    drive_service: Optional[Any] = None,
    qdrant_client: Optional[QdrantClient] = None,
    campaigns_folder_name: Optional[str] = None,
) -> int:
    """
    Syncs (new + updated) documents from a Google Drive folder to a Qdrant collection.
    
    Supports nested folder structure:
    - Outer folder: Contains best practices documentation (PDFs, DOCX files)
    - Inner folder (optional): Contains campaign spreadsheets (.xlsx files)
    
    Steps:
        1. List files in the outer folder (pdf/docx - documentation).
        2. If campaigns_folder_name is specified, find and process spreadsheets in that subfolder.
        3. For each file, check if it needs update.
        4. Download file, extract text, chunk.
        5. Embed chunks and upsert to Qdrant.

    Args:
        folder_id: ID of the main Google Drive folder
        collection_name: Name of the Qdrant collection
        drive_service: Optional Google Drive service (will be created if None)
        qdrant_client: Optional Qdrant client (will be created if None)
        campaigns_folder_name: Optional name of the inner folder containing campaign spreadsheets.
                              If None, only processes documents in the outer folder.
                              Can also be set via CAMPAIGNS_FOLDER_NAME environment variable.

    Returns:
        Number of files processed (indexed or re-indexed).
    """
    if drive_service is None:
        drive_service = get_drive_service()

    if qdrant_client is None:
        qdrant_client = get_qdrant_client()

    # Ensure collection exists with correct vector size (3072 for text-embedding-3-large)
    # This will automatically fix dimension mismatches if they exist
    ensure_qdrant_collection(qdrant_client, collection_name)

    # Get campaigns folder name from parameter or environment variable
    if campaigns_folder_name is None:
        campaigns_folder_name = os.getenv("CAMPAIGNS_FOLDER_NAME", None)

    processed_count = 0

    # Step 1: Process documentation files in the outer folder (PDFs and DOCX, not XLSX)
    doc_mime_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    doc_files = list_files_in_folder(drive_service, folder_id, file_types=doc_mime_types)
    print(f"Found {len(doc_files)} documentation files in outer folder {folder_id}")

    for f in doc_files:
        file_id = f["id"]
        file_name = f["name"]
        mime_type = f["mimeType"]

        if not needs_update(qdrant_client, collection_name, f):
            print(f"[SKIP] {file_name} (id={file_id}) is up to date.")
            continue

        print(f"[UPDATE] Processing documentation: {file_name} (id={file_id}, mime={mime_type})")

        try:
            content = download_file_content(drive_service, file_id, mime_type)
            text, structured_data = extract_text_for_file(mime_type, content, f)
            if not text.strip():
                print(f"[WARN] No text extracted from {file_name}, skipping.")
                continue

            chunks = chunk_text(text, f)
            if not chunks:
                print(f"[WARN] No chunks generated for {file_name}, skipping.")
                continue

            # Documentation files don't have structured_data, so file_type is "document"
            for chunk in chunks:
                chunk["payload"]["file_type"] = "document"

            texts = [c["text"] for c in chunks]
            
            # Generate embeddings in batches to avoid memory issues
            print(f"  Generating embeddings for {len(chunks)} chunks...")
            embeddings = embed_text(texts)
            print(f"  ✓ Generated {len(embeddings)} embeddings")

            # Upsert in batches with retry logic to handle timeouts
            print(f"  Upserting {len(chunks)} chunks in batches of {QDRANT_BATCH_SIZE}...")
            batch_upsert_with_retry(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                batch_size=QDRANT_BATCH_SIZE,
                max_retries=QDRANT_MAX_RETRIES,
                retry_delay=QDRANT_RETRY_DELAY
            )

            print(
                f"[OK] Indexed {len(chunks)} chunks for {file_name} "
                f"into collection '{collection_name}'"
            )
            processed_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {file_name} (id={file_id}): {e}")

    # Step 2: Process campaign spreadsheets in the nested folder structure (if specified)
    if campaigns_folder_name:
        campaigns_folder = find_folder_by_name(drive_service, folder_id, campaigns_folder_name)
        
        if campaigns_folder:
            campaigns_folder_id = campaigns_folder["id"]
            print(f"\nFound campaigns folder '{campaigns_folder_name}' (id={campaigns_folder_id})")
            
            # Process the nested folder structure:
            # Campaigns/
            #   ├── Campaign Update/
            #   │   ├── Actual/
            #   │   └── Previous/
            #   ├── New Creative/
            #   └── Theme/
            
            task_type_folders = ["Campaign Update", "New Creative", "Theme"]
            xlsx_mime_type = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
            
            for task_type_name in task_type_folders:
                task_type_folder = find_folder_by_name(drive_service, campaigns_folder_id, task_type_name)
                
                if not task_type_folder:
                    print(f"[WARN] Task type folder '{task_type_name}' not found in '{campaigns_folder_name}', skipping.")
                    continue
                
                task_type_folder_id = task_type_folder["id"]
                print(f"\n  Processing task type folder: '{task_type_name}' (id={task_type_folder_id})")
                
                # For "Campaign Update", process "Actual" and "Previous" subfolders
                if task_type_name == "Campaign Update":
                    update_subfolders = ["Actual", "Previous"]
                    
                    for subfolder_name in update_subfolders:
                        subfolder = find_folder_by_name(drive_service, task_type_folder_id, subfolder_name)
                        
                        if not subfolder:
                            print(f"    [WARN] Subfolder '{subfolder_name}' not found in 'Campaign Update', skipping.")
                            continue
                        
                        subfolder_id = subfolder["id"]
                        print(f"    Processing subfolder: '{subfolder_name}' (id={subfolder_id})")
                        
                        spreadsheet_files = list_files_in_folder(drive_service, subfolder_id, file_types=xlsx_mime_type)
                        print(f"    Found {len(spreadsheet_files)} spreadsheets in '{task_type_name}/{subfolder_name}'")
                        
                        processed_count += _process_spreadsheet_files(
                            drive_service=drive_service,
                            qdrant_client=qdrant_client,
                            collection_name=collection_name,
                            spreadsheet_files=spreadsheet_files,
                            task_type=task_type_name,
                            folder_path=f"{task_type_name}/{subfolder_name}"
                        )
                else:
                    # For "New Creative" and "Theme", process spreadsheets directly in the folder
                    spreadsheet_files = list_files_in_folder(drive_service, task_type_folder_id, file_types=xlsx_mime_type)
                    print(f"  Found {len(spreadsheet_files)} spreadsheets in '{task_type_name}'")
                    
                    processed_count += _process_spreadsheet_files(
                        drive_service=drive_service,
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        spreadsheet_files=spreadsheet_files,
                        task_type=task_type_name,
                        folder_path=task_type_name
                    )
        else:
            print(f"\n[WARN] Campaigns folder '{campaigns_folder_name}' not found in folder {folder_id}")
            print("       Only processing documentation files in the outer folder.")

    print(f"\nSync completed. Processed {processed_count} file(s).")
    return processed_count
