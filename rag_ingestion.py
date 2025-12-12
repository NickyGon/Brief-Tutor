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

# Adjust to whatever your embedding model outputs.
# e.g. 1536 for OpenAI text-embedding-3-small, 768/1024 for many HF models.
VECTOR_SIZE = int(os.getenv("RAG_VECTOR_SIZE", "1536"))

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
        from graph.tools import load_and_parse_spreadsheet
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
        
        # Use load_and_parse_spreadsheet to parse the file
        # Note: load_and_parse_spreadsheet returns a dict (not a CampaignBrief object)
        campaign_brief = load_and_parse_spreadsheet(temp_file)
        
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
    client: QdrantClient, collection_name: str, vector_size: int = VECTOR_SIZE
) -> None:
    """
    Ensures the Qdrant collection exists with the given vector size.
    If it doesn't, it creates it. Also ensures required payload indexes exist.
    """
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
        print(f"Created collection '{collection_name}'")
    
    # Ensure payload indexes exist for filtering
    # Always try to create indexes - Qdrant will return an error if they already exist
    required_indexes = {
        "file_id": qmodels.PayloadSchemaType.KEYWORD,
        "file_modified_time": qmodels.PayloadSchemaType.KEYWORD,
        "file_type": qmodels.PayloadSchemaType.KEYWORD,
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
    Compute embeddings for a list of texts.

    IMPORTANT: Replace this with your actual embedding provider.

    For example, with OpenAI:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in resp.data]

    Or with a local HuggingFace model, etc.
    """

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
    Process a list of spreadsheet files and index them into Qdrant.
    
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
            text, structured_data = extract_text_for_file(mime_type, content, f)
            if not text.strip():
                print(f"      [WARN] No text extracted from {file_name}, skipping.")
                continue

            chunks = chunk_text(text, f)
            if not chunks:
                print(f"      [WARN] No chunks generated for {file_name}, skipping.")
                continue

            # Add structured data to payload for campaign briefs
            # Include task_type and folder_path from the folder structure
            for chunk in chunks:
                if structured_data:
                    # Add structured brief data to payload for easy retrieval
                    chunk["payload"]["structured_brief"] = structured_data
                    chunk["payload"]["file_type"] = structured_data.get("file_type", "campaign_brief")
                    # Add folder-based task_type (may override spreadsheet's task_type if different)
                    chunk["payload"]["folder_task_type"] = task_type
                    chunk["payload"]["folder_path"] = folder_path
                    # Also update structured_brief with folder information
                    if isinstance(chunk["payload"]["structured_brief"], dict):
                        chunk["payload"]["structured_brief"]["folder_task_type"] = task_type
                        chunk["payload"]["structured_brief"]["folder_path"] = folder_path
                else:
                    chunk["payload"]["file_type"] = "campaign_brief"
                    chunk["payload"]["folder_task_type"] = task_type
                    chunk["payload"]["folder_path"] = folder_path

            texts = [c["text"] for c in chunks]
            embeddings = embed_text(texts)

            # Prepare Qdrant batch
            ids = [ uuid5(NAMESPACE_URL, f"{file_id}_{c['id_suffix']}") for c in chunks ]
            payloads = [c["payload"] for c in chunks]

            qdrant_client.upsert(
                collection_name=collection_name,
                points=qmodels.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads,
                ),
            )

            print(
                f"      [OK] Indexed {len(chunks)} chunks for {file_name} "
                f"into collection '{collection_name}'"
            )
            processed_count += 1

        except Exception as e:
            print(f"      [ERROR] Failed to process {file_name} (id={file_id}): {e}")
    
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

    ensure_qdrant_collection(qdrant_client, collection_name, VECTOR_SIZE)

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
            embeddings = embed_text(texts)

            # Prepare Qdrant batch
            ids = [ uuid5(NAMESPACE_URL, f"{file_id}_{c['id_suffix']}") for c in chunks ]
            payloads = [c["payload"] for c in chunks]

            qdrant_client.upsert(
                collection_name=collection_name,
                points=qmodels.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads,
                ),
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
