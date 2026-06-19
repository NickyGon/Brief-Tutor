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
import json
import hashlib
import math
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from datetime import datetime

from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pypdf import PdfReader
from docx import Document as DocxDocument  # for .docx files
from dotenv import load_dotenv
from uuid import uuid5, NAMESPACE_URL

import pandas as pd
from openpyxl import load_workbook
from graph.llm_provider import create_embeddings, get_primary_provider

# Conditional imports that may fail
try:
    from graph.tools import _parse_spreadsheet_internal
except ImportError:
    _parse_spreadsheet_internal = None

try:
    from graph.models import CampaignDiagnosis
except ImportError:
    CampaignDiagnosis = None

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

def infer_vector_size_from_model(model_name: str) -> int:
    """
    Infer vector dimension from known embedding model names.

    Defaults to 1536 when unknown to preserve backward compatibility.
    """
    model = (model_name or "").strip().lower()

    # OpenAI embedding families
    if "text-embedding-3-large" in model:
        return 3072
    if "text-embedding-3-small" in model:
        return 1536

    # Vertex embedding families
    if "text-embedding-005" in model:
        return 768
    if "text-multilingual-embedding-002" in model:
        return 768
    if "gemini-embedding-001" in model:
        return 3072

    # Legacy/fallback heuristic
    if "large" in model:
        return 3072
    return 1536

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
    default_size = infer_vector_size_from_model(embedding_model)
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


def get_service_account_email() -> Optional[str]:
    """
    Helper function to extract the service account email from the JSON file.
    
    Returns:
        Service account email address or None if not found
    """
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not service_account_file:
        return None
    
    try:
        with open(service_account_file, 'r') as f:
            account_info = json.load(f)
            return account_info.get("client_email")
    except Exception:
        return None


def get_drive_service(write_access: bool = False) -> Any:
    """
    Builds an authenticated Google Drive API service using a service account.

    Expects:
        - GOOGLE_SERVICE_ACCOUNT_FILE: path to your service account JSON.
    
    Args:
        write_access: If True, requests write access (readonly by default for safety)
    """
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not service_account_file:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_FILE env var is not set. "
            "Point it to your service account JSON file."
        )

    if write_access:
        # Use drive scope to allow creating files in existing folders
        scopes = ["https://www.googleapis.com/auth/drive"]  # Full read/write access to Google Drive
    else:
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
                pageToken=page_token
            )
            .execute()
        )

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return files


def share_folder_with_service_account(
    drive_service: Any, folder_id: str, service_account_email: Optional[str] = None
) -> bool:
    """
    Share a Google Drive folder with the service account and grant Editor permissions.
    
    This function requires write access to Google Drive. If the provided drive_service
    has read-only access, a new write-access service will be created automatically.
    
    Args:
        drive_service: Google Drive API service (read-only or write-access)
        folder_id: ID of the folder to share
        service_account_email: Service account email (will be retrieved if not provided)
    
    Returns:
        True if sharing was successful, False otherwise
    """
    if service_account_email is None:
        service_account_email = get_service_account_email()
    
    if not service_account_email:
        print("[Drive] Warning: Could not determine service account email for sharing")
        return False
    
    # Check if we need write access and create a new service if needed
    # We'll try with the provided service first, and if it fails with insufficient scopes,
    # create a new service with write access
    try:
        # Create permission to share folder with service account as Editor
        permission = {
            "type": "user",
            "role": "writer",  # Editor role in Drive API
            "emailAddress": service_account_email
        }
        
        drive_service.permissions().create(
            fileId=folder_id,
            body=permission,
            fields="id"
        ).execute()
        
        print(f"[Drive] ✓ Shared folder (ID: {folder_id}) with service account: {service_account_email}")
        return True
    except Exception as e:
        # If permission already exists, that's fine
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower() or "Permission already exists" in str(e):
            print(f"[Drive] Folder already shared with service account: {service_account_email}")
            return True
        
        # If we get an insufficient scopes error, try with a write-access service
        if "insufficient" in str(e).lower() and ("scope" in str(e).lower() or "permission" in str(e).lower()):
            try:
                # Create a new drive service with write access for sharing
                write_drive_service = get_drive_service(write_access=True)
                write_drive_service.permissions().create(
                    fileId=folder_id,
                    body=permission,
                    fields="id"
                ).execute()
                print(f"[Drive] ✓ Shared folder (ID: {folder_id}) with service account: {service_account_email} (using write-access service)")
                return True
            except Exception as write_error:
                print(f"[Drive] Warning: Failed to share folder with service account: {write_error}")
                return False
        
        print(f"[Drive] Warning: Failed to share folder with service account: {e}")
        return False


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
                pageToken=page_token
            )
            .execute()
        )

        folders.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return folders


def find_folder_by_name(
    drive_service: Any, parent_folder_id: str, folder_name: str, auto_share: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Find a subfolder by name within a parent folder.
    Optionally shares the folder with the service account if found.

    Args:
        drive_service: Google Drive API service
        parent_folder_id: ID of the parent folder
        folder_name: Name of the folder to find
        auto_share: If True, automatically share found folder with service account

    Returns:
        Folder dict with {id, name, mimeType, modifiedTime} or None if not found
    """
    folders = list_folders_in_folder(drive_service, parent_folder_id)
    for folder in folders:
        if folder.get("name") == folder_name:
            # Automatically share the folder with service account if requested
            if auto_share:
                share_folder_with_service_account(drive_service, folder["id"])
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
    reader = PdfReader(io.BytesIO(content))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(content: bytes) -> str:
    """
    Extracts text from a .docx file (bytes) using python-docx.
    """
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
        if _parse_spreadsheet_internal is None:
            raise ImportError("_parse_spreadsheet_internal not available")
    except (ImportError, AttributeError):
        # Fallback if import fails
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
        if _parse_spreadsheet_internal is None:
            raise ImportError("_parse_spreadsheet_internal not available")
    except (ImportError, AttributeError):
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


def extract_diagnosis_metadata(
    content: bytes,
    file_meta: Dict[str, Any]
) -> tuple[Dict[str, Any], str]:
    """
    Extract metadata from diagnosis spreadsheets (one row per diagnosis).
    
    Expected spreadsheet format:
    - Columns: campaign_id, status, diagnosis, issues, recommendations, task_type, dealership_name, diagnosis_date, qa_result
    - One row per diagnosis
    - First row is header
    
    Args:
        content: Raw bytes of the Excel file
        file_meta: File metadata dict from Google Drive with keys: id, name, modifiedTime
    
    Returns:
        tuple: (metadata_dict, compact_text_representation)
        - metadata_dict: Contains file_id, file_name, task_type, dealership_name,
                        statuses (list), campaign_ids (list), total_diagnoses, diagnosis_date, qa_result
        - compact_text_representation: Text string for embedding/search (includes diagnosis text, issues, recommendations)
    """
    temp_file = None
    try:
        # Create temporary file to save the Excel content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
            tmp.write(content)
            temp_file = tmp.name
        
        # Read Excel file
        df = pd.read_excel(temp_file, header=0)
        
        # Expected columns: campaign_id, status, diagnosis, issues, recommendations, task_type, dealership_name, diagnosis_date, qa_result
        required_columns = ['campaign_id', 'status', 'diagnosis']
        optional_columns = ['issues', 'recommendations', 'task_type', 'dealership_name', 'diagnosis_date', 'qa_result']
        
        # Check if required columns exist
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Extract metadata from first row (should be consistent across all rows)
        task_type = None
        dealership_name = None
        diagnosis_date = None
        qa_result = None
        
        if 'task_type' in df.columns:
            task_type = df['task_type'].iloc[0] if not df['task_type'].isna().iloc[0] else None
            if pd.notna(task_type):
                task_type = str(task_type).strip()
        
        if 'dealership_name' in df.columns:
            dealership_name = df['dealership_name'].iloc[0] if not df['dealership_name'].isna().iloc[0] else None
            if pd.notna(dealership_name):
                dealership_name = str(dealership_name).strip()
        
        if 'diagnosis_date' in df.columns:
            diagnosis_date = df['diagnosis_date'].iloc[0] if not df['diagnosis_date'].isna().iloc[0] else None
            if pd.notna(diagnosis_date):
                # Convert to ISO format string if it's a datetime
                if isinstance(diagnosis_date, pd.Timestamp):
                    diagnosis_date = diagnosis_date.isoformat()
                else:
                    diagnosis_date = str(diagnosis_date).strip()
        
        if 'qa_result' in df.columns:
            qa_result_val = df['qa_result'].iloc[0] if not df['qa_result'].isna().iloc[0] else None
            if pd.notna(qa_result_val):
                # Convert to boolean (handle string "true"/"false" or actual boolean)
                if isinstance(qa_result_val, bool):
                    qa_result = qa_result_val
                elif isinstance(qa_result_val, str):
                    qa_result = qa_result_val.strip().lower() in ('true', '1', 'yes')
                else:
                    qa_result = bool(qa_result_val)
        
        # Extract unique statuses and campaign IDs
        statuses = df['status'].dropna().unique().tolist()
        statuses = [str(s).strip() for s in statuses if pd.notna(s)]
        
        campaign_ids = df['campaign_id'].dropna().unique().tolist()
        campaign_ids = [str(cid).strip() for cid in campaign_ids if pd.notna(cid)]
        
        total_diagnoses = len(df)
        
        # Build text representation for embedding (include diagnosis text, issues, recommendations)
        text_parts = [
            f"Campaign Diagnoses: {file_meta.get('name', 'Unknown')}",
            f"Task Type: {task_type or 'Unknown'}",
            f"Dealership: {dealership_name or 'Unknown'}",
            f"Total Diagnoses: {total_diagnoses}",
            f"Statuses: {', '.join(statuses) if statuses else 'None'}",
            f"Campaign IDs: {', '.join(campaign_ids) if campaign_ids else 'None'}"
        ]
        
        # Add diagnosis summaries (first few for compact representation)
        for idx, row in df.head(5).iterrows():
            diag_text = str(row['diagnosis']).strip() if pd.notna(row['diagnosis']) else ""
            status = str(row['status']).strip() if pd.notna(row['status']) else ""
            campaign_id = str(row['campaign_id']).strip() if pd.notna(row['campaign_id']) else ""
            
            text_parts.append(f"\nDiagnosis {idx + 1} ({campaign_id} - {status}):")
            if diag_text:
                text_parts.append(f"  {diag_text[:200]}...")  # Truncate for compact representation
            
            # Add issues if present
            if 'issues' in df.columns and pd.notna(row['issues']):
                issues_val = row['issues']
                if isinstance(issues_val, str):
                    try:
                        issues_list = json.loads(issues_val)
                    except (json.JSONDecodeError, TypeError):
                        issues_list = [i.strip() for i in issues_val.split(',') if i.strip()]
                elif isinstance(issues_val, list):
                    issues_list = issues_val
                else:
                    issues_list = []
                if issues_list:
                    text_parts.append(f"  Issues: {', '.join(str(i)[:50] for i in issues_list[:3])}")
            
            # Add recommendations if present
            if 'recommendations' in df.columns and pd.notna(row['recommendations']):
                recs_val = row['recommendations']
                if isinstance(recs_val, str):
                    try:
                        recs_list = json.loads(recs_val)
                    except (json.JSONDecodeError, TypeError):
                        recs_list = [r.strip() for r in recs_val.split(',') if r.strip()]
                elif isinstance(recs_val, list):
                    recs_list = recs_val
                else:
                    recs_list = []
                if recs_list:
                    text_parts.append(f"  Recommendations: {', '.join(str(r)[:50] for r in recs_list[:3])}")
        
        if total_diagnoses > 5:
            text_parts.append(f"\n... and {total_diagnoses - 5} more diagnoses")
        
        compact_text = "\n".join(text_parts)
        
        metadata = {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "task_type": task_type,
            "dealership_name": dealership_name,
            "statuses": statuses,
            "campaign_ids": campaign_ids,
            "total_diagnoses": total_diagnoses,
            "diagnosis_date": diagnosis_date,
            "qa_result": qa_result,
            "file_type": "campaign_diagnosis",
            "file_modified_time": file_meta.get("modifiedTime"),
        }
        
        return metadata, compact_text
        
    except Exception as e:
        # Fallback: return minimal metadata on error
        print(f"[WARN] Failed to extract diagnosis metadata from {file_meta.get('name', 'Unknown')}: {e}")
        traceback.print_exc()
        return {
            "file_id": file_meta["id"],
            "file_name": file_meta["name"],
            "task_type": None,
            "dealership_name": None,
            "statuses": [],
            "campaign_ids": [],
            "total_diagnoses": 0,
            "diagnosis_date": None,
            "qa_result": None,
            "file_type": "campaign_diagnosis",
            "file_modified_time": file_meta.get("modifiedTime"),
        }, f"Campaign Diagnoses: {file_meta.get('name', 'Unknown')}"
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


def store_diagnoses_to_drive(
    diagnoses: List[Any],
    campaign_brief: Dict[str, Any],
    qa_result: bool,
    drive_service: Optional[Any] = None,
    main_drive_folder_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store diagnoses as Excel spreadsheet in Google Drive after QA passes.
    
    Uses Template-diagnoses.xlsx as a template and fills in diagnosis data.
    Uploads to Google Drive in the appropriate folder structure: Campaigns/[Task Type]/Diagnoses/
    
    Args:
        diagnoses: List of CampaignDiagnosis objects (or dicts with diagnosis data)
        campaign_brief: CampaignBrief metadata dict (must have task_type, dealership_name, spreadsheet_path)
        qa_result: QA result (True if passed, False if failed) - not stored in spreadsheet
        drive_service: Optional Google Drive service (will be created if None)
        main_drive_folder_id: Optional main Drive folder ID (defaults to env var CAMPAIGNS_DRIVE_FOLDER_ID)
    
    Returns:
        Dict with file_id and file metadata from Google Drive
    """
    # Get drive service with write access
    if drive_service is None:
        drive_service = get_drive_service(write_access=True)
    
    # Get main folder ID from parameter or environment
    if main_drive_folder_id is None:
        main_drive_folder_id = os.getenv("CAMPAIGNS_DRIVE_FOLDER_ID")
        if not main_drive_folder_id:
            raise RuntimeError(
                "main_drive_folder_id parameter or CAMPAIGNS_DRIVE_FOLDER_ID env var must be set"
            )
    
    # Extract metadata from campaign_brief
    task_type = campaign_brief.get("task_type", "")
    dealership_name = campaign_brief.get("dealership_name")
    spreadsheet_path = campaign_brief.get("spreadsheet_path", "")
    
    if not task_type:
        raise ValueError("campaign_brief must have a task_type")
    
    # Extract filename from spreadsheet_path for naming pattern: [spreadsheetFileName]-diagnoses.xlsx
    spreadsheet_filename = ""
    if spreadsheet_path:
        # Extract filename from path (handle both local paths and URLs)
        if "/" in spreadsheet_path:
            spreadsheet_filename = spreadsheet_path.split("/")[-1]
        elif "\\" in spreadsheet_path:
            spreadsheet_filename = spreadsheet_path.split("\\")[-1]
        else:
            spreadsheet_filename = spreadsheet_path
        
        # Remove extension if present
        if "." in spreadsheet_filename:
            spreadsheet_filename = spreadsheet_filename.rsplit(".", 1)[0]
    
    if not spreadsheet_filename:
        # Fallback: use date-based filename
        current_date = datetime.now()
        date_str = current_date.strftime("%Y-%m-%d")
        spreadsheet_filename = f"{date_str}-diagnoses"
    
    filename = f"{spreadsheet_filename}-diagnoses.xlsx"
    
    # Get current date for diagnosis_date column (MM/DD/YYYY format)
    current_date = datetime.now()
    date_formatted = current_date.strftime("%m/%d/%Y")
    
    # Load template Excel file
    project_root = Path(__file__).parent
    template_path = project_root / "Template-diagnoses.xlsx"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load template workbook
    template_wb = load_workbook(template_path)
    template_ws = template_wb.active
    
    # Get headers from first row (row 1 in openpyxl, which is 1-indexed)
    headers = []
    header_to_col = {}  # Map normalized header names to column indices
    for col_idx, cell in enumerate(template_ws[1], start=1):
        header_value = cell.value if cell.value else ""
        headers.append(header_value)
        # Normalize header for matching (lowercase, strip whitespace)
        normalized_header = str(header_value).lower().strip() if header_value else ""
        header_to_col[normalized_header] = col_idx
    
    # Build data rows starting from row 2 (after headers)
    row_num = 2
    for diag in diagnoses:
        # Extract data from CampaignDiagnosis object or dict
        if CampaignDiagnosis and isinstance(diag, CampaignDiagnosis):
            campaign_id = diag.campaign_id
            status = diag.status
            diagnosis = diag.diagnosis
            issues_list = diag.issues if diag.issues else []
            recommendations_list = diag.recommendations if diag.recommendations else []
        elif isinstance(diag, dict):
            campaign_id = diag.get("campaign_id", "")
            status = diag.get("status", "")
            diagnosis = diag.get("diagnosis", "")
            issues_list = diag.get("issues", []) if diag.get("issues") else []
            recommendations_list = diag.get("recommendations", []) if diag.get("recommendations") else []
        else:
            continue  # Skip invalid entries
        
        # Format issues and recommendations: join with line breaks
        issues_text = "\n".join(issues_list) if issues_list else ""
        recommendations_text = "\n".join(recommendations_list) if recommendations_list else ""
        
        # Map data fields to normalized header names
        data_mapping = {
            "campaign_id": campaign_id,
            "status": status,
            "diagnosis": diagnosis,
            "issues": issues_text,
            "recommendations": recommendations_text,
            "task_type": task_type,
            "dealership_name": dealership_name or "",
            "diagnosis_date": date_formatted
        }
        
        # Write row data to template worksheet using normalized header matching
        for field_name, field_value in data_mapping.items():
            normalized_field = field_name.lower().strip()
            if normalized_field in header_to_col:
                col_idx = header_to_col[normalized_field]
                template_ws.cell(row=row_num, column=col_idx, value=field_value)
        
        row_num += 1
    
    if row_num == 2:
        raise ValueError("No valid diagnoses provided")
    
    # Convert workbook to bytes
    excel_buffer = io.BytesIO()
    template_wb.save(excel_buffer)
    excel_buffer.seek(0)
    excel_content = excel_buffer.read()
    
    # Find existing folder structure: Campaigns/[Task Type]/Diagnoses/
    # For Campaign Update: Campaigns/Campaign Update/Actual/Diagnoses/
    # Automatically share each folder with service account as we find it
    # Step 1: Find "Campaigns" folder and share it
    campaigns_folder = find_folder_by_name(drive_service, main_drive_folder_id, "Campaigns", auto_share=True)
    if not campaigns_folder:
        raise RuntimeError(
            f"Folder 'Campaigns' not found in parent folder (ID: {main_drive_folder_id}). "
            f"Please ensure the folder structure exists in Google Drive."
        )
    
    campaigns_folder_id = campaigns_folder["id"]
    
    # Step 2: Find task type folder (e.g., "New Creative", "Theme", "Campaign Update") and share it
    task_type_folder = find_folder_by_name(drive_service, campaigns_folder_id, task_type, auto_share=True)
    if not task_type_folder:
        raise RuntimeError(
            f"Folder '{task_type}' not found in 'Campaigns' folder. "
            f"Please ensure the folder structure exists in Google Drive."
        )
    
    task_type_folder_id = task_type_folder["id"]
    
    # Step 3: For Campaign Update, find "Actual" subfolder, then "Diagnoses"
    # For other task types, find "Diagnoses" directly
    if task_type.lower() == "campaign update":
        # Campaign Update has an "Actual" subfolder: Campaigns/Campaign Update/Actual/Diagnoses/
        actual_folder = find_folder_by_name(drive_service, task_type_folder_id, "Actual", auto_share=True)
        if not actual_folder:
            raise RuntimeError(
                f"Folder 'Actual' not found in '{task_type}' folder. "
                f"Please ensure the folder structure 'Campaigns/Campaign Update/Actual/' exists in Google Drive."
            )
        actual_folder_id = actual_folder["id"]
        
        # Find "Diagnoses" folder inside "Actual" and share it
        diagnoses_folder = find_folder_by_name(drive_service, actual_folder_id, "Diagnoses", auto_share=True)
        if not diagnoses_folder:
            raise RuntimeError(
                f"Folder 'Diagnoses' not found in 'Campaigns/{task_type}/Actual/'. "
                f"Please ensure the folder structure exists in Google Drive."
            )
    else:
        # For Theme and New Creative: Campaigns/[Task Type]/Diagnoses/
        diagnoses_folder = find_folder_by_name(drive_service, task_type_folder_id, "Diagnoses", auto_share=True)
        if not diagnoses_folder:
            raise RuntimeError(
                f"Folder 'Diagnoses' not found in '{task_type}' folder. "
                f"Please ensure the folder structure 'Campaigns/{task_type}/Diagnoses/' exists in Google Drive."
            )
    
    diagnoses_folder_id = diagnoses_folder["id"]
    
    # Step 4: Upload Excel file to Diagnoses folder
    file_metadata = {
        "name": filename,
        "parents": [diagnoses_folder_id]
    }
    
    media = MediaIoBaseUpload(
        io.BytesIO(excel_content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=True
    )
    
    try:
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, mimeType, modifiedTime, createdTime"
        ).execute()
        
        print(f"[Diagnosis Storage] Uploaded diagnosis spreadsheet: {filename} (id={file['id']})")
    except Exception as e:
        if "insufficientParentPermissions" in str(e) or "403" in str(e):
            # Get service account email for helpful error message
            service_account_email = get_service_account_email()
            
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Insufficient permissions to upload files to Google Drive.\n"
                f"{'='*80}\n"
                f"The service account needs 'Editor' access to the folder specified by\n"
                f"CAMPAIGNS_DRIVE_FOLDER_ID.\n\n"
                f"To fix this:\n"
                f"1. Go to Google Drive and open the folder (ID: {main_drive_folder_id})\n"
                f"2. Right-click the folder → Share\n"
                f"3. Add the service account email as an Editor\n"
                f"4. Click Send\n\n"
            )
            if service_account_email:
                error_msg += f"Service account email: {service_account_email}\n\n"
            else:
                error_msg += f"Note: Open your service account JSON file to find the 'client_email' field\n\n"
            error_msg += f"{'='*80}\n"
            raise RuntimeError(error_msg) from e
        raise
    
    # Build folder path for return value
    if task_type.lower() == "campaign update":
        folder_path = "Campaigns/Campaign Update/Actual/Diagnoses/"
    else:
        folder_path = f"Campaigns/{task_type}/Diagnoses/"
    
    return {
        "file_id": file["id"],
        "file_name": file["name"],
        "file_metadata": file,
        "folder_path": folder_path
    }


def store_text_document_to_drive(
    content: str,
    file_name: str,
    task_type: str,
    drive_service: Optional[Any] = None,
    main_drive_folder_id: Optional[str] = None
) -> Dict[str, Any]:
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
        Dict with file_id, file_name, and folder_path from Google Drive
    """
    # Get drive service with write access
    if drive_service is None:
        drive_service = get_drive_service(write_access=True)
    
    # Get main folder ID from parameter or environment
    if main_drive_folder_id is None:
        main_drive_folder_id = os.getenv("CAMPAIGNS_DRIVE_FOLDER_ID")
        if not main_drive_folder_id:
            raise RuntimeError(
                "main_drive_folder_id parameter or CAMPAIGNS_DRIVE_FOLDER_ID env var must be set"
            )
    
    if not task_type:
        raise ValueError("task_type must be provided")
    
    # Find existing folder structure: Campaigns/[Task Type]/Diagnoses/
    # For Campaign Update: Campaigns/Campaign Update/Actual/Diagnoses/
    # Automatically share each folder with service account as we find it
    # Step 1: Find "Campaigns" folder and share it
    campaigns_folder = find_folder_by_name(drive_service, main_drive_folder_id, "Campaigns", auto_share=True)
    if not campaigns_folder:
        raise RuntimeError(
            f"Folder 'Campaigns' not found in parent folder (ID: {main_drive_folder_id}). "
            f"Please ensure the folder structure exists in Google Drive."
        )
    
    campaigns_folder_id = campaigns_folder["id"]
    
    # Step 2: Find task type folder (e.g., "New Creative", "Theme", "Campaign Update") and share it
    task_type_folder = find_folder_by_name(drive_service, campaigns_folder_id, task_type, auto_share=True)
    if not task_type_folder:
        raise RuntimeError(
            f"Folder '{task_type}' not found in 'Campaigns' folder. "
            f"Please ensure the folder structure exists in Google Drive."
        )
    
    task_type_folder_id = task_type_folder["id"]
    
    # Step 3: For Campaign Update, find "Actual" subfolder, then "Diagnoses"
    # For other task types, find "Diagnoses" directly
    if task_type.lower() == "campaign update":
        # Campaign Update has an "Actual" subfolder: Campaigns/Campaign Update/Actual/Diagnoses/
        actual_folder = find_folder_by_name(drive_service, task_type_folder_id, "Actual", auto_share=True)
        if not actual_folder:
            raise RuntimeError(
                f"Folder 'Actual' not found in '{task_type}' folder. "
                f"Please ensure the folder structure 'Campaigns/Campaign Update/Actual/' exists in Google Drive."
            )
        actual_folder_id = actual_folder["id"]
        
        # Find "Diagnoses" folder inside "Actual" and share it
        diagnoses_folder = find_folder_by_name(drive_service, actual_folder_id, "Diagnoses", auto_share=True)
        if not diagnoses_folder:
            raise RuntimeError(
                f"Folder 'Diagnoses' not found in 'Campaigns/{task_type}/Actual/'. "
                f"Please ensure the folder structure exists in Google Drive."
            )
    else:
        # For Theme and New Creative: Campaigns/[Task Type]/Diagnoses/
        diagnoses_folder = find_folder_by_name(drive_service, task_type_folder_id, "Diagnoses", auto_share=True)
        if not diagnoses_folder:
            raise RuntimeError(
                f"Folder 'Diagnoses' not found in '{task_type}' folder. "
                f"Please ensure the folder structure 'Campaigns/{task_type}/Diagnoses/' exists in Google Drive."
            )
    
    diagnoses_folder_id = diagnoses_folder["id"]
    
    # Step 4: Upload text file to Diagnoses folder
    file_metadata = {
        "name": file_name,
        "parents": [diagnoses_folder_id]
    }
    
    # Convert text content to bytes
    text_content_bytes = content.encode('utf-8')
    
    media = MediaIoBaseUpload(
        io.BytesIO(text_content_bytes),
        mimetype="text/plain",
        resumable=True
    )
    
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, mimeType, modifiedTime, createdTime"
    ).execute()
    
    print(f"[Document Storage] Uploaded text document: {file_name} (id={file['id']})")
    
    # Build folder path for return value
    if task_type.lower() == "campaign update":
        folder_path = "Campaigns/Campaign Update/Actual/Diagnoses/"
    else:
        folder_path = f"Campaigns/{task_type}/Diagnoses/"
    
    return {
        "file_id": file["id"],
        "file_name": file["name"],
        "file_metadata": file,
        "folder_path": folder_path
    }


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
    
    # Calculate expected size directly from embedding model
    # (don't use get_vector_size() which respects RAG_VECTOR_SIZE overrides).
    expected_size = infer_vector_size_from_model(embedding_model)
    
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
        "status": qmodels.PayloadSchemaType.KEYWORD,  # For diagnosis filtering by status
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
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", get_primary_provider())
    
    try:
        embeddings_client = create_embeddings(provider=embedding_provider, model=embedding_model)
        return embeddings_client.embed_documents(texts)
    
    except (ImportError, NameError):
        print("[WARN] OpenAI library not available. Using fallback hash-based embeddings.")
        print("       Install with: pip install openai")
        # Fallback to hash-based embeddings if OpenAI is not available
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


def sync_diagnoses_from_gdrive_folder(
    main_drive_folder_id: str,
    collection_name: str,
    drive_service: Optional[Any] = None,
    qdrant_client: Optional[QdrantClient] = None,
) -> int:
    """
    Sync diagnosis spreadsheets from Google Drive to Qdrant.
    Scans Campaigns/[Task Type]/Diagnoses/ folders and stores diagnosis metadata in Qdrant.
    
    Args:
        main_drive_folder_id: ID of the main Google Drive folder containing "Campaigns" folder
        collection_name: Name of the Qdrant collection
        drive_service: Optional Google Drive service (will be created if None)
        qdrant_client: Optional Qdrant client (will be created if None)
    
    Returns:
        Number of diagnosis files processed
    """
    if drive_service is None:
        drive_service = get_drive_service()
    
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    
    # Ensure collection exists with correct vector size
    ensure_qdrant_collection(qdrant_client, collection_name)
    
    processed_count = 0
    
    # Find "Campaigns" folder
    campaigns_folder = find_folder_by_name(drive_service, main_drive_folder_id, "Campaigns")
    if not campaigns_folder:
        print(f"[Diagnosis Sync] 'Campaigns' folder not found in {main_drive_folder_id}, skipping diagnosis sync.")
        return 0
    
    campaigns_folder_id = campaigns_folder["id"]
    print(f"[Diagnosis Sync] Found 'Campaigns' folder (id={campaigns_folder_id})")
    
    # Task type folders to scan
    task_type_folders = ["Campaign Update", "New Creative", "Theme"]
    xlsx_mime_type = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    
    for task_type_name in task_type_folders:
        task_type_folder = find_folder_by_name(drive_service, campaigns_folder_id, task_type_name)
        
        if not task_type_folder:
            print(f"[Diagnosis Sync] Task type folder '{task_type_name}' not found, skipping.")
            continue
        
        task_type_folder_id = task_type_folder["id"]
        print(f"\n[Diagnosis Sync] Processing task type folder: '{task_type_name}' (id={task_type_folder_id})")
        
        # For "Campaign Update", look inside "Actual" folder first, then "Diagnoses"
        # For other task types, look directly in "Diagnoses" folder
        if task_type_name == "Campaign Update":
            # Find "Actual" subfolder first
            actual_folder = find_folder_by_name(drive_service, task_type_folder_id, "Actual")
            if not actual_folder:
                print(f"[Diagnosis Sync] 'Actual' folder not found in '{task_type_name}', skipping.")
                continue
            
            actual_folder_id = actual_folder["id"]
            print(f"[Diagnosis Sync] Found 'Actual' folder (id={actual_folder_id})")
            
            # Find "Diagnoses" subfolder inside "Actual"
            diagnoses_folder = find_folder_by_name(drive_service, actual_folder_id, "Diagnoses")
            folder_path_prefix = f"{task_type_name}/Actual"
        else:
            # For other task types, find "Diagnoses" directly under task type folder
            diagnoses_folder = find_folder_by_name(drive_service, task_type_folder_id, "Diagnoses")
            folder_path_prefix = task_type_name
        
        if not diagnoses_folder:
            print(f"[Diagnosis Sync] 'Diagnoses' folder not found in '{folder_path_prefix}', skipping.")
            continue
        
        diagnoses_folder_id = diagnoses_folder["id"]
        print(f"[Diagnosis Sync] Found 'Diagnoses' folder (id={diagnoses_folder_id})")
        
        # List all Excel files in Diagnoses folder
        diagnosis_files = list_files_in_folder(drive_service, diagnoses_folder_id, file_types=xlsx_mime_type)
        print(f"[Diagnosis Sync] Found {len(diagnosis_files)} diagnosis spreadsheet(s) in 'Campaigns/{folder_path_prefix}/Diagnoses'")
        
        for f in diagnosis_files:
            file_id = f["id"]
            file_name = f["name"]
            
            if not needs_update(qdrant_client, collection_name, f):
                print(f"      [SKIP] {file_name} (id={file_id}) is up to date.")
                continue
            
            print(f"      [UPDATE] Processing diagnosis file: {file_name} (id={file_id})")
            
            try:
                content = download_file_content(drive_service, file_id, f["mimeType"])
                
                # Extract diagnosis metadata
                metadata, compact_text = extract_diagnosis_metadata(content, f)
                
                if not compact_text.strip():
                    print(f"      [WARN] No metadata extracted from {file_name}, skipping.")
                    continue
                
                # Create a single embedding for the entire diagnosis spreadsheet metadata
                embedding = embed_text([compact_text])[0]
                
                # Build payload with metadata
                payload = {
                    "file_type": "campaign_diagnosis",
                    "file_id": file_id,
                    "file_name": metadata["file_name"],
                    "task_type": metadata["task_type"],
                    "dealership_name": metadata["dealership_name"],
                    "statuses": metadata["statuses"],  # List of unique statuses
                    "campaign_ids": metadata["campaign_ids"],
                    "total_diagnoses": metadata["total_diagnoses"],
                    "diagnosis_date": metadata["diagnosis_date"],
                    "qa_result": metadata["qa_result"],
                    "file_modified_time": metadata["file_modified_time"],
                    "folder_path": f"Campaigns/{folder_path_prefix}/Diagnoses/",
                    "text": compact_text,
                    "page_content": compact_text,  # For LangChain compatibility
                }
                
                # Use file_id as the unique identifier (single point per diagnosis spreadsheet)
                point_id = uuid5(NAMESPACE_URL, f"campaign_diagnosis_{file_id}")
                
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
                            raise Exception(f"Failed to upsert diagnosis metadata after {QDRANT_MAX_RETRIES} attempts: {e}")
                
                print(
                    f"      [OK] Indexed diagnosis metadata for {file_name} "
                    f"({metadata['total_diagnoses']} diagnoses) "
                    f"into collection '{collection_name}'"
                )
                processed_count += 1
                
            except Exception as e:
                print(f"      [ERROR] Failed to process {file_name} (id={file_id}): {e}")
                traceback.print_exc()
    
    print(f"\nDiagnosis sync completed. Processed {processed_count} file(s).")
    return processed_count
