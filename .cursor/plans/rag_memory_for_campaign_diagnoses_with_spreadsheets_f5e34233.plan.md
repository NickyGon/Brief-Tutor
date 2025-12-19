---
name: RAG Memory for Campaign Diagnoses with Spreadsheets
overview: Implement a RAG memory system that stores campaign diagnoses in Qdrant using spreadsheet format (one row per diagnosis), allows external editing via Google Drive spreadsheets, automatically stores diagnoses after QA passes, and enables the QA agent to retrieve similar past diagnoses for better judgment.
todos:
  - id: extract_diagnosis_metadata
    content: Create extract_diagnosis_metadata() function in rag_ingestion.py to parse diagnosis spreadsheets (one row per diagnosis) and extract metadata
    status: pending
  - id: store_diagnoses_to_drive
    content: Create store_diagnoses_to_drive() function in rag_ingestion.py to auto-store diagnoses as Excel spreadsheet in Google Drive after QA passes
    status: pending
  - id: sync_diagnoses_function
    content: Create sync_diagnoses_from_gdrive_folder() function in rag_ingestion.py to sync diagnosis spreadsheets from Google Drive to Qdrant
    status: pending
  - id: find_similar_diagnoses_tool
    content: Create find_similar_diagnoses() tool in graph/tools.py for QA agent to retrieve similar past diagnoses
    status: pending
  - id: update_qdrant_indexes
    content: Update ensure_qdrant_collection() in rag_ingestion.py to include status payload index
    status: pending
  - id: add_tool_to_qa
    content: Update get_available_tools() in graph/tools.py to add find_similar_diagnoses to QA agent tools
    status: pending
  - id: update_qa_agent_prompt
    content: Update qa_agent.yaml prompt to instruct using find_similar_diagnoses tool for better judgment
    status: pending
  - id: integrate_auto_storage
    content: Update workflow.py QA agent node to automatically store diagnoses to Google Drive after QA passes
    status: pending
---

# RAG Memory System for Campaign Diagnoses (Spreadsheet-Based)

## Overview

Create a RAG memory system that stores completed campaign diagnoses in Qdrant, similar to how campaign metadata is stored. Diagnoses will be stored in spreadsheet format (one row per diagnosis) in Google Drive, automatically saved after QA passes, and made searchable for the QA agent to improve judgment.

## Architecture

The system will:

1. **Store diagnoses in spreadsheets**: One row per diagnosis with columns: `campaign_id`, `status`, `diagnosis`, `issues`, `recommendations`, plus metadata (`task_type`, `dealership_name`, `diagnosis_date`, `qa_result`)
2. **Auto-store after QA**: Automatically create/update diagnosis spreadsheets in Google Drive after QA agent passes
3. **Sync from Google Drive**: Sync diagnosis spreadsheets to Qdrant (similar to campaign metadata sync)
4. **Enable QA retrieval**: Provide tool for QA agent to find similar past diagnoses

## Implementation Steps

### 1. Create Diagnosis Spreadsheet Parser (`rag_ingestion.py`)

**File:** `rag_ingestion.py`Create `extract_diagnosis_metadata()` function similar to `extract_campaign_metadata()`:

- Accepts spreadsheet content (bytes) and file metadata
- Parses spreadsheet where each row is a diagnosis
- Extracts metadata: `task_type`, `dealership_name`, `statuses` (list), `campaign_ids`, `diagnosis_date`, `total_diagnoses`, `qa_result`
- Creates compact text representation for embedding (includes diagnosis text, issues, recommendations)
- Returns metadata dict and text representation

**Expected spreadsheet format:**

- Columns: `campaign_id`, `status`, `diagnosis`, `issues`, `recommendations`, `task_type`, `dealership_name`, `diagnosis_date`, `qa_result`
- One row per diagnosis
- First row is header

**Key fields in metadata:**

```python
{
    "file_id": str,
    "file_name": str,
    "task_type": str,
    "dealership_name": Optional[str],
    "statuses": List[str],  # ["critical", "observed", "passed"]
    "campaign_ids": List[str],
    "total_diagnoses": int,
    "diagnosis_date": str,  # ISO format
    "qa_result": bool,
    "file_type": "campaign_diagnosis"
}
```



### 2. Create Diagnosis Storage Function (`rag_ingestion.py`)

**File:** `rag_ingestion.py`Create `store_diagnoses_to_drive()` function:

- Accepts list of `CampaignDiagnosis` objects, campaign brief metadata, and QA result
- Creates a pandas DataFrame with one row per diagnosis
- Converts DataFrame to Excel format
- Uploads to Google Drive in appropriate folder structure (e.g., `Campaigns/[Task Type]/Diagnoses/`)
- Returns file_id and file metadata

**Folder structure:**

```javascript
Main Drive Folder/
└── Campaigns/
    ├── New Creative/
    │   └── Diagnoses/
    │       └── 2025-01-15-dealership-A-12345678-diagnoses.xlsx
    ├── Theme/
    │   └── Diagnoses/
    └── Campaign Update/
        └── Diagnoses/
```



### 3. Create Diagnosis Sync Function (`rag_ingestion.py`)

**File:** `rag_ingestion.py`Create `sync_diagnoses_from_gdrive_folder()` function:

- Similar to `sync_from_gdrive_folder()` but specifically for diagnosis spreadsheets
- Scans `Campaigns/[Task Type]/Diagnoses/` folders
- Parses diagnosis spreadsheets using `extract_diagnosis_metadata()`
- Stores in Qdrant with `file_type: "campaign_diagnosis"`

### 4. Create Tool for QA Agent (`graph/tools.py`)

**File:** `graph/tools.py`Create `find_similar_diagnoses()` tool:

- Similar to `find_similar_campaigns()` but for diagnoses
- Filters by `file_type: "campaign_diagnosis"`
- Filters by `task_type` (required)
- Optionally filters by `status` (to find similar status patterns)
- Optionally filters by `dealership_name`
- Returns similar past diagnoses with their full content
- Uses semantic search on diagnosis text for relevance

**Tool signature:**

```python
@tool
def find_similar_diagnoses(
    task_type: str,
    status: Optional[str] = None,
    dealership_name: Optional[str] = None,
    query: Optional[str] = None,  # Semantic search query
    top_k: int = 5,
    collection_name: Optional[str] = None
) -> str
```



### 5. Update Qdrant Collection Setup (`rag_ingestion.py`)

**File:** `rag_ingestion.py`Update `ensure_qdrant_collection()` to include payload indexes for diagnosis fields:

- Add `"status"` to `required_indexes` (for filtering by status)
- Ensure `"task_type"` and `"dealership_name"` indexes exist (already should be there)

### 6. Add Tool to QA Agent (`graph/tools.py`)

**File:** `graph/tools.py`Update `get_available_tools()` to add `find_similar_diagnoses` to QA agent tools:

- Add to `rag_tools_with_briefs` list (or create separate list for QA)

### 7. Update QA Agent Prompt (`agents/qa_agent.yaml`)

**File:** `agents/qa_agent.yaml`Update QA agent prompt to:

- Instruct to use `find_similar_diagnoses` tool to retrieve similar past diagnoses
- Compare current diagnoses against past diagnoses for consistency
- Learn from past QA decisions and patterns

### 8. Integrate Auto-Storage in Workflow (`graph/workflow.py`)

**File:** `graph/workflow.py`Add auto-storage step after QA agent passes:

- After QA agent returns `qa_result: true`, call `store_diagnoses_to_drive()`
- Store diagnoses spreadsheet in Google Drive
- Optionally trigger sync to Qdrant immediately (or rely on periodic sync)

**Integration point:**

- In the QA agent node, after successful QA, store diagnoses before returning

## File Structure

```javascript
rag_ingestion.py
├── extract_diagnosis_metadata()  # NEW: Extract metadata from diagnosis spreadsheets
├── store_diagnoses_to_drive()  # NEW: Auto-store diagnoses after QA
├── sync_diagnoses_from_gdrive_folder()  # NEW: Sync from Google Drive
└── ensure_qdrant_collection()  # UPDATE: Add status index

graph/tools.py
├── find_similar_diagnoses()  # NEW: Tool for QA agent
└── get_available_tools()  # UPDATE: Add tool to QA agent

graph/workflow.py
└── qa_agent_node()  # UPDATE: Add auto-storage after QA passes

agents/qa_agent.yaml
└── prompt  # UPDATE: Add instructions to use find_similar_diagnoses
```



## Data Flow

```javascript
1. Task agent creates diagnoses
   ↓
2. QA agent reviews diagnoses
   ↓
3. QA agent calls find_similar_diagnoses() to retrieve past examples
   ↓
4. QA agent compares current vs past diagnoses
   ↓
5. If QA passes: store_diagnoses_to_drive() creates spreadsheet in Google Drive
   ↓
6. Spreadsheet synced to Qdrant via sync_diagnoses_from_gdrive_folder()
   ↓
7. External editing: User edits Google Drive spreadsheet
   ↓
8. Re-sync updates Qdrant with edited diagnoses
```



## Spreadsheet Format

**Diagnosis Spreadsheet Structure:**| campaign_id | status | diagnosis | issues | recommendations | task_type | dealership_name | diagnosis_date | qa_result ||-------------|--------|-----------|--------|-----------------|-----------|-----------------|----------------|-----------|| Content 1 | passed | Campaign complies... | [] | [] | New Creative | Example Dealership | 2025-01-15T10:30:00Z | true || Content 2 | critical | Missing headline... | ["Missing headline"] | ["Add headline"] | New Creative | Example Dealership | 2025-01-15T10:30:00Z | true |**Notes:**

- `issues` and `recommendations` can be JSON arrays or comma-separated strings
- `diagnosis_date` in ISO format
- `qa_result` is boolean (true/false)

## Testing Considerations

- Test diagnosis metadata extraction from spreadsheets
- Test auto-storage to Google Drive after QA passes
- Test syncing from Google Drive diagnosis spreadsheets
- Test `find_similar_diagnoses` tool with various filters
- Test QA agent integration
- Verify payload indexes work correctly