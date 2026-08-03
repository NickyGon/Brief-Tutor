# Brief Tutor - A Creative Campaign Brief Grooming workflow

A LangGraph workflow that evaluates Campaign Brief spreadsheets, generates grounded diagnoses, and produces output documents for the COX DDC MS Grooming team.

## Project Structure

```
.
├── agents/           # Agent configuration files (.yaml)
│   └── agent.yaml    # Basic agent prompt configuration
├── graph/            # Graph workflow files
│   ├── models.py     # Pydantic classes for state and data models
│   ├── tools.py      # Agent tools definitions
│   └── workflow.py   # Main LangGraph workflow
├── requirements.txt  # Python dependencies
├── credentials/      # Credentials folder (has to be created locally)
│   ├── models.py     # Google service account JSON (has to allow Google Drive API)
└── README.md
```


## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the needed API keys and configuration:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with the actual values:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_CHAT_MODEL=gpt-5-nano

   # LLM provider routing
   LLM_PROVIDER=openai
   FALLBACK_LLM_PROVIDER=
   LLM_MODEL=
   FALLBACK_LLM_MODEL=

   # Optional override for embeddings provider (defaults to LLM_PROVIDER)
   EMBEDDING_PROVIDER=

   QDRANT_URL=https://your-cluster-id.qdrant.io  # Qdrant Cloud URL
   QDRANT_API_KEY=your_qdrant_api_key_here  # Required for Qdrant Cloud
   QDRANT_COLLECTION_NAME=my_rag_collection  # Optional, defaults to "my_rag_collection"
   EMBEDDING_MODEL=text-embedding-3-large  # Optional, defaults to "text-embedding-3-large"
   RAG_VECTOR_SIZE=3072  # Optional, auto-set based on embedding model
   GOOGLE_SERVICE_ACCOUNT_FILE=credentials/your-service-account.json
   GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here
   EVAL_MIN_GROUNDEDNESS=0.8
   EVAL_MIN_FIELD_COMPLETENESS=1.0
   ALERT_MAX_NODE_LATENCY_MS=45000
   WORKFLOW_METRICS_FILE=workflow_metrics.jsonl
   ```

3. Configure Google credentials:
   - Google Drive access uses the service account file in `GOOGLE_SERVICE_ACCOUNT_FILE`.
   - If your environment needs ADC-based Google API auth, set:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=credentials/your-service-account.json
   ```
   - Ensure that service account has Drive access to the relevant folders.

## Usage

The workflow is structured to be extended. Key components:

- **agents/agent.yaml**: Define agent prompts and configurations
- **graph/models.py**: Pydantic models for type safety
- **graph/tools.py**: Custom tools for the agent
- **graph/workflow.py**: Main workflow logic with provider fallback and per-node metrics

### Provider Routing (OpenAI)

- `LLM_PROVIDER` should be `openai`.
- `FALLBACK_LLM_PROVIDER` is optional; if configured, the workflow retries node calls on fallback provider when the primary fails.
- `LLM_MODEL` and `FALLBACK_LLM_MODEL` can override defaults per provider.
- `EMBEDDING_PROVIDER` controls embeddings independently from chat provider (defaults to `LLM_PROVIDER` when unset; keep `openai`).

### Post-Brief Route Toggle

After `brief_creator` parses the spreadsheet, the workflow now supports two mutually exclusive routes:

- `BRIEF_POST_PARSE_ROUTE=1`: original analyzer flow (`router` -> task-type agent -> diagnosis formatter -> document creator).
- `BRIEF_POST_PARSE_ROUTE=0`: family-similarity-only flow (same-family local spreadsheet matching -> family similarity agent narrative enrichment -> similarity JSON/TXT outputs), then workflow ends.

Optional thresholds for the similarity route:
- `FAMILY_SIM_STRONG_THRESHOLD` (default `0.80`)
- `FAMILY_SIM_REVIEW_THRESHOLD` (default `0.50`)

Similarity score strategy (dual path + absolute pairing):
- Always `10%` dealership / OEM / group proximity
- Reference path (same account/group + matching A-/D- IDs in Style Direction/related fields):
  high weight on reference strength (cue words like "Copy from"/"Refer to" make it stronger, but are not required when both sides share the same ID)
  light content blend
- Content path (no copy/refer ID signal):
  StyleDirection/assets + campaign wording/structure dominate the remaining weight
- Output is campaign pairing (1:1), not just ranked similarity:
  - `absolute_pairs`: reference lock and/or very strong style-assets agreement
  - `likely_pairs`: strong enough to pair
  - `review_pairs`: human confirmation
  - `unpaired_targets`: no acceptable pair
- File ranking uses strongest assigned pair score so campaign count does not dilute similarity

Narrative enrichment uses three parallel specialists (reference, style/assets, wording) feeding a synthesizer agent.

### Hierarchical Similarity Discovery

The similarity-only branch uses hierarchical candidate discovery with filename parsing:

- Supported filename pattern: `YYYY-MM-[accountID]-[A-|D-]<id>.xlsx`
- Confirmed examples:
  - `2025-11-rogerbeasleyvolvovcna-A-25008537.xlsx`
  - `2026-06-tonydivinousedcarsntrucks-D-94095.xlsx`
- Search order:
  1. same accountID folder
  2. if no qualifying match, widen to:
     - sibling account folders in the same group folder (when grouped), or
     - sibling account/group folders under `Campaigns` (when ungrouped)

Optional discovery controls:
- `FAMILY_SIM_WIDEN_IF_NO_QUALIFYING` (default `true`)
- `FAMILY_SIM_QUALIFYING_THRESHOLD` (default `0.80`)
- `FAMILY_SIM_USE_OEM_FILTER` (default `true`)
- `FAMILY_SIM_OEM_FALLBACK_IF_EMPTY` (default `true`)

The same filename parser is reused by campaign-update previous-brief resolution logic to keep `A-` and `D-` handling consistent.

### Dealership Metadata (Group/Account/OEM)

For wider-scope matching accuracy, this project supports Supabase-seeded dealership metadata:

- `dealership_groups`
- `dealership_accounts`
- `dealership_account_oems`

Behavior in widened searches:
- hard OEM/OEM-family compatibility filter first
- fallback to broader candidates when filter returns empty (configurable)

Special handling:
- multi-OEM accounts use `dealership_account_oems`
- accounts handling broadly set `handles_all_oems=true` (equivalent to `All`)
- non-family accounts use `oem_family='NA'`

### Supabase Connection Scaffold

This project includes a Supabase connection folder at `graph/supabase/`:

- `config.py`: environment-backed settings and readiness checks.
- `client.py`: lazy/cached Supabase client factory (`anon` and `service_role` modes).
- `repository.py`: workflow repository helpers (`upsert_brief`, `upsert_campaigns`, run and similarity persistence).
- `__init__.py`: simple import surface for workflow modules.

Environment variables:
- `SUPABASE_ENABLED` (`true`/`false`)
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_SCHEMA` (default `public`)

Usage example:

```python
from graph.supabase import get_supabase_client, is_supabase_configured

if is_supabase_configured():
    client = get_supabase_client()
    # Example: read from a table
    response = client.table("briefs").select("*").limit(5).execute()
```

Repository example:

```python
from graph.supabase import SupabaseWorkflowRepository

repo = SupabaseWorkflowRepository(use_service_role=True)
brief_row = repo.upsert_brief(campaign_brief)
repo.upsert_campaigns(brief_fk=brief_row["id"], campaign_brief=campaign_brief)
```

## RAG Information Retrieval

The workflow includes a RAG (Retrieval-Augmented Generation) tool that connects to a Qdrant vector database. This tool is available to:
- `theme_agent`
- `new_creative_agent`
- `campaign_update_agent`
- `qa_agent`

The `retrieve_rag_information` and `retrieve_campaign_briefs` tools allow these agents to search and retrieve relevant documentation and campaign briefs from the Qdrant vector Database to guide their work.

### Setting up Qdrant Cloud

This project uses **Qdrant Cloud** (not a local instance). To set up:

1. **Create a Qdrant Cloud account:**
   - Go to [cloud.qdrant.io](https://cloud.qdrant.io)
   - Sign up and create a cluster

2. **Get your cluster credentials:**
   - Copy your cluster URL (format: `https://your-cluster-id.qdrant.io`)
   - Copy your API key from the cluster settings

3. **Configure environment variables:**
   - Set `QDRANT_URL` to your cluster URL
   - Set `QDRANT_API_KEY` to your API key
   - Optionally set `QDRANT_COLLECTION_NAME` (defaults to `my_rag_collection`)

The collection vector size is enforced based on the embedding model in use:
- 3072 dimensions for OpenAI large embeddings
- 1536 dimensions for OpenAI small embeddings

If a collection has mismatched dimensions, ingestion recreates it with the expected size.

### Syncing Documents from Google Drive

The `rag_ingestion.py` script can sync documents from a Google Drive folder to Qdrant. It supports a nested folder structure:

**Folder Structure:**
```
Main Drive Folder/       
├── guidelines.docx # Documentation files (PDFs, DOCX)
└── Campaigns/                  # Inner folder (optional)
    ├── Campaign Update/        # Task type folder
    │   ├── Actual/            # Subfolder for current campaigns
    │   │   └── campaign1.xlsx
    │   └── Previous/          # Subfolder for previous campaigns
    │       └── campaign2.xlsx
    ├── New Creative/          # Task type folder
    │   └── campaign3.xlsx
    └── Theme/                 # Task type folder
        └── campaign4.xlsx
```

**Important:** The task type folder names ("Campaign Update", "New Creative", "Theme") and subfolder names ("Actual", "Previous") are fixed and must match exactly.

**Usage:**
```python
from rag_ingestion import sync_from_gdrive_folder

# Sync from Google Drive folder
processed = sync_from_gdrive_folder(
    folder_id="YOUR_DRIVE_FOLDER_ID",
    collection_name="my_rag_collection",
    campaigns_folder_name="YOUR_INNER_CAMPAIGNS_FOLDER_NAME"
)
```

**Environment Variables:**
- `GOOGLE_SERVICE_ACCOUNT_FILE`: Path to custom Google service account JSON file (e.g., `credentials/your-service-account.json`)
- `GOOGLE_APPLICATION_CREDENTIALS`: Optional ADC path for Google API auth (typically same service account JSON)
- `LLM_PROVIDER`: Model provider for workflow agents (`openai`)
- `FALLBACK_LLM_PROVIDER`: Optional fallback provider if primary provider fails
- `LLM_MODEL`: Optional override for primary provider chat model
- `FALLBACK_LLM_MODEL`: Optional override for fallback provider chat model
- `EMBEDDING_PROVIDER`: Optional override for embedding provider (`openai`)
- `QDRANT_URL`: Qdrant Cloud cluster URL (required, format: `https://your-cluster-id.qdrant.io`)
- `QDRANT_API_KEY`: Qdrant Cloud API key (required for authentication)
- `QDRANT_COLLECTION_NAME`: Qdrant collection name (default: `my_rag_collection`)
- `EMBEDDING_MODEL`: Embedding model to use (default: `text-embedding-3-large`)
- `RAG_VECTOR_SIZE`: Vector dimensions (auto-set based on embedding model, 3072 for large, 1536 for small)
- `CAMPAIGNS_FOLDER_NAME`: Optional name of the inner folder containing campaign spreadsheets (default: `Campaigns`)
- `QDRANT_BATCH_SIZE`: Batch size for upsert operations (default: `50`)
- `QDRANT_MAX_RETRIES`: Maximum retries for failed operations (default: `3`)
- `QDRANT_RETRY_DELAY`: Delay between retries in seconds (default: `5`)
- `EVAL_MIN_GROUNDEDNESS`: Minimum fraction of diagnoses with grounding evidence (default: `0.8`)
- `EVAL_MIN_FIELD_COMPLETENESS`: Minimum fraction of diagnoses with required fields (default: `1.0`)
- `ALERT_MAX_NODE_LATENCY_MS`: Per-node latency threshold for alert logging (default: `45000`)
- `WORKFLOW_METRICS_FILE`: JSONL output file for run metrics and alerts (default: `workflow_metrics.jsonl`)

**What gets indexed:**
- **Outer folder**: PDFs and DOCX files are indexed as "document" type (best practices, guidelines)
- **Inner folder**: XLSX files are parsed and indexed as campaign metadata (`campaign_metadata`) for similarity search

## Development

### Setting up for GitHub

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd brief-tutor
   ```

2. **Set up environment:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Service Account:**
   - Place a Google service account JSON file in the `credentials/` directory
   - Update `GOOGLE_SERVICE_ACCOUNT_FILE` in `.env` to point to the JSON file

5. **Set up Qdrant Cloud:**
   - Create a Qdrant Cloud account at [cloud.qdrant.io](https://cloud.qdrant.io)
   - Create a cluster and get your cluster URL and API key
   - Set `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` file

## Next Steps

1. Expand campaign retrieval sources if needed
2. Continue improving Campaign Update matching/evaluation quality
3. Improve document output quality and consistency checks
4. Prepare delivery for frontend integration

