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
   ENABLE_VERTEXAI=true
   LLM_PROVIDER=openai
   FALLBACK_LLM_PROVIDER=
   LLM_MODEL=
   FALLBACK_LLM_MODEL=

   # Vertex AI (required when LLM_PROVIDER=vertexai)
   VERTEX_PROJECT_ID=your-gcp-project-id
   VERTEX_LOCATION=us-central1
   VERTEX_MODEL=gemini-2.0-flash-001
   VERTEX_EMBEDDING_MODEL=text-embedding-005

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
   - Vertex AI calls require Application Default Credentials (ADC). Set:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=credentials/your-service-account.json
   ```
   - Ensure that service account has IAM needed for Vertex AI (for example, `roles/aiplatform.user` for predict access) and Drive access to the relevant folders.

## Usage

The workflow is structured to be extended. Key components:

- **agents/agent.yaml**: Define agent prompts and configurations
- **graph/models.py**: Pydantic models for type safety
- **graph/tools.py**: Custom tools for the agent
- **graph/workflow.py**: Main workflow logic with provider fallback and per-node metrics

### Provider Routing (OpenAI + Vertex)

- `LLM_PROVIDER` selects the primary chat provider (`openai` or `vertexai`).
- `ENABLE_VERTEXAI` is a hard safety switch; when `false`, all `vertexai` selections are forced to OpenAI.
- `FALLBACK_LLM_PROVIDER` is optional; if configured, the workflow retries node calls on fallback provider when the primary fails.
- `LLM_MODEL` and `FALLBACK_LLM_MODEL` can override defaults per provider.
- `EMBEDDING_PROVIDER` controls embeddings independently from chat provider (defaults to `LLM_PROVIDER` when unset).

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
- 1536 dimensions for small/Vertex-style embedding dimensions

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
- `GOOGLE_APPLICATION_CREDENTIALS`: ADC path for Vertex AI auth (typically same service account JSON)
- `LLM_PROVIDER`: Model provider for workflow agents (`openai` or `vertexai`)
- `ENABLE_VERTEXAI`: Master safety switch for Vertex AI (`true`/`false`). If `false`, provider selection and fallback ignore `vertexai`.
- `FALLBACK_LLM_PROVIDER`: Optional fallback provider if primary provider fails
- `LLM_MODEL`: Optional override for primary provider chat model
- `FALLBACK_LLM_MODEL`: Optional override for fallback provider chat model
- `VERTEX_PROJECT_ID`: GCP project for Vertex AI
- `VERTEX_LOCATION`: Vertex region (default: `us-central1`)
- `VERTEX_MODEL`: Vertex chat model (default: `gemini-2.0-flash-001`)
- `VERTEX_EMBEDDING_MODEL`: Vertex embedding model (default: `text-embedding-005`)
- `EMBEDDING_PROVIDER`: Optional override for embedding provider (`openai` or `vertexai`)
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

