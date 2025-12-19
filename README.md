# Brief Tutor - A Creative Campaign Brief Grooming workflow

A LangGraph agent workflow structure designed to evaluate Campaign Brief spreadsheets from DropBox and give both diagnostics of the promotional campaigns it contains and suggested updated versions for the COX DDC MS Grooming team to use.

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

2. Setting up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the needed API keys and configuration:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with the actual values:
   ```
   OPENAI_API_KEY=your_api_key_here
   QDRANT_URL=https://your-cluster-id.qdrant.io  # Qdrant Cloud URL
   QDRANT_API_KEY=your_qdrant_api_key_here  # Required for Qdrant Cloud
   QDRANT_COLLECTION_NAME=my_rag_collection  # Optional, defaults to "my_rag_collection"
   EMBEDDING_MODEL=text-embedding-3-large  # Optional, defaults to "text-embedding-3-large"
   RAG_VECTOR_SIZE=3072  # Optional, auto-set based on embedding model
   GOOGLE_SERVICE_ACCOUNT_FILE=credentials/your-service-account.json
   GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here
   ```

## Usage

The workflow is structured to be extended. Key components:

- **agents/agent.yaml**: Define agent prompts and configurations
- **graph/models.py**: Pydantic models for type safety
- **graph/tools.py**: Custom tools for the agent
- **graph/workflow.py**: Main workflow logic

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

The system uses `text-embedding-3-large` by default for PDF and Campaign Brief spreadsheet metadata information storing, which requires 3072-dimensional vectors over the regular setting for better performance. The collection will be automatically created with the correct dimensions when running the ingestion script.

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
- `QDRANT_URL`: Qdrant Cloud cluster URL (required, format: `https://your-cluster-id.qdrant.io`)
- `QDRANT_API_KEY`: Qdrant Cloud API key (required for authentication)
- `QDRANT_COLLECTION_NAME`: Qdrant collection name (default: `my_rag_collection`)
- `EMBEDDING_MODEL`: Embedding model to use (default: `text-embedding-3-large`)
- `RAG_VECTOR_SIZE`: Vector dimensions (auto-set based on embedding model, 3072 for large, 1536 for small)
- `CAMPAIGNS_FOLDER_NAME`: Optional name of the inner folder containing campaign spreadsheets (default: `Campaigns`)
- `QDRANT_BATCH_SIZE`: Batch size for upsert operations (default: `50`)
- `QDRANT_MAX_RETRIES`: Maximum retries for failed operations (default: `3`)
- `QDRANT_RETRY_DELAY`: Delay between retries in seconds (default: `5`)

**What gets indexed:**
- **Outer folder**: PDFs and DOCX files are indexed as "document" type (best practices, guidelines)
- **Inner folder**: XLSX files are parsed as campaign briefs and indexed as "campaign_brief" type with structured data (task_type, dealership, campaigns, etc.)

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

1. Implement DropBox API spreadsheet retrieval workflow
2. Adjust the workflow for Campaign Update cases (should use a previous campaign)
3. Create corrected campaign suggestions for the evaluated campaigns
4. Prepare delivery for the frontend implementation

