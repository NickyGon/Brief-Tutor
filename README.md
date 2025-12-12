# Brief Tutor - A Creative Campaign Brief Grooming workflow

A LangGraph agent workflow structure designed to evaluate Campaign Brief spreadsheets from DropBox and give
both diagnostics of the promotional campaigns it contains and suggested updated versions for the COX DDC MS Grooming team to use.

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
└── README.md        # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your actual values:
   ```
   OPENAI_API_KEY=your_api_key_here
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_API_KEY=your_qdrant_api_key_here  # Optional
   QDRANT_COLLECTION_NAME=my_rag_collection  # Optional
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

The `retrieve_rag_information` and `retrieve_campaign_briefs` tools allow these agents to search and retrieve relevant documentation and campaign briefs from your Qdrant vector store to guide their work.

### Setting up Qdrant

1. Install and run Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Ensure your `.env` file has the correct Qdrant configuration

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
- `GOOGLE_SERVICE_ACCOUNT_FILE`: Path to your Google service account JSON file (e.g., `credentials/your-service-account.json`)
- `QDRANT_HOST`: Qdrant server host (default: `localhost`)
- `QDRANT_PORT`: Qdrant server port (default: `6333`)
- `QDRANT_API_KEY`: Optional Qdrant API key for authenticated instances
- `QDRANT_COLLECTION_NAME`: Qdrant collection name (default: `my_rag_collection`)
- `CAMPAIGNS_FOLDER_NAME`: Optional name of the inner folder containing campaign spreadsheets (default: `Campaigns`)

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
   - Place your Google service account JSON file in the `credentials/` directory
   - Update `GOOGLE_SERVICE_ACCOUNT_FILE` in `.env` to point to your file
   - **Note:** The `credentials/` folder is gitignored for security

5. **Run Qdrant:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Next Steps

1. Implement DropBox API spreadsheet retrieval workflow
2. Adjust the workflow for Campaign Update cases (should use a previous campaign)
3. Create corrected campaign suggestions for the evaluated campaigns
4. Prepare delivery for the frontend implementation

