from rag_ingestion import sync_from_gdrive_folder

if __name__ == "__main__":
    # 1. Google Drive folder ID to process
    FOLDER_ID = "1yfd9PUugktO4Hr3y7ngVf6vjAvNY1czu"

    # 2. The Qdrant collection name to use for RAG
    COLLECTION_NAME = "my_rag_collection"

    # 3. Optional: Name of the subfolder containing campaign spreadsheets to process
    CAMPAIGNS_FOLDER_NAME = "Campaigns" 

    processed = sync_from_gdrive_folder(
        FOLDER_ID, 
        COLLECTION_NAME,
        campaigns_folder_name=CAMPAIGNS_FOLDER_NAME
    )
    print(f"Processed {processed} file(s).")
