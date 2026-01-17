from rag_ingestion import sync_from_gdrive_folder, sync_diagnoses_from_gdrive_folder

if __name__ == "__main__":
    # 1. Google Drive folder ID to process
    FOLDER_ID = "1yfd9PUugktO4Hr3y7ngVf6vjAvNY1czu"

    # 2. The Qdrant collection name to use for RAG
    COLLECTION_NAME = "my_rag_collection"

    # 3. Optional: Name of the subfolder containing campaign spreadsheets to process
    CAMPAIGNS_FOLDER_NAME = "Campaigns" 

    print("=" * 80)
    print("RAG Ingestion - Syncing Campaign Briefs and Diagnoses")
    print("=" * 80)
    
    # Sync campaign briefs (campaign examples)
    print("\n[1/2] Syncing campaign briefs...")
    print("-" * 80)
    processed_campaigns = sync_from_gdrive_folder(
        FOLDER_ID, 
        COLLECTION_NAME,
        campaigns_folder_name=CAMPAIGNS_FOLDER_NAME
    )
    print(f"\n✓ Processed {processed_campaigns} campaign brief file(s).")
    
    # Sync diagnoses (diagnosis examples)
    print("\n[2/2] Syncing diagnoses...")
    print("-" * 80)
    processed_diagnoses = sync_diagnoses_from_gdrive_folder(
        FOLDER_ID,
        COLLECTION_NAME
    )
    print(f"\n✓ Processed {processed_diagnoses} diagnosis file(s).")
    
    print("\n" + "=" * 80)
    print(f"RAG Ingestion Complete!")
    print(f"  • Campaign briefs: {processed_campaigns} file(s)")
    print(f"  • Diagnoses: {processed_diagnoses} file(s)")
    print(f"  • Total: {processed_campaigns + processed_diagnoses} file(s)")
    print("=" * 80)
