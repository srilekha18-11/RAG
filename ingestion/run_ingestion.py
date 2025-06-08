import os
import glob
from pathlib import Path
import logging

from config import DATA_REPO_PATH
from ingestion.document_parser import DocumentParser
from ingestion.chunker import Chunker
from ingestion.vector_store_manager import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def process_and_ingest_documents():
    if not os.path.exists(DATA_REPO_PATH):
        logger.error(f"Data repository path does not exist: {DATA_REPO_PATH}")
        return

    parser = DocumentParser()
    chunker = Chunker()
    vsm = VectorStoreManager()

    processed_files_count = 0
    total_chunks_ingested = 0

    # Iterate through subfolders (conference names)
    for conference_folder in Path(DATA_REPO_PATH).iterdir():
        if conference_folder.is_dir():
            conference_name = conference_folder.name
            logger.info(f"Processing conference: {conference_name}")
            
            # Find PDF files in the conference folder
            # Using glob for simplicity, can be Path.glob as well
            pdf_files = glob.glob(os.path.join(conference_folder, "*.pdf"))
            
            for pdf_file_path in pdf_files:
                logger.info(f"--- Processing document: {pdf_file_path} ---")
                try:
                    # 1. Parse Document
                    parsed_pages = parser.parse_document(pdf_file_path)
                    if not parsed_pages:
                        logger.warning(f"No content parsed from {pdf_file_path}. Skipping.")
                        continue

                    # 2. Chunk Parsed Content
                    chunks_with_metadata = chunker.chunk_parsed_document(
                        file_path=pdf_file_path,
                        conference_name=conference_name,
                        parsed_pages=parsed_pages
                    )
                    if not chunks_with_metadata:
                        logger.warning(f"No chunks created for {pdf_file_path}. Skipping.")
                        continue
                    
                    # 3. Add Chunks to Vector Store
                    vsm.add_chunks_to_store(chunks_with_metadata)
                    total_chunks_ingested += len(chunks_with_metadata)
                    processed_files_count += 1
                    logger.info(f"Successfully processed and ingested {pdf_file_path}")

                except Exception as e:
                    logger.error(f"Failed to process document {pdf_file_path}: {e}", exc_info=True)
    
    logger.info("--- Ingestion Process Complete ---")
    logger.info(f"Total files processed: {processed_files_count}")
    logger.info(f"Total chunks ingested: {total_chunks_ingested}")
    if vsm and vsm.collection:
         logger.info(f"Total items in ChromaDB collection '{vsm.collection.name}': {vsm.collection.count()}")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. .env file is set up with GOOGLE_API_KEY and GOOGLE_APPLICATION_CREDENTIALS.
    # 2. google_cloud_creds.json file exists.
    # 3. Your data repository is pointed to by DATA_REPO_PATH and has PDFs.
    #    Example: ./data/MyConference/paper1.pdf
    # 4. Install all requirements from requirements.txt.
    
    # You might want to clear the vector store before a full re-ingestion
    # if os.path.exists(VECTOR_STORE_PATH):
    #     import shutil
    #     logger.warning(f"Removing existing vector store at {VECTOR_STORE_PATH} before re-ingestion.")
    #     shutil.rmtree(VECTOR_STORE_PATH)

    process_and_ingest_documents()