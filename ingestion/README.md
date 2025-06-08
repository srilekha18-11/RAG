# Data Ingestion Module

This module is responsible for processing raw source documents (primarily PDFs), extracting their content, preparing them for retrieval, and storing them in a vector database (ChromaDB).

## Overview

The ingestion pipeline performs the following key steps:

1.  **Document Discovery:** Locates PDF files in a specified source directory.
2.  **Parsing:** Uses the `unstructured` library to parse PDFs, extracting both plain text and structured table content (converted to Markdown). It can leverage OCR (e.g., Google Vision) for scanned documents or image-based content if configured with `GOOGLE_APPLICATION_CREDENTIALS`.
3.  **Chunking:** Splits the extracted content into smaller, semantically relevant chunks suitable for embedding and retrieval.
4.  **Embedding:** Generates vector embeddings for each chunk using a sentence transformer model (or other configured model).
5.  **Storage:** Stores the chunks and their embeddings in a ChromaDB collection, typically persisting to disk.

## Key Files

*   `run_ingestion.py`: The main executable script to trigger the ingestion pipeline. It takes command-line arguments (e.g., `--source_dir`).
*   `document_parser.py`: Contains the `DocumentParser` class responsible for loading and parsing PDF files using `unstructured`. It handles text and table extraction, aiming for `strategy="hi_res"`.
*   `chunker.py`: Implements text splitting logic (e.g., `RecursiveCharacterTextSplitter` from LangChain) to create appropriately sized chunks from the parsed document elements.
*   `vector_store_manager.py`: Contains the `VectorStoreManager` class that handles interactions with ChromaDB, including collection creation/loading, adding documents, and setting up the retriever.

## Workflow

1.  The `run_ingestion.py` script is invoked, typically with a source directory argument (e.g., `data/pdf_documents`).
2.  It iterates through PDF files in the source directory.
3.  For each PDF, `DocumentParser` is used to extract page-level content. `unstructured`'s `hi_res` strategy is preferred, which uses layout models and OCR (potentially Google Vision if credentials are set) as needed. Tables are converted to Markdown.
4.  The extracted content (a list of dictionaries per page, containing text and tables) is then passed to the `Chunker` to be split into smaller text chunks.
5.  `VectorStoreManager` takes these chunks, generates embeddings for them (using a configured embedding model like `all-MiniLM-L6-v2`), and ingests them into the specified ChromaDB collection.
6.  Metadata (e.g., source file name, page number, original table markdown if applicable) is stored alongside each chunk.

## Configuration

*   **Source Directory:** Passed as a command-line argument to `run_ingestion.py` (e.g., `--source_dir data/pdf_documents`).
*   **ChromaDB Collection Name & Persistence:** Configured within `vector_store_manager.py` (e.g., `COLLECTION_NAME = "civil_eng_papers"`, `PERSIST_DIRECTORY = "db"` in the project root).
*   **Embedding Model:** Can be configured in `vector_store_manager.py` or loaded from environment variables.
*   **Chunking Parameters:** Chunk size and overlap can be adjusted in `ingestion/chunker.py`.
*   **`unstructured` Parsing Strategy:** The `DocumentParser` uses `strategy="hi_res"` by default. If `GOOGLE_APPLICATION_CREDENTIALS` are set in the `.env` file, `unstructured` may use Google Vision for OCR.

## How to Run

Navigate to the project's root directory and ensure your virtual environment is activated.

```bash
# Example:
python -m ingestion.run_ingestion --source_dir data/pdf_documents