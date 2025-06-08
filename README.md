# RAG Workflows

This project implements a Retrieval Augmented Generation (RAG) system tailored for querying and understanding documents. It leverages Large Language Models (LLMs) to provide insightful answers based on a corpus of ingested documents.

The system is built using LangChain and LangGraphs for orchestrating complex LLM workflows, `unstructured` for robust document parsing (including tables), ChromaDB for vector storage and retrieval, and Google Gemini as the primary LLM :)

## Features

*   **Document Ingestion:** Processes PDF documents, extracts text and tables, chunks content, generates embeddings, and stores them in a vector database.
*   **Intelligent Retrieval:** Retrieves relevant document chunks based on user queries.
*   **Augmented Generation:** Uses an LLM (Google Gemini) to synthesize answers based on retrieved context.
*   **CLI:** Allows users to ask questions and receive answers.
*   **Stateful Workflows:** Utilizes LangGraphs to manage conversational state and complex multi-step reasoning.

## Tech Stack

*   **Python 3.10+**
*   **LangChain & LangGraphs:** For building and orchestrating LLM applications.
*   **Google Gemini API:** As the core Large Language Model.
*   **ChromaDB:** For vector storage and similarity search.
*   **`unstructured`:** For advanced PDF parsing, including text and table extraction.
    *   Leverages Google Cloud Vision API via `unstructured` for OCR on scanned PDFs/images when configured.
*   **Sentence Transformers (or other embedding model):** For generating text embeddings.
  
## Prerequisites

1.  **Python & Pip:** Ensure Python (3.10 or newer) and pip are installed.
2.  **Git:** For cloning the repository.
3.  **Google Cloud Account & Project:**
    *   You'll need a Google Cloud Project with the **Generative Language API (Gemini API)** enabled.
    *   **Billing Enabled (Highly Recommended):** The Gemini API free tier has very strict rate limits. For any serious development or usage, you'll need to [enable billing on your Google Cloud Project](https://cloud.google.com/billing/docs/how-to/modify-project).
    *   **API Key for Gemini:** [Create an API key](https://ai.google.dev/gemini-api/docs/api-key) for the Gemini API.
    *   **(Optional but Recommended for `unstructured` OCR) Google Application Credentials:** If you want `unstructured` to use Google Vision API for OCR on scanned PDFs:
        *   Enable the Cloud Vision API in your GCP project.
        *   Create a service account with appropriate permissions (e.g., "Cloud Vision AI User").
        *   Download the service account key JSON file.
4.  **System Dependencies for `unstructured` and `pdf2image`:**
    *   `unstructured` (especially with `strategy="hi_res"`) and its underlying PDF processing (which often uses `pdf2image`) may require system libraries:
    *   **Poppler:** For PDF rendering and image conversion.
        *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y poppler-utils`
    *   **Tesseract OCR Engine (Fallback OCR):** Even if using Google Vision, having Tesseract can be a useful fallback or for certain `unstructured` strategies.
        *   Debian/Ubuntu: `sudo apt-get install -y tesseract-ocr libtesseract-dev`
    *   Check the [`unstructured` installation guide](https://unstructured-io.github.io/unstructured/installing.html) and [`pdf2image` documentation](https://pypi.org/project/pdf2image/) for the latest system dependency details.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/srilekha18-11/RAG.git
    cd RAG
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    Ensure you have a `requirements.txt` file in the root of your project (see example below or generate from your environment). Then run:
    ```bash
    pip install -r requirements.txt
    ```
    other dependencies if any


4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your credentials:
    ```env
    GOOGLE_API_KEY="your_gemini_api_key_here"
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    DATA_REPO_PATH="./data"
    VECTOR_STORE_PATH="./chroma_db_data"
    ```

## Running the Application

### 1. Data Ingestion

```bash
python -m ingestion.run_ingestion --source_dir "path/to/your/pdf_documents"
# or
python -m ingestion.run_ingestion 
```


### 2. Running the RAG Pipeline / Application

```bash
python -m rag_pipeline.app
```

## Project Structure

```
.
├── config.py
├── data
│   └── ASME
│       └── D4191.pdf
├── .gitignore
├── ingestion
│   ├── chunker.py
│   ├── document_parser.py
│   ├── __init__.py
│   ├── README.md
│   ├── run_ingestion.py
│   └── vector_store_manager.py
├── main.py
├── rag_pipeline
│   ├── graph_builder.py
│   ├── graph_state.py
│   ├── __init__.py
│   ├── nodes.py
│   ├── prompts.py
│   └── README.md
├── README.md
└── requirements.txt
```

## Configuration

- API keys and paths: `.env`
- Vector store config: `vector_store_manager.py`
- Chunking: `chunker.py`
- RAG pipeline: `rag_pipeline/` modules

## Troubleshooting

- **429 Rate Limit (Gemini):** Enable billing or reduce API usage.
- **PDF Parsing Issues:** Ensure Poppler and Tesseract installed.
- **ChromaDB Issues:** Check installation and file access.
- **Module Errors:** Activate venv and install all deps.
