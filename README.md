# Civil Engineering RAG System

This project implements a Retrieval Augmented Generation (RAG) system tailored for querying and understanding documents related to civil engineering. It leverages Large Language Models (LLMs) to provide insightful answers based on a corpus of ingested documents.

The system is built using LangChain and LangGraphs for orchestrating complex LLM workflows, `unstructured` for robust document parsing (including tables), ChromaDB for vector storage and retrieval, and Google Gemini as the primary LLM.

## Features

*   **Document Ingestion:** Processes PDF documents, extracts text and tables, chunks content, generates embeddings, and stores them in a vector database.
*   **Intelligent Retrieval:** Retrieves relevant document chunks based on user queries.
*   **Augmented Generation:** Uses an LLM (Google Gemini) to synthesize answers based on retrieved context.
*   **Conversational Interface:** (Assuming you'll build one, e.g., with Streamlit or a CLI) Allows users to ask questions and receive answers.
*   **Stateful Workflows:** Utilizes LangGraphs to manage conversational state and complex multi-step reasoning.

## Tech Stack

*   **Python 3.10+**
*   **LangChain & LangGraphs:** For building and orchestrating LLM applications.
*   **Google Gemini API:** As the core Large Language Model.
*   **ChromaDB:** For vector storage and similarity search.
*   **`unstructured`:** For advanced PDF parsing, including text and table extraction.
    *   Leverages Google Cloud Vision API via `unstructured` for OCR on scanned PDFs/images when configured.
*   **Sentence Transformers (or other embedding model):** For generating text embeddings.
*   **(Optional) Streamlit/FastAPI:** For building a user interface or API.

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
        *   macOS: `brew install poppler`
        *   Windows: Download Poppler binaries, add to PATH. [See `pdf2image` docs for Windows setup](https://pypi.org/project/pdf2image/).
    *   **Tesseract OCR Engine (Fallback OCR):** Even if using Google Vision, having Tesseract can be a useful fallback or for certain `unstructured` strategies.
        *   Debian/Ubuntu: `sudo apt-get install -y tesseract-ocr libtesseract-dev`
        *   macOS: `brew install tesseract tesseract-lang`
        *   Windows: Install Tesseract using an MSI installer, add to PATH. Ensure `tesseract` command is available.
    *   Check the [`unstructured` installation guide](https://unstructured-io.github.io/unstructured/installing.html) and [`pdf2image` documentation](https://pypi.org/project/pdf2image/) for the latest system dependency details.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd civil_eng_rag # Or your project's root directory name
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
    *Example `requirements.txt` content:*
    ```txt
    # Core LangChain and LLM components
    langchain
    langchain-core
    langchain-community
    langchain-google-genai
    langgraph

    # Vector Store
    chromadb
    sentence-transformers

    # Document Parsing
    unstructured[local-inference,gcp]
    pypdf
    pandas
    lxml
    html5lib
    beautifulsoup4

    # Environment variable management
    python-dotenv

    # (Optional: streamlit, fastapi, uvicorn)
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your credentials:
    ```env
    GOOGLE_API_KEY="your_gemini_api_key_here"
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    ```

## Running the Application

### 1. Data Ingestion

```bash
python -m ingestion.run_ingestion --source_dir "path/to/your/pdf_documents"
```

### 2. Running the RAG Pipeline / Application

```bash
python -m rag_pipeline.app
# or
streamlit run rag_pipeline/app.py
```

## Project Structure

```
civil_eng_rag/
├── data/
│   └── pdf_documents/
├── db/
├── ingestion/
│   ├── __init__.py
│   ├── document_parser.py
│   ├── chunker.py
│   ├── vector_store_manager.py
│   └── run_ingestion.py
├── rag_pipeline/
│   ├── __init__.py
│   ├── nodes.py
│   ├── graph.py
│   ├── app.py
│   └── (config.py)
├── .env
├── requirements.txt
└── README.md
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

## Contributing

(Add guidelines if you plan for contributions.)

## License

(Specify a license, e.g., MIT, Apache 2.0.)