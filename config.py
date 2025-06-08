
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and Cloud Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") # Optional, often auto-detected

# Paths
DATA_REPO_PATH = os.getenv("DATA_REPO_PATH", "./data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db_data")
CHROMA_COLLECTION_NAME = "civil_eng_papers"

# LLM and Embedding Model Configuration
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_TEMPERATURE = 0.3

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 10 # Let's try 10 for better chance of table retrieval

# Table Extraction
FORCE_VISION_FOR_TABLES = True

# CLI
CLI_HISTORY_LENGTH = 10

# Ensure API key is set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")
if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path '{GOOGLE_APPLICATION_CREDENTIALS}' not found or not set in .env. Please set it and ensure the file exists.")