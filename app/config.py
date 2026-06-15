import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Paths
CHROMA_DB_DIR = os.path.join(BASE_DIR, ".chromadb")
TEMP_DIR = os.path.join(BASE_DIR, "temp_pdf")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# LLM Configurations
OPENAI_MODEL_NAME = "gpt-4o-mini"
LOCAL_MODEL_NAME = "openbmb/MiniCPM5-1B-SFT"

# Embedding Configurations
LOCAL_EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# RAG / Text Splitting Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
RETRIEVER_K = 3
