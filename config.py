# config.py - All settings in one place

# Your PDF folder paths - change these to your actual paths
PDF_FOLDERS = [
    r"G:\My Drive\Major_Project\u1",
    r"G:\My Drive\Major_Project\u2", 
    r"G:\My Drive\Major_Project\u3",
    r"G:\My Drive\Major_Project\u4"
]

# Where to save the database
FAISS_PATH = r"C:\RealEstateRAG\faiss_database"

# Your API key
OPENROUTER_API_KEY = "your open_router_api key"

# Model settings
MODEL_NAME = "openai/gpt-3.5-turbo"
MODEL_BASE_URL = "https://openrouter.ai/api/v1"

# Embedding model (free, runs locally)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 8
FETCH_K = 40



