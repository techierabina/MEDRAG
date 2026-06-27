import os
from dotenv import load_dotenv

load_dotenv()

# Groq LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model (runs locally, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB settings
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "medrag_docs"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval settings
TOP_K = 5

# Data path
DATA_PATH = "./data/sample_docs"