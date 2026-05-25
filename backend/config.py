import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Cross-encoder reranker
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# ChromaDB
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "pgx_knowledge")

# Retrieval
RAG_TOP_K_INITIAL = 20
RAG_TOP_K_RERANK = 5

# Confidence thresholds (cosine distance, 0=perfect match)
# Conservative initial values. Calibrate via: python eval.py
#   - Too many bad results marked 'high' → lower thresholds
#   - Too many good results marked 'low' → raise thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.25
CONFIDENCE_MEDIUM_THRESHOLD = 0.50

# HyDE
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() == "true"

# Query expansion
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
QUERY_EXPANSION_NUM = 3

# Self-RAG
SELF_RAG_ENABLED = os.getenv("SELF_RAG_ENABLED", "true").lower() == "true"

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
