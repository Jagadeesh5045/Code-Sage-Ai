import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# OpenRouter LLM Configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-2024-08-06")
LLM_REASONING_MODEL = os.getenv("LLM_REASONING_MODEL", "moonshotai/kimi-k2.5")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Parallel System API (Web Search & Extract)
# ---------------------------------------------------------------------------
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join(BASE_DIR, "codesage.db")

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMADB_PATH = os.path.join(BASE_DIR, "chromadb_data")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
FINETUNED_EMBEDDING_MODEL = os.path.join(BASE_DIR, "models", "finetuned-embedding")

# ---------------------------------------------------------------------------
# Cross-Encoder Re-Ranking
# ---------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINETUNED_RERANKER_MODEL = os.path.join(BASE_DIR, "models", "finetuned-reranker")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 8
TOP_K_FINAL = 5
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7
SELF_CORRECTION_MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
MAX_FILE_SIZE_MB = 50
SUPPORTED_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
    ".h", ".cs", ".go", ".rb", ".php", ".rs", ".swift", ".kt",
}
SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".rst", ".pdf"}
IGNORED_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode",
}

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "codesage-ai-secret-key-2024")
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
HOST = "0.0.0.0"
PORT = 5000

# ---------------------------------------------------------------------------
# LangSmith Monitoring (optional)
# ---------------------------------------------------------------------------
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "codesage-ai")

# ---------------------------------------------------------------------------
# Vector Store Backend: "chromadb" or "numpy"
# ---------------------------------------------------------------------------
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chromadb")
