"""Configuration module for RAG Document Q&A Pipeline with Evaluation Harness."""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "RAG Document Q&A Pipeline with Evaluation Harness"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    FAISS_INDEX_DIR: Path = BASE_DIR / "data" / "faiss_index"
    SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"
    EVAL_OUTPUT_DIR: Path = BASE_DIR / "evaluation" / "results"

    # Embedding Configuration
    # Supports Hugging Face models like "sentence-transformers/all-MiniLM-L6-v2" or "BAAI/bge-small-en-v1.5"
    EMBEDDING_PROVIDER: str = Field(default="huggingface", description="huggingface, openai, or mock")
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face model repository ID"
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", description="cpu or cuda")
    NORMALIZE_EMBEDDINGS: bool = True

    # Ingestion & Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_CHUNK_STRATEGY: str = "recursive"  # recursive, token, character

    # Vector Search Defaults
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.35  # Min cosine similarity score (0 to 1)
    USE_MMR: bool = False  # Maximal Marginal Relevance for diversity
    MMR_DIVERSITY_LAMBDA: float = 0.7

    # Generation & LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_API_BASE: Optional[str] = None  # Custom base url (e.g. for Ollama / vLLM)
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 600

    # Guardrails & Validation
    ENABLE_OUTPUT_VALIDATION: bool = True
    STRICT_GROUNDING: bool = True
    MIN_CITATION_COUNT: int = 1
    HALLUCINATION_THRESHOLD: float = 0.65  # Flag if ungrounded tokens ratio exceeds this

    # Model Pricing for Cost Estimation ($ / 1M tokens)
    MODEL_PRICING: dict = {
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        "default": {"prompt": 0.15, "completion": 0.60}
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(settings.SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(settings.EVAL_OUTPUT_DIR, exist_ok=True)
