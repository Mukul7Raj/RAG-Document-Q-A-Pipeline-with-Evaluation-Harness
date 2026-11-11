"""FastAPI dependency management for RAG pipeline singleton."""
import logging
from typing import Optional
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Returns the shared RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing RAGPipeline singleton...")
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


def set_pipeline(pipeline: RAGPipeline):
    """Sets custom pipeline instance (useful for testing or reconfiguration)."""
    global _pipeline_instance
    _pipeline_instance = pipeline
