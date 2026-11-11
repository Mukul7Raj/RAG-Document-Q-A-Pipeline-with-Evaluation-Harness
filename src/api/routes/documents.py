"""Document and index management routes."""
from fastapi import APIRouter, Depends, HTTPException
from src.api.deps import get_pipeline
from src.api.schemas import DocumentChunkItem, DocumentListResponse
from src.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    pipeline: RAGPipeline = Depends(get_pipeline)
) -> DocumentListResponse:
    """Returns all currently indexed document chunks and unique document sources."""
    chunks = pipeline.vector_store.chunks
    sources = list(set(c.metadata.get("source", "unknown") for c in chunks))

    chunk_items = [
        DocumentChunkItem(
            chunk_id=c.chunk_id,
            source=str(c.metadata.get("source", "unknown")),
            page_number=int(c.metadata.get("page_number", 1)),
            token_count=c.token_count,
            char_count=c.char_count,
            snippet=c.text[:180] + "..." if len(c.text) > 180 else c.text
        )
        for c in chunks
    ]

    return DocumentListResponse(
        total_chunks=len(chunks),
        unique_sources=sources,
        chunks=chunk_items
    )


@router.delete("/clear")
async def clear_documents(
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Clears all indexed vectors and documents from FAISS store."""
    pipeline.vector_store.clear()
    pipeline.save_index()
    return {"message": "FAISS index cleared successfully."}

