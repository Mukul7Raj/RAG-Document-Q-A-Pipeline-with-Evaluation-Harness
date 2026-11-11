"""Ingestion routes for uploading documents and indexing raw text."""
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form
from src.api.deps import get_pipeline
from src.api.schemas import IngestResponse, IngestTextRequest
from src.pipeline.rag_pipeline import RAGPipeline
from config.settings import settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/text", response_model=IngestResponse)
async def ingest_text_endpoint(
    request: IngestTextRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
) -> IngestResponse:
    """Chunks, embeds, and indexes raw text directly."""
    try:
        chunks_indexed = pipeline.ingest_text(
            text=request.text,
            source_name=request.source_name or "api_input",
            extra_metadata=request.extra_metadata or {},
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            strategy=request.strategy
        )

        return IngestResponse(
            indexed_chunks=chunks_indexed,
            total_documents=1,
            sources=[request.source_name or "api_input"],
            message=f"Successfully indexed {chunks_indexed} chunks."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")


@router.post("/files", response_model=IngestResponse)
async def ingest_files_endpoint(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Form(500),
    chunk_overlap: Optional[int] = Form(50),
    strategy: Optional[str] = Form("recursive"),
    pipeline: RAGPipeline = Depends(get_pipeline)
) -> IngestResponse:
    """Uploads and indexes multiple documents (PDF, TXT, MD, DOCX, JSON)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    temp_paths: List[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))

    try:
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(file_path)

        result = pipeline.ingest_files(
            file_paths=temp_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            persist=True
        )

        return IngestResponse(
            indexed_chunks=int(result["indexed_chunks"]),
            total_documents=int(result["total_documents"]),
            sources=list(result["sources"]),
            message=f"Successfully ingested {len(files)} files ({result['indexed_chunks']} chunks)."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

