"""Query route for asking questions over indexed documents."""
from fastapi import APIRouter, Depends, HTTPException
from src.api.deps import get_pipeline
from src.api.schemas import (
    PerformanceSchema,
    QueryRequest,
    QueryResponse,
    SourceCitationSchema,
    ValidationSchema,
)
from src.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_pipeline(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
) -> QueryResponse:
    """Answers user question using FAISS retrieval, grounded answer generation, and hallucination validation."""
    try:
        response = pipeline.query(
            question=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            prompt_strategy=request.prompt_strategy,
            use_mmr=request.use_mmr,
            custom_instructions=request.custom_instructions or ""
        )

        citation_schemas = [
            SourceCitationSchema(
                doc_index=c.doc_index,
                source=c.source,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                relevance_score=c.relevance_score
            )
            for c in response.citations
        ]

        validation_schema = ValidationSchema(
            is_valid=response.validation.is_valid,
            hallucination_score=response.validation.hallucination_score,
            groundedness_score=response.validation.groundedness_score,
            is_refusal=response.validation.is_refusal,
            valid_citations=response.validation.valid_citations,
            invalid_citations=response.validation.invalid_citations,
            warning_messages=response.validation.warning_messages
        )

        perf_schema = PerformanceSchema(
            retrieval_latency_ms=response.retrieval_latency_ms,
            generation_latency_ms=response.generation_latency_ms,
            total_latency_ms=response.total_latency_ms,
            prompt_tokens=response.tokens.get("prompt", 0),
            completion_tokens=response.tokens.get("completion", 0),
            total_tokens=response.tokens.get("total", 0),
            cost_usd=response.cost_usd
        )

        return QueryResponse(
            query=response.query,
            answer=response.answer,
            citations=citation_schemas,
            validation=validation_schema,
            performance=perf_schema,
            model_name=response.model_name,
            prompt_strategy=response.prompt_strategy
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline query error: {str(e)}")
