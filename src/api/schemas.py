"""Pydantic schemas for FastAPI request and response models."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.generation.prompts import PromptStrategy


class SourceCitationSchema(BaseModel):
    doc_index: int = Field(..., description="Document index referenced in answer (e.g. 1 for [Doc 1])")
    source: str = Field(..., description="Document file name or origin")
    page_number: int = Field(..., description="Page number or section")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    snippet: str = Field(..., description="Extracted context snippet text")
    relevance_score: float = Field(..., description="Cosine similarity score")


class ValidationSchema(BaseModel):
    is_valid: bool = Field(..., description="Whether response passed all validation guardrails")
    hallucination_score: float = Field(..., description="Ratio of unsupported claims (0.0 to 1.0)")
    groundedness_score: float = Field(..., description="Ratio of context-supported claims (0.0 to 1.0)")
    is_refusal: bool = Field(..., description="True if pipeline abstained due to lack of evidence")
    valid_citations: List[int] = Field(default_factory=list, description="Valid citation numbers")
    invalid_citations: List[int] = Field(default_factory=list, description="Phantom or out-of-bounds citations")
    warning_messages: List[str] = Field(default_factory=list, description="Validation warnings")


class PerformanceSchema(BaseModel):
    retrieval_latency_ms: float = Field(..., description="Time taken to retrieve and rank context")
    generation_latency_ms: float = Field(..., description="Time taken for LLM answer generation")
    total_latency_ms: float = Field(..., description="End-to-end query latency in milliseconds")
    prompt_tokens: int = Field(..., description="Prompt tokens consumed")
    completion_tokens: int = Field(..., description="Completion tokens consumed")
    total_tokens: int = Field(..., description="Total tokens consumed")
    cost_usd: float = Field(..., description="Estimated query cost in USD")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question to answer over documents")
    top_k: Optional[int] = Field(4, ge=1, le=20, description="Number of context chunks to retrieve")
    similarity_threshold: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="Minimum similarity threshold")
    prompt_strategy: Optional[PromptStrategy] = Field(PromptStrategy.STRICT_GROUNDED, description="Grounding prompt style")
    use_mmr: Optional[bool] = Field(False, description="Use Maximal Marginal Relevance for diversity")
    custom_instructions: Optional[str] = Field("", description="Optional custom prompt instructions")


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    query: str
    answer: str
    citations: List[SourceCitationSchema]
    validation: ValidationSchema
    performance: PerformanceSchema
    model_name: str
    prompt_strategy: str


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Raw text content to ingest")
    source_name: Optional[str] = Field("api_input", description="Identifier for this document source")
    chunk_size: Optional[int] = Field(500, ge=50, le=4000, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(50, ge=0, le=1000, description="Chunk overlap in characters")
    strategy: Optional[str] = Field("recursive", description="Splitting strategy: recursive, token, character")
    extra_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata tags")


class IngestResponse(BaseModel):
    indexed_chunks: int
    total_documents: int
    sources: List[str]
    message: str


class DocumentChunkItem(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    token_count: int
    char_count: int
    snippet: str


class DocumentListResponse(BaseModel):
    total_chunks: int
    unique_sources: List[str]
    chunks: List[DocumentChunkItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: Dict[str, Any]
    embedding_provider: str
    embedding_model: str
    llm_model: str
