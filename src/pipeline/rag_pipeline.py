"""Unified RAG pipeline combining ingestion, embedding, FAISS indexing, grounded generation, and validation."""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from config.settings import settings
from src.ingestion.loader import DocumentLoader, LoadedDocument
from src.ingestion.chunker import DocumentChunker, DocumentChunk
from src.vectorstore.embeddings import BaseEmbeddingModel, get_embedding_model
from src.vectorstore.faiss_store import FAISSVectorStore, SearchResult
from src.generation.prompts import PromptManager, PromptStrategy
from src.generation.generator import AnswerGenerator, GenerationResult
from src.generation.validator import OutputValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """Represents a validated citation mapped to an original document source."""
    doc_index: int
    source: str
    page_number: int
    chunk_id: str
    snippet: str
    relevance_score: float

    def to_dict(self) -> dict:
        return {
            "doc_index": self.doc_index,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
            "relevance_score": round(self.relevance_score, 4),
        }


@dataclass
class RAGQueryResponse:
    """Complete response returned by the RAG pipeline."""
    query: str
    answer: str
    citations: List[SourceCitation]
    validation: ValidationResult
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    tokens: Dict[str, int]
    cost_usd: float
    model_name: str
    prompt_strategy: str
    retrieved_count: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "validation": self.validation.to_dict(),
            "performance": {
                "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
                "generation_latency_ms": round(self.generation_latency_ms, 2),
                "total_latency_ms": round(self.total_latency_ms, 2),
                "prompt_tokens": self.tokens.get("prompt", 0),
                "completion_tokens": self.tokens.get("completion", 0),
                "total_tokens": self.tokens.get("total", 0),
                "cost_usd": round(self.cost_usd, 6),
            },
            "meta": {
                "model_name": self.model_name,
                "prompt_strategy": self.prompt_strategy,
                "retrieved_count": self.retrieved_count,
            }
        }


class RAGPipeline:
    """Production-grade RAG pipeline orchestrating ingestion, retrieval, generation, and validation."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        generator: Optional[AnswerGenerator] = None,
        validator: Optional[OutputValidator] = None,
        auto_load_index: bool = True
    ):
        self.loader = DocumentLoader()

        # Initialize embedding model
        self.embedding_model = embedding_model or get_embedding_model(
            provider=settings.EMBEDDING_PROVIDER,
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            api_key=settings.OPENAI_API_KEY
        )

        # Initialize or load vector store
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            if auto_load_index and (settings.FAISS_INDEX_DIR / "index.faiss").exists():
                try:
                    self.vector_store = FAISSVectorStore.load(settings.FAISS_INDEX_DIR)
                    logger.info("Loaded persisted FAISS index successfully.")
                except Exception as e:
                    logger.warning("Failed to load existing FAISS index (%s). Initializing fresh index.", e)
                    self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)
            else:
                self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)

        # Initialize generator & validator
        self.generator = generator or AnswerGenerator()
        self.validator = validator or OutputValidator(
            hallucination_threshold=settings.HALLUCINATION_THRESHOLD,
            enforce_citations=settings.STRICT_GROUNDING
        )

    def ingest_files(
        self,
        file_paths: List[Union[str, Path]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        persist: bool = True
    ) -> Dict[str, Union[int, List[str]]]:
        """Ingests, chunks, embeds, and indexes a list of files."""
        chunker = DocumentChunker(
            chunk_size=chunk_size or settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP,
            strategy=strategy or settings.DEFAULT_CHUNK_STRATEGY
        )

        loaded_docs: List[LoadedDocument] = []
        for p in file_paths:
            docs = self.loader.load_file(p)
            loaded_docs.extend(docs)

        chunks = chunker.chunk_documents(loaded_docs)
        if not chunks:
            return {"indexed_chunks": 0, "total_documents": len(loaded_docs), "sources": []}

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        if persist:
            self.save_index()

        sources = list(set(c.metadata.get("source", "unknown") for c in chunks))
        return {
            "indexed_chunks": len(chunks),
            "total_documents": len(loaded_docs),
            "sources": sources
        }

    def ingest_text(
        self,
        text: str,
        source_name: str = "direct_input",
        extra_metadata: Optional[dict] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        persist: bool = True
    ) -> int:
        """Ingests raw text directly into the index."""
        docs = self.loader.load_text(text, source_name=source_name, extra_metadata=extra_metadata)
        chunker = DocumentChunker(
            chunk_size=chunk_size or settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP,
            strategy=strategy or settings.DEFAULT_CHUNK_STRATEGY
        )
        chunks = chunker.chunk_documents(docs)
        if not chunks:
            return 0

        embeddings = self.embedding_model.embed_texts([c.text for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

        if persist:
            self.save_index()

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        prompt_strategy: PromptStrategy = PromptStrategy.STRICT_GROUNDED,
        use_mmr: Optional[bool] = None,
        custom_instructions: str = ""
    ) -> RAGQueryResponse:
        """Executes full RAG workflow: retrieval -> generation -> validation -> citation mapping."""
        k = top_k or settings.DEFAULT_TOP_K
        thresh = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        mmr = use_mmr if use_mmr is not None else settings.USE_MMR

        # 1. Retrieval
        retrieval_start = time.perf_counter()
        q_vec = self.embedding_model.embed_query(question)

        if mmr:
            search_results = self.vector_store.search_mmr(
                query_vector=q_vec,
                top_k=k,
                similarity_threshold=thresh,
                lambda_mult=settings.MMR_DIVERSITY_LAMBDA
            )
        else:
            search_results = self.vector_store.search(
                query_vector=q_vec,
                top_k=k,
                similarity_threshold=thresh
            )

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 2. Generation
        gen_result: GenerationResult = self.generator.generate_answer(
            query=question,
            search_results=search_results,
            strategy=prompt_strategy,
            custom_instructions=custom_instructions
        )

        # 3. Validation & Grounding Verification
        validation = self.validator.validate(
            answer=gen_result.answer,
            retrieved_results=search_results
        )

        # 4. Map Citations to Metadata
        citations: List[SourceCitation] = []
        for i, res in enumerate(search_results, start=1):
            # Include citation if referenced by the model or if all top results are cited
            is_cited = i in validation.cited_doc_indices or not validation.cited_doc_indices
            if is_cited:
                citations.append(
                    SourceCitation(
                        doc_index=i,
                        source=str(res.chunk.metadata.get("source", "Unknown")),
                        page_number=int(res.chunk.metadata.get("page_number", 1)),
                        chunk_id=res.chunk.chunk_id,
                        snippet=res.chunk.text[:220] + "..." if len(res.chunk.text) > 220 else res.chunk.text,
                        relevance_score=res.score
                    )
                )

        total_latency = retrieval_ms + gen_result.latency_ms

        return RAGQueryResponse(
            query=question,
            answer=gen_result.answer,
            citations=citations,
            validation=validation,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_result.latency_ms,
            total_latency_ms=total_latency,
            tokens={
                "prompt": gen_result.prompt_tokens,
                "completion": gen_result.completion_tokens,
                "total": gen_result.total_tokens
            },
            cost_usd=gen_result.cost_usd,
            model_name=gen_result.model_name,
            prompt_strategy=prompt_strategy.value,
            retrieved_count=len(search_results)
        )

    def save_index(self, path: Optional[Union[str, Path]] = None):
        """Saves vector index to disk."""
        target = path or settings.FAISS_INDEX_DIR
        self.vector_store.save(target)

    def get_status(self) -> dict:
        """Returns diagnostic status of pipeline components."""
        stats = self.vector_store.get_stats()
        stats.update({
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "llm_model": self.generator.model_name,
            "validation_enabled": settings.ENABLE_OUTPUT_VALIDATION,
            "strict_grounding": settings.STRICT_GROUNDING,
        })
        return stats
