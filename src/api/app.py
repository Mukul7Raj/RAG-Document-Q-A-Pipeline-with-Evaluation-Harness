"""FastAPI application factory, lifespan management, and router integration."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.api.deps import get_pipeline
from src.api.schemas import HealthResponse
from src.api.routes import documents, evaluation, ingest, query

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to warm up vector store and embedding models on startup."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline ready. Vectors in store: %d", pipeline.vector_store.index.ntotal)
    except Exception as e:
        logger.error("Error during startup pipeline warmup: %s", e)
    yield
    logger.info("Shutting down RAG API service.")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Production RAG Document Q&A Pipeline with Grounded Citations, "
            "FAISS Vector Search, Hallucination Guardrails, and Quantitative Evaluation."
        ),
        lifespan=lifespan
    )

    # Enable CORS for frontend integrations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(query.router, prefix=settings.API_PREFIX)
    app.include_router(ingest.router, prefix=settings.API_PREFIX)
    app.include_router(documents.router, prefix=settings.API_PREFIX)
    app.include_router(evaluation.router, prefix=settings.API_PREFIX)

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            vector_store=pipeline.vector_store.get_stats(),
            embedding_provider=settings.EMBEDDING_PROVIDER,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_model=pipeline.generator.model_name
        )

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": "/health",
            "endpoints": {
                "query": f"{settings.API_PREFIX}/query",
                "ingest_text": f"{settings.API_PREFIX}/ingest/text",
                "ingest_files": f"{settings.API_PREFIX}/ingest/files",
                "documents": f"{settings.API_PREFIX}/documents",
                "evaluation": f"{settings.API_PREFIX}/evaluation/run",
            }
        }

    return app


app = create_app()

