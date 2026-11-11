"""Tests for FastAPI endpoints using TestClient."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.deps import set_pipeline
from src.pipeline.rag_pipeline import RAGPipeline
from src.vectorstore.embeddings import MockEmbeddingModel


@pytest.fixture(scope="module")
def client():
    # Setup test pipeline with fast mock embeddings
    test_pipeline = RAGPipeline(
        embedding_model=MockEmbeddingModel(dim=64),
        auto_load_index=False
    )
    test_pipeline.vector_store.clear()
    test_pipeline.ingest_text(
        "Standard tier allows a baseline rate of 100 requests per second with burst of 20 tokens.",
        source_name="rate_limits.txt",
        persist=False
    )
    set_pipeline(test_pipeline)

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "endpoints" in data


def test_query_endpoint(client):
    payload = {
        "query": "What baseline rate does the Standard Tier allow?",
        "top_k": 2,
        "similarity_threshold": 0.0
    }
    res = client.post("/api/v1/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "citations" in data
    assert "performance" in data
    assert "validation" in data
    assert data["performance"]["total_latency_ms"] > 0


def test_ingest_text_endpoint(client):
    payload = {
        "text": "OpenTelemetry exports traces to Jaeger with a 10% sampling rate.",
        "source_name": "tracing_spec.txt",
        "chunk_size": 200,
        "chunk_overlap": 20
    }
    res = client.post("/api/v1/ingest/text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["indexed_chunks"] > 0
    assert "tracing_spec.txt" in data["sources"]


def test_list_documents_endpoint(client):
    res = client.get("/api/v1/documents")
    assert res.status_code == 200
    data = res.json()
    assert data["total_chunks"] > 0
    assert len(data["unique_sources"]) > 0


def test_evaluation_run_endpoint(client):
    payload = {
        "experiment_name": "TestClient_Benchmark",
        "top_k": 2
    }
    res = client.post("/api/v1/evaluation/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_queries"] > 0
    assert "metrics" in data
    assert "performance" in data

