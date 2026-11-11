"""End-to-end tests for the complete RAGPipeline."""
from src.pipeline.rag_pipeline import RAGPipeline
from src.vectorstore.embeddings import MockEmbeddingModel


def test_pipeline_e2e_ingest_and_query(tmp_path):
    # Use mock embedding for fast offline testing
    embed_model = MockEmbeddingModel(dim=64)
    pipeline = RAGPipeline(embedding_model=embed_model, auto_load_index=False)
    pipeline.vector_store.clear()

    # Ingest text
    sample_text = (
        "Production clusters run Kafka with 3 broker replication. "
        "Failed messages are directed to a Dead Letter Queue after 5 retries."
    )
    c_count = pipeline.ingest_text(sample_text, source_name="kafka_guide.txt", persist=False)
    assert c_count > 0

    # Query with threshold=0.0 for mock embeddings
    response = pipeline.query(
        "How many retries before messages go to Dead Letter Queue?",
        similarity_threshold=0.0
    )

    assert response.query == "How many retries before messages go to Dead Letter Queue?"
    assert response.total_latency_ms > 0
    assert response.retrieval_latency_ms > 0
    assert len(response.citations) > 0
    assert response.citations[0].source == "kafka_guide.txt"
    assert response.validation is not None
    assert response.tokens["total"] > 0
