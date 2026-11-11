"""Tests for embedding models and FAISS vector index."""
import pytest
import numpy as np
from src.ingestion.chunker import DocumentChunk
from src.vectorstore.embeddings import MockEmbeddingModel, HuggingFaceEmbeddingModel
from src.vectorstore.faiss_store import FAISSVectorStore


def test_mock_embedding_model():
    model = MockEmbeddingModel(dim=128)
    assert model.dimension == 128

    vec = model.embed_query("Test query")
    assert vec.shape == (128,)
    # Norm should be approximately 1.0 (unit vector)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)

    batch = model.embed_texts(["First text", "Second text"])
    assert batch.shape == (2, 128)


def test_faiss_vector_store_crud(tmp_path):
    dim = 64
    store = FAISSVectorStore(dimension=dim, index_type="IP")

    chunks = [
        DocumentChunk(chunk_id="chunk_1", text="Cloud microservices architecture", metadata={"source": "doc1.txt"}),
        DocumentChunk(chunk_id="chunk_2", text="E-commerce 30-day refund policy", metadata={"source": "doc2.txt"}),
    ]

    rng = np.random.RandomState(42)
    vectors = rng.randn(2, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    # 1. Add
    added = store.add_chunks(chunks, vectors)
    assert added == 2
    assert store.index.ntotal == 2

    # 2. Search
    q_vec = vectors[0]  # Exact match for chunk_1
    results = store.search(q_vec, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk_1"
    assert np.isclose(results[0].score, 1.0, atol=1e-4)

    # 3. Save & Load
    save_dir = tmp_path / "faiss_test_index"
    store.save(save_dir)

    loaded_store = FAISSVectorStore.load(save_dir)
    assert loaded_store.index.ntotal == 2
    assert len(loaded_store.chunks) == 2
    loaded_results = loaded_store.search(q_vec, top_k=1)
    assert loaded_results[0].chunk.chunk_id == "chunk_1"

    # 4. Clear
    loaded_store.clear()
    assert loaded_store.index.ntotal == 0
    assert len(loaded_store.chunks) == 0


def test_faiss_mmr_search():
    dim = 32
    store = FAISSVectorStore(dimension=dim, index_type="IP")

    chunks = [
        DocumentChunk(chunk_id=f"c_{i}", text=f"Chunk text {i}", metadata={"source": "data.txt"})
        for i in range(5)
    ]
    rng = np.random.RandomState(10)
    vectors = rng.randn(5, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    store.add_chunks(chunks, vectors)
    q_vec = vectors[0]

    mmr_results = store.search_mmr(q_vec, top_k=3, fetch_k=5, lambda_mult=0.6)
    assert len(mmr_results) == 3
