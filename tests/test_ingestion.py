"""Tests for document loading and chunking strategies."""
import pytest
from pathlib import Path
from src.ingestion.loader import DocumentLoader, LoadedDocument
from src.ingestion.chunker import DocumentChunker, DocumentChunk


def test_document_loader_text(tmp_path):
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text("Hello world! This is a test document for RAG ingestion.", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_file(file_path)

    assert len(docs) == 1
    assert docs[0].text.startswith("Hello world!")
    assert docs[0].metadata["source"] == "test_doc.txt"
    assert docs[0].metadata["file_type"] == ".txt"
    assert docs[0].doc_id != ""


def test_document_loader_raw_text():
    loader = DocumentLoader()
    docs = loader.load_text("Direct text ingestion string.", source_name="custom_source")

    assert len(docs) == 1
    assert docs[0].text == "Direct text ingestion string."
    assert docs[0].metadata["source"] == "custom_source"


def test_document_chunker_recursive():
    sample_text = (
        "Section 1: Microservices Topology.\n\n"
        "Our backend services communicate using Istio service mesh.\n"
        "Mutual TLS is enforced across all internal network links.\n\n"
        "Section 2: Circuit Breakers.\n\n"
        "Resilience4j handles downstream fault tolerance with a 50% error threshold."
    )
    doc = LoadedDocument(text=sample_text, metadata={"source": "test.txt"})
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, strategy="recursive")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.chunk_id.startswith(doc.doc_id)
        assert c.char_count > 0
        assert c.token_count > 0
        assert c.metadata["source"] == "test.txt"


def test_document_chunker_token_strategy():
    sample_text = "Word " * 200
    doc = LoadedDocument(text=sample_text, metadata={"source": "token_test.txt"})
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, strategy="token")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    assert chunks[0].metadata["chunk_strategy"] == "token"


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=150)

