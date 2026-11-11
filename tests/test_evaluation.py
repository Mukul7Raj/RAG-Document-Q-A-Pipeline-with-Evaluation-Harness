"""Tests for evaluation metrics and harness execution."""
import pytest
from evaluation.benchmark_data import BenchmarkQuery
from evaluation.metrics import RAGMetrics
from evaluation.harness import EvaluationHarness
from src.pipeline.rag_pipeline import RAGPipeline
from src.vectorstore.embeddings import MockEmbeddingModel


def test_rag_metrics_token_f1_and_rouge():
    pred = "The circuit breaker opens at 50% error rate."
    gt = "A 50% error rate triggers the circuit breaker to open."

    f1 = RAGMetrics.token_f1(pred, gt)
    rouge = RAGMetrics.rouge_l(pred, gt)

    assert f1 > 0.5
    assert rouge > 0.3

    # Identical strings should be 1.0
    assert RAGMetrics.token_f1("exact match sentence", "exact match sentence") == 1.0
    assert RAGMetrics.rouge_l("exact match sentence", "exact match sentence") == 1.0

    # Completely different
    assert RAGMetrics.token_f1("apple banana orange", "car truck bicycle") == 0.0


def test_retrieval_hit_and_mrr():
    chunks = [
        "Unrelated document about cooking recipes.",
        "Kafka message queues forward to DLQ after 5 retries.",
        "Another random snippet."
    ]
    keywords = ["Kafka", "DLQ", "5 retries"]

    hit = RAGMetrics.retrieval_hit_at_k(chunks, keywords)
    mrr = RAGMetrics.reciprocal_rank(chunks, keywords)

    assert hit == 1.0
    assert mrr == 0.5  # Found at rank 2 -> 1/2


def test_evaluation_harness_benchmark_run(tmp_path):
    embed_model = MockEmbeddingModel(dim=64)
    pipeline = RAGPipeline(embedding_model=embed_model, auto_load_index=False)
    pipeline.vector_store.clear()

    pipeline.ingest_text(
        "A 15% restocking fee applies to opened electronics within 30 days.",
        source_name="policy.txt",
        persist=False
    )

    harness = EvaluationHarness(pipeline)
    test_queries = [
        BenchmarkQuery(
            query_id="TEST_01",
            query="What restocking fee applies to opened electronics?",
            ground_truth_answer="A 15% restocking fee applies to opened electronics.",
            expected_sources=["policy.txt"],
            expected_keywords=["15% restocking fee", "opened electronics"],
            category="policy",
            is_unanswerable=False
        ),
        BenchmarkQuery(
            query_id="TEST_02",
            query="What is the speed of light in vacuum?",
            ground_truth_answer="I do not have sufficient information in the provided context to answer this question.",
            expected_sources=[],
            expected_keywords=[],
            category="unanswerable",
            is_unanswerable=True
        )
    ]

    summary = harness.run_benchmark(queries=test_queries, experiment_name="Harness_Unit_Test")
    assert summary.total_queries == 2
    assert summary.mean_groundedness_score >= 0.0
    assert summary.latency_p50_ms > 0
    assert summary.cost_per_1000_queries >= 0.0

    report_path = harness.save_report(summary, output_dir=tmp_path)
    assert report_path.exists()
