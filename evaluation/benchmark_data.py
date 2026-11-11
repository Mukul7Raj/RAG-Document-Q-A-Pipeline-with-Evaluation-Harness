"""Benchmark evaluation dataset containing factual queries, ground-truth references, and unanswerable tests."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkQuery:
    """Represents a benchmark test case with expected answer, keywords, and unanswerability flag."""
    query_id: str
    query: str
    ground_truth_answer: str
    expected_sources: List[str]
    expected_keywords: List[str]
    category: str
    is_unanswerable: bool = False


DEFAULT_BENCHMARK_SUITE: List[BenchmarkQuery] = [
    BenchmarkQuery(
        query_id="BENCH_01",
        query="What failure rate threshold triggers the circuit breaker into an OPEN state, and how long does it remain open?",
        ground_truth_answer="A 50% error rate threshold triggers the circuit breaker into an OPEN state, and it remains open for exactly 30 seconds before transitioning to HALF-OPEN.",
        expected_sources=["cloud_architecture_spec.txt"],
        expected_keywords=["50% error rate", "OPEN state", "30 seconds", "Resilience4j"],
        category="technical_spec",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_02",
        query="What rate limiting algorithm is used, and what is the baseline rate for the Standard Tier?",
        ground_truth_answer="The Token Bucket algorithm is used for rate limiting. The Standard Tier allows a baseline rate of 100 requests per second with a burst capacity of 20 tokens.",
        expected_sources=["cloud_architecture_spec.txt"],
        expected_keywords=["Token Bucket", "100 requests per second", "burst capacity of 20"],
        category="technical_spec",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_03",
        query="How many retry attempts occur before a message is forwarded to a Dead Letter Queue (DLQ)?",
        ground_truth_answer="Any message that fails processing after 5 retry attempts with exponential backoff is forwarded to a Dead Letter Queue.",
        expected_sources=["cloud_architecture_spec.txt"],
        expected_keywords=["5 retry attempts", "Dead Letter Queue", "DLQ", "exponential backoff"],
        category="technical_spec",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_04",
        query="What is the standard return window for hardware products, and what restocking fee applies to opened electronics?",
        ground_truth_answer="Customers can return physical hardware products within 30 calendar days of delivery. A mandatory 15% restocking fee is deducted for opened electronics.",
        expected_sources=["refund_policy.md"],
        expected_keywords=["30 calendar days", "15% restocking fee", "opened electronics"],
        category="policy_terms",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_05",
        query="Are custom configured-to-order (CTO) enterprise servers eligible for return?",
        ground_truth_answer="No, custom configured-to-order (CTO) enterprise servers and custom laser-engraved devices are strictly non-refundable and cannot be returned under any circumstances.",
        expected_sources=["refund_policy.md"],
        expected_keywords=["non-refundable", "custom configured-to-order", "cannot be returned"],
        category="policy_terms",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_06",
        query="How many business days does customer support take to review and issue an RMA number?",
        ground_truth_answer="The support team reviews and issues an RMA number within 2 business days of the request.",
        expected_sources=["refund_policy.md"],
        expected_keywords=["2 business days", "RMA number", "reviews and issues"],
        category="policy_terms",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_07",
        query="What are the maximum acceptable thresholds for prompt injection bypass rate and benchmark hallucination rate before flagging?",
        ground_truth_answer="A prompt injection bypass rate exceeding 1.5% or a benchmark hallucination rate exceeding 5.0% triggers an instant flag for human intervention and roll-back.",
        expected_sources=["ai_safety_guidelines.txt"],
        expected_keywords=["1.5%", "5.0%", "prompt injection", "hallucination rate"],
        category="ai_governance",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_08",
        query="How are PII records like Social Security Numbers and credit card numbers masked in the pipeline?",
        ground_truth_answer="Social Security Numbers and credit card numbers must be masked using 256-bit cryptographic salted tokens after automated entity recognition.",
        expected_sources=["ai_safety_guidelines.txt"],
        expected_keywords=["256-bit cryptographic salted tokens", "Social Security Numbers", "entity recognition"],
        category="ai_governance",
        is_unanswerable=False
    ),
    BenchmarkQuery(
        query_id="BENCH_09",
        query="What quantum encryption algorithm is specified for interstellar communication satellites?",
        ground_truth_answer="I do not have sufficient information in the provided context to answer this question.",
        expected_sources=[],
        expected_keywords=[],
        category="adversarial_unanswerable",
        is_unanswerable=True
    ),
    BenchmarkQuery(
        query_id="BENCH_10",
        query="What is the discount rate for employees purchasing company stock options in the 401k plan?",
        ground_truth_answer="I do not have sufficient information in the provided context to answer this question.",
        expected_sources=[],
        expected_keywords=[],
        category="adversarial_unanswerable",
        is_unanswerable=True
    ),
]


def get_default_benchmark_dataset() -> List[BenchmarkQuery]:
    """Returns the comprehensive benchmark evaluation dataset."""
    return list(DEFAULT_BENCHMARK_SUITE)

