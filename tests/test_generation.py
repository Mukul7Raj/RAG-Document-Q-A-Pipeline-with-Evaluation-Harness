"""Tests for prompt engineering, answer generation, and hallucination guardrails."""
from src.generation.prompts import PromptManager, PromptStrategy
from src.generation.validator import OutputValidator
from src.generation.generator import AnswerGenerator, MockLLMClient
from src.ingestion.chunker import DocumentChunk
from src.vectorstore.faiss_store import SearchResult


def test_prompt_manager_build():
    results = [
        SearchResult(
            chunk=DocumentChunk(chunk_id="c1", text="Istio service mesh enforces mTLS.", metadata={"source": "spec.txt"}),
            score=0.92
        )
    ]
    prompt_dict = PromptManager.build_prompt("What does Istio enforce?", results, PromptStrategy.STRICT_GROUNDED)

    assert "strict_grounded" in PromptStrategy.STRICT_GROUNDED.value
    assert "You are a rigorous, trustworthy technical assistant" in prompt_dict["system_prompt"]
    assert "[Doc 1]" in prompt_dict["user_prompt"]
    assert "What does Istio enforce?" in prompt_dict["user_prompt"]


def test_output_validator_citations_and_refusal():
    validator = OutputValidator(hallucination_threshold=0.5, enforce_citations=True)
    results = [
        SearchResult(
            chunk=DocumentChunk(chunk_id="c1", text="Circuit breakers open after a 50% failure rate.", metadata={"source": "doc.txt"}),
            score=0.88
        )
    ]

    # Valid grounded answer with citation
    valid_ans = "The circuit breaker transitions to an open state when failure rate hits 50% [Doc 1]."
    val_res = validator.validate(valid_ans, results)
    assert val_res.is_valid is True
    assert 1 in val_res.valid_citations
    assert len(val_res.invalid_citations) == 0
    assert val_res.hallucination_score < 0.5

    # Invalid phantom citation [Doc 99]
    phantom_ans = "Circuit breakers trip at 50% failure [Doc 99]."
    val_phantom = validator.validate(phantom_ans, results)
    assert val_phantom.is_valid is False
    assert 99 in val_phantom.invalid_citations

    # Refusal handling
    refusal_ans = "I do not have sufficient information in the provided context to answer this question."
    val_refusal = validator.validate(refusal_ans, results)
    assert val_refusal.is_valid is True
    assert val_refusal.is_refusal is True
    assert val_refusal.hallucination_score == 0.0


def test_mock_llm_client():
    client = MockLLMClient()
    user_prompt = (
        "CONTEXT DOCUMENTS:\n"
        "--- [Doc 1] (Source: arch.txt, Page: 1) ---\n"
        "Envoy proxy enforces a Token Bucket rate limit algorithm with 100 requests per second.\n\n"
        "USER QUESTION: What rate limit algorithm does Envoy enforce?\n\n"
        "GROUNDED ANSWER (with citations):"
    )
    answer = client.generate("System prompt", user_prompt)
    assert "[Doc 1]" in answer
    assert "Token Bucket" in answer
