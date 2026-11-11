"""Quantitative evaluation metrics: Accuracy, Hallucination Rate, Retrieval, Latency, and Cost."""
import re
import string
from dataclasses import dataclass
from typing import Callable, List, Optional, Set
import numpy as np


@dataclass
class EvaluationResultItem:
    """Detailed quantitative evaluation record for a single benchmark query."""
    query_id: str
    query: str
    ground_truth: str
    predicted_answer: str
    category: str
    is_refusal: bool

    # Accuracy / Relevance metrics
    token_f1: float
    rouge_l: float
    semantic_similarity: float

    # Hallucination & Faithfulness metrics
    hallucination_rate: float
    groundedness_score: float
    citations_valid: bool

    # Retrieval metrics
    hit_at_k: float
    reciprocal_rank: float

    # Latency & Economics
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "is_refusal": self.is_refusal,
            "token_f1": round(self.token_f1, 4),
            "rouge_l": round(self.rouge_l, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "citations_valid": self.citations_valid,
            "hit_at_k": round(self.hit_at_k, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class RAGMetrics:
    """Static helper methods to calculate RAG evaluation metrics."""

    @staticmethod
    def normalize_text(s: str) -> str:
        """Lowercases text, removes punctuation and redundant whitespace."""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        return " ".join(s.split())

    @classmethod
    def token_f1(cls, prediction: str, ground_truth: str) -> float:
        """Calculates token-level F1 overlap between prediction and reference."""
        pred_tokens = cls.normalize_text(prediction).split()
        gt_tokens = cls.normalize_text(ground_truth).split()

        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0

        prec = len(common) / len(set(pred_tokens))
        rec = len(common) / len(set(gt_tokens))
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    @classmethod
    def rouge_l(cls, prediction: str, ground_truth: str) -> float:
        """Calculates Longest Common Subsequence (ROUGE-L) F-measure."""
        pred_tokens = cls.normalize_text(prediction).split()
        gt_tokens = cls.normalize_text(ground_truth).split()

        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0

        m, n = len(pred_tokens), len(gt_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == gt_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        if lcs_len == 0:
            return 0.0

        prec = lcs_len / m
        rec = lcs_len / n
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def semantic_similarity(
        prediction: str,
        ground_truth: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None
    ) -> float:
        """Calculates cosine similarity between embeddings of predicted and ground truth answer."""
        if not prediction.strip() or not ground_truth.strip():
            return 0.0

        if embed_fn is not None:
            v1 = embed_fn(prediction)
            v2 = embed_fn(ground_truth)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                return float(np.dot(v1, v2) / (norm1 * norm2))

        # Lexical Jaccard fallback if no embedding function provided
        p_set = set(prediction.lower().split())
        g_set = set(ground_truth.lower().split())
        union = p_set | g_set
        return len(p_set & g_set) / len(union) if union else 0.0

    @classmethod
    def retrieval_hit_at_k(
        cls,
        retrieved_texts: List[str],
        expected_keywords: List[str]
    ) -> float:
        """Returns 1.0 if at least one retrieved chunk contains the expected keywords, else 0.0."""
        if not expected_keywords:
            return 1.0

        for chunk_text in retrieved_texts:
            norm_chunk = cls.normalize_text(chunk_text)
            if any(cls.normalize_text(kw) in norm_chunk for kw in expected_keywords):
                return 1.0
        return 0.0

    @classmethod
    def reciprocal_rank(
        cls,
        retrieved_texts: List[str],
        expected_keywords: List[str]
    ) -> float:
        """Computes 1 / rank of the first relevant retrieved chunk."""
        if not expected_keywords:
            return 1.0

        for rank, chunk_text in enumerate(retrieved_texts, start=1):
            norm_chunk = cls.normalize_text(chunk_text)
            if any(cls.normalize_text(kw) in norm_chunk for kw in expected_keywords):
                return 1.0 / rank
        return 0.0

