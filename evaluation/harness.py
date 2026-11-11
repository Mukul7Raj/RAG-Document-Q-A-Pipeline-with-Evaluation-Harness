"""Evaluation harness orchestrating benchmark runs, metric aggregations, and ablation sweeps."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from evaluation.benchmark_data import BenchmarkQuery, get_default_benchmark_dataset
from evaluation.metrics import EvaluationResultItem, RAGMetrics
from src.generation.prompts import PromptStrategy
from src.pipeline.rag_pipeline import RAGPipeline
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """Aggregated quantitative performance summary over an evaluation benchmark."""
    experiment_name: str
    total_queries: int
    mean_token_f1: float
    mean_rouge_l: float
    mean_semantic_similarity: float
    mean_hallucination_rate: float
    mean_groundedness_score: float
    refusal_accuracy: float
    hit_at_k: float
    mean_reciprocal_rank: float
    latency_p50_ms: float
    latency_p90_ms: float
    mean_total_latency_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    total_cost_usd: float
    cost_per_1000_queries: float
    detailed_items: List[EvaluationResultItem] = field(default_factory=list)
    configuration: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "total_queries": self.total_queries,
            "metrics": {
                "mean_token_f1": round(self.mean_token_f1, 4),
                "mean_rouge_l": round(self.mean_rouge_l, 4),
                "mean_semantic_similarity": round(self.mean_semantic_similarity, 4),
                "mean_hallucination_rate": round(self.mean_hallucination_rate, 4),
                "mean_groundedness_score": round(self.mean_groundedness_score, 4),
                "refusal_accuracy": round(self.refusal_accuracy, 4),
                "hit_at_k": round(self.hit_at_k, 4),
                "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            },
            "performance": {
                "latency_p50_ms": round(self.latency_p50_ms, 2),
                "latency_p90_ms": round(self.latency_p90_ms, 2),
                "mean_total_latency_ms": round(self.mean_total_latency_ms, 2),
                "mean_prompt_tokens": round(self.mean_prompt_tokens, 1),
                "mean_completion_tokens": round(self.mean_completion_tokens, 1),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "cost_per_1000_queries": round(self.cost_per_1000_queries, 4),
            },
            "configuration": self.configuration,
            "queries": [item.to_dict() for item in self.detailed_items]
        }

    def to_markdown_table(self) -> str:
        """Formats the summary into a clean markdown table."""
        return (
            f"### Evaluation Benchmark Results: {self.experiment_name}\n\n"
            f"| Metric | Score / Value |\n"
            f"| :--- | :--- |\n"
            f"| **Evaluated Queries** | {self.total_queries} |\n"
            f"| **Answer Token F1** | {self.mean_token_f1:.2%} |\n"
            f"| **Answer ROUGE-L** | {self.mean_rouge_l:.2%} |\n"
            f"| **Semantic Similarity** | {self.mean_semantic_similarity:.2%} |\n"
            f"| **Hallucination Rate** | **{self.mean_hallucination_rate:.2%}** |\n"
            f"| **Groundedness Score** | **{self.mean_groundedness_score:.2%}** |\n"
            f"| **Refusal Accuracy** | {self.refusal_accuracy:.2%} |\n"
            f"| **Retrieval Hit@K** | {self.hit_at_k:.2%} |\n"
            f"| **Mean Reciprocal Rank (MRR)** | {self.mean_reciprocal_rank:.4f} |\n"
            f"| **Latency P50** | {self.latency_p50_ms:.1f} ms |\n"
            f"| **Latency P90** | {self.latency_p90_ms:.1f} ms |\n"
            f"| **Mean Total Latency** | {self.mean_total_latency_ms:.1f} ms |\n"
            f"| **Avg Tokens / Query** | {self.mean_prompt_tokens + self.mean_completion_tokens:.0f} tokens |\n"
            f"| **Estimated Cost / 1k Queries** | **${self.cost_per_1000_queries:.4f}** |\n"
        )


class EvaluationHarness:
    """Runs benchmark evaluation against a configured RAG pipeline instance."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run_benchmark(
        self,
        queries: Optional[List[BenchmarkQuery]] = None,
        experiment_name: str = "Default_RAG_Benchmark",
        prompt_strategy: PromptStrategy = PromptStrategy.STRICT_GROUNDED,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> EvaluationSummary:
        """Executes full benchmark suite and computes aggregate quantitative metrics."""
        test_queries = queries or get_default_benchmark_dataset()
        results: List[EvaluationResultItem] = []

        logger.info("Executing evaluation benchmark '%s' (%d queries)...", experiment_name, len(test_queries))

        for bq in test_queries:
            # Query the pipeline
            response = self.pipeline.query(
                question=bq.query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                prompt_strategy=prompt_strategy
            )

            # Accuracy metrics
            token_f1 = RAGMetrics.token_f1(response.answer, bq.ground_truth_answer)
            rouge_l = RAGMetrics.rouge_l(response.answer, bq.ground_truth_answer)
            sem_sim = RAGMetrics.semantic_similarity(
                response.answer,
                bq.ground_truth_answer,
                embed_fn=self.pipeline.embedding_model.embed_query
            )

            # Retrieval metrics
            retrieved_texts = [c.snippet for c in response.citations]
            hit = RAGMetrics.retrieval_hit_at_k(retrieved_texts, bq.expected_keywords)
            mrr_score = RAGMetrics.reciprocal_rank(retrieved_texts, bq.expected_keywords)

            # Hallucination & citations
            hallucination_rate = response.validation.hallucination_score
            groundedness = response.validation.groundedness_score
            cites_valid = len(response.validation.invalid_citations) == 0

            is_refusal = response.validation.is_refusal

            results.append(
                EvaluationResultItem(
                    query_id=bq.query_id,
                    query=bq.query,
                    ground_truth=bq.ground_truth_answer,
                    predicted_answer=response.answer,
                    category=bq.category,
                    is_refusal=is_refusal,
                    token_f1=token_f1,
                    rouge_l=rouge_l,
                    semantic_similarity=sem_sim,
                    hallucination_rate=hallucination_rate,
                    groundedness_score=groundedness,
                    citations_valid=cites_valid,
                    hit_at_k=hit,
                    reciprocal_rank=mrr_score,
                    retrieval_latency_ms=response.retrieval_latency_ms,
                    generation_latency_ms=response.generation_latency_ms,
                    total_latency_ms=response.total_latency_ms,
                    prompt_tokens=response.tokens.get("prompt", 0),
                    completion_tokens=response.tokens.get("completion", 0),
                    total_tokens=response.tokens.get("total", 0),
                    cost_usd=response.cost_usd
                )
            )

        # Aggregate metrics
        latencies = [r.total_latency_ms for r in results]
        latencies.sort()
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p90 = float(np.percentile(latencies, 90)) if latencies else 0.0

        # Refusal accuracy: did it refuse unanswerable queries AND answer answerable ones?
        refusal_correct = sum(
            1 for bq, r in zip(test_queries, results)
            if (bq.is_unanswerable and r.is_refusal) or (not bq.is_unanswerable and not r.is_refusal)
        )
        refusal_acc = refusal_correct / len(test_queries) if test_queries else 1.0

        total_cost = sum(r.cost_usd for r in results)
        cost_per_1k = (total_cost / len(results)) * 1000.0 if results else 0.0

        summary = EvaluationSummary(
            experiment_name=experiment_name,
            total_queries=len(results),
            mean_token_f1=float(np.mean([r.token_f1 for r in results])),
            mean_rouge_l=float(np.mean([r.rouge_l for r in results])),
            mean_semantic_similarity=float(np.mean([r.semantic_similarity for r in results])),
            mean_hallucination_rate=float(np.mean([r.hallucination_rate for r in results])),
            mean_groundedness_score=float(np.mean([r.groundedness_score for r in results])),
            refusal_accuracy=refusal_acc,
            hit_at_k=float(np.mean([r.hit_at_k for r in results])),
            mean_reciprocal_rank=float(np.mean([r.reciprocal_rank for r in results])),
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            mean_total_latency_ms=float(np.mean(latencies)),
            mean_prompt_tokens=float(np.mean([r.prompt_tokens for r in results])),
            mean_completion_tokens=float(np.mean([r.completion_tokens for r in results])),
            total_cost_usd=total_cost,
            cost_per_1000_queries=cost_per_1k,
            detailed_items=results,
            configuration={
                "prompt_strategy": prompt_strategy.value,
                "top_k": top_k or settings.DEFAULT_TOP_K,
                "similarity_threshold": similarity_threshold or settings.SIMILARITY_THRESHOLD,
                "model_name": self.pipeline.generator.model_name,
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
            }
        )

        return summary

    def save_report(
        self,
        summary: EvaluationSummary,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Saves evaluation summary as JSON and Markdown reports."""
        out = Path(output_dir or settings.EVAL_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)

        slug = summary.experiment_name.lower().replace(" ", "_")
        json_file = out / f"{slug}_report.json"
        md_file = out / f"{slug}_report.md"

        json_file.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        md_file.write_text(summary.to_markdown_table(), encoding="utf-8")

        logger.info("Saved evaluation reports to %s and %s", json_file, md_file)
        return md_file


class AblationStudyRunner:
    """Executes systematic ablation sweeps over chunking parameters and prompt strategies."""

    def __init__(self, sample_files: List[Path]):
        self.sample_files = sample_files

    def run_chunking_sweep(
        self,
        chunk_configs: Optional[List[Dict[str, int]]] = None
    ) -> List[EvaluationSummary]:
        """Runs ablation comparing different chunk sizes and overlaps."""
        configs = chunk_configs or [
            {"chunk_size": 200, "chunk_overlap": 20, "name": "Small_Chunks_200"},
            {"chunk_size": 500, "chunk_overlap": 50, "name": "Medium_Chunks_500"},
            {"chunk_size": 1000, "chunk_overlap": 100, "name": "Large_Chunks_1000"},
        ]

        summaries: List[EvaluationSummary] = []
        for cfg in configs:
            pipeline = RAGPipeline(auto_load_index=False)
            pipeline.vector_store.clear()
            pipeline.ingest_files(
                file_paths=self.sample_files,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                persist=False
            )

            harness = EvaluationHarness(pipeline)
            summary = harness.run_benchmark(experiment_name=cfg["name"])
            summaries.append(summary)

        return summaries

    def run_prompt_strategy_sweep(
        self,
        pipeline: RAGPipeline
    ) -> List[EvaluationSummary]:
        """Runs ablation comparing Strict Grounded vs Chain-of-Thought vs Concise prompts."""
        strategies = [
            (PromptStrategy.STRICT_GROUNDED, "Strategy_Strict_Grounded"),
            (PromptStrategy.CHAIN_OF_THOUGHT, "Strategy_Chain_of_Thought"),
            (PromptStrategy.CONCISE, "Strategy_Concise"),
        ]

        harness = EvaluationHarness(pipeline)
        summaries: List[EvaluationSummary] = []

        for strat, name in strategies:
            summary = harness.run_benchmark(
                experiment_name=name,
                prompt_strategy=strat
            )
            summaries.append(summary)

        return summaries

    @staticmethod
    def generate_comparison_table(summaries: List[EvaluationSummary]) -> str:
        """Produces a unified comparative markdown table across all ablation variations."""
        header = (
            "| Experiment Variation | Token F1 | Hallucination Rate | Groundedness | Hit@K | Latency P50 | Cost / 1k Queries |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        rows = []
        for s in summaries:
            row = (
                f"| **{s.experiment_name}** | {s.mean_token_f1:.2%} | "
                f"**{s.mean_hallucination_rate:.2%}** | {s.mean_groundedness_score:.2%} | "
                f"{s.hit_at_k:.2%} | {s.latency_p50_ms:.1f}ms | ${s.cost_per_1000_queries:.4f} |"
            )
            rows.append(row)
        return header + "\n".join(rows) + "\n"
