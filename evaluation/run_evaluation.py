"""CLI runner for RAG Evaluation Harness and Ablation Sweeps."""
import argparse
import logging
from pathlib import Path
from config.settings import settings
from evaluation.benchmark_data import get_default_benchmark_dataset
from evaluation.harness import AblationStudyRunner, EvaluationHarness
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_runner")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Harness & Ablation Runner")
    parser.add_argument("--sweep", action="store_true", help="Run ablation sweep across chunk sizes and prompt strategies")
    parser.add_argument("--output-dir", type=str, default=str(settings.EVAL_OUTPUT_DIR), help="Directory to save evaluation reports")
    args = parser.parse_args()

    sample_dir = Path(settings.SAMPLE_DATA_DIR)
    sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))

    if not sample_files:
        logger.error("No sample documents found in %s to evaluate against.", sample_dir)
        return

    logger.info("Found %d sample documents in %s", len(sample_files), sample_dir)

    # 1. Ingest documents into default pipeline
    pipeline = RAGPipeline(auto_load_index=False)
    pipeline.vector_store.clear()
    logger.info("Ingesting sample documents for benchmark...")
    ingest_result = pipeline.ingest_files(sample_files, persist=True)
    logger.info("Ingested %d chunks from %d files.", ingest_result["indexed_chunks"], ingest_result["total_documents"])

    harness = EvaluationHarness(pipeline)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.sweep:
        # Run standard benchmark
        logger.info("Running baseline evaluation benchmark...")
        summary = harness.run_benchmark(experiment_name="Baseline_Production_RAG")
        md_file = harness.save_report(summary, output_dir=out_dir)
        print("\n" + summary.to_markdown_table())
        print(f"Full JSON and Markdown reports saved to: {out_dir}\n")
    else:
        # Run systematic ablation study
        logger.info("Executing comprehensive ablation study across chunking and prompt variations...")
        runner = AblationStudyRunner(sample_files=sample_files)

        # 1. Chunking sweep
        chunk_summaries = runner.run_chunking_sweep()
        for s in chunk_summaries:
            harness.save_report(s, output_dir=out_dir)

        # 2. Prompt strategy sweep
        prompt_summaries = runner.run_prompt_strategy_sweep(pipeline)
        for s in prompt_summaries:
            harness.save_report(s, output_dir=out_dir)

        all_summaries = chunk_summaries + prompt_summaries
        matrix_table = runner.generate_comparison_table(all_summaries)

        matrix_file = out_dir / "ablation_comparison_matrix.md"
        matrix_file.write_text(matrix_table, encoding="utf-8")

        print("\n=== ABLATION STUDY COMPARISON MATRIX ===")
        print(matrix_table)
        print(f"Ablation matrix written to: {matrix_file}\n")


if __name__ == "__main__":
    main()
