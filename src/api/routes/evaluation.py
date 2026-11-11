"""Evaluation API route for triggering benchmark runs."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from src.api.deps import get_pipeline
from src.pipeline.rag_pipeline import RAGPipeline
from evaluation.harness import EvaluationHarness
from src.generation.prompts import PromptStrategy

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationRunRequest(BaseModel):
    experiment_name: Optional[str] = Field("API_Triggered_Benchmark", description="Experiment identifier")
    prompt_strategy: Optional[PromptStrategy] = Field(PromptStrategy.STRICT_GROUNDED, description="Prompt style to evaluate")
    top_k: Optional[int] = Field(4, description="Retrieval top-k for evaluation run")


@router.post("/run")
async def run_evaluation_benchmark(
    request: EvaluationRunRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Executes evaluation harness over current vector store using default benchmark suite."""
    try:
        harness = EvaluationHarness(pipeline)
        summary = harness.run_benchmark(
            experiment_name=request.experiment_name or "API_Triggered_Benchmark",
            prompt_strategy=request.prompt_strategy or PromptStrategy.STRICT_GROUNDED,
            top_k=request.top_k
        )
        return summary.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation execution failed: {str(e)}")
