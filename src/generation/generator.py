"""Grounded answer generator with token tracking, latency metrics, and cost estimation."""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.generation.prompts import PromptManager, PromptStrategy
from src.vectorstore.faiss_store import SearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Encapsulates generated answer, latency, token consumption, and estimated cost."""
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    model_name: str
    strategy: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "model_name": self.model_name,
            "strategy": self.strategy,
        }


class MockLLMClient:
    """Deterministic LLM simulator that extracts grounded answers with citations for offline runs."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Extract question
        q_match = re.search(r"USER QUESTION:\s*(.*?)(?=\n\nGROUNDED ANSWER)", user_prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""

        # Extract context blocks
        doc_blocks = re.findall(r"---\s*\[Doc (\d+)\][^\n]*---\n(.*?)(?=(?:---\s*\[Doc|\n\nUSER QUESTION|\Z))", user_prompt, re.DOTALL)

        if not doc_blocks:
            return "I do not have sufficient information in the provided context to answer this question."

        q_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "a", "an", "how", "why", "when", "where", "are"}

        scored_sentences = []
        for doc_num, text in doc_blocks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap > 0:
                    scored_sentences.append((overlap, int(doc_num), s))

        if not scored_sentences:
            return "I do not have sufficient information in the provided context to answer this question."

        # Sort by overlap descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sentences[:3]

        answer_parts = []
        for _, doc_num, sentence in top_sentences:
            clean_s = sentence.rstrip(".")
            answer_parts.append(f"{clean_s} [Doc {doc_num}].")

        return " ".join(answer_parts)


class AnswerGenerator:
    """Generates grounded answers using OpenAI API or deterministic offline fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
        force_mock: bool = False
    ):
        self.model_name = model_name or settings.OPENAI_MODEL_NAME
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.api_base = api_base or settings.OPENAI_API_BASE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.force_mock = force_mock

        self._mock_client = MockLLMClient()
        self._openai_client = None

        if not self.force_mock and self.api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
            except Exception as e:
                logger.warning("Could not initialize OpenAI client: %s. Using MockLLMClient.", e)

    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        strategy: PromptStrategy = PromptStrategy.STRICT_GROUNDED,
        custom_instructions: str = ""
    ) -> GenerationResult:
        """Constructs prompt, queries model, and tracks token usage and latency."""
        prompt_dict = PromptManager.build_prompt(
            query=query,
            search_results=search_results,
            strategy=strategy,
            custom_instructions=custom_instructions
        )

        system_prompt = prompt_dict["system_prompt"]
        user_prompt = prompt_dict["user_prompt"]

        start_time = time.perf_counter()

        if self._openai_client is not None and not self.force_mock:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                answer_text = response.choices[0].message.content or ""
                p_tokens = response.usage.prompt_tokens if response.usage else 0
                c_tokens = response.usage.completion_tokens if response.usage else 0
                t_tokens = response.usage.total_tokens if response.usage else (p_tokens + c_tokens)
                cost = self._calculate_cost(self.model_name, p_tokens, c_tokens)

                return GenerationResult(
                    answer=answer_text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    model_name=self.model_name,
                    strategy=strategy.value
                )
            except Exception as e:
                logger.warning("OpenAI API call failed: %s. Falling back to local offline mock generation.", e)

        # Mock fallback
        time.sleep(0.015)  # Simulate small compute overhead
        answer_text = self._mock_client.generate(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Estimate token counts
        p_tokens = len(system_prompt.split()) + len(user_prompt.split())
        c_tokens = len(answer_text.split())
        t_tokens = p_tokens + c_tokens
        cost = self._calculate_cost(self.model_name, p_tokens, c_tokens)

        return GenerationResult(
            answer=answer_text,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_name=f"{self.model_name}-offline-mock",
            strategy=strategy.value
        )

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = settings.MODEL_PRICING.get(model, settings.MODEL_PRICING["default"])
        prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
        return prompt_cost + completion_cost
