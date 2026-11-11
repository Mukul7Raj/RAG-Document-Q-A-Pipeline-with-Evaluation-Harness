"""Engineered prompt templates enforcing strict grounding and source citations."""
from enum import Enum
from typing import List
from src.vectorstore.faiss_store import SearchResult


class PromptStrategy(str, Enum):
    STRICT_GROUNDED = "strict_grounded"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CONCISE = "concise"


class PromptManager:
    """Manages prompt templates and context formatting for RAG generation."""

    STRICT_SYSTEM_PROMPT = (
        "You are a rigorous, trustworthy technical assistant answering questions based strictly on the provided context.\n"
        "Guidelines:\n"
        "1. Answer ONLY using the facts explicitly stated in the context below.\n"
        "2. Do NOT use outside knowledge, speculate, extrapolate, or assume unstated details.\n"
        "3. For every substantive claim you make, cite the corresponding document reference using square brackets, e.g., [Doc 1], [Doc 2].\n"
        "4. If multiple documents support a statement, cite all of them: [Doc 1, Doc 3].\n"
        "5. If the provided context does not contain enough information to answer the question truthfully and completely, "
        "explicitly state: 'I do not have sufficient information in the provided context to answer this question.'\n"
        "6. Do not fabricate citations or reference non-existent documents."
    )

    COT_SYSTEM_PROMPT = (
        "You are an analytical assistant that solves questions using retrieved documents through step-by-step reasoning.\n"
        "Follow this exact procedure:\n"
        "Step 1 - Evidence Extraction: Quote the exact sentences from the context that directly address the user's question, noting [Doc X].\n"
        "Step 2 - Synthesis: Reconcile or synthesize the extracted quotes into a coherent explanation.\n"
        "Step 3 - Final Answer: Provide a crisp, directly grounded final answer with inline citations [Doc X].\n"
        "If the context lacks evidence for the query, state: 'Insufficient context available to answer.'"
    )

    CONCISE_SYSTEM_PROMPT = (
        "You are a concise enterprise search assistant. Provide a direct, factual 1-3 sentence answer "
        "derived exclusively from the provided context. Include inline citations like [Doc 1]. "
        "If unknown from context, state: 'I cannot find this information in the provided documents.'"
    )

    @classmethod
    def format_context_blocks(cls, search_results: List[SearchResult]) -> str:
        """Formats retrieved chunks into numbered context blocks for prompt injection."""
        if not search_results:
            return "No relevant context found."

        blocks = []
        for i, res in enumerate(search_results, start=1):
            source = res.chunk.metadata.get("source", "Unknown")
            page = res.chunk.metadata.get("page_number", 1)
            chunk_id = res.chunk.chunk_id

            block = (
                f"--- [Doc {i}] (Source: {source}, Page: {page}, ID: {chunk_id}, Relevance: {res.score:.2f}) ---\n"
                f"{res.chunk.text}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    @classmethod
    def build_prompt(
        cls,
        query: str,
        search_results: List[SearchResult],
        strategy: PromptStrategy = PromptStrategy.STRICT_GROUNDED,
        custom_instructions: str = ""
    ) -> dict:
        """Constructs system and user messages according to the selected strategy."""
        if strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            system_prompt = cls.COT_SYSTEM_PROMPT
        elif strategy == PromptStrategy.CONCISE:
            system_prompt = cls.CONCISE_SYSTEM_PROMPT
        else:
            system_prompt = cls.STRICT_SYSTEM_PROMPT

        if custom_instructions:
            system_prompt += f"\nAdditional Instructions: {custom_instructions}"

        context_str = cls.format_context_blocks(search_results)

        user_content = (
            f"CONTEXT DOCUMENTS:\n"
            f"{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"GROUNDED ANSWER (with citations):"
        )

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_content,
            "context_count": len(search_results)
        }
