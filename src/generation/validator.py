"""Output validation and hallucination detection guardrails."""
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from src.vectorstore.faiss_store import SearchResult


@dataclass
class ValidationResult:
    """Detailed output validation and hallucination diagnostics."""
    is_valid: bool
    hallucination_score: float  # 0.0 = completely grounded, 1.0 = completely ungrounded
    groundedness_score: float   # 1.0 = completely grounded, 0.0 = completely ungrounded
    is_refusal: bool
    cited_doc_indices: List[int] = field(default_factory=list)
    valid_citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    grounded_sentences: List[str] = field(default_factory=list)
    unsupported_sentences: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "hallucination_score": round(self.hallucination_score, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "is_refusal": self.is_refusal,
            "cited_doc_indices": self.cited_doc_indices,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "unsupported_sentences_count": len(self.unsupported_sentences),
            "grounded_sentences_count": len(self.grounded_sentences),
            "warning_messages": self.warning_messages,
        }


class OutputValidator:
    """Validates generated answers for source citations, refusal correctness, and hallucination."""

    REFUSAL_PHRASES = [
        "do not have sufficient information",
        "not enough information",
        "cannot find this information",
        "insufficient context",
        "not mentioned in the provided",
        "context does not contain",
        "provided context does not state",
        "no information provided",
    ]

    CITATION_PATTERN = re.compile(r'\[Doc\s*([0-9,\s]+)\]', re.IGNORECASE)

    def __init__(
        self,
        hallucination_threshold: float = 0.5,
        min_overlap_ratio: float = 0.35,
        enforce_citations: bool = True
    ):
        self.hallucination_threshold = hallucination_threshold
        self.min_overlap_ratio = min_overlap_ratio
        self.enforce_citations = enforce_citations

    def validate(
        self,
        answer: str,
        retrieved_results: List[SearchResult]
    ) -> ValidationResult:
        """Runs comprehensive validation on generated answer against retrieved context."""
        cleaned_answer = answer.strip()
        warnings: List[str] = []

        if not cleaned_answer:
            return ValidationResult(
                is_valid=False,
                hallucination_score=1.0,
                groundedness_score=0.0,
                is_refusal=False,
                warning_messages=["Generated answer was empty."]
            )

        # 1. Check for valid refusal
        is_refusal = self._check_is_refusal(cleaned_answer)
        if is_refusal:
            return ValidationResult(
                is_valid=True,
                hallucination_score=0.0,
                groundedness_score=1.0,
                is_refusal=True,
                warning_messages=[]
            )

        # 2. Extract citations and check validity
        num_docs = len(retrieved_results)
        cited_indices, valid_cites, invalid_cites = self._parse_citations(cleaned_answer, num_docs)

        if invalid_cites:
            warnings.append(
                f"Hallucinated citations detected: {invalid_cites} (retrieved documents: 1..{num_docs})"
            )

        if self.enforce_citations and not cited_indices and num_docs > 0:
            warnings.append("Answer lacks required source citations ([Doc X]).")

        # 3. Sentence-level grounding evaluation
        sentences = self._split_into_sentences(cleaned_answer)
        context_corpus = " ".join([res.chunk.text for res in retrieved_results])
        context_tokens = self._tokenize(context_corpus)

        grounded_sentences: List[str] = []
        unsupported_sentences: List[str] = []

        for s in sentences:
            s_tokens = self._tokenize(s)
            # Ignore very short sentence fragments
            if len(s_tokens) < 4:
                continue

            overlap_ratio = self._calculate_overlap(s_tokens, context_tokens)
            if overlap_ratio >= self.min_overlap_ratio:
                grounded_sentences.append(s)
            else:
                unsupported_sentences.append(s)

        total_evaluated = len(grounded_sentences) + len(unsupported_sentences)
        if total_evaluated > 0:
            hallucination_score = len(unsupported_sentences) / total_evaluated
        else:
            hallucination_score = 0.0

        groundedness_score = 1.0 - hallucination_score

        if hallucination_score > self.hallucination_threshold:
            warnings.append(
                f"High hallucination rate: {hallucination_score:.1%} of sentences lack context grounding."
            )

        is_valid = (
            len(invalid_cites) == 0
            and hallucination_score <= self.hallucination_threshold
            and (not self.enforce_citations or len(cited_indices) > 0 or num_docs == 0)
        )

        return ValidationResult(
            is_valid=is_valid,
            hallucination_score=hallucination_score,
            groundedness_score=groundedness_score,
            is_refusal=False,
            cited_doc_indices=cited_indices,
            valid_citations=valid_cites,
            invalid_citations=invalid_cites,
            grounded_sentences=grounded_sentences,
            unsupported_sentences=unsupported_sentences,
            warning_messages=warnings
        )

    def _check_is_refusal(self, answer: str) -> bool:
        lower = answer.lower()
        return any(phrase in lower for phrase in self.REFUSAL_PHRASES)

    def _parse_citations(self, text: str, max_valid_index: int) -> Tuple[List[int], List[int], List[int]]:
        matches = self.CITATION_PATTERN.findall(text)
        all_cited: List[int] = []

        for m in matches:
            parts = re.split(r'[,;\s]+', m.strip())
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    if idx not in all_cited:
                        all_cited.append(idx)

        valid = [i for i in all_cited if 1 <= i <= max_valid_index]
        invalid = [i for i in all_cited if i < 1 or i > max_valid_index]
        return all_cited, valid, invalid

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        # Clean markdown headers and bullet points
        cleaned = re.sub(r'#+\s*', '', text)
        cleaned = re.sub(r'^\s*[-*]\s*', '', cleaned, flags=re.MULTILINE)
        raw = re.split(r'(?<=[.!?])\s+', cleaned)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        # Lowercase, strip punctuation and stop words
        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "is", "are", "was", "were", "this", "that", "it", "by", "from",
            "as", "be", "has", "have", "had", "doc", "page"
        }
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = set(text.split())
        return tokens - stopwords

    @staticmethod
    def _calculate_overlap(s_tokens: Set[str], context_tokens: Set[str]) -> float:
        if not s_tokens:
            return 1.0
        intersection = s_tokens.intersection(context_tokens)
        return len(intersection) / len(s_tokens)

