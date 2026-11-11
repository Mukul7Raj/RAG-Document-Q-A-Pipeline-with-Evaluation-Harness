"""Document chunking strategies with metadata preservation."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

from src.ingestion.loader import LoadedDocument


@dataclass
class DocumentChunk:
    """Represents an atomic text chunk produced by the chunker."""
    chunk_id: str
    text: str
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    token_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)
        if self.token_count == 0:
            self.token_count = self._estimate_tokens(self.text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if _TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback: rule of thumb 1 token ≈ 4 characters
        return max(1, len(text.split()))


class DocumentChunker:
    """Configurable text chunker supporting multiple splitting strategies."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive"  # recursive, token, character, sentence
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy.lower()

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """Splits a list of LoadedDocument instances into chunks."""
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """Splits a single document into chunks."""
        text = doc.text.strip()
        if not text:
            return []

        if self.strategy == "token":
            raw_splits = self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)
        elif self.strategy == "character":
            raw_splits = self._split_by_characters(text, self.chunk_size, self.chunk_overlap)
        elif self.strategy == "sentence":
            raw_splits = self._split_by_sentences(text, self.chunk_size, self.chunk_overlap)
        else:  # default to recursive
            raw_splits = self._split_recursive(
                text,
                self.chunk_size,
                self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
            )

        chunks: List[DocumentChunk] = []
        total_chunks = len(raw_splits)

        for i, chunk_text in enumerate(raw_splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{doc.doc_id}#chunk_{i:04d}"
            meta = dict(doc.metadata)
            meta.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "doc_id": doc.doc_id,
                "chunk_strategy": self.strategy,
                "chunk_size_config": self.chunk_size,
                "chunk_overlap_config": self.chunk_overlap,
            })

            token_count = DocumentChunk._estimate_tokens(chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=meta,
                    token_count=token_count,
                    char_count=len(chunk_text)
                )
            )

        return chunks

    def _split_recursive(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str]
    ) -> List[str]:
        """Hierarchical recursive character text splitting."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for s in splits:
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator, chunk_size, chunk_overlap)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # Can't split further, split by character
                    final_chunks.extend(self._split_by_characters(s, chunk_size, chunk_overlap))
                else:
                    sub_chunks = self._split_recursive(s, chunk_size, chunk_overlap, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator, chunk_size, chunk_overlap)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Merges sub-splits until chunk_size is reached, maintaining overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            sep_len = len(separator) if current_doc else 0
            if total + len(s) + sep_len > chunk_size and current_doc:
                merged = separator.join(current_doc)
                docs.append(merged)

                # Keep overlap pieces
                overlap_len = 0
                new_current = []
                for item in reversed(current_doc):
                    if overlap_len + len(item) + len(separator) <= chunk_overlap:
                        new_current.insert(0, item)
                        overlap_len += len(item) + len(separator)
                    else:
                        break
                current_doc = new_current
                total = sum(len(x) for x in current_doc) + (len(separator) * max(0, len(current_doc) - 1))

            current_doc.append(s)
            total += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))

        return docs

    def _split_by_characters(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fixed character window splitting with step size."""
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    def _split_by_tokens(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text strictly by token count using tiktoken."""
        if not _TIKTOKEN_AVAILABLE:
            # Fallback to word splitting
            words = text.split()
            step = max(1, chunk_size - chunk_overlap)
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        step = max(1, chunk_size - chunk_overlap)
        chunks = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(enc.decode(chunk_tokens))
        return chunks

    def _split_by_sentences(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text on sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return self._merge_splits(sentences, " ", chunk_size, chunk_overlap)

