"""FAISS vector store wrapper with persistence, cosine similarity, and MMR search."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a retrieved document chunk with its similarity score."""
    chunk: DocumentChunk
    score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "score": round(float(self.score), 4),
            "metadata": self.chunk.metadata,
        }


class FAISSVectorStore:
    """FAISS-based vector index supporting Inner Product (Cosine Similarity) and MMR."""

    def __init__(self, dimension: int, index_type: str = "IP"):
        self.dimension = dimension
        self.index_type = index_type.upper()
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_map: Dict[str, DocumentChunk] = {}
        self.raw_embeddings: List[np.ndarray] = []

        if not _FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")

        self._init_index()

    def _init_index(self):
        if self.index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}. Use 'IP' or 'L2'.")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Adds a list of chunks and their corresponding embedding vectors to the index."""
        if len(chunks) == 0:
            return 0

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Embeddings count ({embeddings.shape[0]}) does not match chunks count ({len(chunks)})")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})")

        # Cast to float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.chunk_id_map[chunk.chunk_id] = chunk
        self.raw_embeddings.append(vectors)

        logger.info("Added %d chunks to FAISS index. Total entries: %d", len(chunks), self.index.ntotal)
        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Performs standard similarity search."""
        if self.index.ntotal == 0:
            return []

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue

            score = float(dist)
            # For L2, convert distance to similarity score: sim = 1 / (1 + dist)
            if self.index_type == "L2":
                score = 1.0 / (1.0 + score)

            # Filter by threshold if specified
            if similarity_threshold is not None and score < similarity_threshold:
                continue

            results.append(SearchResult(chunk=self.chunks[idx], score=score))

        return results

    def search_mmr(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Maximal Marginal Relevance (MMR) search for diverse, non-redundant retrieval.

        Balances relevance to query with diversity against already selected items.
        """
        if self.index.ntotal == 0:
            return []

        fetch_k = min(fetch_k, self.index.ntotal)
        top_k = min(top_k, fetch_k)

        # Initial candidates retrieval
        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(q_vec, fetch_k)

        cand_indices = [int(i) for i in indices[0] if i != -1 and i < len(self.chunks)]
        if not cand_indices:
            return []

        all_vectors = np.vstack(self.raw_embeddings) if self.raw_embeddings else None
        if all_vectors is None or len(all_vectors) != self.index.ntotal:
            # Fall back to standard search if raw vectors missing
            return self.search(query_vector, top_k=top_k, similarity_threshold=similarity_threshold)

        cand_vectors = all_vectors[cand_indices]
        query_sims = np.dot(cand_vectors, query_vector)

        # Greedy selection
        selected_indices = [0]
        selected_results = [cand_indices[0]]

        while len(selected_results) < top_k and len(selected_indices) < len(cand_indices):
            remaining_indices = [i for i in range(len(cand_indices)) if i not in selected_indices]
            best_mmr_score = -float("inf")
            best_idx = None

            for i in remaining_indices:
                sim_to_query = query_sims[i]
                # Max similarity to already selected candidates
                sim_to_selected = max(
                    np.dot(cand_vectors[i], cand_vectors[s]) for s in selected_indices
                )
                mmr_score = lambda_mult * sim_to_query - (1.0 - lambda_mult) * sim_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_results.append(cand_indices[best_idx])
            else:
                break

        results: List[SearchResult] = []
        for idx in selected_results:
            chunk = self.chunks[idx]
            vec = all_vectors[idx]
            sim = float(np.dot(vec, query_vector))
            if similarity_threshold is not None and sim < similarity_threshold:
                continue
            results.append(SearchResult(chunk=chunk, score=sim))

        return results

    def save(self, directory_path: Union[str, Path]):
        """Persists FAISS index binary and metadata JSON to disk."""
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)

        index_file = path / "index.faiss"
        metadata_file = path / "chunks.json"
        config_file = path / "config.json"

        faiss.write_index(self.index, str(index_file))

        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "token_count": c.token_count,
                "char_count": c.char_count,
            }
            for c in self.chunks
        ]
        metadata_file.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

        config_data = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
        }
        config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        logger.info("Saved FAISS index with %d vectors to %s", self.index.ntotal, directory_path)

    @classmethod
    def load(cls, directory_path: Union[str, Path]) -> "FAISSVectorStore":
        """Loads FAISS index and metadata from disk."""
        path = Path(directory_path)
        index_file = path / "index.faiss"
        metadata_file = path / "chunks.json"
        config_file = path / "config.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"FAISS index files not found in {directory_path}")

        config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        dimension = config.get("dimension", 384)
        index_type = config.get("index_type", "IP")

        store = cls(dimension=dimension, index_type=index_type)
        store.index = faiss.read_index(str(index_file))

        chunks_raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        store.chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                metadata=c["metadata"],
                token_count=c.get("token_count", 0),
                char_count=c.get("char_count", len(c["text"]))
            )
            for c in chunks_raw
        ]
        store.chunk_id_map = {c.chunk_id: c for c in store.chunks}

        # Reconstruct vectors if supported by the index
        try:
            vectors = np.empty((store.index.ntotal, store.dimension), dtype=np.float32)
            for i in range(store.index.ntotal):
                vectors[i] = store.index.reconstruct(i)
            store.raw_embeddings = [vectors]
        except Exception:
            logger.debug("Index does not support direct vector reconstruction; MMR will use fallback.")

        logger.info("Loaded FAISS index with %d vectors from %s", store.index.ntotal, directory_path)
        return store

    def clear(self):
        """Clears all vectors and chunks from index."""
        self.chunks.clear()
        self.chunk_id_map.clear()
        self.raw_embeddings.clear()
        self._init_index()

    def get_stats(self) -> dict:
        """Returns statistics on current index state."""
        sources = set(c.metadata.get("source", "unknown") for c in self.chunks)
        total_tokens = sum(c.token_count for c in self.chunks)
        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(sources),
            "sources": list(sources),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_tokens_indexed": total_tokens,
            "avg_chunk_tokens": round(total_tokens / len(self.chunks), 1) if self.chunks else 0,
        }

