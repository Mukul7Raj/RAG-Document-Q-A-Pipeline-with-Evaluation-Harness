"""Embedding model abstraction supporting Hugging Face, OpenAI, and Mock providers."""
import abc
import hashlib
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(abc.ABC):
    """Abstract base class for text embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts into a 2D float32 numpy array (N x D)."""
        pass

    @abc.abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 numpy array (D)."""
        pass

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes vectors to unit L2 norm so inner product equals cosine similarity."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """Hugging Face Sentence-Transformers embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dim = 384  # Default for all-MiniLM-L6-v2

        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading HuggingFace model '%s' on %s...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded model successfully. Embedding dimension: %d", self._dim)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer ('%s'): %s. Using MockEmbeddingModel fallback.", self.model_name, e)
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            return embeddings.astype(np.float32)

        # Fallback to mock embedding
        fallback = MockEmbeddingModel(dim=self.dimension)
        return fallback.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if self._model is not None:
            emb = self._model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            return emb.astype(np.float32)

        fallback = MockEmbeddingModel(dim=self.dimension)
        return fallback.embed_query(query)


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, fast mock embedding model for testing and offline environments."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> np.ndarray:
        # Create deterministic pseudo-random vector from SHA-256 hash seed
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)

        # If keywords match, boost certain dimensions for lexical-semantic correlation
        words = set(text.lower().split())
        for i, w in enumerate(sorted(words)):
            w_idx = hash(w) % self._dim
            vec[w_idx] += 2.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vecs = [self._hash_to_vector(t) for t in texts]
        return np.vstack(vecs).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._hash_to_vector(query).astype(np.float32)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI Embeddings integration."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self._dim = 1536
        if "3-small" in model_name:
            self._dim = 1536
        elif "3-large" in model_name:
            self._dim = 3072

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [item.embedding for item in response.data]
        arr = np.array(embeddings, dtype=np.float32)
        return self._normalize(arr)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def get_embedding_model(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    normalize: bool = True,
    api_key: Optional[str] = None
) -> BaseEmbeddingModel:
    """Factory function for instantiating embedding models."""
    prov = provider.lower()
    if prov in {"huggingface", "hf"}:
        return HuggingFaceEmbeddingModel(model_name=model_name, device=device, normalize=normalize)
    elif prov == "openai":
        return OpenAIEmbeddingModel(model_name=model_name, api_key=api_key)
    elif prov == "mock":
        return MockEmbeddingModel()
    else:
        logger.warning("Unknown embedding provider '%s', defaulting to HuggingFace", provider)
        return HuggingFaceEmbeddingModel(model_name=model_name, device=device, normalize=normalize)

