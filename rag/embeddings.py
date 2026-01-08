"""
Embedding provider abstraction for Smart RAG.

Supports multiple embedding backends:
- local: Bundled sentence-transformers model (BAAI/bge-small-en-v1.5)
- ollama: Ollama embeddings (granite3.2-vision, nomic-embed-text, etc.)
- openai: OpenAI embeddings (text-embedding-3-small)
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation with usage tracking."""

    embeddings: list[list[float]]
    usage: dict = field(default_factory=dict)  # {"prompt_tokens": N, "total_tokens": N}
    model: str = ""
    provider: str = ""


# Available embedding providers
_PROVIDERS: dict[str, type["EmbeddingProvider"]] = {}


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    name: str = "base"
    description: str = "Base embedding provider"
    requires_api_key: bool = False

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and usage info
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single query text.

        Some providers optimize differently for queries vs documents.

        Args:
            text: Query text to embed

        Returns:
            EmbeddingResult with single embedding and usage info
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and configured."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Uses BAAI/bge-small-en-v1.5 by default (~130MB, 384 dimensions).
    """

    name = "local"
    description = "Local embeddings (sentence-transformers)"
    requires_api_key = False

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = 384  # Default for bge-small

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading local embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for documents."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        # Estimate tokens (rough approximation: ~4 chars per token)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            usage={"prompt_tokens": estimated_tokens, "total_tokens": estimated_tokens},
            model=self.model_name,
            provider=self.name,
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding for a query."""
        model = self._get_model()
        # BGE models use a query prefix for better retrieval
        if "bge" in self.model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = model.encode(text, convert_to_numpy=True)
        estimated_tokens = len(text) // 4
        return EmbeddingResult(
            embeddings=[embedding.tolist()],
            usage={"prompt_tokens": estimated_tokens, "total_tokens": estimated_tokens},
            model=self.model_name,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider.

    Supports models like:
    - nomic-embed-text
    - mxbai-embed-large
    - granite3.2-vision (multimodal)
    - all-minilm
    """

    name = "ollama"
    description = "Ollama embeddings"
    requires_api_key = False

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_URL = "http://localhost:11434"

    # Known embedding dimensions for common models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "granite3.2-vision": 1024,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", self.DEFAULT_URL)
        self._dimension = self.MODEL_DIMENSIONS.get(
            self.model_name.split(":")[0], 768
        )  # Default 768

    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Call Ollama API to get embeddings. Returns (embeddings, total_tokens)."""
        import httpx

        embeddings = []
        total_tokens = 0
        for text in texts:
            response = httpx.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
            # Ollama may return prompt_eval_count for tokens
            total_tokens += data.get("prompt_eval_count", len(text) // 4)

        return embeddings, total_tokens

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for documents."""
        embeddings, total_tokens = self._embed_batch(texts)
        return EmbeddingResult(
            embeddings=embeddings,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            model=self.model_name,
            provider=self.name,
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding for a query."""
        embeddings, total_tokens = self._embed_batch([text])
        return EmbeddingResult(
            embeddings=embeddings,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            model=self.model_name,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            import httpx

            response = httpx.get(f"{self.ollama_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                return False

            # Check if our model is in the list
            data = response.json()
            model_names = [m["name"] for m in data.get("models", [])]
            # Check both with and without tag
            base_model = self.model_name.split(":")[0]
            return any(
                m == self.model_name or m.startswith(f"{base_model}:")
                for m in model_names
            )
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.

    Uses text-embedding-3-small by default.
    """

    name = "openai"
    description = "OpenAI embeddings"
    requires_api_key = True

    DEFAULT_MODEL = "text-embedding-3-small"

    # Known dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self._dimension = self.MODEL_DIMENSIONS.get(self.model_name, 1536)

    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        """Call OpenAI API to get embeddings. Returns (embeddings, usage)."""
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in sorted_data]

        # OpenAI returns actual usage
        usage = data.get("usage", {})
        return embeddings, usage

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for documents."""
        embeddings, usage = self._embed_batch(texts)
        return EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            model=self.model_name,
            provider=self.name,
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding for a query."""
        embeddings, usage = self._embed_batch([text])
        return EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            model=self.model_name,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


# Register providers
_PROVIDERS = {
    "local": LocalEmbeddingProvider,
    "ollama": OllamaEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(
    provider_name: str,
    model_name: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Get an embedding provider by name.

    Args:
        provider_name: Provider name ("local", "ollama", "openai")
        model_name: Optional model name override
        ollama_url: Optional Ollama URL override (for ollama provider)

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )

    provider_class = _PROVIDERS[provider_name]

    if provider_name == "ollama":
        return provider_class(model_name=model_name, ollama_url=ollama_url)
    elif provider_name == "local":
        return provider_class(model_name=model_name)
    elif provider_name == "openai":
        return provider_class(model_name=model_name)
    else:
        return provider_class()


def list_embedding_providers() -> list[dict]:
    """
    List available embedding providers with their status.

    Returns:
        List of dicts with provider info and availability status
    """
    result = []
    for name, provider_class in _PROVIDERS.items():
        provider = provider_class()
        result.append(
            {
                "name": name,
                "description": provider.description,
                "requires_api_key": provider.requires_api_key,
                "available": provider.is_available(),
            }
        )
    return result
