"""
Embedding provider abstraction for Smart RAG.

Supports multiple embedding backends:
- local: Bundled sentence-transformers model (BAAI/bge-small-en-v1.5)
- ollama: Any Ollama instance via /api/embeddings
- provider: Any configured LLM provider via OpenAI-compatible /v1/embeddings
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Global lock to prevent concurrent model loading on GPU
_model_load_lock = threading.Lock()


@dataclass
class EmbeddingResult:
    """Result of an embedding operation with usage tracking."""

    embeddings: list[list[float]]
    usage: dict = field(default_factory=dict)  # {"prompt_tokens": N, "total_tokens": N}
    model: str = ""
    provider: str = ""


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    name: str = "base"
    description: str = "Base embedding provider"

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

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = 384  # Default for bge-small

    def _get_model(self):
        """Lazy-load the embedding model with thread-safe locking."""
        if self._model is None:
            with _model_load_lock:
                # Double-check after acquiring lock
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
        # Lock during GPU inference to prevent concurrent access issues
        with _model_load_lock:
            embeddings = model.encode(texts, convert_to_numpy=True)
        # Estimate tokens (rough approximation: ~4 chars per token)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4

        # Clear CUDA cache after batch to prevent memory buildup
        self._clear_cuda_cache()

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
        # Lock during GPU inference to prevent concurrent access issues
        with _model_load_lock:
            embedding = model.encode(text, convert_to_numpy=True)
        estimated_tokens = len(text) // 4
        return EmbeddingResult(
            embeddings=[embedding.tolist()],
            usage={"prompt_tokens": estimated_tokens, "total_tokens": estimated_tokens},
            model=self.model_name,
            provider=self.name,
        )

    def _clear_cuda_cache(self):
        """Clear CUDA memory cache if available."""
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to clear CUDA cache: {e}")

    def unload(self):
        """Unload the model and free GPU memory."""
        if self._model is not None:
            logger.info(f"Unloading local embedding model: {self.model_name}")
            del self._model
            self._model = None
            self._clear_cuda_cache()
            # Force garbage collection to free memory
            import gc

            gc.collect()

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
    Ollama embedding provider via /api/embeddings endpoint.

    Supports any model available on the Ollama instance.
    """

    name = "ollama"
    description = "Ollama embeddings"

    DEFAULT_MODEL = "nomic-embed-text"

    # Known embedding dimensions for common models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        instance_name: Optional[str] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or "http://localhost:11434"
        self.instance_name = instance_name or "ollama"
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
            provider=f"ollama:{self.instance_name}",
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding for a query."""
        embeddings, total_tokens = self._embed_batch([text])
        return EmbeddingResult(
            embeddings=embeddings,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            model=self.model_name,
            provider=f"ollama:{self.instance_name}",
        )

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            import httpx

            response = httpx.get(f"{self.ollama_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider.

    Works with any provider that supports the /v1/embeddings endpoint:
    - OpenAI
    - OpenRouter
    - Together
    - Fireworks
    - Any other OpenAI-compatible API
    """

    name = "openai-compatible"
    description = "OpenAI-compatible embeddings"

    # Known dimensions for common embedding models
    MODEL_DIMENSIONS = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # Together
        "togethercomputer/m2-bert-80M-8k-retrieval": 768,
        "togethercomputer/m2-bert-80M-32k-retrieval": 768,
        # Voyage (via OpenRouter)
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
    }

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model_name: str,
    ):
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)

    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        """Call OpenAI-compatible API to get embeddings."""
        import httpx

        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
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

        # Get usage if available
        usage = data.get("usage", {})
        return embeddings, usage

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for documents."""
        embeddings, usage = self._embed_batch(texts)
        return EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            model=self.model_name,
            provider=self.provider_name,
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding for a query."""
        embeddings, usage = self._embed_batch([text])
        return EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            model=self.model_name,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


def get_embedding_provider(
    provider_type: str,
    model_name: Optional[str] = None,
    provider_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Get an embedding provider.

    Args:
        provider_type: "local", "ollama", or "provider"
        model_name: Model to use for embeddings
        provider_name: For "provider" type, the provider name (e.g., "openai", "openrouter")
        base_url: For "provider" type, the API base URL
        api_key: For "provider" type, the API key
        ollama_url: For "ollama" type, the Ollama instance URL

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ValueError: If required parameters are missing
    """
    if provider_type == "local":
        return LocalEmbeddingProvider(model_name=model_name)

    elif provider_type == "ollama":
        if not ollama_url:
            raise ValueError("ollama_url is required for Ollama embeddings")
        return OllamaEmbeddingProvider(
            model_name=model_name,
            ollama_url=ollama_url,
            instance_name=provider_name,
        )

    elif provider_type == "provider":
        if not provider_name:
            raise ValueError("provider_name is required for provider embeddings")
        if not base_url:
            raise ValueError("base_url is required for provider embeddings")
        if not api_key:
            raise ValueError("api_key is required for provider embeddings")
        if not model_name:
            raise ValueError("model_name is required for provider embeddings")

        return OpenAICompatibleEmbeddingProvider(
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )

    else:
        raise ValueError(
            f"Unknown embedding provider type: {provider_type}. "
            f"Available: local, ollama, provider"
        )
