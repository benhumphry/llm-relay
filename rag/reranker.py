"""
Cross-encoder re-ranking for improved retrieval quality.

Uses a cross-encoder model to re-rank results after initial bi-encoder retrieval,
improving relevance by considering query-document pairs together.

Supports:
- Local: sentence-transformers CrossEncoder (default, ~50-270MB models)
- Jina: Jina Reranker API (requires JINA_API_KEY)
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""

    documents: list[dict]  # Documents with 'rerank_score' added
    model_used: str
    provider: str


class RerankerProvider(ABC):
    """Abstract base class for reranking providers."""

    name: str = "base"
    description: str = "Base reranker"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        content_key: str = "content",
    ) -> list[dict]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: User query
            documents: List of dicts with content
            top_k: Number of results to return
            content_key: Key containing document text

        Returns:
            Top-k documents sorted by rerank_score
        """
        pass


class LocalRerankerProvider(RerankerProvider):
    """
    Local reranker using sentence-transformers CrossEncoder.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default (~80MB).
    Models are lazy-loaded and cached.
    """

    name = "local"
    description = "Local reranking (sentence-transformers CrossEncoder)"

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cache for loaded models
    _model_cache: dict = {}

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self.model_name not in LocalRerankerProvider._model_cache:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder model: {self.model_name}")
                model = CrossEncoder(self.model_name)
                LocalRerankerProvider._model_cache[self.model_name] = model
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return LocalRerankerProvider._model_cache[self.model_name]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        content_key: str = "content",
    ) -> list[dict]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return []

        if not query or not query.strip():
            logger.warning(
                "Empty query provided for reranking, returning original order"
            )
            return documents[:top_k]

        try:
            model = self._get_model()

            # Create query-document pairs
            pairs = [(query, doc.get(content_key, "")) for doc in documents]

            # Get cross-encoder scores
            scores = model.predict(pairs)

            # Attach scores to documents (create copies to avoid mutating originals)
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                scored_docs.append(doc_copy)

            # Sort by rerank score (highest first)
            sorted_docs = sorted(
                scored_docs, key=lambda x: x.get("rerank_score", 0), reverse=True
            )

            logger.debug(f"Re-ranked {len(documents)} documents, returning top {top_k}")
            return sorted_docs[:top_k]

        except Exception as e:
            logger.warning(f"Re-ranking failed, returning original order: {e}")
            return documents[:top_k]


class JinaRerankerProvider(RerankerProvider):
    """
    Jina Reranker API provider.

    Uses Jina's reranking API endpoint. Requires JINA_API_KEY environment variable.
    See: https://jina.ai/reranker/
    """

    name = "jina"
    description = "Jina Reranker API"

    JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
    DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("JINA_API_KEY")

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        content_key: str = "content",
    ) -> list[dict]:
        """Re-rank documents using Jina Reranker API."""
        if not documents:
            return []

        if not self.api_key:
            logger.warning("JINA_API_KEY not set, falling back to original order")
            return documents[:top_k]

        if not query or not query.strip():
            logger.warning(
                "Empty query provided for reranking, returning original order"
            )
            return documents[:top_k]

        try:
            import httpx

            # Extract document texts
            doc_texts = [doc.get(content_key, "") for doc in documents]

            response = httpx.post(
                self.JINA_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": doc_texts,
                    "top_n": top_k,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            # Jina returns results with 'index' and 'relevance_score'
            results = data.get("results", [])

            # Build scored documents using original indices
            scored_docs = []
            for result in results:
                idx = result.get("index", 0)
                score = result.get("relevance_score", 0.0)
                if idx < len(documents):
                    doc_copy = documents[idx].copy()
                    doc_copy["rerank_score"] = float(score)
                    scored_docs.append(doc_copy)

            logger.debug(
                f"Jina re-ranked {len(documents)} documents, returning top {len(scored_docs)}"
            )
            return scored_docs

        except Exception as e:
            logger.warning(f"Jina reranking failed, returning original order: {e}")
            return documents[:top_k]


def get_reranker_provider(
    provider_type: str = "local",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RerankerProvider:
    """
    Get a reranker provider.

    Args:
        provider_type: "local" or "jina"
        model_name: Model to use for reranking
        api_key: For "jina" provider, the API key (or use JINA_API_KEY env var)

    Returns:
        Configured RerankerProvider instance

    Raises:
        ValueError: If provider_type is unknown
    """
    if provider_type == "local":
        return LocalRerankerProvider(model_name=model_name)

    elif provider_type == "jina":
        return JinaRerankerProvider(model_name=model_name, api_key=api_key)

    else:
        raise ValueError(f"Unknown reranker provider: {provider_type}")


# Convenience functions for backwards compatibility and simple usage


def rerank_documents(
    query: str,
    documents: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
    content_key: str = "content",
    provider_type: str = "local",
) -> list[dict]:
    """
    Re-rank documents using specified provider.

    Args:
        query: User query
        documents: List of dicts with content key
        model_name: Model name for reranking
        top_k: Number of results to return
        content_key: Key in document dict containing text content
        provider_type: "local" or "jina"

    Returns:
        Top-k documents sorted by rerank score, with 'rerank_score' added
    """
    provider = get_reranker_provider(provider_type=provider_type, model_name=model_name)
    return provider.rerank(query, documents, top_k, content_key)


def rerank_urls(
    query: str,
    urls_with_context: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 3,
    provider_type: str = "local",
) -> list[str]:
    """
    Re-rank URLs based on title+snippet relevance.

    Args:
        query: User query
        urls_with_context: List of dicts with 'url', 'title', 'snippet' keys
        model_name: Model name for reranking
        top_k: Number of URLs to return
        provider_type: "local" or "jina"

    Returns:
        Top-k URLs sorted by relevance
    """
    if not urls_with_context:
        return []

    # Create content from title + snippet for ranking
    docs = [
        {
            "content": f"{u.get('title', '')} {u.get('snippet', '')}".strip(),
            "url": u.get("url", ""),
        }
        for u in urls_with_context
        if u.get("url")
    ]

    if not docs:
        return []

    ranked = rerank_documents(
        query, docs, model_name, top_k, provider_type=provider_type
    )
    return [d["url"] for d in ranked if d.get("url")]


def get_global_rerank_provider() -> str:
    """
    Get the globally configured rerank provider from Settings.

    Returns:
        Provider name ("local" or "jina"), defaults to "local"
    """
    try:
        from db import Setting, get_db_context

        with get_db_context() as db:
            setting = (
                db.query(Setting)
                .filter(Setting.key == Setting.KEY_WEB_RERANK_PROVIDER)
                .first()
            )
            if setting and setting.value:
                return setting.value
    except Exception as e:
        logger.warning(f"Failed to get rerank provider setting: {e}")
    return "local"  # Default


# Available reranker providers (for UI dropdown)
RERANKER_PROVIDERS = [
    {
        "id": "local",
        "name": "Local (sentence-transformers)",
        "description": "Run cross-encoder locally (~50-270MB models)",
    },
    {
        "id": "jina",
        "name": "Jina Reranker API",
        "description": "Use Jina's cloud reranking service (requires JINA_API_KEY)",
    },
]

# Available local reranker models (for UI dropdown when provider=local)
LOCAL_RERANKER_MODELS = [
    {
        "id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "name": "MiniLM-L6 (Fast, ~50MB)",
        "description": "Fast and lightweight, good quality",
    },
    {
        "id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "name": "MiniLM-L12 (Balanced, ~130MB)",
        "description": "Better quality, slightly slower",
    },
    {
        "id": "BAAI/bge-reranker-base",
        "name": "BGE Reranker Base (Best, ~270MB)",
        "description": "Highest quality, slower",
    },
]

# Available Jina reranker models (for UI dropdown when provider=jina)
JINA_RERANKER_MODELS = [
    {
        "id": "jina-reranker-v2-base-multilingual",
        "name": "Jina Reranker v2 Base (Multilingual)",
        "description": "Supports 100+ languages",
    },
    {
        "id": "jina-reranker-v1-base-en",
        "name": "Jina Reranker v1 Base (English)",
        "description": "English only, faster",
    },
]
