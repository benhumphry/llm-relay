"""
RAG Retriever for Smart RAG.

Handles semantic search and context formatting for query augmentation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)

logger = logging.getLogger(__name__)


def _get_embedding_provider_for_rag(
    embedding_provider: str,
    embedding_model: Optional[str],
    ollama_url: Optional[str],
) -> EmbeddingProvider:
    """
    Get an embedding provider based on RAG configuration.

    Args:
        embedding_provider: "local", "ollama:<instance>", or provider name
        embedding_model: Model name for embeddings
        ollama_url: Ollama URL (for ollama providers)

    Returns:
        Configured EmbeddingProvider
    """
    if embedding_provider == "local":
        return LocalEmbeddingProvider(model_name=embedding_model)

    if embedding_provider.startswith("ollama:") or embedding_provider == "ollama":
        # Extract instance name if present
        if ":" in embedding_provider:
            instance_name = embedding_provider.split(":", 1)[1]
        else:
            instance_name = "ollama"

        if not ollama_url:
            raise ValueError(f"Ollama URL required for {embedding_provider}")

        return OllamaEmbeddingProvider(
            model_name=embedding_model,
            ollama_url=ollama_url,
            instance_name=instance_name,
        )

    # It's a provider name - look up from registry
    from providers import registry

    provider = registry.get_provider(embedding_provider)
    if not provider:
        raise ValueError(f"Provider not found: {embedding_provider}")

    if not provider.is_configured():
        raise ValueError(f"Provider not configured: {embedding_provider}")

    # Get base URL and API key from provider
    base_url = getattr(provider, "base_url", None)
    api_key = getattr(provider, "api_key", None)

    if not base_url:
        # Some providers have different URL attribute names
        base_url = getattr(provider, "url", None)

    if not api_key:
        raise ValueError(f"Provider {embedding_provider} has no API key configured")

    if not embedding_model:
        raise ValueError(f"Embedding model required for provider {embedding_provider}")

    return OpenAICompatibleEmbeddingProvider(
        provider_name=embedding_provider,
        base_url=base_url,
        api_key=api_key,
        model_name=embedding_model,
    )


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""

    content: str
    source_file: str
    chunk_index: int
    score: float  # Similarity score (0-1, higher is more similar)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    chunks: list[RetrievedChunk]
    query: str
    total_tokens: int  # Estimated token count of all chunks
    embedding_usage: dict | None = None  # Usage from embedding call
    embedding_model: str | None = None  # Model used for embedding
    embedding_provider: str | None = None  # Provider used for embedding


class RAGRetriever:
    """
    Retrieves relevant document chunks from ChromaDB.
    """

    def __init__(self):
        self._chroma_client = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            chroma_url = os.environ.get("CHROMA_URL")
            if not chroma_url:
                raise ValueError("CHROMA_URL not configured")

            import chromadb

            # Parse URL
            host = (
                chroma_url.replace("http://", "").replace("https://", "").split(":")[0]
            )
            port_str = (
                chroma_url.split(":")[-1]
                if ":" in chroma_url.split("/")[-1]
                else "8000"
            )
            port = int(port_str.split("/")[0])

            self._chroma_client = chromadb.HttpClient(host=host, port=port)

        return self._chroma_client

    def retrieve(
        self,
        collection_name: str,
        query: str,
        embedding_provider: str,
        embedding_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
        rerank_top_n: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Always reranks results using a cross-encoder for better relevance.

        Args:
            collection_name: ChromaDB collection to search
            query: User query text
            embedding_provider: Provider for query embedding
            embedding_model: Model name for embedding
            ollama_url: Ollama URL override
            max_results: Maximum chunks to return
            similarity_threshold: Minimum similarity score (0-1)
            rerank_provider: Reranking provider ("local" or "jina")
            rerank_model: Model for reranking
            rerank_top_n: Fetch this many from ChromaDB before reranking

        Returns:
            RetrievalResult with relevant chunks
        """
        # Apply defaults for reranking (always on by default)
        # Use global setting if not specified
        if rerank_provider is None:
            from rag.reranker import get_global_rerank_provider

            rerank_provider = get_global_rerank_provider()
        if rerank_model is None:
            rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if rerank_top_n is None:
            rerank_top_n = 20

        from .embeddings import get_embedding_provider

        try:
            client = self._get_client()
            collection = client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            return RetrievalResult(chunks=[], query=query, total_tokens=0)

        # Get embedding provider and embed query
        embedding_usage = None
        embed_model = None
        embed_provider = None
        try:
            provider = _get_embedding_provider_for_rag(
                embedding_provider,
                embedding_model,
                ollama_url,
            )
            embed_result = provider.embed_query(query)
            query_embedding = embed_result.embeddings[0]
            embedding_usage = embed_result.usage
            embed_model = embed_result.model
            embed_provider = embed_result.provider
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return RetrievalResult(chunks=[], query=query, total_tokens=0)

        # Query ChromaDB - fetch more results for reranking
        fetch_count = rerank_top_n
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_count,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return RetrievalResult(chunks=[], query=query, total_tokens=0)

        # Process results
        chunks = []
        total_tokens = 0

        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(documents)
            )
            distances = (
                results["distances"][0]
                if results["distances"]
                else [0.0] * len(documents)
            )

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Convert distance to similarity (ChromaDB uses L2 by default, cosine returns 0-2)
                # For cosine: similarity = 1 - (distance / 2)
                similarity = 1 - (dist / 2) if dist <= 2 else 0

                if similarity >= similarity_threshold:
                    chunk = RetrievedChunk(
                        content=doc,
                        source_file=meta.get("source_file", "unknown"),
                        chunk_index=meta.get("chunk_index", 0),
                        score=similarity,
                    )
                    chunks.append(chunk)

        # Always rerank for better relevance
        if chunks:
            try:
                from .reranker import rerank_documents

                # Convert chunks to dicts for reranking
                chunk_dicts = [
                    {
                        "content": c.content,
                        "source_file": c.source_file,
                        "chunk_index": c.chunk_index,
                        "original_score": c.score,
                    }
                    for c in chunks
                ]

                # Rerank
                reranked = rerank_documents(
                    query=query,
                    documents=chunk_dicts,
                    model_name=rerank_model,
                    top_k=max_results,
                    provider_type=rerank_provider,
                )

                # Convert back to RetrievedChunk, using rerank_score
                chunks = [
                    RetrievedChunk(
                        content=d["content"],
                        source_file=d["source_file"],
                        chunk_index=d["chunk_index"],
                        score=d.get("rerank_score", d.get("original_score", 0)),
                    )
                    for d in reranked
                ]

                logger.debug(
                    f"Reranked {len(chunk_dicts)} chunks to {len(chunks)} using {rerank_provider}/{rerank_model}"
                )
            except Exception as e:
                logger.warning(f"Reranking failed, using original order: {e}")
                # Fall back to original chunks, limited to max_results
                chunks = chunks[:max_results]

        # Calculate total tokens
        for chunk in chunks:
            # Rough token estimate (4 chars per token)
            total_tokens += len(chunk.content) // 4

        logger.debug(
            f"Retrieved {len(chunks)} chunks for query (threshold={similarity_threshold})"
        )

        return RetrievalResult(
            chunks=chunks,
            query=query,
            total_tokens=total_tokens,
            embedding_usage=embedding_usage,
            embedding_model=embed_model,
            embedding_provider=embed_provider,
        )

    def format_context(
        self,
        result: RetrievalResult,
        max_tokens: int = 4000,
        include_sources: bool = True,
    ) -> str:
        """
        Format retrieved chunks into context for injection.

        Args:
            result: RetrievalResult from retrieve()
            max_tokens: Maximum tokens for context
            include_sources: Whether to include source file info

        Returns:
            Formatted context string
        """
        if not result.chunks:
            return ""

        # Sort by relevance (highest score first)
        sorted_chunks = sorted(result.chunks, key=lambda c: c.score, reverse=True)

        # Build context within token limit
        context_parts = []
        current_tokens = 0
        header = "## Relevant Document Context\n\n"
        header_tokens = len(header) // 4
        current_tokens += header_tokens

        for i, chunk in enumerate(sorted_chunks):
            # Estimate tokens for this chunk
            chunk_tokens = len(chunk.content) // 4

            if include_sources:
                source_line = f"[Source: {chunk.source_file}]\n"
                chunk_tokens += len(source_line) // 4
            else:
                source_line = ""

            # Check if we'd exceed limit
            if current_tokens + chunk_tokens > max_tokens:
                # Try to fit partial chunk
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Worth including partial
                    remaining_chars = remaining_tokens * 4
                    truncated = chunk.content[:remaining_chars] + "..."
                    if include_sources:
                        context_parts.append(f"{source_line}{truncated}\n")
                    else:
                        context_parts.append(f"{truncated}\n")
                break

            if include_sources:
                context_parts.append(f"{source_line}{chunk.content}\n")
            else:
                context_parts.append(f"{chunk.content}\n")

            current_tokens += chunk_tokens

        if not context_parts:
            return ""

        return header + "\n---\n\n".join(context_parts)


# Global retriever instance
_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get or create the global RAG retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
