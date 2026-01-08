"""
RAG (Retrieval-Augmented Generation) module for LLM Relay.

Provides document indexing and retrieval using Docling for document parsing
and ChromaDB for vector storage.
"""

from .embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
    list_embedding_providers,
)
from .indexer import (
    RAGIndexer,
    get_indexer,
    start_indexer,
    stop_indexer,
)
from .retriever import (
    RAGRetriever,
    RetrievalResult,
    RetrievedChunk,
    get_retriever,
)

__all__ = [
    # Embeddings
    "EmbeddingProvider",
    "get_embedding_provider",
    "list_embedding_providers",
    # Indexer
    "RAGIndexer",
    "get_indexer",
    "start_indexer",
    "stop_indexer",
    # Retriever
    "RAGRetriever",
    "RetrievalResult",
    "RetrievedChunk",
    "get_retriever",
]
