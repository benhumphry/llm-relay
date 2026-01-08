"""
Context module for LLM Relay.

Provides ChromaDB integration and utilities for Smart Cache,
Smart Augmentor, and Smart RAG features.

Usage:
    from context import is_chroma_available, CollectionWrapper

    # Check if ChromaDB is available
    if is_chroma_available():
        # Use a collection
        cache = CollectionWrapper("response_cache")
        cache.add(
            ids=["doc1"],
            documents=["What is the capital of France?"],
            metadatas=[{"answer": "Paris"}]
        )
        results = cache.query("capital of France", n_results=5)
"""

from .builder import (
    estimate_tokens,
    format_as_context_message,
    format_cached_response,
    format_context_footer,
    format_context_header,
    format_document_chunks,
    format_scraped_content,
    format_search_results,
    inject_context_to_system,
    merge_contexts,
    truncate_to_tokens,
)
from .chroma import (
    CollectionWrapper,
    delete_collection,
    get_chroma_client,
    get_chroma_url,
    get_collection,
    get_collection_prefix,
    is_chroma_available,
    list_collections,
    reset_client,
)

__all__ = [
    # ChromaDB client
    "get_chroma_client",
    "get_chroma_url",
    "get_collection",
    "get_collection_prefix",
    "delete_collection",
    "list_collections",
    "is_chroma_available",
    "reset_client",
    "CollectionWrapper",
    # Context builder utilities
    "estimate_tokens",
    "truncate_to_tokens",
    "format_context_header",
    "format_context_footer",
    "inject_context_to_system",
    "format_as_context_message",
    "format_search_results",
    "format_scraped_content",
    "format_document_chunks",
    "format_cached_response",
    "merge_contexts",
]
