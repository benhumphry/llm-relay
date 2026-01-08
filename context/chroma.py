"""
ChromaDB client wrapper for LLM Relay.

Provides a unified interface for all ChromaDB operations across
Smart Cache, Smart Augmentor, and Smart RAG features.
"""

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton client instance
_client = None


def get_chroma_url() -> str:
    """Get ChromaDB URL from environment."""
    return os.environ.get("CHROMA_URL", "http://localhost:8000")


def get_collection_prefix() -> str:
    """Get collection name prefix from environment."""
    return os.environ.get("CHROMA_COLLECTION_PREFIX", "llmrelay_")


def get_chroma_client():
    """
    Get or create ChromaDB client singleton.

    Returns:
        chromadb.HttpClient instance
    """
    global _client
    if _client is None:
        try:
            import chromadb

            url = get_chroma_url()
            # Parse URL to extract host and port
            url_clean = url.replace("http://", "").replace("https://", "")
            if ":" in url_clean:
                host, port_str = url_clean.split(":", 1)
                port = int(port_str.split("/")[0])
            else:
                host = url_clean.split("/")[0]
                port = 8000

            _client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to ChromaDB at {url}")
        except ImportError:
            logger.error("chromadb package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    return _client


def reset_client():
    """Reset the client singleton (useful for testing or reconnection)."""
    global _client
    _client = None


def is_chroma_available() -> bool:
    """
    Check if ChromaDB is configured and reachable.

    Returns:
        True if ChromaDB is available, False otherwise
    """
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True
    except Exception as e:
        logger.debug(f"ChromaDB not available: {e}")
        return False


def get_collection(name: str, create: bool = True):
    """
    Get a collection by name, optionally creating it.

    Args:
        name: Collection name (without prefix)
        create: If True, create collection if it doesn't exist

    Returns:
        ChromaDB Collection object
    """
    client = get_chroma_client()
    prefix = get_collection_prefix()
    full_name = f"{prefix}{name}"

    if create:
        return client.get_or_create_collection(name=full_name)
    return client.get_collection(name=full_name)


def delete_collection(name: str) -> bool:
    """
    Delete a collection by name.

    Args:
        name: Collection name (without prefix)

    Returns:
        True if deleted successfully, False otherwise
    """
    client = get_chroma_client()
    prefix = get_collection_prefix()
    full_name = f"{prefix}{name}"

    try:
        client.delete_collection(name=full_name)
        logger.info(f"Deleted ChromaDB collection: {full_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection {full_name}: {e}")
        return False


def list_collections() -> list[str]:
    """
    List all collections with our prefix.

    Returns:
        List of collection names (without prefix)
    """
    try:
        client = get_chroma_client()
        prefix = get_collection_prefix()
        all_collections = client.list_collections()
        return [
            c.name[len(prefix) :] for c in all_collections if c.name.startswith(prefix)
        ]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


class CollectionWrapper:
    """
    Wrapper for a ChromaDB collection with convenience methods.

    Usage:
        cache = CollectionWrapper("response_cache")
        cache.add(
            ids=["doc1"],
            documents=["What is the capital of France?"],
            metadatas=[{"answer": "Paris"}]
        )
        results = cache.query("capital of France", n_results=5)
    """

    def __init__(self, name: str):
        """
        Initialize collection wrapper.

        Args:
            name: Collection name (without prefix)
        """
        self.name = name
        self._collection = None

    @property
    def collection(self):
        """Get the underlying ChromaDB collection (lazy loaded)."""
        if self._collection is None:
            self._collection = get_collection(self.name)
        return self._collection

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ):
        """
        Add documents to the collection.

        Args:
            ids: Unique identifiers for documents
            documents: Document texts (used for embedding)
            metadatas: Optional metadata dicts for each document
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"Added {len(ids)} documents to collection {self.name}")

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ):
        """
        Add or update documents in the collection.

        Args:
            ids: Unique identifiers for documents
            documents: Document texts (used for embedding)
            metadatas: Optional metadata dicts for each document
        """
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(ids)} documents in collection {self.name}")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """
        Query for similar documents.

        Args:
            query_text: Text to search for
            n_results: Maximum number of results
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include: What to include in results (default: documents, metadatas, distances)

        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

    def get(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """
        Get documents by ID or filter.

        Args:
            ids: Document IDs to retrieve
            where: Metadata filter conditions
            include: What to include in results

        Returns:
            Dict with keys: ids, documents, metadatas
        """
        if include is None:
            include = ["documents", "metadatas"]

        return self.collection.get(ids=ids, where=where, include=include)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
    ):
        """
        Delete documents by ID or filter.

        Args:
            ids: Document IDs to delete
            where: Metadata filter conditions
        """
        self.collection.delete(ids=ids, where=where)
        logger.debug(f"Deleted documents from collection {self.name}")

    def count(self) -> int:
        """
        Get document count in collection.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def clear(self):
        """Delete all documents in the collection by recreating it."""
        delete_collection(self.name)
        self._collection = None  # Force recreation on next access
        # Recreate empty collection
        self._collection = get_collection(self.name)
        logger.info(f"Cleared collection {self.name}")

    def delete_expired(self, expires_field: str = "expires_at"):
        """
        Delete documents that have expired based on a metadata field.

        Args:
            expires_field: Metadata field containing expiry timestamp (ISO format)
        """
        now = datetime.utcnow().isoformat()
        try:
            self.collection.delete(where={expires_field: {"$lt": now}})
            logger.debug(f"Deleted expired documents from {self.name}")
        except Exception as e:
            # ChromaDB may not support $lt on strings in all versions
            logger.warning(f"Could not delete expired documents: {e}")
