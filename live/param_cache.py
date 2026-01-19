"""
Parameter Cache for Live Data Sources.

Uses ChromaDB for semantic caching of extracted parameters like:
- Location names -> API location keys (AccuWeather)
- Company names -> Stock symbols (Finnhub)
- Station names -> Station codes (TransportAPI)

This enables semantic matching so "weather in the British capital" can
match a cached entry for "London".
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Collection name for parameter cache
PARAM_CACHE_COLLECTION = "live_data_params"


def get_param_cache_collection():
    """Get or create the parameter cache ChromaDB collection."""
    try:
        from context.chroma import get_chroma_client

        client = get_chroma_client()
        if not client:
            return None

        return client.get_or_create_collection(
            name=PARAM_CACHE_COLLECTION,
            metadata={"description": "Live data parameter cache"},
        )
    except Exception as e:
        logger.warning(f"Failed to get param cache collection: {e}")
        return None


def cache_parameter(
    source_type: str,
    param_type: str,
    query_text: str,
    resolved_value: str,
    metadata: dict | None = None,
) -> bool:
    """
    Cache a resolved parameter for future semantic matching.

    Args:
        source_type: e.g., "weather", "stocks", "transport"
        param_type: e.g., "location", "symbol", "station"
        query_text: The original query text (e.g., "London", "Apple stock")
        resolved_value: The resolved API value (e.g., "328328", "AAPL", "KGX")
        metadata: Optional additional metadata

    Returns:
        True if cached successfully
    """
    collection = get_param_cache_collection()
    if not collection:
        return False

    try:
        doc_id = f"{source_type}:{param_type}:{query_text.lower()}"
        doc_metadata = {
            "source_type": source_type,
            "param_type": param_type,
            "resolved_value": resolved_value,
            "cached_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            doc_metadata.update(metadata)

        # Upsert the document
        collection.upsert(
            ids=[doc_id],
            documents=[query_text],
            metadatas=[doc_metadata],
        )
        logger.debug(f"Cached param: {doc_id} -> {resolved_value}")
        return True

    except Exception as e:
        logger.warning(f"Failed to cache parameter: {e}")
        return False


def lookup_parameter(
    source_type: str,
    param_type: str,
    query_text: str,
    similarity_threshold: float = 0.85,
) -> Optional[str]:
    """
    Look up a cached parameter using semantic matching.

    Args:
        source_type: e.g., "weather", "stocks", "transport"
        param_type: e.g., "location", "symbol", "station"
        query_text: The query text to match
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        The resolved value if found, None otherwise
    """
    collection = get_param_cache_collection()
    if not collection:
        return None

    try:
        # Query with semantic search
        results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and": [
                    {"source_type": {"$eq": source_type}},
                    {"param_type": {"$eq": param_type}},
                ]
            },
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return None

        # Check similarity - ChromaDB returns distances, lower is better
        # Convert distance to similarity (assuming L2 distance)
        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            # Rough conversion: similarity = 1 / (1 + distance)
            similarity = 1 / (1 + distance)

            if similarity >= similarity_threshold:
                metadata = results["metadatas"][0][0]
                resolved = metadata.get("resolved_value")
                logger.debug(
                    f"Cache hit for {source_type}:{param_type}:{query_text} "
                    f"-> {resolved} (similarity: {similarity:.2f})"
                )
                return resolved

        return None

    except Exception as e:
        logger.warning(f"Failed to lookup parameter: {e}")
        return None


def clear_param_cache(source_type: str | None = None) -> int:
    """
    Clear the parameter cache.

    Args:
        source_type: If provided, only clear entries for this source type.
                    If None, clear all entries.

    Returns:
        Number of entries cleared
    """
    collection = get_param_cache_collection()
    if not collection:
        return 0

    try:
        if source_type:
            # Get IDs matching the source type
            results = collection.get(
                where={"source_type": {"$eq": source_type}},
            )
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
        else:
            # Clear all - delete and recreate collection
            from context.chroma import get_chroma_client

            client = get_chroma_client()
            if client:
                client.delete_collection(PARAM_CACHE_COLLECTION)
                return -1  # Unknown count

        return 0

    except Exception as e:
        logger.warning(f"Failed to clear param cache: {e}")
        return 0
