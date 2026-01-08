"""
Smart Cache Engine for semantic response caching.

Uses ChromaDB to find semantically similar past queries and returns
cached responses when similarity exceeds the threshold, reducing
token usage and costs.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from context.chroma import CollectionWrapper, is_chroma_available, require_chroma

if TYPE_CHECKING:
    from db.models import SmartCache
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


@dataclass
class CacheResult:
    """Result of a cache lookup or generation."""

    # The resolved target model (for cache miss)
    resolved: "ResolvedModel"
    cache_name: str
    cache_tags: list[str]

    # Cache hit information
    is_cache_hit: bool = False
    cached_response: dict | None = None  # Full response dict on cache hit

    # Cache miss - caller should forward to model
    # After getting response, call store_response() to cache it

    # Stats for tracking
    similarity_score: float | None = None  # Similarity of best match (if any)


class SmartCacheEngine:
    """
    Engine for semantic response caching using ChromaDB.

    The engine:
    1. Builds a cache key from query content (optionally including system prompt)
    2. Searches ChromaDB for semantically similar cached queries
    3. Returns cached response if similarity exceeds threshold
    4. On cache miss, caller forwards to target model and stores result
    """

    def __init__(self, cache: "SmartCache", registry: "ProviderRegistry"):
        """
        Initialize the cache engine.

        Args:
            cache: SmartCache configuration
            registry: Provider registry for model resolution

        Raises:
            RuntimeError: If ChromaDB is not configured
        """
        require_chroma()
        self.cache = cache
        self.registry = registry
        self._collection: Optional[CollectionWrapper] = None

    @property
    def collection(self) -> CollectionWrapper:
        """Get the ChromaDB collection for this cache (lazy loaded)."""
        if self._collection is None:
            # Use stored collection name, or generate consistent one from cache name
            if self.cache.chroma_collection:
                collection_name = self.cache.chroma_collection
            else:
                # Normalize name same way as create_smart_cache does
                normalized = self.cache.name.lower().replace("-", "_").replace(" ", "_")
                collection_name = f"cache_{normalized}"
            logger.debug(
                f"Cache '{self.cache.name}' using collection: {collection_name} "
                f"(stored chroma_collection={self.cache.chroma_collection})"
            )
            self._collection = CollectionWrapper(collection_name)
        return self._collection

    def lookup(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> CacheResult:
        """
        Look up a query in the cache.

        Args:
            messages: List of message dicts
            system: Optional system prompt

        Returns:
            CacheResult with cache hit information or resolved target model
        """
        from providers.registry import ResolvedModel

        # Build the query text for embedding
        query_text = self._build_query_text(messages, system)

        # Resolve the target model (needed regardless of cache hit)
        try:
            target_resolved = self.registry._resolve_actual_model(
                self.cache.target_model
            )
            resolved = ResolvedModel(
                provider=target_resolved.provider,
                model_id=target_resolved.model_id,
                alias_name=self.cache.name,
                alias_tags=self.cache.tags,
            )
        except ValueError as e:
            logger.error(f"Cache '{self.cache.name}' target model not available: {e}")
            raise

        # Search for similar cached queries
        try:
            results = self.collection.query(
                query_text=query_text,
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return CacheResult(
                resolved=resolved,
                cache_name=self.cache.name,
                cache_tags=self.cache.tags,
                is_cache_hit=False,
            )

        # Check if we have a match above threshold
        if results and results.get("ids") and results["ids"][0]:
            # ChromaDB returns L2 distance by default, convert to similarity
            # similarity = 1 / (1 + distance) for L2 distance
            distance = results["distances"][0][0]
            similarity = 1 / (1 + distance)

            logger.debug(
                f"Cache '{self.cache.name}' best match: similarity={similarity:.4f}, "
                f"threshold={self.cache.similarity_threshold}"
            )

            if similarity >= self.cache.similarity_threshold:
                # Check if cache entry is expired
                metadata = results["metadatas"][0][0]
                expires_at = metadata.get("expires_at")

                if expires_at:
                    try:
                        expiry = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expiry:
                            logger.debug(f"Cache entry expired at {expires_at}")
                            # Delete expired entry
                            self.collection.delete(ids=[results["ids"][0][0]])
                            return CacheResult(
                                resolved=resolved,
                                cache_name=self.cache.name,
                                cache_tags=self.cache.tags,
                                is_cache_hit=False,
                                similarity_score=similarity,
                            )
                    except (ValueError, TypeError):
                        pass  # Invalid expiry, treat as valid

                # Cache hit!
                try:
                    cached_response = json.loads(metadata.get("response", "{}"))

                    # Check if cached response meets min token requirement
                    # (for entries cached before min_cached_tokens was set)
                    cached_tokens = metadata.get("output_tokens", 0)
                    if cached_tokens < self.cache.min_cached_tokens:
                        logger.debug(
                            f"Cached response too short: {cached_tokens} < {self.cache.min_cached_tokens}, "
                            "deleting stale entry"
                        )
                        # Delete the too-short entry
                        self.collection.delete(ids=[results["ids"][0][0]])
                        return CacheResult(
                            resolved=resolved,
                            cache_name=self.cache.name,
                            cache_tags=self.cache.tags,
                            is_cache_hit=False,
                            similarity_score=similarity,
                        )

                    logger.info(
                        f"Cache hit for '{self.cache.name}' "
                        f"(similarity={similarity:.4f})"
                    )
                    return CacheResult(
                        resolved=resolved,
                        cache_name=self.cache.name,
                        cache_tags=self.cache.tags,
                        is_cache_hit=True,
                        cached_response=cached_response,
                        similarity_score=similarity,
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to decode cached response")

            return CacheResult(
                resolved=resolved,
                cache_name=self.cache.name,
                cache_tags=self.cache.tags,
                is_cache_hit=False,
                similarity_score=similarity,
            )

        # No matches found
        return CacheResult(
            resolved=resolved,
            cache_name=self.cache.name,
            cache_tags=self.cache.tags,
            is_cache_hit=False,
        )

    def store_response(
        self,
        messages: list[dict],
        system: str | None,
        response: dict,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> bool:
        """
        Store a response in the cache.

        Args:
            messages: Original message list
            system: Original system prompt
            response: Response dict to cache (will be JSON serialized)
            input_tokens: Input tokens used (for stats)
            output_tokens: Output tokens used (for stats)
            cost: Cost incurred (for stats)

        Returns:
            True if stored successfully, False otherwise
        """
        # Check if response is too short to cache (filters titles, etc.)
        if output_tokens < self.cache.min_cached_tokens:
            logger.debug(
                f"Response too short to cache: {output_tokens} < {self.cache.min_cached_tokens}"
            )
            return False

        # Check if response is too long to cache
        if output_tokens > self.cache.max_cached_tokens:
            logger.debug(
                f"Response too long to cache: {output_tokens} > {self.cache.max_cached_tokens}"
            )
            return False

        # Build query text and ID
        query_text = self._build_query_text(messages, system)
        cache_id = self._generate_cache_id(query_text)

        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(hours=self.cache.cache_ttl_hours)

        # Prepare metadata
        metadata = {
            "response": json.dumps(response),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": self.cache.target_model,
        }

        try:
            self.collection.upsert(
                ids=[cache_id],
                documents=[query_text],
                metadatas=[metadata],
            )
            logger.debug(
                f"Stored response in cache '{self.cache.name}' (id={cache_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store in cache: {e}")
            return False

    def clear_cache(self) -> bool:
        """
        Clear all entries from this cache's collection.

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self.collection.clear()
            logger.info(f"Cleared cache '{self.cache.name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        try:
            count = self.collection.count()
            return {
                "name": self.cache.name,
                "entries": count,
                "total_requests": self.cache.total_requests,
                "cache_hits": self.cache.cache_hits,
                "hit_rate": self.cache.hit_rate,
                "tokens_saved": self.cache.tokens_saved,
                "cost_saved": self.cache.cost_saved,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "name": self.cache.name,
                "error": str(e),
            }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        try:
            self.collection.delete_expired("expires_at")
            # ChromaDB doesn't return count, so we can't know how many were deleted
            return 0
        except Exception as e:
            logger.warning(f"Failed to cleanup expired entries: {e}")
            return 0

    def _build_query_text(self, messages: list[dict], system: str | None) -> str:
        """
        Build the query text for embedding.

        Combines system prompt (if configured) and user messages into a
        single text string for semantic similarity matching.

        If match_last_message_only is enabled, only uses the last user message
        (useful for OpenWebUI and other tools that inject varying context).
        """
        parts = []

        # Include system prompt if configured
        if self.cache.match_system_prompt and system:
            parts.append(f"[SYSTEM]: {system}")

        # If match_last_message_only, only use the last user message
        if self.cache.match_last_message_only:
            # Find last user message
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break

            if last_user_msg:
                content = last_user_msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        parts.append(" ".join(text_parts))
        else:
            # Include all message content
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    parts.append(f"[{role.upper()}]: {content}")
                elif isinstance(content, list):
                    # Content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        parts.append(f"[{role.upper()}]: {' '.join(text_parts)}")

        return "\n".join(parts)

    def _generate_cache_id(self, query_text: str) -> str:
        """Generate a unique cache ID from query text."""
        return hashlib.sha256(query_text.encode()).hexdigest()[:32]


def build_cached_response(
    cached_data: dict,
    cache_name: str,
) -> dict:
    """
    Build a response dict from cached data.

    This reconstructs a response that looks like it came from the LLM,
    with additional metadata indicating it was served from cache.

    Args:
        cached_data: The cached response data
        cache_name: Name of the cache that served this response

    Returns:
        Response dict suitable for returning to the client
    """
    # The cached_data should already be a valid response dict
    # Just add cache metadata
    response = cached_data.copy()

    # Add cache indicator (non-standard but useful for debugging)
    if "x_cache" not in response:
        response["x_cache"] = {
            "hit": True,
            "cache_name": cache_name,
        }

    return response
