"""
Smart Cache Engine for semantic response caching.

Uses ChromaDB to find semantically similar past queries and returns
cached responses when similarity exceeds the threshold, reducing
token usage and costs.

This engine can be used by any entity that implements the CacheConfig protocol
(SmartEnricher, Alias, SmartRouter, Redirect).
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from context.chroma import CollectionWrapper, is_chroma_available, require_chroma

if TYPE_CHECKING:
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheConfig(Protocol):
    """
    Protocol for entities that support caching.

    Any model with these attributes can use the SmartCacheEngine:
    - SmartEnricher, Alias, SmartRouter, Redirect
    """

    # Identity
    name: str
    target_model: str

    # Cache toggle
    use_cache: bool

    # Cache behavior
    cache_similarity_threshold: float  # 0.0-1.0, higher = stricter matching
    cache_match_system_prompt: bool  # Include system prompt in cache key
    cache_match_last_message_only: bool  # Only match last user message
    cache_ttl_hours: int  # How long entries remain valid
    cache_min_tokens: int  # Don't cache very short responses
    cache_max_tokens: int  # Don't cache very long responses
    cache_collection: Optional[str]  # ChromaDB collection name

    # Stats
    cache_hits: int
    cache_tokens_saved: int
    cache_cost_saved: float

    # Tags (for tracking)
    @property
    def tags(self) -> list[str]: ...


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
    cached_tokens: int = 0  # Output tokens saved by cache hit
    cached_cost: float = 0.0  # Cost saved by cache hit (from original request)

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

    Can be used with any entity implementing the CacheConfig protocol.
    """

    def __init__(self, config: CacheConfig, registry: "ProviderRegistry"):
        """
        Initialize the cache engine.

        Args:
            config: Entity with cache configuration (SmartEnricher, Alias, etc.)
            registry: Provider registry for model resolution

        Raises:
            RuntimeError: If ChromaDB is not configured
        """
        require_chroma()
        self.config = config
        self.registry = registry
        self._collection: Optional[CollectionWrapper] = None

    @property
    def collection(self) -> CollectionWrapper:
        """Get the ChromaDB collection for this cache (lazy loaded)."""
        if self._collection is None:
            # Use stored collection name, or generate consistent one from entity name
            if self.config.cache_collection:
                collection_name = self.config.cache_collection
            else:
                # Normalize name for collection
                normalized = (
                    self.config.name.lower().replace("-", "_").replace(" ", "_")
                )
                collection_name = f"cache_{normalized}"
            logger.debug(
                f"Cache '{self.config.name}' using collection: {collection_name} "
                f"(stored cache_collection={self.config.cache_collection})"
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
                self.config.target_model
            )
            resolved = ResolvedModel(
                provider=target_resolved.provider,
                model_id=target_resolved.model_id,
                alias_name=self.config.name,
                alias_tags=self.config.tags,
            )
        except ValueError as e:
            logger.error(f"Cache '{self.config.name}' target model not available: {e}")
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
                cache_name=self.config.name,
                cache_tags=self.config.tags,
                is_cache_hit=False,
            )

        # Check if we have a match above threshold
        if results and results.get("ids") and results["ids"][0]:
            # ChromaDB returns L2 distance by default, convert to similarity
            # similarity = 1 / (1 + distance) for L2 distance
            distance = results["distances"][0][0]
            similarity = 1 / (1 + distance)

            logger.debug(
                f"Cache '{self.config.name}' best match: similarity={similarity:.4f}, "
                f"threshold={self.config.cache_similarity_threshold}"
            )

            if similarity >= self.config.cache_similarity_threshold:
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
                                cache_name=self.config.name,
                                cache_tags=self.config.tags,
                                is_cache_hit=False,
                                similarity_score=similarity,
                            )
                    except (ValueError, TypeError):
                        pass  # Invalid expiry, treat as valid

                # Cache hit!
                try:
                    cached_response = json.loads(metadata.get("response", "{}"))

                    # Check if cached response meets min token requirement
                    # (for entries cached before cache_min_tokens was set)
                    cached_tokens = metadata.get("output_tokens", 0)
                    if cached_tokens < self.config.cache_min_tokens:
                        logger.debug(
                            f"Cached response too short: {cached_tokens} < {self.config.cache_min_tokens}, "
                            "deleting stale entry"
                        )
                        # Delete the too-short entry
                        self.collection.delete(ids=[results["ids"][0][0]])
                        return CacheResult(
                            resolved=resolved,
                            cache_name=self.config.name,
                            cache_tags=self.config.tags,
                            is_cache_hit=False,
                            similarity_score=similarity,
                        )

                    # Get the cost from the original request
                    cached_cost = metadata.get("cost", 0.0)

                    logger.info(
                        f"Cache hit for '{self.config.name}' "
                        f"(similarity={similarity:.4f}, tokens={cached_tokens}, cost=${cached_cost:.4f})"
                    )
                    return CacheResult(
                        resolved=resolved,
                        cache_name=self.config.name,
                        cache_tags=self.config.tags,
                        is_cache_hit=True,
                        cached_response=cached_response,
                        cached_tokens=cached_tokens,
                        cached_cost=cached_cost,
                        similarity_score=similarity,
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to decode cached response")

            return CacheResult(
                resolved=resolved,
                cache_name=self.config.name,
                cache_tags=self.config.tags,
                is_cache_hit=False,
                similarity_score=similarity,
            )

        # No matches found
        return CacheResult(
            resolved=resolved,
            cache_name=self.config.name,
            cache_tags=self.config.tags,
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
        if output_tokens < self.config.cache_min_tokens:
            logger.debug(
                f"Response too short to cache: {output_tokens} < {self.config.cache_min_tokens}"
            )
            return False

        # Check if response is too long to cache
        if output_tokens > self.config.cache_max_tokens:
            logger.debug(
                f"Response too long to cache: {output_tokens} > {self.config.cache_max_tokens}"
            )
            return False

        # Build query text and ID
        query_text = self._build_query_text(messages, system)
        cache_id = self._generate_cache_id(query_text)

        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(hours=self.config.cache_ttl_hours)

        # Prepare metadata
        metadata = {
            "response": json.dumps(response),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": self.config.target_model,
        }

        try:
            self.collection.upsert(
                ids=[cache_id],
                documents=[query_text],
                metadatas=[metadata],
            )
            logger.debug(
                f"Stored response in cache '{self.config.name}' (id={cache_id})"
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
            logger.info(f"Cleared cache '{self.config.name}'")
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
                "name": self.config.name,
                "entries": count,
                "cache_hits": self.config.cache_hits,
                "tokens_saved": self.config.cache_tokens_saved,
                "cost_saved": self.config.cache_cost_saved,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "name": self.config.name,
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

        If cache_match_last_message_only is enabled, only uses the last user message
        (useful for OpenWebUI and other tools that inject varying context).
        """
        parts = []

        # Include system prompt if configured
        if self.config.cache_match_system_prompt and system:
            parts.append(f"[SYSTEM]: {system}")

        # If cache_match_last_message_only, only use the last user message
        if self.config.cache_match_last_message_only:
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
