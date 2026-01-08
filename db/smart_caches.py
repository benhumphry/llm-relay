"""
Smart Cache CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart caches.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import SmartCache

logger = logging.getLogger(__name__)


def get_all_smart_caches(db: Optional[Session] = None) -> list[SmartCache]:
    """Get all smart caches from the database."""
    if db:
        return db.query(SmartCache).order_by(SmartCache.name).all()

    with get_db_context() as session:
        caches = session.query(SmartCache).order_by(SmartCache.name).all()
        return [_cache_to_detached(c) for c in caches]


def get_smart_cache_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartCache]:
    """Get a smart cache by its name."""
    if db:
        return db.query(SmartCache).filter(SmartCache.name == name).first()

    with get_db_context() as session:
        cache = session.query(SmartCache).filter(SmartCache.name == name).first()
        return _cache_to_detached(cache) if cache else None


def get_smart_cache_by_id(
    cache_id: int, db: Optional[Session] = None
) -> Optional[SmartCache]:
    """Get a smart cache by its ID."""
    if db:
        return db.query(SmartCache).filter(SmartCache.id == cache_id).first()

    with get_db_context() as session:
        cache = session.query(SmartCache).filter(SmartCache.id == cache_id).first()
        return _cache_to_detached(cache) if cache else None


def create_smart_cache(
    name: str,
    target_model: str,
    similarity_threshold: float = 0.95,
    match_system_prompt: bool = True,
    match_last_message_only: bool = False,
    cache_ttl_hours: int = 168,
    min_cached_tokens: int = 50,
    max_cached_tokens: int = 4000,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> SmartCache:
    """
    Create a new smart cache.

    Args:
        name: Unique name for the cache
        target_model: Model to use on cache miss ("provider/model")
        similarity_threshold: Similarity threshold for cache hits (0.0-1.0)
        match_system_prompt: Whether to include system prompt in cache matching
        match_last_message_only: Only match last user message (ignores history)
        cache_ttl_hours: Cache entry time-to-live in hours
        min_cached_tokens: Minimum tokens in response to cache (filters short responses)
        max_cached_tokens: Maximum tokens in response to cache
        tags: Optional list of tags to assign to requests
        description: Optional description
        enabled: Whether the cache is enabled (default True)
        db: Optional database session

    Returns:
        The created SmartCache object

    Raises:
        ValueError: If a cache with the same name already exists
    """

    def _create(session: Session) -> SmartCache:
        # Check for existing cache with same name
        existing = session.query(SmartCache).filter(SmartCache.name == name).first()
        if existing:
            raise ValueError(f"Smart cache with name '{name}' already exists")

        # Generate collection name
        collection_name = f"cache_{name.lower().replace('-', '_').replace(' ', '_')}"

        cache = SmartCache(
            name=name,
            target_model=target_model,
            similarity_threshold=similarity_threshold,
            match_system_prompt=match_system_prompt,
            match_last_message_only=match_last_message_only,
            cache_ttl_hours=cache_ttl_hours,
            min_cached_tokens=min_cached_tokens,
            max_cached_tokens=max_cached_tokens,
            chroma_collection=collection_name,
            description=description,
            enabled=enabled,
        )
        if tags:
            cache.tags = tags

        session.add(cache)
        session.flush()  # Get the ID
        logger.info(f"Created smart cache: {name}")
        return cache

    if db:
        return _create(db)

    with get_db_context() as session:
        cache = _create(session)
        session.refresh(cache)
        return _cache_to_detached(cache)


def update_smart_cache(
    cache_id: int,
    name: str | None = None,
    target_model: str | None = None,
    similarity_threshold: float | None = None,
    match_system_prompt: bool | None = None,
    match_last_message_only: bool | None = None,
    cache_ttl_hours: int | None = None,
    min_cached_tokens: int | None = None,
    max_cached_tokens: int | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartCache]:
    """
    Update an existing smart cache.

    Args:
        cache_id: ID of the cache to update
        name: New name (optional)
        target_model: New target model (optional)
        similarity_threshold: New similarity threshold (optional)
        match_system_prompt: New match_system_prompt setting (optional)
        match_last_message_only: New match_last_message_only setting (optional)
        cache_ttl_hours: New cache TTL (optional)
        min_cached_tokens: New min cached tokens (optional)
        max_cached_tokens: New max cached tokens (optional)
        tags: New tags list (optional, pass empty list to clear)
        description: New description (optional)
        enabled: New enabled state (optional)
        db: Optional database session

    Returns:
        Updated SmartCache object or None if not found

    Raises:
        ValueError: If new name conflicts with existing cache
    """

    def _update(session: Session) -> Optional[SmartCache]:
        cache = session.query(SmartCache).filter(SmartCache.id == cache_id).first()
        if not cache:
            return None

        # Check name uniqueness if changing
        if name and name != cache.name:
            existing = session.query(SmartCache).filter(SmartCache.name == name).first()
            if existing:
                raise ValueError(f"Smart cache with name '{name}' already exists")
            cache.name = name
            # Update collection name too
            cache.chroma_collection = (
                f"cache_{name.lower().replace('-', '_').replace(' ', '_')}"
            )

        if target_model is not None:
            cache.target_model = target_model

        if similarity_threshold is not None:
            cache.similarity_threshold = similarity_threshold

        if match_system_prompt is not None:
            cache.match_system_prompt = match_system_prompt

        if match_last_message_only is not None:
            cache.match_last_message_only = match_last_message_only

        if cache_ttl_hours is not None:
            cache.cache_ttl_hours = cache_ttl_hours

        if min_cached_tokens is not None:
            cache.min_cached_tokens = min_cached_tokens

        if max_cached_tokens is not None:
            cache.max_cached_tokens = max_cached_tokens

        if tags is not None:
            cache.tags = tags

        if description is not None:
            cache.description = description

        if enabled is not None:
            cache.enabled = enabled

        session.flush()
        logger.info(f"Updated smart cache: {cache.name}")
        return cache

    if db:
        return _update(db)

    with get_db_context() as session:
        cache = _update(session)
        if cache:
            session.refresh(cache)
            return _cache_to_detached(cache)
        return None


def update_smart_cache_stats(
    cache_id: int,
    increment_requests: int = 0,
    increment_hits: int = 0,
    increment_tokens_saved: int = 0,
    increment_cost_saved: float = 0.0,
    db: Optional[Session] = None,
) -> bool:
    """
    Update cache statistics (atomically increment counters).

    Args:
        cache_id: ID of the cache to update
        increment_requests: Number of requests to add
        increment_hits: Number of cache hits to add
        increment_tokens_saved: Number of tokens saved to add
        increment_cost_saved: Cost saved to add

    Returns:
        True if updated, False if cache not found
    """

    def _update_stats(session: Session) -> bool:
        cache = session.query(SmartCache).filter(SmartCache.id == cache_id).first()
        if not cache:
            return False

        cache.total_requests += increment_requests
        cache.cache_hits += increment_hits
        cache.tokens_saved += increment_tokens_saved
        cache.cost_saved += increment_cost_saved

        session.flush()
        return True

    if db:
        return _update_stats(db)

    with get_db_context() as session:
        return _update_stats(session)


def reset_smart_cache_stats(cache_id: int, db: Optional[Session] = None) -> bool:
    """
    Reset cache statistics to zero.

    Returns:
        True if reset, False if cache not found
    """

    def _reset(session: Session) -> bool:
        cache = session.query(SmartCache).filter(SmartCache.id == cache_id).first()
        if not cache:
            return False

        cache.total_requests = 0
        cache.cache_hits = 0
        cache.tokens_saved = 0
        cache.cost_saved = 0.0

        session.flush()
        logger.info(f"Reset stats for smart cache: {cache.name}")
        return True

    if db:
        return _reset(db)

    with get_db_context() as session:
        return _reset(session)


def delete_smart_cache(cache_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart cache by ID.

    Note: This does NOT delete the ChromaDB collection. Use clear_smart_cache_data()
    first if you want to remove the cached data.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        cache = session.query(SmartCache).filter(SmartCache.id == cache_id).first()
        if not cache:
            return False

        name = cache.name
        session.delete(cache)
        logger.info(f"Deleted smart cache: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def cache_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if a cache name is available.

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)
        db: Optional database session

    Returns:
        True if name is available, False if taken
    """

    def _check(session: Session) -> bool:
        query = session.query(SmartCache).filter(SmartCache.name == name)
        if exclude_id:
            query = query.filter(SmartCache.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_smart_caches(db: Optional[Session] = None) -> dict[str, SmartCache]:
    """
    Get all enabled smart caches as a dict keyed by name.

    Useful for quick lookups during request processing.

    Returns:
        Dict mapping cache name to SmartCache object
    """

    def _get(session: Session) -> list[SmartCache]:
        return (
            session.query(SmartCache)
            .filter(SmartCache.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        caches = _get(db)
        return {c.name: c for c in caches}

    with get_db_context() as session:
        caches = _get(session)
        return {c.name: _cache_to_detached(c) for c in caches}


def _cache_to_detached(cache: SmartCache) -> SmartCache:
    """
    Create a detached copy of a smart cache with all data loaded.

    This allows the cache to be used after the session is closed.
    """
    detached = SmartCache(
        name=cache.name,
        target_model=cache.target_model,
        similarity_threshold=cache.similarity_threshold,
        match_system_prompt=cache.match_system_prompt,
        match_last_message_only=cache.match_last_message_only,
        cache_ttl_hours=cache.cache_ttl_hours,
        min_cached_tokens=cache.min_cached_tokens,
        max_cached_tokens=cache.max_cached_tokens,
        chroma_collection=cache.chroma_collection,
        total_requests=cache.total_requests,
        cache_hits=cache.cache_hits,
        tokens_saved=cache.tokens_saved,
        cost_saved=cache.cost_saved,
        tags_json=cache.tags_json,
        description=cache.description,
        enabled=cache.enabled,
    )
    detached.id = cache.id
    detached.created_at = cache.created_at
    detached.updated_at = cache.updated_at
    return detached
