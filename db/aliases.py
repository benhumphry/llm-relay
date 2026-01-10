"""
Alias CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete model aliases.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import Alias

logger = logging.getLogger(__name__)


def get_all_aliases(db: Optional[Session] = None) -> list[Alias]:
    """Get all aliases from the database."""
    if db:
        return db.query(Alias).order_by(Alias.name).all()

    with get_db_context() as session:
        aliases = session.query(Alias).order_by(Alias.name).all()
        return [_alias_to_detached(a) for a in aliases]


def get_alias_by_name(name: str, db: Optional[Session] = None) -> Optional[Alias]:
    """Get an alias by its name."""
    if db:
        return db.query(Alias).filter(Alias.name == name).first()

    with get_db_context() as session:
        alias = session.query(Alias).filter(Alias.name == name).first()
        return _alias_to_detached(alias) if alias else None


def get_alias_by_id(alias_id: int, db: Optional[Session] = None) -> Optional[Alias]:
    """Get an alias by its ID."""
    if db:
        return db.query(Alias).filter(Alias.id == alias_id).first()

    with get_db_context() as session:
        alias = session.query(Alias).filter(Alias.id == alias_id).first()
        return _alias_to_detached(alias) if alias else None


def create_alias(
    name: str,
    target_model: str,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool = True,
    # Caching
    use_cache: bool = False,
    cache_similarity_threshold: float = 0.95,
    cache_match_system_prompt: bool = True,
    cache_match_last_message_only: bool = False,
    cache_ttl_hours: int = 168,
    cache_min_tokens: int = 50,
    cache_max_tokens: int = 4000,
    cache_collection: str | None = None,
    db: Optional[Session] = None,
) -> Alias:
    """
    Create a new alias.

    Args:
        name: Unique name for the alias
        target_model: Target model in "provider_id/model_id" format
        tags: Optional list of tags to assign to requests using this alias
        description: Optional description
        enabled: Whether the alias is enabled (default True)
        db: Optional database session

    Returns:
        The created Alias object

    Raises:
        ValueError: If an alias with the same name already exists
    """

    def _create(session: Session) -> Alias:
        # Check for existing alias with same name
        existing = session.query(Alias).filter(Alias.name == name).first()
        if existing:
            raise ValueError(f"Alias with name '{name}' already exists")

        alias = Alias(
            name=name,
            target_model=target_model,
            description=description,
            enabled=enabled,
            use_cache=use_cache,
            cache_similarity_threshold=cache_similarity_threshold,
            cache_match_system_prompt=cache_match_system_prompt,
            cache_match_last_message_only=cache_match_last_message_only,
            cache_ttl_hours=cache_ttl_hours,
            cache_min_tokens=cache_min_tokens,
            cache_max_tokens=cache_max_tokens,
            cache_collection=cache_collection,
        )
        if tags:
            alias.tags = tags

        session.add(alias)
        session.flush()  # Get the ID
        logger.info(f"Created alias: {name} -> {target_model}")
        return alias

    if db:
        return _create(db)

    with get_db_context() as session:
        alias = _create(session)
        # Refresh to get all attributes before session closes
        session.refresh(alias)
        # Create a detached copy with the data we need
        return _alias_to_detached(alias)


def update_alias(
    alias_id: int,
    name: str | None = None,
    target_model: str | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    # Caching
    use_cache: bool | None = None,
    cache_similarity_threshold: float | None = None,
    cache_match_system_prompt: bool | None = None,
    cache_match_last_message_only: bool | None = None,
    cache_ttl_hours: int | None = None,
    cache_min_tokens: int | None = None,
    cache_max_tokens: int | None = None,
    cache_collection: str | None = None,
    db: Optional[Session] = None,
) -> Optional[Alias]:
    """
    Update an existing alias.

    Args:
        alias_id: ID of the alias to update
        name: New name (optional)
        target_model: New target model (optional)
        tags: New tags list (optional, pass empty list to clear)
        description: New description (optional)
        enabled: New enabled state (optional)
        db: Optional database session

    Returns:
        Updated Alias object or None if not found

    Raises:
        ValueError: If new name conflicts with existing alias
    """

    def _update(session: Session) -> Optional[Alias]:
        alias = session.query(Alias).filter(Alias.id == alias_id).first()
        if not alias:
            return None

        # Check name uniqueness if changing
        if name and name != alias.name:
            existing = session.query(Alias).filter(Alias.name == name).first()
            if existing:
                raise ValueError(f"Alias with name '{name}' already exists")
            alias.name = name

        if target_model is not None:
            alias.target_model = target_model

        if tags is not None:
            alias.tags = tags

        if description is not None:
            alias.description = description

        if enabled is not None:
            alias.enabled = enabled

        # Cache settings
        if use_cache is not None:
            alias.use_cache = use_cache
        if cache_similarity_threshold is not None:
            alias.cache_similarity_threshold = cache_similarity_threshold
        if cache_match_system_prompt is not None:
            alias.cache_match_system_prompt = cache_match_system_prompt
        if cache_match_last_message_only is not None:
            alias.cache_match_last_message_only = cache_match_last_message_only
        if cache_ttl_hours is not None:
            alias.cache_ttl_hours = cache_ttl_hours
        if cache_min_tokens is not None:
            alias.cache_min_tokens = cache_min_tokens
        if cache_max_tokens is not None:
            alias.cache_max_tokens = cache_max_tokens
        if cache_collection is not None:
            alias.cache_collection = cache_collection

        session.flush()
        logger.info(f"Updated alias: {alias.name}")
        return alias

    if db:
        return _update(db)

    with get_db_context() as session:
        alias = _update(session)
        if alias:
            session.refresh(alias)
            return _alias_to_detached(alias)
        return None


def delete_alias(alias_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete an alias by ID.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        alias = session.query(Alias).filter(Alias.id == alias_id).first()
        if not alias:
            return False

        name = alias.name
        session.delete(alias)
        logger.info(f"Deleted alias: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def alias_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if an alias name is available (doesn't conflict with existing aliases).

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)
        db: Optional database session

    Returns:
        True if name is available, False if taken
    """

    def _check(session: Session) -> bool:
        query = session.query(Alias).filter(Alias.name == name)
        if exclude_id:
            query = query.filter(Alias.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_aliases(db: Optional[Session] = None) -> dict[str, Alias]:
    """
    Get all enabled aliases as a dict keyed by name.

    Useful for quick lookups during request processing.

    Returns:
        Dict mapping alias name to Alias object
    """

    def _get(session: Session) -> list[Alias]:
        return session.query(Alias).filter(Alias.enabled == True).all()  # noqa: E712

    if db:
        aliases = _get(db)
        return {a.name: a for a in aliases}

    with get_db_context() as session:
        aliases = _get(session)
        return {a.name: _alias_to_detached(a) for a in aliases}


def _alias_to_detached(alias: Alias) -> Alias:
    """
    Create a detached copy of an alias with all data loaded.

    This allows the alias to be used after the session is closed.
    """
    detached = Alias(
        name=alias.name,
        target_model=alias.target_model,
        tags_json=alias.tags_json,
        description=alias.description,
        enabled=alias.enabled,
        # Cache fields
        use_cache=alias.use_cache,
        cache_similarity_threshold=alias.cache_similarity_threshold,
        cache_match_system_prompt=alias.cache_match_system_prompt,
        cache_match_last_message_only=alias.cache_match_last_message_only,
        cache_ttl_hours=alias.cache_ttl_hours,
        cache_min_tokens=alias.cache_min_tokens,
        cache_max_tokens=alias.cache_max_tokens,
        cache_collection=alias.cache_collection,
        cache_hits=alias.cache_hits,
        cache_tokens_saved=alias.cache_tokens_saved,
        cache_cost_saved=alias.cache_cost_saved,
    )
    detached.id = alias.id
    detached.created_at = alias.created_at
    detached.updated_at = alias.updated_at
    return detached
