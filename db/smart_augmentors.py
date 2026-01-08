"""
Smart Augmentor CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart augmentors.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import SmartAugmentor

logger = logging.getLogger(__name__)


def get_all_smart_augmentors(db: Optional[Session] = None) -> list[SmartAugmentor]:
    """Get all smart augmentors from the database."""
    if db:
        return db.query(SmartAugmentor).order_by(SmartAugmentor.name).all()

    with get_db_context() as session:
        augmentors = session.query(SmartAugmentor).order_by(SmartAugmentor.name).all()
        return [_augmentor_to_detached(a) for a in augmentors]


def get_smart_augmentor_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartAugmentor]:
    """Get a smart augmentor by its name."""
    if db:
        return db.query(SmartAugmentor).filter(SmartAugmentor.name == name).first()

    with get_db_context() as session:
        augmentor = (
            session.query(SmartAugmentor).filter(SmartAugmentor.name == name).first()
        )
        return _augmentor_to_detached(augmentor) if augmentor else None


def get_smart_augmentor_by_id(
    augmentor_id: int, db: Optional[Session] = None
) -> Optional[SmartAugmentor]:
    """Get a smart augmentor by its ID."""
    if db:
        return (
            db.query(SmartAugmentor).filter(SmartAugmentor.id == augmentor_id).first()
        )

    with get_db_context() as session:
        augmentor = (
            session.query(SmartAugmentor)
            .filter(SmartAugmentor.id == augmentor_id)
            .first()
        )
        return _augmentor_to_detached(augmentor) if augmentor else None


def create_smart_augmentor(
    name: str,
    designator_model: str,
    target_model: str,
    purpose: str | None = None,
    search_provider: str = "searxng",
    search_provider_url: str | None = None,
    max_search_results: int = 5,
    max_scrape_urls: int = 3,
    max_context_tokens: int = 4000,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> SmartAugmentor:
    """
    Create a new smart augmentor.

    Args:
        name: Unique name for the augmentor
        designator_model: Model to use for deciding augmentation ("provider/model")
        target_model: Model to forward augmented requests to ("provider/model")
        purpose: Context for augmentation decisions
        search_provider: Search provider to use ("searxng", "perplexity", etc.)
        search_provider_url: Override URL for self-hosted providers
        max_search_results: Maximum search results to include
        max_scrape_urls: Maximum URLs to scrape
        max_context_tokens: Maximum tokens for injected context
        tags: Optional list of tags for usage tracking
        description: Optional description
        enabled: Whether the augmentor is enabled (default True)
        db: Optional database session

    Returns:
        The created SmartAugmentor object

    Raises:
        ValueError: If an augmentor with the same name already exists
    """

    def _create(session: Session) -> SmartAugmentor:
        # Check for existing augmentor with same name
        existing = (
            session.query(SmartAugmentor).filter(SmartAugmentor.name == name).first()
        )
        if existing:
            raise ValueError(f"Smart augmentor with name '{name}' already exists")

        augmentor = SmartAugmentor(
            name=name,
            designator_model=designator_model,
            target_model=target_model,
            purpose=purpose,
            search_provider=search_provider,
            search_provider_url=search_provider_url,
            max_search_results=max_search_results,
            max_scrape_urls=max_scrape_urls,
            max_context_tokens=max_context_tokens,
            description=description,
            enabled=enabled,
        )
        if tags:
            augmentor.tags = tags

        session.add(augmentor)
        session.flush()  # Get the ID
        logger.info(f"Created smart augmentor: {name}")
        return augmentor

    if db:
        return _create(db)

    with get_db_context() as session:
        augmentor = _create(session)
        session.refresh(augmentor)
        return _augmentor_to_detached(augmentor)


def update_smart_augmentor(
    augmentor_id: int,
    name: str | None = None,
    designator_model: str | None = None,
    target_model: str | None = None,
    purpose: str | None = None,
    search_provider: str | None = None,
    search_provider_url: str | None = None,
    max_search_results: int | None = None,
    max_scrape_urls: int | None = None,
    max_context_tokens: int | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartAugmentor]:
    """
    Update an existing smart augmentor.

    Args:
        augmentor_id: ID of the augmentor to update
        name: New name (optional)
        designator_model: New designator model (optional)
        target_model: New target model (optional)
        purpose: New purpose (optional)
        search_provider: New search provider (optional)
        search_provider_url: New search provider URL (optional)
        max_search_results: New max search results (optional)
        max_scrape_urls: New max scrape URLs (optional)
        max_context_tokens: New max context tokens (optional)
        tags: New tags (optional)
        description: New description (optional)
        enabled: New enabled status (optional)
        db: Optional database session

    Returns:
        Updated SmartAugmentor object or None if not found

    Raises:
        ValueError: If new name conflicts with existing augmentor
    """

    def _update(session: Session) -> Optional[SmartAugmentor]:
        augmentor = (
            session.query(SmartAugmentor)
            .filter(SmartAugmentor.id == augmentor_id)
            .first()
        )
        if not augmentor:
            return None

        # Check name uniqueness if changing
        if name and name != augmentor.name:
            existing = (
                session.query(SmartAugmentor)
                .filter(SmartAugmentor.name == name)
                .first()
            )
            if existing:
                raise ValueError(f"Smart augmentor with name '{name}' already exists")
            augmentor.name = name

        if designator_model is not None:
            augmentor.designator_model = designator_model

        if target_model is not None:
            augmentor.target_model = target_model

        if purpose is not None:
            augmentor.purpose = purpose

        if search_provider is not None:
            augmentor.search_provider = search_provider

        if search_provider_url is not None:
            augmentor.search_provider_url = search_provider_url

        if max_search_results is not None:
            augmentor.max_search_results = max_search_results

        if max_scrape_urls is not None:
            augmentor.max_scrape_urls = max_scrape_urls

        if max_context_tokens is not None:
            augmentor.max_context_tokens = max_context_tokens

        if tags is not None:
            augmentor.tags = tags

        if description is not None:
            augmentor.description = description

        if enabled is not None:
            augmentor.enabled = enabled

        session.flush()
        logger.info(f"Updated smart augmentor: {augmentor.name}")
        return augmentor

    if db:
        return _update(db)

    with get_db_context() as session:
        augmentor = _update(session)
        if augmentor:
            session.refresh(augmentor)
            return _augmentor_to_detached(augmentor)
        return None


def delete_smart_augmentor(augmentor_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart augmentor.

    Args:
        augmentor_id: ID of the augmentor to delete
        db: Optional database session

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        augmentor = (
            session.query(SmartAugmentor)
            .filter(SmartAugmentor.id == augmentor_id)
            .first()
        )
        if not augmentor:
            return False

        name = augmentor.name
        session.delete(augmentor)
        logger.info(f"Deleted smart augmentor: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def update_smart_augmentor_stats(
    augmentor_id: int,
    increment_requests: int = 0,
    increment_augmented: int = 0,
    increment_search: int = 0,
    increment_scrape: int = 0,
    db: Optional[Session] = None,
) -> bool:
    """
    Update statistics for a smart augmentor.

    Args:
        augmentor_id: ID of the augmentor
        increment_requests: Amount to add to total_requests
        increment_augmented: Amount to add to augmented_requests
        increment_search: Amount to add to search_requests
        increment_scrape: Amount to add to scrape_requests
        db: Optional database session

    Returns:
        True if updated, False if not found
    """

    def _update(session: Session) -> bool:
        augmentor = (
            session.query(SmartAugmentor)
            .filter(SmartAugmentor.id == augmentor_id)
            .first()
        )
        if not augmentor:
            return False

        if increment_requests:
            augmentor.total_requests += increment_requests
        if increment_augmented:
            augmentor.augmented_requests += increment_augmented
        if increment_search:
            augmentor.search_requests += increment_search
        if increment_scrape:
            augmentor.scrape_requests += increment_scrape

        session.flush()
        return True

    if db:
        return _update(db)

    with get_db_context() as session:
        return _update(session)


def get_smart_augmentor_names() -> dict[str, SmartAugmentor]:
    """
    Get a mapping of augmentor names to augmentor objects.

    Returns:
        Dict mapping lowercase names to SmartAugmentor objects
    """
    with get_db_context() as session:
        augmentors = session.query(SmartAugmentor).all()
        return {a.name.lower(): _augmentor_to_detached(a) for a in augmentors}


def get_enabled_smart_augmentors() -> dict[str, SmartAugmentor]:
    """
    Get a mapping of enabled augmentor names to augmentor objects.

    Returns:
        Dict mapping lowercase names to enabled SmartAugmentor objects
    """
    with get_db_context() as session:
        augmentors = (
            session.query(SmartAugmentor).filter(SmartAugmentor.enabled == True).all()
        )
        return {a.name.lower(): _augmentor_to_detached(a) for a in augmentors}


def _augmentor_to_detached(augmentor: SmartAugmentor) -> SmartAugmentor:
    """
    Create a detached copy of a smart augmentor with all data loaded.

    This allows the augmentor to be used after the session is closed.
    """
    detached = SmartAugmentor(
        name=augmentor.name,
        designator_model=augmentor.designator_model,
        target_model=augmentor.target_model,
        purpose=augmentor.purpose,
        search_provider=augmentor.search_provider,
        search_provider_url=augmentor.search_provider_url,
        max_search_results=augmentor.max_search_results,
        max_scrape_urls=augmentor.max_scrape_urls,
        max_context_tokens=augmentor.max_context_tokens,
        total_requests=augmentor.total_requests,
        augmented_requests=augmentor.augmented_requests,
        search_requests=augmentor.search_requests,
        scrape_requests=augmentor.scrape_requests,
        tags_json=augmentor.tags_json,
        description=augmentor.description,
        enabled=augmentor.enabled,
    )
    detached.id = augmentor.id
    detached.created_at = augmentor.created_at
    detached.updated_at = augmentor.updated_at
    return detached
