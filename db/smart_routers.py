"""
Smart Router CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart routers.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import SmartRouter

logger = logging.getLogger(__name__)


def get_all_smart_routers(db: Optional[Session] = None) -> list[SmartRouter]:
    """Get all smart routers from the database."""
    if db:
        return db.query(SmartRouter).order_by(SmartRouter.name).all()

    with get_db_context() as session:
        routers = session.query(SmartRouter).order_by(SmartRouter.name).all()
        return [_router_to_detached(r) for r in routers]


def get_smart_router_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartRouter]:
    """Get a smart router by its name."""
    if db:
        return db.query(SmartRouter).filter(SmartRouter.name == name).first()

    with get_db_context() as session:
        router = session.query(SmartRouter).filter(SmartRouter.name == name).first()
        return _router_to_detached(router) if router else None


def get_smart_router_by_id(
    router_id: int, db: Optional[Session] = None
) -> Optional[SmartRouter]:
    """Get a smart router by its ID."""
    if db:
        return db.query(SmartRouter).filter(SmartRouter.id == router_id).first()

    with get_db_context() as session:
        router = session.query(SmartRouter).filter(SmartRouter.id == router_id).first()
        return _router_to_detached(router) if router else None


def create_smart_router(
    name: str,
    designator_model: str,
    purpose: str,
    candidates: list[dict],
    fallback_model: str,
    strategy: str = "per_request",
    session_ttl: int = 3600,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool = True,
    use_model_intelligence: bool = False,
    search_provider: str | None = None,
    intelligence_model: str | None = None,
    db: Optional[Session] = None,
) -> SmartRouter:
    """
    Create a new smart router.

    Args:
        name: Unique name for the router
        designator_model: Model to use for routing decisions ("provider/model")
        purpose: Description of routing purpose for designator context
        candidates: List of candidate models [{"model": "provider/model", "notes": "optional"}]
        fallback_model: Fallback model if designator fails ("provider/model")
        strategy: Routing strategy ("per_request" or "per_session")
        session_ttl: Session cache TTL in seconds (for per_session strategy)
        tags: Optional list of tags to assign to requests
        description: Optional description
        enabled: Whether the router is enabled (default True)
        use_model_intelligence: Whether to use web-gathered model intelligence (default False)
        search_provider: Search provider for model intelligence (e.g., "searxng")
        intelligence_model: Model to use for summarizing search results
        db: Optional database session

    Returns:
        The created SmartRouter object

    Raises:
        ValueError: If a router with the same name already exists
    """

    def _create(session: Session) -> SmartRouter:
        # Check for existing router with same name
        existing = session.query(SmartRouter).filter(SmartRouter.name == name).first()
        if existing:
            raise ValueError(f"Smart router with name '{name}' already exists")

        router = SmartRouter(
            name=name,
            designator_model=designator_model,
            purpose=purpose,
            fallback_model=fallback_model,
            strategy=strategy,
            session_ttl=session_ttl,
            description=description,
            enabled=enabled,
            use_model_intelligence=use_model_intelligence,
            search_provider=search_provider,
            intelligence_model=intelligence_model,
        )
        router.candidates = candidates
        if tags:
            router.tags = tags

        session.add(router)
        session.flush()  # Get the ID
        logger.info(f"Created smart router: {name}")
        return router

    if db:
        return _create(db)

    with get_db_context() as session:
        router = _create(session)
        session.refresh(router)
        return _router_to_detached(router)


def update_smart_router(
    router_id: int,
    name: str | None = None,
    designator_model: str | None = None,
    purpose: str | None = None,
    candidates: list[dict] | None = None,
    fallback_model: str | None = None,
    strategy: str | None = None,
    session_ttl: int | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    use_model_intelligence: bool | None = None,
    search_provider: str | None = None,
    intelligence_model: str | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartRouter]:
    """
    Update an existing smart router.

    Args:
        router_id: ID of the router to update
        name: New name (optional)
        designator_model: New designator model (optional)
        purpose: New purpose (optional)
        candidates: New candidates list (optional)
        fallback_model: New fallback model (optional)
        strategy: New routing strategy (optional)
        session_ttl: New session TTL (optional)
        tags: New tags list (optional, pass empty list to clear)
        description: New description (optional)
        enabled: New enabled state (optional)
        use_model_intelligence: New model intelligence state (optional)
        search_provider: New search provider (optional)
        intelligence_model: New intelligence model (optional)
        db: Optional database session

    Returns:
        Updated SmartRouter object or None if not found

    Raises:
        ValueError: If new name conflicts with existing router
    """

    def _update(session: Session) -> Optional[SmartRouter]:
        router = session.query(SmartRouter).filter(SmartRouter.id == router_id).first()
        if not router:
            return None

        # Check name uniqueness if changing
        if name and name != router.name:
            existing = (
                session.query(SmartRouter).filter(SmartRouter.name == name).first()
            )
            if existing:
                raise ValueError(f"Smart router with name '{name}' already exists")
            router.name = name

        if designator_model is not None:
            router.designator_model = designator_model

        if purpose is not None:
            router.purpose = purpose

        if candidates is not None:
            router.candidates = candidates

        if fallback_model is not None:
            router.fallback_model = fallback_model

        if strategy is not None:
            router.strategy = strategy

        if session_ttl is not None:
            router.session_ttl = session_ttl

        if tags is not None:
            router.tags = tags

        if description is not None:
            router.description = description

        if enabled is not None:
            router.enabled = enabled

        if use_model_intelligence is not None:
            router.use_model_intelligence = use_model_intelligence

        if search_provider is not None:
            router.search_provider = search_provider if search_provider else None

        if intelligence_model is not None:
            router.intelligence_model = (
                intelligence_model if intelligence_model else None
            )

        session.flush()
        logger.info(f"Updated smart router: {router.name}")
        return router

    if db:
        return _update(db)

    with get_db_context() as session:
        router = _update(session)
        if router:
            session.refresh(router)
            return _router_to_detached(router)
        return None


def delete_smart_router(router_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart router by ID.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        router = session.query(SmartRouter).filter(SmartRouter.id == router_id).first()
        if not router:
            return False

        name = router.name
        session.delete(router)
        logger.info(f"Deleted smart router: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def router_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if a router name is available.

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)
        db: Optional database session

    Returns:
        True if name is available, False if taken
    """

    def _check(session: Session) -> bool:
        query = session.query(SmartRouter).filter(SmartRouter.name == name)
        if exclude_id:
            query = query.filter(SmartRouter.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_smart_routers(db: Optional[Session] = None) -> dict[str, SmartRouter]:
    """
    Get all enabled smart routers as a dict keyed by name.

    Useful for quick lookups during request processing.

    Returns:
        Dict mapping router name to SmartRouter object
    """

    def _get(session: Session) -> list[SmartRouter]:
        return (
            session.query(SmartRouter)
            .filter(SmartRouter.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        routers = _get(db)
        return {r.name: r for r in routers}

    with get_db_context() as session:
        routers = _get(session)
        return {r.name: _router_to_detached(r) for r in routers}


def _router_to_detached(router: SmartRouter) -> SmartRouter:
    """
    Create a detached copy of a smart router with all data loaded.

    This allows the router to be used after the session is closed.
    """
    detached = SmartRouter(
        name=router.name,
        designator_model=router.designator_model,
        purpose=router.purpose,
        candidates_json=router.candidates_json,
        fallback_model=router.fallback_model,
        strategy=router.strategy,
        session_ttl=router.session_ttl,
        tags_json=router.tags_json,
        description=router.description,
        enabled=router.enabled,
        use_model_intelligence=router.use_model_intelligence,
        search_provider=router.search_provider,
        intelligence_model=router.intelligence_model,
    )
    detached.id = router.id
    detached.created_at = router.created_at
    detached.updated_at = router.updated_at
    return detached
