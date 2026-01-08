"""
Redirect CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete model redirects.
Redirects transparently map one model name to another before normal resolution.
"""

import fnmatch
import logging
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import Redirect

logger = logging.getLogger(__name__)


def get_all_redirects(db: Optional[Session] = None) -> list[Redirect]:
    """Get all redirects from the database."""
    if db:
        return db.query(Redirect).order_by(Redirect.source).all()

    with get_db_context() as session:
        redirects = session.query(Redirect).order_by(Redirect.source).all()
        return [_redirect_to_detached(r) for r in redirects]


def get_redirect_by_id(
    redirect_id: int, db: Optional[Session] = None
) -> Optional[Redirect]:
    """Get a redirect by its ID."""
    if db:
        return db.query(Redirect).filter(Redirect.id == redirect_id).first()

    with get_db_context() as session:
        redirect = session.query(Redirect).filter(Redirect.id == redirect_id).first()
        return _redirect_to_detached(redirect) if redirect else None


def get_redirect_by_source(
    source: str, db: Optional[Session] = None
) -> Optional[Redirect]:
    """Get a redirect by its exact source pattern."""
    source = source.lower().strip()
    if db:
        return db.query(Redirect).filter(Redirect.source == source).first()

    with get_db_context() as session:
        redirect = session.query(Redirect).filter(Redirect.source == source).first()
        return _redirect_to_detached(redirect) if redirect else None


def find_matching_redirect(
    model_name: str, db: Optional[Session] = None
) -> Optional[tuple[Redirect, str]]:
    """
    Find a redirect that matches the given model name.

    Supports both exact matches and wildcard patterns using fnmatch.

    Args:
        model_name: The model name to check for redirects
        db: Optional database session

    Returns:
        Tuple of (Redirect, resolved_target) if found, None otherwise.
        The resolved_target handles wildcard substitution.
    """
    name = model_name.lower().strip()

    def _find(session: Session) -> Optional[tuple[Redirect, str]]:
        # Get all enabled redirects
        redirects = session.query(Redirect).filter(Redirect.enabled == True).all()  # noqa: E712

        for redirect in redirects:
            source = redirect.source.lower()

            # Exact match
            if source == name:
                return redirect, redirect.target

            # Wildcard match using fnmatch
            if "*" in source or "?" in source:
                if fnmatch.fnmatch(name, source):
                    # For wildcard redirects, substitute the matched portion
                    # e.g., source="openrouter/anthropic/*", target="anthropic/*"
                    # name="openrouter/anthropic/claude-3-opus"
                    # result="anthropic/claude-3-opus"
                    resolved_target = _resolve_wildcard_target(
                        name, source, redirect.target
                    )
                    return redirect, resolved_target

        return None

    if db:
        result = _find(db)
        return result

    with get_db_context() as session:
        result = _find(session)
        if result:
            return _redirect_to_detached(result[0]), result[1]
        return None


def _resolve_wildcard_target(
    name: str, source_pattern: str, target_pattern: str
) -> str:
    """
    Resolve a wildcard target pattern based on the matched source.

    For patterns like:
        source: "openrouter/anthropic/*"
        target: "anthropic/*"
        name: "openrouter/anthropic/claude-3-opus"

    Returns: "anthropic/claude-3-opus"
    """
    # If target doesn't have wildcards, return it as-is
    if "*" not in target_pattern:
        return target_pattern

    # Find the position of the wildcard in source
    # Simple case: single * at the end
    if source_pattern.endswith("*") and target_pattern.endswith("*"):
        source_prefix = source_pattern[:-1]  # "openrouter/anthropic/"
        target_prefix = target_pattern[:-1]  # "anthropic/"

        if name.startswith(source_prefix):
            matched_suffix = name[len(source_prefix) :]  # "claude-3-opus"
            return target_prefix + matched_suffix

    # For more complex patterns, just return the target as-is
    # (could be extended with more sophisticated pattern matching)
    return target_pattern.replace("*", "")


def create_redirect(
    source: str,
    target: str,
    description: str | None = None,
    enabled: bool = True,
    tags: list[str] | None = None,
    db: Optional[Session] = None,
) -> Redirect:
    """
    Create a new redirect.

    Args:
        source: Model name pattern to redirect from (supports wildcards)
        target: Model name to redirect to (can include * for wildcard substitution)
        description: Optional description
        enabled: Whether the redirect is enabled (default True)
        tags: Optional list of tags for usage tracking
        db: Optional database session

    Returns:
        The created Redirect object

    Raises:
        ValueError: If a redirect with the same source already exists
    """
    source = source.lower().strip()
    target = target.lower().strip()

    def _create(session: Session) -> Redirect:
        # Check for existing redirect with same source
        existing = session.query(Redirect).filter(Redirect.source == source).first()
        if existing:
            raise ValueError(f"Redirect for source '{source}' already exists")

        redirect = Redirect(
            source=source,
            target=target,
            description=description,
            enabled=enabled,
        )
        if tags:
            redirect.tags = tags

        session.add(redirect)
        session.flush()
        logger.info(f"Created redirect: {source} -> {target}")
        return redirect

    if db:
        return _create(db)

    with get_db_context() as session:
        redirect = _create(session)
        session.refresh(redirect)
        return _redirect_to_detached(redirect)


def update_redirect(
    redirect_id: int,
    source: str | None = None,
    target: str | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    tags: list[str] | None = None,
    db: Optional[Session] = None,
) -> Optional[Redirect]:
    """
    Update an existing redirect.

    Args:
        redirect_id: ID of the redirect to update
        source: New source pattern (optional)
        target: New target (optional)
        description: New description (optional)
        enabled: New enabled state (optional)
        tags: New tags list (optional, pass empty list to clear)
        db: Optional database session

    Returns:
        Updated Redirect object or None if not found

    Raises:
        ValueError: If new source conflicts with existing redirect
    """

    def _update(session: Session) -> Optional[Redirect]:
        redirect = session.query(Redirect).filter(Redirect.id == redirect_id).first()
        if not redirect:
            return None

        # Check source uniqueness if changing
        if source and source.lower().strip() != redirect.source:
            new_source = source.lower().strip()
            existing = (
                session.query(Redirect).filter(Redirect.source == new_source).first()
            )
            if existing:
                raise ValueError(f"Redirect for source '{new_source}' already exists")
            redirect.source = new_source

        if target is not None:
            redirect.target = target.lower().strip()

        if description is not None:
            redirect.description = description

        if enabled is not None:
            redirect.enabled = enabled

        if tags is not None:
            redirect.tags = tags

        session.flush()
        logger.info(f"Updated redirect: {redirect.source} -> {redirect.target}")
        return redirect

    if db:
        return _update(db)

    with get_db_context() as session:
        redirect = _update(session)
        if redirect:
            session.refresh(redirect)
            return _redirect_to_detached(redirect)
        return None


def delete_redirect(redirect_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a redirect by ID.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        redirect = session.query(Redirect).filter(Redirect.id == redirect_id).first()
        if not redirect:
            return False

        source = redirect.source
        session.delete(redirect)
        logger.info(f"Deleted redirect: {source}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def increment_redirect_count(redirect_id: int, db: Optional[Session] = None) -> None:
    """Increment the redirect count for tracking usage."""

    def _increment(session: Session) -> None:
        redirect = session.query(Redirect).filter(Redirect.id == redirect_id).first()
        if redirect:
            redirect.redirect_count += 1

    if db:
        _increment(db)
        return

    with get_db_context() as session:
        _increment(session)


def get_enabled_redirects(db: Optional[Session] = None) -> list[Redirect]:
    """
    Get all enabled redirects.

    Returns:
        List of enabled Redirect objects
    """

    def _get(session: Session) -> list[Redirect]:
        return session.query(Redirect).filter(Redirect.enabled == True).all()  # noqa: E712

    if db:
        return _get(db)

    with get_db_context() as session:
        redirects = _get(session)
        return [_redirect_to_detached(r) for r in redirects]


def _redirect_to_detached(redirect: Redirect) -> Redirect:
    """
    Create a detached copy of a redirect with all data loaded.

    This allows the redirect to be used after the session is closed.
    """
    detached = Redirect(
        source=redirect.source,
        target=redirect.target,
        description=redirect.description,
        enabled=redirect.enabled,
        tags_json=redirect.tags_json,
    )
    detached.id = redirect.id
    detached.redirect_count = redirect.redirect_count
    detached.created_at = redirect.created_at
    detached.updated_at = redirect.updated_at
    return detached
