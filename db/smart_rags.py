"""
Smart RAG CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart RAGs.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import SmartRAG

logger = logging.getLogger(__name__)


def get_all_smart_rags(db: Optional[Session] = None) -> list[SmartRAG]:
    """Get all smart RAGs from the database."""
    if db:
        return db.query(SmartRAG).order_by(SmartRAG.name).all()

    with get_db_context() as session:
        rags = session.query(SmartRAG).order_by(SmartRAG.name).all()
        return [_rag_to_detached(r) for r in rags]


def get_smart_rag_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartRAG]:
    """Get a smart RAG by its name."""
    if db:
        return db.query(SmartRAG).filter(SmartRAG.name == name).first()

    with get_db_context() as session:
        rag = session.query(SmartRAG).filter(SmartRAG.name == name).first()
        return _rag_to_detached(rag) if rag else None


def get_smart_rag_by_id(
    rag_id: int, db: Optional[Session] = None
) -> Optional[SmartRAG]:
    """Get a smart RAG by its ID."""
    if db:
        return db.query(SmartRAG).filter(SmartRAG.id == rag_id).first()

    with get_db_context() as session:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        return _rag_to_detached(rag) if rag else None


def create_smart_rag(
    name: str,
    source_path: str,
    target_model: str,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    ollama_url: str | None = None,
    index_schedule: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    max_results: int = 5,
    similarity_threshold: float = 0.7,
    max_context_tokens: int = 4000,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> SmartRAG:
    """
    Create a new smart RAG.

    Args:
        name: Unique name for the RAG
        source_path: Path to document folder (Docker-mapped)
        target_model: Model to forward requests to ("provider/model")
        embedding_provider: Embedding provider ("local", "ollama", "openai")
        embedding_model: Model name for Ollama/OpenAI embeddings
        ollama_url: Override URL for Ollama instance
        index_schedule: Cron expression for scheduled indexing
        chunk_size: Document chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_results: Maximum chunks to retrieve
        similarity_threshold: Minimum similarity for retrieval (0.0-1.0)
        max_context_tokens: Maximum context tokens to inject
        tags: Optional list of tags to assign to requests
        description: Optional description
        enabled: Whether the RAG is enabled (default True)
        db: Optional database session

    Returns:
        The created SmartRAG object

    Raises:
        ValueError: If a RAG with the same name already exists
    """

    def _create(session: Session) -> SmartRAG:
        # Check for existing RAG with same name
        existing = session.query(SmartRAG).filter(SmartRAG.name == name).first()
        if existing:
            raise ValueError(f"Smart RAG with name '{name}' already exists")

        rag = SmartRAG(
            name=name,
            source_path=source_path,
            target_model=target_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            index_schedule=index_schedule,
            index_status="pending",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            max_context_tokens=max_context_tokens,
            description=description,
            enabled=enabled,
        )
        if tags:
            rag.tags = tags

        session.add(rag)
        session.flush()  # Get the ID

        # Generate collection name using the ID
        rag.collection_name = f"smartrag_{rag.id}"

        logger.info(f"Created smart RAG: {name}")
        return rag

    if db:
        return _create(db)

    with get_db_context() as session:
        rag = _create(session)
        session.refresh(rag)
        return _rag_to_detached(rag)


def update_smart_rag(
    rag_id: int,
    name: str | None = None,
    source_path: str | None = None,
    target_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    ollama_url: str | None = None,
    index_schedule: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_results: int | None = None,
    similarity_threshold: float | None = None,
    max_context_tokens: int | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartRAG]:
    """
    Update an existing smart RAG.

    Args:
        rag_id: ID of the RAG to update
        name: New name (optional)
        source_path: New source path (optional)
        target_model: New target model (optional)
        embedding_provider: New embedding provider (optional)
        embedding_model: New embedding model (optional)
        ollama_url: New Ollama URL (optional)
        index_schedule: New index schedule (optional)
        chunk_size: New chunk size (optional)
        chunk_overlap: New chunk overlap (optional)
        max_results: New max results (optional)
        similarity_threshold: New similarity threshold (optional)
        max_context_tokens: New max context tokens (optional)
        tags: New tags list (optional, pass empty list to clear)
        description: New description (optional)
        enabled: New enabled state (optional)
        db: Optional database session

    Returns:
        Updated SmartRAG object or None if not found

    Raises:
        ValueError: If new name conflicts with existing RAG
    """

    def _update(session: Session) -> Optional[SmartRAG]:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        if not rag:
            return None

        # Check name uniqueness if changing
        if name and name != rag.name:
            existing = session.query(SmartRAG).filter(SmartRAG.name == name).first()
            if existing:
                raise ValueError(f"Smart RAG with name '{name}' already exists")
            rag.name = name

        if source_path is not None:
            rag.source_path = source_path

        if target_model is not None:
            rag.target_model = target_model

        if embedding_provider is not None:
            rag.embedding_provider = embedding_provider

        if embedding_model is not None:
            rag.embedding_model = embedding_model

        if ollama_url is not None:
            rag.ollama_url = ollama_url

        if index_schedule is not None:
            rag.index_schedule = index_schedule

        if chunk_size is not None:
            rag.chunk_size = chunk_size

        if chunk_overlap is not None:
            rag.chunk_overlap = chunk_overlap

        if max_results is not None:
            rag.max_results = max_results

        if similarity_threshold is not None:
            rag.similarity_threshold = similarity_threshold

        if max_context_tokens is not None:
            rag.max_context_tokens = max_context_tokens

        if tags is not None:
            rag.tags = tags

        if description is not None:
            rag.description = description

        if enabled is not None:
            rag.enabled = enabled

        session.flush()
        logger.info(f"Updated smart RAG: {rag.name}")
        return rag

    if db:
        return _update(db)

    with get_db_context() as session:
        rag = _update(session)
        if rag:
            session.refresh(rag)
            return _rag_to_detached(rag)
        return None


def update_smart_rag_index_status(
    rag_id: int,
    status: str,
    error: str | None = None,
    document_count: int | None = None,
    chunk_count: int | None = None,
    collection_name: str | None = None,
    db: Optional[Session] = None,
) -> bool:
    """
    Update RAG index status.

    Args:
        rag_id: ID of the RAG to update
        status: New status ("pending", "indexing", "ready", "error")
        error: Error message if status is "error"
        document_count: Number of documents indexed
        chunk_count: Number of chunks created
        collection_name: ChromaDB collection name

    Returns:
        True if updated, False if RAG not found
    """

    def _update_status(session: Session) -> bool:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        if not rag:
            return False

        rag.index_status = status
        rag.index_error = error if status == "error" else None

        if status == "ready":
            rag.last_indexed = datetime.utcnow()

        if document_count is not None:
            rag.document_count = document_count

        if chunk_count is not None:
            rag.chunk_count = chunk_count

        if collection_name is not None:
            rag.collection_name = collection_name

        session.flush()
        logger.info(f"Updated index status for smart RAG {rag.name}: {status}")
        return True

    if db:
        return _update_status(db)

    with get_db_context() as session:
        return _update_status(session)


def update_smart_rag_stats(
    rag_id: int,
    increment_requests: int = 0,
    increment_injections: int = 0,
    db: Optional[Session] = None,
) -> bool:
    """
    Update RAG statistics (atomically increment counters).

    Args:
        rag_id: ID of the RAG to update
        increment_requests: Number of requests to add
        increment_injections: Number of context injections to add

    Returns:
        True if updated, False if RAG not found
    """

    def _update_stats(session: Session) -> bool:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        if not rag:
            return False

        rag.total_requests += increment_requests
        rag.context_injections += increment_injections

        session.flush()
        return True

    if db:
        return _update_stats(db)

    with get_db_context() as session:
        return _update_stats(session)


def reset_smart_rag_stats(rag_id: int, db: Optional[Session] = None) -> bool:
    """
    Reset RAG statistics to zero.

    Returns:
        True if reset, False if RAG not found
    """

    def _reset(session: Session) -> bool:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        if not rag:
            return False

        rag.total_requests = 0
        rag.context_injections = 0

        session.flush()
        logger.info(f"Reset stats for smart RAG: {rag.name}")
        return True

    if db:
        return _reset(db)

    with get_db_context() as session:
        return _reset(session)


def delete_smart_rag(rag_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart RAG by ID.

    Note: This does NOT delete the ChromaDB collection. Call the indexer's
    delete_collection() method first if you want to remove the indexed data.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        rag = session.query(SmartRAG).filter(SmartRAG.id == rag_id).first()
        if not rag:
            return False

        name = rag.name
        session.delete(rag)
        logger.info(f"Deleted smart RAG: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def rag_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if a RAG name is available.

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)
        db: Optional database session

    Returns:
        True if name is available, False if taken
    """

    def _check(session: Session) -> bool:
        query = session.query(SmartRAG).filter(SmartRAG.name == name)
        if exclude_id:
            query = query.filter(SmartRAG.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_smart_rags(db: Optional[Session] = None) -> dict[str, SmartRAG]:
    """
    Get all enabled smart RAGs as a dict keyed by name.

    Useful for quick lookups during request processing.

    Returns:
        Dict mapping RAG name to SmartRAG object
    """

    def _get(session: Session) -> list[SmartRAG]:
        return (
            session.query(SmartRAG)
            .filter(SmartRAG.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        rags = _get(db)
        return {r.name: r for r in rags}

    with get_db_context() as session:
        rags = _get(session)
        return {r.name: _rag_to_detached(r) for r in rags}


def get_rags_with_schedule(db: Optional[Session] = None) -> list[SmartRAG]:
    """
    Get all enabled RAGs that have an index schedule configured.

    Useful for the scheduler to set up cron jobs.

    Returns:
        List of SmartRAG objects with schedules
    """

    def _get(session: Session) -> list[SmartRAG]:
        return (
            session.query(SmartRAG)
            .filter(SmartRAG.enabled == True)  # noqa: E712
            .filter(SmartRAG.index_schedule != None)  # noqa: E711
            .filter(SmartRAG.index_schedule != "")
            .all()
        )

    if db:
        return _get(db)

    with get_db_context() as session:
        rags = _get(session)
        return [_rag_to_detached(r) for r in rags]


def _rag_to_detached(rag: SmartRAG) -> SmartRAG:
    """
    Create a detached copy of a smart RAG with all data loaded.

    This allows the RAG to be used after the session is closed.
    """
    detached = SmartRAG(
        name=rag.name,
        source_path=rag.source_path,
        target_model=rag.target_model,
        embedding_provider=rag.embedding_provider,
        embedding_model=rag.embedding_model,
        ollama_url=rag.ollama_url,
        index_schedule=rag.index_schedule,
        last_indexed=rag.last_indexed,
        index_status=rag.index_status,
        index_error=rag.index_error,
        chunk_size=rag.chunk_size,
        chunk_overlap=rag.chunk_overlap,
        max_results=rag.max_results,
        similarity_threshold=rag.similarity_threshold,
        max_context_tokens=rag.max_context_tokens,
        collection_name=rag.collection_name,
        document_count=rag.document_count,
        chunk_count=rag.chunk_count,
        total_requests=rag.total_requests,
        context_injections=rag.context_injections,
        tags_json=rag.tags_json,
        description=rag.description,
        enabled=rag.enabled,
    )
    detached.id = rag.id
    detached.created_at = rag.created_at
    detached.updated_at = rag.updated_at
    return detached
