"""
Document Store CRUD operations for LLM Relay.

Handles creation, retrieval, updating, and deletion of document stores,
as well as linking/unlinking stores to Smart Enrichers.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import DocumentStore, SmartEnricher, smart_enricher_stores

logger = logging.getLogger(__name__)


def get_all_document_stores(db=None) -> list[DocumentStore]:
    """Get all document stores ordered by name."""

    def _query(session) -> list[DocumentStore]:
        stmt = select(DocumentStore).order_by(DocumentStore.name)
        return list(session.execute(stmt).scalars().all())

    if db:
        return _query(db)
    with get_db_context() as session:
        stores = _query(session)
        # Detach from session for use outside
        return [_store_to_detached(s, session) for s in stores]


def get_document_store_by_name(name: str, db=None) -> Optional[DocumentStore]:
    """Get a document store by name."""

    def _query(session) -> Optional[DocumentStore]:
        stmt = select(DocumentStore).where(DocumentStore.name == name.lower())
        return session.execute(stmt).scalar_one_or_none()

    if db:
        return _query(db)
    with get_db_context() as session:
        store = _query(session)
        return _store_to_detached(store, session) if store else None


def get_document_store_by_id(store_id: int, db=None) -> Optional[DocumentStore]:
    """Get a document store by ID."""

    def _query(session) -> Optional[DocumentStore]:
        stmt = select(DocumentStore).where(DocumentStore.id == store_id)
        return session.execute(stmt).scalar_one_or_none()

    if db:
        return _query(db)
    with get_db_context() as session:
        store = _query(session)
        return _store_to_detached(store, session) if store else None


def create_document_store(
    name: str,
    source_type: str = "local",
    source_path: Optional[str] = None,
    mcp_server_config: Optional[dict] = None,
    google_account_id: Optional[int] = None,
    gdrive_folder_id: Optional[str] = None,
    gdrive_folder_name: Optional[str] = None,
    gmail_label_id: Optional[str] = None,
    gmail_label_name: Optional[str] = None,
    gcalendar_calendar_id: Optional[str] = None,
    gcalendar_calendar_name: Optional[str] = None,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    vision_provider: str = "local",
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    index_schedule: Optional[str] = None,
    description: Optional[str] = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> DocumentStore:
    """
    Create a new document store.

    The collection_name is auto-generated as 'docstore_{id}' after creation.
    """

    def _create(session: Session) -> DocumentStore:
        store = DocumentStore(
            name=name.lower().strip(),
            source_type=source_type,
            source_path=source_path,
            mcp_server_config_json=json.dumps(mcp_server_config)
            if mcp_server_config
            else None,
            google_account_id=google_account_id,
            gdrive_folder_id=gdrive_folder_id,
            gdrive_folder_name=gdrive_folder_name,
            gmail_label_id=gmail_label_id,
            gmail_label_name=gmail_label_name,
            gcalendar_calendar_id=gcalendar_calendar_id,
            gcalendar_calendar_name=gcalendar_calendar_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_schedule=index_schedule,
            description=description,
            enabled=enabled,
        )
        session.add(store)
        session.flush()  # Get the ID

        # Auto-generate collection name
        store.collection_name = f"docstore_{store.id}"
        session.commit()

        logger.info(f"Created document store: {store.name} (ID: {store.id})")
        return store

    if db:
        return _create(db)
    with get_db_context() as session:
        store = _create(session)
        return _store_to_detached(store)


def update_document_store(
    store_id: int,
    name: Optional[str] = None,
    source_type: Optional[str] = None,
    source_path: Optional[str] = None,
    mcp_server_config: Optional[dict] = None,
    google_account_id: Optional[int] = None,
    gdrive_folder_id: Optional[str] = None,
    gdrive_folder_name: Optional[str] = None,
    gmail_label_id: Optional[str] = None,
    gmail_label_name: Optional[str] = None,
    gcalendar_calendar_id: Optional[str] = None,
    gcalendar_calendar_name: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    vision_provider: Optional[str] = None,
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    index_schedule: Optional[str] = None,
    description: Optional[str] = None,
    enabled: Optional[bool] = None,
    db: Optional[Session] = None,
) -> Optional[DocumentStore]:
    """Update a document store."""

    def _update(session: Session) -> Optional[DocumentStore]:
        stmt = select(DocumentStore).where(DocumentStore.id == store_id)
        store = session.execute(stmt).scalar_one_or_none()
        if not store:
            return None

        if name is not None:
            store.name = name.lower().strip()
        if source_type is not None:
            store.source_type = source_type
        if source_path is not None:
            store.source_path = source_path
        if mcp_server_config is not None:
            store.mcp_server_config_json = json.dumps(mcp_server_config)
        if google_account_id is not None:
            store.google_account_id = google_account_id
        if gdrive_folder_id is not None:
            store.gdrive_folder_id = gdrive_folder_id if gdrive_folder_id else None
        if gdrive_folder_name is not None:
            store.gdrive_folder_name = (
                gdrive_folder_name if gdrive_folder_name else None
            )
        if gmail_label_id is not None:
            store.gmail_label_id = gmail_label_id if gmail_label_id else None
        if gmail_label_name is not None:
            store.gmail_label_name = gmail_label_name if gmail_label_name else None
        if gcalendar_calendar_id is not None:
            store.gcalendar_calendar_id = (
                gcalendar_calendar_id if gcalendar_calendar_id else None
            )
        if gcalendar_calendar_name is not None:
            store.gcalendar_calendar_name = (
                gcalendar_calendar_name if gcalendar_calendar_name else None
            )
        if embedding_provider is not None:
            store.embedding_provider = embedding_provider
        if embedding_model is not None:
            store.embedding_model = embedding_model
        if ollama_url is not None:
            store.ollama_url = ollama_url
        if vision_provider is not None:
            store.vision_provider = vision_provider
        if vision_model is not None:
            store.vision_model = vision_model
        if vision_ollama_url is not None:
            store.vision_ollama_url = vision_ollama_url
        if chunk_size is not None:
            store.chunk_size = chunk_size
        if chunk_overlap is not None:
            store.chunk_overlap = chunk_overlap
        if index_schedule is not None:
            store.index_schedule = index_schedule
        if description is not None:
            store.description = description
        if enabled is not None:
            store.enabled = enabled

        store.updated_at = datetime.utcnow()
        session.commit()

        logger.info(f"Updated document store: {store.name} (ID: {store.id})")
        return store

    if db:
        return _update(db)
    with get_db_context() as session:
        store = _update(session)
        return _store_to_detached(store) if store else None


def update_document_store_index_status(
    store_id: int,
    status: str,
    error: Optional[str] = None,
    document_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    collection_name: Optional[str] = None,
    db: Optional[Session] = None,
) -> bool:
    """Update store index status and statistics."""

    def _update(session: Session) -> bool:
        stmt = select(DocumentStore).where(DocumentStore.id == store_id)
        store = session.execute(stmt).scalar_one_or_none()
        if not store:
            return False

        store.index_status = status
        store.index_error = error

        if status == "ready":
            store.last_indexed = datetime.utcnow()

        if document_count is not None:
            store.document_count = document_count
        if chunk_count is not None:
            store.chunk_count = chunk_count
        if collection_name is not None:
            store.collection_name = collection_name

        store.updated_at = datetime.utcnow()
        session.commit()
        return True

    if db:
        return _update(db)
    with get_db_context() as session:
        return _update(session)


def delete_document_store(store_id: int, db=None) -> bool:
    """
    Delete a document store.

    Fails if any Enrichers are currently linked to it.
    Note: Does NOT delete the ChromaDB collection - that should be handled separately.
    """

    def _delete(session) -> bool:
        stmt = select(DocumentStore).where(DocumentStore.id == store_id)
        store = session.execute(stmt).scalar_one_or_none()
        if not store:
            return False

        # Check if any enrichers are using this store via junction table
        from sqlalchemy import func

        enricher_count = session.execute(
            select(func.count())
            .select_from(smart_enricher_stores)
            .where(smart_enricher_stores.c.document_store_id == store_id)
        ).scalar()

        if enricher_count > 0:
            logger.error(
                f"Cannot delete store {store.name}: used by {enricher_count} enricher(s)"
            )
            return False

        session.delete(store)
        session.commit()
        logger.info(f"Deleted document store: {store.name} (ID: {store_id})")
        return True

    if db:
        return _delete(db)
    with get_db_context() as session:
        return _delete(session)


def store_name_available(
    name: str, exclude_id: Optional[int] = None, db: Optional[Session] = None
) -> bool:
    """Check if a store name is available."""

    def _check(session: Session) -> bool:
        stmt = select(DocumentStore).where(DocumentStore.name == name.lower().strip())
        if exclude_id:
            stmt = stmt.where(DocumentStore.id != exclude_id)
        return session.execute(stmt).scalar_one_or_none() is None

    if db:
        return _check(db)
    with get_db_context() as session:
        return _check(session)


def get_enabled_document_stores(
    db: Optional[Session] = None,
) -> dict[str, DocumentStore]:
    """Get all enabled document stores as a dict keyed by name."""

    def _query(session: Session) -> dict[str, DocumentStore]:
        stmt = select(DocumentStore).where(DocumentStore.enabled == True)
        stores = session.execute(stmt).scalars().all()
        return {s.name: s for s in stores}

    if db:
        return _query(db)
    with get_db_context() as session:
        stores_dict = _query(session)
        return {name: _store_to_detached(s) for name, s in stores_dict.items()}


def get_stores_with_schedule(db: Optional[Session] = None) -> list[DocumentStore]:
    """Get all enabled document stores that have an index schedule."""

    def _query(session: Session) -> list[DocumentStore]:
        stmt = (
            select(DocumentStore)
            .where(DocumentStore.enabled == True)
            .where(DocumentStore.index_schedule.isnot(None))
            .where(DocumentStore.index_schedule != "")
        )
        return list(session.execute(stmt).scalars().all())

    if db:
        return _query(db)
    with get_db_context() as session:
        stores = _query(session)
        return [_store_to_detached(s) for s in stores]


def get_stores_for_enricher(
    enricher_id: int, db: Optional[Session] = None
) -> list[DocumentStore]:
    """Get all document stores linked to an Enricher."""

    def _query(session: Session) -> list[DocumentStore]:
        stmt = (
            select(DocumentStore)
            .join(smart_enricher_stores)
            .where(smart_enricher_stores.c.smart_enricher_id == enricher_id)
            .order_by(DocumentStore.name)
        )
        return list(session.execute(stmt).scalars().all())

    if db:
        return _query(db)
    with get_db_context() as session:
        stores = _query(session)
        return [_store_to_detached(s) for s in stores]


def link_store_to_enricher(
    store_id: int, enricher_id: int, db: Optional[Session] = None
) -> bool:
    """Link a document store to an Enricher."""

    def _link(session: Session) -> bool:
        # Verify both exist
        store = session.get(DocumentStore, store_id)
        enricher = session.get(SmartEnricher, enricher_id)
        if not store or not enricher:
            return False

        # Check if already linked
        if store in enricher.document_stores:
            return True  # Already linked

        enricher.document_stores.append(store)
        session.commit()
        logger.info(f"Linked store {store.name} to Enricher {enricher.name}")
        return True

    if db:
        return _link(db)
    with get_db_context() as session:
        return _link(session)


def unlink_store_from_enricher(
    store_id: int, enricher_id: int, db: Optional[Session] = None
) -> bool:
    """Unlink a document store from an Enricher."""

    def _unlink(session: Session) -> bool:
        # Verify both exist
        store = session.get(DocumentStore, store_id)
        enricher = session.get(SmartEnricher, enricher_id)
        if not store or not enricher:
            return False

        # Check if linked
        if store not in enricher.document_stores:
            return True  # Already unlinked

        enricher.document_stores.remove(store)
        session.commit()
        logger.info(f"Unlinked store {store.name} from Enricher {enricher.name}")
        return True

    if db:
        return _unlink(db)
    with get_db_context() as session:
        return _unlink(session)


def set_enricher_stores(
    enricher_id: int, store_ids: list[int], db: Optional[Session] = None
) -> bool:
    """
    Set the document stores for an Enricher (replaces existing links).

    Args:
        enricher_id: The Enricher ID
        store_ids: List of store IDs to link (replaces current links)

    Returns:
        True if successful, False if Enricher not found
    """

    def _set(session: Session) -> bool:
        enricher = session.get(SmartEnricher, enricher_id)
        if not enricher:
            return False

        # Get the stores
        stores = []
        for store_id in store_ids:
            store = session.get(DocumentStore, store_id)
            if store:
                stores.append(store)
            else:
                logger.warning(f"Store {store_id} not found, skipping")

        # Replace the relationship
        enricher.document_stores = stores
        session.commit()

        logger.info(
            f"Set Enricher {enricher.name} stores to: {[s.name for s in stores]}"
        )
        return True

    if db:
        return _set(db)
    with get_db_context() as session:
        return _set(session)


def _store_to_detached(
    store: Optional[DocumentStore], session=None
) -> Optional[DocumentStore]:
    """
    Create a detached copy of a DocumentStore for use outside sessions.

    This copies all attributes to a new instance that isn't bound to any session.
    """
    if not store:
        return None

    detached = DocumentStore(
        id=store.id,
        name=store.name,
        source_type=store.source_type,
        source_path=store.source_path,
        mcp_server_config_json=store.mcp_server_config_json,
        google_account_id=store.google_account_id,
        gdrive_folder_id=store.gdrive_folder_id,
        gdrive_folder_name=store.gdrive_folder_name,
        gmail_label_id=store.gmail_label_id,
        gmail_label_name=store.gmail_label_name,
        gcalendar_calendar_id=store.gcalendar_calendar_id,
        gcalendar_calendar_name=store.gcalendar_calendar_name,
        embedding_provider=store.embedding_provider,
        embedding_model=store.embedding_model,
        ollama_url=store.ollama_url,
        vision_provider=store.vision_provider,
        vision_model=store.vision_model,
        vision_ollama_url=store.vision_ollama_url,
        chunk_size=store.chunk_size,
        chunk_overlap=store.chunk_overlap,
        index_schedule=store.index_schedule,
        last_indexed=store.last_indexed,
        index_status=store.index_status,
        index_error=store.index_error,
        document_count=store.document_count,
        chunk_count=store.chunk_count,
        collection_name=store.collection_name,
        description=store.description,
        enabled=store.enabled,
        created_at=store.created_at,
        updated_at=store.updated_at,
    )

    # Get enricher count via junction table query if session is provided
    if session:
        from sqlalchemy import func

        enricher_count = session.execute(
            select(func.count())
            .select_from(smart_enricher_stores)
            .where(smart_enricher_stores.c.document_store_id == store.id)
        ).scalar()
        detached._enricher_count = enricher_count
    else:
        detached._enricher_count = 0

    return detached
