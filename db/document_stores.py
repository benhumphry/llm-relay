"""
Document Store CRUD operations for LLM Relay.

Handles creation, retrieval, updating, and deletion of document stores.
Document stores are linked to Smart Aliases via the smart_alias_stores junction table.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import DocumentStore, smart_alias_stores

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
    display_name: Optional[str] = None,
    source_type: str = "local",
    plugin_config_id: Optional[int] = None,
    source_path: Optional[str] = None,
    mcp_server_config: Optional[dict] = None,
    google_account_id: Optional[int] = None,
    gdrive_folder_id: Optional[str] = None,
    gdrive_folder_name: Optional[str] = None,
    gmail_label_id: Optional[str] = None,
    gmail_label_name: Optional[str] = None,
    gcalendar_calendar_id: Optional[str] = None,
    gcalendar_calendar_name: Optional[str] = None,
    gtasks_tasklist_id: Optional[str] = None,
    gtasks_tasklist_name: Optional[str] = None,
    gcontacts_group_id: Optional[str] = None,
    gcontacts_group_name: Optional[str] = None,
    microsoft_account_id: Optional[int] = None,
    onedrive_folder_id: Optional[str] = None,
    onedrive_folder_name: Optional[str] = None,
    outlook_folder_id: Optional[str] = None,
    outlook_folder_name: Optional[str] = None,
    outlook_days_back: int = 90,
    onenote_notebook_id: Optional[str] = None,
    onenote_notebook_name: Optional[str] = None,
    teams_team_id: Optional[str] = None,
    teams_team_name: Optional[str] = None,
    teams_channel_id: Optional[str] = None,
    teams_channel_name: Optional[str] = None,
    teams_days_back: int = 90,
    paperless_url: Optional[str] = None,
    paperless_token: Optional[str] = None,
    paperless_tag_id: Optional[int] = None,
    paperless_tag_name: Optional[str] = None,
    github_repo: Optional[str] = None,
    github_branch: Optional[str] = None,
    github_path: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    notion_page_id: Optional[str] = None,
    nextcloud_folder: Optional[str] = None,
    website_url: Optional[str] = None,
    website_crawl_depth: int = 1,
    website_max_pages: int = 50,
    website_include_pattern: Optional[str] = None,
    website_exclude_pattern: Optional[str] = None,
    website_crawler_override: Optional[str] = None,
    slack_channel_id: Optional[str] = None,
    slack_channel_types: str = "public_channel",
    slack_days_back: int = 90,
    todoist_project_id: Optional[str] = None,
    todoist_project_name: Optional[str] = None,
    todoist_filter: Optional[str] = None,
    todoist_include_completed: bool = False,
    websearch_query: Optional[str] = None,
    websearch_max_results: int = 10,
    websearch_pages_to_scrape: int = 5,
    websearch_time_range: Optional[str] = None,
    websearch_category: Optional[str] = None,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    vision_provider: str = "local",
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    index_schedule: Optional[str] = None,
    max_documents: Optional[int] = None,
    description: Optional[str] = None,
    enabled: bool = True,
    use_temporal_filtering: bool = False,
    db: Optional[Session] = None,
) -> DocumentStore:
    """
    Create a new document store.

    The collection_name is auto-generated as 'docstore_{id}' after creation.
    """

    def _create(session: Session) -> DocumentStore:
        store = DocumentStore(
            name=name.lower().strip(),
            display_name=display_name.strip() if display_name else None,
            source_type=source_type,
            plugin_config_id=plugin_config_id,
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
            gtasks_tasklist_id=gtasks_tasklist_id,
            gtasks_tasklist_name=gtasks_tasklist_name,
            gcontacts_group_id=gcontacts_group_id,
            gcontacts_group_name=gcontacts_group_name,
            microsoft_account_id=microsoft_account_id,
            onedrive_folder_id=onedrive_folder_id,
            onedrive_folder_name=onedrive_folder_name,
            outlook_folder_id=outlook_folder_id,
            outlook_folder_name=outlook_folder_name,
            outlook_days_back=outlook_days_back,
            onenote_notebook_id=onenote_notebook_id,
            onenote_notebook_name=onenote_notebook_name,
            teams_team_id=teams_team_id,
            teams_team_name=teams_team_name,
            teams_channel_id=teams_channel_id,
            teams_channel_name=teams_channel_name,
            teams_days_back=teams_days_back,
            paperless_url=paperless_url,
            paperless_token=paperless_token,
            paperless_tag_id=paperless_tag_id,
            paperless_tag_name=paperless_tag_name,
            github_repo=github_repo,
            github_branch=github_branch,
            github_path=github_path,
            notion_database_id=notion_database_id,
            notion_page_id=notion_page_id,
            nextcloud_folder=nextcloud_folder,
            website_url=website_url,
            website_crawl_depth=website_crawl_depth,
            website_max_pages=website_max_pages,
            website_include_pattern=website_include_pattern,
            website_exclude_pattern=website_exclude_pattern,
            website_crawler_override=website_crawler_override,
            slack_channel_id=slack_channel_id,
            slack_channel_types=slack_channel_types,
            slack_days_back=slack_days_back,
            todoist_project_id=todoist_project_id,
            todoist_project_name=todoist_project_name,
            todoist_filter=todoist_filter,
            todoist_include_completed=todoist_include_completed,
            websearch_query=websearch_query,
            websearch_max_results=websearch_max_results,
            websearch_pages_to_scrape=websearch_pages_to_scrape,
            websearch_time_range=websearch_time_range,
            websearch_category=websearch_category,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_schedule=index_schedule,
            max_documents=max_documents,
            description=description,
            enabled=enabled,
            use_temporal_filtering=use_temporal_filtering,
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
    display_name: Optional[str] = None,
    source_type: Optional[str] = None,
    plugin_config_id: Optional[int] = None,
    source_path: Optional[str] = None,
    mcp_server_config: Optional[dict] = None,
    google_account_id: Optional[int] = None,
    gdrive_folder_id: Optional[str] = None,
    gdrive_folder_name: Optional[str] = None,
    gmail_label_id: Optional[str] = None,
    gmail_label_name: Optional[str] = None,
    gcalendar_calendar_id: Optional[str] = None,
    gcalendar_calendar_name: Optional[str] = None,
    gtasks_tasklist_id: Optional[str] = None,
    gtasks_tasklist_name: Optional[str] = None,
    gcontacts_group_id: Optional[str] = None,
    gcontacts_group_name: Optional[str] = None,
    microsoft_account_id: Optional[int] = None,
    onedrive_folder_id: Optional[str] = None,
    onedrive_folder_name: Optional[str] = None,
    outlook_folder_id: Optional[str] = None,
    outlook_folder_name: Optional[str] = None,
    outlook_days_back: Optional[int] = None,
    onenote_notebook_id: Optional[str] = None,
    onenote_notebook_name: Optional[str] = None,
    teams_team_id: Optional[str] = None,
    teams_team_name: Optional[str] = None,
    teams_channel_id: Optional[str] = None,
    teams_channel_name: Optional[str] = None,
    teams_days_back: Optional[int] = None,
    paperless_url: Optional[str] = None,
    paperless_token: Optional[str] = None,
    paperless_tag_id: Optional[int] = None,
    paperless_tag_name: Optional[str] = None,
    github_repo: Optional[str] = None,
    github_branch: Optional[str] = None,
    github_path: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    notion_page_id: Optional[str] = None,
    nextcloud_folder: Optional[str] = None,
    website_url: Optional[str] = None,
    website_crawl_depth: Optional[int] = None,
    website_max_pages: Optional[int] = None,
    website_include_pattern: Optional[str] = None,
    website_exclude_pattern: Optional[str] = None,
    website_crawler_override: Optional[str] = None,
    slack_channel_id: Optional[str] = None,
    slack_channel_types: Optional[str] = None,
    slack_days_back: Optional[int] = None,
    todoist_project_id: Optional[str] = None,
    todoist_project_name: Optional[str] = None,
    todoist_filter: Optional[str] = None,
    todoist_include_completed: Optional[bool] = None,
    websearch_query: Optional[str] = None,
    websearch_max_results: Optional[int] = None,
    websearch_pages_to_scrape: Optional[int] = None,
    websearch_time_range: Optional[str] = None,
    websearch_category: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    vision_provider: Optional[str] = None,
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    index_schedule: Optional[str] = None,
    max_documents: Optional[int] = None,
    description: Optional[str] = None,
    enabled: Optional[bool] = None,
    use_temporal_filtering: Optional[bool] = None,
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
        if display_name is not None:
            store.display_name = display_name.strip() if display_name else None
        if source_type is not None:
            store.source_type = source_type
        if plugin_config_id is not None:
            store.plugin_config_id = plugin_config_id if plugin_config_id else None
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
        if gtasks_tasklist_id is not None:
            store.gtasks_tasklist_id = (
                gtasks_tasklist_id if gtasks_tasklist_id else None
            )
        if gtasks_tasklist_name is not None:
            store.gtasks_tasklist_name = (
                gtasks_tasklist_name if gtasks_tasklist_name else None
            )
        if gcontacts_group_id is not None:
            store.gcontacts_group_id = (
                gcontacts_group_id if gcontacts_group_id else None
            )
        if gcontacts_group_name is not None:
            store.gcontacts_group_name = (
                gcontacts_group_name if gcontacts_group_name else None
            )
        if microsoft_account_id is not None:
            store.microsoft_account_id = microsoft_account_id
        if onedrive_folder_id is not None:
            store.onedrive_folder_id = (
                onedrive_folder_id if onedrive_folder_id else None
            )
        if onedrive_folder_name is not None:
            store.onedrive_folder_name = (
                onedrive_folder_name if onedrive_folder_name else None
            )
        if outlook_folder_id is not None:
            store.outlook_folder_id = outlook_folder_id if outlook_folder_id else None
        if outlook_folder_name is not None:
            store.outlook_folder_name = (
                outlook_folder_name if outlook_folder_name else None
            )
        if outlook_days_back is not None:
            store.outlook_days_back = outlook_days_back
        if onenote_notebook_id is not None:
            store.onenote_notebook_id = (
                onenote_notebook_id if onenote_notebook_id else None
            )
        if onenote_notebook_name is not None:
            store.onenote_notebook_name = (
                onenote_notebook_name if onenote_notebook_name else None
            )
        if teams_team_id is not None:
            store.teams_team_id = teams_team_id if teams_team_id else None
        if teams_team_name is not None:
            store.teams_team_name = teams_team_name if teams_team_name else None
        if teams_channel_id is not None:
            store.teams_channel_id = teams_channel_id if teams_channel_id else None
        if teams_channel_name is not None:
            store.teams_channel_name = (
                teams_channel_name if teams_channel_name else None
            )
        if teams_days_back is not None:
            store.teams_days_back = teams_days_back
        if paperless_url is not None:
            store.paperless_url = paperless_url if paperless_url else None
        if paperless_token is not None:
            store.paperless_token = paperless_token if paperless_token else None
        if paperless_tag_id is not None:
            store.paperless_tag_id = paperless_tag_id if paperless_tag_id else None
        if paperless_tag_name is not None:
            store.paperless_tag_name = (
                paperless_tag_name if paperless_tag_name else None
            )
        if github_repo is not None:
            store.github_repo = github_repo if github_repo else None
        if github_branch is not None:
            store.github_branch = github_branch if github_branch else None
        if github_path is not None:
            store.github_path = github_path if github_path else None
        if notion_database_id is not None:
            store.notion_database_id = (
                notion_database_id if notion_database_id else None
            )
        if notion_page_id is not None:
            store.notion_page_id = notion_page_id if notion_page_id else None
        if nextcloud_folder is not None:
            store.nextcloud_folder = nextcloud_folder if nextcloud_folder else None
        if website_url is not None:
            store.website_url = website_url if website_url else None
        if website_crawl_depth is not None:
            store.website_crawl_depth = website_crawl_depth
        if website_max_pages is not None:
            store.website_max_pages = website_max_pages
        if website_include_pattern is not None:
            store.website_include_pattern = (
                website_include_pattern if website_include_pattern else None
            )
        if website_exclude_pattern is not None:
            store.website_exclude_pattern = (
                website_exclude_pattern if website_exclude_pattern else None
            )
        if website_crawler_override is not None:
            store.website_crawler_override = (
                website_crawler_override if website_crawler_override else None
            )
        if slack_channel_id is not None:
            store.slack_channel_id = slack_channel_id if slack_channel_id else None
        if slack_channel_types is not None:
            store.slack_channel_types = slack_channel_types
        if slack_days_back is not None:
            store.slack_days_back = slack_days_back
        if todoist_project_id is not None:
            store.todoist_project_id = (
                todoist_project_id if todoist_project_id else None
            )
        if todoist_project_name is not None:
            store.todoist_project_name = (
                todoist_project_name if todoist_project_name else None
            )
        if todoist_filter is not None:
            store.todoist_filter = todoist_filter if todoist_filter else None
        if todoist_include_completed is not None:
            store.todoist_include_completed = todoist_include_completed
        if websearch_query is not None:
            store.websearch_query = websearch_query if websearch_query else None
        if websearch_max_results is not None:
            store.websearch_max_results = websearch_max_results
        if websearch_pages_to_scrape is not None:
            store.websearch_pages_to_scrape = websearch_pages_to_scrape
        if websearch_time_range is not None:
            store.websearch_time_range = (
                websearch_time_range if websearch_time_range else None
            )
        if websearch_category is not None:
            store.websearch_category = (
                websearch_category if websearch_category else None
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
        if max_documents is not None:
            # 0 or None means no limit
            store.max_documents = max_documents if max_documents > 0 else None
        if description is not None:
            store.description = description
        if enabled is not None:
            store.enabled = enabled
        if use_temporal_filtering is not None:
            store.use_temporal_filtering = use_temporal_filtering

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

    Fails if any Smart Aliases are currently linked to it.
    Note: Does NOT delete the ChromaDB collection - that should be handled separately.
    """

    def _delete(session) -> bool:
        stmt = select(DocumentStore).where(DocumentStore.id == store_id)
        store = session.execute(stmt).scalar_one_or_none()
        if not store:
            return False

        # Check if any Smart Aliases are using this store via junction table
        from sqlalchemy import func

        alias_count = session.execute(
            select(func.count())
            .select_from(smart_alias_stores)
            .where(smart_alias_stores.c.document_store_id == store_id)
        ).scalar()

        if alias_count > 0:
            logger.error(
                f"Cannot delete store {store.name}: used by {alias_count} Smart Alias(es)"
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
        display_name=store.display_name,
        source_type=store.source_type,
        plugin_config_id=store.plugin_config_id,
        source_path=store.source_path,
        mcp_server_config_json=store.mcp_server_config_json,
        google_account_id=store.google_account_id,
        gdrive_folder_id=store.gdrive_folder_id,
        gdrive_folder_name=store.gdrive_folder_name,
        gmail_label_id=store.gmail_label_id,
        gmail_label_name=store.gmail_label_name,
        gcalendar_calendar_id=store.gcalendar_calendar_id,
        gcalendar_calendar_name=store.gcalendar_calendar_name,
        gtasks_tasklist_id=store.gtasks_tasklist_id,
        gtasks_tasklist_name=store.gtasks_tasklist_name,
        gcontacts_group_id=store.gcontacts_group_id,
        gcontacts_group_name=store.gcontacts_group_name,
        microsoft_account_id=store.microsoft_account_id,
        onedrive_folder_id=store.onedrive_folder_id,
        onedrive_folder_name=store.onedrive_folder_name,
        outlook_folder_id=store.outlook_folder_id,
        outlook_folder_name=store.outlook_folder_name,
        outlook_days_back=store.outlook_days_back,
        onenote_notebook_id=store.onenote_notebook_id,
        onenote_notebook_name=store.onenote_notebook_name,
        teams_team_id=store.teams_team_id,
        teams_team_name=store.teams_team_name,
        teams_channel_id=store.teams_channel_id,
        teams_channel_name=store.teams_channel_name,
        teams_days_back=store.teams_days_back,
        paperless_url=store.paperless_url,
        paperless_token=store.paperless_token,
        paperless_tag_id=store.paperless_tag_id,
        paperless_tag_name=store.paperless_tag_name,
        github_repo=store.github_repo,
        github_branch=store.github_branch,
        github_path=store.github_path,
        notion_database_id=store.notion_database_id,
        notion_page_id=store.notion_page_id,
        nextcloud_folder=store.nextcloud_folder,
        website_url=store.website_url,
        website_crawl_depth=store.website_crawl_depth,
        website_max_pages=store.website_max_pages,
        website_include_pattern=store.website_include_pattern,
        website_exclude_pattern=store.website_exclude_pattern,
        website_crawler_override=store.website_crawler_override,
        slack_channel_id=store.slack_channel_id,
        slack_channel_types=store.slack_channel_types,
        slack_days_back=store.slack_days_back,
        todoist_project_id=store.todoist_project_id,
        todoist_project_name=store.todoist_project_name,
        todoist_filter=store.todoist_filter,
        todoist_include_completed=store.todoist_include_completed,
        websearch_query=store.websearch_query,
        websearch_max_results=store.websearch_max_results,
        websearch_pages_to_scrape=store.websearch_pages_to_scrape,
        websearch_time_range=store.websearch_time_range,
        websearch_category=store.websearch_category,
        embedding_provider=store.embedding_provider,
        embedding_model=store.embedding_model,
        ollama_url=store.ollama_url,
        vision_provider=store.vision_provider,
        vision_model=store.vision_model,
        vision_ollama_url=store.vision_ollama_url,
        chunk_size=store.chunk_size,
        chunk_overlap=store.chunk_overlap,
        index_schedule=store.index_schedule,
        max_documents=store.max_documents,
        last_indexed=store.last_indexed,
        index_status=store.index_status,
        index_error=store.index_error,
        document_count=store.document_count,
        chunk_count=store.chunk_count,
        collection_name=store.collection_name,
        # Intelligence fields
        themes_json=store.themes_json,
        best_for=store.best_for,
        content_summary=store.content_summary,
        intelligence_updated_at=store.intelligence_updated_at,
        # Metadata
        description=store.description,
        enabled=store.enabled,
        use_temporal_filtering=store.use_temporal_filtering,
        created_at=store.created_at,
        updated_at=store.updated_at,
    )

    # Get alias count via junction table query if session is provided
    if session:
        from sqlalchemy import func

        alias_count = session.execute(
            select(func.count())
            .select_from(smart_alias_stores)
            .where(smart_alias_stores.c.document_store_id == store.id)
        ).scalar()
        detached._alias_count = alias_count
    else:
        detached._alias_count = 0

    return detached
