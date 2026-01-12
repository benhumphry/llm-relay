"""
Smart Alias CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart aliases.
Smart aliases unify routing, enrichment (RAG + Web), and caching into a single concept.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from .connection import get_db_context
from .models import DocumentStore, SmartAlias, smart_alias_stores

logger = logging.getLogger(__name__)


def get_all_smart_aliases(db: Optional[Session] = None) -> list[SmartAlias]:
    """Get all smart aliases from the database."""
    if db:
        return (
            db.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .order_by(SmartAlias.name)
            .all()
        )

    with get_db_context() as session:
        aliases = (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .order_by(SmartAlias.name)
            .all()
        )
        return [_alias_to_detached(a) for a in aliases]


def get_smart_alias_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartAlias]:
    """Get a smart alias by its name."""
    if db:
        return (
            db.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.name == name)
            .first()
        )

    with get_db_context() as session:
        alias = (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.name == name)
            .first()
        )
        return _alias_to_detached(alias) if alias else None


def get_smart_tag_by_name(
    tag_name: str, db: Optional[Session] = None
) -> Optional[SmartAlias]:
    """
    Get a smart alias that is configured as a smart tag with the given name.

    Only returns aliases where is_smart_tag=True and enabled=True.
    """
    if db:
        return (
            db.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(
                SmartAlias.name == tag_name,
                SmartAlias.is_smart_tag == True,
                SmartAlias.enabled == True,
            )
            .first()
        )

    with get_db_context() as session:
        alias = (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(
                SmartAlias.name == tag_name,
                SmartAlias.is_smart_tag == True,
                SmartAlias.enabled == True,
            )
            .first()
        )
        return _alias_to_detached(alias) if alias else None


def get_smart_alias_by_id(
    alias_id: int, db: Optional[Session] = None
) -> Optional[SmartAlias]:
    """Get a smart alias by its ID."""
    if db:
        return (
            db.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.id == alias_id)
            .first()
        )

    with get_db_context() as session:
        alias = (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.id == alias_id)
            .first()
        )
        return _alias_to_detached(alias) if alias else None


def create_smart_alias(
    name: str,
    target_model: str,
    # Feature toggles
    use_routing: bool = False,
    use_rag: bool = False,
    use_web: bool = False,
    use_cache: bool = False,
    # Smart tag settings
    is_smart_tag: bool = False,
    passthrough_model: bool = False,
    # Routing settings
    designator_model: str | None = None,
    purpose: str | None = None,
    candidates: list[dict] | None = None,
    fallback_model: str | None = None,
    routing_strategy: str = "per_request",
    session_ttl: int = 3600,
    use_model_intelligence: bool = False,
    search_provider: str | None = None,
    intelligence_model: str | None = None,
    # RAG settings
    store_ids: list[int] | None = None,
    max_results: int = 5,
    similarity_threshold: float = 0.7,
    # Web settings
    max_search_results: int = 5,
    max_scrape_urls: int = 3,
    # Common enrichment settings
    max_context_tokens: int = 4000,
    rerank_provider: str = "local",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 20,
    context_priority: str = "balanced",
    # Cache settings
    cache_similarity_threshold: float = 0.95,
    cache_match_system_prompt: bool = True,
    cache_match_last_message_only: bool = False,
    cache_ttl_hours: int = 168,
    cache_min_tokens: int = 50,
    cache_max_tokens: int = 4000,
    cache_collection: str | None = None,
    # Metadata
    tags: list[str] | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> SmartAlias:
    """
    Create a new smart alias.

    Args:
        name: Unique name for the alias
        target_model: Default/fallback model ("provider/model")
        use_routing: Enable smart routing
        use_rag: Enable RAG document retrieval
        use_web: Enable realtime web search
        use_cache: Enable response caching (ignored if use_web=True)
        designator_model: Model for routing/web query generation
        purpose: Context for routing decisions
        candidates: List of candidate models for routing
        fallback_model: Fallback model if routing fails
        routing_strategy: "per_request" or "per_session"
        session_ttl: Session cache TTL in seconds
        use_model_intelligence: Enable model intelligence for routing
        system_prompt: Custom system prompt/context to inject
        search_provider: Search provider for model intelligence
        intelligence_model: Model for summarizing intelligence
        store_ids: List of DocumentStore IDs for RAG
        max_results: Max RAG chunks to retrieve
        similarity_threshold: Min similarity for RAG (0.0-1.0)
        max_search_results: Max web search results
        max_scrape_urls: Max URLs to scrape
        max_context_tokens: Max context tokens to inject
        rerank_provider: Reranking provider
        rerank_model: Reranking model name
        rerank_top_n: Results to fetch before reranking
        cache_similarity_threshold: Cache similarity threshold
        cache_match_system_prompt: Include system prompt in cache key
        cache_match_last_message_only: Only match last message
        cache_ttl_hours: Cache TTL in hours
        cache_min_tokens: Min tokens to cache
        cache_max_tokens: Max tokens to cache
        cache_collection: ChromaDB collection name
        tags: Tags for usage tracking
        description: Description
        enabled: Whether alias is enabled

    Returns:
        Created SmartAlias object

    Raises:
        ValueError: If name already exists or invalid combination
    """
    # Validate: cache + web is not allowed
    if use_cache and use_web:
        raise ValueError("Cannot enable caching with realtime web search")

    def _create(session: Session) -> SmartAlias:
        # Check for existing alias with same name
        existing = session.query(SmartAlias).filter(SmartAlias.name == name).first()
        if existing:
            raise ValueError(f"Smart alias with name '{name}' already exists")

        alias = SmartAlias(
            name=name,
            target_model=target_model,
            # Feature toggles
            use_routing=use_routing,
            use_rag=use_rag,
            use_web=use_web,
            use_cache=use_cache,
            # Smart tag
            is_smart_tag=is_smart_tag,
            passthrough_model=passthrough_model,
            # Routing
            designator_model=designator_model,
            purpose=purpose,
            fallback_model=fallback_model,
            routing_strategy=routing_strategy,
            session_ttl=session_ttl,
            use_model_intelligence=use_model_intelligence,
            search_provider=search_provider,
            intelligence_model=intelligence_model,
            # RAG
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            # Web
            max_search_results=max_search_results,
            max_scrape_urls=max_scrape_urls,
            # Common enrichment
            max_context_tokens=max_context_tokens,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
            context_priority=context_priority,
            # Cache
            cache_similarity_threshold=cache_similarity_threshold,
            cache_match_system_prompt=cache_match_system_prompt,
            cache_match_last_message_only=cache_match_last_message_only,
            cache_ttl_hours=cache_ttl_hours,
            cache_min_tokens=cache_min_tokens,
            cache_max_tokens=cache_max_tokens,
            cache_collection=cache_collection,
            # Metadata
            description=description,
            system_prompt=system_prompt,
            enabled=enabled,
        )

        if candidates:
            alias.candidates = candidates
        if tags:
            alias.tags = tags

        session.add(alias)
        session.flush()

        # Link to document stores if provided
        if store_ids:
            stores = (
                session.query(DocumentStore)
                .filter(DocumentStore.id.in_(store_ids))
                .all()
            )
            alias.document_stores = stores
            session.flush()

        logger.info(f"Created smart alias: {name}")
        return alias

    if db:
        return _create(db)

    with get_db_context() as session:
        alias = _create(session)
        session.refresh(alias)
        return _alias_to_detached(alias)


def update_smart_alias(
    alias_id: int,
    name: str | None = None,
    target_model: str | None = None,
    # Feature toggles
    use_routing: bool | None = None,
    use_rag: bool | None = None,
    use_web: bool | None = None,
    use_cache: bool | None = None,
    # Smart tag settings
    is_smart_tag: bool | None = None,
    passthrough_model: bool | None = None,
    # Routing settings
    designator_model: str | None = None,
    purpose: str | None = None,
    candidates: list[dict] | None = None,
    fallback_model: str | None = None,
    routing_strategy: str | None = None,
    session_ttl: int | None = None,
    use_model_intelligence: bool | None = None,
    search_provider: str | None = None,
    intelligence_model: str | None = None,
    # RAG settings
    store_ids: list[int] | None = None,
    max_results: int | None = None,
    similarity_threshold: float | None = None,
    # Web settings
    max_search_results: int | None = None,
    max_scrape_urls: int | None = None,
    # Common enrichment settings
    max_context_tokens: int | None = None,
    rerank_provider: str | None = None,
    rerank_model: str | None = None,
    rerank_top_n: int | None = None,
    context_priority: str | None = None,
    # Cache settings
    cache_similarity_threshold: float | None = None,
    cache_match_system_prompt: bool | None = None,
    cache_match_last_message_only: bool | None = None,
    cache_ttl_hours: int | None = None,
    cache_min_tokens: int | None = None,
    cache_max_tokens: int | None = None,
    cache_collection: str | None = None,
    # Metadata
    tags: list[str] | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    enabled: bool | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartAlias]:
    """
    Update an existing smart alias.

    Returns:
        Updated SmartAlias or None if not found

    Raises:
        ValueError: If name conflicts or invalid combination
    """

    def _update(session: Session) -> Optional[SmartAlias]:
        alias = (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.id == alias_id)
            .first()
        )
        if not alias:
            return None

        # Check name uniqueness if changing
        if name and name != alias.name:
            existing = session.query(SmartAlias).filter(SmartAlias.name == name).first()
            if existing:
                raise ValueError(f"Smart alias with name '{name}' already exists")
            alias.name = name

        # Determine final use_web and use_cache values for validation
        final_use_web = use_web if use_web is not None else alias.use_web
        final_use_cache = use_cache if use_cache is not None else alias.use_cache
        if final_use_cache and final_use_web:
            raise ValueError("Cannot enable caching with realtime web search")

        # Apply updates
        if target_model is not None:
            alias.target_model = target_model

        # Feature toggles
        if use_routing is not None:
            alias.use_routing = use_routing
        if use_rag is not None:
            alias.use_rag = use_rag
        if use_web is not None:
            alias.use_web = use_web
        if use_cache is not None:
            alias.use_cache = use_cache

        # Smart tag
        if is_smart_tag is not None:
            alias.is_smart_tag = is_smart_tag
        if passthrough_model is not None:
            alias.passthrough_model = passthrough_model

        # Routing
        if designator_model is not None:
            alias.designator_model = designator_model
        if purpose is not None:
            alias.purpose = purpose
        if candidates is not None:
            alias.candidates = candidates
        if fallback_model is not None:
            alias.fallback_model = fallback_model
        if routing_strategy is not None:
            alias.routing_strategy = routing_strategy
        if session_ttl is not None:
            alias.session_ttl = session_ttl
        if use_model_intelligence is not None:
            alias.use_model_intelligence = use_model_intelligence
        if search_provider is not None:
            alias.search_provider = search_provider
        if intelligence_model is not None:
            alias.intelligence_model = intelligence_model

        # RAG
        if store_ids is not None:
            stores = (
                session.query(DocumentStore)
                .filter(DocumentStore.id.in_(store_ids))
                .all()
            )
            alias.document_stores = stores
            session.flush()
        if max_results is not None:
            alias.max_results = max_results
        if similarity_threshold is not None:
            alias.similarity_threshold = similarity_threshold

        # Web
        if max_search_results is not None:
            alias.max_search_results = max_search_results
        if max_scrape_urls is not None:
            alias.max_scrape_urls = max_scrape_urls

        # Common enrichment
        if max_context_tokens is not None:
            alias.max_context_tokens = max_context_tokens
        if rerank_provider is not None:
            alias.rerank_provider = rerank_provider
        if rerank_model is not None:
            alias.rerank_model = rerank_model
        if rerank_top_n is not None:
            alias.rerank_top_n = rerank_top_n
        if context_priority is not None:
            alias.context_priority = context_priority

        # Cache
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

        # Metadata
        if tags is not None:
            alias.tags = tags
        if description is not None:
            alias.description = description
        if system_prompt is not None:
            alias.system_prompt = system_prompt
        if enabled is not None:
            alias.enabled = enabled

        session.flush()
        logger.info(f"Updated smart alias: {alias.name}")
        return alias

    if db:
        return _update(db)

    with get_db_context() as session:
        alias = _update(session)
        if alias:
            session.refresh(alias)
            return _alias_to_detached(alias)
        return None


def update_smart_alias_stats(
    alias_id: int,
    increment_requests: int = 0,
    increment_routing: int = 0,
    increment_injections: int = 0,
    increment_search: int = 0,
    increment_scrape: int = 0,
    increment_cache_hits: int = 0,
    increment_tokens_saved: int = 0,
    increment_cost_saved: float = 0.0,
    db: Optional[Session] = None,
) -> bool:
    """
    Update alias statistics (atomically increment counters).

    Returns:
        True if updated, False if alias not found
    """

    def _update_stats(session: Session) -> bool:
        alias = session.query(SmartAlias).filter(SmartAlias.id == alias_id).first()
        if not alias:
            return False

        alias.total_requests += increment_requests
        alias.routing_decisions += increment_routing
        alias.context_injections += increment_injections
        alias.search_requests += increment_search
        alias.scrape_requests += increment_scrape
        alias.cache_hits += increment_cache_hits
        alias.cache_tokens_saved += increment_tokens_saved
        alias.cache_cost_saved += increment_cost_saved

        session.flush()
        return True

    if db:
        return _update_stats(db)

    with get_db_context() as session:
        return _update_stats(session)


def reset_smart_alias_stats(alias_id: int, db: Optional[Session] = None) -> bool:
    """
    Reset alias statistics to zero.

    Returns:
        True if reset, False if alias not found
    """

    def _reset(session: Session) -> bool:
        alias = session.query(SmartAlias).filter(SmartAlias.id == alias_id).first()
        if not alias:
            return False

        alias.total_requests = 0
        alias.routing_decisions = 0
        alias.context_injections = 0
        alias.search_requests = 0
        alias.scrape_requests = 0
        alias.cache_hits = 0
        alias.cache_tokens_saved = 0
        alias.cache_cost_saved = 0.0

        session.flush()
        logger.info(f"Reset stats for smart alias: {alias.name}")
        return True

    if db:
        return _reset(db)

    with get_db_context() as session:
        return _reset(session)


def delete_smart_alias(alias_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart alias by ID.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        alias = session.query(SmartAlias).filter(SmartAlias.id == alias_id).first()
        if not alias:
            return False

        name = alias.name
        session.delete(alias)
        logger.info(f"Deleted smart alias: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def smart_alias_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if an alias name is available.

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)

    Returns:
        True if name is available
    """

    def _check(session: Session) -> bool:
        query = session.query(SmartAlias).filter(SmartAlias.name == name)
        if exclude_id:
            query = query.filter(SmartAlias.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_smart_aliases(
    db: Optional[Session] = None,
) -> dict[str, SmartAlias]:
    """
    Get all enabled smart aliases as a dict keyed by name.

    Returns:
        Dict mapping alias name to SmartAlias object
    """

    def _get(session: Session) -> list[SmartAlias]:
        return (
            session.query(SmartAlias)
            .options(joinedload(SmartAlias.document_stores))
            .filter(SmartAlias.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        aliases = _get(db)
        return {a.name: a for a in aliases}

    with get_db_context() as session:
        aliases = _get(session)
        return {a.name: _alias_to_detached(a) for a in aliases}


def _store_to_dict(store: DocumentStore) -> dict:
    """Convert a DocumentStore to a dict for detached alias."""
    return {
        "id": store.id,
        "name": store.name,
        "source_type": store.source_type,
        "source_path": store.source_path,
        "mcp_server_config_json": store.mcp_server_config_json,
        "google_account_id": store.google_account_id,
        "gdrive_folder_id": store.gdrive_folder_id,
        "gdrive_folder_name": store.gdrive_folder_name,
        "gmail_label_id": store.gmail_label_id,
        "gmail_label_name": store.gmail_label_name,
        "gcalendar_calendar_id": store.gcalendar_calendar_id,
        "gcalendar_calendar_name": store.gcalendar_calendar_name,
        "embedding_provider": store.embedding_provider,
        "embedding_model": store.embedding_model,
        "ollama_url": store.ollama_url,
        "vision_provider": store.vision_provider,
        "vision_model": store.vision_model,
        "vision_ollama_url": store.vision_ollama_url,
        "chunk_size": store.chunk_size,
        "chunk_overlap": store.chunk_overlap,
        "index_schedule": store.index_schedule,
        "index_status": store.index_status,
        "index_error": store.index_error,
        "last_indexed": store.last_indexed,
        "document_count": store.document_count,
        "chunk_count": store.chunk_count,
        "collection_name": store.collection_name,
        "description": store.description,
        "enabled": store.enabled,
    }


class DetachedDocumentStore:
    """Lightweight detached document store for use outside session."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.name = data["name"]
        self.source_type = data["source_type"]
        self.source_path = data["source_path"]
        self.mcp_server_config_json = data["mcp_server_config_json"]
        self.google_account_id = data["google_account_id"]
        self.gdrive_folder_id = data["gdrive_folder_id"]
        self.gdrive_folder_name = data["gdrive_folder_name"]
        self.gmail_label_id = data["gmail_label_id"]
        self.gmail_label_name = data["gmail_label_name"]
        self.gcalendar_calendar_id = data["gcalendar_calendar_id"]
        self.gcalendar_calendar_name = data["gcalendar_calendar_name"]
        self.embedding_provider = data["embedding_provider"]
        self.embedding_model = data["embedding_model"]
        self.ollama_url = data["ollama_url"]
        self.vision_provider = data["vision_provider"]
        self.vision_model = data["vision_model"]
        self.vision_ollama_url = data["vision_ollama_url"]
        self.chunk_size = data["chunk_size"]
        self.chunk_overlap = data["chunk_overlap"]
        self.index_schedule = data["index_schedule"]
        self.index_status = data["index_status"]
        self.index_error = data["index_error"]
        self.last_indexed = data["last_indexed"]
        self.document_count = data["document_count"]
        self.chunk_count = data["chunk_count"]
        self.collection_name = data["collection_name"]
        self.description = data["description"]
        self.enabled = data["enabled"]

    @property
    def mcp_server_config(self) -> dict | None:
        """Get MCP server config as a dict."""
        if not self.mcp_server_config_json:
            return None
        import json

        return json.loads(self.mcp_server_config_json)


def _alias_to_detached(alias: SmartAlias) -> SmartAlias:
    """
    Create a detached copy of a smart alias with all data loaded.
    """
    # Create detached document stores
    detached_stores = []
    if alias.document_stores:
        for store in alias.document_stores:
            detached_stores.append(DetachedDocumentStore(_store_to_dict(store)))

    detached = SmartAlias(
        name=alias.name,
        target_model=alias.target_model,
        # Feature toggles
        use_routing=alias.use_routing,
        use_rag=alias.use_rag,
        use_web=alias.use_web,
        use_cache=alias.use_cache,
        # Smart tag
        is_smart_tag=alias.is_smart_tag,
        passthrough_model=alias.passthrough_model,
        # Routing
        designator_model=alias.designator_model,
        purpose=alias.purpose,
        candidates_json=alias.candidates_json,
        fallback_model=alias.fallback_model,
        routing_strategy=alias.routing_strategy,
        session_ttl=alias.session_ttl,
        use_model_intelligence=alias.use_model_intelligence,
        search_provider=alias.search_provider,
        intelligence_model=alias.intelligence_model,
        # RAG
        max_results=alias.max_results,
        similarity_threshold=alias.similarity_threshold,
        # Web
        max_search_results=alias.max_search_results,
        max_scrape_urls=alias.max_scrape_urls,
        # Common enrichment
        max_context_tokens=alias.max_context_tokens,
        rerank_provider=alias.rerank_provider,
        rerank_model=alias.rerank_model,
        rerank_top_n=alias.rerank_top_n,
        context_priority=alias.context_priority,
        # Cache
        cache_similarity_threshold=alias.cache_similarity_threshold,
        cache_match_system_prompt=alias.cache_match_system_prompt,
        cache_match_last_message_only=alias.cache_match_last_message_only,
        cache_ttl_hours=alias.cache_ttl_hours,
        cache_min_tokens=alias.cache_min_tokens,
        cache_max_tokens=alias.cache_max_tokens,
        cache_collection=alias.cache_collection,
        # Stats
        total_requests=alias.total_requests,
        routing_decisions=alias.routing_decisions,
        context_injections=alias.context_injections,
        search_requests=alias.search_requests,
        scrape_requests=alias.scrape_requests,
        cache_hits=alias.cache_hits,
        cache_tokens_saved=alias.cache_tokens_saved,
        cache_cost_saved=alias.cache_cost_saved,
        # Metadata
        tags_json=alias.tags_json,
        description=alias.description,
        enabled=alias.enabled,
        # System prompt
        system_prompt=alias.system_prompt,
    )
    detached.id = alias.id
    detached.created_at = alias.created_at
    detached.updated_at = alias.updated_at

    # Attach detached stores
    detached._detached_stores = detached_stores

    return detached
