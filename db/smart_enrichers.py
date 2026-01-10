"""
Smart Enricher CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart enrichers.
Smart enrichers unify RAG and web search augmentation into a single concept.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from .connection import get_db_context
from .models import DocumentStore, SmartEnricher, smart_enricher_stores

logger = logging.getLogger(__name__)


def get_all_smart_enrichers(db: Optional[Session] = None) -> list[SmartEnricher]:
    """Get all smart enrichers from the database."""
    if db:
        return (
            db.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .order_by(SmartEnricher.name)
            .all()
        )

    with get_db_context() as session:
        enrichers = (
            session.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .order_by(SmartEnricher.name)
            .all()
        )
        return [_enricher_to_detached(e) for e in enrichers]


def get_smart_enricher_by_name(
    name: str, db: Optional[Session] = None
) -> Optional[SmartEnricher]:
    """Get a smart enricher by its name."""
    if db:
        return (
            db.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.name == name)
            .first()
        )

    with get_db_context() as session:
        enricher = (
            session.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.name == name)
            .first()
        )
        return _enricher_to_detached(enricher) if enricher else None


def get_smart_enricher_by_id(
    enricher_id: int, db: Optional[Session] = None
) -> Optional[SmartEnricher]:
    """Get a smart enricher by its ID."""
    if db:
        return (
            db.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.id == enricher_id)
            .first()
        )

    with get_db_context() as session:
        enricher = (
            session.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.id == enricher_id)
            .first()
        )
        return _enricher_to_detached(enricher) if enricher else None


def create_smart_enricher(
    name: str,
    target_model: str,
    # Enrichment mode toggles
    use_rag: bool = False,
    use_web: bool = False,
    # Document store IDs (for RAG)
    store_ids: list[int] | None = None,
    # Designator model (for web search query optimization)
    designator_model: str | None = None,
    # RAG settings
    max_results: int = 5,
    similarity_threshold: float = 0.7,
    # Web settings
    max_search_results: int = 5,
    max_scrape_urls: int = 3,
    # Common settings
    max_context_tokens: int = 4000,
    # Reranking
    rerank_provider: str = "local",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 20,
    # Caching (disabled when use_web=True)
    use_cache: bool = False,
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
    enabled: bool = True,
    db: Optional[Session] = None,
) -> SmartEnricher:
    """
    Create a new smart enricher.

    Args:
        name: Unique name for the enricher
        target_model: Model to forward requests to ("provider/model")
        use_rag: Enable RAG document retrieval
        use_web: Enable realtime web search
        store_ids: List of DocumentStore IDs to link (for RAG)
        designator_model: Model for generating optimized search queries
        max_results: Max RAG chunks to retrieve
        similarity_threshold: Minimum similarity for RAG retrieval (0.0-1.0)
        max_search_results: Max web search results
        max_scrape_urls: Max URLs to scrape from search results
        max_context_tokens: Maximum context tokens to inject
        rerank_provider: Reranking provider ("local" or "jina")
        rerank_model: Reranking model name
        rerank_top_n: Number of results to fetch before reranking
        use_cache: Enable response caching (disabled when use_web=True)
        cache_similarity_threshold: Similarity threshold for cache hits (0.0-1.0)
        cache_match_system_prompt: Include system prompt in cache key
        cache_match_last_message_only: Only match last user message
        cache_ttl_hours: Cache entry TTL in hours
        cache_min_tokens: Minimum tokens to cache response
        cache_max_tokens: Maximum tokens to cache response
        cache_collection: ChromaDB collection name (auto-generated if None)
        tags: Optional list of tags for usage tracking
        description: Optional description
        enabled: Whether the enricher is enabled (default True)
        db: Optional database session

    Returns:
        The created SmartEnricher object

    Raises:
        ValueError: If an enricher with the same name already exists
    """

    def _create(session: Session) -> SmartEnricher:
        # Check for existing enricher with same name
        existing = (
            session.query(SmartEnricher).filter(SmartEnricher.name == name).first()
        )
        if existing:
            raise ValueError(f"Smart enricher with name '{name}' already exists")

        enricher = SmartEnricher(
            name=name,
            target_model=target_model,
            use_rag=use_rag,
            use_web=use_web,
            designator_model=designator_model,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            max_search_results=max_search_results,
            max_scrape_urls=max_scrape_urls,
            max_context_tokens=max_context_tokens,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
            use_cache=use_cache,
            cache_similarity_threshold=cache_similarity_threshold,
            cache_match_system_prompt=cache_match_system_prompt,
            cache_match_last_message_only=cache_match_last_message_only,
            cache_ttl_hours=cache_ttl_hours,
            cache_min_tokens=cache_min_tokens,
            cache_max_tokens=cache_max_tokens,
            cache_collection=cache_collection,
            description=description,
            enabled=enabled,
        )
        if tags:
            enricher.tags = tags

        session.add(enricher)
        session.flush()  # Get the ID

        # Link to document stores if provided
        if store_ids:
            stores = (
                session.query(DocumentStore)
                .filter(DocumentStore.id.in_(store_ids))
                .all()
            )
            enricher.document_stores = stores
            session.flush()  # Persist junction table entries

        logger.info(f"Created smart enricher: {name}")
        return enricher

    if db:
        return _create(db)

    with get_db_context() as session:
        enricher = _create(session)
        session.refresh(enricher)
        return _enricher_to_detached(enricher)


def update_smart_enricher(
    enricher_id: int,
    name: str | None = None,
    target_model: str | None = None,
    # Enrichment mode toggles
    use_rag: bool | None = None,
    use_web: bool | None = None,
    # Document store IDs
    store_ids: list[int] | None = None,
    # Designator model
    designator_model: str | None = None,
    # RAG settings
    max_results: int | None = None,
    similarity_threshold: float | None = None,
    # Web settings
    max_search_results: int | None = None,
    max_scrape_urls: int | None = None,
    # Common settings
    max_context_tokens: int | None = None,
    # Reranking
    rerank_provider: str | None = None,
    rerank_model: str | None = None,
    rerank_top_n: int | None = None,
    # Caching
    use_cache: bool | None = None,
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
    enabled: bool | None = None,
    db: Optional[Session] = None,
) -> Optional[SmartEnricher]:
    """
    Update an existing smart enricher.

    Args:
        enricher_id: ID of the enricher to update
        name: New name (optional)
        target_model: New target model (optional)
        use_rag: Enable/disable RAG (optional)
        use_web: Enable/disable web search (optional)
        store_ids: List of DocumentStore IDs to link (replaces existing)
        designator_model: New designator model (optional)
        max_results: New max RAG results (optional)
        similarity_threshold: New similarity threshold (optional)
        max_search_results: New max search results (optional)
        max_scrape_urls: New max scrape URLs (optional)
        max_context_tokens: New max context tokens (optional)
        rerank_provider: New rerank provider (optional)
        rerank_model: New rerank model (optional)
        rerank_top_n: New rerank top N (optional)
        use_cache: Enable/disable caching (optional)
        cache_similarity_threshold: New cache similarity threshold (optional)
        cache_match_system_prompt: Include system prompt in cache key (optional)
        cache_match_last_message_only: Only match last message (optional)
        cache_ttl_hours: New cache TTL hours (optional)
        cache_min_tokens: New minimum tokens to cache (optional)
        cache_max_tokens: New maximum tokens to cache (optional)
        cache_collection: New cache collection name (optional)
        tags: New tags list (optional, pass empty list to clear)
        description: New description (optional)
        enabled: New enabled state (optional)
        db: Optional database session

    Returns:
        Updated SmartEnricher object or None if not found

    Raises:
        ValueError: If new name conflicts with existing enricher
    """

    def _update(session: Session) -> Optional[SmartEnricher]:
        enricher = (
            session.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.id == enricher_id)
            .first()
        )
        if not enricher:
            return None

        # Check name uniqueness if changing
        if name and name != enricher.name:
            existing = (
                session.query(SmartEnricher).filter(SmartEnricher.name == name).first()
            )
            if existing:
                raise ValueError(f"Smart enricher with name '{name}' already exists")
            enricher.name = name

        if target_model is not None:
            enricher.target_model = target_model

        if use_rag is not None:
            enricher.use_rag = use_rag

        if use_web is not None:
            enricher.use_web = use_web

        # Update document store links if provided
        if store_ids is not None:
            stores = (
                session.query(DocumentStore)
                .filter(DocumentStore.id.in_(store_ids))
                .all()
            )
            enricher.document_stores = stores
            session.flush()  # Persist junction table entries

        if designator_model is not None:
            enricher.designator_model = designator_model

        if max_results is not None:
            enricher.max_results = max_results

        if similarity_threshold is not None:
            enricher.similarity_threshold = similarity_threshold

        if max_search_results is not None:
            enricher.max_search_results = max_search_results

        if max_scrape_urls is not None:
            enricher.max_scrape_urls = max_scrape_urls

        if max_context_tokens is not None:
            enricher.max_context_tokens = max_context_tokens

        if rerank_provider is not None:
            enricher.rerank_provider = rerank_provider

        if rerank_model is not None:
            enricher.rerank_model = rerank_model

        if rerank_top_n is not None:
            enricher.rerank_top_n = rerank_top_n

        # Cache settings
        if use_cache is not None:
            enricher.use_cache = use_cache

        if cache_similarity_threshold is not None:
            enricher.cache_similarity_threshold = cache_similarity_threshold

        if cache_match_system_prompt is not None:
            enricher.cache_match_system_prompt = cache_match_system_prompt

        if cache_match_last_message_only is not None:
            enricher.cache_match_last_message_only = cache_match_last_message_only

        if cache_ttl_hours is not None:
            enricher.cache_ttl_hours = cache_ttl_hours

        if cache_min_tokens is not None:
            enricher.cache_min_tokens = cache_min_tokens

        if cache_max_tokens is not None:
            enricher.cache_max_tokens = cache_max_tokens

        if cache_collection is not None:
            enricher.cache_collection = cache_collection

        if tags is not None:
            enricher.tags = tags

        if description is not None:
            enricher.description = description

        if enabled is not None:
            enricher.enabled = enabled

        session.flush()
        logger.info(f"Updated smart enricher: {enricher.name}")
        return enricher

    if db:
        return _update(db)

    with get_db_context() as session:
        enricher = _update(session)
        if enricher:
            session.refresh(enricher)
            return _enricher_to_detached(enricher)
        return None


def update_smart_enricher_stats(
    enricher_id: int,
    increment_requests: int = 0,
    increment_injections: int = 0,
    increment_search: int = 0,
    increment_scrape: int = 0,
    increment_cache_hits: int = 0,
    increment_tokens_saved: int = 0,
    increment_cost_saved: float = 0.0,
    db: Optional[Session] = None,
) -> bool:
    """
    Update enricher statistics (atomically increment counters).

    Args:
        enricher_id: ID of the enricher to update
        increment_requests: Number of requests to add
        increment_injections: Number of context injections to add
        increment_search: Number of search requests to add
        increment_scrape: Number of scrape requests to add
        increment_cache_hits: Number of cache hits to add
        increment_tokens_saved: Number of tokens saved by caching
        increment_cost_saved: Cost saved by caching

    Returns:
        True if updated, False if enricher not found
    """

    def _update_stats(session: Session) -> bool:
        enricher = (
            session.query(SmartEnricher).filter(SmartEnricher.id == enricher_id).first()
        )
        if not enricher:
            return False

        enricher.total_requests += increment_requests
        enricher.context_injections += increment_injections
        enricher.search_requests += increment_search
        enricher.scrape_requests += increment_scrape
        enricher.cache_hits += increment_cache_hits
        enricher.cache_tokens_saved += increment_tokens_saved
        enricher.cache_cost_saved += increment_cost_saved

        session.flush()
        return True

    if db:
        return _update_stats(db)

    with get_db_context() as session:
        return _update_stats(session)


def reset_smart_enricher_stats(enricher_id: int, db: Optional[Session] = None) -> bool:
    """
    Reset enricher statistics to zero (including cache stats).

    Returns:
        True if reset, False if enricher not found
    """

    def _reset(session: Session) -> bool:
        enricher = (
            session.query(SmartEnricher).filter(SmartEnricher.id == enricher_id).first()
        )
        if not enricher:
            return False

        enricher.total_requests = 0
        enricher.context_injections = 0
        enricher.search_requests = 0
        enricher.scrape_requests = 0
        enricher.cache_hits = 0
        enricher.cache_tokens_saved = 0
        enricher.cache_cost_saved = 0.0

        session.flush()
        logger.info(f"Reset stats for smart enricher: {enricher.name}")
        return True

    if db:
        return _reset(db)

    with get_db_context() as session:
        return _reset(session)


def delete_smart_enricher(enricher_id: int, db: Optional[Session] = None) -> bool:
    """
    Delete a smart enricher by ID.

    Document store links are automatically removed via CASCADE.

    Returns:
        True if deleted, False if not found
    """

    def _delete(session: Session) -> bool:
        enricher = (
            session.query(SmartEnricher).filter(SmartEnricher.id == enricher_id).first()
        )
        if not enricher:
            return False

        name = enricher.name
        session.delete(enricher)
        logger.info(f"Deleted smart enricher: {name}")
        return True

    if db:
        return _delete(db)

    with get_db_context() as session:
        return _delete(session)


def enricher_name_available(
    name: str, exclude_id: int | None = None, db: Optional[Session] = None
) -> bool:
    """
    Check if an enricher name is available.

    Args:
        name: Name to check
        exclude_id: Optional ID to exclude (for updates)
        db: Optional database session

    Returns:
        True if name is available, False if taken
    """

    def _check(session: Session) -> bool:
        query = session.query(SmartEnricher).filter(SmartEnricher.name == name)
        if exclude_id:
            query = query.filter(SmartEnricher.id != exclude_id)
        return query.first() is None

    if db:
        return _check(db)

    with get_db_context() as session:
        return _check(session)


def get_enabled_smart_enrichers(
    db: Optional[Session] = None,
) -> dict[str, SmartEnricher]:
    """
    Get all enabled smart enrichers as a dict keyed by name.

    Useful for quick lookups during request processing.

    Returns:
        Dict mapping enricher name to SmartEnricher object
    """

    def _get(session: Session) -> list[SmartEnricher]:
        return (
            session.query(SmartEnricher)
            .options(joinedload(SmartEnricher.document_stores))
            .filter(SmartEnricher.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        enrichers = _get(db)
        return {e.name: e for e in enrichers}

    with get_db_context() as session:
        enrichers = _get(session)
        return {e.name: _enricher_to_detached(e) for e in enrichers}


def _store_to_dict(store: DocumentStore) -> dict:
    """Convert a DocumentStore to a minimal dict for embedding in detached enricher."""
    return {
        "id": store.id,
        "name": store.name,
        "source_type": store.source_type,
        "source_path": store.source_path,
        "mcp_server_config_json": store.mcp_server_config_json,
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


def _enricher_to_detached(enricher: SmartEnricher) -> SmartEnricher:
    """
    Create a detached copy of a smart enricher with all data loaded.

    This allows the enricher to be used after the session is closed.
    """
    # Create detached document stores
    detached_stores = []
    if enricher.document_stores:
        for store in enricher.document_stores:
            detached_stores.append(DetachedDocumentStore(_store_to_dict(store)))

    detached = SmartEnricher(
        name=enricher.name,
        use_rag=enricher.use_rag,
        use_web=enricher.use_web,
        target_model=enricher.target_model,
        designator_model=enricher.designator_model,
        max_results=enricher.max_results,
        similarity_threshold=enricher.similarity_threshold,
        max_search_results=enricher.max_search_results,
        max_scrape_urls=enricher.max_scrape_urls,
        max_context_tokens=enricher.max_context_tokens,
        rerank_provider=enricher.rerank_provider,
        rerank_model=enricher.rerank_model,
        rerank_top_n=enricher.rerank_top_n,
        total_requests=enricher.total_requests,
        context_injections=enricher.context_injections,
        search_requests=enricher.search_requests,
        scrape_requests=enricher.scrape_requests,
        tags_json=enricher.tags_json,
        description=enricher.description,
        enabled=enricher.enabled,
        # Cache fields
        use_cache=enricher.use_cache,
        cache_similarity_threshold=enricher.cache_similarity_threshold,
        cache_match_system_prompt=enricher.cache_match_system_prompt,
        cache_match_last_message_only=enricher.cache_match_last_message_only,
        cache_ttl_hours=enricher.cache_ttl_hours,
        cache_min_tokens=enricher.cache_min_tokens,
        cache_max_tokens=enricher.cache_max_tokens,
        cache_collection=enricher.cache_collection,
        cache_hits=enricher.cache_hits,
        cache_tokens_saved=enricher.cache_tokens_saved,
        cache_cost_saved=enricher.cache_cost_saved,
    )
    detached.id = enricher.id
    detached.created_at = enricher.created_at
    detached.updated_at = enricher.updated_at

    # Attach detached stores (as a list, not ORM relationship)
    detached._detached_stores = detached_stores

    return detached
