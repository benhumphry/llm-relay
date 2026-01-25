"""
Smart Alias CRUD operations for LLM Relay.

Provides functions to create, read, update, and delete smart aliases.
Smart aliases unify routing, enrichment (RAG + Web), and caching into a single concept.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from .connection import get_db_context
from .models import DocumentStore, LiveDataSource, SmartAlias, smart_alias_stores

logger = logging.getLogger(__name__)


def get_all_smart_aliases(db: Optional[Session] = None) -> list[SmartAlias]:
    """Get all smart aliases from the database."""
    if db:
        return (
            db.query(SmartAlias)
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
            .order_by(SmartAlias.name)
            .all()
        )

    with get_db_context() as session:
        aliases = (
            session.query(SmartAlias)
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
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
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
            .filter(SmartAlias.name == name)
            .first()
        )

    with get_db_context() as session:
        alias = (
            session.query(SmartAlias)
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
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
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
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
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
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
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
            .filter(SmartAlias.id == alias_id)
            .first()
        )

    with get_db_context() as session:
        alias = (
            session.query(SmartAlias)
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
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
    use_live_data: bool = False,
    # Smart tag settings
    is_smart_tag: bool = False,
    passthrough_model: bool = False,
    # Routing settings
    designator_model: str | None = None,
    router_designator_model: str | None = None,
    rag_designator_model: str | None = None,
    web_designator_model: str | None = None,
    live_designator_model: str | None = None,
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
    # Smart source selection
    use_smart_source_selection: bool = False,
    use_two_pass_retrieval: bool = False,
    # Web settings
    max_search_results: int = 5,
    max_scrape_urls: int = 3,
    # Live data settings
    live_data_source_ids: list[int] | None = None,
    # Common enrichment settings
    max_context_tokens: int = 4000,
    rerank_provider: str = "local",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 20,
    context_priority: str = "balanced",
    show_sources: bool = False,
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
    # Memory
    use_memory: bool = False,
    memory_max_tokens: int = 500,
    # Actions
    use_actions: bool = False,
    allowed_actions: list[str] | None = None,
    action_email_account_id: int | None = None,
    action_calendar_account_id: int | None = None,
    action_calendar_id: str | None = None,
    action_tasks_account_id: int | None = None,
    action_tasks_provider: str | None = None,
    action_tasks_list_id: str | None = None,
    action_notification_urls: list[str] | None = None,
    action_notes_store_id: int | None = None,
    # Scheduled prompts
    scheduled_prompts_enabled: bool = False,
    scheduled_prompts_account_id: int | None = None,
    scheduled_prompts_calendar_id: str | None = None,
    scheduled_prompts_calendar_name: str | None = None,
    scheduled_prompts_lookahead: int = 15,
    scheduled_prompts_store_response: bool = True,
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
        use_live_data: Enable live data source queries
        live_data_source_ids: List of LiveDataSource IDs to query
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
            use_live_data=use_live_data,
            # Smart tag
            is_smart_tag=is_smart_tag,
            passthrough_model=passthrough_model,
            # Routing
            designator_model=designator_model,
            router_designator_model=router_designator_model,
            rag_designator_model=rag_designator_model,
            web_designator_model=web_designator_model,
            live_designator_model=live_designator_model,
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
            # Smart source selection
            use_smart_source_selection=use_smart_source_selection,
            use_two_pass_retrieval=use_two_pass_retrieval,
            # Web
            max_search_results=max_search_results,
            max_scrape_urls=max_scrape_urls,
            # Common enrichment
            max_context_tokens=max_context_tokens,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
            context_priority=context_priority,
            show_sources=show_sources,
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
            # Memory
            use_memory=use_memory,
            memory_max_tokens=memory_max_tokens,
            # Actions
            use_actions=use_actions,
            action_email_account_id=action_email_account_id,
            action_calendar_account_id=action_calendar_account_id,
            action_calendar_id=action_calendar_id,
            action_tasks_account_id=action_tasks_account_id,
            action_tasks_provider=action_tasks_provider,
            action_tasks_list_id=action_tasks_list_id,
            action_notes_store_id=action_notes_store_id,
            # Scheduled prompts
            scheduled_prompts_enabled=scheduled_prompts_enabled,
            scheduled_prompts_account_id=scheduled_prompts_account_id,
            scheduled_prompts_calendar_id=scheduled_prompts_calendar_id,
            scheduled_prompts_calendar_name=scheduled_prompts_calendar_name,
            scheduled_prompts_lookahead=scheduled_prompts_lookahead,
            scheduled_prompts_store_response=scheduled_prompts_store_response,
        )

        if candidates:
            alias.candidates = candidates
        if tags:
            alias.tags = tags
        if allowed_actions:
            alias.allowed_actions = allowed_actions
        if action_notification_urls:
            alias.action_notification_urls = action_notification_urls

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

        # Link to live data sources if provided
        if live_data_source_ids:
            sources = (
                session.query(LiveDataSource)
                .filter(LiveDataSource.id.in_(live_data_source_ids))
                .all()
            )
            alias.live_data_sources = sources
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
    use_live_data: bool | None = None,
    # Smart tag settings
    is_smart_tag: bool | None = None,
    passthrough_model: bool | None = None,
    # Routing settings
    designator_model: str | None = None,
    router_designator_model: str | None = None,
    rag_designator_model: str | None = None,
    web_designator_model: str | None = None,
    live_designator_model: str | None = None,
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
    # Smart source selection
    use_smart_source_selection: bool | None = None,
    use_two_pass_retrieval: bool | None = None,
    # Web settings
    max_search_results: int | None = None,
    max_scrape_urls: int | None = None,
    # Live data settings
    live_data_source_ids: list[int] | None = None,
    # Common enrichment settings
    max_context_tokens: int | None = None,
    rerank_provider: str | None = None,
    rerank_model: str | None = None,
    rerank_top_n: int | None = None,
    context_priority: str | None = None,
    show_sources: bool | None = None,
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
    # Memory
    use_memory: bool | None = None,
    memory: str | None = None,
    memory_max_tokens: int | None = None,
    # Actions
    use_actions: bool | None = None,
    allowed_actions: list[str] | None = None,
    action_email_account_id: int | None = None,
    action_calendar_account_id: int | None = None,
    action_calendar_id: str | None = None,
    action_tasks_account_id: int | None = None,
    action_tasks_provider: str | None = None,
    action_tasks_list_id: str | None = None,
    action_notification_urls: list[str] | None = None,
    action_notes_store_id: int | None = None,
    # Scheduled prompts
    scheduled_prompts_enabled: bool | None = None,
    scheduled_prompts_account_id: int | None = None,
    scheduled_prompts_calendar_id: str | None = None,
    scheduled_prompts_calendar_name: str | None = None,
    scheduled_prompts_lookahead: int | None = None,
    scheduled_prompts_store_response: bool | None = None,
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
        if use_live_data is not None:
            alias.use_live_data = use_live_data

        # Smart tag
        if is_smart_tag is not None:
            alias.is_smart_tag = is_smart_tag
        if passthrough_model is not None:
            alias.passthrough_model = passthrough_model

        # Routing
        if designator_model is not None:
            alias.designator_model = designator_model
        if router_designator_model is not None:
            alias.router_designator_model = router_designator_model
        if rag_designator_model is not None:
            alias.rag_designator_model = rag_designator_model
        if web_designator_model is not None:
            alias.web_designator_model = web_designator_model
        if live_designator_model is not None:
            alias.live_designator_model = live_designator_model
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
        # Live data sources
        if live_data_source_ids is not None:
            sources = (
                session.query(LiveDataSource)
                .filter(LiveDataSource.id.in_(live_data_source_ids))
                .all()
            )
            alias.live_data_sources = sources
            session.flush()
        if max_results is not None:
            alias.max_results = max_results
        if similarity_threshold is not None:
            alias.similarity_threshold = similarity_threshold

        # Smart source selection
        if use_smart_source_selection is not None:
            alias.use_smart_source_selection = use_smart_source_selection
        if use_two_pass_retrieval is not None:
            alias.use_two_pass_retrieval = use_two_pass_retrieval

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
        if show_sources is not None:
            alias.show_sources = show_sources

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

        # Memory
        if use_memory is not None:
            alias.use_memory = use_memory
        if memory is not None:
            alias.memory = memory
        if memory_max_tokens is not None:
            alias.memory_max_tokens = memory_max_tokens

        # Actions
        if use_actions is not None:
            alias.use_actions = use_actions
        if allowed_actions is not None:
            alias.allowed_actions = allowed_actions
        if action_notification_urls is not None:
            alias.action_notification_urls = action_notification_urls
        if action_notes_store_id is not None:
            alias.action_notes_store_id = (
                action_notes_store_id if action_notes_store_id != 0 else None
            )
        if action_email_account_id is not None:
            # Allow setting to 0 to clear the account
            alias.action_email_account_id = (
                action_email_account_id if action_email_account_id != 0 else None
            )
        if action_calendar_account_id is not None:
            alias.action_calendar_account_id = (
                action_calendar_account_id if action_calendar_account_id != 0 else None
            )
        if action_calendar_id is not None:
            alias.action_calendar_id = (
                action_calendar_id if action_calendar_id else None
            )
        if action_tasks_account_id is not None:
            alias.action_tasks_account_id = (
                action_tasks_account_id if action_tasks_account_id != 0 else None
            )
        if action_tasks_provider is not None:
            alias.action_tasks_provider = (
                action_tasks_provider if action_tasks_provider else None
            )
        if action_tasks_list_id is not None:
            alias.action_tasks_list_id = (
                action_tasks_list_id if action_tasks_list_id else None
            )

        # Scheduled prompts
        if scheduled_prompts_enabled is not None:
            alias.scheduled_prompts_enabled = scheduled_prompts_enabled
        if scheduled_prompts_account_id is not None:
            alias.scheduled_prompts_account_id = (
                scheduled_prompts_account_id
                if scheduled_prompts_account_id != 0
                else None
            )
        if scheduled_prompts_calendar_id is not None:
            alias.scheduled_prompts_calendar_id = (
                scheduled_prompts_calendar_id if scheduled_prompts_calendar_id else None
            )
        if scheduled_prompts_calendar_name is not None:
            alias.scheduled_prompts_calendar_name = (
                scheduled_prompts_calendar_name
                if scheduled_prompts_calendar_name
                else None
            )
        if scheduled_prompts_lookahead is not None:
            alias.scheduled_prompts_lookahead = scheduled_prompts_lookahead
        if scheduled_prompts_store_response is not None:
            alias.scheduled_prompts_store_response = scheduled_prompts_store_response

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


def update_smart_alias_memory(
    alias_id: int,
    memory: str,
    db: Optional[Session] = None,
) -> bool:
    """
    Update the memory for a smart alias.

    This is called by the smart enricher when the designator decides
    to update the memory with significant new information.

    Args:
        alias_id: ID of the smart alias
        memory: New memory content

    Returns:
        True if updated, False if alias not found
    """
    from datetime import datetime

    def _update_memory(session: Session) -> bool:
        alias = session.query(SmartAlias).filter(SmartAlias.id == alias_id).first()
        if not alias:
            return False

        alias.memory = memory
        alias.memory_updated_at = datetime.utcnow()
        session.flush()
        logger.info(f"Updated memory for smart alias: {alias.name}")
        return True

    if db:
        return _update_memory(db)

    with get_db_context() as session:
        return _update_memory(session)


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
            .options(
                joinedload(SmartAlias.document_stores),
                joinedload(SmartAlias.live_data_sources),
            )
            .filter(SmartAlias.enabled == True)  # noqa: E712
            .all()
        )

    if db:
        aliases = _get(db)
        return {a.name: a for a in aliases}

    with get_db_context() as session:
        aliases = _get(session)
        return {a.name: _alias_to_detached(a) for a in aliases}


def _live_source_to_dict(source: LiveDataSource) -> dict:
    """Convert a LiveDataSource to a dict for detached alias."""
    return {
        "id": source.id,
        "name": source.name,
        "source_type": source.source_type,
        "enabled": source.enabled,
        "description": source.description,
        "endpoint_url": source.endpoint_url,
        "http_method": source.http_method,
        "headers_json": source.headers_json,
        "auth_type": source.auth_type,
        "auth_config_json": source.auth_config_json,
        "request_template_json": source.request_template_json,
        "query_params_json": source.query_params_json,
        "response_path": source.response_path,
        "response_format_template": source.response_format_template,
        "cache_ttl_seconds": source.cache_ttl_seconds,
        "timeout_seconds": source.timeout_seconds,
        "retry_count": source.retry_count,
        "rate_limit_rpm": source.rate_limit_rpm,
        "data_type": source.data_type,
        "sample_response_json": source.sample_response_json,
        "best_for": source.best_for,
        "last_success": source.last_success,
        "last_error": source.last_error,
        "error_count": source.error_count,
        "created_at": source.created_at,
        "updated_at": source.updated_at,
    }


class DetachedLiveDataSource:
    """Lightweight detached live data source for use outside session."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.name = data["name"]
        self.source_type = data["source_type"]
        self.enabled = data["enabled"]
        self.description = data["description"]
        self.endpoint_url = data["endpoint_url"]
        self.http_method = data["http_method"]
        self.headers_json = data["headers_json"]
        self.auth_type = data["auth_type"]
        self.auth_config_json = data["auth_config_json"]
        self.request_template_json = data["request_template_json"]
        self.query_params_json = data["query_params_json"]
        self.response_path = data["response_path"]
        self.response_format_template = data["response_format_template"]
        self.cache_ttl_seconds = data["cache_ttl_seconds"]
        self.timeout_seconds = data["timeout_seconds"]
        self.retry_count = data["retry_count"]
        self.rate_limit_rpm = data["rate_limit_rpm"]
        self.data_type = data["data_type"]
        self.sample_response_json = data["sample_response_json"]
        self.best_for = data["best_for"]
        self.last_success = data["last_success"]
        self.last_error = data["last_error"]
        self.error_count = data["error_count"]
        self.created_at = data["created_at"]
        self.updated_at = data["updated_at"]

    @property
    def headers(self) -> dict | None:
        """Get headers as a dict."""
        if not self.headers_json:
            return None
        import json

        return json.loads(self.headers_json)

    @property
    def auth_config(self) -> dict | None:
        """Get auth config as a dict."""
        if not self.auth_config_json:
            return None
        import json

        return json.loads(self.auth_config_json)

    @property
    def request_template(self) -> dict | None:
        """Get request template as a dict."""
        if not self.request_template_json:
            return None
        import json

        return json.loads(self.request_template_json)

    @property
    def query_params(self) -> dict | None:
        """Get query params as a dict."""
        if not self.query_params_json:
            return None
        import json

        return json.loads(self.query_params_json)


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
        # Google Tasks fields
        "gtasks_tasklist_id": store.gtasks_tasklist_id,
        "gtasks_tasklist_name": store.gtasks_tasklist_name,
        # Todoist fields
        "todoist_project_id": store.todoist_project_id,
        "todoist_project_name": store.todoist_project_name,
        "todoist_filter": store.todoist_filter,
        "todoist_include_completed": store.todoist_include_completed,
        # Notion fields
        "notion_page_id": store.notion_page_id,
        "notion_database_id": store.notion_database_id,
        "notion_is_task_database": store.notion_is_task_database,
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
        # Intelligence fields
        "themes": store.themes,
        "best_for": store.best_for,
        "content_summary": store.content_summary,
        "intelligence_updated_at": store.intelligence_updated_at,
        # Metadata
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
        # Google Tasks fields
        self.gtasks_tasklist_id = data.get("gtasks_tasklist_id")
        self.gtasks_tasklist_name = data.get("gtasks_tasklist_name")
        # Todoist fields
        self.todoist_project_id = data.get("todoist_project_id")
        self.todoist_project_name = data.get("todoist_project_name")
        self.todoist_filter = data.get("todoist_filter")
        self.todoist_include_completed = data.get("todoist_include_completed", False)
        # Notion fields
        self.notion_page_id = data.get("notion_page_id")
        self.notion_database_id = data.get("notion_database_id")
        self.notion_is_task_database = data.get("notion_is_task_database", False)
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
        # Intelligence fields
        self.themes = data.get("themes")
        self.best_for = data.get("best_for")
        self.content_summary = data.get("content_summary")
        self.intelligence_updated_at = data.get("intelligence_updated_at")
        # Metadata
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

    # Create detached live data sources
    detached_live_sources = []
    if alias.live_data_sources:
        for source in alias.live_data_sources:
            detached_live_sources.append(
                DetachedLiveDataSource(_live_source_to_dict(source))
            )

    detached = SmartAlias(
        name=alias.name,
        target_model=alias.target_model,
        # Feature toggles
        use_routing=alias.use_routing,
        use_rag=alias.use_rag,
        use_web=alias.use_web,
        use_cache=alias.use_cache,
        use_live_data=alias.use_live_data,
        # Smart tag
        is_smart_tag=alias.is_smart_tag,
        passthrough_model=alias.passthrough_model,
        # Routing
        designator_model=alias.designator_model,
        router_designator_model=alias.router_designator_model,
        rag_designator_model=alias.rag_designator_model,
        web_designator_model=alias.web_designator_model,
        live_designator_model=alias.live_designator_model,
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
        # Smart source selection
        use_smart_source_selection=alias.use_smart_source_selection,
        use_two_pass_retrieval=alias.use_two_pass_retrieval,
        # Web
        max_search_results=alias.max_search_results,
        max_scrape_urls=alias.max_scrape_urls,
        # Common enrichment
        max_context_tokens=alias.max_context_tokens,
        rerank_provider=alias.rerank_provider,
        rerank_model=alias.rerank_model,
        rerank_top_n=alias.rerank_top_n,
        context_priority=alias.context_priority,
        show_sources=alias.show_sources,
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
        # Memory
        use_memory=alias.use_memory,
        memory=alias.memory,
        memory_max_tokens=alias.memory_max_tokens,
        memory_updated_at=alias.memory_updated_at,
        # Actions
        use_actions=alias.use_actions,
        allowed_actions_json=alias.allowed_actions_json,
        action_email_account_id=alias.action_email_account_id,
        action_calendar_account_id=alias.action_calendar_account_id,
        action_calendar_id=alias.action_calendar_id,
        action_tasks_account_id=alias.action_tasks_account_id,
        action_tasks_provider=alias.action_tasks_provider,
        action_tasks_list_id=alias.action_tasks_list_id,
        action_notification_urls_json=alias.action_notification_urls_json,
        action_notes_store_id=alias.action_notes_store_id,
        # Scheduled prompts
        scheduled_prompts_enabled=alias.scheduled_prompts_enabled,
        scheduled_prompts_account_id=alias.scheduled_prompts_account_id,
        scheduled_prompts_calendar_id=alias.scheduled_prompts_calendar_id,
        scheduled_prompts_calendar_name=alias.scheduled_prompts_calendar_name,
        scheduled_prompts_lookahead=alias.scheduled_prompts_lookahead,
        scheduled_prompts_store_response=alias.scheduled_prompts_store_response,
    )
    detached.id = alias.id
    detached.created_at = alias.created_at
    detached.updated_at = alias.updated_at

    # Attach detached stores and live sources
    detached._detached_stores = detached_stores
    detached._detached_live_sources = detached_live_sources

    return detached
