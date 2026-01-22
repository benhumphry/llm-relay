"""
Smart Enricher Engine for unified context augmentation.

Combines RAG document retrieval and web search/scraping into a single enrichment pipeline.
Supports RAG-only, web-only, or hybrid (both) modes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from db.models import SmartAlias
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


def _get_live_source_to_unified_map() -> dict[str, str]:
    """
    Get mapping from legacy live source types to unified source types.

    Uses dynamic plugin lookup so custom plugins are automatically included.
    """
    try:
        from plugin_base.loader import get_live_source_to_unified_map

        return get_live_source_to_unified_map()
    except ImportError:
        logger.warning("Plugin loader not available for dynamic live source lookup")
        return {}


@dataclass
class EnrichmentResult:
    """Result of context enrichment."""

    resolved: "ResolvedModel"
    enricher_id: int  # For stats updates
    enricher_name: str
    enricher_tags: list[str]

    # Augmented content
    augmented_system: str | None = None  # Modified system prompt with injected context
    augmented_messages: list[dict] = field(default_factory=list)  # Usually unchanged

    # Enrichment metadata
    enrichment_type: str = "none"  # "rag"|"web"|"hybrid"|"none"
    context_injected: bool = False

    # RAG-specific metadata
    chunks_retrieved: int = 0
    sources: list[str] = field(default_factory=list)  # Source files used
    stores_queried: list[str] = field(default_factory=list)  # Document stores used
    embedding_usage: dict | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None

    # Web-specific metadata
    search_query: str | None = None
    scraped_urls: list[str] = field(default_factory=list)
    designator_usage: dict | None = (
        None  # Aggregated usage (prompt_tokens, completion_tokens)
    )
    designator_model: str | None = None
    designator_calls: list[dict] = field(
        default_factory=list
    )  # Individual call records

    # Live data metadata
    live_sources_queried: list[str] = field(default_factory=list)
    live_data_errors: list[str] = field(default_factory=list)

    # Budget allocations (from smart source selection)
    store_budgets: dict[int, int] | None = None  # {store_id: tokens}
    store_id_to_name: dict[int, str] | None = None  # {store_id: name} for display
    web_budget: int | None = None
    live_budget: int | None = None
    total_budget: int | None = None  # Total context budget for percentage calc
    show_sources: bool = False  # Whether to append sources to response

    # Memory metadata
    memory_included: bool = False
    memory_update_pending: bool = False  # Flag to trigger memory update after response

    # Actions metadata
    actions_enabled: bool = False
    allowed_actions: list[str] = field(default_factory=list)
    # Available accounts derived from linked document stores
    # Keys: "email", "calendar", "tasks" - each contains list of store-based accounts
    available_accounts: dict[str, list[dict]] = field(default_factory=dict)
    # Default accounts configured on the Smart Alias (fallback if LLM doesn't specify)
    default_accounts: dict[str, dict] = field(default_factory=dict)
    # Legacy action-category defaults (for backwards compatibility)
    action_email_account_id: int | None = None
    action_calendar_account_id: int | None = None
    action_calendar_id: str | None = None
    action_tasks_account_id: int | None = None
    action_tasks_list_id: str | None = None
    action_notification_urls: list[str] | None = None
    # Scheduled prompts config (for schedule:prompt action)
    scheduled_prompts_account_id: int | None = None
    scheduled_prompts_calendar_id: str | None = None

    # Session context (for action handlers)
    session_key: str | None = None


class SmartEnricherEngine:
    """
    Unified engine for context enrichment using RAG and/or web search.

    The engine:
    1. Extracts the user's query from messages
    2. If RAG enabled: Retrieves relevant document chunks from linked stores
    3. If Web enabled: Performs web search and scrapes relevant URLs
    4. Merges and reranks combined context
    5. Injects the context into the system prompt
    6. Returns the augmented request for forwarding to the target model
    """

    def __init__(self, enricher: "SmartAlias", registry: "ProviderRegistry"):
        """
        Initialize the enrichment engine.

        Args:
            enricher: SmartAlias (or adapter) with enrichment configuration
            registry: Provider registry for model resolution
        """
        self.enricher = enricher
        self.registry = registry

    def _get_linked_stores(self) -> list:
        """Get document stores linked to this enricher."""
        if (
            hasattr(self.enricher, "_detached_stores")
            and self.enricher._detached_stores
        ):
            return self.enricher._detached_stores
        if hasattr(self.enricher, "document_stores") and self.enricher.document_stores:
            return self.enricher.document_stores
        return []

    def _has_ready_stores(self, stores: list) -> bool:
        """Check if any linked stores are ready for querying."""
        for store in stores:
            enabled = getattr(store, "enabled", True)
            status = getattr(store, "index_status", None)
            collection = getattr(store, "collection_name", None)
            if enabled and status == "ready" and collection:
                return True
        return False

    def _build_available_accounts(self) -> dict[str, list[dict]]:
        """
        Build available_accounts dict from linked document stores.

        Maps document stores to action categories based on source_type:
        - mcp:gmail -> email
        - mcp:gcalendar -> calendar
        - mcp:gtasks, todoist -> tasks

        Returns:
            Dict mapping categories (email, calendar, tasks) to lists of account info.
            Each account has: store_id, name, slug, source_type, oauth_account_id,
            provider, email, and source-specific fields (project_id, tasklist_id, etc.)
        """
        from db.oauth_tokens import get_oauth_token_info
        from plugin_base.common import ContentCategory, get_content_category

        available_accounts: dict[str, list[dict]] = {
            "email": [],
            "calendar": [],
            "tasks": [],
        }

        # Map ContentCategory enum to action handler category keys
        CONTENT_TO_ACTION_CATEGORY = {
            ContentCategory.EMAILS: "email",
            ContentCategory.CALENDARS: "calendar",
            ContentCategory.TASKS: "tasks",
        }

        stores = self._get_linked_stores()
        logger.debug(f"_build_available_accounts: Found {len(stores)} linked stores")

        try:
            for store in stores:
                source_type = getattr(store, "source_type", "")
                store_name = getattr(store, "name", "unknown")

                # Get content category from plugin system
                content_category = get_content_category(source_type)

                # Special case: Notion stores marked as task databases
                # Override category from FILES to TASKS
                if source_type == "notion" and getattr(
                    store, "notion_is_task_database", False
                ):
                    content_category = ContentCategory.TASKS

                category = CONTENT_TO_ACTION_CATEGORY.get(content_category)

                if not category:
                    continue  # Not an actionable source type

                # Use display_name if set, otherwise fall back to name
                display_name = getattr(store, "display_name", None) or store_name
                store_id = getattr(store, "id", None)

                # Determine OAuth info based on source type
                oauth_account_id = None
                oauth_provider = ""
                oauth_email = ""

                if source_type.startswith("mcp:g"):
                    # Google source
                    oauth_account_id = getattr(store, "google_account_id", None)
                    oauth_provider = "google"
                elif source_type.startswith("mcp:o"):
                    # Microsoft/Outlook source
                    oauth_account_id = getattr(store, "microsoft_account_id", None)
                    oauth_provider = "microsoft"
                elif source_type == "todoist":
                    # Todoist uses API key, not OAuth
                    oauth_provider = "todoist"
                elif source_type == "notion":
                    # Notion uses API key, not OAuth
                    oauth_provider = "notion"

                # Get email from OAuth token if we have an account ID
                if oauth_account_id:
                    token_info = get_oauth_token_info(oauth_account_id)
                    if token_info:
                        oauth_email = token_info.get("account_email", "")

                # Build account info with both slug and display name
                account_info = {
                    "store_id": store_id,
                    "name": display_name,  # Friendly name
                    "slug": store_name,  # Original slug for matching
                    "source_type": source_type,
                    "oauth_account_id": oauth_account_id,
                    "provider": oauth_provider,
                    "email": oauth_email,
                }

                # Add source-specific fields
                if source_type == "mcp:gcalendar":
                    account_info["calendar_id"] = getattr(
                        store, "gcalendar_calendar_id", None
                    )
                elif source_type == "mcp:gtasks":
                    account_info["tasklist_id"] = getattr(
                        store, "gtasks_tasklist_id", None
                    )
                elif source_type == "todoist":
                    # Include project_id so tasks go to the right project
                    account_info["project_id"] = getattr(
                        store, "todoist_project_id", None
                    )
                    account_info["project_name"] = getattr(
                        store, "todoist_project_name", None
                    )
                    logger.info(
                        f"  Todoist store '{store_name}': project_id={account_info['project_id']}"
                    )
                elif source_type == "notion" and category == "tasks":
                    # Notion task database - include database_id
                    account_info["database_id"] = getattr(
                        store, "notion_database_id", None
                    )
                    logger.info(
                        f"  Notion tasks store '{store_name}': database_id={account_info['database_id']}"
                    )

                available_accounts[category].append(account_info)
                logger.debug(f"  -> Added to {category}: {store_name}")

        except Exception as e:
            logger.warning(f"Failed to build available_accounts from stores: {e}")

        # Log final result
        for cat, accounts in available_accounts.items():
            if accounts:
                names = [a.get("name", "?") for a in accounts]
                logger.info(f"Available {cat} accounts: {names}")

        return available_accounts

    def _build_default_accounts(self) -> dict[str, dict]:
        """
        Build default_accounts dict from Smart Alias configuration.

        These are the fallback accounts when the LLM doesn't specify which account to use.

        Returns:
            Dict mapping categories (email, calendar, tasks, notification, schedule)
            to default account configuration.
        """
        default_accounts: dict[str, dict] = {}

        # Email default
        email_account_id = getattr(self.enricher, "action_email_account_id", None)
        if email_account_id:
            default_accounts["email"] = {
                "id": email_account_id,
                "provider": "google",  # Legacy: always Google
            }

        # Calendar default
        calendar_account_id = getattr(self.enricher, "action_calendar_account_id", None)
        calendar_id = getattr(self.enricher, "action_calendar_id", None)
        if calendar_account_id:
            default_accounts["calendar"] = {
                "id": calendar_account_id,
                "provider": "google",
                "calendar_id": calendar_id,
            }

        # Tasks default
        tasks_account_id = getattr(self.enricher, "action_tasks_account_id", None)
        tasks_list_id = getattr(self.enricher, "action_tasks_list_id", None)
        if tasks_account_id:
            default_accounts["tasks"] = {
                "id": tasks_account_id,
                "provider": "google",
                "list_id": tasks_list_id,
            }

        # Notes default (from document store)
        notes_store_id = getattr(self.enricher, "action_notes_store_id", None)
        if notes_store_id:
            try:
                from db import get_document_store_by_id

                notes_store = get_document_store_by_id(notes_store_id)
                if notes_store:
                    source_type = getattr(notes_store, "source_type", "")
                    # Map source type to provider name
                    provider = (
                        "notion"
                        if source_type == "notion"
                        else "onenote"
                        if source_type == "onenote"
                        else "gkeep"
                        if source_type == "mcp:gkeep"
                        else source_type
                    )
                    notes_config = {
                        "store_id": notes_store_id,
                        "provider": provider,
                        "name": getattr(notes_store, "name", ""),
                    }
                    # Add provider-specific fields
                    if provider == "notion":
                        notes_config["database_id"] = getattr(
                            notes_store, "notion_database_id", None
                        )
                    elif provider == "onenote":
                        notes_config["oauth_account_id"] = getattr(
                            notes_store, "microsoft_account_id", None
                        )
                        notes_config["notebook_id"] = getattr(
                            notes_store, "onenote_notebook_id", None
                        )
                    elif provider == "gkeep":
                        notes_config["oauth_account_id"] = getattr(
                            notes_store, "google_account_id", None
                        )
                    default_accounts["notes"] = notes_config
            except Exception as e:
                logger.warning(f"Failed to load notes store {notes_store_id}: {e}")

        # Notification URLs
        notification_urls = getattr(self.enricher, "action_notification_urls", None)
        if notification_urls:
            default_accounts["notification"] = {"urls": notification_urls}

        return default_accounts

    def _get_linked_live_sources(self) -> list:
        """Get live data sources linked to this enricher.

        If no specific sources are linked, returns ALL enabled sources.
        """
        if (
            hasattr(self.enricher, "_detached_live_sources")
            and self.enricher._detached_live_sources
        ):
            return self.enricher._detached_live_sources
        if (
            hasattr(self.enricher, "live_data_sources")
            and self.enricher.live_data_sources
        ):
            return self.enricher.live_data_sources
        # Fall back to all enabled live data sources
        from db import get_all_live_data_sources

        return [s for s in get_all_live_data_sources() if s.enabled]

    def _get_unified_source_for_live(self, source: Any) -> Optional[Any]:
        """
        Check if a unified source plugin exists for this live source.

        Unified sources combine RAG + Live into a single plugin with smart routing.
        When available, they provide better results by intelligently combining
        historical (RAG) and real-time (Live) data.

        Args:
            source: LiveDataSource or DetachedLiveDataSource

        Returns:
            Instantiated unified source plugin if available, None otherwise
        """
        source_type = getattr(source, "source_type", "")

        # Map legacy live source type to unified source type using dynamic lookup
        live_to_unified = _get_live_source_to_unified_map()
        unified_type = live_to_unified.get(source_type)
        if not unified_type:
            return None

        try:
            from plugin_base.loader import get_unified_source_plugin

            plugin_class = get_unified_source_plugin(unified_type)
            if not plugin_class:
                return None

            # Build config from the live source attributes
            config = {}

            # Google OAuth sources need account ID
            google_account_id = getattr(source, "google_account_id", None)
            if google_account_id:
                config["oauth_account_id"] = google_account_id

            # Add any other relevant config from the source
            # The unified source will use these for both RAG and Live operations

            return plugin_class(config)

        except Exception as e:
            logger.debug(f"Could not instantiate unified source for {source_type}: {e}")
            return None

    def _get_rag_search_fn_for_unified(self, source: Any) -> Optional[callable]:
        """
        Create a RAG search function for unified source smart routing.

        This function allows unified sources to query the RAG index as part of
        their smart routing logic. It searches linked document stores that
        correspond to the same data source (e.g., Gmail live + Gmail docs).

        Args:
            source: LiveDataSource being queried

        Returns:
            Callable (query, limit) -> list[str], or None if no matching stores
        """
        source_type = getattr(source, "source_type", "")

        # Map legacy live source type to unified source type using dynamic lookup
        live_to_unified = _get_live_source_to_unified_map()
        unified_type = live_to_unified.get(source_type)
        if not unified_type:
            return None

        # Get the document store source types this unified source handles
        try:
            from plugin_base.loader import get_unified_source_plugin

            plugin_class = get_unified_source_plugin(unified_type)
            if not plugin_class:
                return None
            doc_source_types = plugin_class.get_handled_doc_source_types()
        except ImportError:
            return None

        if not doc_source_types:
            return None

        # Find linked document stores that match any of this unified source's doc types
        linked_stores = self._get_linked_stores()
        matching_stores = [
            s
            for s in linked_stores
            if getattr(s, "source_type", "") in doc_source_types
            and getattr(s, "enabled", True)
            and getattr(s, "index_status", "") == "ready"
            and getattr(s, "collection_name", None)
        ]

        if not matching_stores:
            logger.debug(
                f"No matching document stores for unified source {unified_type} "
                f"(looking for {doc_source_types})"
            )
            return None

        def rag_search(query: str, limit: int = 10) -> list[str]:
            """Search matching RAG collections."""
            results = []
            try:
                from context.chroma import get_chroma_client
                from rag.retriever import RetrievalEngine

                chroma_client = get_chroma_client()

                for store in matching_stores:
                    collection_name = store.collection_name
                    if not collection_name:
                        continue

                    try:
                        collection = chroma_client.get_collection(collection_name)
                        # Simple semantic search
                        search_results = collection.query(
                            query_texts=[query],
                            n_results=min(limit, 5),  # Cap per store
                            include=["documents", "metadatas"],
                        )

                        if search_results and search_results.get("documents"):
                            for doc in search_results["documents"][0]:
                                if doc:
                                    results.append(doc)

                    except Exception as e:
                        logger.debug(f"RAG search failed for {collection_name}: {e}")

            except Exception as e:
                logger.debug(f"RAG search initialization failed: {e}")

            return results[:limit]

        return rag_search

    def enrich(
        self,
        messages: list[dict],
        system: str | None = None,
        session_key: str | None = None,
    ) -> EnrichmentResult:
        """
        Enrich a request with document and/or web context.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            session_key: Optional session identifier for session-scoped caching

        Returns:
            EnrichmentResult with augmented system/messages and metadata
        """
        self._session_key = session_key  # Store for use in _retrieve_live_context
        from providers.registry import ResolvedModel

        # Resolve target model first
        try:
            target_resolved = self.registry._resolve_actual_model(
                self.enricher.target_model
            )
            resolved = ResolvedModel(
                provider=target_resolved.provider,
                model_id=target_resolved.model_id,
                alias_name=self.enricher.name,
                alias_tags=self.enricher.tags,
            )
        except ValueError as e:
            logger.error(
                f"Enricher '{self.enricher.name}' target model not available: {e}"
            )
            raise

        # Get the user's query
        query = self._get_query(messages)
        if not query:
            logger.debug(f"Enricher '{self.enricher.name}': No query found in messages")
            return EnrichmentResult(
                resolved=resolved,
                enricher_id=self.enricher.id,
                enricher_name=self.enricher.name,
                enricher_tags=self.enricher.tags,
                augmented_system=system,
                augmented_messages=messages,
                context_injected=False,
                session_key=session_key,
            )

        # Collect context from enabled sources
        context_parts = []
        result_metadata = EnrichmentResult(
            resolved=resolved,
            enricher_id=self.enricher.id,
            enricher_name=self.enricher.name,
            enricher_tags=self.enricher.tags,
            augmented_system=system,
            augmented_messages=messages,
            session_key=session_key,
            # Set designator model upfront if configured
            designator_model=self.enricher.designator_model,
        )

        # Smart source selection - let designator choose which sources to use
        use_rag = self.enricher.use_rag
        use_web = self.enricher.use_web
        selected_stores = None  # None means use all stores
        store_budgets = None  # Token budgets per store (from smart selection)
        store_id_to_name = None  # Mapping for display
        web_budget = None  # Token budget for web search
        search_query = None  # Pre-computed search query from unified selection
        live_params = None  # Live data parameters from unified selection
        selected_model = None  # Model from unified routing selection

        if getattr(self.enricher, "use_smart_source_selection", False):
            # Get routing config if provided (for unified designator call)
            routing_config = getattr(self.enricher, "routing_config", None)

            selection = self._select_sources(
                query, messages=messages, routing_config=routing_config
            )
            if selection:
                use_rag = selection.get("use_rag", use_rag)
                use_web = selection.get("use_web", use_web)
                selected_stores = selection.get("store_ids")  # List of store IDs to use
                store_budgets = selection.get(
                    "store_budgets"
                )  # Token budgets per store
                store_id_to_name = selection.get("store_id_to_name")  # ID -> name
                web_budget = selection.get("web_budget")  # Token budget for web
                search_query = selection.get(
                    "search_query"
                )  # Pre-computed search query
                live_params = selection.get("live_params")  # Live data params
                selected_model = selection.get("selected_model")  # Routed model

                # Capture source selection designator usage
                if selection.get("designator_usage"):
                    result_metadata.designator_calls.append(
                        selection["designator_usage"]
                    )
                # Note: designator_model already set in constructor from self.enricher.designator_model

                # If routing selected a model, update the resolved target
                if selected_model:
                    try:
                        routed_resolved = self.registry._resolve_actual_model(
                            selected_model
                        )
                        resolved = ResolvedModel(
                            provider=routed_resolved.provider,
                            model_id=routed_resolved.model_id,
                            alias_name=self.enricher.name,
                            alias_tags=self.enricher.tags,
                        )
                        result_metadata.resolved = resolved
                        logger.info(f"Unified selection routed to: {selected_model}")
                    except ValueError as e:
                        logger.warning(
                            f"Selected model '{selected_model}' not available: {e}"
                        )

                logger.info(
                    f"Smart source selection: use_rag={use_rag}, use_web={use_web}, "
                    f"store_budgets={store_budgets or 'default'}, web_budget={web_budget or 'default'}, "
                    f"live_params={list(live_params.keys()) if live_params else 'none'}, "
                    f"selected_model={selected_model}"
                )

        # Run retrievals in parallel - web is slowest so benefits most from parallelism
        use_live_data = getattr(self.enricher, "use_live_data", False)

        # Use ThreadPoolExecutor for parallel retrieval
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        rag_context, rag_metadata = None, {}
        web_context, web_metadata = None, {}
        live_context, live_metadata = None, {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks - web first as it's slowest
            if use_web:
                futures["web"] = executor.submit(
                    self._retrieve_web_context,
                    query,
                    max_tokens=web_budget,
                    search_query=search_query,
                )
            if use_live_data:
                logger.info(f"Fetching live data context, live_params={live_params}")
                futures["live"] = executor.submit(
                    self._retrieve_live_context, query, live_params=live_params
                )
            if use_rag:
                futures["rag"] = executor.submit(
                    self._retrieve_rag_context,
                    query,
                    selected_store_ids=selected_stores,
                    store_budgets=store_budgets,
                )

            # Collect results as they complete
            for future in as_completed(futures.values()):
                # Find which key this future belongs to
                for key, f in futures.items():
                    if f == future:
                        try:
                            if key == "rag":
                                rag_context, rag_metadata = future.result()
                            elif key == "web":
                                web_context, web_metadata = future.result()
                            elif key == "live":
                                live_context, live_metadata = future.result()
                        except Exception as e:
                            logger.error(f"Error in {key} retrieval: {e}")
                        break

        # Process RAG results
        if use_rag:
            if rag_context:
                context_parts.append(("rag", rag_context))
            # Copy RAG metadata
            result_metadata.chunks_retrieved = rag_metadata.get("chunks_retrieved", 0)
            result_metadata.sources = rag_metadata.get("sources", [])
            result_metadata.stores_queried = rag_metadata.get("stores_queried", [])
            result_metadata.embedding_usage = rag_metadata.get("embedding_usage")
            result_metadata.embedding_model = rag_metadata.get("embedding_model")
            result_metadata.embedding_provider = rag_metadata.get("embedding_provider")

        # Process Web results
        if use_web:
            if web_context:
                context_parts.append(("web", web_context))
            # Copy web metadata
            result_metadata.search_query = web_metadata.get("search_query")
            result_metadata.scraped_urls = web_metadata.get("scraped_urls", [])
            # Capture web search designator usage
            if web_metadata.get("designator_usage"):
                web_usage = web_metadata["designator_usage"].copy()
                web_usage["purpose"] = "web_search_query"
                result_metadata.designator_calls.append(web_usage)
            # Note: designator_model already set in constructor from self.enricher.designator_model

        # Process Live data results
        if use_live_data:
            if live_context:
                context_parts.append(("live", live_context))

            # Also inject any previous session live context (accumulated from prior queries)
            # This preserves context across multiple live data requests (e.g., weather in London then Cornwall)
            if session_key:
                try:
                    from live.sources import get_session_live_context

                    prior_context = get_session_live_context(session_key)
                    if prior_context and prior_context != live_context:
                        # Only add if there's prior context different from current
                        context_parts.append(("live_history", prior_context))
                        logger.info(
                            f"Injected prior session live context ({len(prior_context)} chars)"
                        )
                except Exception as e:
                    logger.debug(f"Failed to get session live context: {e}")

            # Copy live metadata
            result_metadata.live_sources_queried = live_metadata.get(
                "sources_queried", []
            )
            result_metadata.live_data_errors = live_metadata.get("errors", [])

        # Determine enrichment type based on what was actually used
        if context_parts:
            types_used = [t for t, _ in context_parts]
            # Determine enrichment type - combinations become "hybrid"
            if len(types_used) > 1:
                result_metadata.enrichment_type = "hybrid"
            elif "rag" in types_used:
                result_metadata.enrichment_type = "rag"
            elif "web" in types_used:
                result_metadata.enrichment_type = "web"
            elif "live" in types_used:
                result_metadata.enrichment_type = "live"

            # Merge context
            merged_context = self._merge_context(context_parts)

            # Inject context into system prompt
            augmented_system = self._inject_context(system, merged_context)
            result_metadata.augmented_system = augmented_system
            result_metadata.context_injected = True

            # Store budget allocations for source attribution
            result_metadata.store_budgets = store_budgets
            result_metadata.store_id_to_name = store_id_to_name
            result_metadata.web_budget = web_budget
            result_metadata.total_budget = getattr(
                self.enricher, "max_context_tokens", 4000
            )
            result_metadata.show_sources = getattr(self.enricher, "show_sources", False)

            logger.info(
                f"Enricher '{self.enricher.name}': Injected {result_metadata.enrichment_type} context "
                f"(RAG: {result_metadata.chunks_retrieved} chunks, Web: {len(result_metadata.scraped_urls)} URLs)"
            )
        else:
            result_metadata.enrichment_type = "none"
            result_metadata.context_injected = False
            logger.debug(f"Enricher '{self.enricher.name}': No context to inject")

        # Handle memory if enabled
        if getattr(self.enricher, "use_memory", False):
            memory = self._get_memory_context()
            if memory:
                # Inject existing memory into system prompt
                result_metadata.augmented_system = self._inject_memory(
                    result_metadata.augmented_system, memory
                )
                result_metadata.memory_included = True
                logger.debug(
                    f"Enricher '{self.enricher.name}': Injected memory context"
                )

            # Always flag for memory update when memory is enabled
            # This allows building initial memory even when starting from empty
            result_metadata.memory_update_pending = True

        # Handle actions if enabled - inject action instructions into system prompt
        use_actions = getattr(self.enricher, "use_actions", False)
        allowed_actions = getattr(self.enricher, "allowed_actions", [])
        logger.info(
            f"Enricher '{self.enricher.name}': Actions check - use_actions={use_actions}, allowed_actions={allowed_actions}"
        )
        if use_actions:
            if allowed_actions:
                try:
                    from actions import get_system_prompt_for_actions

                    action_instructions = get_system_prompt_for_actions(allowed_actions)
                    logger.info(
                        f"Enricher '{self.enricher.name}': Action instructions length={len(action_instructions) if action_instructions else 0}"
                    )
                    if action_instructions:
                        current_system = result_metadata.augmented_system or ""
                        result_metadata.augmented_system = (
                            current_system + action_instructions
                        )
                        result_metadata.actions_enabled = True
                        result_metadata.allowed_actions = allowed_actions
                        # Pass default account IDs for action execution
                        result_metadata.action_email_account_id = getattr(
                            self.enricher, "action_email_account_id", None
                        )
                        result_metadata.action_calendar_account_id = getattr(
                            self.enricher, "action_calendar_account_id", None
                        )
                        result_metadata.action_calendar_id = getattr(
                            self.enricher, "action_calendar_id", None
                        )
                        result_metadata.action_tasks_account_id = getattr(
                            self.enricher, "action_tasks_account_id", None
                        )
                        result_metadata.action_tasks_list_id = getattr(
                            self.enricher, "action_tasks_list_id", None
                        )
                        result_metadata.action_notification_urls = getattr(
                            self.enricher, "action_notification_urls", None
                        )
                        # Build available accounts from linked document stores
                        result_metadata.available_accounts = (
                            self._build_available_accounts()
                        )
                        result_metadata.default_accounts = (
                            self._build_default_accounts()
                        )
                        # Scheduled prompts config (for schedule:prompt action)
                        result_metadata.scheduled_prompts_account_id = getattr(
                            self.enricher, "scheduled_prompts_account_id", None
                        )
                        result_metadata.scheduled_prompts_calendar_id = getattr(
                            self.enricher, "scheduled_prompts_calendar_id", None
                        )
                        logger.info(
                            f"Enricher '{self.enricher.name}': Injected action instructions ({len(action_instructions)} chars)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to inject action instructions: {e}")

        # Aggregate designator calls into total usage
        if result_metadata.designator_calls:
            total_prompt = sum(
                c.get("prompt_tokens", 0) for c in result_metadata.designator_calls
            )
            total_completion = sum(
                c.get("completion_tokens", 0) for c in result_metadata.designator_calls
            )
            result_metadata.designator_usage = {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
            }

        return result_metadata

    def _get_query(self, messages: list[dict], max_chars: int = 2000) -> str:
        """
        Extract the user's query from messages.

        For short queries (< 50 chars) that might be follow-ups like "Yes", "No", "Do it",
        include preceding context so the designator understands what the user is responding to.
        """
        # Find the last user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_content = content
                elif isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    last_user_content = " ".join(texts)
                break

        if not last_user_content:
            return ""

        # For short follow-up queries, include conversation context
        if len(last_user_content) < 50 and len(messages) > 1:
            # Build context from recent messages (last assistant message + user query)
            context_parts = []
            for msg in messages[-4:]:  # Last 4 messages for context
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                if content and role in ("user", "assistant"):
                    # Truncate long messages
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_parts.append(f"{role.upper()}: {content}")

            if context_parts:
                context = "\n".join(context_parts)
                return context[:max_chars]

        return last_user_content[:max_chars]

    def _get_preview_samples(
        self, query: str, store_info: list[dict], chunks_per_store: int = 5
    ) -> dict[int, list[str]]:
        """
        Retrieve preview samples from each store for two-pass retrieval.

        Args:
            query: The user's query
            store_info: List of store info dicts with 'id' and 'name'
            chunks_per_store: Number of chunks to retrieve per store

        Returns:
            Dict mapping store_id to list of chunk text samples
        """
        samples = {}
        linked_stores = self._get_linked_stores()

        try:
            from rag import get_retriever

            retriever = get_retriever()

            # Query each store individually to get samples
            for info in store_info:
                store_id = info["id"]
                # Find the store object
                store = next(
                    (s for s in linked_stores if getattr(s, "id", None) == store_id),
                    None,
                )
                if not store:
                    continue

                try:
                    result = retriever.retrieve_from_stores(
                        stores=[store],
                        query=query,
                        max_results=chunks_per_store,
                        similarity_threshold=0.3,  # Lower threshold for preview
                        rerank_provider="none",  # Skip reranking for speed
                    )

                    if result.chunks:
                        # Extract just the content, truncated
                        samples[store_id] = [
                            chunk.content[:300] + "..."
                            if len(chunk.content) > 300
                            else chunk.content
                            for chunk in result.chunks[:chunks_per_store]
                        ]
                except Exception as e:
                    logger.warning(f"Failed to get preview from store {store_id}: {e}")

        except Exception as e:
            logger.error(f"Two-pass preview retrieval failed: {e}")

        return samples

    def _select_sources(
        self,
        query: str,
        messages: list[dict] | None = None,
        routing_config: dict | None = None,
    ) -> dict | None:
        """
        Use the designator model to select sources and allocate budget in a single call.

        Handles RAG stores, web search, live data sources, AND model routing together.

        Args:
            query: The user's query text
            messages: Full message list (needed for model routing context)
            routing_config: If provided, also select model. Dict with:
                - candidates: List of candidate model dicts
                - purpose: Router purpose description
                - token_count: Estimated request tokens
                - has_images: Whether request contains images

        Returns:
            Dict with budget allocations, live data parameters, and optionally selected_model
        """
        if not self.enricher.designator_model:
            logger.warning(
                "Smart source selection enabled but no designator model configured"
            )
            return None

        # Build list of available RAG stores
        linked_stores = self._get_linked_stores()
        store_info = []
        for store in linked_stores:
            enabled = getattr(store, "enabled", True)
            status = getattr(store, "index_status", None)
            if enabled and status == "ready":
                store_id = getattr(store, "id", None)
                store_name = getattr(store, "name", "unknown")
                description = (
                    getattr(store, "description", None)
                    or f"Document store: {store_name}"
                )
                themes = getattr(store, "themes", None) or []
                best_for = getattr(store, "best_for", None)
                content_summary = getattr(store, "content_summary", None)

                store_info.append(
                    {
                        "id": store_id,
                        "name": store_name,
                        "description": description,
                        "themes": themes,
                        "best_for": best_for,
                        "content_summary": content_summary,
                    }
                )

        # Build list of available live data sources
        linked_live_sources = self._get_linked_live_sources()
        live_source_info = []
        mcp_tool_info = []  # Separate list for MCP tools
        use_live_data = getattr(self.enricher, "use_live_data", False)
        if use_live_data:
            for source in linked_live_sources:
                if not getattr(source, "enabled", True):
                    continue
                name = getattr(source, "name", "")
                source_type = getattr(source, "source_type", "")
                data_type = getattr(source, "data_type", "")
                best_for = getattr(source, "best_for", "")
                description = getattr(source, "description", "")

                # MCP sources provide their own tool descriptions
                if source_type == "mcp_server":
                    from live.sources import MCPProvider

                    try:
                        provider = MCPProvider(source)
                        if provider.is_available():
                            tools = provider.list_tools()
                            for tool in tools:
                                tool_name = tool.get("name", "")
                                tool_desc = tool.get("description", "No description")
                                if len(tool_desc) > 150:
                                    tool_desc = tool_desc[:147] + "..."
                                schema = tool.get("inputSchema", {})
                                required = schema.get("required", [])
                                properties = schema.get("properties", {})

                                # Build param hints from schema
                                param_parts = []
                                for param_name in required:
                                    prop = properties.get(param_name, {})
                                    param_type = prop.get("type", "string")
                                    param_parts.append(f"{param_name} ({param_type})")

                                mcp_tool_info.append(
                                    {
                                        "source_name": name,
                                        "source_description": description,
                                        "source_data_type": data_type,
                                        "source_best_for": best_for,
                                        "tool_name": tool_name,
                                        "description": tool_desc,
                                        "params": param_parts,
                                    }
                                )
                    except Exception as e:
                        logger.warning(f"Failed to get MCP tools from {name}: {e}")
                    continue

                # Parameter hints based on source type (non-MCP)
                # First check if there's a plugin that can provide hints
                param_hints = ""
                try:
                    from plugin_base.loader import get_live_source_plugin

                    plugin_class = get_live_source_plugin(source_type)
                    if plugin_class:
                        # Use plugin's designator hint (includes params and best_for)
                        param_hints = plugin_class.get_designator_hint()
                        # Also update data_type and best_for from plugin if not set
                        if not data_type:
                            data_type = getattr(plugin_class, "data_type", "")
                        if not best_for:
                            best_for = getattr(plugin_class, "best_for", "")
                except ImportError:
                    pass

                # Fall back to hardcoded hints for non-plugin sources
                if not param_hints:
                    if source_type == "builtin_weather":
                        param_hints = (
                            "Parameters: location (city name), type (current|forecast)"
                        )
                    elif source_type == "builtin_stocks":
                        param_hints = "Parameters: symbol (US stock ticker like AAPL, MSFT, NVDA). For portfolio queries, extract ALL stock symbols from user context and create separate entries for each."
                    elif source_type == "builtin_alpha_vantage":
                        param_hints = "Parameters: symbol (UK stock ticker like LGEN.LON, TSCO.LON, or fund name like 'L&G Global Technology'), period (optional: 1W, 1M, 3M, 6M, 1Y for historical data)"
                    elif source_type == "builtin_transport":
                        param_hints = "Parameters: station (DEPARTURE station - use user's home location from context), destination (optional), type (departures|arrivals). UK train departures ONLY - for full journey planning with connections, use 'routes' with mode=transit instead."
                    elif source_type == "gmail":
                        param_hints = "Parameters: action (today|unread|recent|search). Use action='today' for today's emails, action='unread' for unread, action='search' with query for specific searches (e.g. from:sender subject:topic)"
                    elif source_type == "builtin_routes":
                        param_hints = "Parameters: origin, destination, mode (drive|walk|bicycle|transit). Optional: arrival_time (ARRIVE BY - for events like '3pm kick off') OR departure_time (LEAVE AT) - use one, the other, or neither. PREFER THIS for all journey planning."
                    elif source_type == "builtin_google_maps":
                        param_hints = "Parameters: action (search|details|nearby|directions), query (search term), location (lat,lng or address), place_id (for details). Use for places search, business info, nearby POIs. For directions between cities, prefer 'routes' source instead."

                live_source_info.append(
                    {
                        "name": name,
                        "source_type": source_type,
                        "data_type": data_type,
                        "best_for": best_for,
                        "description": description,
                        "param_hints": param_hints,
                    }
                )

        # Check if we have anything to select from
        has_rag = bool(store_info)
        has_web = self.enricher.use_web
        has_live = bool(live_source_info)
        has_mcp = bool(mcp_tool_info)

        if has_live:
            logger.info(
                f"Live sources for designator: {[s['name'] + ' [' + s.get('param_hints', '') + ']' for s in live_source_info]}"
            )
        if has_mcp:
            logger.info(
                f"MCP tools for designator: {len(mcp_tool_info)} tools from {set(t['source_name'] for t in mcp_tool_info)}"
            )

        if not has_rag and not has_web and not has_live and not has_mcp:
            logger.debug("No sources available for selection")
            return None

        # Two-pass retrieval: get preview samples from each store first
        preview_samples = {}
        use_two_pass = getattr(self.enricher, "use_two_pass_retrieval", False)
        if use_two_pass and store_info:
            logger.info("Two-pass retrieval: fetching preview samples from each store")
            preview_samples = self._get_preview_samples(
                query, store_info, chunks_per_store=5
            )

        # Try cache lookup for live data params first
        cached_live_params = {}
        if has_live:
            cached_live_params = self._try_cached_params(query, linked_live_sources)

        # Build the prompt sections
        sources_text = ""

        # RAG stores section
        if store_info:
            sources_text += "DOCUMENT STORES (indexed content for semantic search):\n"
            for s in store_info:
                sources_text += f"- store_{s['id']}: {s['name']}\n"
                sources_text += f"  Description: {s['description']}\n"
                if s.get("themes"):
                    sources_text += f"  Themes: {', '.join(s['themes'])}\n"
                if s.get("best_for"):
                    sources_text += f"  Best for: {s['best_for']}\n"
                if s.get("content_summary"):
                    sources_text += f"  Content: {s['content_summary']}\n"
                store_samples = preview_samples.get(s["id"], [])
                if store_samples:
                    sources_text += f"  SAMPLE CONTENT ({len(store_samples)} chunks):\n"
                    for i, sample in enumerate(store_samples, 1):
                        sample_text = sample.replace("\n", " ")[:250]
                        sources_text += f"    [{i}] {sample_text}\n"
            sources_text += "\n"

        # Web search section
        if has_web:
            sources_text += "WEB SEARCH (real-time internet search and scraping):\n"
            sources_text += "- web: Search the internet for current information\n"
            sources_text += "  Best for: Current events, news, recent information not in document stores\n\n"

        # Live data sources section (builtin providers)
        if live_source_info:
            sources_text += "LIVE DATA SOURCES (real-time API queries - READ ONLY):\n"
            sources_text += "These sources FETCH data only. They CANNOT perform actions like complete/update/delete/send.\n"
            sources_text += "For write operations, the target LLM will use Smart Actions separately.\n"
            sources_text += "(Use the source NAME in live_params, e.g., 'weather', 'stocks', 'transport')\n"
            for s in live_source_info:
                sources_text += f"- {s['name']}"
                if s.get("data_type"):
                    sources_text += f" [{s['data_type']}]"
                sources_text += "\n"
                if s.get("description"):
                    sources_text += f"  Description: {s['description']}\n"
                if s.get("best_for"):
                    sources_text += f"  Best for: {s['best_for']}\n"
                if s.get("param_hints"):
                    sources_text += f"  {s['param_hints']}\n"
            sources_text += "\n"

        # MCP API tools section (from MCP servers like RapidAPI)
        if mcp_tool_info:
            # Group tools by source
            mcp_by_source: dict[str, list] = {}
            mcp_source_meta: dict[str, dict] = {}  # Store source-level metadata
            for tool in mcp_tool_info:
                src = tool["source_name"]
                if src not in mcp_by_source:
                    mcp_by_source[src] = []
                    mcp_source_meta[src] = {
                        "description": tool.get("source_description", ""),
                        "data_type": tool.get("source_data_type", ""),
                        "best_for": tool.get("source_best_for", ""),
                    }
                mcp_by_source[src].append(tool)

            sources_text += "MCP API TOOLS (call specific API endpoints):\n"
            sources_text += "IMPORTANT: Most MCP APIs require ID lookups. Use agentic mode for multi-step queries:\n"
            sources_text += '  {"source_name": [{"agentic": true, "goal": "describe what you need"}]}\n'
            sources_text += (
                "Or call specific tools directly if you know the parameters:\n"
            )
            sources_text += '  {"source_name": [{"tool_name": "ToolName", "tool_args": {"param": "value"}}]}\n\n'

            for src_name, tools in mcp_by_source.items():
                meta = mcp_source_meta.get(src_name, {})
                sources_text += f"{src_name} API"
                if meta.get("data_type"):
                    sources_text += f" [{meta['data_type']}]"
                sources_text += f" ({len(tools)} tools):\n"
                if meta.get("description"):
                    sources_text += f"  Description: {meta['description']}\n"
                if meta.get("best_for"):
                    sources_text += f"  Best for: {meta['best_for']}\n"
                sources_text += "  Tools:\n"
                for tool in tools:  # Show ALL tools - designator has large context
                    sources_text += (
                        f"    - {tool['tool_name']}: {tool['description']}\n"
                    )
                    if tool.get("params"):
                        sources_text += f"      Required: {', '.join(tool['params'])}\n"
                sources_text += "\n"

        # Get total context budget - designator allocates the full budget
        max_context = getattr(self.enricher, "max_context_tokens", 4000)
        priority_budget = max_context  # Full budget for designator to allocate

        # Build routing section if routing is enabled
        routing_text = ""
        routing_instructions = ""
        routing_json_field = ""
        routing_rules = ""
        if routing_config and routing_config.get("candidates"):
            candidates = routing_config["candidates"]
            purpose = routing_config.get("purpose", "General purpose routing")
            token_count = routing_config.get("token_count", 0)
            has_images = routing_config.get("has_images", False)

            routing_text = "MODEL ROUTING:\n"
            routing_text += f"Purpose: {purpose}\n"
            routing_text += f"Request: ~{token_count} tokens, images={'yes' if has_images else 'no'}\n"
            routing_text += "Available models:\n"
            for c in candidates:
                model = c.get("model", "")
                ctx = c.get("context_length", 0)
                caps = ", ".join(c.get("capabilities", [])) or "general"
                notes = c.get("notes", "")
                routing_text += f"- {model} (context: {ctx}, caps: {caps})"
                if notes:
                    routing_text += f"\n  {notes[:200]}"
                routing_text += "\n"
            routing_text += "\n"

            routing_instructions = "\n4. Select the best model for this request"
            routing_json_field = ',\n  "selected_model": "provider/model-id"'
            routing_rules = (
                '\n- "selected_model": The best model from the list for this query'
            )

        # Get user context: intelligence (static) and memory (learned)
        # Note: purpose describes the alias role, not user info, so we don't include it here
        intelligence = getattr(self.enricher, "intelligence", "") or ""
        memory = getattr(self.enricher, "memory", "") or ""

        user_context_section = ""
        user_context_parts = []
        if intelligence:
            user_context_parts.append(f"User Profile:\n{intelligence}")
        if memory:
            user_context_parts.append(f"Learned Information:\n{memory}")

        if user_context_parts:
            user_context_section = f"""USER CONTEXT:
{chr(10).join(user_context_parts)}

"""

        prompt = f"""You are a source selection assistant. Analyze the query and:
1. Allocate token budget to document stores and web search based on relevance
2. If web search is relevant, generate an optimized search query
3. For any relevant live data sources, extract the specific parameters needed{routing_instructions}

TOTAL BUDGET: {priority_budget} tokens to distribute across document stores and web search.
Sources only get tokens you explicitly allocate - there is no baseline. Allocate 0 to irrelevant sources.

{user_context_section}{routing_text}{sources_text}
USER QUERY:
{query}

Respond with a JSON object:
{{
  "allocations": {{"store_ID": extra_tokens, "web": extra_tokens}},
  "search_query": "optimized web search query if web is used",
  "live_params": {{"source_name": [{{"param": "value"}}]}}{routing_json_field}
}}

Rules:
- "allocations": Distribute the FULL {priority_budget} token budget across relevant stores/web based on relevance. Aim to use most or all of the budget. Give more tokens to highly relevant sources, fewer to less relevant ones.
- "search_query": If allocating to web, provide an optimized 3-8 word search query. Omit if not using web.
- "live_params": Array of parameter objects per source. Use multiple entries for multiple queries (e.g., multiple stock symbols, multi-leg journeys, multiple cities).
  Examples for builtin sources:
  - Single stock: {{"stocks": [{{"symbol": "AAPL"}}]}}
  - Multiple stocks (portfolio): {{"stocks": [{{"symbol": "NVDA"}}, {{"symbol": "U"}}, {{"symbol": "AAPL"}}]}}
  - Multi-city weather: {{"weather": [{{"location": "London"}}, {{"location": "Paris"}}]}}
  - Multi-leg train journey: {{"transport": [{{"station": "KGX"}}, {{"station": "EDB"}}]}}
  Examples for MCP API tools (PREFER agentic mode - most APIs require ID lookups):
  - Sports data (needs team/event lookup): {{"sofasport": [{{"agentic": true, "goal": "get latest QPR match results and upcoming fixtures"}}]}}
  - News search: {{"google-news-mcp": [{{"tool_name": "Search", "tool_args": {{"keyword": "technology"}}}}]}}
  - Finance data: {{"yahoo-finance15": [{{"agentic": true, "goal": "get Apple stock price and recent performance"}}]}}
  - Direct tool call (when you know exact params): {{"source": [{{"tool_name": "ToolName", "tool_args": {{"id": "123"}}}}]}}
- If no live data needed, use empty object: "live_params": {{}}{routing_rules}
- Live data sources provide real-time API data. PREFER live data sources over web search when a matching source exists.
- IMPORTANT: Live data sources are READ-ONLY. Do NOT include action params like "action": "complete" or "action": "delete". For write operations (complete task, send email, etc.), just fetch the relevant data - the target LLM handles actions separately.
- For MCP APIs, PREFER agentic mode when the query involves entities that need ID lookups (teams, players, companies, locations).
- TIP: It's better to allocate the full budget across multiple sources than to leave budget unused

Respond with ONLY the JSON object."""

        try:
            designator_resolved = self.registry._resolve_actual_model(
                self.enricher.designator_model
            )
            provider = designator_resolved.provider

            result = provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 800, "temperature": 0},
            )

            response_text = result.get("content", "").strip()

            logger.debug(f"Designator prompt:\n{prompt}")
            logger.info(f"Designator response: {response_text}")

            # Parse JSON response - handle nested objects
            import json
            import re

            # Find the outermost JSON object
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                allocations = selection.get("allocations", {})
                search_query = selection.get("search_query")
                live_params = selection.get("live_params", {})
                selected_model = selection.get("selected_model")

                # Parse priority allocations for stores
                valid_store_ids = {s["id"] for s in store_info}
                priority_budgets = {}
                web_priority = 0

                for key, budget in allocations.items():
                    if not isinstance(budget, (int, float)) or budget <= 0:
                        continue
                    budget = int(budget)

                    if key == "web":
                        web_priority = budget if has_web else 0
                    elif key.startswith("store_"):
                        try:
                            store_id = int(key.replace("store_", ""))
                            if store_id in valid_store_ids:
                                priority_budgets[store_id] = budget
                        except ValueError:
                            pass
                    else:
                        try:
                            store_id = int(key)
                            if store_id in valid_store_ids:
                                priority_budgets[store_id] = budget
                        except ValueError:
                            pass

                # Build final store budgets - only include stores with allocation > 0
                store_budgets = {}
                for store_id in valid_store_ids:
                    budget = priority_budgets.get(store_id, 0)
                    if budget > 0:
                        store_budgets[store_id] = budget

                # Web budget
                web_budget = web_priority if has_web else 0

                # Process live params - merge with cache and validate
                # New format: {"source_name": [{"param": "value"}, ...]} (array)
                # Legacy format: {"source_name": {"param": "value"}} (single dict)
                # Include both builtin sources and MCP sources
                valid_live_names = {s["name"] for s in live_source_info}
                valid_live_names.update({t["source_name"] for t in mcp_tool_info})
                final_live_params = {}

                logger.debug(
                    f"Designator live_params raw: {live_params}, valid_names: {valid_live_names}"
                )

                for name, params_data in live_params.items():
                    if name not in valid_live_names:
                        continue

                    # Normalize to list format
                    if isinstance(params_data, list):
                        params_list = params_data
                    elif isinstance(params_data, dict):
                        # Legacy single dict format
                        params_list = [params_data]
                    else:
                        continue

                    # Process each param set in the array
                    processed_params = []
                    for params in params_list:
                        if not isinstance(params, dict):
                            continue
                        # Merge with cached params if available
                        if name in cached_live_params:
                            merged = cached_live_params[name].copy()
                            for k, v in params.items():
                                if k not in merged:
                                    merged[k] = v
                            params = merged
                        else:
                            # Only set query as fallback if designator didn't provide one
                            # Don't overwrite designator-extracted queries (e.g., Gmail search syntax)
                            if (
                                "query" not in params
                                and "action" not in params
                                and "tool_name" not in params
                            ):
                                params["query"] = query
                            # Cache for future use (only cache first param set)
                            if not processed_params:
                                self._cache_extracted_params(
                                    name, params, linked_live_sources
                                )
                        processed_params.append(params)

                    if processed_params:
                        final_live_params[name] = processed_params

                use_rag = bool(store_budgets)
                use_web = web_budget > 0
                store_ids = list(store_budgets.keys()) if store_budgets else None
                store_id_to_name = {s["id"]: s["name"] for s in store_info}

                # Validate selected_model if routing was requested
                if routing_config and selected_model:
                    valid_models = [
                        c.get("model") for c in routing_config.get("candidates", [])
                    ]
                    if selected_model not in valid_models:
                        logger.warning(
                            f"Designator returned invalid model '{selected_model}', ignoring"
                        )
                        selected_model = None

                logger.info(
                    f"Unified source selection: stores={store_budgets}, web={web_budget}, "
                    f"search_query={search_query}, live_params={list(final_live_params.keys())}, "
                    f"selected_model={selected_model}"
                )

                # Capture designator usage
                designator_usage = {
                    "prompt_tokens": result.get("input_tokens", 0),
                    "completion_tokens": result.get("output_tokens", 0),
                    "purpose": "source_selection",
                }

                return {
                    "use_rag": use_rag,
                    "use_web": use_web,
                    "store_ids": store_ids,
                    "store_budgets": store_budgets if store_budgets else None,
                    "store_id_to_name": store_id_to_name,
                    "web_budget": web_budget if web_budget > 0 else None,
                    "search_query": search_query,
                    "live_params": final_live_params if final_live_params else None,
                    "selected_model": selected_model,
                    "designator_usage": designator_usage,
                    "designator_model": self.enricher.designator_model,
                }

            logger.warning(
                f"Could not parse source selection response: {response_text}"
            )
            return None

        except Exception as e:
            logger.error(f"Source selection failed: {e}")
            return None

    def _retrieve_rag_context(
        self,
        query: str,
        selected_store_ids: list[int] | None = None,
        store_budgets: dict[int, int] | None = None,
    ) -> tuple[str, dict]:
        """
        Retrieve context from linked document stores.

        Args:
            query: The user's query
            selected_store_ids: If provided, only use stores with these IDs
            store_budgets: If provided, token budget per store ID (from smart selection)

        Returns:
            Tuple of (context string, metadata dict)
        """
        metadata = {
            "chunks_retrieved": 0,
            "sources": [],
            "stores_queried": [],
            "embedding_usage": None,
            "embedding_model": None,
            "embedding_provider": None,
        }

        linked_stores = self._get_linked_stores()

        # Filter to selected stores if specified
        if selected_store_ids is not None:
            linked_stores = [
                s for s in linked_stores if getattr(s, "id", None) in selected_store_ids
            ]

        if not linked_stores or not self._has_ready_stores(linked_stores):
            logger.debug(f"Enricher '{self.enricher.name}': No ready document stores")
            return "", metadata

        try:
            from rag import get_retriever

            retriever = get_retriever()
            result = retriever.retrieve_from_stores(
                stores=linked_stores,
                query=query,
                max_results=self.enricher.max_results,
                similarity_threshold=self.enricher.similarity_threshold,
                rerank_provider=self.enricher.rerank_provider,
                rerank_model=self.enricher.rerank_model,
                rerank_top_n=self.enricher.rerank_top_n,
            )

            if not result.chunks:
                metadata["stores_queried"] = result.stores_queried or []
                metadata["embedding_usage"] = result.embedding_usage
                metadata["embedding_model"] = result.embedding_model
                metadata["embedding_provider"] = result.embedding_provider
                return "", metadata

            # Calculate max tokens for RAG context
            if store_budgets:
                # Use sum of allocated store budgets
                max_tokens = sum(store_budgets.values())
            elif self.enricher.use_web:
                # Legacy: allocate tokens based on priority when both RAG and Web enabled
                priority = getattr(self.enricher, "context_priority", "balanced")
                if priority == "prefer_rag":
                    max_tokens = int(self.enricher.max_context_tokens * 0.7)
                elif priority == "prefer_web":
                    max_tokens = int(self.enricher.max_context_tokens * 0.3)
                else:  # balanced
                    max_tokens = self.enricher.max_context_tokens // 2
            else:
                max_tokens = self.enricher.max_context_tokens

            context = retriever.format_context(
                result,
                max_tokens=max_tokens,
                include_sources=True,
                include_store_names=len(linked_stores) > 1,
            )

            metadata["chunks_retrieved"] = len(result.chunks)
            metadata["sources"] = list(
                set(chunk.source_file for chunk in result.chunks)
            )
            # Get stores that actually contributed chunks (not just queried)
            metadata["stores_queried"] = list(
                set(chunk.store_name for chunk in result.chunks if chunk.store_name)
            )
            metadata["embedding_usage"] = result.embedding_usage
            metadata["embedding_model"] = result.embedding_model
            metadata["embedding_provider"] = result.embedding_provider

            return context, metadata

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return "", metadata

    def _retrieve_web_context(
        self, query: str, max_tokens: int | None = None, search_query: str | None = None
    ) -> tuple[str, dict]:
        """
        Retrieve context from web search and scraping.

        Args:
            query: The user's query
            max_tokens: If provided, override the default token budget for web context.
                       When from smart source selection, this determines dynamic search/scrape limits.
            search_query: If provided, use this pre-computed search query (from unified selection)

        Returns:
            Tuple of (context string, metadata dict)
        """
        metadata = {
            "search_query": None,
            "scraped_urls": [],
            "designator_usage": None,
        }

        # Use pre-computed search query if provided, otherwise generate one
        if search_query:
            # Search query already computed by unified _select_sources
            pass
        elif self.enricher.designator_model:
            # Fallback: generate optimized search query (non-smart-selection path)
            optimized_query, designator_usage = self._call_designator(query)
            if optimized_query:
                search_query = optimized_query
            else:
                search_query = query
            metadata["designator_usage"] = designator_usage
        else:
            search_query = query

        metadata["search_query"] = search_query

        # Calculate max tokens for web context
        if max_tokens is None:
            # Legacy: allocate tokens based on priority when both RAG and Web enabled
            if self.enricher.use_rag:
                priority = getattr(self.enricher, "context_priority", "balanced")
                if priority == "prefer_rag":
                    max_tokens = int(self.enricher.max_context_tokens * 0.3)
                elif priority == "prefer_web":
                    max_tokens = int(self.enricher.max_context_tokens * 0.7)
                else:  # balanced
                    max_tokens = self.enricher.max_context_tokens // 2
            else:
                max_tokens = self.enricher.max_context_tokens

        # Calculate dynamic search/scrape limits based on token budget
        # When designator allocates tokens, scale search results and scrape URLs accordingly
        # Default values from enricher config serve as maximums
        default_max_results = self.enricher.max_search_results
        default_max_scrape = self.enricher.max_scrape_urls
        total_context = self.enricher.max_context_tokens

        if max_tokens and total_context > 0:
            # Scale based on token allocation ratio
            # Higher allocation = more search results and URLs to scrape
            ratio = max_tokens / total_context

            # At full budget: use configured maximums
            # At low budget: reduce proportionally (minimum 1 result, 1 URL)
            # Use a curve that gives reasonable results at lower allocations
            dynamic_max_results = max(1, int(default_max_results * min(1.0, ratio * 2)))
            dynamic_max_scrape = max(1, int(default_max_scrape * min(1.0, ratio * 2)))

            logger.debug(
                f"Dynamic web limits: {max_tokens}/{total_context} tokens = "
                f"{dynamic_max_results} results (max {default_max_results}), "
                f"{dynamic_max_scrape} URLs (max {default_max_scrape})"
            )
        else:
            # No token budget specified - use defaults
            dynamic_max_results = default_max_results
            dynamic_max_scrape = default_max_scrape

        # Execute web search with dynamic limit
        search_results, urls_with_context = self._execute_search(
            search_query, max_results=dynamic_max_results
        )
        if not search_results:
            return "", metadata

        context_parts = [search_results]

        # Rerank and scrape URLs with dynamic limit
        if urls_with_context:
            urls = self._rerank_urls(search_query, urls_with_context)
            if urls:
                scrape_results, scraped_urls = self._execute_scrape(
                    urls, max_tokens, max_urls=dynamic_max_scrape
                )
                if scrape_results:
                    context_parts.append(scrape_results)
                    metadata["scraped_urls"] = scraped_urls

        return "\n\n".join(context_parts), metadata

    def _retrieve_live_context(
        self, query: str, live_params: dict[str, dict] | None = None
    ) -> tuple[str, dict]:
        """
        Retrieve context from live data sources.

        Args:
            query: The user's query
            live_params: If provided, pre-computed parameters from unified selection.
                        Dict mapping source name to params dict.

        Returns:
            Tuple of (context string, metadata dict)
        """
        metadata = {
            "sources_queried": [],
            "errors": [],
            "params_extracted": {},
        }

        linked_sources = self._get_linked_live_sources()
        if not linked_sources:
            logger.debug(
                f"Enricher '{self.enricher.name}': No live data sources linked"
            )
            return "", metadata

        # Filter to enabled sources only
        enabled_sources = [s for s in linked_sources if getattr(s, "enabled", True)]
        if not enabled_sources:
            logger.debug(
                f"Enricher '{self.enricher.name}': No enabled live data sources"
            )
            return "", metadata

        # Use pre-computed params if provided, otherwise query all sources
        # live_params format: {"source_name": [{"param": "value"}, ...]} (array of param sets)
        if live_params:
            source_params = live_params
            metadata["params_extracted"] = source_params
            # Only query sources that have params (deemed relevant by designator)
            # Each source can have multiple param sets (array) for multiple queries
            sources_to_query = []
            for source in enabled_sources:
                source_name = getattr(source, "name", "")
                if source_name in source_params:
                    params_list = source_params[source_name]
                    # Handle both array format (new) and single dict format (legacy)
                    if isinstance(params_list, list):
                        # For stocks with multiple symbols, consolidate into a portfolio fetch
                        # This ensures funds in user memory are also fetched via web fallback
                        if source_name == "stocks" and len(params_list) > 1:
                            # Multiple stock symbols = portfolio query
                            # Use a single portfolio fetch instead of individual queries
                            sources_to_query.append(
                                (source, {"_portfolio_fetch": True, "query": query})
                            )
                        else:
                            for params in params_list:
                                sources_to_query.append((source, params))
                    elif isinstance(params_list, dict):
                        # Legacy single dict format
                        sources_to_query.append((source, params_list))
        else:
            # No smart selection params - don't query any live sources
            # The designator should explicitly select relevant sources via live_params
            # Querying all sources with raw query leads to spurious API calls
            # (e.g., weather API called with "what size strap does watch use?" as location)
            sources_to_query = []

        if not sources_to_query:
            logger.debug(
                f"Enricher '{self.enricher.name}': Designator found no relevant live sources"
            )
            return "", metadata

        try:
            from live import get_provider_for_source

            # Add user context to params for placeholder resolution
            user_intelligence = getattr(self.enricher, "intelligence", "") or ""
            user_purpose = getattr(self.enricher, "purpose", "") or ""
            user_memory = getattr(self.enricher, "memory", "") or ""

            context_parts = []
            sources_already_logged = set()  # Track which sources we've logged

            for source, params in sources_to_query:
                source_name = getattr(source, "name", "")
                # Only add to sources_queried once per source (not per query)
                if source_name not in sources_already_logged:
                    metadata["sources_queried"].append(source_name)
                    sources_already_logged.add(source_name)

                try:
                    # Check if a unified source plugin exists for this live source
                    # Unified sources combine RAG + Live with smart routing
                    unified_source = self._get_unified_source_for_live(source)
                    if unified_source:
                        # Use unified source's query() method with smart routing
                        logger.debug(
                            f"Using unified source for {source_name} ({getattr(source, 'source_type', '')})"
                        )

                        # Build a RAG search function if we have linked document stores
                        # that match this unified source type
                        rag_search_fn = self._get_rag_search_fn_for_unified(source)

                        import time

                        start_time = time.time()
                        unified_result = unified_source.query(
                            query=params.get("query", query),
                            params=params,
                            rag_search_fn=rag_search_fn,
                        )
                        latency_ms = int((time.time() - start_time) * 1000)

                        if unified_result.success and unified_result.formatted:
                            context_parts.append(unified_result.formatted)
                            # Log unified source usage
                            logger.info(
                                f"Unified source {source_name}: "
                                f"routing={unified_result.routing_used.value if unified_result.routing_used else 'n/a'}, "
                                f"rag={unified_result.rag_count}, live={unified_result.live_count}, "
                                f"dedupe={unified_result.dedupe_count}, time={latency_ms}ms"
                            )
                            # Update source status
                            from db.live_data_sources import (
                                record_live_data_call,
                                update_live_data_source_status,
                            )

                            update_live_data_source_status(source.id, success=True)
                            record_live_data_call(
                                source.id,
                                "unified_query",
                                success=True,
                                latency_ms=latency_ms,
                            )
                        elif unified_result.error:
                            metadata["errors"].append(
                                f"{source_name}: {unified_result.error}"
                            )

                        continue  # Skip legacy provider path

                    # Fall back to legacy provider
                    provider = get_provider_for_source(source)
                    if not provider.is_available():
                        metadata["errors"].append(f"{source_name}: Not available")
                        continue

                    # Pass structured params as context, including user context for placeholder resolution
                    enriched_params = dict(params)
                    if user_intelligence:
                        enriched_params["user_intelligence"] = user_intelligence
                    if user_purpose:
                        enriched_params["purpose"] = user_purpose
                    if user_memory:
                        enriched_params["user_memory"] = user_memory
                    # Pass designator model for MCP agentic fallback
                    if self.enricher.designator_model:
                        enriched_params["designator_model"] = (
                            self.enricher.designator_model
                        )
                    # Pass session key for session-scoped caching (e.g., Gmail)
                    if hasattr(self, "_session_key") and self._session_key:
                        enriched_params["session_key"] = self._session_key

                    # Track timing for stats
                    import time

                    start_time = time.time()
                    result = provider.fetch(params.get("query", query), enriched_params)
                    latency_ms = int((time.time() - start_time) * 1000)

                    # Determine endpoint name for tracking
                    endpoint_name = params.get("_endpoint", "default")
                    if hasattr(provider, "last_endpoint"):
                        endpoint_name = provider.last_endpoint or endpoint_name

                    if result.success and result.formatted:
                        context_parts.append(result.formatted)

                        # Update source status and record call stats
                        from db.live_data_sources import (
                            record_live_data_call,
                            update_live_data_source_status,
                        )

                        update_live_data_source_status(source.id, success=True)
                        record_live_data_call(
                            source.id,
                            endpoint_name,
                            success=True,
                            latency_ms=latency_ms,
                        )
                    elif result.error:
                        metadata["errors"].append(f"{source_name}: {result.error}")
                        logger.warning(
                            f"Live data error from {source_name}: {result.error}"
                        )
                        from db.live_data_sources import (
                            record_live_data_call,
                            update_live_data_source_status,
                        )

                        update_live_data_source_status(
                            source.id, success=False, error=result.error
                        )
                        record_live_data_call(
                            source.id,
                            endpoint_name,
                            success=False,
                            latency_ms=latency_ms,
                            error=result.error,
                        )

                except Exception as e:
                    metadata["errors"].append(f"{source_name}: {str(e)}")
                    logger.error(f"Error fetching from {source_name}: {e}")

            if context_parts:
                live_context = "=== Live Data ===\n" + "\n\n".join(context_parts)
                return live_context, metadata

        except Exception as e:
            logger.error(f"Live data retrieval failed: {e}")
            metadata["errors"].append(str(e))

        return "", metadata

    def _try_cached_params(self, query: str, sources: list) -> dict[str, dict]:
        """
        Try to resolve parameters from ChromaDB cache using semantic matching.

        Args:
            query: The user's query
            sources: List of enabled live data sources

        Returns:
            Dict mapping source name to cached parameters (may be partial)
        """
        cached_params = {}

        try:
            from live.param_cache import lookup_parameter

            for source in sources:
                name = getattr(source, "name", "")
                source_type = getattr(source, "source_type", "")

                # Try to find cached value based on source type
                if source_type == "builtin_weather":
                    cached_loc = lookup_parameter("weather", "location", query)
                    if cached_loc:
                        cached_params[name] = {
                            "location": cached_loc,
                            "type": "current",
                            "query": query,
                            "cache_hit": True,
                        }
                        logger.debug(f"Cache hit for weather location: {cached_loc}")

                elif source_type == "builtin_stocks":
                    cached_symbol = lookup_parameter("stocks", "symbol", query)
                    if cached_symbol:
                        cached_params[name] = {
                            "symbol": cached_symbol,
                            "query": query,
                            "cache_hit": True,
                        }
                        logger.debug(f"Cache hit for stock symbol: {cached_symbol}")

                elif source_type == "builtin_alpha_vantage":
                    cached_symbol = lookup_parameter("stocks-uk", "symbol", query)
                    if cached_symbol:
                        cached_params[name] = {
                            "symbol": cached_symbol,
                            "query": query,
                            "cache_hit": True,
                        }
                        logger.debug(f"Cache hit for UK stock symbol: {cached_symbol}")

                elif source_type == "builtin_transport":
                    cached_station = lookup_parameter("transport", "station", query)
                    if cached_station:
                        cached_params[name] = {
                            "station": cached_station,
                            "type": "departures",
                            "query": query,
                            "cache_hit": True,
                        }
                        logger.debug(
                            f"Cache hit for transport station: {cached_station}"
                        )

        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

        return cached_params

    def _extract_live_data_params(self, query: str, sources: list) -> dict[str, dict]:
        """
        Use the designator to extract structured parameters for live data sources.

        First checks ChromaDB cache for semantic matches (e.g., "British capital"
        matching cached "London"). Falls back to designator for cache misses.

        Args:
            query: The user's query
            sources: List of enabled live data sources

        Returns:
            Dict mapping source name to extracted parameters, e.g.:
            {
                "weather": {"location": "London", "type": "current"},
                "stocks": {"symbol": "AAPL"},
            }
        """
        # First try cache lookup for all sources
        cached_params = self._try_cached_params(query, sources)
        if cached_params:
            # Check if we got cache hits for all sources that might be relevant
            # If so, we can skip the designator call entirely
            logger.info(f"Using cached params for {len(cached_params)} source(s)")
            # Still need designator to determine relevance, but can hint with cached values

        if not self.enricher.designator_model:
            # Fall back to querying all sources with raw query (or cached params)
            result = {}
            for s in sources:
                name = getattr(s, "name", "")
                if name in cached_params:
                    result[name] = cached_params[name]
                else:
                    result[name] = {"query": query}
            return result

        # Build source descriptions for the prompt
        source_info = []
        for source in sources:
            name = getattr(source, "name", "")
            source_type = getattr(source, "source_type", "")
            data_type = getattr(source, "data_type", "")
            best_for = getattr(source, "best_for", "")
            description = getattr(source, "description", "")

            info = f"- {name} ({source_type})"
            if data_type:
                info += f" [data_type: {data_type}]"
            if best_for:
                info += f"\n  Best for: {best_for}"
            if description:
                info += f"\n  Description: {description}"

            # Add parameter hints based on source type
            if source_type == "builtin_weather":
                info += "\n  Parameters: location (city name), type (current|forecast)"
            elif source_type == "builtin_stocks":
                info += "\n  Parameters: symbol (US stock ticker like AAPL, MSFT)"
            elif source_type == "builtin_alpha_vantage":
                info += "\n  Parameters: symbol (UK stock like LGEN.LON or fund name), period (optional: 1W, 1M, 3M, 6M, 1Y)"
            elif source_type == "builtin_transport":
                info += "\n  Parameters: station (3-letter code or name), type (departures|arrivals)"
            else:
                # Extract parameter hints from query_params_json and endpoint_url for custom sources
                params_found = []
                query_params = getattr(source, "query_params_json", "") or ""
                endpoint_url = getattr(source, "endpoint_url", "") or ""
                # Find {{param}} placeholders
                import re

                for text in [query_params, endpoint_url]:
                    matches = re.findall(r"\{\{(\w+)\}\}", text)
                    params_found.extend(matches)
                if params_found:
                    unique_params = list(
                        dict.fromkeys(params_found)
                    )  # Preserve order, remove dupes
                    info += f"\n  Parameters: {', '.join(unique_params)}"

            source_info.append(info)

        sources_text = "\n".join(source_info)

        prompt = f"""Analyze the user's query and determine which live data sources are relevant.
For each relevant source, extract the specific parameters needed to query it.

AVAILABLE LIVE DATA SOURCES:
{sources_text}

USER QUERY:
{query}

Respond with a JSON object where keys are source names and values are parameter objects.
Only include sources that are RELEVANT to the query. If no sources are relevant, return {{}}.

Examples:
- "What's the weather in Paris?" -> {{"weather": {{"location": "Paris", "type": "current"}}}}
- "AAPL stock price" -> {{"stocks": {{"symbol": "AAPL"}}}}
- "Trains from Kings Cross" -> {{"transport": {{"station": "KGX", "type": "departures"}}}}
- "Tell me a joke" -> {{}}  (no live data needed)

Respond with ONLY the JSON object, no other text."""

        try:
            designator_resolved = self.registry._resolve_actual_model(
                self.enricher.designator_model
            )
            provider = designator_resolved.provider

            result = provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 300, "temperature": 0},
            )

            response_text = result.get("content", "").strip()

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())

                # Validate and filter to actual source names
                valid_names = {getattr(s, "name", "") for s in sources}
                filtered_params = {}
                for name, p in params.items():
                    if name in valid_names and isinstance(p, dict):
                        # Merge with cached params if available (cache takes priority)
                        if name in cached_params:
                            # Use cached values but mark source was deemed relevant
                            merged = cached_params[name].copy()
                            # Designator may have additional params, merge them
                            for k, v in p.items():
                                if k not in merged:
                                    merged[k] = v
                            filtered_params[name] = merged
                            logger.debug(f"Merged cached params for {name}")
                        else:
                            # Add raw query as fallback
                            p["query"] = query
                            filtered_params[name] = p
                            # Cache extracted parameters for semantic reuse
                            self._cache_extracted_params(name, p, sources)

                logger.info(f"Designator extracted live data params: {filtered_params}")
                return filtered_params

            logger.warning(f"Could not parse live data params: {response_text}")

        except Exception as e:
            logger.error(f"Live data param extraction failed: {e}")

        # Fall back to querying all sources
        return {getattr(s, "name", ""): {"query": query} for s in sources}

    def _cache_extracted_params(
        self, source_name: str, params: dict, sources: list
    ) -> None:
        """Cache extracted parameters in ChromaDB for semantic reuse."""
        try:
            from live.param_cache import cache_parameter

            # Find source type
            source = next(
                (s for s in sources if getattr(s, "name", "") == source_name), None
            )
            if not source:
                return

            source_type = getattr(source, "source_type", "")

            # Cache based on source type
            if source_type == "builtin_weather" and "location" in params:
                cache_parameter(
                    source_type="weather",
                    param_type="location",
                    query_text=params["location"],
                    resolved_value=params["location"],
                )
            elif source_type == "builtin_stocks" and "symbol" in params:
                cache_parameter(
                    source_type="stocks",
                    param_type="symbol",
                    query_text=params.get("query", params["symbol"]),
                    resolved_value=params["symbol"],
                )
            elif source_type == "builtin_alpha_vantage" and "symbol" in params:
                cache_parameter(
                    source_type="stocks-uk",
                    param_type="symbol",
                    query_text=params.get("query", params["symbol"]),
                    resolved_value=params["symbol"],
                )
            elif source_type == "builtin_transport" and "station" in params:
                cache_parameter(
                    source_type="transport",
                    param_type="station",
                    query_text=params.get("query", params["station"]),
                    resolved_value=params["station"],
                )

        except Exception as e:
            logger.debug(f"Failed to cache params: {e}")

    def _call_designator(self, query: str) -> tuple[Optional[str], Optional[dict]]:
        """Call the designator LLM to generate an optimized search query."""
        prompt = f"""Generate an optimized web search query for the user's question.

RULES:
- Extract key concepts
- Add context like "2024", "latest", "news" where relevant
- Keep it concise (3-8 words ideal)
- Output ONLY the search query, nothing else

USER QUESTION:
{query}

SEARCH QUERY:"""

        try:
            designator_resolved = self.registry._resolve_actual_model(
                self.enricher.designator_model
            )
            provider = designator_resolved.provider

            result = provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 200, "temperature": 0},
            )

            optimized = result.get("content", "").strip()
            usage = {
                "prompt_tokens": result.get("input_tokens", 0),
                "completion_tokens": result.get("output_tokens", 0),
            }

            # Validate the response looks like a search query
            if optimized and len(optimized) < 200 and not optimized.startswith("I "):
                logger.debug(f"Designator search query: {optimized}")
                return optimized, usage

            return None, usage

        except Exception as e:
            logger.error(f"Designator call failed: {e}")
            return None, None

    def _execute_search(
        self, query: str, max_results: int | None = None
    ) -> tuple[str, list[dict]]:
        """Execute web search and return formatted results plus URL data.

        Args:
            query: Search query
            max_results: Maximum results to return (defaults to enricher config)
        """
        from augmentation import get_configured_search_provider
        from augmentation.query_intent import extract_query_intent

        provider = get_configured_search_provider()
        if not provider:
            logger.warning("No search provider configured")
            return "", []

        # Use provided max_results or fall back to enricher config
        if max_results is None:
            max_results = self.enricher.max_search_results

        try:
            # Extract temporal and category intent from query
            intent = extract_query_intent(query)

            if intent.time_range or intent.category:
                logger.debug(
                    f"Query intent detected: time_range={intent.time_range}, "
                    f"category={intent.category}"
                )

            results = provider.search(
                query,
                max_results=max_results,
                time_range=intent.time_range,
                category=intent.category,
            )
            formatted = provider.format_results(results)

            urls_with_context = [
                {"url": r.url, "title": r.title or "", "snippet": r.snippet or ""}
                for r in results
                if r.url
            ]

            return formatted, urls_with_context

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "", []

    def _execute_scrape(
        self, urls: list[str], max_tokens: int, max_urls: int | None = None
    ) -> tuple[str, list[str]]:
        """Scrape URLs and return formatted content.

        Args:
            urls: List of URLs to scrape
            max_tokens: Token budget for scraped content
            max_urls: Maximum URLs to scrape (defaults to enricher config)
        """
        scraper_provider = self._get_scraper_provider()

        if scraper_provider == "jina":
            from augmentation.scraper import JinaScraper

            # Free tier - no API key
            scraper = JinaScraper(api_key=None)
        elif scraper_provider == "jina-api":
            from augmentation.scraper import JinaScraper

            # Use API key from environment
            scraper = JinaScraper()
        else:
            from augmentation import WebScraper

            scraper = WebScraper()

        # Use provided max_urls or fall back to enricher config
        if max_urls is None:
            max_urls = self.enricher.max_scrape_urls

        results = scraper.scrape_multiple(urls, max_urls=max_urls)
        scraped_urls = [r.url for r in results if r.success]

        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        formatted = scraper.format_results(results, max_chars=max_chars)

        return formatted, scraped_urls

    def _get_scraper_provider(self) -> str:
        """Get the globally configured scraper provider from Settings."""
        try:
            from db import Setting, get_db_context

            with get_db_context() as db:
                setting = (
                    db.query(Setting)
                    .filter(Setting.key == Setting.KEY_WEB_SCRAPER_PROVIDER)
                    .first()
                )
                if setting and setting.value:
                    return setting.value
        except Exception as e:
            logger.warning(f"Failed to get scraper provider setting: {e}")
        return "builtin"

    def _rerank_urls(self, query: str, urls_with_context: list[dict]) -> list[str]:
        """Rerank URLs by relevance before scraping."""
        try:
            from rag.reranker import get_global_rerank_provider, rerank_urls

            rerank_provider = get_global_rerank_provider()
            return rerank_urls(
                query=query,
                urls_with_context=urls_with_context,
                model_name=self.enricher.rerank_model,
                top_k=self.enricher.max_scrape_urls,
                provider_type=rerank_provider,
            )
        except Exception as e:
            logger.warning(f"URL reranking failed: {e}")
            return [
                u["url"] for u in urls_with_context[: self.enricher.max_scrape_urls]
            ]

    def _merge_context(self, context_parts: list[tuple[str, str]]) -> str:
        """
        Merge context from multiple sources.

        Args:
            context_parts: List of (source_type, context) tuples

        Returns:
            Merged context string
        """
        if len(context_parts) == 1:
            return context_parts[0][1]

        # Get priority setting (only relevant when both RAG and Web present)
        priority = getattr(self.enricher, "context_priority", "balanced")

        # Define sort order: live data first (most current), then by priority
        # Live data always comes first as it's real-time, history comes after
        def sort_key(item):
            source_type = item[0]
            if source_type == "live":
                return 0  # Current live data first
            elif source_type == "live_history":
                return 1  # Prior session live data second
            elif priority == "prefer_web":
                return 2 if source_type == "web" else 3
            else:  # balanced or prefer_rag
                return 2 if source_type == "rag" else 3

        context_parts = sorted(context_parts, key=sort_key)

        # Build merged context with source labels
        merged = []
        for source_type, context in context_parts:
            if source_type == "rag":
                merged.append(f"=== Document Context ===\n{context}")
            elif source_type == "web":
                merged.append(f"=== Web Context ===\n{context}")
            elif source_type == "live":
                # Live context already has its header from _retrieve_live_context
                merged.append(context)
            elif source_type == "live_history":
                # Prior live data from this session
                merged.append(f"=== Previous Live Data (this session) ===\n{context}")
            else:
                merged.append(context)

        return "\n\n".join(merged)

    def _inject_context(self, original_system: str | None, context: str) -> str:
        """Inject enrichment context into the system prompt."""
        if not context.strip():
            return original_system or ""

        current_datetime = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Build priority-specific instruction when both sources are used
        priority = getattr(self.enricher, "context_priority", "balanced")
        has_both = self.enricher.use_rag and self.enricher.use_web

        if has_both and priority == "prefer_rag":
            priority_hint = "Prioritize information from Document Context over Web Context when answering."
        elif has_both and priority == "prefer_web":
            priority_hint = "Prioritize information from Web Context over Document Context when answering."
        else:
            priority_hint = ""

        instruction = """IMPORTANT: The following information has been specifically retrieved to answer the user's question.
You should strongly prefer information from this context over your general knowledge.
If the context contains relevant details, reference them directly in your response."""

        if priority_hint:
            instruction = f"{instruction}\n{priority_hint}"

        context_block = f"""
<enriched_context>
Current date/time: {current_datetime}

{instruction}

{context.strip()}
</enriched_context>
"""

        if original_system:
            return original_system + "\n\n" + context_block
        else:
            return context_block.strip()

    def _get_memory_context(self) -> str | None:
        """Get the current memory content if memory is enabled."""
        if not getattr(self.enricher, "use_memory", False):
            return None

        memory = getattr(self.enricher, "memory", None)
        if not memory:
            return None

        return memory

    def _inject_memory(self, system: str | None, memory: str) -> str:
        """Inject memory into the system prompt."""
        memory_block = f"""
<user_memory>
The following is your persistent memory about this user's preferences and context.
Use this information to personalize your responses.

{memory.strip()}
</user_memory>
"""

        if system:
            return memory_block + "\n" + system
        else:
            return memory_block.strip()

    def update_memory_after_response(
        self,
        messages: list[dict],
        assistant_response: str,
    ) -> bool:
        """
        Check if memory should be updated after a response and update if needed.

        This is called after the target model generates a response. The designator
        decides whether significant new information was learned that should be
        persisted to memory.

        Args:
            messages: The conversation messages (including user query)
            assistant_response: The response from the target model

        Returns:
            True if memory was updated, False otherwise
        """

        if not getattr(self.enricher, "use_memory", False):
            return False

        if not self.enricher.designator_model:
            logger.warning("Memory enabled but no designator model configured")
            return False

        current_memory = getattr(self.enricher, "memory", None) or ""
        memory_is_empty = not current_memory.strip()
        max_tokens = getattr(self.enricher, "memory_max_tokens", 500) or 500

        # Estimate current token usage (rough: 4 chars per token)
        current_tokens = len(current_memory) // 4 if current_memory else 0
        at_capacity = current_tokens >= max_tokens * 0.9  # 90% threshold

        # Get the user's query
        query = self._get_query(messages)
        if not query:
            return False

        # Build the prompt for memory update decision
        # Guidance varies based on current memory state
        if memory_is_empty:
            capacity_guidance = f"""3. Since memory is currently EMPTY, be more permissive about what to save.
   Any useful information about the user is valuable when starting from nothing.
   Look for: name, role, projects they're working on, preferences, technical stack, goals.

TOKEN LIMIT: {max_tokens} tokens maximum. Current usage: 0 tokens."""
        elif at_capacity:
            capacity_guidance = f"""3. Memory is near capacity ({current_tokens}/{max_tokens} tokens).
   Only add new information if it's MORE IMPORTANT than existing information.
   If adding new info, you may need to REMOVE or CONDENSE less important existing info.
   Prioritize: core identity > active projects > preferences > historical context.

TOKEN LIMIT: {max_tokens} tokens maximum. You MUST stay within this limit."""
        else:
            capacity_guidance = f"""3. Only update memory for SIGNIFICANT new information that adds to what's already known.
   Don't repeat or rephrase existing information.

TOKEN LIMIT: {max_tokens} tokens maximum. Current usage: ~{current_tokens} tokens."""

        prompt = f"""You are a memory management assistant. Your task is to decide whether the USER'S MESSAGE contains explicit information about themselves that should be remembered.

CURRENT MEMORY:
{current_memory if current_memory else "(empty - no information stored yet)"}

USER'S MESSAGE:
{query[:1000]}

IMPORTANT: Only extract information the user EXPLICITLY STATED about themselves. Do NOT infer or assume anything.

Good examples of what to remember:
- "I'm a software engineer at Acme Corp" → Remember: role and company
- "I prefer Python over JavaScript" → Remember: language preference
- "I'm working on a mobile app project" → Remember: current project
- "My name is Ben" → Remember: name
- "I have 100 shares in Apple" → Remember: stock holdings
- "I own a Tesla Model 3" → Remember: possessions
- "My budget is £50,000" → Remember: financial info they shared

Do NOT remember:
- Topics they merely asked questions about without stating personal facts
- Pure inferences from what information they requested
- Anything from the assistant's response - only the user's own statements

IMPORTANT: If the user says "I have X" or "I own X" or "My X is Y", that IS an explicit statement about themselves and SHOULD be remembered.

{capacity_guidance}

If the user explicitly stated new facts about themselves, respond with:
UPDATE: <new complete memory content>

If the user didn't explicitly share information about themselves, respond with:
NO_UPDATE

The memory must be concise and stay within the token limit. Merge new information with relevant existing memory."""

        try:
            designator_resolved = self.registry._resolve_actual_model(
                self.enricher.designator_model
            )
            provider = designator_resolved.provider

            result = provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 800, "temperature": 0},
            )

            response_text = result.get("content", "").strip()

            if response_text.startswith("UPDATE:"):
                new_memory = response_text[7:].strip()
                if new_memory:
                    from db import update_smart_alias_memory

                    success = update_smart_alias_memory(self.enricher.id, new_memory)
                    if success:
                        logger.info(
                            f"Updated memory for smart alias '{self.enricher.name}'"
                        )
                        return True
                    else:
                        logger.error(
                            f"Failed to save memory for smart alias '{self.enricher.name}'"
                        )

            elif response_text.startswith("NO_UPDATE"):
                logger.info(
                    f"No memory update needed for smart alias '{self.enricher.name}'"
                )
            else:
                logger.warning(
                    f"Unexpected memory response format: {response_text[:100]}"
                )

            return False

        except Exception as e:
            logger.error(f"Memory update check failed: {e}")
            return False
