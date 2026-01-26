"""
Base class for unified source plugins.

Unified sources combine document indexing (for RAG) and live querying (for real-time data)
into a single plugin. This simplifies configuration for users who just want to add "Gmail"
or "Calendar" without worrying about separate document stores and live sources.

The smart source decides whether to use RAG, live API, or both based on query characteristics.

Example: A user asks "What emails did I get from John?"
- If asking about recent emails -> Live API (freshest data)
- If asking about historical emails -> RAG index (faster, semantic search)
- If asking about "latest" -> Both, merge and dedupe

Key benefits:
1. Single configuration - One OAuth connection, one set of settings
2. Intelligent routing - System decides RAG vs Live based on query
3. Better results - Can combine indexed history with fresh API data
4. Simpler mental model - Users configure "Gmail" not "Gmail Docs + Gmail Live"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator, Optional

from plugin_base.common import FieldDefinition, ValidationResult, validate_config
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition


class QueryRouting(Enum):
    """How to route a query."""

    RAG_ONLY = "rag_only"  # Only search indexed documents
    LIVE_ONLY = "live_only"  # Only query live API
    RAG_THEN_LIVE = "rag_then_live"  # Search RAG first, supplement with live
    LIVE_THEN_RAG = "live_then_rag"  # Query live first, supplement with RAG
    BOTH_MERGE = "both_merge"  # Query both simultaneously, merge results
    TWO_PASS = "two_pass"  # RAG finds documents, live fetches full content


class MergeStrategy(Enum):
    """How to merge results from RAG and Live."""

    DEDUPE = "dedupe"  # Remove duplicates, prefer fresher data
    RAG_FIRST = "rag_first"  # RAG results first, then live
    LIVE_FIRST = "live_first"  # Live results first, then RAG
    INTERLEAVE = "interleave"  # Alternate between sources


@dataclass
class QueryAnalysis:
    """
    Result of analyzing a user query to determine routing.

    The unified source analyzes each query and decides:
    - Should we search the RAG index?
    - Should we query the live API?
    - How should we combine results?
    """

    routing: QueryRouting = QueryRouting.BOTH_MERGE
    merge_strategy: MergeStrategy = MergeStrategy.DEDUPE

    # Transformed query for RAG search (may differ from original)
    rag_query: str = ""

    # Parameters for live API call
    live_params: dict = field(default_factory=dict)

    # Explanation for routing decision (useful for debugging)
    reason: str = ""

    # Confidence in routing decision (0.0-1.0)
    confidence: float = 1.0

    # Hints for result processing
    max_rag_results: int = 10
    max_live_results: int = 10
    freshness_priority: bool = False  # Prefer fresher results in merge

    # Two-pass retrieval settings
    # When routing=TWO_PASS, RAG is used to find documents, then live fetches full content
    two_pass_fetch_full: bool = False  # If True, fetch full document content in pass 2

    def to_dict(self) -> dict:
        """Convert to dict for logging/debugging."""
        return {
            "routing": self.routing.value,
            "merge_strategy": self.merge_strategy.value,
            "rag_query": self.rag_query,
            "live_params": self.live_params,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class UnifiedResult:
    """
    Result from a unified source query.

    Contains results from both RAG and live sources, plus metadata
    about how they were combined.
    """

    success: bool
    formatted: str = ""  # Combined, formatted result for LLM context
    error: Optional[str] = None

    # Source breakdown
    rag_results: list[str] = field(default_factory=list)
    live_results: list[str] = field(default_factory=list)

    # Metadata
    routing_used: Optional[QueryRouting] = None
    merge_strategy_used: Optional[MergeStrategy] = None
    rag_count: int = 0
    live_count: int = 0
    dedupe_count: int = 0  # How many duplicates were removed

    # Timing
    rag_time_ms: int = 0
    live_time_ms: int = 0
    total_time_ms: int = 0

    # Cache hint
    cache_ttl: int = 300


class PluginUnifiedSource(ABC):
    """
    Base class for unified source plugins.

    Combines document indexing (for RAG) and live querying (for real-time data)
    into a single plugin. The smart source decides whether to use RAG, live API,
    or both based on query characteristics.

    Subclasses implement:
    - Document side: list_documents(), read_document() for indexing
    - Live side: fetch() for real-time queries
    - Router: analyze_query() to decide RAG vs Live vs Both

    The system calls:
    - list_documents()/read_document() during scheduled indexing
    - query() at request time, which uses analyze_query() to route

    Example:
        class SmartGmailSource(OAuthMixin, PluginUnifiedSource):
            source_type = "smart_gmail"
            display_name = "Gmail"

            def analyze_query(self, query, params):
                if "last hour" in query.lower():
                    return QueryAnalysis(routing=QueryRouting.LIVE_ONLY)
                elif "2023" in query:
                    return QueryAnalysis(routing=QueryRouting.RAG_ONLY)
                else:
                    return QueryAnalysis(routing=QueryRouting.BOTH_MERGE)
    """

    # --- Required class attributes (override in subclass) ---
    source_type: str  # Unique identifier (e.g., "gmail")
    display_name: str  # Shown in admin UI (e.g., "Gmail")
    description: str  # Help text for users
    category: str  # "google", "microsoft", "slack", "api", "local"

    # --- Optional class attributes ---
    icon: str = "📦"

    # Document store source types this unified source handles
    # e.g., ["mcp:gmail"] for Gmail, ["local"] for local filesystem
    # If empty, defaults to [source_type]
    handles_doc_source_types: list[str] = []

    # Live data source types this unified source handles
    # e.g., ["google_gmail_live"] for Gmail unified source
    # Used to map legacy live sources to unified sources
    # If empty, this unified source doesn't handle any legacy live sources
    handles_live_source_types: list[str] = []

    # --- Capability flags ---
    supports_rag: bool = True  # Can index documents for RAG
    supports_live: bool = True  # Can query live API
    supports_actions: bool = False  # Can perform actions (links to action plugin)
    supports_incremental: bool = True  # Can detect changed documents

    # --- Default settings ---
    default_cache_ttl: int = 300  # For live results
    default_index_days: int = 90  # How far back to index by default

    # Mark as abstract to prevent direct registration
    _abstract: bool = True

    # =========================================================================
    # Configuration
    # =========================================================================

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """
        Unified configuration for both document and live sides.

        This is called by the admin UI to render the configuration form.
        Include fields for:
        - Authentication (OAuth account, API key)
        - Document indexing settings (folders, labels, date range)
        - Live query settings (defaults, limits)
        - Scheduling (index schedule)

        Returns:
            List of FieldDefinition for admin UI
        """
        pass

    @classmethod
    @abstractmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """
        Parameters the designator can pass for live queries.

        These are dynamic per-request parameters that the designator
        determines based on the user's query.

        Returns:
            List of ParamDefinition for designator prompt
        """
        pass

    @classmethod
    def get_designator_hint(cls) -> str:
        """
        Generate hint for designator prompt.

        Describes what this source can do and what parameters it accepts.
        Override for custom formatting.

        Returns:
            String to include in designator system prompt
        """
        params = cls.get_live_params()
        param_parts = []
        for p in params:
            req = "" if p.required else "optional, "
            examples = f" e.g., {', '.join(p.examples)}" if p.examples else ""
            param_parts.append(f"{p.name} ({req}{p.description}{examples})")
        param_str = "; ".join(param_parts) if param_parts else "no parameters"

        capabilities = []
        if cls.supports_rag:
            capabilities.append("historical search via indexed documents")
        if cls.supports_live:
            capabilities.append("real-time API queries")
        if cls.supports_actions:
            capabilities.append("actions (create, update, send)")
        cap_str = ", ".join(capabilities)

        return f"Unified source with {cap_str}. Parameters: {param_str}"

    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """Validate plugin configuration."""
        return validate_config(cls.get_config_fields(), config)

    @classmethod
    def get_handled_doc_source_types(cls) -> list[str]:
        """
        Get list of document store source_type values this unified source handles.

        Returns handles_doc_source_types if set, otherwise [source_type].
        """
        if cls.handles_doc_source_types:
            return cls.handles_doc_source_types
        return [cls.source_type]

    @classmethod
    def get_handled_live_source_types(cls) -> list[str]:
        """
        Get list of live data source types this unified source handles.

        Returns handles_live_source_types (empty list if not set).
        Unlike doc sources, there's no default - only explicitly declared mappings.
        """
        return cls.handles_live_source_types or []

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """
        Build unified source config from a document store object.

        Override this in subclasses to map document store attributes
        to unified source config fields.

        Args:
            store: DocumentStore model instance

        Returns:
            Config dict suitable for __init__
        """
        # Default implementation - subclasses should override
        return {}

    @classmethod
    def get_config_for_store(cls, store) -> dict:
        """
        Get plugin config for a document store.

        This method first checks if the store has a linked PluginConfig.
        If so, it uses that config. Otherwise, it falls back to the
        legacy build_config_from_store() method.

        This enables gradual migration from hardcoded columns to PluginConfig.

        Args:
            store: DocumentStore model instance

        Returns:
            Config dict suitable for __init__
        """
        # Check if store has a linked PluginConfig
        if store.plugin_config_id:
            try:
                from db.plugin_configs import get_plugin_config

                plugin_config = get_plugin_config(store.plugin_config_id)
                if plugin_config and plugin_config.config:
                    return plugin_config.config
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to load PluginConfig {store.plugin_config_id}: {e}, "
                    "falling back to legacy config"
                )

        # Fall back to legacy column-based config
        return cls.build_config_from_store(store)

    @classmethod
    def get_account_info(cls, store) -> Optional[dict]:
        """
        Extract account info from a document store for action handlers.

        This method enables action handlers to discover available accounts
        from linked document stores. Each unified source knows how to extract
        the account information from its configuration.

        Args:
            store: DocumentStore model instance

        Returns:
            Dict with account info if this store provides an actionable account:
            - provider: str - OAuth provider or source type (e.g., "google", "imap")
            - email: str - Account email or identifier
            - name: str - Friendly display name for the account
            - store_id: int - The document store ID
            - oauth_account_id: Optional[int] - OAuth token ID if applicable
            - Extra source-specific fields (calendar_id, project_id, etc.)

            Returns None if this store doesn't provide an actionable account.

        Example (Gmail):
            return {
                "provider": "google",
                "email": "user@gmail.com",
                "name": store.display_name or store.name,
                "store_id": store.id,
                "oauth_account_id": store.google_account_id,
            }
        """
        # Default implementation - subclasses should override
        # to provide account info for action handlers
        return None

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize with validated config.

        Args:
            config: Dict matching get_config_fields() definitions
        """
        pass

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate documents for indexing.

        Yields DocumentInfo for each document. The indexer uses this to:
        1. Discover new documents
        2. Detect changed documents (via modified_at)
        3. Remove deleted documents

        For large sources, this should be a generator that fetches pages lazily.

        Yields:
            DocumentInfo for each document to index
        """
        pass

    @abstractmethod
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Fetch document content for indexing.

        Args:
            uri: The URI from DocumentInfo.uri

        Returns:
            DocumentContent with text/binary content, or None if not found
        """
        pass

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    @abstractmethod
    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live data based on parameters.

        Called when the query routing indicates live data is needed.

        Args:
            params: Dict from designator or from analyze_query().live_params

        Returns:
            LiveDataResult with formatted string for context injection
        """
        pass

    # =========================================================================
    # Smart Router
    # =========================================================================

    @abstractmethod
    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze a query to determine routing.

        This is the "smart" part of unified sources. Examine the query
        characteristics and decide whether to use RAG, Live, or both.

        Factors to consider:
        - Time expressions: "last hour" → live only, "2023" → RAG only
        - Freshness keywords: "latest", "recent", "new" → prefer live
        - Search complexity: complex semantic search → prefer RAG
        - Specificity: "email from John" → both, dedupe

        Args:
            query: The user's natural language query
            params: Parameters from the designator

        Returns:
            QueryAnalysis indicating how to route the query

        Example implementation:
            def analyze_query(self, query, params):
                query_lower = query.lower()

                # Very recent → live only
                if any(t in query_lower for t in ["last hour", "just now", "right now"]):
                    return QueryAnalysis(
                        routing=QueryRouting.LIVE_ONLY,
                        live_params=params,
                        reason="Very recent time reference"
                    )

                # Historical → RAG only
                if any(t in query_lower for t in ["last year", "2023", "2022"]):
                    return QueryAnalysis(
                        routing=QueryRouting.RAG_ONLY,
                        rag_query=query,
                        reason="Historical time reference"
                    )

                # Default → both
                return QueryAnalysis(
                    routing=QueryRouting.BOTH_MERGE,
                    rag_query=query,
                    live_params=params,
                    merge_strategy=MergeStrategy.LIVE_FIRST,
                    reason="General query - checking both sources"
                )
        """
        pass

    # =========================================================================
    # Query Execution
    # =========================================================================

    def query(
        self,
        query: str,
        params: dict,
        rag_search_fn: Optional[callable] = None,
    ) -> UnifiedResult:
        """
        Execute a smart query using the appropriate path(s).

        This is the main entry point called by the enricher at request time.
        It analyzes the query, routes to the appropriate source(s), and
        merges results.

        Args:
            query: The user's natural language query
            params: Parameters from the designator
            rag_search_fn: Optional function to search RAG index.
                           Signature: (query: str, limit: int, include_metadata: bool = False)
                           - If include_metadata=False: returns list[str] (document content)
                           - If include_metadata=True: returns list[dict] with keys:
                             content, source_uri, source_file, score
                           If not provided, RAG search is skipped.

        Returns:
            UnifiedResult with combined results and metadata
        """
        import time

        start_time = time.time()

        # Analyze query to determine routing
        analysis = self.analyze_query(query, params)

        import logging

        log = logging.getLogger(__name__)
        log.info(
            f"Unified source {self.source_type} routing: {analysis.routing.value}, "
            f"reason: {analysis.reason}"
        )

        rag_results = []
        live_results = []
        rag_time = 0
        live_time = 0

        # Execute based on routing
        if analysis.routing in [
            QueryRouting.RAG_ONLY,
            QueryRouting.RAG_THEN_LIVE,
            QueryRouting.BOTH_MERGE,
        ]:
            if rag_search_fn and self.supports_rag:
                rag_start = time.time()
                try:
                    rag_results = rag_search_fn(
                        analysis.rag_query or query, analysis.max_rag_results
                    )
                except Exception as e:
                    # Log but don't fail - we might still have live results
                    import logging

                    logging.getLogger(__name__).warning(f"RAG search failed: {e}")
                rag_time = int((time.time() - rag_start) * 1000)

        if analysis.routing in [
            QueryRouting.LIVE_ONLY,
            QueryRouting.LIVE_THEN_RAG,
            QueryRouting.BOTH_MERGE,
        ]:
            if self.supports_live:
                live_start = time.time()
                try:
                    live_result = self.fetch(analysis.live_params or params)
                    if live_result.success and live_result.formatted:
                        live_results = [live_result.formatted]
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(f"Live fetch failed: {e}")
                live_time = int((time.time() - live_start) * 1000)

        # Handle RAG_THEN_LIVE: only query live if RAG didn't find enough
        if analysis.routing == QueryRouting.RAG_THEN_LIVE and not rag_results:
            if self.supports_live and not live_results:
                live_start = time.time()
                try:
                    live_result = self.fetch(analysis.live_params or params)
                    if live_result.success and live_result.formatted:
                        live_results = [live_result.formatted]
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Live fetch fallback failed: {e}"
                    )
                live_time = int((time.time() - live_start) * 1000)

        # Handle LIVE_THEN_RAG: only query RAG if live didn't find enough
        if analysis.routing == QueryRouting.LIVE_THEN_RAG and not live_results:
            if rag_search_fn and self.supports_rag and not rag_results:
                rag_start = time.time()
                try:
                    rag_results = rag_search_fn(
                        analysis.rag_query or query, analysis.max_rag_results
                    )
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"RAG search fallback failed: {e}"
                    )
                rag_time = int((time.time() - rag_start) * 1000)

        # Handle TWO_PASS: RAG finds documents, live fetches full content
        if analysis.routing == QueryRouting.TWO_PASS:
            if rag_search_fn and self.supports_rag and self.supports_live:
                import logging

                log = logging.getLogger(__name__)

                # Pass 1: RAG semantic search with metadata
                rag_start = time.time()
                try:
                    # Call RAG with metadata to get source URIs
                    rag_with_meta = rag_search_fn(
                        analysis.rag_query or query,
                        analysis.max_rag_results,
                        include_metadata=True,  # Get source_uri for document resolution
                    )
                    rag_time = int((time.time() - rag_start) * 1000)

                    if rag_with_meta:
                        log.debug(
                            f"Two-pass: RAG found {len(rag_with_meta)} chunks "
                            f"from documents"
                        )

                        # Extract unique document URIs from RAG results
                        doc_uris = []
                        seen = set()
                        for result in rag_with_meta:
                            if isinstance(result, dict):
                                uri = result.get("source_uri", "")
                                if uri and uri not in seen:
                                    doc_uris.append(uri)
                                    seen.add(uri)
                                # Keep chunk content as context
                                if result.get("content"):
                                    rag_results.append(result["content"])
                            else:
                                # Fallback if rag_search_fn doesn't support metadata
                                rag_results.append(result)

                        # Pass 2: Fetch full documents for top matches
                        if doc_uris and analysis.two_pass_fetch_full:
                            live_start = time.time()
                            fetched_docs = []

                            for uri in doc_uris[: analysis.max_live_results]:
                                try:
                                    # Call fetch with the document URI/path
                                    fetch_params = {
                                        "action": "lookup",
                                        "filename": uri,
                                        "_two_pass_uri": uri,  # Direct URI for resolution
                                    }
                                    live_result = self.fetch(fetch_params)
                                    if live_result.success and live_result.formatted:
                                        fetched_docs.append(live_result.formatted)
                                        log.debug(f"Two-pass: Fetched full doc: {uri}")
                                except Exception as e:
                                    log.debug(f"Two-pass: Failed to fetch {uri}: {e}")

                            live_time = int((time.time() - live_start) * 1000)
                            live_results = fetched_docs

                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Two-pass RAG search failed: {e}"
                    )

        # Merge results
        merged, dedupe_count = self._merge_results(
            rag_results,
            live_results,
            analysis.merge_strategy,
            analysis.freshness_priority,
        )

        total_time = int((time.time() - start_time) * 1000)

        if not merged:
            return UnifiedResult(
                success=False,
                error="No results from RAG or Live sources",
                routing_used=analysis.routing,
                merge_strategy_used=analysis.merge_strategy,
                total_time_ms=total_time,
            )

        return UnifiedResult(
            success=True,
            formatted=merged,
            rag_results=rag_results,
            live_results=live_results,
            routing_used=analysis.routing,
            merge_strategy_used=analysis.merge_strategy,
            rag_count=len(rag_results),
            live_count=len(live_results),
            dedupe_count=dedupe_count,
            rag_time_ms=rag_time,
            live_time_ms=live_time,
            total_time_ms=total_time,
            cache_ttl=self.default_cache_ttl if live_results else 3600,
        )

    def _merge_results(
        self,
        rag_results: list[str],
        live_results: list[str],
        strategy: MergeStrategy,
        freshness_priority: bool = False,
    ) -> tuple[str, int]:
        """
        Merge results from RAG and Live sources.

        Args:
            rag_results: Results from RAG search
            live_results: Results from live API
            strategy: How to merge
            freshness_priority: Prefer fresher results in dedupe

        Returns:
            Tuple of (merged_text, dedupe_count)
        """
        if not rag_results and not live_results:
            return "", 0

        if not rag_results:
            return "\n\n".join(live_results), 0

        if not live_results:
            return "\n\n".join(rag_results), 0

        dedupe_count = 0

        if strategy == MergeStrategy.RAG_FIRST:
            merged = rag_results + live_results

        elif strategy == MergeStrategy.LIVE_FIRST:
            merged = live_results + rag_results

        elif strategy == MergeStrategy.INTERLEAVE:
            merged = []
            max_len = max(len(rag_results), len(live_results))
            for i in range(max_len):
                if i < len(live_results):
                    merged.append(live_results[i])
                if i < len(rag_results):
                    merged.append(rag_results[i])

        else:  # DEDUPE (default)
            # Simple deduplication based on content similarity
            # For more sophisticated deduplication, subclasses can override
            merged = []
            seen_content = set()

            # Process live first if freshness_priority, else RAG first
            if freshness_priority:
                all_results = live_results + rag_results
            else:
                all_results = rag_results + live_results

            for result in all_results:
                # Create a simple hash of the content for deduplication
                # Strip whitespace and normalize for comparison
                normalized = " ".join(result.lower().split())[:500]
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    merged.append(result)
                else:
                    dedupe_count += 1

        return "\n\n".join(merged), dedupe_count

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if source is configured and available."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """
        Test the connection for both document and live sides.

        Returns:
            Tuple of (success, message)
        """
        results = []
        overall_success = True

        # Test document side
        if self.supports_rag:
            try:
                docs = list(self.list_documents())
                results.append(f"Documents: Found {len(docs)} documents")
            except Exception as e:
                results.append(f"Documents: Error - {e}")
                overall_success = False

        # Test live side
        if self.supports_live:
            try:
                # Try a minimal fetch
                live_result = self.fetch({})
                if live_result.success:
                    preview = (
                        live_result.formatted[:100] + "..."
                        if len(live_result.formatted) > 100
                        else live_result.formatted
                    )
                    results.append(f"Live: OK - {preview}")
                else:
                    results.append(f"Live: Error - {live_result.error}")
                    overall_success = False
            except Exception as e:
                results.append(f"Live: Error - {e}")
                overall_success = False

        return overall_success, "\n".join(results)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_indexing_stats(self) -> dict:
        """
        Get statistics about indexed documents.

        Override to provide source-specific stats.

        Returns:
            Dict with stats like document_count, last_indexed, etc.
        """
        return {}

    def get_live_stats(self) -> dict:
        """
        Get statistics about live queries.

        Override to provide source-specific stats.

        Returns:
            Dict with stats like query_count, avg_response_time, etc.
        """
        return {}
