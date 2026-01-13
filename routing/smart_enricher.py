"""
Smart Enricher Engine for unified context augmentation.

Combines RAG document retrieval and web search/scraping into a single enrichment pipeline.
Supports RAG-only, web-only, or hybrid (both) modes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from db.models import SmartAlias
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


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
    designator_usage: dict | None = None
    designator_model: str | None = None

    # Budget allocations (from smart source selection)
    store_budgets: dict[int, int] | None = None  # {store_id: tokens}
    store_id_to_name: dict[int, str] | None = None  # {store_id: name} for display
    web_budget: int | None = None
    total_budget: int | None = None  # Total context budget for percentage calc
    show_sources: bool = False  # Whether to append sources to response

    # Memory metadata
    memory_included: bool = False
    memory_update_pending: bool = False  # Flag to trigger memory update after response


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

    def enrich(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> EnrichmentResult:
        """
        Enrich a request with document and/or web context.

        Args:
            messages: List of message dicts
            system: Optional system prompt

        Returns:
            EnrichmentResult with augmented system/messages and metadata
        """
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
        )

        # Smart source selection - let designator choose which sources to use
        use_rag = self.enricher.use_rag
        use_web = self.enricher.use_web
        selected_stores = None  # None means use all stores
        store_budgets = None  # Token budgets per store (from smart selection)
        store_id_to_name = None  # Mapping for display
        web_budget = None  # Token budget for web search

        if getattr(self.enricher, "use_smart_source_selection", False):
            selection = self._select_sources(query)
            if selection:
                use_rag = selection.get("use_rag", use_rag)
                use_web = selection.get("use_web", use_web)
                selected_stores = selection.get("store_ids")  # List of store IDs to use
                store_budgets = selection.get(
                    "store_budgets"
                )  # Token budgets per store
                store_id_to_name = selection.get("store_id_to_name")  # ID -> name
                web_budget = selection.get("web_budget")  # Token budget for web
                logger.info(
                    f"Smart source selection: use_rag={use_rag}, use_web={use_web}, "
                    f"store_budgets={store_budgets or 'default'}, web_budget={web_budget or 'default'}"
                )

        # RAG retrieval (if enabled)
        if use_rag:
            rag_context, rag_metadata = self._retrieve_rag_context(
                query, selected_store_ids=selected_stores, store_budgets=store_budgets
            )
            if rag_context:
                context_parts.append(("rag", rag_context))
            # Copy RAG metadata
            result_metadata.chunks_retrieved = rag_metadata.get("chunks_retrieved", 0)
            result_metadata.sources = rag_metadata.get("sources", [])
            result_metadata.stores_queried = rag_metadata.get("stores_queried", [])
            result_metadata.embedding_usage = rag_metadata.get("embedding_usage")
            result_metadata.embedding_model = rag_metadata.get("embedding_model")
            result_metadata.embedding_provider = rag_metadata.get("embedding_provider")

        # Web search and scrape (if enabled)
        if use_web:
            web_context, web_metadata = self._retrieve_web_context(
                query, max_tokens=web_budget
            )
            if web_context:
                context_parts.append(("web", web_context))
            # Copy web metadata
            result_metadata.search_query = web_metadata.get("search_query")
            result_metadata.scraped_urls = web_metadata.get("scraped_urls", [])
            result_metadata.designator_usage = web_metadata.get("designator_usage")
            result_metadata.designator_model = web_metadata.get("designator_model")

        # Determine enrichment type based on what was actually used
        if context_parts:
            types_used = [t for t, _ in context_parts]
            if "rag" in types_used and "web" in types_used:
                result_metadata.enrichment_type = "hybrid"
            elif "rag" in types_used:
                result_metadata.enrichment_type = "rag"
            elif "web" in types_used:
                result_metadata.enrichment_type = "web"

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

        return result_metadata

    def _get_query(self, messages: list[dict], max_chars: int = 2000) -> str:
        """Extract the user's query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                elif isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    return " ".join(texts)[:max_chars]
        return ""

    def _select_sources(self, query: str) -> dict | None:
        """
        Use the designator model to select which sources to use for a query.

        Returns:
            Dict with 'use_rag', 'use_web', and 'store_ids' keys, or None if selection fails
        """
        if not self.enricher.designator_model:
            logger.warning(
                "Smart source selection enabled but no designator model configured"
            )
            return None

        # Build list of available sources with descriptions and intelligence
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
                # Get intelligence fields if available
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

        if not store_info and not self.enricher.use_web:
            logger.debug("No ready stores and web disabled - nothing to select from")
            return None

        # Build the prompt for the designator
        sources_text = ""
        if store_info:
            sources_text = "DOCUMENT STORES:\n"
            for s in store_info:
                sources_text += f"- ID: {s['id']}, Name: {s['name']}\n"
                sources_text += f"  Description: {s['description']}\n"
                # Add intelligence info if available
                if s.get("themes"):
                    sources_text += f"  Themes: {', '.join(s['themes'])}\n"
                if s.get("best_for"):
                    sources_text += f"  Best for: {s['best_for']}\n"
                if s.get("content_summary"):
                    sources_text += f"  Content: {s['content_summary']}\n"

        web_available = (
            "Web search is AVAILABLE."
            if self.enricher.use_web
            else "Web search is NOT available."
        )

        # Get total context budget
        max_context = getattr(self.enricher, "max_context_tokens", 4000)

        # Split budget: 50% for priority allocation, 50% baseline for all sources
        priority_budget = max_context // 2
        num_sources = len(store_info) + (1 if self.enricher.use_web else 0)
        baseline_per_source = (max_context - priority_budget) // max(num_sources, 1)

        prompt = f"""You are a source selection assistant. Allocate a PRIORITY context token budget across sources most relevant to the user's query.

PRIORITY BUDGET TO ALLOCATE: {priority_budget} tokens (50% of total)
Note: All sources will receive a baseline of {baseline_per_source} tokens each regardless of your allocation. Your allocation adds EXTRA budget to prioritized sources.

{sources_text}
{web_available}

USER QUERY:
{query}

Respond with a JSON object allocating the PRIORITY budget to the most relevant sources:
- "allocations": object mapping source to EXTRA token budget, e.g. {{"store_1": 1500, "web": 500}}
- Use store IDs as keys (e.g. "store_5" for store ID 5)
- Use "web" as the key for web search budget
- Allocations should sum to at most {priority_budget} tokens
- Only include sources that are particularly relevant - others will still get baseline tokens

Guidelines:
- Use the themes, best_for, and content fields to match stores to the query topic
- Allocate more tokens to sources most likely to have relevant information
- Use document stores for internal/archived content matching their descriptions
- Use web search for current events, recent news, or information unlikely to be in document stores
- Focus priority budget on 1-3 most relevant sources

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

            # Try to extract JSON from the response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                allocations = selection.get("allocations", selection)

                # Parse priority allocations from designator
                valid_store_ids = {s["id"] for s in store_info}
                priority_budgets = {}
                web_priority = 0

                for key, budget in allocations.items():
                    if not isinstance(budget, (int, float)) or budget <= 0:
                        continue
                    budget = int(budget)

                    if key == "web":
                        web_priority = budget if self.enricher.use_web else 0
                    elif key.startswith("store_"):
                        try:
                            store_id = int(key.replace("store_", ""))
                            if store_id in valid_store_ids:
                                priority_budgets[store_id] = budget
                        except ValueError:
                            pass
                    else:
                        # Try parsing as plain store ID
                        try:
                            store_id = int(key)
                            if store_id in valid_store_ids:
                                priority_budgets[store_id] = budget
                        except ValueError:
                            pass

                # Build final budgets: baseline + priority for ALL stores
                store_budgets = {}
                for store_id in valid_store_ids:
                    store_budgets[store_id] = (
                        baseline_per_source + priority_budgets.get(store_id, 0)
                    )

                # Web budget: baseline + priority (if web is enabled)
                web_budget = 0
                if self.enricher.use_web:
                    web_budget = baseline_per_source + web_priority

                # Always use RAG if we have stores (they all get baseline)
                use_rag = bool(store_budgets)
                use_web = web_budget > 0
                # Return ALL store IDs since they all get queried now
                store_ids = list(store_budgets.keys()) if store_budgets else None

                # Build store_id -> name mapping for display
                store_id_to_name = {s["id"]: s["name"] for s in store_info}

                logger.info(
                    f"Source selection: priority={priority_budgets}, baseline={baseline_per_source}/source, "
                    f"final budgets: stores={store_budgets}, web={web_budget}"
                )

                return {
                    "use_rag": use_rag,
                    "use_web": use_web,
                    "store_ids": store_ids,
                    "store_budgets": store_budgets if store_budgets else None,
                    "store_id_to_name": store_id_to_name,
                    "web_budget": web_budget if web_budget > 0 else None,
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
            metadata["stores_queried"] = result.stores_queried or []
            metadata["embedding_usage"] = result.embedding_usage
            metadata["embedding_model"] = result.embedding_model
            metadata["embedding_provider"] = result.embedding_provider

            return context, metadata

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return "", metadata

    def _retrieve_web_context(
        self, query: str, max_tokens: int | None = None
    ) -> tuple[str, dict]:
        """
        Retrieve context from web search and scraping.

        Args:
            query: The user's query
            max_tokens: If provided, override the default token budget for web context

        Returns:
            Tuple of (context string, metadata dict)
        """
        metadata = {
            "search_query": None,
            "scraped_urls": [],
            "designator_usage": None,
            "designator_model": None,
        }

        # Generate optimized search query using designator model (if configured)
        search_query = query
        if self.enricher.designator_model:
            optimized_query, designator_usage = self._call_designator(query)
            if optimized_query:
                search_query = optimized_query
            metadata["designator_usage"] = designator_usage
            metadata["designator_model"] = self.enricher.designator_model

        metadata["search_query"] = search_query

        # Execute web search
        search_results, urls_with_context = self._execute_search(search_query)
        if not search_results:
            return "", metadata

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

        context_parts = [search_results]

        # Rerank and scrape URLs
        if urls_with_context:
            urls = self._rerank_urls(search_query, urls_with_context)
            if urls:
                scrape_results, scraped_urls = self._execute_scrape(urls, max_tokens)
                if scrape_results:
                    context_parts.append(scrape_results)
                    metadata["scraped_urls"] = scraped_urls

        return "\n\n".join(context_parts), metadata

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

    def _execute_search(self, query: str) -> tuple[str, list[dict]]:
        """Execute web search and return formatted results plus URL data."""
        from augmentation import get_configured_search_provider
        from augmentation.query_intent import extract_query_intent

        provider = get_configured_search_provider()
        if not provider:
            logger.warning("No search provider configured")
            return "", []

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
                max_results=self.enricher.max_search_results,
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
        self, urls: list[str], max_tokens: int
    ) -> tuple[str, list[str]]:
        """Scrape URLs and return formatted content."""
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

        results = scraper.scrape_multiple(urls, max_urls=self.enricher.max_scrape_urls)
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

        # Order context based on priority (prioritized source first)
        if priority == "prefer_web":
            # Put web first
            context_parts = sorted(
                context_parts, key=lambda x: 0 if x[0] == "web" else 1
            )
        else:
            # Default: put RAG first (for balanced and prefer_rag)
            context_parts = sorted(
                context_parts, key=lambda x: 0 if x[0] == "rag" else 1
            )

        # Build merged context with source labels
        merged = []
        for source_type, context in context_parts:
            if source_type == "rag":
                merged.append(f"=== Document Context ===\n{context}")
            elif source_type == "web":
                merged.append(f"=== Web Context ===\n{context}")
            else:
                merged.append(context)

        return "\n\n".join(merged)

    def _inject_context(self, original_system: str | None, context: str) -> str:
        """Inject enrichment context into the system prompt."""
        if not context.strip():
            return original_system or ""

        current_date = datetime.utcnow().strftime("%Y-%m-%d")

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
Today's date: {current_date}

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

Do NOT remember:
- Topics they asked questions about (asking about X doesn't mean they're interested in X)
- Inferences from what information they requested
- Anything from the assistant's response - only the user's own statements

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
                logger.debug(
                    f"No memory update needed for smart alias '{self.enricher.name}'"
                )

            return False

        except Exception as e:
            logger.error(f"Memory update check failed: {e}")
            return False
