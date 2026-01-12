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

        # RAG retrieval (if enabled)
        if self.enricher.use_rag:
            rag_context, rag_metadata = self._retrieve_rag_context(query)
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
        if self.enricher.use_web:
            web_context, web_metadata = self._retrieve_web_context(query)
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

            logger.info(
                f"Enricher '{self.enricher.name}': Injected {result_metadata.enrichment_type} context "
                f"(RAG: {result_metadata.chunks_retrieved} chunks, Web: {len(result_metadata.scraped_urls)} URLs)"
            )
        else:
            result_metadata.enrichment_type = "none"
            result_metadata.context_injected = False
            logger.debug(f"Enricher '{self.enricher.name}': No context to inject")

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

    def _retrieve_rag_context(self, query: str) -> tuple[str, dict]:
        """
        Retrieve context from linked document stores.

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

            # Format context
            context = retriever.format_context(
                result,
                max_tokens=self.enricher.max_context_tokens // 2
                if self.enricher.use_web
                else self.enricher.max_context_tokens,
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

    def _retrieve_web_context(self, query: str) -> tuple[str, dict]:
        """
        Retrieve context from web search and scraping.

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

        # Allocate tokens for web context
        max_tokens = (
            self.enricher.max_context_tokens // 2
            if self.enricher.use_rag
            else self.enricher.max_context_tokens
        )

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

        provider = get_configured_search_provider()
        if not provider:
            logger.warning("No search provider configured")
            return "", []

        try:
            results = provider.search(
                query, max_results=self.enricher.max_search_results
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

        context_block = f"""
<enriched_context>
Today's date: {current_date}

The following information was retrieved to help answer the user's question.
Use this information to provide an accurate, well-informed response.

{context.strip()}
</enriched_context>
"""

        if original_system:
            return original_system + "\n\n" + context_block
        else:
            return context_block.strip()
