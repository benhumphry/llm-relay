"""
WebSearch Unified Source Plugin.

A RAG-only unified source for indexing web search results.

Features:
- Document side: Run search queries, scrape results, and index content
- Configurable time range and result limits
- Scheduled re-indexing for news/topic monitoring
- No live queries (use existing web enrichment for live search)
"""

import logging
from datetime import datetime, timezone
from typing import Iterator, Optional

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import PluginUnifiedSource, QueryAnalysis, QueryRouting

logger = logging.getLogger(__name__)


class WebSearchUnifiedSource(PluginUnifiedSource):
    """
    WebSearch source - RAG-only for indexed search results.

    Runs configured search queries, scrapes top results, and indexes
    the content for semantic search. Useful for monitoring topics/news.
    """

    source_type = "websearch"
    display_name = "Web Search"
    description = "Index web search results for a configured query"
    category = "web"
    icon = "🔍"
    content_category = ContentCategory.WEBSITES

    # Document store types this unified source handles
    handles_doc_source_types = ["websearch"]

    supports_rag = True
    supports_live = False  # Use existing web enrichment for live search
    supports_actions = False
    supports_incremental = False  # Full re-search each time

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "query": store.websearch_query or "",
            "max_results": store.websearch_max_results or 10,
            "pages_to_scrape": store.websearch_pages_to_scrape or 5,
            "time_range": store.websearch_time_range or "",
            "category": store.websearch_category or "",
            "index_schedule": store.index_schedule or "",
        }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="query",
                label="Search Query",
                field_type=FieldType.TEXT,
                required=True,
                help_text="Search query to run (e.g., 'AI news', 'python tutorials')",
            ),
            FieldDefinition(
                name="max_results",
                label="Max Search Results",
                field_type=FieldType.INTEGER,
                default=10,
                help_text="Maximum number of search results to fetch",
            ),
            FieldDefinition(
                name="pages_to_scrape",
                label="Pages to Scrape",
                field_type=FieldType.INTEGER,
                default=5,
                help_text="How many of the top results to actually scrape and index",
            ),
            FieldDefinition(
                name="time_range",
                label="Time Range",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Any time"},
                    {"value": "day", "label": "Past day"},
                    {"value": "week", "label": "Past week"},
                    {"value": "month", "label": "Past month"},
                    {"value": "year", "label": "Past year"},
                ],
                help_text="Filter results by time (if supported by search provider)",
            ),
            FieldDefinition(
                name="category",
                label="Category",
                field_type=FieldType.SELECT,
                required=False,
                default="general",
                options=[
                    {"value": "general", "label": "General"},
                    {"value": "news", "label": "News"},
                    {"value": "images", "label": "Images"},
                    {"value": "videos", "label": "Videos"},
                ],
                help_text="Search category (if supported by search provider)",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-run the search and update index",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """No live parameters - use web enrichment for live search."""
        return []

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.query = config.get("query", "")
        self.max_results = config.get("max_results", 10)
        self.pages_to_scrape = config.get("pages_to_scrape", 5)
        self.time_range = config.get("time_range", "") or None
        self.category = config.get("category", "general") or None
        self.index_schedule = config.get("index_schedule", "")

        # Cached results
        self._search_results: Optional[list] = None
        self._scraped_content: dict = {}

    def _get_search_provider(self):
        """Get the configured search provider."""
        try:
            from augmentation.search import get_configured_search_provider

            return get_configured_search_provider()
        except ImportError:
            return None

    def _get_scraper(self):
        """Get the web scraper."""
        try:
            from augmentation.scraper import WebScraper

            return WebScraper()
        except ImportError:
            return None

    def _run_search(self) -> list:
        """Run the search query."""
        provider = self._get_search_provider()
        if not provider:
            logger.error("No search provider configured")
            return []

        logger.info(
            f"Running web search: '{self.query}' (max_results={self.max_results}, "
            f"time_range={self.time_range}, category={self.category})"
        )

        try:
            results = provider.search(
                query=self.query,
                max_results=self.max_results,
                time_range=self.time_range,
                category=self.category,
            )
            logger.info(f"Found {len(results)} search results")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _scrape_results(self, results: list) -> dict:
        """Scrape content from search results."""
        if not results:
            return {}

        scraper = self._get_scraper()
        if not scraper:
            logger.error("Web scraper not available")
            return {}

        urls_to_scrape = [r.url for r in results[: self.pages_to_scrape]]
        logger.info(f"Scraping {len(urls_to_scrape)} pages from search results")

        scraped = {}
        try:
            results = scraper.scrape_multiple(
                urls_to_scrape, max_urls=self.pages_to_scrape
            )
            for result in results:
                if result.success and result.content:
                    scraped[result.url] = result
            logger.info(
                f"Successfully scraped {len(scraped)}/{len(urls_to_scrape)} pages"
            )
        except Exception as e:
            logger.error(f"Scraping failed: {e}")

        return scraped

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all scraped search result pages as documents."""
        logger.info(f"Listing web search results for: {self.query}")

        # Run fresh search
        self._search_results = self._run_search()
        self._scraped_content = self._scrape_results(self._search_results)

        # Use current timestamp for all results
        search_timestamp = datetime.now(timezone.utc).isoformat()

        for url, result in self._scraped_content.items():
            yield DocumentInfo(
                uri=f"websearch://{url}",
                title=result.title or url,
                mime_type="text/html",
                modified_at=search_timestamp,
                metadata={
                    "url": url,
                    "query": self.query,
                    "time_range": self.time_range,
                    "category": self.category,
                },
            )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read content from a scraped search result."""
        if not uri.startswith("websearch://"):
            logger.error(f"Invalid websearch URI: {uri}")
            return None

        url = uri.replace("websearch://", "")

        # Check if we have it cached
        if url in self._scraped_content:
            result = self._scraped_content[url]
            return DocumentContent(
                content=result.content,
                mime_type="text/plain",
                metadata={
                    "url": url,
                    "title": result.title or url,
                    "query": self.query,
                    "source_type": "webpage",
                },
            )

        # Try to scrape it fresh
        scraper = self._get_scraper()
        if not scraper:
            return None

        try:
            results = scraper.scrape_multiple([url], max_urls=1)
            if results and results[0].success:
                result = results[0]
                return DocumentContent(
                    content=result.content,
                    mime_type="text/plain",
                    metadata={
                        "url": url,
                        "title": result.title or url,
                        "query": self.query,
                        "source_type": "webpage",
                    },
                )
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")

        return None

    # =========================================================================
    # Live Side (disabled - use web enrichment)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """WebSearch doesn't support live queries - use web enrichment instead."""
        return LiveDataResult(
            success=False,
            error="WebSearch source does not support live queries. "
            "Enable web enrichment on your Smart Alias for live search.",
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """All queries go to RAG for WebSearch source."""
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="WebSearch source - RAG only (use web enrichment for live search)",
            max_rag_results=20,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if search provider is configured."""
        provider = self._get_search_provider()
        return provider is not None

    def test_connection(self) -> tuple[bool, str]:
        """Test search provider access."""
        results = []
        overall_success = True

        try:
            # Check search provider
            provider = self._get_search_provider()
            if not provider:
                return (
                    False,
                    "No search provider configured. Configure SearXNG, Perplexity, or Jina in Web Config.",
                )

            provider_name = type(provider).__name__
            results.append(f"Search provider: {provider_name}")
            results.append(f"Query: {self.query}")

            # Run a test search
            try:
                search_results = provider.search(
                    query=self.query,
                    max_results=3,
                    time_range=self.time_range,
                    category=self.category,
                )
                results.append(f"Search results: Found {len(search_results)} results")

                if search_results:
                    results.append(f"First result: {search_results[0].title}")
            except Exception as e:
                results.append(f"Search error: {e}")
                overall_success = False

            # Check scraper
            scraper = self._get_scraper()
            if scraper:
                results.append("Scraper: Available")
            else:
                results.append("Scraper: Not available")
                overall_success = False

        except Exception as e:
            return False, f"Test failed: {e}"

        return overall_success, "\n".join(results)
