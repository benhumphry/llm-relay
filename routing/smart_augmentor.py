"""
Smart Augmentor Engine for context augmentation.

Uses a designator LLM to decide what augmentation to apply (search, scrape, both, or none),
then fetches external content and injects it into the system prompt.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from db.models import SmartAugmentor
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    """Result of an augmentation decision."""

    resolved: "ResolvedModel"
    augmentor_id: int  # For stats updates
    augmentor_name: str
    augmentor_tags: list[str]

    # Augmented content
    augmented_system: str | None = None  # Modified system prompt with injected context
    augmented_messages: list[dict] = field(default_factory=list)  # Usually unchanged

    # Augmentation metadata
    augmentation_type: str = "direct"  # "direct"|"search"|"scrape"|"search+scrape"
    search_query: str | None = None
    scraped_urls: list[str] = field(default_factory=list)

    # Designator usage for cost tracking
    designator_usage: dict | None = None
    designator_model: str | None = None


class SmartAugmentorEngine:
    """
    Engine for smart context augmentation using LLM-based designator.

    The engine:
    1. Calls a designator LLM to decide what augmentation to apply
    2. Executes search and/or scrape based on the decision
    3. Injects the augmented context into the system prompt
    4. Returns the augmented request for forwarding to the target model
    """

    def __init__(self, augmentor: "SmartAugmentor", registry: "ProviderRegistry"):
        """
        Initialize the augmentation engine.

        Args:
            augmentor: SmartAugmentor configuration
            registry: Provider registry for model resolution
        """
        self.augmentor = augmentor
        self.registry = registry

    def augment(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AugmentationResult:
        """
        Augment a request with external context.

        Args:
            messages: List of message dicts
            system: Optional system prompt

        Returns:
            AugmentationResult with augmented system/messages and metadata
        """
        from providers.registry import ResolvedModel

        # Resolve target model first
        try:
            target_resolved = self.registry._resolve_actual_model(
                self.augmentor.target_model
            )
            resolved = ResolvedModel(
                provider=target_resolved.provider,
                model_id=target_resolved.model_id,
                alias_name=self.augmentor.name,
                alias_tags=self.augmentor.tags,
            )
        except ValueError as e:
            logger.error(
                f"Augmentor '{self.augmentor.name}' target model not available: {e}"
            )
            raise

        # Get the user's query for context
        query_preview = self._get_query_preview(messages)

        # Call designator to decide augmentation
        decision, designator_usage = self._call_designator(query_preview)

        # Extract search query from designator response
        search_query = self._extract_search_query(decision, query_preview)
        logger.info(
            f"Augmentor '{self.augmentor.name}' search query: {search_query[:100]}..."
            if len(search_query) > 100
            else f"Augmentor '{self.augmentor.name}' search query: {search_query}"
        )

        # Always search and scrape
        augmented_context = ""
        scraped_urls = []

        search_results, urls_with_context = self._execute_search(search_query)
        if search_results:
            augmented_context += search_results + "\n\n"

            # Rerank URLs before scraping for better relevance
            if urls_with_context:
                urls = self._rerank_urls(search_query, urls_with_context)
                if urls:
                    scrape_results, scraped_urls = self._execute_scrape(urls)
                    if scrape_results:
                        augmented_context += scrape_results

        # Inject context into system prompt
        augmented_system = self._inject_context(system, augmented_context)

        return AugmentationResult(
            resolved=resolved,
            augmentor_id=self.augmentor.id,
            augmentor_name=self.augmentor.name,
            augmentor_tags=self.augmentor.tags,
            augmented_system=augmented_system,
            augmented_messages=messages,
            augmentation_type="search+scrape",
            search_query=search_query,
            scraped_urls=scraped_urls,
            designator_usage=designator_usage,
            designator_model=self.augmentor.designator_model,
        )

    def _get_query_preview(self, messages: list[dict], max_chars: int = 1000) -> str:
        """Extract a preview of the user's query for the designator."""
        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                elif isinstance(content, list):
                    # Extract text from content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    return " ".join(texts)[:max_chars]
        return ""

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

        return "builtin"  # Default

    def _call_designator(
        self, query_preview: str
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Call the designator LLM to decide augmentation.

        Returns:
            Tuple of (decision string, usage dict)
        """
        # Build the designator prompt
        prompt = f"""Generate an optimized web search query for the user's question.

RULES:
- Extract key concepts
- Add context like "2024", "latest", "news" where relevant
- Keep it concise (3-8 words ideal)
- Output ONLY the search query, nothing else

USER QUESTION:
{query_preview}

SEARCH QUERY:"""

        try:
            # Resolve and call the designator model
            designator_resolved = self.registry._resolve_actual_model(
                self.augmentor.designator_model
            )
            provider = designator_resolved.provider

            # Call the provider's chat_completion method (same as smart router)
            result = provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,  # System is built into the prompt
                options={
                    "max_tokens": 200,
                    "temperature": 0,
                },
            )

            decision = result.get("content", "").strip()
            usage = {
                "prompt_tokens": result.get("input_tokens", 0),
                "completion_tokens": result.get("output_tokens", 0),
            }

            logger.debug(f"Designator response: {decision}")
            return decision, usage

        except Exception as e:
            logger.error(f"Designator call failed: {e}")
            return None, None

    def _extract_search_query(self, decision: str | None, fallback_query: str) -> str:
        """
        Extract the search query from the designator's response.

        Args:
            decision: The designator's response (may be None or malformed)
            fallback_query: The original user query to use as fallback

        Returns:
            The search query to use
        """
        if not decision:
            return fallback_query

        decision = decision.strip()

        # Try to extract query from "search:query" format
        decision_lower = decision.lower()
        if decision_lower.startswith("search:"):
            query = decision[7:].strip()
            if query:
                return query

        # If designator returned something else, use it as-is if it looks like a query
        # (not too long, doesn't look like an explanation)
        if len(decision) < 200 and not decision.startswith("I "):
            return decision

        # Fall back to the original user query
        return fallback_query

    def _execute_search(self, query: str) -> tuple[str, list[dict]]:
        """
        Execute a web search and return formatted results plus raw data for reranking.

        Returns:
            Tuple of (formatted results string, list of {url, title, snippet} dicts)
        """
        from augmentation import get_configured_search_provider

        provider = get_configured_search_provider()

        if not provider:
            logger.warning("No search provider configured or available")
            return "", []

        try:
            results = provider.search(
                query, max_results=self.augmentor.max_search_results
            )
            formatted = provider.format_results(results)

            # Extract structured data for reranking
            urls_with_context = [
                {
                    "url": r.url,
                    "title": r.title or "",
                    "snippet": r.snippet or "",
                }
                for r in results
                if r.url
            ]

            return formatted, urls_with_context
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return "", []

    def _execute_scrape(self, urls: list[str]) -> tuple[str, list[str]]:
        """
        Scrape URLs and return formatted content.

        Uses globally configured scraper provider (builtin or jina) from Settings.

        Returns:
            Tuple of (formatted content, list of successfully scraped URLs)
        """
        # Get scraper provider from global settings
        scraper_provider = self._get_scraper_provider()

        if scraper_provider == "jina":
            from augmentation.scraper import JinaScraper

            scraper = JinaScraper()
        else:
            from augmentation import WebScraper

            scraper = WebScraper()

        results = scraper.scrape_multiple(urls, max_urls=self.augmentor.max_scrape_urls)

        # Calculate available tokens for scraping
        # Rough estimate: 4 chars per token
        max_chars = self.augmentor.max_context_tokens * 4

        scraped_urls = [r.url for r in results if r.success]
        formatted = scraper.format_results(results, max_chars=max_chars)

        return formatted, scraped_urls

    def _rerank_urls(self, query: str, urls_with_context: list[dict]) -> list[str]:
        """
        Rerank URLs by relevance to query before scraping.

        Args:
            query: Search query
            urls_with_context: List of {url, title, snippet} dicts

        Returns:
            List of URLs, reranked by relevance
        """
        # Use global rerank provider setting
        from rag.reranker import get_global_rerank_provider

        rerank_provider = get_global_rerank_provider()
        rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        try:
            from rag.reranker import rerank_urls

            return rerank_urls(
                query=query,
                urls_with_context=urls_with_context,
                model_name=rerank_model,
                top_k=self.augmentor.max_scrape_urls,
                provider_type=rerank_provider,
            )
        except Exception as e:
            logger.warning(f"URL reranking failed, using original order: {e}")
            # Fall back to original order
            return [
                u["url"] for u in urls_with_context[: self.augmentor.max_scrape_urls]
            ]

    def _extract_urls_from_search(self, search_results: str) -> list[str]:
        """Extract URLs from formatted search results (deprecated, use _rerank_urls)."""
        urls = []
        for line in search_results.split("\n"):
            if line.startswith("URL:"):
                url = line[4:].strip()
                if url:
                    urls.append(url)
        return urls

    def _parse_urls(self, context: str) -> list[str]:
        """Parse URLs from the context string."""
        # Split by comma and clean
        urls = []
        for part in context.split(","):
            url = part.strip()
            if url.startswith("http://") or url.startswith("https://"):
                urls.append(url)
        return urls

    def _inject_context(
        self, original_system: str | None, augmented_context: str
    ) -> str:
        """Inject augmented context into the system prompt."""
        from datetime import datetime

        # Always include current date for temporal context
        current_date = datetime.utcnow().strftime("%Y-%m-%d")

        if not augmented_context.strip():
            # Even without augmentation, add the date if we were called
            return original_system or ""

        context_block = f"""
<augmented_context>
Today's date: {current_date}

The following information was retrieved from the web to help answer the user's question.
Use this information to provide an accurate, up-to-date response.

{augmented_context.strip()}
</augmented_context>
"""

        if original_system:
            return original_system + "\n\n" + context_block
        else:
            return context_block.strip()
