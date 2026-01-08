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

        if decision is None or decision.startswith("direct"):
            # No augmentation needed
            logger.info(
                f"Augmentor '{self.augmentor.name}' decision: direct (no augmentation)"
            )
            return AugmentationResult(
                resolved=resolved,
                augmentor_id=self.augmentor.id,
                augmentor_name=self.augmentor.name,
                augmentor_tags=self.augmentor.tags,
                augmented_system=system,
                augmented_messages=messages,
                augmentation_type="direct",
                designator_usage=designator_usage,
                designator_model=self.augmentor.designator_model,
            )

        # Parse the decision
        augmentation_type, context = self._parse_decision(decision)
        logger.info(f"Augmentor '{self.augmentor.name}' decision: {augmentation_type}")

        # Execute augmentation
        augmented_context = ""
        search_query = None
        scraped_urls = []

        if augmentation_type in ("search", "search+scrape"):
            search_query = context or query_preview
            search_results = self._execute_search(search_query)
            if search_results:
                augmented_context += search_results + "\n\n"

                # If search+scrape, scrape top results
                if augmentation_type == "search+scrape":
                    urls = self._extract_urls_from_search(search_results)
                    if urls:
                        scrape_results, scraped_urls = self._execute_scrape(urls)
                        if scrape_results:
                            augmented_context += scrape_results

        elif augmentation_type == "scrape":
            urls = self._parse_urls(context)
            if urls:
                scrape_results, scraped_urls = self._execute_scrape(urls)
                if scrape_results:
                    augmented_context = scrape_results

        # Inject context into system prompt
        augmented_system = self._inject_context(system, augmented_context)

        return AugmentationResult(
            resolved=resolved,
            augmentor_id=self.augmentor.id,
            augmentor_name=self.augmentor.name,
            augmentor_tags=self.augmentor.tags,
            augmented_system=augmented_system,
            augmented_messages=messages,
            augmentation_type=augmentation_type,
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

    def _call_designator(
        self, query_preview: str
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Call the designator LLM to decide augmentation.

        Returns:
            Tuple of (decision string, usage dict)
        """
        # Build the designator prompt
        purpose_context = ""
        if self.augmentor.purpose:
            purpose_context = f"\nPURPOSE: {self.augmentor.purpose}\n"

        prompt = f"""You are an augmentation assistant. Analyze the user's query and decide what external information would help provide a better, more current answer.
{purpose_context}
OPTIONS:
- direct - No external info needed (use for simple questions, creative tasks, coding, etc.)
- search:query terms - Search the web for current information (use for news, current events, recent data)
- scrape:url1,url2 - Fetch specific URLs mentioned by the user
- search+scrape:query - Search then fetch top results for comprehensive research

USER QUERY:
{query_preview}

Respond with ONLY your decision (e.g., "direct" or "search:UK foreign policy 2024"). Do not explain."""

        try:
            # Resolve and call the designator model
            designator_resolved = self.registry._resolve_actual_model(
                self.augmentor.designator_model
            )
            provider = designator_resolved.provider

            # Call the provider's chat method
            response = provider.chat(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,  # Deterministic
                max_tokens=100,  # Short response expected
            )

            decision = response.get("message", {}).get("content", "").strip()
            usage = response.get("usage", {})

            logger.debug(f"Designator response: {decision}")
            return decision, usage

        except Exception as e:
            logger.error(f"Designator call failed: {e}")
            return None, None

    def _parse_decision(self, decision: str) -> tuple[str, str]:
        """
        Parse the designator's decision.

        Returns:
            Tuple of (augmentation_type, context/query)
        """
        decision = decision.lower().strip()

        if decision.startswith("search+scrape:"):
            return "search+scrape", decision[14:].strip()
        elif decision.startswith("search:"):
            return "search", decision[7:].strip()
        elif decision.startswith("scrape:"):
            return "scrape", decision[7:].strip()
        else:
            return "direct", ""

    def _execute_search(self, query: str) -> str:
        """Execute a web search and return formatted results."""
        from augmentation import get_search_provider

        provider = get_search_provider(
            self.augmentor.search_provider,
            url_override=self.augmentor.search_provider_url,
        )

        if not provider:
            logger.warning(
                f"Search provider '{self.augmentor.search_provider}' not available"
            )
            return ""

        try:
            results = provider.search(
                query, max_results=self.augmentor.max_search_results
            )
            return provider.format_results(results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ""

    def _execute_scrape(self, urls: list[str]) -> tuple[str, list[str]]:
        """
        Scrape URLs and return formatted content.

        Returns:
            Tuple of (formatted content, list of successfully scraped URLs)
        """
        from augmentation import WebScraper

        scraper = WebScraper()
        results = scraper.scrape_multiple(urls, max_urls=self.augmentor.max_scrape_urls)

        # Calculate available tokens for scraping
        # Rough estimate: 4 chars per token
        max_chars = self.augmentor.max_context_tokens * 4

        scraped_urls = [r.url for r in results if r.success]
        formatted = scraper.format_results(results, max_chars=max_chars)

        return formatted, scraped_urls

    def _extract_urls_from_search(self, search_results: str) -> list[str]:
        """Extract URLs from formatted search results."""
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
        if not augmented_context.strip():
            return original_system or ""

        context_block = f"""
<augmented_context>
The following information was retrieved from the web to help answer the user's question.
Use this information to provide an accurate, up-to-date response.

{augmented_context.strip()}
</augmented_context>
"""

        if original_system:
            return original_system + "\n\n" + context_block
        else:
            return context_block.strip()
