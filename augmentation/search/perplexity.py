"""
Perplexity search provider.

Uses the Perplexity API (via the existing LLM provider) to get search results.
Perplexity models have built-in web search capabilities.
"""

import logging
import os
from typing import Optional

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class PerplexitySearchProvider(SearchProvider):
    """
    Search provider using Perplexity API.

    Perplexity's models have built-in web search, so we use them directly
    to get search-augmented responses. This is different from SearXNG
    in that it returns synthesized information rather than raw search results.

    Requires PERPLEXITY_API_KEY environment variable.
    """

    name = "perplexity"
    requires_api_key = True

    def __init__(self, url_override: Optional[str] = None):
        super().__init__(url_override)
        self._api_key = os.environ.get("PERPLEXITY_API_KEY")

    def is_configured(self) -> bool:
        """Check if Perplexity API key is configured."""
        return bool(self._api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Search using Perplexity.

        Note: Perplexity doesn't return traditional search results.
        Instead, it returns a synthesized response with citations.
        We extract the citations as "search results".

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects (extracted from citations)
        """
        if not self._api_key:
            logger.error("Perplexity API key not configured")
            return []

        try:
            import httpx

            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Search for: {query}. Provide a brief summary with sources.",
                            }
                        ],
                        "max_tokens": 1024,
                    },
                )
                response.raise_for_status()
                data = response.json()

            # Extract citations from response
            results = []
            citations = data.get("citations", [])

            for i, url in enumerate(citations[:max_results]):
                results.append(
                    SearchResult(
                        title=f"Source {i + 1}",
                        url=url,
                        snippet="",  # Perplexity doesn't provide snippets per-source
                    )
                )

            # Also include the synthesized content as context
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")
                if content and not results:
                    # If no citations, create a single result with the content
                    results.append(
                        SearchResult(
                            title="Perplexity Summary",
                            url="",
                            snippet=content[:500],
                        )
                    )

            logger.info(
                f"Perplexity returned {len(results)} results for query: {query}"
            )
            return results

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return []

    def format_results(self, results: list[SearchResult]) -> str:
        """
        Format Perplexity results for context injection.

        Perplexity results are different - they may include a synthesized summary.
        """
        if not results:
            return "No search results found."

        lines = ["## Web Search Results (via Perplexity)\n"]
        for i, result in enumerate(results, 1):
            if result.title == "Perplexity Summary":
                lines.append(f"### Summary\n{result.snippet}\n")
            else:
                lines.append(f"### {i}. {result.title}")
                if result.url:
                    lines.append(f"URL: {result.url}")
                if result.snippet:
                    lines.append(f"{result.snippet}\n")

        return "\n".join(lines)
