"""
Jina Search providers.

Jina Search (s.jina.ai) is a web search API that returns clean, LLM-friendly
results with optional content extraction.

Two variants:
- JinaFreeSearchProvider: Free tier, no API key required (rate limited)
- JinaApiSearchProvider: Paid tier with API key (higher rate limits)

See: https://jina.ai/search/
"""

import logging
import os
from typing import Optional

import httpx

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class JinaSearchProviderBase(SearchProvider):
    """
    Base class for Jina Search providers.

    Uses the s.jina.ai endpoint which returns search results optimized
    for LLM consumption.
    """

    JINA_SEARCH_URL = "https://s.jina.ai/"

    def __init__(
        self, url_override: Optional[str] = None, api_key: Optional[str] = None
    ):
        super().__init__(url_override)
        self._api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Search using Jina Search API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        try:
            headers = {
                "Accept": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            # Jina Search uses the query as part of the URL path
            # Format: https://s.jina.ai/{query}
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.JINA_SEARCH_URL}{query}",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            results = []

            # Jina returns results in a 'data' array
            items = data.get("data", [])
            for item in items[:max_results]:
                results.append(
                    SearchResult(
                        title=item.get("title", "Untitled"),
                        url=item.get("url", ""),
                        snippet=item.get("description", item.get("content", ""))[:500],
                    )
                )

            logger.info(
                f"Jina Search returned {len(results)} results for query: {query}"
            )
            return results

        except httpx.TimeoutException:
            logger.error(f"Jina Search request timed out for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Jina Search HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Jina Search failed: {e}")
            return []

    def format_results(self, results: list[SearchResult]) -> str:
        """
        Format search results for injection into LLM context.

        Args:
            results: List of SearchResult objects

        Returns:
            Formatted string suitable for context injection
        """
        if not results:
            return "No search results found."

        lines = ["## Web Search Results (via Jina)\n"]
        for i, result in enumerate(results, 1):
            lines.append(f"### {i}. {result.title}")
            lines.append(f"URL: {result.url}")
            lines.append(f"{result.snippet}\n")

        return "\n".join(lines)


class JinaFreeSearchProvider(JinaSearchProviderBase):
    """
    Jina Search - Free tier (no API key required).

    Works without authentication but has lower rate limits.
    Good for testing or low-volume usage.
    """

    name = "jina"
    requires_api_key = False

    def __init__(
        self, url_override: Optional[str] = None, api_key: Optional[str] = None
    ):
        # Free tier doesn't use API key
        super().__init__(url_override, api_key=None)

    def is_configured(self) -> bool:
        """Free tier is always available."""
        return True


class JinaApiSearchProvider(JinaSearchProviderBase):
    """
    Jina Search - API tier (requires JINA_API_KEY).

    Uses authenticated API for higher rate limits and priority access.
    """

    name = "jina-api"
    requires_api_key = True

    def __init__(
        self, url_override: Optional[str] = None, api_key: Optional[str] = None
    ):
        # API tier requires the key
        super().__init__(
            url_override, api_key=api_key or os.environ.get("JINA_API_KEY")
        )

    def is_configured(self) -> bool:
        """API tier requires JINA_API_KEY."""
        return bool(self._api_key)


# Keep backward compatibility - JinaSearchProvider defaults to API version
JinaSearchProvider = JinaApiSearchProvider
