"""
SearXNG search provider.

SearXNG is a free, self-hosted metasearch engine that aggregates results
from multiple search engines while respecting privacy.
"""

import logging
import os
from typing import Optional

import httpx

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class SearXNGProvider(SearchProvider):
    """
    Search provider using SearXNG.

    Requires SEARXNG_URL environment variable to be set, or url_override
    to be provided.
    """

    name = "searxng"
    requires_api_key = False

    def __init__(self, url_override: Optional[str] = None):
        super().__init__(url_override)
        self._base_url = url_override or os.environ.get("SEARXNG_URL")

    def is_configured(self) -> bool:
        """Check if SearXNG URL is configured."""
        return bool(self._base_url)

    def search(
        self,
        query: str,
        max_results: int = 5,
        time_range: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search using SearXNG.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            time_range: Optional time filter (day, week, month, year)
            category: Optional search category (news, images, videos, etc.)

        Returns:
            List of SearchResult objects
        """
        if not self._base_url:
            logger.error("SearXNG URL not configured")
            return []

        # Ensure URL doesn't have trailing slash
        base_url = self._base_url.rstrip("/")

        try:
            # Build search parameters
            params = {
                "q": query,
                "format": "json",
                "categories": category or "general",
            }

            # Add time_range if specified (SearXNG accepts: day, week, month, year)
            if time_range in ("day", "week", "month", "year"):
                params["time_range"] = time_range
                logger.debug(f"SearXNG search with time_range={time_range}")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SearchResult(
                        title=item.get("title", "Untitled"),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                    )
                )

            logger.info(f"SearXNG returned {len(results)} results for query: {query}")
            return results

        except httpx.TimeoutException:
            logger.error(f"SearXNG request timed out for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"SearXNG HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []
