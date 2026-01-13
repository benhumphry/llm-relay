"""
Base class for search providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str  # Brief excerpt/description


class SearchProvider(ABC):
    """
    Abstract base class for search providers.

    Search providers fetch web search results that can be injected into
    LLM context for augmentation.
    """

    # Provider name (used for registration and selection)
    name: str = "base"

    # Whether this provider requires an API key
    requires_api_key: bool = False

    def __init__(self, url_override: Optional[str] = None):
        """
        Initialize the search provider.

        Args:
            url_override: Optional URL override for self-hosted providers
        """
        self.url_override = url_override

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        time_range: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for the given query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            time_range: Optional time filter (day, week, month, year)
            category: Optional search category (news, images, videos, etc.)

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Check if the provider is properly configured.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

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

        lines = ["## Web Search Results\n"]
        for i, result in enumerate(results, 1):
            lines.append(f"### {i}. {result.title}")
            lines.append(f"URL: {result.url}")
            lines.append(f"{result.snippet}\n")

        return "\n".join(lines)
