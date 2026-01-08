"""
Search provider module for LLM Relay augmentation.

Provides an extensible search provider system for fetching web search results.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SearchProvider

logger = logging.getLogger(__name__)

# Registry of available search providers
_providers: dict[str, type["SearchProvider"]] = {}


def register_provider(provider_class: type["SearchProvider"]) -> None:
    """Register a search provider class."""
    _providers[provider_class.name] = provider_class


def get_search_provider(
    name: str, url_override: str | None = None
) -> "SearchProvider | None":
    """
    Get a search provider instance by name.

    Args:
        name: Provider name (e.g., "searxng", "perplexity")
        url_override: Optional URL override for self-hosted providers

    Returns:
        SearchProvider instance or None if not found/not configured
    """
    provider_class = _providers.get(name.lower())
    if not provider_class:
        logger.warning(f"Unknown search provider: {name}")
        return None

    provider = provider_class(url_override=url_override)
    if not provider.is_configured():
        logger.warning(f"Search provider '{name}' is not configured")
        return None

    return provider


def list_search_providers() -> list[dict]:
    """
    List all available search providers with their configuration status.

    Returns:
        List of dicts with provider info: {name, configured, requires_api_key}
    """
    result = []
    for name, provider_class in _providers.items():
        provider = provider_class()
        result.append(
            {
                "name": name,
                "configured": provider.is_configured(),
                "requires_api_key": provider_class.requires_api_key,
            }
        )
    return result


# Import and register providers
from .searxng import SearXNGProvider

register_provider(SearXNGProvider)

# Perplexity provider (optional - uses existing LLM provider)
try:
    from .perplexity import PerplexitySearchProvider

    register_provider(PerplexitySearchProvider)
except ImportError:
    pass

__all__ = [
    "get_search_provider",
    "list_search_providers",
    "register_provider",
]
