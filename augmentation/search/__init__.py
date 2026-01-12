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


def get_default_search_provider() -> "SearchProvider | None":
    """
    Get the first configured search provider.

    Preference order: searxng, then any other configured provider.

    Returns:
        SearchProvider instance or None if no provider is configured
    """
    # Prefer SearXNG as it's self-hosted and doesn't require API keys
    if "searxng" in _providers:
        provider = _providers["searxng"]()
        if provider.is_configured():
            return provider

    # Fall back to any configured provider
    for name, provider_class in _providers.items():
        provider = provider_class()
        if provider.is_configured():
            return provider

    return None


def get_configured_search_provider() -> "SearchProvider | None":
    """
    Get the search provider configured in global settings.

    Falls back to get_default_search_provider() if no provider is configured in settings.

    Returns:
        SearchProvider instance or None if no provider is configured/available
    """
    from db import Setting, get_db_context

    provider_name = None
    url_override = None

    with get_db_context() as db:
        keys = [Setting.KEY_WEB_SEARCH_PROVIDER, Setting.KEY_WEB_SEARCH_URL]
        settings = db.query(Setting).filter(Setting.key.in_(keys)).all()
        settings_dict = {s.key: s.value for s in settings}
        provider_name = settings_dict.get(Setting.KEY_WEB_SEARCH_PROVIDER)
        url_override = settings_dict.get(Setting.KEY_WEB_SEARCH_URL)

    if provider_name:
        provider = get_search_provider(provider_name, url_override=url_override or None)
        if provider:
            return provider
        logger.warning(
            f"Configured search provider '{provider_name}' not available, falling back to default"
        )

    return get_default_search_provider()


# Import and register providers
from .searxng import SearXNGProvider

register_provider(SearXNGProvider)

# Perplexity provider (optional - uses existing LLM provider)
try:
    from .perplexity import PerplexitySearchProvider

    register_provider(PerplexitySearchProvider)
except ImportError:
    pass

# Jina providers - free tier (always available) and API tier (requires key)
try:
    from .jina import JinaApiSearchProvider, JinaFreeSearchProvider

    register_provider(JinaFreeSearchProvider)
    register_provider(JinaApiSearchProvider)
except ImportError:
    pass

__all__ = [
    "get_search_provider",
    "get_default_search_provider",
    "get_configured_search_provider",
    "list_search_providers",
    "register_provider",
]
