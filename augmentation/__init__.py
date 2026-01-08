"""
Augmentation module for LLM Relay.

Provides web search and URL scraping capabilities for enhancing LLM requests
with up-to-date information.
"""

from .scraper import WebScraper
from .search import get_search_provider, list_search_providers

__all__ = [
    "WebScraper",
    "get_search_provider",
    "list_search_providers",
]
