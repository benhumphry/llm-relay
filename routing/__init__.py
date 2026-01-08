"""
Routing module for LLM Relay.

Provides smart routing capabilities that use LLM-based designators
to intelligently route requests to the most appropriate model.

Also provides smart caching using ChromaDB for semantic similarity matching,
and smart augmentation for injecting web search/scrape results into context.
"""

from .smart_augmentor import AugmentationResult, SmartAugmentorEngine
from .smart_cache import CacheResult, SmartCacheEngine, build_cached_response
from .smart_router import RoutingResult, SmartRouterEngine, get_session_key

__all__ = [
    "SmartRouterEngine",
    "RoutingResult",
    "get_session_key",
    "SmartCacheEngine",
    "CacheResult",
    "build_cached_response",
    "SmartAugmentorEngine",
    "AugmentationResult",
]
