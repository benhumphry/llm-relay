"""
Routing module for LLM Relay.

Provides smart routing capabilities that use LLM-based designators
to intelligently route requests to the most appropriate model.

Also provides smart enrichers for unified RAG + web augmentation,
and unified smart aliases that combine routing, enrichment, and caching.
"""

from .smart_alias import SmartAliasEngine
from .smart_enricher import EnrichmentResult, SmartEnricherEngine
from .smart_router import RoutingResult, SmartRouterEngine, get_session_key

__all__ = [
    # Legacy - Smart Router
    "SmartRouterEngine",
    "RoutingResult",
    # Legacy - Smart Enricher
    "SmartEnricherEngine",
    "EnrichmentResult",
    # Unified - Smart Alias (uses EnrichmentResult from SmartEnricher)
    "SmartAliasEngine",
    "get_session_key",
]
