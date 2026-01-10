"""
Routing module for LLM Relay.

Provides smart routing capabilities that use LLM-based designators
to intelligently route requests to the most appropriate model.

Also provides smart enrichers for unified RAG + web augmentation.
"""

from .smart_enricher import EnrichmentResult, SmartEnricherEngine
from .smart_router import RoutingResult, SmartRouterEngine, get_session_key

__all__ = [
    "SmartRouterEngine",
    "RoutingResult",
    "get_session_key",
    "SmartEnricherEngine",
    "EnrichmentResult",
]
