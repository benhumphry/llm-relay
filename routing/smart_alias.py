"""
Smart Alias Engine - unified routing, enrichment, and caching.

This is a thin coordinator that delegates to existing engines:
- SmartRouterEngine for model routing
- SmartEnricherEngine for RAG/web enrichment
- SmartCacheEngine for semantic caching (handled by SmartEnricherEngine)

The key insight is that SmartAlias has similar fields to SmartRouter and
SmartEnricher, so adapter classes can make SmartAlias work with both engines.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.models import SmartAlias
    from providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class AliasAsRouter:
    """
    Adapter to use SmartAlias as SmartRouter.

    Maps SmartAlias fields to SmartRouter field names:
    - routing_strategy -> strategy
    - candidates (list[str]) -> candidates (list[dict] with 'model' key)
    """

    def __init__(self, alias: "SmartAlias"):
        self._alias = alias

    def __getattr__(self, name):
        if name == "strategy":
            return self._alias.routing_strategy
        if name == "candidates":
            # SmartRouter expects list of dicts with 'model' key
            # SmartAlias stores list of strings
            return [{"model": m} for m in (self._alias.candidates or [])]
        return getattr(self._alias, name)


class AliasAsEnricher:
    """
    Adapter to use SmartAlias as SmartEnricher with optional routed target.

    Allows overriding target_model when routing selects a different model.
    """

    def __init__(self, alias: "SmartAlias", routed_target: str | None = None):
        self._alias = alias
        self._routed_target = routed_target
        # Copy _detached_stores if present (for session handling)
        if hasattr(alias, "_detached_stores"):
            self._detached_stores = alias._detached_stores

    def __getattr__(self, name):
        if name == "target_model" and self._routed_target:
            return self._routed_target
        return getattr(self._alias, name)


class SmartAliasEngine:
    """
    Coordinator for smart alias processing.

    Delegates to existing engines rather than reimplementing:
    - SmartRouterEngine handles model routing
    - SmartEnricherEngine handles RAG + Web enrichment + caching
    """

    def __init__(self, alias: "SmartAlias", registry: "ProviderRegistry"):
        """
        Initialize the engine.

        Args:
            alias: SmartAlias configuration
            registry: Provider registry for model resolution
        """
        self.alias = alias
        self.registry = registry

    def process(
        self,
        messages: list[dict],
        system: str | None = None,
        session_key: str | None = None,
    ):
        """
        Process a request through the smart alias pipeline.

        Returns an EnrichmentResult (from SmartEnricherEngine) so that
        proxy.py handles it exactly like a SmartEnricher.
        """
        from .smart_enricher import EnrichmentResult, SmartEnricherEngine
        from .smart_router import SmartRouterEngine

        # Apply alias system_prompt if configured
        effective_system = system
        if self.alias.system_prompt:
            if system:
                effective_system = f"{self.alias.system_prompt}\n\n{system}"
            else:
                effective_system = self.alias.system_prompt

        # Determine target model (may be modified by routing)
        target_model = self.alias.target_model

        # ========================================
        # STEP 1: SMART ROUTING (delegate to SmartRouterEngine)
        # ========================================
        routing_designator_usage = None
        routing_designator_model = None

        if self.alias.use_routing:
            router_adapter = AliasAsRouter(self.alias)
            router_engine = SmartRouterEngine(router_adapter, self.registry)

            try:
                routing_result = router_engine.route(
                    messages, effective_system, session_key
                )
                # Extract the routed model name from the result
                target_model = f"{routing_result.resolved.provider.prefix}/{routing_result.resolved.model_id}"
                routing_designator_usage = routing_result.designator_usage
                routing_designator_model = routing_result.designator_model
                logger.debug(
                    f"Smart alias '{self.alias.name}' routed to: {target_model}"
                )
            except Exception as e:
                logger.error(
                    f"Smart alias '{self.alias.name}' routing failed: {e}, using target_model"
                )

        # ========================================
        # STEP 2: ENRICHMENT (delegate to SmartEnricherEngine)
        # ========================================
        if self.alias.use_rag or self.alias.use_web:
            adapter = AliasAsEnricher(self.alias, target_model)
            enricher_engine = SmartEnricherEngine(adapter, self.registry)
            result = enricher_engine.enrich(messages, effective_system)

            # If routing was used and had designator usage, preserve it
            # (enricher may have its own designator for web query optimization)
            if routing_designator_usage and not result.designator_usage:
                result.designator_usage = routing_designator_usage
                result.designator_model = routing_designator_model

            return result

        # ========================================
        # NO ENRICHMENT - Simple alias/routing only
        # ========================================
        try:
            resolved = self.registry._resolve_actual_model(target_model)
            from providers.registry import ResolvedModel

            return EnrichmentResult(
                resolved=ResolvedModel(
                    provider=resolved.provider,
                    model_id=resolved.model_id,
                    alias_name=self.alias.name,
                    alias_tags=self.alias.tags or [],
                ),
                enricher_id=self.alias.id,
                enricher_name=self.alias.name,
                enricher_tags=self.alias.tags or [],
                augmented_system=effective_system,
                augmented_messages=messages,
                enrichment_type="none",
                context_injected=False,
                designator_usage=routing_designator_usage,
                designator_model=routing_designator_model,
            )
        except ValueError as e:
            logger.error(f"Smart alias '{self.alias.name}' target not available: {e}")
            raise
