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
    - candidates: SmartAlias already stores list[dict] with 'model' key
    """

    def __init__(self, alias: "SmartAlias"):
        self._alias = alias

    def __getattr__(self, name):
        if name == "strategy":
            return self._alias.routing_strategy
        if name == "candidates":
            # SmartAlias.candidates is already list[dict] with 'model' key
            # Same format as SmartRouter expects
            return self._alias.candidates or []
        return getattr(self._alias, name)


class AliasAsEnricher:
    """
    Adapter to use SmartAlias as SmartEnricher with optional routed target.

    Allows overriding target_model when routing selects a different model.
    Also carries routing_config for unified designator calls.
    """

    def __init__(
        self,
        alias: "SmartAlias",
        routed_target: str | None = None,
        routing_config: dict | None = None,
    ):
        self._alias = alias
        self._routed_target = routed_target
        self._routing_config = routing_config
        # Copy _detached_stores if present (for session handling)
        if hasattr(alias, "_detached_stores"):
            self._detached_stores = alias._detached_stores
        # Copy _detached_live_sources if present
        if hasattr(alias, "_detached_live_sources"):
            self._detached_live_sources = alias._detached_live_sources

    def __getattr__(self, name):
        if name == "target_model" and self._routed_target:
            return self._routed_target
        if name == "routing_config":
            return self._routing_config
        return getattr(self._alias, name)


class SmartAliasEngine:
    """
    Coordinator for smart alias processing.

    Delegates to existing engines rather than reimplementing:
    - SmartRouterEngine handles model routing
    - SmartEnricherEngine handles RAG + Web enrichment + caching
    """

    def __init__(
        self,
        alias: "SmartAlias",
        registry: "ProviderRegistry",
        passthrough_model: str | None = None,
    ):
        """
        Initialize the engine.

        Args:
            alias: SmartAlias configuration
            registry: Provider registry for model resolution
            passthrough_model: If set, use this model instead of alias.target_model
                               (used by smart tags with passthrough_model=True)
        """
        self.alias = alias
        self.registry = registry
        self.passthrough_model = passthrough_model

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

        # Determine target model (may be modified by routing or passthrough)
        # If passthrough_model is set, use that instead of alias target
        target_model = (
            self.passthrough_model
            if self.passthrough_model
            else self.alias.target_model
        )

        # Check if we can use unified designator call
        use_smart_selection = getattr(self.alias, "use_smart_source_selection", False)
        use_routing = self.alias.use_routing
        use_enrichment = (
            self.alias.use_rag
            or self.alias.use_web
            or getattr(self.alias, "use_live_data", False)
        )

        # ========================================
        # UNIFIED PATH: Smart source selection handles routing + enrichment
        # ========================================
        if use_smart_selection and (use_routing or use_enrichment):
            # Build routing config if routing is enabled
            routing_config = None
            if use_routing:
                routing_config = self._build_routing_config(messages, effective_system)

            adapter = AliasAsEnricher(self.alias, target_model, routing_config)
            enricher_engine = SmartEnricherEngine(adapter, self.registry)
            result = enricher_engine.enrich(messages, effective_system, session_key)

            logger.debug(f"Smart alias '{self.alias.name}' unified selection complete")
            return result

        # ========================================
        # LEGACY PATH: Separate routing then enrichment
        # ========================================
        routing_designator_usage = None
        routing_designator_model = None

        # STEP 1: SMART ROUTING (separate SmartRouterEngine call)
        if use_routing:
            router_adapter = AliasAsRouter(self.alias)
            router_engine = SmartRouterEngine(router_adapter, self.registry)

            try:
                routing_result = router_engine.route(
                    messages, effective_system, session_key
                )
                target_model = f"{routing_result.resolved.provider.name}/{routing_result.resolved.model_id}"
                routing_designator_usage = routing_result.designator_usage
                routing_designator_model = routing_result.designator_model
                logger.debug(
                    f"Smart alias '{self.alias.name}' routed to: {target_model}"
                )
            except Exception as e:
                logger.error(
                    f"Smart alias '{self.alias.name}' routing failed: {e}, using target_model"
                )

        # STEP 2: ENRICHMENT (delegate to SmartEnricherEngine)
        if use_enrichment:
            adapter = AliasAsEnricher(self.alias, target_model)
            enricher_engine = SmartEnricherEngine(adapter, self.registry)
            result = enricher_engine.enrich(messages, effective_system, session_key)

            if routing_designator_usage and not result.designator_usage:
                result.designator_usage = routing_designator_usage
                result.designator_model = routing_designator_model

            return result

        # ========================================
        # NO ENRICHMENT - Simple alias/routing only
        # ========================================
        # Even without RAG/Web, we may still have memory enabled
        if getattr(self.alias, "use_memory", False):
            adapter = AliasAsEnricher(self.alias, target_model)
            enricher_engine = SmartEnricherEngine(adapter, self.registry)
            result = enricher_engine.enrich(messages, effective_system, session_key)

            if routing_designator_usage and not result.designator_usage:
                result.designator_usage = routing_designator_usage
                result.designator_model = routing_designator_model

            return result

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

    def _build_routing_config(
        self, messages: list[dict], system: str | None
    ) -> dict | None:
        """
        Build routing configuration for the unified designator call.

        Returns dict with candidates, purpose, token_count, has_images.
        """
        if not self.alias.candidates:
            return None

        # Estimate token count
        token_count = self._estimate_tokens(messages, system)
        has_images = self._check_for_images(messages)

        # Build candidates with info
        candidates = []
        for candidate in self.alias.candidates:
            model_ref = candidate.get("model", "")
            notes = candidate.get("notes", "")

            try:
                resolved = self.registry._resolve_actual_model(model_ref)
                model_info = resolved.provider.get_models().get(resolved.model_id)

                if model_info:
                    # Filter by context length
                    if model_info.context_length < token_count * 1.5:
                        continue
                    # Filter by vision if needed
                    if has_images and "vision" not in model_info.capabilities:
                        continue

                    candidates.append(
                        {
                            "model": model_ref,
                            "notes": notes,
                            "context_length": model_info.context_length,
                            "capabilities": model_info.capabilities,
                        }
                    )
                else:
                    candidates.append(
                        {
                            "model": model_ref,
                            "notes": notes,
                            "context_length": 128000,
                            "capabilities": [],
                        }
                    )
            except ValueError:
                logger.debug(f"Candidate model '{model_ref}' not available, skipping")
                continue

        if not candidates:
            return None

        return {
            "candidates": candidates,
            "purpose": self.alias.purpose or "General purpose",
            "token_count": token_count,
            "has_images": has_images,
        }

    def _estimate_tokens(self, messages: list[dict], system: str | None) -> int:
        """Estimate token count using char/4 approximation."""
        total_chars = len(system) if system else 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_chars += len(block.get("text", ""))

        return total_chars // 4

    def _check_for_images(self, messages: list[dict]) -> bool:
        """Check if any messages contain images."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                        "image",
                        "image_url",
                    ):
                        return True
            if msg.get("images"):
                return True
        return False
