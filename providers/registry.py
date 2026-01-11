"""
Provider registry for managing multiple LLM providers.

The registry handles:
- Provider registration and discovery
- Model resolution across all providers
- Aggregating model lists for API responses
- Alias resolution (v3.1)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """Result of model resolution, including alias information."""

    provider: "LLMProvider"
    model_id: str
    alias_name: str | None = None
    alias_tags: list[str] | None = None
    # Internal: routing result from smart router (set dynamically)
    _routing_result: "object | None" = None
    # Track if this resolution used the default fallback
    is_default_fallback: bool = False
    # Cache configuration (set when resolving from cache-enabled entity)
    _cache_config: "object | None" = None

    @property
    def has_alias(self) -> bool:
        """Check if this resolution came from an alias."""
        return self.alias_name is not None

    @property
    def has_router(self) -> bool:
        """Check if this resolution came from a smart router."""
        return self._routing_result is not None

    @property
    def router_name(self) -> str | None:
        """Get the router name if this came from a smart router."""
        if self._routing_result:
            return self._routing_result.router_name
        return None

    @property
    def designator_usage(self) -> dict | None:
        """Get designator usage info if this came from a smart router."""
        if self._routing_result:
            return self._routing_result.designator_usage
        return None

    @property
    def has_cache(self) -> bool:
        """Check if caching is enabled for this resolution."""
        return self._cache_config is not None and self._cache_config.use_cache

    @property
    def cache_config(self) -> "object | None":
        """Get the cache configuration (Alias, Router, Redirect, or Enricher)."""
        return self._cache_config


@dataclass
class EnricherResolvedModel(ResolvedModel):
    """
    Extended ResolvedModel that includes smart enricher information.

    When resolve_model() returns an EnricherResolvedModel, the proxy should:
    1. Use enrichment_result.augmented_system instead of original system
    2. Use enrichment_result.augmented_messages instead of original messages
    3. Forward to the target provider with enriched context
    """

    enrichment_result: "object" = None  # EnrichmentResult from SmartEnricherEngine

    @property
    def has_enrichment(self) -> bool:
        """Check if this resolution came from a smart enricher."""
        return True

    @property
    def augmented_system(self) -> str | None:
        """Get the augmented system prompt with enriched context."""
        if self.enrichment_result:
            return self.enrichment_result.augmented_system
        return None

    @property
    def augmented_messages(self) -> list[dict]:
        """Get the messages (usually unchanged)."""
        if self.enrichment_result:
            return self.enrichment_result.augmented_messages
        return []

    @property
    def context_injected(self) -> bool:
        """Check if context was injected."""
        if self.enrichment_result:
            return self.enrichment_result.context_injected
        return False

    @property
    def enrichment_type(self) -> str:
        """Get the type of enrichment applied (rag, web, hybrid, none)."""
        if self.enrichment_result:
            return self.enrichment_result.enrichment_type
        return "none"


@dataclass
class ProviderRegistry:
    """
    Central registry for all LLM providers.

    Handles model routing, provider discovery, and model aggregation.
    """

    def __init__(self):
        self._providers: dict[str, "LLMProvider"] = {}
        self._default_provider: str | None = None
        self._default_model: str | None = None

    def register(self, provider: "LLMProvider") -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")

    def get_provider(self, name: str) -> "LLMProvider | None":
        """Get a provider by name."""
        return self._providers.get(name)

    def get_configured_providers(self) -> list["LLMProvider"]:
        """Return list of providers that have valid API credentials."""
        return [p for p in self._providers.values() if p.is_configured()]

    def get_available_providers(self) -> list["LLMProvider"]:
        """Return list of providers that are available for API requests.

        A provider is available if it's both enabled AND has valid credentials.
        """
        return [p for p in self._providers.values() if p.is_available()]

    def get_all_providers(self) -> list["LLMProvider"]:
        """Return all registered providers."""
        return list(self._providers.values())

    def set_default(self, provider_name: str, model_id: str) -> None:
        """Set the default provider and model for unknown model names."""
        self._default_provider = provider_name
        self._default_model = model_id

    def resolve_model(
        self,
        model_name: str,
        messages: list[dict] | None = None,
        system: str | None = None,
        session_key: str | None = None,
    ) -> "ResolvedModel | EnricherResolvedModel":
        """
        Resolve a model name to a provider and model ID.

        Resolution order:
        1. Check for redirects (v3.7) - transparent model name mapping
        2. Check if it's a smart alias (unified) - routing + enrichment + caching
        3. Check if it's a smart router (v3.2, legacy) - requires messages
        4. Check if it's a smart enricher (legacy) - returns EnricherResolvedModel
        5. Check if it's an alias (v3.1, legacy)
        6. Check for provider prefix (e.g., "openai-gpt-4o" -> openai provider)
        7. Check each configured provider's models
        8. Fall back to default provider/model

        Args:
            model_name: User-provided model name
            messages: Optional message list (required for smart router/enricher)
            system: Optional system prompt (for smart router/enricher context)
            session_key: Optional session key (for per_session routing)

        Returns:
            ResolvedModel with provider, model_id, and optional alias info
            OR EnricherResolvedModel for smart enricher/smart alias lookups

        Raises:
            ValueError: If model not found and no default configured
        """
        from context.chroma import is_chroma_available
        from db import (
            find_matching_redirect,
            get_alias_by_name,
            get_smart_alias_by_name,
            get_smart_enricher_by_name,
            get_smart_router_by_name,
            increment_redirect_count,
        )
        from routing import (
            SmartAliasEngine,
            SmartEnricherEngine,
            SmartRouterEngine,
        )

        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Step 1: Check for redirects (v3.7)
        # Try exact match first, then try with provider-prefix converted to slash format
        # This handles both "anthropic/claude-3" and "anthropic-claude-3" formats
        redirect_match = find_matching_redirect(name)
        if not redirect_match and "-" in name and "/" not in name:
            # Convert first hyphen to slash for provider-prefix format
            # e.g., "anthropic-claude-opus-4-1" -> "anthropic/claude-opus-4-1"
            for provider_name in self._providers.keys():
                prefix = f"{provider_name}-"
                if name.startswith(prefix):
                    slash_name = f"{provider_name}/{name[len(prefix) :]}"
                    redirect_match = find_matching_redirect(slash_name)
                    break
        if redirect_match:
            redirect, resolved_target = redirect_match
            logger.debug(f"Redirect: {name} -> {resolved_target}")
            # Increment redirect counter (fire and forget)
            try:
                increment_redirect_count(redirect.id)
            except Exception:
                pass
            # Recursively resolve the target (allows chaining)
            result = self.resolve_model(resolved_target, messages, system, session_key)
            # If redirect has tags, merge them with existing tags and set alias
            if redirect.tags:
                # Merge redirect tags with any existing tags from the resolved target
                existing_tags = result.alias_tags or []
                merged_tags = list(redirect.tags) + [
                    t for t in existing_tags if t not in redirect.tags
                ]
                result.alias_name = redirect.source
                result.alias_tags = merged_tags
            # Set cache config if caching is enabled on redirect
            if redirect.use_cache:
                result._cache_config = redirect
            return result

        # Step 2: Check if it's a smart alias (unified routing + enrichment + caching)
        smart_alias = get_smart_alias_by_name(name)
        if smart_alias and smart_alias.enabled:
            # Check if smart alias needs ChromaDB (for RAG functionality)
            needs_chroma = smart_alias.use_rag
            if not needs_chroma or is_chroma_available():
                if messages is not None:
                    logger.debug(f"Using smart alias '{name}'")
                    engine = SmartAliasEngine(smart_alias, self)
                    enrichment_result = engine.process(messages, system, session_key)

                    # Return an EnricherResolvedModel (same as SmartEnricher)
                    # This ensures proxy.py handles it correctly
                    result = EnricherResolvedModel(
                        provider=enrichment_result.resolved.provider,
                        model_id=enrichment_result.resolved.model_id,
                        alias_name=smart_alias.name,
                        alias_tags=smart_alias.tags or [],
                        enrichment_result=enrichment_result,
                    )
                    # Set cache config if caching is enabled
                    if smart_alias.cache_enabled:
                        result._cache_config = smart_alias
                    return result
                else:
                    # No messages - just resolve to target model
                    logger.debug(
                        f"Smart alias '{name}' called without messages, resolving target"
                    )
                    try:
                        target_result = self._resolve_actual_model(
                            smart_alias.target_model
                        )
                        return ResolvedModel(
                            provider=target_result.provider,
                            model_id=target_result.model_id,
                            alias_name=smart_alias.name,
                            alias_tags=smart_alias.tags or [],
                        )
                    except ValueError:
                        pass  # Fall through to normal resolution
            else:
                logger.warning(
                    f"Smart alias '{name}' requires ChromaDB for RAG but it's not available, "
                    "falling back to target model"
                )
                try:
                    target_result = self._resolve_actual_model(smart_alias.target_model)
                    return ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=smart_alias.name,
                        alias_tags=smart_alias.tags or [],
                    )
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 3: Check if it's a smart router (v3.2, legacy)
        router = get_smart_router_by_name(name)
        if router and router.enabled:
            if messages is not None:
                # Use smart routing
                logger.debug(f"Using smart router '{name}'")
                engine = SmartRouterEngine(router, self)
                routing_result = engine.route(messages, system, session_key)
                # Store routing metadata for later use
                routing_result.resolved._routing_result = routing_result
                # Set cache config if caching is enabled
                if router.use_cache:
                    routing_result.resolved._cache_config = router
                return routing_result.resolved
            else:
                # No messages provided - fall back to router's fallback model
                logger.debug(
                    f"Smart router '{name}' called without messages, using fallback"
                )
                try:
                    target_result = self._resolve_actual_model(router.fallback_model)
                    result = ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=router.name,
                        alias_tags=router.tags or [],
                    )
                    if router.use_cache:
                        result._cache_config = router
                    return result
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 4: Check if it's a smart enricher (legacy)
        enricher = get_smart_enricher_by_name(name)
        if enricher and enricher.enabled:
            # Check if enricher needs ChromaDB (for RAG functionality)
            needs_chroma = enricher.use_rag
            if not needs_chroma or is_chroma_available():
                if messages is not None:
                    logger.debug(f"Using smart enricher '{name}'")
                    engine = SmartEnricherEngine(enricher, self)
                    enrichment_result = engine.enrich(messages, system)
                    # Return an EnricherResolvedModel that the proxy can handle
                    result = EnricherResolvedModel(
                        provider=enrichment_result.resolved.provider,
                        model_id=enrichment_result.resolved.model_id,
                        alias_name=enricher.name,
                        alias_tags=enricher.tags or [],
                        enrichment_result=enrichment_result,
                    )
                    # Set cache config if caching is enabled AND use_web is False
                    # (realtime web data shouldn't be cached)
                    if enricher.cache_enabled:
                        result._cache_config = enricher
                    return result
                else:
                    # No messages - just resolve to target model
                    logger.debug(
                        f"Smart enricher '{name}' called without messages, resolving target"
                    )
                    try:
                        target_result = self._resolve_actual_model(
                            enricher.target_model
                        )
                        return ResolvedModel(
                            provider=target_result.provider,
                            model_id=target_result.model_id,
                            alias_name=enricher.name,
                            alias_tags=enricher.tags or [],
                        )
                    except ValueError:
                        pass  # Fall through to normal resolution
            else:
                logger.warning(
                    f"Smart enricher '{name}' requires ChromaDB for RAG but it's not available, "
                    "falling back to target model"
                )
                try:
                    target_result = self._resolve_actual_model(enricher.target_model)
                    return ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=enricher.name,
                        alias_tags=enricher.tags or [],
                    )
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 5: Check if it's an alias (v3.1, legacy)
        alias = get_alias_by_name(name)
        if alias and alias.enabled:
            logger.debug(f"Resolving alias '{name}' -> '{alias.target_model}'")
            # Resolve the target model
            try:
                target_result = self._resolve_actual_model(alias.target_model)
                result = ResolvedModel(
                    provider=target_result.provider,
                    model_id=target_result.model_id,
                    alias_name=alias.name,
                    alias_tags=alias.tags or [],
                )
                # Set cache config if caching is enabled
                if alias.use_cache:
                    result._cache_config = alias
                return result
            except ValueError:
                # Target model not available, fall back to default
                logger.warning(
                    f"Alias '{name}' target '{alias.target_model}' not available, "
                    "falling back to default"
                )
                if self._default_provider and self._default_model:
                    default = self._providers.get(self._default_provider)
                    if default and default.is_available():
                        result = ResolvedModel(
                            provider=default,
                            model_id=self._default_model,
                            alias_name=alias.name,
                            alias_tags=alias.tags or [],
                        )
                        if alias.use_cache:
                            result._cache_config = alias
                        return result

        # Step 6-8: Normal model resolution
        return self._resolve_actual_model(name)

    def _resolve_actual_model(self, model_name: str) -> ResolvedModel:
        """
        Internal method to resolve a model name without alias checking.

        Used by resolve_model for both direct lookups and alias target resolution.
        """
        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Check for explicit provider/model format (e.g., "gemini/gemini-2.5-pro")
        if "/" in name:
            provider_name, model_part = name.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider and provider.is_available():
                resolved = provider.resolve_model(model_part)
                if resolved:
                    return ResolvedModel(provider=provider, model_id=resolved)

        # Check for provider prefix (e.g., "openai-gpt-4o")
        for provider_name, provider in self._providers.items():
            if not provider.is_available():
                continue

            prefix = f"{provider_name}-"
            if name.startswith(prefix):
                model_part = name[len(prefix) :]
                resolved = provider.resolve_model(model_part)
                if resolved:
                    return ResolvedModel(provider=provider, model_id=resolved)
                # Also try the full name without prefix removal
                resolved = provider.resolve_model(name)
                if resolved:
                    return ResolvedModel(provider=provider, model_id=resolved)

        # Check each available provider (enabled + configured)
        for provider in self.get_available_providers():
            resolved = provider.resolve_model(name)
            if resolved:
                return ResolvedModel(provider=provider, model_id=resolved)

        # Fall back to default
        if self._default_provider and self._default_model:
            default = self._providers.get(self._default_provider)
            if default and default.is_available():
                logger.warning(
                    f'Unknown model "{model_name}", using default: '
                    f"{self._default_provider}/{self._default_model}"
                )
                return ResolvedModel(
                    provider=default,
                    model_id=self._default_model,
                    is_default_fallback=True,
                )

        raise ValueError(
            f'Model "{model_name}" not found in any available provider. '
            f"Available providers: {[p.name for p in self.get_available_providers()]}"
        )

    def _get_default_model(self) -> ResolvedModel:
        """
        Get the default model.

        Used by SmartRouterEngine as ultimate fallback.

        Raises:
            ValueError: If no default configured or available
        """
        if self._default_provider and self._default_model:
            default = self._providers.get(self._default_provider)
            if default and default.is_available():
                return ResolvedModel(provider=default, model_id=self._default_model)

        raise ValueError("No default model configured or available")

    def list_all_models(self) -> list[dict]:
        """
        Get combined model list from all available providers, smart aliases, and legacy entities.

        Returns list of model info dicts suitable for /api/tags response.
        Models are already filtered for enabled status in load_models_for_provider().
        """
        from context.chroma import is_chroma_available
        from db import (
            get_enabled_aliases,
            get_enabled_smart_aliases,
            get_enabled_smart_enrichers,
            get_enabled_smart_routers,
        )

        models = []
        seen = set()

        # Add smart aliases first (unified feature)
        smart_aliases = get_enabled_smart_aliases()
        for sa_name, sa in smart_aliases.items():
            # Skip if RAG-only and ChromaDB not available
            if sa.use_rag and not sa.use_web and not is_chroma_available():
                continue
            seen.add(sa_name)
            # Build capabilities list
            capabilities = []
            if sa.use_routing:
                capabilities.append("routing")
            if sa.use_rag:
                capabilities.extend(["rag", "documents"])
            if sa.use_web:
                capabilities.extend(["search", "scrape"])
            if sa.cache_enabled:
                capabilities.append("cache")
            models.append(
                {
                    "name": sa_name,
                    "model": sa_name,
                    "provider": "smart-alias",
                    "details": {
                        "family": "smart-alias",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                    "description": sa.description
                    or f"Smart alias for {sa.target_model}",
                    "context_length": 0,
                    "capabilities": capabilities,
                }
            )

        # Add legacy aliases (user-defined shortcuts)
        aliases = get_enabled_aliases()
        for alias_name, alias in aliases.items():
            if alias_name in seen:
                continue
            seen.add(alias_name)
            models.append(
                {
                    "name": alias_name,
                    "model": alias_name,
                    "provider": "alias",
                    "details": {
                        "family": "alias",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                    "description": alias.description
                    or f"Alias for {alias.target_model}",
                    "context_length": 0,
                    "capabilities": [],
                }
            )

        # Add legacy smart routers
        routers = get_enabled_smart_routers()
        for router_name, router in routers.items():
            if router_name in seen:
                continue
            seen.add(router_name)
            models.append(
                {
                    "name": router_name,
                    "model": router_name,
                    "provider": "smart-router",
                    "details": {
                        "family": "smart-router",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                    "description": router.description or router.purpose,
                    "context_length": 0,
                    "capabilities": [],
                }
            )

        # Add legacy smart enrichers (RAG + Web)
        enrichers = get_enabled_smart_enrichers()
        for enricher_name, enricher in enrichers.items():
            if enricher_name in seen:
                continue
            # Skip if RAG-only enricher and ChromaDB not available
            if enricher.use_rag and not enricher.use_web and not is_chroma_available():
                continue
            seen.add(enricher_name)
            # Build capabilities list based on what's enabled
            capabilities = []
            if enricher.use_rag:
                capabilities.extend(["rag", "documents"])
            if enricher.use_web:
                capabilities.extend(["search", "scrape"])
            models.append(
                {
                    "name": enricher_name,
                    "model": enricher_name,
                    "provider": "smart-enricher",
                    "details": {
                        "family": "smart-enricher",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                    "description": enricher.description
                    or f"Enriched context for {enricher.target_model}",
                    "context_length": 0,
                    "capabilities": capabilities,
                }
            )

        # Add provider models
        for provider in self.get_available_providers():
            for model_id, info in provider.get_models().items():
                full_name = f"{provider.name}-{model_id}"
                if full_name in seen:
                    continue

                seen.add(full_name)

                # Create Ollama-compatible model entry
                models.append(
                    {
                        "name": full_name,
                        "model": full_name,
                        "provider": provider.name,
                        "details": {
                            "family": info.family,
                            "parameter_size": info.parameter_size,
                            "quantization_level": info.quantization_level,
                        },
                        "description": info.description,
                        "context_length": info.context_length,
                        "capabilities": info.capabilities,
                    }
                )

        return models

    def list_openai_models(self) -> list[dict]:
        """
        Get combined model list in OpenAI format.

        Returns list of model info dicts suitable for /v1/models response.
        Includes smart aliases, legacy entities, and provider models.
        """
        from context.chroma import is_chroma_available
        from db import (
            get_enabled_aliases,
            get_enabled_smart_aliases,
            get_enabled_smart_enrichers,
            get_enabled_smart_routers,
        )

        models = []
        seen = set()

        # Add smart aliases first (unified feature)
        smart_aliases = get_enabled_smart_aliases()
        for sa_name, sa in smart_aliases.items():
            # Skip if RAG-only and ChromaDB not available
            if sa.use_rag and not sa.use_web and not is_chroma_available():
                continue
            seen.add(sa_name)
            models.append(
                {
                    "id": sa_name,
                    "object": "model",
                    "owned_by": "smart-alias",
                }
            )

        # Add legacy aliases
        aliases = get_enabled_aliases()
        for alias_name in aliases.keys():
            if alias_name in seen:
                continue
            seen.add(alias_name)
            models.append(
                {
                    "id": alias_name,
                    "object": "model",
                    "owned_by": "alias",
                }
            )

        # Add legacy smart routers
        routers = get_enabled_smart_routers()
        for router_name in routers.keys():
            if router_name in seen:
                continue
            seen.add(router_name)
            models.append(
                {
                    "id": router_name,
                    "object": "model",
                    "owned_by": "smart-router",
                }
            )

        # Add legacy smart enrichers
        enrichers = get_enabled_smart_enrichers()
        for enricher_name, enricher in enrichers.items():
            if enricher_name in seen:
                continue
            # Skip if RAG-only enricher and ChromaDB not available
            if enricher.use_rag and not enricher.use_web and not is_chroma_available():
                continue
            seen.add(enricher_name)
            models.append(
                {
                    "id": enricher_name,
                    "object": "model",
                    "owned_by": "smart-enricher",
                }
            )

        # Add provider models
        for provider in self.get_available_providers():
            for model_id, info in provider.get_models().items():
                full_name = f"{provider.name}-{model_id}"
                if full_name in seen:
                    continue

                seen.add(full_name)

                models.append(
                    {
                        "id": full_name,
                        "object": "model",
                        "owned_by": provider.name,
                    }
                )

        return models


# Global registry instance
registry = ProviderRegistry()
