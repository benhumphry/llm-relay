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

    @property
    def has_alias(self) -> bool:
        """Check if this resolution came from an alias."""
        return self.alias_name is not None

    @property
    def has_router(self) -> bool:
        """Check if this resolution came from a smart router."""
        return self._routing_result is not None

    @property
    def has_cache(self) -> bool:
        """Check if this resolution came from a smart cache."""
        return False  # Regular ResolvedModel is never from cache

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


@dataclass
class CacheResolvedModel(ResolvedModel):
    """
    Extended ResolvedModel that includes smart cache information.

    When resolve_model() returns a CacheResolvedModel, the proxy should:
    1. Check cache_result.is_cache_hit
    2. If hit: return cached_response directly
    3. If miss: forward to provider, then call cache_engine.store_response()
    """

    cache_result: "object" = None  # CacheResult from SmartCacheEngine.lookup()
    cache_engine: "object" = None  # SmartCacheEngine instance for storing responses

    @property
    def has_cache(self) -> bool:
        """Check if this resolution came from a smart cache."""
        return True

    @property
    def is_cache_hit(self) -> bool:
        """Check if this is a cache hit."""
        return self.cache_result and self.cache_result.is_cache_hit

    @property
    def cached_response(self) -> dict | None:
        """Get the cached response if this is a cache hit."""
        if self.cache_result and self.cache_result.is_cache_hit:
            return self.cache_result.cached_response
        return None


@dataclass
class AugmentorResolvedModel(ResolvedModel):
    """
    Extended ResolvedModel that includes smart augmentor information.

    When resolve_model() returns an AugmentorResolvedModel, the proxy should:
    1. Use augmentation_result.augmented_system instead of original system
    2. Use augmentation_result.augmented_messages instead of original messages
    3. Forward to the target provider with augmented context
    """

    augmentation_result: "object" = None  # AugmentationResult from SmartAugmentorEngine

    @property
    def has_augmentation(self) -> bool:
        """Check if this resolution came from a smart augmentor."""
        return True

    @property
    def augmented_system(self) -> str | None:
        """Get the augmented system prompt."""
        if self.augmentation_result:
            return self.augmentation_result.augmented_system
        return None

    @property
    def augmented_messages(self) -> list[dict]:
        """Get the augmented messages."""
        if self.augmentation_result:
            return self.augmentation_result.augmented_messages
        return []

    @property
    def augmentation_type(self) -> str:
        """Get the type of augmentation applied."""
        if self.augmentation_result:
            return self.augmentation_result.augmentation_type
        return "direct"


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
    ) -> "ResolvedModel | CacheResolvedModel":
        """
        Resolve a model name to a provider and model ID.

        Resolution order:
        1. Check if it's a smart router (v3.2) - requires messages
        2. Check if it's a smart cache (v3.3) - requires ChromaDB, returns CacheResolvedModel
        3. Check if it's a smart augmentor (v3.4) - returns AugmentorResolvedModel
        4. Check if it's an alias (v3.1)
        5. Check for provider prefix (e.g., "openai-gpt-4o" -> openai provider)
        6. Check each configured provider's models
        7. Fall back to default provider/model

        Args:
            model_name: User-provided model name
            messages: Optional message list (required for smart router/cache)
            system: Optional system prompt (for smart router/cache context)
            session_key: Optional session key (for per_session routing)

        Returns:
            ResolvedModel with provider, model_id, and optional alias info
            OR CacheResolvedModel for smart cache lookups
            OR AugmentorResolvedModel for smart augmentor lookups

        Raises:
            ValueError: If model not found and no default configured
        """
        from context.chroma import is_chroma_available
        from db import (
            get_alias_by_name,
            get_smart_augmentor_by_name,
            get_smart_cache_by_name,
            get_smart_router_by_name,
        )
        from routing import SmartAugmentorEngine, SmartCacheEngine, SmartRouterEngine

        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Step 1: Check if it's a smart router (v3.2)
        router = get_smart_router_by_name(name)
        if router and router.enabled:
            if messages is not None:
                # Use smart routing
                logger.debug(f"Using smart router '{name}'")
                engine = SmartRouterEngine(router, self)
                result = engine.route(messages, system, session_key)
                # Store routing metadata for later use
                result.resolved._routing_result = result
                return result.resolved
            else:
                # No messages provided - fall back to router's fallback model
                logger.debug(
                    f"Smart router '{name}' called without messages, using fallback"
                )
                try:
                    target_result = self._resolve_actual_model(router.fallback_model)
                    return ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=router.name,
                        alias_tags=router.tags or [],
                    )
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 2: Check if it's a smart cache (v3.3)
        cache = get_smart_cache_by_name(name)
        if cache and cache.enabled:
            # Smart caches require ChromaDB
            if is_chroma_available():
                if messages is not None:
                    logger.debug(f"Using smart cache '{name}'")
                    engine = SmartCacheEngine(cache, self)
                    cache_result = engine.lookup(messages, system)
                    # Return a special CacheResolvedModel that the proxy can handle
                    return CacheResolvedModel(
                        provider=cache_result.resolved.provider,
                        model_id=cache_result.resolved.model_id,
                        alias_name=cache.name,
                        alias_tags=cache.tags or [],
                        cache_result=cache_result,
                        cache_engine=engine,
                    )
                else:
                    # No messages - just resolve to target model
                    logger.debug(
                        f"Smart cache '{name}' called without messages, resolving target"
                    )
                    try:
                        target_result = self._resolve_actual_model(cache.target_model)
                        return ResolvedModel(
                            provider=target_result.provider,
                            model_id=target_result.model_id,
                            alias_name=cache.name,
                            alias_tags=cache.tags or [],
                        )
                    except ValueError:
                        pass  # Fall through to normal resolution
            else:
                logger.warning(
                    f"Smart cache '{name}' requested but ChromaDB not available, "
                    "falling back to target model"
                )
                try:
                    target_result = self._resolve_actual_model(cache.target_model)
                    return ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=cache.name,
                        alias_tags=cache.tags or [],
                    )
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 3: Check if it's a smart augmentor (v3.4)
        augmentor = get_smart_augmentor_by_name(name)
        if augmentor and augmentor.enabled:
            if messages is not None:
                logger.debug(f"Using smart augmentor '{name}'")
                engine = SmartAugmentorEngine(augmentor, self)
                augmentation_result = engine.augment(messages, system)
                # Return an AugmentorResolvedModel that the proxy can handle
                return AugmentorResolvedModel(
                    provider=augmentation_result.resolved.provider,
                    model_id=augmentation_result.resolved.model_id,
                    alias_name=augmentor.name,
                    alias_tags=augmentor.tags or [],
                    augmentation_result=augmentation_result,
                )
            else:
                # No messages - just resolve to target model
                logger.debug(
                    f"Smart augmentor '{name}' called without messages, resolving target"
                )
                try:
                    target_result = self._resolve_actual_model(augmentor.target_model)
                    return ResolvedModel(
                        provider=target_result.provider,
                        model_id=target_result.model_id,
                        alias_name=augmentor.name,
                        alias_tags=augmentor.tags or [],
                    )
                except ValueError:
                    pass  # Fall through to normal resolution

        # Step 4: Check if it's an alias (v3.1)
        alias = get_alias_by_name(name)
        if alias and alias.enabled:
            logger.debug(f"Resolving alias '{name}' -> '{alias.target_model}'")
            # Resolve the target model
            try:
                target_result = self._resolve_actual_model(alias.target_model)
                return ResolvedModel(
                    provider=target_result.provider,
                    model_id=target_result.model_id,
                    alias_name=alias.name,
                    alias_tags=alias.tags or [],
                )
            except ValueError:
                # Target model not available, fall back to default
                logger.warning(
                    f"Alias '{name}' target '{alias.target_model}' not available, "
                    "falling back to default"
                )
                if self._default_provider and self._default_model:
                    default = self._providers.get(self._default_provider)
                    if default and default.is_available():
                        return ResolvedModel(
                            provider=default,
                            model_id=self._default_model,
                            alias_name=alias.name,
                            alias_tags=alias.tags or [],
                        )

        # Step 5-7: Normal model resolution
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
        Get combined model list from all available providers, aliases, smart routers, and smart caches.

        Returns list of model info dicts suitable for /api/tags response.
        Models are already filtered for enabled status in load_models_for_provider().
        """
        from context.chroma import is_chroma_available
        from db import (
            get_enabled_aliases,
            get_enabled_smart_caches,
            get_enabled_smart_routers,
        )

        models = []
        seen = set()

        # Add aliases first (user-defined shortcuts)
        aliases = get_enabled_aliases()
        for alias_name, alias in aliases.items():
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

        # Add smart routers
        routers = get_enabled_smart_routers()
        for router_name, router in routers.items():
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

        # Add smart caches (only if ChromaDB is available)
        if is_chroma_available():
            caches = get_enabled_smart_caches()
            for cache_name, cache in caches.items():
                seen.add(cache_name)
                models.append(
                    {
                        "name": cache_name,
                        "model": cache_name,
                        "provider": "smart-cache",
                        "details": {
                            "family": "smart-cache",
                            "parameter_size": "",
                            "quantization_level": "",
                        },
                        "description": cache.description
                        or f"Cached responses for {cache.target_model}",
                        "context_length": 0,
                        "capabilities": ["caching"],
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
        Includes aliases, smart routers, smart caches, and provider models.
        """
        from context.chroma import is_chroma_available
        from db import (
            get_enabled_aliases,
            get_enabled_smart_caches,
            get_enabled_smart_routers,
        )

        """
        Get combined model list in OpenAI format.

        Returns list of model info dicts suitable for /v1/models response.
        Includes aliases, smart routers, smart caches, and provider models.
        """
        from context.chroma import is_chroma_available
        from db import (
            get_enabled_aliases,
            get_enabled_smart_caches,
            get_enabled_smart_routers,
        )

        models = []
        seen = set()

        # Add aliases first (user-defined shortcuts)
        aliases = get_enabled_aliases()
        for alias_name in aliases.keys():
            seen.add(alias_name)
            models.append(
                {
                    "id": alias_name,
                    "object": "model",
                    "owned_by": "alias",
                }
            )

        # Add smart routers
        routers = get_enabled_smart_routers()
        for router_name in routers.keys():
            seen.add(router_name)
            models.append(
                {
                    "id": router_name,
                    "object": "model",
                    "owned_by": "smart-router",
                }
            )

        # Add smart caches (only if ChromaDB is available)
        if is_chroma_available():
            caches = get_enabled_smart_caches()
            for cache_name in caches.keys():
                seen.add(cache_name)
                models.append(
                    {
                        "id": cache_name,
                        "object": "model",
                        "owned_by": "smart-cache",
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
