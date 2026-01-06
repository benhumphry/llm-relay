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
    ) -> ResolvedModel:
        """
        Resolve a model name to a provider and model ID.

        Resolution order:
        1. Check if it's a smart router (v3.2) - requires messages
        2. Check if it's an alias (v3.1)
        3. Check for provider prefix (e.g., "openai-gpt-4o" -> openai provider)
        4. Check each configured provider's models
        5. Fall back to default provider/model

        Args:
            model_name: User-provided model name
            messages: Optional message list (required for smart router)
            system: Optional system prompt (for smart router context)
            session_key: Optional session key (for per_session routing)

        Returns:
            ResolvedModel with provider, model_id, and optional alias info

        Raises:
            ValueError: If model not found and no default configured
        """
        from db import get_alias_by_name, get_smart_router_by_name
        from routing import SmartRouterEngine

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

        # Step 2: Check if it's an alias (v3.1)
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

        # Step 3-5: Normal model resolution
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
        Get combined model list from all available providers, aliases, and smart routers.

        Returns list of model info dicts suitable for /api/tags response.
        Models are already filtered for enabled status in load_models_for_provider().
        """
        from db import get_enabled_aliases, get_enabled_smart_routers

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
        Includes aliases, smart routers, and provider models.
        """
        from db import get_enabled_aliases, get_enabled_smart_routers

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
