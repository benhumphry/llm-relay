"""
Provider registry for managing multiple LLM providers.

The registry handles:
- Provider registration and discovery
- Model resolution across all providers
- Aggregating model lists for API responses
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider, ModelInfo

logger = logging.getLogger(__name__)


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

    def resolve_model(self, model_name: str) -> tuple["LLMProvider", str]:
        """
        Resolve a model name to a provider and model ID.

        Resolution order:
        1. Check for provider prefix (e.g., "openai-gpt-4o" -> openai provider)
        2. Check each configured provider's models
        3. Fall back to default provider/model

        Args:
            model_name: User-provided model name

        Returns:
            Tuple of (provider, model_id)

        Raises:
            ValueError: If model not found and no default configured
        """
        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Check for provider prefix (e.g., "openai-gpt-4o")
        for provider_name, provider in self._providers.items():
            if not provider.is_available():
                continue

            prefix = f"{provider_name}-"
            if name.startswith(prefix):
                model_part = name[len(prefix) :]
                resolved = provider.resolve_model(model_part)
                if resolved:
                    return provider, resolved
                # Also try the full name without prefix removal
                resolved = provider.resolve_model(name)
                if resolved:
                    return provider, resolved

        # Check each available provider (enabled + configured)
        for provider in self.get_available_providers():
            resolved = provider.resolve_model(name)
            if resolved:
                return provider, resolved

        # Fall back to default
        if self._default_provider and self._default_model:
            default = self._providers.get(self._default_provider)
            if default and default.is_available():
                logger.warning(
                    f'Unknown model "{model_name}", using default: '
                    f"{self._default_provider}/{self._default_model}"
                )
                return default, self._default_model

        raise ValueError(
            f'Model "{model_name}" not found in any available provider. '
            f"Available providers: {[p.name for p in self.get_available_providers()]}"
        )

    def list_all_models(self) -> list[dict]:
        """
        Get combined model list from all available providers.

        Returns list of model info dicts suitable for /api/tags response.
        Models are already filtered for enabled status in load_models_for_provider().
        """
        models = []
        seen = set()

        for provider in self.get_available_providers():
            for model_id, info in provider.get_models().items():
                if model_id in seen:
                    continue

                seen.add(model_id)

                # Create Ollama-compatible model entry
                models.append(
                    {
                        "name": f"{provider.name}-{model_id}",
                        "model": f"{provider.name}-{model_id}",
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
        Models are already filtered for enabled status in load_models_for_provider().
        """
        models = []
        seen = set()

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
