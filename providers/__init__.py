"""
LLM Providers package.

This module loads provider configurations from the database and registers
them with the global registry. Provider and model data is seeded from:
- Providers: Hardcoded DEFAULT_PROVIDERS list (see db/seed.py)
- Models: LiteLLM pricing data on first run, then stored in database

To add a new provider:
1. Add entry to DEFAULT_PROVIDERS in db/seed.py
2. Add provider to LITELLM_PROVIDER_MAPPING if models come from LiteLLM

OpenAI-compatible providers work automatically. Only providers with custom
SDK requirements (like Anthropic) need a Python class.
"""

import logging

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ModelInfo, OpenAICompatibleProvider, get_api_key
from .gemini_provider import GeminiProvider
from .hybrid_loader import load_hybrid_models
from .loader import (
    get_all_provider_names,
    get_default_config,
    get_provider_config,
)
from .ollama_provider import OllamaProvider
from .openrouter_provider import OpenRouterProvider
from .perplexity_provider import PerplexityProvider
from .registry import registry

logger = logging.getLogger(__name__)

# Providers with custom implementations
# - anthropic: Uses Anthropic SDK (not OpenAI-compatible)
# - ollama: Dynamic model discovery from local Ollama instance
# - openrouter: Custom implementation for dynamic cost extraction
# - perplexity: Custom cost calculation (request fees, citation tokens, etc.)
# - gemini: Custom cost calculation (tiered pricing, thinking tokens, caching, search)
CUSTOM_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
    "perplexity": PerplexityProvider,
}


def _create_provider(provider_name: str, config: dict) -> LLMProvider | None:
    """Create a provider instance from config.

    Args:
        provider_name: Name of the provider
        config: Provider configuration dict

    Returns:
        Provider instance or None if creation failed
    """
    provider_type = config.get("type", provider_name)

    # Special handling for Ollama - models are discovered dynamically
    if provider_type == "ollama":
        base_url = config.get("base_url", "http://localhost:11434")
        # Load only aliases from YAML, models come from Ollama API
        _, aliases = load_hybrid_models(provider_name)
        return OllamaProvider(name=provider_name, base_url=base_url, aliases=aliases)

    # Load models and aliases using hybrid loader (YAML + DB overrides)
    models, aliases = load_hybrid_models(provider_name)

    # Note: We no longer skip providers with no enabled models.
    # This allows providers to appear in the admin UI even when all models are disabled.
    if not models:
        logger.info(
            f"No enabled models for provider '{provider_name}' (all may be disabled)"
        )

    # Check if this provider needs a custom class (non-OpenAI SDK)
    if provider_name in CUSTOM_PROVIDER_CLASSES:
        provider_class = CUSTOM_PROVIDER_CLASSES[provider_name]
        return provider_class(models=models, aliases=aliases)

    # OpenAI-compatible providers are created directly from config
    if provider_type == "openai-compatible":
        base_url = config.get("base_url")
        api_key_env = config.get("api_key_env")

        if not base_url or not api_key_env:
            logger.error(f"Provider '{provider_name}' missing base_url or api_key_env")
            return None

        return OpenAICompatibleProvider(
            name=provider_name,
            base_url=base_url,
            api_key_env=api_key_env,
            models=models,
            aliases=aliases,
        )

    logger.error(f"Unknown provider type '{provider_type}' for '{provider_name}'")
    return None


def _register_db_ollama_instances():
    """Load and register Ollama instances from database."""
    from db import OllamaInstance
    from db.connection import check_db_initialized, get_db_context, init_db

    # Ensure DB is initialized
    if not check_db_initialized():
        init_db()

    try:
        with get_db_context() as db:
            instances = (
                db.query(OllamaInstance).filter(OllamaInstance.enabled == True).all()
            )

            for inst in instances:
                # Skip if already registered (e.g., from YAML)
                if registry.get_provider(inst.name):
                    logger.debug(
                        f"Ollama instance '{inst.name}' already registered, skipping DB entry"
                    )
                    continue

                provider = OllamaProvider(name=inst.name, base_url=inst.base_url)
                registry.register(provider)
                logger.info(
                    f"Registered Ollama instance from DB: {inst.name} ({inst.base_url})"
                )

    except Exception as e:
        logger.warning(f"Failed to load Ollama instances from DB: {e}")


def _register_providers():
    """Load and register all providers from config."""
    provider_names = get_all_provider_names()

    for provider_name in provider_names:
        config = get_provider_config(provider_name)
        if not config:
            continue

        provider = _create_provider(provider_name, config)
        if provider:
            registry.register(provider)
            logger.info(f"Registered provider: {provider_name}")

    # Also load Ollama instances from database
    _register_db_ollama_instances()

    # Set default provider/model from config
    default_provider, default_model = get_default_config()
    registry.set_default(default_provider, default_model)
    logger.info(f"Default: {default_provider}/{default_model}")


# Register all providers on module import
_register_providers()

__all__ = [
    "registry",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ModelInfo",
    "get_api_key",
    "AnthropicProvider",
    "OllamaProvider",
    "OpenRouterProvider",
]
