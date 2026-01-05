"""
Configuration loader for providers and models.

Loads provider and model definitions from the database.
On first run, the database is seeded from LiteLLM pricing data.
"""

import logging
from typing import Any

from .base import ModelInfo

logger = logging.getLogger(__name__)


def _ensure_db_ready():
    """Ensure database tables exist and migrations are run before querying."""
    from db.connection import check_db_initialized, init_db, run_migrations

    if not check_db_initialized():
        logger.info("Database not initialized, creating tables...")
        init_db()
    else:
        run_migrations()


# Module-level cache for loaded data
_providers_cache: dict[str, dict[str, Any]] | None = None
_models_cache: dict[str, tuple[dict[str, ModelInfo], dict[str, str]]] = {}


def clear_config_cache(provider_name: str | None = None) -> None:
    """Clear configuration caches.

    Args:
        provider_name: If specified, only clear cache for this provider.
                       If None, clear all caches.
    """
    global _providers_cache, _models_cache

    if provider_name is None:
        _providers_cache = None
        _models_cache = {}
        logger.debug("Cleared all configuration caches")
    else:
        if provider_name in _models_cache:
            del _models_cache[provider_name]
        logger.debug(f"Cleared configuration cache for {provider_name}")


def _db_model_to_model_info(model) -> ModelInfo:
    """Convert a database Model object to a ModelInfo dataclass."""
    return ModelInfo(
        family=model.family,
        description=model.description or model.id,
        context_length=model.context_length,
        capabilities=model.capabilities or [],
        parameter_size="?B",  # Not stored in DB
        quantization_level="none",  # Not stored in DB
        unsupported_params=set(model.unsupported_params or []),
        supports_system_prompt=model.supports_system_prompt,
        use_max_completion_tokens=model.use_max_completion_tokens,
        input_cost=model.input_cost,
        output_cost=model.output_cost,
        cache_read_multiplier=model.cache_read_multiplier,
        cache_write_multiplier=model.cache_write_multiplier,
    )


def load_providers_config() -> dict[str, Any]:
    """Load all providers configuration from database.

    Results are cached in memory.

    Returns:
        Dict with 'default' and 'providers' keys
    """
    global _providers_cache

    if _providers_cache is not None:
        return _providers_cache

    # Ensure DB is ready before querying
    _ensure_db_ready()

    from db import Provider, Setting, get_db_context

    providers_dict = {}
    default_provider = "anthropic"
    default_model = "claude-sonnet-4-20250514"

    with get_db_context() as db:
        # Load all providers
        providers = db.query(Provider).filter(Provider.enabled == True).all()
        for p in providers:
            providers_dict[p.id] = {
                "type": p.type,
                "base_url": p.base_url,
                "api_key_env": p.api_key_env,
                "display_name": p.display_name,
                "source": p.source,
            }

        # Load default settings
        default_provider_setting = (
            db.query(Setting)
            .filter(Setting.key == Setting.KEY_DEFAULT_PROVIDER)
            .first()
        )
        default_model_setting = (
            db.query(Setting).filter(Setting.key == Setting.KEY_DEFAULT_MODEL).first()
        )

        if default_provider_setting and default_provider_setting.value:
            default_provider = default_provider_setting.value
        if default_model_setting and default_model_setting.value:
            default_model = default_model_setting.value

    _providers_cache = {
        "default": {
            "provider": default_provider,
            "model": default_model,
        },
        "providers": providers_dict,
    }

    return _providers_cache


def load_models_for_provider(
    provider_name: str,
) -> tuple[dict[str, ModelInfo], dict[str, str]]:
    """Load and parse models and aliases for a provider from database.

    Results are cached in memory.

    Args:
        provider_name: Name of the provider

    Returns:
        Tuple of (models dict, aliases dict)
    """
    global _models_cache

    # Return cached result if available
    if provider_name in _models_cache:
        return _models_cache[provider_name]

    # Ensure DB is ready before querying
    _ensure_db_ready()

    from db import Alias, Model, get_db_context

    models: dict[str, ModelInfo] = {}
    aliases: dict[str, str] = {}

    with get_db_context() as db:
        # Load enabled models for this provider
        db_models = (
            db.query(Model)
            .filter(Model.provider_id == provider_name, Model.enabled == True)
            .all()
        )

        for m in db_models:
            models[m.id] = _db_model_to_model_info(m)

        # Load enabled aliases for this provider
        db_aliases = (
            db.query(Alias)
            .filter(Alias.provider_id == provider_name, Alias.enabled == True)
            .all()
        )

        for a in db_aliases:
            # Only include alias if target model exists
            if a.model_id in models:
                aliases[a.alias] = a.model_id
            else:
                logger.warning(
                    f"Alias '{a.alias}' points to unknown model '{a.model_id}' "
                    f"in {provider_name}, skipping"
                )

    if models:
        logger.debug(
            f"Loaded {len(models)} models and {len(aliases)} aliases for {provider_name}"
        )

    # Cache the result
    _models_cache[provider_name] = (models, aliases)

    return models, aliases


def get_provider_config(provider_name: str) -> dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider_name: Name of the provider

    Returns:
        Provider config dict with type, base_url, api_key_env, etc.
    """
    config = load_providers_config()
    providers = config.get("providers", {})

    if provider_name not in providers:
        logger.warning(f"Provider '{provider_name}' not found in configuration")
        return {}

    return providers[provider_name]


def get_default_config() -> tuple[str, str]:
    """Get the default provider and model.

    Returns:
        Tuple of (provider_name, model_id)
    """
    config = load_providers_config()
    default = config.get("default", {})

    provider = default.get("provider", "anthropic")
    model = default.get("model", "claude-sonnet-4-20250514")

    return provider, model


def get_all_provider_names() -> list[str]:
    """Get list of all configured provider names.

    Returns:
        List of provider names
    """
    config = load_providers_config()
    providers = config.get("providers", {})
    return list(providers.keys())


# Legacy functions for backwards compatibility during migration
# These can be removed once all code is updated


def load_models_config(provider_name: str) -> dict[str, Any]:
    """Legacy function - loads models config in old format.

    This is kept for backwards compatibility with code that
    expects the old YAML-style dict format.
    """
    models, aliases = load_models_for_provider(provider_name)

    # Convert ModelInfo objects back to dicts
    models_dict = {}
    for model_id, model_info in models.items():
        models_dict[model_id] = {
            "family": model_info.family,
            "description": model_info.description,
            "context_length": model_info.context_length,
            "capabilities": model_info.capabilities,
            "unsupported_params": list(model_info.unsupported_params),
            "supports_system_prompt": model_info.supports_system_prompt,
            "use_max_completion_tokens": model_info.use_max_completion_tokens,
            "input_cost": model_info.input_cost,
            "output_cost": model_info.output_cost,
            "cache_read_multiplier": model_info.cache_read_multiplier,
            "cache_write_multiplier": model_info.cache_write_multiplier,
        }

    return {
        "models": models_dict,
        "aliases": aliases,
    }


def create_model_info(model_id: str, config: dict[str, Any]) -> ModelInfo:
    """Create a ModelInfo instance from a config dict.

    This is kept for backwards compatibility with code that
    creates ModelInfo from dict configs.
    """
    context_length = config.get("context_length", 128000)

    family = config.get("family")
    if not family:
        model_lower = model_id.lower()
        if "llama" in model_lower:
            family = "llama"
        elif "qwen" in model_lower or "qwq" in model_lower:
            family = "qwen"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            family = "mistral"
        elif "gemma" in model_lower:
            family = "gemma"
        elif "deepseek" in model_lower:
            family = "deepseek"
        elif "claude" in model_lower:
            family = "claude"
        elif "gpt" in model_lower:
            family = "gpt"
        elif "phi" in model_lower:
            family = "phi"
        elif "yi" in model_lower:
            family = "yi"
        elif "command" in model_lower:
            family = "command"
        else:
            family = "other"

    description = config.get("description", model_id)
    unsupported_params = set(config.get("unsupported_params", []))
    input_cost = config.get("input_cost") or config.get("cost_per_million_input_tokens")
    output_cost = config.get("output_cost") or config.get(
        "cost_per_million_output_tokens"
    )

    return ModelInfo(
        family=family,
        description=description,
        context_length=context_length,
        capabilities=config.get("capabilities", []),
        parameter_size=config.get("parameter_size", "?B"),
        quantization_level=config.get("quantization_level", "none"),
        unsupported_params=unsupported_params,
        supports_system_prompt=config.get("supports_system_prompt", True),
        use_max_completion_tokens=config.get("use_max_completion_tokens", False),
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_multiplier=config.get("cache_read_multiplier"),
        cache_write_multiplier=config.get("cache_write_multiplier"),
    )
