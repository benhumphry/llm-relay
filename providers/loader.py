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
    """Ensure database tables exist, migrations run, and data seeded before querying."""
    from db import ensure_seeded
    from db.connection import check_db_initialized, init_db, run_migrations

    if not check_db_initialized():
        logger.info("Database not initialized, creating tables...")
        init_db()
        # Seed default providers and models on first run
        ensure_seeded()
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


def _db_model_to_model_info(model, override=None) -> ModelInfo:
    """Convert a database Model object to a ModelInfo dataclass.

    Args:
        model: The Model database object
        override: Optional ModelOverride with user-defined settings that
                  take precedence over base model values. Used for provider
                  quirks (use_max_completion_tokens, unsupported_params) and
                  pricing overrides.

    Returns:
        ModelInfo with base values overridden by any non-null override values
    """

    # Helper to get value from override if set, otherwise from model
    def get_value(field_name, default=None):
        if override is not None:
            override_val = getattr(override, field_name, None)
            if override_val is not None:
                return override_val
        model_val = getattr(model, field_name, None)
        return model_val if model_val is not None else default

    return ModelInfo(
        family=model.family,
        description=get_value("description") or model.id,
        context_length=get_value("context_length") or model.context_length,
        capabilities=get_value("capabilities") or model.capabilities or [],
        parameter_size="?B",  # Not stored in DB
        quantization_level="none",  # Not stored in DB
        unsupported_params=set(
            get_value("unsupported_params") or model.unsupported_params or []
        ),
        supports_system_prompt=get_value(
            "supports_system_prompt", model.supports_system_prompt
        ),
        use_max_completion_tokens=get_value(
            "use_max_completion_tokens", model.use_max_completion_tokens
        ),
        input_cost=get_value("input_cost"),
        output_cost=get_value("output_cost"),
        cache_read_multiplier=get_value("cache_read_multiplier"),
        cache_write_multiplier=get_value("cache_write_multiplier"),
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
        # Load all providers (including disabled - they should still appear in admin UI)
        providers = db.query(Provider).all()
        for p in providers:
            providers_dict[p.id] = {
                "type": p.type,
                "base_url": p.base_url,
                "api_key_env": p.api_key_env,
                "display_name": p.display_name,
                "source": p.source,
                "enabled": p.enabled,
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
) -> dict[str, ModelInfo]:
    """Load and parse models for a provider from database.

    Loads base model data from Model table and applies any overrides
    from ModelOverride table. Overrides persist across LiteLLM syncs
    and are used for provider quirks like use_max_completion_tokens.

    Results are cached in memory.

    Args:
        provider_name: Name of the provider

    Returns:
        Dict of model_id -> ModelInfo
    """
    global _models_cache

    # Return cached result if available
    if provider_name in _models_cache:
        return _models_cache[provider_name]

    # Ensure DB is ready before querying
    _ensure_db_ready()

    from db import Model, ModelOverride, get_db_context

    models: dict[str, ModelInfo] = {}

    with get_db_context() as db:
        # Load enabled models for this provider
        db_models = (
            db.query(Model)
            .filter(Model.provider_id == provider_name, Model.enabled == True)
            .all()
        )

        # Load overrides for this provider (keyed by model_id for fast lookup)
        overrides = {
            o.model_id: o
            for o in db.query(ModelOverride)
            .filter(ModelOverride.provider_id == provider_name)
            .all()
        }

        for m in db_models:
            # Check if there's an override for this model
            override = overrides.get(m.id)

            # Skip disabled models (via override)
            if override and override.disabled:
                continue

            models[m.id] = _db_model_to_model_info(m, override)

    if models:
        logger.debug(f"Loaded {len(models)} models for {provider_name}")

    # Cache the result
    _models_cache[provider_name] = models

    return models


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
