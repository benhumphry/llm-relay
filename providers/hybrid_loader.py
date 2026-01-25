"""
Hybrid configuration loader for providers and models.

This module provides backwards-compatible functions that work with
the new database-driven configuration system.

In the new system:
- All providers are in the Provider table
- All models are in the Model table (source: litellm, custom, ollama)
- No more separate Override tables needed
"""

import logging
import os
from typing import Any

from db import Model, Provider
from db.connection import check_db_initialized, get_db_context, init_db, run_migrations

from .base import ModelInfo
from .loader import get_provider_config, load_models_for_provider

logger = logging.getLogger(__name__)


def is_provider_active(provider_name: str) -> bool:
    """Check if a provider is active (has API key configured or doesn't need one).

    A provider is considered active if:
    - It's an Ollama provider (no API key needed)
    - It has its API key environment variable set

    Args:
        provider_name: Name of the provider

    Returns:
        True if the provider is active, False otherwise
    """
    from . import registry

    # First check if provider exists in registry
    provider = registry.get_provider(provider_name)
    if provider:
        # Ollama providers don't need API keys
        if hasattr(provider, "type") and provider.type == "ollama":
            return True
        # Check provider's api_key_env attribute
        if hasattr(provider, "api_key_env") and provider.api_key_env:
            if os.environ.get(provider.api_key_env):
                return True
            return False
        # Provider exists but has no api_key_env requirement
        return True

    # Fall back to DB config for providers not yet in registry
    config = get_provider_config(provider_name)
    if not config:
        return False

    provider_type = config.get("type", provider_name)

    # Ollama providers don't need API keys
    if provider_type == "ollama":
        return True

    # Check if API key is configured
    api_key_env = config.get("api_key_env")
    if api_key_env and os.environ.get(api_key_env):
        return True

    return False


def _ensure_db_initialized():
    """Ensure database tables exist and migrations are run before querying."""
    if not check_db_initialized():
        logger.info("Database not initialized, creating tables...")
        init_db()
    else:
        run_migrations()


def load_hybrid_models(
    provider_name: str,
) -> dict[str, ModelInfo]:
    """
    Load models for a provider from the database.

    This is now just a wrapper around load_models_for_provider since
    all models are stored in the database.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        Dict of model_id -> ModelInfo
    """
    _ensure_db_initialized()
    return load_models_for_provider(provider_name)


def get_all_models_with_metadata(
    provider_name: str,
) -> list[dict[str, Any]]:
    """
    Get all models for a provider with metadata for the admin UI.

    Returns a list of model dicts with source, enabled status, etc.
    """
    from . import registry

    _ensure_db_initialized()

    # Get provider from registry
    provider = registry.get_provider(provider_name)

    # Check if this is an Ollama provider (dynamic models from API)
    is_ollama = provider and hasattr(provider, "type") and provider.type == "ollama"

    # Check if provider has custom cost calculation
    has_custom_cost = (
        provider
        and hasattr(provider, "has_custom_cost_calculation")
        and provider.has_custom_cost_calculation
    )

    # Check if provider is enabled (toggled on in admin UI)
    provider_enabled = provider and getattr(provider, "enabled", True)

    # Check if provider is configured (has API key)
    provider_configured = is_provider_active(provider_name)

    # Provider is "active" for model usage if it's both enabled AND configured
    provider_active = provider_enabled and provider_configured

    result = []

    with get_db_context() as db:
        from db import ModelOverride

        # Load all models for this provider (including disabled)
        db_models = db.query(Model).filter(Model.provider_id == provider_name).all()

        # Load overrides for this provider (keyed by model_id)
        overrides = {
            o.model_id: o
            for o in db.query(ModelOverride)
            .filter(ModelOverride.provider_id == provider_name)
            .all()
        }

        for m in db_models:
            override = overrides.get(m.id)
            has_override = override is not None

            # Get effective values (override takes precedence)
            def get_effective(field, system_val):
                if override:
                    override_val = getattr(override, field, None)
                    if override_val is not None:
                        return override_val
                return system_val

            result.append(
                {
                    "id": m.id,
                    "provider_id": m.provider_id,
                    "source": m.source,
                    "last_synced": m.last_synced.isoformat() if m.last_synced else None,
                    "family": m.family,
                    # Effective values (with override if present)
                    "description": get_effective("description", m.description),
                    "context_length": get_effective("context_length", m.context_length),
                    "capabilities": get_effective("capabilities", m.capabilities),
                    "unsupported_params": get_effective(
                        "unsupported_params", m.unsupported_params
                    )
                    or [],
                    "supports_system_prompt": get_effective(
                        "supports_system_prompt", m.supports_system_prompt
                    ),
                    "use_max_completion_tokens": get_effective(
                        "use_max_completion_tokens", m.use_max_completion_tokens
                    ),
                    "input_cost": get_effective("input_cost", m.input_cost),
                    "output_cost": get_effective("output_cost", m.output_cost),
                    "cache_read_multiplier": get_effective(
                        "cache_read_multiplier", m.cache_read_multiplier
                    ),
                    "cache_write_multiplier": get_effective(
                        "cache_write_multiplier", m.cache_write_multiplier
                    ),
                    # System values (from base model, before override)
                    "system_description": m.description,
                    "system_context_length": m.context_length,
                    "system_capabilities": m.capabilities or [],
                    "system_unsupported_params": m.unsupported_params or [],
                    "system_supports_system_prompt": m.supports_system_prompt,
                    "system_use_max_completion_tokens": m.use_max_completion_tokens,
                    "system_input_cost": m.input_cost,
                    "system_output_cost": m.output_cost,
                    "system_cache_read_multiplier": m.cache_read_multiplier,
                    "system_cache_write_multiplier": m.cache_write_multiplier,
                    # Status flags
                    "enabled": m.enabled and not (override and override.disabled),
                    "disabled": not m.enabled or (override and override.disabled),
                    "has_override": has_override,
                    "is_system": m.source == "litellm",
                    "is_dynamic": m.source == "ollama",
                    "has_custom_cost": has_custom_cost,
                    "provider_active": provider_active,
                    "provider_enabled": provider_enabled,
                    "provider_configured": provider_configured,
                }
            )

    # For Ollama providers, also get dynamically discovered models
    if is_ollama and provider:
        try:
            dynamic_models = provider.get_models()
            existing_ids = {m["id"] for m in result}

            for model_id, model_info in dynamic_models.items():
                if model_id not in existing_ids:
                    result.append(
                        {
                            "id": model_id,
                            "provider_id": provider_name,
                            "source": "ollama",
                            "last_synced": None,
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
                            "enabled": True,
                            "disabled": False,
                            "is_system": False,
                            "is_dynamic": True,
                            "has_custom_cost": False,
                            "provider_active": provider_active,
                            "provider_enabled": provider_enabled,
                            "provider_configured": provider_configured,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to get dynamic models for {provider_name}: {e}")

    return result
