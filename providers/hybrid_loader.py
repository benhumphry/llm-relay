"""
Hybrid configuration loader for providers and models.

This module provides backwards-compatible functions that work with
the new database-driven configuration system.

In the new system:
- All providers are in the Provider table
- All models are in the Model table (source: litellm, custom, ollama)
- All aliases are in the Alias table
- No more separate Override tables needed
"""

import logging
import os
from typing import Any

from db import Alias, Model, Provider
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
) -> tuple[dict[str, ModelInfo], dict[str, str]]:
    """
    Load models and aliases for a provider from the database.

    This is now just a wrapper around load_models_for_provider since
    all models are stored in the database.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        Tuple of (models dict, aliases dict)
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

    # Check if provider is active (has API key configured)
    provider_active = is_provider_active(provider_name)

    result = []

    with get_db_context() as db:
        # Load all models for this provider (including disabled)
        db_models = db.query(Model).filter(Model.provider_id == provider_name).all()

        for m in db_models:
            result.append(
                {
                    "id": m.id,
                    "provider_id": m.provider_id,
                    "source": m.source,
                    "last_synced": m.last_synced.isoformat() if m.last_synced else None,
                    "family": m.family,
                    "description": m.description,
                    "context_length": m.context_length,
                    "capabilities": m.capabilities,
                    "unsupported_params": m.unsupported_params,
                    "supports_system_prompt": m.supports_system_prompt,
                    "use_max_completion_tokens": m.use_max_completion_tokens,
                    "input_cost": m.input_cost,
                    "output_cost": m.output_cost,
                    "cache_read_multiplier": m.cache_read_multiplier,
                    "cache_write_multiplier": m.cache_write_multiplier,
                    "enabled": m.enabled,
                    "disabled": not m.enabled,  # For backwards compatibility
                    "is_system": m.source == "litellm",
                    "is_dynamic": m.source == "ollama",
                    "has_custom_cost": has_custom_cost,
                    "provider_active": provider_active,
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
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to get dynamic models for {provider_name}: {e}")

    return result


def get_all_aliases_with_metadata(
    provider_name: str,
) -> list[dict[str, Any]]:
    """
    Get all aliases for a provider with metadata for the admin UI.

    Returns a list of alias dicts with source, enabled status, etc.
    """
    from . import registry

    _ensure_db_initialized()

    # Get provider from registry
    provider = registry.get_provider(provider_name)

    # Check if provider is active
    provider_active = is_provider_active(provider_name)

    result = []

    with get_db_context() as db:
        # Load all aliases for this provider (including disabled)
        db_aliases = db.query(Alias).filter(Alias.provider_id == provider_name).all()

        for a in db_aliases:
            result.append(
                {
                    "alias": a.alias,
                    "model_id": a.model_id,
                    "provider_id": a.provider_id,
                    "source": a.source,
                    "enabled": a.enabled,
                    "disabled": not a.enabled,  # For backwards compatibility
                    "is_system": a.source == "system",
                    "provider_active": provider_active,
                }
            )

    # For Ollama providers with dynamic aliases
    if provider and hasattr(provider, "aliases") and provider.aliases:
        existing_aliases = {a["alias"] for a in result}
        for alias, model_id in provider.aliases.items():
            if alias not in existing_aliases:
                result.append(
                    {
                        "alias": alias,
                        "model_id": model_id,
                        "provider_id": provider_name,
                        "source": "dynamic",
                        "enabled": True,
                        "disabled": False,
                        "is_system": False,
                        "provider_active": provider_active,
                    }
                )

    return result


# Legacy functions for backwards compatibility
# These can be removed once all code is updated to use the new Model table directly


def get_model_overrides(provider_id: str) -> dict[str, dict]:
    """Legacy function - returns empty dict since overrides are no longer used."""
    return {}


def get_alias_overrides(provider_id: str) -> dict[str, dict]:
    """Legacy function - returns empty dict since overrides are no longer used."""
    return {}


def get_custom_models(provider_id: str) -> list[dict]:
    """Legacy function - returns models with source='custom' from Model table."""
    _ensure_db_initialized()
    with get_db_context() as db:
        models = (
            db.query(Model)
            .filter(Model.provider_id == provider_id, Model.source == "custom")
            .all()
        )
        return [m.to_dict() for m in models]


def get_custom_aliases(provider_id: str) -> list[dict]:
    """Legacy function - returns aliases with source='custom' from Alias table."""
    _ensure_db_initialized()
    with get_db_context() as db:
        aliases = (
            db.query(Alias)
            .filter(Alias.provider_id == provider_id, Alias.source == "custom")
            .all()
        )
        return [a.to_dict() for a in aliases]
