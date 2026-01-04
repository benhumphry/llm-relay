"""
Hybrid configuration loader for providers and models.

Merges system models (from YAML) with database overrides and custom models.
This allows:
- System models to auto-update with releases
- Users to disable specific system models
- Users to create custom models that persist across updates
"""

import logging
from typing import Any

from db import AliasOverride, CustomAlias, CustomModel, ModelOverride
from db.connection import check_db_initialized, get_db_context, init_db

from .base import ModelInfo
from .loader import load_models_for_provider as load_yaml_models

logger = logging.getLogger(__name__)


def _ensure_db_initialized():
    """Ensure database tables exist before querying."""
    if not check_db_initialized():
        logger.info("Database not initialized, creating tables...")
        init_db()


def get_model_overrides(provider_id: str) -> dict[str, dict]:
    """Get all model overrides for a provider, keyed by model_id.

    Returns dict of model_id -> {"disabled": bool} to avoid SQLAlchemy session issues.
    """
    _ensure_db_initialized()
    with get_db_context() as db:
        overrides = (
            db.query(ModelOverride)
            .filter(ModelOverride.provider_id == provider_id)
            .all()
        )
        # Return dicts instead of ORM objects to avoid detached session issues
        return {o.model_id: {"disabled": o.disabled} for o in overrides}


def get_alias_overrides(provider_id: str) -> dict[str, dict]:
    """Get all alias overrides for a provider, keyed by alias.

    Returns dict of alias -> {"disabled": bool} to avoid SQLAlchemy session issues.
    """
    _ensure_db_initialized()
    with get_db_context() as db:
        overrides = (
            db.query(AliasOverride)
            .filter(AliasOverride.provider_id == provider_id)
            .all()
        )
        # Return dicts instead of ORM objects to avoid detached session issues
        return {o.alias: {"disabled": o.disabled} for o in overrides}


def get_custom_models(provider_id: str) -> list[dict]:
    """Get all custom models for a provider as dicts (to avoid detached session issues)."""
    _ensure_db_initialized()
    with get_db_context() as db:
        models = (
            db.query(CustomModel).filter(CustomModel.provider_id == provider_id).all()
        )
        # Convert to dicts while still in session
        return [
            {
                "id": m.id,
                "provider_id": m.provider_id,
                "model_id": m.model_id,
                "family": m.family,
                "description": m.description,
                "context_length": m.context_length,
                "capabilities": m.capabilities,
                "unsupported_params": m.unsupported_params,
                "supports_system_prompt": m.supports_system_prompt,
                "use_max_completion_tokens": m.use_max_completion_tokens,
                "enabled": m.enabled,
            }
            for m in models
        ]


def get_custom_aliases(provider_id: str) -> list[dict]:
    """Get all custom aliases for a provider as dicts (to avoid detached session issues)."""
    _ensure_db_initialized()
    with get_db_context() as db:
        aliases = (
            db.query(CustomAlias).filter(CustomAlias.provider_id == provider_id).all()
        )
        # Convert to dicts while still in session
        return [
            {
                "id": a.id,
                "provider_id": a.provider_id,
                "alias": a.alias,
                "model_id": a.model_id,
            }
            for a in aliases
        ]


def custom_model_to_model_info(model: dict) -> ModelInfo:
    """Convert a CustomModel dict to a ModelInfo dataclass."""
    return ModelInfo(
        family=model["family"],
        description=model["description"] or "",
        context_length=model["context_length"],
        capabilities=model["capabilities"],
        parameter_size="?B",
        quantization_level="none",
        unsupported_params=set(model["unsupported_params"] or []),
        supports_system_prompt=model["supports_system_prompt"],
        use_max_completion_tokens=model["use_max_completion_tokens"],
    )


def load_hybrid_models(
    provider_name: str,
) -> tuple[dict[str, ModelInfo], dict[str, str]]:
    """
    Load models and aliases by merging YAML config with DB overrides and custom models.

    Flow:
    1. Load system models from YAML
    2. Apply model overrides (disable specific system models)
    3. Add custom models from database
    4. Load system aliases from YAML
    5. Apply alias overrides (disable specific system aliases)
    6. Add custom aliases from database

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        Tuple of (models dict, aliases dict)
    """
    # Step 1: Load from YAML
    yaml_models, yaml_aliases = load_yaml_models(provider_name)

    # Step 2: Apply model overrides
    model_overrides = get_model_overrides(provider_name)
    for model_id, override in model_overrides.items():
        if override["disabled"] and model_id in yaml_models:
            logger.debug(f"Disabling system model {provider_name}/{model_id}")
            del yaml_models[model_id]

    # Step 3: Add custom models (now dicts from get_custom_models)
    custom_models = get_custom_models(provider_name)
    for custom in custom_models:
        if custom["enabled"]:
            yaml_models[custom["model_id"]] = custom_model_to_model_info(custom)
            logger.debug(f"Added custom model {provider_name}/{custom['model_id']}")

    # Step 4: Aliases are already in yaml_aliases

    # Step 5: Apply alias overrides
    alias_overrides = get_alias_overrides(provider_name)
    for alias, override in alias_overrides.items():
        if override["disabled"] and alias in yaml_aliases:
            logger.debug(f"Disabling system alias {provider_name}/{alias}")
            del yaml_aliases[alias]

    # Step 6: Add custom aliases (now dicts from get_custom_aliases)
    custom_aliases = get_custom_aliases(provider_name)
    for custom in custom_aliases:
        # Only add if target model exists
        if custom["model_id"] in yaml_models:
            yaml_aliases[custom["alias"]] = custom["model_id"]
            logger.debug(f"Added custom alias {provider_name}/{custom['alias']}")
        else:
            logger.warning(
                f"Custom alias {custom['alias']} points to non-existent model "
                f"{custom['model_id']}, skipping"
            )

    logger.info(
        f"Hybrid loaded {len(yaml_models)} models and {len(yaml_aliases)} aliases "
        f"for {provider_name}"
    )

    return yaml_models, yaml_aliases


def get_all_models_with_metadata(
    provider_name: str,
) -> list[dict[str, Any]]:
    """
    Get all models for a provider with system/custom metadata for the admin UI.

    Returns a list of model dicts with 'is_system' and 'disabled' flags.
    """
    from . import registry
    from .loader import load_providers_config

    yaml_models = {}
    model_source = (
        "system"  # "system" (YAML), "dynamic" (API-discovered), or "custom" (DB)
    )

    # Get provider from registry
    provider = registry.get_provider(provider_name)

    # Check if this is an Ollama provider (dynamic models from API, not YAML)
    is_ollama = provider and hasattr(provider, "type") and provider.type == "ollama"

    if is_ollama:
        # For Ollama providers, get models dynamically from the registry
        model_source = "dynamic"
        try:
            yaml_models = provider.get_models()
        except Exception as e:
            logger.warning(f"Failed to get dynamic models for {provider_name}: {e}")
    else:
        # Check if this provider exists in YAML config
        providers_config = load_providers_config()
        yaml_providers = providers_config.get("providers", {})

        if provider_name in yaml_providers:
            # Load from YAML (system provider)
            model_source = "system"
            yaml_models, _ = load_yaml_models(provider_name)
        else:
            # Custom provider without YAML - get models from registry if available
            model_source = "dynamic"
            if provider:
                try:
                    yaml_models = provider.get_models()
                except Exception as e:
                    logger.debug(
                        f"No models available for custom provider {provider_name}: {e}"
                    )

    # Get overrides
    model_overrides = get_model_overrides(provider_name)

    # Get custom models
    custom_models = get_custom_models(provider_name)

    result = []

    # Add models from YAML or dynamic discovery
    for model_id, model_info in yaml_models.items():
        override = model_overrides.get(model_id)
        disabled = override["disabled"] if override else False

        result.append(
            {
                "id": model_id,
                "provider_id": provider_name,
                "family": model_info.family,
                "description": model_info.description,
                "context_length": model_info.context_length,
                "capabilities": model_info.capabilities,
                "unsupported_params": list(model_info.unsupported_params),
                "supports_system_prompt": model_info.supports_system_prompt,
                "use_max_completion_tokens": model_info.use_max_completion_tokens,
                "input_cost": getattr(model_info, "input_cost", None),
                "output_cost": getattr(model_info, "output_cost", None),
                "source": model_source,
                "is_system": model_source == "system",
                "is_dynamic": model_source == "dynamic",
                "disabled": disabled,
                "enabled": not disabled,
            }
        )

    # Add custom models (now dicts from get_custom_models)
    for custom in custom_models:
        result.append(
            {
                "id": custom["model_id"],
                "db_id": custom["id"],  # Database ID for edit/delete
                "provider_id": provider_name,
                "family": custom["family"],
                "description": custom["description"],
                "context_length": custom["context_length"],
                "capabilities": custom["capabilities"],
                "unsupported_params": custom["unsupported_params"],
                "supports_system_prompt": custom["supports_system_prompt"],
                "use_max_completion_tokens": custom["use_max_completion_tokens"],
                "input_cost": custom.get("input_cost"),
                "output_cost": custom.get("output_cost"),
                "source": "custom",
                "is_system": False,
                "is_dynamic": False,
                "disabled": not custom["enabled"],
                "enabled": custom["enabled"],
            }
        )

    return result


def get_all_aliases_with_metadata(
    provider_name: str,
) -> list[dict[str, Any]]:
    """
    Get all aliases for a provider with system/custom metadata for the admin UI.

    Returns a list of alias dicts with 'is_system' and 'disabled' flags.
    """
    from . import registry
    from .loader import load_providers_config

    yaml_aliases = {}

    # Get provider from registry
    provider = registry.get_provider(provider_name)

    # Check if this is an Ollama provider (dynamic, no YAML aliases)
    is_ollama = provider and hasattr(provider, "type") and provider.type == "ollama"

    if is_ollama:
        # Ollama providers get aliases from the provider if available
        if provider and hasattr(provider, "aliases"):
            yaml_aliases = provider.aliases or {}
    else:
        # Check if this provider exists in YAML config
        providers_config = load_providers_config()
        yaml_providers = providers_config.get("providers", {})

        if provider_name in yaml_providers:
            # Load from YAML (system provider)
            _, yaml_aliases = load_yaml_models(provider_name)
        else:
            # Custom provider without YAML - get aliases from registry if available
            if provider and hasattr(provider, "aliases"):
                yaml_aliases = provider.aliases or {}

    # Get overrides
    alias_overrides = get_alias_overrides(provider_name)

    # Get custom aliases
    custom_aliases = get_custom_aliases(provider_name)

    result = []

    # Add system aliases
    for alias, model_id in yaml_aliases.items():
        override = alias_overrides.get(alias)
        disabled = override["disabled"] if override else False

        result.append(
            {
                "alias": alias,
                "model_id": model_id,
                "provider_id": provider_name,
                "is_system": True,
                "disabled": disabled,
            }
        )

    # Add custom aliases (now dicts from get_custom_aliases)
    for custom in custom_aliases:
        result.append(
            {
                "alias": custom["alias"],
                "model_id": custom["model_id"],
                "provider_id": provider_name,
                "db_id": custom["id"],  # Database ID for edit/delete
                "is_system": False,
                "disabled": False,  # Custom aliases don't have a disabled state
            }
        )

    return result
