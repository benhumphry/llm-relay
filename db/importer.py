"""
Import configuration from YAML files into the database.

This allows migrating from the v1.x YAML-based config to the v2.x database-backed config.
YAML files serve as "factory defaults" that can be re-imported at any time.
"""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from providers.loader import (
    load_models_config,
    load_providers_config,
)

from .models import Alias, Model, Provider, Setting

logger = logging.getLogger(__name__)


def import_providers_from_yaml(db: Session, overwrite: bool = False) -> dict:
    """
    Import provider definitions from config/providers.yml.

    Args:
        db: Database session
        overwrite: If True, update existing providers. If False, skip existing.

    Returns:
        Dict with import statistics
    """
    stats = {"created": 0, "updated": 0, "skipped": 0}

    config = load_providers_config()

    # Import default settings
    if "default" in config:
        default = config["default"]
        if default.get("provider"):
            _set_setting(
                db, Setting.KEY_DEFAULT_PROVIDER, default["provider"], overwrite
            )
        if default.get("model"):
            _set_setting(db, Setting.KEY_DEFAULT_MODEL, default["model"], overwrite)

    # Import providers
    providers_config = config.get("providers", {})
    for idx, (provider_id, provider_config) in enumerate(providers_config.items()):
        existing = db.query(Provider).filter(Provider.id == provider_id).first()

        if existing and not overwrite:
            stats["skipped"] += 1
            continue

        if existing:
            # Update existing
            existing.type = provider_config.get("type", "openai-compatible")
            existing.base_url = provider_config.get("base_url")
            existing.api_key_env = provider_config.get("api_key_env")
            existing.display_order = idx
            stats["updated"] += 1
        else:
            # Create new
            provider = Provider(
                id=provider_id,
                type=provider_config.get("type", "openai-compatible"),
                base_url=provider_config.get("base_url"),
                api_key_env=provider_config.get("api_key_env"),
                display_order=idx,
                enabled=True,
            )
            db.add(provider)
            stats["created"] += 1

    db.commit()
    logger.info(f"Imported providers: {stats}")
    return stats


def import_models_from_yaml(
    db: Session, provider_id: str, overwrite: bool = False
) -> dict:
    """
    Import model definitions from config/models/{provider_id}.yml.

    Args:
        db: Database session
        provider_id: Provider ID to import models for
        overwrite: If True, update existing models. If False, skip existing.

    Returns:
        Dict with import statistics
    """
    stats = {
        "models_created": 0,
        "models_updated": 0,
        "models_skipped": 0,
        "aliases_created": 0,
        "aliases_updated": 0,
        "aliases_skipped": 0,
    }

    # Check provider exists
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        logger.warning(
            f"Provider '{provider_id}' not found in database, skipping model import"
        )
        return stats

    try:
        config = load_models_config(provider_id)
    except FileNotFoundError:
        logger.warning(f"No model config file found for provider '{provider_id}'")
        return stats

    # Import models
    models_config = config.get("models", {})
    for model_id, model_config in models_config.items():
        existing = (
            db.query(Model)
            .filter(Model.id == model_id, Model.provider_id == provider_id)
            .first()
        )

        if existing and not overwrite:
            stats["models_skipped"] += 1
            continue

        capabilities = model_config.get("capabilities", [])
        unsupported_params = model_config.get("unsupported_params", [])

        if existing:
            # Update existing
            existing.family = model_config.get("family", model_id)
            existing.description = model_config.get("description")
            existing.context_length = model_config.get("context_length", 128000)
            existing.capabilities = capabilities
            existing.unsupported_params = unsupported_params
            existing.supports_system_prompt = model_config.get(
                "supports_system_prompt", True
            )
            existing.use_max_completion_tokens = model_config.get(
                "use_max_completion_tokens", False
            )
            stats["models_updated"] += 1
        else:
            # Create new
            model = Model(
                id=model_id,
                provider_id=provider_id,
                family=model_config.get("family", model_id),
                description=model_config.get("description"),
                context_length=model_config.get("context_length", 128000),
                supports_system_prompt=model_config.get("supports_system_prompt", True),
                use_max_completion_tokens=model_config.get(
                    "use_max_completion_tokens", False
                ),
                enabled=True,
            )
            model.capabilities = capabilities
            model.unsupported_params = unsupported_params
            db.add(model)
            stats["models_created"] += 1

    # Import aliases
    aliases_config = config.get("aliases", {})
    for alias_name, model_id in aliases_config.items():
        existing = db.query(Alias).filter(Alias.alias == alias_name).first()

        if existing and not overwrite:
            stats["aliases_skipped"] += 1
            continue

        if existing:
            # Update existing
            existing.model_id = model_id
            existing.provider_id = provider_id
            stats["aliases_updated"] += 1
        else:
            # Create new
            alias = Alias(
                alias=alias_name,
                model_id=model_id,
                provider_id=provider_id,
            )
            db.add(alias)
            stats["aliases_created"] += 1

    db.commit()
    logger.info(f"Imported models for '{provider_id}': {stats}")
    return stats


def import_all_from_yaml(db: Session, overwrite: bool = False) -> dict:
    """
    Import all configuration from YAML files.

    Args:
        db: Database session
        overwrite: If True, update existing entries. If False, skip existing.

    Returns:
        Dict with complete import statistics
    """
    stats = {
        "providers": import_providers_from_yaml(db, overwrite),
        "models": {},
    }

    # Get all provider IDs from database
    providers = db.query(Provider).all()

    for provider in providers:
        model_stats = import_models_from_yaml(db, provider.id, overwrite)
        stats["models"][provider.id] = model_stats

    return stats


def _set_setting(db: Session, key: str, value: str, overwrite: bool = False) -> bool:
    """Set a setting value, optionally overwriting existing."""
    existing = db.query(Setting).filter(Setting.key == key).first()

    if existing and not overwrite:
        return False

    if existing:
        existing.value = value
    else:
        setting = Setting(key=key, value=value)
        db.add(setting)

    return True


def get_yaml_provider_ids() -> list[str]:
    """Get list of provider IDs from YAML config files."""
    config = load_providers_config()
    return list(config.get("providers", {}).keys())


def get_yaml_model_count(provider_id: str) -> tuple[int, int]:
    """Get count of models and aliases from YAML config file."""
    try:
        config = load_models_config(provider_id)
        models = len(config.get("models", {}))
        aliases = len(config.get("aliases", {}))
        return models, aliases
    except FileNotFoundError:
        return 0, 0
