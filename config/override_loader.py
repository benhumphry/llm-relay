"""
Model Override Loader

Loads model overrides from YAML configuration and applies them to the database.
This ensures provider quirks and deprecated model settings persist across
LiteLLM pricing syncs.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the overrides YAML file
OVERRIDES_FILE = Path(__file__).parent / "model_overrides.yaml"


def load_overrides_config() -> dict[str, Any]:
    """Load the model overrides YAML configuration.

    Returns:
        Dict with provider overrides configuration
    """
    if not OVERRIDES_FILE.exists():
        logger.warning(f"Model overrides file not found: {OVERRIDES_FILE}")
        return {"providers": {}}

    with open(OVERRIDES_FILE, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


def get_model_overrides(provider_id: str, model_id: str) -> dict[str, Any] | None:
    """Get overrides for a specific model from the YAML config.

    Checks in order:
    1. Exact model match in provider.models
    2. Pattern match in provider.patterns
    3. Provider defaults

    Args:
        provider_id: The provider ID (e.g., 'openai')
        model_id: The model ID (e.g., 'gpt-5-mini')

    Returns:
        Dict of override settings, or None if no overrides apply
    """
    config = load_overrides_config()
    providers = config.get("providers", {})

    if provider_id not in providers:
        return None

    provider_config = providers[provider_id]
    overrides = {}

    # 1. Apply provider defaults first
    defaults = provider_config.get("defaults", {})
    if defaults:
        overrides.update(defaults)

    # 2. Check pattern matches
    patterns = provider_config.get("patterns", [])
    for pattern in patterns:
        match_pattern = pattern.get("match", "")
        if fnmatch.fnmatch(model_id, match_pattern):
            # Apply pattern overrides (excluding 'match' key)
            for key, value in pattern.items():
                if key != "match":
                    overrides[key] = value

    # 3. Check exact model match (highest priority)
    models = provider_config.get("models", {})
    if model_id in models:
        model_overrides = models[model_id]
        if model_overrides:
            overrides.update(model_overrides)

    return overrides if overrides else None


def apply_yaml_overrides_to_db() -> dict[str, int]:
    """Apply YAML overrides to the ModelOverride database table.

    This function:
    1. Loads all models from the database
    2. Checks each against the YAML overrides
    3. Creates/updates ModelOverride records as needed

    Returns:
        Dict with counts: {'created': N, 'updated': N, 'skipped': N}
    """
    from db import Model, ModelOverride, get_db_context

    config = load_overrides_config()
    providers = config.get("providers", {})

    stats = {"created": 0, "updated": 0, "skipped": 0}

    with get_db_context() as db:
        # Process each provider in the config
        for provider_id, provider_config in providers.items():
            # Get all models for this provider
            models = db.query(Model).filter(Model.provider_id == provider_id).all()

            for model in models:
                overrides = get_model_overrides(provider_id, model.id)

                if not overrides:
                    stats["skipped"] += 1
                    continue

                # Check if override already exists
                existing = (
                    db.query(ModelOverride)
                    .filter(
                        ModelOverride.provider_id == provider_id,
                        ModelOverride.model_id == model.id,
                    )
                    .first()
                )

                if existing:
                    # Update existing override
                    changed = False
                    if (
                        "disabled" in overrides
                        and existing.disabled != overrides["disabled"]
                    ):
                        existing.disabled = overrides["disabled"]
                        changed = True
                    if (
                        "use_max_completion_tokens" in overrides
                        and existing.use_max_completion_tokens
                        != overrides["use_max_completion_tokens"]
                    ):
                        existing.use_max_completion_tokens = overrides[
                            "use_max_completion_tokens"
                        ]
                        changed = True
                    if (
                        "supports_system_prompt" in overrides
                        and existing.supports_system_prompt
                        != overrides["supports_system_prompt"]
                    ):
                        existing.supports_system_prompt = overrides[
                            "supports_system_prompt"
                        ]
                        changed = True
                    if "unsupported_params" in overrides:
                        new_params = overrides["unsupported_params"]
                        if existing.unsupported_params != new_params:
                            existing.unsupported_params = new_params
                            changed = True

                    if changed:
                        stats["updated"] += 1
                    else:
                        stats["skipped"] += 1
                else:
                    # Create new override
                    new_override = ModelOverride(
                        provider_id=provider_id,
                        model_id=model.id,
                        disabled=overrides.get("disabled", False),
                        use_max_completion_tokens=overrides.get(
                            "use_max_completion_tokens"
                        ),
                        supports_system_prompt=overrides.get("supports_system_prompt"),
                    )
                    if "unsupported_params" in overrides:
                        new_override.unsupported_params = overrides[
                            "unsupported_params"
                        ]

                    db.add(new_override)
                    stats["created"] += 1

        db.commit()

    logger.info(
        f"Applied YAML overrides: {stats['created']} created, "
        f"{stats['updated']} updated, {stats['skipped']} skipped"
    )

    return stats


def get_all_disabled_models() -> list[tuple[str, str, str]]:
    """Get list of all disabled models from the YAML config.

    Returns:
        List of (provider_id, model_id, reason) tuples
    """
    config = load_overrides_config()
    providers = config.get("providers", {})
    disabled = []

    for provider_id, provider_config in providers.items():
        models = provider_config.get("models", {})
        for model_id, model_config in models.items():
            if model_config and model_config.get("disabled"):
                reason = model_config.get("reason", "No reason provided")
                disabled.append((provider_id, model_id, reason))

    return disabled


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    print("Testing override loader...\n")

    # Test getting overrides for specific models
    test_cases = [
        ("openai", "gpt-5-mini"),
        ("openai", "o1-preview"),
        ("openai", "gpt-4o"),
        ("perplexity", "sonar-reasoning"),
        ("perplexity", "sonar"),
        ("deepseek", "deepseek-r1"),
    ]

    for provider, model in test_cases:
        overrides = get_model_overrides(provider, model)
        print(f"{provider}/{model}: {overrides}")

    print("\n\nDisabled models:")
    for provider, model, reason in get_all_disabled_models():
        print(f"  {provider}/{model}: {reason}")
