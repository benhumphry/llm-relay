"""
Configuration loader for providers and models.

Loads provider and model definitions from YAML configuration files,
allowing users to easily customize models without modifying code.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .base import ModelInfo

logger = logging.getLogger(__name__)

# Default config directory (relative to this file's parent)
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


def get_config_dir() -> Path:
    """Get the configuration directory path.

    Checks CONFIG_DIR environment variable first, then falls back to default.
    """
    config_dir = os.environ.get("CONFIG_DIR")
    if config_dir:
        return Path(config_dir)
    return DEFAULT_CONFIG_DIR


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, "r") as f:
        content = yaml.safe_load(f)
        return content if content else {}


def load_providers_config() -> dict[str, Any]:
    """Load the main providers configuration file.

    Returns:
        Dict with 'default' and 'providers' keys
    """
    config_dir = get_config_dir()
    providers_file = config_dir / "providers.yml"

    config = load_yaml_file(providers_file)

    if not config:
        logger.error(f"No providers configuration found at {providers_file}")
        return {"default": {}, "providers": {}}

    return config


def load_models_config(provider_name: str) -> dict[str, Any]:
    """Load models configuration for a specific provider.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        Dict with 'models' and 'aliases' keys
    """
    config_dir = get_config_dir()
    models_file = config_dir / "models" / f"{provider_name}.yml"

    config = load_yaml_file(models_file)

    if not config:
        logger.warning(f"No models configuration found for {provider_name}")
        return {"models": {}, "aliases": {}}

    return config


def create_model_info(model_id: str, config: dict[str, Any]) -> ModelInfo:
    """Create a ModelInfo instance from a config dict.

    Args:
        model_id: The model identifier
        config: Dict with model configuration

    Returns:
        ModelInfo instance

    Raises:
        ValueError: If required fields are missing
    """
    # Required fields
    required = ["family", "description", "context_length"]
    for field in required:
        if field not in config:
            raise ValueError(f"Model '{model_id}' missing required field: {field}")

    # Handle unsupported_params - convert list to set
    unsupported_params = set(config.get("unsupported_params", []))

    # Support both field naming conventions for costs:
    # - input_cost/output_cost (short form)
    # - cost_per_million_input_tokens/cost_per_million_output_tokens (explicit form)
    input_cost = config.get("input_cost") or config.get("cost_per_million_input_tokens")
    output_cost = config.get("output_cost") or config.get(
        "cost_per_million_output_tokens"
    )

    return ModelInfo(
        family=config["family"],
        description=config["description"],
        context_length=config["context_length"],
        capabilities=config.get("capabilities", []),
        parameter_size=config.get("parameter_size", "?B"),
        quantization_level=config.get("quantization_level", "none"),
        unsupported_params=unsupported_params,
        supports_system_prompt=config.get("supports_system_prompt", True),
        use_max_completion_tokens=config.get("use_max_completion_tokens", False),
        input_cost=input_cost,
        output_cost=output_cost,
    )


def load_models_for_provider(
    provider_name: str,
) -> tuple[dict[str, ModelInfo], dict[str, str]]:
    """Load and parse models and aliases for a provider.

    Args:
        provider_name: Name of the provider

    Returns:
        Tuple of (models dict, aliases dict)
    """
    config = load_models_config(provider_name)

    models: dict[str, ModelInfo] = {}
    aliases: dict[str, str] = {}

    # Parse models
    models_config = config.get("models", {})
    for model_id, model_config in models_config.items():
        try:
            models[model_id] = create_model_info(model_id, model_config)
        except ValueError as e:
            logger.error(f"Error loading model '{model_id}' for {provider_name}: {e}")
            continue

    # Parse aliases
    aliases = config.get("aliases", {})

    # Validate aliases point to existing models
    for alias, target in list(aliases.items()):
        if target not in models:
            logger.warning(
                f"Alias '{alias}' points to unknown model '{target}' "
                f"in {provider_name}, skipping"
            )
            del aliases[alias]

    logger.info(
        f"Loaded {len(models)} models and {len(aliases)} aliases for {provider_name}"
    )

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
    model = default.get("model", "claude-sonnet-4-5-20250929")

    return provider, model


def get_all_provider_names() -> list[str]:
    """Get list of all configured provider names.

    Returns:
        List of provider names
    """
    config = load_providers_config()
    providers = config.get("providers", {})
    return list(providers.keys())
