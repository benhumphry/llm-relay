#!/usr/bin/env python3
"""
Sync model pricing from LiteLLM's model_prices_and_context_window.json.

This script fetches the latest pricing data from LiteLLM's GitHub repository
and compares it with our database configuration, generating a report of
differences and optionally applying updates.

Usage:
    python scripts/sync_pricing.py [--update] [--add-new] [--provider PROVIDER]

Options:
    --update            Apply price changes to database (default: report only)
    --add-new           Add new models found in LiteLLM
    --provider NAME     Only sync specific provider (e.g., openai, anthropic)
    --output FORMAT     Output format: text, json (default: text)
    --verbose           Show detailed comparison info
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# LiteLLM pricing data URL
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Map LiteLLM provider names to our provider names
# Note: We only map direct provider prefixes, not hosted versions (vertex_ai, bedrock, azure)
PROVIDER_MAPPING = {
    # First-party providers
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "groq": "groq",
    "mistral": "mistral",
    "xai": "xai",
    "perplexity": "perplexity",
    # Inference providers
    "fireworks_ai": "fireworks",
    "deepinfra": "deepinfra",
    "together_ai": "together",
    "cerebras": "cerebras",
    "sambanova": "sambanova",
    "cohere": "cohere",
    # Meta-providers (pricing is dynamic from API, but models are seeded for discovery)
    "openrouter": "openrouter",
}

# Map LiteLLM capability flags to our capability names
CAPABILITY_MAPPING = {
    "supports_vision": "vision",
    "supports_function_calling": "tools",
    "supports_reasoning": "reasoning",
    "supports_web_search": "search",
    "supports_audio_input": "audio",
    "supports_pdf_input": "pdf",
}

# Model names to filter out (likely errors in LiteLLM data)
INVALID_MODEL_NAMES = {
    "container",  # Not a real model
    "test",
    "sample",
}

# Patterns for models that should be excluded from sync suggestions
# These are typically experimental, preview, or special-purpose models
EXCLUDE_MODEL_PATTERNS = [
    # Dated preview/experimental versions (keep only the latest non-dated version)
    r"-\d{2}-\d{2}$",  # Matches -MM-DD suffix (e.g., -04-17, -09-2025)
    r"-\d{4}$",  # Matches -MMDD suffix (e.g., -0827, -1206)
    r"-\d{6}$",  # Matches -YYMMDD suffix
    r"-\d{3}$",  # Matches -001, -002 version suffixes
    # Experimental models
    r"-exp-\d+",  # Matches -exp-0827, -exp-1114, etc.
    r"-exp$",  # Matches -exp suffix
    # Preview versions with dates
    r"-preview-\d",  # Matches -preview-02-05, -preview-09-2025, etc.
    # Special purpose models that don't work with standard chat API
    r"-live-",  # Live/streaming models
    r"-live$",
    r"-tts",  # Text-to-speech models
    r"-native-audio",  # Audio-specific models
    r"-image-generation",  # Image generation models
    r"-computer-use",  # Computer use models (special API)
    # Legacy/deprecated patterns
    r"-vision$",  # Old vision-specific models (modern models have vision built-in)
    r"-latest$",  # Alias models that point to other versions
]

# Models to always include even if they match exclude patterns
INCLUDE_MODEL_OVERRIDES = {
    # Current flagship models that happen to have dates
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
}


def should_exclude_model(model_id: str) -> bool:
    """Check if a model should be excluded from sync suggestions.

    Returns True if the model matches exclusion patterns and is not
    in the override list.
    """
    import re

    # Check if model is in override list (always include)
    if model_id in INCLUDE_MODEL_OVERRIDES:
        return False

    # Check against exclusion patterns
    for pattern in EXCLUDE_MODEL_PATTERNS:
        if re.search(pattern, model_id):
            return True

    return False


def has_valid_pricing(model: "ModelPricing") -> bool:
    """Check if a model has valid pricing data.

    Models without pricing are typically experimental or not yet available.
    """
    return model.input_cost is not None and model.output_cost is not None


@dataclass
class ModelPricing:
    """Represents pricing data for a model."""

    model_id: str
    provider: str
    input_cost: float | None = None  # per 1M tokens
    output_cost: float | None = None  # per 1M tokens
    cache_read_multiplier: float | None = None
    cache_write_multiplier: float | None = None
    context_length: int | None = None
    capabilities: list[str] = field(default_factory=list)
    source: str = "litellm"

    # Additional fields from LiteLLM
    input_cost_above_128k: float | None = None  # Tiered pricing
    output_cost_above_128k: float | None = None
    search_context_costs: dict | None = None  # Perplexity


@dataclass
class PricingDiff:
    """Represents a difference between local and remote pricing."""

    model_id: str
    provider: str
    field: str
    local_value: Any
    remote_value: Any
    change_type: str  # "added", "removed", "changed"


def fetch_litellm_pricing() -> dict:
    """Fetch pricing data from LiteLLM GitHub."""
    logger.info(f"Fetching pricing data from LiteLLM...")
    response = requests.get(LITELLM_PRICING_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    logger.info(f"Fetched {len(data)} model entries")
    return data


def parse_litellm_model(key: str, data: dict) -> ModelPricing | None:
    """Parse a LiteLLM model entry into our format."""
    # Skip non-chat models
    if data.get("mode") not in ("chat", None):
        return None

    # Extract provider from key or litellm_provider field
    litellm_provider = data.get("litellm_provider", "")

    # Filter out invalid model names
    model_name = key.split("/")[-1] if "/" in key else key
    if model_name.lower() in INVALID_MODEL_NAMES:
        return None

    # Handle prefixed keys like "xai/grok-4" or "groq/llama-3.3-70b"
    if "/" in key:
        prefix = key.split("/")[0]
        model_id = "/".join(key.split("/")[1:])
        # Use prefix as provider hint
        if prefix in PROVIDER_MAPPING:
            provider = PROVIDER_MAPPING[prefix]
        elif litellm_provider in PROVIDER_MAPPING:
            provider = PROVIDER_MAPPING[litellm_provider]
        else:
            return None  # Unknown provider
    else:
        if litellm_provider in PROVIDER_MAPPING:
            provider = PROVIDER_MAPPING[litellm_provider]
        else:
            return None
        model_id = key

    # Convert per-token costs to per-million-token costs
    input_cost = None
    output_cost = None
    if data.get("input_cost_per_token"):
        input_cost = round(data["input_cost_per_token"] * 1_000_000, 4)
    if data.get("output_cost_per_token"):
        output_cost = round(data["output_cost_per_token"] * 1_000_000, 4)

    # Calculate cache multipliers from absolute costs
    cache_read_multiplier = None
    cache_write_multiplier = None

    if input_cost and data.get("cache_read_input_token_cost"):
        cache_read_cost = data["cache_read_input_token_cost"] * 1_000_000
        cache_read_multiplier = round(cache_read_cost / input_cost, 2)

    if input_cost and data.get("cache_creation_input_token_cost"):
        cache_write_cost = data["cache_creation_input_token_cost"] * 1_000_000
        if cache_write_cost > 0:  # Some providers have 0 (free cache creation)
            cache_write_multiplier = round(cache_write_cost / input_cost, 2)

    # Extract context length
    context_length = None
    if data.get("max_input_tokens"):
        context_length = int(data["max_input_tokens"])

    # Map capabilities
    capabilities = []
    for litellm_cap, our_cap in CAPABILITY_MAPPING.items():
        if data.get(litellm_cap):
            capabilities.append(our_cap)

    # Handle tiered pricing (e.g., xAI above 128k)
    input_cost_above_128k = None
    output_cost_above_128k = None
    if data.get("input_cost_per_token_above_128k_tokens"):
        input_cost_above_128k = round(
            data["input_cost_per_token_above_128k_tokens"] * 1_000_000, 4
        )
    if data.get("output_cost_per_token_above_128k_tokens"):
        output_cost_above_128k = round(
            data["output_cost_per_token_above_128k_tokens"] * 1_000_000, 4
        )

    # Handle Perplexity search context costs
    search_context_costs = data.get("search_context_cost_per_query")

    return ModelPricing(
        model_id=model_id,
        provider=provider,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_multiplier=cache_read_multiplier,
        cache_write_multiplier=cache_write_multiplier,
        context_length=context_length,
        capabilities=capabilities,
        input_cost_above_128k=input_cost_above_128k,
        output_cost_above_128k=output_cost_above_128k,
        search_context_costs=search_context_costs,
    )


def get_model_overrides_for_provider(provider: str) -> dict[str, dict]:
    """Get all model overrides from database for a provider.

    Returns dict of model_id -> override fields dict.
    This function is safe to call even if DB is not initialized.
    """
    try:
        # Import here to avoid circular imports and allow CLI usage without DB
        import sys

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from db import ModelOverride
        from db.connection import check_db_initialized, get_db_context, init_db

        if not check_db_initialized():
            init_db()

        with get_db_context() as db:
            overrides = (
                db.query(ModelOverride)
                .filter(ModelOverride.provider_id == provider)
                .all()
            )
            return {
                o.model_id: {
                    "input_cost": o.input_cost,
                    "output_cost": o.output_cost,
                    "cache_read_multiplier": o.cache_read_multiplier,
                    "cache_write_multiplier": o.cache_write_multiplier,
                    "context_length": o.context_length,
                }
                for o in overrides
            }
    except Exception as e:
        logger.debug(f"Could not load DB overrides for {provider}: {e}")
        return {}


def load_effective_pricing(provider: str) -> dict:
    """Load effective pricing from database.

    Returns models with their current pricing and source information.
    """
    try:
        import sys

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from db import Model
        from db.connection import check_db_initialized, get_db_context, init_db

        if not check_db_initialized():
            init_db()

        models = {}
        with get_db_context() as db:
            db_models = (
                db.query(Model)
                .filter(Model.provider_id == provider, Model.enabled == True)
                .all()
            )
            for m in db_models:
                models[m.id] = {
                    "input_cost": m.input_cost,
                    "output_cost": m.output_cost,
                    "cache_read_multiplier": m.cache_read_multiplier,
                    "cache_write_multiplier": m.cache_write_multiplier,
                    "context_length": m.context_length,
                    "source": m.source,  # "litellm", "custom", "ollama"
                }

        return {"models": models}
    except Exception as e:
        logger.warning(f"Could not load models for {provider}: {e}")
        return {"models": {}}


def compare_pricing(
    local: dict, remote: list[ModelPricing], provider: str
) -> list[PricingDiff]:
    """Compare local database config with remote LiteLLM data."""
    diffs = []
    local_models = local.get("models", {})

    # Build lookup by model_id
    remote_by_id = {m.model_id: m for m in remote if m.provider == provider}

    # Check for changes in existing models
    for model_id, local_data in local_models.items():
        if model_id in remote_by_id:
            remote_model = remote_by_id[model_id]

            # Compare input_cost
            local_input = local_data.get("input_cost")
            if remote_model.input_cost and local_input != remote_model.input_cost:
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="input_cost",
                        local_value=local_input,
                        remote_value=remote_model.input_cost,
                        change_type="changed",
                    )
                )

            # Compare output_cost
            local_output = local_data.get("output_cost")
            if remote_model.output_cost and local_output != remote_model.output_cost:
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="output_cost",
                        local_value=local_output,
                        remote_value=remote_model.output_cost,
                        change_type="changed",
                    )
                )

            # Compare cache_read_multiplier
            local_cache_read = local_data.get("cache_read_multiplier")
            if (
                remote_model.cache_read_multiplier
                and local_cache_read != remote_model.cache_read_multiplier
            ):
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="cache_read_multiplier",
                        local_value=local_cache_read,
                        remote_value=remote_model.cache_read_multiplier,
                        change_type="changed" if local_cache_read else "added",
                    )
                )

            # Compare cache_write_multiplier
            local_cache_write = local_data.get("cache_write_multiplier")
            if (
                remote_model.cache_write_multiplier
                and local_cache_write != remote_model.cache_write_multiplier
            ):
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="cache_write_multiplier",
                        local_value=local_cache_write,
                        remote_value=remote_model.cache_write_multiplier,
                        change_type="changed" if local_cache_write else "added",
                    )
                )

            # Compare context_length
            local_context = local_data.get("context_length")
            if (
                remote_model.context_length
                and local_context != remote_model.context_length
            ):
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="context_length",
                        local_value=local_context,
                        remote_value=remote_model.context_length,
                        change_type="changed",
                    )
                )

            # Note tiered pricing if present
            if remote_model.input_cost_above_128k:
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="input_cost_above_128k",
                        local_value=None,
                        remote_value=remote_model.input_cost_above_128k,
                        change_type="info",
                    )
                )

    # Check for new models in remote not in local
    for model_id, remote_model in remote_by_id.items():
        if model_id not in local_models:
            # Skip models without valid pricing
            if not has_valid_pricing(remote_model):
                continue

            # Skip models matching exclusion patterns
            if should_exclude_model(model_id):
                continue

            diffs.append(
                PricingDiff(
                    model_id=model_id,
                    provider=provider,
                    field="(new model)",
                    local_value=None,
                    remote_value=f"${remote_model.input_cost} in / ${remote_model.output_cost} out",
                    change_type="added",
                )
            )

    # Check for models in local that are not in remote (deprecated/removed)
    for model_id, local_data in local_models.items():
        if model_id not in remote_by_id:
            # Only flag models that came from LiteLLM originally
            # (source == "litellm"), not custom models
            source = local_data.get("source", "litellm")
            if source == "litellm":
                input_cost = local_data.get("input_cost")
                output_cost = local_data.get("output_cost")
                diffs.append(
                    PricingDiff(
                        model_id=model_id,
                        provider=provider,
                        field="(deprecated)",
                        local_value=f"${input_cost} in / ${output_cost} out",
                        remote_value=None,
                        change_type="removed",
                    )
                )

    return diffs


def format_diff_report(diffs: list[PricingDiff], verbose: bool = False) -> str:
    """Format diff report for console output."""
    if not diffs:
        return "No pricing differences found."

    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"PRICING DIFFERENCES FOUND: {len(diffs)}")
    lines.append(f"{'=' * 60}\n")

    # Group by provider
    by_provider: dict[str, list[PricingDiff]] = {}
    for diff in diffs:
        by_provider.setdefault(diff.provider, []).append(diff)

    for provider, provider_diffs in sorted(by_provider.items()):
        lines.append(f"\n## {provider.upper()}")
        lines.append("-" * 40)

        # Group by model
        by_model: dict[str, list[PricingDiff]] = {}
        for diff in provider_diffs:
            by_model.setdefault(diff.model_id, []).append(diff)

        for model_id, model_diffs in sorted(by_model.items()):
            lines.append(f"\n  {model_id}:")
            for diff in model_diffs:
                if diff.change_type == "added" and diff.field == "(new model)":
                    lines.append(f"    + NEW MODEL: {diff.remote_value}")
                elif diff.change_type == "added":
                    lines.append(f"    + {diff.field}: {diff.remote_value}")
                elif diff.change_type == "changed":
                    lines.append(
                        f"    ~ {diff.field}: {diff.local_value} -> {diff.remote_value}"
                    )
                elif diff.change_type == "info":
                    lines.append(f"    i {diff.field}: {diff.remote_value}")

    return "\n".join(lines)


def format_json_report(diffs: list[PricingDiff]) -> str:
    """Format diff report as JSON."""
    return json.dumps(
        [
            {
                "model_id": d.model_id,
                "provider": d.provider,
                "field": d.field,
                "local": d.local_value,
                "remote": d.remote_value,
                "change": d.change_type,
            }
            for d in diffs
        ],
        indent=2,
    )


def apply_pricing_overrides(
    diffs: list[PricingDiff],
    dry_run: bool = True,
    update_existing: bool = True,
    add_new: bool = False,
    disable_removed: bool = False,
) -> dict:
    """Apply pricing updates directly to the Model table.

    Updates the Model table directly for LiteLLM-sourced models.
    This ensures the comparison logic sees the updated values.

    Args:
        diffs: List of pricing differences to apply
        dry_run: If True, only report what would be done
        update_existing: Apply price changes to existing models
        add_new: Add new models found in LiteLLM
        disable_removed: Disable models not found in LiteLLM (not implemented yet)

    Returns:
        Dict with counts of updated/created models
    """
    import sys

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from db import Model
    from db.connection import check_db_initialized, get_db_context, init_db

    if not check_db_initialized():
        init_db()

    # Separate diffs into categories
    update_diffs: dict[tuple[str, str], dict[str, Any]] = {}
    new_model_diffs: list[PricingDiff] = []

    for diff in diffs:
        # Skip info-only diffs
        if diff.change_type == "info":
            continue

        # New models
        if diff.field == "(new model)":
            if add_new:
                new_model_diffs.append(diff)
            continue

        # Price updates to existing models
        if update_existing:
            key = (diff.provider, diff.model_id)
            if key not in update_diffs:
                update_diffs[key] = {}
            update_diffs[key][diff.field] = diff.remote_value

    results = {
        "created": 0,
        "updated": 0,
        "new_models_added": 0,
        "skipped": 0,
        "details": [],
    }

    if dry_run:
        logger.info("\n[DRY RUN] Would apply the following changes:")
        if update_existing:
            for (provider, model_id), fields in update_diffs.items():
                logger.info(f"  UPDATE {provider}/{model_id}: {fields}")
                results["details"].append(
                    {
                        "provider": provider,
                        "model_id": model_id,
                        "fields": fields,
                        "action": "would_update",
                    }
                )
        if add_new:
            for diff in new_model_diffs:
                logger.info(f"  ADD NEW {diff.provider}/{diff.model_id}")
                results["details"].append(
                    {
                        "provider": diff.provider,
                        "model_id": diff.model_id,
                        "action": "would_add_new",
                    }
                )
        return results

    with get_db_context() as db:
        # Apply price updates to existing models in the Model table
        if update_existing:
            for (provider, model_id), fields in update_diffs.items():
                model = (
                    db.query(Model)
                    .filter(
                        Model.provider_id == provider,
                        Model.id == model_id,
                    )
                    .first()
                )

                if model:
                    for field, value in fields.items():
                        if hasattr(model, field):
                            setattr(model, field, value)
                    results["updated"] += 1
                    results["details"].append(
                        {
                            "provider": provider,
                            "model_id": model_id,
                            "fields": fields,
                            "action": "updated",
                        }
                    )
                    logger.info(f"  Updated model: {provider}/{model_id}")
                else:
                    # Model not found in database, skip
                    results["skipped"] += 1
                    logger.warning(
                        f"  Model not found, skipping: {provider}/{model_id}"
                    )

        # Add new models to the Model table
        if add_new:
            for diff in new_model_diffs:
                # Check if model already exists
                existing = (
                    db.query(Model)
                    .filter(
                        Model.provider_id == diff.provider,
                        Model.id == diff.model_id,
                    )
                    .first()
                )

                if existing:
                    results["skipped"] += 1
                    continue

                # Parse pricing from remote_value string like "$1.0 in / $2.0 out"
                input_cost = None
                output_cost = None
                if diff.remote_value and isinstance(diff.remote_value, str):
                    parts = diff.remote_value.replace("$", "").split(" / ")
                    if len(parts) == 2:
                        try:
                            input_cost = float(parts[0].replace(" in", ""))
                            output_cost = float(parts[1].replace(" out", ""))
                        except ValueError:
                            pass

                # Infer family from model name
                model_lower = diff.model_id.lower()
                if (
                    "gpt" in model_lower
                    or "o1" in model_lower
                    or "o3" in model_lower
                    or "o4" in model_lower
                ):
                    family = "gpt"
                elif "claude" in model_lower:
                    family = "claude"
                elif "llama" in model_lower:
                    family = "llama"
                elif "gemini" in model_lower or "gemma" in model_lower:
                    family = "gemini"
                elif "mistral" in model_lower or "mixtral" in model_lower:
                    family = "mistral"
                elif "deepseek" in model_lower:
                    family = "deepseek"
                elif "qwen" in model_lower or "qwq" in model_lower:
                    family = "qwen"
                elif "grok" in model_lower:
                    family = "grok"
                elif "command" in model_lower:
                    family = "command"
                elif "sonar" in model_lower:
                    family = "sonar"
                else:
                    family = "other"

                new_model = Model(
                    id=diff.model_id,
                    provider_id=diff.provider,
                    source="litellm",
                    family=family,
                    description=diff.model_id,
                    context_length=128000,  # Default
                    capabilities=[],
                    unsupported_params=[],
                    supports_system_prompt=True,
                    use_max_completion_tokens=False,
                    enabled=True,
                    input_cost=input_cost,
                    output_cost=output_cost,
                )
                db.add(new_model)
                results["new_models_added"] += 1

                results["details"].append(
                    {
                        "provider": diff.provider,
                        "model_id": diff.model_id,
                        "action": "added_new",
                    }
                )
                logger.info(f"  Added new model: {diff.provider}/{diff.model_id}")

        db.commit()

    logger.info(
        f"\nApplied: {results['created']} price overrides created, "
        f"{results['updated']} updated, {results['new_models_added']} new models added"
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sync model pricing from LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Apply price changes to database (default: report only)",
    )
    parser.add_argument(
        "--add-new",
        action="store_true",
        help="Add new models found in LiteLLM to database",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Only sync specific provider (e.g., openai, anthropic)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed comparison info"
    )

    args = parser.parse_args()

    # Fetch remote pricing data
    try:
        litellm_data = fetch_litellm_pricing()
    except Exception as e:
        logger.error(f"Failed to fetch LiteLLM pricing: {e}")
        sys.exit(1)

    # Parse all models
    remote_models = []
    for key, data in litellm_data.items():
        if key == "sample_spec":
            continue
        model = parse_litellm_model(key, data)
        if model:
            remote_models.append(model)

    logger.info(f"Parsed {len(remote_models)} chat models from LiteLLM data")

    # Determine which providers to sync
    providers = (
        [args.provider] if args.provider else list(set(PROVIDER_MAPPING.values()))
    )

    # Compare with database
    all_diffs = []
    for provider in providers:
        local_config = load_effective_pricing(provider)
        provider_models = [m for m in remote_models if m.provider == provider]
        if provider_models:
            diffs = compare_pricing(local_config, provider_models, provider)
            all_diffs.extend(diffs)

    # Output report
    if args.output == "json":
        print(format_json_report(all_diffs))
    else:
        print(format_diff_report(all_diffs, verbose=args.verbose))

    # Apply updates if requested
    if args.update or args.add_new:
        result = apply_pricing_overrides(
            all_diffs,
            dry_run=False,
            update_existing=args.update,
            add_new=args.add_new,
        )
        logger.info(
            f"\nApplied: {result['created']} overrides created, "
            f"{result['updated']} updated, {result['new_models_added']} new models"
        )
    elif all_diffs:
        logger.info("\nRun with --update to apply price changes to database.")
        logger.info("Run with --add-new to add new models to database.")


if __name__ == "__main__":
    main()
