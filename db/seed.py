"""
Database seeding for LLM Relay.

Seeds the database with default providers and models from LiteLLM on first run.
"""

import logging
from datetime import datetime

import requests

from .connection import get_db_context
from .models import Model, Provider

logger = logging.getLogger(__name__)

# LiteLLM pricing data URL
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Default providers to seed on first run
DEFAULT_PROVIDERS = [
    # First-party providers
    {
        "id": "anthropic",
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "display_name": "Anthropic",
    },
    {
        "id": "openai",
        "type": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "display_name": "OpenAI",
    },
    {
        "id": "gemini",
        "type": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "display_name": "Google Gemini",
    },
    {
        "id": "deepseek",
        "type": "openai-compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "display_name": "DeepSeek",
    },
    {
        "id": "mistral",
        "type": "openai-compatible",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "display_name": "Mistral",
    },
    {
        "id": "groq",
        "type": "openai-compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "display_name": "Groq",
    },
    {
        "id": "xai",
        "type": "openai-compatible",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "display_name": "xAI",
    },
    # Aggregator providers
    {
        "id": "openrouter",
        "type": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "display_name": "OpenRouter",
    },
    {
        "id": "perplexity",
        "type": "perplexity",
        "base_url": "https://api.perplexity.ai",
        "api_key_env": "PERPLEXITY_API_KEY",
        "display_name": "Perplexity",
    },
    # Inference providers
    {
        "id": "fireworks",
        "type": "openai-compatible",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "display_name": "Fireworks",
    },
    {
        "id": "together",
        "type": "openai-compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "display_name": "Together AI",
    },
    {
        "id": "deepinfra",
        "type": "openai-compatible",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "display_name": "DeepInfra",
    },
    {
        "id": "cerebras",
        "type": "openai-compatible",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "display_name": "Cerebras",
    },
    {
        "id": "sambanova",
        "type": "openai-compatible",
        "base_url": "https://api.sambanova.ai/v1",
        "api_key_env": "SAMBANOVA_API_KEY",
        "display_name": "SambaNova",
    },
    {
        "id": "cohere",
        "type": "openai-compatible",
        "base_url": "https://api.cohere.ai/v1",
        "api_key_env": "COHERE_API_KEY",
        "display_name": "Cohere",
    },
]

# Map LiteLLM provider names to our provider IDs
LITELLM_PROVIDER_MAPPING = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "groq": "groq",
    "mistral": "mistral",
    "xai": "xai",
    "perplexity": "perplexity",
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
    "container",
    "test",
    "sample",
}


def seed_providers() -> int:
    """
    Seed default providers on first run.

    Returns:
        Number of providers created
    """
    created = 0
    with get_db_context() as db:
        for p in DEFAULT_PROVIDERS:
            existing = db.query(Provider).filter(Provider.id == p["id"]).first()
            if not existing:
                provider = Provider(
                    id=p["id"],
                    type=p["type"],
                    base_url=p.get("base_url"),
                    api_key_env=p.get("api_key_env"),
                    display_name=p.get("display_name"),
                    source="system",
                    enabled=True,
                )
                db.add(provider)
                created += 1
                logger.debug(f"Created provider: {p['id']}")
        db.commit()

    if created:
        logger.info(f"Seeded {created} default providers")
    return created


def fetch_litellm_data() -> dict:
    """Fetch model data from LiteLLM GitHub."""
    logger.info("Fetching model data from LiteLLM...")
    response = requests.get(LITELLM_PRICING_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    logger.info(f"Fetched {len(data)} model entries from LiteLLM")
    return data


def parse_litellm_model(key: str, data: dict) -> dict | None:
    """
    Parse a LiteLLM model entry into our format.

    Returns:
        Dict with model fields, or None if model should be skipped
    """
    # Skip non-chat models
    if data.get("mode") not in ("chat", None):
        return None

    litellm_provider = data.get("litellm_provider", "")

    # Filter out invalid model names
    model_name = key.split("/")[-1] if "/" in key else key
    if model_name.lower() in INVALID_MODEL_NAMES:
        return None

    # Handle prefixed keys like "xai/grok-4" or "groq/llama-3.3-70b"
    if "/" in key:
        prefix = key.split("/")[0]
        model_id = "/".join(key.split("/")[1:])
        if prefix in LITELLM_PROVIDER_MAPPING:
            provider_id = LITELLM_PROVIDER_MAPPING[prefix]
        elif litellm_provider in LITELLM_PROVIDER_MAPPING:
            provider_id = LITELLM_PROVIDER_MAPPING[litellm_provider]
        else:
            return None
    else:
        if litellm_provider in LITELLM_PROVIDER_MAPPING:
            provider_id = LITELLM_PROVIDER_MAPPING[litellm_provider]
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

    # Calculate cache multipliers
    cache_read_multiplier = None
    cache_write_multiplier = None

    if input_cost and data.get("cache_read_input_token_cost"):
        cache_read_cost = data["cache_read_input_token_cost"] * 1_000_000
        cache_read_multiplier = round(cache_read_cost / input_cost, 2)

    if input_cost and data.get("cache_creation_input_token_cost"):
        cache_write_cost = data["cache_creation_input_token_cost"] * 1_000_000
        if cache_write_cost > 0:
            cache_write_multiplier = round(cache_write_cost / input_cost, 2)

    # Extract context length
    context_length = 128000  # default
    if data.get("max_input_tokens"):
        context_length = int(data["max_input_tokens"])

    # Map capabilities
    capabilities = []
    for litellm_cap, our_cap in CAPABILITY_MAPPING.items():
        if data.get(litellm_cap):
            capabilities.append(our_cap)

    # Infer family from model name
    model_lower = model_id.lower()
    if "claude" in model_lower:
        family = "claude"
    elif (
        "gpt" in model_lower
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("o4")
    ):
        family = "gpt"
    elif "gemini" in model_lower or "gemma" in model_lower:
        family = "gemini"
    elif "llama" in model_lower:
        family = "llama"
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
    elif "phi" in model_lower:
        family = "phi"
    else:
        family = "other"

    # Check for reasoning models that need max_completion_tokens
    use_max_completion_tokens = False
    if model_lower.startswith("o1") or model_lower.startswith("o3"):
        use_max_completion_tokens = True

    return {
        "id": model_id,
        "provider_id": provider_id,
        "family": family,
        "description": model_id,
        "context_length": context_length,
        "capabilities": capabilities,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_read_multiplier": cache_read_multiplier,
        "cache_write_multiplier": cache_write_multiplier,
        "supports_system_prompt": True,
        "use_max_completion_tokens": use_max_completion_tokens,
    }


def seed_models_from_litellm(
    provider_id: str | None = None, update_existing: bool = False
) -> dict:
    """
    Seed models from LiteLLM pricing data.

    Args:
        provider_id: Specific provider to seed, or None for all
        update_existing: If True, update existing models; if False, only add new

    Returns:
        Dict with counts: {"created": N, "updated": N, "skipped": N}
    """
    results = {"created": 0, "updated": 0, "skipped": 0}

    try:
        litellm_data = fetch_litellm_data()
    except Exception as e:
        logger.error(f"Failed to fetch LiteLLM data: {e}")
        raise

    # Get set of provider IDs we support
    supported_providers = set(LITELLM_PROVIDER_MAPPING.values())
    if provider_id:
        supported_providers = {provider_id}

    # Parse all models, deduplicating by (provider_id, model_id)
    models_by_key: dict[tuple[str, str], dict] = {}
    for key, data in litellm_data.items():
        if key == "sample_spec":
            continue

        parsed = parse_litellm_model(key, data)
        if parsed and parsed["provider_id"] in supported_providers:
            model_key = (parsed["provider_id"], parsed["id"])
            # Keep the first occurrence (or could merge/update)
            if model_key not in models_by_key:
                models_by_key[model_key] = parsed

    models_to_add = list(models_by_key.values())
    logger.info(f"Parsed {len(models_to_add)} unique models for seeding")

    now = datetime.utcnow()

    with get_db_context() as db:
        # Ensure providers exist first
        existing_providers = {p.id for p in db.query(Provider.id).all()}

        for m in models_to_add:
            # Skip if provider doesn't exist
            if m["provider_id"] not in existing_providers:
                results["skipped"] += 1
                continue

            existing = (
                db.query(Model)
                .filter(Model.provider_id == m["provider_id"], Model.id == m["id"])
                .first()
            )

            if existing:
                if update_existing:
                    # Update existing model
                    existing.input_cost = m["input_cost"]
                    existing.output_cost = m["output_cost"]
                    existing.cache_read_multiplier = m["cache_read_multiplier"]
                    existing.cache_write_multiplier = m["cache_write_multiplier"]
                    existing.context_length = m["context_length"]
                    existing.capabilities = m["capabilities"]
                    existing.last_synced = now
                    results["updated"] += 1
                else:
                    results["skipped"] += 1
            else:
                # Create new model
                model = Model(
                    id=m["id"],
                    provider_id=m["provider_id"],
                    source="litellm",
                    last_synced=now,
                    family=m["family"],
                    description=m["description"],
                    context_length=m["context_length"],
                    capabilities=m["capabilities"],
                    input_cost=m["input_cost"],
                    output_cost=m["output_cost"],
                    cache_read_multiplier=m["cache_read_multiplier"],
                    cache_write_multiplier=m["cache_write_multiplier"],
                    supports_system_prompt=m["supports_system_prompt"],
                    use_max_completion_tokens=m["use_max_completion_tokens"],
                    enabled=True,
                )
                db.add(model)
                results["created"] += 1

        db.commit()

    logger.info(
        f"Seeding complete: {results['created']} created, "
        f"{results['updated']} updated, {results['skipped']} skipped"
    )
    return results


def ensure_seeded() -> None:
    """
    Ensure the database is seeded with defaults.

    Called on application startup. Seeds providers (always works offline),
    then attempts to seed models from LiteLLM (graceful failure if offline).

    This function is idempotent - it will add any missing providers
    without duplicating existing ones, allowing for seamless upgrades when
    new providers are added to DEFAULT_PROVIDERS.
    """
    with get_db_context() as db:
        model_count = db.query(Model).count()

    # Always ensure all default providers exist (adds missing ones on upgrade)
    providers_added = seed_providers()

    if providers_added > 0:
        logger.info(f"Seeded {providers_added} providers")

    # Seed models from LiteLLM if empty
    if model_count == 0:
        logger.info("No models in database, seeding from LiteLLM...")
        try:
            seed_models_from_litellm()
        except Exception as e:
            logger.warning(f"Could not seed models from LiteLLM: {e}")
            logger.info("Models can be added manually or synced later via admin UI")

    # Apply YAML-based model overrides (provider quirks, deprecated models)
    try:
        from config.override_loader import apply_yaml_overrides_to_db

        apply_yaml_overrides_to_db()

        # Clear the model cache so overrides take effect
        from providers.loader import clear_config_cache

        clear_config_cache()
    except Exception as e:
        logger.warning(f"Could not apply model overrides: {e}")
