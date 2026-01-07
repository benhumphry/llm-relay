"""
Sync model descriptions from provider APIs and OpenRouter.

Strategy:
1. Fetch descriptions from each provider's API (if API key available)
2. Overlay with OpenRouter's richer descriptions (public API, no key needed)
3. Store in the Model table's description field
"""

import logging
import os
from typing import Optional

import httpx

from .connection import get_db_context
from .models import Model

logger = logging.getLogger(__name__)

# Timeout for API requests
REQUEST_TIMEOUT = 30.0


def fetch_openrouter_descriptions() -> dict[str, str]:
    """
    Fetch model descriptions from OpenRouter's public API.

    Returns:
        Dict mapping model_id to description
    """
    descriptions = {}

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                model_id = model.get("id", "")
                description = model.get("description", "")

                if model_id and description:
                    # OpenRouter IDs are like "anthropic/claude-3-opus"
                    # Store both the full ID and just the model part
                    descriptions[model_id] = description

                    # Also store by just the model name part for matching
                    if "/" in model_id:
                        model_name = model_id.split("/", 1)[1]
                        # Don't overwrite if we already have a more specific match
                        if model_name not in descriptions:
                            descriptions[model_name] = description

            logger.info(f"Fetched {len(descriptions)} descriptions from OpenRouter")

    except Exception as e:
        logger.warning(f"Failed to fetch OpenRouter descriptions: {e}")

    return descriptions


def _find_best_description_match(
    model_id: str, provider_id: str, descriptions: dict[str, str]
) -> str | None:
    """
    Find the best matching description for a model.

    Matching strategy (in order of preference):
    1. Exact match: provider/model_id
    2. Exact match: model_id only
    3. Cross-provider match: Try common provider name variations
    4. Base model match: Strip date/version suffixes only (not variant suffixes)

    For base model matching, a description for "claude-3-opus" should apply to:
    - claude-3-opus-20250101 (dated version)
    - claude-3-opus-latest (latest alias)
    - claude-3-opus-preview (preview version)

    But should NOT apply to:
    - claude-3-opus-mini (different variant)
    - claude-3-opus-pro (different variant)
    - claude-3-sonnet (different model)
    """
    import re

    # Try exact match with provider prefix
    full_key = f"{provider_id}/{model_id}"
    if full_key in descriptions:
        return descriptions[full_key]

    # Try exact match on model_id only
    if model_id in descriptions:
        return descriptions[model_id]

    # Try cross-provider matching for common provider name variations
    # OpenRouter uses different provider names than we do internally
    provider_aliases = {
        "xai": ["x-ai"],
        "x-ai": ["xai"],
        "google": ["google-ai"],
        "google-ai": ["google"],
        "gemini": ["google", "google-ai"],
        "meta": ["meta-llama"],
        "meta-llama": ["meta"],
    }

    for alias in provider_aliases.get(provider_id, []):
        alias_key = f"{alias}/{model_id}"
        if alias_key in descriptions:
            return descriptions[alias_key]

    # Try base model matching for dated/versioned models ONLY
    # These are suffixes that indicate the same model at a different point in time
    # NOT variant suffixes like -mini, -pro, -turbo which are different models
    version_suffixes = [
        r"-\d{8}$",  # -20240229 (8-digit date suffix)
        r"-\d{4}-\d{2}-\d{2}$",  # -2024-02-29 (ISO date)
        r"-latest$",  # -latest
        r"-preview$",  # -preview
        r"-beta$",  # -beta
        r"-exp$",  # -exp (experimental)
        r"-experimental$",  # -experimental
    ]
    # Note: We intentionally DON'T strip these as they indicate different models:
    # -mini, -pro, -turbo, -plus, -ultra, -lite, -small, -medium, -large
    # -0125, -0314 etc (4-digit) are ambiguous - could be dates or versions, skip them

    for suffix_pattern in version_suffixes:
        match = re.search(suffix_pattern, model_id, re.IGNORECASE)
        if match:
            # Extract base model name
            base_model = model_id[: match.start()]

            # Try to find description for base model (with all provider variations)
            providers_to_try = [provider_id] + provider_aliases.get(provider_id, [])
            for prov in providers_to_try:
                base_full_key = f"{prov}/{base_model}"
                if base_full_key in descriptions:
                    return descriptions[base_full_key]

            if base_model in descriptions:
                return descriptions[base_model]

    # No match found
    return None


def fetch_google_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model descriptions from Google's Gemini API.

    Args:
        api_key: Google API key

    Returns:
        Dict mapping model_id to description
    """
    descriptions = {}

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key},
            )
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                # Google returns names like "models/gemini-1.5-pro"
                name = model.get("name", "")
                description = model.get("description", "")

                if name and description:
                    # Extract just the model ID
                    model_id = name.replace("models/", "")
                    descriptions[model_id] = description

            logger.info(f"Fetched {len(descriptions)} descriptions from Google")

    except Exception as e:
        logger.warning(f"Failed to fetch Google descriptions: {e}")

    return descriptions


def fetch_anthropic_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from Anthropic's API.

    Note: Anthropic only returns display_name, not descriptions.
    We still fetch to get the current model list.

    Args:
        api_key: Anthropic API key

    Returns:
        Dict mapping model_id to display_name (not rich description)
    """
    descriptions = {}

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                model_id = model.get("id", "")
                display_name = model.get("display_name", "")

                if model_id and display_name:
                    # Anthropic doesn't have descriptions, just display names
                    # We'll use this as a fallback if no OpenRouter description
                    descriptions[model_id] = display_name

            logger.info(f"Fetched {len(descriptions)} model names from Anthropic")

    except Exception as e:
        logger.warning(f"Failed to fetch Anthropic model info: {e}")

    return descriptions


def fetch_openai_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from OpenAI's API.

    Note: OpenAI's API returns minimal metadata (id, created, owned_by).
    No descriptions available.

    Args:
        api_key: OpenAI API key

    Returns:
        Empty dict (OpenAI doesn't provide descriptions)
    """
    # OpenAI doesn't provide descriptions in their models endpoint
    # Just log that we checked
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"OpenAI has {len(data.get('data', []))} models (no descriptions available)"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch OpenAI models: {e}")

    return {}


def fetch_all_provider_descriptions() -> dict[str, dict[str, str]]:
    """
    Fetch descriptions from all configured providers.

    Returns:
        Dict mapping provider_id to {model_id: description}
    """
    all_descriptions = {}

    # Google Gemini
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        all_descriptions["gemini"] = fetch_google_descriptions(google_key)

    # Anthropic (display names only, but useful for model list)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        all_descriptions["anthropic"] = fetch_anthropic_descriptions(anthropic_key)

    # OpenAI (no descriptions, but we try anyway)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        all_descriptions["openai"] = fetch_openai_descriptions(openai_key)

    return all_descriptions


def sync_model_descriptions(update_existing: bool = False) -> dict:
    """
    Sync model descriptions from all sources.

    Strategy:
    1. Fetch from individual providers first (most current for new models)
    2. Overlay OpenRouter descriptions (richer content)
    3. Update database

    Args:
        update_existing: If True, overwrite existing descriptions

    Returns:
        Dict with sync statistics
    """
    stats = {"updated": 0, "skipped": 0, "providers_synced": []}

    # Step 1: Fetch from individual providers
    provider_descriptions = fetch_all_provider_descriptions()

    # Step 2: Fetch from OpenRouter (public API)
    openrouter_descriptions = fetch_openrouter_descriptions()

    # Step 3: Update database
    with get_db_context() as db:
        models = db.query(Model).all()

        for model in models:
            new_description = None
            source = None

            # First, try provider-specific description (exact match only)
            if model.provider_id in provider_descriptions:
                provider_desc = provider_descriptions[model.provider_id].get(model.id)
                if provider_desc:
                    new_description = provider_desc
                    source = model.provider_id

            # Then, try OpenRouter with smart matching for dated/versioned models
            # OpenRouter descriptions are typically more detailed
            openrouter_desc = _find_best_description_match(
                model.id, model.provider_id, openrouter_descriptions
            )

            if openrouter_desc:
                # OpenRouter descriptions are usually richer, prefer them
                # unless the provider gave us something longer
                if not new_description or len(openrouter_desc) > len(new_description):
                    new_description = openrouter_desc
                    source = "openrouter"

            # Update if we have a new description
            if new_description:
                if not model.description or update_existing:
                    model.description = new_description
                    stats["updated"] += 1
                    logger.debug(
                        f"Updated {model.provider_id}/{model.id} from {source}"
                    )
                else:
                    stats["skipped"] += 1
            else:
                stats["skipped"] += 1

        db.commit()

    stats["providers_synced"] = list(provider_descriptions.keys()) + ["openrouter"]
    logger.info(
        f"Description sync complete: {stats['updated']} updated, {stats['skipped']} skipped"
    )

    return stats


def get_model_description(provider_id: str, model_id: str) -> Optional[str]:
    """
    Get the description for a specific model from the database.

    Args:
        provider_id: Provider identifier
        model_id: Model identifier

    Returns:
        Description string or None
    """
    with get_db_context() as db:
        model = (
            db.query(Model)
            .filter(Model.provider_id == provider_id, Model.id == model_id)
            .first()
        )

        if model:
            return model.description

    return None


def get_descriptions_for_models(model_ids: list[str]) -> dict[str, str]:
    """
    Get descriptions for multiple models.

    Args:
        model_ids: List of model identifiers (can be "provider/model" or just "model")

    Returns:
        Dict mapping model_id to description
    """
    descriptions = {}

    with get_db_context() as db:
        for model_id in model_ids:
            # Parse provider/model format
            if "/" in model_id:
                provider_id, model_name = model_id.split("/", 1)
            else:
                provider_id = None
                model_name = model_id

            # Query for the model
            query = db.query(Model)
            if provider_id:
                query = query.filter(
                    Model.provider_id == provider_id, Model.id == model_name
                )
            else:
                query = query.filter(Model.id == model_name)

            model = query.first()
            if model and model.description:
                descriptions[model_id] = model.description

    return descriptions
