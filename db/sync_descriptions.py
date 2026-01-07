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
    4. Base model match (both directions):
       - Strip suffixes from our model to find base in descriptions
       - Find descriptions with dated suffixes that match our base model

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

    def normalize_model_name(name: str) -> str:
        """Normalize model name for fuzzy matching.

        Handles variations like:
        - claude-haiku-4.5 vs claude-haiku-4-5 (dot vs hyphen in versions)
        - claude-3.5-sonnet vs claude-3-5-sonnet
        """
        # Replace dots with hyphens in version numbers (e.g., 4.5 -> 4-5)
        return re.sub(r"(\d)\.(\d)", r"\1-\2", name.lower())

    # Try exact match with provider prefix
    full_key = f"{provider_id}/{model_id}"
    if full_key in descriptions:
        return descriptions[full_key]

    # Try exact match on model_id only
    if model_id in descriptions:
        return descriptions[model_id]

    # Try normalized matching (handles 4.5 vs 4-5 variations)
    normalized_model = normalize_model_name(model_id)
    normalized_full_key = f"{provider_id}/{normalized_model}"

    for desc_key, desc_value in descriptions.items():
        if normalize_model_name(desc_key) == normalized_full_key:
            return desc_value
        if "/" not in desc_key and normalize_model_name(desc_key) == normalized_model:
            return desc_value

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

    # Version/date suffixes that indicate the same model at different points in time
    # NOT variant suffixes like -mini, -pro, -turbo which are different models
    version_suffixes = [
        r"-\d{8}$",  # -20240229 (8-digit date suffix)
        r"-\d{2}-\d{4}$",  # -08-2024 (MM-YYYY suffix)
        r"-\d{4}-\d{2}-\d{2}$",  # -2024-02-29 (ISO date)
        r"-latest$",  # -latest
        r"-preview$",  # -preview
        r"-beta$",  # -beta
        r"-exp$",  # -exp (experimental)
        r"-experimental$",  # -experimental
    ]

    providers_to_try = [provider_id] + provider_aliases.get(provider_id, [])

    # Strategy A: Strip suffixes from OUR model to find base in descriptions
    for suffix_pattern in version_suffixes:
        match = re.search(suffix_pattern, model_id, re.IGNORECASE)
        if match:
            base_model = model_id[: match.start()]
            # Try exact match first
            for prov in providers_to_try:
                base_full_key = f"{prov}/{base_model}"
                if base_full_key in descriptions:
                    return descriptions[base_full_key]
            if base_model in descriptions:
                return descriptions[base_model]

            # Try normalized match (handles 4-5 vs 4.5 variations)
            normalized_base = normalize_model_name(base_model)
            for desc_key, desc_value in descriptions.items():
                normalized_desc_key = normalize_model_name(desc_key)
                for prov in providers_to_try:
                    if normalized_desc_key == f"{prov}/{normalized_base}":
                        return desc_value
                if "/" not in desc_key and normalized_desc_key == normalized_base:
                    return desc_value

    # Strategy B: Find descriptions with dated suffixes that match our base model
    # e.g., our model is "command-r-plus", find "cohere/command-r-plus-08-2024"
    for desc_key, desc_value in descriptions.items():
        # Extract provider and model from description key
        if "/" in desc_key:
            desc_provider, desc_model = desc_key.split("/", 1)
        else:
            desc_provider = None
            desc_model = desc_key

        # Check if provider matches (or is an alias)
        if desc_provider and desc_provider not in providers_to_try:
            continue

        # Check if stripping suffix from description model matches our model
        for suffix_pattern in version_suffixes:
            match = re.search(suffix_pattern, desc_model, re.IGNORECASE)
            if match:
                base_desc_model = desc_model[: match.start()]
                if base_desc_model == model_id:
                    return desc_value

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

    Note: Anthropic only returns display_name, not actual descriptions.
    The display_name is just the model ID formatted, not useful as a description.
    We return an empty dict since there are no real descriptions to sync.

    Args:
        api_key: Anthropic API key

    Returns:
        Empty dict - Anthropic doesn't provide descriptions
    """
    # Anthropic's API only returns display_name which is just the model ID
    # (e.g., "claude-3-opus-20240229") - not a useful description.
    # OpenRouter provides actual descriptions for Anthropic models.
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"Anthropic has {len(data.get('data', []))} models (no descriptions in API)"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch Anthropic models: {e}")

    return {}


def fetch_openai_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from OpenAI's API.

    Note: OpenAI's API returns minimal metadata (id, created, owned_by).
    No descriptions available.
    """
    descriptions = {}
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

    return descriptions


def fetch_groq_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from Groq's API.

    Groq's models endpoint returns id, owned_by, and active status.
    No descriptions available.
    """
    descriptions = {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"Groq has {len(data.get('data', []))} models (no descriptions available)"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch Groq models: {e}")

    return descriptions


def fetch_mistral_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from Mistral's API.

    Mistral returns id, owned_by, capabilities, and description.
    """
    descriptions = {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                model_id = model.get("id", "")
                description = model.get("description", "")

                if model_id and description:
                    descriptions[model_id] = description

            logger.info(f"Fetched {len(descriptions)} descriptions from Mistral")

    except Exception as e:
        logger.warning(f"Failed to fetch Mistral models: {e}")

    return descriptions


def fetch_perplexity_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from Perplexity's API.

    Perplexity uses OpenAI-compatible API but doesn't provide descriptions.
    """
    descriptions = {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.perplexity.ai/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"Perplexity has {len(data.get('data', []))} models (no descriptions available)"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch Perplexity models: {e}")

    return descriptions


def fetch_cohere_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from Cohere's API.

    Cohere's models endpoint includes name and description.
    """
    descriptions = {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.cohere.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                model_id = model.get("name", "")
                description = model.get("description", "")

                if model_id and description:
                    descriptions[model_id] = description

            logger.info(f"Fetched {len(descriptions)} descriptions from Cohere")

    except Exception as e:
        logger.warning(f"Failed to fetch Cohere models: {e}")

    return descriptions


def fetch_xai_descriptions(api_key: str) -> dict[str, str]:
    """
    Fetch model info from xAI's API.

    xAI uses OpenAI-compatible API.
    """
    descriptions = {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                model_id = model.get("id", "")
                description = model.get("description", "")

                if model_id and description:
                    descriptions[model_id] = description

            if descriptions:
                logger.info(f"Fetched {len(descriptions)} descriptions from xAI")
            else:
                logger.info(
                    f"xAI has {len(data.get('data', []))} models (no descriptions available)"
                )

    except Exception as e:
        logger.warning(f"Failed to fetch xAI models: {e}")

    return descriptions


# Provider configuration: maps provider_id to (env_var, fetch_function)
PROVIDER_FETCH_CONFIG = {
    "gemini": ("GOOGLE_API_KEY", fetch_google_descriptions),
    "anthropic": ("ANTHROPIC_API_KEY", fetch_anthropic_descriptions),
    "openai": ("OPENAI_API_KEY", fetch_openai_descriptions),
    "groq": ("GROQ_API_KEY", fetch_groq_descriptions),
    "mistral": ("MISTRAL_API_KEY", fetch_mistral_descriptions),
    "perplexity": ("PERPLEXITY_API_KEY", fetch_perplexity_descriptions),
    "cohere": ("COHERE_API_KEY", fetch_cohere_descriptions),
    "xai": ("XAI_API_KEY", fetch_xai_descriptions),
}


def get_available_description_providers() -> list[dict]:
    """
    Get list of providers that can be synced for descriptions.

    Returns all providers from the database (since OpenRouter can provide
    descriptions for any provider's models), plus OpenRouter itself.

    Returns:
        List of dicts with provider_id and has_api_key status
    """
    from .connection import get_db_context
    from .models import Provider

    providers = []
    seen = set()

    # Get all providers from database
    with get_db_context() as db:
        db_providers = db.query(Provider).filter(Provider.enabled == True).all()
        for p in db_providers:
            if p.id not in seen:
                # Check if this provider has a native fetch function
                env_var = PROVIDER_FETCH_CONFIG.get(p.id, (None, None))[0]
                providers.append(
                    {
                        "id": p.id,
                        "has_api_key": True,  # All DB providers can get descriptions from OpenRouter
                    }
                )
                seen.add(p.id)

    # Always include OpenRouter (no API key needed, provides descriptions for all)
    if "openrouter" not in seen:
        providers.append({"id": "openrouter", "has_api_key": True})

    return providers


def fetch_all_provider_descriptions(
    provider_filter: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """
    Fetch descriptions from configured providers.

    Args:
        provider_filter: If set, only fetch from this provider

    Returns:
        Tuple of:
        - Dict mapping provider_id to {model_id: description}
        - Dict mapping provider_id to status ("success", "no_descriptions", "no_api_key", "error")
    """
    all_descriptions = {}
    status = {}

    for provider_id, (env_var, fetch_func) in PROVIDER_FETCH_CONFIG.items():
        # Skip if filtering to a specific provider
        if provider_filter and provider_filter != provider_id:
            continue

        api_key = os.environ.get(env_var)
        if not api_key:
            status[provider_id] = "no_api_key"
            continue

        try:
            descriptions = fetch_func(api_key)
            if descriptions:
                all_descriptions[provider_id] = descriptions
                status[provider_id] = "success"
            else:
                status[provider_id] = "no_descriptions"
        except Exception as e:
            logger.warning(f"Error fetching from {provider_id}: {e}")
            status[provider_id] = "error"

    return all_descriptions, status


def sync_model_descriptions(
    update_existing: bool = False,
    provider: str | None = None,
) -> dict:
    """
    Sync model descriptions from provider APIs and/or OpenRouter.

    Strategy:
    1. Fetch from individual providers first (most current for new models)
    2. Overlay OpenRouter descriptions (richer content)
    3. Update database

    Args:
        update_existing: If True, overwrite existing descriptions
        provider: If set, only sync from this provider (or "openrouter" for OpenRouter only)

    Returns:
        Dict with sync statistics
    """
    stats = {
        "updated": 0,
        "skipped": 0,
        "providers_synced": [],
        "providers_attempted": {},
    }

    # Determine what to fetch
    fetch_providers = provider != "openrouter"
    fetch_openrouter = provider is None or provider == "openrouter"

    # Step 1: Fetch from individual providers (if not OpenRouter-only)
    provider_descriptions = {}
    provider_status = {}
    if fetch_providers:
        provider_descriptions, provider_status = fetch_all_provider_descriptions(
            provider_filter=provider if provider and provider != "openrouter" else None
        )
        stats["providers_attempted"].update(provider_status)

    # Step 2: Fetch from OpenRouter (public API) if included
    openrouter_descriptions = {}
    if fetch_openrouter:
        openrouter_descriptions = fetch_openrouter_descriptions()
        stats["providers_attempted"]["openrouter"] = (
            "success" if openrouter_descriptions else "no_descriptions"
        )

    # Step 3: Update database
    with get_db_context() as db:
        # Filter models if syncing a specific provider
        if provider and provider != "openrouter":
            models = db.query(Model).filter(Model.provider_id == provider).all()
        else:
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
            if openrouter_descriptions:
                openrouter_desc = _find_best_description_match(
                    model.id, model.provider_id, openrouter_descriptions
                )

                if openrouter_desc:
                    # OpenRouter descriptions are usually richer, prefer them
                    # unless the provider gave us something longer
                    if not new_description or len(openrouter_desc) > len(
                        new_description
                    ):
                        new_description = openrouter_desc
                        source = "openrouter"

            # Validate description is actually useful (not just the model name)
            if new_description:
                desc_lower = new_description.lower().strip()
                model_lower = model.id.lower()
                full_name = f"{model.provider_id}/{model.id}".lower()

                # Skip if description is just the model name or provider/model
                if desc_lower in (
                    model_lower,
                    full_name,
                    model_lower.replace("-", " "),
                    model_lower.replace("_", " "),
                ):
                    new_description = None
                # Skip very short descriptions (likely just names)
                elif len(new_description) < 20:
                    new_description = None

            # Update if we have a useful description
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

    # Build list of synced providers
    synced = list(provider_descriptions.keys())
    if openrouter_descriptions:
        synced.append("openrouter")
    stats["providers_synced"] = synced

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
