"""
Perplexity provider.

Perplexity offers an OpenAI-compatible API, so we can use the base class.
Their models are particularly good for search-augmented generation.
"""

from .base import ModelInfo, OpenAICompatibleProvider


class PerplexityProvider(OpenAICompatibleProvider):
    """Provider for Perplexity models."""

    name = "perplexity"
    base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"

    models: dict[str, ModelInfo] = {
        # Sonar models (search-augmented)
        "sonar": ModelInfo(
            family="sonar",
            description="Sonar - Search-augmented model with web access",
            context_length=128000,
            capabilities=["search", "analysis", "writing"],
        ),
        "sonar-pro": ModelInfo(
            family="sonar",
            description="Sonar Pro - Advanced search-augmented model",
            context_length=200000,
            capabilities=["search", "analysis", "writing", "reasoning"],
        ),
        "sonar-reasoning": ModelInfo(
            family="sonar",
            description="Sonar Reasoning - Search with chain-of-thought",
            context_length=128000,
            capabilities=["search", "analysis", "reasoning"],
        ),
        # Legacy Sonar models
        "llama-3.1-sonar-small-128k-online": ModelInfo(
            family="sonar-llama",
            description="Sonar Small - Lightweight search model",
            context_length=128000,
            capabilities=["search", "analysis", "fast"],
        ),
        "llama-3.1-sonar-large-128k-online": ModelInfo(
            family="sonar-llama",
            description="Sonar Large - Full-featured search model",
            context_length=128000,
            capabilities=["search", "analysis", "writing"],
        ),
        "llama-3.1-sonar-huge-128k-online": ModelInfo(
            family="sonar-llama",
            description="Sonar Huge - Most capable search model",
            context_length=128000,
            capabilities=["search", "analysis", "writing", "reasoning"],
        ),
    }

    aliases: dict[str, str] = {
        # Sonar aliases
        "pplx": "sonar",
        "pplx-pro": "sonar-pro",
        "sonar-small": "llama-3.1-sonar-small-128k-online",
        "sonar-large": "llama-3.1-sonar-large-128k-online",
        "sonar-huge": "llama-3.1-sonar-huge-128k-online",
        # Generic alias
        "perplexity": "sonar",
    }
