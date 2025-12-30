"""
OpenAI provider.

Uses the OpenAI SDK directly with their standard API endpoint.
"""

from .base import ModelInfo, OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for OpenAI GPT models."""

    name = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    models: dict[str, ModelInfo] = {
        # GPT-4o family (latest multimodal)
        "gpt-4o": ModelInfo(
            family="gpt-4o",
            description="GPT-4o - Most capable multimodal model",
            context_length=128000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        "gpt-4o-mini": ModelInfo(
            family="gpt-4o",
            description="GPT-4o Mini - Fast and affordable multimodal model",
            context_length=128000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
        # GPT-4 Turbo
        "gpt-4-turbo": ModelInfo(
            family="gpt-4",
            description="GPT-4 Turbo - High capability with vision",
            context_length=128000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        # GPT-4
        "gpt-4": ModelInfo(
            family="gpt-4",
            description="GPT-4 - Original GPT-4 model",
            context_length=8192,
            capabilities=["analysis", "coding", "writing"],
        ),
        # GPT-3.5
        "gpt-3.5-turbo": ModelInfo(
            family="gpt-3.5",
            description="GPT-3.5 Turbo - Fast and cost-effective",
            context_length=16385,
            capabilities=["analysis", "coding", "writing", "fast"],
        ),
        # O1 reasoning models
        "o1": ModelInfo(
            family="o1",
            description="O1 - Advanced reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning"],
        ),
        "o1-mini": ModelInfo(
            family="o1",
            description="O1 Mini - Fast reasoning model",
            context_length=128000,
            capabilities=["analysis", "coding", "reasoning", "fast"],
        ),
        "o1-pro": ModelInfo(
            family="o1",
            description="O1 Pro - Most capable reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning"],
        ),
    }

    aliases: dict[str, str] = {
        # GPT-4o aliases
        "gpt4o": "gpt-4o",
        "gpt-4o-latest": "gpt-4o",
        "gpt4o-mini": "gpt-4o-mini",
        # GPT-4 aliases
        "gpt4": "gpt-4",
        "gpt4-turbo": "gpt-4-turbo",
        # GPT-3.5 aliases
        "gpt35": "gpt-3.5-turbo",
        "gpt-35-turbo": "gpt-3.5-turbo",
        "chatgpt": "gpt-3.5-turbo",
        # O1 aliases
        "o1-preview": "o1",
        # Generic alias
        "openai": "gpt-4o",
    }
