"""
OpenAI provider.

Uses the OpenAI SDK directly with their standard API endpoint.
"""

from .base import (
    REASONING_MODEL_UNSUPPORTED_PARAMS,
    ModelInfo,
    OpenAICompatibleProvider,
)


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for OpenAI GPT models."""

    name = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    models: dict[str, ModelInfo] = {
        # GPT-5 family (reasoning models - no temperature/top_p/etc, uses max_completion_tokens)
        "gpt-5": ModelInfo(
            family="gpt-5",
            description="GPT-5 - Latest flagship reasoning model",
            context_length=128000,
            capabilities=["vision", "analysis", "coding", "writing", "reasoning"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
        "gpt-5-mini": ModelInfo(
            family="gpt-5",
            description="GPT-5 Mini - Fast reasoning model",
            context_length=128000,
            capabilities=[
                "vision",
                "analysis",
                "coding",
                "writing",
                "reasoning",
                "fast",
            ],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
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
        # O3 reasoning models (latest reasoning - no temperature/top_p/etc, uses max_completion_tokens)
        "o3": ModelInfo(
            family="o3",
            description="O3 - Most advanced reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
        "o3-mini": ModelInfo(
            family="o3",
            description="O3 Mini - Fast advanced reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning", "fast"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
        # O1 reasoning models (no temperature/top_p/etc, uses max_completion_tokens)
        "o1": ModelInfo(
            family="o1",
            description="O1 - Advanced reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
        "o1-mini": ModelInfo(
            family="o1",
            description="O1 Mini - Fast reasoning model",
            context_length=128000,
            capabilities=["analysis", "coding", "reasoning", "fast"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
        "o1-pro": ModelInfo(
            family="o1",
            description="O1 Pro - Most capable O1 reasoning model",
            context_length=200000,
            capabilities=["analysis", "coding", "reasoning"],
            unsupported_params=REASONING_MODEL_UNSUPPORTED_PARAMS,
            supports_system_prompt=False,
            use_max_completion_tokens=True,
        ),
    }

    aliases: dict[str, str] = {
        # GPT-5 aliases
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt5-mini": "gpt-5-mini",
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
        # O3 aliases
        "o3-latest": "o3",
        # O1 aliases
        "o1-preview": "o1",
        # Generic alias - point to best non-reasoning model for compatibility
        "openai": "gpt-4o",
    }
