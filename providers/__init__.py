"""
LLM Providers package.

This module auto-registers all available providers with the global registry.
To add a new provider:
1. Create a new provider file (e.g., groq_provider.py)
2. Import and register it below
"""

# Import all providers
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ModelInfo, OpenAICompatibleProvider, get_api_key
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .perplexity_provider import PerplexityProvider
from .registry import registry

# Register all providers
registry.register(AnthropicProvider())
registry.register(OpenAIProvider())
registry.register(GeminiProvider())
registry.register(PerplexityProvider())

# Set default provider/model (Anthropic Claude Sonnet for backward compatibility)
registry.set_default("anthropic", "claude-sonnet-4-5-20250929")

__all__ = [
    "registry",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ModelInfo",
    "get_api_key",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "PerplexityProvider",
]
