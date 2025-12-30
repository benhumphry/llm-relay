"""
Anthropic Claude provider.

Anthropic uses its own SDK format which differs from OpenAI, so this
provider has a custom implementation rather than extending OpenAICompatibleProvider.
"""

import logging
from typing import Generator

import anthropic

from .base import LLMProvider, ModelInfo, get_api_key

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models."""

    name = "anthropic"

    # Model definitions
    models: dict[str, ModelInfo] = {
        # Claude 4.5 family (latest)
        "claude-opus-4-5-20251101": ModelInfo(
            family="claude-4.5",
            description="Claude Opus 4.5 - Most capable model, best for complex analysis and OCR",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing", "ocr"],
        ),
        "claude-sonnet-4-5-20250929": ModelInfo(
            family="claude-4.5",
            description="Claude Sonnet 4.5 - Balanced performance and speed",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        "claude-haiku-4-5-20251001": ModelInfo(
            family="claude-4.5",
            description="Claude Haiku 4.5 - Fastest model, ideal for tagging and quick tasks",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
        # Claude 4 family
        "claude-opus-4-20250514": ModelInfo(
            family="claude-4",
            description="Claude Opus 4 - Previous generation flagship",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        "claude-sonnet-4-20250514": ModelInfo(
            family="claude-4",
            description="Claude Sonnet 4 - Previous generation balanced model",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        # Claude 3.5 family (legacy)
        "claude-3-5-sonnet-20241022": ModelInfo(
            family="claude-3.5",
            description="Claude 3.5 Sonnet - Legacy model",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing"],
        ),
        "claude-3-5-haiku-20241022": ModelInfo(
            family="claude-3.5",
            description="Claude 3.5 Haiku - Legacy fast model",
            context_length=200000,
            capabilities=["vision", "analysis", "coding", "writing", "fast"],
        ),
    }

    # Aliases for user-friendly names
    aliases: dict[str, str] = {
        # Claude 4.5 aliases
        "claude-4.5-opus": "claude-opus-4-5-20251101",
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-4.5-haiku": "claude-haiku-4-5-20251001",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        # Claude 4 aliases
        "claude-4-opus": "claude-opus-4-20250514",
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        # Claude 3.5 aliases
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-haiku-3.5": "claude-3-5-haiku-20241022",
        # Generic aliases (point to latest recommended)
        "claude-opus": "claude-opus-4-5-20251101",
        "claude-sonnet": "claude-sonnet-4-5-20250929",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude": "claude-sonnet-4-5-20250929",
    }

    _client: anthropic.Anthropic | None = None

    def is_configured(self) -> bool:
        """Check if Anthropic API key is available."""
        return get_api_key("ANTHROPIC_API_KEY") is not None

    def get_models(self) -> dict[str, ModelInfo]:
        """Return models dict."""
        return self.models

    def get_aliases(self) -> dict[str, str]:
        """Return aliases dict."""
        return self.aliases

    def get_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            api_key = get_api_key("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _build_kwargs(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """Build kwargs for Anthropic API call."""
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": options.get("num_predict", options.get("max_tokens", 4096)),
        }

        if system:
            kwargs["system"] = system

        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if "top_k" in options:
            kwargs["top_k"] = options["top_k"]
        if "stop" in options:
            stop = options["stop"]
            kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        return kwargs

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """Execute non-streaming chat completion."""
        client = self.get_client()
        kwargs = self._build_kwargs(model, messages, system, options)

        response = client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return {
            "content": content,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> Generator[str, None, None]:
        """Execute streaming chat completion."""
        client = self.get_client()
        kwargs = self._build_kwargs(model, messages, system, options)

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
