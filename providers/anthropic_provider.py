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
    api_key_env = "ANTHROPIC_API_KEY"
    has_custom_cost_calculation = True  # Complex pricing: cache read/write tokens

    _client: anthropic.Anthropic | None = None

    def __init__(
        self,
        models: dict[str, ModelInfo] | None = None,
    ):
        """
        Initialize the provider.

        Args:
            models: Dict of model_id -> ModelInfo (loads from config if not provided)
        """
        self._models = models if models is not None else {}

    def is_configured(self) -> bool:
        """Check if Anthropic API key is available."""
        return get_api_key(self.api_key_env) is not None

    def get_models(self) -> dict[str, ModelInfo]:
        """Return models dict."""
        return self._models

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

        result = {
            "content": content,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Extract Anthropic cache token details if available
        if hasattr(response.usage, "cache_creation_input_tokens"):
            result["cache_creation_tokens"] = response.usage.cache_creation_input_tokens
        if hasattr(response.usage, "cache_read_input_tokens"):
            result["cache_read_tokens"] = response.usage.cache_read_input_tokens

        return result

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
