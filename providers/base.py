"""
Base classes and protocols for LLM providers.

This module defines the interface that all providers must implement,
plus a base class for OpenAI-compatible providers that handles most
of the implementation automatically.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator

from openai import OpenAI

logger = logging.getLogger(__name__)


# Parameters that reasoning models (GPT-5, o1, o3, etc.) don't support
REASONING_MODEL_UNSUPPORTED_PARAMS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "logit_bias",
}


@dataclass
class ModelInfo:
    """Metadata about a model."""

    family: str
    description: str
    context_length: int
    capabilities: list[str] = field(default_factory=list)
    parameter_size: str = "?B"
    quantization_level: str = "none"
    # Parameters that this model does NOT support (will be filtered out)
    unsupported_params: set[str] = field(default_factory=set)
    # Whether the model supports system prompts
    supports_system_prompt: bool = True
    # Whether to use max_completion_tokens instead of max_tokens (for reasoning models)
    use_max_completion_tokens: bool = False
    # Cost per million tokens (USD)
    input_cost: float | None = None
    output_cost: float | None = None
    # Cache cost multipliers (v2.2.3)
    # e.g., 0.1 = 10% of input cost for Anthropic cache reads
    cache_read_multiplier: float | None = None
    # e.g., 1.25 = 125% of input cost for Anthropic cache writes
    cache_write_multiplier: float | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier (e.g., "anthropic", "openai", "gemini")

    # Whether this provider calculates costs dynamically (vs using static database rates)
    # Providers with custom cost calculation (Anthropic, Gemini, Perplexity, OpenRouter)
    # should set this to True. When True, cost overrides in the admin UI are hidden.
    has_custom_cost_calculation: bool = False

    # Whether this provider is enabled (can be toggled via admin UI)
    # Disabled providers still appear in the UI but won't be used for requests
    enabled: bool = True

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if this provider has valid API credentials configured."""
        pass

    def is_available(self) -> bool:
        """Check if this provider is available for API requests.

        A provider is available if it's both enabled AND configured.
        """
        return self.enabled and self.is_configured()

    @abstractmethod
    def get_models(self) -> dict[str, ModelInfo]:
        """Return dict of model_id -> ModelInfo for this provider."""
        pass

    @abstractmethod
    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """
        Execute a non-streaming chat completion.

        Args:
            model: The model ID to use
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            options: Dict of options (max_tokens, temperature, etc.)

        Returns:
            Dict with 'content', 'input_tokens', 'output_tokens'
        """
        pass

    @abstractmethod
    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> Generator[str, None, None]:
        """
        Execute a streaming chat completion.

        Args:
            model: The model ID to use
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            options: Dict of options (max_tokens, temperature, etc.)

        Yields:
            Text chunks as they arrive
        """
        pass

    def resolve_model(self, model_name: str) -> str | None:
        """
        Resolve a model name to a model ID for this provider.

        Returns None if model not found in this provider.
        """
        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Check direct model IDs
        models = self.get_models()
        if name in models:
            return name

        return None


def get_api_key(env_var: str, file_env_var: str | None = None) -> str | None:
    """
    Get API key from environment variable or file.

    Args:
        env_var: Name of environment variable containing the key
        file_env_var: Optional name of env var containing path to key file

    Returns:
        API key string or None if not configured
    """
    # Check direct environment variable
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key

    # Check file-based secret (Docker Swarm secrets)
    if file_env_var:
        api_key_file = os.environ.get(file_env_var)
        if api_key_file and os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                return f.read().strip()

    # Also check default _FILE suffix
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()

    return None


class OpenAICompatibleProvider(LLMProvider):
    """
    Base class for providers that support the OpenAI API format.

    This covers most providers: OpenAI, Google Gemini, Perplexity, Groq,
    Together AI, and many others.

    Can be instantiated directly with config, or subclassed with class attributes.
    """

    name: str
    base_url: str
    api_key_env: str

    _client: OpenAI | None = None

    def __init__(
        self,
        name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        models: dict[str, ModelInfo] | None = None,
    ):
        """
        Initialize the provider.

        Args:
            name: Provider identifier (uses class attribute if not provided)
            base_url: API endpoint URL (uses class attribute if not provided)
            api_key_env: Environment variable name (uses class attribute if not provided)
            models: Dict of model_id -> ModelInfo (uses class attribute if not provided)
        """
        # Use provided values or fall back to class attributes
        if name is not None:
            self.name = name
        if base_url is not None:
            self.base_url = base_url
        if api_key_env is not None:
            self.api_key_env = api_key_env

        # Models - use instance attributes
        self._models = (
            models if models is not None else getattr(self.__class__, "models", {})
        )

    def is_configured(self) -> bool:
        """Check if API key is available."""
        return get_api_key(self.api_key_env) is not None

    def get_models(self) -> dict[str, ModelInfo]:
        """Return models dict."""
        return self._models

    def get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = get_api_key(self.api_key_env)
            if not api_key:
                raise ValueError(f"{self.api_key_env} is required for {self.name}")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _get_model_info(self, model: str) -> ModelInfo | None:
        """Get ModelInfo for a model."""
        if model in self._models:
            return self._models[model]
        return None

    def _convert_image_format(self, content: list) -> list:
        """Convert Anthropic-style image format to OpenAI format.

        Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        OpenAI format: {"type": "image_url", "url": "data:image/jpeg;base64,..."}
        """
        converted = []
        for block in content:
            if block.get("type") == "image" and "source" in block:
                source = block["source"]
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    converted.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                else:
                    converted.append(block)
            else:
                converted.append(block)
        return converted

    def _build_messages(
        self, model: str, messages: list[dict], system: str | None
    ) -> list[dict]:
        """Build messages list with system prompt if supported by model."""
        result = []
        model_info = self._get_model_info(model)

        # Only add system prompt if the model supports it
        if system:
            if model_info and not model_info.supports_system_prompt:
                # Prepend system content to first user message instead
                logger.debug(
                    f"Model {model} doesn't support system prompts, "
                    "prepending to first user message"
                )
                if messages and messages[0].get("role") == "user":
                    first_msg = messages[0].copy()
                    content = first_msg.get("content", "")
                    if isinstance(content, str):
                        first_msg["content"] = f"{system}\n\n{content}"
                    result.append(first_msg)
                    result.extend(self._convert_messages_images(messages[1:]))
                    return result
            else:
                result.append({"role": "system", "content": system})

        result.extend(self._convert_messages_images(messages))
        return result

    def _convert_messages_images(self, messages: list[dict]) -> list[dict]:
        """Convert image formats in messages to OpenAI format."""
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Has content blocks - may contain images
                new_msg = msg.copy()
                new_msg["content"] = self._convert_image_format(content)
                converted.append(new_msg)
            else:
                converted.append(msg)
        return converted

    def _build_kwargs(self, model: str, options: dict) -> dict:
        """Build kwargs for OpenAI API call, filtering unsupported params."""
        model_info = self._get_model_info(model)
        unsupported = model_info.unsupported_params if model_info else set()
        use_max_completion_tokens = (
            model_info.use_max_completion_tokens if model_info else False
        )

        kwargs = {
            "model": model,
        }

        # Handle max_tokens - reasoning models use max_completion_tokens instead
        # Default to 16384 to avoid truncation on models with higher limits
        max_tokens_value = options.get("max_tokens", 16384)
        if use_max_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens_value
        elif "max_tokens" not in unsupported:
            kwargs["max_tokens"] = max_tokens_value

        # Only add parameters if the model supports them
        if "temperature" in options and "temperature" not in unsupported:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options and "top_p" not in unsupported:
            kwargs["top_p"] = options["top_p"]
        if "stop" in options and "stop" not in unsupported:
            stop = options["stop"]
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]
        if "presence_penalty" in options and "presence_penalty" not in unsupported:
            kwargs["presence_penalty"] = options["presence_penalty"]
        if "frequency_penalty" in options and "frequency_penalty" not in unsupported:
            kwargs["frequency_penalty"] = options["frequency_penalty"]
        if "reasoning_effort" in options:
            # For OpenAI o-series reasoning models - controls thinking depth
            # Values: "none", "low", "medium", "high"
            # Non-reasoning models and other providers will ignore this parameter
            kwargs["reasoning_effort"] = options["reasoning_effort"]

        # Log if we filtered any parameters
        filtered = [p for p in options if p in unsupported]
        if filtered:
            logger.debug(f"Filtered unsupported params for {model}: {filtered}")

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

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(model, messages, system)

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        result = {
            "content": content,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        # Extract extended token details if available (OpenAI o1/o3 models)
        if response.usage:
            # OpenAI reasoning tokens (o1, o3 models)
            if hasattr(response.usage, "completion_tokens_details"):
                details = response.usage.completion_tokens_details
                if details and hasattr(details, "reasoning_tokens"):
                    result["reasoning_tokens"] = details.reasoning_tokens

            # OpenAI cached input tokens (prompt_tokens_details.cached_tokens)
            if hasattr(response.usage, "prompt_tokens_details"):
                details = response.usage.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    result["cached_input_tokens"] = details.cached_tokens

            # DeepSeek uses a different format (prompt_cache_hit_tokens at top level)
            # Only use this if we didn't already get cached tokens from OpenAI format
            if "cached_input_tokens" not in result:
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    cache_hit = response.usage.prompt_cache_hit_tokens
                    if cache_hit:
                        result["cached_input_tokens"] = cache_hit

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

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(model, messages, system)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        # Reset last stream result
        self._last_stream_result = None

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            # Yield content if present
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

            # Capture usage from final chunk (when stream_options.include_usage is True)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                result = {
                    "input_tokens": usage.prompt_tokens or 0,
                    "output_tokens": usage.completion_tokens or 0,
                }

                # Extract reasoning tokens (OpenAI o1/o3/o4 models)
                if hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        result["reasoning_tokens"] = details.reasoning_tokens

                # Extract cached tokens (OpenAI format)
                if hasattr(usage, "prompt_tokens_details"):
                    details = usage.prompt_tokens_details
                    if details and hasattr(details, "cached_tokens"):
                        result["cached_input_tokens"] = details.cached_tokens

                # DeepSeek format (prompt_cache_hit_tokens at top level)
                if "cached_input_tokens" not in result:
                    if hasattr(usage, "prompt_cache_hit_tokens"):
                        cache_hit = usage.prompt_cache_hit_tokens
                        if cache_hit:
                            result["cached_input_tokens"] = cache_hit

                self._last_stream_result = result

    def get_last_stream_result(self) -> dict | None:
        """Get the result from the last streaming call.

        Returns dict with 'input_tokens', 'output_tokens', and optionally
        'reasoning_tokens', 'cached_input_tokens' if available.
        """
        return getattr(self, "_last_stream_result", None)
