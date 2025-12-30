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


@dataclass
class ModelInfo:
    """Metadata about a model."""

    family: str
    description: str
    context_length: int
    capabilities: list[str] = field(default_factory=list)
    parameter_size: str = "?B"
    quantization_level: str = "none"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier (e.g., "anthropic", "openai", "gemini")

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if this provider has valid API credentials configured."""
        pass

    @abstractmethod
    def get_models(self) -> dict[str, ModelInfo]:
        """Return dict of model_id -> ModelInfo for this provider."""
        pass

    @abstractmethod
    def get_aliases(self) -> dict[str, str]:
        """Return dict of alias -> model_id for user-friendly names."""
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

        Checks aliases first, then direct model IDs.
        Returns None if model not found in this provider.
        """
        name = model_name.lower().strip()

        # Remove :latest suffix
        if name.endswith(":latest"):
            name = name[:-7]

        # Check aliases
        aliases = self.get_aliases()
        if name in aliases:
            return aliases[name]

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
    Together AI, and many others. Subclasses only need to define:
    - name: Provider identifier
    - base_url: API endpoint URL
    - api_key_env: Environment variable name for API key
    - models: Dict of model_id -> ModelInfo
    - aliases: Dict of alias -> model_id
    """

    name: str
    base_url: str
    api_key_env: str
    models: dict[str, ModelInfo]
    aliases: dict[str, str]

    _client: OpenAI | None = None

    def is_configured(self) -> bool:
        """Check if API key is available."""
        return get_api_key(self.api_key_env) is not None

    def get_models(self) -> dict[str, ModelInfo]:
        """Return models dict."""
        return self.models

    def get_aliases(self) -> dict[str, str]:
        """Return aliases dict."""
        return self.aliases

    def get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = get_api_key(self.api_key_env)
            if not api_key:
                raise ValueError(f"{self.api_key_env} is required for {self.name}")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _build_messages(self, messages: list[dict], system: str | None) -> list[dict]:
        """Build messages list with system prompt if provided."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend(messages)
        return result

    def _build_kwargs(self, model: str, options: dict) -> dict:
        """Build kwargs for OpenAI API call."""
        kwargs = {
            "model": model,
            "max_tokens": options.get("max_tokens", 4096),
        }

        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if "stop" in options:
            stop = options["stop"]
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]

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
        kwargs["messages"] = self._build_messages(messages, system)

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
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

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(messages, system)
        kwargs["stream"] = True

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
