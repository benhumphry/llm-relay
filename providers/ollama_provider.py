"""
Ollama provider for local LLM inference.

This provider connects to a local Ollama instance and provides:
- Dynamic model discovery from the Ollama API
- Model pull/delete operations
- Periodic refresh of available models
- OpenAI-compatible chat completions
"""

import json
import logging
import time
from typing import Any, Generator

import httpx
from openai import OpenAI

from .base import LLMProvider, ModelInfo, get_api_key

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Provider for local Ollama instance with dynamic model discovery.

    Unlike other providers, Ollama models are discovered dynamically from
    the local Ollama API rather than being defined in YAML configuration.

    Multiple Ollama instances can be configured with different names and URLs.
    """

    # How often to refresh the model list (seconds)
    REFRESH_INTERVAL = 60

    # Default context length for discovered models
    DEFAULT_CONTEXT_LENGTH = 8192

    def __init__(
        self,
        name: str = "ollama",
        base_url: str = "http://localhost:11434",
        aliases: dict[str, str] | None = None,
    ):
        """
        Initialize the Ollama provider.

        Args:
            name: Provider name (allows multiple Ollama instances)
            base_url: Ollama API URL (e.g., http://192.168.1.100:11434)
            aliases: Optional dict of alias -> model_id mappings
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._aliases = aliases or {}
        self._models: dict[str, ModelInfo] = {}
        self._last_refresh: float = 0
        self._client: OpenAI | None = None
        self._http_client: httpx.Client | None = None

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for Ollama API calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self.base_url,
                timeout=30.0,
            )
        return self._http_client

    def _get_openai_client(self) -> OpenAI:
        """Get or create OpenAI client for chat completions."""
        if self._client is None:
            # Ollama doesn't require an API key, but OpenAI client needs one
            api_key = get_api_key("OLLAMA_API_KEY") or "ollama"
            self._client = OpenAI(
                api_key=api_key,
                base_url=f"{self.base_url}/v1",
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            client = self._get_http_client()
            response = client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not accessible: {e}")
            return False

    def _should_refresh(self) -> bool:
        """Check if model list should be refreshed."""
        return time.time() - self._last_refresh > self.REFRESH_INTERVAL

    def _refresh_models(self) -> None:
        """Fetch current models from Ollama API."""
        try:
            client = self._get_http_client()
            response = client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            self._models = {}
            for model in models:
                name = model.get("name", "")
                if not name:
                    continue

                details = model.get("details", {})

                # Extract model info from Ollama response
                family = details.get("family", name.split(":")[0])
                parameter_size = details.get("parameter_size", "?B")
                quantization = details.get("quantization_level", "unknown")

                # Build description
                description = f"{name}"
                if parameter_size and parameter_size != "?B":
                    description += f" - {parameter_size}"
                if quantization and quantization != "unknown":
                    description += f" {quantization}"

                # Determine capabilities based on model name
                capabilities = ["analysis", "coding", "writing"]
                name_lower = name.lower()
                if "vision" in name_lower or "llava" in name_lower:
                    capabilities.append("vision")
                if "code" in name_lower:
                    capabilities.insert(0, "coding")

                self._models[name] = ModelInfo(
                    family=family,
                    description=description,
                    context_length=self.DEFAULT_CONTEXT_LENGTH,
                    capabilities=capabilities,
                    parameter_size=parameter_size,
                    quantization_level=quantization,
                    unsupported_params=set(),
                    supports_system_prompt=True,
                    use_max_completion_tokens=False,
                )

            self._last_refresh = time.time()
            logger.info(f"Refreshed Ollama models: {len(self._models)} models found")

        except Exception as e:
            logger.warning(f"Failed to refresh Ollama models: {e}")
            # Keep existing models on failure

    def get_models(self) -> dict[str, ModelInfo]:
        """Return dict of model_id -> ModelInfo, refreshing if stale."""
        if self._should_refresh():
            self._refresh_models()
        return self._models

    def get_aliases(self) -> dict[str, str]:
        """Return aliases dict."""
        return self._aliases

    def get_raw_models(self) -> list[dict[str, Any]]:
        """
        Get raw model data from Ollama API for admin UI.

        Returns more details than get_models() for display purposes.
        """
        try:
            client = self._get_http_client()
            response = client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> Generator[dict[str, Any], None, None]:
        """
        Pull a model from the Ollama library.

        Args:
            model_name: Name of model to pull (e.g., "llama3.2", "mistral:7b")

        Yields:
            Progress dicts with status, completed, total, etc.
        """
        try:
            # Use streaming request for progress updates
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=None,  # Pulls can take a long time
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            continue

            # Refresh model list after successful pull
            self._refresh_models()

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            yield {"error": str(e)}

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from Ollama.

        Args:
            model_name: Name of model to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            client = self._get_http_client()
            response = client.request(
                "DELETE",
                "/api/delete",
                json={"name": model_name},
            )
            response.raise_for_status()

            # Refresh model list after deletion
            self._refresh_models()

            logger.info(f"Deleted Ollama model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """
        Get detailed info about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model details dict or None if not found
        """
        try:
            client = self._get_http_client()
            response = client.post(
                "/api/show",
                json={"name": model_name},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get info for model {model_name}: {e}")
            return None

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """Execute non-streaming chat completion via OpenAI-compatible API."""
        client = self._get_openai_client()

        # Build messages with system prompt
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        # Build kwargs
        kwargs = {
            "model": model,
            "messages": api_messages,
        }

        if "max_tokens" in options:
            kwargs["max_tokens"] = options["max_tokens"]
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if "stop" in options:
            stop = options["stop"]
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Ollama may report inflated token counts (thinking tokens, cumulative counts)
        # Estimate tokens from content length as sanity check (~4 chars per token)

        # Estimate input from messages
        input_chars = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
            for m in api_messages
        )
        estimated_input = input_chars // 4 + 1
        if input_tokens > estimated_input * 3:
            logger.info(
                f"Ollama reported {input_tokens} input tokens for {input_chars} chars - using estimate {estimated_input}"
            )
            input_tokens = estimated_input

        # Estimate output from response
        estimated_output = len(content) // 4 + 1
        if output_tokens > estimated_output * 3:
            logger.info(
                f"Ollama reported {output_tokens} output tokens for {len(content)} chars - using estimate {estimated_output}"
            )
            output_tokens = estimated_output

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> Generator[str, None, None]:
        """Execute streaming chat completion via OpenAI-compatible API."""
        client = self._get_openai_client()

        # Build messages with system prompt
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        # Build kwargs
        kwargs = {
            "model": model,
            "messages": api_messages,
            "stream": True,
        }

        if "max_tokens" in options:
            kwargs["max_tokens"] = options["max_tokens"]
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if "stop" in options:
            stop = options["stop"]
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
