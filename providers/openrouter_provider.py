"""
OpenRouter provider with dynamic cost extraction.

OpenRouter is a meta-provider that routes to many different LLM providers.
Unlike other providers, OpenRouter returns the actual cost in the API response,
so we extract and pass it through rather than calculating from static pricing.
"""

import logging
from typing import Generator

import httpx

from .base import LLMProvider, ModelInfo, get_api_key

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """
    Provider for OpenRouter with dynamic cost extraction.

    OpenRouter provides cost data in the response, either via:
    1. The 'usage' field when requested with extra params
    2. The /api/v1/generation/{id} endpoint after completion

    This provider uses the httpx library directly (instead of OpenAI SDK)
    to access the full response including cost data.
    """

    name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"
    has_custom_cost_calculation = True  # Dynamic pricing from API response

    def __init__(
        self,
        models: dict[str, ModelInfo] | None = None,
        aliases: dict[str, str] | None = None,
    ):
        """
        Initialize the provider.

        Args:
            models: Dict of model_id -> ModelInfo (loads from config if not provided)
            aliases: Dict of alias -> model_id (loads from config if not provided)
        """
        self._models = models if models is not None else {}
        self._aliases = aliases if aliases is not None else {}
        self._client: httpx.Client | None = None

    def is_configured(self) -> bool:
        """Check if OpenRouter API key is available."""
        return get_api_key(self.api_key_env) is not None

    def get_models(self) -> dict[str, ModelInfo]:
        """Return models dict."""
        return self._models

    def get_aliases(self) -> dict[str, str]:
        """Return aliases dict."""
        return self._aliases

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            api_key = get_api_key(self.api_key_env)
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for OpenRouter provider"
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/ollama-claude-proxy",
                    "X-Title": "ollama-claude-proxy",
                },
                timeout=120.0,
            )
        return self._client

    def _get_model_info(self, model: str) -> ModelInfo | None:
        """Get ModelInfo for a model, checking aliases if needed."""
        if model in self._models:
            return self._models[model]
        for alias, model_id in self._aliases.items():
            if model == model_id and model_id in self._models:
                return self._models[model_id]
        return None

    def _build_messages(
        self, model: str, messages: list[dict], system: str | None
    ) -> list[dict]:
        """Build messages list with system prompt."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend(messages)
        return result

    def _build_request_body(
        self, model: str, messages: list[dict], system: str | None, options: dict
    ) -> dict:
        """Build request body for OpenRouter API."""
        model_info = self._get_model_info(model)

        body = {
            "model": model,
            "messages": self._build_messages(model, messages, system),
            # Request usage data including cost
            "usage": {"include": True},
        }

        # Add optional parameters
        max_tokens = options.get("max_tokens", 4096)
        if model_info and model_info.use_max_completion_tokens:
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens

        unsupported = model_info.unsupported_params if model_info else set()

        if "temperature" in options and "temperature" not in unsupported:
            body["temperature"] = options["temperature"]
        if "top_p" in options and "top_p" not in unsupported:
            body["top_p"] = options["top_p"]
        if "stop" in options and "stop" not in unsupported:
            stop = options["stop"]
            body["stop"] = stop if isinstance(stop, list) else [stop]

        return body

    def _extract_cost_from_response(self, response_data: dict) -> float | None:
        """
        Extract cost from OpenRouter response.

        OpenRouter returns cost in the usage object when requested.
        """
        usage = response_data.get("usage", {})

        # OpenRouter returns cost in the usage object
        if "cost" in usage:
            return float(usage["cost"])

        # Also check for total_cost field
        if "total_cost" in usage:
            return float(usage["total_cost"])

        return None

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """
        Execute non-streaming chat completion.

        Returns dict with 'content', 'input_tokens', 'output_tokens', and 'cost'.
        The 'cost' field is extracted directly from OpenRouter's response.
        """
        client = self._get_client()
        body = self._build_request_body(model, messages, system, options)

        response = client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        # Extract content
        content = ""
        if data.get("choices"):
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")

        # Extract token counts
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Extract cost from response
        cost = self._extract_cost_from_response(data)

        result = {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Only include cost if we got it from the response
        if cost is not None:
            result["cost"] = cost
            logger.debug(f"OpenRouter returned cost: ${cost:.6f}")

        return result

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> Generator[str, None, dict]:
        """
        Execute streaming chat completion.

        Yields content strings during streaming.
        Returns a dict with 'cost', 'input_tokens', 'output_tokens' at the end
        (accessible via generator.return_value after exhausting the generator).

        OpenRouter includes usage/cost data in the final SSE chunk.
        """
        import json as json_module

        client = self._get_client()
        body = self._build_request_body(model, messages, system, options)
        body["stream"] = True
        # Also request usage info in streaming mode
        body["stream_options"] = {"include_usage": True}

        cost = None
        input_tokens = 0
        output_tokens = 0

        with client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                # Remove "data: " prefix
                if line.startswith("data: "):
                    line = line[6:]

                if line == "[DONE]":
                    break

                try:
                    chunk = json_module.loads(line)

                    # Check for usage data (usually in final chunk)
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                        # Extract cost
                        if "cost" in usage:
                            cost = float(usage["cost"])
                        elif "total_cost" in usage:
                            cost = float(usage["total_cost"])

                    # Yield content
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except (json_module.JSONDecodeError, KeyError, IndexError):
                    continue

        # Store the final stats for access after iteration
        # The caller can access this via the generator's return value
        self._last_stream_result = {
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        if cost is not None:
            logger.debug(f"OpenRouter streaming returned cost: ${cost:.6f}")

        return self._last_stream_result

    def get_last_stream_result(self) -> dict | None:
        """Get the result from the last streaming call (cost, tokens)."""
        return getattr(self, "_last_stream_result", None)

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
        if name in self._aliases:
            return self._aliases[name]

        # Check direct model IDs (case-sensitive for OpenRouter)
        if model_name in self._models:
            return model_name

        # Also check lowercase
        if name in self._models:
            return name

        return None
