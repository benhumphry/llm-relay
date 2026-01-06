"""
Google Gemini provider with custom cost calculation.

Gemini has a complex pricing model that includes:
1. Tiered input pricing based on prompt length (≤200k vs >200k tokens)
2. Thinking tokens (billed at output rate, reported separately)
3. Context caching (cached tokens at reduced rate)
4. Grounding with Google Search (per-query pricing for Gemini 3)

This provider extracts all usage data from Gemini's API response
and calculates accurate costs.
"""

import logging
from typing import Generator

from openai import OpenAI

from .base import ModelInfo, OpenAICompatibleProvider, get_api_key

logger = logging.getLogger(__name__)


class GeminiProvider(OpenAICompatibleProvider):
    """
    Google Gemini provider with custom cost calculation.

    Extends OpenAICompatibleProvider to handle Gemini's unique pricing:
    - Tiered input pricing (different rates for prompts ≤200k vs >200k tokens)
    - Thinking tokens (separate from output tokens)
    - Cached content tokens (discounted rate)
    - Grounding with Google Search (per-query pricing)
    """

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env = "GOOGLE_API_KEY"
    has_custom_cost_calculation = (
        True  # Complex pricing: tiered by prompt length, thinking, caching
    )

    # Prompt length threshold for tiered pricing (in tokens)
    PROMPT_LENGTH_THRESHOLD = 200_000

    # Models with tiered pricing (input cost doubles above threshold)
    # Format: model_id -> (low_input, high_input, output)
    # "low" = prompts ≤200k tokens, "high" = prompts >200k tokens
    TIERED_PRICING = {
        # Gemini 3 Pro: $2/$4 input, $12/$18 output
        "gemini-3-pro-preview": {
            "input_low": 2.00,
            "input_high": 4.00,
            "output_low": 12.00,
            "output_high": 18.00,
            "cache_low": 0.20,
            "cache_high": 0.40,
        },
        # Gemini 2.5 Pro: $1.25/$2.50 input, $10/$15 output
        "gemini-2.5-pro": {
            "input_low": 1.25,
            "input_high": 2.50,
            "output_low": 10.00,
            "output_high": 15.00,
            "cache_low": 0.125,
            "cache_high": 0.25,
        },
        # Gemini 2.5 Computer Use: same as 2.5 Pro
        "gemini-2.5-computer-use-preview-10-2025": {
            "input_low": 1.25,
            "input_high": 2.50,
            "output_low": 10.00,
            "output_high": 15.00,
        },
    }

    # Models with flat pricing (no tiered input)
    # These use the standard input_cost/output_cost from database

    # Context cache pricing (per 1M tokens)
    # Default cache multiplier for models without specific cache pricing
    DEFAULT_CACHE_MULTIPLIER = 0.1  # 10% of input cost

    # Grounding with Google Search pricing
    # Gemini 3: $14 per 1K queries (starting Jan 5, 2026)
    # Gemini 2.5 and earlier: $35 per 1K grounded prompts
    SEARCH_COST_GEMINI_3 = 14.00  # per 1K queries
    SEARCH_COST_LEGACY = 35.00  # per 1K grounded prompts

    def __init__(
        self,
        name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        models: dict[str, ModelInfo] | None = None,
    ):
        """Initialize the Gemini provider."""
        super().__init__(
            name=name or self.name,
            base_url=base_url or self.base_url,
            api_key_env=api_key_env or self.api_key_env,
            models=models,
        )
        self._last_stream_result: dict | None = None

    def _is_gemini_3(self, model_id: str) -> bool:
        """Check if model is a Gemini 3 model (for search pricing)."""
        return model_id.startswith("gemini-3")

    def _calculate_gemini_cost(
        self,
        model_id: str,
        usage: dict,
    ) -> float | None:
        """
        Calculate total cost for a Gemini request.

        Args:
            model_id: The model identifier
            usage: Usage dict from API response containing token counts

        Returns:
            Total cost in USD, or None if cannot calculate
        """
        model_info = self._models.get(model_id)
        if not model_info:
            logger.debug(f"No model info for {model_id}, cannot calculate cost")
            return None

        # Extract token counts
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        thinking_tokens = usage.get("thoughts_token_count", 0) or 0
        cached_tokens = usage.get("cached_content_token_count", 0) or 0

        # Determine if we're in the high tier (>200k prompt tokens)
        is_high_tier = prompt_tokens > self.PROMPT_LENGTH_THRESHOLD

        # Get pricing rates
        if model_id in self.TIERED_PRICING:
            pricing = self.TIERED_PRICING[model_id]
            input_rate = pricing["input_high"] if is_high_tier else pricing["input_low"]
            output_rate = pricing.get(
                "output_high" if is_high_tier else "output_low",
                model_info.output_cost or 0,
            )
            cache_rate = pricing.get(
                "cache_high" if is_high_tier else "cache_low",
                input_rate * self.DEFAULT_CACHE_MULTIPLIER,
            )
        else:
            # Use flat pricing from model config
            input_rate = model_info.input_cost or 0
            output_rate = model_info.output_cost or 0
            cache_rate = input_rate * self.DEFAULT_CACHE_MULTIPLIER

        # Calculate input cost (subtract cached tokens, they're priced separately)
        regular_input_tokens = max(0, prompt_tokens - cached_tokens)
        input_cost = (regular_input_tokens / 1_000_000) * input_rate

        # Calculate cache cost
        cache_cost = (cached_tokens / 1_000_000) * cache_rate

        # Calculate output cost (completion_tokens may or may not include thinking)
        # In Gemini API, thinking tokens are reported separately in thoughts_token_count
        # and are billed at the output rate
        output_cost = (completion_tokens / 1_000_000) * output_rate

        # Thinking tokens (if not already included in completion_tokens)
        # Note: Gemini API behavior varies - sometimes included, sometimes separate
        # We track them separately for logging but assume they're in completion_tokens
        # unless we detect them separately
        thinking_cost = 0.0
        if thinking_tokens > 0:
            # If thinking tokens are reported separately, they're additional
            # This depends on whether using native Gemini API vs OpenAI-compatible
            # For OpenAI-compatible, they may be included in completion_tokens
            logger.debug(f"Thinking tokens reported: {thinking_tokens}")

        # Calculate search/grounding cost
        search_cost = 0.0
        search_queries = usage.get("web_search_queries", [])
        if search_queries:
            num_queries = len(search_queries) if isinstance(search_queries, list) else 0
            if self._is_gemini_3(model_id):
                # Gemini 3: per-query pricing
                search_cost = (num_queries / 1_000) * self.SEARCH_COST_GEMINI_3
            else:
                # Legacy models: per-prompt pricing (1 prompt = 1 charge if grounded)
                search_cost = (1 / 1_000) * self.SEARCH_COST_LEGACY

        total_cost = input_cost + cache_cost + output_cost + thinking_cost + search_cost

        if is_high_tier:
            logger.debug(
                f"Gemini HIGH TIER cost for {model_id} ({prompt_tokens} tokens): "
                f"input=${input_cost:.6f}, cache=${cache_cost:.6f}, "
                f"output=${output_cost:.6f}, search=${search_cost:.6f}, "
                f"total=${total_cost:.6f}"
            )
        else:
            logger.debug(
                f"Gemini cost for {model_id}: input=${input_cost:.6f}, "
                f"cache=${cache_cost:.6f}, output=${output_cost:.6f}, "
                f"search=${search_cost:.6f}, total=${total_cost:.6f}"
            )

        return total_cost

    def _extract_gemini_usage(self, response) -> dict:
        """
        Extract Gemini-specific usage fields from API response.

        The OpenAI-compatible API may return standard fields, while
        the native Gemini API returns usage_metadata with different field names.
        """
        usage = {}

        if not response.usage:
            return usage

        # Standard OpenAI-compatible fields
        usage["prompt_tokens"] = response.usage.prompt_tokens or 0
        usage["completion_tokens"] = response.usage.completion_tokens or 0

        # Gemini-specific fields (may be in usage or usage_metadata)
        # Try various attribute names for compatibility

        # Thinking tokens
        for attr in ["thoughts_token_count", "thinking_tokens", "reasoning_tokens"]:
            if hasattr(response.usage, attr):
                value = getattr(response.usage, attr)
                if value:
                    usage["thoughts_token_count"] = value
                    break

        # Cached tokens
        for attr in ["cached_content_token_count", "cached_tokens"]:
            if hasattr(response.usage, attr):
                value = getattr(response.usage, attr)
                if value:
                    usage["cached_content_token_count"] = value
                    break

        # Also check prompt_tokens_details for cached tokens (OpenAI format)
        if hasattr(response.usage, "prompt_tokens_details"):
            details = response.usage.prompt_tokens_details
            if details and hasattr(details, "cached_tokens"):
                usage["cached_content_token_count"] = details.cached_tokens

        return usage

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """
        Execute non-streaming chat completion with cost calculation.

        Extracts Gemini-specific usage data and calculates total cost.
        """
        client = self.get_client()

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(model, messages, system)

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Log finish reason for debugging
        logger.info(
            f"Gemini {model} response: finish_reason='{finish_reason}', "
            f"content_length={len(content)} chars"
        )

        # Log if response was truncated
        if finish_reason and finish_reason != "stop":
            logger.warning(
                f"Gemini {model} finished with reason '{finish_reason}' "
                f"(output may be truncated)"
            )

        result = {
            "content": content,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        # Extract Gemini-specific usage and calculate cost
        if response.usage:
            usage_dict = self._extract_gemini_usage(response)
            cost = self._calculate_gemini_cost(model, usage_dict)
            if cost is not None:
                result["cost"] = cost

            # Extract thinking tokens if available
            if "thoughts_token_count" in usage_dict:
                result["reasoning_tokens"] = usage_dict["thoughts_token_count"]

            # Extract cached tokens if available
            if "cached_content_token_count" in usage_dict:
                result["cached_input_tokens"] = usage_dict["cached_content_token_count"]

        return result

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> Generator[str, None, None]:
        """
        Execute streaming chat completion with usage extraction.

        Captures usage data from the final chunk and calculates cost.
        The result is stored in _last_stream_result for the proxy to retrieve.
        """
        client = self.get_client()

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(model, messages, system)
        kwargs["stream"] = True
        # Request usage info in streaming mode
        kwargs["stream_options"] = {"include_usage": True}

        # Reset stream result
        self._last_stream_result = None
        usage_data = None

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            # Check for usage data (usually in final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                }

                # Extract Gemini-specific fields
                for attr in [
                    "thoughts_token_count",
                    "thinking_tokens",
                    "reasoning_tokens",
                ]:
                    if hasattr(chunk.usage, attr):
                        value = getattr(chunk.usage, attr)
                        if value:
                            usage_data["thoughts_token_count"] = value
                            break

                for attr in ["cached_content_token_count", "cached_tokens"]:
                    if hasattr(chunk.usage, attr):
                        value = getattr(chunk.usage, attr)
                        if value:
                            usage_data["cached_content_token_count"] = value
                            break

                # Check prompt_tokens_details
                if hasattr(chunk.usage, "prompt_tokens_details"):
                    details = chunk.usage.prompt_tokens_details
                    if details and hasattr(details, "cached_tokens"):
                        usage_data["cached_content_token_count"] = details.cached_tokens

            # Yield content
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        # After streaming completes, calculate cost from usage data
        if usage_data:
            cost = self._calculate_gemini_cost(model, usage_data)
            self._last_stream_result = {
                "cost": cost,
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            }

            # Include reasoning tokens if available
            if "thoughts_token_count" in usage_data:
                self._last_stream_result["reasoning_tokens"] = usage_data[
                    "thoughts_token_count"
                ]

            logger.debug(
                f"Gemini streaming completed: {usage_data.get('prompt_tokens', 0)} in, "
                f"{usage_data.get('completion_tokens', 0)} out, cost=${cost or 0:.6f}"
            )
        else:
            logger.warning(
                "Gemini streaming did not return usage data, "
                "falling back to character-based estimation"
            )

    def get_last_stream_result(self) -> dict | None:
        """
        Get the result from the last streaming call.

        Returns dict with 'cost', 'input_tokens', 'output_tokens' if available.
        Used by proxy.py to get accurate token counts instead of char-based estimates.
        """
        return self._last_stream_result
