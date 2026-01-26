"""
Perplexity provider with custom cost calculation.

Perplexity has a complex pricing model that includes:
1. Token costs (input/output per 1M tokens)
2. Per-request fees (tiered by search_context_size)
3. Deep Research specific: citation tokens, search queries, reasoning tokens

This provider extracts all usage data from Perplexity's API response
and calculates accurate costs.
"""

import logging
from typing import Generator

from openai import OpenAI

from .base import ModelInfo, OpenAICompatibleProvider, get_api_key

logger = logging.getLogger(__name__)


class PerplexityProvider(OpenAICompatibleProvider):
    """
    Perplexity provider with custom cost calculation.

    Extends OpenAICompatibleProvider to handle Perplexity's unique pricing:
    - Per-request fees based on search_context_size (low/medium/high)
    - Deep Research: citation tokens, search queries, reasoning tokens
    """

    name = "perplexity"
    display_name = "Perplexity"
    icon = "🟣"  # Perplexity purple
    base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"
    has_custom_cost_calculation = (
        True  # Complex pricing: request fees, citation tokens, etc.
    )

    # Per-request fees (per request, converted from per-1K pricing)
    # Applies to: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro
    # Does NOT apply to: sonar-deep-research
    REQUEST_FEES = {
        "sonar": {"low": 0.005, "medium": 0.008, "high": 0.012},
        "sonar-pro": {"low": 0.006, "medium": 0.010, "high": 0.014},
        "sonar-reasoning": {"low": 0.005, "medium": 0.008, "high": 0.012},
        "sonar-reasoning-pro": {"low": 0.006, "medium": 0.010, "high": 0.014},
    }

    # Deep Research specific costs
    CITATION_COST = 2.00  # per 1M citation tokens
    SEARCH_QUERY_COST = 5.00  # per 1K search queries
    REASONING_COST = 3.00  # per 1M reasoning tokens

    def __init__(
        self,
        name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        models: dict[str, ModelInfo] | None = None,
    ):
        """Initialize the Perplexity provider."""
        super().__init__(
            name=name or self.name,
            base_url=base_url or self.base_url,
            api_key_env=api_key_env or self.api_key_env,
            models=models,
        )
        self._last_stream_result: dict | None = None

    def _calculate_perplexity_cost(
        self,
        model_id: str,
        usage: dict,
        search_context_size: str = "low",
    ) -> float | None:
        """
        Calculate total cost for a Perplexity request.

        Args:
            model_id: The model identifier (e.g., "sonar", "sonar-deep-research")
            usage: Usage dict from API response containing token counts
            search_context_size: "low", "medium", or "high" (default: "low")

        Returns:
            Total cost in USD, or None if cannot calculate
        """
        model_info = self._models.get(model_id)
        if not model_info:
            logger.debug(f"No model info for {model_id}, cannot calculate cost")
            return None

        input_rate = model_info.input_cost or 0
        output_rate = model_info.output_cost or 0

        # Extract token counts from usage dict
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        # 1. Basic token costs
        input_cost = (prompt_tokens / 1_000_000) * input_rate
        output_cost = (completion_tokens / 1_000_000) * output_rate

        # 2. Per-request fee (based on search context size)
        request_fee = 0.0
        if model_id in self.REQUEST_FEES:
            fees = self.REQUEST_FEES[model_id]
            request_fee = fees.get(search_context_size, fees["low"])

        # 3. Deep Research specific costs
        citation_cost = 0.0
        search_cost = 0.0
        reasoning_cost = 0.0

        if model_id == "sonar-deep-research":
            citation_tokens = usage.get("citation_tokens", 0) or 0
            num_queries = usage.get("num_search_queries", 0) or 0
            reasoning_tokens = usage.get("reasoning_tokens", 0) or 0

            citation_cost = (citation_tokens / 1_000_000) * self.CITATION_COST
            search_cost = (num_queries / 1_000) * self.SEARCH_QUERY_COST
            reasoning_cost = (reasoning_tokens / 1_000_000) * self.REASONING_COST

            logger.debug(
                f"Deep Research costs: citation=${citation_cost:.6f} "
                f"({citation_tokens} tokens), search=${search_cost:.6f} "
                f"({num_queries} queries), reasoning=${reasoning_cost:.6f} "
                f"({reasoning_tokens} tokens)"
            )

        total_cost = (
            input_cost
            + output_cost
            + request_fee
            + citation_cost
            + search_cost
            + reasoning_cost
        )

        logger.debug(
            f"Perplexity cost for {model_id}: input=${input_cost:.6f}, "
            f"output=${output_cost:.6f}, request_fee=${request_fee:.6f}, "
            f"total=${total_cost:.6f}"
        )

        return total_cost

    def _extract_search_context_size(self, options: dict) -> str:
        """Extract search_context_size from request options."""
        web_search_options = options.get("web_search_options", {})
        return web_search_options.get("search_context_size", "low")

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        system: str | None,
        options: dict,
    ) -> dict:
        """
        Execute non-streaming chat completion with cost calculation.

        Extracts Perplexity-specific usage data and calculates total cost.
        """
        client = self.get_client()

        kwargs = self._build_kwargs(model, options)
        kwargs["messages"] = self._build_messages(model, messages, system)

        # Pass through web_search_options if present
        if "web_search_options" in options:
            kwargs["web_search_options"] = options["web_search_options"]

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        result = {
            "content": content,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        # Calculate cost from Perplexity usage data
        if response.usage:
            # Build usage dict from response
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

            # Extract Perplexity-specific fields if available
            if hasattr(response.usage, "citation_tokens"):
                usage_dict["citation_tokens"] = response.usage.citation_tokens
            if hasattr(response.usage, "num_search_queries"):
                usage_dict["num_search_queries"] = response.usage.num_search_queries
            if hasattr(response.usage, "reasoning_tokens"):
                usage_dict["reasoning_tokens"] = response.usage.reasoning_tokens

            search_context_size = self._extract_search_context_size(options)
            cost = self._calculate_perplexity_cost(
                model, usage_dict, search_context_size
            )
            if cost is not None:
                result["cost"] = cost

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

        # Pass through web_search_options if present
        if "web_search_options" in options:
            kwargs["web_search_options"] = options["web_search_options"]

        search_context_size = self._extract_search_context_size(options)

        # Reset stream result
        self._last_stream_result = None
        usage_data = None

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            # Check for usage data (usually in final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }
                # Extract Perplexity-specific fields
                if hasattr(chunk.usage, "citation_tokens"):
                    usage_data["citation_tokens"] = chunk.usage.citation_tokens
                if hasattr(chunk.usage, "num_search_queries"):
                    usage_data["num_search_queries"] = chunk.usage.num_search_queries
                if hasattr(chunk.usage, "reasoning_tokens"):
                    usage_data["reasoning_tokens"] = chunk.usage.reasoning_tokens

            # Yield content
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        # After streaming completes, calculate cost from usage data
        if usage_data:
            cost = self._calculate_perplexity_cost(
                model, usage_data, search_context_size
            )
            self._last_stream_result = {
                "cost": cost,
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            }
            logger.debug(
                f"Perplexity streaming completed: {usage_data.get('prompt_tokens', 0)} in, "
                f"{usage_data.get('completion_tokens', 0)} out, cost=${cost:.6f}"
            )
        else:
            logger.warning(
                "Perplexity streaming did not return usage data, "
                "falling back to character-based estimation"
            )

    def get_last_stream_result(self) -> dict | None:
        """
        Get the result from the last streaming call.

        Returns dict with 'cost', 'input_tokens', 'output_tokens' if available.
        Used by proxy.py to get accurate token counts instead of char-based estimates.
        """
        return self._last_stream_result
