"""
Smart Router Engine for intelligent model routing.

Uses a designator LLM to analyze incoming requests and route them
to the most appropriate model from a pool of candidates.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.models import SmartRouter
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of a routing decision."""

    resolved: "ResolvedModel"
    router_name: str
    router_tags: list[str]
    designator_usage: dict | None = None  # Token usage from designator call
    designator_model: str | None = (
        None  # The model used for designation (e.g., "gemini/gemini-flash-latest")
    )


class SmartRouterEngine:
    """
    Engine for smart model routing using LLM-based designator.

    The engine:
    1. Estimates token count from the request
    2. Filters candidates by context length
    3. Calls a designator LLM to select the best model
    4. Validates the response and falls back if needed
    5. Caches decisions for per_session strategy
    """

    # Session cache: {session_key: (model_name, timestamp)}
    _session_cache: dict[str, tuple[str, float]] = {}

    def __init__(self, router: "SmartRouter", registry: "ProviderRegistry"):
        """
        Initialize the routing engine.

        Args:
            router: SmartRouter configuration
            registry: Provider registry for model resolution
        """
        self.router = router
        self.registry = registry

    def route(
        self,
        messages: list[dict],
        system: str | None,
        session_key: str | None = None,
    ) -> RoutingResult:
        """
        Route a request to the appropriate model.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            session_key: Optional session key for per_session caching

        Returns:
            RoutingResult with resolved model and metadata
        """
        from providers.registry import ResolvedModel

        # Check session cache for per_session strategy
        if self.router.strategy == "per_session" and session_key:
            cached = self._get_cached_route(session_key)
            if cached:
                logger.debug(
                    f"Router '{self.router.name}' using cached route: {cached}"
                )
                try:
                    resolved = self.registry._resolve_actual_model(cached)
                    return RoutingResult(
                        resolved=ResolvedModel(
                            provider=resolved.provider,
                            model_id=resolved.model_id,
                            alias_name=self.router.name,
                            alias_tags=self.router.tags,
                        ),
                        router_name=self.router.name,
                        router_tags=self.router.tags,
                        designator_usage=None,
                    )
                except ValueError:
                    # Cached model no longer available, continue with routing
                    logger.warning(
                        f"Cached route '{cached}' no longer available, re-routing"
                    )

        # Estimate token count
        token_count = self._estimate_tokens(messages, system)
        has_images = self._check_for_images(messages)

        # Get candidates with their model info
        candidates = self._get_candidates_with_info()

        # Filter by context length
        filtered_candidates = self._filter_candidates_by_context(
            candidates, token_count
        )

        # If images present, filter to vision-capable models
        if has_images:
            filtered_candidates = [
                c for c in filtered_candidates if "vision" in c.get("capabilities", [])
            ]

        # If no candidates left after filtering, use fallback
        if not filtered_candidates:
            logger.warning(
                f"Router '{self.router.name}' has no viable candidates, using fallback"
            )
            return self._use_fallback()

        # If only one candidate, use it directly (skip designator call)
        if len(filtered_candidates) == 1:
            selected_model = filtered_candidates[0]["model"]
            logger.debug(
                f"Router '{self.router.name}' has single viable candidate: {selected_model}"
            )
        else:
            # Call designator LLM
            selected_model, designator_usage = self._call_designator(
                messages, system, filtered_candidates, token_count, has_images
            )

            # Check if designator failed
            if selected_model is None:
                logger.warning(
                    f"Router '{self.router.name}' designator failed, using fallback"
                )
                return self._use_fallback(designator_usage)

            # Validate response
            valid_models = [c["model"] for c in filtered_candidates]
            if selected_model not in valid_models:
                logger.warning(
                    f"Designator returned invalid model '{selected_model}', using fallback"
                )
                return self._use_fallback(designator_usage)

        # Cache the decision for per_session
        if self.router.strategy == "per_session" and session_key:
            self._cache_route(session_key, selected_model)

        # Resolve the selected model
        try:
            resolved = self.registry._resolve_actual_model(selected_model)
            return RoutingResult(
                resolved=ResolvedModel(
                    provider=resolved.provider,
                    model_id=resolved.model_id,
                    alias_name=self.router.name,
                    alias_tags=self.router.tags,
                ),
                router_name=self.router.name,
                router_tags=self.router.tags,
                designator_usage=designator_usage
                if len(filtered_candidates) > 1
                else None,
                designator_model=self.router.designator_model
                if len(filtered_candidates) > 1
                else None,
            )
        except ValueError:
            logger.warning(
                f"Selected model '{selected_model}' not available, using fallback"
            )
            return self._use_fallback(
                designator_usage if len(filtered_candidates) > 1 else None
            )

    def _use_fallback(self, designator_usage: dict | None = None) -> RoutingResult:
        """Use the fallback model."""
        from providers.registry import ResolvedModel

        try:
            resolved = self.registry._resolve_actual_model(self.router.fallback_model)
            return RoutingResult(
                resolved=ResolvedModel(
                    provider=resolved.provider,
                    model_id=resolved.model_id,
                    alias_name=self.router.name,
                    alias_tags=self.router.tags,
                ),
                router_name=self.router.name,
                router_tags=self.router.tags,
                designator_usage=designator_usage,
                designator_model=self.router.designator_model
                if designator_usage
                else None,
            )
        except ValueError:
            # Fallback also failed - use system default
            logger.error(
                f"Router '{self.router.name}' fallback '{self.router.fallback_model}' "
                "not available, using system default"
            )
            default_resolved = self.registry._get_default_model()
            return RoutingResult(
                resolved=ResolvedModel(
                    provider=default_resolved.provider,
                    model_id=default_resolved.model_id,
                    alias_name=self.router.name,
                    alias_tags=self.router.tags,
                ),
                router_name=self.router.name,
                router_tags=self.router.tags,
                designator_usage=designator_usage,
                designator_model=self.router.designator_model
                if designator_usage
                else None,
            )

    def _estimate_tokens(self, messages: list[dict], system: str | None) -> int:
        """
        Estimate token count using char/4 approximation.

        This is a rough estimate suitable for context length filtering.
        """
        total_chars = 0

        if system:
            total_chars += len(system)

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Content blocks (text, images, etc.)
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total_chars += len(block.get("text", ""))
                        # Images don't contribute to text token count
                    elif isinstance(block, str):
                        total_chars += len(block)

        return total_chars // 4

    def _check_for_images(self, messages: list[dict]) -> bool:
        """Check if any messages contain images."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in ("image", "image_url"):
                            return True
                        # Check for images field (Ollama format)
            if msg.get("images"):
                return True
        return False

    def _get_candidates_with_info(self) -> list[dict]:
        """
        Get candidate models with their metadata.

        Returns list of dicts with model info including context_length,
        capabilities, costs, and descriptions.
        """
        candidates = []

        for candidate in self.router.candidates:
            model_ref = candidate.get("model", "")
            notes = candidate.get("notes", "")

            try:
                # Try to resolve and get model info
                resolved = self.registry._resolve_actual_model(model_ref)
                model_info = resolved.provider.get_models().get(resolved.model_id)

                if model_info:
                    candidates.append(
                        {
                            "model": model_ref,
                            "notes": notes,
                            "context_length": model_info.context_length,
                            "capabilities": model_info.capabilities,
                            "input_cost": model_info.input_cost,
                            "output_cost": model_info.output_cost,
                            "description": getattr(model_info, "description", None),
                        }
                    )
                else:
                    # Model exists but no info - include with defaults
                    candidates.append(
                        {
                            "model": model_ref,
                            "notes": notes,
                            "context_length": 128000,  # Assume reasonable default
                            "capabilities": [],
                            "input_cost": None,
                            "output_cost": None,
                            "description": None,
                        }
                    )
            except ValueError:
                # Model not available - skip it
                logger.debug(f"Candidate model '{model_ref}' not available, skipping")
                continue

        return candidates

    def _filter_candidates_by_context(
        self, candidates: list[dict], token_count: int
    ) -> list[dict]:
        """
        Filter candidates that have insufficient context length.

        Uses a 1.5x safety margin to account for response tokens.
        """
        min_context = int(token_count * 1.5)
        return [c for c in candidates if c.get("context_length", 0) >= min_context]

    def _call_designator(
        self,
        messages: list[dict],
        system: str | None,
        candidates: list[dict],
        token_count: int,
        has_images: bool,
    ) -> tuple[str, dict]:
        """
        Call the designator LLM to select a model.

        Returns (selected_model, usage_dict).
        """
        # Build the designator prompt
        prompt = self._build_designator_prompt(
            messages, candidates, token_count, has_images
        )

        # Resolve the designator model
        try:
            designator_resolved = self.registry._resolve_actual_model(
                self.router.designator_model
            )
        except ValueError:
            logger.error(
                f"Designator model '{self.router.designator_model}' not available"
            )
            # Return None to signal failure - caller will use configured fallback
            return None, {
                "error": f"Designator model '{self.router.designator_model}' not available"
            }

        # Call the designator
        try:
            # Disable reasoning/thinking for designator - we just want a direct answer
            result = designator_resolved.provider.chat_completion(
                model=designator_resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,  # System is built into the prompt
                options={
                    "max_tokens": 100,
                    "temperature": 0,
                    "reasoning_effort": "none",  # Disable reasoning for OpenAI o-series
                },
            )

            # Some models (like o1/o3) put the answer in reasoning_content instead of content
            selected = result.get("content", "").strip()
            if not selected:
                # Try reasoning_content as fallback for reasoning models
                selected = result.get("reasoning_content", "").strip()
                # If still empty but we have reasoning, the model may have just "thought" without answering
                if not selected and result.get("reasoning_tokens", 0) > 0:
                    logger.warning(
                        f"Designator used {result.get('reasoning_tokens')} reasoning tokens but produced no output. "
                        "Consider using a non-reasoning model as designator (e.g., gpt-4o-mini)."
                    )

            usage = {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "cost": result.get("cost"),
            }

            logger.info(f"Router '{self.router.name}' designator selected: {selected}")
            logger.debug(f"Designator full response: {result}")
            if not selected:
                logger.warning(
                    f"Designator returned empty response. Full result: {result}"
                )
            return selected, usage

        except Exception as e:
            logger.error(f"Designator call failed: {e}")
            # Return None to signal failure - caller will use configured fallback
            return None, {"error": str(e)}

    def _build_designator_prompt(
        self,
        messages: list[dict],
        candidates: list[dict],
        token_count: int,
        has_images: bool,
    ) -> str:
        """Build the prompt for the designator LLM."""
        # Build model list with notes (which may contain synced descriptions or custom notes)
        model_lines = []
        for c in candidates:
            caps = ", ".join(c.get("capabilities", [])) or "general"
            cost_str = ""
            if c.get("input_cost") is not None and c.get("output_cost") is not None:
                cost_str = f" | Cost: ${c['input_cost']:.2f}/${c['output_cost']:.2f} per 1M tokens"

            line = f"- {c['model']}\n  Context: {c['context_length']} tokens{cost_str}\n  Capabilities: {caps}"

            # Add notes if available (may contain synced description or custom notes)
            notes = c.get("notes")
            if notes:
                # Truncate long notes to keep prompt size reasonable
                if len(notes) > 300:
                    notes = notes[:297] + "..."
                line += f"\n  Notes: {notes}"
            model_lines.append(line)

        models_section = "\n".join(model_lines)

        # Get query preview (truncated for long messages)
        query_preview = self._get_query_preview(messages)

        prompt = f"""You are a model routing assistant. Select the best model for the user's query.

PURPOSE: {self.router.purpose}

AVAILABLE MODELS:
{models_section}

QUERY INFO:
- Messages: {len(messages)}
- Estimated tokens: {token_count}
- Contains images: {has_images}
- Preview: {query_preview}

Respond with ONLY the model identifier (e.g., "anthropic/claude-sonnet-4-20250514"). No explanation."""

        return prompt

    def _get_query_preview(self, messages: list[dict], max_chars: int = 1000) -> str:
        """
        Get a preview of the query for the designator.

        For short queries, returns full content.
        For long queries, returns first 500 + last 500 chars.
        """
        # Combine all user message content
        user_content = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_content.append(block.get("text", ""))
                        elif isinstance(block, str):
                            user_content.append(block)

        full_text = "\n".join(user_content)

        if len(full_text) <= max_chars:
            return full_text

        # Truncate: first 500 + "..." + last 500
        half = max_chars // 2
        return f"{full_text[:half]}\n[... {len(full_text) - max_chars} chars truncated ...]\n{full_text[-half:]}"

    def _get_cached_route(self, session_key: str) -> str | None:
        """Get cached route for session if still valid."""
        if session_key not in self._session_cache:
            return None

        model, timestamp = self._session_cache[session_key]
        if time.time() - timestamp > self.router.session_ttl:
            # Cache expired
            del self._session_cache[session_key]
            return None

        return model

    def _cache_route(self, session_key: str, model: str) -> None:
        """Cache a routing decision for the session."""
        self._session_cache[session_key] = (model, time.time())

        # Clean up old cache entries periodically
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Remove expired entries from session cache."""
        now = time.time()
        # Use a reasonable max TTL for cleanup (1 hour)
        max_ttl = 3600

        expired = [
            key
            for key, (_, timestamp) in self._session_cache.items()
            if now - timestamp > max_ttl
        ]

        for key in expired:
            del self._session_cache[key]


def get_session_key(client_ip: str, user_agent: str | None = None) -> str:
    """
    Generate a session key from client info.

    Uses client IP and user agent hash for implicit session tracking.
    """
    data = client_ip
    if user_agent:
        data += user_agent

    return hashlib.sha256(data.encode()).hexdigest()[:16]
