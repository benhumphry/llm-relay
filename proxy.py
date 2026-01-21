#!/usr/bin/env python3
"""
LLM Relay - Multi-Provider Proxy

Presents multiple LLM providers (Anthropic, OpenAI, Google Gemini, Perplexity, etc.)
via both Ollama and OpenAI-compatible API interfaces.

This allows any Ollama or OpenAI-compatible application to use models from
multiple providers seamlessly.

The proxy runs two separate servers:
- API server (default port 11434): Ollama and OpenAI compatible endpoints
- Admin server (default port 8080): Web UI for configuration and management
"""

import json
import logging
import os
import random
import string
import threading
import time
from datetime import datetime, timezone
from typing import Generator

from flask import Flask, Response, jsonify, request

# Import database and admin
from db import ensure_seeded, init_db
from db.connection import get_db_context

# Import the provider registry and registration function
from providers import register_all_providers, registry

# Import routing utilities for smart routers
from routing import get_session_key

# Import usage tracking
from tracking import (
    extract_relay_commands_from_messages,
    extract_tag,
    get_client_ip,
    normalize_tags,
    resolve_hostname,
    tracker,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Install debug log handler for admin UI streaming
from admin.debug_logs import install_debug_handler

install_debug_handler()

# Debug mode for response comparison
DEBUG_RESPONSES = os.environ.get("DEBUG_RESPONSES", "false").lower() == "true"


def debug_compare_response(endpoint: str, llm_response: str, client_response: str):
    """Compare LLM response with what we send to client, showing diff if different."""
    if not DEBUG_RESPONSES:
        return

    logger.info("=" * 80)
    logger.info(f"DEBUG RESPONSE COMPARISON [{endpoint}]")
    logger.info("=" * 80)

    # Show raw LLM response
    logger.info("--- RAW LLM RESPONSE ---")
    logger.info(llm_response[:2000] if len(llm_response) > 2000 else llm_response)
    if len(llm_response) > 2000:
        logger.info(f"... (truncated, total length: {len(llm_response)} chars)")

    logger.info("")
    logger.info("--- CLIENT RESPONSE ---")
    logger.info(
        client_response[:2000] if len(client_response) > 2000 else client_response
    )
    if len(client_response) > 2000:
        logger.info(f"... (truncated, total length: {len(client_response)} chars)")

    # Compare and show differences
    if llm_response == client_response:
        logger.info("")
        logger.info(">>> MATCH: LLM response and client response are IDENTICAL")
    else:
        logger.info("")
        logger.info(">>> DIFFERENCE DETECTED:")
        logger.info(f"    LLM response length:    {len(llm_response)} chars")
        logger.info(f"    Client response length: {len(client_response)} chars")

        # Show character-level diff for first difference
        min_len = min(len(llm_response), len(client_response))
        for i in range(min_len):
            if llm_response[i] != client_response[i]:
                context_start = max(0, i - 20)
                context_end = min(min_len, i + 20)
                logger.info(f"    First difference at position {i}:")
                logger.info(
                    f"    LLM:    ...{repr(llm_response[context_start:context_end])}..."
                )
                logger.info(
                    f"    Client: ...{repr(client_response[context_start:context_end])}..."
                )
                break
        else:
            # No char difference found - one is a prefix of the other
            if len(llm_response) > len(client_response):
                logger.info(
                    f"    Client response is TRUNCATED (missing {len(llm_response) - len(client_response)} chars)"
                )
                logger.info(
                    f"    Missing content: {repr(llm_response[len(client_response) : len(client_response) + 100])}..."
                )
            else:
                logger.info(
                    f"    Client response has EXTRA content ({len(client_response) - len(llm_response)} chars)"
                )

    logger.info("=" * 80)


def create_api_app():
    """Create the API Flask application (Ollama/OpenAI compatible endpoints)."""
    application = Flask(__name__)

    # Initialize database and seed default data (including YAML overrides)
    init_db()
    ensure_seeded()

    # Register providers AFTER seeding so YAML overrides are applied first
    register_all_providers()

    # Load action handlers for Smart Actions
    try:
        from actions import load_action_handlers

        load_action_handlers()
    except Exception as e:
        logger.warning(f"Failed to load action handlers: {e}")

    return application


def create_admin_app():
    """Create the Admin Flask application (Web UI)."""
    from admin import create_admin_blueprint
    from admin.auth import get_session_secret

    # Disable default static folder to avoid conflict with blueprint's static route
    application = Flask(__name__, static_folder=None)

    # Initialize database (idempotent - providers already registered by create_api_app)
    init_db()

    # Configure session for admin authentication
    application.secret_key = get_session_secret()

    # Register admin blueprint at root (since this is a dedicated admin server)
    admin_blueprint = create_admin_blueprint(url_prefix="")
    application.register_blueprint(admin_blueprint)

    return application


# Create the API app - database will be initialized here
app = create_api_app()

# Start usage tracker
tracker.start()


# ============================================================================
# Request Logging Middleware
# ============================================================================


@app.before_request
def log_request():
    """Log all incoming requests and set up tracking context."""
    from flask import g

    logger.info(f">>> {request.method} {request.path}")

    # Set up tracking context using Flask's request-scoped g object
    g.start_time = time.time()
    g.client_ip = get_client_ip(request)

    if request.data:
        try:
            data = request.get_json(silent=True)
            if data:
                # Truncate large messages for logging
                log_data = {
                    k: (v if k != "messages" else f"[{len(v)} messages]")
                    for k, v in data.items()
                }
                logger.info(f"    Request data: {log_data}")
        except Exception:
            pass


@app.after_request
def log_response(response):
    """Log response status for debugging."""
    logger.info(f"<<< {response.status_code} {request.path}")
    return response


def track_completion(
    provider_id: str,
    model_id: str,
    model_name: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    status_code: int,
    error_message: str | None = None,
    is_streaming: bool = False,
    cost: float | None = None,
    # Extended token tracking (v2.2.3)
    reasoning_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    # Alias tracking (v3.1)
    tag: str | None = None,
    alias: str | None = None,
    # Smart router tracking (v3.2)
    is_designator: bool = False,
    router_name: str | None = None,
    # Cache tracking (v1.6)
    is_cache_hit: bool = False,
    cache_name: str | None = None,
    cache_tokens_saved: int = 0,
    cache_cost_saved: float = 0.0,
    # Request type tracking (v3.11)
    request_type: str = "main",
):
    """
    Track a completed request.

    Args:
        provider_id: Provider that handled the request
        model_id: Actual model ID used
        model_name: Original model name from request (for tag extraction)
        endpoint: API endpoint called
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        status_code: HTTP status code
        error_message: Error message if request failed
        is_streaming: Whether this was a streaming request
        cost: Actual cost from provider (if available, e.g., OpenRouter)
        reasoning_tokens: OpenAI reasoning tokens (o1/o3 models)
        cached_input_tokens: OpenAI cached prompt tokens
        cache_creation_tokens: Anthropic cache creation tokens
        cache_read_tokens: Anthropic cache read tokens
        tag: Pre-computed tag (with alias tags merged) - if None, extracted from request
        alias: Alias name if request used an alias (v3.1)
        is_designator: Whether this was a designator call (v3.2)
        router_name: Smart router name if request used a router (v3.2)
        is_cache_hit: Whether this was served from cache (v1.6)
        cache_name: Name of cache entity if cache hit
        cache_tokens_saved: Output tokens saved by cache hit
        cache_cost_saved: Estimated cost saved by cache hit
        request_type: Type of request (v3.11): "inbound", "main", "designator", "embedding"
    """
    from flask import g

    start_time = getattr(g, "start_time", None)
    if start_time is None:
        logger.warning("No start_time found in Flask g - using current time")
        start_time = time.time()

    response_time_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"Track: {provider_id}/{model_id} - {response_time_ms}ms, streaming={is_streaming}"
    )

    client_ip = getattr(g, "client_ip", "unknown")

    # Extract tag from request if not provided
    if tag is None:
        tag, _ = extract_tag(request, model_name)

    # Resolve hostname (cached)
    hostname = resolve_hostname(client_ip)

    tracker.log_request(
        timestamp=datetime.now(timezone.utc),
        client_ip=client_ip,
        hostname=hostname,
        tag=tag,
        provider_id=provider_id,
        model_id=model_id,
        endpoint=endpoint,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time_ms=response_time_ms,
        status_code=status_code,
        error_message=error_message,
        is_streaming=is_streaming,
        alias=alias,
        cost=cost,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        is_designator=is_designator,
        router_name=router_name,
        is_cache_hit=is_cache_hit,
        cache_name=cache_name,
        cache_tokens_saved=cache_tokens_saved,
        cache_cost_saved=cache_cost_saved,
        request_type=request_type,
    )


def log_designator_usage(resolved, endpoint: str, tag: str = ""):
    """
    Log designator usage for smart router requests.

    This logs the designator LLM call separately from the main request,
    allowing tracking of routing overhead costs and failures.

    Args:
        resolved: The resolved model with routing information
        endpoint: The API endpoint being called
        tag: Request tags to attribute the designator cost to
    """
    if not resolved.has_router or not resolved.designator_usage:
        return

    usage = resolved.designator_usage

    # Check if designator failed
    designator_error = usage.get("error")

    # Skip logging if no tokens and no error
    if (
        not usage.get("input_tokens")
        and not usage.get("output_tokens")
        and not designator_error
    ):
        return

    from flask import g

    client_ip = getattr(g, "client_ip", "unknown")
    hostname = resolve_hostname(client_ip)

    # Parse designator model to get provider/model
    designator_model = resolved._routing_result.designator_model or ""
    if "/" in designator_model:
        des_provider, des_model = designator_model.split("/", 1)
    else:
        des_provider = "designator"
        des_model = designator_model or resolved._routing_result.router_name

    # Log the designator call as a separate request entry
    tracker.log_request(
        timestamp=datetime.now(timezone.utc),
        client_ip=client_ip,
        hostname=hostname,
        tag=tag,
        provider_id=des_provider,
        model_id=des_model,
        endpoint=endpoint,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        response_time_ms=0,  # Not tracked separately
        status_code=500 if designator_error else 200,
        error_message=designator_error,
        is_streaming=False,
        cost=usage.get("cost"),
        is_designator=True,
        router_name=resolved.router_name,
        request_type="designator",
    )


def log_inbound_request(
    model_name: str,
    endpoint: str,
    tag: str | None = None,
):
    """
    Log an inbound client request before processing.

    This creates a request_type="inbound" entry to track what clients
    are requesting, separate from the actual LLM calls made.

    Args:
        model_name: The model name requested by the client (before resolution)
        endpoint: The API endpoint being called
        tag: Request tag (if already extracted)
    """
    from flask import g

    client_ip = getattr(g, "client_ip", "unknown")
    hostname = resolve_hostname(client_ip)

    # Extract tag if not provided
    if tag is None:
        tag, _ = extract_tag(request, model_name)

    tracker.log_request(
        timestamp=datetime.now(timezone.utc),
        client_ip=client_ip,
        hostname=hostname,
        tag=tag,
        provider_id="client",  # Special provider for inbound
        model_id=model_name,  # The requested model (before resolution)
        endpoint=endpoint,
        input_tokens=0,
        output_tokens=0,
        response_time_ms=0,
        status_code=0,  # Not yet known
        error_message=None,
        is_streaming=False,
        request_type="inbound",
    )


# ============================================================================
# Error Handlers - Return JSON instead of HTML for all errors
# ============================================================================


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 Bad Request errors."""
    return jsonify(
        {"error": str(e.description) if hasattr(e, "description") else "Bad request"}
    ), 400


@app.errorhandler(404)
def not_found(e):
    """Handle 404 Not Found errors."""
    return jsonify({"error": f"Endpoint not found: {request.path}"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 Method Not Allowed errors."""
    return jsonify(
        {"error": f"Method {request.method} not allowed for {request.path}"}
    ), 405


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 Internal Server errors."""
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions."""
    logger.exception(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500


# ============================================================================
# Message Conversion Utilities
# ============================================================================


def provider_supports_vision(provider, model_id: str) -> bool:
    """Check if a provider/model supports vision (image inputs)."""
    models = provider.get_models()
    model_info = models.get(model_id)
    if model_info and hasattr(model_info, "capabilities"):
        return "vision" in model_info.capabilities
    return False


def filter_images_from_messages(messages: list) -> list:
    """Remove image content from messages, keeping only text."""
    filtered = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Filter out image blocks, keep only text
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                # Skip image types
            # Combine text parts
            filtered.append({"role": role, "content": " ".join(text_parts)})
        else:
            filtered.append(msg)

    return filtered


def convert_ollama_messages(ollama_messages: list) -> tuple[str | None, list]:
    """
    Convert Ollama message format to provider-agnostic format.

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = None
    messages = []

    for msg in ollama_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role in ("user", "assistant"):
            # Handle images if present
            if "images" in msg and msg["images"]:
                content_blocks = []
                for img in msg["images"]:
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img,
                            },
                        }
                    )
                content_blocks.append({"type": "text", "text": content})
                messages.append({"role": role, "content": content_blocks})
            else:
                messages.append({"role": role, "content": content})

    return system_prompt, messages


def convert_openai_messages(openai_messages: list) -> tuple[str | None, list]:
    """
    Convert OpenAI message format to provider-agnostic format.

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = None
    messages = []

    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                system_prompt = " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
        elif role in ("user", "assistant"):
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                content_blocks = []
                for part in content:
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        content_blocks.append(
                            {"type": "text", "text": part.get("text", "")}
                        )
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = (
                            image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else image_url
                        )

                        if url.startswith("data:"):
                            try:
                                header, data = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                content_blocks.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        },
                                    }
                                )
                            except (ValueError, IndexError):
                                logger.warning("Failed to parse image data URL")
                        else:
                            content_blocks.append(
                                {"type": "image", "source": {"type": "url", "url": url}}
                            )

                if content_blocks:
                    messages.append({"role": role, "content": content_blocks})

    return system_prompt, messages


def generate_openai_id(prefix: str = "chatcmpl") -> str:
    """Generate an OpenAI-style ID."""
    chars = string.ascii_letters + string.digits
    suffix = "".join(random.choices(chars, k=24))
    return f"{prefix}-{suffix}"


# ============================================================================
# Anthropic Message Conversion
# ============================================================================


def convert_anthropic_messages(
    anthropic_messages: list, system: str | None = None
) -> tuple[str | None, list]:
    """
    Convert Anthropic message format to provider-agnostic format.

    Anthropic format differences:
    - System prompt is a separate field, not in messages array
    - Content can be string or array of content blocks
    - Images use {"type": "image", "source": {"type": "base64", ...}}

    Returns (system_prompt, messages) tuple.
    """
    messages = []

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role in ("user", "assistant"):
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Anthropic content blocks - handle each type
                content_blocks = []
                for block in content:
                    block_type = block.get("type", "text")
                    if block_type == "text":
                        content_blocks.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
                    elif block_type == "image":
                        # Already in correct internal format
                        content_blocks.append(block)
                    elif block_type == "tool_use":
                        # Pass through for tool support
                        content_blocks.append(block)
                    elif block_type == "tool_result":
                        content_blocks.append(block)
                if content_blocks:
                    messages.append({"role": role, "content": content_blocks})

    return system, messages


def generate_anthropic_id(prefix: str = "msg") -> str:
    """Generate an Anthropic-style message ID."""
    chars = string.ascii_letters + string.digits
    suffix = "".join(random.choices(chars, k=24))
    return f"{prefix}_{suffix}"


def map_finish_reason_to_anthropic(finish_reason: str | None) -> str:
    """Map internal/OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",  # Best approximation
        None: "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


def build_anthropic_response(
    message_id: str,
    model: str,
    content: str,
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Build non-streaming Anthropic response."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def anthropic_error_response(
    error_type: str,
    message: str,
    status_code: int = 400,
) -> tuple[Response, int]:
    """Return error in Anthropic format."""
    return (
        jsonify(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            }
        ),
        status_code,
    )


# ============================================================================
# Response Formatters
# ============================================================================


def estimate_input_chars(messages: list, system: str | None = None) -> int:
    """Estimate input character count from messages and system prompt."""
    total = 0
    if system:
        total += len(system)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            # Multi-modal content
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
    return total


def build_source_attribution(enrichment_result) -> str | None:
    """
    Build a source attribution string showing budget percentages.

    Returns something like:
    "\n\n---\nSources: my-docs (50%), company-wiki (30%), web (20%)"
    """
    if not enrichment_result or not enrichment_result.show_sources:
        return None

    if not enrichment_result.context_injected:
        return None

    parts = []
    total_budget = enrichment_result.total_budget or 4000

    # Get store names for the budget display
    store_budgets = enrichment_result.store_budgets
    store_id_to_name = enrichment_result.store_id_to_name or {}
    stores_queried = enrichment_result.stores_queried or []
    web_budget = enrichment_result.web_budget

    if store_budgets and store_id_to_name:
        # Smart source selection with budgets - only show stores that actually
        # had chunks retrieved (are in stores_queried)
        queried_names = set(stores_queried)

        # Sort by budget descending so highest priority sources appear first
        sorted_stores = sorted(store_budgets.items(), key=lambda x: x[1], reverse=True)
        for store_id, store_budget in sorted_stores:
            store_name = store_id_to_name.get(store_id, f"store_{store_id}")
            # Only include if store actually had chunks retrieved
            if store_name in queried_names:
                parts.append(f"{store_name} ({store_budget})")
    elif stores_queried:
        # No budgets (legacy mode) - just list stores
        parts.extend(stores_queried)

    if web_budget and web_budget > 0:
        parts.append(f"web ({web_budget})")
    elif enrichment_result.scraped_urls:
        # Web was used but no budget info
        parts.append("web")

    if not parts:
        return None

    return f"\n\n---\nSources: {', '.join(parts)}"


class ActionStreamFilter:
    """
    Filters smart_action blocks from streaming responses.

    Buffers content when it detects the start of an action block (<smart_action)
    and holds it until the block is complete (</smart_action>). Non-action content
    is passed through immediately.
    """

    def __init__(self, strip_actions: bool = False):
        self.strip_actions = strip_actions
        self.buffer = ""
        self.in_action_block = False
        self.action_blocks = []  # Collected complete action blocks

    def process(self, text: str) -> str:
        """
        Process incoming text chunk.

        Returns text that should be sent to client (may be empty if buffering).
        """
        if not self.strip_actions:
            return text

        self.buffer += text
        output = ""

        while True:
            if not self.in_action_block:
                # Look for start of action block
                start_idx = self.buffer.find("<smart_action")
                if start_idx == -1:
                    # No action start found - but keep last 20 chars in case
                    # "<smart_action" is split across chunks
                    safe_len = max(0, len(self.buffer) - 20)
                    output += self.buffer[:safe_len]
                    self.buffer = self.buffer[safe_len:]
                    break
                else:
                    # Found start - output everything before it
                    output += self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx:]
                    self.in_action_block = True
            else:
                # Look for end of action block
                end_idx = self.buffer.find("</smart_action>")
                if end_idx == -1:
                    # End not found yet - keep buffering
                    break
                else:
                    # Found end - extract and store the action block
                    end_pos = end_idx + len("</smart_action>")
                    action_block = self.buffer[:end_pos]
                    self.action_blocks.append(action_block)
                    self.buffer = self.buffer[end_pos:]
                    self.in_action_block = False

        return output

    def flush(self) -> str:
        """
        Flush any remaining buffer at end of stream.

        Returns remaining content (may include incomplete action block if malformed).
        """
        if not self.strip_actions:
            return ""

        # If we're still in an action block, the LLM output was malformed
        # Include it in output so nothing is silently lost
        remaining = self.buffer
        self.buffer = ""
        return remaining


def _stream_response_base(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    format_chunk: callable,
    format_final: callable,
    format_error: callable,
    debug_endpoint: str,
    on_complete: callable = None,
    input_char_count: int = 0,
    source_attribution: str | None = None,
    strip_actions: bool = False,
) -> Generator[str, None, None]:
    """
    Base streaming function with shared logic.

    Args:
        format_chunk: (text) -> str - Format a content chunk
        format_final: (output_chars, input_char_count) -> list[str] - Format final chunk(s)
        format_error: (error_msg) -> str - Format error response
        debug_endpoint: Endpoint name for debug logging
        source_attribution: Optional text to append at end of response (e.g., "Sources: doc1, doc2")
        strip_actions: If True, filter out <smart_action> blocks from client stream
    """
    output_chars = 0
    llm_chunks = []
    client_chunks = []
    action_filter = ActionStreamFilter(strip_actions=strip_actions)

    try:
        for text in provider.chat_completion_stream(model, messages, system, options):
            output_chars += len(text)
            llm_chunks.append(text)

            # Filter actions if enabled
            filtered_text = action_filter.process(text)
            if filtered_text:
                client_chunks.append(filtered_text)
                yield format_chunk(filtered_text)

        # Flush any remaining buffered content
        remaining = action_filter.flush()
        if remaining:
            client_chunks.append(remaining)
            yield format_chunk(remaining)

        # Append source attribution if provided
        if source_attribution:
            output_chars += len(source_attribution)
            client_chunks.append(source_attribution)
            yield format_chunk(source_attribution)

        # Yield final chunk(s)
        for final_str in format_final(output_chars, input_char_count):
            yield final_str

        # Debug comparison
        llm_full = "".join(llm_chunks)
        client_full = "".join(client_chunks)
        debug_compare_response(debug_endpoint, llm_full, client_full)

        # Get streaming result (cost, tokens) if provider supports it
        stream_result = None
        if hasattr(provider, "get_last_stream_result"):
            stream_result = provider.get_last_stream_result()

        # Call completion callback with full LLM content (includes action blocks)
        if on_complete:
            input_tokens = (
                stream_result.get("input_tokens")
                if stream_result and stream_result.get("input_tokens")
                else input_char_count // 4
            )
            output_tokens = (
                stream_result.get("output_tokens")
                if stream_result and stream_result.get("output_tokens")
                else output_chars // 4
            )
            cost = stream_result.get("cost") if stream_result else None
            on_complete(
                input_tokens,
                output_tokens,
                cost=cost,
                stream_result=stream_result,
                full_content=llm_full,
                action_blocks=action_filter.action_blocks if strip_actions else None,
            )

    except Exception as e:
        logger.error(f"Provider error during streaming: {e}")
        yield format_error(str(e))
        if on_complete:
            on_complete(0, 0, error=str(e), stream_result=None)


def stream_ollama_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    on_complete: callable = None,
    input_char_count: int = 0,
    source_attribution: str | None = None,
    strip_actions: bool = False,
) -> Generator[str, None, None]:
    """Stream response in Ollama NDJSON format."""

    def format_chunk(text: str) -> str:
        return (
            json.dumps(
                {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": text},
                    "done": False,
                }
            )
            + "\n"
        )

    def format_final(output_chars: int, input_char_count: int) -> list[str]:
        return [
            json.dumps(
                {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": input_char_count // 4,
                    "eval_count": output_chars // 4,
                    "eval_duration": 0,
                }
            )
            + "\n"
        ]

    def format_error(error_msg: str) -> str:
        return json.dumps({"error": error_msg, "done": True}) + "\n"

    yield from _stream_response_base(
        provider,
        model,
        messages,
        system,
        options,
        format_chunk,
        format_final,
        format_error,
        "/api/chat (stream)",
        on_complete,
        input_char_count,
        source_attribution,
        strip_actions,
    )


def stream_openai_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    request_model: str,
    on_complete: callable = None,
    input_char_count: int = 0,
    source_attribution: str | None = None,
    strip_actions: bool = False,
) -> Generator[str, None, None]:
    """Stream response in OpenAI SSE format."""
    response_id = generate_openai_id()
    created = int(time.time())

    def format_chunk(text: str) -> str:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": None}
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_final(output_chars: int, input_char_count: int) -> list[str]:
        final_chunk = json.dumps(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        return [f"data: {final_chunk}\n\n", "data: [DONE]\n\n"]

    def format_error(error_msg: str) -> str:
        return f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error'}})}\n\n"

    yield from _stream_response_base(
        provider,
        model,
        messages,
        system,
        options,
        format_chunk,
        format_final,
        format_error,
        "/v1/chat/completions (stream)",
        on_complete,
        input_char_count,
        source_attribution,
        strip_actions,
    )


def stream_anthropic_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    request_model: str,
    on_complete: callable = None,
    input_char_count: int = 0,
    source_attribution: str | None = None,
    strip_actions: bool = False,
) -> Generator[str, None, None]:
    """
    Stream response in Anthropic SSE format with named events.

    Anthropic streaming uses named events in this sequence:
    1. message_start - Initial message metadata
    2. content_block_start - Start of content block
    3. content_block_delta - Text chunks (multiple)
    4. content_block_stop - End of content block
    5. message_delta - Final metadata (stop_reason, usage)
    6. message_stop - Stream termination
    """
    message_id = generate_anthropic_id()
    output_chars = 0
    llm_chunks = []
    client_chunks = []
    action_filter = ActionStreamFilter(strip_actions=strip_actions)
    content_started = False

    try:
        # message_start event
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_char_count // 4,
                    "output_tokens": 0,
                },
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Stream content from provider
        for text in provider.chat_completion_stream(model, messages, system, options):
            output_chars += len(text)
            llm_chunks.append(text)

            # Filter actions if enabled
            filtered_text = action_filter.process(text)
            if filtered_text:
                # Start content block on first text
                if not content_started:
                    content_block_start = {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
                    content_started = True

                client_chunks.append(filtered_text)
                delta = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": filtered_text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"

        # Flush any remaining buffered content
        remaining = action_filter.flush()
        if remaining:
            if not content_started:
                content_block_start = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
                content_started = True
            client_chunks.append(remaining)
            delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": remaining},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"

        # Append source attribution if provided
        if source_attribution:
            if not content_started:
                content_block_start = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
                content_started = True
            output_chars += len(source_attribution)
            client_chunks.append(source_attribution)
            delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": source_attribution},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"

        # Ensure we started content block (even for empty response)
        if not content_started:
            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

        # content_block_stop event
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

        # Get streaming result (cost, tokens) if provider supports it
        stream_result = None
        if hasattr(provider, "get_last_stream_result"):
            stream_result = provider.get_last_stream_result()

        # Calculate tokens
        input_tokens = (
            stream_result.get("input_tokens")
            if stream_result and stream_result.get("input_tokens")
            else input_char_count // 4
        )
        output_tokens = (
            stream_result.get("output_tokens")
            if stream_result and stream_result.get("output_tokens")
            else output_chars // 4
        )

        # message_delta event (with stop_reason and usage)
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

        # message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        # Debug comparison
        llm_full = "".join(llm_chunks)
        client_full = "".join(client_chunks)
        debug_compare_response("/v1/messages (stream)", llm_full, client_full)

        # Call completion callback with full LLM content
        if on_complete:
            cost = stream_result.get("cost") if stream_result else None
            on_complete(
                input_tokens,
                output_tokens,
                cost=cost,
                stream_result=stream_result,
                full_content=llm_full,
                action_blocks=action_filter.action_blocks if strip_actions else None,
            )

    except Exception as e:
        logger.error(f"Provider error during Anthropic streaming: {e}")
        # Emit error event
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        if on_complete:
            on_complete(0, 0, error=str(e), stream_result=None)


# ============================================================================
# Ollama API Endpoints
# ============================================================================


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return "Ollama is running"


def generate_fake_digest(name: str) -> str:
    """Generate a fake SHA256-like digest from a model name for Ollama compatibility."""
    import hashlib

    return hashlib.sha256(name.encode()).hexdigest()[:12]


@app.route("/api/tags", methods=["GET"])
def list_models():
    """List available models in Ollama format."""
    all_models = registry.list_all_models()

    models = []
    for model in all_models:
        model_name = model["name"]
        # Ollama models typically have :latest suffix
        model_name_with_tag = f"{model_name}:latest"
        models.append(
            {
                "name": model_name_with_tag,
                "model": model_name_with_tag,
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "size": 1000000000,  # 1GB fake size
                "digest": generate_fake_digest(model_name),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model["details"]["family"],
                    "families": [model["details"]["family"]],
                    "parameter_size": model["details"]["parameter_size"],
                    "quantization_level": model["details"]["quantization_level"],
                },
            }
        )

    return jsonify({"models": models})


@app.route("/api/show", methods=["POST"])
def show_model():
    """Show model details in Ollama format."""
    data = request.get_json() or {}
    model_name = data.get("name", data.get("model", ""))

    try:
        resolved = registry.resolve_model(model_name)
        provider, model_id = resolved.provider, resolved.model_id
        info = provider.get_models().get(model_id)

        if info:
            # Build capabilities list in Ollama format
            # Ollama uses: "completion", "vision", "tools", "embedding"
            capabilities = ["completion"]  # All chat models support completion
            if "vision" in info.capabilities:
                capabilities.append("vision")
            if "tools" in info.capabilities:
                capabilities.append("tools")

            return jsonify(
                {
                    "modelfile": f"# {provider.name} Model: {model_id}",
                    "parameters": f"temperature 0.7\nnum_ctx {info.context_length}",
                    "template": "{{ .System }}\n\n{{ .Prompt }}",
                    "details": {
                        "parent_model": "",
                        "format": "api",
                        "family": info.family,
                        "families": [info.family],
                        "parameter_size": info.parameter_size,
                        "quantization_level": info.quantization_level,
                    },
                    "capabilities": capabilities,
                    "model_info": {
                        "provider": provider.name,
                        "model_id": model_id,
                        "context_length": info.context_length,
                        "description": info.description,
                    },
                }
            )
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    return jsonify({"error": "Model not found"}), 404


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat completion endpoint - main inference endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    # Log inbound request (before resolution/processing)
    log_inbound_request(model_name, "/api/chat", tag)

    # Convert messages for resolution (needed for smart routers)
    ollama_messages = data.get("messages", [])
    system_prompt, messages = convert_ollama_messages(ollama_messages)

    # Extract @relay commands from messages and get cleaned messages
    messages, relay_commands, relay_tags = extract_relay_commands_from_messages(
        messages
    )
    if relay_commands:
        logger.info(
            f"Extracted @relay commands: {[c.command + ':' + c.raw_value for c in relay_commands]}"
        )

    # Merge relay tags with existing tags
    if relay_tags:
        all_tags = tag.split(",") if tag else []
        all_tags.extend(relay_tags)
        tag = normalize_tags(",".join(all_tags))

    # Generate session key for smart router caching
    from flask import g

    client_ip = getattr(g, "client_ip", get_client_ip(request))
    user_agent = request.headers.get("User-Agent", "")
    session_key = get_session_key(client_ip, user_agent)

    try:
        resolved = registry.resolve_model(
            clean_model_name,
            messages=messages,
            system=system_prompt,
            session_key=session_key,
            tags=tag,
        )
        provider, model_id = resolved.provider, resolved.model_id
        alias_name = resolved.alias_name
        alias_tags = resolved.alias_tags or []
        router_name = resolved.router_name  # Smart router tracking (v3.2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Handle Smart Enricher context injection (unified RAG + Web)
    enricher_name = None
    enricher_tags = []
    enricher_type = None
    enricher_query = None
    enricher_urls = None
    enrichment_result = None  # For source attribution
    original_messages = messages.copy()  # Save for memory update
    if getattr(resolved, "has_enrichment", False):
        enrichment_result = resolved.enrichment_result
        enricher_name = enrichment_result.enricher_name
        enricher_tags = enrichment_result.enricher_tags or []

        # Capture enrichment details for logging
        enricher_type = enrichment_result.enrichment_type
        enricher_query = enrichment_result.search_query
        enricher_urls = enrichment_result.scraped_urls or None

        # Swap in augmented content
        if enrichment_result.augmented_system is not None:
            system_prompt = enrichment_result.augmented_system
        if enrichment_result.augmented_messages:
            messages = enrichment_result.augmented_messages

        # Log enrichment
        if enrichment_result.context_injected:
            logger.info(
                f"Enricher '{enricher_name}' applied {enricher_type} enrichment "
                f"(RAG: {enrichment_result.chunks_retrieved} chunks, Web: {len(enrichment_result.scraped_urls)} URLs)"
            )

        # Update Smart Alias statistics
        from db import get_smart_alias_by_name, update_smart_alias_stats

        smart_alias = get_smart_alias_by_name(enricher_name)
        if smart_alias:
            # Track routing decision if this alias uses routing
            is_routing = smart_alias.use_routing
            if is_routing:
                router_name = enricher_name  # For request log tracking
            update_smart_alias_stats(
                alias_id=enrichment_result.enricher_id,
                increment_requests=1,
                increment_routing=1 if is_routing else 0,
                increment_injections=1 if enrichment_result.context_injected else 0,
                increment_search=1 if enrichment_result.search_query else 0,
                increment_scrape=1 if enrichment_result.scraped_urls else 0,
            )

        # Log designator usage for enricher - log each call separately
        designator_calls = getattr(enrichment_result, "designator_calls", [])
        if designator_calls:
            designator_model = enrichment_result.designator_model or "unknown"

            # Merge enricher tags with request tags for designator logging
            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            for call in designator_calls:
                track_completion(
                    provider_id=designator_model.split("/")[0]
                    if "/" in designator_model
                    else "unknown",
                    model_id=designator_model.split("/")[1]
                    if "/" in designator_model
                    else designator_model,
                    model_name=designator_model,
                    endpoint="/api/chat",
                    input_tokens=call.get("prompt_tokens", 0),
                    output_tokens=call.get("completion_tokens", 0),
                    status_code=200,
                    tag=designator_tag,
                    is_designator=True,
                    request_type="designator",
                )
        # Fallback for legacy single designator_usage (shouldn't happen with new code)
        elif enrichment_result.designator_usage:
            designator_model = enrichment_result.designator_model or "unknown"
            designator_usage = enrichment_result.designator_usage

            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            track_completion(
                provider_id=designator_model.split("/")[0]
                if "/" in designator_model
                else "unknown",
                model_id=designator_model.split("/")[1]
                if "/" in designator_model
                else designator_model,
                model_name=designator_model,
                endpoint="/api/chat",
                input_tokens=designator_usage.get("prompt_tokens", 0),
                output_tokens=designator_usage.get("completion_tokens", 0),
                status_code=200,
                tag=designator_tag,
                is_designator=True,
                request_type="designator",
            )

        # Log embedding usage for paid providers (OpenAI)
        if (
            enrichment_result.embedding_usage
            and enrichment_result.embedding_provider == "openai"
        ):
            embed_model = enrichment_result.embedding_model or "text-embedding-3-small"
            embed_usage = enrichment_result.embedding_usage

            # Merge enricher tags with request tags for embedding logging
            embed_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                embed_tag = ",".join(all_tags)

            track_completion(
                provider_id="openai",
                model_id=embed_model,
                model_name=f"openai/{embed_model}",
                endpoint="/v1/embeddings",
                input_tokens=embed_usage.get("prompt_tokens", 0),
                output_tokens=0,
                status_code=200,
                tag=embed_tag,
                request_type="embedding",
            )

    # Log designator usage for smart routers (v3.2)
    log_designator_usage(resolved, "/api/chat", tag)

    # Merge alias/enricher tags with request tags
    all_entity_tags = alias_tags + enricher_tags
    if all_entity_tags:
        existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
        all_tags = list(set(existing_tags + all_entity_tags))
        tag = ",".join(all_tags)

    # Filter out images if provider doesn't support vision
    if not provider_supports_vision(provider, model_id):
        original_count = len(messages)
        messages = filter_images_from_messages(messages)
        # Check if any messages had images
        has_images = any(
            isinstance(m.get("content"), list)
            for m in convert_ollama_messages(ollama_messages)[1]
        )
        if has_images:
            logger.warning(
                f"Model {model_id} doesn't support vision - images removed from request"
            )

    options = data.get("options", {})
    stream = data.get("stream", True)

    logger.info(
        f"Chat request: provider={provider.name}, model={model_id}, "
        f"messages={len(messages)}, stream={stream}"
    )
    # Check for cache hit (if caching is enabled on the resolved entity)
    cache_engine = None
    if resolved.has_cache:
        from context.chroma import is_chroma_available
        from routing.smart_cache import SmartCacheEngine

        if is_chroma_available():
            try:
                cache_engine = SmartCacheEngine(resolved.cache_config, registry)
                cache_result = cache_engine.lookup(messages, system_prompt)

                if cache_result.is_cache_hit:
                    # Cache hit! Return cached response
                    cached_response = cache_result.cached_response
                    cached_content = cached_response.get("content", "")
                    cached_tokens = cache_result.cached_tokens
                    cached_cost_saved = (
                        cache_result.cached_cost
                    )  # Exact cost from original request

                    logger.info(
                        f"Cache hit for '{cache_result.cache_name}' "
                        f"(similarity={cache_result.similarity_score:.4f}, "
                        f"tokens={cached_tokens}, cost_saved=${cached_cost_saved:.4f})"
                    )

                    # Update cache stats on the SmartAlias
                    config = resolved.cache_config
                    if hasattr(config, "id"):
                        from db import update_smart_alias_stats

                        update_smart_alias_stats(
                            alias_id=config.id,
                            increment_cache_hits=1,
                            increment_tokens_saved=cached_tokens,
                            increment_cost_saved=cached_cost_saved,
                        )

                    # Build Ollama-format response
                    response_obj = {
                        "model": model_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": cached_content},
                        "done": True,
                        "total_duration": 0,
                        "load_duration": 0,
                        "prompt_eval_count": 0,
                        "eval_count": cached_tokens,
                        "eval_duration": 0,
                    }

                    # Track the cache hit
                    track_completion(
                        provider_id=provider.name,
                        model_id=model_id,
                        model_name=model_name,
                        endpoint="/api/chat",
                        input_tokens=0,
                        output_tokens=0,
                        status_code=200,
                        tag=tag,
                        alias=alias_name,
                        router_name=router_name,
                        is_cache_hit=True,
                        cache_name=cache_result.cache_name,
                        cache_tokens_saved=cached_tokens,
                        cache_cost_saved=cached_cost_saved,
                    )

                    return jsonify(response_obj)
            except Exception as cache_err:
                logger.warning(f"Cache lookup failed: {cache_err}")
                cache_engine = None  # Proceed without caching

    try:
        if stream:
            # Calculate input chars for token estimation
            input_chars = estimate_input_chars(messages, system_prompt)

            # Capture request context before streaming starts (context won't exist after response starts)
            from flask import g

            start_time = getattr(g, "start_time", time.time())
            client_ip = getattr(g, "client_ip", "unknown")
            # tag already extracted above
            hostname = resolve_hostname(client_ip)
            prov_name = provider.name  # Capture for closure
            captured_alias = alias_name  # Capture alias for closure
            captured_router = router_name  # Capture router for closure

            # Capture cache engine and messages for streaming cache storage
            captured_cache_engine = cache_engine
            captured_messages = messages
            captured_system = system_prompt
            captured_enrichment_result = enrichment_result
            captured_original_messages = original_messages  # Pre-augmentation messages

            # Create callback to track after stream completes
            def on_stream_complete(
                input_tokens,
                output_tokens,
                error=None,
                cost=None,
                stream_result=None,
                full_content=None,
                action_blocks=None,
            ):
                response_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Track: {prov_name}/{model_id} - {response_time_ms}ms, streaming=True"
                )
                tracker.log_request(
                    timestamp=datetime.now(timezone.utc),
                    client_ip=client_ip,
                    hostname=hostname,
                    tag=tag,
                    provider_id=prov_name,
                    model_id=model_id,
                    endpoint="/api/chat",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=response_time_ms,
                    status_code=200 if not error else 500,
                    error_message=error,
                    is_streaming=True,
                    cost=cost,
                    reasoning_tokens=stream_result.get("reasoning_tokens")
                    if stream_result
                    else None,
                    cached_input_tokens=stream_result.get("cached_input_tokens")
                    if stream_result
                    else None,
                    cache_creation_tokens=stream_result.get("cache_creation_tokens")
                    if stream_result
                    else None,
                    cache_read_tokens=stream_result.get("cache_read_tokens")
                    if stream_result
                    else None,
                    alias=captured_alias,
                    router_name=captured_router,
                )

                # Store streaming response in cache
                if captured_cache_engine and full_content and not error:
                    try:
                        captured_cache_engine.store_response(
                            messages=captured_messages,
                            system=captured_system,
                            response={"content": full_content},
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost or 0.0,
                        )
                    except Exception as cache_err:
                        logger.warning(
                            f"Failed to store streaming response in cache: {cache_err}"
                        )

                # Update memory if enabled
                if (
                    captured_enrichment_result
                    and getattr(
                        captured_enrichment_result, "memory_update_pending", False
                    )
                    and full_content
                    and not error
                ):
                    try:
                        from db import get_smart_alias_by_name
                        from routing.smart_enricher import SmartEnricherEngine

                        alias = get_smart_alias_by_name(
                            captured_enrichment_result.enricher_name
                        )
                        if alias:
                            engine = SmartEnricherEngine(alias, registry)
                            engine.update_memory_after_response(
                                captured_original_messages, full_content
                            )
                    except Exception as mem_err:
                        logger.warning(f"Failed to update memory: {mem_err}")

                # Execute actions if enabled and actions were detected
                if (
                    captured_enrichment_result
                    and getattr(captured_enrichment_result, "actions_enabled", False)
                    and full_content
                    and not error
                ):
                    try:
                        from actions import execute_actions

                        allowed = getattr(
                            captured_enrichment_result, "allowed_actions", []
                        )
                        results, _ = execute_actions(
                            response_text=full_content,
                            alias_name=captured_enrichment_result.enricher_name,
                            allowed_actions=allowed,
                            session_key=getattr(
                                captured_enrichment_result, "session_key", None
                            ),
                            default_email_account_id=getattr(
                                captured_enrichment_result,
                                "action_email_account_id",
                                None,
                            ),
                            default_calendar_account_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_account_id",
                                None,
                            ),
                            default_calendar_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_id",
                                None,
                            ),
                            default_tasks_account_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_account_id",
                                None,
                            ),
                            default_tasks_list_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_list_id",
                                None,
                            ),
                            default_notification_urls=getattr(
                                captured_enrichment_result,
                                "action_notification_urls",
                                None,
                            ),
                            scheduled_prompts_account_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_account_id",
                                None,
                            ),
                            scheduled_prompts_calendar_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_calendar_id",
                                None,
                            ),
                        )
                        if results:
                            logger.info(
                                f"Executed {len(results)} actions for alias '{captured_enrichment_result.enricher_name}'"
                            )
                    except Exception as action_err:
                        logger.warning(f"Failed to execute actions: {action_err}")

            # Build source attribution if enabled
            source_attribution = build_source_attribution(enrichment_result)

            # Check if actions are enabled for stripping from response
            strip_actions = enrichment_result and getattr(
                enrichment_result, "actions_enabled", False
            )

            return Response(
                stream_ollama_response(
                    provider,
                    model_id,
                    messages,
                    system_prompt,
                    options,
                    on_complete=on_stream_complete,
                    input_char_count=input_chars,
                    source_attribution=source_attribution,
                    strip_actions=strip_actions,
                ),
                mimetype="application/x-ndjson",
            )
        else:
            result = provider.chat_completion(
                model_id, messages, system_prompt, options
            )

            # Debug comparison for non-streaming
            llm_content = result.get("content", "")
            response_obj = {
                "model": model_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": llm_content},
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": result.get("input_tokens", 0),
                "eval_count": result.get("output_tokens", 0),
                "eval_duration": 0,
            }
            client_content = response_obj["message"]["content"]
            debug_compare_response("/api/chat", llm_content, client_content)

            # Track non-streaming request
            track_completion(
                provider_id=provider.name,
                model_id=model_id,
                model_name=model_name,
                endpoint="/api/chat",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status_code=200,
                cost=result.get("cost"),
                reasoning_tokens=result.get("reasoning_tokens"),
                cached_input_tokens=result.get("cached_input_tokens"),
                cache_creation_tokens=result.get("cache_creation_tokens"),
                cache_read_tokens=result.get("cache_read_tokens"),
                tag=tag,
                alias=alias_name,
                router_name=router_name,
            )

            # Store response in cache (if caching is enabled and we have a cache engine)
            if cache_engine:
                try:
                    cache_engine.store_response(
                        messages=messages,
                        system=system_prompt,
                        response={"content": llm_content},
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        cost=result.get("cost", 0.0) or 0.0,
                    )
                except Exception as cache_err:
                    logger.warning(f"Failed to store response in cache: {cache_err}")

            # Update memory if enabled (non-streaming)
            if (
                enrichment_result
                and getattr(enrichment_result, "memory_update_pending", False)
                and llm_content
            ):
                try:
                    from db import get_smart_alias_by_name
                    from routing.smart_enricher import SmartEnricherEngine

                    alias = get_smart_alias_by_name(enrichment_result.enricher_name)
                    if alias:
                        engine = SmartEnricherEngine(alias, registry)
                        engine.update_memory_after_response(
                            original_messages, llm_content
                        )
                except Exception as mem_err:
                    logger.warning(f"Failed to update memory: {mem_err}")

            # Execute actions if enabled (non-streaming)
            if (
                enrichment_result
                and getattr(enrichment_result, "actions_enabled", False)
                and llm_content
            ):
                try:
                    from actions import execute_actions, strip_actions

                    allowed = getattr(enrichment_result, "allowed_actions", [])
                    results, cleaned_content = execute_actions(
                        response_text=llm_content,
                        alias_name=enrichment_result.enricher_name,
                        allowed_actions=allowed,
                        session_key=getattr(enrichment_result, "session_key", None),
                        default_email_account_id=getattr(
                            enrichment_result, "action_email_account_id", None
                        ),
                        default_calendar_account_id=getattr(
                            enrichment_result, "action_calendar_account_id", None
                        ),
                        default_calendar_id=getattr(
                            enrichment_result, "action_calendar_id", None
                        ),
                        default_tasks_account_id=getattr(
                            enrichment_result, "action_tasks_account_id", None
                        ),
                        default_tasks_list_id=getattr(
                            enrichment_result, "action_tasks_list_id", None
                        ),
                        default_notification_urls=getattr(
                            enrichment_result, "action_notification_urls", None
                        ),
                        scheduled_prompts_account_id=getattr(
                            enrichment_result, "scheduled_prompts_account_id", None
                        ),
                        scheduled_prompts_calendar_id=getattr(
                            enrichment_result, "scheduled_prompts_calendar_id", None
                        ),
                    )
                    if results:
                        logger.info(
                            f"Executed {len(results)} actions for alias '{enrichment_result.enricher_name}'"
                        )
                    # Update response with cleaned content (action blocks removed)
                    response_obj["message"]["content"] = cleaned_content
                except Exception as action_err:
                    logger.warning(f"Failed to execute actions: {action_err}")

            return jsonify(response_obj)

    except Exception as e:
        logger.error(f"Provider error: {e}")
        # Track error
        track_completion(
            provider_id=provider.name,
            model_id=model_id,
            model_name=model_name,
            endpoint="/api/chat",
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            error_message=str(e),
            tag=tag,
            alias=alias_name,
            router_name=router_name,
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate endpoint for compatibility. Converts to chat format internally."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    # Log inbound request (before resolution/processing)
    log_inbound_request(model_name, "/api/generate", tag)

    prompt = data.get("prompt", "")
    system = data.get("system", None)

    # Build messages for resolution (needed for smart routers)
    # We build a simple message list first, then potentially add images after resolution
    messages_for_resolution = [{"role": "user", "content": prompt}]

    # Extract @relay commands from messages and get cleaned messages
    messages_for_resolution, relay_commands, relay_tags = (
        extract_relay_commands_from_messages(messages_for_resolution)
    )
    if relay_commands:
        logger.info(
            f"Extracted @relay commands: {[c.command + ':' + c.raw_value for c in relay_commands]}"
        )
        # Update prompt with cleaned content
        prompt = (
            messages_for_resolution[0].get("content", prompt)
            if messages_for_resolution
            else prompt
        )

    # Merge relay tags with existing tags
    if relay_tags:
        all_tags = tag.split(",") if tag else []
        all_tags.extend(relay_tags)
        tag = normalize_tags(",".join(all_tags))

    # Generate session key for smart router caching
    from flask import g

    client_ip = getattr(g, "client_ip", get_client_ip(request))
    user_agent = request.headers.get("User-Agent", "")
    session_key = get_session_key(client_ip, user_agent)

    try:
        resolved = registry.resolve_model(
            clean_model_name,
            messages=messages_for_resolution,
            system=system,
            session_key=session_key,
            tags=tag,
        )
        provider, model_id = resolved.provider, resolved.model_id
        alias_name = resolved.alias_name
        alias_tags = resolved.alias_tags or []
        router_name = resolved.router_name  # Smart router tracking (v3.2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Log designator usage for smart routers (v3.2)
    log_designator_usage(resolved, "/api/generate", tag)

    # Merge alias tags with request tags (v3.1)
    if alias_tags:
        existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
        all_tags = list(set(existing_tags + alias_tags))
        tag = ",".join(all_tags)

    # Build actual messages (may include images)
    if "images" in data and data["images"]:
        # Check if provider supports vision
        if provider_supports_vision(provider, model_id):
            content_blocks = []
            for img in data["images"]:
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img,
                        },
                    }
                )
            content_blocks.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_blocks}]
        else:
            # Provider doesn't support vision - ignore images
            logger.warning(
                f"Model {model_id} doesn't support vision - images removed from request"
            )
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    options = data.get("options", {})
    stream = data.get("stream", True)

    logger.info(
        f"Generate request: provider={provider.name}, model={model_id}, stream={stream}"
    )

    try:
        if stream:
            # Calculate input chars for token estimation
            input_chars = estimate_input_chars(messages, system)

            # Capture request context before streaming starts
            from flask import g

            start_time = getattr(g, "start_time", time.time())
            client_ip = getattr(g, "client_ip", "unknown")
            # tag already extracted above
            hostname = resolve_hostname(client_ip)
            prov_name = provider.name
            captured_alias = alias_name  # Capture alias for closure (v3.1)
            captured_router = router_name  # Capture router for closure (v3.2)

            # Create callback to track after stream completes
            def on_stream_complete(
                input_tokens, output_tokens, error=None, cost=None, stream_result=None
            ):
                response_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Track: {prov_name}/{model_id} - {response_time_ms}ms, streaming=True"
                )
                tracker.log_request(
                    timestamp=datetime.now(timezone.utc),
                    client_ip=client_ip,
                    hostname=hostname,
                    tag=tag,
                    provider_id=prov_name,
                    model_id=model_id,
                    endpoint="/api/generate",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=response_time_ms,
                    status_code=200 if not error else 500,
                    error_message=error,
                    is_streaming=True,
                    cost=cost,
                    reasoning_tokens=stream_result.get("reasoning_tokens")
                    if stream_result
                    else None,
                    cached_input_tokens=stream_result.get("cached_input_tokens")
                    if stream_result
                    else None,
                    cache_creation_tokens=stream_result.get("cache_creation_tokens")
                    if stream_result
                    else None,
                    cache_read_tokens=stream_result.get("cache_read_tokens")
                    if stream_result
                    else None,
                    alias=captured_alias,
                    router_name=captured_router,
                )

            return Response(
                stream_ollama_response(
                    provider,
                    model_id,
                    messages,
                    system,
                    options,
                    on_complete=on_stream_complete,
                    input_char_count=input_chars,
                ),
                mimetype="application/x-ndjson",
            )
        else:
            result = provider.chat_completion(model_id, messages, system, options)

            # Debug comparison for non-streaming
            llm_content = result.get("content", "")
            response_obj = {
                "model": model_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": llm_content,
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": result.get("input_tokens", 0),
                "eval_count": result.get("output_tokens", 0),
                "eval_duration": 0,
            }
            client_content = response_obj["response"]
            debug_compare_response("/api/generate", llm_content, client_content)

            # Track non-streaming request
            track_completion(
                provider_id=provider.name,
                model_id=model_id,
                model_name=model_name,
                endpoint="/api/generate",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status_code=200,
                cost=result.get("cost"),
                reasoning_tokens=result.get("reasoning_tokens"),
                cached_input_tokens=result.get("cached_input_tokens"),
                cache_creation_tokens=result.get("cache_creation_tokens"),
                cache_read_tokens=result.get("cache_read_tokens"),
                tag=tag,
                alias=alias_name,
                router_name=router_name,
            )
            return jsonify(response_obj)

    except Exception as e:
        logger.error(f"Provider error: {e}")
        # Track error
        track_completion(
            provider_id=provider.name,
            model_id=model_id,
            model_name=model_name,
            endpoint="/api/generate",
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            error_message=str(e),
            tag=tag,
            alias=alias_name,
            router_name=router_name,
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    """Embeddings endpoint - not supported by most providers."""
    return jsonify(
        {
            "error": "Embeddings are not supported. Use a dedicated embedding service or local model."
        }
    ), 501


@app.route("/api/pull", methods=["POST"])
def pull_model():
    """Pull endpoint - not applicable for API models."""
    data = request.get_json() or {}
    model = data.get("name", "")

    return Response(
        json.dumps(
            {"status": f"success: {model} is an API model and requires no download"}
        )
        + "\n",
        mimetype="application/x-ndjson",
    )


@app.route("/api/version", methods=["GET"])
def version():
    """Return version information."""
    return jsonify({"version": "1.2.1-multi-provider"})


@app.route("/api/ps", methods=["GET"])
def list_running_models():
    """List running models - for API models, all configured models are 'running'."""
    all_models = registry.list_all_models()

    models = []
    for model in all_models:
        model_name = model["name"]
        models.append(
            {
                "name": model_name,
                "model": model_name,
                "size": 1000000000,
                "digest": generate_fake_digest(model_name),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model["details"]["family"],
                    "families": [model["details"]["family"]],
                    "parameter_size": model["details"]["parameter_size"],
                    "quantization_level": model["details"]["quantization_level"],
                },
                "expires_at": "2099-12-31T23:59:59.999999999Z",
                "size_vram": 1000000000,
            }
        )

    return jsonify({"models": models})


# ============================================================================
# OpenAI-Compatible API Endpoints (/v1/*)
# ============================================================================


@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """List available models in OpenAI format."""
    models = registry.list_openai_models()

    # Add created timestamp
    created = int(time.time())
    for model in models:
        model["created"] = created

    return jsonify({"object": "list", "data": models})


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/api/chat/completions", methods=["POST"])
def openai_chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    # Log inbound request (before resolution/processing)
    log_inbound_request(model_name, "/v1/chat/completions", tag)

    # Convert messages for resolution (needed for smart routers)
    openai_messages = data.get("messages", [])
    system_prompt, messages = convert_openai_messages(openai_messages)

    # Extract @relay commands from messages and get cleaned messages
    messages, relay_commands, relay_tags = extract_relay_commands_from_messages(
        messages
    )
    if relay_commands:
        logger.info(
            f"Extracted @relay commands: {[c.command + ':' + c.raw_value for c in relay_commands]}"
        )

    # Merge relay tags with existing tags
    if relay_tags:
        all_tags = tag.split(",") if tag else []
        all_tags.extend(relay_tags)
        tag = normalize_tags(",".join(all_tags))

    # Generate session key for smart router caching
    from flask import g

    client_ip = getattr(g, "client_ip", get_client_ip(request))
    user_agent = request.headers.get("User-Agent", "")
    session_key = get_session_key(client_ip, user_agent)

    try:
        resolved = registry.resolve_model(
            clean_model_name,
            messages=messages,
            system=system_prompt,
            session_key=session_key,
            tags=tag,
        )
        provider, model_id = resolved.provider, resolved.model_id
        alias_name = resolved.alias_name
        alias_tags = resolved.alias_tags or []
        router_name = resolved.router_name  # Smart router tracking (v3.2)
    except ValueError as e:
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        ), 400

    # Handle Smart Enricher context injection (unified RAG + Web)
    enricher_name = None
    enricher_tags = []
    enricher_type = None
    enricher_query = None
    enricher_urls = None
    enrichment_result = None  # For source attribution
    original_messages = messages.copy()  # Save for memory update
    if getattr(resolved, "has_enrichment", False):
        enrichment_result = resolved.enrichment_result
        enricher_name = enrichment_result.enricher_name
        enricher_tags = enrichment_result.enricher_tags or []

        # Capture enrichment details for logging
        enricher_type = enrichment_result.enrichment_type
        enricher_query = enrichment_result.search_query
        enricher_urls = enrichment_result.scraped_urls or None

        # Swap in augmented content
        if enrichment_result.augmented_system is not None:
            system_prompt = enrichment_result.augmented_system
        if enrichment_result.augmented_messages:
            messages = enrichment_result.augmented_messages

        # Log enrichment
        if enrichment_result.context_injected:
            logger.info(
                f"Enricher '{enricher_name}' applied {enricher_type} enrichment "
                f"(RAG: {enrichment_result.chunks_retrieved} chunks, Web: {len(enrichment_result.scraped_urls)} URLs)"
            )

        # Update Smart Alias statistics
        from db import get_smart_alias_by_name, update_smart_alias_stats

        smart_alias = get_smart_alias_by_name(enricher_name)
        if smart_alias:
            # Track routing decision if this alias uses routing
            is_routing = smart_alias.use_routing
            if is_routing:
                router_name = enricher_name  # For request log tracking
            update_smart_alias_stats(
                alias_id=enrichment_result.enricher_id,
                increment_requests=1,
                increment_routing=1 if is_routing else 0,
                increment_injections=1 if enrichment_result.context_injected else 0,
                increment_search=1 if enrichment_result.search_query else 0,
                increment_scrape=1 if enrichment_result.scraped_urls else 0,
            )

        # Log designator usage for enricher - log each call separately
        designator_calls = getattr(enrichment_result, "designator_calls", [])
        if designator_calls:
            designator_model = enrichment_result.designator_model or "unknown"

            # Merge enricher tags with request tags for designator logging
            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            for call in designator_calls:
                track_completion(
                    provider_id=designator_model.split("/")[0]
                    if "/" in designator_model
                    else "unknown",
                    model_id=designator_model.split("/")[1]
                    if "/" in designator_model
                    else designator_model,
                    model_name=designator_model,
                    endpoint="/v1/chat/completions",
                    input_tokens=call.get("prompt_tokens", 0),
                    output_tokens=call.get("completion_tokens", 0),
                    status_code=200,
                    tag=designator_tag,
                    is_designator=True,
                    request_type="designator",
                )
        # Fallback for legacy single designator_usage
        elif enrichment_result.designator_usage:
            designator_model = enrichment_result.designator_model or "unknown"
            designator_usage = enrichment_result.designator_usage

            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            track_completion(
                provider_id=designator_model.split("/")[0]
                if "/" in designator_model
                else "unknown",
                model_id=designator_model.split("/")[1]
                if "/" in designator_model
                else designator_model,
                model_name=designator_model,
                endpoint="/v1/chat/completions",
                input_tokens=designator_usage.get("prompt_tokens", 0),
                output_tokens=designator_usage.get("completion_tokens", 0),
                status_code=200,
                tag=designator_tag,
                is_designator=True,
                request_type="designator",
            )

        # Log embedding usage for paid providers (OpenAI)
        if (
            enrichment_result.embedding_usage
            and enrichment_result.embedding_provider == "openai"
        ):
            embed_model = enrichment_result.embedding_model or "text-embedding-3-small"
            embed_usage = enrichment_result.embedding_usage

            # Merge enricher tags with request tags for embedding logging
            embed_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                embed_tag = ",".join(all_tags)

            track_completion(
                provider_id="openai",
                model_id=embed_model,
                model_name=f"openai/{embed_model}",
                endpoint="/v1/embeddings",
                input_tokens=embed_usage.get("prompt_tokens", 0),
                output_tokens=0,
                status_code=200,
                tag=embed_tag,
                request_type="embedding",
            )

    # Log designator usage for smart routers (v3.2)
    log_designator_usage(resolved, "/v1/chat/completions", tag)

    # Merge alias/enricher tags with request tags
    all_entity_tags = alias_tags + enricher_tags
    if all_entity_tags:
        existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
        all_tags = list(set(existing_tags + all_entity_tags))
        tag = ",".join(all_tags)

    # Filter out images if provider doesn't support vision
    if not provider_supports_vision(provider, model_id):
        has_images = any(isinstance(m.get("content"), list) for m in messages)
        if has_images:
            messages = filter_images_from_messages(messages)
            logger.warning(
                f"Model {model_id} doesn't support vision - images removed from request"
            )

    # Check for cache hit (if caching is enabled on the resolved entity)
    cache_engine = None
    if resolved.has_cache:
        from context.chroma import is_chroma_available
        from routing.smart_cache import SmartCacheEngine

        if is_chroma_available():
            try:
                cache_engine = SmartCacheEngine(resolved.cache_config, registry)
                cache_result = cache_engine.lookup(messages, system_prompt)

                if cache_result.is_cache_hit:
                    # Cache hit! Return cached response
                    cached_response = cache_result.cached_response
                    cached_content = cached_response.get("content", "")
                    cached_tokens = cache_result.cached_tokens
                    cached_cost_saved = (
                        cache_result.cached_cost
                    )  # Exact cost from original request

                    logger.info(
                        f"Cache hit for '{cache_result.cache_name}' "
                        f"(similarity={cache_result.similarity_score:.4f}, "
                        f"tokens={cached_tokens}, cost_saved=${cached_cost_saved:.4f})"
                    )

                    # Update cache stats on the SmartAlias
                    config = resolved.cache_config
                    if hasattr(config, "id"):
                        from db import update_smart_alias_stats

                        update_smart_alias_stats(
                            alias_id=config.id,
                            increment_cache_hits=1,
                            increment_tokens_saved=cached_tokens,
                            increment_cost_saved=cached_cost_saved,
                        )

                    # Build OpenAI-format response
                    response_obj = {
                        "id": generate_openai_id(),
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": cached_content,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": cached_tokens,
                            "total_tokens": cached_tokens,
                        },
                    }

                    # Track the cache hit
                    track_completion(
                        provider_id=provider.name,
                        model_id=model_id,
                        model_name=model_name,
                        endpoint="/v1/chat/completions",
                        input_tokens=0,
                        output_tokens=0,
                        status_code=200,
                        tag=tag,
                        alias=alias_name,
                        router_name=router_name,
                        is_cache_hit=True,
                        cache_name=cache_result.cache_name,
                        cache_tokens_saved=cached_tokens,
                        cache_cost_saved=cached_cost_saved,
                    )

                    return jsonify(response_obj)
            except Exception as cache_err:
                logger.warning(f"Cache lookup failed: {cache_err}")
                cache_engine = None  # Proceed without caching

    # Build options
    options = {}
    if "max_tokens" in data:
        options["max_tokens"] = data["max_tokens"]
    elif "max_completion_tokens" in data:
        options["max_tokens"] = data["max_completion_tokens"]
    else:
        options["max_tokens"] = 4096

    if "temperature" in data:
        options["temperature"] = data["temperature"]
    if "top_p" in data:
        options["top_p"] = data["top_p"]
    if "stop" in data:
        options["stop"] = data["stop"]

    stream = data.get("stream", False)

    logger.info(
        f"OpenAI chat request: provider={provider.name}, model={model_id}, "
        f"messages={len(messages)}, stream={stream}"
    )

    try:
        if stream:
            # Calculate input chars for token estimation
            input_chars = estimate_input_chars(messages, system_prompt)

            # Capture request context before streaming starts
            from flask import g

            start_time = getattr(g, "start_time", time.time())
            client_ip = getattr(g, "client_ip", "unknown")
            # tag already extracted above
            hostname = resolve_hostname(client_ip)
            prov_name = provider.name
            captured_alias = alias_name  # Capture alias for closure
            captured_router = router_name  # Capture router for closure

            # Capture cache engine and messages for streaming cache storage
            captured_cache_engine = cache_engine
            captured_messages = messages
            captured_system = system_prompt
            captured_enrichment_result = enrichment_result
            captured_original_messages = original_messages  # Pre-augmentation messages

            # Create callback to track after stream completes
            def on_stream_complete(
                input_tokens,
                output_tokens,
                error=None,
                cost=None,
                stream_result=None,
                full_content=None,
                action_blocks=None,
            ):
                response_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Track: {prov_name}/{model_id} - {response_time_ms}ms, streaming=True"
                )
                tracker.log_request(
                    timestamp=datetime.now(timezone.utc),
                    client_ip=client_ip,
                    hostname=hostname,
                    tag=tag,
                    provider_id=prov_name,
                    model_id=model_id,
                    endpoint="/v1/chat/completions",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=response_time_ms,
                    status_code=200 if not error else 500,
                    error_message=error,
                    is_streaming=True,
                    cost=cost,
                    reasoning_tokens=stream_result.get("reasoning_tokens")
                    if stream_result
                    else None,
                    cached_input_tokens=stream_result.get("cached_input_tokens")
                    if stream_result
                    else None,
                    cache_creation_tokens=stream_result.get("cache_creation_tokens")
                    if stream_result
                    else None,
                    cache_read_tokens=stream_result.get("cache_read_tokens")
                    if stream_result
                    else None,
                    alias=captured_alias,
                    router_name=captured_router,
                )

                # Store streaming response in cache
                if captured_cache_engine and full_content and not error:
                    try:
                        captured_cache_engine.store_response(
                            messages=captured_messages,
                            system=captured_system,
                            response={"content": full_content},
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost or 0.0,
                        )
                    except Exception as cache_err:
                        logger.warning(
                            f"Failed to store streaming response in cache: {cache_err}"
                        )

                # Update memory if enabled
                if (
                    captured_enrichment_result
                    and getattr(
                        captured_enrichment_result, "memory_update_pending", False
                    )
                    and full_content
                    and not error
                ):
                    try:
                        from db import get_smart_alias_by_name
                        from routing.smart_enricher import SmartEnricherEngine

                        alias = get_smart_alias_by_name(
                            captured_enrichment_result.enricher_name
                        )
                        if alias:
                            engine = SmartEnricherEngine(alias, registry)
                            engine.update_memory_after_response(
                                captured_original_messages, full_content
                            )
                    except Exception as mem_err:
                        logger.warning(f"Failed to update memory: {mem_err}")

                # Execute actions if enabled and actions were detected
                if (
                    captured_enrichment_result
                    and getattr(captured_enrichment_result, "actions_enabled", False)
                    and full_content
                    and not error
                ):
                    try:
                        from actions import execute_actions

                        allowed = getattr(
                            captured_enrichment_result, "allowed_actions", []
                        )
                        results, _ = execute_actions(
                            response_text=full_content,
                            alias_name=captured_enrichment_result.enricher_name,
                            allowed_actions=allowed,
                            session_key=getattr(
                                captured_enrichment_result, "session_key", None
                            ),
                            default_email_account_id=getattr(
                                captured_enrichment_result,
                                "action_email_account_id",
                                None,
                            ),
                            default_calendar_account_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_account_id",
                                None,
                            ),
                            default_calendar_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_id",
                                None,
                            ),
                            default_tasks_account_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_account_id",
                                None,
                            ),
                            default_tasks_list_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_list_id",
                                None,
                            ),
                            default_notification_urls=getattr(
                                captured_enrichment_result,
                                "action_notification_urls",
                                None,
                            ),
                            scheduled_prompts_account_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_account_id",
                                None,
                            ),
                            scheduled_prompts_calendar_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_calendar_id",
                                None,
                            ),
                        )
                        if results:
                            logger.info(
                                f"Executed {len(results)} actions for alias '{captured_enrichment_result.enricher_name}'"
                            )
                    except Exception as action_err:
                        logger.warning(f"Failed to execute actions: {action_err}")

            # Build source attribution if enabled
            source_attribution = build_source_attribution(enrichment_result)

            # Check if actions are enabled for stripping from response
            strip_actions = enrichment_result and getattr(
                enrichment_result, "actions_enabled", False
            )

            return Response(
                stream_openai_response(
                    provider,
                    model_id,
                    messages,
                    system_prompt,
                    options,
                    model_name,
                    on_complete=on_stream_complete,
                    input_char_count=input_chars,
                    source_attribution=source_attribution,
                    strip_actions=strip_actions,
                ),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = provider.chat_completion(
                model_id, messages, system_prompt, options
            )

            # Debug comparison for non-streaming
            llm_content = result.get("content", "")
            response_obj = {
                "id": generate_openai_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": llm_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.get("input_tokens", 0),
                    "completion_tokens": result.get("output_tokens", 0),
                    "total_tokens": result.get("input_tokens", 0)
                    + result.get("output_tokens", 0),
                },
            }
            client_content = response_obj["choices"][0]["message"]["content"]
            debug_compare_response("/v1/chat/completions", llm_content, client_content)

            # Track non-streaming request
            track_completion(
                provider_id=provider.name,
                model_id=model_id,
                model_name=model_name,
                endpoint="/v1/chat/completions",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status_code=200,
                cost=result.get("cost"),
                reasoning_tokens=result.get("reasoning_tokens"),
                cached_input_tokens=result.get("cached_input_tokens"),
                cache_creation_tokens=result.get("cache_creation_tokens"),
                cache_read_tokens=result.get("cache_read_tokens"),
                tag=tag,
                alias=alias_name,
                router_name=router_name,
            )

            # Store response in cache (if caching is enabled and we have a cache engine)
            if cache_engine:
                try:
                    cache_engine.store_response(
                        messages=messages,
                        system=system_prompt,
                        response={"content": llm_content},
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        cost=result.get("cost", 0.0) or 0.0,
                    )
                except Exception as cache_err:
                    logger.warning(f"Failed to store response in cache: {cache_err}")

            # Update memory if enabled (non-streaming)
            if (
                enrichment_result
                and getattr(enrichment_result, "memory_update_pending", False)
                and llm_content
            ):
                try:
                    from db import get_smart_alias_by_name
                    from routing.smart_enricher import SmartEnricherEngine

                    alias = get_smart_alias_by_name(enrichment_result.enricher_name)
                    if alias:
                        engine = SmartEnricherEngine(alias, registry)
                        engine.update_memory_after_response(
                            original_messages, llm_content
                        )
                except Exception as mem_err:
                    logger.warning(f"Failed to update memory: {mem_err}")

            # Execute actions if enabled (non-streaming)
            if (
                enrichment_result
                and getattr(enrichment_result, "actions_enabled", False)
                and llm_content
            ):
                try:
                    from actions import execute_actions

                    allowed = getattr(enrichment_result, "allowed_actions", [])
                    results, cleaned_content = execute_actions(
                        response_text=llm_content,
                        alias_name=enrichment_result.enricher_name,
                        allowed_actions=allowed,
                        session_key=getattr(enrichment_result, "session_key", None),
                        default_email_account_id=getattr(
                            enrichment_result, "action_email_account_id", None
                        ),
                        default_calendar_account_id=getattr(
                            enrichment_result, "action_calendar_account_id", None
                        ),
                        default_calendar_id=getattr(
                            enrichment_result, "action_calendar_id", None
                        ),
                        default_tasks_account_id=getattr(
                            enrichment_result, "action_tasks_account_id", None
                        ),
                        default_tasks_list_id=getattr(
                            enrichment_result, "action_tasks_list_id", None
                        ),
                        default_notification_urls=getattr(
                            enrichment_result, "action_notification_urls", None
                        ),
                        scheduled_prompts_account_id=getattr(
                            enrichment_result, "scheduled_prompts_account_id", None
                        ),
                        scheduled_prompts_calendar_id=getattr(
                            enrichment_result, "scheduled_prompts_calendar_id", None
                        ),
                    )
                    if results:
                        logger.info(
                            f"Executed {len(results)} actions for alias '{enrichment_result.enricher_name}'"
                        )
                    # Update response with cleaned content (action blocks removed)
                    response_obj["choices"][0]["message"]["content"] = cleaned_content
                except Exception as action_err:
                    logger.warning(f"Failed to execute actions: {action_err}")

            return jsonify(response_obj)

    except Exception as e:
        logger.error(f"Provider error: {e}")
        # Track error
        track_completion(
            provider_id=provider.name,
            model_id=model_id,
            model_name=model_name,
            endpoint="/v1/chat/completions",
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            error_message=str(e),
            tag=tag,
            alias=alias_name,
            router_name=router_name,
        )
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "provider_error",
                }
            }
        ), 500


@app.route("/v1/completions", methods=["POST"])
def openai_completions():
    """OpenAI-compatible text completions endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    # Log inbound request (before resolution/processing)
    log_inbound_request(model_name, "/v1/completions", tag)

    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    messages = [{"role": "user", "content": prompt}]

    # Extract @relay commands from messages and get cleaned messages
    messages, relay_commands, relay_tags = extract_relay_commands_from_messages(
        messages
    )
    if relay_commands:
        logger.info(
            f"Extracted @relay commands: {[c.command + ':' + c.raw_value for c in relay_commands]}"
        )
        # Update prompt with cleaned content
        prompt = messages[0].get("content", prompt) if messages else prompt

    # Merge relay tags with existing tags
    if relay_tags:
        all_tags = tag.split(",") if tag else []
        all_tags.extend(relay_tags)
        tag = normalize_tags(",".join(all_tags))

    # Generate session key for smart router caching
    from flask import g

    client_ip = getattr(g, "client_ip", get_client_ip(request))
    user_agent = request.headers.get("User-Agent", "")
    session_key = get_session_key(client_ip, user_agent)

    try:
        resolved = registry.resolve_model(
            clean_model_name,
            messages=messages,
            system=None,
            session_key=session_key,
            tags=tag,
        )
        provider, model_id = resolved.provider, resolved.model_id
        alias_name = resolved.alias_name
        alias_tags = resolved.alias_tags or []
        router_name = resolved.router_name  # Smart router tracking (v3.2)
    except ValueError as e:
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        ), 400

    # Log designator usage for smart routers (v3.2)
    log_designator_usage(resolved, "/v1/completions", tag)

    # Merge alias tags with request tags (v3.1)
    if alias_tags:
        existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
        all_tags = list(set(existing_tags + alias_tags))
        tag = ",".join(all_tags)

    # Build options
    options = {"max_tokens": data.get("max_tokens", 4096)}
    if "temperature" in data:
        options["temperature"] = data["temperature"]
    if "top_p" in data:
        options["top_p"] = data["top_p"]
    if "stop" in data:
        options["stop"] = data["stop"]

    stream = data.get("stream", False)

    logger.info(
        f"OpenAI completions request: provider={provider.name}, model={model_id}, stream={stream}"
    )

    try:
        if stream:
            # Calculate input chars for token estimation
            input_chars = estimate_input_chars(messages, None)

            # Capture request context before streaming starts (context won't exist after response starts)
            from flask import g

            start_time = getattr(g, "start_time", time.time())
            client_ip = getattr(g, "client_ip", "unknown")
            # tag already extracted above
            hostname = resolve_hostname(client_ip)
            prov_name = provider.name  # Capture for closure
            captured_model_id = model_id  # Capture for closure
            captured_alias = alias_name  # Capture alias for closure (v3.1)
            captured_router = router_name  # Capture router for closure (v3.2)

            def stream_completions():
                response_id = generate_openai_id("cmpl")
                created = int(time.time())
                output_chars = 0
                llm_chunks = []  # Collect raw LLM response for debug comparison
                client_chunks = []  # Collect what we send to client

                try:
                    for text in provider.chat_completion_stream(
                        captured_model_id, messages, None, options
                    ):
                        output_chars += len(text)
                        llm_chunks.append(text)  # Raw LLM text

                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "text": text, "finish_reason": None}
                            ],
                        }
                        client_chunks.append(text)  # Track content we're sending
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final_chunk = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    # Debug comparison: raw LLM response vs what client receives
                    llm_full = "".join(llm_chunks)
                    client_full = "".join(client_chunks)
                    debug_compare_response(
                        "/v1/completions (stream)", llm_full, client_full
                    )

                    # Get streaming result (actual tokens, cost) if provider supports it
                    stream_result = None
                    if hasattr(provider, "get_last_stream_result"):
                        stream_result = provider.get_last_stream_result()

                    # Use actual tokens if available, fall back to char estimation
                    actual_input = (
                        stream_result.get("input_tokens")
                        if stream_result and stream_result.get("input_tokens")
                        else input_chars // 4
                    )
                    actual_output = (
                        stream_result.get("output_tokens")
                        if stream_result and stream_result.get("output_tokens")
                        else output_chars // 4
                    )

                    # Track after stream completes using captured context
                    response_time_ms = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"Track: {prov_name}/{captured_model_id} - {response_time_ms}ms, streaming=True"
                    )
                    tracker.log_request(
                        timestamp=datetime.now(timezone.utc),
                        client_ip=client_ip,
                        hostname=hostname,
                        tag=tag,
                        provider_id=prov_name,
                        model_id=captured_model_id,
                        endpoint="/v1/completions",
                        input_tokens=actual_input,
                        output_tokens=actual_output,
                        response_time_ms=response_time_ms,
                        status_code=200,
                        error_message=None,
                        is_streaming=True,
                        cost=stream_result.get("cost") if stream_result else None,
                        reasoning_tokens=stream_result.get("reasoning_tokens")
                        if stream_result
                        else None,
                        cached_input_tokens=stream_result.get("cached_input_tokens")
                        if stream_result
                        else None,
                        cache_creation_tokens=stream_result.get("cache_creation_tokens")
                        if stream_result
                        else None,
                        cache_read_tokens=stream_result.get("cache_read_tokens")
                        if stream_result
                        else None,
                        alias=captured_alias,
                        router_name=captured_router,
                    )

                except Exception as e:
                    logger.error(f"Provider error during streaming: {e}")
                    error_chunk = {"error": {"message": str(e), "type": "api_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    # Track error using captured context
                    response_time_ms = int((time.time() - start_time) * 1000)
                    tracker.log_request(
                        timestamp=datetime.now(timezone.utc),
                        client_ip=client_ip,
                        hostname=hostname,
                        tag=tag,
                        provider_id=prov_name,
                        model_id=captured_model_id,
                        endpoint="/v1/completions",
                        input_tokens=0,
                        output_tokens=0,
                        response_time_ms=response_time_ms,
                        status_code=500,
                        error_message=str(e),
                        is_streaming=True,
                        alias=captured_alias,
                        router_name=captured_router,
                    )

            return Response(
                stream_completions(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = provider.chat_completion(model_id, messages, None, options)

            # Debug comparison for non-streaming
            llm_content = result.get("content", "")
            response_obj = {
                "id": generate_openai_id("cmpl"),
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "text": llm_content, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": result.get("input_tokens", 0),
                    "completion_tokens": result.get("output_tokens", 0),
                    "total_tokens": result.get("input_tokens", 0)
                    + result.get("output_tokens", 0),
                },
            }
            client_content = response_obj["choices"][0]["text"]
            debug_compare_response("/v1/completions", llm_content, client_content)

            # Track non-streaming request
            track_completion(
                provider_id=provider.name,
                model_id=model_id,
                model_name=model_name,
                endpoint="/v1/completions",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status_code=200,
                cost=result.get("cost"),
                reasoning_tokens=result.get("reasoning_tokens"),
                cached_input_tokens=result.get("cached_input_tokens"),
                cache_creation_tokens=result.get("cache_creation_tokens"),
                cache_read_tokens=result.get("cache_read_tokens"),
                tag=tag,
                alias=alias_name,
                router_name=router_name,
            )

            return jsonify(response_obj)

    except Exception as e:
        logger.error(f"Provider error: {e}")
        # Track error
        track_completion(
            provider_id=provider.name,
            model_id=model_id,
            model_name=model_name,
            endpoint="/v1/completions",
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            error_message=str(e),
            tag=tag,
            alias=alias_name,
            router_name=router_name,
        )
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "provider_error",
                }
            }
        ), 500


# ============================================================================
# Anthropic API Endpoint
# ============================================================================


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    """Anthropic-compatible messages endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    # Log inbound request (before resolution/processing)
    log_inbound_request(model_name, "/v1/messages", tag)

    # Anthropic API has system as separate field
    anthropic_system = data.get("system")
    anthropic_messages = data.get("messages", [])

    # Convert messages - system is already separate in Anthropic API
    system_prompt, messages = convert_anthropic_messages(
        anthropic_messages, anthropic_system
    )

    # Extract @relay commands from messages and get cleaned messages
    messages, relay_commands, relay_tags = extract_relay_commands_from_messages(
        messages
    )
    if relay_commands:
        logger.info(
            f"Extracted @relay commands: {[c.command + ':' + c.raw_value for c in relay_commands]}"
        )

    # Merge relay tags with existing tags
    if relay_tags:
        all_tags = tag.split(",") if tag else []
        all_tags.extend(relay_tags)
        tag = normalize_tags(",".join(all_tags))

    # Generate session key for smart router caching
    from flask import g

    client_ip = getattr(g, "client_ip", get_client_ip(request))
    user_agent = request.headers.get("User-Agent", "")
    session_key = get_session_key(client_ip, user_agent)

    try:
        resolved = registry.resolve_model(
            clean_model_name,
            messages=messages,
            system=system_prompt,
            session_key=session_key,
            tags=tag,
        )
        provider, model_id = resolved.provider, resolved.model_id
        alias_name = resolved.alias_name
        alias_tags = resolved.alias_tags or []
        router_name = resolved.router_name
    except ValueError as e:
        return anthropic_error_response("not_found_error", str(e), 404)

    # Handle Smart Enricher context injection (unified RAG + Web)
    enricher_name = None
    enricher_tags = []
    enricher_type = None
    enricher_query = None
    enricher_urls = None
    enrichment_result = None
    original_messages = messages.copy()
    if getattr(resolved, "has_enrichment", False):
        enrichment_result = resolved.enrichment_result
        enricher_name = enrichment_result.enricher_name
        enricher_tags = enrichment_result.enricher_tags or []

        enricher_type = enrichment_result.enrichment_type
        enricher_query = enrichment_result.search_query
        enricher_urls = enrichment_result.scraped_urls or None

        if enrichment_result.augmented_system is not None:
            system_prompt = enrichment_result.augmented_system
        if enrichment_result.augmented_messages:
            messages = enrichment_result.augmented_messages

        if enrichment_result.context_injected:
            logger.info(
                f"Enricher '{enricher_name}' applied {enricher_type} enrichment "
                f"(RAG: {enrichment_result.chunks_retrieved} chunks, Web: {len(enrichment_result.scraped_urls)} URLs)"
            )

        from db import get_smart_alias_by_name, update_smart_alias_stats

        smart_alias = get_smart_alias_by_name(enricher_name)
        if smart_alias:
            is_routing = smart_alias.use_routing
            if is_routing:
                router_name = enricher_name
            update_smart_alias_stats(
                alias_id=enrichment_result.enricher_id,
                increment_requests=1,
                increment_routing=1 if is_routing else 0,
                increment_injections=1 if enrichment_result.context_injected else 0,
                increment_search=1 if enrichment_result.search_query else 0,
                increment_scrape=1 if enrichment_result.scraped_urls else 0,
            )

        # Log designator usage for enricher
        designator_calls = getattr(enrichment_result, "designator_calls", [])
        if designator_calls:
            designator_model = enrichment_result.designator_model or "unknown"
            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            for call in designator_calls:
                track_completion(
                    provider_id=designator_model.split("/")[0]
                    if "/" in designator_model
                    else "unknown",
                    model_id=designator_model.split("/")[1]
                    if "/" in designator_model
                    else designator_model,
                    model_name=designator_model,
                    endpoint="/v1/messages",
                    input_tokens=call.get("prompt_tokens", 0),
                    output_tokens=call.get("completion_tokens", 0),
                    status_code=200,
                    tag=designator_tag,
                    is_designator=True,
                    request_type="designator",
                )
        elif enrichment_result.designator_usage:
            designator_model = enrichment_result.designator_model or "unknown"
            designator_usage = enrichment_result.designator_usage
            designator_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                designator_tag = ",".join(all_tags)

            track_completion(
                provider_id=designator_model.split("/")[0]
                if "/" in designator_model
                else "unknown",
                model_id=designator_model.split("/")[1]
                if "/" in designator_model
                else designator_model,
                model_name=designator_model,
                endpoint="/v1/messages",
                input_tokens=designator_usage.get("prompt_tokens", 0),
                output_tokens=designator_usage.get("completion_tokens", 0),
                status_code=200,
                tag=designator_tag,
                is_designator=True,
                request_type="designator",
            )

        # Log embedding usage for paid providers (OpenAI)
        if (
            enrichment_result.embedding_usage
            and enrichment_result.embedding_provider == "openai"
        ):
            embed_model = enrichment_result.embedding_model or "text-embedding-3-small"
            embed_usage = enrichment_result.embedding_usage
            embed_tag = tag
            if enricher_tags:
                existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
                all_tags = list(set(existing_tags + enricher_tags))
                embed_tag = ",".join(all_tags)

            track_completion(
                provider_id="openai",
                model_id=embed_model,
                model_name=f"openai/{embed_model}",
                endpoint="/v1/embeddings",
                input_tokens=embed_usage.get("prompt_tokens", 0),
                output_tokens=0,
                status_code=200,
                tag=embed_tag,
                request_type="embedding",
            )

    # Log designator usage for smart routers
    log_designator_usage(resolved, "/v1/messages", tag)

    # Merge alias/enricher tags with request tags
    all_entity_tags = alias_tags + enricher_tags
    if all_entity_tags:
        existing_tags = [t.strip() for t in tag.split(",") if t.strip()]
        all_tags = list(set(existing_tags + all_entity_tags))
        tag = ",".join(all_tags)

    # Filter out images if provider doesn't support vision
    if not provider_supports_vision(provider, model_id):
        has_images = any(isinstance(m.get("content"), list) for m in messages)
        if has_images:
            messages = filter_images_from_messages(messages)
            logger.warning(
                f"Model {model_id} doesn't support vision - images removed from request"
            )

    # Check for cache hit
    cache_engine = None
    if resolved.has_cache:
        from context.chroma import is_chroma_available
        from routing.smart_cache import SmartCacheEngine

        if is_chroma_available():
            try:
                cache_engine = SmartCacheEngine(resolved.cache_config, registry)
                cache_result = cache_engine.lookup(messages, system_prompt)

                if cache_result.is_cache_hit:
                    cached_response = cache_result.cached_response
                    cached_content = cached_response.get("content", "")
                    cached_tokens = cache_result.cached_tokens
                    cached_cost_saved = cache_result.cached_cost

                    logger.info(
                        f"Cache hit for '{cache_result.cache_name}' "
                        f"(similarity={cache_result.similarity_score:.4f}, "
                        f"tokens={cached_tokens}, cost_saved=${cached_cost_saved:.4f})"
                    )

                    config = resolved.cache_config
                    if hasattr(config, "id"):
                        from db import update_smart_alias_stats

                        update_smart_alias_stats(
                            alias_id=config.id,
                            increment_cache_hits=1,
                            increment_tokens_saved=cached_tokens,
                            increment_cost_saved=cached_cost_saved,
                        )

                    # Build Anthropic-format response for cache hit
                    response_obj = build_anthropic_response(
                        message_id=generate_anthropic_id(),
                        model=model_name,
                        content=cached_content,
                        stop_reason="end_turn",
                        input_tokens=0,
                        output_tokens=cached_tokens,
                    )

                    track_completion(
                        provider_id=provider.name,
                        model_id=model_id,
                        model_name=model_name,
                        endpoint="/v1/messages",
                        input_tokens=0,
                        output_tokens=0,
                        status_code=200,
                        tag=tag,
                        alias=alias_name,
                        router_name=router_name,
                        is_cache_hit=True,
                        cache_name=cache_result.cache_name,
                        cache_tokens_saved=cached_tokens,
                        cache_cost_saved=cached_cost_saved,
                    )

                    return jsonify(response_obj)
            except Exception as cache_err:
                logger.warning(f"Cache lookup failed: {cache_err}")
                cache_engine = None

    # Build options - note max_tokens is required in Anthropic API
    options = {}
    max_tokens = data.get("max_tokens")
    if max_tokens is None:
        # Default to a reasonable value if not provided
        options["max_tokens"] = 4096
    else:
        options["max_tokens"] = max_tokens

    if "temperature" in data:
        options["temperature"] = data["temperature"]
    if "top_p" in data:
        options["top_p"] = data["top_p"]
    if "top_k" in data:
        options["top_k"] = data["top_k"]
    if "stop_sequences" in data:
        options["stop"] = data["stop_sequences"]

    stream = data.get("stream", False)

    logger.info(
        f"Anthropic messages request: provider={provider.name}, model={model_id}, "
        f"messages={len(messages)}, stream={stream}"
    )

    try:
        if stream:
            input_chars = estimate_input_chars(messages, system_prompt)

            from flask import g

            start_time = getattr(g, "start_time", time.time())
            client_ip = getattr(g, "client_ip", "unknown")
            hostname = resolve_hostname(client_ip)
            prov_name = provider.name
            captured_alias = alias_name
            captured_router = router_name
            captured_cache_engine = cache_engine
            captured_messages = messages
            captured_system = system_prompt
            captured_enrichment_result = enrichment_result
            captured_original_messages = original_messages

            def on_stream_complete(
                input_tokens,
                output_tokens,
                error=None,
                cost=None,
                stream_result=None,
                full_content=None,
                action_blocks=None,
            ):
                response_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Track: {prov_name}/{model_id} - {response_time_ms}ms, streaming=True"
                )
                tracker.log_request(
                    timestamp=datetime.now(timezone.utc),
                    client_ip=client_ip,
                    hostname=hostname,
                    tag=tag,
                    provider_id=prov_name,
                    model_id=model_id,
                    endpoint="/v1/messages",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=response_time_ms,
                    status_code=200 if not error else 500,
                    error_message=error,
                    is_streaming=True,
                    cost=cost,
                    reasoning_tokens=stream_result.get("reasoning_tokens")
                    if stream_result
                    else None,
                    cached_input_tokens=stream_result.get("cached_input_tokens")
                    if stream_result
                    else None,
                    cache_creation_tokens=stream_result.get("cache_creation_tokens")
                    if stream_result
                    else None,
                    cache_read_tokens=stream_result.get("cache_read_tokens")
                    if stream_result
                    else None,
                    alias=captured_alias,
                    router_name=captured_router,
                )

                # Store streaming response in cache
                if captured_cache_engine and full_content and not error:
                    try:
                        captured_cache_engine.store_response(
                            messages=captured_messages,
                            system=captured_system,
                            response={"content": full_content},
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost or 0.0,
                        )
                    except Exception as cache_err:
                        logger.warning(
                            f"Failed to store streaming response in cache: {cache_err}"
                        )

                # Update memory if enabled
                if (
                    captured_enrichment_result
                    and getattr(
                        captured_enrichment_result, "memory_update_pending", False
                    )
                    and full_content
                    and not error
                ):
                    try:
                        from db import get_smart_alias_by_name
                        from routing.smart_enricher import SmartEnricherEngine

                        alias = get_smart_alias_by_name(
                            captured_enrichment_result.enricher_name
                        )
                        if alias:
                            engine = SmartEnricherEngine(alias, registry)
                            engine.update_memory_after_response(
                                captured_original_messages, full_content
                            )
                    except Exception as mem_err:
                        logger.warning(f"Failed to update memory: {mem_err}")

                # Execute actions if enabled
                if (
                    captured_enrichment_result
                    and getattr(captured_enrichment_result, "actions_enabled", False)
                    and full_content
                    and not error
                ):
                    try:
                        from actions import execute_actions

                        allowed = getattr(
                            captured_enrichment_result, "allowed_actions", []
                        )
                        results, _ = execute_actions(
                            response_text=full_content,
                            alias_name=captured_enrichment_result.enricher_name,
                            allowed_actions=allowed,
                            session_key=getattr(
                                captured_enrichment_result, "session_key", None
                            ),
                            default_email_account_id=getattr(
                                captured_enrichment_result,
                                "action_email_account_id",
                                None,
                            ),
                            default_calendar_account_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_account_id",
                                None,
                            ),
                            default_calendar_id=getattr(
                                captured_enrichment_result,
                                "action_calendar_id",
                                None,
                            ),
                            default_tasks_account_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_account_id",
                                None,
                            ),
                            default_tasks_list_id=getattr(
                                captured_enrichment_result,
                                "action_tasks_list_id",
                                None,
                            ),
                            default_notification_urls=getattr(
                                captured_enrichment_result,
                                "action_notification_urls",
                                None,
                            ),
                            scheduled_prompts_account_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_account_id",
                                None,
                            ),
                            scheduled_prompts_calendar_id=getattr(
                                captured_enrichment_result,
                                "scheduled_prompts_calendar_id",
                                None,
                            ),
                        )
                        if results:
                            logger.info(
                                f"Executed {len(results)} actions for alias '{captured_enrichment_result.enricher_name}'"
                            )
                    except Exception as action_err:
                        logger.warning(f"Failed to execute actions: {action_err}")

            source_attribution = build_source_attribution(enrichment_result)
            strip_actions = enrichment_result and getattr(
                enrichment_result, "actions_enabled", False
            )

            return Response(
                stream_anthropic_response(
                    provider,
                    model_id,
                    messages,
                    system_prompt,
                    options,
                    model_name,
                    on_complete=on_stream_complete,
                    input_char_count=input_chars,
                    source_attribution=source_attribution,
                    strip_actions=strip_actions,
                ),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            result = provider.chat_completion(
                model_id, messages, system_prompt, options
            )

            llm_content = result.get("content", "")
            finish_reason = result.get("finish_reason", "stop")
            stop_reason = map_finish_reason_to_anthropic(finish_reason)

            response_obj = build_anthropic_response(
                message_id=generate_anthropic_id(),
                model=model_name,
                content=llm_content,
                stop_reason=stop_reason,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
            )

            debug_compare_response(
                "/v1/messages", llm_content, response_obj["content"][0]["text"]
            )

            track_completion(
                provider_id=provider.name,
                model_id=model_id,
                model_name=model_name,
                endpoint="/v1/messages",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status_code=200,
                cost=result.get("cost"),
                reasoning_tokens=result.get("reasoning_tokens"),
                cached_input_tokens=result.get("cached_input_tokens"),
                cache_creation_tokens=result.get("cache_creation_tokens"),
                cache_read_tokens=result.get("cache_read_tokens"),
                tag=tag,
                alias=alias_name,
                router_name=router_name,
            )

            # Store response in cache
            if cache_engine:
                try:
                    cache_engine.store_response(
                        messages=messages,
                        system=system_prompt,
                        response={"content": llm_content},
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        cost=result.get("cost", 0.0) or 0.0,
                    )
                except Exception as cache_err:
                    logger.warning(f"Failed to store response in cache: {cache_err}")

            # Update memory if enabled
            if (
                enrichment_result
                and getattr(enrichment_result, "memory_update_pending", False)
                and llm_content
            ):
                try:
                    from db import get_smart_alias_by_name
                    from routing.smart_enricher import SmartEnricherEngine

                    alias = get_smart_alias_by_name(enrichment_result.enricher_name)
                    if alias:
                        engine = SmartEnricherEngine(alias, registry)
                        engine.update_memory_after_response(
                            original_messages, llm_content
                        )
                except Exception as mem_err:
                    logger.warning(f"Failed to update memory: {mem_err}")

            # Execute actions if enabled
            if (
                enrichment_result
                and getattr(enrichment_result, "actions_enabled", False)
                and llm_content
            ):
                try:
                    from actions import execute_actions

                    allowed = getattr(enrichment_result, "allowed_actions", [])
                    results, cleaned_content = execute_actions(
                        response_text=llm_content,
                        alias_name=enrichment_result.enricher_name,
                        allowed_actions=allowed,
                        session_key=getattr(enrichment_result, "session_key", None),
                        default_email_account_id=getattr(
                            enrichment_result, "action_email_account_id", None
                        ),
                        default_calendar_account_id=getattr(
                            enrichment_result, "action_calendar_account_id", None
                        ),
                        default_calendar_id=getattr(
                            enrichment_result, "action_calendar_id", None
                        ),
                        default_tasks_account_id=getattr(
                            enrichment_result, "action_tasks_account_id", None
                        ),
                        default_tasks_list_id=getattr(
                            enrichment_result, "action_tasks_list_id", None
                        ),
                        default_notification_urls=getattr(
                            enrichment_result, "action_notification_urls", None
                        ),
                        scheduled_prompts_account_id=getattr(
                            enrichment_result, "scheduled_prompts_account_id", None
                        ),
                        scheduled_prompts_calendar_id=getattr(
                            enrichment_result, "scheduled_prompts_calendar_id", None
                        ),
                    )
                    if results:
                        logger.info(
                            f"Executed {len(results)} actions for alias '{enrichment_result.enricher_name}'"
                        )
                    # Update response with cleaned content
                    response_obj["content"][0]["text"] = cleaned_content
                except Exception as action_err:
                    logger.warning(f"Failed to execute actions: {action_err}")

            return jsonify(response_obj)

    except Exception as e:
        logger.error(f"Provider error: {e}")
        track_completion(
            provider_id=provider.name,
            model_id=model_id,
            model_name=model_name,
            endpoint="/v1/messages",
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            error_message=str(e),
            tag=tag,
            alias=alias_name,
            router_name=router_name,
        )
        return anthropic_error_response("api_error", str(e), 500)


@app.route("/v1/embeddings", methods=["POST"])
def openai_embeddings():
    """OpenAI-compatible embeddings endpoint - not supported."""
    return jsonify(
        {
            "error": {
                "message": "Embeddings are not supported. Use a dedicated embedding service.",
                "type": "invalid_request_error",
                "code": "unsupported_operation",
            }
        }
    ), 501


# ============================================================================
# Main
# ============================================================================


def run_admin_server(host: str, port: int, debug: bool):
    """Run the admin server in a separate thread."""
    admin_app = create_admin_app()

    # Use werkzeug directly for the admin server to avoid Flask's reloader issues
    from werkzeug.serving import make_server

    server = make_server(host, port, admin_app, threaded=True)
    logger.info(f"Admin UI server started on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    # API server configuration
    api_port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 11434)))
    api_host = os.environ.get("HOST", "0.0.0.0")

    # Admin server configuration
    admin_port = int(os.environ.get("ADMIN_PORT", 8080))
    admin_host = os.environ.get("ADMIN_HOST", "0.0.0.0")
    admin_enabled = os.environ.get("ADMIN_ENABLED", "true").lower() == "true"

    debug = os.environ.get("DEBUG", "false").lower() == "true"

    # Note: Models load from database via hybrid loader.
    # Initial seeding happens on first startup from LiteLLM data.

    # Initialize admin password
    from admin import init_admin_password

    init_admin_password()

    # Check that at least one provider is configured
    configured = registry.get_configured_providers()
    if not configured:
        logger.warning(
            "No LLM providers configured. Add API keys to .env or "
            "add custom/Ollama providers via the Admin UI."
        )

    provider_names = [p.name for p in configured]

    # Discover and register plugins
    try:
        from plugin_base import discover_plugins

        plugin_counts = discover_plugins()
        total_plugins = sum(plugin_counts.values())
        if total_plugins > 0:
            logger.info(
                f"Loaded {total_plugins} plugins: "
                f"{plugin_counts['document_sources']} document sources, "
                f"{plugin_counts['live_sources']} live sources, "
                f"{plugin_counts['actions']} actions"
            )

            # Seed database entries for live source plugins
            # (must happen AFTER plugin discovery so registry is populated)
            if plugin_counts["live_sources"] > 0:
                try:
                    from db.live_data_sources import seed_plugin_sources

                    created = seed_plugin_sources()
                    if created:
                        logger.info(f"Seeded plugin live data sources: {created}")
                except Exception as e:
                    logger.warning(f"Failed to seed plugin sources: {e}")
    except Exception as e:
        logger.warning(f"Failed to discover plugins: {e}")

    from version import VERSION

    logger.info("=" * 60)
    logger.info(f"LLM Relay v{VERSION}")
    logger.info("=" * 60)
    logger.info(f"API server:   http://{api_host}:{api_port}")

    if admin_enabled:
        logger.info(f"Admin UI:     http://{admin_host}:{admin_port}")
    else:
        logger.info("Admin UI:     disabled (set ADMIN_ENABLED=true to enable)")

    if DEBUG_RESPONSES:
        logger.info("Debug mode:   ENABLED (set DEBUG_RESPONSES=false to disable)")

    logger.info("-" * 60)

    if configured:
        logger.info(f"Configured providers: {', '.join(provider_names)}")
        for provider in configured:
            model_count = len(provider.get_models())
            logger.info(f"  {provider.name}: {model_count} models")
    else:
        logger.info("No providers configured - use Admin UI to configure")

    logger.info("=" * 60)

    # Start admin server in a separate thread
    if admin_enabled:
        admin_thread = threading.Thread(
            target=run_admin_server, args=(admin_host, admin_port, debug), daemon=True
        )
        admin_thread.start()

    # Start RAG indexer (resets stuck jobs and enables scheduled indexing)
    try:
        from rag import start_indexer

        start_indexer()
    except Exception as e:
        logger.warning(f"Failed to start RAG indexer: {e}")

    # Start scheduled prompts scheduler
    try:
        from scheduling import start_prompt_scheduler

        start_prompt_scheduler()
    except Exception as e:
        logger.warning(f"Failed to start prompt scheduler: {e}")

    # Run API server in main thread
    app.run(host=api_host, port=api_port, debug=debug, threaded=True)
