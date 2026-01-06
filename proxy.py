#!/usr/bin/env python3
"""
Multi-Provider LLM Proxy

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

# Import usage tracking
from tracking import extract_tag, get_client_ip, resolve_hostname, tracker

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

    # Extract tag from request
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
        cost=cost,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
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


def stream_ollama_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    on_complete: callable = None,
    input_char_count: int = 0,
) -> Generator[str, None, None]:
    """Stream response in Ollama NDJSON format."""
    output_chars = 0
    llm_chunks = []  # Collect raw LLM response for debug comparison
    client_chunks = []  # Collect what we send to client

    try:
        for text in provider.chat_completion_stream(model, messages, system, options):
            output_chars += len(text)
            llm_chunks.append(text)  # Raw LLM text

            chunk = {
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": text},
                "done": False,
            }
            chunk_str = json.dumps(chunk) + "\n"
            client_chunks.append(text)  # Track content we're sending
            yield chunk_str

        # Final message
        final = {
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
        yield json.dumps(final) + "\n"

        # Debug comparison: raw LLM response vs what client receives
        llm_full = "".join(llm_chunks)
        client_full = "".join(client_chunks)
        debug_compare_response("/api/chat (stream)", llm_full, client_full)

        # Get streaming result (cost, tokens) if provider supports it (e.g., OpenRouter)
        stream_result = None
        if hasattr(provider, "get_last_stream_result"):
            stream_result = provider.get_last_stream_result()

        # Call completion callback with tokens and optional cost
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
            # Pass the full stream_result for extended token info
            on_complete(
                input_tokens, output_tokens, cost=cost, stream_result=stream_result
            )

    except Exception as e:
        logger.error(f"Provider error during streaming: {e}")
        error_response = {"error": str(e), "done": True}
        yield json.dumps(error_response) + "\n"
        if on_complete:
            on_complete(0, 0, error=str(e))


def stream_openai_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    request_model: str,
    on_complete: callable = None,
    input_char_count: int = 0,
) -> Generator[str, None, None]:
    """Stream response in OpenAI SSE format."""
    response_id = generate_openai_id()
    created = int(time.time())
    output_chars = 0
    llm_chunks = []  # Collect raw LLM response for debug comparison
    client_chunks = []  # Collect what we send to client

    try:
        for text in provider.chat_completion_stream(model, messages, system, options):
            output_chars += len(text)
            llm_chunks.append(text)  # Raw LLM text

            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
            client_chunks.append(text)  # Track content we're sending
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        # Debug comparison: raw LLM response vs what client receives
        llm_full = "".join(llm_chunks)
        client_full = "".join(client_chunks)
        debug_compare_response("/v1/chat/completions (stream)", llm_full, client_full)

        # Get streaming result (cost, tokens) if provider supports it (e.g., OpenRouter)
        stream_result = None
        if hasattr(provider, "get_last_stream_result"):
            stream_result = provider.get_last_stream_result()

        # Call completion callback with tokens and optional cost
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
            # Pass the full stream_result for extended token info
            on_complete(
                input_tokens, output_tokens, cost=cost, stream_result=stream_result
            )

    except Exception as e:
        logger.error(f"Provider error during streaming: {e}")
        error_chunk = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
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
        provider, model_id = registry.resolve_model(model_name)
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

    try:
        provider, model_id = registry.resolve_model(clean_model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ollama_messages = data.get("messages", [])
    system_prompt, messages = convert_ollama_messages(ollama_messages)

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
            )
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
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate endpoint for compatibility. Converts to chat format internally."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    # Extract tag early so we use clean model name for resolution
    tag, clean_model_name = extract_tag(request, model_name)

    try:
        provider, model_id = registry.resolve_model(clean_model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prompt = data.get("prompt", "")
    system = data.get("system", None)

    # Build messages
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
    return jsonify({"version": "1.2.0-multi-provider"})


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

    try:
        provider, model_id = registry.resolve_model(clean_model_name)
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

    openai_messages = data.get("messages", [])
    system_prompt, messages = convert_openai_messages(openai_messages)

    # Filter out images if provider doesn't support vision
    if not provider_supports_vision(provider, model_id):
        has_images = any(isinstance(m.get("content"), list) for m in messages)
        if has_images:
            messages = filter_images_from_messages(messages)
            logger.warning(
                f"Model {model_id} doesn't support vision - images removed from request"
            )

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
            )

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

    try:
        provider, model_id = registry.resolve_model(clean_model_name)
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

    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    messages = [{"role": "user", "content": prompt}]

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

    logger.info("=" * 60)
    logger.info("Multi-Provider LLM Proxy v3.0")
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

    # Run API server in main thread
    app.run(host=api_host, port=api_port, debug=debug, threaded=True)
