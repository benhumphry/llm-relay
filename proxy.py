#!/usr/bin/env python3
"""
Multi-Provider LLM Proxy

Presents multiple LLM providers (Anthropic, OpenAI, Google Gemini, Perplexity, etc.)
via both Ollama and OpenAI-compatible API interfaces.

This allows any Ollama or OpenAI-compatible application to use models from
multiple providers seamlessly.
"""

import json
import logging
import random
import string
import time
from datetime import datetime, timezone
from typing import Generator

from flask import Flask, Response, jsonify, request

# Import the provider registry
from providers import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ============================================================================
# Request Logging Middleware
# ============================================================================


@app.before_request
def log_request():
    """Log all incoming requests for debugging."""
    logger.info(f">>> {request.method} {request.path}")
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


def stream_ollama_response(
    provider, model: str, messages: list, system: str | None, options: dict
) -> Generator[str, None, None]:
    """Stream response in Ollama NDJSON format."""
    try:
        for text in provider.chat_completion_stream(model, messages, system, options):
            chunk = {
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": text},
                "done": False,
            }
            yield json.dumps(chunk) + "\n"

        # Final message
        final = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }
        yield json.dumps(final) + "\n"

    except Exception as e:
        logger.error(f"Provider error during streaming: {e}")
        error_response = {"error": str(e), "done": True}
        yield json.dumps(error_response) + "\n"


def stream_openai_response(
    provider,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    request_model: str,
) -> Generator[str, None, None]:
    """Stream response in OpenAI SSE format."""
    response_id = generate_openai_id()
    created = int(time.time())

    try:
        for text in provider.chat_completion_stream(model, messages, system, options):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
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

    except Exception as e:
        logger.error(f"Provider error during streaming: {e}")
        error_chunk = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


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
                    "model_info": {
                        "provider": provider.name,
                        "model_id": model_id,
                        "context_length": info.context_length,
                        "description": info.description,
                        "capabilities": info.capabilities,
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

    try:
        provider, model_id = registry.resolve_model(model_name)
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
            return Response(
                stream_ollama_response(
                    provider, model_id, messages, system_prompt, options
                ),
                mimetype="application/x-ndjson",
            )
        else:
            result = provider.chat_completion(
                model_id, messages, system_prompt, options
            )
            return jsonify(
                {
                    "model": model_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": result["content"]},
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": result.get("input_tokens", 0),
                    "eval_count": result.get("output_tokens", 0),
                    "eval_duration": 0,
                }
            )

    except Exception as e:
        logger.error(f"Provider error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate endpoint for compatibility. Converts to chat format internally."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")

    try:
        provider, model_id = registry.resolve_model(model_name)
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
            return Response(
                stream_ollama_response(provider, model_id, messages, system, options),
                mimetype="application/x-ndjson",
            )
        else:
            result = provider.chat_completion(model_id, messages, system, options)
            return jsonify(
                {
                    "model": model_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": result["content"],
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": result.get("input_tokens", 0),
                    "eval_count": result.get("output_tokens", 0),
                    "eval_duration": 0,
                }
            )

    except Exception as e:
        logger.error(f"Provider error: {e}")
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

    try:
        provider, model_id = registry.resolve_model(model_name)
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
            return Response(
                stream_openai_response(
                    provider, model_id, messages, system_prompt, options, model_name
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

            return jsonify(
                {
                    "id": generate_openai_id(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result["content"],
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
            )

    except Exception as e:
        logger.error(f"Provider error: {e}")
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

    try:
        provider, model_id = registry.resolve_model(model_name)
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

            def stream_completions():
                response_id = generate_openai_id("cmpl")
                created = int(time.time())

                try:
                    for text in provider.chat_completion_stream(
                        model_id, messages, None, options
                    ):
                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "text": text, "finish_reason": None}
                            ],
                        }
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

                except Exception as e:
                    logger.error(f"Provider error during streaming: {e}")
                    error_chunk = {"error": {"message": str(e), "type": "api_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n"

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

            return jsonify(
                {
                    "id": generate_openai_id("cmpl"),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {"index": 0, "text": result["content"], "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": result.get("input_tokens", 0),
                        "completion_tokens": result.get("output_tokens", 0),
                        "total_tokens": result.get("input_tokens", 0)
                        + result.get("output_tokens", 0),
                    },
                }
            )

    except Exception as e:
        logger.error(f"Provider error: {e}")
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

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 11434)))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    # Check that at least one provider is configured
    configured = registry.get_configured_providers()
    if not configured:
        logger.error(
            "No LLM providers configured. Set at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, PERPLEXITY_API_KEY"
        )
        exit(1)

    provider_names = [p.name for p in configured]
    logger.info(f"Starting Multi-Provider LLM Proxy on {host}:{port}")
    logger.info(f"Configured providers: {provider_names}")

    # Log available models count per provider
    for provider in configured:
        model_count = len(provider.get_models())
        alias_count = len(provider.get_aliases())
        logger.info(f"  - {provider.name}: {model_count} models, {alias_count} aliases")

    app.run(host=host, port=port, debug=debug, threaded=True)
