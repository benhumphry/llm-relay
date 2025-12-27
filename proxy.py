#!/usr/bin/env python3
"""
Ollama API Proxy for Anthropic Claude

Presents Claude models via the Ollama API interface, allowing any Ollama-compatible
application to use Claude models seamlessly.

Supports Claude 4.5, Claude 4, and Claude 3.5 model families.
"""

import json
import logging
import os
import random
import string
import time
from datetime import datetime, timezone
from typing import Generator

import anthropic
from flask import Flask, Response, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model mappings: Ollama model name -> Anthropic model ID
# Organised by model family for clarity
MODEL_MAPPINGS = {
    # Claude 4.5 family (latest)
    "claude-4.5-opus": "claude-opus-4-5-20251101",
    "claude-4.5-opus:latest": "claude-opus-4-5-20251101",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-4.5-sonnet:latest": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-4.5-haiku": "claude-haiku-4-5-20251001",
    "claude-4.5-haiku:latest": "claude-haiku-4-5-20251001",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    # Claude 4 family
    "claude-4-opus": "claude-opus-4-20250514",
    "claude-4-opus:latest": "claude-opus-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-4-sonnet": "claude-sonnet-4-20250514",
    "claude-4-sonnet:latest": "claude-sonnet-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    # Claude 3.5 family (for compatibility)
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet:latest": "claude-3-5-sonnet-20241022",
    "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.5-haiku:latest": "claude-3-5-haiku-20241022",
    "claude-haiku-3.5": "claude-3-5-haiku-20241022",
    # Convenience aliases (maps to recommended models)
    "claude-opus": "claude-opus-4-5-20251101",
    "claude-opus:latest": "claude-opus-4-5-20251101",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet:latest": "claude-sonnet-4-5-20250929",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-haiku:latest": "claude-haiku-4-5-20251001",
    # Generic aliases for apps that use simple names
    "claude": "claude-sonnet-4-5-20250929",
    "claude:latest": "claude-sonnet-4-5-20250929",
}

# Model metadata for /api/tags and /api/show endpoints
MODEL_INFO = {
    "claude-opus-4-5-20251101": {
        "family": "claude-4.5",
        "parameter_size": "?B",  # Not publicly disclosed
        "quantization_level": "none",
        "description": "Claude Opus 4.5 - Most capable model, best for complex analysis and OCR",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing", "ocr"],
    },
    "claude-sonnet-4-5-20250929": {
        "family": "claude-4.5",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude Sonnet 4.5 - Balanced performance and speed",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing"],
    },
    "claude-haiku-4-5-20251001": {
        "family": "claude-4.5",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude Haiku 4.5 - Fastest model, ideal for tagging and quick tasks",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing", "fast"],
    },
    "claude-opus-4-20250514": {
        "family": "claude-4",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude Opus 4 - Previous generation flagship",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing"],
    },
    "claude-sonnet-4-20250514": {
        "family": "claude-4",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude Sonnet 4 - Previous generation balanced model",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing"],
    },
    "claude-3-5-sonnet-20241022": {
        "family": "claude-3.5",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude 3.5 Sonnet - Legacy model",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing"],
    },
    "claude-3-5-haiku-20241022": {
        "family": "claude-3.5",
        "parameter_size": "?B",
        "quantization_level": "none",
        "description": "Claude 3.5 Haiku - Legacy fast model",
        "context_length": 200000,
        "capabilities": ["vision", "analysis", "coding", "writing", "fast"],
    },
}


# Initialise Anthropic client
def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    return anthropic.Anthropic(api_key=api_key)


def resolve_model(ollama_model: str) -> str:
    """Resolve an Ollama model name to an Anthropic model ID."""
    model = ollama_model.lower().strip()

    if model in MODEL_MAPPINGS:
        return MODEL_MAPPINGS[model]

    # Check if it's already a valid Anthropic model ID
    if model in MODEL_INFO:
        return model

    # Try without :latest suffix
    if model.endswith(":latest"):
        base_model = model[:-7]
        if base_model in MODEL_MAPPINGS:
            return MODEL_MAPPINGS[base_model]

    logger.warning(f'Unknown model "{ollama_model}", defaulting to claude-sonnet-4.5')
    return "claude-sonnet-4-5-20250929"


def convert_messages(ollama_messages: list) -> tuple[str | None, list]:
    """
    Convert Ollama message format to Anthropic format.

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = None
    anthropic_messages = []

    for msg in ollama_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Anthropic handles system prompts separately
            system_prompt = content
        elif role in ("user", "assistant"):
            # Handle images if present
            if "images" in msg and msg["images"]:
                content_blocks = []
                for img in msg["images"]:
                    # Ollama sends base64 images
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Default, could be detected
                                "data": img,
                            },
                        }
                    )
                content_blocks.append(
                    {
                        "type": "text",
                        "text": content,
                    }
                )
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": content_blocks,
                    }
                )
            else:
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

    return system_prompt, anthropic_messages


def stream_response(
    client: anthropic.Anthropic,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
) -> Generator[str, None, None]:
    """Stream response from Claude in Ollama format."""

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": options.get("num_predict", options.get("max_tokens", 4096)),
        "stream": True,
    }

    if system:
        kwargs["system"] = system

    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]
    if "top_p" in options:
        kwargs["top_p"] = options["top_p"]
    if "top_k" in options:
        kwargs["top_k"] = options["top_k"]
    if "stop" in options:
        kwargs["stop_sequences"] = (
            options["stop"] if isinstance(options["stop"], list) else [options["stop"]]
        )

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                # Ollama streaming format
                chunk = {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "done": False,
                }
                yield json.dumps(chunk) + "\n"

        # Final message
        final = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }
        yield json.dumps(final) + "\n"

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        error_response = {
            "error": str(e),
            "done": True,
        }
        yield json.dumps(error_response) + "\n"


def non_streaming_response(
    client: anthropic.Anthropic,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
) -> dict:
    """Get non-streaming response from Claude in Ollama format."""

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": options.get("num_predict", options.get("max_tokens", 4096)),
    }

    if system:
        kwargs["system"] = system

    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]
    if "top_p" in options:
        kwargs["top_p"] = options["top_p"]
    if "top_k" in options:
        kwargs["top_k"] = options["top_k"]
    if "stop" in options:
        kwargs["stop_sequences"] = (
            options["stop"] if isinstance(options["stop"], list) else [options["stop"]]
        )

    response = client.messages.create(**kwargs)

    content = ""
    for block in response.content:
        if hasattr(block, "text"):
            content += block.text

    return {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.usage.input_tokens,
        "eval_count": response.usage.output_tokens,
        "eval_duration": 0,
    }


# ============================================================================
# Ollama API Endpoints
# ============================================================================


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return "Ollama is running"


@app.route("/api/tags", methods=["GET"])
def list_models():
    """List available models in Ollama format."""
    models = []

    # Create unique model entries (avoid duplicates from aliases)
    seen_models = set()
    for ollama_name, anthropic_id in MODEL_MAPPINGS.items():
        if anthropic_id in seen_models:
            continue
        if ":" in ollama_name:  # Skip :latest variants for cleaner list
            continue

        seen_models.add(anthropic_id)
        info = MODEL_INFO.get(anthropic_id, {})

        models.append(
            {
                "name": ollama_name,
                "model": ollama_name,
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "size": 0,  # Not applicable for API models
                "digest": anthropic_id,
                "details": {
                    "parent_model": "",
                    "format": "api",
                    "family": info.get("family", "claude"),
                    "families": [info.get("family", "claude")],
                    "parameter_size": info.get("parameter_size", "?B"),
                    "quantization_level": info.get("quantization_level", "none"),
                },
            }
        )

    return jsonify({"models": models})


@app.route("/api/show", methods=["POST"])
def show_model():
    """Show model details in Ollama format."""
    data = request.get_json() or {}
    model_name = data.get("name", data.get("model", ""))

    anthropic_id = resolve_model(model_name)
    info = MODEL_INFO.get(anthropic_id, {})

    return jsonify(
        {
            "modelfile": f"# Anthropic Claude Model: {anthropic_id}",
            "parameters": f"temperature 0.7\nnum_ctx {info.get('context_length', 200000)}",
            "template": "{{ .System }}\n\n{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "api",
                "family": info.get("family", "claude"),
                "families": [info.get("family", "claude")],
                "parameter_size": info.get("parameter_size", "?B"),
                "quantization_level": info.get("quantization_level", "none"),
            },
            "model_info": {
                "anthropic.model_id": anthropic_id,
                "anthropic.context_length": info.get("context_length", 200000),
                "anthropic.description": info.get("description", ""),
                "anthropic.capabilities": info.get("capabilities", []),
            },
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat completion endpoint - main inference endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")
    anthropic_model = resolve_model(model_name)

    ollama_messages = data.get("messages", [])
    system_prompt, messages = convert_messages(ollama_messages)

    options = data.get("options", {})
    stream = data.get("stream", True)

    logger.info(
        f"Chat request: model={anthropic_model}, messages={len(messages)}, stream={stream}"
    )

    try:
        client = get_client()

        if stream:
            return Response(
                stream_response(
                    client, anthropic_model, messages, system_prompt, options
                ),
                mimetype="application/x-ndjson",
            )
        else:
            response = non_streaming_response(
                client, anthropic_model, messages, system_prompt, options
            )
            return jsonify(response)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({"error": str(e)}), 500
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate endpoint for compatibility.
    Converts to chat format internally.
    """
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")
    anthropic_model = resolve_model(model_name)

    prompt = data.get("prompt", "")
    system = data.get("system", None)

    messages = [{"role": "user", "content": prompt}]

    # Handle images
    if "images" in data and data["images"]:
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

    options = data.get("options", {})
    stream = data.get("stream", True)

    logger.info(f"Generate request: model={anthropic_model}, stream={stream}")

    try:
        client = get_client()

        if stream:
            return Response(
                stream_response(client, anthropic_model, messages, system, options),
                mimetype="application/x-ndjson",
            )
        else:
            response = non_streaming_response(
                client, anthropic_model, messages, system, options
            )
            # Generate format uses 'response' instead of 'message'
            response["response"] = response.pop("message", {}).get("content", "")
            return jsonify(response)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({"error": str(e)}), 500
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    """
    Embeddings endpoint - not supported by Claude.
    Returns an error directing users to use a local embedding model.
    """
    return jsonify(
        {
            "error": "Embeddings are not supported by Claude. Use a local embedding model like nomic-embed-text with Ollama."
        }
    ), 501


@app.route("/api/pull", methods=["POST"])
def pull_model():
    """
    Pull endpoint - not applicable for API models.
    Returns success to satisfy clients that check for model availability.
    """
    data = request.get_json() or {}
    model = data.get("name", "claude")

    # Return a fake successful pull response
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
    return jsonify({"version": "0.1.0-claude-proxy"})


# ============================================================================
# OpenAI-Compatible API Endpoints (/v1/*)
# ============================================================================


def convert_openai_messages(openai_messages: list) -> tuple[str | None, list]:
    """
    Convert OpenAI message format to Anthropic format.

    OpenAI format supports:
    - role: system, user, assistant
    - content: string or array of content parts (text, image_url)

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = None
    anthropic_messages = []

    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Anthropic handles system prompts separately
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                # Extract text from content array
                system_prompt = " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
        elif role in ("user", "assistant"):
            if isinstance(content, str):
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            elif isinstance(content, list):
                # Handle content array with text and images
                content_blocks = []
                for part in content:
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": part.get("text", ""),
                            }
                        )
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = (
                            image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else image_url
                        )

                        # Handle base64 data URLs
                        if url.startswith("data:"):
                            # Parse data URL: data:image/jpeg;base64,<data>
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
                                logger.warning(f"Failed to parse image data URL")
                        else:
                            # URL-based image - Anthropic supports this directly
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url,
                                    },
                                }
                            )

                if content_blocks:
                    anthropic_messages.append(
                        {
                            "role": role,
                            "content": content_blocks,
                        }
                    )

    return system_prompt, anthropic_messages


def generate_openai_id(prefix: str = "chatcmpl") -> str:
    """Generate an OpenAI-style ID."""
    chars = string.ascii_letters + string.digits
    suffix = "".join(random.choices(chars, k=24))
    return f"{prefix}-{suffix}"


def stream_openai_response(
    client: anthropic.Anthropic,
    model: str,
    messages: list,
    system: str | None,
    options: dict,
    request_model: str,
) -> Generator[str, None, None]:
    """Stream response from Claude in OpenAI SSE format."""

    response_id = generate_openai_id()
    created = int(time.time())

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": options.get("max_tokens", 4096),
        "stream": True,
    }

    if system:
        kwargs["system"] = system

    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]
    if "top_p" in options:
        kwargs["top_p"] = options["top_p"]
    if "top_k" in options:
        kwargs["top_k"] = options["top_k"]
    if "stop" in options:
        stop = options["stop"]
        kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": text,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error during streaming: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "api_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """List available models in OpenAI format."""
    models = []

    # Create unique model entries
    seen_models = set()
    for ollama_name, anthropic_id in MODEL_MAPPINGS.items():
        if anthropic_id in seen_models:
            continue
        if ":" in ollama_name:  # Skip :latest variants
            continue

        seen_models.add(anthropic_id)
        info = MODEL_INFO.get(anthropic_id, {})

        models.append(
            {
                "id": ollama_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic",
            }
        )

    return jsonify(
        {
            "object": "list",
            "data": models,
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")
    anthropic_model = resolve_model(model_name)

    openai_messages = data.get("messages", [])
    system_prompt, messages = convert_openai_messages(openai_messages)

    # Build options from OpenAI parameters
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
        f"OpenAI chat request: model={anthropic_model}, messages={len(messages)}, stream={stream}"
    )

    try:
        client = get_client()

        if stream:
            return Response(
                stream_openai_response(
                    client,
                    anthropic_model,
                    messages,
                    system_prompt,
                    options,
                    model_name,
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
            kwargs = {
                "model": anthropic_model,
                "messages": messages,
                "max_tokens": options.get("max_tokens", 4096),
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if "temperature" in options:
                kwargs["temperature"] = options["temperature"]
            if "top_p" in options:
                kwargs["top_p"] = options["top_p"]
            if "stop" in options:
                stop = options["stop"]
                kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

            response = client.messages.create(**kwargs)

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

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
                                "content": content,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    },
                }
            )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "configuration_error",
                }
            }
        ), 500
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "anthropic_error",
                }
            }
        ), 500


@app.route("/v1/completions", methods=["POST"])
def openai_completions():
    """OpenAI-compatible text completions endpoint."""
    data = request.get_json() or {}

    model_name = data.get("model", "claude-sonnet")
    anthropic_model = resolve_model(model_name)

    prompt = data.get("prompt", "")

    # Handle prompt as string or array
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    messages = [{"role": "user", "content": prompt}]

    # Build options from OpenAI parameters
    options = {}
    if "max_tokens" in data:
        options["max_tokens"] = data["max_tokens"]
    else:
        options["max_tokens"] = 4096

    if "temperature" in data:
        options["temperature"] = data["temperature"]
    if "top_p" in data:
        options["top_p"] = data["top_p"]
    if "stop" in data:
        options["stop"] = data["stop"]

    stream = data.get("stream", False)

    logger.info(f"OpenAI completions request: model={anthropic_model}, stream={stream}")

    try:
        client = get_client()

        if stream:
            # Streaming completions
            def stream_completions():
                response_id = generate_openai_id("cmpl")
                created = int(time.time())

                kwargs = {
                    "model": anthropic_model,
                    "messages": messages,
                    "max_tokens": options.get("max_tokens", 4096),
                    "stream": True,
                }

                if "temperature" in options:
                    kwargs["temperature"] = options["temperature"]
                if "top_p" in options:
                    kwargs["top_p"] = options["top_p"]
                if "stop" in options:
                    stop = options["stop"]
                    kwargs["stop_sequences"] = (
                        stop if isinstance(stop, list) else [stop]
                    )

                try:
                    with client.messages.stream(**kwargs) as stream_obj:
                        for text in stream_obj.text_stream:
                            chunk = {
                                "id": response_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": text,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Final chunk
                    final_chunk = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                except anthropic.APIError as e:
                    logger.error(f"Anthropic API error during streaming: {e}")
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
            # Non-streaming response
            kwargs = {
                "model": anthropic_model,
                "messages": messages,
                "max_tokens": options.get("max_tokens", 4096),
            }

            if "temperature" in options:
                kwargs["temperature"] = options["temperature"]
            if "top_p" in options:
                kwargs["top_p"] = options["top_p"]
            if "stop" in options:
                stop = options["stop"]
                kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

            response = client.messages.create(**kwargs)

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return jsonify(
                {
                    "id": generate_openai_id("cmpl"),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": content,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    },
                }
            )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "configuration_error",
                }
            }
        ), 500
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "anthropic_error",
                }
            }
        ), 500


@app.route("/v1/embeddings", methods=["POST"])
def openai_embeddings():
    """
    OpenAI-compatible embeddings endpoint - not supported by Claude.
    Returns an error directing users to use a different embedding service.
    """
    return jsonify(
        {
            "error": {
                "message": "Embeddings are not supported by Claude. Use a dedicated embedding service or local model.",
                "type": "invalid_request_error",
                "code": "unsupported_operation",
            }
        }
    ), 501


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 11434)))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    # Validate API key on startup
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        exit(1)

    logger.info(f"Starting Ollama-Claude proxy on {host}:{port}")
    logger.info(f"Available models: {list(set(MODEL_MAPPINGS.values()))}")

    app.run(host=host, port=port, debug=debug, threaded=True)
