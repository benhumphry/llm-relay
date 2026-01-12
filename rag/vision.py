"""
Vision model provider abstraction for Docling document parsing.

Supports multiple vision backends for document understanding:
- local: Bundled Docling models (default, runs on CPU/GPU)
- ollama: Any Ollama instance with a vision model (e.g., granite3.2-vision)
- provider: Any configured LLM provider with vision support via OpenAI-compatible API
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision model provider."""

    provider_type: str  # "local", "ollama", or provider name
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 300  # Vision processing can be slow


def get_document_converter(
    vision_provider: str = "local",
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
) -> "DocumentConverter":
    """
    Get a configured Docling DocumentConverter.

    Args:
        vision_provider: "local", "ollama:<instance>", or provider name
        vision_model: Model to use for vision processing
        vision_ollama_url: Ollama URL when using ollama provider

    Returns:
        Configured DocumentConverter instance
    """
    from docling.document_converter import DocumentConverter

    # Local processing (default) - use standard DocumentConverter with Tesseract OCR
    if vision_provider == "local" or not vision_provider:
        logger.debug("Using local Docling processing with Tesseract OCR")
        return _get_local_converter()

    # Ollama instance
    if vision_provider.startswith("ollama:") or vision_provider == "ollama":
        return _get_ollama_converter(vision_provider, vision_model, vision_ollama_url)

    # External provider via OpenAI-compatible API
    return _get_provider_converter(vision_provider, vision_model)


def _get_local_converter() -> "DocumentConverter":
    """Get DocumentConverter configured for local processing with Tesseract OCR."""
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Configure Tesseract OCR for scanned documents
    ocr_options = TesseractOcrOptions(lang=["eng"])

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
    )

    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _get_ollama_converter(
    vision_provider: str,
    vision_model: Optional[str],
    vision_ollama_url: Optional[str],
) -> "DocumentConverter":
    """Get DocumentConverter configured for Ollama vision model."""
    from docling.datamodel.pipeline_options import (
        ApiVlmOptions,
        ResponseFormat,
        VlmPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Parse ollama URL
    if vision_provider.startswith("ollama:"):
        instance_name = vision_provider.split(":", 1)[1]
        # Look up the Ollama instance URL
        ollama_url = _get_ollama_instance_url(instance_name)
    else:
        ollama_url = vision_ollama_url

    if not ollama_url:
        logger.warning("No Ollama URL configured, falling back to local processing")
        return DocumentConverter()

    model = vision_model or "granite3.2-vision:latest"

    logger.info(f"Using Ollama vision model: {model} at {ollama_url}")

    # Configure API options for Ollama
    # prompt is required - this is the system prompt for document understanding
    vlm_options = ApiVlmOptions(
        url=f"{ollama_url.rstrip('/')}/v1/chat/completions",
        params={"model": model},
        prompt="You are a document understanding assistant. Extract and describe the content of this document image accurately, preserving structure and formatting.",
        timeout=300,
        scale=2.0,
        response_format=ResponseFormat.MARKDOWN,
    )

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,
        vlm_options=vlm_options,
    )

    # Create converter with VLM pipeline for PDFs and images
    # Must use VlmPipeline class with VlmPipelineOptions
    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
            "image": ImageFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
        }
    )


def _get_provider_converter(
    provider_name: str,
    vision_model: Optional[str],
) -> "DocumentConverter":
    """Get DocumentConverter configured for an external provider."""
    from docling.datamodel.pipeline_options import (
        ApiVlmOptions,
        ResponseFormat,
        VlmPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Look up provider from registry
    from providers import registry

    provider = registry.get_provider(provider_name)
    if not provider:
        logger.warning(
            f"Provider '{provider_name}' not found, falling back to local processing"
        )
        return DocumentConverter()

    if not provider.is_available():
        logger.warning(
            f"Provider '{provider_name}' not available, falling back to local processing"
        )
        return DocumentConverter()

    # Get provider config
    base_url = provider.base_url
    api_key = provider.get_api_key()

    if not base_url or not api_key:
        logger.warning(
            f"Provider '{provider_name}' missing URL or API key, falling back to local"
        )
        return DocumentConverter()

    # Default models for known providers
    if not vision_model:
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-1.5-flash",
            "openrouter": "openai/gpt-4o-mini",
            "together": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        }
        vision_model = default_models.get(provider_name.lower(), "gpt-4o-mini")

    logger.info(f"Using {provider_name} vision model: {vision_model}")

    # Build the chat completions URL
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    if "chat/completions" in base_url:
        chat_url = base_url

    # Configure API options
    # prompt is required - this is the system prompt for document understanding
    vlm_options = ApiVlmOptions(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        params={"model": vision_model},
        prompt="You are a document understanding assistant. Extract and describe the content of this document image accurately, preserving structure and formatting.",
        timeout=300,
        scale=2.0,
        response_format=ResponseFormat.MARKDOWN,
    )

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,
        vlm_options=vlm_options,
    )

    # Must use VlmPipeline class with VlmPipelineOptions
    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
            "image": ImageFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
        }
    )


def _get_ollama_instance_url(instance_name: str) -> Optional[str]:
    """Look up Ollama instance URL from database or registry."""
    from providers import registry

    # Check if it's a registered provider
    provider = registry.get_provider(instance_name)
    if provider and hasattr(provider, "base_url"):
        return provider.base_url

    # Check database for custom Ollama instances
    try:
        from db import get_db_context
        from db.models import OllamaInstance

        with get_db_context() as session:
            instance = (
                session.query(OllamaInstance)
                .filter(OllamaInstance.name == instance_name)
                .first()
            )
            if instance:
                return instance.base_url
    except Exception as e:
        logger.warning(f"Error looking up Ollama instance: {e}")

    return None


def get_vision_config_from_settings() -> VisionConfig:
    """
    Get vision model configuration from application settings.

    Used for web scraping in Smart Augmentors.
    """
    from db import get_db_context
    from db.models import Setting

    with get_db_context() as session:
        vision_provider = (
            session.query(Setting)
            .filter(Setting.key == Setting.KEY_VISION_PROVIDER)
            .first()
        )
        vision_model = (
            session.query(Setting)
            .filter(Setting.key == Setting.KEY_VISION_MODEL)
            .first()
        )
        vision_ollama_url = (
            session.query(Setting)
            .filter(Setting.key == Setting.KEY_VISION_OLLAMA_URL)
            .first()
        )

        return VisionConfig(
            provider_type=vision_provider.value if vision_provider else "local",
            model_name=vision_model.value if vision_model else None,
            base_url=vision_ollama_url.value if vision_ollama_url else None,
        )
