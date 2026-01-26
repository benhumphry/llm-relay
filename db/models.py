"""
SQLAlchemy models for LLM Relay.

These models store provider configuration, model definitions, and settings
in the database.
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Provider(Base):
    """
    LLM provider configuration.

    All providers are stored in the database, seeded from defaults on first run.
    """

    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'anthropic', 'openai-compatible', 'ollama', 'openrouter', 'perplexity', 'gemini'
    base_url: Mapped[Optional[str]] = mapped_column(String(500))
    api_key_env: Mapped[Optional[str]] = mapped_column(String(100))  # Env var name
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(
        Text
    )  # Encrypted key stored in DB

    # Display
    display_name: Mapped[Optional[str]] = mapped_column(String(100))  # "Anthropic"

    # Source tracking
    source: Mapped[str] = mapped_column(
        String(20), default="system"
    )  # "system", "custom"

    # State
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    models: Mapped[list["Model"]] = relationship(
        "Model", back_populates="provider", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        import os

        # Check if API key is available (either from env var or stored encrypted)
        has_api_key = bool(self.api_key_encrypted)
        if not has_api_key and self.api_key_env:
            has_api_key = bool(os.environ.get(self.api_key_env))

        return {
            "id": self.id,
            "type": self.type,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "display_name": self.display_name,
            "source": self.source,
            "has_api_key": has_api_key,
            "enabled": self.enabled,
            "display_order": self.display_order,
            "model_count": len(self.models) if self.models else 0,
        }


class Model(Base):
    """
    Model definition for a provider.

    All models are stored in the database:
    - source="litellm": Seeded from LiteLLM pricing data
    - source="custom": User-created models
    - source="ollama": Dynamically discovered from Ollama API
    """

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    provider_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("providers.id"), primary_key=True
    )

    # Source tracking
    source: Mapped[str] = mapped_column(
        String(20), default="litellm"
    )  # "litellm", "custom", "ollama"
    last_synced: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )  # When last updated from LiteLLM

    # Basic info
    family: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    context_length: Mapped[int] = mapped_column(Integer, default=128000)
    capabilities_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    unsupported_params_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    supports_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True)
    use_max_completion_tokens: Mapped[bool] = mapped_column(Boolean, default=False)

    # Cost per million tokens (USD)
    input_cost: Mapped[Optional[float]] = mapped_column(Float)
    output_cost: Mapped[Optional[float]] = mapped_column(Float)
    # Cache read multiplier (e.g., 0.1 = 10% of input cost for Anthropic)
    cache_read_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Cache write multiplier (e.g., 1.25 = 125% of input cost for Anthropic)
    cache_write_multiplier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # State
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    provider: Mapped["Provider"] = relationship("Provider", back_populates="models")

    @property
    def capabilities(self) -> list[str]:
        """Get capabilities as a list."""
        return json.loads(self.capabilities_json) if self.capabilities_json else []

    @capabilities.setter
    def capabilities(self, value: list[str]):
        """Set capabilities from a list."""
        self.capabilities_json = json.dumps(value)

    @property
    def unsupported_params(self) -> list[str]:
        """Get unsupported params as a list."""
        return (
            json.loads(self.unsupported_params_json)
            if self.unsupported_params_json
            else []
        )

    @unsupported_params.setter
    def unsupported_params(self, value: list[str]):
        """Set unsupported params from a list."""
        self.unsupported_params_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "source": self.source,
            "last_synced": self.last_synced.isoformat() if self.last_synced else None,
            "family": self.family,
            "description": self.description,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "unsupported_params": self.unsupported_params,
            "supports_system_prompt": self.supports_system_prompt,
            "use_max_completion_tokens": self.use_max_completion_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_read_multiplier": self.cache_read_multiplier,
            "cache_write_multiplier": self.cache_write_multiplier,
            "enabled": self.enabled,
        }


class Setting(Base):
    """
    Key-value settings storage.

    Stores configuration like default provider/model, admin password hash, etc.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Well-known setting keys
    KEY_DEFAULT_PROVIDER = "default_provider"
    KEY_DEFAULT_MODEL = "default_model"
    KEY_ADMIN_PASSWORD_HASH = "admin_password_hash"
    KEY_SESSION_SECRET = "session_secret"

    # Web search settings (v1.2.1)
    KEY_WEB_SEARCH_PROVIDER = "web_search_provider"  # "searxng" or "perplexity"
    KEY_WEB_SEARCH_URL = "web_search_url"  # Override URL for SearXNG
    KEY_WEB_SCRAPER_PROVIDER = "web_scraper_provider"  # "builtin" or "jina"
    KEY_WEB_PDF_PARSER = "web_pdf_parser"  # "docling", "pypdf", "jina" for web scraping
    KEY_RAG_PDF_PARSER = "rag_pdf_parser"  # "docling", "pypdf" for RAG indexing
    KEY_WEB_RERANK_PROVIDER = "web_rerank_provider"  # "local" or "jina"
    KEY_WEB_CRAWL_PROVIDER = (
        "web_crawl_provider"  # "builtin" or "jina" for website crawling
    )
    # Note: Jina API key is configured via JINA_API_KEY env var

    # Embedding settings for Smart RAGs (v1.6)
    KEY_EMBEDDING_PROVIDER = (
        "embedding_provider"  # "local", "ollama:<instance>", or provider name
    )
    KEY_EMBEDDING_MODEL = "embedding_model"  # Model name (e.g., "nomic-embed-text")
    KEY_EMBEDDING_OLLAMA_URL = "embedding_ollama_url"  # Ollama URL when using ollama

    # Vision model settings for document parsing (v3.9)
    KEY_VISION_PROVIDER = (
        "vision_provider"  # "local", "ollama:<instance>", or provider name
    )
    KEY_VISION_MODEL = "vision_model"  # Model name (e.g., "granite3.2-vision:latest")
    KEY_VISION_OLLAMA_URL = "vision_ollama_url"  # Ollama URL when using ollama provider
    # Document store intelligence (auto-generated themes/summary)
    KEY_DOCSTORE_INTELLIGENCE_MODEL = "docstore_intelligence_model"
    # Usage tracking settings (v2.1)
    KEY_TRACKING_ENABLED = "tracking_enabled"
    KEY_DNS_RESOLUTION_ENABLED = "dns_resolution_enabled"
    KEY_RETENTION_DAYS = "retention_days"

    # Live Data Cache settings (TTLs in seconds, 0 = never expires)
    KEY_LIVE_CACHE_TTL_PRICE = (
        "live_cache_ttl_price"  # Stock prices, default 900 (15 min)
    )
    KEY_LIVE_CACHE_TTL_WEATHER_CURRENT = (
        "live_cache_ttl_weather_current"  # Default 3600 (1 hour)
    )
    KEY_LIVE_CACHE_TTL_WEATHER_FORECAST = (
        "live_cache_ttl_weather_forecast"  # Default 21600 (6 hours)
    )
    KEY_LIVE_CACHE_TTL_SCORE_LIVE = "live_cache_ttl_score_live"  # Default 120 (2 min)
    KEY_LIVE_CACHE_TTL_DEFAULT = "live_cache_ttl_default"  # Default 3600 (1 hour)
    KEY_LIVE_CACHE_ENTITY_TTL_DAYS = (
        "live_cache_entity_ttl_days"  # Entity cache, default 90 days
    )
    KEY_LIVE_CACHE_ENABLED = (
        "live_cache_enabled"  # Enable/disable caching, default true
    )
    KEY_LIVE_CACHE_MAX_SIZE_MB = (
        "live_cache_max_size_mb"  # Max cache size, default 100MB
    )


class ModelOverride(Base):
    """
    Override settings for LiteLLM-sourced models.

    Allows users to disable models or override specific properties
    like pricing. LiteLLM models auto-update with sync;
    overrides persist user preferences.
    """

    __tablename__ = "model_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Override fields (null means use system default)
    input_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    output_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Extended cost parameters (v2.2.3)
    cache_read_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cache_write_multiplier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    capabilities_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array
    context_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Provider quirks (API-specific behavior overrides)
    # These persist across LiteLLM syncs since LiteLLM doesn't track them
    use_max_completion_tokens: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )  # Use max_completion_tokens instead of max_tokens (OpenAI o1/o3/gpt-5)
    supports_system_prompt: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )  # Whether model supports system prompts
    unsupported_params_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array of unsupported parameter names (e.g., ["temperature", "top_p"])

    @property
    def capabilities(self) -> list | None:
        """Get capabilities as a list."""
        if self.capabilities_json is None:
            return None
        return json.loads(self.capabilities_json)

    @capabilities.setter
    def capabilities(self, value: list | None):
        """Set capabilities from a list."""
        self.capabilities_json = json.dumps(value) if value is not None else None

    @property
    def unsupported_params(self) -> list | None:
        """Get unsupported_params as a list."""
        if self.unsupported_params_json is None:
            return None
        return json.loads(self.unsupported_params_json)

    @unsupported_params.setter
    def unsupported_params(self, value: list | None):
        """Set unsupported_params from a list."""
        self.unsupported_params_json = json.dumps(value) if value is not None else None

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Unique constraint on provider + model
    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "disabled": self.disabled,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_read_multiplier": self.cache_read_multiplier,
            "cache_write_multiplier": self.cache_write_multiplier,
            "capabilities": self.capabilities,
            "context_length": self.context_length,
            "description": self.description,
            "use_max_completion_tokens": self.use_max_completion_tokens,
            "supports_system_prompt": self.supports_system_prompt,
            "unsupported_params": self.unsupported_params,
        }


class CustomModel(Base):
    """
    User-created custom models.

    Custom models are fully editable and persist across updates.
    """

    __tablename__ = "custom_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    family: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    context_length: Mapped[int] = mapped_column(Integer, default=128000)
    capabilities_json: Mapped[str] = mapped_column(Text, default="[]")
    unsupported_params_json: Mapped[Optional[str]] = mapped_column(Text)
    supports_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True)
    use_max_completion_tokens: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Cost per million tokens (USD)
    input_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    output_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Extended cost parameters (v2.2.3)
    cache_read_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cache_write_multiplier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    @property
    def capabilities(self) -> list[str]:
        """Get capabilities as a list."""
        return json.loads(self.capabilities_json) if self.capabilities_json else []

    @capabilities.setter
    def capabilities(self, value: list[str]):
        """Set capabilities from a list."""
        self.capabilities_json = json.dumps(value)

    @property
    def unsupported_params(self) -> list[str]:
        """Get unsupported params as a list."""
        return (
            json.loads(self.unsupported_params_json)
            if self.unsupported_params_json
            else []
        )

    @unsupported_params.setter
    def unsupported_params(self, value: list[str]):
        """Set unsupported params from a list."""
        self.unsupported_params_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "family": self.family,
            "description": self.description,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "unsupported_params": self.unsupported_params,
            "supports_system_prompt": self.supports_system_prompt,
            "use_max_completion_tokens": self.use_max_completion_tokens,
            "enabled": self.enabled,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_read_multiplier": self.cache_read_multiplier,
            "cache_write_multiplier": self.cache_write_multiplier,
            "is_system": False,  # Always false for custom models
        }


class OllamaInstance(Base):
    """
    User-configured Ollama instances.

    Allows users to add Ollama instances via the UI.
    """

    __tablename__ = "ollama_instances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "is_system": False,  # DB instances are user-created
        }


class CustomProvider(Base):
    """
    User-configured custom providers (Anthropic or OpenAI-compatible).

    Allows users to add providers via the UI.
    """

    __tablename__ = "custom_providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'anthropic' or 'openai-compatible'
    base_url: Mapped[Optional[str]] = mapped_column(String(500))
    api_key_env: Mapped[Optional[str]] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "enabled": self.enabled,
            "is_system": False,
        }


# ============================================================================
# Usage Tracking Models (v2.1)
# ============================================================================


class RequestLog(Base):
    """
    Log of individual proxy requests for usage tracking.

    Stores detailed information about each request including client IP,
    tag attribution, token counts, and response time.
    """

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )

    # Client identification
    client_ip: Mapped[str] = mapped_column(
        String(45), nullable=False
    )  # IPv6 max length
    hostname: Mapped[Optional[str]] = mapped_column(String(255))  # Reverse DNS
    tag: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Alias tracking (v3.1)
    alias: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Smart router tracking (v3.2)
    is_designator: Mapped[bool] = mapped_column(Boolean, default=False)
    router_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Request type tracking (v3.11)
    # "inbound" = client request received, "main" = primary LLM call,
    # "designator" = routing/source selection call, "embedding" = embedding call
    request_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, index=True, default="main"
    )

    # Smart augmentor tracking (v3.5)
    augmentor_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Augmentation details (v3.5.1) - what augmentation was applied and how
    augmentation_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # "direct"|"search"|"scrape"|"search+scrape"
    augmentation_query: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Search query used (if any)
    augmentation_urls: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array of scraped URLs (if any)

    # Smart RAG tracking (v3.8)
    rag_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Cache tracking (v1.6)
    is_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Name of entity that provided cache (alias/router/enricher/redirect)
    cache_tokens_saved: Mapped[int] = mapped_column(
        Integer, default=0
    )  # Output tokens that would have been generated
    cache_cost_saved: Mapped[float] = mapped_column(
        Float, default=0.0
    )  # Estimated cost saved

    # Request details
    provider_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # /api/chat, /v1/chat/completions, etc.

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Extended token tracking (v2.2.3)
    # OpenAI reasoning tokens (o1, o3 models) - billed at output rate
    reasoning_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # OpenAI cached input tokens (discounted rate)
    cached_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Anthropic cache tokens
    cache_creation_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # 1.25x input rate
    cache_read_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # 0.1x input rate

    # Cost tracking (provider-reported cost, e.g., from OpenRouter)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Performance & status
    response_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "client_ip": self.client_ip,
            "hostname": self.hostname,
            "tag": self.tag,
            "alias": self.alias,
            "is_designator": self.is_designator,
            "request_type": self.request_type,
            "router_name": self.router_name,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost": self.cost,
            "response_time_ms": self.response_time_ms,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "is_streaming": self.is_streaming,
            # Cache tracking
            "is_cache_hit": self.is_cache_hit,
            "cache_name": self.cache_name,
            "cache_tokens_saved": self.cache_tokens_saved,
            "cache_cost_saved": self.cache_cost_saved,
        }


class ModelCost(Base):
    """
    Per-model pricing configuration for cost estimation.

    Stores input/output token costs per million tokens.
    """

    __tablename__ = "model_costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    input_cost_per_million: Mapped[float] = mapped_column(Float, default=0.0)
    output_cost_per_million: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "currency": self.currency,
        }


class DailyStats(Base):
    """
    Pre-aggregated daily statistics for fast dashboard queries.

    Stores aggregated metrics by date and optional dimensions (tag, provider, model).
    Null dimensions represent totals at that aggregation level.
    """

    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    # Dimensions (nullable for different aggregation levels)
    tag: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    provider_id: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    alias: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # v3.1
    router_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # v3.2

    # Aggregated metrics
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_response_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = ({"sqlite_autoincrement": True},)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "date": self.date.strftime("%Y-%m-%d") if self.date else None,
            "tag": self.tag,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "alias": self.alias,
            "router_name": self.router_name,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_response_time_ms": self.total_response_time_ms,
            "estimated_cost": self.estimated_cost,
            "avg_response_time_ms": (
                self.total_response_time_ms // self.request_count
                if self.request_count > 0
                else 0
            ),
        }


# ============================================================================
# Redirect Model (v3.7)
# ============================================================================


class Redirect(Base):
    """
    Model redirects that transparently map one model name to another.

    Redirects allow migrating from old model names to new ones, or pointing
    provider-prefixed names to other providers. Unlike aliases, redirects:
    - Are checked first in resolution order (before smart routers)
    - Support wildcard patterns (e.g., "openrouter/anthropic/*" -> "anthropic/*")
    - Can optionally track usage via tags (like aliases)

    Examples:
    - "gpt-4" -> "gpt-5" (model upgrade)
    - "openrouter/anthropic/claude-3-opus" -> "anthropic/claude-3-opus" (provider switch)
    """

    __tablename__ = "redirects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(
        String(150), unique=True, nullable=False, index=True
    )  # Model name to redirect from (can include wildcards)
    target: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # Model name to redirect to
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Tags for usage tracking (stored as JSON array string)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Statistics
    redirect_count: Mapped[int] = mapped_column(Integer, default=0)

    # Response caching (semantic cache using ChromaDB)
    use_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_similarity_threshold: Mapped[float] = mapped_column(Float, default=0.95)
    cache_match_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True)
    cache_match_last_message_only: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_ttl_hours: Mapped[int] = mapped_column(Integer, default=168)
    cache_min_tokens: Mapped[int] = mapped_column(Integer, default=50)
    cache_max_tokens: Mapped[int] = mapped_column(Integer, default=4000)
    cache_collection: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Cache statistics
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cache_cost_saved: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    @property
    def tags(self) -> list[str]:
        """Get tags as a list."""
        if not self.tags_json:
            return []
        try:
            return json.loads(self.tags_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @tags.setter
    def tags(self, value: list[str]) -> None:
        """Set tags from a list."""
        self.tags_json = json.dumps(value) if value else None

    @property
    def target_model(self) -> str:
        """Alias for target to satisfy CacheConfig protocol."""
        return self.target

    @property
    def name(self) -> str:
        """Alias for source to satisfy CacheConfig protocol."""
        return self.source

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "enabled": self.enabled,
            "tags": self.tags,
            "redirect_count": self.redirect_count,
            "use_cache": self.use_cache,
            "cache_similarity_threshold": self.cache_similarity_threshold,
            "cache_match_system_prompt": self.cache_match_system_prompt,
            "cache_match_last_message_only": self.cache_match_last_message_only,
            "cache_ttl_hours": self.cache_ttl_hours,
            "cache_min_tokens": self.cache_min_tokens,
            "cache_max_tokens": self.cache_max_tokens,
            "cache_collection": self.cache_collection,
            "cache_hits": self.cache_hits,
            "cache_tokens_saved": self.cache_tokens_saved,
            "cache_cost_saved": self.cache_cost_saved,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Plugin Configuration Model
# ============================================================================


class PluginConfig(Base):
    """
    Configuration for a plugin instance.

    This table stores configuration for plugins that have been set up by the user.
    Each row represents a configured instance of a plugin (e.g., a specific Todoist
    account, a particular Google Drive folder source, etc.).

    The config_json field stores plugin-specific configuration as JSON, matching
    the plugin's get_config_fields() definition. This allows plugins to define
    their own configuration without requiring schema changes.
    """

    __tablename__ = "plugin_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Plugin identification
    plugin_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # 'document_source', 'live_source', 'action'
    source_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # Plugin's source_type/action_type identifier

    # User-friendly name for this configuration
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Configuration data as JSON
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # State
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_plugin_configs_type", "plugin_type", "source_type"),
        {"sqlite_autoincrement": True},
    )

    @property
    def config(self) -> dict:
        """Parse and return the config JSON as a dict."""
        if self.config_json:
            try:
                return json.loads(self.config_json)
            except json.JSONDecodeError:
                return {}
        return {}

    @config.setter
    def config(self, value: dict):
        """Set the config from a dict."""
        self.config_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "plugin_type": self.plugin_type,
            "source_type": self.source_type,
            "name": self.name,
            "config": self.config,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Document Store Model (v3.9)
# ============================================================================


class DocumentStore(Base):
    """
    Independent document store configuration.

    A document store represents a single source of documents that can be
    indexed into ChromaDB and shared across multiple Smart RAGs.
    """

    __tablename__ = "document_stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Friendly display name for LLM to identify this as an account
    # e.g., "Work Email", "Personal Calendar", "Home Tasks"
    # If not set, falls back to `name`
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Source configuration
    source_type: Mapped[str] = mapped_column(
        String(20), default="local"
    )  # "local" | "mcp"

    # Link to PluginConfig for plugin-specific configuration (unified sources)
    # When set, plugin config is read from the linked PluginConfig instead of
    # the legacy hardcoded columns. This enables custom plugins without schema changes.
    plugin_config_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("plugin_configs.id", ondelete="SET NULL"), nullable=True
    )

    # Unified configuration JSON (v2.1 - Plugin Cleanup)
    # Stores all source-specific configuration as JSON, replacing 50+ legacy columns.
    # Legacy columns are kept for backwards compatibility during migration period.
    # When config_json is set, it takes precedence over legacy columns.
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Local source (Docker-mapped folder)
    source_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # e.g., "/data/documents"

    # MCP source configuration (JSON object with server config)
    mcp_server_config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Google OAuth account (for MCP:gdrive sources)
    google_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )

    # Google Drive folder filter (optional - if set, only index files from this folder)
    gdrive_folder_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    gdrive_folder_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Gmail label filter (optional - if set, only index emails with this label)
    gmail_label_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    gmail_label_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes (e.g., "INBOX", "SENT")

    # Google Calendar filter (optional - if set, only index events from this calendar)
    gcalendar_calendar_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    gcalendar_calendar_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Google Tasks filter (optional - if set, only index tasks from this task list)
    gtasks_tasklist_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    gtasks_tasklist_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Google Contacts filter (optional - if set, only index contacts from this group)
    gcontacts_group_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    gcontacts_group_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Microsoft OAuth account (for mcp:onedrive, mcp:outlook, mcp:onenote sources)
    microsoft_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )

    # OneDrive folder filter (optional - if set, only index files from this folder)
    # Microsoft Graph API uses very long IDs (can be 150+ chars)
    onedrive_folder_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    onedrive_folder_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Outlook folder filter (optional - e.g., "inbox", "sentitems")
    # Microsoft Graph API uses very long IDs (can be 150+ chars)
    outlook_folder_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    outlook_folder_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes
    outlook_days_back: Mapped[int] = mapped_column(
        Integer, default=90
    )  # Days of email history to index

    # OneNote notebook filter (optional - if set, only index pages from this notebook)
    # Microsoft Graph API uses very long IDs (can be 150+ chars)
    onenote_notebook_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    onenote_notebook_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes

    # Microsoft Teams configuration (for source_type="mcp:teams")
    teams_team_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Team ID to filter (None = all teams)
    teams_team_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes
    teams_channel_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Channel ID to filter (None = all channels in team)
    teams_channel_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display purposes
    teams_days_back: Mapped[int] = mapped_column(
        Integer, default=90
    )  # Days of message history to fetch

    # Paperless-ngx configuration (for source_type="paperless")
    paperless_url: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Base URL (e.g., "http://paperless:8000") - DEPRECATED, use env var
    paperless_token: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # API token - DEPRECATED, use env var
    paperless_tag_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Filter by tag ID (optional)
    paperless_tag_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Tag name for display

    # Todoist configuration (for source_type="todoist")
    todoist_project_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Filter by project ID (optional)
    todoist_project_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Project name for display
    todoist_filter: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Todoist filter expression (e.g., "today", "priority 1")
    todoist_include_completed: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Whether to include completed tasks

    # GitHub configuration (for source_type="mcp:github")
    github_repo: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Repository in "owner/repo" format
    github_branch: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Branch name (defaults to repo default)
    github_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Path filter pattern

    # Notion configuration (for source_type="notion")
    notion_database_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Database ID to query pages from
    notion_page_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Root page ID (indexes children)
    notion_is_task_database: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Treat as task database for Smart Actions

    # Nextcloud configuration (for source_type="nextcloud")
    nextcloud_folder: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Folder path to restrict indexing (e.g., "/Documents")

    # Website configuration (for source_type="website")
    website_url: Mapped[Optional[str]] = mapped_column(
        String(1000), nullable=True
    )  # Starting URL to crawl
    website_crawl_depth: Mapped[int] = mapped_column(
        Integer, default=1
    )  # How many links deep to follow (0 = single page only)
    website_max_pages: Mapped[int] = mapped_column(
        Integer, default=50
    )  # Maximum pages to crawl
    website_include_pattern: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Regex pattern - only crawl URLs matching this
    website_exclude_pattern: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Regex pattern - skip URLs matching this
    website_crawler_override: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # Override global crawler: None (use default), "builtin", or "jina"

    # Slack configuration (for source_type="slack")
    slack_channel_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Specific channel ID to index (None = all accessible channels)
    slack_channel_types: Mapped[str] = mapped_column(
        String(100), default="public_channel"
    )  # Comma-separated: public_channel, private_channel
    slack_days_back: Mapped[int] = mapped_column(
        Integer, default=90
    )  # Days of history to fetch (max 90 for free plan)

    # Web Search configuration (for source_type="websearch")
    websearch_query: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Search query string
    websearch_max_results: Mapped[int] = mapped_column(
        Integer, default=10
    )  # Max search results to fetch
    websearch_pages_to_scrape: Mapped[int] = mapped_column(
        Integer, default=5
    )  # How many results to actually scrape
    websearch_time_range: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # Time filter: day, week, month, year
    websearch_category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Search category: news, general, etc.

    # IMAP/SMTP configuration (for source_type="imap")
    imap_host: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # IMAP server hostname
    imap_port: Mapped[int] = mapped_column(
        Integer, default=993
    )  # IMAP port (993 for SSL, 143 for STARTTLS)
    imap_username: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Email/username for authentication
    imap_password: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Password (consider using env var instead)
    imap_use_ssl: Mapped[bool] = mapped_column(Boolean, default=True)  # Use SSL/TLS
    imap_allow_insecure: Mapped[bool] = mapped_column(Boolean, default=False)  # Skip cert verification
    imap_folders: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Comma-separated folders to index (e.g., "INBOX,Sent")
    imap_index_days: Mapped[int] = mapped_column(
        Integer, default=90
    )  # Days of email history to index
    smtp_host: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # SMTP server for sending
    smtp_port: Mapped[int] = mapped_column(
        Integer, default=587
    )  # SMTP port (587 for STARTTLS, 465 for SSL)

    # Embedding configuration
    embedding_provider: Mapped[str] = mapped_column(String(100), default="local")
    embedding_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    ollama_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Vision model configuration (for document parsing with Docling)
    vision_provider: Mapped[str] = mapped_column(String(100), default="local")
    vision_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    vision_ollama_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Chunking configuration
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=50)

    # Indexing configuration
    index_schedule: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_indexed: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    index_status: Mapped[str] = mapped_column(String(20), default="pending")
    index_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Document limit (None/0 = no limit)
    max_documents: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=None
    )

    # Statistics
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # ChromaDB collection name (auto-generated as "docstore_{id}")
    collection_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Intelligence - auto-generated analysis of indexed content
    # Helps routing decisions when multiple RAG sources are available
    themes_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array of topic themes
    best_for: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Types of questions this store answers
    content_summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Brief description of content
    intelligence_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Temporal filtering - when enabled, documents can be filtered by date
    # Uses document_date metadata (derived from modified_time) for date-based queries
    use_temporal_filtering: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    @property
    def mcp_server_config(self) -> dict | None:
        """Get MCP server config as a dict."""
        if not self.mcp_server_config_json:
            return None
        return json.loads(self.mcp_server_config_json)

    @mcp_server_config.setter
    def mcp_server_config(self, value: dict | None):
        """Set MCP server config from a dict."""
        self.mcp_server_config_json = json.dumps(value) if value else None

    @property
    def themes(self) -> list[str]:
        """Get themes as a list."""
        if not self.themes_json:
            return []
        return json.loads(self.themes_json)

    @themes.setter
    def themes(self, value: list[str] | None):
        """Set themes from a list."""
        self.themes_json = json.dumps(value) if value else None

    @property
    def config(self) -> dict:
        """
        Get unified configuration dict.

        If config_json is set, returns that. Otherwise, builds config from
        legacy columns for backwards compatibility.
        """
        if self.config_json:
            try:
                return json.loads(self.config_json)
            except json.JSONDecodeError:
                pass

        # Build from legacy columns
        return self._build_config_from_legacy_columns()

    @config.setter
    def config(self, value: dict | None):
        """Set unified configuration from a dict."""
        self.config_json = json.dumps(value) if value else None

    def _build_config_from_legacy_columns(self) -> dict:
        """Build config dict from legacy source-specific columns."""
        config = {}

        # Local source
        if self.source_path:
            config["source_path"] = self.source_path

        # MCP config
        if self.mcp_server_config_json:
            config["mcp_server_config"] = self.mcp_server_config

        # Google OAuth
        if self.google_account_id:
            config["google_account_id"] = self.google_account_id
        if self.gdrive_folder_id:
            config["gdrive_folder_id"] = self.gdrive_folder_id
            config["gdrive_folder_name"] = self.gdrive_folder_name
        if self.gmail_label_id:
            config["gmail_label_id"] = self.gmail_label_id
            config["gmail_label_name"] = self.gmail_label_name
        if self.gcalendar_calendar_id:
            config["gcalendar_calendar_id"] = self.gcalendar_calendar_id
            config["gcalendar_calendar_name"] = self.gcalendar_calendar_name
        if self.gtasks_tasklist_id:
            config["gtasks_tasklist_id"] = self.gtasks_tasklist_id
            config["gtasks_tasklist_name"] = self.gtasks_tasklist_name
        if self.gcontacts_group_id:
            config["gcontacts_group_id"] = self.gcontacts_group_id
            config["gcontacts_group_name"] = self.gcontacts_group_name

        # Microsoft OAuth
        if self.microsoft_account_id:
            config["microsoft_account_id"] = self.microsoft_account_id
        if self.onedrive_folder_id:
            config["onedrive_folder_id"] = self.onedrive_folder_id
            config["onedrive_folder_name"] = self.onedrive_folder_name
        if self.outlook_folder_id:
            config["outlook_folder_id"] = self.outlook_folder_id
            config["outlook_folder_name"] = self.outlook_folder_name
        if self.outlook_days_back and self.outlook_days_back != 90:
            config["outlook_days_back"] = self.outlook_days_back
        if self.onenote_notebook_id:
            config["onenote_notebook_id"] = self.onenote_notebook_id
            config["onenote_notebook_name"] = self.onenote_notebook_name
        if self.teams_team_id:
            config["teams_team_id"] = self.teams_team_id
            config["teams_team_name"] = self.teams_team_name
        if self.teams_channel_id:
            config["teams_channel_id"] = self.teams_channel_id
            config["teams_channel_name"] = self.teams_channel_name
        if self.teams_days_back and self.teams_days_back != 90:
            config["teams_days_back"] = self.teams_days_back

        # Paperless
        if self.paperless_url:
            config["paperless_url"] = self.paperless_url
        if self.paperless_token:
            config["paperless_token"] = self.paperless_token
        if self.paperless_tag_id:
            config["paperless_tag_id"] = self.paperless_tag_id
            config["paperless_tag_name"] = self.paperless_tag_name

        # GitHub
        if self.github_repo:
            config["github_repo"] = self.github_repo
        if self.github_branch:
            config["github_branch"] = self.github_branch
        if self.github_path:
            config["github_path"] = self.github_path

        # Notion
        if self.notion_database_id:
            config["notion_database_id"] = self.notion_database_id
        if self.notion_page_id:
            config["notion_page_id"] = self.notion_page_id
        if self.notion_is_task_database:
            config["notion_is_task_database"] = self.notion_is_task_database

        # Nextcloud
        if self.nextcloud_folder:
            config["nextcloud_folder"] = self.nextcloud_folder

        # Website
        if self.website_url:
            config["website_url"] = self.website_url
        if self.website_crawl_depth and self.website_crawl_depth != 1:
            config["website_crawl_depth"] = self.website_crawl_depth
        if self.website_max_pages and self.website_max_pages != 50:
            config["website_max_pages"] = self.website_max_pages
        if self.website_include_pattern:
            config["website_include_pattern"] = self.website_include_pattern
        if self.website_exclude_pattern:
            config["website_exclude_pattern"] = self.website_exclude_pattern
        if self.website_crawler_override:
            config["website_crawler_override"] = self.website_crawler_override

        # Slack
        if self.slack_channel_id:
            config["slack_channel_id"] = self.slack_channel_id
        if self.slack_channel_types and self.slack_channel_types != "public_channel":
            config["slack_channel_types"] = self.slack_channel_types
        if self.slack_days_back and self.slack_days_back != 90:
            config["slack_days_back"] = self.slack_days_back

        # Todoist
        if self.todoist_project_id:
            config["todoist_project_id"] = self.todoist_project_id
            config["todoist_project_name"] = self.todoist_project_name
        if self.todoist_filter:
            config["todoist_filter"] = self.todoist_filter
        if self.todoist_include_completed:
            config["todoist_include_completed"] = self.todoist_include_completed

        # WebSearch
        if self.websearch_query:
            config["websearch_query"] = self.websearch_query
        if self.websearch_max_results and self.websearch_max_results != 10:
            config["websearch_max_results"] = self.websearch_max_results
        if self.websearch_pages_to_scrape and self.websearch_pages_to_scrape != 5:
            config["websearch_pages_to_scrape"] = self.websearch_pages_to_scrape
        if self.websearch_time_range:
            config["websearch_time_range"] = self.websearch_time_range
        if self.websearch_category:
            config["websearch_category"] = self.websearch_category

        # IMAP/SMTP
        if self.imap_host:
            config["imap_host"] = self.imap_host
        if self.imap_port and self.imap_port != 993:
            config["imap_port"] = self.imap_port
        if self.imap_username:
            config["imap_username"] = self.imap_username
        if self.imap_password:
            config["imap_password"] = self.imap_password
        if not self.imap_use_ssl:
            config["imap_use_ssl"] = self.imap_use_ssl
        if self.imap_folders:
            config["imap_folders"] = self.imap_folders
        if self.imap_index_days and self.imap_index_days != 90:
            config["imap_index_days"] = self.imap_index_days
        if self.smtp_host:
            config["smtp_host"] = self.smtp_host
        if self.smtp_port and self.smtp_port != 587:
            config["smtp_port"] = self.smtp_port

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "plugin_config_id": self.plugin_config_id,
            "config": self.config,  # Unified config dict
            "source_path": self.source_path,
            "mcp_server_config": self.mcp_server_config,
            "google_account_id": self.google_account_id,
            "gdrive_folder_id": self.gdrive_folder_id,
            "gdrive_folder_name": self.gdrive_folder_name,
            "gmail_label_id": self.gmail_label_id,
            "gmail_label_name": self.gmail_label_name,
            "gcalendar_calendar_id": self.gcalendar_calendar_id,
            "gcalendar_calendar_name": self.gcalendar_calendar_name,
            "gtasks_tasklist_id": self.gtasks_tasklist_id,
            "gtasks_tasklist_name": self.gtasks_tasklist_name,
            "gcontacts_group_id": self.gcontacts_group_id,
            "gcontacts_group_name": self.gcontacts_group_name,
            "microsoft_account_id": self.microsoft_account_id,
            "onedrive_folder_id": self.onedrive_folder_id,
            "onedrive_folder_name": self.onedrive_folder_name,
            "outlook_folder_id": self.outlook_folder_id,
            "outlook_folder_name": self.outlook_folder_name,
            "outlook_days_back": self.outlook_days_back,
            "onenote_notebook_id": self.onenote_notebook_id,
            "onenote_notebook_name": self.onenote_notebook_name,
            "teams_team_id": self.teams_team_id,
            "teams_team_name": self.teams_team_name,
            "teams_channel_id": self.teams_channel_id,
            "teams_channel_name": self.teams_channel_name,
            "teams_days_back": self.teams_days_back,
            "paperless_url": self.paperless_url,
            "paperless_token": self.paperless_token,
            "paperless_tag_id": self.paperless_tag_id,
            "paperless_tag_name": self.paperless_tag_name,
            "todoist_project_id": self.todoist_project_id,
            "todoist_project_name": self.todoist_project_name,
            "todoist_filter": self.todoist_filter,
            "todoist_include_completed": self.todoist_include_completed,
            "github_repo": self.github_repo,
            "github_branch": self.github_branch,
            "github_path": self.github_path,
            "notion_database_id": self.notion_database_id,
            "notion_page_id": self.notion_page_id,
            "notion_is_task_database": self.notion_is_task_database,
            "nextcloud_folder": self.nextcloud_folder,
            "website_url": self.website_url,
            "website_crawl_depth": self.website_crawl_depth,
            "website_max_pages": self.website_max_pages,
            "website_include_pattern": self.website_include_pattern,
            "website_exclude_pattern": self.website_exclude_pattern,
            "website_crawler_override": self.website_crawler_override,
            "slack_channel_id": self.slack_channel_id,
            "slack_channel_types": self.slack_channel_types,
            "slack_days_back": self.slack_days_back,
            # IMAP fields
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
            "imap_username": self.imap_username,
            "imap_password": self.imap_password,
            "imap_use_ssl": self.imap_use_ssl,
            "imap_allow_insecure": self.imap_allow_insecure,
            "imap_folders": self.imap_folders,
            "imap_index_days": self.imap_index_days,
            "websearch_query": self.websearch_query,
            "websearch_max_results": self.websearch_max_results,
            "websearch_pages_to_scrape": self.websearch_pages_to_scrape,
            "websearch_time_range": self.websearch_time_range,
            "websearch_category": self.websearch_category,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "ollama_url": self.ollama_url,
            "vision_provider": self.vision_provider,
            "vision_model": self.vision_model,
            "vision_ollama_url": self.vision_ollama_url,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_schedule": self.index_schedule,
            "last_indexed": self.last_indexed.isoformat()
            if self.last_indexed
            else None,
            "index_status": self.index_status,
            "index_error": self.index_error,
            "max_documents": self.max_documents,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "collection_name": self.collection_name,
            # Intelligence
            "themes": self.themes,
            "best_for": self.best_for,
            "content_summary": self.content_summary,
            "intelligence_updated_at": self.intelligence_updated_at.isoformat()
            if self.intelligence_updated_at
            else None,
            # Metadata
            "description": self.description,
            "enabled": self.enabled,
            "use_temporal_filtering": self.use_temporal_filtering,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Smart Alias Model (unified routing + enrichment + caching)
# ============================================================================

# Association table for SmartAlias <-> DocumentStore many-to-many relationship
smart_alias_stores = Table(
    "smart_alias_stores",
    Base.metadata,
    Column(
        "smart_alias_id",
        Integer,
        ForeignKey("smart_aliases.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "document_store_id",
        Integer,
        ForeignKey("document_stores.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

# Association table for SmartAlias <-> LiveDataSource many-to-many relationship
smart_alias_live_sources = Table(
    "smart_alias_live_sources",
    Base.metadata,
    Column(
        "smart_alias_id",
        Integer,
        ForeignKey("smart_aliases.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "live_data_source_id",
        Integer,
        ForeignKey("live_data_sources.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class SmartAlias(Base):
    """
    Unified Smart Alias - combines routing, enrichment, and caching.

    A SmartAlias is the single entry point for all smart features:
    - Simple alias (no features enabled) - just forwards to target_model
    - Smart routing (use_routing=True) - designator picks from candidates
    - RAG enrichment (use_rag=True) - injects document context
    - Web enrichment (use_web=True) - injects realtime web context
    - Response caching (use_cache=True) - semantic response caching

    Note: use_cache is only permitted when use_web=False (realtime data shouldn't be cached)
    """

    __tablename__ = "smart_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # ===== FEATURE TOGGLES =====
    use_routing: Mapped[bool] = mapped_column(Boolean, default=False)
    use_rag: Mapped[bool] = mapped_column(Boolean, default=False)
    use_web: Mapped[bool] = mapped_column(Boolean, default=False)
    use_live_data: Mapped[bool] = mapped_column(Boolean, default=False)
    use_cache: Mapped[bool] = mapped_column(Boolean, default=False)

    # ===== SMART TAG SETTINGS =====
    # When enabled, requests tagged with this alias name trigger the alias
    is_smart_tag: Mapped[bool] = mapped_column(Boolean, default=False)
    # When enabled with smart tag, use request's original model instead of target_model
    passthrough_model: Mapped[bool] = mapped_column(Boolean, default=False)

    # ===== TARGET CONFIGURATION =====
    # Always required - the default/fallback target model
    target_model: Mapped[str] = mapped_column(String(150), nullable=False)

    # ===== ROUTING SETTINGS (when use_routing=True) =====
    designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    # Parallel designators (optional - if not set, falls back to unified designator_model)
    router_designator_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )
    rag_designator_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )
    web_designator_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )
    live_designator_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    candidates_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fallback_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    routing_strategy: Mapped[str] = mapped_column(String(20), default="per_request")
    session_ttl: Mapped[int] = mapped_column(Integer, default=3600)

    # Model Intelligence (for routing)
    use_model_intelligence: Mapped[bool] = mapped_column(Boolean, default=False)
    search_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    intelligence_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )

    # ===== RAG SETTINGS (when use_rag=True) =====
    # document_stores via junction table
    max_results: Mapped[int] = mapped_column(Integer, default=5)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    # ===== SMART SOURCE SELECTION =====
    # When enabled, uses designator_model to decide which document stores
    # and whether web search should be used for each query
    use_smart_source_selection: Mapped[bool] = mapped_column(Boolean, default=False)
    # Two-pass retrieval: first retrieve sample chunks from each store,
    # then let designator see actual content before allocating tokens
    use_two_pass_retrieval: Mapped[bool] = mapped_column(Boolean, default=False)

    # ===== WEB SETTINGS (when use_web=True) =====
    max_search_results: Mapped[int] = mapped_column(Integer, default=5)
    max_scrape_urls: Mapped[int] = mapped_column(Integer, default=3)

    # ===== COMMON ENRICHMENT SETTINGS =====
    max_context_tokens: Mapped[int] = mapped_column(Integer, default=4000)
    rerank_provider: Mapped[str] = mapped_column(String(50), default="local")
    rerank_model: Mapped[str] = mapped_column(
        String(150), default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_n: Mapped[int] = mapped_column(Integer, default=20)
    # Context priority when both RAG and Web are enabled
    # "balanced" = 50/50, "prefer_rag" = 70/30, "prefer_web" = 30/70
    # Note: This is ignored when use_smart_source_selection=True (dynamic allocation)
    context_priority: Mapped[str] = mapped_column(String(20), default="balanced")

    # Show sources attribution at end of response (for debugging/transparency)
    show_sources: Mapped[bool] = mapped_column(Boolean, default=False)

    # ===== CACHE SETTINGS (when use_cache=True) =====
    cache_similarity_threshold: Mapped[float] = mapped_column(Float, default=0.95)
    cache_match_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True)
    cache_match_last_message_only: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_ttl_hours: Mapped[int] = mapped_column(Integer, default=168)
    cache_min_tokens: Mapped[int] = mapped_column(Integer, default=50)
    cache_max_tokens: Mapped[int] = mapped_column(Integer, default=4000)
    cache_collection: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # ===== STATISTICS =====
    total_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Routing stats
    routing_decisions: Mapped[int] = mapped_column(Integer, default=0)

    # Enrichment stats
    context_injections: Mapped[int] = mapped_column(Integer, default=0)
    search_requests: Mapped[int] = mapped_column(Integer, default=0)
    scrape_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Cache stats
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cache_cost_saved: Mapped[float] = mapped_column(Float, default=0.0)

    # ===== METADATA =====
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # ===== MEMORY =====
    # Persistent memory maintained by the designator model
    # Contains learned information about user preferences, context, etc.
    use_memory: Mapped[bool] = mapped_column(Boolean, default=False)
    memory: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    memory_max_tokens: Mapped[int] = mapped_column(Integer, default=500)
    memory_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # ===== ACTIONS SETTINGS =====
    # Enable LLM to perform actions (create drafts, etc.)
    use_actions: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON array of allowed action patterns, e.g., ["email:draft_*", "calendar:*"]
    # Empty array means no actions allowed even if use_actions=True
    allowed_actions_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Default accounts/resources for each action category
    # When set, LLM doesn't need to specify account/resource in action blocks

    # Email: default account for email actions (draft_new, draft_reply, draft_forward)
    action_email_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )

    # Calendar: default account and calendar for calendar actions
    action_calendar_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )
    action_calendar_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # e.g., "primary" or specific calendar ID

    # Tasks: default account and task list for task actions
    # For OAuth providers (google, microsoft): use action_tasks_account_id
    # For API key providers (todoist): set action_tasks_provider="todoist", account_id=NULL
    action_tasks_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )
    action_tasks_provider: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # "todoist" for API key providers, NULL for OAuth
    action_tasks_list_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Task list ID or Todoist project ID

    # Notification: Apprise URL(s) for notification actions
    # JSON array of Apprise URLs, e.g., ["gotify://host/token", "slack://token"]
    action_notification_urls_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Notes: default document store for note creation
    # References a document store (Notion database, OneNote notebook, etc.)
    action_notes_store_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("document_stores.id", ondelete="SET NULL"), nullable=True
    )

    # ===== SCHEDULED PROMPTS SETTINGS =====
    # Calendar-based scheduled prompts: events in this calendar are executed as LLM prompts
    # The event title becomes the prompt, description provides additional context
    scheduled_prompts_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # OAuth account for accessing the prompts calendar
    scheduled_prompts_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"), nullable=True
    )
    # Calendar ID containing prompt events (user creates dedicated calendar manually)
    scheduled_prompts_calendar_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    scheduled_prompts_calendar_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For display
    # How far ahead to look for upcoming prompts (minutes)
    scheduled_prompts_lookahead: Mapped[int] = mapped_column(Integer, default=15)
    # Whether to store full response in execution log
    scheduled_prompts_store_response: Mapped[bool] = mapped_column(
        Boolean, default=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document_stores: Mapped[list["DocumentStore"]] = relationship(
        "DocumentStore",
        secondary=smart_alias_stores,
        backref="smart_aliases",
    )
    live_data_sources: Mapped[list["LiveDataSource"]] = relationship(
        "LiveDataSource",
        secondary=smart_alias_live_sources,
        backref="smart_aliases",
    )

    __table_args__ = ({"sqlite_autoincrement": True},)

    @property
    def tags(self) -> list[str]:
        """Get tags as a list."""
        if not self.tags_json:
            return []
        try:
            return json.loads(self.tags_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @tags.setter
    def tags(self, value: list[str]) -> None:
        """Set tags from a list."""
        self.tags_json = json.dumps(value) if value else None

    @property
    def candidates(self) -> list[dict]:
        """Get candidates as a list of dicts."""
        if not self.candidates_json:
            return []
        try:
            return json.loads(self.candidates_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @candidates.setter
    def candidates(self, value: list[dict]) -> None:
        """Set candidates from a list of dicts."""
        self.candidates_json = json.dumps(value) if value else None

    @property
    def allowed_actions(self) -> list[str]:
        """Get allowed actions as a list."""
        if not self.allowed_actions_json:
            return []
        try:
            return json.loads(self.allowed_actions_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @allowed_actions.setter
    def allowed_actions(self, value: list[str]) -> None:
        """Set allowed actions from a list."""
        self.allowed_actions_json = json.dumps(value) if value else None

    @property
    def action_notification_urls(self) -> list[str]:
        """Get notification URLs as a list."""
        if not self.action_notification_urls_json:
            return []
        try:
            return json.loads(self.action_notification_urls_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @action_notification_urls.setter
    def action_notification_urls(self, value: list[str]) -> None:
        """Set notification URLs from a list."""
        self.action_notification_urls_json = json.dumps(value) if value else None

    @property
    def cache_enabled(self) -> bool:
        """
        Check if caching is actually enabled.

        Caching is only permitted when use_web=False (realtime web data
        shouldn't be cached). Returns True only if use_cache=True AND
        caching is permitted.
        """
        return self.use_cache and not self.use_web

    @property
    def feature_type(self) -> str:
        """Get a description of enabled features."""
        features = []
        if self.use_routing:
            features.append("routing")
        if self.use_rag:
            features.append("rag")
        if self.use_web:
            features.append("web")
        if self.use_live_data:
            features.append("live")
        if self.cache_enabled:
            features.append("cache")
        return "+".join(features) if features else "simple"

    @property
    def injection_rate(self) -> float:
        """Calculate context injection rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.context_injections / self.total_requests) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        # Get stores from either relationship or detached stores
        stores = getattr(self, "_detached_stores", None) or self.document_stores or []
        live_sources = (
            getattr(self, "_detached_live_sources", None)
            or self.live_data_sources
            or []
        )

        return {
            "id": self.id,
            "name": self.name,
            # Feature toggles
            "use_routing": self.use_routing,
            "use_rag": self.use_rag,
            "use_web": self.use_web,
            "use_live_data": self.use_live_data,
            "use_cache": self.use_cache,
            "is_smart_tag": self.is_smart_tag,
            "passthrough_model": self.passthrough_model,
            "feature_type": self.feature_type,
            # Target
            "target_model": self.target_model,
            # Routing settings
            "designator_model": self.designator_model,
            # Parallel designators
            "router_designator_model": self.router_designator_model,
            "rag_designator_model": self.rag_designator_model,
            "web_designator_model": self.web_designator_model,
            "live_designator_model": self.live_designator_model,
            "purpose": self.purpose,
            "candidates": self.candidates,
            "fallback_model": self.fallback_model,
            "routing_strategy": self.routing_strategy,
            "session_ttl": self.session_ttl,
            "use_model_intelligence": self.use_model_intelligence,
            "search_provider": self.search_provider,
            "intelligence_model": self.intelligence_model,
            # RAG settings
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            # Smart source selection
            "use_smart_source_selection": self.use_smart_source_selection,
            "use_two_pass_retrieval": self.use_two_pass_retrieval,
            # Web settings
            "max_search_results": self.max_search_results,
            "max_scrape_urls": self.max_scrape_urls,
            # Common enrichment
            "max_context_tokens": self.max_context_tokens,
            "rerank_provider": self.rerank_provider,
            "rerank_model": self.rerank_model,
            "rerank_top_n": self.rerank_top_n,
            "context_priority": self.context_priority,
            "show_sources": self.show_sources,
            # Cache settings
            "cache_enabled": self.cache_enabled,
            "cache_allowed": not self.use_web,
            "cache_similarity_threshold": self.cache_similarity_threshold,
            "cache_match_system_prompt": self.cache_match_system_prompt,
            "cache_match_last_message_only": self.cache_match_last_message_only,
            "cache_ttl_hours": self.cache_ttl_hours,
            "cache_min_tokens": self.cache_min_tokens,
            "cache_max_tokens": self.cache_max_tokens,
            "cache_collection": self.cache_collection,
            # Statistics
            "total_requests": self.total_requests,
            "routing_decisions": self.routing_decisions,
            "context_injections": self.context_injections,
            "search_requests": self.search_requests,
            "scrape_requests": self.scrape_requests,
            "injection_rate": self.injection_rate,
            "cache_hits": self.cache_hits,
            "cache_tokens_saved": self.cache_tokens_saved,
            "cache_cost_saved": self.cache_cost_saved,
            # Metadata
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
            "system_prompt": self.system_prompt,
            # Memory
            "use_memory": self.use_memory,
            "memory": self.memory,
            "memory_max_tokens": self.memory_max_tokens,
            "memory_updated_at": self.memory_updated_at.isoformat()
            if self.memory_updated_at
            else None,
            # Actions
            "use_actions": self.use_actions,
            "allowed_actions": self.allowed_actions,
            "action_email_account_id": self.action_email_account_id,
            "action_calendar_account_id": self.action_calendar_account_id,
            "action_calendar_id": self.action_calendar_id,
            "action_tasks_account_id": self.action_tasks_account_id,
            "action_tasks_provider": self.action_tasks_provider,
            "action_tasks_list_id": self.action_tasks_list_id,
            "action_notification_urls": self.action_notification_urls,
            "action_notes_store_id": self.action_notes_store_id,
            # Scheduled prompts
            "scheduled_prompts_enabled": self.scheduled_prompts_enabled,
            "scheduled_prompts_account_id": self.scheduled_prompts_account_id,
            "scheduled_prompts_calendar_id": self.scheduled_prompts_calendar_id,
            "scheduled_prompts_calendar_name": self.scheduled_prompts_calendar_name,
            "scheduled_prompts_lookahead": self.scheduled_prompts_lookahead,
            "scheduled_prompts_store_response": self.scheduled_prompts_store_response,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            # Document stores
            "document_store_ids": [s.id for s in stores] if stores else [],
            "document_stores": [
                {
                    "id": s.id,
                    "name": s.name,
                    "source_type": s.source_type,
                    "index_status": s.index_status,
                    "document_count": s.document_count,
                    "chunk_count": s.chunk_count,
                }
                for s in stores
            ]
            if stores
            else [],
            # Live data sources
            "live_data_source_ids": [s.id for s in live_sources]
            if live_sources
            else [],
            "live_data_sources": [
                {
                    "id": s.id,
                    "name": s.name,
                    "source_type": s.source_type,
                    "enabled": s.enabled,
                    "data_type": s.data_type,
                }
                for s in live_sources
            ]
            if live_sources
            else [],
        }


class OAuthToken(Base):
    """
    OAuth tokens for external service integrations (Google, etc.).

    Stores refresh tokens and metadata for MCP server authentication.
    Multiple tokens can be stored per provider (e.g., multiple Google accounts).
    """

    __tablename__ = "oauth_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Provider identification
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "google", "microsoft", etc.
    account_email: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # User's email for this account
    account_name: Mapped[Optional[str]] = mapped_column(
        String(255)
    )  # Display name (optional)

    # Token data (encrypted JSON containing access_token, refresh_token, etc.)
    token_data_encrypted: Mapped[str] = mapped_column(Text, nullable=False)

    # Scopes granted
    scopes_json: Mapped[Optional[str]] = mapped_column(Text)

    # Status
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_refreshed: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def scopes(self) -> list[str]:
        """Get scopes as a list."""
        if not self.scopes_json:
            return []
        return json.loads(self.scopes_json)

    @scopes.setter
    def scopes(self, value: list[str]):
        """Set scopes from a list."""
        self.scopes_json = json.dumps(value) if value else "[]"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses (excludes sensitive token data)."""
        return {
            "id": self.id,
            "provider": self.provider,
            "account_email": self.account_email,
            "account_name": self.account_name,
            "scopes": self.scopes,
            "is_valid": self.is_valid,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "last_refreshed": self.last_refreshed.isoformat()
            if self.last_refreshed
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Live Data Source Model (real-time API data)
# ============================================================================


class LiveDataSource(Base):
    """
    Live Data Source - real-time API queries for current data.

    Unlike Document Stores (which index and embed documents), Live Data Sources
    query external APIs at request time to inject real-time data like weather,
    transport status, stock prices, etc.

    The designator model decides which live sources to query based on the
    user's question, similar to smart source selection for document stores.
    """

    __tablename__ = "live_data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Source type: "rest_api", "builtin_weather", "builtin_stocks", "builtin_transport"
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== CONNECTION SETTINGS (for rest_api type) =====
    endpoint_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    http_method: Mapped[str] = mapped_column(String(10), default="GET")
    headers_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Auth configuration
    auth_type: Mapped[str] = mapped_column(
        String(20), default="none"
    )  # none, api_key, bearer, basic
    auth_config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== REQUEST CONFIGURATION =====
    # Template for request body/params with {{query}} placeholder
    request_template_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    query_params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== RESPONSE PROCESSING =====
    # JSONPath to extract data from response
    response_path: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Template for formatting response for LLM context
    response_format_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== BEHAVIOR =====
    cache_ttl_seconds: Mapped[int] = mapped_column(Integer, default=60)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=10)
    retry_count: Mapped[int] = mapped_column(Integer, default=1)
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, default=60)

    # ===== INTELLIGENCE (for smart source selection) =====
    # Category for quick filtering: weather, transport, finance, sports, etc.
    data_type: Mapped[str] = mapped_column(String(50), default="general")
    # Example response for designator to understand the data format
    sample_response_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Description of what queries this source is good for
    best_for: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== PLUGIN CONFIGURATION =====
    # JSON object storing plugin-specific config (e.g., oura_account_id, withings_account_id)
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ===== STATUS =====
    last_success: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    tool_count: Mapped[int] = mapped_column(Integer, default=0)  # MCP tools discovered

    # ===== USAGE STATS =====
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    successful_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[int] = mapped_column(
        Integer, default=0
    )  # For avg calculation

    # ===== TIMESTAMPS =====
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def headers(self) -> dict:
        """Get headers as a dict."""
        if not self.headers_json:
            return {}
        try:
            return json.loads(self.headers_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @headers.setter
    def headers(self, value: dict) -> None:
        """Set headers from a dict."""
        self.headers_json = json.dumps(value) if value else None

    @property
    def auth_config(self) -> dict:
        """Get auth config as a dict."""
        if not self.auth_config_json:
            return {}
        try:
            return json.loads(self.auth_config_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @auth_config.setter
    def auth_config(self, value: dict) -> None:
        """Set auth config from a dict."""
        self.auth_config_json = json.dumps(value) if value else None

    @property
    def request_template(self) -> dict:
        """Get request template as a dict."""
        if not self.request_template_json:
            return {}
        try:
            return json.loads(self.request_template_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @request_template.setter
    def request_template(self, value: dict) -> None:
        """Set request template from a dict."""
        self.request_template_json = json.dumps(value) if value else None

    @property
    def query_params(self) -> dict:
        """Get query params as a dict."""
        if not self.query_params_json:
            return {}
        try:
            return json.loads(self.query_params_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @query_params.setter
    def query_params(self, value: dict) -> None:
        """Set query params from a dict."""
        self.query_params_json = json.dumps(value) if value else None

    @property
    def sample_response(self) -> dict:
        """Get sample response as a dict."""
        if not self.sample_response_json:
            return {}
        try:
            return json.loads(self.sample_response_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @sample_response.setter
    def sample_response(self, value: dict) -> None:
        """Set sample response from a dict."""
        self.sample_response_json = json.dumps(value) if value else None

    @property
    def config(self) -> dict:
        """Get plugin config as a dict."""
        if not self.config_json:
            return {}
        try:
            return json.loads(self.config_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @config.setter
    def config(self, value: dict) -> None:
        """Set plugin config from a dict."""
        self.config_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "enabled": self.enabled,
            "description": self.description,
            # Connection
            "endpoint_url": self.endpoint_url,
            "http_method": self.http_method,
            "headers": self.headers,
            "auth_type": self.auth_type,
            "auth_config": self.auth_config,
            # Request
            "request_template": self.request_template,
            "query_params": self.query_params,
            # Response
            "response_path": self.response_path,
            "response_format_template": self.response_format_template,
            # Behavior
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "rate_limit_rpm": self.rate_limit_rpm,
            # Intelligence
            "data_type": self.data_type,
            "sample_response": self.sample_response,
            "best_for": self.best_for,
            # Plugin config
            "config": self.config,
            # Status
            "last_success": self.last_success.isoformat()
            if self.last_success
            else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "tool_count": self.tool_count,
            # Usage stats
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": (
                self.total_latency_ms // self.total_calls if self.total_calls > 0 else 0
            ),
            # Timestamps
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LiveDataSourceEndpointStats(Base):
    """
    Per-endpoint/tool statistics for Live Data Sources.

    Tracks usage at the endpoint level for MCP tools or REST endpoints.
    """

    __tablename__ = "live_data_source_endpoint_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_data_sources.id", ondelete="CASCADE"), nullable=False
    )
    endpoint_name: Mapped[str] = mapped_column(
        String(200), nullable=False
    )  # Tool name or endpoint path

    # Stats
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    successful_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    last_called: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Broken tool tracking - tool is marked broken when it returns 404 or similar
    # "doesn't exist" errors. Broken tools are filtered from designator tool lists.
    is_broken: Mapped[bool] = mapped_column(Boolean, default=False)
    broken_reason: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True
    )  # e.g., "404: Endpoint does not exist"
    broken_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Composite unique constraint
    __table_args__ = (
        Index("ix_endpoint_stats_source_endpoint", "source_id", "endpoint_name"),
        {"sqlite_autoincrement": True},
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "endpoint_name": self.endpoint_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": (
                self.total_latency_ms // self.total_calls if self.total_calls > 0 else 0
            ),
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "last_error": self.last_error,
            "is_broken": self.is_broken,
            "broken_reason": self.broken_reason,
            "broken_at": self.broken_at.isoformat() if self.broken_at else None,
        }


class LiveDataCache(Base):
    """
    Cache for live data API responses.

    Stores API responses with TTL-based expiration. Different data types
    have different TTLs:
    - Current prices/quotes: 15 minutes
    - Historical data: Forever (immutable)
    - Weather current: 1 hour
    - Weather forecast: 6 hours
    - Default: 1 hour
    """

    __tablename__ = "live_data_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Cache key components
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "stocks", "weather"
    tool_name: Mapped[str] = mapped_column(
        String(200), nullable=False
    )  # API tool/endpoint name
    args_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # SHA256 of normalized args JSON

    # The cached data
    response_data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON response
    response_summary: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Human-readable summary

    # TTL and timestamps
    ttl_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # 0 = never expires
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )  # NULL = never expires

    # Metadata
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Indexes for efficient lookup
    __table_args__ = (
        Index(
            "ix_live_cache_lookup", "source_type", "tool_name", "args_hash", unique=True
        ),
        Index("ix_live_cache_expires", "expires_at"),
        {"sqlite_autoincrement": True},
    )

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False  # Never expires
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "tool_name": self.tool_name,
            "args_hash": self.args_hash,
            "response_summary": self.response_summary,
            "ttl_seconds": self.ttl_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
            "is_expired": self.is_expired(),
        }


class LiveEntityCache(Base):
    """
    Cache for entity resolution (names -> API identifiers).

    Stores mappings like:
    - "Apple" -> "AAPL" (stocks:symbol)
    - "London" -> "328328" (weather:location)
    - "Kings Cross" -> "KGX" (transport:station)

    Uses SQLite/PostgreSQL instead of ChromaDB for simpler exact matching.
    Long TTL since entities rarely change.
    """

    __tablename__ = "live_entity_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Entity identification
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "stocks", "weather"
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "symbol", "location"
    query_text: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # Original query (lowercase)
    resolved_value: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # Resolved API value

    # Additional context for disambiguation
    display_name: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True
    )  # e.g., "Apple Inc."
    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Additional metadata

    # TTL and timestamps
    ttl_days: Mapped[int] = mapped_column(
        Integer, default=90
    )  # Default 90 days, 0 = never expires
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Usage stats
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index(
            "ix_entity_cache_lookup",
            "source_type",
            "entity_type",
            "query_text",
            unique=True,
        ),
        Index("ix_entity_cache_expires", "expires_at"),
        {"sqlite_autoincrement": True},
    )

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "entity_type": self.entity_type,
            "query_text": self.query_text,
            "resolved_value": self.resolved_value,
            "display_name": self.display_name,
            "ttl_days": self.ttl_days,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
            "is_expired": self.is_expired(),
        }


# ============================================================================
# Scheduled Prompts Model (calendar-based LLM prompt execution)
# ============================================================================


class ScheduledPromptExecution(Base):
    """
    Tracks execution of calendar-based scheduled prompts.

    Each Smart Alias can be linked to a dedicated calendar containing
    prompt events. The scheduler polls these calendars and executes
    prompts at their scheduled times. For recurring events, this table
    tracks the last execution to avoid duplicate runs.

    The event_id is the Google Calendar event ID. For recurring events,
    the iCalUID is used to identify the master event, and we track
    which instances have been executed.
    """

    __tablename__ = "scheduled_prompt_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Link to Smart Alias that owns this calendar
    smart_alias_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("smart_aliases.id", ondelete="CASCADE"), nullable=False
    )

    # Calendar event identification
    # event_id: The Google Calendar event ID (unique per instance for recurring)
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # ical_uid: The iCalendar UID (same for all instances of a recurring event)
    ical_uid: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # For recurring events, this is the start time of the specific instance
    instance_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Event details (cached from calendar for display/debugging)
    event_title: Mapped[str] = mapped_column(String(500), nullable=False)
    event_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scheduled_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Execution tracking
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, running, completed, failed, skipped
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    execution_duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Response tracking
    response_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )  # Model that handled the request
    response_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_preview: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # First ~500 chars of response
    response_full: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Full response (optional)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Indexes for efficient queries
    __table_args__ = (
        # Fast lookup by alias + event for deduplication
        Index("ix_scheduled_exec_alias_event", "smart_alias_id", "event_id"),
        # Fast lookup for recurring event instances
        Index("ix_scheduled_exec_ical_uid", "ical_uid"),
        # Fast lookup for pending executions
        Index("ix_scheduled_exec_status", "status", "scheduled_time"),
        {"sqlite_autoincrement": True},
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "smart_alias_id": self.smart_alias_id,
            "event_id": self.event_id,
            "ical_uid": self.ical_uid,
            "instance_start": self.instance_start.isoformat()
            if self.instance_start
            else None,
            "event_title": self.event_title,
            "event_description": self.event_description,
            "scheduled_time": self.scheduled_time.isoformat()
            if self.scheduled_time
            else None,
            "status": self.status,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_duration_ms": self.execution_duration_ms,
            "response_model": self.response_model,
            "response_tokens": self.response_tokens,
            "response_preview": self.response_preview,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
