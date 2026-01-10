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
    KEY_WEB_RERANK_PROVIDER = "web_rerank_provider"  # "local" or "jina"
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
    # Usage tracking settings (v2.1)
    KEY_TRACKING_ENABLED = "tracking_enabled"
    KEY_DNS_RESOLUTION_ENABLED = "dns_resolution_enabled"
    KEY_RETENTION_DAYS = "retention_days"


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
# Alias Model (v3.1)
# ============================================================================


class Alias(Base):
    """
    Model aliases that map a user-defined name to an actual model.

    Aliases allow users to create shortcuts like "gpt4" that point to
    "openai/gpt-4-turbo", with optional tag assignment for usage tracking.
    """

    __tablename__ = "aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    target_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # "provider_id/model_id"
    tags_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of tags
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Response caching (semantic cache using ChromaDB)
    use_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_similarity_threshold: Mapped[float] = mapped_column(
        Float, default=0.95
    )  # 0.0-1.0, higher = stricter matching
    cache_match_system_prompt: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # Include system prompt in cache key
    cache_match_last_message_only: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Only match last user message
    cache_ttl_hours: Mapped[int] = mapped_column(Integer, default=168)  # 7 days default
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
        return json.loads(self.tags_json) if self.tags_json else []

    @tags.setter
    def tags(self, value: list[str]):
        """Set tags from a list."""
        self.tags_json = json.dumps(value) if value else "[]"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "target_model": self.target_model,
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
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
# Smart Router Model (v3.2)
# ============================================================================


class SmartRouter(Base):
    """
    Smart routers that intelligently select models based on query analysis.

    A smart router uses a designator LLM to analyze incoming requests and
    route them to the most appropriate model from a pool of candidates.
    """

    __tablename__ = "smart_routers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Designator configuration
    designator_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # "provider_id/model_id"
    purpose: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # User-provided context for routing decisions

    # Candidate models: JSON array of {"model": "provider/model", "notes": "optional"}
    candidates_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Routing behavior
    strategy: Mapped[str] = mapped_column(
        String(20), default="per_request"
    )  # "per_request" | "per_session"
    fallback_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # Used if designator fails
    session_ttl: Mapped[int] = mapped_column(
        Integer, default=3600
    )  # Session cache TTL in seconds

    # Optional tags for usage tracking
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Model Intelligence (v3.6) - enhance designator with web-gathered model assessments
    use_model_intelligence: Mapped[bool] = mapped_column(Boolean, default=False)
    # Search provider for model intelligence (e.g., "searxng", "perplexity")
    search_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Model to use for summarizing search results into intelligence
    intelligence_model: Mapped[Optional[str]] = mapped_column(
        String(150), nullable=True
    )

    # Response caching (semantic cache using ChromaDB)
    # Note: For routers, caching caches the final routed response
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
        return json.loads(self.tags_json) if self.tags_json else []

    @tags.setter
    def tags(self, value: list[str]):
        """Set tags from a list."""
        self.tags_json = json.dumps(value) if value else "[]"

    @property
    def candidates(self) -> list[dict]:
        """Get candidates as a list of dicts."""
        return json.loads(self.candidates_json) if self.candidates_json else []

    @candidates.setter
    def candidates(self, value: list[dict]):
        """Set candidates from a list of dicts."""
        self.candidates_json = json.dumps(value) if value else "[]"

    @property
    def target_model(self) -> str:
        """Alias for fallback_model to satisfy CacheConfig protocol."""
        return self.fallback_model

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "designator_model": self.designator_model,
            "purpose": self.purpose,
            "candidates": self.candidates,
            "strategy": self.strategy,
            "fallback_model": self.fallback_model,
            "session_ttl": self.session_ttl,
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
            "use_model_intelligence": self.use_model_intelligence,
            "search_provider": self.search_provider,
            "intelligence_model": self.intelligence_model,
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

    # Source configuration
    source_type: Mapped[str] = mapped_column(
        String(20), default="local"
    )  # "local" | "mcp"

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

    # Statistics
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # ChromaDB collection name (auto-generated as "docstore_{id}")
    collection_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

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

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "mcp_server_config": self.mcp_server_config,
            "google_account_id": self.google_account_id,
            "gdrive_folder_id": self.gdrive_folder_id,
            "gdrive_folder_name": self.gdrive_folder_name,
            "gmail_label_id": self.gmail_label_id,
            "gmail_label_name": self.gmail_label_name,
            "gcalendar_calendar_id": self.gcalendar_calendar_id,
            "gcalendar_calendar_name": self.gcalendar_calendar_name,
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
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "collection_name": self.collection_name,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Smart Enricher (unified RAG + Web enrichment)
# ============================================================================

# Association table for SmartEnricher <-> DocumentStore many-to-many relationship
smart_enricher_stores = Table(
    "smart_enricher_stores",
    Base.metadata,
    Column(
        "smart_enricher_id",
        Integer,
        ForeignKey("smart_enrichers.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "document_store_id",
        Integer,
        ForeignKey("document_stores.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class SmartEnricher(Base):
    """
    Unified smart enrichment configuration (replaces SmartRAG + SmartAugmentor).

    Smart Enrichers can use RAG (document retrieval), realtime web search,
    or both to augment LLM requests with relevant context.
    """

    __tablename__ = "smart_enrichers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Enrichment mode toggles
    use_rag: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Enable RAG retrieval
    use_web: Mapped[bool] = mapped_column(Boolean, default=False)  # Enable web search

    # Target model (receives augmented context)
    target_model: Mapped[str] = mapped_column(String(150), nullable=False)

    # Designator model for web search query optimization (optional)
    designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    # RAG settings (when use_rag=True)
    max_results: Mapped[int] = mapped_column(
        Integer, default=5
    )  # Max chunks to retrieve
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    # Web settings (when use_web=True)
    # Uses global search_provider and scraper from Settings
    max_search_results: Mapped[int] = mapped_column(Integer, default=5)
    max_scrape_urls: Mapped[int] = mapped_column(Integer, default=3)

    # Common settings
    max_context_tokens: Mapped[int] = mapped_column(Integer, default=4000)

    # Reranking configuration
    rerank_provider: Mapped[str] = mapped_column(String(50), default="local")
    rerank_model: Mapped[str] = mapped_column(
        String(150), default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_n: Mapped[int] = mapped_column(Integer, default=20)

    # Statistics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    context_injections: Mapped[int] = mapped_column(Integer, default=0)
    search_requests: Mapped[int] = mapped_column(Integer, default=0)
    scrape_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Response caching (semantic cache using ChromaDB)
    # Note: Caching is only effective when use_web=False (realtime data shouldn't be cached)
    use_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_similarity_threshold: Mapped[float] = mapped_column(
        Float, default=0.95
    )  # 0.0-1.0, higher = stricter matching
    cache_match_system_prompt: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # Include system prompt in cache key
    cache_match_last_message_only: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Only match last user message (ignores conversation history)
    cache_ttl_hours: Mapped[int] = mapped_column(Integer, default=168)  # 7 days default
    cache_min_tokens: Mapped[int] = mapped_column(
        Integer, default=50
    )  # Don't cache very short responses
    cache_max_tokens: Mapped[int] = mapped_column(
        Integer, default=4000
    )  # Don't cache very long responses
    cache_collection: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # ChromaDB collection name (auto-generated if null)

    # Cache statistics
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cache_cost_saved: Mapped[float] = mapped_column(Float, default=0.0)

    # Tags for usage tracking (stored as JSON array string)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document_stores: Mapped[list["DocumentStore"]] = relationship(
        "DocumentStore",
        secondary=smart_enricher_stores,
        backref="smart_enrichers",
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
    def injection_rate(self) -> float:
        """Calculate context injection rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.context_injections / self.total_requests) * 100

    @property
    def enrichment_type(self) -> str:
        """Get enrichment type description."""
        if self.use_rag and self.use_web:
            return "hybrid"
        elif self.use_rag:
            return "rag"
        elif self.use_web:
            return "web"
        return "none"

    @property
    def cache_enabled(self) -> bool:
        """
        Check if caching is actually enabled.

        Caching is only permitted when use_web=False (realtime web data
        shouldn't be cached). Returns True only if use_cache=True AND
        caching is permitted.
        """
        return self.use_cache and not self.use_web

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        # Get stores from either relationship or detached stores
        stores = getattr(self, "_detached_stores", None) or self.document_stores or []

        return {
            "id": self.id,
            "name": self.name,
            "use_rag": self.use_rag,
            "use_web": self.use_web,
            "enrichment_type": self.enrichment_type,
            "target_model": self.target_model,
            "designator_model": self.designator_model,
            # RAG settings
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            # Web settings
            "max_search_results": self.max_search_results,
            "max_scrape_urls": self.max_scrape_urls,
            # Common settings
            "max_context_tokens": self.max_context_tokens,
            # Reranking
            "rerank_provider": self.rerank_provider,
            "rerank_model": self.rerank_model,
            "rerank_top_n": self.rerank_top_n,
            # Stats
            "total_requests": self.total_requests,
            "context_injections": self.context_injections,
            "search_requests": self.search_requests,
            "scrape_requests": self.scrape_requests,
            "injection_rate": self.injection_rate,
            # Caching (disabled when use_web=True)
            "use_cache": self.use_cache,
            "cache_allowed": not self.use_web,  # Caching not permitted with realtime web
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
            # Metadata
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
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
