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
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Smart Cache Model (v3.3)
# ============================================================================


class SmartCache(Base):
    """
    Smart caches that return cached responses for semantically similar queries.

    A smart cache uses ChromaDB to find semantically similar past queries and
    returns cached responses when similarity exceeds the threshold, reducing
    token usage and costs.

    Requires ChromaDB to be configured (CHROMA_URL environment variable).
    """

    __tablename__ = "smart_caches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Target model to use on cache miss
    target_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # "provider_id/model_id"

    # Cache behavior
    similarity_threshold: Mapped[float] = mapped_column(
        Float, default=0.95
    )  # 0.0-1.0, higher = stricter matching
    match_system_prompt: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # Include system prompt in cache key
    match_last_message_only: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Only match last user message (ignores conversation history)
    cache_ttl_hours: Mapped[int] = mapped_column(Integer, default=168)  # 7 days default
    min_cached_tokens: Mapped[int] = mapped_column(
        Integer, default=50
    )  # Don't cache very short responses (filters out titles, etc.)
    max_cached_tokens: Mapped[int] = mapped_column(
        Integer, default=4000
    )  # Don't cache very long responses

    # ChromaDB collection name (auto-generated if null)
    chroma_collection: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Statistics (updated periodically)
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cost_saved: Mapped[float] = mapped_column(Float, default=0.0)

    # Optional tags for usage tracking
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

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
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "target_model": self.target_model,
            "similarity_threshold": self.similarity_threshold,
            "match_system_prompt": self.match_system_prompt,
            "match_last_message_only": self.match_last_message_only,
            "cache_ttl_hours": self.cache_ttl_hours,
            "min_cached_tokens": self.min_cached_tokens,
            "max_cached_tokens": self.max_cached_tokens,
            "chroma_collection": self.chroma_collection,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "tokens_saved": self.tokens_saved,
            "cost_saved": self.cost_saved,
            "hit_rate": self.hit_rate,
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Smart Augmentor Model (v3.4)
# ============================================================================


class SmartAugmentor(Base):
    """
    Smart augmentors that enhance LLM requests with web search and URL scraping.

    A smart augmentor uses a designator LLM to decide what augmentation to apply:
    - direct: pass through unchanged
    - search:query: search via configured provider, inject results
    - scrape:url1,url2: fetch specific URLs, inject content
    - search+scrape:query: search then scrape top results

    The augmented context is injected into the system prompt before forwarding
    to the target model.
    """

    __tablename__ = "smart_augmentors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Designator LLM config
    designator_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # "provider_id/model_id" for deciding augmentation
    purpose: Mapped[Optional[str]] = mapped_column(
        Text
    )  # Context for augmentation decisions (e.g., "research assistant for current events")

    # Target model to forward augmented requests to
    target_model: Mapped[str] = mapped_column(
        String(150), nullable=False
    )  # "provider_id/model_id"

    # Search provider config
    search_provider: Mapped[str] = mapped_column(
        String(50), default="searxng"
    )  # "searxng" | "perplexity" | future providers
    search_provider_url: Mapped[Optional[str]] = mapped_column(
        String(500)
    )  # Override default URL for self-hosted providers
    max_search_results: Mapped[int] = mapped_column(Integer, default=5)
    max_scrape_urls: Mapped[int] = mapped_column(Integer, default=3)

    # Context injection config
    max_context_tokens: Mapped[int] = mapped_column(
        Integer, default=4000
    )  # Max tokens for injected context

    # Statistics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    augmented_requests: Mapped[int] = mapped_column(Integer, default=0)  # Non-direct
    search_requests: Mapped[int] = mapped_column(Integer, default=0)
    scrape_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Optional tags for usage tracking
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

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
    def augmentation_rate(self) -> float:
        """Calculate augmentation rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.augmented_requests / self.total_requests) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "designator_model": self.designator_model,
            "purpose": self.purpose,
            "target_model": self.target_model,
            "search_provider": self.search_provider,
            "search_provider_url": self.search_provider_url,
            "max_search_results": self.max_search_results,
            "max_scrape_urls": self.max_scrape_urls,
            "max_context_tokens": self.max_context_tokens,
            "total_requests": self.total_requests,
            "augmented_requests": self.augmented_requests,
            "search_requests": self.search_requests,
            "scrape_requests": self.scrape_requests,
            "augmentation_rate": self.augmentation_rate,
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Future tables for v2.2+ (defined here for reference, not created yet)
#
# class ApiKey(Base):
#     """API keys for proxy access (v2.2)"""
#     __tablename__ = "api_keys"
#     ...
#
# class Quota(Base):
#     """Usage quotas per tag (v2.2)"""
#     __tablename__ = "quotas"
#     ...
