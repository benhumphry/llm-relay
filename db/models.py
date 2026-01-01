"""
SQLAlchemy models for ollama-llm-proxy.

These models store provider configuration, model definitions, and settings
that were previously stored in YAML files.
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
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

    Replaces entries in config/providers.yml
    """

    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'anthropic' or 'openai-compatible'
    base_url: Mapped[Optional[str]] = mapped_column(String(500))
    api_key_env: Mapped[Optional[str]] = mapped_column(String(100))  # Env var name
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(
        Text
    )  # Encrypted key stored in DB
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
    aliases: Mapped[list["Alias"]] = relationship(
        "Alias", back_populates="provider", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "type": self.type,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "has_api_key": bool(self.api_key_encrypted),
            "enabled": self.enabled,
            "display_order": self.display_order,
            "model_count": len(self.models) if self.models else 0,
            "alias_count": len(self.aliases) if self.aliases else 0,
        }


class Model(Base):
    """
    Model definition for a provider.

    Replaces entries in config/models/*.yml
    """

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    provider_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("providers.id"), primary_key=True
    )
    family: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    context_length: Mapped[int] = mapped_column(Integer, default=128000)
    capabilities_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    unsupported_params_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    supports_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True)
    use_max_completion_tokens: Mapped[bool] = mapped_column(Boolean, default=False)
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
            "family": self.family,
            "description": self.description,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "unsupported_params": self.unsupported_params,
            "supports_system_prompt": self.supports_system_prompt,
            "use_max_completion_tokens": self.use_max_completion_tokens,
            "enabled": self.enabled,
        }


class Alias(Base):
    """
    Model alias mapping.

    Allows short names like 'claude' to map to 'claude-sonnet-4-5-20250929'
    """

    __tablename__ = "aliases"

    alias: Mapped[str] = mapped_column(String(100), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("providers.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    provider: Mapped["Provider"] = relationship("Provider", back_populates="aliases")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
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
    KEY_TRACKING_ENABLED = "tracking_enabled"
    KEY_DEFAULT_TAG = "default_tag"


# Future tables for v2.1+ (defined here for reference, not created yet)
#
# class Request(Base):
#     """Request log for usage tracking (v2.1)"""
#     __tablename__ = "requests"
#     ...
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
