"""
Database layer for ollama-llm-proxy.

Provides SQLAlchemy models and database connection management.
"""

from .connection import (
    check_db_initialized,
    get_db,
    get_db_context,
    get_engine,
    init_db,
    run_migrations,
)
from .models import (
    Alias,
    AliasOverride,
    Base,
    CustomAlias,
    CustomModel,
    CustomProvider,
    DailyStats,
    Model,
    ModelOverride,
    OllamaInstance,
    Provider,
    RequestLog,
    Setting,
)

__all__ = [
    "get_db",
    "get_db_context",
    "init_db",
    "run_migrations",
    "check_db_initialized",
    "get_engine",
    "Provider",
    "Model",
    "Alias",
    "Setting",
    "Base",
    "ModelOverride",
    "AliasOverride",
    "CustomModel",
    "CustomAlias",
    "OllamaInstance",
    "CustomProvider",
    # Usage tracking (v2.1)
    "RequestLog",
    "DailyStats",
]
