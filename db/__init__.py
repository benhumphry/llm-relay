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
from .seed import (
    ensure_seeded,
    seed_aliases,
    seed_models_from_litellm,
    seed_providers,
)

__all__ = [
    # Connection
    "get_db",
    "get_db_context",
    "init_db",
    "run_migrations",
    "check_db_initialized",
    "get_engine",
    # Core models
    "Provider",
    "Model",
    "Alias",
    "Setting",
    "Base",
    # Legacy models (to be removed in future)
    "ModelOverride",
    "AliasOverride",
    "CustomModel",
    "CustomAlias",
    "OllamaInstance",
    "CustomProvider",
    # Usage tracking
    "RequestLog",
    "DailyStats",
    # Seeding
    "ensure_seeded",
    "seed_providers",
    "seed_aliases",
    "seed_models_from_litellm",
]
