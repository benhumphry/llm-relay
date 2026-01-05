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
    Base,
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
    "Setting",
    "Base",
    # Override/custom models
    "ModelOverride",
    "CustomModel",
    "OllamaInstance",
    "CustomProvider",
    # Usage tracking
    "RequestLog",
    "DailyStats",
    # Seeding
    "ensure_seeded",
    "seed_providers",
    "seed_models_from_litellm",
]
