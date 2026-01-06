"""
Database layer for LLM Relay.

Provides SQLAlchemy models and database connection management.
"""

from .aliases import (
    alias_name_available,
    create_alias,
    delete_alias,
    get_alias_by_id,
    get_alias_by_name,
    get_all_aliases,
    get_enabled_aliases,
    update_alias,
)
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
    SmartRouter,
)
from .seed import (
    ensure_seeded,
    seed_models_from_litellm,
    seed_providers,
)
from .smart_routers import (
    create_smart_router,
    delete_smart_router,
    get_all_smart_routers,
    get_enabled_smart_routers,
    get_smart_router_by_id,
    get_smart_router_by_name,
    router_name_available,
    update_smart_router,
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
    # Aliases (v3.1)
    "Alias",
    "get_all_aliases",
    "get_alias_by_name",
    "get_alias_by_id",
    "create_alias",
    "update_alias",
    "delete_alias",
    "alias_name_available",
    "get_enabled_aliases",
    # Smart Routers (v3.2)
    "SmartRouter",
    "get_all_smart_routers",
    "get_smart_router_by_name",
    "get_smart_router_by_id",
    "create_smart_router",
    "update_smart_router",
    "delete_smart_router",
    "router_name_available",
    "get_enabled_smart_routers",
    # Seeding
    "ensure_seeded",
    "seed_providers",
    "seed_models_from_litellm",
]
