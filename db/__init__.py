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
from .document_stores import (
    create_document_store,
    delete_document_store,
    get_all_document_stores,
    get_document_store_by_id,
    get_document_store_by_name,
    get_enabled_document_stores,
    get_stores_with_schedule,
    store_name_available,
    update_document_store,
    update_document_store_index_status,
)
from .models import (
    Alias,
    Base,
    CustomModel,
    CustomProvider,
    DailyStats,
    DocumentStore,
    Model,
    ModelOverride,
    OllamaInstance,
    Provider,
    Redirect,
    RequestLog,
    Setting,
    SmartEnricher,
    SmartRouter,
    smart_enricher_stores,
)
from .redirects import (
    create_redirect,
    delete_redirect,
    find_matching_redirect,
    get_all_redirects,
    get_enabled_redirects,
    get_redirect_by_id,
    get_redirect_by_source,
    increment_redirect_count,
    update_redirect,
)
from .seed import (
    ensure_seeded,
    seed_models_from_litellm,
    seed_providers,
)
from .settings import (
    delete_setting,
    get_all_settings,
    get_setting,
    set_setting,
)
from .smart_enrichers import (
    create_smart_enricher,
    delete_smart_enricher,
    enricher_name_available,
    get_all_smart_enrichers,
    get_enabled_smart_enrichers,
    get_smart_enricher_by_id,
    get_smart_enricher_by_name,
    reset_smart_enricher_stats,
    update_smart_enricher,
    update_smart_enricher_stats,
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
from .sync_descriptions import (
    get_available_description_providers,
    get_descriptions_for_models,
    get_model_description,
    sync_model_descriptions,
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
    # Description sync
    "sync_model_descriptions",
    "get_model_description",
    "get_descriptions_for_models",
    "get_available_description_providers",
    # Redirects (v3.7)
    "Redirect",
    "get_all_redirects",
    "get_redirect_by_id",
    "get_redirect_by_source",
    "find_matching_redirect",
    "create_redirect",
    "update_redirect",
    "delete_redirect",
    "increment_redirect_count",
    "get_enabled_redirects",
    # Document Stores
    "DocumentStore",
    "get_all_document_stores",
    "get_document_store_by_name",
    "get_document_store_by_id",
    "create_document_store",
    "update_document_store",
    "update_document_store_index_status",
    "delete_document_store",
    "store_name_available",
    "get_enabled_document_stores",
    "get_stores_with_schedule",
    # Smart Enrichers (unified RAG + Web)
    "SmartEnricher",
    "smart_enricher_stores",
    "get_all_smart_enrichers",
    "get_smart_enricher_by_name",
    "get_smart_enricher_by_id",
    "create_smart_enricher",
    "update_smart_enricher",
    "update_smart_enricher_stats",
    "reset_smart_enricher_stats",
    "delete_smart_enricher",
    "enricher_name_available",
    "get_enabled_smart_enrichers",
    # Settings
    "get_setting",
    "set_setting",
    "delete_setting",
    "get_all_settings",
]
