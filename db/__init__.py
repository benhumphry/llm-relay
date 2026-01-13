"""
Database layer for LLM Relay.

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
    SmartAlias,
    smart_alias_stores,
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
from .smart_aliases import (
    create_smart_alias,
    delete_smart_alias,
    get_all_smart_aliases,
    get_enabled_smart_aliases,
    get_smart_alias_by_id,
    get_smart_alias_by_name,
    get_smart_tag_by_name,
    reset_smart_alias_stats,
    smart_alias_name_available,
    update_smart_alias,
    update_smart_alias_memory,
    update_smart_alias_stats,
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
    # Seeding
    "ensure_seeded",
    "seed_providers",
    "seed_models_from_litellm",
    # Description sync
    "sync_model_descriptions",
    "get_model_description",
    "get_descriptions_for_models",
    "get_available_description_providers",
    # Redirects
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
    # Settings
    "get_setting",
    "set_setting",
    "delete_setting",
    "get_all_settings",
    # Smart Aliases (unified routing + enrichment + caching)
    "SmartAlias",
    "smart_alias_stores",
    "get_all_smart_aliases",
    "get_smart_alias_by_name",
    "get_smart_alias_by_id",
    "get_smart_tag_by_name",
    "create_smart_alias",
    "update_smart_alias",
    "update_smart_alias_memory",
    "update_smart_alias_stats",
    "reset_smart_alias_stats",
    "delete_smart_alias",
    "smart_alias_name_available",
    "get_enabled_smart_aliases",
]
