"""
Live Data Source CRUD operations for LLM Relay.

Handles creation, retrieval, updating, and deletion of live data sources.
Live data sources are linked to Smart Aliases via the smart_alias_live_sources junction table.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import (
    LiveDataSource,
    LiveDataSourceEndpointStats,
    smart_alias_live_sources,
)

logger = logging.getLogger(__name__)


def get_all_live_data_sources(db=None) -> list[LiveDataSource]:
    """Get all live data sources ordered by name."""

    def _query(session) -> list[LiveDataSource]:
        stmt = select(LiveDataSource).order_by(LiveDataSource.name)
        return list(session.execute(stmt).scalars().all())

    if db:
        return _query(db)
    with get_db_context() as session:
        sources = _query(session)
        return [_source_to_detached(s) for s in sources]


def get_live_data_source_by_name(name: str, db=None) -> Optional[LiveDataSource]:
    """Get a live data source by name."""

    def _query(session) -> Optional[LiveDataSource]:
        stmt = select(LiveDataSource).where(LiveDataSource.name == name.lower())
        return session.execute(stmt).scalar_one_or_none()

    if db:
        return _query(db)
    with get_db_context() as session:
        source = _query(session)
        return _source_to_detached(source) if source else None


def get_live_data_source_by_id(source_id: int, db=None) -> Optional[LiveDataSource]:
    """Get a live data source by ID."""

    def _query(session) -> Optional[LiveDataSource]:
        stmt = select(LiveDataSource).where(LiveDataSource.id == source_id)
        return session.execute(stmt).scalar_one_or_none()

    if db:
        return _query(db)
    with get_db_context() as session:
        source = _query(session)
        return _source_to_detached(source) if source else None


def get_live_data_source_by_type(source_type: str, db=None) -> Optional[LiveDataSource]:
    """Get the first live data source with a given source_type."""

    def _query(session) -> Optional[LiveDataSource]:
        stmt = select(LiveDataSource).where(LiveDataSource.source_type == source_type)
        return session.execute(stmt).scalar_one_or_none()

    if db:
        return _query(db)
    with get_db_context() as session:
        source = _query(session)
        return _source_to_detached(source) if source else None


def create_live_data_source(
    name: str,
    source_type: str,
    description: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    http_method: str = "GET",
    headers: Optional[dict] = None,
    auth_type: str = "none",
    auth_config: Optional[dict] = None,
    request_template: Optional[dict] = None,
    query_params: Optional[dict] = None,
    response_path: Optional[str] = None,
    response_format_template: Optional[str] = None,
    cache_ttl_seconds: int = 60,
    timeout_seconds: int = 10,
    retry_count: int = 1,
    rate_limit_rpm: int = 60,
    data_type: str = "general",
    sample_response: Optional[dict] = None,
    best_for: Optional[str] = None,
    enabled: bool = True,
    config: Optional[dict] = None,
    db: Optional[Session] = None,
) -> LiveDataSource:
    """Create a new live data source."""

    def _create(session: Session) -> LiveDataSource:
        source = LiveDataSource(
            name=name.lower().strip(),
            source_type=source_type,
            description=description,
            endpoint_url=endpoint_url,
            http_method=http_method,
            headers_json=json.dumps(headers) if headers else None,
            auth_type=auth_type,
            auth_config_json=json.dumps(auth_config) if auth_config else None,
            request_template_json=json.dumps(request_template)
            if request_template
            else None,
            query_params_json=json.dumps(query_params) if query_params else None,
            response_path=response_path,
            response_format_template=response_format_template,
            cache_ttl_seconds=cache_ttl_seconds,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            rate_limit_rpm=rate_limit_rpm,
            data_type=data_type,
            sample_response_json=json.dumps(sample_response)
            if sample_response
            else None,
            best_for=best_for,
            enabled=enabled,
            config_json=json.dumps(config) if config else None,
        )
        session.add(source)
        session.commit()

        logger.info(f"Created live data source: {source.name} (ID: {source.id})")
        return source

    if db:
        return _create(db)
    with get_db_context() as session:
        source = _create(session)
        return _source_to_detached(source)


def update_live_data_source(
    source_id: int,
    name: Optional[str] = None,
    source_type: Optional[str] = None,
    description: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    http_method: Optional[str] = None,
    headers: Optional[dict] = None,
    auth_type: Optional[str] = None,
    auth_config: Optional[dict] = None,
    request_template: Optional[dict] = None,
    query_params: Optional[dict] = None,
    response_path: Optional[str] = None,
    response_format_template: Optional[str] = None,
    cache_ttl_seconds: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    rate_limit_rpm: Optional[int] = None,
    data_type: Optional[str] = None,
    sample_response: Optional[dict] = None,
    best_for: Optional[str] = None,
    enabled: Optional[bool] = None,
    config: Optional[dict] = None,
    db: Optional[Session] = None,
) -> Optional[LiveDataSource]:
    """Update a live data source."""

    def _update(session: Session) -> Optional[LiveDataSource]:
        stmt = select(LiveDataSource).where(LiveDataSource.id == source_id)
        source = session.execute(stmt).scalar_one_or_none()
        if not source:
            return None

        if name is not None:
            source.name = name.lower().strip()
        if source_type is not None:
            source.source_type = source_type
        if description is not None:
            source.description = description if description else None
        if endpoint_url is not None:
            source.endpoint_url = endpoint_url if endpoint_url else None
        if http_method is not None:
            source.http_method = http_method
        if headers is not None:
            source.headers_json = json.dumps(headers) if headers else None
        if auth_type is not None:
            source.auth_type = auth_type
        if auth_config is not None:
            source.auth_config_json = json.dumps(auth_config) if auth_config else None
        if request_template is not None:
            source.request_template_json = (
                json.dumps(request_template) if request_template else None
            )
        if query_params is not None:
            source.query_params_json = (
                json.dumps(query_params) if query_params else None
            )
        if response_path is not None:
            source.response_path = response_path if response_path else None
        if response_format_template is not None:
            source.response_format_template = (
                response_format_template if response_format_template else None
            )
        if cache_ttl_seconds is not None:
            source.cache_ttl_seconds = cache_ttl_seconds
        if timeout_seconds is not None:
            source.timeout_seconds = timeout_seconds
        if retry_count is not None:
            source.retry_count = retry_count
        if rate_limit_rpm is not None:
            source.rate_limit_rpm = rate_limit_rpm
        if data_type is not None:
            source.data_type = data_type
        if sample_response is not None:
            source.sample_response_json = (
                json.dumps(sample_response) if sample_response else None
            )
        if best_for is not None:
            source.best_for = best_for if best_for else None
        if enabled is not None:
            source.enabled = enabled
        if config is not None:
            source.config_json = json.dumps(config) if config else None

        source.updated_at = datetime.utcnow()
        session.commit()

        logger.info(f"Updated live data source: {source.name} (ID: {source.id})")
        return source

    if db:
        return _update(db)
    with get_db_context() as session:
        source = _update(session)
        return _source_to_detached(source) if source else None


def update_live_data_source_status(
    source_id: int,
    success: bool,
    error: Optional[str] = None,
    tool_count: Optional[int] = None,
    db: Optional[Session] = None,
) -> bool:
    """Update source status after a query attempt."""

    def _update(session: Session) -> bool:
        stmt = select(LiveDataSource).where(LiveDataSource.id == source_id)
        source = session.execute(stmt).scalar_one_or_none()
        if not source:
            return False

        if success:
            source.last_success = datetime.utcnow()
            source.last_error = None
            source.error_count = 0
            if tool_count is not None:
                source.tool_count = tool_count
        else:
            source.last_error = error
            source.error_count = (source.error_count or 0) + 1

        source.updated_at = datetime.utcnow()
        session.commit()
        return True

    if db:
        return _update(db)
    with get_db_context() as session:
        return _update(session)


def delete_live_data_source(source_id: int, db=None) -> bool:
    """
    Delete a live data source.

    Fails if any Smart Aliases are currently linked to it.
    """

    def _delete(session) -> bool:
        stmt = select(LiveDataSource).where(LiveDataSource.id == source_id)
        source = session.execute(stmt).scalar_one_or_none()
        if not source:
            return False

        # Check if any Smart Aliases are using this source via junction table
        from sqlalchemy import func

        alias_count = session.execute(
            select(func.count())
            .select_from(smart_alias_live_sources)
            .where(smart_alias_live_sources.c.live_data_source_id == source_id)
        ).scalar()

        if alias_count > 0:
            logger.error(
                f"Cannot delete source {source.name}: used by {alias_count} Smart Alias(es)"
            )
            return False

        session.delete(source)
        session.commit()
        logger.info(f"Deleted live data source: {source.name} (ID: {source_id})")
        return True

    if db:
        return _delete(db)
    with get_db_context() as session:
        return _delete(session)


def source_name_available(
    name: str, exclude_id: Optional[int] = None, db: Optional[Session] = None
) -> bool:
    """Check if a source name is available."""

    def _check(session: Session) -> bool:
        stmt = select(LiveDataSource).where(LiveDataSource.name == name.lower().strip())
        if exclude_id:
            stmt = stmt.where(LiveDataSource.id != exclude_id)
        return session.execute(stmt).scalar_one_or_none() is None

    if db:
        return _check(db)
    with get_db_context() as session:
        return _check(session)


def get_enabled_live_data_sources(
    db: Optional[Session] = None,
) -> dict[str, LiveDataSource]:
    """Get all enabled live data sources as a dict keyed by name."""

    def _query(session: Session) -> dict[str, LiveDataSource]:
        stmt = select(LiveDataSource).where(LiveDataSource.enabled == True)
        sources = session.execute(stmt).scalars().all()
        return {s.name: s for s in sources}

    if db:
        return _query(db)
    with get_db_context() as session:
        sources_dict = _query(session)
        return {name: _source_to_detached(s) for name, s in sources_dict.items()}


def _source_to_detached(
    source: Optional[LiveDataSource],
) -> Optional[LiveDataSource]:
    """
    Create a detached copy of a LiveDataSource for use outside sessions.

    This copies all attributes to a new instance that isn't bound to any session.
    """
    if not source:
        return None

    detached = LiveDataSource(
        id=source.id,
        name=source.name,
        source_type=source.source_type,
        enabled=source.enabled,
        description=source.description,
        endpoint_url=source.endpoint_url,
        http_method=source.http_method,
        headers_json=source.headers_json,
        auth_type=source.auth_type,
        auth_config_json=source.auth_config_json,
        request_template_json=source.request_template_json,
        query_params_json=source.query_params_json,
        response_path=source.response_path,
        response_format_template=source.response_format_template,
        cache_ttl_seconds=source.cache_ttl_seconds,
        timeout_seconds=source.timeout_seconds,
        retry_count=source.retry_count,
        rate_limit_rpm=source.rate_limit_rpm,
        data_type=source.data_type,
        sample_response_json=source.sample_response_json,
        best_for=source.best_for,
        config_json=source.config_json,  # Plugin config
        last_success=source.last_success,
        last_error=source.last_error,
        error_count=source.error_count,
        tool_count=source.tool_count,
        total_calls=source.total_calls,
        successful_calls=source.successful_calls,
        failed_calls=source.failed_calls,
        total_latency_ms=source.total_latency_ms,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )

    return detached


def seed_builtin_sources() -> list[str]:
    """
    Auto-create built-in live data sources if API keys are configured.

    Returns list of created source names.

    NOTE: All live data sources are now handled by the plugin system via
    seed_plugin_sources(). This function is deprecated and returns empty list.

    The plugin system (builtin_plugins/live_sources/) automatically creates
    sources for registered plugins when API keys are available.
    """
    # All live data sources now come from plugins - see seed_plugin_sources()
    return []


def seed_plugin_sources() -> list[str]:
    """
    Auto-create live data sources from registered live source plugins.

    Iterates through all registered live source plugins and creates
    database entries for those that don't already exist. This allows
    plugin-based sources (like smart_weather, smart_sports) to appear
    in the admin UI automatically.

    Returns list of created source names.
    """
    import os

    created = []

    try:
        from plugin_base.loader import live_source_registry
    except ImportError:
        logger.warning("Plugin loader not available, skipping plugin source seeding")
        return created

    plugins = live_source_registry.get_all()

    for source_type, plugin_class in plugins.items():
        # Generate a default name from the source_type (e.g., smart_weather -> smart-weather)
        default_name = source_type.replace("_", "-")

        # Check if a source with this source_type already exists (by name or type)
        existing = get_live_data_source_by_name(default_name)
        if existing:
            continue

        # Also check if any source with this source_type exists (user may have renamed it)
        existing_by_type = get_live_data_source_by_type(source_type)
        if existing_by_type:
            continue

        # Get plugin metadata
        display_name = getattr(plugin_class, "display_name", source_type)
        description = getattr(plugin_class, "description", "")
        data_type = getattr(plugin_class, "data_type", "general")
        best_for = getattr(plugin_class, "best_for", "")
        default_cache_ttl = getattr(plugin_class, "default_cache_ttl", 300)

        # Check if plugin requires specific configuration (API keys, OAuth)
        # that we can't auto-provide. For these, we'll create a disabled entry.
        # Exception: if RAPIDAPI_KEY is set, plugins with api_key fields can use it.
        requires_config = False
        config_fields = []
        try:
            config_fields = plugin_class.get_config_fields()
        except Exception:
            pass

        has_rapidapi_key = bool(os.environ.get("RAPIDAPI_KEY"))

        for field in config_fields:
            # Check for required fields that need user input
            if field.required:
                field_type = (
                    field.field_type.value
                    if hasattr(field.field_type, "value")
                    else str(field.field_type)
                )
                # OAuth accounts always require manual configuration
                if field_type == "oauth_account":
                    requires_config = True
                    break
                # API keys can use RAPIDAPI_KEY as fallback if available
                if field_type == "password":
                    field_name = getattr(field, "name", "").lower()
                    if "api_key" in field_name and has_rapidapi_key:
                        # RAPIDAPI_KEY available, plugin can use it as fallback
                        continue
                    requires_config = True
                    break

        # Create the source (disabled if it requires manual config)
        try:
            create_live_data_source(
                name=default_name,
                source_type=source_type,
                description=description,
                data_type=data_type,
                best_for=best_for,
                cache_ttl_seconds=default_cache_ttl,
                enabled=not requires_config,  # Enabled only if no manual config needed
            )
            created.append(default_name)
            status = "disabled (needs config)" if requires_config else "enabled"
            logger.info(f"Created plugin live data source: {default_name} ({status})")
        except Exception as e:
            logger.error(f"Failed to create plugin source {default_name}: {e}")

    return created


def _is_broken_tool_error(error: Optional[str]) -> tuple[bool, str]:
    """
    Check if an error indicates the tool/endpoint doesn't exist.

    Returns (is_broken, reason) tuple.
    """
    if not error:
        return False, ""

    error_lower = error.lower()

    # Patterns that indicate the endpoint doesn't exist
    broken_patterns = [
        ("404", "404: Not Found"),
        ("endpoint does not exist", "Endpoint does not exist"),
        ("not found", "404: Not Found"),
        ("no such endpoint", "Endpoint does not exist"),
        ("unknown endpoint", "Unknown endpoint"),
        ("invalid endpoint", "Invalid endpoint"),
        ("method not allowed", "405: Method not allowed"),
    ]

    for pattern, reason in broken_patterns:
        if pattern in error_lower:
            return True, reason

    return False, ""


def record_live_data_call(
    source_id: int,
    endpoint_name: str,
    success: bool,
    latency_ms: int,
    error: Optional[str] = None,
    db: Optional[Session] = None,
) -> bool:
    """
    Record a live data API call for statistics tracking.

    Updates both the source-level stats and endpoint-level stats.
    Also detects and marks tools that return 404/not-found errors as broken.
    """

    def _record(session: Session) -> bool:
        # Update source-level stats
        stmt = select(LiveDataSource).where(LiveDataSource.id == source_id)
        source = session.execute(stmt).scalar_one_or_none()
        if not source:
            return False

        source.total_calls = (source.total_calls or 0) + 1
        source.total_latency_ms = (source.total_latency_ms or 0) + latency_ms
        if success:
            source.successful_calls = (source.successful_calls or 0) + 1
        else:
            source.failed_calls = (source.failed_calls or 0) + 1

        # Update endpoint-level stats
        endpoint_stmt = select(LiveDataSourceEndpointStats).where(
            LiveDataSourceEndpointStats.source_id == source_id,
            LiveDataSourceEndpointStats.endpoint_name == endpoint_name,
        )
        endpoint_stats = session.execute(endpoint_stmt).scalar_one_or_none()

        if not endpoint_stats:
            # Create new endpoint stats record
            endpoint_stats = LiveDataSourceEndpointStats(
                source_id=source_id,
                endpoint_name=endpoint_name,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                total_latency_ms=0,
            )
            session.add(endpoint_stats)

        endpoint_stats.total_calls = (endpoint_stats.total_calls or 0) + 1
        endpoint_stats.total_latency_ms = (
            endpoint_stats.total_latency_ms or 0
        ) + latency_ms
        endpoint_stats.last_called = datetime.utcnow()

        if success:
            endpoint_stats.successful_calls = (endpoint_stats.successful_calls or 0) + 1
            endpoint_stats.last_error = None
            # If tool succeeds, unmark it as broken (it may have been fixed)
            if endpoint_stats.is_broken:
                endpoint_stats.is_broken = False
                endpoint_stats.broken_reason = None
                endpoint_stats.broken_at = None
                logger.info(
                    f"Tool {endpoint_name} unmarked as broken after successful call"
                )
        else:
            endpoint_stats.failed_calls = (endpoint_stats.failed_calls or 0) + 1
            endpoint_stats.last_error = error

            # Check if this error indicates the tool doesn't exist
            is_broken, broken_reason = _is_broken_tool_error(error)
            if is_broken and not endpoint_stats.is_broken:
                endpoint_stats.is_broken = True
                endpoint_stats.broken_reason = broken_reason
                endpoint_stats.broken_at = datetime.utcnow()
                logger.warning(
                    f"Tool {endpoint_name} marked as broken: {broken_reason}"
                )

        session.commit()
        return True

    if db:
        return _record(db)
    with get_db_context() as session:
        return _record(session)


def get_endpoint_stats(source_id: int, db: Optional[Session] = None) -> list[dict]:
    """Get endpoint stats for a live data source."""

    def _query(session: Session) -> list[dict]:
        stmt = (
            select(LiveDataSourceEndpointStats)
            .where(LiveDataSourceEndpointStats.source_id == source_id)
            .order_by(LiveDataSourceEndpointStats.total_calls.desc())
        )
        stats = session.execute(stmt).scalars().all()
        return [s.to_dict() for s in stats]

    if db:
        return _query(db)
    with get_db_context() as session:
        return _query(session)


def get_broken_tools(source_id: int, db: Optional[Session] = None) -> list[str]:
    """
    Get list of broken tool names for a source.

    These tools should be filtered out of designator tool lists.
    """

    def _query(session: Session) -> list[str]:
        stmt = (
            select(LiveDataSourceEndpointStats.endpoint_name)
            .where(LiveDataSourceEndpointStats.source_id == source_id)
            .where(LiveDataSourceEndpointStats.is_broken == True)
        )
        results = session.execute(stmt).scalars().all()
        return list(results)

    if db:
        return _query(db)
    with get_db_context() as session:
        return _query(session)


def reset_broken_tool(
    source_id: int, endpoint_name: str, db: Optional[Session] = None
) -> bool:
    """
    Reset a broken tool so it can be tried again.

    Returns True if reset, False if tool not found.
    """

    def _reset(session: Session) -> bool:
        stmt = select(LiveDataSourceEndpointStats).where(
            LiveDataSourceEndpointStats.source_id == source_id,
            LiveDataSourceEndpointStats.endpoint_name == endpoint_name,
        )
        endpoint_stats = session.execute(stmt).scalar_one_or_none()
        if not endpoint_stats:
            return False

        endpoint_stats.is_broken = False
        endpoint_stats.broken_reason = None
        endpoint_stats.broken_at = None
        session.commit()
        logger.info(f"Reset broken tool: {endpoint_name}")
        return True

    if db:
        return _reset(db)
    with get_db_context() as session:
        return _reset(session)


def reset_all_broken_tools(source_id: int, db: Optional[Session] = None) -> int:
    """
    Reset all broken tools for a source.

    Returns count of tools reset.
    """

    def _reset(session: Session) -> int:
        stmt = (
            select(LiveDataSourceEndpointStats)
            .where(LiveDataSourceEndpointStats.source_id == source_id)
            .where(LiveDataSourceEndpointStats.is_broken == True)
        )
        broken_tools = session.execute(stmt).scalars().all()

        count = 0
        for tool in broken_tools:
            tool.is_broken = False
            tool.broken_reason = None
            tool.broken_at = None
            count += 1

        if count > 0:
            session.commit()
            logger.info(f"Reset {count} broken tools for source {source_id}")

        return count

    if db:
        return _reset(db)
    with get_db_context() as session:
        return _reset(session)


def get_top_live_data_sources(
    limit: int = 5, db: Optional[Session] = None
) -> list[dict]:
    """Get top live data sources ranked by total API calls."""

    def _query(session: Session) -> list[dict]:
        stmt = (
            select(LiveDataSource)
            .where(LiveDataSource.enabled == True)
            .where(LiveDataSource.total_calls > 0)
            .order_by(LiveDataSource.total_calls.desc())
            .limit(limit)
        )
        sources = session.execute(stmt).scalars().all()
        return [
            {
                "id": s.id,
                "name": s.name,
                "source_type": s.source_type,
                "total_calls": s.total_calls,
                "successful_calls": s.successful_calls,
                "failed_calls": s.failed_calls,
                "success_rate": (
                    round(s.successful_calls / s.total_calls * 100, 1)
                    if s.total_calls > 0
                    else 0
                ),
                "avg_latency_ms": (
                    s.total_latency_ms // s.total_calls if s.total_calls > 0 else 0
                ),
            }
            for s in sources
        ]

    if db:
        return _query(db)
    with get_db_context() as session:
        return _query(session)


def _get_live_source_info_for_doc_type(doc_source_type: str) -> Optional[dict]:
    """
    Get live source info for a document store type using dynamic plugin lookup.

    Returns dict with live_type, data_type, description, best_for - or None if no mapping.
    """
    try:
        from plugin_base.loader import get_doc_to_live_source_info

        return get_doc_to_live_source_info(doc_source_type)
    except ImportError:
        logger.warning("Plugin loader not available for dynamic live source lookup")
        return None


def sync_live_source_for_document_store(
    doc_store_name: str,
    doc_store_source_type: str,
    google_account_id: int,
    google_account_email: str,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> Optional[LiveDataSource]:
    """
    Auto-create or update a live data source when a document store is created/updated.

    Uses dynamic plugin lookup to map document store types to their corresponding
    live source types. Custom plugins can register their own mappings.

    Returns the created/updated live source, or None if not applicable.
    """
    mapping = _get_live_source_info_for_doc_type(doc_store_source_type)
    if not mapping:
        return None  # Not a Google source type that has a live equivalent

    # Generate a unique name based on the document store name
    # e.g., "My Gmail" -> "my-gmail-live"
    live_source_name = f"{doc_store_name.lower().replace(' ', '-')}-live"

    # Check if this live source already exists
    existing = get_live_data_source_by_name(live_source_name, db=db)

    auth_config = {"account_id": google_account_id}

    if existing:
        # Update existing source with current account_id
        updated = update_live_data_source(
            source_id=existing.id,
            auth_config=auth_config,
            enabled=enabled,
            db=db,
        )
        logger.info(
            f"Updated live source '{live_source_name}' for document store '{doc_store_name}'"
        )
        return updated
    else:
        # Create new live source
        created = create_live_data_source(
            name=live_source_name,
            source_type=mapping["live_type"],
            description=f"{mapping['description']} ({google_account_email})",
            data_type=mapping["data_type"],
            best_for=mapping["best_for"],
            auth_config=auth_config,
            cache_ttl_seconds=60,
            timeout_seconds=30,
            enabled=enabled,
            db=db,
        )
        logger.info(
            f"Created live source '{live_source_name}' for document store '{doc_store_name}'"
        )
        return created


def delete_live_source_for_document_store(
    doc_store_name: str,
    db: Optional[Session] = None,
) -> bool:
    """
    Delete the auto-created live source when a Google document store is deleted.

    Returns True if deleted, False if not found or couldn't delete.
    """
    live_source_name = f"{doc_store_name.lower().replace(' ', '-')}-live"
    existing = get_live_data_source_by_name(live_source_name, db=db)

    if existing:
        return delete_live_data_source(existing.id, db=db)
    return False


def sync_all_google_live_sources() -> list[str]:
    """
    Create live sources for all existing document stores that have live equivalents.

    This is a one-time migration utility to backfill live sources for stores
    created before the auto-sync feature was added.

    Uses dynamic plugin lookup - works with both builtin and custom plugins.

    Returns list of created live source names.
    """
    from .document_stores import get_all_document_stores
    from .oauth_tokens import get_oauth_token_info

    created = []
    stores = get_all_document_stores()

    for store in stores:
        # Check if this doc store type has a live equivalent via dynamic lookup
        mapping = _get_live_source_info_for_doc_type(store.source_type)
        if not mapping:
            continue

        if not store.google_account_id:
            logger.warning(f"Skipping {store.name}: no Google account ID")
            continue

        # Check if live source already exists
        live_source_name = f"{store.name.lower().replace(' ', '-')}-live"
        existing = get_live_data_source_by_name(live_source_name)
        if existing:
            logger.info(f"Live source already exists: {live_source_name}")
            continue

        # Get account email
        oauth_token = get_oauth_token_info(store.google_account_id)
        if not oauth_token:
            logger.warning(
                f"Skipping {store.name}: OAuth token not found for account {store.google_account_id}"
            )
            continue

        # Create live source
        live_source = sync_live_source_for_document_store(
            doc_store_name=store.name,
            doc_store_source_type=store.source_type,
            google_account_id=store.google_account_id,
            google_account_email=oauth_token.get("account_email") or "unknown",
            enabled=store.enabled,
        )

        if live_source:
            created.append(live_source.name)

    logger.info(
        f"Synced {len(created)} live sources for existing Google document stores"
    )
    return created
