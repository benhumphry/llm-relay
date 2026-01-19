"""
Live Data Cache CRUD operations.

Provides caching for:
1. Data Cache - API responses with TTL-based expiration
2. Entity Cache - Name-to-identifier mappings (e.g., "Apple" -> "AAPL")

TTL strategies:
- Current/real-time data: Short TTL (15 min - 1 hour)
- Historical data: Never expires (immutable)
- Entity resolution: Long TTL (90 days default)
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, select, update

from .connection import get_db_context
from .models import LiveDataCache, LiveEntityCache
from .settings import get_setting

logger = logging.getLogger(__name__)

# Default TTL values (in seconds) - can be overridden via settings
DEFAULT_TTLS = {
    # Stock/financial data
    "price": 900,  # 15 minutes for current prices
    "quote": 900,
    "ticker": 900,
    "market": 900,
    # Weather
    "weather_current": 3600,  # 1 hour
    "weather_forecast": 21600,  # 6 hours
    "weather_history": 0,  # Never expires
    # Sports
    "score_live": 120,  # 2 minutes during live games
    "score_final": 0,  # Never expires
    # Generic
    "default": 3600,  # 1 hour default
    "historical": 0,  # Never expires for any historical data
}

# Patterns to detect historical queries (dates in the past)
HISTORICAL_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",  # ISO date
    r"history|historical|past|ago|yesterday|last\s+(week|month|year)",
]


def _compute_args_hash(args: dict) -> str:
    """Compute a stable hash of tool arguments for cache key."""
    # Sort keys for consistent hashing
    normalized = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _determine_ttl(tool_name: str, args: dict, source_type: str) -> int:
    """
    Determine appropriate TTL based on tool name, args, and data type.

    Returns TTL in seconds. 0 means never expires.
    """
    tool_lower = tool_name.lower()
    args_str = json.dumps(args).lower()

    # Check for historical data patterns
    for pattern in HISTORICAL_PATTERNS:
        if re.search(pattern, args_str):
            # Check if the date is in the past
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", args_str)
            if date_match:
                try:
                    date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                    if date.date() < datetime.utcnow().date():
                        return DEFAULT_TTLS["historical"]  # Never expires
                except ValueError:
                    pass

    # Check for specific tool patterns
    if any(term in tool_lower for term in ["history", "historical", "past"]):
        return DEFAULT_TTLS["historical"]

    # Weather patterns
    if "weather" in tool_lower or source_type == "weather":
        if "forecast" in tool_lower:
            return _get_ttl_setting("weather_forecast")
        elif "history" in tool_lower or "historical" in tool_lower:
            return DEFAULT_TTLS["historical"]
        else:
            return _get_ttl_setting("weather_current")

    # Stock/financial patterns
    if any(term in tool_lower for term in ["price", "quote", "ticker", "stock"]):
        return _get_ttl_setting("price")

    # Sports patterns
    if any(term in tool_lower for term in ["score", "match", "game"]):
        # Check if it's a final/completed result
        if any(term in args_str for term in ["final", "completed", "finished"]):
            return DEFAULT_TTLS["score_final"]
        return _get_ttl_setting("score_live")

    return _get_ttl_setting("default")


def _get_ttl_setting(key: str) -> int:
    """Get TTL from settings, falling back to defaults."""
    setting_key = f"live_cache_ttl_{key}"
    value = get_setting(setting_key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return DEFAULT_TTLS.get(key, DEFAULT_TTLS["default"])


def is_cache_enabled() -> bool:
    """Check if live data caching is enabled."""
    value = get_setting("live_cache_enabled")
    if value is None:
        return True  # Enabled by default
    return value.lower() in ("true", "1", "yes", "enabled")


def get_entity_ttl_days() -> int:
    """Get the TTL for entity cache entries in days."""
    value = get_setting("live_cache_entity_ttl_days")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return 90  # Default 90 days


# =============================================================================
# Data Cache Operations
# =============================================================================


def cache_data_response(
    source_type: str,
    tool_name: str,
    args: dict,
    response_data: str,
    response_summary: str | None = None,
    ttl_seconds: int | None = None,
) -> bool:
    """
    Cache an API response.

    Args:
        source_type: e.g., "stocks", "weather", "transport"
        tool_name: API tool/endpoint name
        args: Tool arguments (will be hashed)
        response_data: JSON string of the response
        response_summary: Optional human-readable summary
        ttl_seconds: Override TTL (None = auto-determine)

    Returns:
        True if cached successfully
    """
    if not is_cache_enabled():
        return False

    try:
        args_hash = _compute_args_hash(args)

        if ttl_seconds is None:
            ttl_seconds = _determine_ttl(tool_name, args, source_type)

        expires_at = None
        if ttl_seconds > 0:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        with get_db_context() as db:
            # Upsert the cache entry
            existing = db.execute(
                select(LiveDataCache).where(
                    LiveDataCache.source_type == source_type,
                    LiveDataCache.tool_name == tool_name,
                    LiveDataCache.args_hash == args_hash,
                )
            ).scalar_one_or_none()

            if existing:
                existing.response_data = response_data
                existing.response_summary = response_summary
                existing.ttl_seconds = ttl_seconds
                existing.expires_at = expires_at
                existing.created_at = datetime.utcnow()
                # Reset hit count on refresh
                existing.hit_count = 0
                existing.last_hit_at = None
            else:
                entry = LiveDataCache(
                    source_type=source_type,
                    tool_name=tool_name,
                    args_hash=args_hash,
                    response_data=response_data,
                    response_summary=response_summary,
                    ttl_seconds=ttl_seconds,
                    expires_at=expires_at,
                )
                db.add(entry)

            db.commit()
            logger.debug(
                f"Cached {source_type}:{tool_name} (TTL: {ttl_seconds}s, "
                f"expires: {expires_at})"
            )
            return True

    except Exception as e:
        logger.warning(f"Failed to cache data response: {e}")
        return False


def lookup_data_cache(
    source_type: str,
    tool_name: str,
    args: dict,
) -> Optional[str]:
    """
    Look up a cached API response.

    Args:
        source_type: e.g., "stocks", "weather"
        tool_name: API tool/endpoint name
        args: Tool arguments

    Returns:
        Cached response data (JSON string) if found and not expired, None otherwise
    """
    if not is_cache_enabled():
        return None

    try:
        args_hash = _compute_args_hash(args)

        with get_db_context() as db:
            entry = db.execute(
                select(LiveDataCache).where(
                    LiveDataCache.source_type == source_type,
                    LiveDataCache.tool_name == tool_name,
                    LiveDataCache.args_hash == args_hash,
                )
            ).scalar_one_or_none()

            if not entry:
                return None

            if entry.is_expired():
                # Delete expired entry
                db.delete(entry)
                db.commit()
                logger.debug(f"Cache expired for {source_type}:{tool_name}")
                return None

            # Update hit stats
            entry.hit_count += 1
            entry.last_hit_at = datetime.utcnow()
            db.commit()

            logger.debug(
                f"Cache hit for {source_type}:{tool_name} (hits: {entry.hit_count})"
            )
            return entry.response_data

    except Exception as e:
        logger.warning(f"Failed to lookup data cache: {e}")
        return None


def clear_data_cache(
    source_type: str | None = None,
    tool_name: str | None = None,
    expired_only: bool = False,
) -> int:
    """
    Clear data cache entries.

    Args:
        source_type: Filter by source type (None = all)
        tool_name: Filter by tool name (None = all)
        expired_only: Only clear expired entries

    Returns:
        Number of entries deleted
    """
    try:
        with get_db_context() as db:
            stmt = delete(LiveDataCache)

            conditions = []
            if source_type:
                conditions.append(LiveDataCache.source_type == source_type)
            if tool_name:
                conditions.append(LiveDataCache.tool_name == tool_name)
            if expired_only:
                conditions.append(LiveDataCache.expires_at < datetime.utcnow())
                conditions.append(LiveDataCache.expires_at.isnot(None))

            if conditions:
                stmt = stmt.where(*conditions)

            result = db.execute(stmt)
            db.commit()

            count = result.rowcount
            logger.info(f"Cleared {count} data cache entries")
            return count

    except Exception as e:
        logger.warning(f"Failed to clear data cache: {e}")
        return 0


def get_data_cache_stats() -> dict:
    """Get statistics about the data cache."""
    try:
        with get_db_context() as db:
            total = db.execute(select(func.count()).select_from(LiveDataCache)).scalar()

            expired = db.execute(
                select(func.count())
                .select_from(LiveDataCache)
                .where(
                    LiveDataCache.expires_at < datetime.utcnow(),
                    LiveDataCache.expires_at.isnot(None),
                )
            ).scalar()

            permanent = db.execute(
                select(func.count())
                .select_from(LiveDataCache)
                .where(LiveDataCache.expires_at.is_(None))
            ).scalar()

            total_hits = (
                db.execute(select(func.sum(LiveDataCache.hit_count))).scalar() or 0
            )

            # Get by source type
            by_source = {}
            source_counts = db.execute(
                select(
                    LiveDataCache.source_type,
                    func.count().label("count"),
                    func.sum(LiveDataCache.hit_count).label("hits"),
                ).group_by(LiveDataCache.source_type)
            ).all()
            for row in source_counts:
                by_source[row.source_type] = {
                    "count": row.count,
                    "hits": row.hits or 0,
                }

            return {
                "total_entries": total,
                "expired_entries": expired,
                "permanent_entries": permanent,
                "active_entries": total - expired,
                "total_hits": total_hits,
                "by_source": by_source,
            }

    except Exception as e:
        logger.warning(f"Failed to get data cache stats: {e}")
        return {"error": str(e)}


# =============================================================================
# Entity Cache Operations
# =============================================================================


def cache_entity(
    source_type: str,
    entity_type: str,
    query_text: str,
    resolved_value: str,
    display_name: str | None = None,
    metadata: dict | None = None,
    ttl_days: int | None = None,
) -> bool:
    """
    Cache an entity resolution.

    Args:
        source_type: e.g., "stocks", "weather", "transport"
        entity_type: e.g., "symbol", "location", "station"
        query_text: Original query text (e.g., "Apple", "London")
        resolved_value: Resolved API value (e.g., "AAPL", "328328")
        display_name: Human-readable name (e.g., "Apple Inc.")
        metadata: Additional metadata dict
        ttl_days: Days until expiry (0 = never), None = use default from settings

    Returns:
        True if cached successfully
    """
    if not is_cache_enabled():
        return False

    if ttl_days is None:
        ttl_days = get_entity_ttl_days()

    try:
        query_normalized = query_text.lower().strip()

        expires_at = None
        if ttl_days > 0:
            expires_at = datetime.utcnow() + timedelta(days=ttl_days)

        metadata_json = json.dumps(metadata) if metadata else None

        with get_db_context() as db:
            existing = db.execute(
                select(LiveEntityCache).where(
                    LiveEntityCache.source_type == source_type,
                    LiveEntityCache.entity_type == entity_type,
                    LiveEntityCache.query_text == query_normalized,
                )
            ).scalar_one_or_none()

            if existing:
                existing.resolved_value = resolved_value
                existing.display_name = display_name
                existing.metadata_json = metadata_json
                existing.ttl_days = ttl_days
                existing.expires_at = expires_at
                existing.created_at = datetime.utcnow()
            else:
                entry = LiveEntityCache(
                    source_type=source_type,
                    entity_type=entity_type,
                    query_text=query_normalized,
                    resolved_value=resolved_value,
                    display_name=display_name,
                    metadata_json=metadata_json,
                    ttl_days=ttl_days,
                    expires_at=expires_at,
                )
                db.add(entry)

            db.commit()
            logger.debug(
                f"Cached entity: {source_type}:{entity_type}:{query_normalized} "
                f"-> {resolved_value}"
            )
            return True

    except Exception as e:
        logger.warning(f"Failed to cache entity: {e}")
        return False


def lookup_entity(
    source_type: str,
    entity_type: str,
    query_text: str,
) -> Optional[str]:
    """
    Look up a cached entity resolution.

    Args:
        source_type: e.g., "stocks", "weather"
        entity_type: e.g., "symbol", "location"
        query_text: Query text to match

    Returns:
        Resolved value if found and not expired, None otherwise
    """
    try:
        query_normalized = query_text.lower().strip()

        with get_db_context() as db:
            entry = db.execute(
                select(LiveEntityCache).where(
                    LiveEntityCache.source_type == source_type,
                    LiveEntityCache.entity_type == entity_type,
                    LiveEntityCache.query_text == query_normalized,
                )
            ).scalar_one_or_none()

            if not entry:
                return None

            if entry.is_expired():
                db.delete(entry)
                db.commit()
                logger.debug(
                    f"Entity cache expired: {source_type}:{entity_type}:{query_normalized}"
                )
                return None

            # Update hit stats
            entry.hit_count += 1
            entry.last_hit_at = datetime.utcnow()
            db.commit()

            logger.debug(
                f"Entity cache hit: {source_type}:{entity_type}:{query_normalized} "
                f"-> {entry.resolved_value} (hits: {entry.hit_count})"
            )
            return entry.resolved_value

    except Exception as e:
        logger.warning(f"Failed to lookup entity: {e}")
        return None


def lookup_entity_with_metadata(
    source_type: str,
    entity_type: str,
    query_text: str,
) -> Optional[dict]:
    """
    Look up a cached entity with full metadata.

    Returns dict with resolved_value, display_name, and metadata if found.
    """
    try:
        query_normalized = query_text.lower().strip()

        with get_db_context() as db:
            entry = db.execute(
                select(LiveEntityCache).where(
                    LiveEntityCache.source_type == source_type,
                    LiveEntityCache.entity_type == entity_type,
                    LiveEntityCache.query_text == query_normalized,
                )
            ).scalar_one_or_none()

            if not entry or entry.is_expired():
                return None

            # Update hit stats
            entry.hit_count += 1
            entry.last_hit_at = datetime.utcnow()
            db.commit()

            metadata = None
            if entry.metadata_json:
                try:
                    metadata = json.loads(entry.metadata_json)
                except json.JSONDecodeError:
                    pass

            return {
                "resolved_value": entry.resolved_value,
                "display_name": entry.display_name,
                "metadata": metadata,
            }

    except Exception as e:
        logger.warning(f"Failed to lookup entity with metadata: {e}")
        return None


def clear_entity_cache(
    source_type: str | None = None,
    entity_type: str | None = None,
    expired_only: bool = False,
) -> int:
    """
    Clear entity cache entries.

    Args:
        source_type: Filter by source type (None = all)
        entity_type: Filter by entity type (None = all)
        expired_only: Only clear expired entries

    Returns:
        Number of entries deleted
    """
    try:
        with get_db_context() as db:
            stmt = delete(LiveEntityCache)

            conditions = []
            if source_type:
                conditions.append(LiveEntityCache.source_type == source_type)
            if entity_type:
                conditions.append(LiveEntityCache.entity_type == entity_type)
            if expired_only:
                conditions.append(LiveEntityCache.expires_at < datetime.utcnow())
                conditions.append(LiveEntityCache.expires_at.isnot(None))

            if conditions:
                stmt = stmt.where(*conditions)

            result = db.execute(stmt)
            db.commit()

            count = result.rowcount
            logger.info(f"Cleared {count} entity cache entries")
            return count

    except Exception as e:
        logger.warning(f"Failed to clear entity cache: {e}")
        return 0


def get_entity_cache_stats() -> dict:
    """Get statistics about the entity cache."""
    try:
        with get_db_context() as db:
            total = db.execute(
                select(func.count()).select_from(LiveEntityCache)
            ).scalar()

            expired = db.execute(
                select(func.count())
                .select_from(LiveEntityCache)
                .where(
                    LiveEntityCache.expires_at < datetime.utcnow(),
                    LiveEntityCache.expires_at.isnot(None),
                )
            ).scalar()

            total_hits = (
                db.execute(select(func.sum(LiveEntityCache.hit_count))).scalar() or 0
            )

            # Get by source type
            by_source = {}
            source_counts = db.execute(
                select(
                    LiveEntityCache.source_type,
                    func.count().label("count"),
                    func.sum(LiveEntityCache.hit_count).label("hits"),
                ).group_by(LiveEntityCache.source_type)
            ).all()
            for row in source_counts:
                by_source[row.source_type] = {
                    "count": row.count,
                    "hits": row.hits or 0,
                }

            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "total_hits": total_hits,
                "by_source": by_source,
            }

    except Exception as e:
        logger.warning(f"Failed to get entity cache stats: {e}")
        return {"error": str(e)}


# =============================================================================
# Combined Operations
# =============================================================================


def get_all_cache_stats() -> dict:
    """Get combined statistics for both caches."""
    return {
        "data_cache": get_data_cache_stats(),
        "entity_cache": get_entity_cache_stats(),
    }


def clear_all_caches(expired_only: bool = False) -> dict:
    """Clear both caches."""
    return {
        "data_cache_cleared": clear_data_cache(expired_only=expired_only),
        "entity_cache_cleared": clear_entity_cache(expired_only=expired_only),
    }


def cleanup_expired_caches() -> dict:
    """
    Clean up expired entries from both caches.

    This should be called periodically (e.g., hourly) to prevent
    the cache tables from growing indefinitely.
    """
    return clear_all_caches(expired_only=True)
