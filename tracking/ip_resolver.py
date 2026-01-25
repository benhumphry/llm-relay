"""
IP address resolution and hostname lookup.

Provides utilities for extracting client IP addresses from requests
and optionally resolving them to hostnames via reverse DNS.
"""

import logging
import socket
import threading
import time
from typing import Optional

from flask import Request

from db import Setting, get_db_context

logger = logging.getLogger(__name__)

# DNS cache with TTL
_dns_cache: dict[str, tuple[Optional[str], float]] = {}
_dns_cache_lock = threading.Lock()
_DNS_CACHE_TTL = 3600  # 1 hour
_DNS_TIMEOUT = 2  # seconds


def get_client_ip(request: Request) -> str:
    """
    Get client IP address, handling proxy headers.

    Checks headers in order:
    1. X-Forwarded-For (leftmost IP is original client)
    2. X-Real-IP
    3. request.remote_addr

    Args:
        request: Flask request object

    Returns:
        Client IP address string
    """
    # Check X-Forwarded-For header (leftmost is original client)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first (leftmost) IP
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to remote_addr
    return request.remote_addr or "unknown"


def resolve_hostname(ip: str) -> Optional[str]:
    """
    Resolve IP to hostname with caching.

    Uses a thread-safe cache with TTL to avoid repeated DNS lookups.
    Returns None if resolution fails, times out, or DNS resolution is disabled.

    Args:
        ip: IP address to resolve

    Returns:
        Hostname string or None
    """
    # Check if DNS resolution is enabled
    if not _is_dns_resolution_enabled():
        return None

    # Skip invalid IPs
    if not ip or ip == "unknown":
        return None

    # Check cache first
    now = time.time()
    with _dns_cache_lock:
        if ip in _dns_cache:
            hostname, timestamp = _dns_cache[ip]
            if now - timestamp < _DNS_CACHE_TTL:
                return hostname

    # Perform DNS lookup
    hostname = _do_dns_lookup(ip)

    # Update cache
    with _dns_cache_lock:
        _dns_cache[ip] = (hostname, now)

    return hostname


def _do_dns_lookup(ip: str) -> Optional[str]:
    """
    Perform actual DNS reverse lookup.

    Args:
        ip: IP address to resolve

    Returns:
        Hostname string or None
    """
    try:
        # Set a timeout for the lookup
        socket.setdefaulttimeout(_DNS_TIMEOUT)
        hostname = socket.gethostbyaddr(ip)[0]
        return hostname
    except (socket.herror, socket.gaierror, socket.timeout, OSError) as e:
        logger.debug(f"DNS lookup failed for {ip}: {e}")
        return None
    finally:
        # Reset timeout
        socket.setdefaulttimeout(None)


def _is_dns_resolution_enabled() -> bool:
    """Check if DNS resolution is enabled in settings."""
    try:
        with get_db_context() as db:
            setting = (
                db.query(Setting)
                .filter(Setting.key == Setting.KEY_DNS_RESOLUTION_ENABLED)
                .first()
            )
            if setting:
                return setting.value.lower() == "true"
    except Exception:
        pass
    # Default to enabled
    return True
