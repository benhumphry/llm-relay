"""
Usage tracking module for LLM Relay.

Provides request logging, tag extraction, and usage statistics.
"""

from .ip_resolver import get_client_ip, resolve_hostname
from .tag_extractor import (
    RelayCommand,
    RelayCommandResult,
    extract_relay_commands_from_messages,
    extract_tag,
    normalize_tags,
    parse_relay_commands,
)
from .usage_tracker import tracker

__all__ = [
    "tracker",
    "extract_tag",
    "normalize_tags",
    "parse_relay_commands",
    "extract_relay_commands_from_messages",
    "RelayCommand",
    "RelayCommandResult",
    "get_client_ip",
    "resolve_hostname",
]
