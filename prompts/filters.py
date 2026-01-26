"""
Custom Jinja2 filters for prompt templates.

These filters can be used in templates like:
    {{ current_time | format_datetime('%A, %B %d') }}
    {{ context_window | format_number }}
    {{ long_text | truncate(100) }}
"""

from datetime import date, datetime
from typing import Any


def format_datetime(
    value: datetime | date | str | None, fmt: str = "%Y-%m-%d %H:%M"
) -> str:
    """
    Format a datetime object or ISO string.

    Args:
        value: datetime, date, or ISO format string
        fmt: strftime format string

    Examples:
        {{ current_time | format_datetime('%A, %B %d, %Y') }}
        {{ event_date | format_datetime('%Y-%m-%d') }}
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    return value.strftime(fmt)


def format_number(value: int | float | None, sep: str = ",") -> str:
    """
    Format a number with thousand separators.

    Args:
        value: Number to format
        sep: Separator character (default: comma)

    Examples:
        {{ context_window | format_number }}  -> "128,000"
        {{ price | format_number('.') }}      -> "1.234.567"
    """
    if value is None:
        return ""
    return f"{value:,}".replace(",", sep)


def truncate(value: str | None, length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        value: String to truncate
        length: Maximum length including suffix
        suffix: String to append when truncated

    Examples:
        {{ long_description | truncate(50) }}
        {{ text | truncate(100, '…') }}
    """
    if value is None:
        return ""
    if len(value) <= length:
        return value
    return value[: length - len(suffix)] + suffix


def indent(value: str | None, width: int = 2, first: bool = True) -> str:
    """
    Indent all lines in a string.

    Args:
        value: String to indent
        width: Number of spaces
        first: Whether to indent the first line

    Examples:
        {{ multiline_text | indent(4) }}
        {{ code | indent(2, first=False) }}
    """
    if value is None:
        return ""
    prefix = " " * width
    lines = value.split("\n")
    if first:
        return "\n".join(prefix + line for line in lines)
    if not lines:
        return ""
    return lines[0] + "\n" + "\n".join(prefix + line for line in lines[1:])


def join_lines(value: list | None, separator: str = "\n") -> str:
    """
    Join a list of strings with a separator.

    Args:
        value: List of strings
        separator: Separator between items

    Examples:
        {{ items | join_lines }}
        {{ items | join_lines(', ') }}
    """
    if value is None:
        return ""
    return separator.join(str(v) for v in value)


def default(value: Any, default_value: Any) -> Any:
    """
    Return default value if value is None or empty.

    Args:
        value: Value to check
        default_value: Value to return if empty

    Examples:
        {{ user_name | default('Anonymous') }}
    """
    if value is None or value == "":
        return default_value
    return value


def json_escape(value: str | None) -> str:
    """
    Escape a string for safe inclusion in JSON.

    Args:
        value: String to escape

    Examples:
        {{ user_input | json_escape }}
    """
    if value is None:
        return ""
    import json

    # json.dumps adds quotes, so we strip them
    return json.dumps(value)[1:-1]


def pluralize(value: int, singular: str = "", plural: str = "s") -> str:
    """
    Return singular or plural suffix based on count.

    Args:
        value: Count to check
        singular: Suffix for singular (default: empty)
        plural: Suffix for plural (default: 's')

    Examples:
        {{ count }} task{{ count | pluralize }}
        {{ count }} {{ count | pluralize('entry', 'entries') }}
    """
    if value == 1:
        return singular
    return plural


def register_filters(env):
    """Register all custom filters with a Jinja2 environment."""
    env.filters["format_datetime"] = format_datetime
    env.filters["format_number"] = format_number
    env.filters["truncate"] = truncate
    env.filters["indent"] = indent
    env.filters["join_lines"] = join_lines
    env.filters["default"] = default
    env.filters["json_escape"] = json_escape
    env.filters["pluralize"] = pluralize
