"""
Base class for live data source plugins.

Live sources fetch real-time data at request time (unlike document sources
which index content for RAG). The designator selects relevant sources and
provides parameters based on the user's query.

Full implementation in Phase 3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from plugin_base.common import FieldDefinition, ValidationResult, validate_config


@dataclass
class ParamDefinition:
    """
    Defines a parameter the designator can pass at query time.

    Unlike FieldDefinition (static config), these are dynamic per-request.
    """

    name: str
    description: str  # Included in designator prompt
    param_type: str  # "string", "integer", "number", "boolean", "datetime"
    required: bool = False
    default: Any = None
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "param_type": self.param_type,
            "required": self.required,
            "default": self.default,
            "examples": self.examples,
        }


@dataclass
class LiveDataResult:
    """Result from fetch()."""

    success: bool
    data: Any = None  # Raw data (for caching)
    formatted: str = ""  # Formatted string for LLM context
    error: Optional[str] = None
    cache_ttl: int = 300  # Seconds, 0 = don't cache, -1 = cache forever
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PluginLiveSource(ABC):
    """
    Base class for live data source plugins.

    Live sources fetch real-time data at request time (unlike document sources
    which index content for RAG). The designator selects relevant sources and
    provides parameters based on the user's query.

    Subclasses define:
    - source_type: Unique identifier
    - display_name: Human-readable name
    - description: Help text
    - data_type: Category (weather, finance, sports, etc.)
    - best_for: Description for designator prompt
    - get_config_fields(): Static configuration (API keys, etc.)
    - get_param_definitions(): Dynamic parameters for designator
    - fetch(): Data fetching logic
    """

    # --- Required class attributes ---
    source_type: str
    display_name: str
    description: str
    data_type: str  # Category for grouping
    best_for: str  # Description for designator prompt

    # --- Optional ---
    icon: str = "🔌"
    default_cache_ttl: int = 300

    # Mark as abstract to prevent direct registration
    _abstract: bool = True

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI (API keys, settings, etc.)."""
        pass

    @classmethod
    @abstractmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can pass at query time."""
        pass

    @classmethod
    def get_designator_hint(cls) -> str:
        """
        Generate hint for designator prompt.

        Override for custom formatting.
        """
        params = cls.get_param_definitions()

        param_parts = []
        for p in params:
            req = "" if p.required else "optional, "
            examples = f" e.g., {', '.join(p.examples)}" if p.examples else ""
            param_parts.append(f"{p.name} ({req}{p.description}{examples})")

        param_str = "; ".join(param_parts) if param_parts else "no parameters"
        return f"Parameters: {param_str}. {cls.best_for}"

    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """Validate plugin configuration."""
        return validate_config(cls.get_config_fields(), config)

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with validated config."""
        pass

    @abstractmethod
    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live data based on designator-provided params.

        Args:
            params: Dict from designator, matches get_param_definitions()

        Returns:
            LiveDataResult with formatted string for context injection
        """
        pass

    def validate_params(self, params: dict) -> ValidationResult:
        """Validate query parameters."""
        from plugin_base.common import ValidationError

        errors = []
        for param_def in self.get_param_definitions():
            if param_def.required and params.get(param_def.name) is None:
                errors.append(ValidationError(param_def.name, "Required parameter"))
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def is_available(self) -> bool:
        """Check if source is configured and available."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test the connection with a sample query."""
        try:
            result = self.fetch({})
            if result.success:
                preview = (
                    result.formatted[:100] + "..."
                    if len(result.formatted) > 100
                    else result.formatted
                )
                return True, f"OK: {preview}"
            return False, result.error or "Unknown error"
        except Exception as e:
            return False, str(e)
