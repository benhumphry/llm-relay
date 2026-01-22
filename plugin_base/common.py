"""
Common dataclasses shared across all plugin types.

This module defines the building blocks for plugin configuration:
- FieldDefinition: Declares a configuration field for the admin UI
- ValidationResult: Result of config validation
- FieldType: Supported field types for rendering
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FieldType(Enum):
    """Supported field types for admin UI rendering."""

    TEXT = "text"
    PASSWORD = "password"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXTAREA = "textarea"
    JSON = "json"
    OAUTH_ACCOUNT = "oauth_account"
    FOLDER_PICKER = "folder_picker"
    CALENDAR_PICKER = "calendar_picker"
    CHANNEL_PICKER = "channel_picker"
    LABEL_PICKER = "label_picker"
    TASKLIST_PICKER = "tasklist_picker"


class ContentCategory(Enum):
    """
    Content categories for document sources.

    Used for:
    1. Admin UI filtering of document stores
    2. Action handler account grouping (email, calendar, tasks)

    Document source plugins should set their content_category class attribute
    to indicate what type of content they provide.
    """

    FILES = "files"  # General documents, files, notes (Drive, OneDrive, Notion, etc.)
    EMAILS = "emails"  # Email messages (Gmail, Outlook)
    CALENDARS = "calendars"  # Calendar events (Google Calendar, Outlook Calendar)
    TASKS = "tasks"  # Tasks/todos (Google Tasks, Todoist)
    WEBSITES = "websites"  # Crawled web content
    CONTACTS = "contacts"  # Contact information
    MESSAGES = "messages"  # Chat messages (Slack, Teams)
    OTHER = "other"  # Anything else


# Legacy source_type to ContentCategory mapping
# Used for legacy sources that haven't been migrated to plugins yet
# Plugins should define content_category directly as a class attribute
LEGACY_SOURCE_TYPE_CATEGORIES: dict[str, ContentCategory] = {
    # Files (documents, notes, etc.)
    "local": ContentCategory.FILES,
    "mcp:gdrive": ContentCategory.FILES,
    "mcp:onedrive": ContentCategory.FILES,
    "notion": ContentCategory.FILES,
    "paperless": ContentCategory.FILES,
    "nextcloud": ContentCategory.FILES,
    "mcp:onenote": ContentCategory.FILES,
    "mcp:github": ContentCategory.FILES,
    # Emails
    "mcp:gmail": ContentCategory.EMAILS,
    "mcp:outlook": ContentCategory.EMAILS,
    # Calendars
    "mcp:gcalendar": ContentCategory.CALENDARS,
    "mcp:ocalendar": ContentCategory.CALENDARS,
    # Tasks
    "mcp:gtasks": ContentCategory.TASKS,
    "todoist": ContentCategory.TASKS,
    # Websites
    "website": ContentCategory.WEBSITES,
    "websearch": ContentCategory.WEBSITES,
    # Contacts
    "mcp:gcontacts": ContentCategory.CONTACTS,
    # Messages
    "slack": ContentCategory.MESSAGES,
    "mcp:teams": ContentCategory.MESSAGES,
}


def get_content_category(source_type: str) -> ContentCategory:
    """
    Get the content category for a source type.

    Args:
        source_type: The source type identifier (e.g., "mcp:gmail", "local")

    Returns:
        ContentCategory enum value, defaults to OTHER if not mapped
    """
    return LEGACY_SOURCE_TYPE_CATEGORIES.get(source_type, ContentCategory.OTHER)


@dataclass
class SelectOption:
    """Option for select/multiselect fields."""

    value: str
    label: str

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {"value": self.value, "label": self.label}


@dataclass
class FieldDefinition:
    """
    Defines a configuration field for the admin UI.

    Used by all plugin types to declare their configuration requirements.
    The admin UI dynamically renders forms based on these definitions.

    Attributes:
        name: Field identifier (used as key in config dict)
        label: Human-readable label for UI
        field_type: Type of field (determines UI widget)
        required: Whether field must have a value
        default: Default value if not provided
        help_text: Help text shown below field
        placeholder: Placeholder text in input
        options: For select/multiselect - list of SelectOption or dicts
        picker_options: Options for picker fields (oauth provider, etc.)
        depends_on: Conditional visibility - {"field_name": "expected_value"}
        min_value/max_value: For numeric fields
        min_length/max_length: For text fields
        pattern: Regex pattern for validation
    """

    name: str
    label: str
    field_type: FieldType | str  # FieldType enum or string for flexibility
    required: bool = False
    default: Any = None
    help_text: str = ""
    placeholder: str = ""

    # For select/multiselect fields
    options: list[SelectOption | dict] = field(default_factory=list)

    # For oauth_account, folder_picker, etc.
    # Example: {"provider": "google", "scopes": ["gmail"]}
    picker_options: dict = field(default_factory=dict)

    # Conditional visibility: {"field_name": "expected_value"}
    # Field is only shown when all conditions are met
    depends_on: dict = field(default_factory=dict)

    # Validation constraints
    min_value: Optional[int | float] = None
    max_value: Optional[int | float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern

    # Environment variable fallback - if set and env var exists, field is hidden
    env_var: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization to admin UI."""
        result = {
            "name": self.name,
            "label": self.label,
            "field_type": (
                self.field_type.value
                if isinstance(self.field_type, FieldType)
                else self.field_type
            ),
            "required": self.required,
            "help_text": self.help_text,
            "placeholder": self.placeholder,
        }

        if self.default is not None:
            result["default"] = self.default

        if self.options:
            result["options"] = [
                o.to_dict() if isinstance(o, SelectOption) else o for o in self.options
            ]

        if self.picker_options:
            result["picker_options"] = self.picker_options

        if self.depends_on:
            result["depends_on"] = self.depends_on

        # Add validation constraints if set
        for attr in ["min_value", "max_value", "min_length", "max_length", "pattern"]:
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val

        # Add env_var if set
        if self.env_var:
            result["env_var"] = self.env_var

        return result


@dataclass
class ValidationError:
    """Validation error for a specific field."""

    field: str
    message: str

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {"field": self.field, "message": self.message}


@dataclass
class ValidationResult:
    """Result of config validation."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        """Combined error message for logging/display."""
        return "; ".join(f"{e.field}: {e.message}" for e in self.errors)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
        }


def validate_field_value(
    field_def: FieldDefinition, value: Any
) -> list[ValidationError]:
    """
    Validate a single field value against its definition.

    Returns a list of ValidationErrors (empty if valid).
    """
    errors = []

    # Required check
    if field_def.required and (value is None or value == ""):
        errors.append(ValidationError(field_def.name, "This field is required"))
        return errors  # Skip other validations if required and missing

    if value is None or value == "":
        return errors  # Optional field with no value is valid

    # Type-specific validation
    field_type = (
        field_def.field_type
        if isinstance(field_def.field_type, FieldType)
        else FieldType(field_def.field_type)
    )

    if field_type == FieldType.INTEGER:
        try:
            int_val = int(value)
            if field_def.min_value is not None and int_val < field_def.min_value:
                errors.append(
                    ValidationError(
                        field_def.name, f"Minimum value is {field_def.min_value}"
                    )
                )
            if field_def.max_value is not None and int_val > field_def.max_value:
                errors.append(
                    ValidationError(
                        field_def.name, f"Maximum value is {field_def.max_value}"
                    )
                )
        except (ValueError, TypeError):
            errors.append(ValidationError(field_def.name, "Must be an integer"))

    elif field_type == FieldType.NUMBER:
        try:
            num_val = float(value)
            if field_def.min_value is not None and num_val < field_def.min_value:
                errors.append(
                    ValidationError(
                        field_def.name, f"Minimum value is {field_def.min_value}"
                    )
                )
            if field_def.max_value is not None and num_val > field_def.max_value:
                errors.append(
                    ValidationError(
                        field_def.name, f"Maximum value is {field_def.max_value}"
                    )
                )
        except (ValueError, TypeError):
            errors.append(ValidationError(field_def.name, "Must be a number"))

    elif field_type in (FieldType.TEXT, FieldType.PASSWORD, FieldType.TEXTAREA):
        str_val = str(value)
        if field_def.min_length is not None and len(str_val) < field_def.min_length:
            errors.append(
                ValidationError(
                    field_def.name, f"Minimum length is {field_def.min_length}"
                )
            )
        if field_def.max_length is not None and len(str_val) > field_def.max_length:
            errors.append(
                ValidationError(
                    field_def.name, f"Maximum length is {field_def.max_length}"
                )
            )
        if field_def.pattern is not None:
            import re

            if not re.match(field_def.pattern, str_val):
                errors.append(
                    ValidationError(
                        field_def.name, f"Must match pattern: {field_def.pattern}"
                    )
                )

    elif field_type == FieldType.SELECT:
        # Validate value is one of the options
        if field_def.options:
            valid_values = [
                o.value if isinstance(o, SelectOption) else o.get("value")
                for o in field_def.options
            ]
            if value not in valid_values:
                errors.append(
                    ValidationError(field_def.name, f"Invalid option: {value}")
                )

    elif field_type == FieldType.MULTISELECT:
        # Value should be a list
        if not isinstance(value, list):
            errors.append(ValidationError(field_def.name, "Must be a list"))
        elif field_def.options:
            valid_values = [
                o.value if isinstance(o, SelectOption) else o.get("value")
                for o in field_def.options
            ]
            for v in value:
                if v not in valid_values:
                    errors.append(
                        ValidationError(field_def.name, f"Invalid option: {v}")
                    )

    elif field_type == FieldType.BOOLEAN:
        if not isinstance(value, bool):
            # Allow string representations
            if str(value).lower() not in ("true", "false", "1", "0"):
                errors.append(ValidationError(field_def.name, "Must be a boolean"))

    return errors


def validate_config(
    field_definitions: list[FieldDefinition], config: dict
) -> ValidationResult:
    """
    Validate a config dict against a list of field definitions.

    Args:
        field_definitions: List of FieldDefinition for the plugin
        config: Config dict to validate

    Returns:
        ValidationResult with valid flag and any errors
    """
    all_errors = []

    for field_def in field_definitions:
        # Check conditional visibility
        if field_def.depends_on:
            visible = all(
                config.get(dep_field) == dep_value
                for dep_field, dep_value in field_def.depends_on.items()
            )
            if not visible:
                continue  # Skip validation for hidden fields

        value = config.get(field_def.name)
        errors = validate_field_value(field_def, value)
        all_errors.extend(errors)

    return ValidationResult(valid=len(all_errors) == 0, errors=all_errors)
