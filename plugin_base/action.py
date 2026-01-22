"""
Base class for action handler plugins.

Actions allow the LLM to perform side effects like sending emails,
creating calendar events, or managing tasks.

Full implementation in Phase 2.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from plugin_base.common import FieldDefinition, ValidationResult, validate_config


class ResourceType(Enum):
    """Types of resources that action handlers can require."""

    OAUTH_ACCOUNT = "oauth_account"  # OAuth account selector (filtered by provider)
    CALENDAR_PICKER = "calendar_picker"  # Calendar selector (depends on account)
    TASKLIST_PICKER = "tasklist_picker"  # Task list selector (depends on account)
    TEXT = "text"  # Free text input
    TEXTAREA = "textarea"  # Multi-line text
    PASSWORD = "password"  # Hidden text (API keys)
    SELECT = "select"  # Dropdown with options


@dataclass
class ResourceRequirement:
    """
    Defines a resource that an action handler needs from the Smart Alias.

    These are rendered dynamically in the Smart Alias edit modal's
    "Default Resources" section.

    Keys map to context.default_accounts structure:
    - email: {"id": account_id, "provider": "google|microsoft"}
    - calendar: {"id": account_id, "provider": "...", "calendar_id": "..."}
    - tasks: {"id": account_id, "provider": "gtasks", "list_id": "..."} or
             {"provider": "todoist", "api_token": "...", "list_id": "..."}
    - notification: {"urls": ["..."]}
    - schedule: {"account_id": ..., "calendar_id": "...", "provider": "..."}
    """

    key: str  # Context key (e.g., "email", "calendar", "tasks")
    label: str  # Display label
    resource_type: ResourceType  # Type of input
    help_text: str = ""  # Help text shown below input
    required: bool = False  # Whether this resource is required
    providers: list[str] = field(
        default_factory=list
    )  # For oauth_account: ["google", "microsoft"]
    depends_on: Optional[str] = None  # Show only when this other key has a value
    options: list[dict] = field(
        default_factory=list
    )  # For select type: [{"value": "x", "label": "X"}]
    sub_key: Optional[str] = None  # Sub-key within the context (e.g., "calendar_id")

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "resource_type": self.resource_type.value,
            "help_text": self.help_text,
            "required": self.required,
            "providers": self.providers,
            "depends_on": self.depends_on,
            "options": self.options,
            "sub_key": self.sub_key,
        }


class ActionRisk(Enum):
    """
    Risk level determines approval requirements.

    - READ_ONLY: No side effects, never needs approval
    - LOW: Minor side effects, can be pre-approved
    - MEDIUM: Visible side effects, confirmation recommended
    - HIGH: Significant side effects, always confirm
    - DESTRUCTIVE: Irreversible, cannot be automated
    """

    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


@dataclass
class ActionDefinition:
    """
    Defines an action that can be invoked by the LLM.

    Each plugin declares its available actions with parameters,
    risk levels, and examples for the LLM prompt.
    """

    name: str  # e.g., "create", "send", "delete"
    description: str  # Human-readable, included in LLM prompt
    risk: ActionRisk
    params: list[FieldDefinition]  # Parameters the LLM must provide
    examples: list[dict] = field(default_factory=list)  # Example param values

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "params": [p.to_dict() for p in self.params],
            "examples": self.examples,
        }


@dataclass
class ActionResult:
    """Result from action execution."""

    success: bool
    message: str  # Human-readable result message
    data: Any = None  # Structured result data (for chaining)
    error: Optional[str] = None


@dataclass
class ActionContext:
    """Context passed to execute()."""

    session_key: Optional[str] = None
    user_tags: list[str] = field(default_factory=list)
    smart_alias_name: str = ""
    conversation_id: Optional[str] = None

    # All available accounts by category - LLM can choose any of these
    # Keys: "email", "calendar", "tasks"
    # Each account has: {"id": "...", "type": "oauth|api|smtp", "provider": "google|microsoft|todoist|smtp", "email": "...", "name": "..."}
    # e.g.:
    #   available_accounts["email"] = [
    #       {"id": 1, "type": "oauth", "provider": "google", "email": "user@gmail.com"},
    #       {"id": 2, "type": "oauth", "provider": "microsoft", "email": "user@outlook.com"},
    #   ]
    #   available_accounts["tasks"] = [
    #       {"id": 1, "type": "oauth", "provider": "google", "email": "user@gmail.com"},
    #       {"id": "todoist", "type": "api", "provider": "todoist", "name": "Todoist"},
    #   ]
    available_accounts: dict[str, list[dict]] = field(default_factory=dict)

    # Default accounts from Smart Alias configuration (fallback if LLM doesn't specify)
    # Keys: "email", "calendar", "tasks", "notification", "schedule"
    # Values vary by type, e.g.:
    #   email/calendar: {"id": 1, "email": "user@gmail.com", "provider": "google"}
    #   tasks: {"id": 1, "email": "...", "provider": "google", "list_id": "..."}
    #          or {"provider": "todoist", "api_token": "..."}
    #   notification: {"urls": ["https://..."]}
    default_accounts: dict[str, dict] = field(default_factory=dict)


class PluginActionHandler(ABC):
    """
    Base class for action handler plugins.

    Subclasses define:
    - action_type: Unique identifier (e.g., "email", "slack", "todoist")
    - display_name: Human-readable name for admin UI
    - description: Help text
    - get_config_fields(): Configuration required (API keys, OAuth, etc.)
    - get_actions(): Available actions with parameters
    - execute(): Action execution logic

    Example usage in LLM response:
    ```xml
    <smart_action type="todoist" action="create">
    {"content": "Review PR", "due_string": "tomorrow"}
    </smart_action>
    ```
    """

    # --- Required class attributes (override in subclass) ---
    action_type: str  # Unique identifier
    display_name: str  # Shown in admin UI
    description: str  # Help text

    # --- Optional class attributes ---
    icon: str = "⚡"
    category: str = "other"  # For grouping: "communication", "productivity", etc.
    supported_sources: list[str] = []  # Document source types this handler works with

    # Mark as abstract to prevent direct registration
    _abstract: bool = True

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """
        Define configuration fields for the admin UI.

        These are set once when configuring the plugin, not per-request.
        Examples: API keys, OAuth accounts, default settings.

        NOTE: Most handlers should return [] here - resources come from
        Smart Alias context via get_resource_requirements() instead.
        Only truly global settings (like Apprise URL) belong here.
        """
        pass

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """
        Define resources this handler needs from the Smart Alias.

        These are rendered dynamically in the Smart Alias edit modal's
        "Default Resources" section when actions are enabled.

        The values are passed at execution time via context.default_accounts.

        Example:
            return [
                ResourceRequirement(
                    key="email",
                    label="Email Account",
                    resource_type=ResourceType.OAUTH_ACCOUNT,
                    providers=["google", "microsoft"],
                    help_text="Account for sending emails",
                ),
                ResourceRequirement(
                    key="calendar",
                    label="Calendar Account",
                    resource_type=ResourceType.OAUTH_ACCOUNT,
                    providers=["google", "microsoft"],
                ),
                ResourceRequirement(
                    key="calendar",
                    sub_key="calendar_id",
                    label="Default Calendar",
                    resource_type=ResourceType.CALENDAR_PICKER,
                    depends_on="calendar",
                    help_text="Calendar for creating events",
                ),
            ]
        """
        return []

    @classmethod
    @abstractmethod
    def get_actions(cls) -> list[ActionDefinition]:
        """
        Define available actions and their parameters.

        These are exposed to the LLM via system prompt injection.
        """
        pass

    @classmethod
    def get_action(cls, action_name: str) -> Optional[ActionDefinition]:
        """Get a specific action definition by name."""
        return next((a for a in cls.get_actions() if a.name == action_name), None)

    @classmethod
    def get_llm_instructions(cls) -> str:
        """
        Generate instructions for LLM system prompt.

        Override for custom formatting. Default format:

        ## Email Actions

        ### email:draft_new
        Create a new email draft.
        Parameters: account (required), to (required), subject (required), body (required)

        Example:
        ```xml
        <smart_action type="email" action="draft_new">
        {"account": "user@gmail.com", "to": ["recipient@example.com"], ...}
        </smart_action>
        ```
        """
        actions = cls.get_actions()
        if not actions:
            return ""

        lines = [f"## {cls.display_name}", ""]

        for action in actions:
            lines.append(f"### {cls.action_type}:{action.name}")
            lines.append(action.description)

            # Parameter list
            param_parts = []
            for p in action.params:
                req = "required" if p.required else "optional"
                param_parts.append(f"{p.name} ({req})")
            if param_parts:
                lines.append(f"Parameters: {', '.join(param_parts)}")

            # Example
            if action.examples:
                lines.append("")
                lines.append("Example:")
                lines.append("```xml")
                lines.append(
                    f'<smart_action type="{cls.action_type}" action="{action.name}">'
                )
                # Pretty-print first example
                lines.append(json.dumps(action.examples[0], indent=2))
                lines.append("</smart_action>")
                lines.append("```")

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """
        Validate plugin configuration.

        Override for custom validation logic.
        """
        return validate_config(cls.get_config_fields(), config)

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize with validated configuration.

        Args:
            config: Dict matching get_config_fields() definitions
        """
        pass

    @abstractmethod
    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """
        Execute an action.

        Args:
            action: Action name (from get_actions())
            params: Parameters provided by LLM
            context: Execution context (session, user, etc.)

        Returns:
            ActionResult with success/failure and message
        """
        pass

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """
        Validate action parameters before execution.

        Override for custom validation.
        """
        from plugin_base.common import ValidationError

        action_def = self.get_action(action)
        if not action_def:
            return ValidationResult(
                valid=False,
                errors=[ValidationError("action", f"Unknown action: {action}")],
            )

        errors = []
        for param_def in action_def.params:
            value = params.get(param_def.name)
            if param_def.required and value is None:
                errors.append(
                    ValidationError(param_def.name, "Required parameter missing")
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """
        Generate human-readable summary for approval UI.

        Override for action-specific summaries.
        """
        action_def = self.get_action(action)
        if action_def and action_def.risk == ActionRisk.DESTRUCTIVE:
            return f"⚠️ DESTRUCTIVE: {self.action_type}:{action}"
        return f"{self.action_type}:{action} with {len(params)} parameters"

    def get_action_risk(self, action: str) -> ActionRisk:
        """Get risk level for an action."""
        action_def = self.get_action(action)
        return action_def.risk if action_def else ActionRisk.HIGH

    def is_available(self) -> bool:
        """Check if plugin is properly configured and available."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test the plugin configuration."""
        return True, "OK"
