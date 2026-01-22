"""
Base classes for Smart Actions.

Defines the ActionHandler abstract base class and supporting types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an action execution."""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    INVALID = "invalid"


@dataclass
class ActionContext:
    """
    Context passed to action handlers during execution.

    Contains information about the request, user, and available resources.
    """

    # Request context
    request_id: Optional[str] = None
    session_key: Optional[str] = None  # For session-scoped caching (email lookups)
    alias_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Available OAuth accounts (by provider) - LEGACY
    # e.g., {"google": [{"id": 1, "email": "user@gmail.com"}, ...], "microsoft": [...]}
    oauth_accounts: dict[str, list[dict]] = field(default_factory=dict)

    # Available accounts by category (derived from linked document stores)
    # Keys: "email", "calendar", "tasks"
    # Each account has: store_id, name (friendly), source_type, oauth_account_id, email
    available_accounts: dict[str, list[dict]] = field(default_factory=dict)

    # Default accounts configured on the Smart Alias (fallback if LLM doesn't specify)
    # Keys: "email", "calendar", "tasks", "notification", "schedule"
    default_accounts: dict[str, dict] = field(default_factory=dict)

    # Action settings from Smart Alias
    allowed_actions: list[str] = field(default_factory=list)  # e.g., ["email:draft_*"]
    require_approval: bool = True

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActionResult:
    """
    Result of an action execution.

    Returned by ActionHandler.execute() to indicate success/failure and details.
    """

    status: ActionStatus
    action_type: str
    action: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    # For tracking
    executed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "action_type": self.action_type,
            "action": self.action,
            "message": self.message,
            "details": self.details,
            "executed_at": self.executed_at.isoformat(),
        }


@dataclass
class ParsedAction:
    """
    A parsed action from LLM output.

    Represents a single <smart_action> block extracted from response text.
    """

    action_type: str  # e.g., "email"
    action: str  # e.g., "draft_new"
    params: dict[str, Any]  # The JSON payload
    raw_block: str  # Original XML block for reference

    @property
    def full_action(self) -> str:
        """Return full action identifier like 'email:draft_new'."""
        return f"{self.action_type}:{self.action}"


class ActionHandler(ABC):
    """
    Abstract base class for action handlers.

    Each handler manages a specific action type (email, calendar, task, etc.)
    and implements the logic to validate and execute actions.
    """

    @property
    @abstractmethod
    def action_type(self) -> str:
        """
        The action type this handler manages.

        Examples: 'email', 'calendar', 'task', 'slack'
        """
        pass

    @property
    @abstractmethod
    def supported_actions(self) -> list[str]:
        """
        List of actions this handler supports.

        Examples for email: ['draft_new', 'draft_reply', 'draft_forward', 'send']
        """
        pass

    @property
    def requires_oauth(self) -> bool:
        """Whether this handler requires OAuth authentication."""
        return True

    @property
    def oauth_provider(self) -> Optional[str]:
        """
        The OAuth provider required (if any).

        Examples: 'google', 'microsoft', None
        """
        return None

    @abstractmethod
    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """
        Validate action parameters before execution.

        Args:
            action: The specific action (e.g., 'draft_new')
            params: The action parameters from the LLM
            context: Execution context

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message should be empty string.
        """
        pass

    @abstractmethod
    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """
        Execute the action.

        Args:
            action: The specific action (e.g., 'draft_new')
            params: The validated action parameters
            context: Execution context

        Returns:
            ActionResult indicating success/failure and details.
        """
        pass

    @abstractmethod
    def get_approval_summary(self, action: str, params: dict) -> str:
        """
        Generate a human-readable summary for approval UI.

        Args:
            action: The specific action
            params: The action parameters

        Returns:
            A concise description like "Create email draft to john@example.com"
        """
        pass

    def get_system_prompt_instructions(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """
        Get instructions to inject into the system prompt.

        Override to provide specific instructions for LLMs about
        how to use this handler's actions.

        Args:
            available_accounts: Optional dict of available accounts by category.
                Keys: "email", "calendar", "tasks"
                Each account has: {"id", "type", "provider", "email", "name"}
        """
        actions = ", ".join(self.supported_actions)
        return f"- {self.action_type}: {actions}"
