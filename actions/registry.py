"""
Action Handler Registry.

Manages registration and lookup of action handlers.
"""

import fnmatch
import logging
from typing import Optional

from .base import ActionHandler

logger = logging.getLogger(__name__)

# Global registry of action handlers
_handlers: dict[str, ActionHandler] = {}


def register_handler(handler: ActionHandler) -> None:
    """
    Register an action handler.

    Args:
        handler: The handler instance to register
    """
    action_type = handler.action_type
    if action_type in _handlers:
        logger.warning(f"Overwriting existing handler for action type: {action_type}")

    _handlers[action_type] = handler
    logger.info(
        f"Registered action handler: {action_type} "
        f"(actions: {', '.join(handler.supported_actions)})"
    )


def get_handler(action_type: str) -> Optional[ActionHandler]:
    """
    Get a handler by action type.

    Args:
        action_type: The action type (e.g., 'email')

    Returns:
        The handler instance, or None if not found.
    """
    return _handlers.get(action_type)


def list_handlers() -> dict[str, ActionHandler]:
    """
    Get all registered handlers.

    Returns:
        Dictionary mapping action types to handlers.
    """
    return dict(_handlers)


def is_action_allowed(full_action: str, allowed_patterns: list[str]) -> bool:
    """
    Check if an action is allowed based on patterns.

    Args:
        full_action: Full action like 'email:draft_new'
        allowed_patterns: List of patterns like ['email:draft_*', 'calendar:*']

    Returns:
        True if the action matches any allowed pattern.
    """
    if not allowed_patterns:
        return False

    # Special case: '*' allows everything
    if "*" in allowed_patterns:
        return True

    for pattern in allowed_patterns:
        if fnmatch.fnmatch(full_action, pattern):
            return True

    return False


def get_available_actions(
    allowed_patterns: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """
    Get all available actions, optionally filtered by allowed patterns.

    Args:
        allowed_patterns: Optional list of patterns to filter by

    Returns:
        Dictionary mapping action types to lists of allowed actions.
    """
    result = {}

    for action_type, handler in _handlers.items():
        allowed_actions = []
        for action in handler.supported_actions:
            full_action = f"{action_type}:{action}"
            if allowed_patterns is None or is_action_allowed(
                full_action, allowed_patterns
            ):
                allowed_actions.append(action)

        if allowed_actions:
            result[action_type] = allowed_actions

    return result


def get_system_prompt_for_actions(allowed_patterns: Optional[list[str]] = None) -> str:
    """
    Generate system prompt instructions for available actions.

    Args:
        allowed_patterns: Optional list of patterns to filter which actions to include

    Returns:
        String to inject into system prompt describing available actions.
    """
    available = get_available_actions(allowed_patterns)

    if not available:
        return ""

    lines = [
        "",
        "## Available Actions",
        "",
        "You can perform actions by including <smart_action> blocks in your response:",
        "",
        "```xml",
        '<smart_action type="ACTION_TYPE" action="ACTION_NAME">',
        '{"param1": "value1", "param2": "value2"}',
        "</smart_action>",
        "```",
        "",
        "Available actions:",
    ]

    for action_type, actions in available.items():
        handler = _handlers.get(action_type)
        if handler:
            instructions = handler.get_system_prompt_instructions()
            if instructions:
                lines.append(instructions)

    lines.append("")

    return "\n".join(lines)
