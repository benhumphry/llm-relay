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


def get_system_prompt_for_actions(
    allowed_patterns: Optional[list[str]] = None,
    available_accounts: Optional[dict[str, list[dict]]] = None,
) -> str:
    """
    Generate system prompt instructions for available actions.

    Args:
        allowed_patterns: Optional list of patterns to filter which actions to include
        available_accounts: Optional dict of available accounts by category for dynamic instructions.
            Keys: "email", "calendar", "tasks"
            Each account has: {"id", "type", "provider", "email", "name"}

    Returns:
        String to inject into system prompt describing available actions.
    """
    available = get_available_actions(allowed_patterns)

    if not available:
        return ""

    lines = [
        "",
        "## Available Actions - YOU HAVE THESE CAPABILITIES",
        "",
        "You ARE ABLE to perform the following actions on behalf of the user. These are real capabilities you have access to.",
        "When the user asks you to perform one of these actions, YOU CAN AND SHOULD DO IT by including a <smart_action> block.",
        "",
        "Do NOT claim you cannot perform these actions - you CAN. The system will execute them for you.",
        "",
        "CRITICAL FORMAT REQUIREMENT:",
        "You MUST use the exact XML format shown below. Do NOT use raw JSON, markdown code blocks, or any other format.",
        "Actions in incorrect formats will be ignored and the user's request will fail.",
        "",
        "CORRECT format (you MUST use this exact structure):",
        '<smart_action type="ACTION_TYPE" action="ACTION_NAME">',
        '{"param1": "value1", "param2": "value2"}',
        "</smart_action>",
        "",
        "WRONG formats (do NOT use these):",
        '- Raw JSON like {"action": "..."}',
        "- Markdown code blocks with json",
        "- Any format without the <smart_action> tags",
        "",
        "IMPORTANT RULES:",
        "- When the user asks you to perform an action (send email, create event, delete task, etc.), USE the action - do not say you cannot",
        "- Do NOT use actions for information queries - just answer the question directly",
        "- Only use the exact action types listed below - do not invent new types",
        "",
        "CRITICAL - RESPONSE FORMAT:",
        "You MUST write a conversational response to the user BEFORE any action block.",
        "The action block is hidden from the user, so they only see your text response.",
        "NEVER start your response with <smart_action> - always write text first!",
        "",
        "CORRECT: 'I'll send that notification now.' followed by <smart_action>...</smart_action>",
        "WRONG: Starting directly with <smart_action> (user sees nothing)",
        "",
    ]

    # List exact valid type values
    valid_types = list(available.keys())
    lines.append(
        f"VALID type VALUES (use EXACTLY one of these): {', '.join(valid_types)}"
    )
    lines.append("")
    lines.append("Available actions:")

    for action_type, actions in available.items():
        handler = _handlers.get(action_type)
        if handler:
            instructions = handler.get_system_prompt_instructions(available_accounts)
            if instructions:
                lines.append(instructions)

    lines.append("")

    return "\n".join(lines)
