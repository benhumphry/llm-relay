"""
Smart Actions Framework for LLM Relay.

Enables LLMs to take actions (create drafts, send emails, create tasks, etc.)
through structured action blocks in their responses.
"""

from .base import ActionContext, ActionHandler, ActionResult
from .executor import execute_actions, get_action_instructions_for_alias
from .loader import load_action_handlers
from .parser import has_actions, parse_actions, strip_actions
from .registry import (
    get_handler,
    get_system_prompt_for_actions,
    list_handlers,
    register_handler,
)

__all__ = [
    "ActionHandler",
    "ActionContext",
    "ActionResult",
    "parse_actions",
    "has_actions",
    "strip_actions",
    "register_handler",
    "get_handler",
    "list_handlers",
    "get_system_prompt_for_actions",
    "load_action_handlers",
    "execute_actions",
    "get_action_instructions_for_alias",
]
