"""
Action Parser for Smart Actions.

Detects and parses <smart_action> blocks from LLM responses.
"""

import json
import logging
import re
from typing import Iterator

from .base import ParsedAction

logger = logging.getLogger(__name__)

# Regex to match <smart_action type="..." action="...">...</smart_action>
# Supports both single-line and multi-line JSON content
ACTION_PATTERN = re.compile(
    r'<smart_action\s+type=["\']([^"\']+)["\']\s+action=["\']([^"\']+)["\']>\s*'
    r"(.*?)"
    r"</smart_action>",
    re.DOTALL | re.IGNORECASE,
)


def parse_actions(text: str) -> list[ParsedAction]:
    """
    Parse all <smart_action> blocks from text.

    Args:
        text: The full LLM response text

    Returns:
        List of ParsedAction objects. Invalid blocks are logged but skipped.
    """
    actions = []

    for match in ACTION_PATTERN.finditer(text):
        action_type = match.group(1).strip().lower()
        action = match.group(2).strip().lower()
        json_content = match.group(3).strip()
        raw_block = match.group(0)

        try:
            params = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in smart_action block ({action_type}:{action}): {e}"
            )
            logger.debug(f"Raw JSON content: {json_content[:200]}")
            continue

        if not isinstance(params, dict):
            logger.warning(
                f"smart_action params must be a dict, got {type(params).__name__}"
            )
            continue

        actions.append(
            ParsedAction(
                action_type=action_type,
                action=action,
                params=params,
                raw_block=raw_block,
            )
        )

    if actions:
        logger.info(f"Parsed {len(actions)} smart_action blocks from response")

    return actions


def strip_actions(text: str) -> str:
    """
    Remove all <smart_action> blocks from text.

    Useful for showing clean response to user after extracting actions.

    Args:
        text: The full LLM response text

    Returns:
        Text with all smart_action blocks removed.
    """
    return ACTION_PATTERN.sub("", text).strip()


def has_actions(text: str) -> bool:
    """
    Quick check if text contains any <smart_action> blocks.

    More efficient than parse_actions() when you just need to know
    if actions exist.

    Args:
        text: The text to check

    Returns:
        True if at least one smart_action block is found.
    """
    return bool(ACTION_PATTERN.search(text))
