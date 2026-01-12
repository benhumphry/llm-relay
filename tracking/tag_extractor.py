"""
Tag extraction for usage attribution.

Extracts usage tags from requests using multiple strategies:
1. X-Proxy-Tag header (supports comma-separated tags)
2. Model name suffix (model@tag or model@tag1,tag2)
3. Authorization Bearer token (for apps like Open WebUI)
4. @relay[tag:...] commands embedded in message content

Multiple tags can be specified with commas: "alice,project-x"
Requests without tags will have an empty tag string.

@relay command format:
    @relay[command:value]
    @relay[command:key=value,key2="quoted value"]

Supported commands:
    - tag: Add tags to the request (e.g., @relay[tag:cached,analytics])
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from flask import Request

logger = logging.getLogger(__name__)

# Regex to match @relay[command:value] patterns
# Captures: command name and the value/args portion
RELAY_COMMAND_PATTERN = re.compile(r"@relay\[(\w+):([^\]]*)\]", re.IGNORECASE)


@dataclass
class RelayCommand:
    """Represents a parsed @relay command."""

    command: str
    raw_value: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.command = self.command.lower()


@dataclass
class RelayCommandResult:
    """Result of extracting @relay commands from content."""

    commands: list[RelayCommand]
    cleaned_content: str
    tags: list[str]  # Tags extracted from tag commands


def parse_relay_commands(content: str) -> RelayCommandResult:
    """
    Parse and extract @relay commands from message content.

    Args:
        content: Message content that may contain @relay commands

    Returns:
        RelayCommandResult with extracted commands, cleaned content, and tags
    """
    commands = []
    tags = []

    # Find all @relay commands
    for match in RELAY_COMMAND_PATTERN.finditer(content):
        command_name = match.group(1).lower()
        raw_value = match.group(2).strip()

        cmd = RelayCommand(command=command_name, raw_value=raw_value)

        # Parse command-specific values
        if command_name == "tag":
            # Tag command: comma-separated tag names
            cmd_tags = [t.strip() for t in raw_value.split(",") if t.strip()]
            tags.extend(cmd_tags)
            cmd.args["tags"] = cmd_tags
        else:
            # Future commands: parse key=value pairs
            cmd.args = _parse_command_args(raw_value)

        commands.append(cmd)
        logger.debug(f"Parsed @relay command: {command_name}:{raw_value}")

    # Strip commands from content
    cleaned_content = RELAY_COMMAND_PATTERN.sub("", content).strip()

    # Clean up extra whitespace that may result from stripping
    cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)
    cleaned_content = re.sub(r"  +", " ", cleaned_content)

    return RelayCommandResult(
        commands=commands, cleaned_content=cleaned_content, tags=tags
    )


def _parse_command_args(value: str) -> dict[str, Any]:
    """
    Parse key=value pairs from command arguments.

    Handles:
        - bare values: "value" -> {"_value": "value"}
        - key=value: "key=value" -> {"key": "value"}
        - quoted values: 'key="value with spaces"' -> {"key": "value with spaces"}
        - multiple: "a=1,b=2" -> {"a": "1", "b": "2"}

    Args:
        value: Raw argument string

    Returns:
        Dictionary of parsed arguments
    """
    args = {}

    if not value:
        return args

    # Simple case: no = means it's a bare value
    if "=" not in value:
        args["_value"] = value
        return args

    # Parse key=value pairs (handling quoted values)
    # This is a simple parser - could be enhanced for complex cases
    current_key = ""
    current_value = ""
    in_quotes = False
    quote_char = None
    parsing_key = True

    i = 0
    while i < len(value):
        char = value[i]

        if parsing_key:
            if char == "=":
                parsing_key = False
            elif char == ",":
                # Key without value
                if current_key.strip():
                    args[current_key.strip()] = True
                current_key = ""
            else:
                current_key += char
        else:
            if not in_quotes:
                if char in ('"', "'"):
                    in_quotes = True
                    quote_char = char
                elif char == ",":
                    # End of this key=value pair
                    if current_key.strip():
                        args[current_key.strip()] = current_value.strip()
                    current_key = ""
                    current_value = ""
                    parsing_key = True
                else:
                    current_value += char
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_value += char

        i += 1

    # Handle last pair
    if current_key.strip():
        if parsing_key:
            args[current_key.strip()] = True
        else:
            args[current_key.strip()] = current_value.strip()

    return args


def extract_relay_commands_from_messages(
    messages: list[dict],
) -> tuple[list[dict], list[RelayCommand], list[str]]:
    """
    Extract @relay commands from all messages and return cleaned messages.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Tuple of (cleaned_messages, all_commands, all_tags)
    """
    all_commands = []
    all_tags = []
    cleaned_messages = []

    for msg in messages:
        content = msg.get("content", "")

        # Handle string content
        if isinstance(content, str):
            result = parse_relay_commands(content)
            all_commands.extend(result.commands)
            all_tags.extend(result.tags)

            cleaned_msg = msg.copy()
            cleaned_msg["content"] = result.cleaned_content
            cleaned_messages.append(cleaned_msg)

        # Handle multimodal content (list of content parts)
        elif isinstance(content, list):
            cleaned_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    result = parse_relay_commands(text)
                    all_commands.extend(result.commands)
                    all_tags.extend(result.tags)

                    cleaned_part = part.copy()
                    cleaned_part["text"] = result.cleaned_content
                    cleaned_parts.append(cleaned_part)
                else:
                    # Keep non-text parts as-is (images, etc.)
                    cleaned_parts.append(part)

            cleaned_msg = msg.copy()
            cleaned_msg["content"] = cleaned_parts
            cleaned_messages.append(cleaned_msg)
        else:
            # Unknown content type, keep as-is
            cleaned_messages.append(msg)

    # Deduplicate tags
    unique_tags = list(dict.fromkeys(all_tags))

    return cleaned_messages, all_commands, unique_tags


def extract_tag(request: Request, model_name: str) -> tuple[str, str]:
    """
    Extract tag(s) and clean model name from request.

    Priority:
    1. X-Proxy-Tag header - explicit tag specification
    2. Model name suffix - model@tag format
    3. Authorization Bearer token - use token value as tag

    Multiple tags can be comma-separated: "alice,project-x"
    Tags are normalized (trimmed, deduplicated, sorted) and stored as comma-separated.
    Requests without explicit tags will have an empty tag string.

    Args:
        request: Flask request object
        model_name: Original model name from request

    Returns:
        Tuple of (tag_string, cleaned_model_name)
        tag_string may contain multiple comma-separated tags, or empty string if no tags
    """
    raw_tag = None
    clean_model = model_name

    # Priority 1: X-Proxy-Tag header
    header_tag = request.headers.get("X-Proxy-Tag")
    if header_tag:
        raw_tag = header_tag

    # Priority 2: Model name suffix (model@tag)
    elif "@" in model_name:
        parts = model_name.rsplit("@", 1)
        if len(parts) == 2 and parts[1]:
            clean_model = parts[0].strip()
            tag_part = parts[1].strip()
            # Handle :version suffix after tag (e.g., model@tag:latest)
            if ":" in tag_part:
                tag_part = tag_part.split(":")[0]
            raw_tag = tag_part

    # Priority 3: Authorization Bearer token as tag
    # Apps like Open WebUI send API key as "Authorization: Bearer <key>"
    # We use the key value directly as the tag
    elif auth_header := request.headers.get("Authorization"):
        # Handle "Bearer <token>" format
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                raw_tag = token
        # Handle plain token (no Bearer prefix)
        elif auth_header.strip():
            raw_tag = auth_header.strip()

    # Normalize tags if we found any
    if raw_tag:
        normalized = normalize_tags(raw_tag)
        if normalized:
            return normalized, clean_model

    # No tag found - return empty string
    return "", clean_model


def normalize_tags(tag_string: str) -> str:
    """
    Normalize a tag string: trim whitespace, deduplicate, sort.

    Args:
        tag_string: Raw tag string, possibly comma-separated

    Returns:
        Normalized comma-separated tag string, or empty string if no valid tags
    """
    # Split on comma, trim whitespace, filter empty
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]
    # Deduplicate while preserving order, then sort for consistency
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    unique_tags.sort()
    return ",".join(unique_tags)
