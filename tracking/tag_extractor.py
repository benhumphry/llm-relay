"""
Tag extraction for usage attribution.

Extracts usage tags from requests using multiple strategies:
1. X-Proxy-Tag header (supports comma-separated tags)
2. Model name suffix (model@tag or model@tag1,tag2)
3. Authorization Bearer token (for apps like Open WebUI)

Multiple tags can be specified with commas: "alice,project-x"
Requests without tags will have an empty tag string.
"""

from flask import Request


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
