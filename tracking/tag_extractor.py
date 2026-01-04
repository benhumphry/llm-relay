"""
Tag extraction for usage attribution.

Extracts usage tags from requests using multiple strategies:
1. X-Proxy-Tag header
2. Model name suffix (model@tag)
3. Authorization Bearer token (for apps like Open WebUI)
4. Default tag from settings
"""

from flask import Request

from db import Setting, get_db_context


def extract_tag(request: Request, model_name: str) -> tuple[str, str]:
    """
    Extract tag and clean model name from request.

    Priority:
    1. X-Proxy-Tag header - explicit tag specification
    2. Model name suffix - model@tag format
    3. Authorization Bearer token - use token value as tag
    4. Default tag from settings

    Args:
        request: Flask request object
        model_name: Original model name from request

    Returns:
        Tuple of (tag, cleaned_model_name)
    """
    # Priority 1: X-Proxy-Tag header
    header_tag = request.headers.get("X-Proxy-Tag")
    if header_tag:
        return header_tag.strip(), model_name

    # Priority 2: Model name suffix (model@tag)
    if "@" in model_name:
        parts = model_name.rsplit("@", 1)
        if len(parts) == 2 and parts[1]:
            model_part = parts[0].strip()
            tag_part = parts[1].strip()
            # Handle :version suffix after tag (e.g., model@tag:latest)
            if ":" in tag_part:
                tag_part = tag_part.split(":")[0]
            return tag_part, model_part

    # Priority 3: Authorization Bearer token as tag
    # Apps like Open WebUI send API key as "Authorization: Bearer <key>"
    # We use the key value directly as the tag
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # Handle "Bearer <token>" format
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                return token, model_name
        # Handle plain token (no Bearer prefix)
        elif auth_header.strip():
            return auth_header.strip(), model_name

    # Priority 4: Default tag from settings
    default_tag = get_default_tag()
    return default_tag, model_name


def get_default_tag() -> str:
    """Get the default tag from settings."""
    try:
        with get_db_context() as db:
            setting = (
                db.query(Setting).filter(Setting.key == Setting.KEY_DEFAULT_TAG).first()
            )
            if setting and setting.value:
                return setting.value
    except Exception:
        pass
    return "default"
