"""
Prompt Library - Centralized prompt and keyword management.

This module provides a structured way to manage:
- Designator prompts (router, RAG, web, live)
- Context injection templates
- Action instructions
- Keyword synonyms (dates, query types, etc.)

Templates use Jinja2 for variable substitution, conditionals, and loops.

Usage:
    from prompts import render, get_config, match_keyword

    # Render a template
    prompt = render("designators", "live",
        current_time=datetime.now(),
        query="what tasks do I have?",
        task_accounts=[...],
    )

    # Get raw config
    config = get_config("designators", "live")
    system_prompt = config.get("system_prompt", "")

    # Match keywords
    canonical, groups = match_keyword("keywords", "dates", "this weekend")
    # canonical = "weekend", groups = None
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global singleton
_library: "PromptLibrary | None" = None


def get_library() -> "PromptLibrary":
    """Get the global PromptLibrary instance."""
    global _library
    if _library is None:
        from prompts.loader import PromptLibrary

        _library = PromptLibrary()
    return _library


def render(
    category: str, name: str, template_key: str = "user_template", **variables
) -> str:
    """
    Render a prompt template with variables.

    Args:
        category: Prompt category (e.g., "designators", "context", "keywords")
        name: Prompt name (e.g., "live", "router", "dates")
        template_key: Key in the YAML for the template (default: "user_template")
        **variables: Variables to pass to the Jinja2 template

    Returns:
        Rendered template string
    """
    return get_library().render(category, name, template_key, **variables)


def get_config(category: str, name: str) -> dict[str, Any]:
    """
    Get raw config dict for a prompt.

    Useful for accessing non-template fields like system_prompt, examples, etc.

    Args:
        category: Prompt category
        name: Prompt name

    Returns:
        Config dictionary from YAML file
    """
    return get_library().get_config(category, name)


def get_keywords(category: str, name: str) -> dict[str, list[str]]:
    """
    Get keyword mappings from a keywords config.

    Args:
        category: Should be "keywords"
        name: Keyword set name (e.g., "dates", "query_types")

    Returns:
        Dict mapping canonical keywords to lists of synonyms
    """
    return get_library().get_keywords(category, name)


def match_keyword(
    category: str, name: str, value: str
) -> tuple[str | None, dict | None]:
    """
    Match a value to its canonical keyword using synonyms.

    First checks direct synonym matches, then regex patterns.

    Args:
        category: Should be "keywords"
        name: Keyword set name (e.g., "dates")
        value: Value to match

    Returns:
        (canonical_keyword, match_groups) or (None, None) if no match
        match_groups contains regex capture groups if matched via pattern
    """
    return get_library().match_keyword(category, name, value)


def reload():
    """Reload all prompts from disk (for hot-reload)."""
    global _library
    from prompts.loader import PromptLibrary

    _library = PromptLibrary()
    logger.info("Prompt library reloaded")
