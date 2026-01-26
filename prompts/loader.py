"""
Prompt Library Loader - YAML loading with Jinja2 rendering.

Loads prompt templates from defaults/ directory, with overrides/ taking precedence.
Templates are rendered using Jinja2 for variable substitution, conditionals, and loops.
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment, TemplateNotFound

from prompts.filters import register_filters

logger = logging.getLogger(__name__)


class YAMLTemplateLoader(BaseLoader):
    """Custom Jinja2 loader that loads templates from cached YAML configs."""

    def __init__(self, library: "PromptLibrary"):
        self.library = library

    def get_source(self, environment, template):
        """
        Get template source from YAML config.

        Template name format: "category/name:template_key"
        """
        try:
            parts = template.rsplit(":", 1)
            if len(parts) == 2:
                path, key = parts
            else:
                path, key = parts[0], "user_template"

            category, name = path.split("/", 1)
            config = self.library.get_config(category, name)
            source = config.get(key, "")

            return source, template, lambda: True
        except Exception:
            raise TemplateNotFound(template)


class PromptLibrary:
    """
    Manages prompt templates and keyword mappings.

    Loads from defaults/, with overrides/ taking precedence.
    Templates are rendered using Jinja2.

    Directory structure:
        prompts/
        ├── defaults/           # Built-in prompts
        │   ├── designators/
        │   ├── context/
        │   ├── actions/
        │   └── keywords/
        └── overrides/          # User overrides (gitignored)
    """

    def __init__(
        self,
        defaults_dir: str | Path | None = None,
        overrides_dir: str | Path | None = None,
        locale: str = "en",
    ):
        # Find the prompts directory relative to this file
        base_dir = Path(__file__).parent

        self.defaults_dir = (
            Path(defaults_dir) if defaults_dir else base_dir / "defaults"
        )
        self.overrides_dir = (
            Path(overrides_dir) if overrides_dir else base_dir / "overrides"
        )
        self.locale = locale
        self._cache: dict[str, dict] = {}

        # Set up Jinja2 environment
        self._env = Environment(
            loader=YAMLTemplateLoader(self),
            autoescape=False,  # We're generating prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )
        register_filters(self._env)

        # Load all configs
        self._load_all()

    def _load_all(self):
        """Load all YAML files from defaults and overrides."""
        # Load defaults first
        if self.defaults_dir.exists():
            self._load_directory(self.defaults_dir)

        # Then load overrides (will replace defaults)
        if self.overrides_dir.exists():
            self._load_directory(self.overrides_dir)

        logger.info(f"Loaded {len(self._cache)} prompt configs")

    def _load_directory(self, base_dir: Path):
        """Recursively load YAML files from a directory."""
        for yaml_file in base_dir.rglob("*.yaml"):
            # Get relative path as cache key
            rel_path = yaml_file.relative_to(base_dir)
            # Remove .yaml extension and convert to category/name format
            key = str(rel_path.with_suffix("")).replace("\\", "/")

            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                self._cache[key] = config
                logger.debug(f"Loaded prompt config: {key}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

    def get_config(self, category: str, name: str) -> dict[str, Any]:
        """
        Get raw config dict for a prompt.

        Args:
            category: Prompt category (e.g., "designators", "context")
            name: Prompt name (e.g., "live", "router")

        Returns:
            Config dictionary from YAML file, or empty dict if not found
        """
        key = f"{category}/{name}"
        return self._cache.get(key, {})

    def render(
        self,
        category: str,
        name: str,
        template_key: str = "user_template",
        **variables,
    ) -> str:
        """
        Render a Jinja2 template from a prompt config.

        Args:
            category: Prompt category (e.g., "designators", "context")
            name: Prompt name (e.g., "live", "router")
            template_key: Key in the YAML for the template (default: "user_template")
            **variables: Variables to pass to the template

        Returns:
            Rendered template string
        """
        config = self.get_config(category, name)
        template_str = config.get(template_key, "")

        if not template_str:
            logger.warning(f"No template '{template_key}' found in {category}/{name}")
            return ""

        try:
            template = self._env.from_string(template_str)

            # Include other config values as variables (e.g., section_headers)
            # But don't override explicitly passed variables
            context = {**config, **variables}

            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render {category}/{name}:{template_key}: {e}")
            return f"[Template error: {e}]"

    def get_keywords(self, category: str, name: str) -> dict[str, list[str]]:
        """
        Get keyword mappings from a keywords config.

        Filters out non-keyword entries (patterns, metadata fields).

        Args:
            category: Should be "keywords"
            name: Keyword set name (e.g., "dates", "query_types")

        Returns:
            Dict mapping canonical keywords to lists of synonyms
        """
        config = self.get_config(category, name)
        # Filter out non-keyword entries
        excluded_keys = {"patterns", "name", "description", "variables"}
        return {
            k: v
            for k, v in config.items()
            if isinstance(v, list) and k not in excluded_keys
        }

    def match_keyword(
        self, category: str, name: str, value: str
    ) -> tuple[str | None, dict | None]:
        """
        Match input to canonical keyword using synonyms.

        First checks direct synonym matches, then regex patterns.

        Args:
            category: Should be "keywords"
            name: Keyword set name (e.g., "dates")
            value: Value to match

        Returns:
            (canonical_keyword, match_groups) or (None, None) if no match
            match_groups contains regex capture groups if matched via pattern
        """
        config = self.get_config(category, name)
        value_lower = value.lower().strip()

        # Check direct synonyms first
        excluded_keys = {"patterns", "name", "description", "variables"}
        for canonical, synonyms in config.items():
            if canonical in excluded_keys:
                continue
            if isinstance(synonyms, list):
                if value_lower in [s.lower() for s in synonyms]:
                    return canonical, None

        # Check regex patterns
        patterns = config.get("patterns", [])
        for p in patterns:
            pattern = p.get("pattern", "")
            if not pattern:
                continue
            match = re.match(pattern, value_lower, re.IGNORECASE)
            if match:
                # Return named groups if available, else indexed groups
                groups = match.groupdict() or {
                    i: g for i, g in enumerate(match.groups())
                }
                return p.get("handler", "pattern"), groups

        return None, None

    def list_configs(self, category: str | None = None) -> list[str]:
        """
        List available config keys.

        Args:
            category: Optional category to filter by

        Returns:
            List of config keys (e.g., ["designators/live", "keywords/dates"])
        """
        if category:
            return [k for k in self._cache.keys() if k.startswith(f"{category}/")]
        return list(self._cache.keys())
