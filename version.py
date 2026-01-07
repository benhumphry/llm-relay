"""Version information for LLM Relay."""

from pathlib import Path

_VERSION_FILE = Path(__file__).parent / "VERSION"


def get_version() -> str:
    """Read version from VERSION file."""
    try:
        return _VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        return "unknown"


VERSION = get_version()
