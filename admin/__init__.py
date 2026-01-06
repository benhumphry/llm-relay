"""
Admin module for LLM Relay.

Provides web UI and API for managing providers, models, and settings.
"""

from .app import create_admin_blueprint
from .auth import init_admin_password

__all__ = [
    "create_admin_blueprint",
    "init_admin_password",
]
