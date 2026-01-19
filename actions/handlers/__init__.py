"""
Action Handlers for Smart Actions.

Each handler implements a specific action type (email, calendar, etc.).
"""

from .email import EmailActionHandler

__all__ = ["EmailActionHandler"]
