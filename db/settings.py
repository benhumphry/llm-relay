"""
Settings CRUD operations.

Simple key-value storage for configuration and temporary state.
"""

import logging
from typing import Optional

from sqlalchemy import select

from .connection import get_db_context
from .models import Setting

logger = logging.getLogger(__name__)


def get_setting(key: str) -> Optional[str]:
    """Get a setting value by key."""
    with get_db_context() as db:
        result = db.execute(
            select(Setting).where(Setting.key == key)
        ).scalar_one_or_none()
        return result.value if result else None


def set_setting(key: str, value: str) -> None:
    """Set a setting value. Creates or updates the setting."""
    with get_db_context() as db:
        existing = db.execute(
            select(Setting).where(Setting.key == key)
        ).scalar_one_or_none()

        if existing:
            existing.value = value
        else:
            setting = Setting(key=key, value=value)
            db.add(setting)

        db.commit()


def delete_setting(key: str) -> bool:
    """Delete a setting by key. Returns True if deleted, False if not found."""
    with get_db_context() as db:
        existing = db.execute(
            select(Setting).where(Setting.key == key)
        ).scalar_one_or_none()

        if existing:
            db.delete(existing)
            db.commit()
            return True
        return False


def get_all_settings() -> dict[str, str]:
    """Get all settings as a dictionary."""
    with get_db_context() as db:
        results = db.execute(select(Setting)).scalars().all()
        return {s.key: s.value for s in results if s.value is not None}
