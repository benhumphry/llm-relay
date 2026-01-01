"""
Authentication for the admin interface.

Uses bcrypt for password hashing and session cookies for authentication.
"""

import logging
import os
import secrets
from functools import wraps
from typing import Optional

import bcrypt
from flask import g, redirect, request, session, url_for

from db import Setting, get_db_context

logger = logging.getLogger(__name__)

# Session configuration
SESSION_COOKIE_NAME = "admin_session"
SESSION_LIFETIME_HOURS = 24


def get_session_secret() -> str:
    """Get or generate the session secret key."""
    # First check environment variable
    secret = os.environ.get("ADMIN_SESSION_SECRET")
    if secret:
        return secret

    # Then check database
    with get_db_context() as db:
        setting = (
            db.query(Setting).filter(Setting.key == Setting.KEY_SESSION_SECRET).first()
        )
        if setting and setting.value:
            return setting.value

        # Generate and store a new secret
        secret = secrets.token_hex(32)
        if setting:
            setting.value = secret
        else:
            db.add(Setting(key=Setting.KEY_SESSION_SECRET, value=secret))
        db.commit()
        return secret


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def get_admin_password_hash() -> Optional[str]:
    """Get the admin password hash from the database."""
    with get_db_context() as db:
        setting = (
            db.query(Setting)
            .filter(Setting.key == Setting.KEY_ADMIN_PASSWORD_HASH)
            .first()
        )
        return setting.value if setting else None


def set_admin_password(password: str) -> None:
    """Set the admin password."""
    password_hash = hash_password(password)

    with get_db_context() as db:
        setting = (
            db.query(Setting)
            .filter(Setting.key == Setting.KEY_ADMIN_PASSWORD_HASH)
            .first()
        )
        if setting:
            setting.value = password_hash
        else:
            db.add(Setting(key=Setting.KEY_ADMIN_PASSWORD_HASH, value=password_hash))
        db.commit()

    logger.info("Admin password updated")


def init_admin_password() -> bool:
    """
    Initialize admin password from environment variable if not set.

    Returns:
        True if password was initialized, False if already set
    """
    # Check if already set
    if get_admin_password_hash():
        return False

    # Check for environment variable
    password = os.environ.get("ADMIN_PASSWORD")
    if password:
        set_admin_password(password)
        logger.info(
            "Admin password initialized from ADMIN_PASSWORD environment variable"
        )
        return True

    # Generate a random password and log it
    password = secrets.token_urlsafe(16)
    set_admin_password(password)
    logger.warning("=" * 60)
    logger.warning("ADMIN PASSWORD NOT SET - Generated random password:")
    logger.warning(f"  {password}")
    logger.warning("Set ADMIN_PASSWORD environment variable to use your own password.")
    logger.warning("=" * 60)
    return True


def authenticate(password: str) -> bool:
    """
    Authenticate with the admin password.

    Args:
        password: Password to verify

    Returns:
        True if authentication successful
    """
    password_hash = get_admin_password_hash()
    if not password_hash:
        # No password set - deny access
        logger.warning("Admin authentication attempted but no password is set")
        return False

    return verify_password(password, password_hash)


def login_user() -> None:
    """Mark the current session as authenticated."""
    session["authenticated"] = True
    session.permanent = True


def logout_user() -> None:
    """Clear the current session."""
    session.clear()


def is_authenticated() -> bool:
    """Check if the current session is authenticated."""
    return session.get("authenticated", False)


def require_auth(f):
    """
    Decorator to require authentication for a route.

    For API routes (Accept: application/json), returns 401.
    For page routes, redirects to login page.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            # Check if this is an API request
            if request.headers.get("Accept", "").startswith("application/json"):
                return {"error": "Authentication required"}, 401
            # Redirect to login for page requests
            return redirect(url_for("admin.login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def require_auth_api(f):
    """
    Decorator to require authentication for API routes.

    Always returns JSON error response if not authenticated.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            return {"error": "Authentication required"}, 401
        return f(*args, **kwargs)

    return decorated_function
