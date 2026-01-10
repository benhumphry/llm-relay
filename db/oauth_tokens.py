"""
CRUD operations for OAuth tokens.

Handles storage and retrieval of OAuth tokens for external service integrations.
Tokens are encrypted at rest using Fernet symmetric encryption.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from cryptography.fernet import Fernet
from sqlalchemy import select

from .connection import get_db_context
from .models import OAuthToken

logger = logging.getLogger(__name__)

# Encryption key - generated once and stored in env
# If not set, generates a new one (tokens won't survive restarts without it)
_ENCRYPTION_KEY: Optional[bytes] = None


def _get_encryption_key() -> bytes:
    """Get or generate the encryption key for token storage."""
    global _ENCRYPTION_KEY
    if _ENCRYPTION_KEY is None:
        key_str = os.environ.get("OAUTH_ENCRYPTION_KEY")
        if key_str:
            _ENCRYPTION_KEY = key_str.encode()
        else:
            # Generate a new key - warn that tokens won't persist
            logger.warning(
                "OAUTH_ENCRYPTION_KEY not set - generating temporary key. "
                "Set this env var to persist OAuth tokens across restarts."
            )
            _ENCRYPTION_KEY = Fernet.generate_key()
    return _ENCRYPTION_KEY


def _encrypt(data: str) -> str:
    """Encrypt a string using Fernet."""
    f = Fernet(_get_encryption_key())
    return f.encrypt(data.encode()).decode()


def _decrypt(data: str) -> str:
    """Decrypt a string using Fernet."""
    f = Fernet(_get_encryption_key())
    return f.decrypt(data.encode()).decode()


def store_oauth_token(
    provider: str,
    account_email: str,
    token_data: dict,
    scopes: list[str],
    account_name: Optional[str] = None,
) -> int:
    """
    Store or update an OAuth token.

    Args:
        provider: Provider name (e.g., "google")
        account_email: User's email for this account
        token_data: Token dictionary containing access_token, refresh_token, etc.
        scopes: List of granted scopes
        account_name: Optional display name

    Returns:
        The ID of the created or updated token
    """
    with get_db_context() as session:
        # Check if token already exists for this provider/email
        stmt = select(OAuthToken).where(
            OAuthToken.provider == provider,
            OAuthToken.account_email == account_email,
        )
        existing = session.execute(stmt).scalar_one_or_none()

        encrypted_data = _encrypt(json.dumps(token_data))

        if existing:
            existing.token_data_encrypted = encrypted_data
            existing.scopes = scopes
            existing.account_name = account_name or existing.account_name
            existing.is_valid = True
            existing.last_refreshed = datetime.utcnow()
            session.commit()
            logger.info(f"Updated OAuth token for {provider}/{account_email}")
            return existing.id
        else:
            token = OAuthToken(
                provider=provider,
                account_email=account_email,
                account_name=account_name,
                token_data_encrypted=encrypted_data,
                scopes=scopes,
                is_valid=True,
            )
            session.add(token)
            session.commit()
            session.refresh(token)
            logger.info(f"Stored new OAuth token for {provider}/{account_email}")
            return token.id


def get_oauth_token(provider: str, account_email: str) -> Optional[dict]:
    """
    Get decrypted token data for a provider/email.

    Args:
        provider: Provider name
        account_email: User's email

    Returns:
        Decrypted token dictionary or None if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(
            OAuthToken.provider == provider,
            OAuthToken.account_email == account_email,
            OAuthToken.is_valid == True,
        )
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return None

        try:
            token_data = json.loads(_decrypt(token.token_data_encrypted))
            # Update last_used
            token.last_used = datetime.utcnow()
            session.commit()
            return token_data
        except Exception as e:
            logger.error(f"Failed to decrypt token for {provider}/{account_email}: {e}")
            return None


def get_oauth_token_by_id(token_id: int) -> Optional[dict]:
    """
    Get decrypted token data by ID.

    Args:
        token_id: Token ID

    Returns:
        Decrypted token dictionary or None if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(
            OAuthToken.id == token_id,
            OAuthToken.is_valid == True,
        )
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return None

        try:
            token_data = json.loads(_decrypt(token.token_data_encrypted))
            token.last_used = datetime.utcnow()
            session.commit()
            return token_data
        except Exception as e:
            logger.error(f"Failed to decrypt token {token_id}: {e}")
            return None


def get_oauth_token_info(token_id: int) -> Optional[dict]:
    """
    Get OAuth token metadata (not decrypted token data) by ID.

    Args:
        token_id: Token ID

    Returns:
        Token metadata dictionary or None if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(OAuthToken.id == token_id)
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return None

        return token.to_dict()


def list_oauth_tokens(provider: Optional[str] = None) -> list[dict]:
    """
    List all OAuth tokens (metadata only, not decrypted token data).

    Args:
        provider: Optional filter by provider

    Returns:
        List of token metadata dictionaries
    """
    with get_db_context() as session:
        stmt = select(OAuthToken)
        if provider:
            stmt = stmt.where(OAuthToken.provider == provider)
        stmt = stmt.order_by(OAuthToken.provider, OAuthToken.account_email)

        tokens = session.execute(stmt).scalars().all()
        return [t.to_dict() for t in tokens]


def delete_oauth_token(token_id: int) -> bool:
    """
    Delete an OAuth token.

    Args:
        token_id: Token ID to delete

    Returns:
        True if deleted, False if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(OAuthToken.id == token_id)
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return False

        session.delete(token)
        session.commit()
        logger.info(
            f"Deleted OAuth token {token_id} ({token.provider}/{token.account_email})"
        )
        return True


def invalidate_oauth_token(token_id: int) -> bool:
    """
    Mark an OAuth token as invalid (e.g., when refresh fails).

    Args:
        token_id: Token ID

    Returns:
        True if updated, False if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(OAuthToken.id == token_id)
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return False

        token.is_valid = False
        session.commit()
        logger.info(f"Invalidated OAuth token {token_id}")
        return True


def update_oauth_token_data(token_id: int, token_data: dict) -> bool:
    """
    Update the token data (e.g., after refresh).

    Args:
        token_id: Token ID
        token_data: New token dictionary

    Returns:
        True if updated, False if not found
    """
    with get_db_context() as session:
        stmt = select(OAuthToken).where(OAuthToken.id == token_id)
        token = session.execute(stmt).scalar_one_or_none()

        if not token:
            return False

        token.token_data_encrypted = _encrypt(json.dumps(token_data))
        token.last_refreshed = datetime.utcnow()
        session.commit()
        logger.info(f"Updated token data for {token_id}")
        return True
