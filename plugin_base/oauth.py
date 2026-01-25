"""
OAuth mixin for plugins that use OAuth authentication.

Provides common token refresh and authenticated request methods.
"""

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OAuthMixin:
    """
    Mixin for plugins that use OAuth authentication.

    Provides common token refresh and authenticated request methods.

    OAuth can be configured in two ways:
    1. Config-based: Set oauth_account_id in plugin config during __init__
    2. Context-based: Pass default_accounts in ActionContext at execute time

    For action plugins, context-based is preferred as it allows the Smart Alias
    to specify which account to use without requiring plugin configuration.

    Usage (config-based):
        class MyPlugin(OAuthMixin, PluginActionHandler):
            def __init__(self, config: dict):
                self.oauth_account_id = config.get("oauth_account_id")
                self.oauth_provider = "google"
                if self.oauth_account_id:
                    self._init_oauth_client()

    Usage (context-based):
        class MyPlugin(OAuthMixin, PluginActionHandler):
            def __init__(self, config: dict):
                self.oauth_account_id = config.get("oauth_account_id")
                self.oauth_provider = "google"

            def execute(self, action, params, context):
                # Configure from context if not already set
                self.configure_oauth_from_context(context, "email")
                response = self.oauth_get("https://api.example.com/data")
                ...
    """

    oauth_account_id: Optional[int] = None
    oauth_provider: str = "google"
    _oauth_client: Optional[httpx.Client] = None
    _access_token: Optional[str] = None
    _token_expires_at: float = 0

    def configure_oauth_from_context(
        self, context, account_key: str, force: bool = False
    ) -> bool:
        """
        Configure OAuth from ActionContext default_accounts.

        This allows plugins to get their OAuth account from Smart Alias
        configuration rather than requiring plugin-level config.

        Args:
            context: ActionContext with default_accounts dict
            account_key: Key in default_accounts (e.g., "email", "calendar", "tasks")
            force: If True, override existing oauth_account_id

        Returns:
            True if OAuth was configured successfully
        """
        # Skip if already configured (unless force=True)
        if self.oauth_account_id and not force:
            return True

        # Get default account from context
        default_accounts = getattr(context, "default_accounts", {})
        if not default_accounts:
            logger.debug(f"No default_accounts in context for {account_key}")
            return False

        account_info = default_accounts.get(account_key, {})
        if not account_info:
            logger.debug(f"No {account_key} account in context.default_accounts")
            return False

        account_id = account_info.get("id")
        if not account_id:
            logger.debug(f"No account ID in {account_key} default account")
            return False

        # Configure OAuth
        self.oauth_account_id = account_id
        self.oauth_provider = account_info.get("provider", "google")

        logger.info(
            f"Configured OAuth from context: {account_key} -> "
            f"account {account_id} ({account_info.get('email', 'unknown')})"
        )

        # Initialize/reinitialize OAuth client
        self._init_oauth_client()
        return True

    def _init_oauth_client(self, timeout: int = 30) -> None:
        """
        Initialize the OAuth HTTP client.

        Args:
            timeout: Request timeout in seconds
        """
        self._oauth_client = httpx.Client(timeout=timeout)
        self._refresh_token_if_needed()

    def _get_token_data(self) -> Optional[dict]:
        """
        Get token data from the database.

        Returns:
            Token data dict or None if not found
        """
        try:
            from db.oauth_tokens import get_oauth_token_by_id

            token_data = get_oauth_token_by_id(self.oauth_account_id)
            return token_data
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            return None

    def _update_token_data(self, token_data: dict) -> bool:
        """
        Update token data in the database.

        Args:
            token_data: New token data to store

        Returns:
            True if successful
        """
        try:
            from db.oauth_tokens import update_oauth_token_data

            update_oauth_token_data(self.oauth_account_id, token_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update OAuth token: {e}")
            return False

    def _refresh_token_if_needed(self) -> bool:
        """
        Check token expiry and refresh if needed.

        Returns:
            True if we have a valid token
        """
        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"OAuth token not found: {self.oauth_account_id}")
            return False

        # Check if token is expired (with 5 minute buffer)
        # Also refresh if expires_at is 0/missing (unknown expiry)
        expires_at = token_data.get("expires_at", 0)
        current_time = time.time()

        # Refresh if: no expiry set (unknown state) OR expired/expiring soon
        if not expires_at or current_time > expires_at - 300:
            logger.info(
                f"OAuth token expired or expiring soon for account {self.oauth_account_id}, refreshing..."
            )
            refreshed = self._do_token_refresh(token_data)
            if refreshed:
                self._update_token_data(refreshed)
                token_data = refreshed
            else:
                logger.error("Token refresh failed")
                return False

        self._access_token = token_data.get("access_token")
        self._token_expires_at = token_data.get("expires_at", 0)
        return bool(self._access_token)

    def _do_token_refresh(self, token_data: dict) -> Optional[dict]:
        """
        Perform OAuth token refresh.

        Override for provider-specific refresh logic.

        Args:
            token_data: Current token data containing refresh_token

        Returns:
            Updated token data dict, or None on failure
        """
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            logger.error("No refresh token available")
            return None

        # Get OAuth credentials
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not client_id or not client_secret:
            logger.error("Missing client_id or client_secret in token data")
            return None

        # Determine token URL based on provider
        token_url = token_data.get("token_url")
        if not token_url:
            token_url = self._get_default_token_url()

        if not token_url:
            logger.error(f"No token URL for provider: {self.oauth_provider}")
            return None

        try:
            response = httpx.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=30,
            )
            response.raise_for_status()
            new_token = response.json()

            # Merge with existing data (preserve refresh_token if not returned)
            result = token_data.copy()
            result["access_token"] = new_token["access_token"]

            if "refresh_token" in new_token:
                result["refresh_token"] = new_token["refresh_token"]

            if "expires_in" in new_token:
                result["expires_at"] = time.time() + new_token["expires_in"]

            logger.info(
                f"Successfully refreshed OAuth token for account {self.oauth_account_id}"
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Token refresh HTTP error: {e.response.status_code}")
            try:
                error_body = e.response.json()
                logger.error(f"Error response: {error_body}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    def _get_default_token_url(self) -> Optional[str]:
        """
        Get default token URL for known providers.

        Returns:
            Token URL string or None
        """
        token_urls = {
            "google": "https://oauth2.googleapis.com/token",
            "slack": "https://slack.com/api/oauth.v2.access",
            "microsoft": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "github": "https://github.com/login/oauth/access_token",
        }
        return token_urls.get(self.oauth_provider)

    def _get_auth_headers(self) -> dict:
        """
        Get authorization headers for requests.

        Refreshes token if needed before returning headers.

        Returns:
            Dict with Authorization header
        """
        # Check if we need to refresh (or if we don't have a token yet)
        current_time = time.time()
        if not self._access_token or current_time > self._token_expires_at - 60:
            self._refresh_token_if_needed()

        if not self._access_token:
            logger.error("No access token available after refresh attempt")

        return {"Authorization": f"Bearer {self._access_token}"}

    def oauth_get(self, url: str, **kwargs) -> httpx.Response:
        """
        Make authenticated GET request.

        Args:
            url: Request URL
            **kwargs: Additional arguments for httpx.get

        Returns:
            httpx.Response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.get(url, headers=headers, **kwargs)

    def oauth_post(self, url: str, **kwargs) -> httpx.Response:
        """
        Make authenticated POST request.

        Args:
            url: Request URL
            **kwargs: Additional arguments for httpx.post

        Returns:
            httpx.Response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.post(url, headers=headers, **kwargs)

    def oauth_put(self, url: str, **kwargs) -> httpx.Response:
        """
        Make authenticated PUT request.

        Args:
            url: Request URL
            **kwargs: Additional arguments for httpx.put

        Returns:
            httpx.Response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.put(url, headers=headers, **kwargs)

    def oauth_patch(self, url: str, **kwargs) -> httpx.Response:
        """
        Make authenticated PATCH request.

        Args:
            url: Request URL
            **kwargs: Additional arguments for httpx.patch

        Returns:
            httpx.Response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.patch(url, headers=headers, **kwargs)

    def oauth_delete(self, url: str, **kwargs) -> httpx.Response:
        """
        Make authenticated DELETE request.

        Args:
            url: Request URL
            **kwargs: Additional arguments for httpx.delete

        Returns:
            httpx.Response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.delete(url, headers=headers, **kwargs)

    def get_account_email(self) -> Optional[str]:
        """
        Get the email address associated with the OAuth account.

        Returns:
            Email string or None
        """
        try:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(self.oauth_account_id)
            if token_info:
                return token_info.get("account_email")
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        if self._oauth_client:
            self._oauth_client.close()
            self._oauth_client = None


class GoogleOAuthMixin(OAuthMixin):
    """
    OAuth mixin specifically for Google APIs.

    Uses Google's token endpoint by default.
    """

    oauth_provider: str = "google"


class MicrosoftOAuthMixin(OAuthMixin):
    """
    OAuth mixin specifically for Microsoft Graph API.

    Uses Microsoft's consumer token endpoint by default.
    For enterprise/work accounts, override _get_default_token_url.
    """

    oauth_provider: str = "microsoft"

    def _get_default_token_url(self) -> Optional[str]:
        """Use Microsoft consumer token endpoint."""
        # For consumer accounts (personal Microsoft accounts)
        # Enterprise would use: https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token
        return "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
