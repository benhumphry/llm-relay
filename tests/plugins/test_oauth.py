"""
Unit tests for plugin_base/oauth.py

Tests the OAuth mixin for token refresh and authenticated requests.
"""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from plugin_base.oauth import OAuthMixin


class MockOAuthPlugin(OAuthMixin):
    """Mock plugin using OAuthMixin for testing."""

    def __init__(self, oauth_account_id: int, oauth_provider: str = "google"):
        self.oauth_account_id = oauth_account_id
        self.oauth_provider = oauth_provider


class TestOAuthMixinTokenUrls:
    """Tests for default token URL resolution."""

    def test_google_token_url(self):
        plugin = MockOAuthPlugin(1, "google")
        url = plugin._get_default_token_url()
        assert url == "https://oauth2.googleapis.com/token"

    def test_slack_token_url(self):
        plugin = MockOAuthPlugin(1, "slack")
        url = plugin._get_default_token_url()
        assert url == "https://slack.com/api/oauth.v2.access"

    def test_microsoft_token_url(self):
        plugin = MockOAuthPlugin(1, "microsoft")
        url = plugin._get_default_token_url()
        assert "login.microsoftonline.com" in url

    def test_github_token_url(self):
        plugin = MockOAuthPlugin(1, "github")
        url = plugin._get_default_token_url()
        assert url == "https://github.com/login/oauth/access_token"

    def test_unknown_provider_token_url(self):
        plugin = MockOAuthPlugin(1, "unknown_provider")
        url = plugin._get_default_token_url()
        assert url is None


class TestOAuthMixinGetTokenData:
    """Tests for getting token data from database."""

    @patch("db.oauth_tokens.get_oauth_token")
    def test_get_token_data_success(self, mock_get_token):
        """Test getting token data when token exists."""
        mock_record = MagicMock()
        mock_record.token_data = {
            "access_token": "test_access",
            "refresh_token": "test_refresh",
            "expires_at": time.time() + 3600,
        }
        mock_get_token.return_value = mock_record

        plugin = MockOAuthPlugin(123, "google")
        token_data = plugin._get_token_data()

        assert token_data is not None
        assert token_data["access_token"] == "test_access"
        mock_get_token.assert_called_once_with(123)

    @patch("db.oauth_tokens.get_oauth_token")
    def test_get_token_data_not_found(self, mock_get_token):
        """Test getting token data when token doesn't exist."""
        mock_get_token.return_value = None

        plugin = MockOAuthPlugin(999, "google")
        token_data = plugin._get_token_data()

        assert token_data is None


class TestOAuthMixinTokenRefresh:
    """Tests for token refresh logic."""

    @patch("plugin_base.oauth.httpx.post")
    def test_do_token_refresh_success(self, mock_post):
        """Test successful token refresh."""
        # Mock successful refresh response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        plugin = MockOAuthPlugin(1, "google")
        token_data = {
            "access_token": "old_access",
            "refresh_token": "test_refresh",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        result = plugin._do_token_refresh(token_data)

        assert result is not None
        assert result["access_token"] == "new_access_token"
        assert "expires_at" in result
        # Original refresh_token preserved (not in response)
        assert result["refresh_token"] == "test_refresh"

    @patch("plugin_base.oauth.httpx.post")
    def test_do_token_refresh_with_new_refresh_token(self, mock_post):
        """Test token refresh when response includes new refresh token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        plugin = MockOAuthPlugin(1, "google")
        token_data = {
            "access_token": "old",
            "refresh_token": "old_refresh",
            "client_id": "id",
            "client_secret": "secret",
        }

        result = plugin._do_token_refresh(token_data)

        assert result["refresh_token"] == "new_refresh"

    def test_do_token_refresh_no_refresh_token(self):
        """Test refresh fails without refresh token."""
        plugin = MockOAuthPlugin(1, "google")
        token_data = {
            "access_token": "test",
            "client_id": "id",
            "client_secret": "secret",
        }

        result = plugin._do_token_refresh(token_data)
        assert result is None

    def test_do_token_refresh_no_credentials(self):
        """Test refresh fails without client credentials."""
        plugin = MockOAuthPlugin(1, "google")
        token_data = {
            "access_token": "test",
            "refresh_token": "refresh",
        }

        result = plugin._do_token_refresh(token_data)
        assert result is None

    @patch("plugin_base.oauth.httpx.post")
    def test_do_token_refresh_http_error(self, mock_post):
        """Test refresh handles HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "invalid_grant"}
        mock_post.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        plugin = MockOAuthPlugin(1, "google")
        token_data = {
            "access_token": "old",
            "refresh_token": "refresh",
            "client_id": "id",
            "client_secret": "secret",
        }

        result = plugin._do_token_refresh(token_data)
        assert result is None


class TestOAuthMixinRefreshIfNeeded:
    """Tests for _refresh_token_if_needed method."""

    @patch.object(MockOAuthPlugin, "_get_token_data")
    def test_refresh_if_needed_valid_token(self, mock_get_token):
        """Test no refresh when token is valid."""
        mock_get_token.return_value = {
            "access_token": "valid_token",
            "expires_at": time.time() + 3600,  # Expires in 1 hour
        }

        plugin = MockOAuthPlugin(1, "google")
        result = plugin._refresh_token_if_needed()

        assert result is True
        assert plugin._access_token == "valid_token"

    @patch.object(MockOAuthPlugin, "_get_token_data")
    def test_refresh_if_needed_no_token(self, mock_get_token):
        """Test returns False when no token exists."""
        mock_get_token.return_value = None

        plugin = MockOAuthPlugin(1, "google")
        result = plugin._refresh_token_if_needed()

        assert result is False

    @patch.object(MockOAuthPlugin, "_update_token_data")
    @patch.object(MockOAuthPlugin, "_do_token_refresh")
    @patch.object(MockOAuthPlugin, "_get_token_data")
    def test_refresh_if_needed_expired_token(
        self, mock_get_token, mock_refresh, mock_update
    ):
        """Test refresh is triggered for expired token."""
        # First call returns expired token, second call returns refreshed data
        mock_get_token.return_value = {
            "access_token": "old_token",
            "refresh_token": "refresh",
            "expires_at": time.time() - 100,  # Already expired
            "client_id": "id",
            "client_secret": "secret",
        }
        mock_refresh.return_value = {
            "access_token": "new_token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
            "client_id": "id",
            "client_secret": "secret",
        }
        mock_update.return_value = True

        plugin = MockOAuthPlugin(1, "google")
        result = plugin._refresh_token_if_needed()

        assert result is True
        mock_refresh.assert_called_once()
        mock_update.assert_called_once()

    @patch.object(MockOAuthPlugin, "_do_token_refresh")
    @patch.object(MockOAuthPlugin, "_get_token_data")
    def test_refresh_if_needed_refresh_fails(self, mock_get_token, mock_refresh):
        """Test returns False when refresh fails."""
        mock_get_token.return_value = {
            "access_token": "old",
            "refresh_token": "refresh",
            "expires_at": time.time() - 100,
            "client_id": "id",
            "client_secret": "secret",
        }
        mock_refresh.return_value = None  # Refresh failed

        plugin = MockOAuthPlugin(1, "google")
        result = plugin._refresh_token_if_needed()

        assert result is False


class TestOAuthMixinAuthHeaders:
    """Tests for _get_auth_headers method."""

    @patch.object(MockOAuthPlugin, "_refresh_token_if_needed")
    def test_get_auth_headers(self, mock_refresh):
        """Test getting auth headers."""
        mock_refresh.return_value = True

        plugin = MockOAuthPlugin(1, "google")
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        headers = plugin._get_auth_headers()

        assert headers["Authorization"] == "Bearer test_token"

    @patch.object(MockOAuthPlugin, "_refresh_token_if_needed")
    def test_get_auth_headers_refreshes_if_expiring(self, mock_refresh):
        """Test headers trigger refresh when token expiring soon."""
        mock_refresh.return_value = True

        plugin = MockOAuthPlugin(1, "google")
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 30  # Expires in 30 seconds

        plugin._get_auth_headers()

        # Should have called refresh because token expires within 60 seconds
        mock_refresh.assert_called_once()


class TestOAuthMixinHttpMethods:
    """Tests for HTTP request methods."""

    def test_oauth_get(self):
        """Test oauth_get makes authenticated GET request."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_get("https://api.example.com/data", params={"foo": "bar"})

        plugin._oauth_client.get.assert_called_once()
        call_args = plugin._oauth_client.get.call_args
        assert call_args[0][0] == "https://api.example.com/data"
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["params"] == {"foo": "bar"}

    def test_oauth_post(self):
        """Test oauth_post makes authenticated POST request."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_post("https://api.example.com/data", json={"key": "value"})

        plugin._oauth_client.post.assert_called_once()
        call_args = plugin._oauth_client.post.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["json"] == {"key": "value"}

    def test_oauth_put(self):
        """Test oauth_put makes authenticated PUT request."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_put("https://api.example.com/data")

        plugin._oauth_client.put.assert_called_once()

    def test_oauth_patch(self):
        """Test oauth_patch makes authenticated PATCH request."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_patch("https://api.example.com/data")

        plugin._oauth_client.patch.assert_called_once()

    def test_oauth_delete(self):
        """Test oauth_delete makes authenticated DELETE request."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_delete("https://api.example.com/data/123")

        plugin._oauth_client.delete.assert_called_once()

    def test_oauth_request_merges_headers(self):
        """Test that custom headers are merged with auth headers."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = MagicMock()
        plugin._access_token = "test_token"
        plugin._token_expires_at = time.time() + 3600

        plugin.oauth_get(
            "https://api.example.com/data",
            headers={"X-Custom-Header": "custom_value"},
        )

        call_args = plugin._oauth_client.get.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["X-Custom-Header"] == "custom_value"


class TestOAuthMixinGetAccountEmail:
    """Tests for get_account_email method."""

    @patch("db.oauth_tokens.get_oauth_token")
    def test_get_account_email_success(self, mock_get_token):
        """Test getting account email."""
        mock_record = MagicMock()
        mock_record.account_email = "user@example.com"
        mock_get_token.return_value = mock_record

        plugin = MockOAuthPlugin(1, "google")
        email = plugin.get_account_email()

        assert email == "user@example.com"

    @patch("db.oauth_tokens.get_oauth_token")
    def test_get_account_email_not_found(self, mock_get_token):
        """Test getting email when token not found."""
        mock_get_token.return_value = None

        plugin = MockOAuthPlugin(1, "google")
        email = plugin.get_account_email()

        assert email is None


class TestOAuthMixinClose:
    """Tests for close method."""

    def test_close_client(self):
        """Test closing the HTTP client."""
        plugin = MockOAuthPlugin(1, "google")
        mock_client = MagicMock()
        plugin._oauth_client = mock_client

        plugin.close()

        mock_client.close.assert_called_once()
        assert plugin._oauth_client is None

    def test_close_no_client(self):
        """Test close when no client exists."""
        plugin = MockOAuthPlugin(1, "google")
        plugin._oauth_client = None

        # Should not raise
        plugin.close()
