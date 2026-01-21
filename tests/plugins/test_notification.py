"""
Tests for the Notification action plugin.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from builtin_plugins.actions.notification import NotificationActionHandler
from plugin_base.action import ActionContext, ActionRisk


class TestNotificationPluginMetadata:
    """Tests for plugin metadata."""

    def test_action_type(self):
        assert NotificationActionHandler.action_type == "notification"

    def test_display_name(self):
        assert NotificationActionHandler.display_name == "Notifications"

    def test_category(self):
        assert NotificationActionHandler.category == "communication"

    def test_not_abstract(self):
        assert NotificationActionHandler._abstract is False

    def test_config_fields(self):
        fields = NotificationActionHandler.get_config_fields()
        field_names = [f.name for f in fields]

        assert "apprise_api_url" in field_names
        assert "default_urls" in field_names

    def test_apprise_url_required(self):
        fields = NotificationActionHandler.get_config_fields()
        apprise_field = next(f for f in fields if f.name == "apprise_api_url")

        assert apprise_field.required is True

    def test_actions(self):
        actions = NotificationActionHandler.get_actions()
        action_names = [a.name for a in actions]

        assert "send" in action_names

    def test_send_action_risk(self):
        actions = NotificationActionHandler.get_actions()
        send_action = next(a for a in actions if a.name == "send")

        assert send_action.risk == ActionRisk.LOW


class TestNotificationPluginValidation:
    """Tests for config validation."""

    def test_valid_config(self):
        result = NotificationActionHandler.validate_config(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )

        assert result.valid is True

    def test_missing_api_url(self):
        result = NotificationActionHandler.validate_config({})

        assert result.valid is False
        assert any(e.field == "apprise_api_url" for e in result.errors)

    def test_with_default_urls(self):
        result = NotificationActionHandler.validate_config(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "ntfy://ntfy.sh/mytopic",
            }
        )

        assert result.valid is True


class TestNotificationInit:
    """Tests for plugin initialization."""

    def test_init_with_api_url(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )

        assert handler.api_url == "http://apprise:8000"
        assert handler.default_urls == []

    def test_init_with_default_urls(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "ntfy://ntfy.sh/topic1\ngotify://gotify/token",
            }
        )

        assert handler.api_url == "http://apprise:8000"
        assert len(handler.default_urls) == 2
        assert "ntfy://ntfy.sh/topic1" in handler.default_urls
        assert "gotify://gotify/token" in handler.default_urls

    def test_init_with_empty_lines_in_urls(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "url1\n\nurl2\n  \nurl3",
            }
        )

        assert len(handler.default_urls) == 3

    @patch.dict("os.environ", {"APPRISE_API_URL": "http://env-apprise:8000"})
    def test_init_from_env_fallback(self):
        handler = NotificationActionHandler({})

        assert handler.api_url == "http://env-apprise:8000"


class TestNotificationSend:
    """Tests for send action."""

    @pytest.fixture
    def handler(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "ntfy://ntfy.sh/mytopic",
            }
        )
        handler.client = MagicMock()
        return handler

    def test_send_simple_notification(self, handler):
        """Test sending a simple notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Test notification"},
            ActionContext(),
        )

        assert result.success is True
        assert "sent to 1 service" in result.message

        # Check the API was called correctly
        handler.client.post.assert_called_once()
        call_args = handler.client.post.call_args
        assert call_args[0][0] == "http://apprise:8000/notify/"
        assert call_args[1]["json"]["body"] == "Test notification"
        assert call_args[1]["json"]["type"] == "info"

    def test_send_with_title(self, handler):
        """Test sending notification with title."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Message content", "title": "Important Alert"},
            ActionContext(),
        )

        assert result.success is True
        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["title"] == "Important Alert"

    def test_send_with_type(self, handler):
        """Test sending notification with type."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Success!", "type": "success"},
            ActionContext(),
        )

        assert result.success is True
        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["type"] == "success"

    def test_send_with_custom_urls(self, handler):
        """Test sending to custom URLs instead of defaults."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Test", "urls": "slack://webhook/url"},
            ActionContext(),
        )

        assert result.success is True
        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["urls"] == ["slack://webhook/url"]

    def test_send_with_multiple_urls_string(self, handler):
        """Test parsing multiple URLs from string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Test", "urls": "url1\nurl2,url3"},
            ActionContext(),
        )

        assert result.success is True
        call_args = handler.client.post.call_args
        assert len(call_args[1]["json"]["urls"]) == 3

    def test_send_invalid_type_defaults_to_info(self, handler):
        """Test that invalid type defaults to info."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Test", "type": "invalid_type"},
            ActionContext(),
        )

        assert result.success is True
        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["type"] == "info"


class TestNotificationErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "ntfy://ntfy.sh/mytopic",
            }
        )
        handler.client = MagicMock()
        return handler

    def test_missing_body(self, handler):
        """Test error when body is missing."""
        result = handler.execute("send", {}, ActionContext())

        assert result.success is False
        assert "body" in result.error.lower()

    def test_no_urls_configured(self):
        """Test error when no URLs available."""
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )
        handler.client = MagicMock()

        result = handler.execute(
            "send",
            {"body": "Test"},
            ActionContext(),
        )

        assert result.success is False
        assert "url" in result.error.lower()

    @patch.dict("os.environ", {"APPRISE_API_URL": ""}, clear=False)
    def test_no_api_url(self):
        """Test error when API URL not configured."""
        # Clear env var to ensure no fallback
        import os

        old_val = os.environ.pop("APPRISE_API_URL", None)
        try:
            handler = NotificationActionHandler({})
            handler.client = MagicMock()

            result = handler.execute(
                "send",
                {"body": "Test", "urls": "ntfy://test"},
                ActionContext(),
            )

            assert result.success is False
            assert "not configured" in result.error.lower()
        finally:
            if old_val is not None:
                os.environ["APPRISE_API_URL"] = old_val

    def test_api_error_response(self, handler):
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "send",
            {"body": "Test"},
            ActionContext(),
        )

        assert result.success is False
        assert "500" in result.error

    def test_connection_error(self, handler):
        """Test handling connection error."""
        handler.client.post.side_effect = httpx.ConnectError("Connection refused")

        result = handler.execute(
            "send",
            {"body": "Test"},
            ActionContext(),
        )

        assert result.success is False
        assert "cannot connect" in result.error.lower()

    def test_timeout_error(self, handler):
        """Test handling timeout error."""
        handler.client.post.side_effect = httpx.TimeoutException("Timeout")

        result = handler.execute(
            "send",
            {"body": "Test"},
            ActionContext(),
        )

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_unknown_action(self, handler):
        """Test handling unknown action."""
        result = handler.execute("unknown", {}, ActionContext())

        assert result.success is False
        assert "unknown action" in result.error.lower()


class TestNotificationApprovalSummary:
    """Tests for approval summary generation."""

    @pytest.fixture
    def handler(self):
        return NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )

    def test_send_summary_with_title(self, handler):
        summary = handler.get_approval_summary(
            "send",
            {"title": "Alert", "body": "Something happened"},
        )

        assert "Alert" in summary
        assert "notification" in summary.lower()

    def test_send_summary_body_only(self, handler):
        summary = handler.get_approval_summary(
            "send",
            {
                "body": "A very long message that should be truncated after fifty characters"
            },
        )

        assert "A very long message" in summary
        assert "..." in summary

    def test_send_summary_short_body(self, handler):
        summary = handler.get_approval_summary(
            "send",
            {"body": "Short message"},
        )

        assert "Short message" in summary


class TestNotificationParamValidation:
    """Tests for parameter validation."""

    @pytest.fixture
    def handler(self):
        return NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "ntfy://test",
            }
        )

    def test_validate_valid_params(self, handler):
        result = handler.validate_action_params("send", {"body": "Test"})

        assert result.valid is True

    def test_validate_missing_body(self, handler):
        result = handler.validate_action_params("send", {})

        assert result.valid is False
        assert any(e.field == "body" for e in result.errors)

    def test_validate_with_message_alias(self, handler):
        """Test that 'message' is accepted as alias for 'body'."""
        result = handler.validate_action_params("send", {"message": "Test"})

        assert result.valid is True

    def test_validate_no_urls_without_defaults(self):
        """Test validation fails when no URLs available."""
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )

        result = handler.validate_action_params("send", {"body": "Test"})

        assert result.valid is False
        assert any(e.field == "urls" for e in result.errors)


class TestNotificationTestConnection:
    """Tests for connection testing."""

    def test_connection_success(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
                "default_urls": "url1\nurl2",
            }
        )
        handler.client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        handler.client.get.return_value = mock_response

        success, message = handler.test_connection()

        assert success is True
        assert "Connected" in message
        assert "2 default URL" in message

    def test_connection_failure(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )
        handler.client = MagicMock()
        handler.client.get.side_effect = httpx.ConnectError("Connection refused")

        success, message = handler.test_connection()

        assert success is False
        assert "cannot connect" in message.lower()

    def test_connection_no_url_configured(self):
        import os

        old_val = os.environ.pop("APPRISE_API_URL", None)
        try:
            handler = NotificationActionHandler({})

            success, message = handler.test_connection()

            assert success is False
            assert "not configured" in message.lower()
        finally:
            if old_val is not None:
                os.environ["APPRISE_API_URL"] = old_val


class TestNotificationLLMInstructions:
    """Tests for LLM instruction generation."""

    def test_instructions_contain_action(self):
        instructions = NotificationActionHandler.get_llm_instructions()

        assert "notification:send" in instructions

    def test_instructions_contain_parameters(self):
        instructions = NotificationActionHandler.get_llm_instructions()

        assert "body" in instructions
        assert "title" in instructions
        assert "type" in instructions

    def test_instructions_contain_example(self):
        instructions = NotificationActionHandler.get_llm_instructions()

        assert "<smart_action" in instructions
        assert "notification" in instructions


class TestNotificationAvailability:
    """Tests for availability check."""

    def test_available_when_configured(self):
        handler = NotificationActionHandler(
            {
                "apprise_api_url": "http://apprise:8000",
            }
        )

        assert handler.is_available() is True

    def test_not_available_without_url(self):
        import os

        old_val = os.environ.pop("APPRISE_API_URL", None)
        try:
            handler = NotificationActionHandler({})

            assert handler.is_available() is False
        finally:
            if old_val is not None:
                os.environ["APPRISE_API_URL"] = old_val
