"""
Tests for the Calendar action plugin.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from builtin_plugins.actions.calendar import CalendarActionHandler
from plugin_base.action import ActionContext, ActionRisk


class TestCalendarPluginMetadata:
    """Tests for plugin metadata."""

    def test_action_type(self):
        assert CalendarActionHandler.action_type == "calendar"

    def test_display_name(self):
        assert CalendarActionHandler.display_name == "Calendar"

    def test_category(self):
        assert CalendarActionHandler.category == "productivity"

    def test_not_abstract(self):
        assert CalendarActionHandler._abstract is False

    def test_config_fields(self):
        fields = CalendarActionHandler.get_config_fields()
        field_names = [f.name for f in fields]

        assert "oauth_account_id" in field_names
        assert "default_calendar_id" in field_names
        assert "default_timezone" in field_names

    def test_oauth_account_required(self):
        fields = CalendarActionHandler.get_config_fields()
        oauth_field = next(f for f in fields if f.name == "oauth_account_id")

        assert oauth_field.required is True

    def test_actions(self):
        actions = CalendarActionHandler.get_actions()
        action_names = [a.name for a in actions]

        assert "create" in action_names
        assert "update" in action_names
        assert "delete" in action_names

    def test_action_risks(self):
        actions = CalendarActionHandler.get_actions()

        create_action = next(a for a in actions if a.name == "create")
        assert create_action.risk == ActionRisk.LOW

        update_action = next(a for a in actions if a.name == "update")
        assert update_action.risk == ActionRisk.MEDIUM

        delete_action = next(a for a in actions if a.name == "delete")
        assert delete_action.risk == ActionRisk.HIGH


class TestCalendarPluginValidation:
    """Tests for config validation."""

    def test_valid_config(self):
        result = CalendarActionHandler.validate_config(
            {
                "oauth_account_id": 123,
            }
        )

        assert result.valid is True

    def test_missing_oauth_account(self):
        result = CalendarActionHandler.validate_config({})

        assert result.valid is False
        assert any(e.field == "oauth_account_id" for e in result.errors)

    def test_with_optional_fields(self):
        result = CalendarActionHandler.validate_config(
            {
                "oauth_account_id": 123,
                "default_calendar_id": "work",
                "default_timezone": "America/New_York",
            }
        )

        assert result.valid is True


class TestCalendarInit:
    """Tests for plugin initialization."""

    @patch("plugin_base.oauth.OAuthMixin._init_oauth_client")
    def test_init_with_oauth_account(self, mock_init_oauth):
        handler = CalendarActionHandler(
            {
                "oauth_account_id": 123,
            }
        )

        assert handler.oauth_account_id == 123
        assert handler.default_calendar_id == "primary"
        assert handler.default_timezone == "Europe/London"
        mock_init_oauth.assert_called_once()

    def test_init_without_oauth_account(self):
        handler = CalendarActionHandler({})

        assert handler.oauth_account_id is None
        assert handler.default_calendar_id == "primary"

    @patch("plugin_base.oauth.OAuthMixin._init_oauth_client")
    def test_init_with_custom_defaults(self, mock_init_oauth):
        handler = CalendarActionHandler(
            {
                "oauth_account_id": 123,
                "default_calendar_id": "work@group.calendar.google.com",
                "default_timezone": "America/Los_Angeles",
            }
        )

        assert handler.default_calendar_id == "work@group.calendar.google.com"
        assert handler.default_timezone == "America/Los_Angeles"


class TestCalendarCreateEvent:
    """Tests for create action."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler(
                {
                    "oauth_account_id": 123,
                    "default_timezone": "Europe/London",
                }
            )
            handler._oauth_client = MagicMock()
            handler._access_token = "test-token"
            handler._token_expires_at = 9999999999
            return handler

    def test_create_simple_event(self, handler):
        """Test creating a simple timed event."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "event123",
            "htmlLink": "https://calendar.google.com/event?eid=event123",
        }
        mock_response.raise_for_status = MagicMock()
        handler._oauth_client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {
                "title": "Team Meeting",
                "start": "2026-01-20T14:00:00",
                "end": "2026-01-20T15:00:00",
            },
            ActionContext(),
        )

        assert result.success is True
        assert "Team Meeting" in result.message
        assert result.data["event_id"] == "event123"

    def test_create_event_with_description(self, handler):
        """Test creating event with description and location."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "event123"}
        mock_response.raise_for_status = MagicMock()
        handler._oauth_client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {
                "title": "Project Review",
                "start": "2026-01-20T10:00:00",
                "description": "Quarterly review meeting",
                "location": "Conference Room B",
            },
            ActionContext(),
        )

        assert result.success is True

        # Check the request body
        call_args = handler._oauth_client.post.call_args
        request_body = call_args[1]["json"]
        assert request_body["description"] == "Quarterly review meeting"
        assert request_body["location"] == "Conference Room B"

    def test_create_event_default_end_time(self, handler):
        """Test that end time defaults to 1 hour after start."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "event123"}
        mock_response.raise_for_status = MagicMock()
        handler._oauth_client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {
                "title": "Quick Call",
                "start": "2026-01-20T14:00:00",
            },
            ActionContext(),
        )

        assert result.success is True

        call_args = handler._oauth_client.post.call_args
        request_body = call_args[1]["json"]
        # End should be 1 hour after start
        assert "15:00:00" in request_body["end"]["dateTime"]

    def test_create_all_day_event(self, handler):
        """Test creating an all-day event."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "event123"}
        mock_response.raise_for_status = MagicMock()
        handler._oauth_client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {
                "title": "Holiday",
                "date": "2026-01-20",
                "all_day": True,
            },
            ActionContext(),
        )

        assert result.success is True

        call_args = handler._oauth_client.post.call_args
        request_body = call_args[1]["json"]
        assert request_body["start"]["date"] == "2026-01-20"
        assert "dateTime" not in request_body["start"]

    def test_create_recurring_event(self, handler):
        """Test creating a recurring event."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "event123"}
        mock_response.raise_for_status = MagicMock()
        handler._oauth_client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {
                "title": "Daily Standup",
                "start": "2026-01-20T09:00:00",
                "recurrence": "weekdays",
            },
            ActionContext(),
        )

        assert result.success is True

        call_args = handler._oauth_client.post.call_args
        request_body = call_args[1]["json"]
        assert "recurrence" in request_body
        assert "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR" in request_body["recurrence"]

    def test_create_event_missing_title(self, handler):
        """Test error when title is missing."""
        result = handler.execute(
            "create",
            {"start": "2026-01-20T14:00:00"},
            ActionContext(),
        )

        assert result.success is False
        assert "title" in result.error.lower()

    def test_create_event_missing_start(self, handler):
        """Test error when start time is missing."""
        result = handler.execute(
            "create",
            {"title": "Test Event"},
            ActionContext(),
        )

        assert result.success is False
        assert "start" in result.error.lower()


class TestCalendarUpdateEvent:
    """Tests for update action."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler(
                {
                    "oauth_account_id": 123,
                }
            )
            handler._oauth_client = MagicMock()
            handler._access_token = "test-token"
            handler._token_expires_at = 9999999999
            return handler

    def test_update_event_title(self, handler):
        """Test updating event title."""
        # Mock GET response (existing event)
        get_response = MagicMock()
        get_response.status_code = 200
        get_response.json.return_value = {
            "id": "event123",
            "summary": "Old Title",
            "start": {"dateTime": "2026-01-20T14:00:00"},
            "end": {"dateTime": "2026-01-20T15:00:00"},
        }

        # Mock PUT response
        put_response = MagicMock()
        put_response.status_code = 200
        put_response.raise_for_status = MagicMock()

        handler._oauth_client.get.return_value = get_response
        handler._oauth_client.put.return_value = put_response

        result = handler.execute(
            "update",
            {
                "event_id": "event123",
                "title": "New Title",
            },
            ActionContext(),
        )

        assert result.success is True
        assert result.data["event_id"] == "event123"

    def test_update_event_not_found(self, handler):
        """Test error when event not found."""
        get_response = MagicMock()
        get_response.status_code = 404
        handler._oauth_client.get.return_value = get_response

        result = handler.execute(
            "update",
            {"event_id": "nonexistent", "title": "New Title"},
            ActionContext(),
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_update_event_missing_id(self, handler):
        """Test error when event_id is missing."""
        result = handler.execute(
            "update",
            {"title": "New Title"},
            ActionContext(),
        )

        assert result.success is False
        assert "event_id" in result.error.lower()


class TestCalendarDeleteEvent:
    """Tests for delete action."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler(
                {
                    "oauth_account_id": 123,
                }
            )
            handler._oauth_client = MagicMock()
            handler._access_token = "test-token"
            handler._token_expires_at = 9999999999
            return handler

    def test_delete_event(self, handler):
        """Test deleting an event."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        handler._oauth_client.delete.return_value = mock_response

        result = handler.execute(
            "delete",
            {"event_id": "event123"},
            ActionContext(),
        )

        assert result.success is True
        assert "deleted" in result.message.lower()

    def test_delete_event_missing_id(self, handler):
        """Test error when event_id is missing."""
        result = handler.execute(
            "delete",
            {},
            ActionContext(),
        )

        assert result.success is False
        assert "event_id" in result.error.lower()


class TestCalendarErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler(
                {
                    "oauth_account_id": 123,
                }
            )
            handler._oauth_client = MagicMock()
            handler._access_token = "test-token"
            handler._token_expires_at = 9999999999
            return handler

    def test_no_oauth_configured(self):
        """Test error when OAuth not configured."""
        handler = CalendarActionHandler({})

        result = handler.execute(
            "create",
            {"title": "Test", "start": "2026-01-20T14:00:00"},
            ActionContext(),
        )

        assert result.success is False
        assert "not configured" in result.error.lower()

    def test_api_error(self, handler):
        """Test handling API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        error = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=mock_response
        )
        error = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=mock_response
        )
        handler._oauth_client.post.return_value.raise_for_status.side_effect = error

        result = handler.execute(
            "create",
            {"title": "Test", "start": "2026-01-20T14:00:00"},
            ActionContext(),
        )

        assert result.success is False
        assert "403" in result.error or "error" in result.error.lower()

    def test_unknown_action(self, handler):
        """Test handling unknown action."""
        result = handler.execute("unknown", {}, ActionContext())

        assert result.success is False
        assert "unknown action" in result.error.lower()


class TestCalendarApprovalSummary:
    """Tests for approval summary generation."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            return CalendarActionHandler({"oauth_account_id": 123})

    def test_create_summary(self, handler):
        summary = handler.get_approval_summary(
            "create",
            {"title": "Team Meeting", "start": "2026-01-20T14:00:00"},
        )

        assert "Team Meeting" in summary
        assert "2026-01-20" in summary

    def test_update_summary(self, handler):
        summary = handler.get_approval_summary(
            "update",
            {"event_id": "abc123xyz", "title": "New Title"},
        )

        assert "abc123xyz" in summary
        assert "update" in summary.lower()

    def test_delete_summary(self, handler):
        summary = handler.get_approval_summary(
            "delete",
            {"event_id": "abc123xyz"},
        )

        assert "abc123xyz" in summary
        assert "delete" in summary.lower()


class TestCalendarParamValidation:
    """Tests for parameter validation."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            return CalendarActionHandler({"oauth_account_id": 123})

    def test_validate_create_valid(self, handler):
        result = handler.validate_action_params(
            "create",
            {"title": "Test", "start": "2026-01-20T14:00:00"},
        )

        assert result.valid is True

    def test_validate_create_missing_title(self, handler):
        result = handler.validate_action_params(
            "create",
            {"start": "2026-01-20T14:00:00"},
        )

        assert result.valid is False
        assert any(e.field == "title" for e in result.errors)

    def test_validate_create_missing_start(self, handler):
        result = handler.validate_action_params(
            "create",
            {"title": "Test"},
        )

        assert result.valid is False
        assert any(e.field == "start" for e in result.errors)

    def test_validate_update_valid(self, handler):
        result = handler.validate_action_params(
            "update",
            {"event_id": "abc123", "title": "New Title"},
        )

        assert result.valid is True

    def test_validate_update_missing_event_id(self, handler):
        result = handler.validate_action_params(
            "update",
            {"title": "New Title"},
        )

        assert result.valid is False
        assert any(e.field == "event_id" for e in result.errors)

    def test_validate_update_no_fields(self, handler):
        result = handler.validate_action_params(
            "update",
            {"event_id": "abc123"},
        )

        assert result.valid is False

    def test_validate_delete_valid(self, handler):
        result = handler.validate_action_params(
            "delete",
            {"event_id": "abc123"},
        )

        assert result.valid is True

    def test_validate_delete_missing_event_id(self, handler):
        result = handler.validate_action_params(
            "delete",
            {},
        )

        assert result.valid is False


class TestCalendarTestConnection:
    """Tests for connection testing."""

    def test_connection_success(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler({"oauth_account_id": 123})
            handler._oauth_client = MagicMock()
            handler._access_token = "test-token"
            handler._token_expires_at = 9999999999

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {"id": "primary", "summary": "Main Calendar"},
                    {"id": "work", "summary": "Work Calendar"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            handler._oauth_client.get.return_value = mock_response

            success, message = handler.test_connection()

            assert success is True
            assert "2 calendars" in message

    def test_connection_no_oauth(self):
        handler = CalendarActionHandler({})

        success, message = handler.test_connection()

        assert success is False
        assert "not configured" in message.lower()


class TestCalendarLLMInstructions:
    """Tests for LLM instruction generation."""

    def test_instructions_contain_actions(self):
        instructions = CalendarActionHandler.get_llm_instructions()

        assert "calendar:create" in instructions
        assert "calendar:update" in instructions
        assert "calendar:delete" in instructions

    def test_instructions_contain_parameters(self):
        instructions = CalendarActionHandler.get_llm_instructions()

        assert "title" in instructions
        assert "start" in instructions
        assert "event_id" in instructions

    def test_instructions_contain_examples(self):
        instructions = CalendarActionHandler.get_llm_instructions()

        assert "<smart_action" in instructions
        assert "calendar" in instructions


class TestCalendarAvailability:
    """Tests for availability check."""

    def test_available_when_configured(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            handler = CalendarActionHandler({"oauth_account_id": 123})

        assert handler.is_available() is True

    def test_not_available_without_oauth(self):
        handler = CalendarActionHandler({})

        assert handler.is_available() is False


class TestCalendarDateTimeParsing:
    """Tests for datetime parsing helper."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            return CalendarActionHandler({"oauth_account_id": 123})

    def test_parse_iso_format(self, handler):
        dt = handler._parse_datetime("2026-01-20T14:00:00")
        assert dt == datetime(2026, 1, 20, 14, 0, 0)

    def test_parse_iso_short(self, handler):
        dt = handler._parse_datetime("2026-01-20T14:00")
        assert dt == datetime(2026, 1, 20, 14, 0, 0)

    def test_parse_date_only(self, handler):
        dt = handler._parse_datetime("2026-01-20")
        assert dt == datetime(2026, 1, 20, 0, 0, 0)

    def test_parse_with_timezone(self, handler):
        dt = handler._parse_datetime("2026-01-20T14:00:00Z")
        assert dt is not None
        assert dt.hour == 14

    def test_parse_invalid(self, handler):
        dt = handler._parse_datetime("invalid")
        assert dt is None


class TestCalendarRecurrenceConversion:
    """Tests for recurrence rule conversion."""

    @pytest.fixture
    def handler(self):
        with patch("plugin_base.oauth.OAuthMixin._init_oauth_client"):
            return CalendarActionHandler({"oauth_account_id": 123})

    def test_daily(self, handler):
        rrule = handler._simple_to_rrule("daily")
        assert rrule == "RRULE:FREQ=DAILY"

    def test_weekly(self, handler):
        rrule = handler._simple_to_rrule("weekly")
        assert rrule == "RRULE:FREQ=WEEKLY"

    def test_weekdays(self, handler):
        rrule = handler._simple_to_rrule("weekdays")
        assert "BYDAY=MO,TU,WE,TH,FR" in rrule

    def test_unknown(self, handler):
        rrule = handler._simple_to_rrule("every 3 days")
        assert rrule is None
