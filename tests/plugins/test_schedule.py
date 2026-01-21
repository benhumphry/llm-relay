"""
Tests for the Schedule action plugin.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from builtin_plugins.actions.schedule import ScheduleActionHandler
from plugin_base.action import ActionContext, ActionRisk
from plugin_base.common import FieldType


class TestSchedulePluginMetadata:
    """Test plugin metadata."""

    def test_action_type(self):
        assert ScheduleActionHandler.action_type == "schedule"

    def test_display_name(self):
        assert ScheduleActionHandler.display_name == "Scheduled Prompts"

    def test_category(self):
        assert ScheduleActionHandler.category == "automation"

    def test_not_abstract(self):
        assert ScheduleActionHandler._abstract is False

    def test_config_fields(self):
        fields = ScheduleActionHandler.get_config_fields()
        field_names = [f.name for f in fields]
        assert "oauth_account_id" in field_names
        assert "calendar_id" in field_names
        assert "default_timezone" in field_names

    def test_oauth_account_required(self):
        fields = ScheduleActionHandler.get_config_fields()
        oauth_field = next(f for f in fields if f.name == "oauth_account_id")
        assert oauth_field.required is True

    def test_actions(self):
        actions = ScheduleActionHandler.get_actions()
        action_names = [a.name for a in actions]
        assert "prompt" in action_names
        assert "cancel" in action_names

    def test_action_risks(self):
        actions = ScheduleActionHandler.get_actions()
        action_map = {a.name: a for a in actions}

        # Both are low risk
        assert action_map["prompt"].risk == ActionRisk.LOW
        assert action_map["cancel"].risk == ActionRisk.LOW


class TestSchedulePluginValidation:
    """Test plugin config validation."""

    def test_valid_config(self):
        result = ScheduleActionHandler.validate_config({"oauth_account_id": 123})
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_config_with_options(self):
        result = ScheduleActionHandler.validate_config(
            {
                "oauth_account_id": 123,
                "calendar_id": "my-calendar",
                "default_timezone": "America/New_York",
            }
        )
        assert result.valid is True

    def test_missing_oauth_account(self):
        result = ScheduleActionHandler.validate_config({})
        assert result.valid is False
        assert any("oauth_account_id" in str(e.field) for e in result.errors)


class TestScheduleInit:
    """Test plugin initialization."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_init_with_oauth_account(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler.oauth_account_id == 123
        assert handler.oauth_provider == "google"
        assert handler.calendar_id == "primary"
        mock_init.assert_called_once()

    def test_init_without_oauth_account(self):
        handler = ScheduleActionHandler({})
        assert handler.oauth_account_id is None

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_init_with_custom_calendar(self, mock_init):
        handler = ScheduleActionHandler(
            {
                "oauth_account_id": 123,
                "calendar_id": "custom-calendar",
                "default_timezone": "America/Los_Angeles",
            }
        )
        assert handler.calendar_id == "custom-calendar"
        assert handler.default_timezone == "America/Los_Angeles"


class TestSchedulePromptAction:
    """Test schedule prompt action."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_schedule_simple_prompt(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "event123"}

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "prompt",
                {"prompt": "Send me the weather forecast", "time": "06:30"},
                context,
            )

            assert result.success is True
            assert "event123" in str(result.data)

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_schedule_prompt_with_title(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "event456"}

        with patch.object(
            handler, "oauth_post", return_value=mock_response
        ) as mock_post:
            context = ActionContext()
            result = handler.execute(
                "prompt",
                {
                    "prompt": "Check the news",
                    "time": "07:00",
                    "title": "Morning News Check",
                },
                context,
            )

            assert result.success is True
            # Verify title is used in the event
            call_args = mock_post.call_args
            json_arg = call_args[1]["json"]
            assert "Morning News Check" in json_arg["summary"]

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_schedule_recurring_prompt(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "recurring123"}

        with patch.object(
            handler, "oauth_post", return_value=mock_response
        ) as mock_post:
            context = ActionContext()
            result = handler.execute(
                "prompt",
                {
                    "prompt": "Daily standup summary",
                    "time": "09:00",
                    "recurrence": "weekdays",
                },
                context,
            )

            assert result.success is True
            # Verify recurrence is set
            call_args = mock_post.call_args
            json_arg = call_args[1]["json"]
            assert "recurrence" in json_arg
            assert "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR" in json_arg["recurrence"]

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_schedule_prompt_api_failure(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "prompt", {"prompt": "Test prompt", "time": "10:00"}, context
            )

            assert result.success is False
            assert "400" in result.error

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_schedule_prompt_invalid_time(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        context = ActionContext()
        result = handler.execute(
            "prompt",
            {"prompt": "Test prompt", "time": "invalid time format xyz"},
            context,
        )

        assert result.success is False
        assert "parse" in result.error.lower()


class TestScheduleCancelAction:
    """Test cancel prompt action."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_cancel_prompt(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 204

        with patch.object(handler, "oauth_delete", return_value=mock_response):
            context = ActionContext()
            result = handler.execute("cancel", {"event_id": "event123"}, context)

            assert result.success is True
            assert "event123" in str(result.data)

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_cancel_prompt_not_found(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(handler, "oauth_delete", return_value=mock_response):
            context = ActionContext()
            result = handler.execute("cancel", {"event_id": "nonexistent"}, context)

            assert result.success is False


class TestScheduleErrorHandling:
    """Test error handling."""

    def test_no_oauth_configured(self):
        handler = ScheduleActionHandler({})
        context = ActionContext()
        result = handler.execute("prompt", {"prompt": "Test", "time": "10:00"}, context)

        assert result.success is False
        assert "not configured" in result.error.lower()

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_unknown_action(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        context = ActionContext()
        result = handler.execute("invalid_action", {}, context)

        assert result.success is False
        assert "unknown" in result.error.lower()


class TestScheduleApprovalSummary:
    """Test approval summary generation."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_prompt_summary_with_title(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "prompt",
            {
                "title": "Morning Weather",
                "prompt": "Send me the weather",
                "time": "06:30",
            },
        )
        assert "Morning Weather" in summary
        assert "06:30" in summary

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_prompt_summary_without_title(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "prompt",
            {"prompt": "Send me the weather forecast for today", "time": "07:00"},
        )
        assert "weather" in summary.lower()

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_prompt_summary_with_recurrence(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "prompt",
            {"prompt": "Daily summary", "time": "08:00", "recurrence": "daily"},
        )
        assert "daily" in summary.lower()

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_cancel_summary(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary("cancel", {"event_id": "event123"})
        assert "cancel" in summary.lower()
        assert "event123" in summary


class TestScheduleParamValidation:
    """Test parameter validation."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_validate_prompt_valid(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "prompt", {"prompt": "Test prompt", "time": "06:30"}
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_validate_prompt_missing_prompt(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params("prompt", {"time": "06:30"})
        assert result.valid is False
        assert any("prompt" in str(e.field) for e in result.errors)

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_validate_prompt_missing_time(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params("prompt", {"prompt": "Test prompt"})
        assert result.valid is False
        assert any("time" in str(e.field) for e in result.errors)

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_validate_cancel_valid(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params("cancel", {"event_id": "event123"})
        assert result.valid is True

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_validate_cancel_missing_event_id(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params("cancel", {})
        assert result.valid is False
        assert any("event_id" in str(e.field) for e in result.errors)


class TestScheduleTestConnection:
    """Test connection testing."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_connection_success(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch.object(handler, "oauth_get", return_value=mock_response):
            success, message = handler.test_connection()
            assert success is True
            assert "primary" in message

    def test_connection_no_oauth(self):
        handler = ScheduleActionHandler({})
        success, message = handler.test_connection()
        assert success is False
        assert "not configured" in message.lower()


class TestScheduleLLMInstructions:
    """Test LLM instruction generation."""

    def test_instructions_contain_actions(self):
        instructions = ScheduleActionHandler.get_llm_instructions()
        assert "schedule:prompt" in instructions
        assert "schedule:cancel" in instructions

    def test_instructions_contain_parameters(self):
        instructions = ScheduleActionHandler.get_llm_instructions()
        assert "prompt" in instructions
        assert "time" in instructions
        assert "recurrence" in instructions

    def test_instructions_contain_examples(self):
        instructions = ScheduleActionHandler.get_llm_instructions()
        assert "<smart_action" in instructions
        assert 'type="schedule"' in instructions
        assert 'type="schedule"' in instructions


class TestScheduleAvailability:
    """Test availability checks."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_available_when_configured(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler.is_available() is True

    def test_not_available_without_oauth(self):
        handler = ScheduleActionHandler({})
        assert handler.is_available() is False


class TestScheduleTimeParsing:
    """Test time parsing functionality."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_hhmm(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("06:30", "Europe/London")
        assert start is not None
        assert start.hour == 6
        assert start.minute == 30
        assert end == start + timedelta(minutes=5)

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_iso_format(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("2026-01-20T14:30:00", "Europe/London")
        assert start is not None
        assert start.year == 2026
        assert start.month == 1
        assert start.day == 20
        assert start.hour == 14
        assert start.minute == 30

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_tomorrow(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("tomorrow at 9am", "Europe/London")
        assert start is not None
        now = datetime.now()
        expected = (now + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        assert start.date() == expected.date()
        assert start.hour == 9

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_in_minutes(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("in 30 minutes", "Europe/London")
        assert start is not None
        now = datetime.now()
        # Should be approximately 30 minutes from now
        diff = start - now
        assert 29 <= diff.total_seconds() / 60 <= 31

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_in_hours(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("in 2 hours", "Europe/London")
        assert start is not None
        now = datetime.now()
        diff = start - now
        assert 119 <= diff.total_seconds() / 60 <= 121

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_invalid(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("invalid gibberish xyz", "Europe/London")
        assert start is None
        assert end is None

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_time_empty(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        start, end = handler._parse_time("", "Europe/London")
        assert start is None
        assert end is None


class TestScheduleRecurrenceParsing:
    """Test recurrence parsing."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_daily(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler._parse_recurrence("daily") == "RRULE:FREQ=DAILY"

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_weekly(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler._parse_recurrence("weekly") == "RRULE:FREQ=WEEKLY"

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_weekdays(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler._parse_recurrence("weekdays")
        assert "BYDAY=MO,TU,WE,TH,FR" in result

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_monthly(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler._parse_recurrence("monthly") == "RRULE:FREQ=MONTHLY"

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_rrule_passthrough(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        result = handler._parse_recurrence("RRULE:FREQ=DAILY;COUNT=5")
        assert result == "RRULE:FREQ=DAILY;COUNT=5"

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_unknown_recurrence(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler._parse_recurrence("biweekly") is None

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_parse_empty_recurrence(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        assert handler._parse_recurrence("") is None
        assert handler._parse_recurrence(None) is None


class TestScheduleTitleGeneration:
    """Test title generation from prompt."""

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_short_prompt_unchanged(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        title = handler._generate_title("Check weather")
        assert title == "Check weather"

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_long_prompt_truncated(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        long_prompt = "This is a very long prompt that needs to be truncated because it exceeds fifty characters"
        title = handler._generate_title(long_prompt)
        assert len(title) <= 54  # 50 + "..."
        assert title.endswith("...")

    @patch("builtin_plugins.actions.schedule.OAuthMixin._init_oauth_client")
    def test_truncate_at_word_boundary(self, mock_init):
        handler = ScheduleActionHandler({"oauth_account_id": 123})
        prompt = "Send me a weather forecast for the upcoming week ahead"
        title = handler._generate_title(prompt)
        assert "..." in title
        # Should cut at a word boundary, not mid-word
        assert not title.endswith("upco...")
