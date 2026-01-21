"""
Tests for the Email action plugin.
"""

import base64
from unittest.mock import MagicMock, Mock, patch

import pytest

from builtin_plugins.actions.email import EmailActionHandler
from plugin_base.action import ActionContext, ActionRisk
from plugin_base.common import FieldType


class TestEmailPluginMetadata:
    """Test plugin metadata."""

    def test_action_type(self):
        assert EmailActionHandler.action_type == "email"

    def test_display_name(self):
        assert EmailActionHandler.display_name == "Email"

    def test_category(self):
        assert EmailActionHandler.category == "communication"

    def test_not_abstract(self):
        assert EmailActionHandler._abstract is False

    def test_config_fields(self):
        fields = EmailActionHandler.get_config_fields()
        assert len(fields) == 1
        assert fields[0].name == "oauth_account_id"
        assert fields[0].field_type == FieldType.OAUTH_ACCOUNT

    def test_oauth_account_required(self):
        fields = EmailActionHandler.get_config_fields()
        oauth_field = next(f for f in fields if f.name == "oauth_account_id")
        assert oauth_field.required is True

    def test_actions(self):
        actions = EmailActionHandler.get_actions()
        action_names = [a.name for a in actions]
        assert "draft_new" in action_names
        assert "draft_reply" in action_names
        assert "draft_forward" in action_names
        assert "send_new" in action_names
        assert "send_reply" in action_names
        assert "send_forward" in action_names
        assert "label" in action_names
        assert "archive" in action_names
        assert "mark_read" in action_names
        assert "mark_unread" in action_names

    def test_action_risks(self):
        actions = EmailActionHandler.get_actions()
        action_map = {a.name: a for a in actions}

        # Drafts are low risk
        assert action_map["draft_new"].risk == ActionRisk.LOW
        assert action_map["draft_reply"].risk == ActionRisk.LOW
        assert action_map["draft_forward"].risk == ActionRisk.LOW

        # Sends are medium risk
        assert action_map["send_new"].risk == ActionRisk.MEDIUM
        assert action_map["send_reply"].risk == ActionRisk.MEDIUM
        assert action_map["send_forward"].risk == ActionRisk.MEDIUM

        # Label/archive are low risk
        assert action_map["label"].risk == ActionRisk.LOW
        assert action_map["archive"].risk == ActionRisk.LOW

        # Mark read/unread are read-only
        assert action_map["mark_read"].risk == ActionRisk.READ_ONLY
        assert action_map["mark_unread"].risk == ActionRisk.READ_ONLY


class TestEmailPluginValidation:
    """Test plugin config validation."""

    def test_valid_config(self):
        result = EmailActionHandler.validate_config({"oauth_account_id": 123})
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_oauth_account(self):
        result = EmailActionHandler.validate_config({})
        assert result.valid is False
        assert any("oauth_account_id" in str(e.field) for e in result.errors)


class TestEmailInit:
    """Test plugin initialization."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_init_with_oauth_account(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        assert handler.oauth_account_id == 123
        assert handler.oauth_provider == "google"
        mock_init.assert_called_once()

    def test_init_without_oauth_account(self):
        handler = EmailActionHandler({})
        assert handler.oauth_account_id is None


class TestEmailDraftNew:
    """Test draft_new action."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_create_draft(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "draft123"}

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "draft_new",
                {
                    "to": ["test@example.com"],
                    "subject": "Test Subject",
                    "body": "Test body content",
                },
                context,
            )

            assert result.success is True
            assert "draft123" in str(result.data)

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_create_draft_with_cc_bcc(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "draft123"}

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "draft_new",
                {
                    "to": ["test@example.com"],
                    "cc": ["cc@example.com"],
                    "bcc": ["bcc@example.com"],
                    "subject": "Test Subject",
                    "body": "Test body",
                },
                context,
            )

            assert result.success is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_create_draft_failure(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "draft_new",
                {"to": ["test@example.com"], "subject": "Test", "body": "Body"},
                context,
            )

            assert result.success is False
            assert "400" in result.error


class TestEmailDraftReply:
    """Test draft_reply action."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_create_reply_draft(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        # Mock getting original message
        orig_response = Mock()
        orig_response.status_code = 200
        orig_response.json.return_value = {
            "threadId": "thread123",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Original Subject"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Message-ID", "value": "<msg123@example.com>"},
                ]
            },
        }

        # Mock creating draft
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {"id": "draft456"}

        with patch.object(handler, "oauth_get", return_value=orig_response):
            with patch.object(handler, "oauth_post", return_value=create_response):
                context = ActionContext()
                result = handler.execute(
                    "draft_reply",
                    {
                        "message_id": "msg123",
                        "to": ["sender@example.com"],
                        "body": "Reply content",
                    },
                    context,
                )

                assert result.success is True
                assert "draft456" in str(result.data)

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_reply_all(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        orig_response = Mock()
        orig_response.status_code = 200
        orig_response.json.return_value = {
            "threadId": "thread123",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Original"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "me@example.com, other@example.com"},
                    {"name": "Cc", "value": "cc@example.com"},
                    {"name": "Message-ID", "value": "<msg123@example.com>"},
                ]
            },
        }

        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {"id": "draft789"}

        with patch.object(handler, "oauth_get", return_value=orig_response):
            with patch.object(handler, "oauth_post", return_value=create_response):
                context = ActionContext()
                result = handler.execute(
                    "draft_reply",
                    {"message_id": "msg123", "body": "Reply to all", "reply_all": True},
                    context,
                )

                assert result.success is True


class TestEmailSendNew:
    """Test send_new action."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_send_email(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "sent123"}

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "send_new",
                {
                    "to": ["recipient@example.com"],
                    "subject": "Important Message",
                    "body": "Hello!",
                },
                context,
            )

            assert result.success is True
            assert "sent123" in str(result.data)
            assert "sent" in result.message.lower()


class TestEmailLabels:
    """Test label action."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_add_labels(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "label",
                {"message_id": "msg123", "add_labels": ["STARRED", "IMPORTANT"]},
                context,
            )

            assert result.success is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_remove_labels(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute(
                "label", {"message_id": "msg123", "remove_labels": ["UNREAD"]}, context
            )

            assert result.success is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_label_name_resolution(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            handler, "oauth_post", return_value=mock_response
        ) as mock_post:
            context = ActionContext()
            handler.execute(
                "label",
                {
                    "message_id": "msg123",
                    "add_labels": ["starred"],  # lowercase
                    "remove_labels": ["unread"],  # lowercase
                },
                context,
            )

            # Check that labels were resolved to uppercase
            call_args = mock_post.call_args
            json_arg = call_args[1]["json"]
            assert "STARRED" in json_arg.get("addLabelIds", [])
            assert "UNREAD" in json_arg.get("removeLabelIds", [])


class TestEmailArchive:
    """Test archive action."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_archive_message(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(handler, "oauth_post", return_value=mock_response):
            context = ActionContext()
            result = handler.execute("archive", {"message_id": "msg123"}, context)

            assert result.success is True
            assert "archived" in result.message.lower()


class TestEmailMarkRead:
    """Test mark_read and mark_unread actions."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_mark_read(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            handler, "oauth_post", return_value=mock_response
        ) as mock_post:
            context = ActionContext()
            result = handler.execute("mark_read", {"message_id": "msg123"}, context)

            assert result.success is True
            # Check UNREAD was removed
            call_args = mock_post.call_args
            json_arg = call_args[1]["json"]
            assert "UNREAD" in json_arg.get("removeLabelIds", [])

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_mark_unread(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            handler, "oauth_post", return_value=mock_response
        ) as mock_post:
            context = ActionContext()
            result = handler.execute("mark_unread", {"message_id": "msg123"}, context)

            assert result.success is True
            # Check UNREAD was added
            call_args = mock_post.call_args
            json_arg = call_args[1]["json"]
            assert "UNREAD" in json_arg.get("addLabelIds", [])


class TestEmailErrorHandling:
    """Test error handling."""

    def test_no_oauth_configured(self):
        handler = EmailActionHandler({})
        context = ActionContext()
        result = handler.execute(
            "draft_new",
            {"to": ["test@example.com"], "subject": "Test", "body": "Test"},
            context,
        )

        assert result.success is False
        assert "not configured" in result.error.lower()

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_unknown_action(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        context = ActionContext()
        result = handler.execute("invalid_action", {}, context)

        assert result.success is False
        assert "unknown" in result.error.lower()


class TestEmailApprovalSummary:
    """Test approval summary generation."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_draft_new_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "draft_new", {"to": ["recipient@example.com"], "subject": "Important"}
        )
        assert "draft" in summary.lower()
        assert "recipient@example.com" in summary
        assert "Important" in summary

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_send_new_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "send_new", {"to": ["recipient@example.com"], "subject": "Quick Note"}
        )
        assert "send" in summary.lower()
        assert "recipient@example.com" in summary

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_reply_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "draft_reply", {"to": ["sender@example.com"], "reply_all": False}
        )
        assert "reply" in summary.lower()

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_reply_all_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary("draft_reply", {"reply_all": True})
        assert "reply-all" in summary.lower()

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_label_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary(
            "label",
            {
                "message_id": "msg123",
                "add_labels": ["STARRED"],
                "remove_labels": ["UNREAD"],
            },
        )
        assert "label" in summary.lower()
        assert "STARRED" in summary

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_archive_summary(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        summary = handler.get_approval_summary("archive", {"message_id": "msg123"})
        assert "archive" in summary.lower()


class TestEmailParamValidation:
    """Test parameter validation."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_draft_new_valid(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_new", {"to": ["test@example.com"], "subject": "Test", "body": "Body"}
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_draft_new_missing_to(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_new", {"subject": "Test", "body": "Body"}
        )
        assert result.valid is False
        assert any("to" in str(e.field) for e in result.errors)

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_draft_new_missing_subject(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_new", {"to": ["test@example.com"], "body": "Body"}
        )
        assert result.valid is False
        assert any("subject" in str(e.field) for e in result.errors)

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_reply_valid(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_reply",
            {"message_id": "msg123", "to": ["sender@example.com"], "body": "Reply"},
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_reply_missing_message_id(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_reply", {"to": ["sender@example.com"], "body": "Reply"}
        )
        assert result.valid is False

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_forward_valid_with_id(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_forward", {"message_id": "msg123", "to": ["forward@example.com"]}
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_forward_valid_with_hints(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_forward",
            {"subject_hint": "Meeting notes", "to": ["forward@example.com"]},
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_forward_missing_identifier(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "draft_forward", {"to": ["forward@example.com"]}
        )
        assert result.valid is False

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_label_valid(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params(
            "label", {"message_id": "msg123", "add_labels": ["STARRED"]}
        )
        assert result.valid is True

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_validate_label_no_operations(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler.validate_action_params("label", {"message_id": "msg123"})
        assert result.valid is False


class TestEmailTestConnection:
    """Test connection testing."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_connection_success(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"emailAddress": "test@example.com"}
        mock_response.raise_for_status = Mock()

        with patch.object(handler, "oauth_get", return_value=mock_response):
            success, message = handler.test_connection()
            assert success is True
            assert "test@example.com" in message

    def test_connection_no_oauth(self):
        handler = EmailActionHandler({})
        success, message = handler.test_connection()
        assert success is False
        assert "not configured" in message.lower()


class TestEmailLLMInstructions:
    """Test LLM instruction generation."""

    def test_instructions_contain_actions(self):
        instructions = EmailActionHandler.get_llm_instructions()
        assert "draft_new" in instructions
        assert "draft_reply" in instructions
        assert "draft_forward" in instructions
        assert "send_new" in instructions
        assert "send_reply" in instructions
        assert "label" in instructions
        assert "archive" in instructions
        assert "mark_read" in instructions

    def test_instructions_contain_guidance(self):
        instructions = EmailActionHandler.get_llm_instructions()
        assert "Message Identification" in instructions
        assert "subject_hint" in instructions
        assert "PLAIN TEXT" in instructions

    def test_instructions_contain_examples(self):
        instructions = EmailActionHandler.get_llm_instructions()
        assert "<smart_action" in instructions
        assert 'type="email"' in instructions
        assert 'type="email"' in instructions


class TestEmailAvailability:
    """Test availability checks."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_available_when_configured(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        assert handler.is_available() is True

    def test_not_available_without_oauth(self):
        handler = EmailActionHandler({})
        assert handler.is_available() is False


class TestEmailRecipientNormalization:
    """Test recipient normalization."""

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_normalize_string_recipients(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler._normalize_recipients("a@test.com, b@test.com, c@test.com")
        assert result == ["a@test.com", "b@test.com", "c@test.com"]

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_normalize_list_recipients(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler._normalize_recipients(["a@test.com", "b@test.com"])
        assert result == ["a@test.com", "b@test.com"]

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_normalize_empty(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler._normalize_recipients("")
        assert result == []

    @patch("builtin_plugins.actions.email.OAuthMixin._init_oauth_client")
    def test_normalize_none(self, mock_init):
        handler = EmailActionHandler({"oauth_account_id": 123})
        result = handler._normalize_recipients(None)
        assert result == []
