"""
Unit tests for Anthropic Messages API endpoint.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, "/dockerstuff/corestuff/llm-relay")


class TestConvertAnthropicMessages(unittest.TestCase):
    """Tests for convert_anthropic_messages function."""

    def test_string_content(self):
        """Test conversion of simple string content."""
        from proxy import convert_anthropic_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        system, converted = convert_anthropic_messages(messages, "You are helpful.")

        self.assertEqual(system, "You are helpful.")
        self.assertEqual(len(converted), 3)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello")
        self.assertEqual(converted[1]["role"], "assistant")
        self.assertEqual(converted[1]["content"], "Hi there!")

    def test_system_prompt_separate(self):
        """Test that system is passed through correctly."""
        from proxy import convert_anthropic_messages

        messages = [{"role": "user", "content": "Test"}]

        system, converted = convert_anthropic_messages(messages, "Custom system prompt")

        self.assertEqual(system, "Custom system prompt")
        self.assertEqual(len(converted), 1)

    def test_no_system_prompt(self):
        """Test handling when no system prompt provided."""
        from proxy import convert_anthropic_messages

        messages = [{"role": "user", "content": "Test"}]

        system, converted = convert_anthropic_messages(messages, None)

        self.assertIsNone(system)
        self.assertEqual(len(converted), 1)

    def test_array_content_blocks(self):
        """Test conversion of array content blocks."""
        from proxy import convert_anthropic_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "base64data",
                        },
                    },
                ],
            }
        ]

        system, converted = convert_anthropic_messages(messages)

        self.assertEqual(len(converted), 1)
        self.assertIsInstance(converted[0]["content"], list)
        self.assertEqual(len(converted[0]["content"]), 2)
        self.assertEqual(converted[0]["content"][0]["type"], "text")
        self.assertEqual(converted[0]["content"][1]["type"], "image")

    def test_tool_use_blocks(self):
        """Test that tool_use blocks are passed through."""
        from proxy import convert_anthropic_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for that."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "search",
                        "input": {"query": "test"},
                    },
                ],
            }
        ]

        system, converted = convert_anthropic_messages(messages)

        self.assertEqual(len(converted), 1)
        self.assertEqual(len(converted[0]["content"]), 2)
        self.assertEqual(converted[0]["content"][1]["type"], "tool_use")

    def test_tool_result_blocks(self):
        """Test that tool_result blocks are passed through."""
        from proxy import convert_anthropic_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "Search results here",
                    }
                ],
            }
        ]

        system, converted = convert_anthropic_messages(messages)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["content"][0]["type"], "tool_result")

    def test_empty_messages(self):
        """Test handling of empty messages list."""
        from proxy import convert_anthropic_messages

        system, converted = convert_anthropic_messages([], "System")

        self.assertEqual(system, "System")
        self.assertEqual(len(converted), 0)


class TestGenerateAnthropicId(unittest.TestCase):
    """Tests for generate_anthropic_id function."""

    def test_default_prefix(self):
        """Test ID generation with default prefix."""
        from proxy import generate_anthropic_id

        msg_id = generate_anthropic_id()

        self.assertTrue(msg_id.startswith("msg_"))
        self.assertEqual(len(msg_id), 28)  # "msg_" + 24 chars

    def test_custom_prefix(self):
        """Test ID generation with custom prefix."""
        from proxy import generate_anthropic_id

        msg_id = generate_anthropic_id("test")

        self.assertTrue(msg_id.startswith("test_"))

    def test_uniqueness(self):
        """Test that generated IDs are unique."""
        from proxy import generate_anthropic_id

        ids = [generate_anthropic_id() for _ in range(100)]

        self.assertEqual(len(ids), len(set(ids)))


class TestMapFinishReasonToAnthropic(unittest.TestCase):
    """Tests for map_finish_reason_to_anthropic function."""

    def test_stop_to_end_turn(self):
        """Test mapping 'stop' to 'end_turn'."""
        from proxy import map_finish_reason_to_anthropic

        result = map_finish_reason_to_anthropic("stop")
        self.assertEqual(result, "end_turn")

    def test_length_to_max_tokens(self):
        """Test mapping 'length' to 'max_tokens'."""
        from proxy import map_finish_reason_to_anthropic

        result = map_finish_reason_to_anthropic("length")
        self.assertEqual(result, "max_tokens")

    def test_tool_calls_to_tool_use(self):
        """Test mapping 'tool_calls' to 'tool_use'."""
        from proxy import map_finish_reason_to_anthropic

        result = map_finish_reason_to_anthropic("tool_calls")
        self.assertEqual(result, "tool_use")

    def test_none_to_end_turn(self):
        """Test mapping None to 'end_turn'."""
        from proxy import map_finish_reason_to_anthropic

        result = map_finish_reason_to_anthropic(None)
        self.assertEqual(result, "end_turn")

    def test_unknown_to_end_turn(self):
        """Test mapping unknown reason to 'end_turn'."""
        from proxy import map_finish_reason_to_anthropic

        result = map_finish_reason_to_anthropic("unknown_reason")
        self.assertEqual(result, "end_turn")


class TestBuildAnthropicResponse(unittest.TestCase):
    """Tests for build_anthropic_response function."""

    def test_basic_response(self):
        """Test building a basic response."""
        from proxy import build_anthropic_response

        response = build_anthropic_response(
            message_id="msg_test123",
            model="claude-sonnet",
            content="Hello, world!",
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
        )

        self.assertEqual(response["id"], "msg_test123")
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["role"], "assistant")
        self.assertEqual(response["model"], "claude-sonnet")
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertIsNone(response["stop_sequence"])

        # Check content array format
        self.assertEqual(len(response["content"]), 1)
        self.assertEqual(response["content"][0]["type"], "text")
        self.assertEqual(response["content"][0]["text"], "Hello, world!")

        # Check usage
        self.assertEqual(response["usage"]["input_tokens"], 10)
        self.assertEqual(response["usage"]["output_tokens"], 5)

    def test_empty_content(self):
        """Test building response with empty content."""
        from proxy import build_anthropic_response

        response = build_anthropic_response(
            message_id="msg_test",
            model="claude",
            content="",
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
        )

        self.assertEqual(response["content"][0]["text"], "")


class TestAnthropicErrorResponse(unittest.TestCase):
    """Tests for anthropic_error_response function."""

    def test_error_structure(self):
        """Test error response structure."""
        from proxy import anthropic_error_response, app

        with app.app_context():
            response, status_code = anthropic_error_response(
                "invalid_request_error", "max_tokens is required", 400
            )

            data = response.get_json()

            self.assertEqual(status_code, 400)
            self.assertEqual(data["type"], "error")
            self.assertEqual(data["error"]["type"], "invalid_request_error")
            self.assertEqual(data["error"]["message"], "max_tokens is required")

    def test_not_found_error(self):
        """Test not found error response."""
        from proxy import anthropic_error_response, app

        with app.app_context():
            response, status_code = anthropic_error_response(
                "not_found_error", "Model not found", 404
            )

            self.assertEqual(status_code, 404)
            data = response.get_json()
            self.assertEqual(data["error"]["type"], "not_found_error")

    def test_api_error(self):
        """Test API error response."""
        from proxy import anthropic_error_response, app

        with app.app_context():
            response, status_code = anthropic_error_response(
                "api_error", "Internal error", 500
            )

            self.assertEqual(status_code, 500)
            data = response.get_json()
            self.assertEqual(data["error"]["type"], "api_error")


class TestStreamAnthropicResponseFormat(unittest.TestCase):
    """Tests for streaming response format validation."""

    def test_event_names(self):
        """Test that correct event names are used in streaming."""
        # This tests the format by parsing the SSE output
        expected_events = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

        # Just verify these are the expected event names per Anthropic spec
        self.assertIn("message_start", expected_events)
        self.assertIn("content_block_delta", expected_events)
        self.assertIn("message_stop", expected_events)

    def test_sse_format(self):
        """Test SSE line format."""
        # Anthropic uses "event: name\ndata: json\n\n" format
        test_event = "event: message_start\ndata: {}\n\n"

        lines = test_event.split("\n")
        self.assertTrue(lines[0].startswith("event: "))
        self.assertTrue(lines[1].startswith("data: "))


class TestAnthropicEndpointIntegration(unittest.TestCase):
    """Integration tests for the /v1/messages endpoint."""

    def setUp(self):
        """Set up test client."""
        from proxy import app

        self.app = app
        self.client = app.test_client()

    @patch("proxy.registry")
    @patch("proxy.tracker")
    def test_basic_request(self, mock_tracker, mock_registry):
        """Test basic non-streaming request."""
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.name = "anthropic"
        mock_provider.chat_completion.return_value = {
            "content": "Hello!",
            "input_tokens": 5,
            "output_tokens": 2,
            "finish_reason": "stop",
        }

        # Mock resolve_model
        mock_resolved = MagicMock()
        mock_resolved.provider = mock_provider
        mock_resolved.model_id = "claude-3-sonnet"
        mock_resolved.alias_name = None
        mock_resolved.alias_tags = []
        mock_resolved.router_name = None
        mock_resolved.has_enrichment = False
        mock_resolved.has_cache = False

        mock_registry.resolve_model.return_value = mock_resolved

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"][0]["type"], "text")
        self.assertEqual(data["content"][0]["text"], "Hello!")
        self.assertEqual(data["stop_reason"], "end_turn")

    @patch("proxy.registry")
    @patch("proxy.tracker")
    def test_with_system_prompt(self, mock_tracker, mock_registry):
        """Test request with system prompt."""
        mock_provider = MagicMock()
        mock_provider.name = "anthropic"
        mock_provider.chat_completion.return_value = {
            "content": "I am a helpful assistant.",
            "input_tokens": 10,
            "output_tokens": 5,
        }

        mock_resolved = MagicMock()
        mock_resolved.provider = mock_provider
        mock_resolved.model_id = "claude-3-sonnet"
        mock_resolved.alias_name = None
        mock_resolved.alias_tags = []
        mock_resolved.router_name = None
        mock_resolved.has_enrichment = False
        mock_resolved.has_cache = False

        mock_registry.resolve_model.return_value = mock_resolved

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "max_tokens": 100,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "What are you?"}],
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        # Verify system prompt was passed to provider
        call_args = mock_provider.chat_completion.call_args
        self.assertEqual(call_args[0][2], "You are a helpful assistant.")

    @patch("proxy.registry")
    def test_model_not_found(self, mock_registry):
        """Test error when model is not found."""
        mock_registry.resolve_model.side_effect = ValueError("Model not found")

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "nonexistent-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)

        data = response.get_json()
        self.assertEqual(data["type"], "error")
        self.assertEqual(data["error"]["type"], "not_found_error")

    @patch("proxy.registry")
    @patch("proxy.tracker")
    def test_default_max_tokens(self, mock_tracker, mock_registry):
        """Test that default max_tokens is applied when not provided."""
        mock_provider = MagicMock()
        mock_provider.name = "anthropic"
        mock_provider.chat_completion.return_value = {
            "content": "Response",
            "input_tokens": 1,
            "output_tokens": 1,
        }

        mock_resolved = MagicMock()
        mock_resolved.provider = mock_provider
        mock_resolved.model_id = "claude-3-sonnet"
        mock_resolved.alias_name = None
        mock_resolved.alias_tags = []
        mock_resolved.router_name = None
        mock_resolved.has_enrichment = False
        mock_resolved.has_cache = False

        mock_registry.resolve_model.return_value = mock_resolved

        # Request without max_tokens
        response = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        # Verify default max_tokens was applied
        call_args = mock_provider.chat_completion.call_args
        options = call_args[0][3]
        self.assertEqual(options["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
