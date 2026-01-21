"""
Email action plugin.

Handles email actions:
- draft_new, draft_reply, draft_forward (create drafts)
- send_new, send_reply, send_forward (send immediately)
- label (add/remove labels)
- archive (archive message)
- mark_read, mark_unread (toggle read status)

Supports Gmail via OAuth.
"""

import base64
import email
import email.encoders
import logging
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    ValidationError,
    ValidationResult,
)
from plugin_base.oauth import OAuthMixin

logger = logging.getLogger(__name__)


def _create_body_part(plain_text: str, html_text: str = None) -> MIMEBase:
    """
    Create a proper email body part with both plain text and HTML.

    If HTML is provided, creates a multipart/alternative with both versions.
    If only plain text, returns a simple text/plain part.
    """
    if html_text:
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_text, "plain"))
        alt.attach(MIMEText(html_text, "html"))
        return alt
    else:
        return MIMEText(plain_text, "plain")


def _plain_to_html(text: str) -> str:
    """Convert plain text to basic HTML (escape and convert newlines)."""
    import html

    escaped = html.escape(text)
    return escaped.replace("\n", "<br>\n")


class EmailActionHandler(OAuthMixin, PluginActionHandler):
    """
    Create drafts, send emails, manage labels and archive messages.

    Supports Gmail via OAuth authentication.
    """

    action_type = "email"
    display_name = "Gmail"
    description = "Create drafts, send emails, manage labels in Gmail"
    icon = "✉️"
    category = "communication"

    # Mark as non-abstract so it can be registered
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                help_text="Select a connected Google account with Gmail access",
                picker_options={"provider": "google"},
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="draft_new",
                description="Create a new email draft",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Recipient email addresses (comma-separated)",
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="bcc",
                        label="BCC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="subject",
                        label="Subject",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Body",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                    ),
                    FieldDefinition(
                        name="body_type",
                        label="Body Type",
                        field_type=FieldType.SELECT,
                        required=False,
                        default="text",
                        options=[
                            {"value": "text", "label": "Plain Text"},
                            {"value": "html", "label": "HTML"},
                        ],
                    ),
                ],
                examples=[
                    {
                        "to": ["recipient@example.com"],
                        "subject": "Meeting Tomorrow",
                        "body": "Hi,\n\nJust confirming our meeting tomorrow at 2pm.\n\nBest regards",
                    },
                ],
            ),
            ActionDefinition(
                name="draft_reply",
                description="Create a reply draft to an existing email",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="ID of the email to reply to",
                    ),
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Override recipients (required if reply_all is false)",
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Body",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                    ),
                    FieldDefinition(
                        name="reply_all",
                        label="Reply All",
                        field_type=FieldType.BOOLEAN,
                        required=False,
                        default=False,
                    ),
                ],
                examples=[
                    {
                        "message_id": "19bd01ea622a1040",
                        "to": ["sender@example.com"],
                        "body": "Thanks for the update!",
                        "reply_all": False,
                    },
                ],
            ),
            ActionDefinition(
                name="draft_forward",
                description="Create a forward draft of an existing email",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="ID of the email to forward",
                    ),
                    FieldDefinition(
                        name="subject_hint",
                        label="Subject Hint",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Partial subject to look up the email",
                    ),
                    FieldDefinition(
                        name="sender_hint",
                        label="Sender Hint",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Sender name or email to look up the email",
                    ),
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Forward recipients",
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Comment",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                        help_text="Optional comment before forwarded content",
                    ),
                ],
                examples=[
                    {
                        "message_id": "19bd01ea622a1040",
                        "to": ["colleague@example.com"],
                        "body": "FYI - see below",
                    },
                ],
            ),
            ActionDefinition(
                name="send_new",
                description="Send a new email immediately",
                risk=ActionRisk.MEDIUM,
                params=[
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="bcc",
                        label="BCC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="subject",
                        label="Subject",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Body",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                    ),
                ],
                examples=[
                    {
                        "to": ["recipient@example.com"],
                        "subject": "Quick question",
                        "body": "Hi, do you have a moment to chat?",
                    },
                ],
            ),
            ActionDefinition(
                name="send_reply",
                description="Send a reply immediately",
                risk=ActionRisk.MEDIUM,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Body",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                    ),
                    FieldDefinition(
                        name="reply_all",
                        label="Reply All",
                        field_type=FieldType.BOOLEAN,
                        required=False,
                        default=False,
                    ),
                ],
                examples=[
                    {
                        "message_id": "19bd01ea622a1040",
                        "to": ["sender@example.com"],
                        "body": "Sounds good, see you then!",
                    },
                ],
            ),
            ActionDefinition(
                name="send_forward",
                description="Forward an email immediately",
                risk=ActionRisk.MEDIUM,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="subject_hint",
                        label="Subject Hint",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="sender_hint",
                        label="Sender Hint",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="cc",
                        label="CC",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="body",
                        label="Comment",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                ],
                examples=[
                    {
                        "message_id": "19bd01ea622a1040",
                        "to": ["colleague@example.com"],
                        "body": "FYI",
                    },
                ],
            ),
            ActionDefinition(
                name="label",
                description="Add or remove labels from an email",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="add_labels",
                        label="Add Labels",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Labels to add (comma-separated)",
                    ),
                    FieldDefinition(
                        name="remove_labels",
                        label="Remove Labels",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Labels to remove (comma-separated)",
                    ),
                ],
                examples=[
                    {
                        "message_id": "19bd01ea622a1040",
                        "add_labels": ["STARRED", "IMPORTANT"],
                        "remove_labels": ["UNREAD"],
                    },
                ],
            ),
            ActionDefinition(
                name="archive",
                description="Archive an email (remove from inbox)",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[
                    {"message_id": "19bd01ea622a1040"},
                ],
            ),
            ActionDefinition(
                name="mark_read",
                description="Mark an email as read",
                risk=ActionRisk.READ_ONLY,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[
                    {"message_id": "19bd01ea622a1040"},
                ],
            ),
            ActionDefinition(
                name="mark_unread",
                description="Mark an email as unread",
                risk=ActionRisk.READ_ONLY,
                params=[
                    FieldDefinition(
                        name="message_id",
                        label="Message ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[
                    {"message_id": "19bd01ea622a1040"},
                ],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Custom LLM instructions for email actions."""
        return """## Email

**Message Identification - Two Options:**
1. Use the exact message ID if you have it (e.g., "ID: 19bd01ea622a1040")
2. Use subject_hint and/or sender_hint - the system will automatically look up the message

**Requirements:**
- Email body must be PLAIN TEXT only - no markdown, no asterisks for bold

**Action Selection - Draft vs Send:**
- "forward this email to X" or "send this to X" → use send_forward (sends immediately)
- "reply to this" or "tell them X" → use send_reply (sends immediately)
- "draft a forward" or "prepare a forward for review" → use draft_forward (saves to drafts)
- "write up a reply" or "draft a response" → use draft_reply (saves to drafts)
- When in doubt about user intent, prefer SEND actions for direct requests

### email:draft_new
Create a new email draft.

**Example:**
```xml
<smart_action type="email" action="draft_new">
{"to": ["recipient@example.com"], "cc": [], "subject": "Subject", "body": "Email body"}
</smart_action>
```

### email:draft_reply
Create a reply draft.

**Example:**
```xml
<smart_action type="email" action="draft_reply">
{"message_id": "19436abc123def", "to": ["sender@example.com"], "reply_all": false, "body": "Reply text"}
</smart_action>
```

### email:draft_forward
Create a forward draft.

**Example:**
```xml
<smart_action type="email" action="draft_forward">
{"message_id": "19436abc123def", "to": ["forward-to@example.com"], "body": "FYI - see below", "subject_hint": "Original subject line"}
</smart_action>
```

### email:send_new
Send a new email immediately.

**Example:**
```xml
<smart_action type="email" action="send_new">
{"to": ["recipient@example.com"], "subject": "Subject", "body": "Email body"}
</smart_action>
```

### email:send_reply
Send a reply immediately.

**Example:**
```xml
<smart_action type="email" action="send_reply">
{"message_id": "19436abc123def", "to": ["sender@example.com"], "reply_all": false, "body": "Reply text"}
</smart_action>
```

### email:send_forward
Forward an email immediately.

**Example:**
```xml
<smart_action type="email" action="send_forward">
{"message_id": "19436abc123def", "to": ["forward-to@example.com"], "body": "FYI", "subject_hint": "Original subject line"}
</smart_action>
```

### email:label
Add or remove labels.

**Example:**
```xml
<smart_action type="email" action="label">
{"message_id": "19436abc123def", "add_labels": ["STARRED", "IMPORTANT"], "remove_labels": ["UNREAD"]}
</smart_action>
```

### email:archive
Archive an email.

**Example:**
```xml
<smart_action type="email" action="archive">
{"message_id": "19436abc123def"}
</smart_action>
```

### email:mark_read / email:mark_unread
Mark email as read or unread.

**Example:**
```xml
<smart_action type="email" action="mark_read">
{"message_id": "19436abc123def"}
</smart_action>
```
"""

    def __init__(self, config: dict):
        """Initialize the email handler."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        # Initialize OAuth client if account configured
        if self.oauth_account_id:
            self._init_oauth_client()

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the email action."""
        if not self.oauth_account_id:
            return ActionResult(
                success=False,
                message="",
                error="OAuth account not configured",
            )

        try:
            if action == "draft_new":
                return self._create_draft(params)
            elif action == "draft_reply":
                return self._create_reply_draft(params)
            elif action == "draft_forward":
                return self._create_forward_draft(params, context)
            elif action == "send_new":
                return self._send_new(params)
            elif action == "send_reply":
                return self._send_reply(params)
            elif action == "send_forward":
                return self._send_forward(params, context)
            elif action == "label":
                return self._modify_labels(params)
            elif action == "archive":
                return self._archive(params)
            elif action == "mark_read":
                return self._mark_read(params, read=True)
            elif action == "mark_unread":
                return self._mark_read(params, read=False)
            else:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Unknown action: {action}",
                )
        except Exception as e:
            logger.exception(f"Email action failed: {action}")
            return ActionResult(
                success=False,
                message="",
                error=f"Action failed: {str(e)}",
            )

    def _normalize_recipients(self, recipients) -> list[str]:
        """Normalize recipients to a list of email addresses."""
        if isinstance(recipients, str):
            return [r.strip() for r in recipients.split(",") if r.strip()]
        elif isinstance(recipients, list):
            return recipients
        return []

    def _create_draft(self, params: dict) -> ActionResult:
        """Create a new email draft in Gmail."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        bcc = self._normalize_recipients(params.get("bcc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")
        body_type = params.get("body_type", "text")

        # Build the message with multipart/alternative (plain + HTML)
        if body_type == "html":
            html_body = body
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            plain_body = body
            html_body = _plain_to_html(body)

        message = MIMEMultipart("alternative")
        message.attach(MIMEText(plain_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
        message["to"] = ", ".join(to)
        if cc:
            message["cc"] = ", ".join(cc)
        if bcc:
            message["bcc"] = ", ".join(bcc)
        message["subject"] = subject

        # Encode for Gmail API
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Create draft
        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={"message": {"raw": raw}},
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail draft creation failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create draft: {response.status_code}",
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        logger.info(f"Created Gmail draft: {draft_id}")
        return ActionResult(
            success=True,
            message="Draft created successfully",
            data={
                "draft_id": draft_id,
                "to": to,
                "subject": subject,
            },
        )

    def _create_reply_draft(self, params: dict) -> ActionResult:
        """Create a reply draft in Gmail."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # Get the original message to get thread ID and headers
        orig_response = self.oauth_get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={
                "format": "metadata",
                "metadataHeaders": ["Subject", "From", "To", "Cc", "Message-ID"],
            },
        )

        if orig_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to get original message: {orig_response.status_code}",
            )

        orig_msg = orig_response.json()
        thread_id = orig_msg.get("threadId")

        # Extract headers
        headers = {
            h["name"].lower(): h["value"]
            for h in orig_msg.get("payload", {}).get("headers", [])
        }
        original_subject = headers.get("subject", "")
        original_message_id = headers.get("message-id", "")

        # Build reply subject
        subject = original_subject
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        # If reply_all, populate to/cc from original
        if reply_all:
            to = [headers.get("from", "")]
            original_to = headers.get("to", "")
            original_cc = headers.get("cc", "")
            if original_to:
                cc = [addr.strip() for addr in original_to.split(",")]
            if original_cc:
                cc.extend([addr.strip() for addr in original_cc.split(",")])

        # Build the message
        body_type = params.get("body_type", "text")
        if body_type == "html":
            html_body = body
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            plain_body = body
            html_body = _plain_to_html(body)

        message = MIMEMultipart("alternative")
        message.attach(MIMEText(plain_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
        message["to"] = ", ".join(to)
        if cc:
            message["cc"] = ", ".join(cc)
        message["subject"] = subject
        if original_message_id:
            message["In-Reply-To"] = original_message_id
            message["References"] = original_message_id

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Create draft with thread ID
        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={
                "message": {
                    "raw": raw,
                    "threadId": thread_id,
                }
            },
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail reply draft creation failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create reply draft: {response.status_code}",
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        logger.info(f"Created Gmail reply draft: {draft_id}")
        return ActionResult(
            success=True,
            message="Reply draft created successfully",
            data={
                "draft_id": draft_id,
                "thread_id": thread_id,
                "to": to,
                "subject": subject,
            },
        )

    def _create_forward_draft(
        self, params: dict, context: ActionContext
    ) -> ActionResult:
        """Create a forward draft in Gmail with original attachments."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        body = params.get("body", "")

        # Verify message ID exists, or try to look it up
        resolved_id, error = self._verify_or_lookup_message_id(
            message_id, params, context
        )
        if error:
            return ActionResult(
                success=False,
                message="",
                error=error,
            )
        message_id = resolved_id

        # Get the original message in RAW format
        orig_response = self.oauth_get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={"format": "raw"},
        )

        if orig_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to get original message: {orig_response.status_code}",
            )

        orig_data = orig_response.json()
        raw_email = orig_data.get("raw", "")

        # Decode and parse the raw email
        try:
            email_bytes = base64.urlsafe_b64decode(raw_email)
            original_email = email.message_from_bytes(email_bytes)
        except Exception as e:
            logger.error(f"Failed to parse original email: {e}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to parse original email: {str(e)}",
            )

        # Extract original headers
        original_subject = original_email.get("Subject", "")
        original_from = original_email.get("From", "")
        original_to = original_email.get("To", "")
        original_date = original_email.get("Date", "")

        # Build forward subject
        subject = original_subject
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}"

        # Extract body and attachments from original
        original_body, original_html, attachments = self._parse_email_parts(
            original_email
        )

        # Build forward header
        forward_header = f"\n\n---------- Forwarded message ----------\nFrom: {original_from}\nDate: {original_date}\nSubject: {original_subject}\nTo: {original_to}\n\n"

        # Create the new message
        if attachments:
            message = MIMEMultipart("mixed")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                combined_body = f"<p>{body}</p><hr>{html_forward_header}{original_html}"
                body_part = MIMEText(combined_body, "html")
            else:
                combined_body = body + forward_header + original_body
                body_part = MIMEText(combined_body, "plain")
            message.attach(body_part)

            for att in attachments:
                message.attach(att)

            logger.info(f"Forwarding email with {len(attachments)} attachment(s)")
        else:
            plain_forward = body + forward_header + (original_body or "")

            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                html_body_escaped = _plain_to_html(body) if body else ""
                html_forward = (
                    f"{html_body_escaped}<hr>{html_forward_header}{original_html}"
                )

                message = MIMEMultipart("alternative")
                message.attach(MIMEText(plain_forward, "plain"))
                message.attach(MIMEText(html_forward, "html"))
            else:
                message = MIMEText(plain_forward, "plain")

            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Create draft
        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={"message": {"raw": raw}},
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail forward draft creation failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create forward draft: {response.status_code}",
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        att_count = len(attachments)
        logger.info(
            f"Created Gmail forward draft: {draft_id} with {att_count} attachment(s)"
        )
        return ActionResult(
            success=True,
            message=f"Forward draft created"
            + (f" with {att_count} attachment(s)" if att_count else ""),
            data={
                "draft_id": draft_id,
                "to": to,
                "subject": subject,
                "attachments": att_count,
            },
        )

    def _send_new(self, params: dict) -> ActionResult:
        """Send a new email immediately via Gmail."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        bcc = self._normalize_recipients(params.get("bcc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")
        body_type = params.get("body_type", "text")

        if body_type == "html":
            html_body = body
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            plain_body = body
            html_body = _plain_to_html(body)

        message = MIMEMultipart("alternative")
        message.attach(MIMEText(plain_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
        message["to"] = ", ".join(to)
        if cc:
            message["cc"] = ", ".join(cc)
        if bcc:
            message["bcc"] = ", ".join(bcc)
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            json={"raw": raw},
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send email: {response.status_code}",
            )

        msg_data = response.json()
        message_id = msg_data.get("id")

        logger.info(f"Sent Gmail message: {message_id}")
        return ActionResult(
            success=True,
            message="Email sent successfully",
            data={
                "message_id": message_id,
                "to": to,
                "subject": subject,
            },
        )

    def _send_reply(self, params: dict) -> ActionResult:
        """Send a reply immediately via Gmail."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # Get original message headers
        orig_response = self.oauth_get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={
                "format": "metadata",
                "metadataHeaders": ["Subject", "From", "To", "Cc", "Message-ID"],
            },
        )

        if orig_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to get original message: {orig_response.status_code}",
            )

        orig_msg = orig_response.json()
        thread_id = orig_msg.get("threadId")

        headers = {
            h["name"].lower(): h["value"]
            for h in orig_msg.get("payload", {}).get("headers", [])
        }
        original_subject = headers.get("subject", "")
        original_message_id = headers.get("message-id", "")

        subject = original_subject
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        if reply_all:
            to = [headers.get("from", "")]
            original_to = headers.get("to", "")
            original_cc = headers.get("cc", "")
            if original_to:
                cc = [addr.strip() for addr in original_to.split(",")]
            if original_cc:
                cc.extend([addr.strip() for addr in original_cc.split(",")])

        body_type = params.get("body_type", "text")
        if body_type == "html":
            html_body = body
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            plain_body = body
            html_body = _plain_to_html(body)

        message = MIMEMultipart("alternative")
        message.attach(MIMEText(plain_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
        message["to"] = ", ".join(to)
        if cc:
            message["cc"] = ", ".join(cc)
        message["subject"] = subject
        if original_message_id:
            message["In-Reply-To"] = original_message_id
            message["References"] = original_message_id

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            json={"raw": raw, "threadId": thread_id},
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send reply failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send reply: {response.status_code}",
            )

        msg_data = response.json()
        new_message_id = msg_data.get("id")

        logger.info(f"Sent Gmail reply: {new_message_id}")
        return ActionResult(
            success=True,
            message="Reply sent successfully",
            data={
                "message_id": new_message_id,
                "thread_id": thread_id,
                "to": to,
                "subject": subject,
            },
        )

    def _send_forward(self, params: dict, context: ActionContext) -> ActionResult:
        """Send a forwarded email immediately via Gmail with original attachments."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        body = params.get("body", "")

        # Verify message ID exists, or try to look it up
        resolved_id, error = self._verify_or_lookup_message_id(
            message_id, params, context
        )
        if error:
            return ActionResult(
                success=False,
                message="",
                error=error,
            )
        message_id = resolved_id

        # Get original message in RAW format
        orig_response = self.oauth_get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={"format": "raw"},
        )

        if orig_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to get original message: {orig_response.status_code}",
            )

        orig_data = orig_response.json()
        raw_email = orig_data.get("raw", "")

        try:
            email_bytes = base64.urlsafe_b64decode(raw_email)
            original_email = email.message_from_bytes(email_bytes)
        except Exception as e:
            logger.error(f"Failed to parse original email: {e}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to parse original email: {str(e)}",
            )

        original_subject = original_email.get("Subject", "")
        original_from = original_email.get("From", "")
        original_to = original_email.get("To", "")
        original_date = original_email.get("Date", "")

        subject = original_subject
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}"

        original_body, original_html, attachments = self._parse_email_parts(
            original_email
        )

        forward_header = f"\n\n---------- Forwarded message ----------\nFrom: {original_from}\nDate: {original_date}\nSubject: {original_subject}\nTo: {original_to}\n\n"

        if attachments:
            message = MIMEMultipart("mixed")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                combined_body = f"<p>{body}</p><hr>{html_forward_header}{original_html}"
                body_part = MIMEText(combined_body, "html")
            else:
                combined_body = body + forward_header + original_body
                body_part = MIMEText(combined_body, "plain")
            message.attach(body_part)

            for att in attachments:
                message.attach(att)

            logger.info(f"Sending forward with {len(attachments)} attachment(s)")
        else:
            plain_forward = body + forward_header + (original_body or "")

            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                html_body_escaped = _plain_to_html(body) if body else ""
                html_forward = (
                    f"{html_body_escaped}<hr>{html_forward_header}{original_html}"
                )

                message = MIMEMultipart("alternative")
                message.attach(MIMEText(plain_forward, "plain"))
                message.attach(MIMEText(html_forward, "html"))
            else:
                message = MIMEText(plain_forward, "plain")

            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            json={"raw": raw},
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send forward failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send forward: {response.status_code}",
            )

        msg_data = response.json()
        new_message_id = msg_data.get("id")

        att_count = len(attachments)
        logger.info(
            f"Sent Gmail forward: {new_message_id} with {att_count} attachment(s)"
        )
        return ActionResult(
            success=True,
            message=f"Forward sent"
            + (f" with {att_count} attachment(s)" if att_count else ""),
            data={
                "message_id": new_message_id,
                "to": to,
                "subject": subject,
                "attachments": att_count,
            },
        )

    def _modify_labels(self, params: dict) -> ActionResult:
        """Add or remove labels from a Gmail message."""
        message_id = params.get("message_id")
        add_labels = params.get("add_labels", [])
        remove_labels = params.get("remove_labels", [])

        # Normalize to lists
        if isinstance(add_labels, str):
            add_labels = [l.strip() for l in add_labels.split(",") if l.strip()]
        if isinstance(remove_labels, str):
            remove_labels = [l.strip() for l in remove_labels.split(",") if l.strip()]

        # Gmail uses label IDs
        label_name_to_id = {
            "inbox": "INBOX",
            "starred": "STARRED",
            "important": "IMPORTANT",
            "sent": "SENT",
            "drafts": "DRAFT",
            "spam": "SPAM",
            "trash": "TRASH",
            "unread": "UNREAD",
            "read": "READ",
        }

        def resolve_label(label: str) -> str:
            lower = label.lower()
            if lower in label_name_to_id:
                return label_name_to_id[lower]
            return label

        add_ids = [resolve_label(l) for l in add_labels]
        remove_ids = [resolve_label(l) for l in remove_labels]

        # Handle "read" as removing UNREAD
        if "READ" in add_ids:
            add_ids.remove("READ")
            if "UNREAD" not in remove_ids:
                remove_ids.append("UNREAD")

        body = {}
        if add_ids:
            body["addLabelIds"] = add_ids
        if remove_ids:
            body["removeLabelIds"] = remove_ids

        response = self.oauth_post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            json=body,
        )

        if response.status_code != 200:
            logger.error(f"Gmail label modification failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to modify labels: {response.status_code}",
            )

        logger.info(f"Modified Gmail labels for message {message_id}")
        return ActionResult(
            success=True,
            message="Labels modified successfully",
            data={
                "message_id": message_id,
                "added": add_labels,
                "removed": remove_labels,
            },
        )

    def _archive(self, params: dict) -> ActionResult:
        """Archive a Gmail message by removing INBOX label."""
        message_id = params.get("message_id")

        response = self.oauth_post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            json={"removeLabelIds": ["INBOX"]},
        )

        if response.status_code != 200:
            logger.error(f"Gmail archive failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to archive message: {response.status_code}",
            )

        logger.info(f"Archived Gmail message {message_id}")
        return ActionResult(
            success=True,
            message="Message archived successfully",
            data={"message_id": message_id},
        )

    def _mark_read(self, params: dict, read: bool = True) -> ActionResult:
        """Mark a Gmail message as read or unread."""
        message_id = params.get("message_id")
        action_name = "mark_read" if read else "mark_unread"

        if read:
            body = {"removeLabelIds": ["UNREAD"]}
        else:
            body = {"addLabelIds": ["UNREAD"]}

        response = self.oauth_post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            json=body,
        )

        if response.status_code != 200:
            logger.error(f"Gmail {action_name} failed: {response.text}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to mark message as {'read' if read else 'unread'}: {response.status_code}",
            )

        logger.info(
            f"Marked Gmail message {message_id} as {'read' if read else 'unread'}"
        )
        return ActionResult(
            success=True,
            message=f"Message marked as {'read' if read else 'unread'}",
            data={"message_id": message_id},
        )

    def _parse_email_parts(self, msg: email.message.Message) -> tuple[str, str, list]:
        """
        Parse an email.message.Message to extract body and attachments.

        Returns:
            tuple: (plain_text_body, html_body, list_of_attachment_MIMEBase_objects)
        """
        plain_body = ""
        html_body = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if content_type.startswith("multipart/"):
                    continue

                if "attachment" in content_disposition or part.get_filename():
                    filename = part.get_filename() or "attachment"
                    payload = part.get_payload(decode=True)

                    if payload:
                        maintype, subtype = (
                            content_type.split("/", 1)
                            if "/" in content_type
                            else ("application", "octet-stream")
                        )
                        att = MIMEBase(maintype, subtype)
                        att.set_payload(payload)
                        email.encoders.encode_base64(att)
                        att.add_header(
                            "Content-Disposition", "attachment", filename=filename
                        )
                        attachments.append(att)
                elif content_type == "text/plain" and not plain_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        plain_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
                elif content_type == "text/html" and not html_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        html_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
        else:
            content_type = msg.get_content_type()
            try:
                charset = msg.get_content_charset() or "utf-8"
                body_text = msg.get_payload(decode=True).decode(
                    charset, errors="ignore"
                )
                if content_type == "text/html":
                    html_body = body_text
                else:
                    plain_body = body_text
            except Exception:
                pass

        return plain_body, html_body, attachments

    def _verify_or_lookup_message_id(
        self, message_id: str, params: dict, context: ActionContext
    ) -> tuple[str | None, str | None]:
        """
        Verify a message ID exists, or try to look it up from session cache.

        Returns:
            Tuple of (resolved_message_id, error_message)
        """
        if not message_id:
            # No message ID - try to look up from hints
            subject_hint = params.get("subject_hint") or params.get("subject", "")
            sender_hint = params.get("sender_hint") or params.get("from", "")

            if not subject_hint and not sender_hint:
                return None, "No message_id or subject/sender hint provided"

            # Try session cache lookup
            try:
                from live.sources import lookup_email_from_session_cache

                session_key = context.session_key if context else None
                account_email = (
                    self._get_account_email()
                    if hasattr(self, "_get_account_email")
                    else ""
                )

                result = lookup_email_from_session_cache(
                    session_key=session_key,
                    account_email=account_email,
                    subject_hint=subject_hint,
                    sender_hint=sender_hint,
                )

                if result:
                    cached_id = result.get("message_id")
                    logger.info(f"Session cache lookup found: {cached_id}")
                    return cached_id, None
            except Exception as e:
                logger.warning(f"Session cache lookup failed: {e}")

            return None, "Could not find message by subject/sender hint"

        # Verify the provided message_id exists
        check_response = self.oauth_get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={"format": "minimal"},
        )

        if check_response.status_code == 200:
            return message_id, None

        # Message ID failed - try to look up from session cache
        logger.warning(
            f"Message ID {message_id} not found (404), attempting session cache lookup"
        )

        subject_hint = params.get("subject_hint") or params.get("subject", "")
        sender_hint = params.get("sender_hint") or params.get("from", "")

        if not subject_hint and not sender_hint:
            return (
                None,
                f"Message ID {message_id} not found and no subject/sender hint available",
            )

        try:
            from live.sources import lookup_email_from_session_cache

            session_key = context.session_key if context else None
            account_email = (
                self._get_account_email() if hasattr(self, "_get_account_email") else ""
            )

            result = lookup_email_from_session_cache(
                session_key=session_key,
                account_email=account_email,
                subject_hint=subject_hint,
                sender_hint=sender_hint,
            )

            if result:
                cached_id = result.get("message_id")
                logger.info(f"Session cache lookup found: {cached_id}")

                # Verify this ID works
                verify_response = self.oauth_get(
                    f"https://www.googleapis.com/gmail/v1/users/me/messages/{cached_id}",
                    params={"format": "minimal"},
                )
                if verify_response.status_code == 200:
                    return cached_id, None
                else:
                    logger.warning(f"Cached message ID {cached_id} verification failed")

        except Exception as e:
            logger.warning(f"Session cache lookup failed: {e}")

        return (
            None,
            f"Message ID {message_id} not found and session cache lookup failed",
        )

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action in ("draft_new", "send_new"):
            to = params.get("to")
            if not to:
                errors.append(ValidationError("to", "Recipient list is required"))
            if not params.get("subject"):
                errors.append(ValidationError("subject", "Subject is required"))
            if not params.get("body"):
                errors.append(ValidationError("body", "Body is required"))

        elif action in ("draft_reply", "send_reply"):
            if not params.get("message_id"):
                errors.append(ValidationError("message_id", "Message ID is required"))
            if not params.get("body"):
                errors.append(ValidationError("body", "Body is required"))
            reply_all = params.get("reply_all", False)
            if not reply_all and not params.get("to"):
                errors.append(
                    ValidationError("to", "Recipients required when reply_all is false")
                )

        elif action in ("draft_forward", "send_forward"):
            message_id = params.get("message_id")
            subject_hint = params.get("subject_hint")
            sender_hint = params.get("sender_hint")

            if not message_id and not subject_hint and not sender_hint:
                errors.append(
                    ValidationError(
                        "message_id",
                        "Provide message_id OR subject_hint/sender_hint to identify the email",
                    )
                )
            if not params.get("to"):
                errors.append(ValidationError("to", "Forward recipients are required"))

        elif action == "label":
            if not params.get("message_id"):
                errors.append(ValidationError("message_id", "Message ID is required"))
            add_labels = params.get("add_labels", [])
            remove_labels = params.get("remove_labels", [])
            if not add_labels and not remove_labels:
                errors.append(
                    ValidationError(
                        "add_labels", "At least one label operation required"
                    )
                )

        elif action in ("archive", "mark_read", "mark_unread"):
            if not params.get("message_id"):
                errors.append(ValidationError("message_id", "Message ID is required"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "draft_new":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            subject = params.get("subject", "(no subject)")
            return f'Create email draft to {to}: "{subject}"'

        elif action == "draft_reply":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            reply_all = params.get("reply_all", False)
            if reply_all:
                return "Create reply-all draft"
            return f"Create reply draft to {to}"

        elif action == "draft_forward":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            return f"Create forward draft to {to}"

        elif action == "send_new":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            subject = params.get("subject", "(no subject)")
            return f'Send email to {to}: "{subject}"'

        elif action == "send_reply":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            reply_all = params.get("reply_all", False)
            if reply_all:
                return "Send reply-all"
            return f"Send reply to {to}"

        elif action == "send_forward":
            to = params.get("to", [])
            if isinstance(to, list):
                to = ", ".join(to)
            return f"Send forward to {to}"

        elif action == "label":
            message_id = params.get("message_id", "?")[:20]
            add = params.get("add_labels", [])
            remove = params.get("remove_labels", [])
            parts = []
            if add:
                parts.append(f"add {', '.join(add) if isinstance(add, list) else add}")
            if remove:
                parts.append(
                    f"remove {', '.join(remove) if isinstance(remove, list) else remove}"
                )
            return f"Modify labels on message {message_id}: {'; '.join(parts)}"

        elif action == "archive":
            message_id = params.get("message_id", "?")[:20]
            return f"Archive message {message_id}"

        elif action == "mark_read":
            message_id = params.get("message_id", "?")[:20]
            return f"Mark message {message_id} as read"

        elif action == "mark_unread":
            message_id = params.get("message_id", "?")[:20]
            return f"Mark message {message_id} as unread"

        return f"Email action: {action}"

    def is_available(self) -> bool:
        """Check if plugin is configured."""
        return bool(self.oauth_account_id)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection by getting Gmail profile."""
        if not self.oauth_account_id:
            return False, "OAuth account not configured"

        try:
            response = self.oauth_get(
                "https://www.googleapis.com/gmail/v1/users/me/profile"
            )
            response.raise_for_status()

            profile = response.json()
            email_addr = profile.get("emailAddress", "unknown")
            return True, f"Connected as {email_addr}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
