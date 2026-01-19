"""
Email Action Handler.

Handles email actions:
- draft_new, draft_reply, draft_forward (create drafts)
- send_new, send_reply, send_forward (send immediately)
- label (add/remove labels)
- archive (archive message)

Supports Gmail and Outlook via OAuth.
"""

import base64
import email
import email.encoders
import logging
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from actions.base import ActionContext, ActionHandler, ActionResult, ActionStatus

logger = logging.getLogger(__name__)


def _create_body_part(plain_text: str, html_text: str = None) -> MIMEBase:
    """
    Create a proper email body part with both plain text and HTML.

    If HTML is provided, creates a multipart/alternative with both versions.
    If only plain text, returns a simple text/plain part.

    Args:
        plain_text: Plain text version of the body
        html_text: Optional HTML version of the body

    Returns:
        MIMEBase object (either MIMEText or MIMEMultipart)
    """
    if html_text:
        # Create multipart/alternative with both plain and HTML
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


class EmailActionHandler(ActionHandler):
    """
    Handler for email actions.

    Supports creating drafts in Gmail and Outlook.
    """

    @property
    def action_type(self) -> str:
        return "email"

    @property
    def supported_actions(self) -> list[str]:
        return [
            "draft_new",
            "draft_reply",
            "draft_forward",
            "send_new",
            "send_reply",
            "send_forward",
            "label",
            "archive",
            "mark_read",
            "mark_unread",
        ]

    @property
    def requires_oauth(self) -> bool:
        return True

    @property
    def oauth_provider(self) -> Optional[str]:
        # Can work with either Google or Microsoft
        return None

    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate email action parameters."""

        # Check action is supported
        if action not in self.supported_actions:
            return False, f"Unknown email action: {action}"

        # Get account - use default if not specified
        account = params.get("account")
        if not account:
            # Use default email account if configured
            default_email = context.default_accounts.get("email", {})
            if default_email.get("email"):
                account = default_email["email"]
                params["account"] = account  # Update params with default
                logger.info(f"Using default email account: {account}")
            else:
                return (
                    False,
                    "Missing required parameter: account (email address). No default account configured.",
                )

        # Find the OAuth account
        oauth_account = self._find_oauth_account(account, context)
        if not oauth_account:
            available = self._get_available_accounts(context)
            return (
                False,
                f"No OAuth account found for '{account}'. Available: {available}",
            )

        # Validate based on action type
        if action == "draft_new" or action == "send_new":
            return self._validate_draft_new(params)
        elif action == "draft_reply" or action == "send_reply":
            return self._validate_draft_reply(params)
        elif action == "draft_forward" or action == "send_forward":
            return self._validate_draft_forward(params)
        elif action == "label":
            return self._validate_label(params)
        elif action == "archive":
            return self._validate_archive(params)
        elif action in ("mark_read", "mark_unread"):
            return self._validate_mark_read(params)

        return False, f"Validation not implemented for action: {action}"

    def _validate_label(self, params: dict) -> tuple[bool, str]:
        """Validate label action parameters."""
        message_id = params.get("message_id")
        if not message_id:
            return False, "Missing required parameter: message_id"

        add_labels = params.get("add_labels", [])
        remove_labels = params.get("remove_labels", [])

        if not add_labels and not remove_labels:
            return False, "Must specify at least one of: add_labels, remove_labels"

        if add_labels and not isinstance(add_labels, list):
            return False, "Parameter 'add_labels' must be a list"
        if remove_labels and not isinstance(remove_labels, list):
            return False, "Parameter 'remove_labels' must be a list"

        return True, ""

    def _validate_archive(self, params: dict) -> tuple[bool, str]:
        """Validate archive action parameters."""
        message_id = params.get("message_id")
        if not message_id:
            return False, "Missing required parameter: message_id"

        return True, ""

    def _validate_mark_read(self, params: dict) -> tuple[bool, str]:
        """Validate mark_read/mark_unread action parameters."""
        message_id = params.get("message_id")
        if not message_id:
            return False, "Missing required parameter: message_id"

        return True, ""

    def _validate_draft_new(self, params: dict) -> tuple[bool, str]:
        """Validate draft_new parameters."""
        to = params.get("to")
        if not to:
            return False, "Missing required parameter: to (recipient list)"
        if not isinstance(to, list):
            return False, "Parameter 'to' must be a list of email addresses"
        if len(to) == 0:
            return False, "Parameter 'to' must contain at least one recipient"

        subject = params.get("subject")
        if not subject:
            return False, "Missing required parameter: subject"

        body = params.get("body")
        if not body:
            return False, "Missing required parameter: body"

        return True, ""

    def _validate_draft_reply(self, params: dict) -> tuple[bool, str]:
        """Validate draft_reply parameters."""
        message_id = params.get("message_id")
        if not message_id:
            return (
                False,
                "Missing required parameter: message_id (ID of email to reply to)",
            )

        body = params.get("body")
        if not body:
            return False, "Missing required parameter: body"

        # If not reply_all, need explicit 'to' list
        reply_all = params.get("reply_all", False)
        if not reply_all:
            to = params.get("to")
            if not to:
                return (
                    False,
                    "Missing required parameter: to (required when reply_all is false)",
                )
            if not isinstance(to, list) or len(to) == 0:
                return False, "Parameter 'to' must be a non-empty list"

        return True, ""

    def _validate_draft_forward(self, params: dict) -> tuple[bool, str]:
        """Validate draft_forward parameters."""
        message_id = params.get("message_id")
        subject_hint = params.get("subject_hint")
        sender_hint = params.get("sender_hint")

        # Need either message_id OR a hint to look up the email
        if not message_id and not subject_hint and not sender_hint:
            return (
                False,
                "Missing message identifier: provide message_id OR subject_hint/sender_hint to identify the email",
            )

        to = params.get("to")
        if not to:
            return False, "Missing required parameter: to (forward recipients)"
        if not isinstance(to, list) or len(to) == 0:
            return False, "Parameter 'to' must be a non-empty list"

        # Body is optional for forward (can just forward without comment)
        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the email action."""

        account = params.get("account")
        oauth_account = self._find_oauth_account(account, context)

        if not oauth_account:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message=f"OAuth account not found: {account}",
            )

        provider = oauth_account.get("provider", "google")
        account_id = oauth_account.get("id")

        try:
            if provider == "google":
                return self._execute_gmail(action, params, account_id, context)
            elif provider == "microsoft":
                return self._execute_outlook(action, params, account_id, context)
            else:
                return ActionResult(
                    status=ActionStatus.FAILED,
                    action_type=self.action_type,
                    action=action,
                    message=f"Unsupported email provider: {provider}",
                )
        except Exception as e:
            logger.exception(f"Email action failed: {e}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message=f"Action failed: {str(e)}",
            )

    def _execute_gmail(
        self, action: str, params: dict, account_id: int, context: ActionContext
    ) -> ActionResult:
        """Execute action via Gmail API."""
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message="OAuth token not found",
            )

        access_token = self._get_gmail_access_token(token_data)
        if not access_token:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message="Failed to get valid access token",
            )

        if action == "draft_new":
            return self._gmail_create_draft(access_token, params)
        elif action == "draft_reply":
            return self._gmail_create_reply_draft(access_token, params)
        elif action == "draft_forward":
            return self._gmail_create_forward_draft(access_token, params, context)
        elif action == "send_new":
            return self._gmail_send_new(access_token, params)
        elif action == "send_reply":
            return self._gmail_send_reply(access_token, params)
        elif action == "send_forward":
            return self._gmail_send_forward(access_token, params, context)
        elif action == "label":
            return self._gmail_modify_labels(access_token, params)
        elif action == "archive":
            return self._gmail_archive(access_token, params)
        elif action == "mark_read":
            return self._gmail_mark_read(access_token, params, read=True)
        elif action == "mark_unread":
            return self._gmail_mark_read(access_token, params, read=False)

        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message=f"Action not implemented: {action}",
        )

    def _get_gmail_access_token(self, token_data: dict) -> Optional[str]:
        """Get valid Gmail access token, refreshing if needed."""
        import requests as http_requests

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return None

        # Test the token
        test_response = http_requests.get(
            "https://www.googleapis.com/gmail/v1/users/me/profile",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error("Cannot refresh Gmail token - missing credentials")
            return None

        refresh_response = http_requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if refresh_response.status_code != 200:
            logger.error(f"Gmail token refresh failed: {refresh_response.text}")
            return None

        new_token = refresh_response.json().get("access_token")

        # Update stored token
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {**token_data, "access_token": new_token}
        update_oauth_token_data(token_data.get("id"), updated_data)

        return new_token

    def _gmail_create_draft(self, access_token: str, params: dict) -> ActionResult:
        """Create a new email draft in Gmail."""
        import requests as http_requests

        to = params.get("to", [])
        cc = params.get("cc", [])
        bcc = params.get("bcc", [])
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
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"message": {"raw": raw}},
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail draft creation failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_new",
                message=f"Failed to create draft: {response.status_code}",
                details={"error": response.text},
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        logger.info(f"Created Gmail draft: {draft_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="draft_new",
            message=f"Draft created successfully",
            details={
                "draft_id": draft_id,
                "to": to,
                "subject": subject,
            },
        )

    def _gmail_create_reply_draft(
        self, access_token: str, params: dict
    ) -> ActionResult:
        """Create a reply draft in Gmail."""
        import requests as http_requests

        message_id = params.get("message_id")
        to = params.get("to", [])
        cc = params.get("cc", [])
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # First, get the original message to get thread ID and headers
        orig_response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "format": "metadata",
                "metadataHeaders": ["Subject", "From", "To", "Cc", "Message-ID"],
            },
            timeout=30,
        )

        if orig_response.status_code != 200:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_reply",
                message=f"Failed to get original message: {orig_response.status_code}",
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

        # Build the message with multipart/alternative (plain + HTML)
        body_type = params.get("body_type", "text")

        if body_type == "html":
            # User provided HTML - use it directly and generate plain text
            html_body = body
            # Strip HTML tags for plain text version (simple approach)
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            # User provided plain text - generate HTML version
            plain_body = body
            html_body = _plain_to_html(body)

        # Create multipart/alternative message
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

        # Encode for Gmail API
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Create draft with thread ID
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "message": {
                    "raw": raw,
                    "threadId": thread_id,
                }
            },
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail reply draft creation failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_reply",
                message=f"Failed to create reply draft: {response.status_code}",
                details={"error": response.text},
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        logger.info(f"Created Gmail reply draft: {draft_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="draft_reply",
            message=f"Reply draft created successfully",
            details={
                "draft_id": draft_id,
                "thread_id": thread_id,
                "to": to,
                "subject": subject,
            },
        )

    def _gmail_create_forward_draft(
        self, access_token: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Create a forward draft in Gmail with original attachments."""
        import requests as http_requests

        message_id = params.get("message_id")
        to = params.get("to", [])
        cc = params.get("cc", [])
        body = params.get("body", "")  # Optional comment before forwarded content

        # Verify message ID exists, or try to look it up from cache
        resolved_id, error = self._verify_or_lookup_message_id(
            access_token, message_id, params, context
        )
        if error:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_forward",
                message=error,
            )
        message_id = resolved_id

        # Get the original message in RAW format (full RFC 2822 with attachments)
        orig_response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"format": "raw"},
            timeout=60,
        )

        if orig_response.status_code != 200:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_forward",
                message=f"Failed to get original message: {orig_response.status_code}",
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
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_forward",
                message=f"Failed to parse original email: {str(e)}",
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
            # Multipart message with attachments
            message = MIMEMultipart("mixed")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

            # Add body (prefer HTML if available for better formatting)
            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                combined_body = f"<p>{body}</p><hr>{html_forward_header}{original_html}"
                body_part = MIMEText(combined_body, "html")
            else:
                combined_body = body + forward_header + original_body
                body_part = MIMEText(combined_body, "plain")
            message.attach(body_part)

            # Add all original attachments
            for att in attachments:
                message.attach(att)

            logger.info(f"Forwarding email with {len(attachments)} attachment(s)")
        else:
            # Simple message without attachments - use multipart/alternative if HTML available
            plain_forward = body + forward_header + (original_body or "")

            if original_html:
                # Build HTML version with forwarded content
                html_forward_header = forward_header.replace("\n", "<br>")
                html_body_escaped = _plain_to_html(body) if body else ""
                html_forward = (
                    f"{html_body_escaped}<hr>{html_forward_header}{original_html}"
                )

                # Create multipart/alternative with both plain and HTML
                message = MIMEMultipart("alternative")
                message.attach(MIMEText(plain_forward, "plain"))
                message.attach(MIMEText(html_forward, "html"))
                logger.info("Forwarding as multipart/alternative (plain + HTML)")
            else:
                # Plain text only email
                message = MIMEText(plain_forward, "plain")
                logger.info("Forwarding as plain text only")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

        # Encode for Gmail API
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Create draft
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"message": {"raw": raw}},
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail forward draft creation failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="draft_forward",
                message=f"Failed to create forward draft: {response.status_code}",
                details={"error": response.text},
            )

        draft_data = response.json()
        draft_id = draft_data.get("id")

        att_count = len(attachments)
        logger.info(
            f"Created Gmail forward draft: {draft_id} with {att_count} attachment(s)"
        )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="draft_forward",
            message=f"Forward draft created"
            + (f" with {att_count} attachment(s)" if att_count else ""),
            details={
                "draft_id": draft_id,
                "to": to,
                "subject": subject,
                "attachments": att_count,
            },
        )

    def _gmail_send_new(self, access_token: str, params: dict) -> ActionResult:
        """Send a new email immediately via Gmail."""
        import requests as http_requests

        to = params.get("to", [])
        cc = params.get("cc", [])
        bcc = params.get("bcc", [])
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

        # Send message
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"raw": raw},
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_new",
                message=f"Failed to send email: {response.status_code}",
                details={"error": response.text},
            )

        msg_data = response.json()
        message_id = msg_data.get("id")

        logger.info(f"Sent Gmail message: {message_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="send_new",
            message=f"Email sent successfully",
            details={
                "message_id": message_id,
                "to": to,
                "subject": subject,
            },
        )

    def _gmail_send_reply(self, access_token: str, params: dict) -> ActionResult:
        """Send a reply immediately via Gmail."""
        import requests as http_requests

        message_id = params.get("message_id")
        to = params.get("to", [])
        cc = params.get("cc", [])
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # Get original message headers
        orig_response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "format": "metadata",
                "metadataHeaders": ["Subject", "From", "To", "Cc", "Message-ID"],
            },
            timeout=30,
        )

        if orig_response.status_code != 200:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_reply",
                message=f"Failed to get original message: {orig_response.status_code}",
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

        # Build the message with multipart/alternative (plain + HTML)
        body_type = params.get("body_type", "text")

        if body_type == "html":
            # User provided HTML - use it directly and generate plain text
            html_body = body
            import re

            plain_body = re.sub(r"<[^>]+>", "", body)
        else:
            # User provided plain text - generate HTML version
            plain_body = body
            html_body = _plain_to_html(body)

        # Create multipart/alternative message
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

        # Send with thread ID
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"raw": raw, "threadId": thread_id},
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send reply failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_reply",
                message=f"Failed to send reply: {response.status_code}",
                details={"error": response.text},
            )

        msg_data = response.json()
        new_message_id = msg_data.get("id")

        logger.info(f"Sent Gmail reply: {new_message_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="send_reply",
            message=f"Reply sent successfully",
            details={
                "message_id": new_message_id,
                "thread_id": thread_id,
                "to": to,
                "subject": subject,
            },
        )

    def _gmail_send_forward(
        self, access_token: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Send a forwarded email immediately via Gmail with original attachments."""
        import requests as http_requests

        message_id = params.get("message_id")
        to = params.get("to", [])
        cc = params.get("cc", [])
        body = params.get("body", "")

        # Verify message ID exists, or try to look it up from cache
        resolved_id, error = self._verify_or_lookup_message_id(
            access_token, message_id, params, context
        )
        if error:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_forward",
                message=error,
            )
        message_id = resolved_id

        # Get original message in RAW format (full RFC 2822 with attachments)
        orig_response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"format": "raw"},
            timeout=60,
        )

        if orig_response.status_code != 200:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_forward",
                message=f"Failed to get original message: {orig_response.status_code}",
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
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_forward",
                message=f"Failed to parse original email: {str(e)}",
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
            # Multipart message with attachments
            message = MIMEMultipart("mixed")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

            # Add body (prefer HTML if available for better formatting)
            if original_html:
                html_forward_header = forward_header.replace("\n", "<br>")
                combined_body = f"<p>{body}</p><hr>{html_forward_header}{original_html}"
                body_part = MIMEText(combined_body, "html")
            else:
                combined_body = body + forward_header + original_body
                body_part = MIMEText(combined_body, "plain")
            message.attach(body_part)

            # Add all original attachments
            for att in attachments:
                message.attach(att)

            logger.info(f"Sending forward with {len(attachments)} attachment(s)")
        else:
            # Simple message without attachments - use multipart/alternative if HTML available
            plain_forward = body + forward_header + (original_body or "")

            if original_html:
                # Build HTML version with forwarded content
                html_forward_header = forward_header.replace("\n", "<br>")
                html_body_escaped = _plain_to_html(body) if body else ""
                html_forward = (
                    f"{html_body_escaped}<hr>{html_forward_header}{original_html}"
                )

                # Create multipart/alternative with both plain and HTML
                message = MIMEMultipart("alternative")
                message.attach(MIMEText(plain_forward, "plain"))
                message.attach(MIMEText(html_forward, "html"))
                logger.info("Sending forward as multipart/alternative (plain + HTML)")
            else:
                # Plain text only email
                message = MIMEText(plain_forward, "plain")
                logger.info("Sending forward as plain text only")
            message["to"] = ", ".join(to)
            if cc:
                message["cc"] = ", ".join(cc)
            message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Send message
        response = http_requests.post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"raw": raw},
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Gmail send forward failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send_forward",
                message=f"Failed to send forward: {response.status_code}",
                details={"error": response.text},
            )

        msg_data = response.json()
        new_message_id = msg_data.get("id")

        att_count = len(attachments)
        logger.info(
            f"Sent Gmail forward: {new_message_id} with {att_count} attachment(s)"
        )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="send_forward",
            message=f"Forward sent"
            + (f" with {att_count} attachment(s)" if att_count else ""),
            details={
                "message_id": new_message_id,
                "to": to,
                "subject": subject,
                "attachments": att_count,
            },
        )

    def _gmail_modify_labels(self, access_token: str, params: dict) -> ActionResult:
        """Add or remove labels from a Gmail message."""
        import requests as http_requests

        message_id = params.get("message_id")
        add_labels = params.get("add_labels", [])
        remove_labels = params.get("remove_labels", [])

        # Gmail uses label IDs, but users might provide names
        # Common label mappings (case-insensitive)
        label_name_to_id = {
            "inbox": "INBOX",
            "starred": "STARRED",
            "important": "IMPORTANT",
            "sent": "SENT",
            "drafts": "DRAFT",
            "spam": "SPAM",
            "trash": "TRASH",
            "unread": "UNREAD",
            "read": "READ",  # Not a real label, but useful alias
        }

        def resolve_label(label: str) -> str:
            """Resolve label name to ID."""
            lower = label.lower()
            if lower in label_name_to_id:
                return label_name_to_id[lower]
            # Assume it's already a label ID or custom label name
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

        response = http_requests.post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Gmail label modification failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="label",
                message=f"Failed to modify labels: {response.status_code}",
                details={"error": response.text},
            )

        logger.info(f"Modified Gmail labels for message {message_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="label",
            message=f"Labels modified successfully",
            details={
                "message_id": message_id,
                "added": add_labels,
                "removed": remove_labels,
            },
        )

    def _gmail_archive(self, access_token: str, params: dict) -> ActionResult:
        """Archive a Gmail message by removing INBOX label."""
        import requests as http_requests

        message_id = params.get("message_id")

        response = http_requests.post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"removeLabelIds": ["INBOX"]},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Gmail archive failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="archive",
                message=f"Failed to archive message: {response.status_code}",
                details={"error": response.text},
            )

        logger.info(f"Archived Gmail message {message_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="archive",
            message=f"Message archived successfully",
            details={"message_id": message_id},
        )

    def _gmail_mark_read(
        self, access_token: str, params: dict, read: bool = True
    ) -> ActionResult:
        """Mark a Gmail message as read or unread."""
        import requests as http_requests

        message_id = params.get("message_id")
        action_name = "mark_read" if read else "mark_unread"

        # UNREAD is a label - remove it to mark as read, add it to mark as unread
        if read:
            body = {"removeLabelIds": ["UNREAD"]}
        else:
            body = {"addLabelIds": ["UNREAD"]}

        response = http_requests.post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Gmail {action_name} failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action_name,
                message=f"Failed to mark message as {'read' if read else 'unread'}: {response.status_code}",
                details={"error": response.text},
            )

        logger.info(
            f"Marked Gmail message {message_id} as {'read' if read else 'unread'}"
        )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action=action_name,
            message=f"Message marked as {'read' if read else 'unread'}",
            details={"message_id": message_id},
        )

    def _extract_gmail_body(self, payload: dict) -> str:
        """Extract plain text body from Gmail message payload."""
        import base64

        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        parts = payload.get("parts", [])

        # Direct body data
        if body.get("data"):
            try:
                return base64.urlsafe_b64decode(body["data"]).decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                pass

        # Multipart - look for text/plain
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    part_body = part.get("body", {})
                    if part_body.get("data"):
                        try:
                            return base64.urlsafe_b64decode(part_body["data"]).decode(
                                "utf-8", errors="ignore"
                            )
                        except Exception:
                            pass

            # Recurse into nested parts
            for part in parts:
                nested = self._extract_gmail_body(part)
                if nested:
                    return nested

        return ""

    def _parse_email_parts(self, msg: email.message.Message) -> tuple[str, str, list]:
        """
        Parse an email.message.Message to extract body and attachments.

        Returns:
            tuple: (plain_text_body, html_body, list_of_attachment_MIMEBase_objects)
        """
        plain_body = ""
        html_body = ""
        attachments = []

        logger.info(
            f"Parsing email: is_multipart={msg.is_multipart()}, content_type={msg.get_content_type()}"
        )

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                logger.info(
                    f"  Part: type={content_type}, disposition={content_disposition[:50] if content_disposition else 'None'}, filename={part.get_filename()}"
                )

                # Skip multipart containers
                if content_type.startswith("multipart/"):
                    continue

                # Check if it's an attachment
                if "attachment" in content_disposition or part.get_filename():
                    # It's an attachment - preserve it
                    filename = part.get_filename() or "attachment"
                    payload = part.get_payload(decode=True)

                    if payload:
                        # Create a new MIME attachment
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
            # Not multipart - just get the body
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

        logger.info(
            f"Parsed email: plain_body={len(plain_body)} chars, html_body={len(html_body)} chars, attachments={len(attachments)}"
        )
        return plain_body, html_body, attachments

    def _execute_outlook(
        self, action: str, params: dict, account_id: int, context: ActionContext
    ) -> ActionResult:
        """Execute action via Outlook/Microsoft Graph API."""
        # TODO: Implement Outlook draft creation
        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message="Outlook actions not yet implemented",
        )

    def _find_oauth_account(self, email: str, context: ActionContext) -> Optional[dict]:
        """Find OAuth account by email address."""
        email_lower = email.lower()

        for provider, accounts in context.oauth_accounts.items():
            for account in accounts:
                if account.get("email", "").lower() == email_lower:
                    return {"provider": provider, **account}

        return None

    def _get_available_accounts(self, context: ActionContext) -> str:
        """Get comma-separated list of available email accounts."""
        emails = []
        for provider, accounts in context.oauth_accounts.items():
            for account in accounts:
                email = account.get("email", "")
                if email:
                    emails.append(email)
        return ", ".join(emails) if emails else "none"

    def _verify_or_lookup_message_id(
        self, access_token: str, message_id: str, params: dict, context: ActionContext
    ) -> tuple[str | None, str | None]:
        """
        Verify a message ID exists, or try to look it up from session cache.

        LLMs sometimes hallucinate message IDs. This method:
        1. First tries the provided message_id
        2. If 404, tries to look up the correct ID from session cache by subject/sender

        Uses session-scoped caching to prevent email matching from leaking across sessions
        and to return the most recently viewed email when there are duplicates.

        Returns:
            Tuple of (resolved_message_id, error_message)
            - If success: (message_id, None)
            - If failure: (None, error_message)
        """
        import requests as http_requests

        # First, verify the provided message_id exists
        check_response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"format": "minimal"},
            timeout=15,
        )

        if check_response.status_code == 200:
            return message_id, None

        # Message ID failed - try to look up from session cache
        logger.warning(
            f"Message ID {message_id} not found (404), attempting session cache lookup"
        )

        # Try to find the message by subject from params
        subject_hint = params.get("subject_hint") or params.get("subject", "")
        sender_hint = params.get("sender_hint") or params.get("from", "")
        account_email = params.get("account", "")

        if not subject_hint and not sender_hint:
            return (
                None,
                f"Message ID {message_id} not found and no subject/sender hint available for lookup",
            )

        try:
            from live.sources import lookup_email_from_session_cache

            # Get session key from context
            session_key = context.session_key if context else None

            # Look up from session cache
            result = lookup_email_from_session_cache(
                session_key=session_key,
                account_email=account_email,
                subject_hint=subject_hint,
                sender_hint=sender_hint,
            )

            if result:
                cached_id = result.get("message_id")
                logger.info(
                    f"Session cache lookup found: {cached_id} "
                    f"(subject: '{subject_hint[:30] if subject_hint else 'N/A'}', "
                    f"sender: '{sender_hint[:30] if sender_hint else 'N/A'}')"
                )
                # Verify this ID works
                verify_response = http_requests.get(
                    f"https://www.googleapis.com/gmail/v1/users/me/messages/{cached_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"format": "minimal"},
                    timeout=15,
                )
                if verify_response.status_code == 200:
                    return cached_id, None
                else:
                    logger.warning(
                        f"Cached message ID {cached_id} verification failed: {verify_response.status_code}"
                    )

        except Exception as e:
            logger.warning(f"Session cache lookup failed: {e}")

        return (
            None,
            f"Message ID {message_id} not found and session cache lookup failed",
        )

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        account = params.get("account", "unknown")

        if action == "draft_new":
            to = params.get("to", [])
            subject = params.get("subject", "(no subject)")
            return f'Create email draft from {account} to {", ".join(to)}: "{subject}"'

        elif action == "draft_reply":
            to = params.get("to", [])
            reply_all = params.get("reply_all", False)
            if reply_all:
                return f"Create reply-all draft from {account}"
            return f"Create reply draft from {account} to {', '.join(to)}"

        elif action == "draft_forward":
            to = params.get("to", [])
            return f"Create forward draft from {account} to {', '.join(to)}"

        elif action == "send_new":
            to = params.get("to", [])
            subject = params.get("subject", "(no subject)")
            return f'Send email from {account} to {", ".join(to)}: "{subject}"'

        elif action == "send_reply":
            to = params.get("to", [])
            reply_all = params.get("reply_all", False)
            if reply_all:
                return f"Send reply-all from {account}"
            return f"Send reply from {account} to {', '.join(to)}"

        elif action == "send_forward":
            to = params.get("to", [])
            return f"Send forward from {account} to {', '.join(to)}"

        elif action == "label":
            message_id = params.get("message_id", "?")
            add = params.get("add_labels", [])
            remove = params.get("remove_labels", [])
            parts = []
            if add:
                parts.append(f"add {', '.join(add)}")
            if remove:
                parts.append(f"remove {', '.join(remove)}")
            return f"Modify labels on message {message_id}: {'; '.join(parts)}"

        elif action == "archive":
            message_id = params.get("message_id", "?")
            return f"Archive message {message_id}"

        elif action == "mark_read":
            message_id = params.get("message_id", "?")
            return f"Mark message {message_id} as read"

        elif action == "mark_unread":
            message_id = params.get("message_id", "?")
            return f"Mark message {message_id} as unread"

        return f"Email action: {action}"

    def get_system_prompt_instructions(self) -> str:
        """Get email-specific instructions for system prompt."""
        return """- email: draft_new, draft_reply, draft_forward, send_new, send_reply, send_forward, label, archive, mark_read, mark_unread

  **Message Identification - Two Options:**
  1. Use the exact message ID if you have it (e.g., "ID: 19bd01ea622a1040")
  2. Use subject_hint and/or sender_hint - the system will automatically look up the message from recently viewed emails

  For reply/forward actions, you can identify the email using EITHER:
  - "message_id": "exact_id_here" (if you have it)
  - "subject_hint": "partial subject text" and/or "sender_hint": "sender name or email" (system looks it up)

  The system caches recently viewed emails, so if the user just listed emails and asks to forward one, use subject_hint with the subject text - no need to search again.

  **Requirements:**
  - Use the "account" field matching the email's account (e.g., "account": "user@gmail.com")
  - The account MUST match where the message is stored
  - Email body must be PLAIN TEXT only - no markdown, no asterisks for bold

  **Action Selection - Draft vs Send:**
  - "forward this email to X" or "send this to X" → use send_forward (sends immediately)
  - "reply to this" or "tell them X" → use send_reply (sends immediately)
  - "draft a forward" or "prepare a forward for review" → use draft_forward (saves to drafts)
  - "write up a reply" or "draft a response" → use draft_reply (saves to drafts)
  - When in doubt about user intent, prefer SEND actions for direct requests

  **Drafts** (saves to drafts folder for user review - use when user says "draft", "prepare", or wants to review first):
  ```xml
  <smart_action type="email" action="draft_new">
  {"to": ["recipient@example.com"], "cc": [], "subject": "Subject", "body": "Email body"}
  </smart_action>
  ```

  ```xml
  <smart_action type="email" action="draft_reply">
  {"account": "user@gmail.com", "message_id": "19436abc123def", "to": ["sender@example.com"], "reply_all": false, "body": "Reply text"}
  </smart_action>
  ```

  ```xml
  <smart_action type="email" action="draft_forward">
  {"account": "user@gmail.com", "message_id": "19436abc123def", "to": ["forward-to@example.com"], "body": "FYI - see below", "subject_hint": "Original subject line"}
  </smart_action>
  ```

  **Send** (sends immediately - use for direct action requests like "forward this", "reply to them", "send X"):
  ```xml
  <smart_action type="email" action="send_new">
  {"to": ["recipient@example.com"], "subject": "Subject", "body": "Email body"}
  </smart_action>
  ```

  ```xml
  <smart_action type="email" action="send_reply">
  {"account": "user@gmail.com", "message_id": "19436abc123def", "to": ["sender@example.com"], "reply_all": false, "body": "Reply text"}
  </smart_action>
  ```

  ```xml
  <smart_action type="email" action="send_forward">
  {"account": "user@gmail.com", "message_id": "19436abc123def", "to": ["forward-to@example.com"], "body": "FYI", "subject_hint": "Original subject line"}
  </smart_action>
  ```

  **Labels** (add/remove labels - use Gmail label names like "STARRED", "IMPORTANT", or custom labels):
  ```xml
  <smart_action type="email" action="label">
  {"account": "user@gmail.com", "message_id": "19436abc123def", "add_labels": ["STARRED", "IMPORTANT"], "remove_labels": ["UNREAD"]}
  </smart_action>
  ```

  **Archive** (removes from inbox):
  ```xml
  <smart_action type="email" action="archive">
  {"account": "user@gmail.com", "message_id": "19436abc123def"}
  </smart_action>
  ```

  **Mark read/unread**:
  ```xml
  <smart_action type="email" action="mark_read">
  {"account": "user@gmail.com", "message_id": "19436abc123def"}
  </smart_action>
  ```

  ```xml
  <smart_action type="email" action="mark_unread">
  {"account": "user@gmail.com", "message_id": "19436abc123def"}
  </smart_action>
  ```"""
