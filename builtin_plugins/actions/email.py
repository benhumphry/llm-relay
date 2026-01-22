"""
Unified Email action plugin.

Handles email actions across multiple providers:
- Gmail (via OAuth)
- Outlook (via OAuth)

The LLM uses generic actions (draft_new, send_new, etc.)
and the system routes to the configured provider based on Smart Alias settings.

NO CONFIG FIELDS - all configuration comes from Smart Alias context at runtime:
- default_accounts["email"]["id"] = OAuth account ID
- default_accounts["email"]["provider"] = "google" or "microsoft"
"""

import base64
import email
import email.encoders
import logging
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
    ResourceRequirement,
    ResourceType,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    ValidationError,
    ValidationResult,
)
from plugin_base.oauth import OAuthMixin

logger = logging.getLogger(__name__)


def _plain_to_html(text: str) -> str:
    """Convert plain text to basic HTML (escape and convert newlines)."""
    import html

    escaped = html.escape(text)
    return escaped.replace("\n", "<br>\n")


class EmailActionHandler(OAuthMixin, PluginActionHandler):
    """
    Unified email management across Gmail and Outlook.

    Provider is determined at runtime from Smart Alias context - no plugin config needed.
    """

    action_type = "email"
    display_name = "Email"
    description = "Create drafts, send emails, manage labels (Gmail or Outlook)"
    icon = "✉️"
    category = "communication"
    supported_sources = ["Gmail", "Outlook"]

    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """No config fields - everything comes from Smart Alias context."""
        return []

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """Define resources needed from Smart Alias."""
        return [
            ResourceRequirement(
                key="email",
                label="Email Account",
                resource_type=ResourceType.OAUTH_ACCOUNT,
                providers=["google", "microsoft"],
                help_text="Account for email actions (Gmail or Outlook)",
                required=True,
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
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Forward recipients",
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
                        name="to",
                        label="To",
                        field_type=FieldType.TEXT,
                        required=True,
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
                        "add_labels": ["STARRED"],
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
                examples=[{"message_id": "19bd01ea622a1040"}],
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
                examples=[{"message_id": "19bd01ea622a1040"}],
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
                examples=[{"message_id": "19bd01ea622a1040"}],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Static LLM instructions for email actions (no account info)."""
        return cls._build_instructions(None)

    def get_llm_instructions_with_context(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """Dynamic LLM instructions with available accounts listed."""
        return self._build_instructions(available_accounts)

    @staticmethod
    def _build_instructions(
        available_accounts: Optional[dict[str, list[dict]]] = None,
    ) -> str:
        """Build LLM instructions, optionally with account info."""
        # Build account selection text if accounts available
        account_text = ""
        if available_accounts:
            email_accounts = available_accounts.get("email", [])
            if email_accounts:
                account_list = []
                for acc in email_accounts:
                    email = acc.get("email", "")
                    name = acc.get("name", "")  # Store name (friendly name)
                    provider = acc.get("provider", "")
                    # Show store name as primary identifier, email in parentheses
                    if name:
                        provider_note = f", {provider}" if provider else ""
                        account_list.append(f'  - "{name}" ({email}{provider_note})')
                    elif email:
                        account_list.append(f'  - "{email}" ({provider})')
                if account_list:
                    account_text = f"""
**Available Email Accounts:**
{chr(10).join(account_list)}

To use a specific account, add `"account": "Account Name"` or `"account": "email@example.com"`.
If only one account is available, it will be used automatically.
"""

        return f"""## Email
{account_text}
**Message Identification - Two Options:**
1. Use the exact message ID if you have it (e.g., "ID: 19bd01ea622a1040")
2. Use subject_hint - the system will automatically look up the message by subject

**Requirements:**
- Email body must be PLAIN TEXT only - no markdown, no asterisks for bold
- YOU MUST include a <smart_action> block to actually perform the action - just saying "I'll forward it" does nothing!

**Action Selection - Draft vs Send:**
- "forward this email to X" or "send this to X" -> use send_forward (sends immediately)
- "reply to this" or "tell them X" -> use send_reply (sends immediately)
- "draft a forward" or "prepare a forward for review" -> use draft_forward (saves to drafts)
- "write up a reply" or "draft a response" -> use draft_reply (saves to drafts)
- When in doubt about user intent, prefer SEND actions for direct requests

### email:send_forward - Forward an email immediately
REQUIRED when user says "forward this to X", "send this to X", etc.

**Example (with message ID):**
```xml
<smart_action type="email" action="send_forward">
{{"message_id": "19bd01ea622a1040", "to": ["colleague@example.com"], "body": "FYI - see the details below"}}
</smart_action>
```

**Example (with subject hint - system will find the email):**
```xml
<smart_action type="email" action="send_forward">
{{"subject_hint": "Octopus Energy", "to": ["colleague@example.com"], "body": "Here's the statement"}}
</smart_action>
```

### email:send_reply - Reply to an email immediately
**Example:**
```xml
<smart_action type="email" action="send_reply">
{{"message_id": "19bd01ea622a1040", "body": "Thanks for letting me know!"}}
</smart_action>
```

### email:send_new - Send a new email
**Example:**
```xml
<smart_action type="email" action="send_new">
{{"to": ["recipient@example.com"], "subject": "Quick question", "body": "Hi, do you have a moment to chat?"}}
</smart_action>
```

### email:draft_new / email:draft_reply / email:draft_forward
Same as above but saves to drafts instead of sending.

### email:archive - Archive an email
**Example:**
```xml
<smart_action type="email" action="archive">
{{"subject_hint": "Password Confirmation"}}
</smart_action>
```

### email:mark_read / email:mark_unread - Mark email read status
**Example:**
```xml
<smart_action type="email" action="mark_read">
{{"subject_hint": "Meeting notes"}}
</smart_action>
```

### email:label - Add/remove labels (requires message_id)
"""

    def __init__(self, config: dict):
        """Initialize the email handler - config is ignored, uses context."""
        # These will be set from context at execution time
        self.oauth_account_id: Optional[int] = None
        self.oauth_provider: Optional[str] = None

    def _find_account(
        self, account_identifier: str, context: ActionContext
    ) -> Optional[dict]:
        """
        Find an account by email address or name.

        Args:
            account_identifier: Email address or account name to find
            context: Action context with available_accounts

        Returns:
            Account dict if found, None otherwise
        """
        available = context.available_accounts.get("email", [])
        identifier_lower = account_identifier.lower()

        for account in available:
            email = account.get("email", "").lower()
            name = account.get("name", "").lower()
            if email == identifier_lower or name == identifier_lower:
                return account
            # Also match partial email (before @)
            if email and identifier_lower == email.split("@")[0]:
                return account

        return None

    def _get_available_accounts_str(self, context: ActionContext) -> str:
        """Get comma-separated list of available email accounts."""
        available = context.available_accounts.get("email", [])
        emails = [a.get("email", a.get("name", "unknown")) for a in available]
        return ", ".join(emails) if emails else "none"

    def _configure_from_params_or_context(
        self, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """
        Configure provider from params (account field) or fall back to default.

        The account can be specified by:
        - Store name (friendly name from document store)
        - Email address
        - Partial email (before @)

        Returns:
            (success, error_message)
        """
        # Check if LLM specified an account
        account_param = params.get("account")

        if account_param:
            # Look up the specified account
            account = self._find_account(account_param, context)
            if not account:
                available = self._get_available_accounts_str(context)
                return (
                    False,
                    f"Account '{account_param}' not found. Available: {available}",
                )

            # Store-based accounts use oauth_account_id, legacy uses id
            self.oauth_account_id = account.get("oauth_account_id", account.get("id"))
            self.oauth_provider = account.get("provider", "google")
        else:
            # Fall back to default account
            default_accounts = getattr(context, "default_accounts", {})
            email_config = default_accounts.get("email", {})

            if not email_config:
                # Check if there's exactly one available account - use it as default
                available_accounts = context.available_accounts.get("email", [])
                if len(available_accounts) == 1:
                    account = available_accounts[0]
                    self.oauth_account_id = account.get(
                        "oauth_account_id", account.get("id")
                    )
                    self.oauth_provider = account.get("provider", "google")
                    logger.info(
                        f"Email: Using only available account: {account.get('name', account.get('email'))}"
                    )
                elif not available_accounts:
                    return False, "No email accounts available"
                else:
                    available = self._get_available_accounts_str(context)
                    return (
                        False,
                        f"Multiple email accounts available. Specify 'account' parameter: {available}",
                    )
            else:
                # Default account uses oauth_account_id or id
                self.oauth_account_id = email_config.get(
                    "oauth_account_id", email_config.get("id")
                )
                self.oauth_provider = email_config.get("provider", "google")

        if not self.oauth_account_id:
            return False, "Could not determine email account ID"

        logger.info(
            f"Email: Using provider '{self.oauth_provider}' account {self.oauth_account_id}"
        )

        self._init_oauth_client()
        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the email action."""
        # Configure from params or context
        success, error = self._configure_from_params_or_context(params, context)
        if not success:
            return ActionResult(
                success=False,
                message="",
                error=error,
            )

        try:
            if self.oauth_provider == "microsoft":
                return self._execute_outlook(action, params, context)
            else:
                return self._execute_gmail(action, params, context)
        except httpx.HTTPStatusError as e:
            logger.error(f"Email API error: {e.response.status_code}")
            error_text = e.response.text[:200] if e.response.text else str(e)
            return ActionResult(
                success=False,
                message="",
                error=f"Email API error ({e.response.status_code}): {error_text}",
            )
        except Exception as e:
            logger.exception(f"Email action failed: {action}")
            return ActionResult(
                success=False,
                message="",
                error=f"Action failed: {str(e)}",
            )

    # -------------------------------------------------------------------------
    # Gmail Implementation
    # -------------------------------------------------------------------------

    def _execute_gmail(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute action via Gmail API."""
        if action == "draft_new":
            return self._gmail_draft_new(params)
        elif action == "draft_reply":
            return self._gmail_draft_reply(params)
        elif action == "draft_forward":
            return self._gmail_draft_forward(params, context)
        elif action == "send_new":
            return self._gmail_send_new(params)
        elif action == "send_reply":
            return self._gmail_send_reply(params)
        elif action == "send_forward":
            return self._gmail_send_forward(params, context)
        elif action == "label":
            return self._gmail_label(params)
        elif action == "archive":
            return self._gmail_archive(params, context)
        elif action == "mark_read":
            return self._gmail_mark_read(params, context, read=True)
        elif action == "mark_unread":
            return self._gmail_mark_read(params, context, read=False)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _normalize_recipients(self, recipients) -> list[str]:
        """Normalize recipients to a list of email addresses."""
        if isinstance(recipients, str):
            return [r.strip() for r in recipients.split(",") if r.strip()]
        elif isinstance(recipients, list):
            return recipients
        return []

    def _gmail_draft_new(self, params: dict) -> ActionResult:
        """Create a new email draft in Gmail."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        bcc = self._normalize_recipients(params.get("bcc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")

        # Build message with plain + HTML
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
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={"message": {"raw": raw}},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create draft: {response.status_code}",
            )

        draft_data = response.json()
        return ActionResult(
            success=True,
            message="Draft created successfully",
            data={"draft_id": draft_data.get("id"), "to": to, "subject": subject},
        )

    def _gmail_draft_reply(self, params: dict) -> ActionResult:
        """Create a reply draft in Gmail."""
        message_id = params.get("message_id")
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # Get original message
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

        # Build recipients
        to = [headers.get("from", "")]
        cc = []
        if reply_all:
            original_to = headers.get("to", "")
            original_cc = headers.get("cc", "")
            if original_to:
                cc = [addr.strip() for addr in original_to.split(",")]
            if original_cc:
                cc.extend([addr.strip() for addr in original_cc.split(",")])

        # Build message
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
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={"message": {"raw": raw, "threadId": thread_id}},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create reply draft: {response.status_code}",
            )

        draft_data = response.json()
        return ActionResult(
            success=True,
            message="Reply draft created",
            data={"draft_id": draft_data.get("id"), "thread_id": thread_id},
        )

    def _gmail_draft_forward(
        self, params: dict, context: ActionContext
    ) -> ActionResult:
        """Create a forward draft in Gmail."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        body = params.get("body", "")

        # Resolve message ID if needed
        if not message_id:
            message_id = self._lookup_message_by_hint(params, context)
            if not message_id:
                return ActionResult(
                    success=False,
                    message="",
                    error="Could not find message - provide message_id or subject_hint",
                )

        # Get original message
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
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to parse original email: {str(e)}",
            )

        original_subject = original_email.get("Subject", "")
        original_from = original_email.get("From", "")
        original_date = original_email.get("Date", "")
        original_to = original_email.get("To", "")

        subject = original_subject
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}"

        # Build forward header
        forward_header = (
            f"---------- Forwarded message ----------\n"
            f"From: {original_from}\n"
            f"Date: {original_date}\n"
            f"Subject: {original_subject}\n"
            f"To: {original_to}\n\n"
        )

        # Extract original body content and attachments
        original_text_body = ""
        original_html_body = ""
        attachments = []

        if original_email.is_multipart():
            for part in original_email.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    attachments.append(part)
                elif content_type == "text/plain" and not original_text_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        original_text_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
                elif content_type == "text/html" and not original_html_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        original_html_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
        else:
            content_type = original_email.get_content_type()
            try:
                charset = original_email.get_content_charset() or "utf-8"
                payload = original_email.get_payload(decode=True).decode(
                    charset, errors="ignore"
                )
                if content_type == "text/html":
                    original_html_body = payload
                else:
                    original_text_body = payload
            except Exception:
                pass

        # Build combined body
        if body:
            combined_text = f"{body}\n\n{forward_header}{original_text_body}"
            combined_html = (
                f"{_plain_to_html(body)}<br><br>"
                f"{_plain_to_html(forward_header)}"
                f"{original_html_body or _plain_to_html(original_text_body)}"
            )
        else:
            combined_text = f"{forward_header}{original_text_body}"
            combined_html = (
                f"{_plain_to_html(forward_header)}"
                f"{original_html_body or _plain_to_html(original_text_body)}"
            )

        # Build the forwarded message
        if attachments:
            message = MIMEMultipart("mixed")
            body_part = MIMEMultipart("alternative")
            body_part.attach(MIMEText(combined_text, "plain"))
            body_part.attach(MIMEText(combined_html, "html"))
            message.attach(body_part)
            for att in attachments:
                message.attach(att)
        else:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(combined_text, "plain"))
            message.attach(MIMEText(combined_html, "html"))

        message["to"] = ", ".join(to)
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/drafts",
            json={"message": {"raw": raw}},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create forward draft: {response.status_code}",
            )

        draft_data = response.json()
        return ActionResult(
            success=True,
            message="Forward draft created",
            data={"draft_id": draft_data.get("id"), "to": to},
        )

    def _gmail_send_new(self, params: dict) -> ActionResult:
        """Send a new email via Gmail."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")

        plain_body = body
        html_body = _plain_to_html(body)

        message = MIMEMultipart("alternative")
        message.attach(MIMEText(plain_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
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
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send email: {response.status_code}",
            )

        msg_data = response.json()
        return ActionResult(
            success=True,
            message="Email sent successfully",
            data={"message_id": msg_data.get("id"), "to": to, "subject": subject},
        )

    def _gmail_send_reply(self, params: dict) -> ActionResult:
        """Send a reply via Gmail."""
        message_id = params.get("message_id")
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        # Get original message
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

        to = [headers.get("from", "")]
        cc = []
        if reply_all:
            original_to = headers.get("to", "")
            original_cc = headers.get("cc", "")
            if original_to:
                cc = [addr.strip() for addr in original_to.split(",")]
            if original_cc:
                cc.extend([addr.strip() for addr in original_cc.split(",")])

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
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send reply: {response.status_code}",
            )

        msg_data = response.json()
        return ActionResult(
            success=True,
            message="Reply sent successfully",
            data={"message_id": msg_data.get("id"), "thread_id": thread_id},
        )

    def _gmail_send_forward(self, params: dict, context: ActionContext) -> ActionResult:
        """Send a forward via Gmail, preserving attachments and full content."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        body = params.get("body", "")

        if not message_id:
            message_id = self._lookup_message_by_hint(params, context)
            if not message_id:
                return ActionResult(
                    success=False,
                    message="",
                    error="Could not find message - provide message_id or subject_hint",
                )

        # Get original message in raw format
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
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to parse original email: {str(e)}",
            )

        original_subject = original_email.get("Subject", "")
        original_from = original_email.get("From", "")
        original_date = original_email.get("Date", "")
        original_to = original_email.get("To", "")

        subject = original_subject
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}"

        # Build forward header
        forward_header = (
            f"---------- Forwarded message ----------\n"
            f"From: {original_from}\n"
            f"Date: {original_date}\n"
            f"Subject: {original_subject}\n"
            f"To: {original_to}\n\n"
        )

        # Extract original body content and attachments
        original_text_body = ""
        original_html_body = ""
        attachments = []

        if original_email.is_multipart():
            for part in original_email.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    # Collect attachments to re-attach
                    attachments.append(part)
                elif content_type == "text/plain" and not original_text_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        original_text_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
                elif content_type == "text/html" and not original_html_body:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        original_html_body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
        else:
            content_type = original_email.get_content_type()
            try:
                charset = original_email.get_content_charset() or "utf-8"
                payload = original_email.get_payload(decode=True).decode(
                    charset, errors="ignore"
                )
                if content_type == "text/html":
                    original_html_body = payload
                else:
                    original_text_body = payload
            except Exception:
                pass

        # Build combined body: user comment + forward header + original content
        if body:
            combined_text = f"{body}\n\n{forward_header}{original_text_body}"
            combined_html = (
                f"{_plain_to_html(body)}<br><br>"
                f"{_plain_to_html(forward_header)}"
                f"{original_html_body or _plain_to_html(original_text_body)}"
            )
        else:
            combined_text = f"{forward_header}{original_text_body}"
            combined_html = (
                f"{_plain_to_html(forward_header)}"
                f"{original_html_body or _plain_to_html(original_text_body)}"
            )

        # Build the forwarded message
        if attachments:
            # Mixed multipart for body + attachments
            message = MIMEMultipart("mixed")

            # Add body as alternative (plain + html)
            body_part = MIMEMultipart("alternative")
            body_part.attach(MIMEText(combined_text, "plain"))
            body_part.attach(MIMEText(combined_html, "html"))
            message.attach(body_part)

            # Re-attach all original attachments
            for att in attachments:
                message.attach(att)
        else:
            # Just alternative for plain + html
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(combined_text, "plain"))
            message.attach(MIMEText(combined_html, "html"))

        message["to"] = ", ".join(to)
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        response = self.oauth_post(
            "https://www.googleapis.com/gmail/v1/users/me/messages/send",
            json={"raw": raw},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send forward: {response.status_code}",
            )

        msg_data = response.json()
        return ActionResult(
            success=True,
            message="Forward sent successfully",
            data={"message_id": msg_data.get("id"), "to": to},
        )

    def _gmail_label(self, params: dict) -> ActionResult:
        """Modify labels on a Gmail message."""
        message_id = params.get("message_id")
        add_labels = params.get("add_labels", [])
        remove_labels = params.get("remove_labels", [])

        if isinstance(add_labels, str):
            add_labels = [l.strip() for l in add_labels.split(",") if l.strip()]
        if isinstance(remove_labels, str):
            remove_labels = [l.strip() for l in remove_labels.split(",") if l.strip()]

        # Map label names to IDs
        label_map = {
            "inbox": "INBOX",
            "starred": "STARRED",
            "important": "IMPORTANT",
            "sent": "SENT",
            "drafts": "DRAFT",
            "spam": "SPAM",
            "trash": "TRASH",
            "unread": "UNREAD",
        }

        add_ids = [label_map.get(l.lower(), l) for l in add_labels]
        remove_ids = [label_map.get(l.lower(), l) for l in remove_labels]

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
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to modify labels: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Labels modified",
            data={
                "message_id": message_id,
                "added": add_labels,
                "removed": remove_labels,
            },
        )

    def _gmail_archive(self, params: dict, context: ActionContext) -> ActionResult:
        """Archive a Gmail message."""
        message_id = params.get("message_id")

        # Resolve message ID if not provided
        if not message_id:
            message_id = self._lookup_message_by_hint(params, context)
            if not message_id:
                return ActionResult(
                    success=False,
                    message="",
                    error="Could not find message - provide message_id or subject_hint",
                )

        response = self.oauth_post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            json={"removeLabelIds": ["INBOX"]},
        )

        if response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to archive: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Message archived",
            data={"message_id": message_id},
        )

    def _gmail_mark_read(
        self, params: dict, context: ActionContext, read: bool = True
    ) -> ActionResult:
        """Mark a Gmail message as read or unread."""
        message_id = params.get("message_id")

        # Resolve message ID if not provided
        if not message_id:
            message_id = self._lookup_message_by_hint(params, context)
            if not message_id:
                return ActionResult(
                    success=False,
                    message="",
                    error="Could not find message - provide message_id or subject_hint",
                )

        if read:
            body = {"removeLabelIds": ["UNREAD"]}
        else:
            body = {"addLabelIds": ["UNREAD"]}

        response = self.oauth_post(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            json=body,
        )

        if response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to mark {'read' if read else 'unread'}: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message=f"Message marked as {'read' if read else 'unread'}",
            data={"message_id": message_id},
        )

    # -------------------------------------------------------------------------
    # Outlook Implementation
    # -------------------------------------------------------------------------

    def _execute_outlook(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute action via Microsoft Graph API."""
        if action == "draft_new":
            return self._outlook_draft_new(params)
        elif action == "draft_reply":
            return self._outlook_draft_reply(params)
        elif action == "draft_forward":
            return self._outlook_draft_forward(params)
        elif action == "send_new":
            return self._outlook_send_new(params)
        elif action == "send_reply":
            return self._outlook_send_reply(params)
        elif action == "send_forward":
            return self._outlook_send_forward(params)
        elif action == "archive":
            return self._outlook_archive(params)
        elif action == "mark_read":
            return self._outlook_mark_read(params, read=True)
        elif action == "mark_unread":
            return self._outlook_mark_read(params, read=False)
        elif action == "label":
            # Outlook uses categories instead of labels
            return self._outlook_category(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _outlook_draft_new(self, params: dict) -> ActionResult:
        """Create a new email draft in Outlook."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")

        message_body = {
            "subject": subject,
            "body": {"contentType": "text", "content": body},
            "toRecipients": [{"emailAddress": {"address": addr}} for addr in to],
        }

        if cc:
            message_body["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in cc
            ]

        response = self.oauth_post(
            "https://graph.microsoft.com/v1.0/me/messages",
            json=message_body,
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create draft: {response.status_code}",
            )

        msg_data = response.json()
        return ActionResult(
            success=True,
            message="Draft created successfully",
            data={"message_id": msg_data.get("id"), "to": to, "subject": subject},
        )

    def _outlook_draft_reply(self, params: dict) -> ActionResult:
        """Create a reply draft in Outlook."""
        message_id = params.get("message_id")
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        endpoint = (
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/createReply"
        )
        if reply_all:
            endpoint = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/createReplyAll"

        response = self.oauth_post(endpoint, json={})

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create reply draft: {response.status_code}",
            )

        draft = response.json()
        draft_id = draft.get("id")

        # Update the draft with the body
        update_response = self.oauth_patch(
            f"https://graph.microsoft.com/v1.0/me/messages/{draft_id}",
            json={"body": {"contentType": "text", "content": body}},
        )

        if update_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to update reply draft: {update_response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Reply draft created",
            data={"message_id": draft_id},
        )

    def _outlook_draft_forward(self, params: dict) -> ActionResult:
        """Create a forward draft in Outlook."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        body = params.get("body", "")

        response = self.oauth_post(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/createForward",
            json={},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create forward draft: {response.status_code}",
            )

        draft = response.json()
        draft_id = draft.get("id")

        # Update the draft with recipients and comment
        update_body = {
            "toRecipients": [{"emailAddress": {"address": addr}} for addr in to],
        }
        if body:
            update_body["body"] = {"contentType": "text", "content": body}

        update_response = self.oauth_patch(
            f"https://graph.microsoft.com/v1.0/me/messages/{draft_id}",
            json=update_body,
        )

        if update_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to update forward draft: {update_response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Forward draft created",
            data={"message_id": draft_id, "to": to},
        )

    def _outlook_send_new(self, params: dict) -> ActionResult:
        """Send a new email via Outlook."""
        to = self._normalize_recipients(params.get("to", []))
        cc = self._normalize_recipients(params.get("cc", []))
        subject = params.get("subject", "")
        body = params.get("body", "")

        message_body = {
            "message": {
                "subject": subject,
                "body": {"contentType": "text", "content": body},
                "toRecipients": [{"emailAddress": {"address": addr}} for addr in to],
            }
        }

        if cc:
            message_body["message"]["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in cc
            ]

        response = self.oauth_post(
            "https://graph.microsoft.com/v1.0/me/sendMail",
            json=message_body,
        )

        if response.status_code not in (200, 202):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send email: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Email sent successfully",
            data={"to": to, "subject": subject},
        )

    def _outlook_send_reply(self, params: dict) -> ActionResult:
        """Send a reply via Outlook."""
        message_id = params.get("message_id")
        body = params.get("body", "")
        reply_all = params.get("reply_all", False)

        endpoint = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/reply"
        if reply_all:
            endpoint = (
                f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/replyAll"
            )

        response = self.oauth_post(
            endpoint,
            json={"comment": body},
        )

        if response.status_code not in (200, 202):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send reply: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Reply sent successfully",
            data={"original_message_id": message_id},
        )

    def _outlook_send_forward(self, params: dict) -> ActionResult:
        """Send a forward via Outlook."""
        message_id = params.get("message_id")
        to = self._normalize_recipients(params.get("to", []))
        body = params.get("body", "")

        response = self.oauth_post(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/forward",
            json={
                "comment": body,
                "toRecipients": [{"emailAddress": {"address": addr}} for addr in to],
            },
        )

        if response.status_code not in (200, 202):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to send forward: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Forward sent successfully",
            data={"original_message_id": message_id, "to": to},
        )

    def _outlook_archive(self, params: dict) -> ActionResult:
        """Archive an Outlook message (move to Archive folder)."""
        message_id = params.get("message_id")

        # Get archive folder ID
        folders_response = self.oauth_get(
            "https://graph.microsoft.com/v1.0/me/mailFolders",
            params={"$filter": "displayName eq 'Archive'"},
        )

        if folders_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error="Failed to find Archive folder",
            )

        folders = folders_response.json().get("value", [])
        if not folders:
            return ActionResult(
                success=False,
                message="",
                error="Archive folder not found",
            )

        archive_id = folders[0]["id"]

        # Move message to Archive
        response = self.oauth_post(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/move",
            json={"destinationId": archive_id},
        )

        if response.status_code not in (200, 201):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to archive: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Message archived",
            data={"message_id": message_id},
        )

    def _outlook_mark_read(self, params: dict, read: bool = True) -> ActionResult:
        """Mark an Outlook message as read or unread."""
        message_id = params.get("message_id")

        response = self.oauth_patch(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}",
            json={"isRead": read},
        )

        if response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to mark {'read' if read else 'unread'}: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message=f"Message marked as {'read' if read else 'unread'}",
            data={"message_id": message_id},
        )

    def _outlook_category(self, params: dict) -> ActionResult:
        """Add or remove categories from an Outlook message."""
        message_id = params.get("message_id")
        add_labels = params.get("add_labels", [])
        remove_labels = params.get("remove_labels", [])

        if isinstance(add_labels, str):
            add_labels = [l.strip() for l in add_labels.split(",") if l.strip()]
        if isinstance(remove_labels, str):
            remove_labels = [l.strip() for l in remove_labels.split(",") if l.strip()]

        # Get current categories
        msg_response = self.oauth_get(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}",
            params={"$select": "categories"},
        )

        if msg_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to get message: {msg_response.status_code}",
            )

        current = set(msg_response.json().get("categories", []))
        current.update(add_labels)
        current -= set(remove_labels)

        response = self.oauth_patch(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}",
            json={"categories": list(current)},
        )

        if response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to update categories: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Categories updated",
            data={"message_id": message_id, "categories": list(current)},
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _extract_email_body(self, msg: email.message.Message) -> str:
        """Extract plain text body from email message."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        return part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        pass
        else:
            try:
                charset = msg.get_content_charset() or "utf-8"
                return msg.get_payload(decode=True).decode(charset, errors="ignore")
            except Exception:
                pass
        return ""

    def _lookup_message_by_hint(
        self, params: dict, context: ActionContext
    ) -> Optional[str]:
        """Try to lookup message ID from session cache, then fall back to Gmail search."""
        subject_hint = params.get("subject_hint") or params.get("subject", "")
        sender_hint = params.get("sender_hint") or params.get("from", "")

        if not subject_hint and not sender_hint:
            return None

        # Try session cache first (fast)
        try:
            from live.sources import lookup_email_from_session_cache

            session_key = context.session_key if context else None
            account_email = self.get_account_email() if self.oauth_account_id else ""

            result = lookup_email_from_session_cache(
                session_key=session_key,
                account_email=account_email,
                subject_hint=subject_hint,
                sender_hint=sender_hint,
            )

            if result:
                logger.info(
                    f"Found message in session cache: {result.get('message_id')}"
                )
                return result.get("message_id")
        except Exception as e:
            logger.warning(f"Session cache lookup failed: {e}")

        # Fall back to Gmail API search
        if self.oauth_provider == "google":
            try:
                return self._gmail_search_for_message(subject_hint, sender_hint)
            except Exception as e:
                logger.warning(f"Gmail search failed: {e}")

        return None

    def _gmail_search_for_message(
        self, subject_hint: str, sender_hint: str
    ) -> Optional[str]:
        """Search Gmail for a message by subject/sender hints."""
        # Build Gmail search query
        query_parts = []
        if subject_hint:
            # Use quotes for exact phrase matching
            query_parts.append(f'subject:"{subject_hint}"')
        if sender_hint:
            query_parts.append(f"from:{sender_hint}")

        if not query_parts:
            return None

        query = " ".join(query_parts)
        logger.info(f"Gmail search for message: {query}")

        response = self.oauth_get(
            "https://www.googleapis.com/gmail/v1/users/me/messages",
            params={"q": query, "maxResults": 1},
        )

        if response.status_code != 200:
            logger.warning(f"Gmail search failed: {response.status_code}")
            return None

        messages = response.json().get("messages", [])
        if messages:
            message_id = messages[0].get("id")
            logger.info(f"Gmail search found message: {message_id}")
            return message_id

        logger.warning(f"Gmail search returned no results for: {query}")
        return None

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action in ("draft_new", "send_new"):
            if not params.get("to"):
                errors.append(ValidationError("to", "Recipients required"))
            if not params.get("subject"):
                errors.append(ValidationError("subject", "Subject required"))
            if not params.get("body"):
                errors.append(ValidationError("body", "Body required"))

        elif action in ("draft_reply", "send_reply"):
            if not params.get("message_id"):
                errors.append(ValidationError("message_id", "Message ID required"))
            if not params.get("body"):
                errors.append(ValidationError("body", "Body required"))

        elif action in ("draft_forward", "send_forward"):
            if not params.get("message_id") and not params.get("subject_hint"):
                errors.append(
                    ValidationError("message_id", "Message ID or subject_hint required")
                )
            if not params.get("to"):
                errors.append(ValidationError("to", "Recipients required"))

        elif action in ("archive", "mark_read", "mark_unread"):
            if not params.get("message_id") and not params.get("subject_hint"):
                errors.append(
                    ValidationError("message_id", "Message ID or subject_hint required")
                )

        elif action == "label":
            if not params.get("message_id"):
                errors.append(ValidationError("message_id", "Message ID required"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary."""
        to = params.get("to", [])
        if isinstance(to, list):
            to = ", ".join(to)

        if action == "draft_new":
            return f'Create draft to {to}: "{params.get("subject", "")}"'
        elif action == "send_new":
            return f'Send email to {to}: "{params.get("subject", "")}"'
        elif action == "draft_reply":
            return "Create reply draft"
        elif action == "send_reply":
            return "Send reply"
        elif action == "draft_forward":
            return f"Create forward draft to {to}"
        elif action == "send_forward":
            return f"Forward to {to}"
        elif action == "archive":
            return f"Archive message {params.get('message_id', '?')[:20]}"
        elif action == "mark_read":
            return f"Mark message as read"
        elif action == "mark_unread":
            return f"Mark message as unread"
        elif action == "label":
            return "Modify labels"

        return f"Email: {action}"

    def is_available(self) -> bool:
        """Check if plugin is available (always true - config comes from context)."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection - cannot test without context."""
        return (
            True,
            "Email handler ready (provider configured per Smart Alias)",
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
