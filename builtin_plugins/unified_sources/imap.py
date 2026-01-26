"""
IMAP/SMTP Unified Source Plugin.

Provides email access via standard IMAP/SMTP protocols - works with any email
provider that supports IMAP (Gmail, Outlook, Fastmail, self-hosted, etc.)
without requiring OAuth.

Features:
- Document side: Index emails from IMAP folders for RAG semantic search
- Live side: Query recent emails, search, read specific messages
- Actions: Send via SMTP, draft/archive/mark via IMAP

Configuration via environment variables or per-store config:
- IMAP_HOST, IMAP_PORT, IMAP_USERNAME, IMAP_PASSWORD
- SMTP_HOST, SMTP_PORT (for sending)
"""

import email
import html as html_module
import imaplib
import ssl
import logging
import os
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email import policy
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, parseaddr, parsedate_to_datetime
from typing import Any, Iterator, Optional

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import PluginUnifiedSource, QueryAnalysis, QueryRouting

logger = logging.getLogger(__name__)


class IMAPUnifiedSource(PluginUnifiedSource):
    """
    Unified IMAP/SMTP source - RAG for history, Live for recent/specific.

    Provides email access via standard protocols without OAuth complexity.
    Works with any IMAP-compatible email provider.
    """

    source_type = "imap"
    display_name = "IMAP Email"
    description = "Email via IMAP/SMTP (works with any email provider)"
    category = "email"
    icon = "📬"

    content_category = ContentCategory.EMAILS

    # Document store types this unified source handles
    handles_doc_source_types = ["imap"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Draft, send, archive, mark read/unread
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results
    default_index_days = 90

    _abstract = False

    @classmethod
    def get_account_info(cls, store) -> dict | None:
        """Extract account info for action handlers."""
        # IMAP uses credential-based auth, not OAuth
        # Get email from store config
        username = store.imap_username or os.environ.get("IMAP_USERNAME", "")
        if not username:
            return None

        return {
            "provider": "imap",
            "email": username,
            "name": store.display_name or store.name,
            "store_id": store.id,
            "oauth_account_id": None,  # No OAuth for IMAP
            # IMAP-specific fields for action handler
            "imap_host": store.imap_host or os.environ.get("IMAP_HOST", ""),
            "smtp_host": store.smtp_host or os.environ.get("SMTP_HOST", ""),
        }

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME IMAP email access. Actions: "
            "action='recent' for latest emails, "
            "action='unread' for unread emails, "
            "action='today' for today's emails, "
            "action='search' with query='...' for IMAP search "
            "(e.g. query='FROM john@example.com', query='SUBJECT invoice'), "
            "action='read' with message_id='...' to fetch full email content. "
            "Optional: max_results=N to limit results, folder='FolderName' to search specific folder."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "imap_host": store.imap_host or os.environ.get("IMAP_HOST", ""),
            "imap_port": store.imap_port or int(os.environ.get("IMAP_PORT", "993")),
            "username": store.imap_username or os.environ.get("IMAP_USERNAME", ""),
            "password": store.imap_password or os.environ.get("IMAP_PASSWORD", ""),
            "use_ssl": store.imap_use_ssl if hasattr(store, "imap_use_ssl") else True,
            "allow_insecure": getattr(store, "imap_allow_insecure", False),
            "smtp_host": store.smtp_host or os.environ.get("SMTP_HOST", ""),
            "smtp_port": store.smtp_port or int(os.environ.get("SMTP_PORT", "587")),
            "folders": store.imap_folders or "INBOX",
            "index_days": store.imap_index_days or 90,
            "index_schedule": store.index_schedule or "",
        }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            # IMAP Settings
            FieldDefinition(
                name="imap_host",
                label="IMAP Host",
                field_type=FieldType.TEXT,
                required=False,
                placeholder="imap.example.com",
                help_text="IMAP server hostname (leave empty to use env var)",
                env_var="IMAP_HOST",
            ),
            FieldDefinition(
                name="imap_port",
                label="IMAP Port",
                field_type=FieldType.INTEGER,
                default=993,
                help_text="IMAP port (993 for SSL, 143 for STARTTLS)",
                env_var="IMAP_PORT",
            ),
            FieldDefinition(
                name="username",
                label="Username/Email",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Email address or username for authentication",
                env_var="IMAP_USERNAME",
            ),
            FieldDefinition(
                name="password",
                label="Password/App Password",
                field_type=FieldType.PASSWORD,
                required=False,
                help_text="Password or app-specific password",
                env_var="IMAP_PASSWORD",
            ),
            FieldDefinition(
                name="use_ssl",
                label="Use SSL/TLS",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Use SSL/TLS encryption (recommended)",
            ),
            FieldDefinition(
                name="allow_insecure",
                label="Allow Insecure Connection",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Skip SSL certificate verification (for self-signed certs or ProtonMail Bridge)",

            ),
            # SMTP Settings (for sending)
            FieldDefinition(
                name="smtp_host",
                label="SMTP Host",
                field_type=FieldType.TEXT,
                required=False,
                placeholder="smtp.example.com",
                help_text="SMTP server for sending (leave empty to use env var)",
                env_var="SMTP_HOST",
            ),
            FieldDefinition(
                name="smtp_port",
                label="SMTP Port",
                field_type=FieldType.INTEGER,
                default=587,
                help_text="SMTP port (587 for STARTTLS, 465 for SSL)",
                env_var="SMTP_PORT",
            ),
            # Indexing Settings
            FieldDefinition(
                name="folders",
                label="Folders to Index",
                field_type=FieldType.TEXT,
                default="INBOX",
                help_text="Comma-separated folder names (e.g., INBOX,Sent,Archive)",
            ),
            FieldDefinition(
                name="index_days",
                label="Days to Index",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of email history to index (max 365)",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-index emails",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum emails to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: recent, unread, today, search, read",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "unread", "today", "search", "read"],
            ),
            ParamDefinition(
                name="query",
                description="IMAP search query (e.g., FROM sender, SUBJECT text, SINCE date)",
                param_type="string",
                required=False,
                examples=[
                    "FROM john@example.com",
                    "SUBJECT invoice",
                    "SINCE 01-Jan-2024",
                ],
            ),
            ParamDefinition(
                name="message_id",
                description="Message ID for action='read' to fetch full content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="folder",
                description="Folder to search (default: INBOX)",
                param_type="string",
                required=False,
                default="INBOX",
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum emails to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        # IMAP settings
        self.imap_host = config.get("imap_host") or os.environ.get("IMAP_HOST", "")
        self.imap_port = config.get("imap_port") or int(
            os.environ.get("IMAP_PORT", "993")
        )
        self.username = config.get("username") or os.environ.get("IMAP_USERNAME", "")
        self.password = config.get("password") or os.environ.get("IMAP_PASSWORD", "")
        self.use_ssl = config.get("use_ssl", True)
        self.allow_insecure = config.get("allow_insecure", False)

        # SMTP settings
        self.smtp_host = config.get("smtp_host") or os.environ.get("SMTP_HOST", "")
        self.smtp_port = config.get("smtp_port") or int(
            os.environ.get("SMTP_PORT", "587")
        )

        # Indexing settings
        folders_str = config.get("folders", "INBOX")
        self.folders = [f.strip() for f in folders_str.split(",") if f.strip()]
        self.index_days = min(config.get("index_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        # Connection cache
        self._imap_conn: Optional[imaplib.IMAP4_SSL | imaplib.IMAP4] = None

    def _get_imap_connection(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        """Get or create IMAP connection."""
        if self._imap_conn:
            try:
                # Test if connection is still alive
                self._imap_conn.noop()
                return self._imap_conn
            except Exception:
                self._imap_conn = None

        if not self.imap_host or not self.username or not self.password:
            raise Exception("IMAP credentials not configured")

        if self.use_ssl:
            if self.allow_insecure:
                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self._imap_conn = imaplib.IMAP4_SSL(
                    self.imap_host, self.imap_port, ssl_context=ssl_context
                )
            else:
                self._imap_conn = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
        else:
            self._imap_conn = imaplib.IMAP4(self.imap_host, self.imap_port)
            if self.allow_insecure:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self._imap_conn.starttls(ssl_context=ssl_context)
            else:
                self._imap_conn.starttls()

        self._imap_conn.login(self.username, self.password)
        return self._imap_conn

    def _close_imap_connection(self):
        """Close IMAP connection."""
        if self._imap_conn:
            try:
                self._imap_conn.logout()
            except Exception:
                pass
            self._imap_conn = None

    def _decode_header_value(self, value: str) -> str:
        """Decode MIME encoded header value."""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                charset = charset or "utf-8"
                try:
                    result.append(part.decode(charset, errors="replace"))
                except Exception:
                    result.append(part.decode("utf-8", errors="replace"))
            else:
                result.append(part)
        return "".join(result)

    def _parse_email_message(self, msg_data: bytes, uid: str) -> dict:
        """Parse raw email bytes into a structured dict."""
        msg = email.message_from_bytes(msg_data, policy=policy.default)

        # Extract headers
        subject = self._decode_header_value(msg.get("Subject", "(No Subject)"))
        from_header = self._decode_header_value(msg.get("From", ""))
        to_header = self._decode_header_value(msg.get("To", ""))
        date_header = msg.get("Date", "")
        message_id = msg.get("Message-ID", "")

        # Parse date
        email_date = None
        if date_header:
            try:
                parsed_date = parsedate_to_datetime(date_header)
                email_date = parsed_date
            except Exception:
                pass

        # Extract body
        body_text = ""
        body_html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and not body_text:
                    try:
                        body_text = part.get_content()
                    except Exception:
                        pass
                elif content_type == "text/html" and not body_html:
                    try:
                        body_html = part.get_content()
                    except Exception:
                        pass
        else:
            content_type = msg.get_content_type()
            try:
                if content_type == "text/plain":
                    body_text = msg.get_content()
                elif content_type == "text/html":
                    body_html = msg.get_content()
            except Exception:
                pass

        # Prefer plain text, fall back to stripped HTML
        body = body_text
        if not body and body_html:
            body = self._strip_html(body_html)

        # Parse from address
        from_name, from_email = parseaddr(from_header)
        if not from_name:
            from_name = from_email.split("@")[0] if "@" in from_email else from_email

        return {
            "uid": uid,
            "message_id": message_id,
            "subject": subject,
            "from": from_header,
            "from_name": from_name,
            "from_email": from_email,
            "to": to_header,
            "date": date_header,
            "date_parsed": email_date,
            "body": body,
            "snippet": (body[:200] + "...") if len(body) > 200 else body,
        }

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags and decode entities."""
        # Remove script and style elements
        html_content = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Replace block elements with newlines
        html_content = re.sub(
            r"<(br|p|div|tr|li)[^>]*>", "\n", html_content, flags=re.IGNORECASE
        )

        # Remove remaining tags
        html_content = re.sub(r"<[^>]+>", "", html_content)

        # Decode entities
        html_content = html_module.unescape(html_content)

        # Clean whitespace
        html_content = re.sub(r"\n\s*\n", "\n\n", html_content)
        return html_content.strip()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate emails for indexing.

        Lists emails from configured folders within the index_days window.
        """
        try:
            imap = self._get_imap_connection()
        except Exception as e:
            logger.error(f"Failed to connect to IMAP: {e}")
            return

        # Calculate date range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.index_days)
        since_date = cutoff_date.strftime("%d-%b-%Y")

        for folder in self.folders:
            try:
                # Select folder
                status, _ = imap.select(folder, readonly=True)
                if status != "OK":
                    logger.warning(f"Could not select folder {folder}")
                    continue

                logger.info(f"Listing emails from folder {folder} since {since_date}")

                # Search for emails since cutoff date
                status, data = imap.search(None, f"SINCE {since_date}")
                if status != "OK":
                    logger.warning(f"Search failed in folder {folder}")
                    continue

                message_ids = data[0].split()
                logger.info(f"Found {len(message_ids)} emails in {folder}")

                for msg_id in message_ids:
                    # Get UID for stable identification
                    status, uid_data = imap.fetch(msg_id, "(UID)")
                    if status != "OK":
                        continue

                    uid_match = re.search(rb"UID\s+(\d+)", uid_data[0])
                    if not uid_match:
                        continue

                    uid = uid_match.group(1).decode()

                    yield DocumentInfo(
                        uri=f"imap://{folder}/{uid}",
                        title=f"Email {uid}",  # Will be replaced with subject when reading
                        mime_type="message/rfc822",
                        metadata={"folder": folder, "uid": uid},
                    )

            except Exception as e:
                logger.error(f"Error listing emails in {folder}: {e}")

        self._close_imap_connection()

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read email content for indexing.

        Fetches the full email and formats it for embedding.
        """
        if not uri.startswith("imap://"):
            logger.error(f"Invalid IMAP URI: {uri}")
            return None

        # Parse URI: imap://folder/uid
        path = uri.replace("imap://", "")
        parts = path.rsplit("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid IMAP URI format: {uri}")
            return None

        folder, uid = parts

        try:
            imap = self._get_imap_connection()
            imap.select(folder, readonly=True)

            # Fetch email by UID
            status, data = imap.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                logger.error(f"Failed to fetch email UID {uid} from {folder}")
                return None

            raw_email = data[0][1]
            email_data = self._parse_email_message(raw_email, uid)

        except Exception as e:
            logger.error(f"Error reading email {uri}: {e}")
            return None

        # Format for indexing
        content = f"""Subject: {email_data["subject"]}
From: {email_data["from"]}
To: {email_data["to"]}
Date: {email_data["date"]}

{email_data["body"]}
"""

        # Parse date for metadata
        email_date = None
        if email_data.get("date_parsed"):
            email_date = email_data["date_parsed"].strftime("%Y-%m-%d")

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "uid": uid,
                "folder": folder,
                "message_id": email_data.get("message_id", ""),
                "email_date": email_date,
                "from": email_data["from"],
                "from_name": email_data["from_name"],
                "to": email_data["to"],
                "subject": email_data["subject"],
                "source_type": "email",
                "account_email": self.username,
            },
        )

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live email data.

        Supports actions: recent, unread, today, search, read
        """
        start_time = time.time()

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        message_id = params.get("message_id", "")
        folder = params.get("folder", "INBOX")
        max_results = min(params.get("max_results", self.live_max_results), 50)

        try:
            imap = self._get_imap_connection()

            # Handle read action separately
            if action == "read" and message_id:
                return self._fetch_full_email(imap, folder, message_id)

            # Select folder
            status, _ = imap.select(folder, readonly=True)
            if status != "OK":
                return LiveDataResult(
                    success=False, error=f"Could not access folder: {folder}"
                )

            # Build IMAP search criteria
            search_criteria = self._build_search_criteria(action, search_query)

            # Search emails
            status, data = imap.search(None, search_criteria)
            if status != "OK":
                return LiveDataResult(success=False, error="Search failed")

            message_ids = data[0].split()

            # Get most recent (last N) messages
            message_ids = message_ids[-max_results:]
            message_ids.reverse()  # Most recent first

            # Fetch email metadata
            emails = []
            for msg_id in message_ids:
                try:
                    # Fetch headers only for speed
                    status, data = imap.fetch(
                        msg_id, "(UID BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE)])"
                    )
                    if status != "OK" or not data or not data[0]:
                        continue

                    # Parse UID
                    uid_match = re.search(rb"UID\s+(\d+)", data[0][0])
                    uid = uid_match.group(1).decode() if uid_match else msg_id.decode()

                    # Parse headers
                    header_data = data[0][1]
                    msg = email.message_from_bytes(header_data, policy=policy.default)

                    emails.append(
                        {
                            "uid": uid,
                            "folder": folder,
                            "from": self._decode_header_value(msg.get("From", "")),
                            "to": self._decode_header_value(msg.get("To", "")),
                            "subject": self._decode_header_value(
                                msg.get("Subject", "(No Subject)")
                            ),
                            "date": msg.get("Date", ""),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse email {msg_id}: {e}")

            # Format for LLM context
            formatted = self._format_emails(emails, action, folder)

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"IMAP live query: action={action}, folder={folder}, "
                f"results={len(emails)}, latency={latency_ms}ms"
            )

            return LiveDataResult(
                success=True,
                data=emails,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"IMAP live query error: {e}")
            return LiveDataResult(success=False, error=str(e))

    def _build_search_criteria(self, action: str, search_query: str) -> str:
        """Build IMAP search criteria based on action and query."""
        today = datetime.now(timezone.utc)

        if action == "unread":
            return "UNSEEN"
        elif action == "today":
            return f"SINCE {today.strftime('%d-%b-%Y')}"
        elif action == "search" and search_query:
            # Pass through IMAP search syntax
            return search_query
        else:
            # Recent - last 7 days
            week_ago = (today - timedelta(days=7)).strftime("%d-%b-%Y")
            return f"SINCE {week_ago}"

    def _fetch_full_email(
        self, imap: imaplib.IMAP4_SSL | imaplib.IMAP4, folder: str, uid: str
    ) -> LiveDataResult:
        """Fetch full content of a specific email."""
        try:
            status, _ = imap.select(folder, readonly=True)
            if status != "OK":
                return LiveDataResult(
                    success=False, error=f"Could not access folder: {folder}"
                )

            # Fetch by UID
            status, data = imap.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                return LiveDataResult(
                    success=False, error=f"Email not found: {uid} in {folder}"
                )

            raw_email = data[0][1]
            email_data = self._parse_email_message(raw_email, uid)

            # Format full email content
            formatted = f"""### Email Details

**From:** {email_data["from"]}
**To:** {email_data["to"]}
**Subject:** {email_data["subject"]}
**Date:** {email_data["date"]}

---

{email_data["body"]}
"""

            return LiveDataResult(
                success=True,
                data=email_data,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Error fetching email {uid}: {e}")
            return LiveDataResult(success=False, error=str(e))

    def _format_emails(self, emails: list[dict], action: str, folder: str) -> str:
        """Format emails for LLM context."""
        if not emails:
            action_msgs = {
                "unread": "No unread emails.",
                "today": "No emails received today.",
                "recent": "No recent emails.",
                "search": "No emails found matching your search.",
            }
            return (
                f"### IMAP ({folder}) - {action_msgs.get(action, 'No emails found.')}"
            )

        action_titles = {
            "unread": "Unread Emails",
            "today": "Today's Emails",
            "recent": "Recent Emails",
            "search": "Email Search Results",
        }

        lines = [f"### {action_titles.get(action, 'Emails')} ({folder})"]
        lines.append(f"Account: {self.username}")
        lines.append("")

        for i, email_data in enumerate(emails, 1):
            from_name, from_email = parseaddr(email_data["from"])
            display_from = from_name or from_email

            lines.append(f"**{i}. {email_data['subject']}**")
            lines.append(f"   From: {display_from} | Date: {email_data['date']}")
            lines.append(f"   [UID: {email_data['uid']} in {email_data['folder']}]")
            lines.append("")

        lines.append(
            "_To read full email content, use action='read' with the message UID._"
        )

        return "\n".join(lines)

    # =========================================================================
    # Action Support (draft, send, archive, mark read/unread)
    # =========================================================================

    def create_draft(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] = None,
        bcc: list[str] = None,
        html_body: str = None,
        in_reply_to: str = None,
    ) -> dict:
        """
        Create a draft email via IMAP (append to Drafts folder).

        Returns dict with success status and draft info.
        """
        try:
            imap = self._get_imap_connection()

            # Build email message
            if html_body:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
            else:
                msg = MIMEText(body, "plain")

            msg["From"] = self.username
            msg["To"] = ", ".join(to)
            msg["Subject"] = subject
            msg["Date"] = formatdate(localtime=True)

            if cc:
                msg["Cc"] = ", ".join(cc)
            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                msg["References"] = in_reply_to

            # Append to Drafts folder
            # Try common draft folder names
            draft_folders = ["Drafts", "INBOX.Drafts", "[Gmail]/Drafts", "Draft"]
            appended = False

            for draft_folder in draft_folders:
                try:
                    status, _ = imap.append(
                        draft_folder, "\\Draft", None, msg.as_bytes()
                    )
                    if status == "OK":
                        appended = True
                        logger.info(f"Draft created in {draft_folder}")
                        break
                except Exception:
                    continue

            if not appended:
                return {"success": False, "error": "Could not find Drafts folder"}

            return {
                "success": True,
                "message": f"Draft created: {subject}",
                "folder": draft_folder,
            }

        except Exception as e:
            logger.error(f"Error creating draft: {e}")
            return {"success": False, "error": str(e)}

    def send_email(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] = None,
        bcc: list[str] = None,
        html_body: str = None,
        in_reply_to: str = None,
    ) -> dict:
        """
        Send an email via SMTP.

        Returns dict with success status.
        """
        if not self.smtp_host:
            return {"success": False, "error": "SMTP not configured"}

        try:
            # Build email message
            if html_body:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
            else:
                msg = MIMEText(body, "plain")

            msg["From"] = self.username
            msg["To"] = ", ".join(to)
            msg["Subject"] = subject
            msg["Date"] = formatdate(localtime=True)

            if cc:
                msg["Cc"] = ", ".join(cc)
            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                msg["References"] = in_reply_to

            # All recipients
            all_recipients = list(to)
            if cc:
                all_recipients.extend(cc)
            if bcc:
                all_recipients.extend(bcc)

            # Connect and send
            if self.smtp_port == 465:
                # SSL
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as smtp:
                    smtp.login(self.username, self.password)
                    smtp.sendmail(self.username, all_recipients, msg.as_string())
            else:
                # STARTTLS
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
                    smtp.starttls()
                    smtp.login(self.username, self.password)
                    smtp.sendmail(self.username, all_recipients, msg.as_string())

            logger.info(f"Email sent to {to}: {subject}")
            return {
                "success": True,
                "message": f"Email sent to {', '.join(to)}: {subject}",
            }

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {"success": False, "error": str(e)}

    def mark_read(self, folder: str, uid: str) -> dict:
        """Mark an email as read via IMAP."""
        try:
            imap = self._get_imap_connection()
            imap.select(folder)
            status, _ = imap.uid("store", uid, "+FLAGS", "\\Seen")
            if status == "OK":
                return {"success": True, "message": f"Marked email {uid} as read"}
            return {"success": False, "error": "Failed to mark as read"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mark_unread(self, folder: str, uid: str) -> dict:
        """Mark an email as unread via IMAP."""
        try:
            imap = self._get_imap_connection()
            imap.select(folder)
            status, _ = imap.uid("store", uid, "-FLAGS", "\\Seen")
            if status == "OK":
                return {"success": True, "message": f"Marked email {uid} as unread"}
            return {"success": False, "error": "Failed to mark as unread"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def archive_email(self, folder: str, uid: str, archive_folder: str = None) -> dict:
        """Archive an email (move to Archive folder) via IMAP."""
        try:
            imap = self._get_imap_connection()
            imap.select(folder)

            # Find archive folder
            if not archive_folder:
                archive_folders = [
                    "Archive",
                    "INBOX.Archive",
                    "[Gmail]/All Mail",
                    "All Mail",
                ]
                for af in archive_folders:
                    try:
                        status, _ = imap.select(af, readonly=True)
                        if status == "OK":
                            archive_folder = af
                            imap.select(folder)  # Re-select original
                            break
                    except Exception:
                        continue

            if not archive_folder:
                return {"success": False, "error": "Archive folder not found"}

            # Copy to archive
            status, _ = imap.uid("copy", uid, archive_folder)
            if status != "OK":
                return {"success": False, "error": "Failed to copy to archive"}

            # Mark original as deleted
            imap.uid("store", uid, "+FLAGS", "\\Deleted")
            imap.expunge()

            return {
                "success": True,
                "message": f"Archived email {uid} to {archive_folder}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_email(self, folder: str, uid: str) -> dict:
        """Delete an email (move to Trash) via IMAP."""
        try:
            imap = self._get_imap_connection()
            imap.select(folder)

            # Find trash folder
            trash_folders = ["Trash", "INBOX.Trash", "[Gmail]/Trash", "Deleted Items"]
            trash_folder = None

            for tf in trash_folders:
                try:
                    status, _ = imap.select(tf, readonly=True)
                    if status == "OK":
                        trash_folder = tf
                        imap.select(folder)  # Re-select original
                        break
                except Exception:
                    continue

            if trash_folder:
                # Move to trash
                status, _ = imap.uid("copy", uid, trash_folder)
                if status == "OK":
                    imap.uid("store", uid, "+FLAGS", "\\Deleted")
                    imap.expunge()
                    return {
                        "success": True,
                        "message": f"Moved email {uid} to {trash_folder}",
                    }

            # If no trash folder, just mark as deleted
            imap.uid("store", uid, "+FLAGS", "\\Deleted")
            imap.expunge()
            return {"success": True, "message": f"Deleted email {uid}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_available(self) -> bool:
        """Check if IMAP is configured."""
        return bool(self.imap_host and self.username and self.password)

    def test_connection(self) -> tuple[bool, str]:
        """Test IMAP connection."""
        try:
            imap = self._get_imap_connection()
            # List folders to verify connection works
            status, folders = imap.list()
            if status == "OK":
                folder_count = len(folders) if folders else 0
                self._close_imap_connection()
                return True, f"Connected. Found {folder_count} folders."
            self._close_imap_connection()
            return False, "Could not list folders"
        except Exception as e:
            return False, str(e)

    def get_account_email(self) -> str:
        """Get the account email address."""
        return self.username

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - Very recent (last hour, just now) -> Live only
        - Historical (last year, specific dates in past) -> RAG only
        - Unread queries -> Live only (real-time status)
        - Latest/new/recent -> Both, prefer live
        - Default -> Both with deduplication
        """
        import re as regex

        query_lower = query.lower()
        action = params.get("action", "")

        # Very recent queries -> Live only
        very_recent_patterns = [
            "last hour",
            "past hour",
            "last 30 minutes",
            "just now",
            "right now",
            "just received",
            "just got",
        ]
        if any(p in query_lower for p in very_recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Very recent time reference - using live API only",
                max_live_results=self.live_max_results,
            )

        # Unread action -> Live only (real-time status)
        if action == "unread" or "unread" in query_lower:
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "unread"},
                reason="Unread status requires live API",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"q[1-4] 20\d{2}",
            r"january|february|march|april|may|june",
            r"july|august|september|october|november|december",
        ]
        for pattern in historical_patterns:
            if regex.search(pattern, query_lower):
                return QueryAnalysis(
                    routing=QueryRouting.RAG_ONLY,
                    rag_query=query,
                    reason=f"Historical reference ({pattern}) - using RAG only",
                )

        # Recent/latest patterns -> Both, prefer live
        recent_patterns = ["latest", "recent", "new", "today", "this week"]
        if any(p in query_lower for p in recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_PREFER_LIVE,
                rag_query=query,
                live_params=params if params.get("action") else {**params, "action": "recent"},
                reason="Recent time reference - checking both sources",
                max_live_results=self.live_max_results,
                max_rag_results=10,
            )

        # Default -> Both with deduplication
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params if params.get("action") else {**params, "action": "recent"},
            reason="General query - checking both sources with deduplication",
            max_live_results=self.live_max_results,
            max_rag_results=10,
        )
