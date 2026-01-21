"""
Built-in Unified Source Plugins.

These plugins combine RAG document indexing with Live API querying
into single unified sources with intelligent query routing.
"""

# Google sources
from .gcalendar import GCalendarUnifiedSource
from .gcontacts import GContactsUnifiedSource
from .gdrive import GDriveUnifiedSource
from .github import GitHubUnifiedSource
from .gmail import GmailUnifiedSource
from .gtasks import GTasksUnifiedSource

# RAG-only sources
from .local_filesystem import LocalFilesystemUnifiedSource
from .nextcloud import NextcloudUnifiedSource
from .notion import NotionUnifiedSource
from .onedrive import OneDriveUnifiedSource
from .onenote import OneNoteUnifiedSource

# Microsoft sources
from .outlook import OutlookUnifiedSource
from .outlook_calendar import OutlookCalendarUnifiedSource
from .paperless import PaperlessUnifiedSource

# Third-party sources
from .slack import SlackUnifiedSource
from .teams import TeamsUnifiedSource
from .todoist import TodoistUnifiedSource
from .websearch import WebSearchUnifiedSource
from .website import WebsiteUnifiedSource

__all__ = [
    # Google
    "GmailUnifiedSource",
    "GDriveUnifiedSource",
    "GCalendarUnifiedSource",
    "GTasksUnifiedSource",
    "GContactsUnifiedSource",
    # Microsoft
    "OutlookUnifiedSource",
    "OutlookCalendarUnifiedSource",
    "OneDriveUnifiedSource",
    "OneNoteUnifiedSource",
    "TeamsUnifiedSource",
    # Third-party
    "SlackUnifiedSource",
    "GitHubUnifiedSource",
    "NotionUnifiedSource",
    "TodoistUnifiedSource",
    # RAG-only
    "LocalFilesystemUnifiedSource",
    "WebsiteUnifiedSource",
    "PaperlessUnifiedSource",
    "NextcloudUnifiedSource",
    "WebSearchUnifiedSource",
]
