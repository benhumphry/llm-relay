"""
Google Contacts Unified Source Plugin.

Combines Google Contacts document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index contacts for semantic search (names, emails, phones, orgs)
- Live side: Search contacts by name, lookup specific contact, list recent
- Intelligent routing: Analyze queries to choose optimal data source

Query routing examples:
- "John Smith's phone number" -> Live only (specific lookup)
- "people who work at Acme Corp" -> RAG (semantic search)
- "contact info for Sarah" -> Both, prefer live for accuracy
- "my contacts" -> Live only (listing)
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import OAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class GContactsUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Google Contacts source - RAG for search, Live for lookup.

    Single configuration provides:
    - Document indexing: Contacts indexed for RAG semantic search
    - Live queries: Search by name, lookup specific contact, list contacts
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "gcontacts"
    display_name = "Google Contacts"
    description = "Google Contacts with semantic search (RAG) and real-time lookup"
    category = "google"
    icon = "👤"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:gcontacts"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # No contact actions yet
    supports_incremental = True

    default_cache_ttl = 600  # 10 minutes for contacts (change less often)

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.google_account_id,
            "contact_groups": store.gcontacts_group_id or "",
            "index_schedule": store.index_schedule or "",
        }

    # People API endpoint
    PEOPLE_API_BASE = "https://people.googleapis.com/v1"

    # Fields to request from API
    PERSON_FIELDS = (
        "names,emailAddresses,phoneNumbers,organizations,addresses,"
        "birthdays,biographies,urls,relations,occupations,nicknames,metadata"
    )

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["contacts"]},
                help_text="Select a connected Google account with Contacts access",
            ),
            FieldDefinition(
                name="include_other_contacts",
                label="Include Other Contacts",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Include 'Other contacts' (auto-saved from interactions)",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 0 * * *", "label": "Daily"},
                    {"value": "0 0 * * 0", "label": "Weekly"},
                ],
                help_text="How often to re-index contacts",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum contacts to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: search, lookup, list",
                param_type="string",
                required=False,
                default="search",
                examples=["search", "lookup", "list"],
            ),
            ParamDefinition(
                name="query",
                description="Search query (name, email, phone)",
                param_type="string",
                required=False,
                examples=["John Smith", "john@example.com", "Acme Corp"],
            ),
            ParamDefinition(
                name="resource_name",
                description="Specific contact resource name for lookup",
                param_type="string",
                required=False,
                examples=["people/c12345678"],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum contacts to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        self.include_other_contacts = config.get("include_other_contacts", False)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        self._client = httpx.Client(timeout=30)
        self._init_oauth_client()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate contacts for indexing.

        Lists all contacts from the user's Google account.
        """
        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot list contacts - no valid access token")
            return

        logger.info("Listing Google Contacts")

        total_contacts = 0
        page_token = None

        while True:
            params = {
                "pageSize": 100,
                "personFields": "names,emailAddresses,metadata",
                "sortOrder": "LAST_MODIFIED_DESCENDING",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = self._oauth_client.get(
                    f"{self.PEOPLE_API_BASE}/people/me/connections",
                    headers=self._get_auth_headers(),
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"People API error: {e}")
                break

            connections = data.get("connections", [])

            for person in connections:
                resource_name = person.get("resourceName", "")
                names = person.get("names", [])
                name = names[0].get("displayName", "Unknown") if names else "Unknown"

                # Get modification time from metadata
                metadata = person.get("metadata", {})
                sources = metadata.get("sources", [])
                modified_time = None
                if sources:
                    modified_time = sources[0].get("updateTime")

                total_contacts += 1
                yield DocumentInfo(
                    uri=f"gcontacts://{resource_name}",
                    title=name[:100],
                    mime_type="text/vcard",
                    modified_at=modified_time,
                    metadata={"resource_name": resource_name},
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        # Include other contacts if configured
        if self.include_other_contacts:
            yield from self._list_other_contacts()

        logger.info(f"Found {total_contacts} contacts to index")

    def _list_other_contacts(self) -> Iterator[DocumentInfo]:
        """List 'Other contacts' (auto-saved from interactions)."""
        page_token = None

        while True:
            params = {
                "pageSize": 100,
                "readMask": "names,emailAddresses",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = self._oauth_client.get(
                    f"{self.PEOPLE_API_BASE}/otherContacts",
                    headers=self._get_auth_headers(),
                    params=params,
                )
                if response.status_code != 200:
                    logger.warning(f"Other contacts API error: {response.status_code}")
                    break
                data = response.json()
            except Exception as e:
                logger.warning(f"Other contacts error: {e}")
                break

            other_contacts = data.get("otherContacts", [])

            for person in other_contacts:
                resource_name = person.get("resourceName", "")
                names = person.get("names", [])
                name = names[0].get("displayName", "Unknown") if names else "Unknown"

                yield DocumentInfo(
                    uri=f"gcontacts://{resource_name}",
                    title=f"(Other) {name[:90]}",
                    mime_type="text/vcard",
                    metadata={"resource_name": resource_name, "is_other": True},
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read contact content for indexing.

        Fetches comprehensive contact details for embedding.
        """
        if not uri.startswith("gcontacts://"):
            logger.error(f"Invalid Contacts URI: {uri}")
            return None

        resource_name = uri.replace("gcontacts://", "")

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot read contact - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/{resource_name}",
                headers=self._get_auth_headers(),
                params={"personFields": self.PERSON_FIELDS},
            )
            response.raise_for_status()
            person = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch contact {resource_name}: {e}")
            return None

        # Format contact as text
        content, metadata = self._format_contact_for_index(person)

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                **metadata,
                "resource_name": resource_name,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "contact",
            },
        )

    def _format_contact_for_index(self, person: dict) -> tuple[str, dict]:
        """Format a contact for indexing, returning (content, metadata)."""
        names = person.get("names", [])
        display_name = names[0].get("displayName", "Unknown") if names else "Unknown"

        emails = person.get("emailAddresses", [])
        phones = person.get("phoneNumbers", [])
        orgs = person.get("organizations", [])
        addresses = person.get("addresses", [])
        birthdays = person.get("birthdays", [])
        bios = person.get("biographies", [])
        urls = person.get("urls", [])
        nicknames = person.get("nicknames", [])

        # Build content
        content_parts = [f"Contact: {display_name}"]

        if nicknames:
            nick_list = ", ".join(n.get("value", "") for n in nicknames)
            content_parts.append(f"Nicknames: {nick_list}")

        if emails:
            for email in emails:
                email_type = email.get("type", "other")
                email_val = email.get("value", "")
                content_parts.append(f"Email ({email_type}): {email_val}")

        if phones:
            for phone in phones:
                phone_type = phone.get("type", "other")
                phone_val = phone.get("value", "")
                content_parts.append(f"Phone ({phone_type}): {phone_val}")

        if orgs:
            for org in orgs:
                org_name = org.get("name", "")
                org_title = org.get("title", "")
                if org_name or org_title:
                    org_str = (
                        f"{org_title} at {org_name}"
                        if org_title and org_name
                        else (org_title or org_name)
                    )
                    content_parts.append(f"Organization: {org_str}")

        if addresses:
            for addr in addresses:
                addr_type = addr.get("type", "other")
                formatted = addr.get("formattedValue", "")
                if formatted:
                    content_parts.append(f"Address ({addr_type}): {formatted}")

        if birthdays:
            for bday in birthdays:
                date = bday.get("date", {})
                if date:
                    year = date.get("year", "????")
                    month = date.get("month", 0)
                    day = date.get("day", 0)
                    bday_str = f"{year}-{month:02d}-{day:02d}"
                    content_parts.append(f"Birthday: {bday_str}")

        if urls:
            for url in urls:
                url_type = url.get("type", "other")
                url_val = url.get("value", "")
                content_parts.append(f"URL ({url_type}): {url_val}")

        if bios:
            for bio in bios:
                bio_val = bio.get("value", "")
                if bio_val:
                    content_parts.append(f"\nNotes: {bio_val}")

        content = "\n".join(content_parts)

        # Build metadata
        metadata = {
            "name": display_name,
            "email": emails[0].get("value") if emails else None,
            "phone": phones[0].get("value") if phones else None,
            "organization": orgs[0].get("name") if orgs else None,
        }

        return content, metadata

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live contact data.

        Supports actions: search, lookup, list
        """
        start_time = time.time()

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            return LiveDataResult(
                success=False,
                error="No valid Google Contacts access token",
            )

        action = params.get("action", "search")
        search_query = params.get("query", "")
        resource_name = params.get("resource_name", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            if action == "lookup" and resource_name:
                # Lookup specific contact
                contacts = self._lookup_contact(resource_name)
            elif action == "search" and search_query:
                # Search for contacts
                contacts = self._search_contacts(search_query, max_results)
            else:
                # List contacts
                contacts = self._list_contacts_live(max_results)

            # Format for LLM context
            formatted = self._format_contacts(contacts, action, search_query)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=contacts,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Contacts live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _search_contacts(self, query: str, max_results: int) -> list[dict]:
        """Search contacts using People API search endpoint."""
        try:
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/people:searchContacts",
                headers=self._get_auth_headers(),
                params={
                    "query": query,
                    "pageSize": max_results,
                    "readMask": self.PERSON_FIELDS,
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            return [r.get("person", {}) for r in results]
        except Exception as e:
            logger.error(f"Contact search failed: {e}")
            return []

    def _lookup_contact(self, resource_name: str) -> list[dict]:
        """Lookup a specific contact by resource name."""
        try:
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/{resource_name}",
                headers=self._get_auth_headers(),
                params={"personFields": self.PERSON_FIELDS},
            )
            response.raise_for_status()
            return [response.json()]
        except Exception as e:
            logger.error(f"Contact lookup failed: {e}")
            return []

    def _list_contacts_live(self, max_results: int) -> list[dict]:
        """List recent contacts."""
        try:
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/people/me/connections",
                headers=self._get_auth_headers(),
                params={
                    "pageSize": max_results,
                    "personFields": self.PERSON_FIELDS,
                    "sortOrder": "LAST_MODIFIED_DESCENDING",
                },
            )
            response.raise_for_status()
            return response.json().get("connections", [])
        except Exception as e:
            logger.error(f"Contact list failed: {e}")
            return []

    def _format_contacts(
        self, contacts: list[dict], action: str, search_query: str
    ) -> str:
        """Format contacts for LLM context."""
        account_email = self.get_account_email()

        if not contacts:
            action_msgs = {
                "search": f"No contacts found matching '{search_query}'.",
                "lookup": "Contact not found.",
                "list": "No contacts found.",
            }
            return (
                f"### Google Contacts\n{action_msgs.get(action, 'No contacts found.')}"
            )

        action_titles = {
            "search": f"Contact Search: {search_query}",
            "lookup": "Contact Details",
            "list": "Recent Contacts",
        }

        lines = [f"### {action_titles.get(action, 'Contacts')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(contacts)} contact(s):\n")

        for person in contacts:
            resource_name = person.get("resourceName", "")
            names = person.get("names", [])
            display_name = (
                names[0].get("displayName", "Unknown") if names else "Unknown"
            )

            emails = person.get("emailAddresses", [])
            phones = person.get("phoneNumbers", [])
            orgs = person.get("organizations", [])

            lines.append(f"**{display_name}**")
            lines.append(f"   ID: {resource_name}")

            if emails:
                for email in emails[:2]:
                    email_type = email.get("type", "")
                    type_str = f" ({email_type})" if email_type else ""
                    lines.append(f"   Email{type_str}: {email.get('value', '')}")

            if phones:
                for phone in phones[:2]:
                    phone_type = phone.get("type", "")
                    type_str = f" ({phone_type})" if phone_type else ""
                    lines.append(f"   Phone{type_str}: {phone.get('value', '')}")

            if orgs:
                org = orgs[0]
                org_name = org.get("name", "")
                org_title = org.get("title", "")
                if org_name or org_title:
                    org_str = (
                        f"{org_title} at {org_name}"
                        if org_title and org_name
                        else (org_title or org_name)
                    )
                    lines.append(f"   Work: {org_str}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Query Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - Specific contact lookup -> Live only
        - Name/email/phone lookup -> Live only (exact match)
        - Semantic queries (who works at, find people who) -> RAG
        - Default -> Both
        """
        query_lower = query.lower()
        action = params.get("action", "")

        # Specific lookup -> Live only
        if action == "lookup" or params.get("resource_name"):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Specific contact lookup uses live API",
                max_live_results=1,
            )

        # Contact listing -> Live only
        if (
            action == "list"
            or "my contacts" in query_lower
            or "all contacts" in query_lower
        ):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "list"},
                reason="Contact listing uses live API",
                max_live_results=self.live_max_results,
            )

        # Specific name/email search -> Live (better for exact matches)
        specific_patterns = [
            r"phone number",
            r"email address",
            r"contact info",
            r"'s phone",
            r"'s email",
            r"call\s+\w+",
        ]
        if any(re.search(p, query_lower) for p in specific_patterns):
            search_query = params.get("query", query)
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "search", "query": search_query},
                reason="Contact info lookup uses live search",
                max_live_results=self.live_max_results,
            )

        # Semantic queries -> RAG
        semantic_patterns = [
            r"who works at",
            r"people at",
            r"colleagues",
            r"everyone from",
            r"contacts from",
            r"find people",
        ]
        if any(re.search(p, query_lower) for p in semantic_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_ONLY,
                rag_query=query,
                reason="Semantic contact search uses RAG",
                max_rag_results=20,
            )

        # Default -> Both, prefer live for accuracy
        search_query = params.get("query", query)
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params={**params, "action": "search", "query": search_query},
            merge_strategy=MergeStrategy.LIVE_FIRST,
            reason="Contact search - live for accuracy, RAG for context",
            max_rag_results=10,
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Google Contacts is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Google Contacts API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return (
                    False,
                    "Failed to get access token - check OAuth configuration",
                )

            # Test API access - get user profile
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/people/me",
                headers=self._get_auth_headers(),
                params={"personFields": "names,emailAddresses"},
            )
            response.raise_for_status()
            me = response.json()

            names = me.get("names", [])
            name = names[0].get("displayName", "Unknown") if names else "Unknown"
            results.append(f"Connected as: {name}")

            # Count contacts
            response = self._oauth_client.get(
                f"{self.PEOPLE_API_BASE}/people/me/connections",
                headers=self._get_auth_headers(),
                params={"pageSize": 1, "personFields": "names"},
            )
            response.raise_for_status()
            total = response.json().get("totalPeople", 0)
            results.append(f"Total contacts: {total}")

            # Test document listing (RAG side)
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found contacts to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "list", "max_results": 5})
                    if live_result.success:
                        contact_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {contact_count} contacts")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
