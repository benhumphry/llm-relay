"""
Smart News live source plugin.

High-level news interface with intelligent multi-step lookups.
Combines headlines with full story coverage for comprehensive news context.

Uses Real-Time News Data API via RapidAPI (powered by Google News).

Smart features:
- Auto-detects topic/search/local from natural language
- Fetches headlines then enriches with full story coverage
- Geo-based local news support
- Combines multiple sources for comprehensive coverage
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)

logger = logging.getLogger(__name__)

# Global cache with TTL
_news_cache: dict[str, tuple[Any, float]] = {}  # 15-min TTL for news


class SmartNewsLiveSource(PluginLiveSource):
    """
    Smart News Provider - intelligent multi-step news fetching.

    Unlike basic news APIs, this provider:
    - Auto-detects query type (headlines, topic, search, local)
    - Fetches top stories then enriches with full coverage
    - Supports geo-based local news
    - Combines headlines + related articles for context

    Examples:
    - "Latest tech news" -> Topic headlines + full coverage
    - "UK headlines" -> Country headlines
    - "News about OpenAI" -> Search + full story coverage
    - "Local news in Manchester" -> Geo-based headlines
    """

    source_type = "realtime_news"
    display_name = "Real-Time News"
    description = "Real-time news with smart headline + full coverage aggregation"
    category = "news"
    data_type = "news"
    best_for = "Breaking news, headlines, topic news (tech, sports, business), deep coverage on stories. Use 'tech news', 'UK headlines', 'news about [topic]', 'local news in [city]'."
    icon = "📰"
    default_cache_ttl = 900  # 15 minutes

    _abstract = False

    # API configuration
    BASE_URL = "https://real-time-news-data.p.rapidapi.com"

    # Cache TTL
    NEWS_CACHE_TTL = 900  # 15 minutes

    # Supported topics
    TOPICS = {
        "world": "WORLD",
        "national": "NATIONAL",
        "business": "BUSINESS",
        "technology": "TECHNOLOGY",
        "tech": "TECHNOLOGY",
        "entertainment": "ENTERTAINMENT",
        "sports": "SPORTS",
        "sport": "SPORTS",
        "science": "SCIENCE",
        "health": "HEALTH",
    }

    # Country codes
    COUNTRIES = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "uk": "GB",
        "gb": "GB",
        "united kingdom": "GB",
        "britain": "GB",
        "canada": "CA",
        "australia": "AU",
        "germany": "DE",
        "france": "FR",
        "india": "IN",
        "japan": "JP",
        "china": "CN",
        "brazil": "BR",
        "mexico": "MX",
        "spain": "ES",
        "italy": "IT",
        "netherlands": "NL",
        "sweden": "SE",
        "norway": "NO",
        "denmark": "DK",
        "finland": "FI",
        "poland": "PL",
        "russia": "RU",
        "south korea": "KR",
        "korea": "KR",
        "singapore": "SG",
        "hong kong": "HK",
        "new zealand": "NZ",
        "ireland": "IE",
        "south africa": "ZA",
        "uae": "AE",
        "israel": "IL",
    }

    # Time ranges for search
    TIME_RANGES = {
        "hour": "1h",
        "1h": "1h",
        "day": "1d",
        "1d": "1d",
        "today": "1d",
        "week": "7d",
        "7d": "7d",
        "month": "30d",
        "30d": "30d",
        "year": "1y",
        "1y": "1y",
        "anytime": "anytime",
        "any": "anytime",
    }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields for admin UI."""
        return [
            FieldDefinition(
                name="name",
                label="Source Name",
                field_type=FieldType.TEXT,
                required=True,
                default="Smart News",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="RapidAPI Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="RAPIDAPI_KEY",
                help_text="RapidAPI key for Real-Time News Data API",
            ),
            FieldDefinition(
                name="default_country",
                label="Default Country",
                field_type=FieldType.SELECT,
                required=False,
                default="GB",
                options=[
                    {"value": "GB", "label": "United Kingdom"},
                    {"value": "US", "label": "United States"},
                    {"value": "CA", "label": "Canada"},
                    {"value": "AU", "label": "Australia"},
                    {"value": "NZ", "label": "New Zealand"},
                    {"value": "IE", "label": "Ireland"},
                    {"value": "DE", "label": "Germany"},
                    {"value": "FR", "label": "France"},
                    {"value": "ES", "label": "Spain"},
                    {"value": "IT", "label": "Italy"},
                    {"value": "NL", "label": "Netherlands"},
                    {"value": "SE", "label": "Sweden"},
                    {"value": "NO", "label": "Norway"},
                    {"value": "DK", "label": "Denmark"},
                    {"value": "FI", "label": "Finland"},
                    {"value": "PL", "label": "Poland"},
                    {"value": "IN", "label": "India"},
                    {"value": "JP", "label": "Japan"},
                    {"value": "CN", "label": "China"},
                    {"value": "KR", "label": "South Korea"},
                    {"value": "SG", "label": "Singapore"},
                    {"value": "HK", "label": "Hong Kong"},
                    {"value": "BR", "label": "Brazil"},
                    {"value": "MX", "label": "Mexico"},
                    {"value": "ZA", "label": "South Africa"},
                    {"value": "AE", "label": "UAE"},
                    {"value": "IL", "label": "Israel"},
                ],
                help_text="Default country for news headlines",
            ),
            FieldDefinition(
                name="default_language",
                label="Default Language",
                field_type=FieldType.SELECT,
                required=False,
                default="en",
                options=[
                    {"value": "en", "label": "English"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "fr", "label": "French"},
                    {"value": "de", "label": "German"},
                    {"value": "it", "label": "Italian"},
                    {"value": "pt", "label": "Portuguese"},
                    {"value": "ja", "label": "Japanese"},
                    {"value": "zh", "label": "Chinese"},
                ],
                help_text="Default language for news",
            ),
            FieldDefinition(
                name="include_full_coverage",
                label="Include Full Coverage",
                field_type=FieldType.BOOLEAN,
                required=False,
                default=True,
                help_text="Fetch full story coverage for top headlines (more comprehensive but slower)",
            ),
            FieldDefinition(
                name="max_articles",
                label="Max Articles",
                field_type=FieldType.INTEGER,
                required=False,
                default=8,
                min_value=1,
                max_value=20,
                help_text="Maximum number of articles to return",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query",
                description="Search query, topic, or location for news",
                param_type="string",
                required=True,
                examples=[
                    "tech news",
                    "UK headlines",
                    "news about artificial intelligence",
                    "local news in London",
                ],
            ),
            ParamDefinition(
                name="country",
                description="Country for headlines (2-letter code or name)",
                param_type="string",
                required=False,
                examples=["US", "UK", "Germany", "Japan"],
            ),
            ParamDefinition(
                name="time_range",
                description="Time range for search results",
                param_type="string",
                required=False,
                default="anytime",
                examples=["hour", "day", "week", "month"],
            ),
            ParamDefinition(
                name="full_coverage",
                description="Whether to include full story coverage",
                param_type="boolean",
                required=False,
                default=True,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-news")
        self.api_key = config.get("api_key") or os.environ.get("RAPIDAPI_KEY", "")
        self.default_country = config.get("default_country", "US")
        self.default_language = config.get("default_language", "en")
        self.include_full_coverage = config.get("include_full_coverage", True)
        self.max_articles = config.get("max_articles", 8)

        self._client = httpx.Client(timeout=15)

    def _get_headers(self) -> dict:
        """Get API headers."""
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com",
        }

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get from cache if still valid."""
        if key in _news_cache:
            data, cached_at = _news_cache[key]
            if time.time() - cached_at < self.NEWS_CACHE_TTL:
                return data
            del _news_cache[key]
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        """Store in cache."""
        _news_cache[key] = (data, time.time())

    def _detect_request_type(self, query: str) -> tuple[str, dict]:
        """
        Detect the type of news request from natural language query.

        Returns (request_type, extracted_params)
        """
        query_lower = query.lower().strip()

        # Check for local news pattern
        local_patterns = ["local news in", "local news for", "news in", "news from"]
        for pattern in local_patterns:
            if pattern in query_lower:
                # Extract location
                location = query_lower.split(pattern)[-1].strip()
                # Clean up common suffixes
                for suffix in [" area", " region", " city"]:
                    location = location.replace(suffix, "")
                return "local", {"location": location.strip()}

        # Check for topic keywords
        for topic_key, topic_value in self.TOPICS.items():
            if topic_key in query_lower and (
                "news" in query_lower or "headlines" in query_lower
            ):
                return "topic", {"topic": topic_value}

        # Check for headlines + country
        if "headline" in query_lower:
            for country_key, country_code in self.COUNTRIES.items():
                if country_key in query_lower:
                    return "headlines", {"country": country_code}
            return "headlines", {"country": self.default_country}

        # Check for "news about X" pattern (search with full coverage)
        if "news about" in query_lower or "news on" in query_lower:
            search_term = (
                query_lower.replace("news about", "").replace("news on", "").strip()
            )
            return "search_deep", {"query": search_term}

        # Default to search
        return "search", {"query": query}

    def _resolve_country(self, country: str) -> str:
        """Resolve country name to code."""
        if not country:
            return self.default_country
        country_lower = country.lower().strip()
        return self.COUNTRIES.get(country_lower, country.upper()[:2])

    def _resolve_time_range(self, time_range: str) -> str:
        """Resolve time range to API format."""
        if not time_range:
            return "anytime"
        return self.TIME_RANGES.get(time_range.lower().strip(), "anytime")

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch news with smart multi-step aggregation.

        For headlines/topics: Fetches top stories, then optionally gets
        full coverage for the top story to provide deeper context.

        For search: Searches articles, then fetches full coverage for
        the most relevant result.

        For local: Uses geo-based headlines.
        """
        if not self.api_key:
            return LiveDataResult(
                success=False,
                error="RapidAPI key not configured. Set RAPIDAPI_KEY environment variable.",
            )

        query = params.get("query", "").strip()
        if not query:
            return LiveDataResult(
                success=False,
                error="Query is required. Specify a topic, search term, or 'headlines'.",
            )

        country = params.get("country", "")
        time_range = params.get("time_range", "anytime")
        include_coverage = params.get("full_coverage", self.include_full_coverage)

        # Auto-detect request type
        request_type, detected_params = self._detect_request_type(query)

        # Merge detected params
        topic = detected_params.get("topic")
        location = detected_params.get("location")
        search_query = detected_params.get("query", query)

        if "country" in detected_params and not country:
            country = detected_params["country"]

        # Resolve parameters
        country = self._resolve_country(country)
        time_range = self._resolve_time_range(time_range)

        # Check cache
        cache_key = (
            f"news:{request_type}:{query}:{country}:{time_range}:{include_coverage}"
        )
        cached = self._get_cached(cache_key)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=cached.get("_formatted", ""),
                cache_ttl=self.NEWS_CACHE_TTL,
            )

        try:
            result_data = {}
            formatted_parts = []

            # Step 1: Fetch primary news based on type
            if request_type == "local" and location:
                logger.info(f"Fetching local news for: {location}")
                headlines = self._fetch_local_news(location)
                formatted_parts.append(f"**Local News: {location.title()}**\n")

            elif request_type == "topic" and topic:
                logger.info(f"Fetching topic news: {topic}")
                headlines = self._fetch_topic_news(topic, country)
                formatted_parts.append(f"**{topic.title()} News**\n")

            elif request_type == "headlines":
                logger.info(f"Fetching headlines for: {country}")
                headlines = self._fetch_headlines(country)
                country_name = [k for k, v in self.COUNTRIES.items() if v == country]
                country_display = country_name[0].title() if country_name else country
                formatted_parts.append(f"**{country_display} Headlines**\n")

            elif request_type == "search_deep":
                logger.info(f"Deep search for: {search_query}")
                headlines = self._fetch_search(search_query, country, time_range)
                formatted_parts.append(f"**News: {search_query}**\n")
                # Force full coverage for deep search
                include_coverage = True

            else:  # regular search
                logger.info(f"Searching news: {search_query}")
                headlines = self._fetch_search(search_query, country, time_range)
                formatted_parts.append(f"**News: {search_query}**\n")

            if not headlines or not headlines.get("data"):
                return LiveDataResult(
                    success=False,
                    error=f"No news found for: {query}",
                )

            articles = headlines.get("data", [])[: self.max_articles]
            result_data["headlines"] = articles

            # Format headlines
            formatted_parts.append(self._format_headlines(articles))

            # Step 2: Get full coverage for top story if enabled
            if include_coverage and articles:
                top_article = articles[0]
                source_url = top_article.get("link") or top_article.get("source_url")

                if source_url:
                    logger.info(f"Fetching full coverage for top story")
                    coverage = self._fetch_full_coverage(source_url)

                    if coverage and coverage.get("data"):
                        result_data["full_coverage"] = coverage.get("data")
                        formatted_parts.append(
                            "\n" + self._format_full_coverage(coverage.get("data"))
                        )

            # Combine formatted output
            formatted = "\n".join(formatted_parts)
            result_data["_formatted"] = formatted

            # Cache and return
            self._set_cached(cache_key, result_data)

            return LiveDataResult(
                success=True,
                data=result_data,
                formatted=formatted,
                cache_ttl=self.NEWS_CACHE_TTL,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"News API HTTP error: {e}")
            return LiveDataResult(
                success=False,
                error=f"API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"News API error: {e}", exc_info=True)
            return LiveDataResult(
                success=False,
                error=f"Failed to fetch news: {str(e)}",
            )

    def _fetch_headlines(self, country: str) -> Optional[dict]:
        """Fetch top headlines for a country."""
        response = self._client.get(
            f"{self.BASE_URL}/top-headlines",
            headers=self._get_headers(),
            params={
                "country": country,
                "lang": self.default_language,
            },
        )
        response.raise_for_status()
        return response.json()

    def _fetch_topic_news(self, topic: str, country: str) -> Optional[dict]:
        """Fetch news for a specific topic."""
        response = self._client.get(
            f"{self.BASE_URL}/topic-headlines",
            headers=self._get_headers(),
            params={
                "topic": topic,
                "country": country,
                "lang": self.default_language,
            },
        )
        response.raise_for_status()
        return response.json()

    def _fetch_search(
        self, query: str, country: str, time_range: str
    ) -> Optional[dict]:
        """Search for news articles."""
        params = {
            "query": query,
            "country": country,
            "lang": self.default_language,
        }
        if time_range != "anytime":
            params["time_published"] = time_range

        response = self._client.get(
            f"{self.BASE_URL}/search",
            headers=self._get_headers(),
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_local_news(self, location: str) -> Optional[dict]:
        """Fetch geo-based local news."""
        response = self._client.get(
            f"{self.BASE_URL}/local-headlines",
            headers=self._get_headers(),
            params={
                "query": location,
                "lang": self.default_language,
            },
        )
        response.raise_for_status()
        return response.json()

    def _fetch_full_coverage(self, source_url: str) -> Optional[dict]:
        """Fetch full story coverage for an article URL."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/full-story-coverage",
                headers=self._get_headers(),
                params={
                    "source_url": source_url,
                    "lang": self.default_language,
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch full coverage: {e}")
            return None

    def _format_time_ago(self, published: str) -> str:
        """Format published time as relative time."""
        if not published:
            return ""
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = now - pub_dt
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            else:
                return f"{diff.seconds // 60}m ago"
        except:
            return published[:10] if len(published) >= 10 else published

    def _format_headlines(self, articles: list) -> str:
        """Format headline articles."""
        lines = []

        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            source = article.get("source_name", "Unknown")
            snippet = article.get("snippet", "")
            published = article.get("published_datetime_utc", "")
            time_ago = self._format_time_ago(published)

            lines.append(f"**{i}. {title}**")
            lines.append(f"   _Source: {source}_ {f'| {time_ago}' if time_ago else ''}")

            if snippet:
                if len(snippet) > 180:
                    snippet = snippet[:180] + "..."
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)

    def _format_full_coverage(self, coverage_data: dict) -> str:
        """Format full story coverage."""
        lines = []
        lines.append("---")
        lines.append("**Full Story Coverage:**")

        # Main story summary
        summary = coverage_data.get("summary")
        if summary:
            lines.append(f"\n_{summary}_\n")

        # Related articles from other sources
        related = coverage_data.get("articles", [])[:5]
        if related:
            lines.append("**Other perspectives:**")
            for article in related:
                title = article.get("title", "")
                source = article.get("source_name", "")
                if title:
                    lines.append(f"- {title} _({source})_")

        # Social/Twitter posts if available
        posts = coverage_data.get("posts", [])[:3]
        if posts:
            lines.append("\n**Social discussion:**")
            for post in posts:
                text = post.get("text", "")[:150]
                if text:
                    lines.append(f'- "{text}..."')

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to news API."""
        if not self.api_key:
            return False, "RapidAPI key not configured"

        try:
            response = self._client.get(
                f"{self.BASE_URL}/top-headlines",
                headers=self._get_headers(),
                params={"country": "US", "lang": "en"},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "OK" or data.get("data"):
                article_count = len(data.get("data", []))
                return True, f"Connected. Found {article_count} headlines."
            else:
                return False, f"Unexpected response: {data.get('status', 'unknown')}"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
