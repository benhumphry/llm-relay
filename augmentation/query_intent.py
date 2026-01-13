"""
Query intent extraction for search optimization.

Extracts temporal and categorical intent from user queries to improve
web search results. Maps natural language to search engine parameters.

Temporal intent maps to search engine time_range values:
- "day" - Last 24 hours (yesterday, today, last night)
- "week" - Last 7 days (this week, last week, recent)
- "month" - Last 30 days (this month, last month, lately)
- "year" - Last year (this year, last year)

Category intent maps to search engine categories:
- "general" - Default web search
- "news" - News articles (headlines, breaking, latest news)
- "images" - Image search
- "videos" - Video content (YouTube, tutorials)
- "science" - Scientific/academic content
- "it" - Technology/programming
- "files" - File downloads
- "social media" - Social platforms
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class QueryIntent:
    """Extracted intent from a user query."""

    time_range: Optional[str] = None  # day, week, month, year
    category: Optional[str] = None  # news, images, videos, etc.
    has_temporal_reference: bool = False
    has_category_reference: bool = False


def extract_query_intent(query: str) -> QueryIntent:
    """
    Extract temporal and categorical intent from a query.

    Args:
        query: User's search query

    Returns:
        QueryIntent with extracted time_range and category
    """
    query_lower = query.lower()
    intent = QueryIntent()

    # Extract temporal intent
    intent.time_range = _extract_time_range(query_lower)
    intent.has_temporal_reference = intent.time_range is not None

    # Extract category intent
    intent.category = _extract_category(query_lower)
    intent.has_category_reference = intent.category is not None

    return intent


def _extract_time_range(query_lower: str) -> Optional[str]:
    """
    Extract time range from query, mapped to search engine values.

    Returns one of: "day", "week", "month", "year", or None
    """
    today = datetime.now().date()

    # Day-level patterns (last 24 hours)
    day_patterns = [
        "today",
        "today's",
        "yesterday",
        "last night",
        "this morning",
        "this afternoon",
        "this evening",
        "tonight",
        "past 24 hours",
        "last 24 hours",
        "in the last day",
    ]
    if any(pattern in query_lower for pattern in day_patterns):
        return "day"

    # Week-level patterns (last 7 days)
    week_patterns = [
        "this week",
        "last week",
        "past week",
        "recent",
        "recently",
        "latest",
        "last few days",
        "past few days",
        "in the last week",
        "over the past week",
    ]
    if any(pattern in query_lower for pattern in week_patterns):
        return "week"

    # Check for "last N days" where N <= 7
    match = re.search(r"last (\d+) days?", query_lower)
    if match:
        days = int(match.group(1))
        if days <= 1:
            return "day"
        elif days <= 7:
            return "week"
        elif days <= 30:
            return "month"
        else:
            return "year"

    # Month-level patterns (last 30 days)
    month_patterns = [
        "this month",
        "last month",
        "past month",
        "lately",
        "in the last month",
        "over the past month",
        "last few weeks",
        "past few weeks",
    ]
    if any(pattern in query_lower for pattern in month_patterns):
        return "month"

    # Year-level patterns
    year_patterns = [
        "this year",
        "last year",
        "past year",
        "in the last year",
        "over the past year",
        f"in {today.year}",
        f"in {today.year - 1}",
    ]
    if any(pattern in query_lower for pattern in year_patterns):
        return "year"

    # News-related queries often imply recency
    news_recency_patterns = [
        "breaking",
        "headlines",
        "current",
        "happening now",
        "just happened",
        "announced",
    ]
    if any(pattern in query_lower for pattern in news_recency_patterns):
        return "day"

    return None


def _extract_category(query_lower: str) -> Optional[str]:
    """
    Extract search category from query.

    Returns one of: "news", "images", "videos", "science", "it", "files", "social media", or None
    """
    # News patterns
    news_patterns = [
        "news",
        "headlines",
        "breaking",
        "announced",
        "reports",
        "reporting",
        "press release",
        "media coverage",
        "latest on",
        "update on",
        "what happened",
        "current events",
    ]
    if any(pattern in query_lower for pattern in news_patterns):
        return "news"

    # Image patterns
    image_patterns = [
        "image of",
        "images of",
        "picture of",
        "pictures of",
        "photo of",
        "photos of",
        "photograph",
        "what does .* look like",
        "show me",
    ]
    if any(pattern in query_lower for pattern in image_patterns):
        return "images"

    # Video patterns
    video_patterns = [
        "video of",
        "videos of",
        "youtube",
        "watch",
        "tutorial",
        "how to video",
        "clip of",
        "footage",
    ]
    if any(pattern in query_lower for pattern in video_patterns):
        return "videos"

    # Science/academic patterns
    science_patterns = [
        "research on",
        "study on",
        "scientific",
        "paper on",
        "journal",
        "academic",
        "peer reviewed",
        "publication",
    ]
    if any(pattern in query_lower for pattern in science_patterns):
        return "science"

    # IT/technology patterns
    it_patterns = [
        "programming",
        "code for",
        "github",
        "stackoverflow",
        "documentation",
        "api",
        "library for",
        "framework",
        "bug fix",
        "error message",
    ]
    if any(pattern in query_lower for pattern in it_patterns):
        return "it"

    # File download patterns
    file_patterns = [
        "download",
        "pdf of",
        "file for",
        "installer",
        "binary",
        "package",
    ]
    if any(pattern in query_lower for pattern in file_patterns):
        return "files"

    # Social media patterns
    social_patterns = [
        "twitter",
        "tweet",
        "facebook",
        "instagram",
        "reddit",
        "linkedin",
        "tiktok",
        "social media",
    ]
    if any(pattern in query_lower for pattern in social_patterns):
        return "social media"

    return None


def get_searxng_time_range(time_range: Optional[str]) -> Optional[str]:
    """
    Map time_range to SearXNG time_range parameter values.

    SearXNG accepts: "day", "week", "month", "year"

    Args:
        time_range: Extracted time range

    Returns:
        SearXNG-compatible time_range value or None
    """
    # SearXNG uses the same values we extract
    if time_range in ("day", "week", "month", "year"):
        return time_range
    return None


def get_searxng_categories(category: Optional[str]) -> Optional[str]:
    """
    Map category to SearXNG categories parameter value.

    SearXNG categories: general, news, images, videos, music, files, it, science, social media

    Args:
        category: Extracted category

    Returns:
        SearXNG-compatible category value or None (defaults to "general")
    """
    valid_categories = {
        "general",
        "news",
        "images",
        "videos",
        "music",
        "files",
        "it",
        "science",
        "social media",
    }
    if category in valid_categories:
        return category
    return None
