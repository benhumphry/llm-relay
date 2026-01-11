"""
Date extraction utility for hybrid date+semantic search.

Parses natural language date references in queries like:
- "tomorrow", "yesterday", "today"
- "this morning", "this afternoon", "tonight"
- "last week", "next month"
- "recent emails", "latest"
- Specific dates: "January 11", "2026-01-11", "11-01-2026"
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


def extract_date_range(query: str) -> Optional[Tuple[str, str]]:
    """
    Extract a date range from a query.

    Args:
        query: User's search query

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format, or None if no date reference found
    """
    query_lower = query.lower()
    today = datetime.now().date()

    # Today (including time-of-day references)
    today_phrases = [
        "today",
        "today's",
        "this morning",
        "this afternoon",
        "this evening",
        "lunchtime",
        "tonight",
    ]
    if any(phrase in query_lower for phrase in today_phrases):
        date_str = today.isoformat()
        return (date_str, date_str)

    # Tomorrow (including time-of-day references)
    tomorrow_phrases = [
        "tomorrow morning",
        "tomorrow afternoon",
        "tomorrow evening",
        "tomorrow night",
        "tomorrow",
    ]
    if any(phrase in query_lower for phrase in tomorrow_phrases):
        tomorrow = today + timedelta(days=1)
        return (tomorrow.isoformat(), tomorrow.isoformat())

    # Yesterday
    if "yesterday" in query_lower:
        yesterday = today - timedelta(days=1)
        return (yesterday.isoformat(), yesterday.isoformat())

    # This week (Mon-Sun containing today)
    if "this week" in query_lower:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return (start.isoformat(), end.isoformat())

    # Next week
    if "next week" in query_lower:
        start = today + timedelta(days=(7 - today.weekday()))
        end = start + timedelta(days=6)
        return (start.isoformat(), end.isoformat())

    # Last week
    if "last week" in query_lower:
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start.isoformat(), end.isoformat())

    # Recent / latest (default to last 7 days)
    if any(word in query_lower for word in ["recent", "latest", "last few days"]):
        start = today - timedelta(days=7)
        return (start.isoformat(), today.isoformat())

    # Last N days
    match = re.search(r"last (\d+) days?", query_lower)
    if match:
        days = int(match.group(1))
        start = today - timedelta(days=days)
        return (start.isoformat(), today.isoformat())

    # Next N days
    match = re.search(r"next (\d+) days?", query_lower)
    if match:
        days = int(match.group(1))
        end = today + timedelta(days=days)
        return (today.isoformat(), end.isoformat())

    # This month
    if "this month" in query_lower:
        start = today.replace(day=1)
        # Last day of month
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    # Last month
    if "last month" in query_lower:
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return (start.isoformat(), end.isoformat())

    # Specific day names (this coming X or last X)
    days_of_week = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for i, day_name in enumerate(days_of_week):
        if day_name in query_lower:
            days_ahead = i - today.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                if "last" in query_lower:
                    days_ahead -= 7
                else:
                    days_ahead += 7  # Default to next occurrence
            target = today + timedelta(days=days_ahead)
            return (target.isoformat(), target.isoformat())

    # ISO date format YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if match:
        date_str = match.group(1)
        return (date_str, date_str)

    # DD-MM-YYYY or DD/MM/YYYY format (common in UK/EU)
    match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", query)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            target = datetime(year, month, day).date()
            return (target.isoformat(), target.isoformat())
        except ValueError:
            pass  # Invalid date

    # Month name + day (e.g., "January 11", "Jan 11th")
    months = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    for month_name, month_num in months.items():
        match = re.search(rf"{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?", query_lower)
        if match:
            day = int(match.group(1))
            try:
                # Assume current year, or next year if date has passed
                target = today.replace(month=month_num, day=day)
                if target < today:
                    target = target.replace(year=today.year + 1)
                return (target.isoformat(), target.isoformat())
            except ValueError:
                pass  # Invalid date

    return None


def has_temporal_reference(query: str) -> bool:
    """
    Check if a query contains any temporal reference.

    Args:
        query: User's search query

    Returns:
        True if the query contains date/time references
    """
    return extract_date_range(query) is not None
