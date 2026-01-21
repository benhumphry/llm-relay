"""
Smart Health live source plugin.

Unified health data interface supporting Oura Ring and Withings devices.
Works with either provider independently or combines data from both.

Accepts natural language queries about sleep, activity, weight, readiness, etc.
Automatically routes to the appropriate provider(s) based on query type.

Supports:
- Oura: Sleep, readiness, activity, heart rate, HRV
- Withings: Weight, body composition, sleep, activity, blood pressure

Requires OAuth tokens configured for at least one provider.
"""

import hashlib
import hmac
import json
import logging
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

# Cache for health data (short TTL since it's personal data)
_health_cache: dict[str, tuple[Any, float]] = {}


class SmartHealthLiveSource(PluginLiveSource):
    """
    Smart Health Provider - unified interface for Oura and Withings.

    Works with either or both providers. Automatically detects which
    providers are configured and routes queries appropriately.

    Examples:
    - "How did I sleep last night?"
    - "What's my weight?"
    - "How's my readiness today?"
    - "Show me my activity this week"
    - "What's my body composition?"
    - "Am I recovered enough to work out?"
    """

    source_type = "oura_withings"
    display_name = "Oura + Withings"
    description = (
        "Health data from Oura Ring and Withings devices with natural language queries"
    )
    category = "health"
    data_type = "health"
    best_for = "Sleep quality, readiness scores, activity tracking, weight, body composition, heart rate, HRV. Works with Oura Ring and/or Withings devices."
    icon = "❤️"
    default_cache_ttl = 300  # 5 minutes

    _abstract = False

    # API endpoints
    OURA_API_URL = "https://api.ouraring.com/v2/usercollection"
    WITHINGS_API_URL = "https://wbsapi.withings.net"

    # Withings measurement type codes
    WITHINGS_MEAS_TYPES = {
        1: ("weight", "kg"),
        4: ("height", "m"),
        5: ("fat_free_mass", "kg"),
        6: ("fat_ratio", "%"),
        8: ("fat_mass", "kg"),
        9: ("diastolic_bp", "mmHg"),
        10: ("systolic_bp", "mmHg"),
        11: ("heart_pulse", "bpm"),
        12: ("temperature", "°C"),
        54: ("spo2", "%"),
        76: ("muscle_mass", "kg"),
        77: ("hydration", "kg"),
        88: ("bone_mass", "kg"),
        91: ("pulse_wave_velocity", "m/s"),
        123: ("vo2_max", "ml/min/kg"),
    }

    # Query type to provider mapping
    QUERY_PROVIDERS = {
        # Oura-specific
        "readiness": ["oura"],
        "hrv": ["oura"],
        "heart_rate": ["oura"],
        "recovery": ["oura"],
        # Withings-specific
        "weight": ["withings"],
        "body_composition": ["withings"],
        "blood_pressure": ["withings"],
        # Both can provide (prefer based on availability)
        "sleep": ["oura", "withings"],
        "activity": ["oura", "withings"],
        # Combined
        "summary": ["oura", "withings"],
        "overview": ["oura", "withings"],
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
                default="Smart Health",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="oura_account_id",
                label="Oura Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=False,
                help_text="Connect your Oura Ring for sleep, readiness, activity, and heart rate data",
                picker_options={"provider": "oura"},
            ),
            FieldDefinition(
                name="withings_account_id",
                label="Withings Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=False,
                help_text="Connect your Withings devices for weight, body composition, and sleep data",
                picker_options={"provider": "withings"},
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query_type",
                description="Type of health data to retrieve",
                param_type="string",
                required=False,
                default="summary",
                examples=[
                    "sleep",
                    "readiness",
                    "activity",
                    "weight",
                    "body_composition",
                    "heart_rate",
                    "hrv",
                    "recovery",
                    "blood_pressure",
                    "summary",
                    "trends",
                ],
            ),
            ParamDefinition(
                name="period",
                description="Time period for the data",
                param_type="string",
                required=False,
                default="today",
                examples=[
                    "today",
                    "yesterday",
                    "last_night",
                    "this_week",
                    "last_week",
                    "this_month",
                    "7_days",
                    "30_days",
                ],
            ),
            ParamDefinition(
                name="date",
                description="Specific date in YYYY-MM-DD format",
                param_type="string",
                required=False,
                examples=["2024-01-15", "2024-01-01"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-health")
        # Convert account IDs to int (they may come as strings from JSON config)
        oura_id = config.get("oura_account_id")
        self.oura_account_id = int(oura_id) if oura_id else None
        withings_id = config.get("withings_account_id")
        self.withings_account_id = int(withings_id) if withings_id else None

        # Token caches
        self._oura_token: Optional[str] = None
        self._oura_token_data: Optional[dict] = None
        self._withings_token: Optional[str] = None
        self._withings_token_data: Optional[dict] = None

        self._client = httpx.Client(timeout=15)

    def _get_cached(self, key: str, ttl: int) -> Optional[Any]:
        """Get from cache if still valid."""
        if key in _health_cache:
            data, cached_at = _health_cache[key]
            if time.time() - cached_at < ttl:
                return data
            del _health_cache[key]
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        """Store in cache."""
        _health_cache[key] = (data, time.time())

    # ==================== OAUTH TOKEN MANAGEMENT ====================

    def _get_oura_token(self) -> Optional[str]:
        """Get valid Oura access token, refreshing if needed."""
        if not self.oura_account_id:
            return None

        # Get stored token
        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(self.oura_account_id)
        if not token_data:
            logger.warning(f"Oura OAuth token {self.oura_account_id} not found")
            return None

        access_token = token_data.get("access_token")
        if not access_token:
            return None

        # Test token validity
        try:
            response = self._client.get(
                f"{self.OURA_API_URL}/personal_info",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code == 200:
                return access_token
        except Exception:
            pass

        # Token expired, try refresh
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not all([refresh_token, client_id, client_secret]):
            logger.error("Cannot refresh Oura token - missing credentials")
            return None

        try:
            response = self._client.post(
                "https://api.ouraring.com/oauth/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            if response.status_code != 200:
                logger.error(f"Oura token refresh failed: {response.text}")
                return None

            new_data = response.json()
            updated = {**token_data, "access_token": new_data.get("access_token")}
            if new_data.get("refresh_token"):
                updated["refresh_token"] = new_data["refresh_token"]

            update_oauth_token_data(self.oura_account_id, updated)
            logger.info("Oura token refreshed successfully")
            return new_data.get("access_token")

        except Exception as e:
            logger.error(f"Oura token refresh error: {e}")
            return None

    def _get_withings_token(self) -> Optional[str]:
        """Get valid Withings access token, refreshing if needed."""
        if not self.withings_account_id:
            return None

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(self.withings_account_id)
        if not token_data:
            logger.warning(f"Withings OAuth token {self.withings_account_id} not found")
            return None

        access_token = token_data.get("access_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return None

        # Test token with a simple call
        try:
            nonce = str(int(time.time()))
            sign_data = f"getdevice,{client_id},{nonce}"
            signature = hmac.new(
                (client_secret or "").encode(),
                sign_data.encode(),
                hashlib.sha256,
            ).hexdigest()

            response = self._client.post(
                f"{self.WITHINGS_API_URL}/v2/user",
                headers={"Authorization": f"Bearer {access_token}"},
                data={
                    "action": "getdevice",
                    "nonce": nonce,
                    "signature": signature,
                    "client_id": client_id,
                },
            )
            if response.json().get("status") == 0:
                self._withings_token_data = token_data
                return access_token
        except Exception:
            pass

        # Token expired, refresh
        refresh_token = token_data.get("refresh_token")
        if not all([refresh_token, client_id, client_secret]):
            logger.error("Cannot refresh Withings token - missing credentials")
            return None

        try:
            response = self._client.post(
                f"{self.WITHINGS_API_URL}/v2/oauth2",
                data={
                    "action": "requesttoken",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            result = response.json()
            if result.get("status") != 0:
                logger.error(f"Withings token refresh failed: {result}")
                return None

            new_data = result.get("body", {})
            updated = {
                **token_data,
                "access_token": new_data.get("access_token"),
                "refresh_token": new_data.get("refresh_token", refresh_token),
            }
            update_oauth_token_data(self.withings_account_id, updated)
            self._withings_token_data = updated
            logger.info("Withings token refreshed successfully")
            return new_data.get("access_token")

        except Exception as e:
            logger.error(f"Withings token refresh error: {e}")
            return None

    def _withings_api_call(
        self, endpoint: str, action: str, params: dict = None
    ) -> dict:
        """Make authenticated Withings API call."""
        access_token = self._get_withings_token()
        if not access_token:
            raise Exception("No valid Withings token")

        if not self._withings_token_data:
            from db.oauth_tokens import get_oauth_token_by_id

            self._withings_token_data = get_oauth_token_by_id(self.withings_account_id)

        client_id = self._withings_token_data.get("client_id", "")
        client_secret = self._withings_token_data.get("client_secret", "")

        nonce = str(int(time.time()))
        sign_data = f"{action},{client_id},{nonce}"
        signature = hmac.new(
            client_secret.encode() if client_secret else b"",
            sign_data.encode(),
            hashlib.sha256,
        ).hexdigest()

        data = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "client_id": client_id,
            **(params or {}),
        }

        response = self._client.post(
            f"{self.WITHINGS_API_URL}/{endpoint}",
            headers={"Authorization": f"Bearer {access_token}"},
            data=data,
        )
        response.raise_for_status()

        result = response.json()
        if result.get("status") != 0:
            raise Exception(f"Withings API error: {result.get('error', 'Unknown')}")

        return result.get("body", {})

    # ==================== DATE PARSING ====================

    def _parse_period(
        self, period: str, specific_date: str = None
    ) -> tuple[datetime, datetime, int]:
        """
        Parse period string into start/end dates and number of days.

        Returns: (start_date, end_date, days)
        """
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if specific_date:
            try:
                target = datetime.strptime(specific_date, "%Y-%m-%d")
                return target, target, 1
            except ValueError:
                pass

        period_lower = period.lower().replace(" ", "_").replace("-", "_")

        if period_lower in ["today", "now"]:
            return today, today, 1
        elif period_lower in ["yesterday", "last_night"]:
            yesterday = today - timedelta(days=1)
            return yesterday, yesterday, 1
        elif period_lower in ["this_week", "week", "7_days", "7d"]:
            start = today - timedelta(days=6)
            return start, today, 7
        elif period_lower in ["last_week"]:
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
            return start, end, 7
        elif period_lower in ["this_month", "month", "30_days", "30d"]:
            start = today - timedelta(days=29)
            return start, today, 30
        elif period_lower in ["last_month"]:
            first_of_month = today.replace(day=1)
            end = first_of_month - timedelta(days=1)
            start = end.replace(day=1)
            days = (end - start).days + 1
            return start, end, days
        elif period_lower in ["3_days", "3d"]:
            start = today - timedelta(days=2)
            return start, today, 3
        elif period_lower in ["14_days", "2_weeks", "2w"]:
            start = today - timedelta(days=13)
            return start, today, 14
        else:
            # Default to today
            return today, today, 1

    def _get_comparison_periods(
        self, period: str
    ) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime], str]:
        """
        Get current and previous periods for comparison.

        Returns: ((curr_start, curr_end), (prev_start, prev_end), period_label)
        """
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        period_lower = period.lower().replace(" ", "_").replace("-", "_")

        if period_lower in ["today", "now", "yesterday"]:
            # Compare today vs yesterday, or yesterday vs day before
            if period_lower == "yesterday":
                curr_end = today - timedelta(days=1)
            else:
                curr_end = today
            curr_start = curr_end
            prev_end = curr_end - timedelta(days=1)
            prev_start = prev_end
            return (curr_start, curr_end), (prev_start, prev_end), "day over day"

        elif period_lower in ["this_week", "week", "7_days", "7d"]:
            # This week vs last week
            curr_start = today - timedelta(days=6)
            curr_end = today
            prev_end = curr_start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=6)
            return (curr_start, curr_end), (prev_start, prev_end), "week over week"

        elif period_lower in ["last_week"]:
            # Last week vs week before
            curr_end = today - timedelta(days=today.weekday() + 1)
            curr_start = curr_end - timedelta(days=6)
            prev_end = curr_start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=6)
            return (curr_start, curr_end), (prev_start, prev_end), "week over week"

        elif period_lower in ["this_month", "month", "30_days", "30d"]:
            # This month vs last month
            curr_start = today - timedelta(days=29)
            curr_end = today
            prev_end = curr_start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=29)
            return (curr_start, curr_end), (prev_start, prev_end), "month over month"

        elif period_lower in ["14_days", "2_weeks", "2w"]:
            # Last 2 weeks vs 2 weeks before
            curr_start = today - timedelta(days=13)
            curr_end = today
            prev_end = curr_start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=13)
            return (curr_start, curr_end), (prev_start, prev_end), "2 weeks comparison"

        else:
            # Default: today vs yesterday
            return (
                (today, today),
                (today - timedelta(days=1), today - timedelta(days=1)),
                "day over day",
            )

    def _trend_indicator(
        self, current: float, previous: float, higher_is_better: bool = True
    ) -> str:
        """
        Generate trend indicator with arrow and percentage change.

        Args:
            current: Current value
            previous: Previous value
            higher_is_better: True if increasing values are positive
        """
        if previous == 0:
            if current > 0:
                return "📈 New" if higher_is_better else "📉 New"
            return "➡️ No change"

        change = ((current - previous) / previous) * 100
        abs_change = abs(change)

        if abs_change < 1:
            return "➡️ Stable"

        if change > 0:
            if higher_is_better:
                if abs_change >= 20:
                    return f"📈📈 +{abs_change:.1f}%"
                return f"📈 +{abs_change:.1f}%"
            else:
                if abs_change >= 20:
                    return f"📉📉 +{abs_change:.1f}%"
                return f"📉 +{abs_change:.1f}%"
        else:
            if higher_is_better:
                if abs_change >= 20:
                    return f"📉📉 {change:.1f}%"
                return f"📉 {change:.1f}%"
            else:
                if abs_change >= 20:
                    return f"📈📈 {change:.1f}%"
                return f"📈 {change:.1f}%"

    def _calculate_avg(self, items: list, key: str) -> float:
        """Calculate average of a key from list of dicts."""
        values = [item.get(key, 0) for item in items if item.get(key) is not None]
        return sum(values) / len(values) if values else 0

    def _calculate_total(self, items: list, key: str) -> float:
        """Calculate total of a key from list of dicts."""
        return sum(item.get(key, 0) for item in items if item.get(key) is not None)

    # ==================== OURA DATA FETCHING ====================

    def _fetch_oura_sleep(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Oura sleep data."""
        token = self._get_oura_token()
        if not token:
            return None

        try:
            response = self._client.get(
                f"{self.OURA_API_URL}/daily_sleep",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Oura sleep fetch error: {e}")
            return None

    def _fetch_oura_readiness(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Oura readiness data."""
        token = self._get_oura_token()
        if not token:
            return None

        try:
            response = self._client.get(
                f"{self.OURA_API_URL}/daily_readiness",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Oura readiness fetch error: {e}")
            return None

    def _fetch_oura_activity(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Oura activity data."""
        token = self._get_oura_token()
        if not token:
            return None

        try:
            response = self._client.get(
                f"{self.OURA_API_URL}/daily_activity",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Oura activity fetch error: {e}")
            return None

    def _fetch_oura_heart_rate(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Oura heart rate data."""
        token = self._get_oura_token()
        if not token:
            return None

        try:
            response = self._client.get(
                f"{self.OURA_API_URL}/heartrate",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start_datetime": start_date.strftime("%Y-%m-%dT00:00:00+00:00"),
                    "end_datetime": end_date.strftime("%Y-%m-%dT23:59:59+00:00"),
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Oura heart rate fetch error: {e}")
            return None

    # ==================== WITHINGS DATA FETCHING ====================

    def _fetch_withings_weight(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Withings weight and body composition."""
        if not self._get_withings_token():
            return None

        try:
            return self._withings_api_call(
                "measure",
                "getmeas",
                {
                    "startdate": int(start_date.timestamp()),
                    "enddate": int(end_date.timestamp()) + 86400,
                    "meastypes": "1,5,6,8,76,77,88",
                    "category": 1,
                },
            )
        except Exception as e:
            logger.warning(f"Withings weight fetch error: {e}")
            return None

    def _fetch_withings_sleep(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Withings sleep data."""
        if not self._get_withings_token():
            return None

        try:
            return self._withings_api_call(
                "v2/sleep",
                "getsummary",
                {
                    "startdateymd": start_date.strftime("%Y-%m-%d"),
                    "enddateymd": end_date.strftime("%Y-%m-%d"),
                },
            )
        except Exception as e:
            logger.warning(f"Withings sleep fetch error: {e}")
            return None

    def _fetch_withings_activity(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Withings activity data."""
        if not self._get_withings_token():
            return None

        try:
            return self._withings_api_call(
                "v2/measure",
                "getactivity",
                {
                    "startdateymd": start_date.strftime("%Y-%m-%d"),
                    "enddateymd": end_date.strftime("%Y-%m-%d"),
                },
            )
        except Exception as e:
            logger.warning(f"Withings activity fetch error: {e}")
            return None

    def _fetch_withings_blood_pressure(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[dict]:
        """Fetch Withings blood pressure data."""
        if not self._get_withings_token():
            return None

        try:
            return self._withings_api_call(
                "measure",
                "getmeas",
                {
                    "startdate": int(start_date.timestamp()),
                    "enddate": int(end_date.timestamp()) + 86400,
                    "meastypes": "9,10,11",  # diastolic, systolic, heart rate
                    "category": 1,
                },
            )
        except Exception as e:
            logger.warning(f"Withings BP fetch error: {e}")
            return None

    # ==================== FORMATTING ====================

    def _format_sleep(
        self, oura_data: Optional[dict], withings_data: Optional[dict], days: int
    ) -> str:
        """Format sleep data from available sources."""
        lines = ["**Sleep Data**"]

        # Oura sleep (preferred for scores)
        oura_items = oura_data.get("data", []) if oura_data else []
        for item in oura_items[:days]:
            day = item.get("day", "Unknown")
            score = item.get("score")
            contributors = item.get("contributors", {})

            lines.append(f"\n📅 **{day}** (Oura)")
            if score:
                emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                lines.append(f"  {emoji} Sleep Score: {score}/100")

            if contributors:
                deep = contributors.get("deep_sleep")
                rem = contributors.get("rem_sleep")
                efficiency = contributors.get("efficiency")
                if deep:
                    lines.append(f"  Deep Sleep: {deep}/100")
                if rem:
                    lines.append(f"  REM Sleep: {rem}/100")
                if efficiency:
                    lines.append(f"  Efficiency: {efficiency}/100")

        # Withings sleep (for duration details)
        withings_items = withings_data.get("series", []) if withings_data else []
        if withings_items and not oura_items:
            # Only show Withings if no Oura data
            for item in withings_items[:days]:
                day = item.get("date", "Unknown")
                data = item.get("data", {})
                total = data.get("total_sleep_time", 0)

                if total:
                    hours = total // 3600
                    minutes = (total % 3600) // 60
                    lines.append(f"\n📅 **{day}** (Withings)")
                    lines.append(f"  🛏️ Total Sleep: {hours}h {minutes}m")

                    deep = data.get("deepsleepduration", 0) // 60
                    light = data.get("lightsleepduration", 0) // 60
                    rem = data.get("remsleepduration", 0) // 60
                    if deep or light or rem:
                        lines.append(
                            f"  Phases: {deep}m deep / {light}m light / {rem}m REM"
                        )

                    score = data.get("sleep_score")
                    if score:
                        lines.append(f"  Score: {score}/100")

        if len(lines) == 1:
            lines.append("\nNo sleep data available for this period.")

        return "\n".join(lines)

    def _format_readiness(self, data: Optional[dict], days: int) -> str:
        """Format Oura readiness data."""
        lines = ["**Readiness & Recovery**"]

        items = data.get("data", []) if data else []
        if not items:
            lines.append("\nNo readiness data available (requires Oura Ring).")
            return "\n".join(lines)

        for item in items[:days]:
            day = item.get("day", "Unknown")
            score = item.get("score")
            contributors = item.get("contributors", {})
            temp = item.get("temperature_deviation")

            lines.append(f"\n📅 **{day}**")
            if score:
                emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                lines.append(f"  {emoji} Readiness Score: {score}/100")

                # Workout recommendation
                if score >= 85:
                    lines.append(f"  💪 Great day for intense training!")
                elif score >= 70:
                    lines.append(f"  🚶 Moderate activity recommended")
                else:
                    lines.append(f"  🧘 Focus on recovery today")

            if temp is not None:
                lines.append(f"  🌡️ Body Temp: {temp:+.1f}°C from baseline")

            if contributors:
                hrv = contributors.get("hrv_balance")
                recovery = contributors.get("recovery_index")
                rhr = contributors.get("resting_heart_rate")
                if hrv:
                    lines.append(f"  HRV Balance: {hrv}/100")
                if recovery:
                    lines.append(f"  Recovery Index: {recovery}/100")
                if rhr:
                    lines.append(f"  Resting HR: {rhr}/100")

        return "\n".join(lines)

    def _format_activity(
        self, oura_data: Optional[dict], withings_data: Optional[dict], days: int
    ) -> str:
        """Format activity data from available sources."""
        lines = ["**Activity Data**"]

        # Oura activity
        oura_items = oura_data.get("data", []) if oura_data else []
        for item in oura_items[:days]:
            day = item.get("day", "Unknown")
            score = item.get("score")
            steps = item.get("steps", 0)
            calories = item.get("active_calories", 0)
            distance = item.get("equivalent_walking_distance", 0)

            lines.append(f"\n📅 **{day}** (Oura)")
            if score:
                emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                lines.append(f"  {emoji} Activity Score: {score}/100")
            lines.append(f"  👟 Steps: {steps:,}")
            if distance:
                lines.append(f"  📏 Distance: {distance / 1000:.1f} km")
            if calories:
                lines.append(f"  🔥 Active Calories: {calories:,}")

        # Withings activity (if no Oura)
        withings_items = withings_data.get("activities", []) if withings_data else []
        if withings_items and not oura_items:
            for item in withings_items[:days]:
                day = item.get("date", "Unknown")
                steps = item.get("steps", 0)
                distance = item.get("distance", 0)
                calories = item.get("calories", 0)

                lines.append(f"\n📅 **{day}** (Withings)")
                lines.append(f"  👟 Steps: {steps:,}")
                if distance:
                    lines.append(f"  📏 Distance: {distance / 1000:.1f} km")
                if calories:
                    lines.append(f"  🔥 Calories: {calories:,}")

        if len(lines) == 1:
            lines.append("\nNo activity data available for this period.")

        return "\n".join(lines)

    def _format_weight(self, data: Optional[dict]) -> str:
        """Format Withings weight and body composition."""
        lines = ["**Weight & Body Composition**"]

        if not data:
            lines.append("\nNo weight data available (requires Withings scale).")
            return "\n".join(lines)

        measuregrps = data.get("measuregrps", [])
        if not measuregrps:
            lines.append("\nNo recent weight measurements.")
            return "\n".join(lines)

        # Parse measurements
        for grp in measuregrps[:5]:
            date = datetime.fromtimestamp(grp.get("date", 0)).strftime("%Y-%m-%d")
            measures = {}

            for m in grp.get("measures", []):
                mtype = m.get("type")
                value = m.get("value", 0) * (10 ** m.get("unit", 0))
                if mtype in self.WITHINGS_MEAS_TYPES:
                    name, unit = self.WITHINGS_MEAS_TYPES[mtype]
                    measures[name] = round(value, 2)

            if measures:
                lines.append(f"\n📅 **{date}**")
                if "weight" in measures:
                    lines.append(f"  ⚖️ Weight: {measures['weight']:.1f} kg")
                if "fat_ratio" in measures:
                    lines.append(f"  📊 Body Fat: {measures['fat_ratio']:.1f}%")
                if "muscle_mass" in measures:
                    lines.append(f"  💪 Muscle Mass: {measures['muscle_mass']:.1f} kg")
                if "bone_mass" in measures:
                    lines.append(f"  🦴 Bone Mass: {measures['bone_mass']:.1f} kg")
                if "hydration" in measures:
                    lines.append(f"  💧 Hydration: {measures['hydration']:.1f} kg")

        return "\n".join(lines)

    def _format_heart_rate(self, data: Optional[dict]) -> str:
        """Format Oura heart rate data."""
        lines = ["**Heart Rate**"]

        if not data:
            lines.append("\nNo heart rate data available (requires Oura Ring).")
            return "\n".join(lines)

        items = data.get("data", [])
        if not items:
            lines.append("\nNo recent heart rate readings.")
            return "\n".join(lines)

        bpms = [item.get("bpm", 0) for item in items if item.get("bpm")]
        if bpms:
            avg = sum(bpms) / len(bpms)
            lines.append(f"\n❤️ Latest: {bpms[-1]} bpm")
            lines.append(f"📊 Average: {avg:.0f} bpm")
            lines.append(f"📈 Range: {min(bpms)} - {max(bpms)} bpm")
            lines.append(f"🔢 Readings: {len(bpms)}")

            # Recent readings
            lines.append("\n**Recent:**")
            for item in items[-5:]:
                ts = item.get("timestamp", "")
                bpm = item.get("bpm")
                if ts and bpm:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        time_str = dt.strftime("%H:%M")
                        lines.append(f"  {time_str}: {bpm} bpm")
                    except ValueError:
                        pass

        return "\n".join(lines)

    def _format_blood_pressure(self, data: Optional[dict]) -> str:
        """Format Withings blood pressure data."""
        lines = ["**Blood Pressure**"]

        if not data:
            lines.append("\nNo blood pressure data available (requires Withings BPM).")
            return "\n".join(lines)

        measuregrps = data.get("measuregrps", [])
        if not measuregrps:
            lines.append("\nNo recent blood pressure readings.")
            return "\n".join(lines)

        for grp in measuregrps[:5]:
            date = datetime.fromtimestamp(grp.get("date", 0)).strftime("%Y-%m-%d %H:%M")
            systolic = None
            diastolic = None
            pulse = None

            for m in grp.get("measures", []):
                mtype = m.get("type")
                value = m.get("value", 0) * (10 ** m.get("unit", 0))
                if mtype == 10:
                    systolic = int(value)
                elif mtype == 9:
                    diastolic = int(value)
                elif mtype == 11:
                    pulse = int(value)

            if systolic and diastolic:
                lines.append(f"\n📅 **{date}**")
                lines.append(f"  🩺 {systolic}/{diastolic} mmHg")
                if pulse:
                    lines.append(f"  ❤️ Pulse: {pulse} bpm")

                # Classification
                if systolic < 120 and diastolic < 80:
                    lines.append(f"  ✅ Normal")
                elif systolic < 130 and diastolic < 80:
                    lines.append(f"  🟡 Elevated")
                elif systolic < 140 or diastolic < 90:
                    lines.append(f"  🟠 High (Stage 1)")
                else:
                    lines.append(f"  🔴 High (Stage 2)")

        return "\n".join(lines)

    def _format_summary(
        self,
        oura_sleep: Optional[dict],
        oura_readiness: Optional[dict],
        oura_activity: Optional[dict],
        withings_weight: Optional[dict],
        withings_sleep: Optional[dict],
        withings_activity: Optional[dict],
    ) -> str:
        """Format combined health summary."""
        lines = ["**Health Summary**"]
        lines.append(f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")

        has_data = False

        # Sleep (prefer Oura)
        oura_sleep_items = oura_sleep.get("data", []) if oura_sleep else []
        if oura_sleep_items:
            latest = oura_sleep_items[-1]
            score = latest.get("score")
            day = latest.get("day", "")
            if score:
                has_data = True
                emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                lines.append(f"\n**🛏️ Sleep** ({day})")
                lines.append(f"  {emoji} Score: {score}/100")
        elif withings_sleep:
            series = withings_sleep.get("series", [])
            if series:
                latest = series[-1]
                total = latest.get("data", {}).get("total_sleep_time", 0)
                if total:
                    has_data = True
                    hours = total // 3600
                    minutes = (total % 3600) // 60
                    lines.append(f"\n**🛏️ Sleep** ({latest.get('date', '')})")
                    lines.append(f"  Duration: {hours}h {minutes}m")

        # Readiness (Oura only)
        readiness_items = oura_readiness.get("data", []) if oura_readiness else []
        if readiness_items:
            latest = readiness_items[-1]
            score = latest.get("score")
            day = latest.get("day", "")
            temp = latest.get("temperature_deviation")
            if score:
                has_data = True
                emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                lines.append(f"\n**💪 Readiness** ({day})")
                lines.append(f"  {emoji} Score: {score}/100")
                if temp is not None:
                    lines.append(f"  🌡️ Body Temp: {temp:+.1f}°C")

        # Activity (prefer Oura)
        oura_activity_items = oura_activity.get("data", []) if oura_activity else []
        if oura_activity_items:
            latest = oura_activity_items[-1]
            steps = latest.get("steps", 0)
            score = latest.get("score")
            day = latest.get("day", "")
            if steps:
                has_data = True
                lines.append(f"\n**🏃 Activity** ({day})")
                lines.append(f"  👟 Steps: {steps:,}")
                if score:
                    lines.append(f"  Score: {score}/100")
        elif withings_activity:
            activities = withings_activity.get("activities", [])
            if activities:
                latest = activities[-1]
                steps = latest.get("steps", 0)
                if steps:
                    has_data = True
                    lines.append(f"\n**🏃 Activity** ({latest.get('date', '')})")
                    lines.append(f"  👟 Steps: {steps:,}")

        # Weight (Withings only)
        if withings_weight:
            measuregrps = withings_weight.get("measuregrps", [])
            if measuregrps:
                latest = measuregrps[0]
                date = datetime.fromtimestamp(latest.get("date", 0)).strftime(
                    "%Y-%m-%d"
                )
                for m in latest.get("measures", []):
                    if m.get("type") == 1:  # Weight
                        weight = m.get("value", 0) * (10 ** m.get("unit", 0))
                        has_data = True
                        lines.append(f"\n**⚖️ Weight** ({date})")
                        lines.append(f"  {weight:.1f} kg")
                        break

        if not has_data:
            lines.append(
                "\nNo health data available. Connect Oura Ring and/or Withings devices."
            )

        return "\n".join(lines)

    def _fetch_trends(
        self, period: str, has_oura: bool, has_withings: bool
    ) -> LiveDataResult:
        """
        Fetch and compare health data across two periods.

        Compares current period vs previous period (e.g., this week vs last week).
        """
        # Get comparison periods
        (curr_start, curr_end), (prev_start, prev_end), period_label = (
            self._get_comparison_periods(period)
        )

        lines = ["**Health Trends**"]
        lines.append(f"_Comparing {period_label}_")
        lines.append(
            f"_Current: {curr_start.strftime('%b %d')} - {curr_end.strftime('%b %d')}_"
        )
        lines.append(
            f"_Previous: {prev_start.strftime('%b %d')} - {prev_end.strftime('%b %d')}_"
        )

        has_data = False
        data = {"current_period": {}, "previous_period": {}, "trends": {}}

        # Sleep trends
        if has_oura:
            curr_sleep = self._fetch_oura_sleep(curr_start, curr_end)
            prev_sleep = self._fetch_oura_sleep(prev_start, prev_end)

            curr_items = curr_sleep.get("data", []) if curr_sleep else []
            prev_items = prev_sleep.get("data", []) if prev_sleep else []

            if curr_items or prev_items:
                has_data = True
                lines.append("\n**🛏️ Sleep**")

                # Sleep score
                curr_score = self._calculate_avg(curr_items, "score")
                prev_score = self._calculate_avg(prev_items, "score")
                if curr_score or prev_score:
                    trend = self._trend_indicator(
                        curr_score, prev_score, higher_is_better=True
                    )
                    lines.append(
                        f"  Score: {curr_score:.0f} vs {prev_score:.0f} {trend}"
                    )
                    data["trends"]["sleep_score"] = {
                        "current": curr_score,
                        "previous": prev_score,
                    }

                # Total sleep time (in seconds)
                curr_duration = self._calculate_avg(curr_items, "total_sleep_duration")
                prev_duration = self._calculate_avg(prev_items, "total_sleep_duration")
                if curr_duration or prev_duration:
                    curr_hrs = curr_duration / 3600
                    prev_hrs = prev_duration / 3600
                    trend = self._trend_indicator(
                        curr_hrs, prev_hrs, higher_is_better=True
                    )
                    lines.append(
                        f"  Duration: {curr_hrs:.1f}h vs {prev_hrs:.1f}h {trend}"
                    )

        elif has_withings:
            curr_sleep = self._fetch_withings_sleep(curr_start, curr_end)
            prev_sleep = self._fetch_withings_sleep(prev_start, prev_end)

            curr_series = curr_sleep.get("series", []) if curr_sleep else []
            prev_series = prev_sleep.get("series", []) if prev_sleep else []

            if curr_series or prev_series:
                has_data = True
                lines.append("\n**🛏️ Sleep**")

                # Calculate average sleep time
                curr_durations = [
                    s.get("data", {}).get("total_sleep_time", 0) for s in curr_series
                ]
                prev_durations = [
                    s.get("data", {}).get("total_sleep_time", 0) for s in prev_series
                ]

                curr_avg = (
                    sum(curr_durations) / len(curr_durations) if curr_durations else 0
                )
                prev_avg = (
                    sum(prev_durations) / len(prev_durations) if prev_durations else 0
                )

                if curr_avg or prev_avg:
                    curr_hrs = curr_avg / 3600
                    prev_hrs = prev_avg / 3600
                    trend = self._trend_indicator(
                        curr_hrs, prev_hrs, higher_is_better=True
                    )
                    lines.append(
                        f"  Avg Duration: {curr_hrs:.1f}h vs {prev_hrs:.1f}h {trend}"
                    )

        # Readiness trends (Oura only)
        if has_oura:
            curr_readiness = self._fetch_oura_readiness(curr_start, curr_end)
            prev_readiness = self._fetch_oura_readiness(prev_start, prev_end)

            curr_items = curr_readiness.get("data", []) if curr_readiness else []
            prev_items = prev_readiness.get("data", []) if prev_readiness else []

            if curr_items or prev_items:
                has_data = True
                lines.append("\n**💪 Readiness**")

                curr_score = self._calculate_avg(curr_items, "score")
                prev_score = self._calculate_avg(prev_items, "score")
                if curr_score or prev_score:
                    trend = self._trend_indicator(
                        curr_score, prev_score, higher_is_better=True
                    )
                    lines.append(
                        f"  Score: {curr_score:.0f} vs {prev_score:.0f} {trend}"
                    )
                    data["trends"]["readiness_score"] = {
                        "current": curr_score,
                        "previous": prev_score,
                    }

                # Temperature deviation
                curr_temps = [
                    i.get("temperature_deviation", 0)
                    for i in curr_items
                    if i.get("temperature_deviation") is not None
                ]
                prev_temps = [
                    i.get("temperature_deviation", 0)
                    for i in prev_items
                    if i.get("temperature_deviation") is not None
                ]
                if curr_temps or prev_temps:
                    curr_temp = sum(curr_temps) / len(curr_temps) if curr_temps else 0
                    prev_temp = sum(prev_temps) / len(prev_temps) if prev_temps else 0
                    # For temp deviation, closer to 0 is better
                    trend = self._trend_indicator(
                        abs(prev_temp), abs(curr_temp), higher_is_better=True
                    )
                    lines.append(
                        f"  Body Temp: {curr_temp:+.2f}°C vs {prev_temp:+.2f}°C {trend}"
                    )

        # Activity trends
        if has_oura:
            curr_activity = self._fetch_oura_activity(curr_start, curr_end)
            prev_activity = self._fetch_oura_activity(prev_start, prev_end)

            curr_items = curr_activity.get("data", []) if curr_activity else []
            prev_items = prev_activity.get("data", []) if prev_activity else []

            if curr_items or prev_items:
                has_data = True
                lines.append("\n**🏃 Activity**")

                # Steps
                curr_steps = self._calculate_avg(curr_items, "steps")
                prev_steps = self._calculate_avg(prev_items, "steps")
                if curr_steps or prev_steps:
                    trend = self._trend_indicator(
                        curr_steps, prev_steps, higher_is_better=True
                    )
                    lines.append(
                        f"  Avg Steps: {curr_steps:,.0f} vs {prev_steps:,.0f} {trend}"
                    )
                    data["trends"]["steps"] = {
                        "current": curr_steps,
                        "previous": prev_steps,
                    }

                # Calories
                curr_cal = self._calculate_avg(curr_items, "active_calories")
                prev_cal = self._calculate_avg(prev_items, "active_calories")
                if curr_cal or prev_cal:
                    trend = self._trend_indicator(
                        curr_cal, prev_cal, higher_is_better=True
                    )
                    lines.append(
                        f"  Avg Calories: {curr_cal:,.0f} vs {prev_cal:,.0f} {trend}"
                    )

                # Activity score
                curr_score = self._calculate_avg(curr_items, "score")
                prev_score = self._calculate_avg(prev_items, "score")
                if curr_score or prev_score:
                    trend = self._trend_indicator(
                        curr_score, prev_score, higher_is_better=True
                    )
                    lines.append(
                        f"  Score: {curr_score:.0f} vs {prev_score:.0f} {trend}"
                    )

        elif has_withings:
            curr_activity = self._fetch_withings_activity(curr_start, curr_end)
            prev_activity = self._fetch_withings_activity(prev_start, prev_end)

            curr_items = curr_activity.get("activities", []) if curr_activity else []
            prev_items = prev_activity.get("activities", []) if prev_activity else []

            if curr_items or prev_items:
                has_data = True
                lines.append("\n**🏃 Activity**")

                curr_steps = self._calculate_avg(curr_items, "steps")
                prev_steps = self._calculate_avg(prev_items, "steps")
                if curr_steps or prev_steps:
                    trend = self._trend_indicator(
                        curr_steps, prev_steps, higher_is_better=True
                    )
                    lines.append(
                        f"  Avg Steps: {curr_steps:,.0f} vs {prev_steps:,.0f} {trend}"
                    )

        # Weight trends (Withings only)
        if has_withings:
            curr_weight = self._fetch_withings_weight(curr_start, curr_end)
            prev_weight = self._fetch_withings_weight(prev_start, prev_end)

            curr_grps = curr_weight.get("measuregrps", []) if curr_weight else []
            prev_grps = prev_weight.get("measuregrps", []) if prev_weight else []

            # Extract weight values
            def extract_weights(grps):
                weights = []
                for grp in grps:
                    for m in grp.get("measures", []):
                        if m.get("type") == 1:  # Weight
                            weights.append(m.get("value", 0) * (10 ** m.get("unit", 0)))
                return weights

            curr_weights = extract_weights(curr_grps)
            prev_weights = extract_weights(prev_grps)

            if curr_weights or prev_weights:
                has_data = True
                lines.append("\n**⚖️ Weight**")

                curr_avg = sum(curr_weights) / len(curr_weights) if curr_weights else 0
                prev_avg = sum(prev_weights) / len(prev_weights) if prev_weights else 0

                if curr_avg and prev_avg:
                    # For weight, lower is often considered better (weight loss goal)
                    # But we'll show neutral trend and let the value speak
                    change = curr_avg - prev_avg
                    if abs(change) < 0.1:
                        trend = "➡️ Stable"
                    elif change > 0:
                        trend = f"📈 +{change:.1f} kg"
                    else:
                        trend = f"📉 {change:.1f} kg"
                    lines.append(
                        f"  Avg: {curr_avg:.1f} kg vs {prev_avg:.1f} kg {trend}"
                    )
                    data["trends"]["weight"] = {
                        "current": curr_avg,
                        "previous": prev_avg,
                    }

                # Body fat if available
                def extract_body_fat(grps):
                    fats = []
                    for grp in grps:
                        for m in grp.get("measures", []):
                            if m.get("type") == 6:  # Fat ratio
                                fats.append(
                                    m.get("value", 0) * (10 ** m.get("unit", 0))
                                )
                    return fats

                curr_fats = extract_body_fat(curr_grps)
                prev_fats = extract_body_fat(prev_grps)

                if curr_fats and prev_fats:
                    curr_fat = sum(curr_fats) / len(curr_fats)
                    prev_fat = sum(prev_fats) / len(prev_fats)
                    change = curr_fat - prev_fat
                    if abs(change) < 0.1:
                        trend = "➡️ Stable"
                    elif change > 0:
                        trend = f"📈 +{change:.1f}%"
                    else:
                        trend = f"📉 {change:.1f}%"
                    lines.append(
                        f"  Body Fat: {curr_fat:.1f}% vs {prev_fat:.1f}% {trend}"
                    )

        if not has_data:
            lines.append("\nNo trend data available for this period.")

        # Summary insight
        if has_data and data.get("trends"):
            lines.append("\n**📊 Summary**")
            improving = []
            declining = []

            for metric, values in data["trends"].items():
                curr = values["current"]
                prev = values["previous"]
                if prev > 0:
                    change = ((curr - prev) / prev) * 100
                    metric_name = metric.replace("_", " ").title()
                    if change > 5:
                        improving.append(metric_name)
                    elif change < -5:
                        declining.append(metric_name)

            if improving:
                lines.append(f"  ✅ Improving: {', '.join(improving)}")
            if declining:
                lines.append(f"  ⚠️ Declining: {', '.join(declining)}")
            if not improving and not declining:
                lines.append("  ➡️ All metrics stable")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data=data,
            cache_ttl=self.default_cache_ttl,
        )

    # ==================== MAIN FETCH ====================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch health data based on parameters."""
        query_type = params.get("query_type", "summary").lower()
        period = params.get("period", "today")
        specific_date = params.get("date")

        # Check what providers are available
        has_oura = self._get_oura_token() is not None
        has_withings = self._get_withings_token() is not None

        if not has_oura and not has_withings:
            return LiveDataResult(
                success=False,
                error="No health providers configured. Please connect Oura Ring or Withings in admin settings.",
            )

        # Parse time period
        start_date, end_date, days = self._parse_period(period, specific_date)

        # Route to appropriate handler
        try:
            if query_type in ["sleep", "last_night"]:
                oura_data = (
                    self._fetch_oura_sleep(start_date, end_date) if has_oura else None
                )
                withings_data = (
                    self._fetch_withings_sleep(start_date, end_date)
                    if has_withings
                    else None
                )
                formatted = self._format_sleep(oura_data, withings_data, days)
                data = {"oura": oura_data, "withings": withings_data}

            elif query_type in ["readiness", "recovery"]:
                oura_data = (
                    self._fetch_oura_readiness(start_date, end_date)
                    if has_oura
                    else None
                )
                formatted = self._format_readiness(oura_data, days)
                data = {"oura": oura_data}

            elif query_type in ["activity", "steps", "exercise"]:
                oura_data = (
                    self._fetch_oura_activity(start_date, end_date)
                    if has_oura
                    else None
                )
                withings_data = (
                    self._fetch_withings_activity(start_date, end_date)
                    if has_withings
                    else None
                )
                formatted = self._format_activity(oura_data, withings_data, days)
                data = {"oura": oura_data, "withings": withings_data}

            elif query_type in ["weight", "body_composition", "body"]:
                withings_data = (
                    self._fetch_withings_weight(start_date, end_date)
                    if has_withings
                    else None
                )
                formatted = self._format_weight(withings_data)
                data = {"withings": withings_data}

            elif query_type in ["heart_rate", "hr", "pulse"]:
                oura_data = (
                    self._fetch_oura_heart_rate(start_date, end_date)
                    if has_oura
                    else None
                )
                formatted = self._format_heart_rate(oura_data)
                data = {"oura": oura_data}

            elif query_type in ["blood_pressure", "bp"]:
                withings_data = (
                    self._fetch_withings_blood_pressure(start_date, end_date)
                    if has_withings
                    else None
                )
                formatted = self._format_blood_pressure(withings_data)
                data = {"withings": withings_data}

            elif query_type in ["hrv"]:
                # HRV is part of readiness data
                oura_data = (
                    self._fetch_oura_readiness(start_date, end_date)
                    if has_oura
                    else None
                )
                formatted = self._format_readiness(oura_data, days)
                data = {"oura": oura_data}

            elif query_type in ["trends", "trend", "comparison", "compare", "progress"]:
                # Trends comparison - handles its own data fetching
                return self._fetch_trends(period, has_oura, has_withings)

            else:  # summary / overview
                oura_sleep = (
                    self._fetch_oura_sleep(start_date, end_date) if has_oura else None
                )
                oura_readiness = (
                    self._fetch_oura_readiness(start_date, end_date)
                    if has_oura
                    else None
                )
                oura_activity = (
                    self._fetch_oura_activity(start_date, end_date)
                    if has_oura
                    else None
                )
                withings_weight = (
                    self._fetch_withings_weight(start_date, end_date)
                    if has_withings
                    else None
                )
                withings_sleep = (
                    self._fetch_withings_sleep(start_date, end_date)
                    if has_withings
                    else None
                )
                withings_activity = (
                    self._fetch_withings_activity(start_date, end_date)
                    if has_withings
                    else None
                )

                formatted = self._format_summary(
                    oura_sleep,
                    oura_readiness,
                    oura_activity,
                    withings_weight,
                    withings_sleep,
                    withings_activity,
                )
                data = {
                    "oura": {
                        "sleep": oura_sleep,
                        "readiness": oura_readiness,
                        "activity": oura_activity,
                    },
                    "withings": {
                        "weight": withings_weight,
                        "sleep": withings_sleep,
                        "activity": withings_activity,
                    },
                }

            return LiveDataResult(
                success=True,
                formatted=formatted,
                data=data,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Smart Health fetch error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if at least one provider is configured."""
        return bool(self.oura_account_id or self.withings_account_id)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to configured providers."""
        results = []

        if self.oura_account_id:
            if self._get_oura_token():
                results.append("Oura: ✓")
            else:
                results.append("Oura: ✗ (token invalid)")

        if self.withings_account_id:
            if self._get_withings_token():
                results.append("Withings: ✓")
            else:
                results.append("Withings: ✗ (token invalid)")

        if not results:
            return False, "No health providers configured"

        success = any("✓" in r for r in results)
        return success, " | ".join(results)
