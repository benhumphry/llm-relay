"""
Smart Weather live source plugin.

Unified weather interface combining Open-Meteo (free) and AccuWeather (premium).
Accepts natural language locations and time periods.

Uses:
- Open-Meteo API (free, no API key) - Always available for forecasts
- AccuWeather API (optional, requires key) - Enhanced current conditions and forecasts

When both are configured, AccuWeather is preferred for current conditions
(more detailed) and can provide extended forecasts up to 15 days.
"""

import logging
import os
import re
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

# Global caches with TTL
_geocode_cache: dict[str, tuple[Any, float]] = {}  # 30-day TTL
_weather_cache: dict[str, tuple[Any, float]] = {}  # 30-minute TTL
_location_key_cache: dict[
    str, tuple[str, float]
] = {}  # 30-day TTL for AccuWeather keys


class SmartWeatherLiveSource(PluginLiveSource):
    """
    Smart Weather Provider - unified Open-Meteo + AccuWeather.

    Accepts natural language locations and time periods:
    - "Weather in London tomorrow"
    - "Paris forecast this weekend"
    - "Will it rain in Tokyo next week?"

    Data sources:
    - Open-Meteo (free): Always available, good forecasts up to 16 days
    - AccuWeather (optional): Better current conditions, forecasts up to 15 days

    When AccuWeather API key is configured, it's used for current conditions
    and can supplement Open-Meteo data. Without it, Open-Meteo provides
    everything needed.
    """

    source_type = "open_meteo_enhanced"
    display_name = "Open-Meteo + AccuWeather"
    description = (
        "Weather forecasts with Open-Meteo (free) and optional AccuWeather enhancement"
    )
    category = "weather"
    data_type = "weather"
    best_for = "Weather forecasts, current conditions, rain predictions. Supports natural language like 'weather in Paris tomorrow' or 'London this weekend'."
    icon = "🌤️"
    default_cache_ttl = 1800  # 30 minutes

    _abstract = False

    # API endpoints
    OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    ACCUWEATHER_BASE_URL = "https://dataservice.accuweather.com"

    # Cache TTLs
    GEOCODE_CACHE_TTL = 86400 * 30  # 30 days
    WEATHER_CACHE_TTL = 1800  # 30 minutes
    LOCATION_KEY_CACHE_TTL = 86400 * 30  # 30 days for AccuWeather location keys

    # WMO Weather codes (Open-Meteo)
    WMO_CODES = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Light rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
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
                default="Smart Weather",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="accuweather_api_key",
                label="AccuWeather API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="ACCUWEATHER_API_KEY",
                help_text="Optional AccuWeather API key for enhanced data. Open-Meteo works without any key.",
            ),
            FieldDefinition(
                name="default_location",
                label="Default Location",
                field_type=FieldType.TEXT,
                required=False,
                default="",
                help_text="Default location if none specified (e.g., 'London')",
            ),
            FieldDefinition(
                name="units",
                label="Temperature Units",
                field_type=FieldType.SELECT,
                required=False,
                default="celsius",
                options=[
                    {"value": "celsius", "label": "Celsius (°C)"},
                    {"value": "fahrenheit", "label": "Fahrenheit (°F)"},
                ],
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="location",
                description="Location name (city, address, landmark)",
                param_type="string",
                required=True,
                examples=["London", "Paris, France", "Central Park, NYC", "Tokyo"],
            ),
            ParamDefinition(
                name="when",
                description="Time period for forecast",
                param_type="string",
                required=False,
                default="now",
                examples=[
                    "now",
                    "today",
                    "tomorrow",
                    "this weekend",
                    "next 3 days",
                    "next week",
                ],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-weather")
        self.accuweather_api_key = config.get("accuweather_api_key") or os.environ.get(
            "ACCUWEATHER_API_KEY", ""
        )
        self.default_location = config.get("default_location", "")
        self.units = config.get("units", "celsius")

        self._client = httpx.Client(timeout=15)

    def _get_cached(self, cache: dict, key: str, ttl: int) -> Optional[Any]:
        """Get from cache if still valid."""
        if key in cache:
            data, cached_at = cache[key]
            if time.time() - cached_at < ttl:
                return data
            del cache[key]
        return None

    def _set_cached(self, cache: dict, key: str, data: Any) -> None:
        """Store in cache."""
        cache[key] = (data, time.time())

    # ============================================
    # Geocoding (Open-Meteo - free)
    # ============================================

    def _geocode_location(self, location: str) -> Optional[dict]:
        """
        Geocode a location name to coordinates using Open-Meteo.

        Returns dict with lat, lon, display_name, country, timezone or None.
        Results cached for 30 days.
        """
        cache_key = f"geocode:{location.lower().strip()}"
        cached = self._get_cached(_geocode_cache, cache_key, self.GEOCODE_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._client.get(
                self.OPEN_METEO_GEOCODING_URL,
                params={"name": location, "count": 1, "language": "en"},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                result = data["results"][0]
                geocode_data = {
                    "lat": result["latitude"],
                    "lon": result["longitude"],
                    "display_name": result.get("name", location),
                    "country": result.get("country", ""),
                    "admin1": result.get("admin1", ""),
                    "timezone": result.get("timezone", "auto"),
                }
                self._set_cached(_geocode_cache, cache_key, geocode_data)
                logger.info(
                    f"Geocoded '{location}' -> {geocode_data['display_name']}, {geocode_data['country']}"
                )
                return geocode_data

        except Exception as e:
            logger.warning(f"Geocoding failed for '{location}': {e}")

        return None

    # ============================================
    # AccuWeather API methods
    # ============================================

    def _get_accuweather_location_key(self, location: str) -> Optional[str]:
        """Get AccuWeather location key for a city (cached 30 days)."""
        if not self.accuweather_api_key:
            return None

        cache_key = f"aw_loc:{location.lower().strip()}"
        cached = self._get_cached(
            _location_key_cache, cache_key, self.LOCATION_KEY_CACHE_TTL
        )
        if cached:
            return cached

        try:
            response = self._client.get(
                f"{self.ACCUWEATHER_BASE_URL}/locations/v1/cities/search",
                params={"apikey": self.accuweather_api_key, "q": location},
            )
            response.raise_for_status()
            locations = response.json()

            if locations:
                key = locations[0]["Key"]
                self._set_cached(_location_key_cache, cache_key, key)
                logger.info(f"AccuWeather location key for '{location}': {key}")
                return key

        except Exception as e:
            logger.warning(f"AccuWeather location lookup failed: {e}")

        return None

    def _get_accuweather_current(self, location_key: str) -> Optional[dict]:
        """Get current conditions from AccuWeather."""
        if not self.accuweather_api_key or not location_key:
            return None

        try:
            response = self._client.get(
                f"{self.ACCUWEATHER_BASE_URL}/currentconditions/v1/{location_key}",
                params={"apikey": self.accuweather_api_key, "details": "true"},
            )
            response.raise_for_status()
            conditions = response.json()

            if conditions:
                return conditions[0]

        except Exception as e:
            logger.warning(f"AccuWeather current conditions failed: {e}")

        return None

    def _get_accuweather_forecast(
        self, location_key: str, days: int = 5
    ) -> Optional[dict]:
        """
        Get daily forecast from AccuWeather.

        Free tier supports 1 and 5 day forecasts.
        Premium tiers support 10, 15, 25, 45 days.
        """
        if not self.accuweather_api_key or not location_key:
            return None

        # Map requested days to available endpoints
        if days <= 1:
            endpoint = "1day"
        elif days <= 5:
            endpoint = "5day"
        elif days <= 10:
            endpoint = "10day"
        else:
            endpoint = "15day"

        try:
            metric = self.units == "celsius"
            response = self._client.get(
                f"{self.ACCUWEATHER_BASE_URL}/forecasts/v1/daily/{endpoint}/{location_key}",
                params={
                    "apikey": self.accuweather_api_key,
                    "details": "true",
                    "metric": str(metric).lower(),
                },
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            # 401/403 might mean endpoint not available on free tier
            if e.response.status_code in [401, 403]:
                logger.debug(
                    f"AccuWeather {endpoint} forecast not available (may require premium)"
                )
                # Fall back to 5day if we tried a longer forecast
                if endpoint != "5day" and endpoint != "1day":
                    return self._get_accuweather_forecast(location_key, 5)
            else:
                logger.warning(f"AccuWeather forecast failed: {e}")
        except Exception as e:
            logger.warning(f"AccuWeather forecast failed: {e}")

        return None

    def _get_accuweather_hourly(
        self, location_key: str, hours: int = 12
    ) -> Optional[list]:
        """Get hourly forecast from AccuWeather."""
        if not self.accuweather_api_key or not location_key:
            return None

        # Map to available endpoints: 1hour, 12hour, 24hour, 72hour, 120hour
        if hours <= 1:
            endpoint = "1hour"
        elif hours <= 12:
            endpoint = "12hour"
        elif hours <= 24:
            endpoint = "24hour"
        else:
            endpoint = "72hour"

        try:
            metric = self.units == "celsius"
            response = self._client.get(
                f"{self.ACCUWEATHER_BASE_URL}/forecasts/v1/hourly/{endpoint}/{location_key}",
                params={
                    "apikey": self.accuweather_api_key,
                    "details": "true",
                    "metric": str(metric).lower(),
                },
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.debug(f"AccuWeather hourly forecast failed: {e}")

        return None

    # ============================================
    # Open-Meteo API methods (free, always available)
    # ============================================

    def _get_open_meteo_weather(
        self, lat: float, lon: float, days: int = 7, timezone: str = "auto"
    ) -> Optional[dict]:
        """Get weather from Open-Meteo (free, no API key)."""
        try:
            response = self._client.get(
                self.OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m,precipitation",
                    "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,sunrise,sunset,uv_index_max,wind_speed_10m_max",
                    "timezone": timezone,
                    "forecast_days": min(days, 16),  # Open-Meteo max is 16
                },
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Open-Meteo API error: {e}")
            return None

    # ============================================
    # Time period parsing
    # ============================================

    def _parse_forecast_period(self, when: str) -> tuple[int, list[str]]:
        """
        Parse natural language time period into forecast days and focus days.

        Returns (forecast_days_needed, list_of_focus_day_names)
        """
        when = when.lower().strip()
        now = datetime.now()
        day_names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        today_idx = now.weekday()

        def days_until(target_day: str) -> int:
            target_idx = day_names.index(target_day)
            diff = target_idx - today_idx
            if diff <= 0:
                diff += 7
            return diff

        if when in ["now", "current", "currently", ""]:
            return 1, ["today"]

        if when == "today":
            return 1, ["today"]

        if when == "tomorrow":
            return 2, ["tomorrow"]

        if when in ["this weekend", "weekend"]:
            days_to_sat = days_until("saturday")
            return days_to_sat + 2, ["saturday", "sunday"]

        if when == "next weekend":
            days_to_sat = days_until("saturday")
            if days_to_sat <= 2:
                days_to_sat += 7
            return days_to_sat + 2, ["saturday", "sunday"]

        if when in ["this week", "week"]:
            return 7, []

        if when == "next week":
            return 14, []

        # "next N days" pattern
        match = re.match(r"next\s+(\d+)\s*days?", when)
        if match:
            days = min(int(match.group(1)), 16)
            return days, []

        # Specific day name
        for day in day_names:
            if day in when:
                days_needed = days_until(day)
                return days_needed + 1, [day]

        # Default to 3-day forecast
        return 3, []

    # ============================================
    # Temperature formatting
    # ============================================

    def _format_temp(self, temp_c: float) -> str:
        """Format temperature in configured units."""
        if self.units == "fahrenheit":
            return f"{(temp_c * 9 / 5) + 32:.0f}°F"
        return f"{temp_c:.0f}°C"

    def _format_temp_range(self, low: float, high: float) -> str:
        """Format temperature range."""
        if self.units == "fahrenheit":
            low_f = (low * 9 / 5) + 32
            high_f = (high * 9 / 5) + 32
            return f"{low_f:.0f}°F to {high_f:.0f}°F"
        return f"{low:.0f}°C to {high:.0f}°C"

    def _get_wmo_description(self, code: int) -> str:
        """Convert WMO weather code to description."""
        return self.WMO_CODES.get(code, f"Unknown ({code})")

    # ============================================
    # Main fetch method
    # ============================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch weather data combining Open-Meteo and AccuWeather.

        Strategy:
        - Always use Open-Meteo for base forecast (free, reliable)
        - If AccuWeather configured, enhance with current conditions
        - AccuWeather provides more detailed "feels like", UV, air quality
        """
        location = params.get("location", "").strip()
        if not location:
            location = self.default_location
        if not location:
            return LiveDataResult(
                success=False,
                error="Location is required. Please specify a city or place name.",
            )

        when = params.get("when", "now").strip()
        forecast_days, focus_days = self._parse_forecast_period(when)

        # Geocode location (using Open-Meteo - free)
        geo = self._geocode_location(location)
        if not geo:
            return LiveDataResult(
                success=False,
                error=f"Could not find location: {location}. Try a city name like 'London' or 'New York'.",
            )

        # Check cache
        cache_key = f"weather:{geo['lat']:.2f},{geo['lon']:.2f}:{forecast_days}:{self.accuweather_api_key[:8] if self.accuweather_api_key else 'om'}"
        cached = self._get_cached(_weather_cache, cache_key, self.WEATHER_CACHE_TTL)
        if cached:
            formatted = self._format_weather_response(cached, geo, when, focus_days)
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=formatted,
                cache_ttl=self.WEATHER_CACHE_TTL,
            )

        # Fetch from Open-Meteo (always available)
        open_meteo_data = self._get_open_meteo_weather(
            geo["lat"], geo["lon"], forecast_days, geo.get("timezone", "auto")
        )

        if not open_meteo_data:
            return LiveDataResult(
                success=False,
                error="Failed to fetch weather data. Please try again.",
            )

        # Try to enhance with AccuWeather if configured
        accuweather_data = None
        if self.accuweather_api_key:
            aw_location_key = self._get_accuweather_location_key(location)
            if aw_location_key:
                accuweather_data = {
                    "current": self._get_accuweather_current(aw_location_key),
                    "forecast": self._get_accuweather_forecast(
                        aw_location_key, forecast_days
                    ),
                    "hourly": self._get_accuweather_hourly(aw_location_key, 12)
                    if when in ["now", "today", ""]
                    else None,
                }

        # Combine data
        combined_data = {
            "open_meteo": open_meteo_data,
            "accuweather": accuweather_data,
            "location": geo,
        }

        # Cache the combined result
        self._set_cached(_weather_cache, cache_key, combined_data)

        # Format response
        formatted = self._format_weather_response(combined_data, geo, when, focus_days)

        return LiveDataResult(
            success=True,
            data=combined_data,
            formatted=formatted,
            cache_ttl=self.WEATHER_CACHE_TTL,
        )

    def _format_weather_response(
        self,
        data: dict,
        geo: dict,
        when: str,
        focus_days: list[str],
    ) -> str:
        """Format combined weather data into a context-ready summary."""
        lines = []

        # Build location string
        location_str = geo["display_name"]
        if geo.get("admin1"):
            location_str += f", {geo['admin1']}"
        if geo.get("country"):
            location_str += f", {geo['country']}"

        om_data = data.get("open_meteo", {})
        aw_data = data.get("accuweather")

        # Current conditions
        if when.lower() in ["now", "current", "currently", "today", ""]:
            lines.append(f"**Current weather in {location_str}:**")

            # Prefer AccuWeather for current conditions if available
            if aw_data and aw_data.get("current"):
                aw_current = aw_data["current"]
                text = aw_current.get("WeatherText", "")
                temp = (
                    aw_current.get("Temperature", {}).get("Metric", {}).get("Value", 0)
                )
                feels = (
                    aw_current.get("RealFeelTemperature", {})
                    .get("Metric", {})
                    .get("Value", temp)
                )
                humidity = aw_current.get("RelativeHumidity", 0)
                wind = (
                    aw_current.get("Wind", {})
                    .get("Speed", {})
                    .get("Metric", {})
                    .get("Value", 0)
                )
                wind_dir = (
                    aw_current.get("Wind", {}).get("Direction", {}).get("English", "")
                )
                uv = aw_current.get("UVIndex", 0)
                precip = (
                    aw_current.get("PrecipitationSummary", {})
                    .get("Precipitation", {})
                    .get("Metric", {})
                    .get("Value", 0)
                )

                lines.append(
                    f"- {text}, {self._format_temp(temp)} (feels like {self._format_temp(feels)})"
                )
                lines.append(f"- Humidity: {humidity}%")
                lines.append(f"- Wind: {wind:.0f} km/h {wind_dir}")
                if uv > 0:
                    uv_level = (
                        "Low"
                        if uv <= 2
                        else "Moderate"
                        if uv <= 5
                        else "High"
                        if uv <= 7
                        else "Very High"
                        if uv <= 10
                        else "Extreme"
                    )
                    lines.append(f"- UV Index: {uv} ({uv_level})")
                if precip > 0:
                    lines.append(f"- Recent precipitation: {precip} mm")
                lines.append("_via AccuWeather_")

            elif om_data.get("current"):
                om_current = om_data["current"]
                temp = om_current.get("temperature_2m", 0)
                feels = om_current.get("apparent_temperature", temp)
                humidity = om_current.get("relative_humidity_2m", 0)
                wind = om_current.get("wind_speed_10m", 0)
                gusts = om_current.get("wind_gusts_10m", 0)
                code = om_current.get("weather_code", 0)
                condition = self._get_wmo_description(code)

                lines.append(
                    f"- {condition}, {self._format_temp(temp)} (feels like {self._format_temp(feels)})"
                )
                lines.append(f"- Humidity: {humidity}%")
                wind_str = f"- Wind: {wind:.0f} km/h"
                if gusts and gusts > wind + 10:
                    wind_str += f" (gusts {gusts:.0f} km/h)"
                lines.append(wind_str)

            lines.append("")

        # Daily forecast
        daily = om_data.get("daily", {})
        if daily and daily.get("time"):
            dates = daily["time"]

            if focus_days:
                lines.append(f"**Forecast for {when}:**")
            else:
                lines.append(f"**{len(dates)}-day forecast:**")

            # If AccuWeather forecast available, use it for richer data
            aw_forecast = (
                aw_data.get("forecast", {}).get("DailyForecasts", []) if aw_data else []
            )

            for i, date_str in enumerate(dates):
                date = datetime.strptime(date_str, "%Y-%m-%d")
                day_name = date.strftime("%A")

                # Skip non-focus days after first 2
                if focus_days and day_name.lower() not in focus_days and i > 1:
                    continue

                # Day label
                if i == 0:
                    day_label = "Today"
                elif i == 1:
                    day_label = "Tomorrow"
                else:
                    day_label = f"{day_name} ({date_str})"

                # Get Open-Meteo data
                high = (
                    daily.get("temperature_2m_max", [0])[i]
                    if daily.get("temperature_2m_max")
                    else 0
                )
                low = (
                    daily.get("temperature_2m_min", [0])[i]
                    if daily.get("temperature_2m_min")
                    else 0
                )
                code = (
                    daily.get("weather_code", [0])[i]
                    if daily.get("weather_code")
                    else 0
                )
                condition = self._get_wmo_description(code)
                precip_prob = (
                    daily.get("precipitation_probability_max", [0])[i]
                    if daily.get("precipitation_probability_max")
                    else 0
                )
                precip_sum = (
                    daily.get("precipitation_sum", [0])[i]
                    if daily.get("precipitation_sum")
                    else 0
                )
                uv = (
                    daily.get("uv_index_max", [0])[i]
                    if daily.get("uv_index_max")
                    else 0
                )
                wind_max = (
                    daily.get("wind_speed_10m_max", [0])[i]
                    if daily.get("wind_speed_10m_max")
                    else 0
                )

                # Try to enhance with AccuWeather data
                aw_day = aw_forecast[i] if i < len(aw_forecast) else None
                if aw_day:
                    # AccuWeather has better condition text
                    day_phrase = aw_day.get("Day", {}).get("IconPhrase", "")
                    night_phrase = aw_day.get("Night", {}).get("IconPhrase", "")
                    if day_phrase:
                        condition = f"{day_phrase}"
                        if night_phrase and night_phrase != day_phrase:
                            condition += f", {night_phrase.lower()} at night"

                # Build day summary
                day_line = f"- **{day_label}**: {condition}, {self._format_temp_range(low, high)}"

                # Add rain probability if significant
                if precip_prob and precip_prob > 20:
                    day_line += f", {precip_prob}% rain"
                    if precip_sum > 1:
                        day_line += f" ({precip_sum:.0f}mm)"

                # Add wind if strong
                if wind_max > 30:
                    day_line += f", windy ({wind_max:.0f} km/h)"

                # Add UV warning if high
                if uv >= 8:
                    day_line += f" ⚠️ Very high UV ({uv:.0f})"
                elif uv >= 6:
                    day_line += f", high UV ({uv:.0f})"

                lines.append(day_line)

        # Data source attribution
        lines.append("")
        if aw_data and aw_data.get("current"):
            lines.append("_Data: AccuWeather + Open-Meteo_")
        else:
            lines.append("_Data: Open-Meteo_")

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Always available - Open-Meteo requires no API key."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to weather APIs."""
        results = []

        # Test Open-Meteo (always available)
        try:
            result = self._get_open_meteo_weather(51.5, -0.1, 1)  # London
            if result:
                results.append("✓ Open-Meteo: OK")
            else:
                results.append("✗ Open-Meteo: Failed")
        except Exception as e:
            results.append(f"✗ Open-Meteo: {e}")

        # Test AccuWeather if configured
        if self.accuweather_api_key:
            try:
                loc_key = self._get_accuweather_location_key("London")
                if loc_key:
                    current = self._get_accuweather_current(loc_key)
                    if current:
                        results.append("✓ AccuWeather: OK")
                    else:
                        results.append("✗ AccuWeather: No data")
                else:
                    results.append("✗ AccuWeather: Location lookup failed")
            except Exception as e:
                results.append(f"✗ AccuWeather: {e}")
        else:
            results.append("- AccuWeather: Not configured (optional)")

        success = "✓ Open-Meteo" in results[0]
        return success, "\n".join(results)
