"""
Smart Routes live source plugin.

High-level route/directions interface with natural language support.
Unlike GoogleMapsProvider which requires structured parameters,
this provider accepts natural language inputs for locations and times.

Migrated from live/sources.py RoutesSmartProvider.

Requires GOOGLE_MAPS_API_KEY environment variable.
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)

logger = logging.getLogger(__name__)

# Simple in-memory cache (matches original provider behavior)
_cache: dict[str, tuple[Any, float]] = {}


class RoutesLiveSource(PluginLiveSource):
    """
    Smart Routes Provider - high-level route/directions interface.

    Unlike GoogleMapsLiveSource which requires structured parameters,
    this provider accepts natural language inputs and handles:
    - Geocoding location names to coordinates (with caching)
    - Parsing natural time expressions ("tomorrow 9am", "Monday 3pm")
    - Computing routes with traffic-aware timing

    The designator can simply provide:
    - origin: "London" or "10 Downing Street"
    - destination: "Manchester Airport"
    - arrival_time: "tomorrow 9am" or "2024-01-20T09:00:00" (optional)
    - departure_time: "in 2 hours" (optional)
    - mode: "drive", "walk", "bicycle", "transit" (optional, default: drive)

    Migrated from live/sources.py RoutesSmartProvider with full functionality preserved.
    """

    source_type = "google_routes_enhanced"
    display_name = "Google Routes (Enhanced)"
    description = (
        "Route planning via Google Routes API with natural language locations and times"
    )
    category = "transport"
    data_type = "routes"
    best_for = "Driving directions, travel time with traffic, route planning with arrival/departure times, alternative routes comparison"
    icon = "🛣️"
    default_cache_ttl = 300  # 5 minutes for routes (traffic changes)

    _abstract = False  # Allow registration

    ROUTES_BASE_URL = "https://routes.googleapis.com/directions/v2"
    GEOCODING_BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    # Cache TTLs
    GEOCODE_CACHE_TTL = 86400 * 30  # 30 days for geocoding (addresses rarely change)
    ROUTE_CACHE_TTL = 300  # 5 minutes for routes (traffic changes)

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields for admin UI."""
        return [
            FieldDefinition(
                name="name",
                label="Source Name",
                field_type=FieldType.TEXT,
                required=True,
                default="Smart Routes",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="Google Maps API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="GOOGLE_MAPS_API_KEY",
                help_text="Your Google Maps API key (leave empty to use GOOGLE_MAPS_API_KEY env var)",
            ),
            FieldDefinition(
                name="cache_ttl_seconds",
                label="Route Cache TTL (seconds)",
                field_type=FieldType.INTEGER,
                required=False,
                default=300,
                min_value=0,
                max_value=3600,
                help_text="How long to cache route results (0 to disable)",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="origin",
                description="Starting location (city, address, or place name)",
                param_type="string",
                required=True,
                examples=["London", "10 Downing Street", "Heathrow Airport"],
            ),
            ParamDefinition(
                name="destination",
                description="End location (city, address, or place name)",
                param_type="string",
                required=True,
                examples=["Manchester", "Birmingham Airport", "Edinburgh"],
            ),
            ParamDefinition(
                name="arrival_time",
                description="Time to arrive BY (for transit/events) - supports natural language",
                param_type="string",
                required=False,
                examples=["tomorrow 9am", "Monday 3pm", "2024-01-20T09:00:00"],
            ),
            ParamDefinition(
                name="departure_time",
                description="Time to leave AT (for driving) - supports natural language",
                param_type="string",
                required=False,
                examples=["in 2 hours", "today 5pm", "tomorrow morning"],
            ),
            ParamDefinition(
                name="mode",
                description="Travel mode",
                param_type="string",
                required=False,
                default="drive",
                examples=["drive", "walk", "bicycle", "transit"],
            ),
        ]

    def __init__(self, config: dict):
        """
        Initialize with configuration.

        Args:
            config: Dict with 'name', 'api_key', and optional 'cache_ttl_seconds'
        """
        self.name = config.get("name", "routes")
        self.api_key = config.get("api_key") or os.environ.get(
            "GOOGLE_MAPS_API_KEY", ""
        )
        self.cache_ttl = config.get("cache_ttl_seconds", self.ROUTE_CACHE_TTL)

    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for the query."""
        return f"RoutesLiveSource:{query}"

    def _get_cached(self, query: str, ttl_seconds: int) -> Optional[Any]:
        """Get cached result if still valid."""
        key = self._get_cache_key(query)
        if key in _cache:
            data, cached_at = _cache[key]
            if time.time() - cached_at < ttl_seconds:
                return data
            del _cache[key]
        return None

    def _set_cached(self, query: str, data: Any) -> None:
        """Cache a result."""
        key = self._get_cache_key(query)
        _cache[key] = (data, time.time())

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch route information between two locations.

        Args:
            params: Dict with origin, destination, and optional time/mode params

        Returns:
            LiveDataResult with route data
        """
        if not self.api_key:
            return LiveDataResult(
                success=False,
                error="GOOGLE_MAPS_API_KEY not configured",
            )

        # Extract parameters
        origin = params.get("origin", "")
        destination = params.get("destination", "")
        arrival_time_str = params.get("arrival_time", "")
        departure_time_str = params.get("departure_time", "")
        mode = params.get("mode", "drive").lower()

        if not origin or not destination:
            return LiveDataResult(
                success=False,
                error="Both origin and destination are required",
            )

        # Normalize travel mode
        mode_map = {
            "drive": "DRIVE",
            "driving": "DRIVE",
            "car": "DRIVE",
            "walk": "WALK",
            "walking": "WALK",
            "bicycle": "BICYCLE",
            "bike": "BICYCLE",
            "cycling": "BICYCLE",
            "transit": "TRANSIT",
            "public": "TRANSIT",
            "bus": "TRANSIT",
            "train": "TRANSIT",
        }
        travel_mode = mode_map.get(mode.lower(), "DRIVE")

        # Parse arrival and departure times
        arrival_time = None
        departure_time = None
        if arrival_time_str:
            arrival_time = self._parse_time(arrival_time_str)
        if departure_time_str:
            departure_time = self._parse_time(departure_time_str)

        try:
            # Geocode origin and destination (with caching)
            origin_coords = self._geocode_location(origin)
            if not origin_coords:
                return LiveDataResult(
                    success=False,
                    error=f"Could not geocode origin: {origin}",
                )

            dest_coords = self._geocode_location(destination)
            if not dest_coords:
                return LiveDataResult(
                    success=False,
                    error=f"Could not geocode destination: {destination}",
                )

            # Build cache key - include both arrival and departure time
            cache_key = f"smart_route:{origin_coords}:{dest_coords}:{travel_mode}"
            if arrival_time:
                rounded_time = (arrival_time // 900) * 900
                cache_key += f":arr{rounded_time}"
            if departure_time:
                rounded_time = (departure_time // 900) * 900
                cache_key += f":dep{rounded_time}"

            # Check cache
            if self.cache_ttl > 0:
                cached = self._get_cached(cache_key, self.cache_ttl)
                if cached is not None:
                    return LiveDataResult(
                        success=True,
                        data=cached,
                        formatted=self._format_route(
                            cached,
                            origin,
                            destination,
                            travel_mode,
                            arrival_time,
                            departure_time,
                        ),
                        cache_ttl=self.cache_ttl,
                    )

            # Call Routes API
            route_data = self._compute_route(
                origin_coords, dest_coords, travel_mode, arrival_time, departure_time
            )

            # Cache the result
            if self.cache_ttl > 0:
                self._set_cached(cache_key, route_data)

            return LiveDataResult(
                success=True,
                data=route_data,
                formatted=self._format_route(
                    route_data,
                    origin,
                    destination,
                    travel_mode,
                    arrival_time,
                    departure_time,
                ),
                cache_ttl=self.cache_ttl,
            )

        except Exception as e:
            logger.error(f"Routes API error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _geocode_location(self, location: str) -> str | None:
        """
        Geocode a location name to "lat,lng" string.

        Uses caching to avoid repeated geocoding of the same location.
        """
        # Check if already coordinates
        if self._is_coordinates(location):
            return location

        # Check cache first
        cache_key = f"geocode:{location.lower().strip()}"
        cached = self._get_cached(cache_key, self.GEOCODE_CACHE_TTL)
        if cached is not None:
            return cached

        # Call Geocoding API
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    self.GEOCODING_BASE_URL,
                    params={
                        "address": location,
                        "key": self.api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            if not results:
                logger.warning(f"No geocoding results for: {location}")
                return None

            # Get first result's coordinates
            loc = results[0]["geometry"]["location"]
            coords = f"{loc['lat']},{loc['lng']}"

            # Cache the result (30 days)
            self._set_cached(cache_key, coords)

            logger.info(f"Geocoded '{location}' -> {coords}")
            return coords

        except Exception as e:
            logger.error(f"Geocoding error for '{location}': {e}")
            return None

    def _is_coordinates(self, location: str) -> bool:
        """Check if location string is already in lat,lng format."""
        parts = location.replace(" ", "").split(",")
        if len(parts) != 2:
            return False
        try:
            lat = float(parts[0])
            lng = float(parts[1])
            return -90 <= lat <= 90 and -180 <= lng <= 180
        except ValueError:
            return False

    def _parse_time(self, time_str: str) -> int | None:
        """
        Parse natural language time to Unix timestamp.

        Supports:
        - ISO format: "2024-01-20T09:00:00"
        - Natural: "tomorrow 9am", "Monday 3pm", "in 2 hours"
        - Time only: "9:30", "14:00" (assumes today)
        """
        time_str = time_str.strip().lower()

        # Try ISO format first
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except ValueError:
            pass

        now = datetime.now()

        # Parse "in X hours/minutes"
        in_match = re.match(r"in\s+(\d+)\s*(hour|hr|minute|min)s?", time_str)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)
            if unit.startswith("hour") or unit.startswith("hr"):
                return int((now + timedelta(hours=amount)).timestamp())
            else:
                return int((now + timedelta(minutes=amount)).timestamp())

        # Parse day references
        target_date = now
        if "tomorrow" in time_str:
            target_date = now + timedelta(days=1)
            time_str = time_str.replace("tomorrow", "").strip()
        elif "today" in time_str:
            time_str = time_str.replace("today", "").strip()
        else:
            # Check for day names
            days = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            for i, day in enumerate(days):
                if day in time_str:
                    days_ahead = i - now.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = now + timedelta(days=days_ahead)
                    time_str = time_str.replace(day, "").strip()
                    break

        # Parse time component
        time_str = time_str.strip()
        if time_str:
            # Handle "9am", "3pm", "14:00", "9:30am"
            time_match = re.match(
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str, re.IGNORECASE
            )
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                period = time_match.group(3)

                if period:
                    period = period.lower()
                    if period == "pm" and hour != 12:
                        hour += 12
                    elif period == "am" and hour == 12:
                        hour = 0

                target_date = target_date.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
            else:
                # Default to 9am if no time specified
                target_date = target_date.replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
        else:
            # No time specified, use 9am
            target_date = target_date.replace(hour=9, minute=0, second=0, microsecond=0)

        return int(target_date.timestamp())

    def _compute_route(
        self,
        origin_coords: str,
        dest_coords: str,
        travel_mode: str,
        arrival_time: int | None,
        departure_time: int | None,
    ) -> dict:
        """Call Google Routes API to compute the route."""
        # Parse coordinates
        origin_lat, origin_lng = map(float, origin_coords.split(","))
        dest_lat, dest_lng = map(float, dest_coords.split(","))

        request_body = {
            "origin": {
                "location": {
                    "latLng": {"latitude": origin_lat, "longitude": origin_lng}
                }
            },
            "destination": {
                "location": {"latLng": {"latitude": dest_lat, "longitude": dest_lng}}
            },
            "travelMode": travel_mode,
            "computeAlternativeRoutes": True,
            "languageCode": "en",
            "units": "METRIC",
        }

        # routingPreference is only valid for DRIVE and TWO_WHEELER modes
        if travel_mode in ["DRIVE", "TWO_WHEELER"]:
            request_body["routingPreference"] = "TRAFFIC_AWARE"

        # Add arrival or departure time (arrival takes precedence for transit)
        # Note: Google Routes API only allows one of arrivalTime or departureTime
        if arrival_time:
            time_str = datetime.utcfromtimestamp(arrival_time).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            request_body["arrivalTime"] = time_str
        elif departure_time:
            time_str = datetime.utcfromtimestamp(departure_time).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            request_body["departureTime"] = time_str

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.staticDuration,routes.distanceMeters,routes.polyline,routes.legs,routes.travelAdvisory,routes.description,routes.routeLabels",
        }

        with httpx.Client(timeout=15) as client:
            response = client.post(
                f"{self.ROUTES_BASE_URL}:computeRoutes",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            return response.json()

    def _format_route(
        self,
        data: dict,
        origin: str,
        destination: str,
        travel_mode: str,
        arrival_time: int | None,
        departure_time: int | None,
    ) -> str:
        """Format route data for context injection."""
        routes = data.get("routes", [])
        if not routes:
            return f"### Route: {origin} → {destination}\nNo route found."

        mode_names = {
            "DRIVE": "Driving",
            "WALK": "Walking",
            "BICYCLE": "Cycling",
            "TRANSIT": "Transit",
        }
        mode_name = mode_names.get(travel_mode, travel_mode.title())

        lines = [f"### {mode_name} Route: {origin} → {destination}"]

        # Add arrival or departure time if specified
        if arrival_time:
            dt = datetime.fromtimestamp(arrival_time)
            lines.append(f"**Arrive by:** {dt.strftime('%A, %B %d at %I:%M %p')}")
            lines.append("")
        elif departure_time:
            dt = datetime.fromtimestamp(departure_time)
            lines.append(f"**Depart at:** {dt.strftime('%A, %B %d at %I:%M %p')}")
            lines.append("")

        # Show primary route
        route = routes[0]
        lines.extend(self._format_single_route(route, is_primary=True))

        # Show alternatives if available
        if len(routes) > 1:
            lines.append("\n---")
            lines.append(f"**Alternative Routes ({len(routes) - 1} available):**")
            for i, alt_route in enumerate(routes[1:3], 1):  # Show up to 2 alternatives
                lines.append(f"\n**Option {i + 1}:**")
                lines.extend(self._format_single_route(alt_route, is_primary=False))

        return "\n".join(lines)

    def _format_single_route(self, route: dict, is_primary: bool) -> list[str]:
        """Format a single route."""
        lines = []

        # Duration (with traffic)
        duration = route.get("duration", "")
        static_duration = route.get("staticDuration", "")

        if duration:
            duration_str = self._format_duration(duration)
            if static_duration and static_duration != duration:
                static_str = self._format_duration(static_duration)
                # Calculate delay due to traffic
                traffic_delay = self._get_seconds(duration) - self._get_seconds(
                    static_duration
                )
                if traffic_delay > 60:  # Only show if more than 1 minute delay
                    delay_str = self._format_duration(f"{traffic_delay}s")
                    lines.append(
                        f"**Duration:** {duration_str} (normally {static_str}, +{delay_str} due to traffic)"
                    )
                else:
                    lines.append(f"**Duration:** {duration_str}")
            else:
                lines.append(f"**Duration:** {duration_str}")

        # Distance
        distance = route.get("distanceMeters", 0)
        if distance:
            if distance >= 1000:
                lines.append(
                    f"**Distance:** {distance / 1000:.1f} km ({distance / 1609.34:.1f} miles)"
                )
            else:
                lines.append(f"**Distance:** {distance} m")

        # Route description (via road)
        description = route.get("description")
        if description:
            lines.append(f"**Via:** {description}")

        # Route labels (e.g., "FUEL_EFFICIENT")
        labels = route.get("routeLabels", [])
        if labels:
            label_map = {
                "FUEL_EFFICIENT": "🌿 Most fuel efficient",
                "DEFAULT_ROUTE": "⭐ Recommended",
                "DEFAULT_ROUTE_ALTERNATE": "Alternative",
            }
            label_strs = [label_map.get(l, l) for l in labels]
            lines.append(f"**Type:** {', '.join(label_strs)}")

        # Traffic/travel advisory
        advisory = route.get("travelAdvisory", {})
        if advisory.get("tollInfo"):
            lines.append("⚠️ **Note:** Route includes tolls")

        # Step-by-step directions for primary route
        if is_primary:
            legs = route.get("legs", [])
            if legs:
                lines.append("\n**Turn-by-turn directions:**")
                step_num = 1
                for leg in legs:
                    steps = leg.get("steps", [])
                    for step in steps[:15]:  # Limit to 15 steps
                        instruction = step.get("navigationInstruction", {})
                        instructions_text = instruction.get("instructions", "")

                        step_distance = step.get("distanceMeters", 0)
                        if step_distance >= 1000:
                            dist_str = f"{step_distance / 1000:.1f} km"
                        elif step_distance > 0:
                            dist_str = f"{step_distance} m"
                        else:
                            dist_str = ""

                        if instructions_text:
                            if dist_str:
                                lines.append(
                                    f"  {step_num}. {instructions_text} ({dist_str})"
                                )
                            else:
                                lines.append(f"  {step_num}. {instructions_text}")
                            step_num += 1

                    total_steps = sum(len(leg.get("steps", [])) for leg in legs)
                    if total_steps > 15:
                        lines.append(f"  ... and {total_steps - 15} more steps")

        return lines

    def _format_duration(self, duration_str: str) -> str:
        """Format duration string (e.g., '3600s') to human readable."""
        seconds = self._get_seconds(duration_str)
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60

        if hours > 0:
            if minutes > 0:
                return f"{hours}h {minutes}min"
            return f"{hours}h"
        return f"{minutes} min"

    def _get_seconds(self, duration_str: str) -> int:
        """Extract seconds from duration string like '3600s'."""
        if isinstance(duration_str, str):
            return int(duration_str.rstrip("s"))
        return int(duration_str)

    def is_available(self) -> bool:
        """Check if Google Maps API key is configured."""
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Google Routes API."""
        if not self.api_key:
            return False, "GOOGLE_MAPS_API_KEY not set"

        # Test geocoding
        coords = self._geocode_location("London, UK")
        if coords:
            return True, f"Connected - Geocoded 'London, UK' -> {coords}"
        return False, "Failed to geocode test location"
