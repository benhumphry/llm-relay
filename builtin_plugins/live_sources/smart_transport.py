"""
Smart UK Transport live source plugin.

High-level UK public transport interface with natural language support.
Handles trains, buses, and multi-modal journey planning.

Unlike the basic transport plugin which only provides train departures,
this provider accepts natural language inputs and handles:
- Journey planning between any two UK locations
- Train departures with destination filtering
- Service status queries ("Is my train running on time?")
- Bus departures from nearby stops
- Station/stop name resolution with caching

Uses TransportAPI (https://transportapi.com)

Requires TRANSPORTAPI_APP_KEY and TRANSPORTAPI_APP_ID environment variables.
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
_station_cache: dict[str, tuple[Any, float]] = {}  # 30-day TTL for station lookups
_stop_cache: dict[str, tuple[Any, float]] = {}  # 30-day TTL for bus stop lookups
_timetable_cache: dict[str, tuple[Any, float]] = {}  # 1-min TTL for live data


class SmartTransportLiveSource(PluginLiveSource):
    """
    Smart UK Transport Provider - natural language public transport queries.

    Supports multiple query types:
    - journey: Plan a public transport journey between two locations
    - departures: Train departures from a station (optionally filtered by destination)
    - service: Check if a specific train service is running/delayed
    - bus: Bus departures from a stop or nearby location

    The designator provides:
    - query_type: "journey", "departures", "service", or "bus"
    - origin: Starting station/stop/location (for journey/departures)
    - destination: End station/location (for journey) or filter (for departures - 'trains TO X')
    - time: Natural language time ("tomorrow 9am", "in 30 minutes")
    - service_id: Specific train service to check (for service queries)

    Examples:
    - "How do I get from Harpenden to Gatwick Airport?"
    - "What time are trains to Luton from St Albans?"
    - "Is the 10:30 to Brighton running on time?"
    - "Next bus from Oxford Circus"
    """

    source_type = "transportapi_enhanced"
    display_name = "TransportAPI (Enhanced)"
    description = (
        "UK trains, buses, and journey planning via TransportAPI with natural language"
    )
    category = "transport"
    data_type = "transport"
    best_for = "UK public transport: journey planning, train departures, service delays, bus times. Use for 'how to get to X', 'trains to London', 'is my train delayed', 'next bus'"
    icon = "🚆"
    default_cache_ttl = 60  # 1 minute for live data

    _abstract = False

    BASE_URL = "https://transportapi.com/v3/uk"

    # Cache TTLs
    STATION_CACHE_TTL = 86400 * 30  # 30 days for station name resolution
    LIVE_CACHE_TTL = 60  # 1 minute for live departure data
    JOURNEY_CACHE_TTL = 300  # 5 minutes for journey plans

    # Common UK station codes for quick lookup (avoids API call)
    KNOWN_STATIONS = {
        # London terminals
        "kings cross": "KGX",
        "king's cross": "KGX",
        "st pancras": "STP",
        "saint pancras": "STP",
        "euston": "EUS",
        "paddington": "PAD",
        "victoria": "VIC",
        "waterloo": "WAT",
        "liverpool street": "LST",
        "london bridge": "LBG",
        "charing cross": "CHX",
        "marylebone": "MYB",
        "blackfriars": "BFR",
        "cannon street": "CST",
        "fenchurch street": "FST",
        "moorgate": "MOG",
        "city thameslink": "CTK",
        # Major airports
        "gatwick": "GTW",
        "gatwick airport": "GTW",
        "heathrow": "HXX",
        "heathrow terminal 5": "HWV",
        "heathrow terminals 2 & 3": "HXX",
        "luton airport": "LTN",
        "luton airport parkway": "LTN",
        "stansted": "SSD",
        "stansted airport": "SSD",
        "birmingham airport": "BHI",
        "manchester airport": "MIA",
        # Home counties
        "harpenden": "HPD",
        "st albans": "SAC",
        "st albans city": "SAC",
        "luton": "LUT",
        "bedford": "BDM",
        "stevenage": "SVG",
        "welwyn garden city": "WGC",
        "hatfield": "HAT",
        "potters bar": "PBR",
        "radlett": "RDT",
        "elstree": "ELS",
        "mill hill broadway": "MIL",
        "hendon": "HEN",
        "cricklewood": "CRI",
        "west hampstead thameslink": "WHP",
        "kentish town": "KTW",
        # Major cities
        "birmingham": "BHM",
        "birmingham new street": "BHM",
        "manchester": "MAN",
        "manchester piccadilly": "MAN",
        "manchester victoria": "MCV",
        "leeds": "LDS",
        "sheffield": "SHF",
        "bristol": "BRI",
        "bristol temple meads": "BRI",
        "edinburgh": "EDB",
        "edinburgh waverley": "EDB",
        "glasgow": "GLC",
        "glasgow central": "GLC",
        "glasgow queen street": "GLQ",
        "liverpool": "LIV",
        "liverpool lime street": "LIV",
        "newcastle": "NCL",
        "york": "YRK",
        "cambridge": "CBG",
        "oxford": "OXF",
        "reading": "RDG",
        "brighton": "BTN",
        "southampton": "SOU",
        "portsmouth": "PMS",
        "cardiff": "CDF",
        "cardiff central": "CDF",
        "nottingham": "NOT",
        "leicester": "LEI",
        "coventry": "COV",
        "peterborough": "PBO",
        "milton keynes": "MKC",
        "milton keynes central": "MKC",
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
                default="Smart UK Transport",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="app_id",
                label="TransportAPI App ID",
                field_type=FieldType.TEXT,
                required=False,
                env_var="TRANSPORTAPI_APP_ID",
                help_text="Leave empty to use TRANSPORTAPI_APP_ID env var",
            ),
            FieldDefinition(
                name="app_key",
                label="TransportAPI App Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="TRANSPORTAPI_APP_KEY",
                help_text="Leave empty to use TRANSPORTAPI_APP_KEY env var",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query_type",
                description="Type of query: 'journey' (A to B planning), 'departures' (trains from station), 'service' (specific train status), 'bus' (bus departures)",
                param_type="string",
                required=True,
                examples=["journey", "departures", "service", "bus"],
            ),
            ParamDefinition(
                name="origin",
                description="Starting station, stop, or location name",
                param_type="string",
                required=False,
                examples=["Harpenden", "Kings Cross", "St Albans", "London"],
            ),
            ParamDefinition(
                name="destination",
                description="End station/location (for journey) or destination filter (for departures - 'trains TO X')",
                param_type="string",
                required=False,
                examples=["Gatwick Airport", "London", "Brighton", "Luton"],
            ),
            ParamDefinition(
                name="time",
                description="Departure/arrival time - supports natural language",
                param_type="string",
                required=False,
                examples=["now", "tomorrow 9am", "Monday 8:30", "in 30 minutes"],
            ),
            ParamDefinition(
                name="arrive_by",
                description="Set to 'true' if time is arrival time (default: departure time)",
                param_type="string",
                required=False,
                default="false",
                examples=["true", "false"],
            ),
            ParamDefinition(
                name="service_id",
                description="Specific train service ID to check status (for service queries)",
                param_type="string",
                required=False,
                examples=["G12345", "service running at 10:30"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-transport")
        self.app_id = config.get("app_id") or os.environ.get("TRANSPORTAPI_APP_ID", "")
        self.app_key = config.get("app_key") or os.environ.get(
            "TRANSPORTAPI_APP_KEY", ""
        )

        self._client = httpx.Client(
            timeout=15,
            follow_redirects=True,
        )

    def _get_auth_params(self) -> dict:
        """Get authentication parameters for API calls."""
        return {
            "app_id": self.app_id,
            "app_key": self.app_key,
        }

    def _get_cached(self, cache: dict, key: str, ttl: int) -> Optional[Any]:
        """Get from cache if still valid."""
        if key in cache:
            data, cached_at = cache[key]
            if time.time() - cached_at < ttl:
                logger.debug(f"Cache hit: {key}")
                return data
            del cache[key]
        return None

    def _set_cached(self, cache: dict, key: str, data: Any) -> None:
        """Cache a result."""
        cache[key] = (data, time.time())

    def _resolve_station(self, station_name: str) -> Optional[dict]:
        """
        Resolve a station name to station code and details.

        Returns dict with 'code', 'name', 'latitude', 'longitude' or None.
        Results are cached for 30 days.
        """
        if not station_name:
            return None

        name_lower = station_name.lower().strip()

        # Check if it's already a 3-letter code
        if len(station_name) == 3 and station_name.isupper():
            return {"code": station_name, "name": station_name}

        # Check hardcoded lookup
        if name_lower in self.KNOWN_STATIONS:
            code = self.KNOWN_STATIONS[name_lower]
            return {"code": code, "name": station_name.title()}

        # Check cache
        cache_key = f"station:{name_lower}"
        cached = self._get_cached(_station_cache, cache_key, self.STATION_CACHE_TTL)
        if cached:
            return cached

        # Query TransportAPI places endpoint
        try:
            response = self._client.get(
                f"{self.BASE_URL}/places.json",
                params={
                    **self._get_auth_params(),
                    "query": station_name,
                    "type": "train_station",
                },
            )
            response.raise_for_status()
            data = response.json()

            members = data.get("member", [])
            for member in members:
                if member.get("type") == "train_station":
                    result = {
                        "code": member.get("station_code"),
                        "name": member.get("name", station_name),
                        "latitude": member.get("latitude"),
                        "longitude": member.get("longitude"),
                    }
                    if result["code"]:
                        self._set_cached(_station_cache, cache_key, result)
                        logger.info(
                            f"Resolved station '{station_name}' -> {result['code']} ({result['name']})"
                        )
                        return result

        except Exception as e:
            logger.warning(f"Station lookup failed for '{station_name}': {e}")

        return None

    def _resolve_place(self, place_name: str) -> Optional[dict]:
        """
        Resolve any place name to coordinates for journey planning.

        Returns dict with 'name', 'latitude', 'longitude', and optionally 'code' for stations.
        """
        # First try as a train station
        station = self._resolve_station(place_name)
        if station and station.get("latitude"):
            return station

        # Try general place search
        cache_key = f"place:{place_name.lower().strip()}"
        cached = self._get_cached(_station_cache, cache_key, self.STATION_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._client.get(
                f"{self.BASE_URL}/places.json",
                params={
                    **self._get_auth_params(),
                    "query": place_name,
                },
            )
            response.raise_for_status()
            data = response.json()

            members = data.get("member", [])
            if members:
                member = members[0]
                result = {
                    "name": member.get("name", place_name),
                    "latitude": member.get("latitude"),
                    "longitude": member.get("longitude"),
                    "type": member.get("type"),
                }
                if member.get("station_code"):
                    result["code"] = member["station_code"]

                if result.get("latitude"):
                    self._set_cached(_station_cache, cache_key, result)
                    logger.info(f"Resolved place '{place_name}' -> {result}")
                    return result

        except Exception as e:
            logger.warning(f"Place lookup failed for '{place_name}': {e}")

        return None

    def _parse_time(self, time_str: str) -> tuple[str, str]:
        """
        Parse natural language time to date and time strings.

        Returns (date_str "YYYY-MM-DD", time_str "HH:MM").
        """
        if not time_str or time_str.lower() == "now":
            now = datetime.now()
            return now.strftime("%Y-%m-%d"), now.strftime("%H:%M")

        time_str = time_str.strip().lower()
        now = datetime.now()
        target_date = now

        # Parse "in X minutes/hours"
        in_match = re.match(r"in\s+(\d+)\s*(hour|hr|minute|min)s?", time_str)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)
            if unit.startswith("hour") or unit.startswith("hr"):
                target_date = now + timedelta(hours=amount)
            else:
                target_date = now + timedelta(minutes=amount)
            return target_date.strftime("%Y-%m-%d"), target_date.strftime("%H:%M")

        # Parse day references
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

        return target_date.strftime("%Y-%m-%d"), target_date.strftime("%H:%M")

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch UK transport data based on query type.

        Args:
            params: Dict with query_type and relevant parameters

        Returns:
            LiveDataResult with transport data
        """
        if not self.app_id or not self.app_key:
            return LiveDataResult(
                success=False,
                error="TransportAPI credentials not configured. Set TRANSPORTAPI_APP_ID and TRANSPORTAPI_APP_KEY.",
            )

        query_type = params.get("query_type", "departures").lower()

        if query_type == "journey":
            return self._fetch_journey(params)
        elif query_type == "departures":
            return self._fetch_departures(params)
        elif query_type == "service":
            return self._fetch_service(params)
        elif query_type == "bus":
            return self._fetch_bus(params)
        else:
            return LiveDataResult(
                success=False,
                error=f"Unknown query_type: {query_type}. Use 'journey', 'departures', 'service', or 'bus'.",
            )

    def _fetch_journey(self, params: dict) -> LiveDataResult:
        """Fetch journey plan between two locations."""
        origin_str = params.get("origin", "")
        dest_str = params.get("destination", "")
        time_str = params.get("time", "now")
        arrive_by = params.get("arrive_by", "false").lower() == "true"

        if not origin_str or not dest_str:
            return LiveDataResult(
                success=False,
                error="Both origin and destination are required for journey planning.",
            )

        # Resolve locations
        origin = self._resolve_place(origin_str)
        if not origin or not origin.get("latitude"):
            return LiveDataResult(
                success=False,
                error=f"Could not find location: {origin_str}",
            )

        destination = self._resolve_place(dest_str)
        if not destination or not destination.get("latitude"):
            return LiveDataResult(
                success=False,
                error=f"Could not find location: {dest_str}",
            )

        # Parse time
        date_str, time_val = self._parse_time(time_str)

        # Build cache key
        cache_key = f"journey:{origin['latitude']},{origin['longitude']}:{destination['latitude']},{destination['longitude']}:{date_str}:{time_val}:{arrive_by}"

        # Check cache
        cached = self._get_cached(_timetable_cache, cache_key, self.JOURNEY_CACHE_TTL)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=self._format_journey(cached, origin, destination, arrive_by),
                cache_ttl=self.JOURNEY_CACHE_TTL,
            )

        # Call journey planner API
        try:
            from_loc = f"lonlat:{origin['longitude']},{origin['latitude']}"
            to_loc = f"lonlat:{destination['longitude']},{destination['latitude']}"

            api_params = {
                **self._get_auth_params(),
                "from": from_loc,
                "to": to_loc,
                "date": date_str,
                "time": time_val,
            }

            if arrive_by:
                api_params["time_is"] = "arriving"
            else:
                api_params["time_is"] = "departing"

            response = self._client.get(
                f"{self.BASE_URL}/public_journey.json",
                params=api_params,
            )
            response.raise_for_status()
            data = response.json()

            self._set_cached(_timetable_cache, cache_key, data)

            return LiveDataResult(
                success=True,
                data=data,
                formatted=self._format_journey(data, origin, destination, arrive_by),
                cache_ttl=self.JOURNEY_CACHE_TTL,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Journey API error: {e.response.status_code} - {e.response.text}"
            )
            if e.response.status_code == 403:
                return LiveDataResult(
                    success=False,
                    error="TransportAPI usage limits exceeded or invalid credentials",
                )
            return LiveDataResult(
                success=False,
                error=f"Journey planning failed: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Journey planning error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_departures(self, params: dict) -> LiveDataResult:
        """Fetch train departures from a station, optionally filtered by destination."""
        origin_str = params.get("origin", "")
        dest_filter = params.get("destination", "")  # Optional filter
        time_str = params.get("time", "now")

        if not origin_str:
            return LiveDataResult(
                success=False,
                error="Origin station is required for departure queries.",
            )

        # Resolve station
        station = self._resolve_station(origin_str)
        if not station or not station.get("code"):
            return LiveDataResult(
                success=False,
                error=f"Could not find station: {origin_str}",
            )

        station_code = station["code"]

        # Resolve destination filter if provided
        dest_station = None
        if dest_filter:
            dest_station = self._resolve_station(dest_filter)

        # Parse time
        date_str, time_val = self._parse_time(time_str)

        # Build cache key
        cache_key = f"departures:{station_code}:{dest_station['code'] if dest_station else ''}:{date_str}:{time_val}"

        # Check cache
        cached = self._get_cached(_timetable_cache, cache_key, self.LIVE_CACHE_TTL)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=self._format_departures(cached, station, dest_station),
                cache_ttl=self.LIVE_CACHE_TTL,
            )

        # Fetch live departures
        try:
            api_params = {
                **self._get_auth_params(),
                "darwin": "true",  # Get real-time data
                "train_status": "passenger",
            }

            # Add destination filter if specified
            if dest_station:
                api_params["calling_at"] = dest_station["code"]

            # If time is in the future, use timetable endpoint
            now = datetime.now()
            req_date = datetime.strptime(date_str, "%Y-%m-%d")
            req_time = datetime.strptime(time_val, "%H:%M")
            req_datetime = req_date.replace(hour=req_time.hour, minute=req_time.minute)

            if req_datetime > now + timedelta(minutes=5):
                # Use timetable endpoint for future times
                api_params["date"] = date_str
                api_params["time"] = time_val
                endpoint = (
                    f"{self.BASE_URL}/train/station/{station_code}/timetable.json"
                )
            else:
                # Use live endpoint for current departures
                endpoint = f"{self.BASE_URL}/train/station/{station_code}/live.json"

            response = self._client.get(endpoint, params=api_params)
            response.raise_for_status()
            data = response.json()

            self._set_cached(_timetable_cache, cache_key, data)

            return LiveDataResult(
                success=True,
                data=data,
                formatted=self._format_departures(data, station, dest_station),
                cache_ttl=self.LIVE_CACHE_TTL,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return LiveDataResult(
                    success=False,
                    error=f"Station not found: {station_code}",
                )
            return LiveDataResult(
                success=False,
                error=f"Departures API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Departures error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_service(self, params: dict) -> LiveDataResult:
        """Fetch status of a specific train service."""
        service_id = params.get("service_id", "")
        origin_str = params.get("origin", "")
        dest_str = params.get("destination", "")
        time_str = params.get("time", "")

        # If no service_id, try to find the service from departures
        if not service_id and origin_str and dest_str and time_str:
            # Find the service by looking at departures
            deps_result = self._fetch_departures(
                {
                    "origin": origin_str,
                    "destination": dest_str,
                    "time": time_str,
                }
            )

            if deps_result.success and deps_result.data:
                departures = deps_result.data.get("departures", {}).get("all", [])
                if departures:
                    # Find the closest service to requested time
                    _, time_val = self._parse_time(time_str)
                    for dep in departures:
                        aimed_time = dep.get("aimed_departure_time", "")
                        if aimed_time == time_val or (not service_id and departures):
                            service_id = dep.get("service_toc", "") + dep.get(
                                "train_uid", ""
                            )
                            # Return enhanced departure info as service status
                            return LiveDataResult(
                                success=True,
                                data=dep,
                                formatted=self._format_service_status(
                                    dep, origin_str, dest_str
                                ),
                                cache_ttl=self.LIVE_CACHE_TTL,
                            )

            return LiveDataResult(
                success=False,
                error="Could not find the specified service. Please provide origin, destination, and approximate time.",
            )

        if not service_id:
            return LiveDataResult(
                success=False,
                error="Service ID or origin/destination/time required for service status queries.",
            )

        # Fetch service details
        try:
            date_str, _ = self._parse_time(time_str or "now")

            response = self._client.get(
                f"{self.BASE_URL}/train/service/{service_id}/{date_str}/timetable.json",
                params=self._get_auth_params(),
            )
            response.raise_for_status()
            data = response.json()

            return LiveDataResult(
                success=True,
                data=data,
                formatted=self._format_service_detail(data),
                cache_ttl=self.LIVE_CACHE_TTL,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return LiveDataResult(
                    success=False,
                    error=f"Service not found: {service_id}",
                )
            return LiveDataResult(
                success=False,
                error=f"Service API error: {e.response.status_code}",
            )
        except Exception as e:
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_bus(self, params: dict) -> LiveDataResult:
        """Fetch bus departures from a stop or nearby location."""
        origin_str = params.get("origin", "")

        if not origin_str:
            return LiveDataResult(
                success=False,
                error="Location or stop name required for bus queries.",
            )

        # Try to find bus stops near the location
        try:
            # First resolve the place to get coordinates
            place = self._resolve_place(origin_str)

            if place and place.get("latitude"):
                # Search for bus stops near this location
                response = self._client.get(
                    f"{self.BASE_URL}/places.json",
                    params={
                        **self._get_auth_params(),
                        "lat": place["latitude"],
                        "lon": place["longitude"],
                        "type": "bus_stop",
                        "rpp": 5,  # Get nearest 5 stops
                    },
                )
                response.raise_for_status()
                places_data = response.json()

                stops = places_data.get("member", [])
                if not stops:
                    return LiveDataResult(
                        success=False,
                        error=f"No bus stops found near: {origin_str}",
                    )

                # Get departures from the nearest stop
                stop = stops[0]
                atcocode = stop.get("atcocode")

                if atcocode:
                    dep_response = self._client.get(
                        f"{self.BASE_URL}/bus/stop/{atcocode}/live.json",
                        params={
                            **self._get_auth_params(),
                            "group": "route",
                            "nextbuses": "yes",
                        },
                    )
                    dep_response.raise_for_status()
                    data = dep_response.json()

                    return LiveDataResult(
                        success=True,
                        data=data,
                        formatted=self._format_bus_departures(data, stop),
                        cache_ttl=self.LIVE_CACHE_TTL,
                    )

            return LiveDataResult(
                success=False,
                error=f"Could not find bus stops near: {origin_str}",
            )

        except httpx.HTTPStatusError as e:
            return LiveDataResult(
                success=False,
                error=f"Bus API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Bus departures error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _format_journey(
        self, data: dict, origin: dict, destination: dict, arrive_by: bool
    ) -> str:
        """Format journey plan for LLM context."""
        lines = [
            f"### Journey: {origin.get('name', 'Origin')} → {destination.get('name', 'Destination')}"
        ]

        routes = data.get("routes", [])
        if not routes:
            lines.append("No routes found for this journey.")
            return "\n".join(lines)

        for i, route in enumerate(routes[:3]):  # Show up to 3 options
            if i > 0:
                lines.append("")
                lines.append(f"**Alternative {i + 1}:**")

            # Journey summary
            departure = route.get("departure_datetime", "")
            arrival = route.get("arrival_datetime", "")
            duration = route.get("duration", "")

            if departure and arrival:
                dep_time = (
                    departure.split("T")[1][:5] if "T" in departure else departure
                )
                arr_time = arrival.split("T")[1][:5] if "T" in arrival else arrival
                lines.append(f"**Depart:** {dep_time} → **Arrive:** {arr_time}")

            if duration:
                # Duration is in seconds
                try:
                    mins = int(duration) // 60
                    hours = mins // 60
                    mins = mins % 60
                    if hours > 0:
                        lines.append(f"**Duration:** {hours}h {mins}min")
                    else:
                        lines.append(f"**Duration:** {mins} minutes")
                except (ValueError, TypeError):
                    pass

            # Route legs
            route_legs = route.get("route_parts", [])
            if route_legs:
                lines.append("**Route:**")
                for leg in route_legs:
                    mode = leg.get("mode", "")
                    from_name = leg.get("from_point_name", "")
                    to_name = leg.get("to_point_name", "")
                    line_name = leg.get("line_name", "")
                    departure_time = leg.get("departure_time", "")
                    arrival_time = leg.get("arrival_time", "")

                    if mode == "foot":
                        duration_mins = leg.get("duration", 0) // 60
                        lines.append(f"  🚶 Walk {duration_mins} min to {to_name}")
                    elif mode in ("train", "rail", "national-rail"):
                        operator = leg.get("operator", {}).get("name", "")
                        lines.append(
                            f"  🚆 {departure_time} {line_name or 'Train'} to {to_name}"
                            + (f" ({operator})" if operator else "")
                        )
                    elif mode in ("tube", "underground"):
                        lines.append(f"  🚇 {line_name} line to {to_name}")
                    elif mode == "bus":
                        lines.append(f"  🚌 Bus {line_name} to {to_name}")
                    elif mode in ("dlr", "overground", "tram"):
                        lines.append(f"  🚃 {line_name or mode.upper()} to {to_name}")
                    else:
                        lines.append(f"  {mode}: {from_name} → {to_name}")

        return "\n".join(lines)

    def _format_departures(
        self, data: dict, station: dict, dest_filter: Optional[dict]
    ) -> str:
        """Format train departures for LLM context."""
        station_name = data.get("station_name", station.get("name", "Station"))
        lines = [f"### Train Departures: {station_name}"]

        if dest_filter:
            lines[0] += f" → {dest_filter.get('name', dest_filter.get('code', ''))}"

        departures = data.get("departures", {}).get("all", [])

        if not departures:
            lines.append("No departures found.")
            return "\n".join(lines)

        lines.append("")

        for dep in departures[:8]:  # Show up to 8 departures
            aimed_time = dep.get("aimed_departure_time", "??:??")
            expected_time = dep.get("expected_departure_time")
            destination_name = dep.get("destination_name", "Unknown")
            platform = dep.get("platform", "?")
            status = dep.get("status", "")
            operator = dep.get("operator_name", "")

            # Build departure line
            line = f"**{aimed_time}** to **{destination_name}**"

            # Add delay info
            if expected_time and expected_time != aimed_time:
                if expected_time == "Cancelled":
                    line += " - ❌ CANCELLED"
                else:
                    # Calculate delay
                    try:
                        aimed = datetime.strptime(aimed_time, "%H:%M")
                        expected = datetime.strptime(expected_time, "%H:%M")
                        # Use total_seconds() to get signed value
                        delay_mins = int((expected - aimed).total_seconds()) // 60
                        if delay_mins > 0:
                            line += (
                                f" - ⚠️ Expected {expected_time} (+{delay_mins}min late)"
                            )
                        elif delay_mins < 0:
                            line += (
                                f" - Expected {expected_time} ({delay_mins}min early)"
                            )
                    except ValueError:
                        line += f" - Expected {expected_time}"
            elif status:
                if "on time" in status.lower():
                    line += " - ✅ On time"
                elif "delayed" in status.lower():
                    line += f" - ⚠️ {status}"
                elif "cancelled" in status.lower():
                    line += " - ❌ CANCELLED"

            # Platform info
            if platform and platform != "?":
                line += f" (Plat {platform})"

            lines.append(line)

            # Add operator on separate line if useful
            if operator and len(departures) <= 5:
                lines.append(f"  _{operator}_")

        # Add timestamp
        lines.append("")
        lines.append(f"_Updated: {datetime.now().strftime('%H:%M')}_")

        return "\n".join(lines)

    def _format_service_status(self, dep: dict, origin: str, destination: str) -> str:
        """Format a single service status."""
        lines = [f"### Service Status: {origin} → {destination}"]
        lines.append("")

        aimed_time = dep.get("aimed_departure_time", "??:??")
        expected_time = dep.get("expected_departure_time")
        platform = dep.get("platform", "?")
        status = dep.get("status", "")
        operator = dep.get("operator_name", "")

        lines.append(f"**Scheduled:** {aimed_time}")

        if expected_time:
            if expected_time == "Cancelled":
                lines.append("**Status:** ❌ CANCELLED")
            elif expected_time != aimed_time:
                try:
                    aimed = datetime.strptime(aimed_time, "%H:%M")
                    expected = datetime.strptime(expected_time, "%H:%M")
                    delay_mins = (expected - aimed).seconds // 60
                    lines.append(
                        f"**Expected:** {expected_time} (⚠️ {delay_mins} minutes late)"
                    )
                except ValueError:
                    lines.append(f"**Expected:** {expected_time}")
            else:
                lines.append("**Status:** ✅ On time")
        elif status:
            lines.append(f"**Status:** {status}")

        if platform and platform != "?":
            lines.append(f"**Platform:** {platform}")

        if operator:
            lines.append(f"**Operator:** {operator}")

        return "\n".join(lines)

    def _format_service_detail(self, data: dict) -> str:
        """Format detailed service information."""
        lines = ["### Service Details"]

        # This would format the full service timetable
        # For now, just show key info
        stops = data.get("stops", [])
        if stops:
            lines.append("")
            lines.append("**Calling at:**")
            for stop in stops:
                name = stop.get("station_name", "")
                aimed_arr = stop.get("aimed_arrival_time", "")
                aimed_dep = stop.get("aimed_departure_time", "")
                time_str = aimed_dep or aimed_arr or ""
                if name and time_str:
                    lines.append(f"  {time_str} - {name}")

        return "\n".join(lines)

    def _format_bus_departures(self, data: dict, stop: dict) -> str:
        """Format bus departures for LLM context."""
        stop_name = data.get("name", stop.get("name", "Bus Stop"))
        lines = [f"### Bus Departures: {stop_name}"]

        departures = data.get("departures", {})
        if not departures:
            lines.append("No bus departures found.")
            return "\n".join(lines)

        lines.append("")

        # Departures are grouped by route
        for route, deps in departures.items():
            if not deps:
                continue

            for dep in deps[:3]:  # Show up to 3 per route
                aimed_time = dep.get("aimed_departure_time", "")
                expected_time = dep.get("expected_departure_time", aimed_time)
                direction = dep.get("direction", "")
                operator = dep.get("operator_name", "")

                line = f"**{route}** ({aimed_time})"
                if direction:
                    line += f" → {direction}"

                if expected_time and expected_time != aimed_time:
                    line += f" - Expected {expected_time}"

                lines.append(line)

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if TransportAPI credentials are configured."""
        return bool(self.app_id and self.app_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to TransportAPI."""
        if not self.app_id or not self.app_key:
            return False, "TransportAPI credentials not set"

        result = self.fetch({"query_type": "departures", "origin": "Kings Cross"})
        if result.success:
            return True, "Connected - fetched Kings Cross departures"
        return False, result.error or "Unknown error"
