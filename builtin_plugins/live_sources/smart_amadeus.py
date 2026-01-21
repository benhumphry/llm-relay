"""
Smart Amadeus Air Travel live source plugin.

Comprehensive air travel data using Amadeus Self-Service APIs:
- Flight search (one-way, round-trip, multi-city)
- Flight status (real-time departure/arrival info)
- Flight inspiration (cheapest destinations from origin)
- Cheapest dates (best prices for a route)
- Airport information (search, routes, on-time performance)
- Airline information (routes, codes)
- Price analysis (how current prices compare historically)

All APIs use the same API key/secret for OAuth2 authentication.
Supports both test and production API environments.

Requires AMADEUS_API_KEY and AMADEUS_API_SECRET environment variables.
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType, SelectOption
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)

logger = logging.getLogger(__name__)

# Global caches with TTL
_token_cache: dict[str, tuple[str, float]] = {}  # OAuth tokens
_airport_cache: dict[str, tuple[Any, float]] = {}  # Airport/city lookups (30-day TTL)
_airline_cache: dict[str, tuple[Any, float]] = {}  # Airline lookups (30-day TTL)


class SmartAmadeusLiveSource(PluginLiveSource):
    """
    Smart Amadeus Air Travel Provider - comprehensive flight data.

    Supports multiple query types:
    - search: Search for flights between cities/airports
    - status: Real-time flight status by flight number
    - inspiration: Find cheapest destinations from an origin
    - cheapest_dates: Find cheapest travel dates for a route
    - airport_info: Airport details, routes, on-time performance
    - airline_info: Airline details and routes
    - price_analysis: Compare current prices to historical data

    The designator provides:
    - query_type: Type of query to perform
    - origin: Origin airport/city (IATA code or name)
    - destination: Destination airport/city (for search/cheapest_dates)
    - date: Departure date (natural language or YYYY-MM-DD)
    - return_date: Return date for round-trip (optional)
    - passengers: Number of adult passengers (default: 1)
    - cabin_class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    - nonstop: Whether to search nonstop flights only
    - flight_number: For status queries (e.g., "BA123")
    - carrier: Airline code for status/airline_info queries

    Examples:
    - "Find flights from London to New York next Friday"
    - "What's the status of BA117 today?"
    - "Cheapest destinations from Paris in March"
    - "Best dates to fly from LAX to Tokyo"
    - "What airlines fly from Heathrow to JFK?"
    """

    source_type = "amadeus_flights"
    display_name = "Amadeus Flights"
    description = "Air travel data: flight search, status, prices, airports via Amadeus"
    category = "travel"
    data_type = "flights"
    best_for = (
        "Flight searches, flight status, airport info, airline routes, ticket prices. "
        "Use for 'flights to X', 'cheapest flights', 'flight status BA123', "
        "'what airlines fly to', 'best time to fly to'"
    )
    icon = "✈️"
    default_cache_ttl = 300  # 5 minutes for flight data

    _abstract = False

    # API base URLs
    API_BASE_TEST = "https://test.api.amadeus.com"
    API_BASE_PROD = "https://api.amadeus.com"

    # Cache TTLs
    TOKEN_CACHE_TTL = 1700  # ~28 minutes (tokens expire in 30 min)
    AIRPORT_CACHE_TTL = 86400 * 30  # 30 days for airport/airline lookups
    FLIGHT_CACHE_TTL = 300  # 5 minutes for flight searches
    STATUS_CACHE_TTL = 60  # 1 minute for flight status

    # Common airport codes for quick lookup
    KNOWN_AIRPORTS = {
        # London
        "london": "LON",
        "heathrow": "LHR",
        "gatwick": "LGW",
        "stansted": "STN",
        "luton": "LTN",
        "london city": "LCY",
        # UK
        "manchester": "MAN",
        "birmingham": "BHX",
        "edinburgh": "EDI",
        "glasgow": "GLA",
        "bristol": "BRS",
        "newcastle": "NCL",
        "belfast": "BFS",
        "leeds": "LBA",
        "liverpool": "LPL",
        "southampton": "SOU",
        "east midlands": "EMA",
        # Europe
        "paris": "PAR",
        "paris cdg": "CDG",
        "paris orly": "ORY",
        "amsterdam": "AMS",
        "frankfurt": "FRA",
        "munich": "MUC",
        "berlin": "BER",
        "rome": "ROM",
        "rome fiumicino": "FCO",
        "milan": "MIL",
        "madrid": "MAD",
        "barcelona": "BCN",
        "lisbon": "LIS",
        "dublin": "DUB",
        "zurich": "ZRH",
        "geneva": "GVA",
        "vienna": "VIE",
        "brussels": "BRU",
        "copenhagen": "CPH",
        "stockholm": "ARN",
        "oslo": "OSL",
        "helsinki": "HEL",
        "athens": "ATH",
        "istanbul": "IST",
        "prague": "PRG",
        "warsaw": "WAW",
        "budapest": "BUD",
        # North America
        "new york": "NYC",
        "jfk": "JFK",
        "newark": "EWR",
        "laguardia": "LGA",
        "los angeles": "LAX",
        "san francisco": "SFO",
        "chicago": "ORD",
        "miami": "MIA",
        "boston": "BOS",
        "washington": "WAS",
        "washington dulles": "IAD",
        "washington reagan": "DCA",
        "seattle": "SEA",
        "denver": "DEN",
        "dallas": "DFW",
        "atlanta": "ATL",
        "las vegas": "LAS",
        "toronto": "YYZ",
        "vancouver": "YVR",
        "montreal": "YUL",
        # Asia
        "tokyo": "TYO",
        "tokyo narita": "NRT",
        "tokyo haneda": "HND",
        "hong kong": "HKG",
        "singapore": "SIN",
        "bangkok": "BKK",
        "dubai": "DXB",
        "abu dhabi": "AUH",
        "doha": "DOH",
        "delhi": "DEL",
        "mumbai": "BOM",
        "beijing": "PEK",
        "shanghai": "PVG",
        "seoul": "ICN",
        "taipei": "TPE",
        "kuala lumpur": "KUL",
        "sydney": "SYD",
        "melbourne": "MEL",
        "auckland": "AKL",
        # Caribbean/Americas
        "cancun": "CUN",
        "mexico city": "MEX",
        "sao paulo": "GRU",
        "rio": "GIG",
        "buenos aires": "EZE",
        "lima": "LIM",
        "bogota": "BOG",
        # Africa
        "johannesburg": "JNB",
        "cape town": "CPT",
        "cairo": "CAI",
        "nairobi": "NBO",
        "casablanca": "CMN",
    }

    # Common airline codes
    KNOWN_AIRLINES = {
        "british airways": "BA",
        "ba": "BA",
        "virgin atlantic": "VS",
        "easyjet": "U2",
        "ryanair": "FR",
        "air france": "AF",
        "klm": "KL",
        "lufthansa": "LH",
        "emirates": "EK",
        "qatar": "QR",
        "qatar airways": "QR",
        "etihad": "EY",
        "american": "AA",
        "american airlines": "AA",
        "united": "UA",
        "united airlines": "UA",
        "delta": "DL",
        "jetblue": "B6",
        "southwest": "WN",
        "air canada": "AC",
        "qantas": "QF",
        "singapore airlines": "SQ",
        "cathay pacific": "CX",
        "japan airlines": "JL",
        "ana": "NH",
        "korean air": "KE",
        "turkish airlines": "TK",
        "iberia": "IB",
        "swiss": "LX",
        "austrian": "OS",
        "finnair": "AY",
        "sas": "SK",
        "norwegian": "DY",
        "aer lingus": "EI",
        "tap portugal": "TP",
        "alitalia": "AZ",
        "ita airways": "AZ",
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
                default="Amadeus Flights",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="Amadeus API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="AMADEUS_API_KEY",
                help_text="Leave empty to use AMADEUS_API_KEY env var",
            ),
            FieldDefinition(
                name="api_secret",
                label="Amadeus API Secret",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="AMADEUS_API_SECRET",
                help_text="Leave empty to use AMADEUS_API_SECRET env var",
            ),
            FieldDefinition(
                name="environment",
                label="API Environment",
                field_type=FieldType.SELECT,
                required=True,
                default="test",
                options=[
                    SelectOption(value="test", label="Test (Limited data, free)"),
                    SelectOption(
                        value="production", label="Production (Full data, paid)"
                    ),
                ],
                help_text="Test environment has limited data but is free",
            ),
            FieldDefinition(
                name="default_currency",
                label="Default Currency",
                field_type=FieldType.SELECT,
                required=False,
                default="GBP",
                options=[
                    SelectOption(value="GBP", label="British Pound (GBP)"),
                    SelectOption(value="USD", label="US Dollar (USD)"),
                    SelectOption(value="EUR", label="Euro (EUR)"),
                ],
                help_text="Currency for price results",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query_type",
                description=(
                    "Type of query: 'search' (find flights), 'status' (flight status), "
                    "'inspiration' (cheapest destinations), 'cheapest_dates' (best prices for route), "
                    "'airport_info' (airport details/routes), 'airline_info' (airline routes), "
                    "'price_analysis' (historical price comparison)"
                ),
                param_type="string",
                required=True,
                examples=[
                    "search",
                    "status",
                    "inspiration",
                    "cheapest_dates",
                    "airport_info",
                ],
            ),
            ParamDefinition(
                name="origin",
                description="Origin airport or city (IATA code or name like 'London', 'JFK', 'Paris')",
                param_type="string",
                required=False,
                examples=["LHR", "London", "JFK", "New York", "Paris"],
            ),
            ParamDefinition(
                name="destination",
                description="Destination airport or city",
                param_type="string",
                required=False,
                examples=["LAX", "Tokyo", "Dubai", "NYC"],
            ),
            ParamDefinition(
                name="date",
                description="Departure date - natural language or YYYY-MM-DD",
                param_type="string",
                required=False,
                examples=["2026-03-15", "next Friday", "March 15", "tomorrow"],
            ),
            ParamDefinition(
                name="return_date",
                description="Return date for round-trip flights",
                param_type="string",
                required=False,
                examples=["2026-03-22", "next Sunday", "a week later"],
            ),
            ParamDefinition(
                name="passengers",
                description="Number of adult passengers",
                param_type="integer",
                required=False,
                default=1,
                examples=["1", "2", "4"],
            ),
            ParamDefinition(
                name="cabin_class",
                description="Cabin class preference",
                param_type="string",
                required=False,
                examples=["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
            ),
            ParamDefinition(
                name="nonstop",
                description="Only show nonstop/direct flights",
                param_type="boolean",
                required=False,
                default=False,
            ),
            ParamDefinition(
                name="max_price",
                description="Maximum price per person",
                param_type="integer",
                required=False,
                examples=["500", "1000", "2000"],
            ),
            ParamDefinition(
                name="flight_number",
                description="Flight number for status queries (e.g., 'BA117', 'AA100')",
                param_type="string",
                required=False,
                examples=["BA117", "AA100", "EK1"],
            ),
            ParamDefinition(
                name="carrier",
                description="Airline code for status or airline_info queries",
                param_type="string",
                required=False,
                examples=["BA", "AA", "EK", "LH"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "amadeus-flights")
        self.api_key = config.get("api_key") or os.environ.get("AMADEUS_API_KEY", "")
        self.api_secret = config.get("api_secret") or os.environ.get(
            "AMADEUS_API_SECRET", ""
        )
        self.environment = config.get("environment", "test")
        self.default_currency = config.get("default_currency", "GBP")

        # Set base URL based on environment
        self.base_url = (
            self.API_BASE_PROD
            if self.environment == "production"
            else self.API_BASE_TEST
        )

        self._client = httpx.Client(timeout=30, follow_redirects=True)

    def _get_access_token(self) -> Optional[str]:
        """
        Get OAuth2 access token, using cache if valid.

        Amadeus uses client credentials grant type.
        Tokens expire in 30 minutes.
        """
        cache_key = f"{self.environment}:{self.api_key}"

        # Check cache
        if cache_key in _token_cache:
            token, cached_at = _token_cache[cache_key]
            if time.time() - cached_at < self.TOKEN_CACHE_TTL:
                return token

        # Request new token
        try:
            response = self._client.post(
                f"{self.base_url}/v1/security/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

            token = data.get("access_token")
            if token:
                _token_cache[cache_key] = (token, time.time())
                logger.info(f"Amadeus OAuth token obtained for {self.environment}")
                return token

        except Exception as e:
            logger.error(f"Failed to get Amadeus OAuth token: {e}")

        return None

    def _api_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        json_data: dict = None,
    ) -> Optional[dict]:
        """Make an authenticated API request."""
        token = self._get_access_token()
        if not token:
            raise ValueError("Failed to authenticate with Amadeus API")

        url = f"{self.base_url}{endpoint}"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            if method.upper() == "GET":
                response = self._client.get(url, params=params, headers=headers)
            else:
                response = self._client.post(url, json=json_data, headers=headers)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                errors = error_data.get("errors", [])
                if errors:
                    error_detail = "; ".join(
                        err.get("detail", str(err)) for err in errors
                    )
            except Exception:
                error_detail = e.response.text[:200]
            logger.error(
                f"Amadeus API error: {e.response.status_code} - {error_detail}"
            )
            raise ValueError(
                f"Amadeus API error: {error_detail or e.response.status_code}"
            )

    def _resolve_airport(self, location: str) -> Optional[str]:
        """
        Resolve a location name to IATA airport/city code.

        Returns IATA code or None.
        """
        if not location:
            return None

        location_clean = location.strip()
        location_lower = location_clean.lower()

        # Already a valid IATA code (3 letters)
        if len(location_clean) == 3 and location_clean.isalpha():
            return location_clean.upper()

        # Check hardcoded lookup
        if location_lower in self.KNOWN_AIRPORTS:
            return self.KNOWN_AIRPORTS[location_lower]

        # Check cache
        cache_key = f"airport:{location_lower}"
        if cache_key in _airport_cache:
            data, cached_at = _airport_cache[cache_key]
            if time.time() - cached_at < self.AIRPORT_CACHE_TTL:
                return data

        # Query Amadeus API
        try:
            data = self._api_request(
                "GET",
                "/v1/reference-data/locations",
                params={
                    "subType": "AIRPORT,CITY",
                    "keyword": location_clean,
                    "page[limit]": 5,
                },
            )

            locations = data.get("data", [])
            if locations:
                # Prefer city codes, then airports
                for loc in locations:
                    if loc.get("subType") == "CITY":
                        code = loc.get("iataCode")
                        if code:
                            _airport_cache[cache_key] = (code, time.time())
                            logger.info(f"Resolved '{location}' -> {code} (city)")
                            return code

                # Fall back to first airport
                code = locations[0].get("iataCode")
                if code:
                    _airport_cache[cache_key] = (code, time.time())
                    logger.info(f"Resolved '{location}' -> {code}")
                    return code

        except Exception as e:
            logger.warning(f"Airport lookup failed for '{location}': {e}")

        return None

    def _resolve_airline(self, airline: str) -> Optional[str]:
        """Resolve airline name to IATA code."""
        if not airline:
            return None

        airline_clean = airline.strip()
        airline_lower = airline_clean.lower()

        # Already a valid code (2 letters)
        if len(airline_clean) == 2 and airline_clean.isalpha():
            return airline_clean.upper()

        # Check hardcoded lookup
        if airline_lower in self.KNOWN_AIRLINES:
            return self.KNOWN_AIRLINES[airline_lower]

        return airline_clean.upper()[:2]  # Best guess

    def _parse_date(
        self, date_str: str, reference_date: datetime = None
    ) -> Optional[str]:
        """
        Parse natural language date to YYYY-MM-DD format.

        Returns date string or None.
        """
        if not date_str:
            return None

        date_str = date_str.strip().lower()
        now = reference_date or datetime.now()

        # Already in YYYY-MM-DD format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # Handle relative dates
        if date_str == "today":
            return now.strftime("%Y-%m-%d")
        elif date_str == "tomorrow":
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next week" in date_str:
            return (now + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "a week later" in date_str or "week later" in date_str:
            return (now + timedelta(days=7)).strftime("%Y-%m-%d")

        # Parse "next [day]"
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
            if day in date_str:
                current_day = now.weekday()
                days_ahead = i - current_day
                if "next" in date_str or days_ahead <= 0:
                    days_ahead += 7
                target = now + timedelta(days=days_ahead)
                return target.strftime("%Y-%m-%d")

        # Parse "in X days/weeks"
        in_match = re.match(r"in\s+(\d+)\s*(day|week)s?", date_str)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)
            if unit == "week":
                amount *= 7
            return (now + timedelta(days=amount)).strftime("%Y-%m-%d")

        # Parse month name + day (e.g., "March 15", "15 March")
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        for month_name, month_num in months.items():
            if month_name in date_str:
                # Extract day number
                day_match = re.search(r"(\d{1,2})", date_str)
                if day_match:
                    day = int(day_match.group(1))
                    year = now.year
                    # If the date has passed this year, use next year
                    target = datetime(year, month_num, min(day, 28))
                    if target < now:
                        target = datetime(year + 1, month_num, min(day, 28))
                    return target.strftime("%Y-%m-%d")
                else:
                    # Just month name - use 1st of month
                    year = now.year
                    target = datetime(year, month_num, 1)
                    if target < now:
                        target = datetime(year + 1, month_num, 1)
                    return target.strftime("%Y-%m-%d")

        return None

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch air travel data based on query type.

        Args:
            params: Dict with query_type and relevant parameters

        Returns:
            LiveDataResult with flight/travel data
        """
        if not self.api_key or not self.api_secret:
            return LiveDataResult(
                success=False,
                error="Amadeus API credentials not configured. Set AMADEUS_API_KEY and AMADEUS_API_SECRET.",
            )

        query_type = params.get("query_type", "search").lower()

        try:
            if query_type == "search":
                return self._fetch_flight_search(params)
            elif query_type == "status":
                return self._fetch_flight_status(params)
            elif query_type == "inspiration":
                return self._fetch_inspiration(params)
            elif query_type == "cheapest_dates":
                return self._fetch_cheapest_dates(params)
            elif query_type == "airport_info":
                return self._fetch_airport_info(params)
            elif query_type == "airline_info":
                return self._fetch_airline_info(params)
            elif query_type == "price_analysis":
                return self._fetch_price_analysis(params)
            else:
                return LiveDataResult(
                    success=False,
                    error=f"Unknown query type: {query_type}. Use: search, status, inspiration, cheapest_dates, airport_info, airline_info, price_analysis",
                )

        except ValueError as e:
            return LiveDataResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Amadeus fetch error: {e}", exc_info=True)
            return LiveDataResult(success=False, error=f"Flight data error: {e}")

    def _fetch_flight_search(self, params: dict) -> LiveDataResult:
        """Search for flights between two locations."""
        origin = self._resolve_airport(params.get("origin", ""))
        destination = self._resolve_airport(params.get("destination", ""))

        if not origin:
            return LiveDataResult(
                success=False, error="Origin airport/city is required"
            )
        if not destination:
            return LiveDataResult(
                success=False, error="Destination airport/city is required"
            )

        # Parse dates
        dep_date = self._parse_date(params.get("date", "")) or (
            datetime.now() + timedelta(days=7)
        ).strftime("%Y-%m-%d")
        return_date = self._parse_date(
            params.get("return_date", ""), datetime.strptime(dep_date, "%Y-%m-%d")
        )

        # Build query
        query_params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": dep_date,
            "adults": params.get("passengers", 1),
            "currencyCode": self.default_currency,
            "max": 10,  # Limit results
        }

        if return_date:
            query_params["returnDate"] = return_date

        if params.get("cabin_class"):
            query_params["travelClass"] = params["cabin_class"].upper()

        if params.get("nonstop"):
            query_params["nonStop"] = "true"

        if params.get("max_price"):
            query_params["maxPrice"] = params["max_price"]

        data = self._api_request(
            "GET", "/v2/shopping/flight-offers", params=query_params
        )

        # Format results
        offers = data.get("data", [])
        dictionaries = data.get("dictionaries", {})

        if not offers:
            return LiveDataResult(
                success=True,
                data=[],
                formatted=f"No flights found from {origin} to {destination} on {dep_date}.",
                cache_ttl=self.FLIGHT_CACHE_TTL,
            )

        formatted = self._format_flight_offers(
            offers, dictionaries, origin, destination, dep_date, return_date
        )

        return LiveDataResult(
            success=True,
            data=offers,
            formatted=formatted,
            cache_ttl=self.FLIGHT_CACHE_TTL,
        )

    def _format_flight_offers(
        self,
        offers: list,
        dictionaries: dict,
        origin: str,
        destination: str,
        dep_date: str,
        return_date: str = None,
    ) -> str:
        """Format flight offers for LLM context."""
        carriers = dictionaries.get("carriers", {})
        aircraft = dictionaries.get("aircraft", {})

        trip_type = "round-trip" if return_date else "one-way"
        lines = [
            f"### Flight Search Results",
            f"**Route**: {origin} → {destination} ({trip_type})",
            f"**Departure**: {dep_date}"
            + (f" | **Return**: {return_date}" if return_date else ""),
            f"**Found**: {len(offers)} options",
            "",
        ]

        for i, offer in enumerate(offers[:8], 1):  # Limit to 8 for context
            price = offer.get("price", {})
            total = price.get("grandTotal", price.get("total", "N/A"))
            currency = price.get("currency", self.default_currency)

            itineraries = offer.get("itineraries", [])

            lines.append(f"**Option {i}: {currency} {total}**")

            for j, itinerary in enumerate(itineraries):
                direction = "Outbound" if j == 0 else "Return"
                segments = itinerary.get("segments", [])
                duration = itinerary.get("duration", "")

                # Parse duration (PT2H30M -> 2h 30m)
                dur_match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration)
                if dur_match:
                    hours = dur_match.group(1) or "0"
                    mins = dur_match.group(2) or "0"
                    duration = f"{hours}h {mins}m"

                stops = len(segments) - 1
                stop_text = (
                    "Nonstop"
                    if stops == 0
                    else f"{stops} stop{'s' if stops > 1 else ''}"
                )

                if segments:
                    first_seg = segments[0]
                    last_seg = segments[-1]

                    dep_time = (
                        first_seg.get("departure", {})
                        .get("at", "")[:16]
                        .replace("T", " ")
                    )
                    arr_time = (
                        last_seg.get("arrival", {}).get("at", "")[:16].replace("T", " ")
                    )
                    dep_airport = first_seg.get("departure", {}).get("iataCode", "")
                    arr_airport = last_seg.get("arrival", {}).get("iataCode", "")

                    carrier_code = first_seg.get("carrierCode", "")
                    carrier_name = carriers.get(carrier_code, carrier_code)
                    flight_num = first_seg.get("number", "")

                    lines.append(
                        f"  {direction}: {dep_airport} {dep_time} → {arr_airport} {arr_time}"
                    )
                    lines.append(
                        f"    {carrier_name} {carrier_code}{flight_num} | {duration} | {stop_text}"
                    )

                    if stops > 0:
                        via = [
                            s.get("arrival", {}).get("iataCode", "")
                            for s in segments[:-1]
                        ]
                        lines.append(f"    Via: {' → '.join(via)}")

            lines.append("")

        return "\n".join(lines)

    def _fetch_flight_status(self, params: dict) -> LiveDataResult:
        """Get real-time flight status."""
        flight_number = params.get("flight_number", "")
        carrier = params.get("carrier", "")
        date = self._parse_date(params.get("date", "today"))

        if not flight_number and not carrier:
            return LiveDataResult(
                success=False,
                error="Flight number (e.g., BA117) or carrier code is required",
            )

        # Parse flight number if provided (e.g., "BA117" -> carrier=BA, number=117)
        if flight_number:
            match = re.match(r"([A-Z]{2})(\d+)", flight_number.upper())
            if match:
                carrier = match.group(1)
                flight_num = match.group(2)
            else:
                return LiveDataResult(
                    success=False,
                    error=f"Invalid flight number format: {flight_number}. Use format like 'BA117'",
                )
        else:
            flight_num = params.get("flight_number", "").replace(carrier, "")

        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        data = self._api_request(
            "GET",
            "/v2/schedule/flights",
            params={
                "carrierCode": carrier,
                "flightNumber": flight_num,
                "scheduledDepartureDate": date,
            },
        )

        flights = data.get("data", [])

        if not flights:
            return LiveDataResult(
                success=True,
                data=[],
                formatted=f"No flight status found for {carrier}{flight_num} on {date}.",
                cache_ttl=self.STATUS_CACHE_TTL,
            )

        formatted = self._format_flight_status(flights, carrier, flight_num, date)

        return LiveDataResult(
            success=True,
            data=flights,
            formatted=formatted,
            cache_ttl=self.STATUS_CACHE_TTL,
        )

    def _format_flight_status(
        self, flights: list, carrier: str, flight_num: str, date: str
    ) -> str:
        """Format flight status for LLM context."""
        lines = [
            f"### Flight Status: {carrier}{flight_num}",
            f"**Date**: {date}",
            "",
        ]

        for flight in flights:
            flight_points = flight.get("flightPoints", [])
            segments = flight.get("segments", [])

            if len(flight_points) >= 2:
                dep = flight_points[0]
                arr = flight_points[-1]

                dep_airport = dep.get("iataCode", "")
                arr_airport = arr.get("iataCode", "")

                dep_times = dep.get("departure", {})
                arr_times = arr.get("arrival", {})

                sched_dep = dep_times.get("timings", [{}])[0].get("value", "")[:16]
                sched_arr = arr_times.get("timings", [{}])[0].get("value", "")[:16]

                lines.append(f"**{dep_airport} → {arr_airport}**")
                lines.append(f"  Scheduled Departure: {sched_dep.replace('T', ' ')}")
                lines.append(f"  Scheduled Arrival: {sched_arr.replace('T', ' ')}")

                # Check for delays/updates in timings
                for timing in dep_times.get("timings", []):
                    qualifier = timing.get("qualifier", "")
                    if qualifier not in ["STD"]:  # Not standard time
                        val = timing.get("value", "")[:16]
                        lines.append(f"  {qualifier}: {val.replace('T', ' ')}")

                if segments:
                    seg = segments[0]
                    aircraft_code = seg.get("boardPointIataCode", "")
                    lines.append(
                        f"  Aircraft: {seg.get('equipment', {}).get('aircraftType', 'N/A')}"
                    )

            lines.append("")

        return "\n".join(lines)

    def _fetch_inspiration(self, params: dict) -> LiveDataResult:
        """Find cheapest destinations from an origin."""
        origin = self._resolve_airport(params.get("origin", ""))

        if not origin:
            return LiveDataResult(
                success=False, error="Origin airport/city is required"
            )

        query_params = {
            "origin": origin,
            "maxPrice": params.get("max_price", 500),
        }

        # Add date if provided
        if params.get("date"):
            dep_date = self._parse_date(params["date"])
            if dep_date:
                query_params["departureDate"] = dep_date

        if params.get("nonstop"):
            query_params["nonStop"] = "true"

        data = self._api_request(
            "GET", "/v1/shopping/flight-destinations", params=query_params
        )

        destinations = data.get("data", [])

        if not destinations:
            return LiveDataResult(
                success=True,
                data=[],
                formatted=f"No destination inspiration found from {origin}. Try adjusting max_price or dates.",
                cache_ttl=self.FLIGHT_CACHE_TTL,
            )

        formatted = self._format_inspiration(destinations, origin)

        return LiveDataResult(
            success=True,
            data=destinations,
            formatted=formatted,
            cache_ttl=self.FLIGHT_CACHE_TTL,
        )

    def _format_inspiration(self, destinations: list, origin: str) -> str:
        """Format destination inspiration for LLM context."""
        lines = [
            f"### Cheapest Destinations from {origin}",
            f"**Found**: {len(destinations)} destinations",
            "",
        ]

        for dest in destinations[:15]:  # Limit for context
            dest_code = dest.get("destination", "")
            price = dest.get("price", {})
            total = price.get("total", "N/A")
            dep_date = dest.get("departureDate", "")
            return_date = dest.get("returnDate", "")

            lines.append(f"**{dest_code}**: £{total}")
            if dep_date:
                lines.append(
                    f"  Dates: {dep_date}"
                    + (f" - {return_date}" if return_date else "")
                )

        return "\n".join(lines)

    def _fetch_cheapest_dates(self, params: dict) -> LiveDataResult:
        """Find cheapest travel dates for a route."""
        origin = self._resolve_airport(params.get("origin", ""))
        destination = self._resolve_airport(params.get("destination", ""))

        if not origin:
            return LiveDataResult(
                success=False, error="Origin airport/city is required"
            )
        if not destination:
            return LiveDataResult(
                success=False, error="Destination airport/city is required"
            )

        data = self._api_request(
            "GET",
            "/v1/shopping/flight-dates",
            params={
                "origin": origin,
                "destination": destination,
            },
        )

        dates = data.get("data", [])

        if not dates:
            return LiveDataResult(
                success=True,
                data=[],
                formatted=f"No price data found for {origin} → {destination}.",
                cache_ttl=self.FLIGHT_CACHE_TTL,
            )

        formatted = self._format_cheapest_dates(dates, origin, destination)

        return LiveDataResult(
            success=True,
            data=dates,
            formatted=formatted,
            cache_ttl=self.FLIGHT_CACHE_TTL,
        )

    def _format_cheapest_dates(self, dates: list, origin: str, destination: str) -> str:
        """Format cheapest dates for LLM context."""
        lines = [
            f"### Cheapest Dates: {origin} → {destination}",
            f"**Found**: {len(dates)} date options",
            "",
        ]

        # Sort by price
        sorted_dates = sorted(
            dates, key=lambda x: float(x.get("price", {}).get("total", 999999))
        )

        for d in sorted_dates[:12]:  # Limit for context
            dep_date = d.get("departureDate", "")
            return_date = d.get("returnDate", "")
            price = d.get("price", {})
            total = price.get("total", "N/A")

            lines.append(f"**{dep_date}** → {return_date}: £{total}")

        return "\n".join(lines)

    def _fetch_airport_info(self, params: dict) -> LiveDataResult:
        """Get airport information and routes."""
        origin = self._resolve_airport(params.get("origin", ""))

        if not origin:
            return LiveDataResult(
                success=False, error="Airport code or name is required"
            )

        # Get direct destinations from this airport
        data = self._api_request(
            "GET",
            "/v1/airport/direct-destinations",
            params={
                "departureAirportCode": origin,
                "max": 50,
            },
        )

        destinations = data.get("data", [])

        # Also try to get on-time performance if date provided
        on_time = None
        if params.get("date"):
            date = self._parse_date(params["date"])
            if date:
                try:
                    ot_data = self._api_request(
                        "GET",
                        "/v1/airport/predictions/on-time",
                        params={
                            "airportCode": origin,
                            "date": date,
                        },
                    )
                    on_time = ot_data.get("data", [])
                except Exception:
                    pass

        formatted = self._format_airport_info(origin, destinations, on_time)

        return LiveDataResult(
            success=True,
            data={"destinations": destinations, "on_time": on_time},
            formatted=formatted,
            cache_ttl=self.AIRPORT_CACHE_TTL,
        )

    def _format_airport_info(
        self, airport: str, destinations: list, on_time: list = None
    ) -> str:
        """Format airport info for LLM context."""
        lines = [
            f"### Airport Information: {airport}",
            "",
        ]

        if on_time:
            for ot in on_time:
                prob = ot.get("probability", "N/A")
                lines.append(
                    f"**On-Time Performance**: {float(prob) * 100:.0f}% flights on time"
                )
                lines.append("")

        if destinations:
            lines.append(f"**Direct Destinations**: {len(destinations)} routes")
            lines.append("")

            # Group by region/distance if possible
            dest_codes = [d.get("destination", "") for d in destinations[:40]]
            lines.append(f"Destinations: {', '.join(dest_codes)}")

        return "\n".join(lines)

    def _fetch_airline_info(self, params: dict) -> LiveDataResult:
        """Get airline information and routes."""
        carrier = self._resolve_airline(params.get("carrier", ""))

        if not carrier:
            return LiveDataResult(
                success=False, error="Airline code or name is required"
            )

        # Get airline details
        try:
            airline_data = self._api_request(
                "GET",
                "/v1/reference-data/airlines",
                params={"airlineCodes": carrier},
            )
            airlines = airline_data.get("data", [])
        except Exception:
            airlines = []

        # Get routes
        try:
            routes_data = self._api_request(
                "GET",
                "/v1/airline/destinations",
                params={
                    "airlineCode": carrier,
                    "max": 100,
                },
            )
            routes = routes_data.get("data", [])
        except Exception:
            routes = []

        formatted = self._format_airline_info(carrier, airlines, routes)

        return LiveDataResult(
            success=True,
            data={"airline": airlines, "routes": routes},
            formatted=formatted,
            cache_ttl=self.AIRPORT_CACHE_TTL,
        )

    def _format_airline_info(self, carrier: str, airlines: list, routes: list) -> str:
        """Format airline info for LLM context."""
        lines = [f"### Airline Information: {carrier}", ""]

        if airlines:
            for airline in airlines:
                name = airline.get("businessName") or airline.get("commonName", carrier)
                lines.append(f"**Name**: {name}")
                iata = airline.get("iataCode", "")
                icao = airline.get("icaoCode", "")
                if iata or icao:
                    lines.append(f"**Codes**: IATA: {iata}, ICAO: {icao}")
                lines.append("")

        if routes:
            lines.append(f"**Destinations**: {len(routes)} routes")
            dest_codes = [r.get("destination", "") for r in routes[:50]]
            lines.append(f"Serves: {', '.join(dest_codes)}")

        return "\n".join(lines)

    def _fetch_price_analysis(self, params: dict) -> LiveDataResult:
        """Get price analysis comparing current prices to historical."""
        origin = self._resolve_airport(params.get("origin", ""))
        destination = self._resolve_airport(params.get("destination", ""))
        date = self._parse_date(params.get("date", ""))

        if not origin:
            return LiveDataResult(
                success=False, error="Origin airport/city is required"
            )
        if not destination:
            return LiveDataResult(
                success=False, error="Destination airport/city is required"
            )
        if not date:
            date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        data = self._api_request(
            "GET",
            "/v1/analytics/itinerary-price-metrics",
            params={
                "originIataCode": origin,
                "destinationIataCode": destination,
                "departureDate": date,
                "currencyCode": self.default_currency,
            },
        )

        metrics = data.get("data", [])

        if not metrics:
            return LiveDataResult(
                success=True,
                data=[],
                formatted=f"No price analysis available for {origin} → {destination} on {date}.",
                cache_ttl=self.FLIGHT_CACHE_TTL,
            )

        formatted = self._format_price_analysis(metrics, origin, destination, date)

        return LiveDataResult(
            success=True,
            data=metrics,
            formatted=formatted,
            cache_ttl=self.FLIGHT_CACHE_TTL,
        )

    def _format_price_analysis(
        self, metrics: list, origin: str, destination: str, date: str
    ) -> str:
        """Format price analysis for LLM context."""
        lines = [
            f"### Price Analysis: {origin} → {destination}",
            f"**Date**: {date}",
            "",
        ]

        for m in metrics:
            price_metrics = m.get("priceMetrics", [])
            for pm in price_metrics:
                quartile = pm.get("quartileRanking", "")
                amount = pm.get("amount", "N/A")
                lines.append(f"**{quartile}**: £{amount}")

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if Amadeus API is accessible."""
        return bool(self.api_key and self.api_secret)

    def test_connection(self) -> tuple[bool, str]:
        """Test Amadeus API connection."""
        if not self.api_key or not self.api_secret:
            return False, "API credentials not configured"

        try:
            token = self._get_access_token()
            if not token:
                return False, "Failed to obtain OAuth token"

            env_label = "Production" if self.environment == "production" else "Test"

            # Try multiple endpoints - test environment can be flaky
            test_endpoints = [
                # Airlines endpoint is most reliable
                ("/v1/reference-data/airlines", {"airlineCodes": "BA"}),
                # Locations as fallback
                (
                    "/v1/reference-data/locations",
                    {"subType": "AIRPORT", "keyword": "LHR", "page[limit]": 1},
                ),
            ]

            for endpoint, params in test_endpoints:
                try:
                    data = self._api_request("GET", endpoint, params=params)
                    if data.get("data"):
                        return True, f"Connected to Amadeus ({env_label}). API working."
                except ValueError as e:
                    # 500 errors from test environment - try next endpoint
                    if "500" in str(e) or "internal error" in str(e).lower():
                        continue
                    raise

            # If we got here, all endpoints failed but OAuth worked
            return (
                True,
                f"Connected to Amadeus ({env_label}). OAuth OK, but API endpoints returning errors (test environment may be experiencing issues).",
            )

        except Exception as e:
            return False, f"Connection failed: {e}"
