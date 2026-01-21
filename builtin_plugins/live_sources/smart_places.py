"""
Smart Places live source plugin.

High-level places/POI interface with natural language support.
Unlike GoogleMapsLiveSource which requires structured parameters (lat/lng, place IDs),
this provider accepts natural language inputs for locations and queries.

Requires GOOGLE_MAPS_API_KEY environment variable.
"""

import logging
import os
import time
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
_geocode_cache: dict[str, tuple[Any, float]] = {}  # 30-day TTL for geocoding
_places_cache: dict[str, tuple[Any, float]] = {}  # 5-min TTL for places


class SmartPlacesLiveSource(PluginLiveSource):
    """
    Smart Places Provider - natural language place search.

    Unlike GoogleMapsLiveSource which requires structured parameters,
    this provider accepts natural language inputs and handles:
    - Geocoding location names to coordinates (with 30-day caching)
    - Combining query + location into smart searches
    - Automatic nearby search when location context is provided

    Examples:
    - "Italian restaurants in Soho"
    - "coffee shops near Tower Bridge"
    - "24 hour pharmacy in Manchester"
    - "best sushi near Canary Wharf"
    - "petrol stations on M1 near Leicester"
    """

    source_type = "google_places_enhanced"
    display_name = "Google Places (Enhanced)"
    description = (
        "Find places, restaurants, businesses using natural language locations"
    )
    category = "places"
    data_type = "places"
    best_for = "Finding restaurants, shops, services, attractions. Use natural language like 'pizza near King's Cross' or 'pharmacies in Camden'."
    icon = "📍"
    default_cache_ttl = 300  # 5 minutes

    _abstract = False  # Allow registration

    PLACES_BASE_URL = "https://places.googleapis.com/v1/places"
    GEOCODING_BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    # Cache TTLs
    GEOCODE_CACHE_TTL = 86400 * 30  # 30 days for geocoding
    PLACES_CACHE_TTL = 300  # 5 minutes for places

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields for admin UI."""
        return [
            FieldDefinition(
                name="name",
                label="Source Name",
                field_type=FieldType.TEXT,
                required=True,
                default="Smart Places",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="Google Maps API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="GOOGLE_MAPS_API_KEY",
                help_text="Leave empty to use GOOGLE_MAPS_API_KEY env var",
            ),
            FieldDefinition(
                name="default_location",
                label="Default Location",
                field_type=FieldType.TEXT,
                required=False,
                default="",
                help_text="Default location for searches when none specified (e.g., 'London')",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query",
                description="What to search for (type of place, business name, etc.)",
                param_type="string",
                required=True,
                examples=[
                    "Italian restaurants",
                    "coffee shops",
                    "24 hour pharmacy",
                    "Tesco",
                    "petrol station",
                ],
            ),
            ParamDefinition(
                name="location",
                description="Where to search (area, landmark, address, or 'near me')",
                param_type="string",
                required=False,
                examples=[
                    "Soho",
                    "near Tower Bridge",
                    "in Manchester",
                    "Canary Wharf",
                    "SW1A 1AA",
                ],
            ),
            ParamDefinition(
                name="open_now",
                description="Only show places that are currently open",
                param_type="boolean",
                required=False,
                default=False,
                examples=["true", "false"],
            ),
            ParamDefinition(
                name="min_rating",
                description="Minimum rating filter (1-5)",
                param_type="number",
                required=False,
                examples=["4", "4.5"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-places")
        self.api_key = config.get("api_key") or os.environ.get(
            "GOOGLE_MAPS_API_KEY", ""
        )
        self.default_location = config.get("default_location", "")

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

    def _geocode(self, location: str) -> Optional[tuple[float, float]]:
        """
        Geocode a location string to lat/lng coordinates.

        Results cached for 30 days.
        """
        if not location:
            return None

        location = location.strip()

        # Check cache
        cache_key = f"geocode:{location.lower()}"
        cached = self._get_cached(_geocode_cache, cache_key, self.GEOCODE_CACHE_TTL)
        if cached:
            logger.debug(f"Geocode cache hit: '{location}' -> {cached}")
            return cached

        try:
            response = self._client.get(
                self.GEOCODING_BASE_URL,
                params={
                    "address": location,
                    "key": self.api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if results:
                loc = results[0].get("geometry", {}).get("location", {})
                lat = loc.get("lat")
                lng = loc.get("lng")
                if lat and lng:
                    coords = (lat, lng)
                    self._set_cached(_geocode_cache, cache_key, coords)
                    logger.info(f"Geocoded '{location}' -> {coords}")
                    return coords

            logger.warning(f"Could not geocode: {location}")
            return None

        except Exception as e:
            logger.warning(f"Geocoding failed for '{location}': {e}")
            return None

    def _clean_location(self, location: str) -> str:
        """Clean up location string by removing common prefixes."""
        if not location:
            return ""

        location = location.strip()

        # Remove common prefixes
        prefixes = ["near ", "in ", "around ", "at ", "close to ", "by "]
        location_lower = location.lower()
        for prefix in prefixes:
            if location_lower.startswith(prefix):
                location = location[len(prefix) :]
                break

        return location.strip()

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Search for places using natural language.

        Combines query and location intelligently:
        - "Italian restaurants" + "Soho" -> searches for Italian restaurants near Soho
        - "Tesco" + "near Tower Bridge" -> finds Tesco stores near Tower Bridge
        """
        if not self.api_key:
            return LiveDataResult(
                success=False,
                error="GOOGLE_MAPS_API_KEY not configured",
            )

        query = params.get("query", "").strip()
        location = params.get("location", "").strip()
        open_now = params.get("open_now", False)
        min_rating = params.get("min_rating")

        if not query:
            return LiveDataResult(
                success=False,
                error="Please specify what you're looking for (e.g., 'coffee shops', 'Italian restaurants')",
            )

        # Use default location if none provided
        if not location and self.default_location:
            location = self.default_location

        # Clean up location string
        location = self._clean_location(location)

        # Build the search query
        if location:
            # Combine query and location for text search
            search_text = f"{query} in {location}"
        else:
            search_text = query

        # Check cache
        cache_key = f"places:{search_text}:{open_now}:{min_rating}"
        cached = self._get_cached(_places_cache, cache_key, self.PLACES_CACHE_TTL)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=self._format_places(cached, query, location),
                cache_ttl=self.PLACES_CACHE_TTL,
            )

        # Try to geocode location for location bias
        location_bias = None
        if location:
            coords = self._geocode(location)
            if coords:
                location_bias = {
                    "circle": {
                        "center": {"latitude": coords[0], "longitude": coords[1]},
                        "radius": 5000.0,  # 5km bias radius
                    }
                }

        # Build Places API request
        request_body = {
            "textQuery": search_text,
            "maxResultCount": 10,
        }

        if location_bias:
            request_body["locationBias"] = location_bias

        # Add open now filter if requested
        if open_now:
            request_body["openNow"] = True

        # Add minimum rating filter if specified
        if min_rating:
            try:
                request_body["minRating"] = float(min_rating)
            except (ValueError, TypeError):
                pass

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.currentOpeningHours,places.websiteUri,places.nationalPhoneNumber,places.types,places.editorialSummary",
        }

        try:
            response = self._client.post(
                f"{self.PLACES_BASE_URL}:searchText",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            places = data.get("places", [])

            # Cache result
            self._set_cached(_places_cache, cache_key, places)

            return LiveDataResult(
                success=True,
                data=places,
                formatted=self._format_places(places, query, location),
                cache_ttl=self.PLACES_CACHE_TTL,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Google Places API error: {e.response.status_code} - {e.response.text}"
            )
            return LiveDataResult(
                success=False,
                error=f"Places API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Places search failed: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _format_places(self, places: list, query: str, location: str = "") -> str:
        """Format places results for context injection."""
        if location:
            header = f"**{query.title()} in {location}**"
        else:
            header = f"**{query.title()}**"

        if not places:
            return f"{header}\n\nNo places found matching your search."

        lines = [header]
        lines.append(f"_Found {len(places)} result(s)_\n")

        for i, place in enumerate(places, 1):
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "")
            rating = place.get("rating")
            rating_count = place.get("userRatingCount", 0)
            price_level = place.get("priceLevel", "")
            phone = place.get("nationalPhoneNumber", "")
            website = place.get("websiteUri", "")
            summary = place.get("editorialSummary", {}).get("text", "")

            lines.append(f"**{i}. {name}**")

            if summary:
                lines.append(f"   _{summary}_")

            if address:
                lines.append(f"   📍 {address}")

            # Rating and price on same line
            info_parts = []
            if rating:
                stars = "⭐" * int(rating)
                info_parts.append(f"{rating}/5 {stars} ({rating_count} reviews)")
            if price_level:
                price_map = {
                    "PRICE_LEVEL_FREE": "Free",
                    "PRICE_LEVEL_INEXPENSIVE": "£",
                    "PRICE_LEVEL_MODERATE": "££",
                    "PRICE_LEVEL_EXPENSIVE": "£££",
                    "PRICE_LEVEL_VERY_EXPENSIVE": "££££",
                }
                info_parts.append(price_map.get(price_level, ""))
            if info_parts:
                lines.append(f"   {' | '.join(info_parts)}")

            # Opening status
            hours = place.get("currentOpeningHours", {})
            if hours.get("openNow") is not None:
                status = "🟢 Open now" if hours["openNow"] else "🔴 Closed"
                lines.append(f"   {status}")

            # Contact info
            if phone:
                lines.append(f"   📞 {phone}")
            if website:
                # Truncate long URLs
                display_url = website if len(website) < 50 else website[:47] + "..."
                lines.append(f"   🌐 {display_url}")

            lines.append("")  # Blank line between results

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection with a sample search."""
        if not self.api_key:
            return False, "GOOGLE_MAPS_API_KEY not configured"

        try:
            result = self.fetch({"query": "coffee shop", "location": "London"})
            if result.success:
                count = len(result.data) if result.data else 0
                return True, f"Connected - found {count} coffee shops in London"
            return False, result.error or "Unknown error"
        except Exception as e:
            return False, str(e)
