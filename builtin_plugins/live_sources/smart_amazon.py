"""
Smart Amazon live source plugin.

High-level Amazon product interface with intelligent multi-step lookups.
Searches for products, resolves ASINs, then fetches detailed info and reviews.

Uses Real-Time Amazon Data API via RapidAPI.

Smart features:
- Natural language product search
- Auto-resolves product names to ASINs
- Combines search + details + reviews in one query
- Caches ASIN resolutions for 24 hours
"""

import logging
import os
import re
import time
from datetime import datetime
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
_asin_cache: dict[str, tuple[str, float]] = {}  # 24-hour TTL for product name -> ASIN
_product_cache: dict[str, tuple[Any, float]] = {}  # 1-hour TTL for product details
_search_cache: dict[str, tuple[Any, float]] = {}  # 30-min TTL for search results


class SmartAmazonLiveSource(PluginLiveSource):
    """
    Smart Amazon Provider - intelligent multi-step product lookups.

    Unlike basic product APIs, this provider:
    - Accepts natural language product queries
    - Auto-resolves product names to ASINs with caching
    - Combines search + details + reviews for comprehensive info
    - Handles "best selling", "deals", and comparison queries

    Examples:
    - "iPhone 15 Pro Max" -> Search, get ASIN, fetch details + reviews
    - "Reviews for AirPods Pro" -> Search, get ASIN, fetch reviews
    - "Compare Sony WH-1000XM5 vs Bose QC45" -> Search both, compare
    - "Best selling headphones" -> Best sellers in category
    - "Amazon deals on electronics" -> Current deals
    """

    source_type = "realtime_amazon"
    display_name = "Real-Time Amazon"
    description = (
        "Amazon product search with smart ASIN resolution and review aggregation"
    )
    category = "shopping"
    data_type = "products"
    best_for = "BEST FOR: 'How much does X cost?', product prices, Amazon product lookups, reviews, comparisons. USE THIS when user asks about product pricing, buying products, or product details. Use natural product names like 'OnePlus Watch 3', 'MacBook Pro M3', 'best wireless earbuds'."
    icon = "🛒"
    default_cache_ttl = 1800  # 30 minutes

    _abstract = False

    # API configuration
    BASE_URL = "https://real-time-amazon-data.p.rapidapi.com"

    # Cache TTLs
    ASIN_CACHE_TTL = 86400  # 24 hours for name -> ASIN resolution
    PRODUCT_CACHE_TTL = 3600  # 1 hour for product details
    SEARCH_CACHE_TTL = 1800  # 30 minutes for search
    DEALS_CACHE_TTL = 900  # 15 minutes for deals

    # Amazon country codes (API expects 2-letter codes, not domains)
    COUNTRIES = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "uk": "GB",
        "gb": "GB",
        "united kingdom": "GB",
        "germany": "DE",
        "de": "DE",
        "france": "FR",
        "fr": "FR",
        "italy": "IT",
        "it": "IT",
        "spain": "ES",
        "es": "ES",
        "canada": "CA",
        "ca": "CA",
        "japan": "JP",
        "jp": "JP",
        "australia": "AU",
        "au": "AU",
        "india": "IN",
        "in": "IN",
        "mexico": "MX",
        "mx": "MX",
        "brazil": "BR",
        "br": "BR",
        "netherlands": "NL",
        "nl": "NL",
        "singapore": "SG",
        "sg": "SG",
        "saudi arabia": "SA",
        "sa": "SA",
        "uae": "AE",
        "ae": "AE",
        "turkey": "TR",
        "tr": "TR",
        "sweden": "SE",
        "se": "SE",
        "poland": "PL",
        "pl": "PL",
        "belgium": "BE",
        "be": "BE",
        "egypt": "EG",
        "eg": "EG",
    }

    # Best seller categories
    CATEGORIES = {
        "electronics": "electronics",
        "computers": "computers",
        "laptops": "computers",
        "phones": "mobile-phones",
        "mobile": "mobile-phones",
        "headphones": "headphones",
        "audio": "electronics",
        "books": "books",
        "clothing": "fashion",
        "fashion": "fashion",
        "home": "home",
        "kitchen": "kitchen",
        "toys": "toys",
        "games": "video-games",
        "video games": "video-games",
        "sports": "sports",
        "beauty": "beauty",
        "health": "health",
        "garden": "garden",
        "automotive": "automotive",
        "tools": "tools",
        "office": "office-products",
        "pet": "pet-supplies",
        "baby": "baby",
        "grocery": "grocery",
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
                default="Smart Amazon",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="RapidAPI Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="RAPIDAPI_KEY",
                help_text="RapidAPI key for Real-Time Amazon Data API",
            ),
            FieldDefinition(
                name="default_country",
                label="Default Amazon Store",
                field_type=FieldType.SELECT,
                required=False,
                default="GB",
                options=[
                    {"value": "US", "label": "United States (amazon.com)"},
                    {"value": "GB", "label": "United Kingdom (amazon.co.uk)"},
                    {"value": "DE", "label": "Germany (amazon.de)"},
                    {"value": "FR", "label": "France (amazon.fr)"},
                    {"value": "IT", "label": "Italy (amazon.it)"},
                    {"value": "ES", "label": "Spain (amazon.es)"},
                    {"value": "CA", "label": "Canada (amazon.ca)"},
                    {"value": "JP", "label": "Japan (amazon.co.jp)"},
                    {"value": "AU", "label": "Australia (amazon.com.au)"},
                    {"value": "IN", "label": "India (amazon.in)"},
                    {"value": "MX", "label": "Mexico (amazon.com.mx)"},
                    {"value": "BR", "label": "Brazil (amazon.com.br)"},
                    {"value": "NL", "label": "Netherlands (amazon.nl)"},
                    {"value": "SG", "label": "Singapore (amazon.sg)"},
                    {"value": "SA", "label": "Saudi Arabia (amazon.sa)"},
                    {"value": "AE", "label": "UAE (amazon.ae)"},
                    {"value": "TR", "label": "Turkey (amazon.com.tr)"},
                    {"value": "SE", "label": "Sweden (amazon.se)"},
                    {"value": "PL", "label": "Poland (amazon.pl)"},
                    {"value": "BE", "label": "Belgium (amazon.com.be)"},
                    {"value": "EG", "label": "Egypt (amazon.eg)"},
                ],
                help_text="Default Amazon marketplace for product searches",
            ),
            FieldDefinition(
                name="include_reviews",
                label="Include Reviews",
                field_type=FieldType.BOOLEAN,
                required=False,
                default=True,
                help_text="Fetch top reviews for products (more comprehensive but slower)",
            ),
            FieldDefinition(
                name="max_results",
                label="Max Search Results",
                field_type=FieldType.INTEGER,
                required=False,
                default=5,
                min_value=1,
                max_value=20,
                help_text="Maximum products to show in search results",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="query",
                description="Product name, ASIN, or search query",
                param_type="string",
                required=True,
                examples=[
                    "iPhone 15 Pro Max",
                    "Sony WH-1000XM5",
                    "best wireless earbuds",
                    "reviews for Kindle Paperwhite",
                ],
            ),
            ParamDefinition(
                name="type",
                description="Query type: search, details, reviews, bestsellers, deals",
                param_type="string",
                required=False,
                default="auto",
                examples=["search", "details", "reviews", "bestsellers", "deals"],
            ),
            ParamDefinition(
                name="country",
                description="Amazon marketplace (country code or name)",
                param_type="string",
                required=False,
                examples=["us", "uk", "germany"],
            ),
            ParamDefinition(
                name="include_reviews",
                description="Whether to fetch reviews for products",
                param_type="boolean",
                required=False,
                default=True,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-amazon")
        self.api_key = config.get("api_key") or os.environ.get("RAPIDAPI_KEY", "")
        # Resolve default_country to ensure it's a country code, not a domain
        raw_country = config.get("default_country", "GB")
        self.default_country = self._normalize_country_code(raw_country)
        self.include_reviews = config.get("include_reviews", True)
        self.max_results = config.get("max_results", 5)

        self._client = httpx.Client(timeout=20)

    @staticmethod
    def _normalize_country_code(country: str) -> str:
        """Normalize country to 2-letter code. Handles legacy domain values."""
        if not country:
            return "GB"

        country_clean = country.strip().upper()

        # Common aliases that need mapping
        code_aliases = {
            "UK": "GB",  # API uses GB not UK
        }
        if country_clean in code_aliases:
            return code_aliases[country_clean]

        # Already a valid 2-letter code
        if len(country_clean) == 2:
            return country_clean

        # Legacy domain -> code mapping
        domain_to_code = {
            "amazon.com": "US",
            "amazon.co.uk": "GB",
            "amazon.de": "DE",
            "amazon.fr": "FR",
            "amazon.it": "IT",
            "amazon.es": "ES",
            "amazon.ca": "CA",
            "amazon.co.jp": "JP",
            "amazon.com.au": "AU",
            "amazon.in": "IN",
        }
        country_lower = country.strip().lower()
        if country_lower in domain_to_code:
            return domain_to_code[country_lower]

        # Try lowercase lookup in COUNTRIES
        return SmartAmazonLiveSource.COUNTRIES.get(country_lower, "GB")

    def _get_headers(self) -> dict:
        """Get API headers."""
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com",
        }

    def _get_cached_asin(self, product_name: str) -> Optional[str]:
        """Get cached ASIN for a product name."""
        key = product_name.lower().strip()
        if key in _asin_cache:
            asin, cached_at = _asin_cache[key]
            if time.time() - cached_at < self.ASIN_CACHE_TTL:
                return asin
            del _asin_cache[key]
        return None

    def _cache_asin(self, product_name: str, asin: str) -> None:
        """Cache ASIN for a product name."""
        key = product_name.lower().strip()
        _asin_cache[key] = (asin, time.time())

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

    def _is_asin(self, text: str) -> bool:
        """Check if text looks like an Amazon ASIN."""
        text = text.strip().upper()
        # ASIN is 10 characters, alphanumeric, often starts with B0
        if len(text) == 10 and text.isalnum():
            return True
        return False

    def _detect_request_type(self, query: str) -> tuple[str, dict]:
        """
        Detect the type of Amazon request from natural language.

        Returns (request_type, extracted_params)
        """
        query_lower = query.lower().strip()

        # Check for explicit ASIN
        words = query.split()
        for word in words:
            if self._is_asin(word):
                return "details", {"asin": word.upper()}

        # Check for reviews request
        if "review" in query_lower:
            # Extract product name
            product = query_lower.replace("reviews for", "").replace("reviews of", "")
            product = product.replace("review for", "").replace("review of", "")
            product = product.replace("reviews", "").replace("review", "").strip()
            return "reviews", {"product": product}

        # Check for deals
        if "deal" in query_lower:
            return "deals", {"query": query}

        # Check for best sellers
        if (
            "best sell" in query_lower
            or "bestsell" in query_lower
            or "top sell" in query_lower
        ):
            for cat_key, cat_value in self.CATEGORIES.items():
                if cat_key in query_lower:
                    return "bestsellers", {"category": cat_value}
            return "bestsellers", {"category": ""}

        # Check for comparison
        if (
            " vs " in query_lower
            or " versus " in query_lower
            or "compare" in query_lower
        ):
            # Extract products to compare
            query_clean = query_lower.replace("compare", "").strip()
            if " vs " in query_clean:
                products = query_clean.split(" vs ")
            elif " versus " in query_clean:
                products = query_clean.split(" versus ")
            else:
                products = [query_clean]
            return "compare", {"products": [p.strip() for p in products if p.strip()]}

        # Check for price check / specific product lookup
        if "price" in query_lower or "how much" in query_lower or "cost" in query_lower:
            product = query_lower.replace("price of", "").replace("price for", "")
            product = product.replace("how much is", "").replace("how much does", "")
            product = product.replace("cost", "").replace("price", "").strip()
            return "details", {"product": product}

        # Default: product details (search -> get details)
        return "product", {"product": query}

    def _resolve_country(self, country: str) -> str:
        """Resolve country name to API country code."""
        if not country:
            return self.default_country

        # Use the static normalizer which handles UK->GB and other edge cases
        return self._normalize_country_code(country)

    def _resolve_asin(self, product_name: str, country: str) -> Optional[str]:
        """
        Resolve a product name to an ASIN.

        First checks cache, then searches Amazon.
        """
        # Check cache
        cached = self._get_cached_asin(product_name)
        if cached:
            logger.info(f"ASIN cache hit: {product_name} -> {cached}")
            return cached

        # Search for product
        logger.info(f"Resolving ASIN for: {product_name}")
        try:
            response = self._client.get(
                f"{self.BASE_URL}/search",
                headers=self._get_headers(),
                params={
                    "query": product_name,
                    "country": country,
                    "page": 1,
                },
            )
            response.raise_for_status()
            data = response.json()

            products = data.get("data", {}).get("products", [])
            if products:
                asin = products[0].get("asin")
                if asin:
                    self._cache_asin(product_name, asin)
                    logger.info(f"Resolved ASIN: {product_name} -> {asin}")
                    return asin

        except Exception as e:
            logger.error(f"Failed to resolve ASIN for {product_name}: {e}")

        return None

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch Amazon data with smart multi-step lookups.

        For product queries:
        1. Search for product
        2. Get ASIN of best match
        3. Fetch detailed product info
        4. Optionally fetch reviews

        For comparisons:
        1. Resolve ASINs for both products
        2. Fetch details for both
        3. Format as comparison
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
                error="Query is required. Specify a product name or search term.",
            )

        request_type = params.get("type", "auto")
        country = params.get("country", "")
        include_reviews = params.get("include_reviews", self.include_reviews)

        # Auto-detect request type
        if request_type == "auto":
            request_type, detected_params = self._detect_request_type(query)
        else:
            detected_params = {"product": query}

        # Resolve country
        country = self._resolve_country(country)

        # Check cache
        cache_key = f"amazon:{request_type}:{query}:{country}:{include_reviews}"

        try:
            if request_type == "details" and detected_params.get("asin"):
                # Direct ASIN lookup
                return self._fetch_product_with_reviews(
                    detected_params["asin"], country, include_reviews
                )

            elif request_type == "reviews" and detected_params.get("product"):
                # Resolve ASIN then get reviews
                asin = self._resolve_asin(detected_params["product"], country)
                if not asin:
                    return LiveDataResult(
                        success=False,
                        error=f"Could not find product: {detected_params['product']}",
                    )
                return self._fetch_product_with_reviews(asin, country, True)

            elif request_type == "compare" and detected_params.get("products"):
                # Compare multiple products
                return self._fetch_comparison(
                    detected_params["products"], country, include_reviews
                )

            elif request_type == "bestsellers":
                # Best sellers
                return self._fetch_best_sellers(
                    detected_params.get("category", ""), country
                )

            elif request_type == "deals":
                # Current deals
                return self._fetch_deals(country)

            elif request_type == "product" or request_type == "details":
                # Product lookup: search -> ASIN -> details
                product_name = detected_params.get("product", query)
                asin = self._resolve_asin(product_name, country)
                if not asin:
                    # Fall back to search results
                    return self._fetch_search_results(product_name, country)
                return self._fetch_product_with_reviews(asin, country, include_reviews)

            else:
                # Default search
                return self._fetch_search_results(query, country)

        except httpx.HTTPStatusError as e:
            logger.error(f"Amazon API HTTP error: {e}")
            return LiveDataResult(
                success=False,
                error=f"API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Amazon API error: {e}", exc_info=True)
            return LiveDataResult(
                success=False,
                error=f"Failed to fetch Amazon data: {str(e)}",
            )

    def _fetch_product_with_reviews(
        self, asin: str, country: str, include_reviews: bool
    ) -> LiveDataResult:
        """Fetch product details and optionally reviews."""
        cache_key = f"product:{asin}:{country}:{include_reviews}"
        cached = self._get_cached(_product_cache, cache_key, self.PRODUCT_CACHE_TTL)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=cached.get("_formatted", ""),
                cache_ttl=self.PRODUCT_CACHE_TTL,
            )

        # Fetch product details
        logger.info(f"Fetching product details: {asin}")
        response = self._client.get(
            f"{self.BASE_URL}/product-details",
            headers=self._get_headers(),
            params={"asin": asin, "country": country},
        )
        response.raise_for_status()
        product_data = response.json()

        result_data = {"product": product_data.get("data", {})}
        formatted_parts = [self._format_product_details(product_data.get("data", {}))]

        # Fetch reviews if requested
        if include_reviews:
            logger.info(f"Fetching reviews for: {asin}")
            try:
                reviews_response = self._client.get(
                    f"{self.BASE_URL}/product-reviews",
                    headers=self._get_headers(),
                    params={
                        "asin": asin,
                        "country": country,
                        "sort_by": "TOP_REVIEWS",
                        "page": 1,
                    },
                )
                reviews_response.raise_for_status()
                reviews_data = reviews_response.json()
                result_data["reviews"] = reviews_data.get("data", {}).get("reviews", [])
                formatted_parts.append(
                    "\n" + self._format_reviews(reviews_data.get("data", {}))
                )
            except Exception as e:
                logger.warning(f"Failed to fetch reviews: {e}")

        formatted = "\n".join(formatted_parts)
        result_data["_formatted"] = formatted
        self._set_cached(_product_cache, cache_key, result_data)

        return LiveDataResult(
            success=True,
            data=result_data,
            formatted=formatted,
            cache_ttl=self.PRODUCT_CACHE_TTL,
        )

    def _fetch_comparison(
        self, products: list, country: str, include_reviews: bool
    ) -> LiveDataResult:
        """Fetch and compare multiple products."""
        if len(products) < 2:
            return LiveDataResult(
                success=False,
                error="Need at least 2 products to compare",
            )

        results = []
        for product_name in products[:3]:  # Max 3 products
            asin = self._resolve_asin(product_name, country)
            if asin:
                try:
                    response = self._client.get(
                        f"{self.BASE_URL}/product-details",
                        headers=self._get_headers(),
                        params={"asin": asin, "country": country},
                    )
                    response.raise_for_status()
                    data = response.json().get("data", {})
                    data["_search_name"] = product_name
                    results.append(data)
                except Exception as e:
                    logger.warning(f"Failed to fetch {product_name}: {e}")

        if not results:
            return LiveDataResult(
                success=False,
                error=f"Could not find products: {', '.join(products)}",
            )

        formatted = self._format_comparison(results)

        return LiveDataResult(
            success=True,
            data={"products": results},
            formatted=formatted,
            cache_ttl=self.PRODUCT_CACHE_TTL,
        )

    def _fetch_search_results(self, query: str, country: str) -> LiveDataResult:
        """Fetch search results."""
        cache_key = f"search:{query}:{country}"
        cached = self._get_cached(_search_cache, cache_key, self.SEARCH_CACHE_TTL)
        if cached:
            return LiveDataResult(
                success=True,
                data=cached,
                formatted=cached.get("_formatted", ""),
                cache_ttl=self.SEARCH_CACHE_TTL,
            )

        response = self._client.get(
            f"{self.BASE_URL}/search",
            headers=self._get_headers(),
            params={"query": query, "country": country, "page": 1},
        )
        response.raise_for_status()
        data = response.json()

        formatted = self._format_search_results(data, query)
        data["_formatted"] = formatted
        self._set_cached(_search_cache, cache_key, data)

        return LiveDataResult(
            success=True,
            data=data,
            formatted=formatted,
            cache_ttl=self.SEARCH_CACHE_TTL,
        )

    def _fetch_best_sellers(self, category: str, country: str) -> LiveDataResult:
        """Fetch best sellers."""
        params = {"country": country}
        if category:
            params["category"] = category

        response = self._client.get(
            f"{self.BASE_URL}/best-sellers",
            headers=self._get_headers(),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        formatted = self._format_search_results(
            data, f"Best Sellers{f' - {category}' if category else ''}"
        )

        return LiveDataResult(
            success=True,
            data=data,
            formatted=formatted,
            cache_ttl=self.SEARCH_CACHE_TTL,
        )

    def _fetch_deals(self, country: str) -> LiveDataResult:
        """Fetch current deals."""
        response = self._client.get(
            f"{self.BASE_URL}/deals-v2",
            headers=self._get_headers(),
            params={"country": country, "page": 1},
        )
        response.raise_for_status()
        data = response.json()

        formatted = self._format_deals(data)

        return LiveDataResult(
            success=True,
            data=data,
            formatted=formatted,
            cache_ttl=self.DEALS_CACHE_TTL,
        )

    def _format_price(self, price_data: Any) -> str:
        """Format price data."""
        if isinstance(price_data, str):
            return price_data
        if isinstance(price_data, dict):
            return price_data.get("raw", price_data.get("value", "N/A"))
        if isinstance(price_data, (int, float)):
            return f"${price_data:.2f}"
        return "N/A"

    def _format_product_details(self, product: dict) -> str:
        """Format product details."""
        if not product:
            return "Product not found"

        lines = []
        title = product.get("product_title", "No title")
        price = self._format_price(product.get("product_price"))
        original_price = product.get("product_original_price")
        rating = product.get("product_star_rating", "")
        num_ratings = product.get("product_num_ratings", 0)
        asin = product.get("asin", "")
        availability = product.get("product_availability", "")
        # API returns about_product (list of bullet points)
        about_product = product.get("about_product", [])
        # product_description may be None or a string
        description = product.get("product_description")
        # product_information is a dict of specs
        product_info = product.get("product_information", {})

        lines.append(f"**{title}**")
        lines.append("")

        price_line = f"**Price:** {price}"
        if original_price and original_price != price:
            orig_formatted = self._format_price(original_price)
            if orig_formatted != price:
                price_line += f" ~~{orig_formatted}~~"
        lines.append(price_line)

        if rating:
            stars = "⭐" * int(float(rating))
            lines.append(f"**Rating:** {rating}/5 {stars} ({num_ratings:,} reviews)")

        if availability:
            lines.append(f"**Availability:** {availability}")

        lines.append(f"**ASIN:** {asin}")

        # Add product description if available
        if description:
            lines.append("")
            desc_text = (
                description if len(description) <= 300 else description[:300] + "..."
            )
            lines.append(f"**Description:** {desc_text}")

        # Add about product bullet points
        if about_product and isinstance(about_product, list):
            lines.append("\n**About this product:**")
            for item in about_product[:6]:
                if item and isinstance(item, str):
                    item_text = item if len(item) <= 150 else item[:150] + "..."
                    lines.append(f"- {item_text}")

        # Add key specifications from product_information
        if product_info and isinstance(product_info, dict):
            # Pick most useful specs
            useful_keys = [
                "Brand",
                "Manufacturer",
                "Model",
                "Item model number",
                "Product Dimensions",
                "Item Weight",
                "Screen Size",
                "Memory Storage Capacity",
                "RAM",
                "Processor",
                "Battery",
                "Connectivity",
                "Operating System",
            ]
            specs = []
            for key in useful_keys:
                if key in product_info:
                    val = product_info[key]
                    if val and isinstance(val, str) and len(val) < 100:
                        specs.append(f"- **{key}:** {val}")
            if specs:
                lines.append("\n**Specifications:**")
                lines.extend(specs[:6])

        return "\n".join(lines)

    def _format_reviews(self, reviews_data: dict) -> str:
        """Format product reviews."""
        lines = []
        lines.append("---")
        lines.append("**Customer Reviews:**")

        reviews = reviews_data.get("reviews", [])[:4]
        if not reviews:
            lines.append("_No reviews available_")
            return "\n".join(lines)

        for review in reviews:
            title = review.get("review_title", "")
            rating = review.get("review_star_rating", "")
            text = review.get("review_comment", "")
            verified = review.get("is_verified_purchase", False)

            stars = "⭐" * int(float(rating)) if rating else ""
            lines.append(f"\n**{title}** {stars}")
            if verified:
                lines.append("_(Verified Purchase)_")

            if text:
                if len(text) > 250:
                    text = text[:250] + "..."
                lines.append(text)

        return "\n".join(lines)

    def _format_search_results(self, data: dict, query: str) -> str:
        """Format search results."""
        lines = []
        lines.append(f"**Amazon: {query}**")
        lines.append("")

        products = data.get("data", {}).get("products", [])
        if not products:
            return f"No products found for: {query}"

        products = products[: self.max_results]

        for i, product in enumerate(products, 1):
            title = product.get("product_title", "No title")
            if len(title) > 70:
                title = title[:70] + "..."

            price = self._format_price(product.get("product_price"))
            rating = product.get("product_star_rating", "")
            num_ratings = product.get("product_num_ratings", 0)
            asin = product.get("asin", "")
            is_prime = product.get("is_prime", False)

            lines.append(f"**{i}. {title}**")
            price_line = f"   {price}"
            if is_prime:
                price_line += " (Prime)"
            lines.append(price_line)

            if rating:
                lines.append(f"   {rating}/5 ({num_ratings:,} reviews) | ASIN: {asin}")
            lines.append("")

        return "\n".join(lines)

    def _format_comparison(self, products: list) -> str:
        """Format product comparison."""
        lines = []
        lines.append("**Product Comparison**")
        lines.append("")

        for i, product in enumerate(products, 1):
            title = product.get("product_title", product.get("_search_name", "Product"))
            if len(title) > 60:
                title = title[:60] + "..."

            price = self._format_price(product.get("product_price"))
            rating = product.get("product_star_rating", "N/A")
            num_ratings = product.get("product_num_ratings", 0)

            lines.append(f"**{i}. {title}**")
            lines.append(f"   Price: {price}")
            lines.append(f"   Rating: {rating}/5 ({num_ratings:,} reviews)")
            lines.append("")

        return "\n".join(lines)

    def _format_deals(self, data: dict) -> str:
        """Format deals."""
        lines = []
        lines.append("**Amazon Deals**")
        lines.append("")

        deals = data.get("data", {}).get("deals", [])
        if not deals:
            return "No deals currently available"

        deals = deals[: self.max_results]

        for i, deal in enumerate(deals, 1):
            title = deal.get("deal_title", deal.get("product_title", "No title"))
            if len(title) > 70:
                title = title[:70] + "..."

            price = self._format_price(
                deal.get("deal_price", deal.get("product_price"))
            )
            discount = deal.get("savings_percentage", "")

            lines.append(f"**{i}. {title}**")
            price_line = f"   {price}"
            if discount:
                price_line += f" ({discount}% off)"
            lines.append(price_line)
            lines.append("")

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Amazon API."""
        if not self.api_key:
            return False, "RapidAPI key not configured"

        try:
            response = self._client.get(
                f"{self.BASE_URL}/search",
                headers=self._get_headers(),
                params={"query": "test", "country": "amazon.com", "page": 1},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "OK" or data.get("data"):
                count = len(data.get("data", {}).get("products", []))
                return True, f"Connected. Found {count} products."
            else:
                return False, "Unexpected API response"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
