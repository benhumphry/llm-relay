"""
Smart Finance live source plugin.

Unified stock market interface combining Finnhub (primary) and Alpha Vantage (UK/international).
Accepts company names, resolves to symbols internally,
provides quotes, company profiles, news, analyst recommendations,
earnings, historical data, and more.

Uses:
- Finnhub API (primary, 500 calls/min free tier) for US stocks
- Alpha Vantage API (fallback, 25 calls/day free tier) for UK/international stocks

Environment variables:
- FINNHUB_API_KEY - Required for US stocks
- ALPHA_VANTAGE_API_KEY - Optional, enables UK stocks like Tesco, Lloyds, BP
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
_symbol_cache: dict[
    str, tuple[Any, float]
] = {}  # 30-day TTL for company name -> symbol
_quote_cache: dict[str, tuple[Any, float]] = {}  # 1-min TTL for quotes
_profile_cache: dict[str, tuple[Any, float]] = {}  # 24-hour TTL for profiles
_news_cache: dict[str, tuple[Any, float]] = {}  # 15-min TTL for news


class SmartStocksLiveSource(PluginLiveSource):
    """
    Smart Finance Provider - unified Finnhub + Alpha Vantage.

    Unlike basic stock APIs requiring ticker symbols, this provider accepts
    company names and handles:
    - Company name -> Symbol resolution with caching (30-day cache)
    - Multiple query types: quote, profile, news, recommendations, earnings, history
    - Context-formatted output optimized for LLM consumption
    - Automatic fallback to Alpha Vantage for UK/international stocks

    Examples:
    - "Apple stock price" (US - Finnhub)
    - "Tesla news" (US - Finnhub)
    - "Tesco share price" (UK - Alpha Vantage)
    - "Lloyds bank stock" (UK - Alpha Vantage)
    - "What do analysts think about Nvidia?" (US - Finnhub)
    """

    source_type = "finnhub_enhanced"
    display_name = "Finnhub + Alpha Vantage"
    description = "Stock market data with US (Finnhub) and UK/international (Alpha Vantage) support"
    category = "finance"
    data_type = "finance"
    best_for = "Stock prices, company info, analyst recommendations, news, earnings. Supports US stocks (Apple, Tesla) and UK stocks (Tesco, Lloyds, BP)."
    icon = "📈"
    default_cache_ttl = 60  # 1 minute for quotes

    _abstract = False  # Allow registration

    # API configuration
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    # Cache TTLs
    SYMBOL_CACHE_TTL = 86400 * 30  # 30 days for symbol resolution
    QUOTE_CACHE_TTL = 60  # 1 minute for quotes
    PROFILE_CACHE_TTL = 86400  # 24 hours for company profiles
    NEWS_CACHE_TTL = 900  # 15 minutes for news
    RECOMMENDATIONS_CACHE_TTL = 3600  # 1 hour for recommendations
    EARNINGS_CACHE_TTL = 3600  # 1 hour for earnings

    # Common company name to symbol mappings
    # US stocks go to Finnhub, UK stocks (.LON suffix) go to Alpha Vantage
    KNOWN_SYMBOLS = {
        # Tech giants (US - Finnhub)
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        # Alphabet/Google: GOOGL = Class A (voting), GOOG = Class C (non-voting)
        # Default to GOOGL as it's more commonly traded
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "alphabet class a": "GOOGL",
        "alphabet class c": "GOOG",
        "googl": "GOOGL",
        "goog": "GOOG",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "netflix": "NFLX",
        # Semiconductors (US)
        "amd": "AMD",
        "intel": "INTC",
        "qualcomm": "QCOM",
        "broadcom": "AVGO",
        "micron": "MU",
        "arm": "ARM",
        # Software/Cloud (US)
        "salesforce": "CRM",
        "adobe": "ADBE",
        "oracle": "ORCL",
        "sap": "SAP",
        "servicenow": "NOW",
        "workday": "WDAY",
        "snowflake": "SNOW",
        "datadog": "DDOG",
        "crowdstrike": "CRWD",
        "palantir": "PLTR",
        "mongodb": "MDB",
        # Consumer tech (US)
        "spotify": "SPOT",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "doordash": "DASH",
        "pinterest": "PINS",
        "snap": "SNAP",
        "snapchat": "SNAP",
        "twitter": "X",
        "x": "X",
        "reddit": "RDDT",
        "roblox": "RBLX",
        "unity": "U",
        "unity software": "U",
        "unity technologies": "U",
        "roku": "ROKU",
        "zoom": "ZM",
        # Fintech (US)
        "paypal": "PYPL",
        "square": "SQ",
        "block": "SQ",
        "coinbase": "COIN",
        "robinhood": "HOOD",
        "affirm": "AFRM",
        "sofi": "SOFI",
        # E-commerce (US)
        "shopify": "SHOP",
        "etsy": "ETSY",
        "ebay": "EBAY",
        "mercadolibre": "MELI",
        # Banks (US)
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "citigroup": "C",
        "citi": "C",
        "goldman sachs": "GS",
        "morgan stanley": "MS",
        # Traditional (US)
        "walmart": "WMT",
        "target": "TGT",
        "costco": "COST",
        "home depot": "HD",
        "lowes": "LOW",
        "mcdonalds": "MCD",
        "starbucks": "SBUX",
        "nike": "NKE",
        "disney": "DIS",
        "coca cola": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "johnson & johnson": "JNJ",
        "j&j": "JNJ",
        "procter & gamble": "PG",
        "p&g": "PG",
        # Auto (US)
        "ford": "F",
        "gm": "GM",
        "general motors": "GM",
        "rivian": "RIVN",
        "lucid": "LCID",
        # Airlines (US)
        "delta": "DAL",
        "united": "UAL",
        "american airlines": "AAL",
        "southwest": "LUV",
        # Pharma/Healthcare (US)
        "pfizer": "PFE",
        "moderna": "MRNA",
        "eli lilly": "LLY",
        "merck": "MRK",
        "abbvie": "ABBV",
        "bristol myers": "BMY",
        "unitedhealth": "UNH",
        # Energy (US)
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "chevron": "CVX",
        # ETFs/Indices (US)
        "spy": "SPY",
        "qqq": "QQQ",
        "dia": "DIA",
        "iwm": "IWM",
        "voo": "VOO",
        "vti": "VTI",
        "arkk": "ARKK",
        # Crypto-adjacent (US)
        "microstrategy": "MSTR",
        # ============================================
        # UK stocks (Alpha Vantage - .LON suffix)
        # ============================================
        # Retail
        "tesco": "TSCO.LON",
        "sainsbury": "SBRY.LON",
        "sainsburys": "SBRY.LON",
        "marks and spencer": "MKS.LON",
        "marks & spencer": "MKS.LON",
        "m&s": "MKS.LON",
        "next": "NXT.LON",
        "jd sports": "JD.LON",
        "ocado": "OCDO.LON",
        # Banks (UK)
        "lloyds": "LLOY.LON",
        "lloyds bank": "LLOY.LON",
        "barclays": "BARC.LON",
        "hsbc": "HSBA.LON",
        "natwest": "NWG.LON",
        "standard chartered": "STAN.LON",
        # Insurance/Financial
        "legal & general": "LGEN.LON",
        "legal and general": "LGEN.LON",
        "l&g": "LGEN.LON",
        "aviva": "AV.LON",
        "prudential": "PRU.LON",
        # Energy (UK)
        "bp": "BP.LON",
        "shell": "SHEL.LON",
        "centrica": "CNA.LON",
        "british gas": "CNA.LON",
        "sse": "SSE.LON",
        "national grid": "NG.LON",
        # Telecom (UK)
        "vodafone": "VOD.LON",
        "bt": "BT-A.LON",
        "british telecom": "BT-A.LON",
        # Pharma (UK)
        "astrazeneca": "AZN.LON",
        "gsk": "GSK.LON",
        "glaxosmithkline": "GSK.LON",
        # Consumer goods (UK)
        "unilever": "ULVR.LON",
        "diageo": "DGE.LON",
        "reckitt": "RKT.LON",
        "reckitt benckiser": "RKT.LON",
        "british american tobacco": "BATS.LON",
        "bat": "BATS.LON",
        "imperial brands": "IMB.LON",
        # Mining/Materials (UK)
        "rio tinto": "RIO.LON",
        "anglo american": "AAL.LON",
        "glencore": "GLEN.LON",
        "bhp": "BHP.LON",
        "antofagasta": "ANTO.LON",
        # Real Estate (UK)
        "land securities": "LAND.LON",
        "british land": "BLND.LON",
        "segro": "SGRO.LON",
        # Other (UK)
        "rolls royce": "RR.LON",
        "bae systems": "BA.LON",
        "relx": "REL.LON",
        "compass group": "CPG.LON",
        "intertek": "ITRK.LON",
        "experian": "EXPN.LON",
        "smith & nephew": "SN.LON",
    }

    # Known UK funds that need web fallback (not available via stock APIs)
    # Maps search terms to fund display name for web search
    FUND_WEB_LOOKUP = {
        # L&G Global Technology
        "l&g global technology": "L&G Global Technology Index Fund",
        "legal & general global technology": "L&G Global Technology Index Fund",
        "legal and general global technology": "L&G Global Technology Index Fund",
        "l&g global tech": "L&G Global Technology Index Fund",
        "l&g technology": "L&G Global Technology Index Fund",
        "l&g tech": "L&G Global Technology Index Fund",
        "l&g global technology index": "L&G Global Technology Index Fund",
        "l&g global technology index fund": "L&G Global Technology Index Fund",
        "0p000023mw": "L&G Global Technology Index Fund",  # Morningstar ticker
        # Common fund patterns - add more as needed
        "vanguard lifestrategy": "Vanguard LifeStrategy",
        "vanguard ftse global": "Vanguard FTSE Global All Cap Index Fund",
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
                default="Smart Finance",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="finnhub_api_key",
                label="Finnhub API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="FINNHUB_API_KEY",
                help_text="For US stocks. Free tier: 500 calls/min. Leave empty to use env var.",
            ),
            FieldDefinition(
                name="alpha_vantage_api_key",
                label="Alpha Vantage API Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="ALPHA_VANTAGE_API_KEY",
                help_text="For UK/international stocks. Free tier: 25 calls/day. Leave empty to use env var.",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="company",
                description="Company name or stock symbol to look up (single company)",
                param_type="string",
                required=False,
                examples=["Apple", "AAPL", "Tesla", "Tesco", "Lloyds", "BP"],
            ),
            ParamDefinition(
                name="companies",
                description="Multiple company names or symbols for portfolio/comparison (comma-separated or list)",
                param_type="string",
                required=False,
                examples=["Apple, Microsoft, Google", "Tesco, Sainsbury, M&S"],
            ),
            ParamDefinition(
                name="query_type",
                description="Type of information needed",
                param_type="string",
                required=False,
                default="quote",
                examples=[
                    "quote",
                    "profile",
                    "news",
                    "recommendations",
                    "earnings",
                    "history",
                    "peers",
                    "financials",
                    "market_news",
                    "overview",
                    "portfolio",
                    "compare",
                ],
            ),
            ParamDefinition(
                name="period",
                description="Time period for historical data",
                param_type="string",
                required=False,
                default="1M",
                examples=["1D", "1W", "1M", "3M", "6M", "1Y", "YTD"],
            ),
        ]

    @classmethod
    def get_designator_hint(cls) -> str:
        """Provide guidance for the designator on how to use this source."""
        return (
            "Parameters: company (use simple company name like 'Apple', 'Tesla', 'Nvidia' - "
            "NOT 'Alphabet (GOOGL)' format); query_type (quote|profile|news|recommendations|earnings|history). "
            "For UK stocks use Alpha Vantage names like 'Tesco', 'Lloyds', 'BP'. "
            "For portfolio queries, use 'companies' param with comma-separated names."
        )

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-finance")
        self.finnhub_api_key = config.get("finnhub_api_key") or os.environ.get(
            "FINNHUB_API_KEY", ""
        )
        self.alpha_vantage_api_key = config.get(
            "alpha_vantage_api_key"
        ) or os.environ.get("ALPHA_VANTAGE_API_KEY", "")

        # Finnhub client (for US stocks)
        self._finnhub_client = httpx.Client(
            timeout=15,
            params={"token": self.finnhub_api_key},
        )

        # Alpha Vantage client (for UK/international stocks)
        self._alpha_vantage_client = httpx.Client(timeout=15)

    def _is_uk_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a UK stock (has .LON suffix)."""
        return symbol.upper().endswith(".LON")

    def _get_fund_name(self, query: str) -> Optional[str]:
        """Check if query matches a known fund that needs web lookup."""
        query_lower = query.lower().strip()
        for term, fund_name in self.FUND_WEB_LOOKUP.items():
            if term in query_lower:
                return fund_name
        return None

    def _fetch_fund_via_web(self, fund_name: str) -> Optional[dict]:
        """
        Fetch fund price via web search and scrape.

        Uses globally configured search provider and scraper from Settings.

        Returns dict with keys: price, currency, change_pct, fund_name
        """
        try:
            from augmentation import get_configured_search_provider
            from augmentation.scraper import JinaScraper, WebScraper
            from db import Setting, get_db_context

            # Get configured search provider
            searcher = get_configured_search_provider()
            if not searcher:
                logger.warning("No search provider configured for fund price lookup")
                return None

            # Get configured scraper provider
            scraper_provider = "builtin"
            try:
                with get_db_context() as db:
                    setting = (
                        db.query(Setting)
                        .filter(Setting.key == Setting.KEY_WEB_SCRAPER_PROVIDER)
                        .first()
                    )
                    if setting and setting.value:
                        scraper_provider = setting.value
            except Exception as e:
                logger.debug(f"Failed to get scraper provider setting: {e}")

            # Initialize appropriate scraper
            if scraper_provider == "jina":
                import os

                api_key = os.environ.get("JINA_API_KEY", "")
                if api_key:
                    scraper = JinaScraper(api_key=api_key)
                else:
                    logger.debug(
                        "Jina API key not set, falling back to builtin scraper"
                    )
                    scraper = WebScraper()
            else:
                scraper = WebScraper()

            # Search for fund price
            search_query = f"{fund_name} price GBP"
            logger.info(f"Searching for fund price: {search_query}")

            results = searcher.search(search_query, max_results=3)
            if not results:
                logger.warning(f"No search results for fund: {fund_name}")
                return None

            # Try scraping each result until we find a price
            for result in results:
                try:
                    scrape_result = scraper.scrape(result.url)
                    if not scrape_result.success or not scrape_result.content:
                        continue

                    # Try to extract price from scraped content
                    price_data = self._extract_fund_price(
                        scrape_result.content, fund_name
                    )

                    if price_data and price_data.get("price", 0) > 0:
                        price_data["fund_name"] = fund_name
                        logger.info(
                            f"Found fund price for {fund_name}: "
                            f"{price_data.get('currency', '£')}{price_data['price']:.2f}"
                        )
                        return price_data

                except Exception as e:
                    logger.debug(f"Error scraping {result.url}: {e}")
                    continue

            logger.warning(f"Could not extract price for fund: {fund_name}")
            return None

        except ImportError as e:
            logger.warning(f"Web search/scraping not available for fund lookup: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching fund price via web: {e}")
            return None

    def _extract_fund_price(self, content: str, fund_name: str) -> Optional[dict]:
        """Extract fund price from scraped web content."""
        # Common price patterns for UK funds
        patterns = [
            # Pence format: "123.45p" or "223.00 p"
            r"(?:price|nav|value|bid)[:\s]*(\d+(?:\.\d+)?)\s*p(?:ence)?",
            # Pound format with £ symbol
            r"[£]\s*(\d+(?:\.\d+)?)",
            # NAV specific
            r"nav[:\s]+[£]?(\d+(?:\.\d+)?)",
            # Bid price
            r"bid[:\s]+[£]?(\d+(?:\.\d+)?)\s*p?",
        ]

        content_lower = content.lower()

        for pattern in patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                try:
                    price = float(matches[0])
                    # If price is > 100, it's likely in pence
                    if price > 100:
                        price = price / 100  # Convert pence to pounds

                    # Sanity check - fund prices are typically 0.50 to 50
                    if 0.1 < price < 100:
                        return {
                            "price": price,
                            "currency": "GBP",
                            "change_pct": 0,
                        }
                except ValueError:
                    continue

        return None

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

    def _resolve_symbol(self, company: str) -> Optional[str]:
        """
        Resolve company name to stock symbol.

        Checks known mappings first, then searches via Finnhub API.
        Results cached for 30 days.
        """
        original = company.strip()
        company_lower = original.lower()

        # Handle annotated names like "Alphabet (GOOGL)" or "Apple - AAPL"
        # Extract the symbol from parentheses or after dash if present
        symbol_match = re.search(r"\(([A-Z]{1,5})\)|\s-\s([A-Z]{1,5})$", original)
        if symbol_match:
            extracted_symbol = symbol_match.group(1) or symbol_match.group(2)
            logger.debug(
                f"Extracted symbol from annotation: '{original}' -> {extracted_symbol}"
            )
            return extracted_symbol

        # Also extract and try just the company name part (before parentheses)
        name_only = re.sub(r"\s*\([^)]*\)\s*$", "", company_lower).strip()
        name_only = re.sub(
            r"\s*-\s*[A-Z]{1,5}\s*$", "", name_only, flags=re.IGNORECASE
        ).strip()

        # Check known mappings FIRST - handles cases like "UNITY" -> "U"
        # where the company name looks like a ticker but isn't
        if name_only in self.KNOWN_SYMBOLS:
            return self.KNOWN_SYMBOLS[name_only]
        if company_lower in self.KNOWN_SYMBOLS:
            return self.KNOWN_SYMBOLS[company_lower]

        # Check if it's already a valid symbol (not in known mappings)
        # US: all caps, 1-5 chars | UK: includes .LON suffix
        if original.isupper() and (
            (1 <= len(original) <= 5 and "." not in original)
            or original.endswith(".LON")
        ):
            return original

        # Check cache
        cache_key = f"symbol:{company_lower}"
        cached = self._get_cached(_symbol_cache, cache_key, self.SYMBOL_CACHE_TTL)
        if cached:
            logger.debug(f"Symbol cache hit: '{company}' -> {cached}")
            return cached

        # Search via Finnhub API (for US stocks)
        if self.finnhub_api_key:
            try:
                response = self._finnhub_client.get(
                    f"{self.FINNHUB_BASE_URL}/search",
                    params={"q": company},
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("result", [])
                if results:
                    # Filter for stocks (not funds, bonds, etc.) and prefer US exchanges
                    stocks = [r for r in results if r.get("type") == "Common Stock"]

                    if not stocks:
                        stocks = results  # Fall back to all results

                    # Prefer US exchanges (symbols without dots)
                    us_stocks = [r for r in stocks if r.get("symbol", "").isalpha()]

                    if us_stocks:
                        stocks = us_stocks

                    if stocks:
                        symbol = stocks[0].get("symbol")
                        if symbol:
                            self._set_cached(_symbol_cache, cache_key, symbol)
                            logger.info(f"Resolved '{company}' -> {symbol}")
                            return symbol

            except Exception as e:
                logger.warning(f"Finnhub symbol resolution failed for '{company}': {e}")

        # If no Finnhub result and Alpha Vantage is configured, try symbol search
        if self.alpha_vantage_api_key:
            try:
                response = self._alpha_vantage_client.get(
                    self.ALPHA_VANTAGE_BASE_URL,
                    params={
                        "function": "SYMBOL_SEARCH",
                        "keywords": company,
                        "apikey": self.alpha_vantage_api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

                matches = data.get("bestMatches", [])
                if matches:
                    # Prefer UK matches for UK-sounding companies
                    uk_matches = [
                        m for m in matches if ".LON" in m.get("1. symbol", "")
                    ]
                    if uk_matches:
                        symbol = uk_matches[0].get("1. symbol")
                    else:
                        symbol = matches[0].get("1. symbol")

                    if symbol:
                        self._set_cached(_symbol_cache, cache_key, symbol)
                        logger.info(
                            f"Resolved '{company}' -> {symbol} via Alpha Vantage"
                        )
                        return symbol

            except Exception as e:
                logger.warning(
                    f"Alpha Vantage symbol resolution failed for '{company}': {e}"
                )

        return None

    # ============================================
    # Finnhub API methods (US stocks)
    # ============================================

    def _get_finnhub_quote(self, symbol: str) -> Optional[dict]:
        """Get real-time quote from Finnhub for US stocks."""
        cache_key = f"quote:{symbol}"
        cached = self._get_cached(_quote_cache, cache_key, self.QUOTE_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/quote",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("c", 0) == 0:  # No current price
                return None

            self._set_cached(_quote_cache, cache_key, data)
            return data

        except Exception as e:
            logger.warning(f"Finnhub quote fetch failed for {symbol}: {e}")
            return None

    def _get_profile(self, symbol: str) -> Optional[dict]:
        """Get company profile from Finnhub."""
        cache_key = f"profile:{symbol}"
        cached = self._get_cached(_profile_cache, cache_key, self.PROFILE_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/profile2",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("name"):
                return None

            self._set_cached(_profile_cache, cache_key, data)
            return data

        except Exception as e:
            logger.warning(f"Profile fetch failed for {symbol}: {e}")
            return None

    def _get_peers(self, symbol: str) -> list[str]:
        """Get company peers/competitors from Finnhub."""
        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/peers",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            return response.json() or []

        except Exception as e:
            logger.debug(f"Peers fetch failed for {symbol}: {e}")
            return []

    def _get_recommendations(self, symbol: str) -> list[dict]:
        """Get analyst recommendations from Finnhub."""
        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/recommendation",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            return response.json() or []

        except Exception as e:
            logger.debug(f"Recommendations fetch failed for {symbol}: {e}")
            return []

    def _get_price_target(self, symbol: str) -> Optional[dict]:
        """Get analyst price targets from Finnhub."""
        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/price-target",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("targetHigh"):
                return data
            return None

        except Exception as e:
            logger.debug(f"Price target fetch failed for {symbol}: {e}")
            return None

    def _get_company_news(self, symbol: str, days: int = 7) -> list[dict]:
        """Get company-specific news from Finnhub."""
        cache_key = f"news:{symbol}:{days}"
        cached = self._get_cached(_news_cache, cache_key, self.NEWS_CACHE_TTL)
        if cached:
            return cached

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/company-news",
                params={
                    "symbol": symbol,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                },
            )
            response.raise_for_status()
            news = response.json() or []

            self._set_cached(_news_cache, cache_key, news)
            return news

        except Exception as e:
            logger.debug(f"News fetch failed for {symbol}: {e}")
            return []

    def _get_market_news(self, category: str = "general") -> list[dict]:
        """Get general market news from Finnhub."""
        cache_key = f"market_news:{category}"
        cached = self._get_cached(_news_cache, cache_key, self.NEWS_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/news",
                params={"category": category},
            )
            response.raise_for_status()
            news = response.json() or []

            self._set_cached(_news_cache, cache_key, news)
            return news

        except Exception as e:
            logger.debug(f"Market news fetch failed: {e}")
            return []

    def _get_earnings(self, symbol: str) -> list[dict]:
        """Get earnings surprises from Finnhub."""
        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/earnings",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            return response.json() or []

        except Exception as e:
            logger.debug(f"Earnings fetch failed for {symbol}: {e}")
            return []

    def _get_basic_financials(self, symbol: str) -> Optional[dict]:
        """Get basic financial metrics from Finnhub."""
        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/metric",
                params={"symbol": symbol, "metric": "all"},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("metric"):
                return data
            return None

        except Exception as e:
            logger.debug(f"Financials fetch failed for {symbol}: {e}")
            return None

    def _get_candles(self, symbol: str, period: str = "1M") -> Optional[dict]:
        """Get historical OHLCV data from Finnhub."""
        # Parse period to resolution and from/to dates
        now = datetime.now()
        resolution = "D"  # Daily by default

        period_upper = period.upper()
        if period_upper == "1D":
            from_date = now - timedelta(days=1)
            resolution = "5"  # 5-minute candles
        elif period_upper == "1W":
            from_date = now - timedelta(weeks=1)
            resolution = "60"  # Hourly
        elif period_upper == "1M":
            from_date = now - timedelta(days=30)
        elif period_upper == "3M":
            from_date = now - timedelta(days=90)
        elif period_upper == "6M":
            from_date = now - timedelta(days=180)
        elif period_upper == "1Y":
            from_date = now - timedelta(days=365)
        elif period_upper == "YTD":
            from_date = datetime(now.year, 1, 1)
        else:
            from_date = now - timedelta(days=30)

        try:
            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/stock/candle",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": int(from_date.timestamp()),
                    "to": int(now.timestamp()),
                },
            )
            response.raise_for_status()
            data = response.json()

            if data.get("s") == "no_data":
                return None

            return data

        except Exception as e:
            logger.debug(f"Candles fetch failed for {symbol}: {e}")
            return None

    def _get_earnings_calendar(self, days_ahead: int = 7) -> list[dict]:
        """Get upcoming earnings calendar from Finnhub."""
        try:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_ahead)

            response = self._finnhub_client.get(
                f"{self.FINNHUB_BASE_URL}/calendar/earnings",
                params={
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("earningsCalendar", [])

        except Exception as e:
            logger.debug(f"Earnings calendar fetch failed: {e}")
            return []

    # ============================================
    # Alpha Vantage API methods (UK/international stocks)
    # ============================================

    def _get_alpha_vantage_quote(self, symbol: str) -> Optional[dict]:
        """Get quote from Alpha Vantage for UK/international stocks."""
        if not self.alpha_vantage_api_key:
            return None

        cache_key = f"av_quote:{symbol}"
        cached = self._get_cached(_quote_cache, cache_key, self.QUOTE_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._alpha_vantage_client.get(
                self.ALPHA_VANTAGE_BASE_URL,
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Check for API limit message
            if "Note" in data or "Information" in data:
                logger.warning(
                    f"Alpha Vantage API limit: {data.get('Note') or data.get('Information')}"
                )
                return None

            quote = data.get("Global Quote", {})
            if not quote or "05. price" not in quote:
                return None

            # Normalize to Finnhub-like format for easier handling
            normalized = {
                "c": float(quote.get("05. price", 0)),  # current
                "d": float(quote.get("09. change", 0)),  # change
                "dp": float(
                    quote.get("10. change percent", "0%").replace("%", "")
                ),  # change %
                "h": float(quote.get("03. high", 0)),  # high
                "l": float(quote.get("04. low", 0)),  # low
                "o": float(quote.get("02. open", 0)),  # open
                "pc": float(quote.get("08. previous close", 0)),  # prev close
                "_source": "alpha_vantage",
                "_symbol": quote.get("01. symbol", symbol),
            }

            self._set_cached(_quote_cache, cache_key, normalized)
            return normalized

        except Exception as e:
            logger.warning(f"Alpha Vantage quote fetch failed for {symbol}: {e}")
            return None

    def _get_alpha_vantage_historical(
        self, symbol: str, period: str = "1M"
    ) -> Optional[dict]:
        """Get historical data from Alpha Vantage."""
        if not self.alpha_vantage_api_key:
            return None

        # Determine function and outputsize based on period
        period_upper = period.upper()
        if period_upper in ("1W", "1M", "3M", "6M", "1Y"):
            function = "TIME_SERIES_DAILY"
            outputsize = "compact" if period_upper in ("1W", "1M") else "full"
        else:  # YTD or other
            function = "TIME_SERIES_DAILY"
            outputsize = "full"

        try:
            response = self._alpha_vantage_client.get(
                self.ALPHA_VANTAGE_BASE_URL,
                params={
                    "function": function,
                    "symbol": symbol,
                    "outputsize": outputsize,
                    "apikey": self.alpha_vantage_api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Check for API limit
            if "Note" in data or "Information" in data:
                logger.warning(
                    f"Alpha Vantage API limit: {data.get('Note') or data.get('Information')}"
                )
                return None

            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                return None

            # Convert to Finnhub-like candle format
            dates = sorted(time_series.keys(), reverse=True)

            # Limit based on period
            days_limit = {
                "1W": 7,
                "1M": 30,
                "3M": 90,
                "6M": 180,
                "1Y": 365,
                "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
            }.get(period_upper, 30)

            dates = dates[:days_limit]
            dates.reverse()  # Oldest first

            closes = []
            highs = []
            lows = []
            opens = []
            volumes = []
            timestamps = []

            for date_str in dates:
                day_data = time_series[date_str]
                closes.append(float(day_data.get("4. close", 0)))
                highs.append(float(day_data.get("2. high", 0)))
                lows.append(float(day_data.get("3. low", 0)))
                opens.append(float(day_data.get("1. open", 0)))
                volumes.append(int(day_data.get("5. volume", 0)))
                timestamps.append(
                    int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
                )

            return {
                "c": closes,
                "h": highs,
                "l": lows,
                "o": opens,
                "v": volumes,
                "t": timestamps,
                "s": "ok",
                "_source": "alpha_vantage",
            }

        except Exception as e:
            logger.warning(f"Alpha Vantage historical fetch failed for {symbol}: {e}")
            return None

    # ============================================
    # Unified quote method with automatic fallback
    # ============================================

    def _get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get quote for a symbol, using appropriate API based on symbol type.

        UK stocks (.LON) use Alpha Vantage, US stocks use Finnhub.
        """
        if self._is_uk_symbol(symbol):
            # UK stock - use Alpha Vantage
            return self._get_alpha_vantage_quote(symbol)
        else:
            # US stock - use Finnhub (with Alpha Vantage fallback)
            quote = self._get_finnhub_quote(symbol)
            if quote:
                return quote
            # Fallback to Alpha Vantage if Finnhub fails
            return self._get_alpha_vantage_quote(symbol)

    def _parse_companies(self, companies_param: str) -> list[str]:
        """Parse companies parameter into a list of company names/symbols."""
        if not companies_param:
            return []

        # Handle both comma-separated and JSON array formats
        companies_param = companies_param.strip()

        # If it looks like a JSON array, parse it
        if companies_param.startswith("["):
            try:
                import json

                return [c.strip() for c in json.loads(companies_param) if c.strip()]
            except Exception:
                pass

        # Otherwise split by comma
        return [c.strip() for c in companies_param.split(",") if c.strip()]

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch stock data based on parameters.

        Intelligently routes to appropriate query based on context.
        """
        # Check for at least one API key
        if not self.finnhub_api_key and not self.alpha_vantage_api_key:
            return LiveDataResult(
                success=False,
                error="No API keys configured. Set FINNHUB_API_KEY and/or ALPHA_VANTAGE_API_KEY.",
            )

        company = params.get("company", "").strip()
        companies = params.get("companies", "").strip()
        query_type = params.get("query_type", "quote").lower()
        period = params.get("period", "1M")

        # Handle market-wide queries (no company needed)
        if query_type == "market_news":
            return self._fetch_market_news()

        if query_type == "earnings_calendar":
            return self._fetch_earnings_calendar()

        # Handle multi-company queries
        if companies or query_type in ["portfolio", "compare"]:
            company_list = self._parse_companies(companies)
            # Also add single company if specified
            if company and company not in company_list:
                company_list.insert(0, company)

            if not company_list:
                return LiveDataResult(
                    success=False,
                    error="Please specify companies for portfolio/comparison, e.g., 'Apple, Microsoft, Google'",
                )

            if query_type == "compare":
                return self._fetch_comparison(company_list)
            else:
                return self._fetch_portfolio(company_list)

        # Company-specific queries require a symbol
        if not company:
            return LiveDataResult(
                success=False,
                error="Please specify a company name or stock symbol, e.g., 'Apple' or 'AAPL' (US) or 'Tesco' or 'TSCO.LON' (UK)",
            )

        # Check if this is a fund that needs web lookup (not a stock)
        fund_name = self._get_fund_name(company)
        if fund_name:
            return self._fetch_fund_quote(fund_name)

        # Resolve company name to symbol
        symbol = self._resolve_symbol(company)
        if not symbol:
            return LiveDataResult(
                success=False,
                error=f"Could not find stock symbol for: {company}. Try the ticker symbol directly (e.g., AAPL for US, TSCO.LON for UK).",
            )

        # Route to appropriate handler
        if query_type == "quote" or query_type == "price":
            return self._fetch_quote(symbol, company)

        elif (
            query_type == "profile" or query_type == "company" or query_type == "about"
        ):
            return self._fetch_profile(symbol)

        elif query_type == "news":
            return self._fetch_company_news(symbol)

        elif (
            query_type == "recommendations"
            or query_type == "analysts"
            or query_type == "ratings"
        ):
            return self._fetch_recommendations(symbol)

        elif query_type == "earnings":
            return self._fetch_earnings(symbol)

        elif (
            query_type == "history"
            or query_type == "historical"
            or query_type == "chart"
        ):
            return self._fetch_history(symbol, period)

        elif query_type == "peers" or query_type == "competitors":
            return self._fetch_peers(symbol)

        elif query_type == "financials" or query_type == "metrics":
            return self._fetch_financials(symbol)

        elif query_type == "overview" or query_type == "summary":
            return self._fetch_overview(symbol)

        else:
            # Default to quote
            return self._fetch_quote(symbol, company)

    def _fetch_fund_quote(self, fund_name: str) -> LiveDataResult:
        """Fetch fund price via web lookup."""
        fund_data = self._fetch_fund_via_web(fund_name)

        if not fund_data:
            return LiveDataResult(
                success=False,
                error=f"Could not find price for fund: {fund_name}. Web search may not be configured or the fund data is not available online.",
            )

        price = fund_data.get("price", 0)
        currency = "£" if fund_data.get("currency") == "GBP" else "$"

        lines = [f"**{fund_name}**"]
        lines.append(f"**{currency}{price:.2f}**")
        lines.append("")
        lines.append(f"Currency: {fund_data.get('currency', 'GBP')}")
        lines.append("_Source: Web lookup_")
        lines.append(
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_"
        )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data=fund_data,
            cache_ttl=1800,  # 30 minutes for fund prices
        )

    def _fetch_quote(self, symbol: str, original_query: str = "") -> LiveDataResult:
        """Fetch stock quote."""
        quote = self._get_quote(symbol)
        if not quote:
            return LiveDataResult(
                success=False,
                error=f"No quote data available for {symbol}",
            )

        # Get profile for company name (only for US stocks via Finnhub)
        profile = None
        if not self._is_uk_symbol(symbol):
            profile = self._get_profile(symbol)
        company_name = profile.get("name", symbol) if profile else symbol

        # For UK stocks, clean up the symbol for display
        display_symbol = (
            symbol.replace(".LON", "") if self._is_uk_symbol(symbol) else symbol
        )

        current = quote.get("c", 0)
        change = quote.get("d", 0)
        change_pct = quote.get("dp", 0)
        high = quote.get("h", 0)
        low = quote.get("l", 0)
        open_price = quote.get("o", 0)
        prev_close = quote.get("pc", 0)

        direction = "📈" if change >= 0 else "📉"
        sign = "+" if change >= 0 else ""

        # Determine currency based on symbol
        currency = "£" if self._is_uk_symbol(symbol) else "$"
        currency_label = "GBP" if self._is_uk_symbol(symbol) else "USD"

        lines = [f"**{company_name} ({display_symbol})** {direction}"]
        lines.append(
            f"**{currency}{current:.2f}** ({sign}{currency}{change:.2f} / {sign}{change_pct:.2f}%)"
        )
        lines.append("")
        lines.append(f"Day Range: {currency}{low:.2f} - {currency}{high:.2f}")
        lines.append(
            f"Open: {currency}{open_price:.2f} | Previous Close: {currency}{prev_close:.2f}"
        )

        if self._is_uk_symbol(symbol):
            lines.append(
                f"Exchange: London Stock Exchange | Currency: {currency_label}"
            )
            lines.append("_Data via Alpha Vantage_")
        else:
            lines.append("_Data via Finnhub_")

        lines.append(
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_"
        )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "quote": quote, "profile": profile},
            cache_ttl=self.QUOTE_CACHE_TTL,
        )

    def _fetch_profile(self, symbol: str) -> LiveDataResult:
        """Fetch company profile."""
        # Profile only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            # For UK stocks, we can only provide basic quote data
            quote = self._get_quote(symbol)
            if not quote:
                return LiveDataResult(
                    success=False,
                    error=f"No data available for {symbol}",
                )

            display_symbol = symbol.replace(".LON", "")
            lines = [f"**{display_symbol}** (London Stock Exchange)"]
            lines.append("")
            lines.append(
                "_Detailed company profiles not available for UK stocks via Alpha Vantage._"
            )
            lines.append("_Use query_type=quote to get current price data._")

            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol},
                cache_ttl=self.PROFILE_CACHE_TTL,
            )

        profile = self._get_profile(symbol)
        if not profile:
            return LiveDataResult(
                success=False,
                error=f"No profile data available for {symbol}",
            )

        lines = [f"**{profile.get('name', symbol)} ({symbol})**"]

        if profile.get("finnhubIndustry"):
            lines.append(f"🏢 Industry: {profile['finnhubIndustry']}")

        if profile.get("exchange"):
            lines.append(f"📊 Exchange: {profile['exchange']}")

        if profile.get("marketCapitalization"):
            # Finnhub returns market cap in millions
            market_cap = profile["marketCapitalization"]
            if market_cap >= 1000000:
                market_cap_str = f"${market_cap / 1000000:.2f}T"
            elif market_cap >= 1000:
                market_cap_str = f"${market_cap / 1000:.1f}B"
            else:
                market_cap_str = f"${market_cap:.0f}M"
            lines.append(f"💰 Market Cap: {market_cap_str}")

        if profile.get("shareOutstanding"):
            shares = profile["shareOutstanding"]
            if shares >= 1000:
                shares_str = f"{shares / 1000:.2f}B"
            else:
                shares_str = f"{shares:.2f}M"
            lines.append(f"📈 Shares Outstanding: {shares_str}")

        if profile.get("country"):
            lines.append(f"🌍 Country: {profile['country']}")

        if profile.get("ipo"):
            lines.append(f"📅 IPO Date: {profile['ipo']}")

        if profile.get("weburl"):
            lines.append(f"🔗 {profile['weburl']}")

        # Add peers
        peers = self._get_peers(symbol)
        if peers:
            lines.append("")
            lines.append(f"**Competitors:** {', '.join(peers[:8])}")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "profile": profile, "peers": peers},
            cache_ttl=self.PROFILE_CACHE_TTL,
        )

    def _fetch_recommendations(self, symbol: str) -> LiveDataResult:
        """Fetch analyst recommendations."""
        # Only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            return LiveDataResult(
                success=False,
                error=f"Analyst recommendations not available for UK stocks. Try a US stock like Apple or Tesla.",
            )

        recommendations = self._get_recommendations(symbol)
        price_target = self._get_price_target(symbol)
        profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol

        lines = [f"**{company_name} ({symbol}) - Analyst Ratings**"]

        if recommendations:
            # Get most recent recommendation
            latest = recommendations[0]
            lines.append("")
            lines.append(f"**Latest ({latest.get('period', 'N/A')}):**")
            lines.append(f"  Strong Buy: {latest.get('strongBuy', 0)}")
            lines.append(f"  Buy: {latest.get('buy', 0)}")
            lines.append(f"  Hold: {latest.get('hold', 0)}")
            lines.append(f"  Sell: {latest.get('sell', 0)}")
            lines.append(f"  Strong Sell: {latest.get('strongSell', 0)}")

            # Calculate consensus
            total = (
                latest.get("strongBuy", 0)
                + latest.get("buy", 0)
                + latest.get("hold", 0)
                + latest.get("sell", 0)
                + latest.get("strongSell", 0)
            )
            if total > 0:
                buy_pct = (
                    (latest.get("strongBuy", 0) + latest.get("buy", 0)) / total * 100
                )
                if buy_pct >= 70:
                    consensus = "Strong Buy"
                elif buy_pct >= 50:
                    consensus = "Buy"
                elif buy_pct >= 30:
                    consensus = "Hold"
                else:
                    consensus = "Sell"
                lines.append(f"  **Consensus: {consensus}** ({total} analysts)")

        if price_target:
            lines.append("")
            lines.append("**Price Targets:**")
            lines.append(f"  High: ${price_target.get('targetHigh', 0):.2f}")
            lines.append(f"  Mean: ${price_target.get('targetMean', 0):.2f}")
            lines.append(f"  Low: ${price_target.get('targetLow', 0):.2f}")
            lines.append(f"  Median: ${price_target.get('targetMedian', 0):.2f}")

            # Compare to current price
            quote = self._get_quote(symbol)
            if quote:
                current = quote.get("c", 0)
                target_mean = price_target.get("targetMean", 0)
                if current > 0 and target_mean > 0:
                    upside = ((target_mean - current) / current) * 100
                    direction = "upside" if upside >= 0 else "downside"
                    lines.append(
                        f"  _Current: ${current:.2f} ({abs(upside):.1f}% {direction} to mean)_"
                    )

        if not recommendations and not price_target:
            lines.append("")
            lines.append("No analyst data available for this stock.")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={
                "symbol": symbol,
                "recommendations": recommendations,
                "price_target": price_target,
            },
            cache_ttl=self.RECOMMENDATIONS_CACHE_TTL,
        )

    def _fetch_company_news(self, symbol: str) -> LiveDataResult:
        """Fetch company news."""
        # Only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            return LiveDataResult(
                success=False,
                error=f"Company news not available for UK stocks. Try a US stock like Apple or Tesla.",
            )

        news = self._get_company_news(symbol, days=7)
        profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol

        lines = [f"**{company_name} ({symbol}) - Recent News**"]

        if not news:
            lines.append("")
            lines.append("No recent news found.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol, "news": []},
            )

        lines.append("")
        for article in news[:10]:  # Top 10 articles
            headline = article.get("headline", "")
            source = article.get("source", "")
            ts = article.get("datetime", 0)

            date_str = ""
            if ts:
                dt = datetime.fromtimestamp(ts)
                date_str = dt.strftime("%b %d")

            lines.append(f"- **{headline}**")
            if source or date_str:
                lines.append(f"  _{source} | {date_str}_")
            lines.append("")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "news": news[:10]},
            cache_ttl=self.NEWS_CACHE_TTL,
        )

    def _fetch_market_news(self) -> LiveDataResult:
        """Fetch general market news."""
        if not self.finnhub_api_key:
            return LiveDataResult(
                success=False,
                error="Market news requires Finnhub API key.",
            )

        news = self._get_market_news("general")

        lines = ["**Market News**"]

        if not news:
            lines.append("")
            lines.append("No recent market news found.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"news": []},
            )

        lines.append("")
        for article in news[:15]:  # Top 15 articles
            headline = article.get("headline", "")
            source = article.get("source", "")
            ts = article.get("datetime", 0)

            date_str = ""
            if ts:
                dt = datetime.fromtimestamp(ts)
                date_str = dt.strftime("%b %d %H:%M")

            lines.append(f"- **{headline}**")
            if source or date_str:
                lines.append(f"  _{source} | {date_str}_")
            lines.append("")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"news": news[:15]},
            cache_ttl=self.NEWS_CACHE_TTL,
        )

    def _fetch_earnings(self, symbol: str) -> LiveDataResult:
        """Fetch earnings data."""
        # Only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            return LiveDataResult(
                success=False,
                error=f"Earnings data not available for UK stocks. Try a US stock like Apple or Tesla.",
            )

        earnings = self._get_earnings(symbol)
        profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol

        lines = [f"**{company_name} ({symbol}) - Earnings**"]

        if not earnings:
            lines.append("")
            lines.append("No earnings data available.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol, "earnings": []},
            )

        lines.append("")
        lines.append("**Recent Quarters:**")
        for q in earnings[:4]:  # Last 4 quarters
            period = q.get("period", "")
            actual = q.get("actual")
            estimate = q.get("estimate")
            surprise = q.get("surprise")
            surprise_pct = q.get("surprisePercent")

            line = f"- **{period}**: "
            if actual is not None:
                line += f"EPS ${actual:.2f}"
                if estimate is not None:
                    line += f" (Est: ${estimate:.2f})"
                if surprise is not None and surprise_pct is not None:
                    beat_miss = "Beat" if surprise >= 0 else "Miss"
                    line += f" - {beat_miss} by {abs(surprise_pct):.1f}%"
            else:
                line += "Pending"

            lines.append(line)

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "earnings": earnings},
            cache_ttl=self.EARNINGS_CACHE_TTL,
        )

    def _fetch_earnings_calendar(self) -> LiveDataResult:
        """Fetch upcoming earnings calendar."""
        if not self.finnhub_api_key:
            return LiveDataResult(
                success=False,
                error="Earnings calendar requires Finnhub API key.",
            )

        calendar = self._get_earnings_calendar(days_ahead=7)

        lines = ["**Upcoming Earnings (Next 7 Days)**"]

        if not calendar:
            lines.append("")
            lines.append("No upcoming earnings announcements found.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"calendar": []},
            )

        # Group by date
        by_date: dict[str, list] = {}
        for entry in calendar[:50]:  # Limit to 50
            date = entry.get("date", "Unknown")
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(entry)

        for date, entries in sorted(by_date.items()):
            lines.append("")
            lines.append(f"**{date}:**")
            for entry in entries[:10]:  # Max 10 per day
                symbol = entry.get("symbol", "?")
                eps_est = entry.get("epsEstimate")
                hour = entry.get("hour", "")

                line = f"  {symbol}"
                if eps_est is not None:
                    line += f" (Est: ${eps_est:.2f})"
                if hour:
                    line += f" [{hour}]"
                lines.append(line)

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"calendar": calendar[:50]},
            cache_ttl=self.EARNINGS_CACHE_TTL,
        )

    def _fetch_history(self, symbol: str, period: str) -> LiveDataResult:
        """Fetch historical price data."""
        # Use appropriate API based on symbol type
        if self._is_uk_symbol(symbol):
            candles = self._get_alpha_vantage_historical(symbol, period)
            currency = "£"
        else:
            candles = self._get_candles(symbol, period)
            # Fallback to Alpha Vantage if Finnhub fails
            if not candles and self.alpha_vantage_api_key:
                candles = self._get_alpha_vantage_historical(symbol, period)
            currency = "$"

        profile = None
        if not self._is_uk_symbol(symbol):
            profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol
        display_symbol = (
            symbol.replace(".LON", "") if self._is_uk_symbol(symbol) else symbol
        )

        lines = [f"**{company_name} ({display_symbol}) - {period} History**"]

        if not candles or candles.get("s") == "no_data":
            lines.append("")
            lines.append("No historical data available for this period.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol},
            )

        closes = candles.get("c", [])
        highs = candles.get("h", [])
        lows = candles.get("l", [])
        timestamps = candles.get("t", [])

        if not closes:
            lines.append("")
            lines.append("No historical data available.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol},
            )

        # Calculate summary stats
        start_price = closes[0]
        end_price = closes[-1]
        period_change = end_price - start_price
        period_change_pct = (period_change / start_price) * 100 if start_price else 0
        period_high = max(highs) if highs else end_price
        period_low = min(lows) if lows else end_price

        direction = "📈" if period_change >= 0 else "📉"
        sign = "+" if period_change >= 0 else ""

        lines.append("")
        lines.append(
            f"**Performance:** {direction} {sign}{currency}{period_change:.2f} ({sign}{period_change_pct:.2f}%)"
        )
        lines.append(
            f"Start: {currency}{start_price:.2f} → End: {currency}{end_price:.2f}"
        )
        lines.append(
            f"Period High: {currency}{period_high:.2f} | Period Low: {currency}{period_low:.2f}"
        )

        # Add some recent data points
        if len(closes) > 5:
            lines.append("")
            lines.append("**Recent Closes:**")
            for i in range(-5, 0):
                if timestamps and len(timestamps) > abs(i):
                    ts = timestamps[i]
                    dt = datetime.fromtimestamp(ts)
                    date_str = dt.strftime("%b %d")
                else:
                    date_str = f"Day {len(closes) + i + 1}"
                lines.append(f"  {date_str}: {currency}{closes[i]:.2f}")

        if self._is_uk_symbol(symbol):
            lines.append("")
            lines.append("_Data via Alpha Vantage_")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "candles": candles},
            cache_ttl=300,  # 5 minutes for historical
        )

    def _fetch_peers(self, symbol: str) -> LiveDataResult:
        """Fetch company peers/competitors."""
        # Only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            return LiveDataResult(
                success=False,
                error=f"Peer comparison not available for UK stocks. Try a US stock like Apple or Tesla.",
            )

        peers = self._get_peers(symbol)
        profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol

        lines = [f"**{company_name} ({symbol}) - Competitors**"]

        if not peers:
            lines.append("")
            lines.append("No peer data available.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol, "peers": []},
            )

        lines.append("")
        lines.append(f"**Similar Companies:** {', '.join(peers)}")

        # Get quotes for top peers for comparison
        lines.append("")
        lines.append("**Peer Comparison:**")
        for peer_symbol in peers[:5]:
            peer_quote = self._get_quote(peer_symbol)
            if peer_quote:
                price = peer_quote.get("c", 0)
                change_pct = peer_quote.get("dp", 0)
                sign = "+" if change_pct >= 0 else ""
                lines.append(f"  {peer_symbol}: ${price:.2f} ({sign}{change_pct:.2f}%)")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "peers": peers},
            cache_ttl=self.PROFILE_CACHE_TTL,
        )

    def _fetch_financials(self, symbol: str) -> LiveDataResult:
        """Fetch basic financial metrics."""
        # Only available via Finnhub for US stocks
        if self._is_uk_symbol(symbol):
            return LiveDataResult(
                success=False,
                error=f"Financial metrics not available for UK stocks. Try a US stock like Apple or Tesla.",
            )

        financials = self._get_basic_financials(symbol)
        profile = self._get_profile(symbol)

        company_name = profile.get("name", symbol) if profile else symbol

        lines = [f"**{company_name} ({symbol}) - Key Metrics**"]

        if not financials or not financials.get("metric"):
            lines.append("")
            lines.append("No financial metrics available.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"symbol": symbol},
            )

        metrics = financials["metric"]
        lines.append("")

        # Valuation metrics
        valuation_metrics = [
            ("peBasicExclExtraTTM", "P/E Ratio"),
            ("psTTM", "P/S Ratio"),
            ("pbQuarterly", "P/B Ratio"),
            ("evToRevenue", "EV/Revenue"),
        ]
        lines.append("**Valuation:**")
        for key, label in valuation_metrics:
            if metrics.get(key):
                lines.append(f"  {label}: {metrics[key]:.2f}")

        # Growth metrics
        lines.append("")
        lines.append("**Growth & Margins:**")
        growth_metrics = [
            ("revenueGrowthTTMYoy", "Revenue Growth (YoY)"),
            ("epsGrowthTTMYoy", "EPS Growth (YoY)"),
            ("grossMarginTTM", "Gross Margin"),
            ("operatingMarginTTM", "Operating Margin"),
            ("netProfitMarginTTM", "Net Profit Margin"),
        ]
        for key, label in growth_metrics:
            if metrics.get(key):
                val = metrics[key]
                if "Growth" in label or "Margin" in label:
                    lines.append(f"  {label}: {val:.1f}%")
                else:
                    lines.append(f"  {label}: {val:.2f}")

        # Other metrics
        lines.append("")
        lines.append("**Other:**")
        other_metrics = [
            ("dividendYieldIndicatedAnnual", "Dividend Yield"),
            ("roeTTM", "ROE"),
            ("roaTTM", "ROA"),
            ("currentRatioQuarterly", "Current Ratio"),
            ("52WeekHigh", "52-Week High"),
            ("52WeekLow", "52-Week Low"),
        ]
        for key, label in other_metrics:
            if metrics.get(key):
                val = metrics[key]
                if "Yield" in label or "ROE" in label or "ROA" in label:
                    lines.append(f"  {label}: {val:.2f}%")
                elif "Week" in label:
                    lines.append(f"  {label}: ${val:.2f}")
                else:
                    lines.append(f"  {label}: {val:.2f}")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"symbol": symbol, "metrics": metrics},
            cache_ttl=self.PROFILE_CACHE_TTL,
        )

    def _fetch_portfolio(self, companies: list[str]) -> LiveDataResult:
        """Fetch quotes for multiple companies as a portfolio summary."""
        results = []
        errors = []
        total_data = []

        for company in companies[:20]:  # Limit to 20 companies
            symbol = self._resolve_symbol(company)
            if not symbol:
                errors.append(company)
                continue

            quote = self._get_quote(symbol)
            if not quote:
                errors.append(company)
                continue

            # Get profile only for US stocks
            profile = None
            if not self._is_uk_symbol(symbol):
                profile = self._get_profile(symbol)
            company_name = (
                profile.get("name", symbol) if profile else symbol.replace(".LON", "")
            )

            current = quote.get("c", 0)
            change = quote.get("d", 0)
            change_pct = quote.get("dp", 0)
            currency = "£" if self._is_uk_symbol(symbol) else "$"

            results.append(
                {
                    "symbol": symbol,
                    "name": company_name,
                    "price": current,
                    "change": change,
                    "change_pct": change_pct,
                    "currency": currency,
                }
            )

            total_data.append(
                {
                    "symbol": symbol,
                    "quote": quote,
                    "profile": profile,
                }
            )

        if not results:
            return LiveDataResult(
                success=False,
                error=f"Could not fetch any quotes. Failed: {', '.join(errors)}",
            )

        # Sort by absolute change percentage (most volatile first)
        results.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

        lines = ["**Portfolio Summary**"]
        lines.append(
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_"
        )
        lines.append("")

        # Summary stats
        gainers = [r for r in results if r["change_pct"] > 0]
        losers = [r for r in results if r["change_pct"] < 0]
        unchanged = [r for r in results if r["change_pct"] == 0]

        lines.append(
            f"📊 {len(results)} stocks: {len(gainers)} 📈 up, {len(losers)} 📉 down, {len(unchanged)} unchanged"
        )
        lines.append("")

        # Individual stocks
        for r in results:
            direction = "📈" if r["change_pct"] >= 0 else "📉"
            sign = "+" if r["change_pct"] >= 0 else ""
            display_symbol = (
                r["symbol"].replace(".LON", "")
                if r["symbol"].endswith(".LON")
                else r["symbol"]
            )
            lines.append(
                f"{direction} **{display_symbol}** ({r['name'][:20]}): "
                f"{r['currency']}{r['price']:.2f} ({sign}{r['change_pct']:.2f}%)"
            )

        if errors:
            lines.append("")
            lines.append(f"_Could not fetch: {', '.join(errors)}_")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"stocks": total_data, "errors": errors},
            cache_ttl=self.QUOTE_CACHE_TTL,
        )

    def _fetch_comparison(self, companies: list[str]) -> LiveDataResult:
        """Fetch detailed comparison of multiple companies."""
        results = []
        errors = []

        for company in companies[:10]:  # Limit to 10 for detailed comparison
            symbol = self._resolve_symbol(company)
            if not symbol:
                errors.append(company)
                continue

            quote = self._get_quote(symbol)
            if not quote:
                errors.append(company)
                continue

            # Get profile and financials only for US stocks
            profile = None
            financials = None
            if not self._is_uk_symbol(symbol):
                profile = self._get_profile(symbol)
                financials = self._get_basic_financials(symbol)

            company_name = (
                profile.get("name", symbol) if profile else symbol.replace(".LON", "")
            )
            metrics = financials.get("metric", {}) if financials else {}
            currency = "£" if self._is_uk_symbol(symbol) else "$"

            results.append(
                {
                    "symbol": symbol,
                    "name": company_name,
                    "price": quote.get("c", 0),
                    "change_pct": quote.get("dp", 0),
                    "market_cap": profile.get("marketCapitalization", 0)
                    if profile
                    else 0,
                    "pe_ratio": metrics.get("peBasicExclExtraTTM"),
                    "ps_ratio": metrics.get("psTTM"),
                    "gross_margin": metrics.get("grossMarginTTM"),
                    "revenue_growth": metrics.get("revenueGrowthTTMYoy"),
                    "industry": profile.get("finnhubIndustry", "") if profile else "",
                    "currency": currency,
                }
            )

        if not results:
            return LiveDataResult(
                success=False,
                error=f"Could not fetch data for comparison. Failed: {', '.join(errors)}",
            )

        lines = ["**Stock Comparison**"]
        lines.append(
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_"
        )
        lines.append("")

        # Price comparison
        lines.append("**Current Prices:**")
        for r in results:
            direction = "📈" if r["change_pct"] >= 0 else "📉"
            sign = "+" if r["change_pct"] >= 0 else ""
            display_symbol = (
                r["symbol"].replace(".LON", "")
                if r["symbol"].endswith(".LON")
                else r["symbol"]
            )
            lines.append(
                f"  {direction} {display_symbol}: {r['currency']}{r['price']:.2f} ({sign}{r['change_pct']:.2f}%)"
            )

        # Market cap comparison (US stocks only)
        us_results = [r for r in results if r["market_cap"] > 0]
        if us_results:
            lines.append("")
            lines.append("**Market Cap (US stocks only):**")
            results_by_mcap = sorted(
                us_results, key=lambda x: x["market_cap"], reverse=True
            )
            for r in results_by_mcap:
                mc = r["market_cap"]
                if mc >= 1000000:
                    mc_str = f"${mc / 1000000:.2f}T"
                elif mc >= 1000:
                    mc_str = f"${mc / 1000:.1f}B"
                elif mc > 0:
                    mc_str = f"${mc:.0f}M"
                else:
                    mc_str = "N/A"
                lines.append(f"  {r['symbol']}: {mc_str}")

        # Valuation comparison (US stocks only)
        has_pe = any(r["pe_ratio"] for r in results)
        if has_pe:
            lines.append("")
            lines.append("**Valuation (P/E Ratio, US stocks only):**")
            results_by_pe = sorted(
                [r for r in results if r["pe_ratio"]], key=lambda x: x["pe_ratio"]
            )
            for r in results_by_pe:
                lines.append(f"  {r['symbol']}: {r['pe_ratio']:.1f}")

        # Growth comparison (US stocks only)
        has_growth = any(r["revenue_growth"] for r in results)
        if has_growth:
            lines.append("")
            lines.append("**Revenue Growth (YoY, US stocks only):**")
            results_by_growth = sorted(
                [r for r in results if r["revenue_growth"]],
                key=lambda x: x["revenue_growth"],
                reverse=True,
            )
            for r in results_by_growth:
                sign = "+" if r["revenue_growth"] >= 0 else ""
                lines.append(f"  {r['symbol']}: {sign}{r['revenue_growth']:.1f}%")

        # Margin comparison (US stocks only)
        has_margin = any(r["gross_margin"] for r in results)
        if has_margin:
            lines.append("")
            lines.append("**Gross Margin (US stocks only):**")
            results_by_margin = sorted(
                [r for r in results if r["gross_margin"]],
                key=lambda x: x["gross_margin"],
                reverse=True,
            )
            for r in results_by_margin:
                lines.append(f"  {r['symbol']}: {r['gross_margin']:.1f}%")

        if errors:
            lines.append("")
            lines.append(f"_Could not fetch: {', '.join(errors)}_")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"comparison": results, "errors": errors},
            cache_ttl=300,  # 5 minutes for comparison
        )

    def _fetch_overview(self, symbol: str) -> LiveDataResult:
        """Fetch comprehensive overview combining multiple data points."""
        quote = self._get_quote(symbol)

        if not quote:
            return LiveDataResult(
                success=False,
                error=f"No data available for {symbol}",
            )

        # Get profile only for US stocks
        profile = None
        recommendations = None
        if not self._is_uk_symbol(symbol):
            profile = self._get_profile(symbol)
            recommendations = self._get_recommendations(symbol)

        company_name = (
            profile.get("name", symbol) if profile else symbol.replace(".LON", "")
        )
        display_symbol = (
            symbol.replace(".LON", "") if self._is_uk_symbol(symbol) else symbol
        )
        currency = "£" if self._is_uk_symbol(symbol) else "$"

        # Quote info
        current = quote.get("c", 0)
        change = quote.get("d", 0)
        change_pct = quote.get("dp", 0)
        direction = "📈" if change >= 0 else "📉"
        sign = "+" if change >= 0 else ""

        lines = [f"**{company_name} ({display_symbol})** {direction}"]
        lines.append(
            f"**{currency}{current:.2f}** ({sign}{currency}{change:.2f} / {sign}{change_pct:.2f}%)"
        )
        lines.append("")

        # Profile info (US only)
        if profile:
            if profile.get("finnhubIndustry"):
                lines.append(f"🏢 {profile['finnhubIndustry']}")
            if profile.get("marketCapitalization"):
                mc = profile["marketCapitalization"]  # In millions
                if mc >= 1000000:
                    mc_str = f"${mc / 1000000:.2f}T"
                elif mc >= 1000:
                    mc_str = f"${mc / 1000:.1f}B"
                else:
                    mc_str = f"${mc:.0f}M"
                lines.append(f"💰 Market Cap: {mc_str}")

        # Analyst summary (US only)
        if recommendations:
            latest = recommendations[0]
            total = (
                latest.get("strongBuy", 0)
                + latest.get("buy", 0)
                + latest.get("hold", 0)
                + latest.get("sell", 0)
                + latest.get("strongSell", 0)
            )
            if total > 0:
                buy_pct = (
                    (latest.get("strongBuy", 0) + latest.get("buy", 0)) / total * 100
                )
                if buy_pct >= 70:
                    consensus = "Strong Buy"
                elif buy_pct >= 50:
                    consensus = "Buy"
                elif buy_pct >= 30:
                    consensus = "Hold"
                else:
                    consensus = "Sell"
                lines.append(f"📊 Analyst Consensus: {consensus} ({total} analysts)")

        # Recent news headline (US only)
        if not self._is_uk_symbol(symbol):
            news = self._get_company_news(symbol, days=3)
            if news:
                lines.append("")
                lines.append(f"📰 Latest: {news[0].get('headline', '')[:80]}...")

        # UK stock note
        if self._is_uk_symbol(symbol):
            lines.append("")
            lines.append("_London Stock Exchange | Data via Alpha Vantage_")
            lines.append(
                "_Note: Detailed profile, news, and analyst data not available for UK stocks._"
            )

        lines.append("")
        lines.append(
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_"
        )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={
                "symbol": symbol,
                "quote": quote,
                "profile": profile,
                "recommendations": recommendations,
                "news": self._get_company_news(symbol, days=3)[:3]
                if not self._is_uk_symbol(symbol)
                else [],
            },
            cache_ttl=self.QUOTE_CACHE_TTL,
        )

    def is_available(self) -> bool:
        """Check if at least one API key is configured."""
        return bool(self.finnhub_api_key or self.alpha_vantage_api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection with sample queries."""
        results = []

        # Test Finnhub (US stocks)
        if self.finnhub_api_key:
            try:
                result = self._fetch_quote("AAPL", "Apple")
                if result.success:
                    results.append("✓ Finnhub: AAPL quote OK")
                else:
                    results.append(f"✗ Finnhub: {result.error}")
            except Exception as e:
                results.append(f"✗ Finnhub: {e}")
        else:
            results.append("- Finnhub: Not configured (US stocks disabled)")

        # Test Alpha Vantage (UK stocks)
        if self.alpha_vantage_api_key:
            try:
                quote = self._get_alpha_vantage_quote("TSCO.LON")
                if quote:
                    results.append("✓ Alpha Vantage: TSCO.LON quote OK")
                else:
                    results.append("✗ Alpha Vantage: No quote data for TSCO.LON")
            except Exception as e:
                results.append(f"✗ Alpha Vantage: {e}")
        else:
            results.append("- Alpha Vantage: Not configured (UK stocks disabled)")

        # Determine overall success
        has_finnhub = (
            self.finnhub_api_key and "✓ Finnhub" in results[0]
            if self.finnhub_api_key
            else False
        )
        has_alpha = (
            self.alpha_vantage_api_key
            and "✓ Alpha Vantage" in (results[1] if len(results) > 1 else "")
            if self.alpha_vantage_api_key
            else False
        )

        if has_finnhub or has_alpha:
            return True, "\n".join(results)
        elif not self.finnhub_api_key and not self.alpha_vantage_api_key:
            return False, "No API keys configured"
        else:
            return False, "\n".join(results)
