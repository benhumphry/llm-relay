"""
Smart Exchange Rates live source plugin.

Provides currency exchange rates using the Frankfurter API (free, no API key required).
Supports latest rates, historical rates, time series, and currency conversion.

API: https://frankfurter.dev/
Data source: European Central Bank reference rates
"""

import logging
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

# Global caches with TTL
_rates_cache: dict[str, tuple[Any, float]] = {}  # 1-hour TTL for rates
_currencies_cache: dict[str, tuple[Any, float]] = {}  # 24-hour TTL for currency list


class SmartExchangeLiveSource(PluginLiveSource):
    """
    Smart Exchange Rate Provider using Frankfurter API.

    Provides currency exchange rates from the European Central Bank.
    No API key required, free to use with no rate limits.

    Features:
    - Latest exchange rates (updated daily ~16:00 CET)
    - Historical rates (back to 1999)
    - Time series data for trends
    - Currency conversion calculations
    - Natural language date parsing

    Examples:
    - "EUR to USD rate"
    - "Convert 100 GBP to EUR"
    - "Dollar to pound exchange rate"
    - "How has EUR/USD changed this month"
    - "Exchange rate on 2024-01-15"
    """

    source_type = "frankfurter"
    display_name = "Exchange Rates (Frankfurter)"
    description = (
        "Currency exchange rates from European Central Bank via Frankfurter API"
    )
    category = "finance"
    data_type = "finance"
    best_for = "Currency exchange rates, forex, currency conversion. Supports EUR, USD, GBP, JPY, and 30+ currencies. Use for 'EUR to USD', 'convert 100 GBP to EUR', 'exchange rate history'."
    icon = "💱"
    default_cache_ttl = 3600  # 1 hour - rates update daily

    _abstract = False

    # API configuration
    BASE_URL = "https://api.frankfurter.dev/v1"

    # Cache TTLs
    RATES_CACHE_TTL = 3600  # 1 hour for current rates
    HISTORICAL_CACHE_TTL = 86400 * 30  # 30 days for historical (won't change)
    CURRENCIES_CACHE_TTL = 86400  # 24 hours for currency list

    # Common currency name mappings
    CURRENCY_ALIASES = {
        # Full names
        "euro": "EUR",
        "euros": "EUR",
        "dollar": "USD",
        "dollars": "USD",
        "us dollar": "USD",
        "us dollars": "USD",
        "american dollar": "USD",
        "pound": "GBP",
        "pounds": "GBP",
        "british pound": "GBP",
        "sterling": "GBP",
        "pound sterling": "GBP",
        "yen": "JPY",
        "japanese yen": "JPY",
        "yuan": "CNY",
        "chinese yuan": "CNY",
        "renminbi": "CNY",
        "rmb": "CNY",
        "swiss franc": "CHF",
        "franc": "CHF",
        "australian dollar": "AUD",
        "aussie dollar": "AUD",
        "canadian dollar": "CAD",
        "loonie": "CAD",
        "swedish krona": "SEK",
        "krona": "SEK",
        "norwegian krone": "NOK",
        "krone": "NOK",
        "danish krone": "DKK",
        "indian rupee": "INR",
        "rupee": "INR",
        "rupees": "INR",
        "singapore dollar": "SGD",
        "hong kong dollar": "HKD",
        "mexican peso": "MXN",
        "peso": "MXN",
        "brazilian real": "BRL",
        "real": "BRL",
        "south african rand": "ZAR",
        "rand": "ZAR",
        "korean won": "KRW",
        "won": "KRW",
        "turkish lira": "TRY",
        "lira": "TRY",
        "polish zloty": "PLN",
        "zloty": "PLN",
        "thai baht": "THB",
        "baht": "THB",
        "new zealand dollar": "NZD",
        "kiwi dollar": "NZD",
        "czech koruna": "CZK",
        "koruna": "CZK",
        "hungarian forint": "HUF",
        "forint": "HUF",
        "israeli shekel": "ILS",
        "shekel": "ILS",
        "romanian leu": "RON",
        "leu": "RON",
        "philippine peso": "PHP",
        "indonesian rupiah": "IDR",
        "rupiah": "IDR",
        "malaysian ringgit": "MYR",
        "ringgit": "MYR",
        "icelandic krona": "ISK",
        "bulgarian lev": "BGN",
        "lev": "BGN",
    }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields - minimal since Frankfurter API is free and keyless."""
        return [
            FieldDefinition(
                name="name",
                label="Source Name",
                field_type=FieldType.TEXT,
                required=True,
                default="Exchange Rates",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="default_base",
                label="Default Base Currency",
                field_type=FieldType.TEXT,
                required=False,
                default="EUR",
                help_text="Default base currency for conversions (e.g., EUR, USD, GBP)",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="from_currency",
                description="Base/source currency (code or name like 'EUR', 'dollar', 'pound')",
                param_type="string",
                required=False,
                examples=["EUR", "USD", "GBP", "dollar", "pound", "yen"],
            ),
            ParamDefinition(
                name="to_currency",
                description="Target currency or currencies (code or name, comma-separated for multiple)",
                param_type="string",
                required=False,
                examples=["USD", "GBP,JPY,CHF", "pound", "euro,yen"],
            ),
            ParamDefinition(
                name="amount",
                description="Amount to convert (optional, for conversion calculations)",
                param_type="number",
                required=False,
                examples=["100", "1000", "50.50"],
            ),
            ParamDefinition(
                name="date",
                description="Specific date for historical rates (YYYY-MM-DD or natural language)",
                param_type="string",
                required=False,
                examples=["2024-01-15", "yesterday", "last week", "2023-12-01"],
            ),
            ParamDefinition(
                name="query_type",
                description="Type of query: latest, historical, series, convert, currencies",
                param_type="string",
                required=False,
                default="latest",
                examples=["latest", "historical", "series", "convert", "currencies"],
            ),
            ParamDefinition(
                name="period",
                description="Time period for series data",
                param_type="string",
                required=False,
                examples=["1W", "1M", "3M", "6M", "1Y", "YTD"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize the exchange rate provider."""
        self.default_base = config.get("default_base", "EUR").upper()
        self._client = httpx.Client(timeout=15)
        self._currencies: dict[str, str] = {}  # Code -> Name mapping

    def _get_cached(self, cache: dict, key: str, ttl: int) -> Optional[Any]:
        """Get value from cache if not expired."""
        import time

        if key in cache:
            value, timestamp = cache[key]
            if time.time() - timestamp < ttl:
                return value
        return None

    def _set_cached(self, cache: dict, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        import time

        cache[key] = (value, time.time())

    def _resolve_currency(self, currency: str) -> Optional[str]:
        """Resolve currency name or alias to ISO code."""
        if not currency:
            return None

        currency_clean = currency.strip()

        # Already a valid 3-letter code?
        if len(currency_clean) == 3 and currency_clean.isalpha():
            return currency_clean.upper()

        # Check aliases
        currency_lower = currency_clean.lower()
        if currency_lower in self.CURRENCY_ALIASES:
            return self.CURRENCY_ALIASES[currency_lower]

        # Try to find partial match in currency names
        if not self._currencies:
            self._load_currencies()

        for code, name in self._currencies.items():
            if currency_lower in name.lower():
                return code

        return None

    def _load_currencies(self) -> None:
        """Load the list of supported currencies."""
        cache_key = "currencies"
        cached = self._get_cached(
            _currencies_cache, cache_key, self.CURRENCIES_CACHE_TTL
        )
        if cached:
            self._currencies = cached
            return

        try:
            response = self._client.get(f"{self.BASE_URL}/currencies")
            response.raise_for_status()
            self._currencies = response.json()
            self._set_cached(_currencies_cache, cache_key, self._currencies)
            logger.debug(f"Loaded {len(self._currencies)} currencies")
        except Exception as e:
            logger.warning(f"Failed to load currencies: {e}")
            # Use a minimal fallback
            self._currencies = {
                "EUR": "Euro",
                "USD": "US Dollar",
                "GBP": "British Pound",
                "JPY": "Japanese Yen",
                "CHF": "Swiss Franc",
                "AUD": "Australian Dollar",
                "CAD": "Canadian Dollar",
            }

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse natural language date to YYYY-MM-DD format."""
        if not date_str:
            return None

        date_lower = date_str.lower().strip()
        today = datetime.now()

        # Handle relative dates
        if date_lower in ("today", "now"):
            return today.strftime("%Y-%m-%d")
        elif date_lower == "yesterday":
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_lower == "last week":
            return (today - timedelta(days=7)).strftime("%Y-%m-%d")
        elif date_lower == "last month":
            return (today - timedelta(days=30)).strftime("%Y-%m-%d")
        elif date_lower == "last year":
            return (today - timedelta(days=365)).strftime("%Y-%m-%d")

        # Try parsing as YYYY-MM-DD
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            pass

        # Try other common formats
        for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    def _get_period_dates(self, period: str) -> tuple[str, str]:
        """Convert period string to start and end dates."""
        today = datetime.now()
        end_date = today.strftime("%Y-%m-%d")

        period_upper = period.upper().strip()

        if period_upper == "1W":
            start = today - timedelta(days=7)
        elif period_upper == "2W":
            start = today - timedelta(days=14)
        elif period_upper == "1M":
            start = today - timedelta(days=30)
        elif period_upper == "3M":
            start = today - timedelta(days=90)
        elif period_upper == "6M":
            start = today - timedelta(days=180)
        elif period_upper == "1Y":
            start = today - timedelta(days=365)
        elif period_upper == "YTD":
            start = datetime(today.year, 1, 1)
        elif period_upper == "5Y":
            start = today - timedelta(days=365 * 5)
        else:
            # Default to 1 month
            start = today - timedelta(days=30)

        return start.strftime("%Y-%m-%d"), end_date

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch exchange rate data."""
        try:
            query_type = params.get("query_type", "latest").lower()

            if query_type == "currencies":
                return self._fetch_currencies()
            elif query_type == "convert":
                return self._fetch_conversion(params)
            elif query_type == "series":
                return self._fetch_series(params)
            elif query_type == "historical":
                return self._fetch_historical(params)
            else:  # latest
                return self._fetch_latest(params)

        except httpx.HTTPStatusError as e:
            error_msg = f"API error: {e.response.status_code}"
            logger.error(f"Frankfurter API error: {e}")
            return LiveDataResult(success=False, error=error_msg)
        except Exception as e:
            logger.error(f"Exchange rate fetch failed: {e}")
            return LiveDataResult(success=False, error=str(e))

    def _fetch_currencies(self) -> LiveDataResult:
        """Fetch list of supported currencies."""
        self._load_currencies()

        lines = ["**Supported Currencies**", ""]
        for code, name in sorted(self._currencies.items()):
            lines.append(f"- **{code}**: {name}")

        return LiveDataResult(
            success=True,
            data=self._currencies,
            formatted="\n".join(lines),
            cache_ttl=self.CURRENCIES_CACHE_TTL,
        )

    def _fetch_latest(self, params: dict) -> LiveDataResult:
        """Fetch latest exchange rates."""
        from_currency = self._resolve_currency(
            params.get("from_currency") or self.default_base
        )
        to_currencies = params.get("to_currency")

        if not from_currency:
            from_currency = self.default_base

        # Build request params
        api_params = {"base": from_currency}
        if to_currencies:
            # Resolve each target currency
            targets = []
            for curr in to_currencies.split(","):
                resolved = self._resolve_currency(curr.strip())
                if resolved:
                    targets.append(resolved)
            if targets:
                api_params["symbols"] = ",".join(targets)

        # Check cache
        cache_key = f"latest:{from_currency}:{api_params.get('symbols', 'all')}"
        cached = self._get_cached(_rates_cache, cache_key, self.RATES_CACHE_TTL)
        if cached:
            return self._format_rates_response(cached, params)

        # Fetch from API
        response = self._client.get(f"{self.BASE_URL}/latest", params=api_params)
        response.raise_for_status()
        data = response.json()

        self._set_cached(_rates_cache, cache_key, data)
        return self._format_rates_response(data, params)

    def _fetch_historical(self, params: dict) -> LiveDataResult:
        """Fetch historical exchange rates for a specific date."""
        date_str = params.get("date")
        date = self._parse_date(date_str) if date_str else None

        if not date:
            return LiveDataResult(
                success=False,
                error="Please provide a valid date (e.g., 2024-01-15, yesterday)",
            )

        from_currency = self._resolve_currency(
            params.get("from_currency") or self.default_base
        )
        to_currencies = params.get("to_currency")

        if not from_currency:
            from_currency = self.default_base

        api_params = {"base": from_currency}
        if to_currencies:
            targets = []
            for curr in to_currencies.split(","):
                resolved = self._resolve_currency(curr.strip())
                if resolved:
                    targets.append(resolved)
            if targets:
                api_params["symbols"] = ",".join(targets)

        # Historical data is immutable, use long cache
        cache_key = (
            f"historical:{date}:{from_currency}:{api_params.get('symbols', 'all')}"
        )
        cached = self._get_cached(_rates_cache, cache_key, self.HISTORICAL_CACHE_TTL)
        if cached:
            return self._format_rates_response(cached, params, is_historical=True)

        response = self._client.get(f"{self.BASE_URL}/{date}", params=api_params)
        response.raise_for_status()
        data = response.json()

        self._set_cached(_rates_cache, cache_key, data)
        return self._format_rates_response(data, params, is_historical=True)

    def _fetch_series(self, params: dict) -> LiveDataResult:
        """Fetch time series exchange rate data."""
        period = params.get("period", "1M")
        start_date, end_date = self._get_period_dates(period)

        from_currency = self._resolve_currency(
            params.get("from_currency") or self.default_base
        )
        to_currencies = params.get("to_currency")

        if not from_currency:
            from_currency = self.default_base

        api_params = {"base": from_currency}

        # For series, we need specific target currencies
        targets = []
        if to_currencies:
            for curr in to_currencies.split(","):
                resolved = self._resolve_currency(curr.strip())
                if resolved:
                    targets.append(resolved)
        if not targets:
            # Default to USD if base is EUR, else EUR
            targets = ["USD"] if from_currency != "USD" else ["EUR"]

        api_params["symbols"] = ",".join(targets)

        # Fetch time series
        cache_key = (
            f"series:{start_date}:{end_date}:{from_currency}:{api_params['symbols']}"
        )
        cached = self._get_cached(_rates_cache, cache_key, self.RATES_CACHE_TTL)
        if cached:
            return self._format_series_response(cached, from_currency, targets, period)

        response = self._client.get(
            f"{self.BASE_URL}/{start_date}..{end_date}", params=api_params
        )
        response.raise_for_status()
        data = response.json()

        self._set_cached(_rates_cache, cache_key, data)
        return self._format_series_response(data, from_currency, targets, period)

    def _fetch_conversion(self, params: dict) -> LiveDataResult:
        """Calculate currency conversion."""
        amount = params.get("amount")
        if not amount:
            return LiveDataResult(
                success=False,
                error="Please provide an amount to convert",
            )

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return LiveDataResult(
                success=False,
                error=f"Invalid amount: {amount}",
            )

        from_currency = self._resolve_currency(
            params.get("from_currency") or self.default_base
        )
        to_currency_raw = params.get("to_currency")

        if not from_currency:
            from_currency = self.default_base

        if not to_currency_raw:
            return LiveDataResult(
                success=False,
                error="Please specify a target currency",
            )

        # Resolve target currencies
        targets = []
        for curr in to_currency_raw.split(","):
            resolved = self._resolve_currency(curr.strip())
            if resolved:
                targets.append(resolved)

        if not targets:
            return LiveDataResult(
                success=False,
                error=f"Could not resolve currency: {to_currency_raw}",
            )

        # Fetch latest rates
        api_params = {"base": from_currency, "symbols": ",".join(targets)}

        response = self._client.get(f"{self.BASE_URL}/latest", params=api_params)
        response.raise_for_status()
        data = response.json()

        # Calculate conversions
        rates = data.get("rates", {})
        lines = [
            f"**Currency Conversion** (as of {data.get('date', 'today')})",
            f"Amount: **{amount:,.2f} {from_currency}**",
            "",
        ]

        for target in targets:
            if target in rates:
                rate = rates[target]
                converted = amount * rate
                lines.append(f"- **{converted:,.2f} {target}** (rate: {rate:.6f})")

        return LiveDataResult(
            success=True,
            data={"amount": amount, "from": from_currency, "rates": rates},
            formatted="\n".join(lines),
            cache_ttl=self.RATES_CACHE_TTL,
        )

    def _format_rates_response(
        self, data: dict, params: dict, is_historical: bool = False
    ) -> LiveDataResult:
        """Format exchange rates response."""
        base = data.get("base", "EUR")
        date = data.get("date", "today")
        rates = data.get("rates", {})
        amount = params.get("amount")

        if is_historical:
            title = f"**Exchange Rates on {date}** (Base: {base})"
        else:
            title = f"**Current Exchange Rates** (Base: {base}, as of {date})"

        lines = [title, ""]

        # If amount provided, show conversions
        if amount:
            try:
                amt = float(amount)
                lines.append(f"Conversion of **{amt:,.2f} {base}**:")
                lines.append("")
                for currency, rate in sorted(rates.items()):
                    converted = amt * rate
                    lines.append(
                        f"- **{converted:,.2f} {currency}** (rate: {rate:.6f})"
                    )
            except (ValueError, TypeError):
                # Just show rates
                for currency, rate in sorted(rates.items()):
                    lines.append(f"- **{currency}**: {rate:.6f}")
        else:
            for currency, rate in sorted(rates.items()):
                lines.append(f"- **{currency}**: {rate:.6f}")

        return LiveDataResult(
            success=True,
            data=data,
            formatted="\n".join(lines),
            cache_ttl=self.HISTORICAL_CACHE_TTL
            if is_historical
            else self.RATES_CACHE_TTL,
        )

    def _format_series_response(
        self, data: dict, base: str, targets: list[str], period: str
    ) -> LiveDataResult:
        """Format time series response with trend analysis."""
        rates_by_date = data.get("rates", {})

        if not rates_by_date:
            return LiveDataResult(
                success=False,
                error="No data available for the requested period",
            )

        dates = sorted(rates_by_date.keys())
        start_date = dates[0]
        end_date = dates[-1]

        lines = [
            f"**{base} Exchange Rate Trend** ({period})",
            f"Period: {start_date} to {end_date}",
            "",
        ]

        for target in targets:
            # Get first and last rates
            first_rate = rates_by_date[start_date].get(target)
            last_rate = rates_by_date[end_date].get(target)

            if first_rate and last_rate:
                change = last_rate - first_rate
                pct_change = (change / first_rate) * 100

                # Calculate high/low
                all_rates = [
                    rates_by_date[d].get(target)
                    for d in dates
                    if rates_by_date[d].get(target)
                ]
                high = max(all_rates)
                low = min(all_rates)

                trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"

                lines.append(f"**{base}/{target}**")
                lines.append(f"- Current: {last_rate:.6f}")
                lines.append(f"- Change: {change:+.6f} ({pct_change:+.2f}%) {trend}")
                lines.append(f"- Period High: {high:.6f}")
                lines.append(f"- Period Low: {low:.6f}")
                lines.append("")

        return LiveDataResult(
            success=True,
            data=data,
            formatted="\n".join(lines),
            cache_ttl=self.RATES_CACHE_TTL,
        )

    def is_available(self) -> bool:
        """Always available - no API key required."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Frankfurter API."""
        try:
            response = self._client.get(f"{self.BASE_URL}/latest")
            response.raise_for_status()
            data = response.json()
            rates_count = len(data.get("rates", {}))
            return True, f"Connected. {rates_count} currencies available."
        except Exception as e:
            return False, str(e)
