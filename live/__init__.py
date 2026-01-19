"""
Live Data Sources for LLM Relay.

Provides real-time API data injection for Smart Aliases.
"""

from .sources import (
    LiveDataProvider,
    LiveDataResult,
    MCPProvider,
    StocksProvider,
    TransportProvider,
    WeatherProvider,
    fetch_live_data,
    get_provider_for_source,
)

__all__ = [
    "LiveDataProvider",
    "LiveDataResult",
    "MCPProvider",
    "StocksProvider",
    "WeatherProvider",
    "TransportProvider",
    "get_provider_for_source",
    "fetch_live_data",
]
