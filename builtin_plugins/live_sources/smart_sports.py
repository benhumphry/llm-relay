"""
Smart Sports live source plugin.

High-level multi-sport interface with natural language support.
Accepts team names in natural language, resolves to IDs internally,
provides match fixtures, results, standings, and live scores.

Uses SportAPI7 via RapidAPI.

Supports: Football (Soccer), Basketball (NBA), Ice Hockey (NHL),
American Football (NFL), Tennis, and more.
"""

import logging
import os
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

# Global caches with TTL
_team_cache: dict[str, tuple[Any, float]] = {}  # 90-day TTL for team name -> ID
_standings_cache: dict[str, tuple[Any, float]] = {}  # 30-min TTL for standings
_tournament_cache: dict[str, tuple[Any, float]] = {}  # 24-hour TTL for tournament info


class SmartSportsLiveSource(PluginLiveSource):
    """
    Smart Sports Provider - accepts natural language team names.

    Unlike basic sports APIs requiring team IDs, this provider accepts
    team names and handles:
    - Team name → ID resolution with caching (90-day cache)
    - Auto-detection of sport from team name
    - Multiple query types: next match, recent results, standings, live scores
    - Context-formatted output optimized for LLM consumption

    Examples:
    - "Arsenal next match"
    - "Lakers score" (checks live, then recent)
    - "Premier League standings"
    - "Is there any football on now?"
    """

    source_type = "sportapi7_enhanced"
    display_name = "SportAPI7 (Enhanced)"
    description = "Multi-sport data via SportAPI7 with natural language team names"
    category = "sports"
    data_type = "sports"
    best_for = "Live scores, upcoming fixtures, recent results, league standings for football, basketball, hockey, and more. Use team names like 'Arsenal', 'Lakers', 'Patriots'."
    icon = "⚽"
    default_cache_ttl = 60  # 1 minute for live data

    _abstract = False  # Allow registration

    # API configuration
    BASE_URL = "https://sportapi7.p.rapidapi.com/api/v1"

    # Cache TTLs
    TEAM_CACHE_TTL = 86400 * 90  # 90 days for team resolution
    STANDINGS_CACHE_TTL = 1800  # 30 minutes for standings
    TOURNAMENT_CACHE_TTL = 86400  # 24 hours for tournament info
    LIVE_CACHE_TTL = 60  # 1 minute for live data

    # Sport ID mapping
    SPORT_IDS = {
        "football": 1,
        "soccer": 1,
        "basketball": 2,
        "ice-hockey": 4,
        "hockey": 4,
        "tennis": 5,
        "american-football": 63,
        "nfl": 63,
        "baseball": 64,
        "mlb": 64,
        "rugby": 12,
        "cricket": 62,
        "motorsport": 11,
        "f1": 11,
    }

    # Common tournament IDs for quick access
    TOURNAMENT_IDS = {
        # English Football
        "premier league": 17,
        "epl": 17,
        "championship": 18,
        "efl championship": 18,
        "english championship": 18,
        "league one": 19,
        "efl league one": 19,
        "league two": 20,
        "efl league two": 20,
        "fa cup": 29,
        "efl cup": 21,
        "league cup": 21,
        "carabao cup": 21,
        # European Top Leagues
        "la liga": 8,
        "serie a": 23,
        "bundesliga": 35,
        "ligue 1": 34,
        "eredivisie": 37,
        # European Competitions
        "champions league": 7,
        "ucl": 7,
        "europa league": 679,
        "conference league": 17015,
        # Scottish Football
        "scottish premiership": 36,
        "spfl": 36,
        # International
        "world cup": 16,
        "euros": 1,
        "euro 2024": 1,
        "european championship": 1,
        # US Sports
        "nba": 132,
        "nhl": 234,
        "nfl": 9464,
        "mlb": 11205,
        "mls": 242,
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
                default="Smart Sports",
                help_text="Display name for this source",
            ),
            FieldDefinition(
                name="api_key",
                label="RapidAPI Key",
                field_type=FieldType.PASSWORD,
                required=False,
                env_var="RAPIDAPI_KEY",
                help_text="Leave empty to use RAPIDAPI_KEY env var",
            ),
            FieldDefinition(
                name="default_sport",
                label="Default Sport",
                field_type=FieldType.SELECT,
                required=False,
                default="football",
                options=[
                    {"value": "football", "label": "Football (Soccer)"},
                    {"value": "basketball", "label": "Basketball"},
                    {"value": "ice-hockey", "label": "Ice Hockey"},
                    {"value": "american-football", "label": "American Football (NFL)"},
                    {"value": "tennis", "label": "Tennis"},
                    {"value": "baseball", "label": "Baseball"},
                ],
                help_text="Default sport when not specified",
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide at query time."""
        return [
            ParamDefinition(
                name="team",
                description="Team name to look up",
                param_type="string",
                required=False,
                examples=["Arsenal", "Lakers", "Patriots", "Real Madrid"],
            ),
            ParamDefinition(
                name="player",
                description="Player name to look up",
                param_type="string",
                required=False,
                examples=["Mbappe", "Haaland", "LeBron James", "Patrick Mahomes"],
            ),
            ParamDefinition(
                name="league",
                description="League/tournament name (helps disambiguation)",
                param_type="string",
                required=False,
                examples=["Premier League", "NBA", "NFL", "Champions League"],
            ),
            ParamDefinition(
                name="opponent",
                description="Opponent team for head-to-head or specific fixture queries",
                param_type="string",
                required=False,
                examples=["Wrexham", "Manchester United", "Real Madrid"],
            ),
            ParamDefinition(
                name="query_type",
                description="Type of information needed",
                param_type="string",
                required=False,
                default="auto",
                examples=[
                    "next_match",
                    "recent_results",
                    "standings",
                    "live_scores",
                    "score",
                    "squad",
                    "player",
                    "transfers",
                    "fixture",
                    "h2h",
                    "auto",
                ],
            ),
            ParamDefinition(
                name="sport",
                description="Sport type if not clear from team/league",
                param_type="string",
                required=False,
                examples=["football", "basketball", "hockey", "nfl"],
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.name = config.get("name", "smart-sports")
        # Fall back to env vars: SPORTAPI7_API_KEY or generic RAPIDAPI_KEY
        self.api_key = (
            config.get("api_key")
            or os.environ.get("SPORTAPI7_API_KEY")
            or os.environ.get("RAPIDAPI_KEY", "")
        )
        self.default_sport = config.get("default_sport", "football")

        self._client = httpx.Client(
            timeout=15,
            headers={
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "sportapi7.p.rapidapi.com",
            },
        )

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

    def _resolve_team(self, team_name: str, sport_hint: str = None) -> Optional[dict]:
        """
        Resolve team name to team data including ID.

        Returns dict with: id, name, shortName, sport, country, tournament info
        Results cached for 90 days.
        """
        cache_key = f"team:{team_name.lower().strip()}"
        if sport_hint:
            cache_key += f":{sport_hint}"

        # Check cache
        cached = self._get_cached(_team_cache, cache_key, self.TEAM_CACHE_TTL)
        if cached:
            logger.debug(f"Team cache hit: '{team_name}' -> {cached.get('id')}")
            return cached

        try:
            response = self._client.get(f"{self.BASE_URL}/search/teams/{team_name}")
            response.raise_for_status()
            data = response.json()

            teams = data.get("teams", [])
            if not teams:
                return None

            # Filter by sport if hint provided
            if sport_hint:
                sport_id = self.SPORT_IDS.get(sport_hint.lower())
                if sport_id:
                    sport_filtered = [
                        t for t in teams if t.get("sport", {}).get("id") == sport_id
                    ]
                    if sport_filtered:
                        teams = sport_filtered

            # Prefer men's teams and higher user count
            teams.sort(
                key=lambda t: (
                    t.get("gender", "M") == "M",  # Men's teams first
                    t.get("userCount", 0),  # Higher popularity
                ),
                reverse=True,
            )

            team = teams[0]
            team_data = {
                "id": team.get("id"),
                "name": team.get("name"),
                "shortName": team.get("shortName", team.get("name")),
                "nameCode": team.get("nameCode"),
                "sport": team.get("sport", {}),
                "country": team.get("country", {}),
                "gender": team.get("gender", "M"),
            }

            # Cache the result
            self._set_cached(_team_cache, cache_key, team_data)
            logger.info(
                f"Resolved team '{team_name}' -> {team_data['name']} (ID: {team_data['id']})"
            )
            return team_data

        except Exception as e:
            logger.warning(f"Team resolution failed for '{team_name}': {e}")
            return None

    def _resolve_tournament(self, league_name: str) -> Optional[int]:
        """Resolve league name to tournament ID."""
        league_lower = league_name.lower().strip()

        # Check known tournaments first
        if league_lower in self.TOURNAMENT_IDS:
            return self.TOURNAMENT_IDS[league_lower]

        # TODO: Add API search for tournaments if needed
        return None

    def _get_live_events(self, sport: str = None) -> list[dict]:
        """Get currently live events, optionally filtered by sport."""
        sport_slug = sport or self.default_sport
        if sport_slug in ["soccer"]:
            sport_slug = "football"

        cache_key = f"live:{sport_slug}"
        cached = self._get_cached(_standings_cache, cache_key, self.LIVE_CACHE_TTL)
        if cached:
            return cached

        try:
            response = self._client.get(
                f"{self.BASE_URL}/sport/{sport_slug}/events/live"
            )
            response.raise_for_status()
            data = response.json()
            events = data.get("events", [])

            self._set_cached(_standings_cache, cache_key, events)
            return events

        except Exception as e:
            logger.warning(f"Failed to fetch live events: {e}")
            return []

    def _get_team_live_match(self, team_id: int) -> Optional[dict]:
        """Check if a team is currently playing."""
        try:
            # Get team's current/recent events
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}/events/last/0")
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            for event in events:
                status = event.get("status", {})
                if status.get("type") == "inprogress":
                    return event

            return None

        except Exception as e:
            logger.warning(f"Failed to check live match for team {team_id}: {e}")
            return None

    def _get_team_next_match(self, team_id: int) -> Optional[dict]:
        """Get team's next scheduled match."""
        try:
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}/events/next/0")
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            return events[0] if events else None

        except Exception as e:
            logger.warning(f"Failed to fetch next match for team {team_id}: {e}")
            return None

    def _get_team_recent_results(self, team_id: int, limit: int = 5) -> list[dict]:
        """Get team's recent match results."""
        try:
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}/events/last/0")
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            # Filter to finished matches only
            finished = [
                e for e in events if e.get("status", {}).get("type") == "finished"
            ]
            return finished[:limit]

        except Exception as e:
            logger.warning(f"Failed to fetch recent results for team {team_id}: {e}")
            return []

    def _get_tournament_standings(
        self, tournament_id: int, season_id: int = None
    ) -> Optional[dict]:
        """Get league standings."""
        cache_key = f"standings:{tournament_id}:{season_id or 'current'}"
        cached = self._get_cached(_standings_cache, cache_key, self.STANDINGS_CACHE_TTL)
        if cached:
            return cached

        try:
            # First get current season if not provided
            if not season_id:
                response = self._client.get(
                    f"{self.BASE_URL}/unique-tournament/{tournament_id}/seasons"
                )
                response.raise_for_status()
                seasons = response.json().get("seasons", [])
                if seasons:
                    season_id = seasons[0].get("id")
                else:
                    return None

            # Get standings
            response = self._client.get(
                f"{self.BASE_URL}/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
            )
            response.raise_for_status()
            data = response.json()

            self._set_cached(_standings_cache, cache_key, data)
            return data

        except Exception as e:
            logger.warning(
                f"Failed to fetch standings for tournament {tournament_id}: {e}"
            )
            return None

    # ==================== COMPREHENSIVE EVENT DATA ====================

    def _get_event_details(self, event_id: int) -> Optional[dict]:
        """Get full event details including venue, referee, etc."""
        try:
            response = self._client.get(f"{self.BASE_URL}/event/{event_id}")
            response.raise_for_status()
            return response.json().get("event")
        except Exception as e:
            logger.warning(f"Failed to fetch event details for {event_id}: {e}")
            return None

    def _get_event_lineups(self, event_id: int) -> Optional[dict]:
        """Get team lineups for an event."""
        try:
            response = self._client.get(f"{self.BASE_URL}/event/{event_id}/lineups")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No lineups for event {event_id}: {e}")
            return None

    def _get_event_incidents(self, event_id: int) -> list[dict]:
        """Get match incidents (goals, cards, substitutions)."""
        try:
            response = self._client.get(f"{self.BASE_URL}/event/{event_id}/incidents")
            response.raise_for_status()
            return response.json().get("incidents", [])
        except Exception as e:
            logger.debug(f"No incidents for event {event_id}: {e}")
            return []

    def _get_event_statistics(self, event_id: int) -> Optional[dict]:
        """Get match statistics (possession, shots, etc.)."""
        try:
            response = self._client.get(f"{self.BASE_URL}/event/{event_id}/statistics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No statistics for event {event_id}: {e}")
            return None

    def _get_event_h2h(self, event_id: int) -> Optional[dict]:
        """Get head-to-head history for the teams."""
        try:
            response = self._client.get(f"{self.BASE_URL}/event/{event_id}/h2h")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No H2H data for event {event_id}: {e}")
            return None

    def _get_event_best_players(self, event_id: int) -> Optional[dict]:
        """Get best players/ratings for the match."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/event/{event_id}/best-players"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No best players for event {event_id}: {e}")
            return None

    def _get_event_pregame_form(self, event_id: int) -> Optional[dict]:
        """Get pre-game form for both teams."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/event/{event_id}/pregame-form"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No pregame form for event {event_id}: {e}")
            return None

    def _get_player_details(self, player_id: int) -> Optional[dict]:
        """Get detailed player information."""
        try:
            response = self._client.get(f"{self.BASE_URL}/player/{player_id}")
            response.raise_for_status()
            return response.json().get("player")
        except Exception as e:
            logger.warning(f"Failed to fetch player {player_id}: {e}")
            return None

    def _get_player_statistics(self, player_id: int) -> Optional[dict]:
        """Get player's current season statistics."""
        try:
            # Get player's statistics for all competitions
            response = self._client.get(
                f"{self.BASE_URL}/player/{player_id}/statistics/season/last"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No statistics for player {player_id}: {e}")
            return None

    def _get_player_transfer_history(self, player_id: int) -> list[dict]:
        """Get player's transfer history."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/player/{player_id}/transfer-history"
            )
            response.raise_for_status()
            return response.json().get("transferHistory", [])
        except Exception as e:
            logger.debug(f"No transfer history for player {player_id}: {e}")
            return []

    def _get_player_national_stats(self, player_id: int) -> Optional[dict]:
        """Get player's national team statistics."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/player/{player_id}/national-team-statistics"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No national stats for player {player_id}: {e}")
            return None

    def _get_team_details(self, team_id: int) -> Optional[dict]:
        """Get detailed team information."""
        try:
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}")
            response.raise_for_status()
            return response.json().get("team")
        except Exception as e:
            logger.warning(f"Failed to fetch team {team_id}: {e}")
            return None

    def _get_team_players(self, team_id: int) -> list[dict]:
        """Get team's current squad."""
        try:
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}/players")
            response.raise_for_status()
            return response.json().get("players", [])
        except Exception as e:
            logger.debug(f"No players for team {team_id}: {e}")
            return []

    def _get_team_transfers(self, team_id: int) -> Optional[dict]:
        """Get team's recent transfers."""
        try:
            response = self._client.get(f"{self.BASE_URL}/team/{team_id}/transfers")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No transfers for team {team_id}: {e}")
            return None

    def _get_team_statistics(
        self, team_id: int, tournament_id: int, season_id: int
    ) -> Optional[dict]:
        """Get team's season statistics for a tournament."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/team/{team_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"No team stats for team {team_id}: {e}")
            return None

    def _get_h2h_events(self, team1_id: int, team2_id: int) -> list[dict]:
        """Get head-to-head history between two teams."""
        try:
            # Get team1's recent and upcoming events, filter for matches vs team2
            events = []

            # Recent events
            response = self._client.get(
                f"{self.BASE_URL}/team/{team1_id}/events/last/0"
            )
            response.raise_for_status()
            events.extend(response.json().get("events", []))

            # Upcoming events
            response = self._client.get(
                f"{self.BASE_URL}/team/{team1_id}/events/next/0"
            )
            response.raise_for_status()
            events.extend(response.json().get("events", []))

            # Filter for matches against team2
            h2h_events = []
            for event in events:
                home_id = event.get("homeTeam", {}).get("id")
                away_id = event.get("awayTeam", {}).get("id")
                if home_id == team2_id or away_id == team2_id:
                    h2h_events.append(event)

            return h2h_events

        except Exception as e:
            logger.warning(
                f"Failed to get H2H events for {team1_id} vs {team2_id}: {e}"
            )
            return []

    def _find_fixture_between_teams(
        self, team1_id: int, team2_id: int
    ) -> Optional[dict]:
        """Find the next scheduled fixture between two teams."""
        try:
            # Get team1's upcoming events
            response = self._client.get(
                f"{self.BASE_URL}/team/{team1_id}/events/next/0"
            )
            response.raise_for_status()
            events = response.json().get("events", [])

            for event in events:
                home_id = event.get("homeTeam", {}).get("id")
                away_id = event.get("awayTeam", {}).get("id")
                if home_id == team2_id or away_id == team2_id:
                    return event

            return None

        except Exception as e:
            logger.warning(
                f"Failed to find fixture between {team1_id} and {team2_id}: {e}"
            )
            return None

    def _find_live_or_recent_match_between_teams(
        self, team1_id: int, team2_id: int
    ) -> Optional[dict]:
        """Find a live or recent match between two teams."""
        try:
            # Check if team1 is currently playing team2
            live_match = self._get_team_live_match(team1_id)
            if live_match:
                home_id = live_match.get("homeTeam", {}).get("id")
                away_id = live_match.get("awayTeam", {}).get("id")
                if home_id == team2_id or away_id == team2_id:
                    return live_match

            # Check recent matches
            response = self._client.get(
                f"{self.BASE_URL}/team/{team1_id}/events/last/0"
            )
            response.raise_for_status()
            events = response.json().get("events", [])

            for event in events:
                home_id = event.get("homeTeam", {}).get("id")
                away_id = event.get("awayTeam", {}).get("id")
                if home_id == team2_id or away_id == team2_id:
                    return event

            return None

        except Exception as e:
            logger.warning(
                f"Failed to find match between {team1_id} and {team2_id}: {e}"
            )
            return None

    def _search_player(self, player_name: str) -> Optional[dict]:
        """Search for a player by name."""
        try:
            response = self._client.get(f"{self.BASE_URL}/search/players/{player_name}")
            response.raise_for_status()
            players = response.json().get("players", [])
            if players:
                # Sort by user count to get most relevant
                players.sort(key=lambda p: p.get("userCount", 0), reverse=True)
                return players[0]
            return None
        except Exception as e:
            logger.warning(f"Player search failed for '{player_name}': {e}")
            return None

    def _format_player_info(
        self,
        player: dict,
        stats: Optional[dict] = None,
        transfers: list = None,
        national_stats: Optional[dict] = None,
    ) -> str:
        """Format comprehensive player information for LLM context."""
        lines = []

        # Basic info
        name = player.get("name", "Unknown")
        short_name = player.get("shortName", name)
        lines.append(f"**{name}**")

        # Position and nationality
        position = player.get("position", "")
        nationality = player.get("country", {}).get("name", "")
        if position or nationality:
            info_parts = []
            if position:
                info_parts.append(position)
            if nationality:
                info_parts.append(f"🌍 {nationality}")
            lines.append(" | ".join(info_parts))

        # Current team
        team = player.get("team", {})
        if team.get("name"):
            lines.append(f"🏟️ {team['name']}")

        # Age and DOB
        dob = player.get("dateOfBirthTimestamp")
        if dob:
            birth_date = datetime.fromtimestamp(dob)
            age = (datetime.now() - birth_date).days // 365
            lines.append(f"📅 Born: {birth_date.strftime('%d %B %Y')} (Age: {age})")

        # Physical attributes
        height = player.get("height")
        if height:
            lines.append(f"📏 Height: {height} cm")

        # Preferred foot
        foot = player.get("preferredFoot")
        if foot:
            lines.append(f"🦶 Preferred foot: {foot}")

        # Contract info
        contract_until = player.get("contractUntilTimestamp")
        if contract_until:
            contract_date = datetime.fromtimestamp(contract_until)
            lines.append(f"📝 Contract until: {contract_date.strftime('%B %Y')}")

        # Market value
        market_value = player.get("proposedMarketValue")
        if market_value:
            if market_value >= 1000000:
                value_str = f"€{market_value / 1000000:.1f}M"
            else:
                value_str = f"€{market_value / 1000:.0f}K"
            lines.append(f"💰 Market value: {value_str}")

        # Season statistics
        if stats and stats.get("statistics"):
            lines.append("")
            lines.append("**Current Season Statistics:**")
            for stat_group in stats.get("statistics", [])[
                :3
            ]:  # Limit to 3 competitions
                tournament = stat_group.get("tournament", {}).get("name", "Competition")
                stat_data = stat_group.get("statistics", {})

                appearances = stat_data.get("appearances", 0)
                goals = stat_data.get("goals", 0)
                assists = stat_data.get("assists", 0)
                rating = stat_data.get("rating")

                stat_line = f"  {tournament}: {appearances} apps"
                if goals or assists:
                    stat_line += f", {goals}G {assists}A"
                if rating:
                    stat_line += f" (Rating: {rating:.2f})"
                lines.append(stat_line)

                # Additional stats if available
                minutes = stat_data.get("minutesPlayed", 0)
                yellow_cards = stat_data.get("yellowCards", 0)
                red_cards = stat_data.get("redCards", 0)

                if minutes or yellow_cards or red_cards:
                    extra_parts = []
                    if minutes:
                        extra_parts.append(f"{minutes} mins")
                    if yellow_cards:
                        extra_parts.append(f"🟨{yellow_cards}")
                    if red_cards:
                        extra_parts.append(f"🟥{red_cards}")
                    lines.append(f"    {', '.join(extra_parts)}")

        # National team stats (can be a list of national team records)
        if national_stats:
            nt_list = (
                national_stats if isinstance(national_stats, list) else [national_stats]
            )
            for nt_entry in nt_list[:1]:  # Just the primary national team
                if not isinstance(nt_entry, dict):
                    continue
                nt_stats = nt_entry.get("statistics", {})
                if not isinstance(nt_stats, dict):
                    continue
                nt_appearances = nt_stats.get("appearances", 0)
                nt_goals = nt_stats.get("goals", 0)
                nt_team = nt_entry.get("team", {}).get("name", "National Team")
                if nt_appearances > 0:
                    lines.append("")
                    lines.append(
                        f"**{nt_team}:** {nt_appearances} caps, {nt_goals} goals"
                    )

        # Transfer history
        if transfers:
            lines.append("")
            lines.append("**Recent Transfers:**")
            for transfer in transfers[:5]:  # Last 5 transfers
                from_team = transfer.get("transferFrom", {}).get("name", "?")
                to_team = transfer.get("transferTo", {}).get("name", "?")
                ts = transfer.get("transferDateTimestamp")
                fee = transfer.get("transferFee")

                date_str = ""
                if ts:
                    date_str = datetime.fromtimestamp(ts).strftime("%b %Y")

                fee_str = ""
                if fee:
                    if fee >= 1000000:
                        fee_str = f" (€{fee / 1000000:.1f}M)"
                    elif fee > 0:
                        fee_str = f" (€{fee / 1000:.0f}K)"

                lines.append(f"  {date_str}: {from_team} → {to_team}{fee_str}")

        return "\n".join(lines)

    def _format_team_squad(self, team: dict, players: list) -> str:
        """Format team squad information."""
        lines = []

        team_name = team.get("name", "Team")
        lines.append(f"**{team_name} Squad**")

        # Group players by position
        # API returns single-letter codes: G=Goalkeeper, D=Defender, M=Midfielder, F=Forward
        positions = {
            "Goalkeepers": [],
            "Defenders": [],
            "Midfielders": [],
            "Forwards": [],
        }
        other = []

        for player_data in players:
            player = player_data.get("player", {})
            position = player.get("position", "")

            player_info = {
                "name": player.get("shortName", player.get("name", "?")),
                "age": None,
                "nationality": player.get("country", {}).get("name", ""),
                "number": player.get("shirtNumber"),
            }

            # Calculate age
            dob = player.get("dateOfBirthTimestamp")
            if dob:
                birth_date = datetime.fromtimestamp(dob)
                player_info["age"] = (datetime.now() - birth_date).days // 365

            # Match single-letter codes or full position names
            pos_upper = position.upper() if position else ""
            if pos_upper == "G" or "Goalkeeper" in position:
                positions["Goalkeepers"].append(player_info)
            elif pos_upper == "D" or "Defender" in position or "Back" in position:
                positions["Defenders"].append(player_info)
            elif pos_upper == "M" or "Midfield" in position:
                positions["Midfielders"].append(player_info)
            elif (
                pos_upper == "F"
                or "Forward" in position
                or "Striker" in position
                or "Wing" in position
            ):
                positions["Forwards"].append(player_info)
            else:
                other.append(player_info)

        for pos_name, pos_players in positions.items():
            if pos_players:
                lines.append("")
                lines.append(f"**{pos_name}:**")
                for p in pos_players:
                    parts = []
                    if p["number"]:
                        parts.append(f"#{p['number']}")
                    parts.append(p["name"])
                    if p["age"]:
                        parts.append(f"({p['age']})")
                    if p["nationality"]:
                        parts.append(f"- {p['nationality']}")
                    lines.append("  " + " ".join(parts))

        if other:
            lines.append("")
            lines.append("**Other:**")
            for p in other:
                lines.append(f"  {p['name']}")

        return "\n".join(lines)

    def _get_comprehensive_event_data(self, event_id: int) -> dict:
        """
        Fetch all available data for an event.
        Returns a dict with all event-related information.
        """
        data = {
            "event_id": event_id,
            "details": self._get_event_details(event_id),
            "lineups": self._get_event_lineups(event_id),
            "incidents": self._get_event_incidents(event_id),
            "statistics": self._get_event_statistics(event_id),
            "h2h": self._get_event_h2h(event_id),
            "best_players": self._get_event_best_players(event_id),
        }
        return data

    def _format_comprehensive_event(self, event_data: dict) -> str:
        """Format comprehensive event data for LLM context."""
        lines = []
        details = event_data.get("details") or {}

        # Basic match info
        home = details.get("homeTeam", {})
        away = details.get("awayTeam", {})
        home_name = home.get("name", "Home")
        away_name = away.get("name", "Away")
        status = details.get("status", {})
        home_score = details.get("homeScore", {}).get("current", 0)
        away_score = details.get("awayScore", {}).get("current", 0)

        # Header with score
        if status.get("type") == "inprogress":
            lines.append(
                f"🔴 **LIVE: {home_name} {home_score} - {away_score} {away_name}**"
            )
            lines.append(f"⏱️ {status.get('description', 'In Progress')}")
        elif status.get("type") == "finished":
            lines.append(f"**FT: {home_name} {home_score} - {away_score} {away_name}**")
        else:
            ts = details.get("startTimestamp", 0)
            if ts:
                dt = datetime.fromtimestamp(ts)
                lines.append(f"**{home_name} vs {away_name}**")
                lines.append(f"📅 {dt.strftime('%A %d %B at %H:%M')}")

        # Tournament & Venue
        tournament = details.get("tournament", {}).get("name", "")
        if tournament:
            lines.append(f"🏆 {tournament}")

        venue = details.get("venue", {})
        if venue.get("name"):
            venue_str = venue.get("name")
            if venue.get("city", {}).get("name"):
                venue_str += f", {venue['city']['name']}"
            lines.append(f"🏟️ {venue_str}")

        referee = details.get("referee", {})
        if referee.get("name"):
            lines.append(f"👨‍⚖️ Referee: {referee['name']}")

        # Incidents (goals, cards)
        incidents = event_data.get("incidents") or []
        goals = [i for i in incidents if i.get("incidentType") == "goal"]
        cards = [
            i
            for i in incidents
            if i.get("incidentType") in ["card", "yellowCard", "redCard"]
        ]

        if goals:
            lines.append("")
            lines.append("**Goals:**")
            for goal in goals:
                player = goal.get("player", {}).get("shortName", "Unknown")
                time_val = goal.get("time", "?")
                is_home = goal.get("isHome", True)
                team_indicator = "🏠" if is_home else "✈️"
                assist = goal.get("assist1", {}).get("shortName")
                goal_line = f"  {team_indicator} ⚽ {time_val}' {player}"
                if assist:
                    goal_line += f" (assist: {assist})"
                lines.append(goal_line)

        if cards:
            lines.append("")
            lines.append("**Cards:**")
            for card in cards[:6]:  # Limit to 6 cards
                player = card.get("player", {}).get("shortName", "Unknown")
                time_val = card.get("time", "?")
                card_type = (
                    "🟨"
                    if "yellow" in str(card.get("incidentType", "")).lower()
                    else "🟥"
                )
                lines.append(f"  {card_type} {time_val}' {player}")

        # Statistics
        stats = event_data.get("statistics")
        if stats and stats.get("statistics"):
            lines.append("")
            lines.append("**Match Statistics:**")
            stat_groups = stats["statistics"]
            for group in stat_groups:
                if group.get("period") == "ALL":
                    for stat_group in group.get("groups", [])[:2]:  # Limit groups
                        for item in stat_group.get("statisticsItems", [])[
                            :8
                        ]:  # Limit items
                            name = item.get("name", "")
                            home_val = item.get("home", "")
                            away_val = item.get("away", "")
                            if name and home_val and away_val:
                                lines.append(f"  {home_val} - {name} - {away_val}")
                    break

        # Lineups
        lineups = event_data.get("lineups")
        if lineups and lineups.get("confirmed"):
            lines.append("")
            lines.append("**Starting XI:**")

            for side, label in [("home", home_name), ("away", away_name)]:
                side_data = lineups.get(side, {})
                players = side_data.get("players", [])
                starters = [p for p in players if not p.get("substitute")]
                if starters:
                    names = [
                        p.get("player", {}).get("shortName", "?") for p in starters[:11]
                    ]
                    lines.append(f"  {label}: {', '.join(names)}")

        # Best players (if available and match finished/in progress)
        best = event_data.get("best_players")
        if best and (status.get("type") in ["inprogress", "finished"]):
            home_best = best.get("bestHomeTeamPlayer", {})
            away_best = best.get("bestAwayTeamPlayer", {})
            if home_best or away_best:
                lines.append("")
                lines.append("**Player Ratings:**")
                if home_best.get("player"):
                    rating = home_best.get("value", "")
                    name = home_best["player"].get("shortName", "")
                    lines.append(f"  ⭐ {home_name}: {name} ({rating})")
                if away_best.get("player"):
                    rating = away_best.get("value", "")
                    name = away_best["player"].get("shortName", "")
                    lines.append(f"  ⭐ {away_name}: {name} ({rating})")

        # H2H summary
        h2h = event_data.get("h2h")
        if h2h:
            events = h2h.get("events", [])
            if events:
                lines.append("")
                lines.append(f"**Head to Head (last {min(5, len(events))}):**")
                for match in events[:5]:
                    h_team = match.get("homeTeam", {}).get("shortName", "?")
                    a_team = match.get("awayTeam", {}).get("shortName", "?")
                    h_score = match.get("homeScore", {}).get("current", "?")
                    a_score = match.get("awayScore", {}).get("current", "?")
                    lines.append(f"  {h_team} {h_score}-{a_score} {a_team}")

        return "\n".join(lines)

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch sports data based on parameters.

        Intelligently routes to appropriate query based on context.
        """
        if not self.api_key:
            return LiveDataResult(
                success=False,
                error="SportAPI7 API key not configured. Set SPORTAPI7_API_KEY or configure in admin.",
            )

        team_name = params.get("team", "").strip()
        player_name = params.get("player", "").strip()
        opponent_name = params.get("opponent", "").strip()
        league_name = params.get("league", "").strip()
        query_type = params.get("query_type", "auto").lower()
        sport = params.get("sport", "").strip() or self.default_sport

        # Handle player queries first
        if player_name or query_type == "player":
            return self._fetch_player_info(player_name or team_name)

        # Handle two-team queries (fixture, h2h, score between teams)
        if team_name and opponent_name:
            return self._fetch_matchup(team_name, opponent_name, query_type, sport)

        # If no team specified, check for live events or standings
        if not team_name:
            if league_name:
                # Get league standings
                tournament_id = self._resolve_tournament(league_name)
                if tournament_id:
                    return self._fetch_standings(tournament_id, league_name)
                return LiveDataResult(
                    success=False,
                    error=f"Could not find league: {league_name}",
                )
            elif query_type == "live_scores":
                return self._fetch_live_scores(sport)
            else:
                return LiveDataResult(
                    success=False,
                    error="Please specify a team name, player name, or league, e.g., 'Arsenal', 'Mbappe', or 'Premier League'",
                )

        # Resolve team name to ID
        team = self._resolve_team(team_name, sport)
        if not team:
            return LiveDataResult(
                success=False,
                error=f"Could not find team: {team_name}. Try the full team name.",
            )

        team_id = team["id"]
        sport_name = team.get("sport", {}).get("name", "Sport")

        # Auto-detect query type if not specified
        if query_type == "auto":
            # Check if team is currently playing
            live_match = self._get_team_live_match(team_id)
            if live_match:
                return self._format_live_match(team, live_match)

            # Otherwise show next match + recent form
            return self._fetch_team_overview(team)

        # Handle specific query types
        if query_type == "live_scores" or query_type == "live" or query_type == "score":
            # "score" query = user is asking about a current/live game
            live_match = self._get_team_live_match(team_id)
            if live_match:
                return self._format_live_match(team, live_match)
            return LiveDataResult(
                success=True,
                formatted=f"**{team['name']}** is not currently playing a match.",
                data={"team": team, "live": False},
            )

        elif (
            query_type == "next_match"
            or query_type == "next"
            or query_type == "fixture"
        ):
            return self._fetch_next_match(team)

        elif (
            query_type == "recent_results"
            or query_type == "results"
            or query_type == "form"
        ):
            return self._fetch_recent_results(team)

        elif query_type == "standings" or query_type == "table":
            # Try to get standings from team's primary tournament
            # For now, fall back to team overview
            return self._fetch_team_overview(team)

        elif query_type == "squad" or query_type == "players" or query_type == "roster":
            return self._fetch_team_squad(team)

        elif query_type == "transfers":
            return self._fetch_team_transfers(team)

        else:
            return self._fetch_team_overview(team)

    def _fetch_matchup(
        self, team_name: str, opponent_name: str, query_type: str, sport: str
    ) -> LiveDataResult:
        """Fetch information about a matchup between two teams."""
        # Resolve both teams
        team1 = self._resolve_team(team_name, sport)
        if not team1:
            return LiveDataResult(
                success=False,
                error=f"Could not find team: {team_name}",
            )

        team2 = self._resolve_team(opponent_name, sport)
        if not team2:
            return LiveDataResult(
                success=False,
                error=f"Could not find team: {opponent_name}",
            )

        team1_id = team1["id"]
        team2_id = team2["id"]
        team1_name = team1["name"]
        team2_name = team2["name"]

        # Handle "score" query - only return live match, not fixtures
        if query_type == "score":
            live_match = self._get_team_live_match(team1_id)
            if live_match:
                # Check if it's against team2
                home_id = live_match.get("homeTeam", {}).get("id")
                away_id = live_match.get("awayTeam", {}).get("id")
                if home_id == team2_id or away_id == team2_id:
                    return self._format_live_match(team1, live_match)
            # No live match between these teams
            return LiveDataResult(
                success=True,
                formatted=f"**{team1_name}** and **{team2_name}** are not currently playing each other.",
                data={"team1": team1, "team2": team2, "live": False},
            )

        # Check for live or recent match first (for general queries)
        if query_type in ["auto", "live", "result"]:
            match = self._find_live_or_recent_match_between_teams(team1_id, team2_id)
            if match:
                status = match.get("status", {}).get("type", "")
                if status == "inprogress":
                    # Live match - get comprehensive data
                    return self._format_live_match(team1, match)
                elif status == "finished" and query_type != "auto":
                    # Recent result - only show for explicit result queries, not auto
                    event_id = match.get("id")
                    if event_id:
                        event_data = self._get_comprehensive_event_data(event_id)
                        formatted = self._format_comprehensive_event(event_data)
                    else:
                        home = match.get("homeTeam", {})
                        away = match.get("awayTeam", {})
                        home_score = match.get("homeScore", {}).get("current", 0)
                        away_score = match.get("awayScore", {}).get("current", 0)
                        formatted = f"**{home.get('name')} {home_score} - {away_score} {away.get('name')}** (FT)"

                    return LiveDataResult(
                        success=True,
                        formatted=formatted,
                        data={"team1": team1, "team2": team2, "match": match},
                        cache_ttl=300,
                    )

        # Check for upcoming fixture
        if query_type in ["auto", "fixture", "next", "when"]:
            fixture = self._find_fixture_between_teams(team1_id, team2_id)
            if fixture:
                home = fixture.get("homeTeam", {})
                away = fixture.get("awayTeam", {})
                ts = fixture.get("startTimestamp", 0)
                tournament = fixture.get("tournament", {}).get("name", "")

                lines = [f"**{home.get('name')} vs {away.get('name')}**"]
                if ts:
                    dt = datetime.fromtimestamp(ts)
                    lines.append(f"📅 {dt.strftime('%A %d %B %Y at %H:%M')}")
                if tournament:
                    lines.append(f"🏆 {tournament}")

                return LiveDataResult(
                    success=True,
                    formatted="\n".join(lines),
                    data={"team1": team1, "team2": team2, "fixture": fixture},
                    cache_ttl=300,
                )

        # Get H2H history
        if query_type in ["h2h", "history", "head_to_head"]:
            h2h_events = self._get_h2h_events(team1_id, team2_id)
            return self._format_h2h(team1, team2, h2h_events)

        # No specific match found - show H2H history and next fixture info
        h2h_events = self._get_h2h_events(team1_id, team2_id)
        fixture = self._find_fixture_between_teams(team1_id, team2_id)

        lines = [f"**{team1_name} vs {team2_name}**"]

        if fixture:
            ts = fixture.get("startTimestamp", 0)
            tournament = fixture.get("tournament", {}).get("name", "")
            if ts:
                dt = datetime.fromtimestamp(ts)
                lines.append("")
                lines.append("**Next Meeting:**")
                lines.append(f"📅 {dt.strftime('%A %d %B %Y at %H:%M')}")
                if tournament:
                    lines.append(f"🏆 {tournament}")

        # Recent H2H
        finished = [
            e for e in h2h_events if e.get("status", {}).get("type") == "finished"
        ]
        if finished:
            lines.append("")
            lines.append("**Recent Meetings:**")
            for match in finished[:5]:
                home = match.get("homeTeam", {})
                away = match.get("awayTeam", {})
                home_score = match.get("homeScore", {}).get("current", "?")
                away_score = match.get("awayScore", {}).get("current", "?")
                lines.append(
                    f"  {home.get('shortName', home.get('name'))} {home_score}-{away_score} {away.get('shortName', away.get('name'))}"
                )

        if len(lines) == 1:
            lines.append("")
            lines.append("No upcoming fixture or recent meetings found.")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={
                "team1": team1,
                "team2": team2,
                "h2h": h2h_events,
                "fixture": fixture,
            },
            cache_ttl=300,
        )

    def _format_h2h(
        self, team1: dict, team2: dict, h2h_events: list[dict]
    ) -> LiveDataResult:
        """Format head-to-head history."""
        team1_name = team1["name"]
        team2_name = team2["name"]
        team1_id = team1["id"]

        lines = [f"**{team1_name} vs {team2_name} - Head to Head**"]

        finished = [
            e for e in h2h_events if e.get("status", {}).get("type") == "finished"
        ]

        if not finished:
            lines.append("")
            lines.append("No recent meetings found.")
            return LiveDataResult(
                success=True,
                formatted="\n".join(lines),
                data={"team1": team1, "team2": team2, "h2h": []},
            )

        # Calculate stats
        team1_wins = 0
        team2_wins = 0
        draws = 0

        for match in finished:
            home_id = match.get("homeTeam", {}).get("id")
            home_score = match.get("homeScore", {}).get("current", 0)
            away_score = match.get("awayScore", {}).get("current", 0)

            if home_score == away_score:
                draws += 1
            elif home_id == team1_id:
                if home_score > away_score:
                    team1_wins += 1
                else:
                    team2_wins += 1
            else:
                if away_score > home_score:
                    team1_wins += 1
                else:
                    team2_wins += 1

        lines.append("")
        lines.append(
            f"**Record:** {team1_name} {team1_wins}W - {draws}D - {team2_wins}W {team2_name}"
        )

        lines.append("")
        lines.append("**Recent Meetings:**")
        for match in finished[:10]:
            home = match.get("homeTeam", {})
            away = match.get("awayTeam", {})
            home_score = match.get("homeScore", {}).get("current", "?")
            away_score = match.get("awayScore", {}).get("current", "?")
            ts = match.get("startTimestamp")
            date_str = ""
            if ts:
                date_str = datetime.fromtimestamp(ts).strftime("%d %b %Y") + ": "
            lines.append(
                f"  {date_str}{home.get('shortName', home.get('name'))} {home_score}-{away_score} {away.get('shortName', away.get('name'))}"
            )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"team1": team1, "team2": team2, "h2h": finished},
            cache_ttl=3600,
        )

    def _fetch_player_info(self, player_name: str) -> LiveDataResult:
        """Fetch comprehensive player information."""
        if not player_name:
            return LiveDataResult(
                success=False,
                error="Please specify a player name, e.g., 'Mbappe' or 'Haaland'",
            )

        # Search for player
        player = self._search_player(player_name)
        if not player:
            return LiveDataResult(
                success=False,
                error=f"Could not find player: {player_name}. Try the full name.",
            )

        player_id = player.get("id")

        # Get full player details
        details = self._get_player_details(player_id)
        if details:
            player = details  # Use more detailed version

        # Get additional data
        stats = self._get_player_statistics(player_id)
        transfers = self._get_player_transfer_history(player_id)
        national_stats = self._get_player_national_stats(player_id)

        formatted = self._format_player_info(player, stats, transfers, national_stats)

        return LiveDataResult(
            success=True,
            formatted=formatted,
            data={
                "player": player,
                "statistics": stats,
                "transfers": transfers,
                "national_stats": national_stats,
            },
            cache_ttl=3600,  # 1 hour for player data
        )

    def _fetch_team_squad(self, team: dict) -> LiveDataResult:
        """Fetch team's current squad."""
        team_id = team["id"]

        # Get detailed team info and players
        team_details = self._get_team_details(team_id) or team
        players = self._get_team_players(team_id)

        if not players:
            return LiveDataResult(
                success=True,
                formatted=f"No squad information available for **{team['name']}**.",
                data={"team": team},
            )

        formatted = self._format_team_squad(team_details, players)

        return LiveDataResult(
            success=True,
            formatted=formatted,
            data={"team": team_details, "players": players},
            cache_ttl=3600,  # 1 hour
        )

    def _fetch_team_transfers(self, team: dict) -> LiveDataResult:
        """Fetch team's recent transfers."""
        team_id = team["id"]

        transfers_data = self._get_team_transfers(team_id)
        if not transfers_data:
            return LiveDataResult(
                success=True,
                formatted=f"No transfer information available for **{team['name']}**.",
                data={"team": team},
            )

        lines = [f"**{team['name']} - Recent Transfers**"]

        # Incoming transfers
        transfers_in = transfers_data.get("transfersIn", [])
        if transfers_in:
            lines.append("")
            lines.append("**Arrivals:**")
            for t in transfers_in[:10]:
                player = t.get("player", {})
                from_team = t.get("transferFrom", {}).get("name", "?")
                fee = t.get("transferFee")
                ts = t.get("transferDateTimestamp")

                date_str = ""
                if ts:
                    date_str = datetime.fromtimestamp(ts).strftime("%b %Y") + ": "

                fee_str = ""
                if fee:
                    if fee >= 1000000:
                        fee_str = f" (€{fee / 1000000:.1f}M)"
                    elif fee > 0:
                        fee_str = f" (€{fee / 1000:.0f}K)"
                    else:
                        fee_str = " (Free)"

                lines.append(
                    f"  {date_str}{player.get('name', '?')} from {from_team}{fee_str}"
                )

        # Outgoing transfers
        transfers_out = transfers_data.get("transfersOut", [])
        if transfers_out:
            lines.append("")
            lines.append("**Departures:**")
            for t in transfers_out[:10]:
                player = t.get("player", {})
                to_team = t.get("transferTo", {}).get("name", "?")
                fee = t.get("transferFee")
                ts = t.get("transferDateTimestamp")

                date_str = ""
                if ts:
                    date_str = datetime.fromtimestamp(ts).strftime("%b %Y") + ": "

                fee_str = ""
                if fee:
                    if fee >= 1000000:
                        fee_str = f" (€{fee / 1000000:.1f}M)"
                    elif fee > 0:
                        fee_str = f" (€{fee / 1000:.0f}K)"
                    else:
                        fee_str = " (Free)"

                lines.append(
                    f"  {date_str}{player.get('name', '?')} to {to_team}{fee_str}"
                )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"team": team, "transfers": transfers_data},
            cache_ttl=3600,  # 1 hour
        )

    def _fetch_team_overview(self, team: dict) -> LiveDataResult:
        """Get comprehensive team overview: live/next match + recent form."""
        team_id = team["id"]
        lines = [f"**{team['name']}** ({team.get('sport', {}).get('name', 'Sport')})"]

        # Check for live match
        live_match = self._get_team_live_match(team_id)
        if live_match:
            lines.append("")
            lines.append("🔴 **LIVE NOW:**")
            lines.append(self._format_match_line(live_match, include_status=True))
        else:
            # Next match
            next_match = self._get_team_next_match(team_id)
            if next_match:
                lines.append("")
                lines.append("**Next Match:**")
                lines.append(self._format_match_line(next_match, include_time=True))

        # Recent results
        recent = self._get_team_recent_results(team_id, limit=5)
        if recent:
            lines.append("")
            lines.append("**Recent Results:**")
            for match in recent:
                lines.append(self._format_match_line(match, include_result=True))

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"team": team, "live": live_match, "recent": recent},
            cache_ttl=self.LIVE_CACHE_TTL if live_match else 300,
        )

    def _fetch_next_match(self, team: dict) -> LiveDataResult:
        """Get team's next scheduled match."""
        next_match = self._get_team_next_match(team["id"])
        if not next_match:
            return LiveDataResult(
                success=True,
                formatted=f"No upcoming matches found for **{team['name']}**.",
                data={"team": team},
            )

        lines = [f"**{team['name']}** - Next Match:"]
        lines.append("")
        lines.append(
            self._format_match_line(next_match, include_time=True, verbose=True)
        )

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"team": team, "next_match": next_match},
            cache_ttl=300,  # 5 minutes
        )

    def _fetch_recent_results(self, team: dict, limit: int = 5) -> LiveDataResult:
        """Get team's recent match results."""
        recent = self._get_team_recent_results(team["id"], limit=limit)
        if not recent:
            return LiveDataResult(
                success=True,
                formatted=f"No recent results found for **{team['name']}**.",
                data={"team": team},
            )

        # Calculate form
        wins = sum(1 for m in recent if self._get_match_result(m, team["id"]) == "W")
        draws = sum(1 for m in recent if self._get_match_result(m, team["id"]) == "D")
        losses = sum(1 for m in recent if self._get_match_result(m, team["id"]) == "L")
        form = "".join(self._get_match_result(m, team["id"]) for m in recent)

        lines = [
            f"**{team['name']}** - Recent Form: {form} ({wins}W {draws}D {losses}L)"
        ]
        lines.append("")
        for match in recent:
            lines.append(self._format_match_line(match, include_result=True))

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"team": team, "recent": recent, "form": form},
            cache_ttl=300,
        )

    def _fetch_live_scores(self, sport: str) -> LiveDataResult:
        """Get all live matches for a sport."""
        events = self._get_live_events(sport)

        if not events:
            return LiveDataResult(
                success=True,
                formatted=f"No live {sport} matches at the moment.",
                data={"sport": sport, "events": []},
            )

        lines = [f"**Live {sport.title()} Scores:**"]
        lines.append("")

        # Group by tournament
        tournaments: dict[str, list] = {}
        for event in events[:20]:  # Limit to 20 matches
            tournament = event.get("tournament", {}).get("name", "Other")
            if tournament not in tournaments:
                tournaments[tournament] = []
            tournaments[tournament].append(event)

        for tournament, matches in tournaments.items():
            lines.append(f"**{tournament}:**")
            for match in matches[:5]:  # Max 5 per tournament
                lines.append(self._format_match_line(match, include_status=True))
            lines.append("")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"sport": sport, "events": events},
            cache_ttl=self.LIVE_CACHE_TTL,
        )

    def _fetch_standings(self, tournament_id: int, league_name: str) -> LiveDataResult:
        """Get league standings."""
        data = self._get_tournament_standings(tournament_id)
        if not data:
            return LiveDataResult(
                success=False,
                error=f"Could not fetch standings for {league_name}",
            )

        standings = data.get("standings", [])
        if not standings:
            return LiveDataResult(
                success=False,
                error=f"No standings available for {league_name}",
            )

        rows = standings[0].get("rows", [])

        lines = [f"**{league_name} Standings:**"]
        lines.append("")
        lines.append("```")
        lines.append(
            f"{'Pos':<4} {'Team':<20} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'GD':<6} {'Pts':<4}"
        )
        lines.append("-" * 60)

        for row in rows[:20]:  # Top 20
            team_name = row.get("team", {}).get(
                "shortName", row.get("team", {}).get("name", "?")
            )
            lines.append(
                f"{row.get('position', '?'):<4} "
                f"{team_name[:19]:<20} "
                f"{row.get('matches', 0):<4} "
                f"{row.get('wins', 0):<4} "
                f"{row.get('draws', 0):<4} "
                f"{row.get('losses', 0):<4} "
                f"{row.get('scoreDiffFormatted', '0'):<6} "
                f"{row.get('points', 0):<4}"
            )
        lines.append("```")

        return LiveDataResult(
            success=True,
            formatted="\n".join(lines),
            data={"standings": rows},
            cache_ttl=self.STANDINGS_CACHE_TTL,
        )

    def _format_live_match(self, team: dict, match: dict) -> LiveDataResult:
        """Format a live match with comprehensive data."""
        event_id = match.get("id")

        # Fetch comprehensive event data for richer context
        if event_id:
            event_data = self._get_comprehensive_event_data(event_id)
            formatted = self._format_comprehensive_event(event_data)
        else:
            # Fallback to basic formatting
            home = match.get("homeTeam", {})
            away = match.get("awayTeam", {})
            home_score = match.get("homeScore", {}).get("current", 0)
            away_score = match.get("awayScore", {}).get("current", 0)
            status = match.get("status", {}).get("description", "Live")
            tournament = match.get("tournament", {}).get("name", "")

            lines = [
                f"🔴 **LIVE: {home.get('name')} {home_score} - {away_score} {away.get('name')}**"
            ]
            lines.append(f"⏱️ {status}")
            if tournament:
                lines.append(f"🏆 {tournament}")
            formatted = "\n".join(lines)

        return LiveDataResult(
            success=True,
            formatted=formatted,
            data={"team": team, "match": match, "live": True},
            cache_ttl=self.LIVE_CACHE_TTL,
        )

    def _format_match_line(
        self,
        match: dict,
        include_time: bool = False,
        include_result: bool = False,
        include_status: bool = False,
        verbose: bool = False,
    ) -> str:
        """Format a single match line."""
        home = match.get("homeTeam", {})
        away = match.get("awayTeam", {})
        home_name = home.get("shortName", home.get("name", "?"))
        away_name = away.get("shortName", away.get("name", "?"))

        parts = []

        if include_status:
            status = match.get("status", {})
            if status.get("type") == "inprogress":
                home_score = match.get("homeScore", {}).get("current", 0)
                away_score = match.get("awayScore", {}).get("current", 0)
                status_desc = status.get("description", "Live")
                parts.append(
                    f"🔴 {home_name} {home_score}-{away_score} {away_name} ({status_desc})"
                )
            else:
                parts.append(f"{home_name} vs {away_name}")
        elif include_result:
            home_score = match.get("homeScore", {}).get("current", "?")
            away_score = match.get("awayScore", {}).get("current", "?")
            parts.append(f"- {home_name} {home_score}-{away_score} {away_name}")
        elif include_time:
            ts = match.get("startTimestamp", 0)
            if ts:
                dt = datetime.fromtimestamp(ts)
                if verbose:
                    time_str = dt.strftime("%A %d %B at %H:%M")
                else:
                    time_str = dt.strftime("%a %d %b %H:%M")
                parts.append(f"- {home_name} vs {away_name} ({time_str})")
            else:
                parts.append(f"- {home_name} vs {away_name}")
        else:
            parts.append(f"- {home_name} vs {away_name}")

        if verbose:
            tournament = match.get("tournament", {}).get("name", "")
            if tournament:
                parts.append(f"  🏆 {tournament}")

        return "\n".join(parts) if verbose else parts[0]

    def _get_match_result(self, match: dict, team_id: int) -> str:
        """Get W/D/L result for a team in a match."""
        home_id = match.get("homeTeam", {}).get("id")
        home_score = match.get("homeScore", {}).get("current", 0)
        away_score = match.get("awayScore", {}).get("current", 0)

        is_home = home_id == team_id
        team_score = home_score if is_home else away_score
        opponent_score = away_score if is_home else home_score

        if team_score > opponent_score:
            return "W"
        elif team_score < opponent_score:
            return "L"
        else:
            return "D"

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection with a sample query."""
        if not self.api_key:
            return False, "API key not configured"

        try:
            team = self._resolve_team("Arsenal")
            if team:
                return True, f"Connected - found {team['name']} (ID: {team['id']})"
            return False, "Could not resolve test team"
        except Exception as e:
            return False, str(e)
