"""
BallDontLie API Provider - Backup/secondary data source.

Free API for NBA stats when nba_api is rate-limited or unavailable.
https://www.balldontlie.io/

Rate limits: 60 requests/minute on free tier
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from src.utils.cache import DiskCache
from src.utils.team_map import resolve_team_name

logger = logging.getLogger(__name__)


class BallDontLieProvider:
    """
    Provider for BallDontLie API (free tier).
    
    Backup data source with caching.
    """
    
    BASE_URL = "https://api.balldontlie.io/v1"
    RATE_LIMIT_DELAY = 1.0  # 1 second between requests (60/min limit)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[DiskCache] = None
    ):
        """
        Initialize provider.
        
        Args:
            api_key: Optional API key for higher rate limits
            cache: Disk cache instance
        """
        self.api_key = api_key
        self.cache = cache or DiskCache(cache_dir="data/cache/balldontlie")
        self._last_request_time = 0
        
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = api_key
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _request(
        self,
        endpoint: str,
        params: dict = None,
        max_retries: int = 3
    ) -> dict:
        """Make API request with retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
                
                if response.status_code == 429:
                    # Rate limited
                    wait_time = 60  # Wait a minute
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Failed to fetch {endpoint} after {max_retries} attempts")
    
    # ========================================================================
    # Teams
    # ========================================================================
    
    def get_teams(self) -> pd.DataFrame:
        """Get all NBA teams."""
        cache_key = "teams"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self._request("teams")
        
        teams = []
        for team in data.get("data", []):
            teams.append({
                "team_id": team["id"],
                "abbreviation": team["abbreviation"],
                "city": team["city"],
                "name": team["name"],
                "full_name": team["full_name"],
                "conference": team["conference"],
                "division": team["division"],
            })
        
        df = pd.DataFrame(teams)
        self.cache.set(cache_key, df, ttl_hours=168)  # Cache for a week
        return df
    
    def get_team_id(self, team: str) -> Optional[int]:
        """Get BallDontLie team ID from team name/abbreviation."""
        teams_df = self.get_teams()
        
        team_norm = team.upper().strip()
        
        # Try abbreviation
        match = teams_df[teams_df["abbreviation"] == team_norm]
        if not match.empty:
            return match.iloc[0]["team_id"]
        
        # Try full name
        match = teams_df[teams_df["full_name"].str.upper() == team_norm]
        if not match.empty:
            return match.iloc[0]["team_id"]
        
        # Try partial match
        for _, row in teams_df.iterrows():
            if team_norm in row["full_name"].upper():
                return row["team_id"]
        
        return None
    
    # ========================================================================
    # Players
    # ========================================================================
    
    def get_players(
        self,
        team_id: Optional[int] = None,
        search: Optional[str] = None,
        per_page: int = 100
    ) -> pd.DataFrame:
        """
        Get players.
        
        Args:
            team_id: Filter by team
            search: Search by name
            per_page: Results per page
        """
        params = {"per_page": per_page}
        if team_id:
            params["team_ids[]"] = team_id
        if search:
            params["search"] = search
        
        cache_key = f"players_{team_id}_{search}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self._request("players", params)
        
        players = []
        for player in data.get("data", []):
            players.append({
                "player_id": player["id"],
                "first_name": player["first_name"],
                "last_name": player["last_name"],
                "full_name": f"{player['first_name']} {player['last_name']}",
                "position": player.get("position", ""),
                "team_id": player.get("team", {}).get("id"),
                "team_abbr": player.get("team", {}).get("abbreviation"),
            })
        
        df = pd.DataFrame(players)
        self.cache.set(cache_key, df, ttl_hours=24)
        return df
    
    def search_player(self, name: str) -> Optional[dict]:
        """Search for a player by name."""
        players = self.get_players(search=name)
        if players.empty:
            return None
        return players.iloc[0].to_dict()
    
    # ========================================================================
    # Games
    # ========================================================================
    
    def get_games(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        seasons: Optional[list[int]] = None,
        team_ids: Optional[list[int]] = None,
        per_page: int = 100
    ) -> pd.DataFrame:
        """
        Get games.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            seasons: List of seasons (e.g., [2024, 2025])
            team_ids: Filter by teams
            per_page: Results per page
        """
        params = {"per_page": per_page}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if seasons:
            params["seasons[]"] = seasons
        if team_ids:
            params["team_ids[]"] = team_ids
        
        cache_key = f"games_{start_date}_{end_date}_{seasons}_{team_ids}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self._request("games", params)
        
        games = []
        for game in data.get("data", []):
            games.append({
                "game_id": game["id"],
                "date": game["date"],
                "season": game["season"],
                "home_team_id": game["home_team"]["id"],
                "home_team": game["home_team"]["abbreviation"],
                "home_score": game["home_team_score"],
                "away_team_id": game["visitor_team"]["id"],
                "away_team": game["visitor_team"]["abbreviation"],
                "away_score": game["visitor_team_score"],
                "status": game.get("status"),
            })
        
        df = pd.DataFrame(games)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        
        self.cache.set(cache_key, df, ttl_hours=6)
        return df
    
    # ========================================================================
    # Stats
    # ========================================================================
    
    def get_player_stats(
        self,
        player_id: Optional[int] = None,
        game_ids: Optional[list[int]] = None,
        seasons: Optional[list[int]] = None,
        per_page: int = 100
    ) -> pd.DataFrame:
        """
        Get player stats (box scores).
        
        Args:
            player_id: Filter by player
            game_ids: Filter by games
            seasons: Filter by seasons
            per_page: Results per page
        """
        params = {"per_page": per_page}
        if player_id:
            params["player_ids[]"] = player_id
        if game_ids:
            params["game_ids[]"] = game_ids
        if seasons:
            params["seasons[]"] = seasons
        
        cache_key = f"stats_{player_id}_{seasons}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self._request("stats", params)
        
        stats = []
        for stat in data.get("data", []):
            stats.append({
                "player_id": stat["player"]["id"],
                "player_name": f"{stat['player']['first_name']} {stat['player']['last_name']}",
                "team_id": stat["team"]["id"],
                "team_abbr": stat["team"]["abbreviation"],
                "game_id": stat["game"]["id"],
                "game_date": stat["game"]["date"],
                "minutes": stat.get("min", "0"),
                "pts": stat.get("pts", 0),
                "reb": stat.get("reb", 0),
                "ast": stat.get("ast", 0),
                "fgm": stat.get("fgm", 0),
                "fga": stat.get("fga", 0),
                "fg3m": stat.get("fg3m", 0),
                "fg3a": stat.get("fg3a", 0),
                "ftm": stat.get("ftm", 0),
                "fta": stat.get("fta", 0),
                "stl": stat.get("stl", 0),
                "blk": stat.get("blk", 0),
                "tov": stat.get("turnover", 0),
                "pf": stat.get("pf", 0),
            })
        
        df = pd.DataFrame(stats)
        if not df.empty:
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["minutes"] = df["minutes"].apply(self._parse_minutes)
        
        self.cache.set(cache_key, df, ttl_hours=6)
        return df
    
    def get_season_averages(
        self,
        player_id: int,
        season: int = None
    ) -> dict:
        """
        Get player season averages.
        
        Args:
            player_id: BallDontLie player ID
            season: Season year
        """
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        
        cache_key = f"season_avg_{player_id}_{season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        params = {
            "player_ids[]": player_id,
            "season": season,
        }
        
        data = self._request("season_averages", params)
        
        if not data.get("data"):
            return {}
        
        avg = data["data"][0]
        result = {
            "player_id": player_id,
            "season": season,
            "games_played": avg.get("games_played", 0),
            "minutes": avg.get("min", 0),
            "pts": avg.get("pts", 0),
            "reb": avg.get("reb", 0),
            "ast": avg.get("ast", 0),
            "fg_pct": avg.get("fg_pct", 0),
            "fg3_pct": avg.get("fg3_pct", 0),
            "ft_pct": avg.get("ft_pct", 0),
        }
        
        self.cache.set(cache_key, result, ttl_hours=24)
        return result
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _parse_minutes(self, val) -> float:
        """Parse minutes from string format."""
        if pd.isna(val) or val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val = val.strip()
            if not val or val == "":
                return 0.0
            if ":" in val:
                parts = val.split(":")
                try:
                    return float(parts[0]) + float(parts[1]) / 60
                except (ValueError, IndexError):
                    return 0.0
            try:
                return float(val)
            except ValueError:
                return 0.0
        return 0.0
    
    def check_availability(self) -> bool:
        """Check if API is available."""
        try:
            self.get_teams()
            return True
        except Exception:
            return False

