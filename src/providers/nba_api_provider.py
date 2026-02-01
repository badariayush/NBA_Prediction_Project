"""
NBA API Provider - Primary data source using nba_api package.

Fetches:
- Team rosters
- Player game logs (box scores)
- Season standings
- Schedule data

All calls are cached to disk to reduce API load.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils.cache import DiskCache, with_retry
from src.utils.team_map import get_team_id, abbr_from_id, TEAM_IDS

logger = logging.getLogger(__name__)


class NbaApiProvider:
    """
    Provider for NBA.com data via nba_api package.
    
    Primary data source with caching and rate limiting.
    """
    
    # Rate limiting
    API_DELAY = 0.6  # seconds between requests
    
    def __init__(self, cache: Optional[DiskCache] = None):
        self.cache = cache or DiskCache(cache_dir="data/cache/nba_api")
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.API_DELAY:
            time.sleep(self.API_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _season_str(self, year: int) -> str:
        """Convert year to NBA season string (e.g., 2024 -> '2024-25')."""
        return f"{year}-{str(year + 1)[-2:]}"
    
    def _parse_season(self, season_str: str) -> int:
        """Parse season string to year (e.g., '2024-25' -> 2024)."""
        return int(season_str.split("-")[0])
    
    # ========================================================================
    # Roster Methods
    # ========================================================================
    
    def get_team_roster(
        self,
        team: str,
        season: int = None
    ) -> pd.DataFrame:
        """
        Get team roster for a season.
        
        Args:
            team: Team name, abbreviation, or ID
            season: Season year (e.g., 2024 for 2024-25). Defaults to current.
            
        Returns:
            DataFrame with columns: player_id, player_name, position, jersey_number
        """
        from nba_api.stats.endpoints import commonteamroster
        
        team_id = get_team_id(team) if isinstance(team, str) else team
        if not team_id:
            raise ValueError(f"Unknown team: {team}")
        
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        
        cache_key = f"roster_{team_id}_{season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        self._rate_limit()
        
        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=self._season_str(season)
            )
            df = roster.get_data_frames()[0]
            
            result = pd.DataFrame({
                "player_id": df["PLAYER_ID"],
                "player_name": df["PLAYER"],
                "position": df.get("POSITION", ""),
                "jersey_number": df.get("NUM", ""),
                "team_id": team_id,
                "season": season,
            })
            
            self.cache.set(cache_key, result, ttl_hours=24)
            logger.info(f"Fetched roster for team {team_id} season {season}: {len(result)} players")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch roster for {team_id}: {e}")
            raise
    
    # ========================================================================
    # Player Stats Methods
    # ========================================================================
    
    def get_player_game_logs(
        self,
        player_id: int,
        season: int = None,
        last_n_games: int = None
    ) -> pd.DataFrame:
        """
        Get player's game-by-game stats.
        
        Args:
            player_id: NBA player ID
            season: Season year. If None, uses current season.
            last_n_games: Limit to last N games
            
        Returns:
            DataFrame with per-game stats
        """
        from nba_api.stats.endpoints import playergamelog
        
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        
        cache_key = f"player_log_{player_id}_{season}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            df = cached
        else:
            self._rate_limit()
            
            try:
                log = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=self._season_str(season)
                )
                df = log.get_data_frames()[0]
                
                if df.empty:
                    return pd.DataFrame()
                
                # Normalize columns
                df = df.rename(columns={
                    "GAME_ID": "game_id",
                    "GAME_DATE": "game_date",
                    "MATCHUP": "matchup",
                    "WL": "result",
                    "MIN": "minutes",
                    "PTS": "pts",
                    "REB": "reb",
                    "AST": "ast",
                    "FGM": "fgm",
                    "FGA": "fga",
                    "FG3M": "fg3m",
                    "FG3A": "fg3a",
                    "FTM": "ftm",
                    "FTA": "fta",
                    "STL": "stl",
                    "BLK": "blk",
                    "TOV": "tov",
                    "PF": "pf",
                    "PLUS_MINUS": "plus_minus",
                })
                
                df["game_date"] = pd.to_datetime(df["game_date"])
                df["player_id"] = player_id
                df["season"] = season
                
                # Parse minutes (handle "MM:SS" format)
                df["minutes"] = df["minutes"].apply(self._parse_minutes)
                
                self.cache.set(cache_key, df, ttl_hours=6)
                
            except Exception as e:
                logger.warning(f"Failed to fetch game log for player {player_id}: {e}")
                return pd.DataFrame()
        
        # Sort by date descending and limit
        df = df.sort_values("game_date", ascending=False)
        
        if last_n_games:
            df = df.head(last_n_games)
        
        return df
    
    def get_player_season_stats(
        self,
        player_id: int,
        season: int = None
    ) -> dict:
        """
        Get player's season averages.
        
        Returns dict with avg stats.
        """
        game_logs = self.get_player_game_logs(player_id, season)
        
        if game_logs.empty:
            return {}
        
        stat_cols = ["minutes", "pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
        available_cols = [c for c in stat_cols if c in game_logs.columns]
        
        stats = game_logs[available_cols].mean().to_dict()
        stats["games_played"] = len(game_logs)
        
        # Calculate percentages from totals (not rounded averages)
        if "fgm" in game_logs.columns and "fga" in game_logs.columns:
            total_fgm = game_logs["fgm"].sum()
            total_fga = game_logs["fga"].sum()
            stats["fg_pct"] = total_fgm / max(total_fga, 1)
        
        if "fg3m" in game_logs.columns and "fg3a" in game_logs.columns:
            total_fg3m = game_logs["fg3m"].sum()
            total_fg3a = game_logs["fg3a"].sum()
            stats["fg3_pct"] = total_fg3m / max(total_fg3a, 1)
        
        return stats
    
    def get_player_recent_form(
        self,
        player_id: int,
        before_date: datetime,
        n_games: int = 10,
        season: int = None
    ) -> dict:
        """
        Get player's recent form stats before a specific date.
        
        Returns dict with avg stats over last N games.
        """
        game_logs = self.get_player_game_logs(player_id, season)
        
        if game_logs.empty:
            return {}
        
        # Filter to games before the target date
        game_logs = game_logs[game_logs["game_date"] < before_date]
        game_logs = game_logs.head(n_games)
        
        if game_logs.empty:
            return {}
        
        stat_cols = ["minutes", "pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a"]
        available_cols = [c for c in stat_cols if c in game_logs.columns]
        
        stats = game_logs[available_cols].mean().to_dict()
        stats["n_games"] = len(game_logs)
        
        # Percentages from totals
        if "fgm" in game_logs.columns and "fga" in game_logs.columns:
            stats["fg_pct"] = game_logs["fgm"].sum() / max(game_logs["fga"].sum(), 1)
        if "fg3m" in game_logs.columns and "fg3a" in game_logs.columns:
            stats["fg3_pct"] = game_logs["fg3m"].sum() / max(game_logs["fg3a"].sum(), 1)
        
        return stats
    
    # ========================================================================
    # Team Stats Methods
    # ========================================================================
    
    def get_team_game_logs(
        self,
        team: str,
        season: int = None
    ) -> pd.DataFrame:
        """
        Get team's game-by-game results.
        
        Returns DataFrame with game results.
        """
        from nba_api.stats.endpoints import teamgamelog
        
        team_id = get_team_id(team)
        if not team_id:
            raise ValueError(f"Unknown team: {team}")
        
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        
        cache_key = f"team_log_{team_id}_{season}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        self._rate_limit()
        
        try:
            log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=self._season_str(season)
            )
            df = log.get_data_frames()[0]
            
            df["game_date"] = pd.to_datetime(df["GAME_DATE"])
            df["team_id"] = team_id
            df["season"] = season
            
            self.cache.set(cache_key, df, ttl_hours=6)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch team game log for {team}: {e}")
            raise
    
    # ========================================================================
    # Bulk Player Stats (All Players for a Season)
    # ========================================================================
    
    def get_all_player_stats_for_season(
        self,
        season: int = None
    ) -> pd.DataFrame:
        """
        Get all player boxscore stats for a season.
        
        Uses PlayerGameLogs endpoint for efficiency.
        """
        from nba_api.stats.endpoints import playergamelogs
        
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        
        cache_key = f"all_player_logs_{season}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            logger.info(f"Using cached player logs for {season}")
            return cached
        
        self._rate_limit()
        
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=self._season_str(season)
            )
            df = logs.get_data_frames()[0]
            
            if df.empty:
                return pd.DataFrame()
            
            # Normalize
            df = df.rename(columns={
                "PLAYER_ID": "player_id",
                "PLAYER_NAME": "player_name",
                "TEAM_ID": "team_id",
                "TEAM_ABBREVIATION": "team_abbr",
                "GAME_ID": "game_id",
                "GAME_DATE": "game_date",
                "MIN": "minutes",
                "PTS": "pts",
                "REB": "reb",
                "AST": "ast",
                "FGM": "fgm",
                "FGA": "fga",
                "FG3M": "fg3m",
                "FG3A": "fg3a",
                "FTM": "ftm",
                "FTA": "fta",
            })
            
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["minutes"] = df["minutes"].apply(self._parse_minutes)
            df["season"] = season
            
            self.cache.set(cache_key, df, ttl_hours=12)
            logger.info(f"Fetched {len(df)} player-game records for {season}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch all player logs for {season}: {e}")
            raise
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _parse_minutes(self, val) -> float:
        """Parse minutes from various formats."""
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
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
        """Check if nba_api is available and working."""
        try:
            from nba_api.stats.endpoints import leaguegamefinder
            return True
        except ImportError:
            return False

