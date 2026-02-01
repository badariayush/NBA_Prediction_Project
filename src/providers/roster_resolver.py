"""
Roster Resolver - Determines active roster and expected minutes for a team.

Combines data from multiple providers:
- nba_api for official roster
- BallDontLie as backup
- Basketball-Reference for injuries

Rotation inference uses minutes heuristics since we don't have paid projections.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.providers.nba_api_provider import NbaApiProvider
from src.providers.balldontlie_provider import BallDontLieProvider
from src.providers.injury_provider import BasketballReferenceInjuryProvider, INJURY_MINUTES_MAP
from src.utils.cache import DiskCache
from src.utils.team_map import get_team_id, resolve_team_name
from src.utils.names import normalize_player_name

logger = logging.getLogger(__name__)


class RosterResolver:
    """
    Resolves active roster with expected minutes for a team on a given date.
    
    Determines:
    - Active players (not injured/out)
    - Expected starters (top 5 by minutes)
    - Expected minutes per player
    - Injury adjustments
    """
    
    # Thresholds
    MIN_BENCH_MINUTES = 8.0  # Minimum expected minutes to include player
    MAX_PLAYER_MINUTES = 40.0  # Cap on expected minutes
    
    # Recent form weighting
    RECENT_WEIGHT = 0.7  # Weight for last 5 games
    EXTENDED_WEIGHT = 0.3  # Weight for last 10 games
    
    def __init__(
        self,
        nba_provider: Optional[NbaApiProvider] = None,
        bdl_provider: Optional[BallDontLieProvider] = None,
        injury_provider: Optional[BasketballReferenceInjuryProvider] = None,
        cache: Optional[DiskCache] = None
    ):
        self.nba = nba_provider or NbaApiProvider()
        self.bdl = bdl_provider or BallDontLieProvider()
        self.injury = injury_provider or BasketballReferenceInjuryProvider()
        self.cache = cache or DiskCache(cache_dir="data/cache/rosters")
    
    def get_active_roster(
        self,
        team: str,
        game_date: datetime,
        include_injuries: bool = True
    ) -> pd.DataFrame:
        """
        Get active roster for a team on a specific date.
        
        Args:
            team: Team name or abbreviation
            game_date: Target game date
            include_injuries: Whether to apply injury adjustments
            
        Returns:
            DataFrame with columns:
                player_id, player_name, position, expected_minutes,
                is_starter, injury_status, availability_prob
        """
        team_abbr = resolve_team_name(team)
        if not team_abbr:
            raise ValueError(f"Unknown team: {team}")
        
        # Determine season
        season = game_date.year if game_date.month >= 10 else game_date.year - 1
        
        # Get base roster
        try:
            roster = self.nba.get_team_roster(team_abbr, season)
        except Exception as e:
            logger.warning(f"nba_api roster failed, trying BallDontLie: {e}")
            roster = self._get_roster_from_bdl(team_abbr)
        
        if roster.empty:
            logger.warning(f"Could not get roster for {team_abbr}")
            return pd.DataFrame()
        
        # Get player minutes data
        roster = self._add_minutes_data(roster, game_date, season)
        
        # Apply injury adjustments
        if include_injuries:
            roster = self._apply_injury_adjustments(roster)
        
        # Calculate expected minutes
        roster = self._calculate_expected_minutes(roster)
        
        # Determine starters
        roster = self._determine_starters(roster)
        
        # Filter to active players
        active = roster[roster["expected_minutes"] >= self.MIN_BENCH_MINUTES].copy()
        active = active.sort_values("expected_minutes", ascending=False)
        
        logger.info(f"Resolved {len(active)} active players for {team_abbr} on {game_date.date()}")
        return active
    
    def _get_roster_from_bdl(self, team: str) -> pd.DataFrame:
        """Fallback roster from BallDontLie."""
        try:
            team_id = self.bdl.get_team_id(team)
            if not team_id:
                return pd.DataFrame()
            
            players = self.bdl.get_players(team_id=team_id)
            
            return pd.DataFrame({
                "player_id": players["player_id"],
                "player_name": players["full_name"],
                "position": players["position"],
            })
        except Exception as e:
            logger.error(f"BallDontLie roster failed: {e}")
            return pd.DataFrame()
    
    def _add_minutes_data(
        self,
        roster: pd.DataFrame,
        game_date: datetime,
        season: int
    ) -> pd.DataFrame:
        """Add recent minutes data for each player."""
        roster = roster.copy()
        
        # Initialize columns
        roster["avg_min_L5"] = 0.0
        roster["avg_min_L10"] = 0.0
        roster["games_started_L5"] = 0
        roster["games_played_L5"] = 0
        
        for idx, player in roster.iterrows():
            player_id = player["player_id"]
            
            try:
                # Get recent game logs
                logs = self.nba.get_player_game_logs(player_id, season)
                
                if logs.empty:
                    continue
                
                # Filter to games before target date
                logs = logs[logs["game_date"] < game_date]
                
                if logs.empty:
                    continue
                
                # Last 5 games
                last_5 = logs.head(5)
                if len(last_5) > 0:
                    roster.loc[idx, "avg_min_L5"] = last_5["minutes"].mean()
                    roster.loc[idx, "games_played_L5"] = len(last_5)
                
                # Last 10 games
                last_10 = logs.head(10)
                if len(last_10) > 0:
                    roster.loc[idx, "avg_min_L10"] = last_10["minutes"].mean()
                
            except Exception as e:
                logger.debug(f"Could not get minutes for player {player_id}: {e}")
        
        return roster
    
    def _apply_injury_adjustments(self, roster: pd.DataFrame) -> pd.DataFrame:
        """Apply injury status adjustments to roster."""
        roster = roster.copy()
        
        roster["injury_status"] = "ACTIVE"
        roster["injury_note"] = ""
        roster["availability_prob"] = 1.0
        roster["minutes_factor"] = 1.0
        
        # Fetch injury report
        injuries = self.injury.get_injuries()
        
        if injuries.empty:
            return roster
        
        for idx, player in roster.iterrows():
            player_name = player["player_name"]
            player_norm = normalize_player_name(player_name)
            
            # Check if player is on injury report
            injury_match = injuries[injuries["player_name_normalized"] == player_norm]
            
            if injury_match.empty:
                # Try fuzzy match
                for _, inj in injuries.iterrows():
                    if normalize_player_name(inj["player_name"]) == player_norm:
                        injury_match = pd.DataFrame([inj])
                        break
            
            if not injury_match.empty:
                injury = injury_match.iloc[0]
                status = injury["status"].upper()
                
                roster.loc[idx, "injury_status"] = status
                roster.loc[idx, "injury_note"] = injury.get("note", "")
                
                # Set availability probability and minutes factor
                status_lower = status.lower()
                roster.loc[idx, "minutes_factor"] = INJURY_MINUTES_MAP.get(status_lower, 1.0)
                
                if status == "OUT":
                    roster.loc[idx, "availability_prob"] = 0.0
                elif status == "DOUBTFUL":
                    roster.loc[idx, "availability_prob"] = 0.1
                elif status == "QUESTIONABLE":
                    roster.loc[idx, "availability_prob"] = 0.5
                elif status == "PROBABLE":
                    roster.loc[idx, "availability_prob"] = 0.9
                elif status == "DAY-TO-DAY":
                    roster.loc[idx, "availability_prob"] = 0.7
        
        return roster
    
    def _calculate_expected_minutes(self, roster: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected minutes using weighted average of recent form."""
        roster = roster.copy()
        
        # Weighted average of L5 and L10
        roster["base_expected_minutes"] = (
            self.RECENT_WEIGHT * roster["avg_min_L5"] +
            self.EXTENDED_WEIGHT * roster["avg_min_L10"]
        )
        
        # Apply injury factor
        roster["expected_minutes"] = (
            roster["base_expected_minutes"] * roster["minutes_factor"]
        )
        
        # Cap minutes
        roster["expected_minutes"] = roster["expected_minutes"].clip(0, self.MAX_PLAYER_MINUTES)
        
        return roster
    
    def _determine_starters(self, roster: pd.DataFrame) -> pd.DataFrame:
        """Determine likely starters (top 5 by expected minutes)."""
        roster = roster.copy()
        
        # Sort by expected minutes (only available players)
        available = roster[roster["availability_prob"] > 0].sort_values(
            "expected_minutes", ascending=False
        )
        
        # Top 5 are starters
        starter_ids = available.head(5)["player_id"].tolist()
        roster["is_starter"] = roster["player_id"].isin(starter_ids)
        
        return roster
    
    def get_rotation_summary(
        self,
        team: str,
        game_date: datetime
    ) -> dict:
        """
        Get a summary of team rotation for display.
        
        Returns:
            Dict with starters, bench, injuries, and totals
        """
        roster = self.get_active_roster(team, game_date)
        
        if roster.empty:
            return {
                "team": team,
                "date": game_date.strftime("%Y-%m-%d"),
                "starters": [],
                "bench": [],
                "injuries": [],
                "total_expected_minutes": 0,
            }
        
        starters = roster[roster["is_starter"]].sort_values("expected_minutes", ascending=False)
        bench = roster[~roster["is_starter"]].sort_values("expected_minutes", ascending=False)
        
        injured = roster[roster["injury_status"] != "ACTIVE"]
        
        return {
            "team": resolve_team_name(team),
            "date": game_date.strftime("%Y-%m-%d"),
            "starters": starters[["player_name", "position", "expected_minutes", "injury_status"]].to_dict("records"),
            "bench": bench[["player_name", "position", "expected_minutes", "injury_status"]].to_dict("records"),
            "injuries": injured[["player_name", "injury_status", "injury_note"]].to_dict("records"),
            "total_expected_minutes": roster["expected_minutes"].sum(),
            "n_active_players": len(roster),
        }


def get_roster_resolver() -> RosterResolver:
    """Get a singleton roster resolver instance."""
    global _roster_resolver
    if "_roster_resolver" not in globals():
        _roster_resolver = RosterResolver()
    return _roster_resolver

