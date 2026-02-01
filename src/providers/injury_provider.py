"""
Basketball-Reference Injury Provider - Free injury data scraping.

Scrapes the injury report page from Basketball-Reference.
https://www.basketball-reference.com/friv/injuries.fcgi

Injury statuses:
- OUT: Player will not play
- DOUBTFUL: Very unlikely to play (10% chance)
- QUESTIONABLE: May or may not play (50% scenario-based)
- PROBABLE: Likely to play with slight limitations (90% chance)
- DAY-TO-DAY: Monitored daily
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.cache import DiskCache
from src.utils.names import normalize_player_name, fuzzy_match_name
from src.utils.team_map import resolve_team_name

logger = logging.getLogger(__name__)


# Injury status mapping to expected minutes fraction
INJURY_MINUTES_MAP = {
    "out": 0.0,
    "doubtful": 0.10,
    "questionable": 0.50,  # Scenario-based
    "probable": 0.90,
    "day-to-day": 0.70,
    "active": 1.0,
    "available": 1.0,
}


class BasketballReferenceInjuryProvider:
    """
    Scrapes injury data from Basketball-Reference.
    
    Free source for NBA injury reports.
    """
    
    INJURY_URL = "https://www.basketball-reference.com/friv/injuries.fcgi"
    
    def __init__(self, cache: Optional[DiskCache] = None):
        self.cache = cache or DiskCache(cache_dir="data/cache/injuries")
        self._last_fetch_time = 0
    
    def get_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch current injury report.
        
        Returns:
            DataFrame with columns: player_name, team, status, note, update_date
        """
        cache_key = "bbref_injuries"
        
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        logger.info("Fetching injuries from Basketball-Reference...")
        
        try:
            # Rate limit (be nice to the server)
            elapsed = time.time() - self._last_fetch_time
            if elapsed < 5:
                time.sleep(5 - elapsed)
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(self.INJURY_URL, headers=headers, timeout=30)
            response.raise_for_status()
            
            self._last_fetch_time = time.time()
            
            df = self._parse_injury_page(response.text)
            
            self.cache.set(cache_key, df, ttl_hours=4)  # Cache for 4 hours
            logger.info(f"Fetched {len(df)} injury records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch injuries: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=["player_name", "team", "status", "note", "update_date"])
    
    def _parse_injury_page(self, html: str) -> pd.DataFrame:
        """Parse the injury report HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        
        injuries = []
        
        # Find the injury table
        table = soup.find("table", {"id": "injuries"})
        
        if not table:
            # Try alternative parsing
            table = soup.find("table")
        
        if not table:
            logger.warning("Could not find injury table")
            return pd.DataFrame(columns=["player_name", "team", "status", "note", "update_date"])
        
        tbody = table.find("tbody")
        if not tbody:
            tbody = table
        
        for row in tbody.find_all("tr"):
            cells = row.find_all(["td", "th"])
            
            if len(cells) < 3:
                continue
            
            try:
                # Parse player name
                player_cell = cells[0]
                player_link = player_cell.find("a")
                player_name = player_link.get_text(strip=True) if player_link else player_cell.get_text(strip=True)
                
                # Parse team
                team_cell = cells[1]
                team_link = team_cell.find("a")
                team = team_link.get_text(strip=True) if team_link else team_cell.get_text(strip=True)
                
                # Parse status/note
                if len(cells) >= 4:
                    status_raw = cells[2].get_text(strip=True)
                    note = cells[3].get_text(strip=True)
                else:
                    # Combined status and note
                    combined = cells[2].get_text(strip=True)
                    status_raw, note = self._parse_status_note(combined)
                
                # Normalize status
                status = self._normalize_status(status_raw)
                
                # Resolve team to abbreviation
                team_abbr = resolve_team_name(team) or team
                
                injuries.append({
                    "player_name": player_name,
                    "player_name_normalized": normalize_player_name(player_name),
                    "team": team_abbr,
                    "status": status,
                    "status_raw": status_raw,
                    "note": note,
                    "update_date": datetime.now().strftime("%Y-%m-%d"),
                })
                
            except Exception as e:
                logger.debug(f"Error parsing injury row: {e}")
                continue
        
        return pd.DataFrame(injuries)
    
    def _parse_status_note(self, text: str) -> tuple[str, str]:
        """
        Parse combined status and note text.
        
        Examples:
            "Out (knee)" -> ("Out", "knee")
            "Day-To-Day (rest)" -> ("Day-To-Day", "rest")
        """
        # Try to match pattern "Status (note)"
        match = re.match(r"^([\w\-]+)\s*\((.+)\)$", text)
        if match:
            return match.group(1), match.group(2)
        
        # Check for common statuses
        for status in ["Out", "Doubtful", "Questionable", "Probable", "Day-To-Day"]:
            if text.lower().startswith(status.lower()):
                note = text[len(status):].strip(" -()")
                return status, note
        
        return text, ""
    
    def _normalize_status(self, status: str) -> str:
        """Normalize injury status to standard form."""
        status = status.lower().strip()
        
        if "out" in status:
            return "OUT"
        elif "doubtful" in status:
            return "DOUBTFUL"
        elif "questionable" in status:
            return "QUESTIONABLE"
        elif "probable" in status:
            return "PROBABLE"
        elif "day" in status and "day" in status:
            return "DAY-TO-DAY"
        else:
            return status.upper()
    
    def get_team_injuries(self, team: str) -> pd.DataFrame:
        """
        Get injuries for a specific team.
        
        Args:
            team: Team name or abbreviation
        """
        all_injuries = self.get_injuries()
        
        if all_injuries.empty:
            return all_injuries
        
        team_abbr = resolve_team_name(team)
        if not team_abbr:
            team_abbr = team.upper()
        
        return all_injuries[all_injuries["team"] == team_abbr]
    
    def get_player_injury(self, player_name: str) -> Optional[dict]:
        """
        Get injury status for a specific player.
        
        Args:
            player_name: Player name to search for
            
        Returns:
            Dict with injury info or None if not on injury report
        """
        all_injuries = self.get_injuries()
        
        if all_injuries.empty:
            return None
        
        # Normalize search name
        search_norm = normalize_player_name(player_name)
        
        # Exact match first
        match = all_injuries[all_injuries["player_name_normalized"] == search_norm]
        if not match.empty:
            return match.iloc[0].to_dict()
        
        # Fuzzy match
        matched_name = fuzzy_match_name(
            search_norm,
            all_injuries["player_name_normalized"].tolist(),
            threshold=0.8
        )
        
        if matched_name:
            match = all_injuries[all_injuries["player_name_normalized"] == matched_name]
            if not match.empty:
                return match.iloc[0].to_dict()
        
        return None
    
    def get_minutes_factor(self, player_name: str) -> float:
        """
        Get expected minutes factor for a player based on injury status.
        
        Returns:
            Float 0-1 representing expected fraction of normal minutes
        """
        injury = self.get_player_injury(player_name)
        
        if not injury:
            return 1.0  # Not on injury report = full minutes
        
        status = injury["status"].lower()
        return INJURY_MINUTES_MAP.get(status, 1.0)
    
    def is_player_available(self, player_name: str) -> tuple[bool, float, str]:
        """
        Check if player is likely to be available.
        
        Returns:
            Tuple of (is_available, probability, status_note)
        """
        injury = self.get_player_injury(player_name)
        
        if not injury:
            return True, 1.0, "Active"
        
        status = injury["status"]
        note = injury.get("note", "")
        
        if status == "OUT":
            return False, 0.0, f"OUT - {note}"
        elif status == "DOUBTFUL":
            return False, 0.10, f"DOUBTFUL - {note}"
        elif status == "QUESTIONABLE":
            return True, 0.50, f"QUESTIONABLE - {note}"
        elif status == "PROBABLE":
            return True, 0.90, f"PROBABLE - {note}"
        elif status == "DAY-TO-DAY":
            return True, 0.70, f"DAY-TO-DAY - {note}"
        else:
            return True, 0.80, f"{status} - {note}"


def get_injury_provider() -> BasketballReferenceInjuryProvider:
    """Get a singleton injury provider instance."""
    global _injury_provider
    if "_injury_provider" not in globals():
        _injury_provider = BasketballReferenceInjuryProvider()
    return _injury_provider

