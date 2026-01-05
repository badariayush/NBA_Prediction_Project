"""
NBA API Client for fetching season schedules and game results.

This module provides functions to fetch NBA game data from the official NBA API
and normalize it to the project's expected schema.

Usage:
    from src.data.nba_api_client import fetch_season_games, upsert_games_csv
    
    # Fetch 2024 season games
    games_df = fetch_season_games(2024)
    
    # Update local CSV
    upsert_games_csv(games_df, "data/raw/games.csv")
"""

import os
import time
from typing import Optional

import pandas as pd


# ============================================================================
# NBA Team ID to Abbreviation Mapping
# ============================================================================

# Official NBA team IDs from nba_api
NBA_TEAM_ID_TO_ABBREV = {
    1610612737: "ATL",  # Atlanta Hawks
    1610612738: "BOS",  # Boston Celtics
    1610612739: "CLE",  # Cleveland Cavaliers
    1610612740: "NOP",  # New Orleans Pelicans
    1610612741: "CHI",  # Chicago Bulls
    1610612742: "DAL",  # Dallas Mavericks
    1610612743: "DEN",  # Denver Nuggets
    1610612744: "GSW",  # Golden State Warriors
    1610612745: "HOU",  # Houston Rockets
    1610612746: "LAC",  # LA Clippers
    1610612747: "LAL",  # Los Angeles Lakers
    1610612748: "MIA",  # Miami Heat
    1610612749: "MIL",  # Milwaukee Bucks
    1610612750: "MIN",  # Minnesota Timberwolves
    1610612751: "BKN",  # Brooklyn Nets
    1610612752: "NYK",  # New York Knicks
    1610612753: "ORL",  # Orlando Magic
    1610612754: "IND",  # Indiana Pacers
    1610612755: "PHI",  # Philadelphia 76ers
    1610612756: "PHX",  # Phoenix Suns
    1610612757: "POR",  # Portland Trail Blazers
    1610612758: "SAC",  # Sacramento Kings
    1610612759: "SAS",  # San Antonio Spurs
    1610612760: "OKC",  # Oklahoma City Thunder
    1610612761: "TOR",  # Toronto Raptors
    1610612762: "UTA",  # Utah Jazz
    1610612763: "MEM",  # Memphis Grizzlies
    1610612764: "WAS",  # Washington Wizards
    1610612765: "DET",  # Detroit Pistons
    1610612766: "CHA",  # Charlotte Hornets
}

# Reverse mapping for lookups
NBA_ABBREV_TO_TEAM_ID = {v: k for k, v in NBA_TEAM_ID_TO_ABBREV.items()}

# Some historical team abbreviations that may appear
TEAM_ABBREV_ALIASES = {
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn Nets
    "NOH": "NOP",  # New Orleans Hornets -> Pelicans
    "NOK": "NOP",  # New Orleans/Oklahoma City Hornets
    "SEA": "OKC",  # Seattle SuperSonics -> Thunder
    "VAN": "MEM",  # Vancouver Grizzlies -> Memphis
    "CHH": "CHA",  # Charlotte Hornets (old)
    "WSB": "WAS",  # Washington Bullets
}


def _check_nba_api_installed() -> None:
    """Check if nba_api is installed, raise helpful error if not."""
    try:
        import nba_api  # noqa: F401
    except ImportError:
        raise ImportError(
            "nba_api package is not installed.\n"
            "Install it with: pip install nba_api\n"
            "Then retry your command."
        )


def team_id_to_abbrev(team_id: int) -> str:
    """Convert NBA team ID to standard abbreviation."""
    return NBA_TEAM_ID_TO_ABBREV.get(team_id, f"UNK_{team_id}")


def normalize_team_abbrev(abbrev: str) -> str:
    """Normalize team abbreviation, handling historical aliases."""
    abbrev = abbrev.upper().strip()
    return TEAM_ABBREV_ALIASES.get(abbrev, abbrev)


def normalize_team_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize team codes in a DataFrame to standard abbreviations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with home_team and away_team columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized team abbreviations
    """
    df = df.copy()
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team_abbrev)
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(normalize_team_abbrev)
    return df


def fetch_season_games(
    season: int,
    season_type: str = "Regular Season",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch all NBA games for a given season from the NBA API.
    
    Parameters
    ----------
    season : int
        Season year (e.g., 2024 for the 2024-25 season)
    season_type : str
        Type of games to fetch: "Regular Season", "Playoffs", "Pre Season"
    verbose : bool
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, season, home_team, away_team, 
        home_score, away_score, game_id
        
    Raises
    ------
    ImportError
        If nba_api is not installed
    """
    _check_nba_api_installed()
    
    from nba_api.stats.endpoints import LeagueGameLog
    from nba_api.stats.static import teams as nba_teams
    
    # NBA API uses format like "2024-25" for the 2024 season
    season_str = f"{season}-{str(season + 1)[-2:]}"
    
    if verbose:
        print(f"Fetching {season_type} games for {season_str} season...")
    
    # Add delay to avoid rate limiting
    time.sleep(0.6)
    
    try:
        # Fetch game log for all teams
        game_log = LeagueGameLog(
            season=season_str,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T"  # Team stats
        )
        
        games_df = game_log.get_data_frames()[0]
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch games from NBA API: {e}")
    
    if games_df.empty:
        if verbose:
            print(f"No games found for {season_str} {season_type}")
        return pd.DataFrame(columns=[
            "date", "season", "home_team", "away_team", 
            "home_score", "away_score", "game_id"
        ])
    
    if verbose:
        print(f"Retrieved {len(games_df)} team-game records")
    
    # Parse the data - each game appears twice (once per team)
    # We need to reconstruct home/away from MATCHUP column
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    
    # MATCHUP format: "BOS vs. LAL" (home) or "BOS @ LAL" (away)
    games_df["is_home"] = games_df["MATCHUP"].str.contains(" vs. ")
    
    # Filter to home games only to get one record per game
    home_games = games_df[games_df["is_home"]].copy()
    
    # Extract opponent from matchup
    home_games["away_team"] = home_games["MATCHUP"].str.extract(r"vs\. (\w+)")[0]
    
    # Build the result DataFrame
    result = pd.DataFrame({
        "date": home_games["GAME_DATE"].dt.strftime("%Y-%m-%d"),
        "season": season,
        "home_team": home_games["TEAM_ABBREVIATION"],
        "away_team": home_games["away_team"],
        "home_score": home_games["PTS"],
        "away_score": None,  # Will fill from away games
        "game_id": home_games["GAME_ID"],
    })
    
    # Get away team scores by joining back
    away_games = games_df[~games_df["is_home"]][["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
    away_games = away_games.rename(columns={"PTS": "away_score_lookup"})
    
    result = result.merge(
        away_games[["GAME_ID", "away_score_lookup"]], 
        left_on="game_id", 
        right_on="GAME_ID", 
        how="left"
    )
    result["away_score"] = result["away_score_lookup"]
    result = result.drop(columns=["GAME_ID", "away_score_lookup"])
    
    # Normalize team codes
    result = normalize_team_codes(result)
    
    # Sort by date
    result = result.sort_values("date").reset_index(drop=True)
    
    if verbose:
        print(f"Processed {len(result)} games")
        date_range = f"{result['date'].min()} to {result['date'].max()}"
        print(f"Date range: {date_range}")
    
    return result


def fetch_season_schedule(
    season: int,
    include_results: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch the season schedule, optionally including game results.
    
    This is a convenience wrapper around fetch_season_games that 
    can also fetch future scheduled games.
    
    Parameters
    ----------
    season : int
        Season year (e.g., 2024 for 2024-25 season)
    include_results : bool
        If True, include scores for completed games
    verbose : bool
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Schedule DataFrame
    """
    games_df = fetch_season_games(season, verbose=verbose)
    
    if not include_results:
        games_df = games_df.drop(columns=["home_score", "away_score"], errors="ignore")
    
    return games_df


def upsert_games_csv(
    df_new: pd.DataFrame,
    path: str = "data/raw/games.csv",
    verbose: bool = True
) -> int:
    """
    Upsert new games into the games CSV file.
    
    Adds new games and updates existing ones (matched by game_id or 
    date + home_team + away_team).
    
    Parameters
    ----------
    df_new : pd.DataFrame
        New games to add/update
    path : str
        Path to games CSV file
    verbose : bool
        Print progress messages
        
    Returns
    -------
    int
        Number of new games added
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    if df_new.empty:
        if verbose:
            print("No new games to add.")
        return 0
    
    # Ensure required columns
    required_cols = ["date", "season", "home_team", "away_team"]
    missing = set(required_cols) - set(df_new.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Load existing data if file exists
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        if verbose:
            print(f"Loaded {len(df_existing)} existing games from {path}")
    else:
        df_existing = pd.DataFrame()
        if verbose:
            print(f"Creating new games file: {path}")
    
    # Create match key for deduplication
    df_new = df_new.copy()
    df_new["_match_key"] = (
        df_new["date"].astype(str) + "_" + 
        df_new["home_team"].astype(str) + "_" + 
        df_new["away_team"].astype(str)
    )
    
    if not df_existing.empty:
        df_existing["_match_key"] = (
            df_existing["date"].astype(str) + "_" + 
            df_existing["home_team"].astype(str) + "_" + 
            df_existing["away_team"].astype(str)
        )
        
        # Find truly new games
        existing_keys = set(df_existing["_match_key"])
        new_games = df_new[~df_new["_match_key"].isin(existing_keys)].copy()
        
        # Remove match key columns
        df_existing = df_existing.drop(columns=["_match_key"])
        new_games = new_games.drop(columns=["_match_key"])
        
        # Combine
        df_combined = pd.concat([df_existing, new_games], ignore_index=True)
    else:
        new_games = df_new.drop(columns=["_match_key"])
        df_combined = new_games.copy()
    
    # Sort by date
    df_combined = df_combined.sort_values("date").reset_index(drop=True)
    
    # Save
    df_combined.to_csv(path, index=False)
    
    n_new = len(new_games)
    if verbose:
        print(f"Added {n_new} new games. Total: {len(df_combined)} games.")
        print(f"Saved to: {path}")
    
    return n_new


def get_available_seasons() -> list[int]:
    """
    Get list of NBA seasons available from the API.
    
    Returns
    -------
    list[int]
        List of season years (e.g., [2020, 2021, 2022, 2023, 2024])
    """
    # NBA API typically has data from 1996-97 onwards
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # If we're past October, current season has started
    if current_month >= 10:
        latest_season = current_year
    else:
        latest_season = current_year - 1
    
    # Return last 10 seasons
    return list(range(latest_season - 9, latest_season + 1))


# ============================================================================
# CLI for standalone testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA API Client")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 2024)")
    parser.add_argument("--save", action="store_true", help="Save to data/raw/games.csv")
    
    args = parser.parse_args()
    
    print(f"Fetching {args.season} season games...")
    games = fetch_season_games(args.season)
    
    print(f"\nSample of fetched games:")
    print(games.head(10).to_string(index=False))
    
    if args.save:
        upsert_games_csv(games)

