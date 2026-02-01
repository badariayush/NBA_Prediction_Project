"""
NBA Data Pipeline - Pull games and player boxscore data from NBA API.

Usage:
    python -m src.pull_nba_data --seasons 2023-24 2024-25 2025-26
    python -m src.pull_nba_data --seasons 2024-25 --force-refresh
    python -m src.pull_nba_data --start-year 2020 --end-year 2025
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
DEFAULT_OUTDIR = "data/raw"
API_DELAY = 0.6  # Seconds between API requests to avoid rate limiting


# ============================================================================
# Helper Functions
# ============================================================================

def season_str(year: int) -> str:
    """Convert 2024 -> '2024-25' (NBA season string format)."""
    return f"{year}-{str(year + 1)[-2:]}"


def parse_season_str(season: str) -> int:
    """Convert '2024-25' -> 2024 (start year)."""
    return int(season.split("-")[0])


def check_nba_api() -> bool:
    """Check if nba_api is installed."""
    try:
        import nba_api  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# Game Data Fetching
# ============================================================================

def fetch_season_games(year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch all games for a season using LeagueGameFinder.
    
    Returns raw dataframe (one row per TEAM per game).
    """
    from nba_api.stats.endpoints import leaguegamefinder
    
    season = season_str(year)
    logger.info(f"Fetching {season} {season_type} games...")
    
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",
        )
        df = gf.get_data_frames()[0]
        time.sleep(API_DELAY)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch games for {season}: {e}")
        raise


def to_game_level(df_team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level rows (2 rows per game) into game-level rows.
    
    Returns: DataFrame with columns:
        date, season, home_team, away_team, home_score, away_score, home_win, game_id
    """
    if df_team_rows.empty:
        return pd.DataFrame()
    
    keep = ["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "PTS"]
    df = df_team_rows[keep].copy()
    
    # Parse season year from SEASON_ID (e.g., 22024 -> 2024)
    df["season"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    
    # HOME/AWAY from MATCHUP: "LAL vs. BOS" -> LAL is home, "LAL @ BOS" -> LAL is away
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)
    
    # Build home side
    home = df[df["is_home"]].rename(columns={
        "TEAM_ABBREVIATION": "home_team",
        "TEAM_ID": "home_team_id",
        "PTS": "home_score",
    })[["GAME_ID", "GAME_DATE", "season", "home_team", "home_team_id", "home_score"]]
    
    # Build away side
    away = df[~df["is_home"]].rename(columns={
        "TEAM_ABBREVIATION": "away_team",
        "TEAM_ID": "away_team_id",
        "PTS": "away_score",
    })[["GAME_ID", "away_team", "away_team_id", "away_score"]]
    
    games = home.merge(away, on="GAME_ID", how="inner")
    
    # Clean columns
    games = games.rename(columns={"GAME_DATE": "date", "GAME_ID": "game_id"})
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    
    # Order and sort
    cols = ["game_id", "date", "season", "home_team", "home_team_id", "away_team", 
            "away_team_id", "home_score", "away_score", "home_win"]
    games = games[[c for c in cols if c in games.columns]]
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    
    return games


# ============================================================================
# Player Boxscore Fetching
# ============================================================================

def fetch_player_boxscores_for_season(year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch player boxscore stats for all games in a season.
    
    Uses PlayerGameLogs endpoint for efficiency.
    
    Returns: DataFrame with columns including:
        game_id, game_date, team_id, team_abbreviation, player_id, player_name,
        MIN, PTS, REB, AST, FGM, FGA, FG3M, FG3A, FTM, FTA, TOV, PF, PLUS_MINUS
    """
    from nba_api.stats.endpoints import playergamelogs
    
    season = season_str(year)
    logger.info(f"Fetching {season} {season_type} player boxscores...")
    
    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
        )
        df = logs.get_data_frames()[0]
        time.sleep(API_DELAY)
        
        if df.empty:
            logger.warning(f"No player boxscores found for {season}")
            return pd.DataFrame()
        
        # Normalize column names
        df = df.rename(columns={
            "SEASON_YEAR": "season_str",
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbreviation",
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
        })
        
        # Parse minutes to float (format: "MM:SS" or just float)
        if "MIN" in df.columns:
            df["MIN"] = df["MIN"].apply(_parse_minutes)
        
        # Add season year
        df["season"] = year
        
        # Select and order columns
        stat_cols = ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", 
                     "FTM", "FTA", "TOV", "PF", "PLUS_MINUS", "STL", "BLK"]
        base_cols = ["game_id", "game_date", "season", "team_id", "team_abbreviation", 
                     "player_id", "player_name"]
        
        available_stat_cols = [c for c in stat_cols if c in df.columns]
        df = df[base_cols + available_stat_cols]
        
        # Sort
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} player-game records for {season}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch player boxscores for {season}: {e}")
        raise


def _parse_minutes(val) -> float:
    """Parse minutes from various formats (MM:SS string or numeric)."""
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


# ============================================================================
# Caching and File I/O
# ============================================================================

def get_games_path(season: str, outdir: str) -> str:
    """Get path for games CSV file."""
    return os.path.join(outdir, f"games_{season.replace('-', '_')}.csv")


def get_player_box_path(season: str, outdir: str) -> str:
    """Get path for player boxscore CSV file."""
    return os.path.join(outdir, f"player_box_{season.replace('-', '_')}.csv")


def load_cached_games(season: str, outdir: str) -> Optional[pd.DataFrame]:
    """Load cached games if available."""
    path = get_games_path(season, outdir)
    if os.path.exists(path):
        logger.info(f"Loading cached games from {path}")
        return pd.read_csv(path)
    return None


def load_cached_player_box(season: str, outdir: str) -> Optional[pd.DataFrame]:
    """Load cached player boxscores if available."""
    path = get_player_box_path(season, outdir)
    if os.path.exists(path):
        logger.info(f"Loading cached player boxscores from {path}")
        return pd.read_csv(path)
    return None


def save_games(df: pd.DataFrame, season: str, outdir: str) -> str:
    """Save games DataFrame to CSV."""
    os.makedirs(outdir, exist_ok=True)
    path = get_games_path(season, outdir)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} games to {path}")
    return path


def save_player_box(df: pd.DataFrame, season: str, outdir: str) -> str:
    """Save player boxscores DataFrame to CSV."""
    os.makedirs(outdir, exist_ok=True)
    path = get_player_box_path(season, outdir)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} player-game records to {path}")
    return path


# ============================================================================
# Main Pipeline
# ============================================================================

def pull_season_data(
    season: str,
    outdir: str = DEFAULT_OUTDIR,
    force_refresh: bool = False,
    include_playoffs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull all data for a single season (games + player boxscores).
    
    Args:
        season: Season string (e.g., "2024-25")
        outdir: Output directory for cached files
        force_refresh: If True, re-download even if cached
        include_playoffs: If True, also fetch playoff games
        
    Returns:
        Tuple of (games_df, player_box_df)
    """
    year = parse_season_str(season)
    
    # --- Games ---
    games_df = None if force_refresh else load_cached_games(season, outdir)
    
    if games_df is None:
        df_reg = fetch_season_games(year, "Regular Season")
        games_reg = to_game_level(df_reg)
        games_reg["season_type"] = "regular"
        
        all_games = [games_reg]
        
        if include_playoffs:
            df_po = fetch_season_games(year, "Playoffs")
            games_po = to_game_level(df_po)
            if not games_po.empty:
                games_po["season_type"] = "playoffs"
                all_games.append(games_po)
        
        games_df = pd.concat(all_games, ignore_index=True)
        games_df = games_df.drop_duplicates(subset=["game_id"])
        save_games(games_df, season, outdir)
    
    # --- Player Boxscores ---
    player_box_df = None if force_refresh else load_cached_player_box(season, outdir)
    
    if player_box_df is None:
        player_box_reg = fetch_player_boxscores_for_season(year, "Regular Season")
        
        all_player_box = [player_box_reg]
        
        if include_playoffs:
            player_box_po = fetch_player_boxscores_for_season(year, "Playoffs")
            if not player_box_po.empty:
                all_player_box.append(player_box_po)
        
        player_box_df = pd.concat(all_player_box, ignore_index=True)
        player_box_df = player_box_df.drop_duplicates(subset=["game_id", "player_id"])
        save_player_box(player_box_df, season, outdir)
    
    return games_df, player_box_df


def pull_all_seasons(
    seasons: list[str],
    outdir: str = DEFAULT_OUTDIR,
    force_refresh: bool = False,
    include_playoffs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull data for multiple seasons and combine.
    
    Returns:
        Tuple of (combined_games_df, combined_player_box_df)
    """
    all_games = []
    all_player_box = []
    
    for season in seasons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing season: {season}")
        logger.info(f"{'='*60}")
        
        try:
            games_df, player_box_df = pull_season_data(
                season, outdir, force_refresh, include_playoffs
            )
            all_games.append(games_df)
            all_player_box.append(player_box_df)
        except Exception as e:
            logger.error(f"Failed to process season {season}: {e}")
            continue
    
    if not all_games:
        raise RuntimeError("No data was successfully pulled for any season")
    
    combined_games = pd.concat(all_games, ignore_index=True)
    combined_player_box = pd.concat(all_player_box, ignore_index=True)
    
    # Remove duplicates
    combined_games = combined_games.drop_duplicates(subset=["game_id"])
    combined_player_box = combined_player_box.drop_duplicates(subset=["game_id", "player_id"])
    
    return combined_games, combined_player_box


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pull NBA games and player boxscore data from NBA API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pull_nba_data --seasons 2023-24 2024-25 2025-26
  python -m src.pull_nba_data --seasons 2024-25 --force-refresh
  python -m src.pull_nba_data --start-year 2020 --end-year 2025
        """
    )
    
    # Season selection (two methods)
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=None,
        help="List of seasons in format YYYY-YY (e.g., 2024-25)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year (alternative to --seasons)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year (alternative to --seasons)"
    )
    
    # Options
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if cached files exist"
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Also fetch playoff games"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})"
    )
    
    args = parser.parse_args()
    
    # Check nba_api
    if not check_nba_api():
        logger.error("nba_api package not installed. Run: pip install nba_api")
        sys.exit(1)
    
    # Determine seasons to fetch
    if args.seasons:
        seasons = args.seasons
    elif args.start_year and args.end_year:
        seasons = [season_str(y) for y in range(args.start_year, args.end_year + 1)]
    else:
        seasons = DEFAULT_SEASONS
        logger.info(f"Using default seasons: {seasons}")
    
    logger.info(f"Seasons to process: {seasons}")
    logger.info(f"Output directory: {args.outdir}")
    logger.info(f"Force refresh: {args.force_refresh}")
    
    # Pull data
    try:
        combined_games, combined_player_box = pull_all_seasons(
            seasons=seasons,
            outdir=args.outdir,
            force_refresh=args.force_refresh,
            include_playoffs=args.include_playoffs
        )
        
        # Also save combined files
        combined_games_path = os.path.join(args.outdir, "games.csv")
        combined_games.to_csv(combined_games_path, index=False)
        logger.info(f"\nSaved combined games to {combined_games_path}")
        
        combined_player_box_path = os.path.join(args.outdir, "player_boxscores.csv")
        combined_player_box.to_csv(combined_player_box_path, index=False)
        logger.info(f"Saved combined player boxscores to {combined_player_box_path}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total games: {len(combined_games)}")
        logger.info(f"Total player-game records: {len(combined_player_box)}")
        logger.info(f"Date range: {combined_games['date'].min()} to {combined_games['date'].max()}")
        logger.info(f"Seasons: {sorted(combined_games['season'].unique())}")
        
        # Show sample
        logger.info("\nSample games:")
        print(combined_games.tail(5).to_string(index=False))
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
