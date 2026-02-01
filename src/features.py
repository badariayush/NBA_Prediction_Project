"""
Feature Engineering Module for NBA Game Prediction.

This module builds player-level rolling stats and aggregates them to team-level
features for game prediction. All features are computed using only data from
BEFORE the game date to prevent leakage.

Key Features:
- Player rolling stats (last N games): PTS, REB, AST, PRA, shooting percentages
- Team aggregation: minutes-weighted averages and shooting efficiency
- Matchup features: home vs away differences

Usage:
    python -m src.features --seasons 2023-24 2024-25 2025-26
    python -m src.features --raw-dir data/raw --out-path data/processed/training_features.csv
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
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

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUTPUT_PATH = "data/processed/training_features.csv"
DEFAULT_PLAYER_FEATURES_PATH = "data/processed/player_rolling_features.csv"

# Rolling windows for player stats
ROLLING_WINDOWS = [5, 10]

# Stats to compute rolling features for
PLAYER_STAT_COLS = ["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN", "TOV", "PF"]


# ============================================================================
# Data Loading
# ============================================================================

def load_games(path: str) -> pd.DataFrame:
    """Load games CSV and parse dates."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    return df


def load_player_boxscores(path: str) -> pd.DataFrame:
    """Load player boxscores CSV and parse dates."""
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    return df


def load_season_data(
    raw_dir: str,
    seasons: Optional[list[str]] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load games and player boxscores for specified seasons.
    
    If seasons is None, loads the combined files.
    """
    if seasons:
        # Load individual season files
        all_games = []
        all_player_box = []
        
        for season in seasons:
            season_file = season.replace("-", "_")
            games_path = os.path.join(raw_dir, f"games_{season_file}.csv")
            player_path = os.path.join(raw_dir, f"player_box_{season_file}.csv")
            
            if os.path.exists(games_path):
                all_games.append(load_games(games_path))
            if os.path.exists(player_path):
                all_player_box.append(load_player_boxscores(player_path))
        
        if not all_games:
            raise FileNotFoundError(f"No game files found in {raw_dir} for seasons {seasons}")
        
        games_df = pd.concat(all_games, ignore_index=True)
        player_box_df = pd.concat(all_player_box, ignore_index=True) if all_player_box else pd.DataFrame()
    else:
        # Load combined files
        games_path = os.path.join(raw_dir, "games.csv")
        player_path = os.path.join(raw_dir, "player_boxscores.csv")
        
        games_df = load_games(games_path)
        player_box_df = load_player_boxscores(player_path) if os.path.exists(player_path) else pd.DataFrame()
    
    return games_df, player_box_df


# ============================================================================
# Player Rolling Features (Pre-Game)
# ============================================================================

def compute_player_rolling_features(
    player_box_df: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Compute rolling features for each player, shifted to avoid leakage.
    
    For each player, computes rolling means over their last N games
    BEFORE the current game (using shift(1)).
    
    Args:
        player_box_df: Player boxscore DataFrame
        windows: Rolling window sizes (e.g., [5, 10])
        
    Returns:
        DataFrame with player rolling features added
    """
    logger.info(f"Computing player rolling features with windows {windows}...")
    
    df = player_box_df.copy()
    
    # Ensure sorted by player and date
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    
    # Add PRA (Points + Rebounds + Assists)
    df["PRA"] = df["PTS"].fillna(0) + df["REB"].fillna(0) + df["AST"].fillna(0)
    
    # Stats to compute rolling features for
    stat_cols = PLAYER_STAT_COLS + ["PRA"]
    available_stats = [c for c in stat_cols if c in df.columns]
    
    for window in windows:
        suffix = f"_L{window}"
        
        for col in available_stats:
            new_col = f"{col}{suffix}"
            
            # Rolling mean with shift(1) to use only PRIOR games
            df[new_col] = (
                df.groupby("player_id")[col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
    
    logger.info(f"Added {len(windows) * len(available_stats)} rolling feature columns")
    return df


# ============================================================================
# Team Aggregation (Per-Game)
# ============================================================================

def aggregate_player_features_to_team(
    player_features_df: pd.DataFrame,
    games_df: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
    min_minutes_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Aggregate player rolling features to team-level for each game.
    
    Uses minutes-weighted averaging for rate stats and sums for counting stats.
    Only includes players who played >= min_minutes_threshold minutes.
    
    Args:
        player_features_df: Player features with rolling stats
        games_df: Games DataFrame
        windows: Rolling windows used
        min_minutes_threshold: Minimum minutes to include player
        
    Returns:
        DataFrame with team-level aggregated features per game
    """
    logger.info("Aggregating player features to team level...")
    
    # Filter to players with meaningful minutes
    df = player_features_df[player_features_df["MIN"] >= min_minutes_threshold].copy()
    
    if df.empty:
        logger.warning("No players with sufficient minutes found")
        return pd.DataFrame()
    
    # Group by game and team
    grouped = df.groupby(["game_id", "team_id"])
    
    team_features = []
    
    for (game_id, team_id), group in grouped:
        row = {"game_id": game_id, "team_id": team_id}
        
        # Get total minutes for weighting
        total_min = group["MIN"].sum()
        if total_min == 0:
            continue
        
        minutes = group["MIN"].values
        weights = minutes / total_min
        
        for window in windows:
            suffix = f"_L{window}"
            
            # Minutes-weighted averages for rate stats
            for stat in ["PTS", "REB", "AST", "PRA"]:
                col = f"{stat}{suffix}"
                if col in group.columns:
                    values = group[col].fillna(0).values
                    row[f"team_{stat.lower()}_per_min{suffix}"] = np.average(values / np.maximum(minutes, 1), weights=weights)
                    row[f"team_{stat.lower()}_avg{suffix}"] = np.average(values, weights=weights)
            
            # Shooting percentages (sum-based)
            for made, att, name in [("FGM", "FGA", "fg"), ("FG3M", "FG3A", "3p"), ("FTM", "FTA", "ft")]:
                made_col = f"{made}{suffix}"
                att_col = f"{att}{suffix}"
                if made_col in group.columns and att_col in group.columns:
                    total_made = (group[made_col].fillna(0) * weights).sum()
                    total_att = (group[att_col].fillna(0) * weights).sum()
                    row[f"team_{name}_pct{suffix}"] = total_made / max(total_att, 1)
            
            # Total projected minutes
            min_col = f"MIN{suffix}"
            if min_col in group.columns:
                row[f"team_min_avg{suffix}"] = group[min_col].fillna(0).mean()
        
        # Number of players
        row["team_n_players"] = len(group)
        
        team_features.append(row)
    
    team_df = pd.DataFrame(team_features)
    logger.info(f"Created {len(team_df)} team-game feature rows")
    
    return team_df


# ============================================================================
# Matchup Feature Building
# ============================================================================

def build_matchup_features(
    games_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Build matchup features for each game with home/away team features and diffs.
    
    Args:
        games_df: Games DataFrame with game_id, home_team_id, away_team_id
        team_features_df: Team-level features per game
        windows: Rolling windows used
        
    Returns:
        DataFrame with matchup features and home_win label
    """
    logger.info("Building matchup features...")
    
    # Get home team features
    home_features = team_features_df.copy()
    home_features = home_features.rename(columns={"team_id": "home_team_id"})
    
    # Rename columns to home_*
    rename_cols = {c: f"home_{c}" for c in home_features.columns 
                   if c not in ["game_id", "home_team_id"]}
    home_features = home_features.rename(columns=rename_cols)
    
    # Get away team features
    away_features = team_features_df.copy()
    away_features = away_features.rename(columns={"team_id": "away_team_id"})
    
    # Rename columns to away_*
    rename_cols = {c: f"away_{c}" for c in away_features.columns 
                   if c not in ["game_id", "away_team_id"]}
    away_features = away_features.rename(columns=rename_cols)
    
    # Merge with games
    result = games_df.copy()
    
    # Need to match by game_id and team_id
    result = result.merge(home_features, on=["game_id", "home_team_id"], how="left")
    result = result.merge(away_features, on=["game_id", "away_team_id"], how="left")
    
    # Compute diff features (home - away)
    feature_cols = [c for c in team_features_df.columns if c not in ["game_id", "team_id"]]
    
    for col in feature_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in result.columns and away_col in result.columns:
            diff_col = f"diff_{col}"
            result[diff_col] = result[home_col] - result[away_col]
    
    logger.info(f"Built matchup features: {len(result)} games, {len(result.columns)} columns")
    
    return result


# ============================================================================
# Legacy Feature Support (Team-Level Rolling Stats)
# ============================================================================

def _games_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert games to long format (one row per team per game)."""
    home_rows = pd.DataFrame({
        "game_idx": df.index,
        "date": df["date"],
        "season": df["season"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": 1,
        "points_for": df["home_score"],
        "points_against": df["away_score"],
    })
    
    away_rows = pd.DataFrame({
        "game_idx": df.index,
        "date": df["date"],
        "season": df["season"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "points_for": df["away_score"],
        "points_against": df["home_score"],
    })
    
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df["point_diff"] = long_df["points_for"] - long_df["points_against"]
    long_df = long_df.sort_values(["date", "game_idx"]).reset_index(drop=True)
    
    return long_df


def _compute_team_rolling_features_legacy(
    long_df: pd.DataFrame, 
    windows: list[int]
) -> pd.DataFrame:
    """Compute team-level rolling features (legacy method)."""
    result = long_df.copy()
    result = result.sort_values(["team", "season", "date"]).reset_index(drop=True)
    
    for window in windows:
        suffix = f"_roll{window}"
        
        for col in ["points_for", "points_against", "point_diff"]:
            col_abbrev = {"points_for": "pf", "points_against": "pa", "point_diff": "pd"}[col]
            new_col = f"{col_abbrev}{suffix}"
            
            result[new_col] = (
                result.groupby(["team", "season"])[col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
    
    return result


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add leakage-free team-level rolling features (legacy support).
    
    This is the original team-level rolling feature method, kept for
    backwards compatibility.
    """
    result = df.copy()
    result["home_win"] = (result["home_score"] > result["away_score"]).astype(int)
    
    long_df = _games_to_long_format(df)
    long_df = _compute_team_rolling_features_legacy(long_df, windows)
    
    # Build feature column list
    roll_cols = []
    for window in windows:
        for stat in ["pf", "pa", "pd"]:
            roll_cols.append(f"{stat}_roll{window}")
    
    # Extract home features
    home_features = long_df[long_df["is_home"] == 1][["game_idx"] + roll_cols].copy()
    home_features = home_features.rename(columns={col: f"home_{col}" for col in roll_cols})
    home_features = home_features.set_index("game_idx")
    
    # Extract away features
    away_features = long_df[long_df["is_home"] == 0][["game_idx"] + roll_cols].copy()
    away_features = away_features.rename(columns={col: f"away_{col}" for col in roll_cols})
    away_features = away_features.set_index("game_idx")
    
    result = result.join(home_features)
    result = result.join(away_features)
    
    # Add diff features
    for window in windows:
        for stat in ["pf", "pa", "pd"]:
            home_col = f"home_{stat}_roll{window}"
            away_col = f"away_{stat}_roll{window}"
            diff_col = f"{stat}_roll{window}_diff"
            result[diff_col] = result[home_col] - result[away_col]
    
    return result


# ============================================================================
# Main Feature Building Pipeline
# ============================================================================

def build_training_dataset(
    seasons: Optional[list[str]] = None,
    raw_dir: str = DEFAULT_RAW_DIR,
    out_path: str = DEFAULT_OUTPUT_PATH,
    use_player_features: bool = True,
    windows: list[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Build complete training dataset with features.
    
    Args:
        seasons: List of seasons (e.g., ["2023-24", "2024-25"])
        raw_dir: Directory with raw data files
        out_path: Output path for training features CSV
        use_player_features: If True, use player-level features; else use legacy team features
        windows: Rolling window sizes
        
    Returns:
        DataFrame with all features and home_win label
    """
    logger.info(f"Building training dataset...")
    logger.info(f"Seasons: {seasons or 'all'}")
    logger.info(f"Raw dir: {raw_dir}")
    logger.info(f"Use player features: {use_player_features}")
    
    # Load data
    games_df, player_box_df = load_season_data(raw_dir, seasons)
    logger.info(f"Loaded {len(games_df)} games, {len(player_box_df)} player-game records")
    
    if use_player_features and not player_box_df.empty:
        # New player-based feature pipeline
        logger.info("\n--- PLAYER FEATURE PIPELINE ---")
        
        # Compute player rolling features
        player_features = compute_player_rolling_features(player_box_df, windows)
        
        # Aggregate to team level
        team_features = aggregate_player_features_to_team(player_features, games_df, windows)
        
        if not team_features.empty:
            # Build matchup features
            result = build_matchup_features(games_df, team_features, windows)
        else:
            logger.warning("Team aggregation failed, falling back to legacy features")
            result = add_rolling_features(games_df, [1, 2])
    else:
        # Legacy team-level feature pipeline
        logger.info("\n--- LEGACY TEAM FEATURE PIPELINE ---")
        result = add_rolling_features(games_df, [1, 2])
    
    # Ensure home_win label exists
    if "home_win" not in result.columns:
        result["home_win"] = (result["home_score"] > result["away_score"]).astype(int)
    
    # Sort by date
    result = result.sort_values("date").reset_index(drop=True)
    
    # Report stats
    n_missing = result.isnull().any(axis=1).sum()
    logger.info(f"\nDataset shape: {result.shape}")
    logger.info(f"Rows with missing values: {n_missing}")
    logger.info(f"Date range: {result['date'].min()} to {result['date'].max()}")
    logger.info(f"Seasons: {sorted(result['season'].unique())}")
    
    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(f"\nSaved training features to {out_path}")
    
    return result


def build_feature_table(raw_path: str = "data/raw/games.csv") -> pd.DataFrame:
    """
    Build feature table from raw games (legacy interface).
    
    This function is kept for backwards compatibility with existing code.
    """
    logger.info("Building feature table (legacy mode)...")
    
    df = load_games(raw_path)
    logger.info(f"Loaded {len(df)} games from {raw_path}")
    
    windows = [1, 2]
    df = add_rolling_features(df, windows)
    logger.info(f"Added rolling features with windows {windows}")
    
    n_missing = df.isnull().any(axis=1).sum()
    logger.info(f"Rows with missing values (early season): {n_missing}")
    
    return df


def save_feature_table(
    df: pd.DataFrame, 
    out_path: str = "data/processed/games_features.csv"
) -> None:
    """Save feature table to CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Feature table saved to {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build training features from NBA game and player data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.features --seasons 2023-24 2024-25 2025-26
  python -m src.features --raw-dir data/raw --out-path data/processed/training_features.csv
  python -m src.features --legacy  # Use team-level features only
        """
    )
    
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=None,
        help="List of seasons to include (e.g., 2023-24 2024-25)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help=f"Directory with raw data (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for training features (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy team-level features instead of player features"
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=ROLLING_WINDOWS,
        help=f"Rolling window sizes (default: {ROLLING_WINDOWS})"
    )
    
    args = parser.parse_args()
    
    try:
        feature_df = build_training_dataset(
            seasons=args.seasons,
            raw_dir=args.raw_dir,
            out_path=args.out_path,
            use_player_features=not args.legacy,
            windows=args.windows
        )
        
        logger.info("\nSample of feature table:")
        print(feature_df.head(5).to_string())
        
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        raise


if __name__ == "__main__":
    main()
