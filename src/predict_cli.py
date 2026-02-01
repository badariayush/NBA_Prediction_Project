"""
Interactive CLI for NBA Game Prediction.

Predicts game outcomes using player-based rolling features.
Supports both new player-level features and legacy team-level features.

Usage:
    python -m src.predict_cli
    python -m src.predict_cli --home BOS --away NYK --date 2025-01-15
    python -m src.predict_cli --legacy  # Use old team-level features
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import joblib
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

MODEL_PATH = "models/logreg.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"
FEATURES_PATH = "data/processed/games_features.csv"
PLAYER_BOX_PATH = "data/raw/player_boxscores.csv"
GAMES_PATH = "data/raw/games.csv"

# Rolling windows (must match training)
ROLLING_WINDOWS = [5, 10]

# Minimum minutes threshold for active players
MIN_MINUTES_ACTIVE = 10.0

# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# NBA Team mappings
NBA_TEAM_ABBREVS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

# Reverse mapping
NBA_TEAM_NAMES = {v.upper(): k for k, v in NBA_TEAM_ABBREVS.items()}


# ============================================================================
# Data Loading
# ============================================================================

def load_model(path: str = MODEL_PATH):
    """Load trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Run 'python -m src.train' first."
        )
    return joblib.load(path)


def load_feature_columns(path: str = FEATURE_COLS_PATH) -> list[str]:
    """Load expected feature columns."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_player_boxscores(path: str = PLAYER_BOX_PATH) -> pd.DataFrame:
    """Load player boxscore data."""
    if not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_games_data(path: str = GAMES_PATH) -> pd.DataFrame:
    """Load games data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Games data not found at '{path}'")
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_legacy_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load legacy feature table."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features not found at '{path}'")
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================================
# Team Resolution
# ============================================================================

def resolve_team(team_input: str, available_teams: set) -> Optional[str]:
    """
    Resolve team input to standard abbreviation.
    
    Accepts: abbreviation (BOS), full name (Boston Celtics), or partial match.
    """
    team_input = team_input.strip().upper()
    
    # Direct abbreviation match
    if team_input in available_teams:
        return team_input
    
    # Check standard abbreviations
    if team_input in NBA_TEAM_ABBREVS:
        return team_input
    
    # Check full name
    if team_input in NBA_TEAM_NAMES:
        return NBA_TEAM_NAMES[team_input]
    
    # Partial match
    for abbrev, name in NBA_TEAM_ABBREVS.items():
        if team_input in name.upper() or team_input in abbrev:
            return abbrev
    
    return None


def get_available_teams(games_df: pd.DataFrame = None, player_box_df: pd.DataFrame = None) -> set:
    """Get set of available team abbreviations from data."""
    teams = set()
    
    if games_df is not None and not games_df.empty:
        if "home_team" in games_df.columns:
            teams.update(games_df["home_team"].unique())
        if "away_team" in games_df.columns:
            teams.update(games_df["away_team"].unique())
    
    if player_box_df is not None and not player_box_df.empty:
        if "team_abbreviation" in player_box_df.columns:
            teams.update(player_box_df["team_abbreviation"].unique())
    
    return teams


# ============================================================================
# Player-Based Feature Building
# ============================================================================

def get_active_players(
    player_box_df: pd.DataFrame,
    team_abbrev: str,
    before_date: pd.Timestamp,
    n_games: int = 3,
    min_minutes: float = MIN_MINUTES_ACTIVE
) -> pd.DataFrame:
    """
    Get likely active players for a team based on recent games.
    
    Returns players who played >= min_minutes in any of the team's
    last n_games before the target date.
    """
    # Filter to team's games before the date
    team_games = player_box_df[
        (player_box_df["team_abbreviation"] == team_abbrev) &
        (player_box_df["game_date"] < before_date)
    ].copy()
    
    if team_games.empty:
        return pd.DataFrame()
    
    # Get the most recent n games
    recent_game_dates = team_games["game_date"].drop_duplicates().nlargest(n_games)
    recent_games = team_games[team_games["game_date"].isin(recent_game_dates)]
    
    # Find players with meaningful minutes
    active_players = recent_games.groupby("player_id").agg({
        "player_name": "first",
        "MIN": "mean",
        "game_date": "count"
    }).reset_index()
    
    active_players = active_players[active_players["MIN"] >= min_minutes]
    active_players = active_players.rename(columns={"game_date": "n_recent_games"})
    
    return active_players


def compute_player_rolling_stats(
    player_box_df: pd.DataFrame,
    player_id: int,
    before_date: pd.Timestamp,
    windows: list[int] = ROLLING_WINDOWS
) -> dict:
    """
    Compute rolling stats for a single player up to (not including) the target date.
    """
    player_games = player_box_df[
        (player_box_df["player_id"] == player_id) &
        (player_box_df["game_date"] < before_date)
    ].sort_values("game_date", ascending=False)
    
    if player_games.empty:
        return {}
    
    # Add PRA
    player_games = player_games.copy()
    player_games["PRA"] = (
        player_games["PTS"].fillna(0) + 
        player_games["REB"].fillna(0) + 
        player_games["AST"].fillna(0)
    )
    
    stats = {}
    stat_cols = ["PTS", "REB", "AST", "PRA", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN"]
    
    for window in windows:
        suffix = f"_L{window}"
        recent = player_games.head(window)
        
        for col in stat_cols:
            if col in recent.columns:
                stats[f"{col}{suffix}"] = recent[col].mean()
    
    # Also include latest game minutes for weighting
    stats["latest_MIN"] = player_games.iloc[0]["MIN"] if len(player_games) > 0 else 0
    
    return stats


def build_team_features_for_prediction(
    player_box_df: pd.DataFrame,
    team_abbrev: str,
    game_date: pd.Timestamp,
    windows: list[int] = ROLLING_WINDOWS
) -> dict:
    """
    Build team-level features for prediction by aggregating player rolling stats.
    """
    # Get active players
    active_players = get_active_players(player_box_df, team_abbrev, game_date)
    
    if active_players.empty:
        logger.warning(f"No active players found for {team_abbrev} before {game_date}")
        return {}
    
    # Compute rolling stats for each active player
    player_stats = []
    for _, player in active_players.iterrows():
        stats = compute_player_rolling_stats(
            player_box_df, player["player_id"], game_date, windows
        )
        if stats:
            stats["player_id"] = player["player_id"]
            stats["player_name"] = player["player_name"]
            player_stats.append(stats)
    
    if not player_stats:
        logger.warning(f"Could not compute stats for any player on {team_abbrev}")
        return {}
    
    stats_df = pd.DataFrame(player_stats)
    
    # Aggregate to team level (minutes-weighted)
    team_features = {}
    total_min = stats_df["latest_MIN"].sum()
    
    if total_min > 0:
        weights = stats_df["latest_MIN"].values / total_min
    else:
        weights = np.ones(len(stats_df)) / len(stats_df)
    
    for window in windows:
        suffix = f"_L{window}"
        
        # Weighted averages
        for stat in ["PTS", "REB", "AST", "PRA"]:
            col = f"{stat}{suffix}"
            if col in stats_df.columns:
                values = stats_df[col].fillna(0).values
                min_col = f"MIN{suffix}"
                if min_col in stats_df.columns:
                    mins = stats_df[min_col].fillna(1).values
                    team_features[f"team_{stat.lower()}_per_min{suffix}"] = np.average(
                        values / np.maximum(mins, 1), weights=weights
                    )
                team_features[f"team_{stat.lower()}_avg{suffix}"] = np.average(values, weights=weights)
        
        # Shooting percentages
        for made, att, name in [("FGM", "FGA", "fg"), ("FG3M", "FG3A", "3p"), ("FTM", "FTA", "ft")]:
            made_col = f"{made}{suffix}"
            att_col = f"{att}{suffix}"
            if made_col in stats_df.columns and att_col in stats_df.columns:
                total_made = (stats_df[made_col].fillna(0) * weights).sum()
                total_att = (stats_df[att_col].fillna(0) * weights).sum()
                team_features[f"team_{name}_pct{suffix}"] = total_made / max(total_att, 1)
    
    team_features["n_active_players"] = len(stats_df)
    
    return team_features


def build_prediction_features(
    player_box_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    expected_features: list[str] = None,
    windows: list[int] = ROLLING_WINDOWS
) -> tuple[pd.DataFrame, dict]:
    """
    Build feature vector for a matchup prediction.
    
    Returns (feature_df, info_dict)
    """
    info = {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "method": "player_based",
        "home_n_players": 0,
        "away_n_players": 0,
    }
    
    # Build home team features
    home_features = build_team_features_for_prediction(
        player_box_df, home_team, game_date, windows
    )
    info["home_n_players"] = home_features.get("n_active_players", 0)
    
    # Build away team features
    away_features = build_team_features_for_prediction(
        player_box_df, away_team, game_date, windows
    )
    info["away_n_players"] = away_features.get("n_active_players", 0)
    
    # Compute diff features
    features = {}
    
    feature_keys = set(home_features.keys()) | set(away_features.keys())
    feature_keys = {k for k in feature_keys if k != "n_active_players"}
    
    for key in feature_keys:
        home_val = home_features.get(key, 0)
        away_val = away_features.get(key, 0)
        features[f"home_{key}"] = home_val
        features[f"away_{key}"] = away_val
        features[f"diff_{key}"] = home_val - away_val
    
    # Match expected features if provided
    if expected_features:
        final_features = {}
        for col in expected_features:
            if col in features:
                final_features[col] = features[col]
            else:
                # Try to find a matching feature
                final_features[col] = 0.0
                logger.debug(f"Feature '{col}' not found, using 0.0")
        features = final_features
    
    feature_df = pd.DataFrame([features])
    
    return feature_df, info


# ============================================================================
# Legacy Feature Building
# ============================================================================

def build_legacy_prediction_features(
    features_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    """
    Build features using legacy team-level rolling stats.
    """
    info = {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "method": "legacy_team_level",
    }
    
    # Find most recent stats for each team
    def get_latest_team_stats(team: str, is_home: bool):
        if is_home:
            team_games = features_df[
                (features_df["home_team"] == team) &
                (features_df["date"] < game_date)
            ].sort_values("date", ascending=False)
        else:
            team_games = features_df[
                (features_df["away_team"] == team) &
                (features_df["date"] < game_date)
            ].sort_values("date", ascending=False)
        
        if team_games.empty:
            # Try other side
            other_col = "away_team" if is_home else "home_team"
            team_games = features_df[
                (features_df[other_col] == team) &
                (features_df["date"] < game_date)
            ].sort_values("date", ascending=False)
        
        return team_games.iloc[0] if len(team_games) > 0 else None
    
    home_row = get_latest_team_stats(home_team, True)
    away_row = get_latest_team_stats(away_team, False)
    
    if home_row is None or away_row is None:
        logger.warning("Insufficient historical data for legacy features")
        return pd.DataFrame(), info
    
    # Build features
    features = {}
    for window in [1, 2]:
        for stat in ["pf", "pa", "pd"]:
            home_col = f"home_{stat}_roll{window}"
            away_col = f"away_{stat}_roll{window}"
            diff_col = f"{stat}_roll{window}_diff"
            
            home_val = home_row.get(home_col, 0) if home_row is not None else 0
            away_val = away_row.get(away_col, 0) if away_row is not None else 0
            
            features[diff_col] = home_val - away_val
    
    info["home_source"] = f"{home_team} game on {home_row['date'].strftime('%Y-%m-%d')}" if home_row is not None else "N/A"
    info["away_source"] = f"{away_team} game on {away_row['date'].strftime('%Y-%m-%d')}" if away_row is not None else "N/A"
    
    return pd.DataFrame([features]), info


# ============================================================================
# Prediction
# ============================================================================

def predict_game(
    model,
    features: pd.DataFrame,
    expected_cols: list[str] = None
) -> tuple[float, float]:
    """
    Make prediction and return (home_win_prob, away_win_prob).
    """
    if expected_cols:
        # Ensure columns are in correct order
        for col in expected_cols:
            if col not in features.columns:
                features[col] = 0.0
        features = features[expected_cols]
    
    # Handle any remaining NaNs
    features = features.fillna(0.0)
    
    home_prob = model.predict_proba(features)[0, 1]
    away_prob = 1 - home_prob
    
    return home_prob, away_prob


# ============================================================================
# Display
# ============================================================================

def print_prediction_result(
    home_team: str,
    away_team: str,
    game_date: str,
    home_prob: float,
    away_prob: float,
    info: dict
):
    """Print formatted prediction result."""
    print()
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 NBA GAME PREDICTION{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print()
    
    print(f"  {Colors.BOLD}Matchup:{Colors.RESET}  {away_team} @ {home_team}")
    print(f"  {Colors.BOLD}Date:{Colors.RESET}     {game_date}")
    print()
    
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  WIN PROBABILITIES{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print()
    
    # Home probability
    home_color = Colors.GREEN if home_prob > 0.5 else Colors.DIM
    print(f"    {Colors.BOLD}{home_team}{Colors.RESET} (home):  {home_color}{home_prob:.1%}{Colors.RESET}")
    
    # Away probability
    away_color = Colors.GREEN if away_prob > 0.5 else Colors.DIM
    print(f"    {Colors.BOLD}{away_team}{Colors.RESET} (away):  {away_color}{away_prob:.1%}{Colors.RESET}")
    print()
    
    # Predicted winner
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    winner = home_team if home_prob > 0.5 else away_team
    role = "home" if home_prob > 0.5 else "away"
    confidence = abs(home_prob - 0.5) * 2
    
    if confidence >= 0.3:
        conf_label = "High confidence"
        conf_color = Colors.GREEN
    elif confidence >= 0.15:
        conf_label = "Moderate confidence"
        conf_color = Colors.CYAN
    else:
        conf_label = "Low confidence"
        conf_color = Colors.YELLOW
    
    print(f"  {Colors.BOLD}Predicted Winner:{Colors.RESET} {Colors.BOLD}{winner}{Colors.RESET} ({role})")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {conf_color}{conf_label} ({confidence:.0%}){Colors.RESET}")
    print()
    
    # Method info
    print(f"  {Colors.DIM}Method: {info.get('method', 'N/A')}{Colors.RESET}")
    if "home_n_players" in info:
        print(f"  {Colors.DIM}Active players: {home_team}={info['home_n_players']}, {away_team}={info['away_n_players']}{Colors.RESET}")
    
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print()


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode(model, expected_cols, player_box_df, features_df, available_teams):
    """Run interactive prediction mode."""
    print()
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 Interactive NBA Game Prediction{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}Enter team codes (e.g., BOS, LAL) and date (YYYY-MM-DD){Colors.RESET}")
    print()
    
    # Get home team
    while True:
        home_input = input("  Enter HOME team: ").strip()
        home_team = resolve_team(home_input, available_teams)
        if home_team:
            break
        print(f"    {Colors.YELLOW}⚠ Team not found. Try: BOS, LAL, NYK, etc.{Colors.RESET}")
    
    # Get away team
    while True:
        away_input = input("  Enter AWAY team: ").strip()
        away_team = resolve_team(away_input, available_teams)
        if away_team:
            if away_team != home_team:
                break
            print(f"    {Colors.YELLOW}⚠ Away team must be different from home team{Colors.RESET}")
        else:
            print(f"    {Colors.YELLOW}⚠ Team not found. Try: BOS, LAL, NYK, etc.{Colors.RESET}")
    
    # Get date
    while True:
        date_input = input("  Enter game date (YYYY-MM-DD): ").strip()
        try:
            game_date = pd.to_datetime(date_input)
            break
        except:
            print(f"    {Colors.YELLOW}⚠ Invalid date format. Use YYYY-MM-DD{Colors.RESET}")
    
    print()
    print(f"  {Colors.DIM}Computing prediction for {away_team} @ {home_team} on {game_date.strftime('%Y-%m-%d')}...{Colors.RESET}")
    
    return home_team, away_team, game_date


# ============================================================================
# Main
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Game Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.predict_cli
  python -m src.predict_cli --home BOS --away NYK --date 2025-01-15
  python -m src.predict_cli --legacy
        """
    )
    
    parser.add_argument("--home", type=str, help="Home team abbreviation")
    parser.add_argument("--away", type=str, help="Away team abbreviation")
    parser.add_argument("--date", type=str, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy team-level features")
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {e}")
        sys.exit(1)
    
    # Load expected feature columns
    expected_cols = load_feature_columns()
    
    # Load data
    player_box_df = load_player_boxscores()
    features_df = None
    
    # Determine if we should use legacy mode
    use_legacy = args.legacy or player_box_df.empty
    
    if use_legacy:
        try:
            features_df = load_legacy_features()
            logger.info("Using legacy team-level features")
        except FileNotFoundError as e:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {e}")
            sys.exit(1)
    
    # Get available teams
    if not player_box_df.empty:
        available_teams = get_available_teams(player_box_df=player_box_df)
    elif features_df is not None:
        available_teams = get_available_teams(games_df=features_df)
    else:
        available_teams = set(NBA_TEAM_ABBREVS.keys())
    
    # Get matchup details
    if args.home and args.away and args.date:
        # Command-line mode
        home_team = resolve_team(args.home, available_teams)
        away_team = resolve_team(args.away, available_teams)
        
        if not home_team:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid home team: {args.home}")
            sys.exit(1)
        if not away_team:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid away team: {args.away}")
            sys.exit(1)
        
        try:
            game_date = pd.to_datetime(args.date)
        except:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid date format: {args.date}")
            sys.exit(1)
    else:
        # Interactive mode
        home_team, away_team, game_date = interactive_mode(
            model, expected_cols, player_box_df, features_df, available_teams
        )
    
    # Build features
    if use_legacy:
        feature_df, info = build_legacy_prediction_features(
            features_df, home_team, away_team, game_date
        )
    else:
        feature_df, info = build_prediction_features(
            player_box_df, home_team, away_team, game_date, expected_cols
        )
    
    if feature_df.empty:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Could not build features for this matchup")
        print(f"  Try a more recent date or use --legacy mode")
        sys.exit(1)
    
    # Make prediction
    try:
        home_prob, away_prob = predict_game(model, feature_df, expected_cols)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Prediction failed: {e}")
        sys.exit(1)
    
    # Display result
    print_prediction_result(
        home_team, away_team, game_date.strftime("%Y-%m-%d"),
        home_prob, away_prob, info
    )


if __name__ == "__main__":
    main()
