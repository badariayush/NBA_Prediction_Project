"""
Combined NBA Game Prediction Tool.

This module combines logistic regression model predictions with Monte Carlo
simulation-derived season context to produce ensemble win probabilities.

The combined probability is:
    p_final = alpha * p_logreg + (1 - alpha) * p_mc

Where:
    - p_logreg: Direct model prediction based on rolling stats
    - p_mc: Season-context probability derived from Monte Carlo projected wins

Usage:
    python -m src.predict_combined --home BOS --away NYK --date 2024-11-05
    python -m src.predict_combined --home LAL --away GSW --date 2024-12-01 --alpha 0.8
    python -m src.predict_combined --home MIA --away PHI --n_sims 20000

Author: NBA Prediction Project
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "models/logreg.pkl"
FEATURES_PATH = "data/processed/games_features.csv"

REQUIRED_DIFF_COLS = [
    "pf_roll1_diff",
    "pa_roll1_diff",
    "pd_roll1_diff",
    "pf_roll2_diff",
    "pa_roll2_diff",
    "pd_roll2_diff",
]

# ANSI color codes for terminal output
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


# ============================================================================
# Helper Functions (reused from predict_cli.py patterns)
# ============================================================================

def load_model(path: str = MODEL_PATH):
    """Load the trained logistic regression model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Run 'python -m src.train' first."
        )
    return joblib.load(path)


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load the historical features table."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Features file not found at '{path}'. "
            "Run 'python -m src.features' first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_all_teams(df: pd.DataFrame) -> set:
    """Get all unique team abbreviations from the dataset."""
    return set(df["home_team"].unique()) | set(df["away_team"].unique())


def get_team_latest_stats(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp
) -> tuple[pd.Series | None, str | None]:
    """
    Get the most recent rolling stats for a team before a given date.
    
    Returns (stats_series, source_date_str) or (None, None) if not found.
    No data leakage: only uses games strictly before the input date.
    """
    team_games = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)  # Strict inequality - no leakage
    ].sort_values("date", ascending=False)
    
    if len(team_games) == 0:
        return None, None
    
    latest_game = team_games.iloc[0]
    game_date = latest_game["date"].strftime("%Y-%m-%d")
    
    if latest_game["home_team"] == team:
        stats = {
            "pf_roll1": latest_game.get("home_pf_roll1"),
            "pa_roll1": latest_game.get("home_pa_roll1"),
            "pd_roll1": latest_game.get("home_pd_roll1"),
            "pf_roll2": latest_game.get("home_pf_roll2"),
            "pa_roll2": latest_game.get("home_pa_roll2"),
            "pd_roll2": latest_game.get("home_pd_roll2"),
        }
    else:
        stats = {
            "pf_roll1": latest_game.get("away_pf_roll1"),
            "pa_roll1": latest_game.get("away_pa_roll1"),
            "pd_roll1": latest_game.get("away_pd_roll1"),
            "pf_roll2": latest_game.get("away_pf_roll2"),
            "pa_roll2": latest_game.get("away_pa_roll2"),
            "pd_roll2": latest_game.get("away_pd_roll2"),
        }
    
    return pd.Series(stats), game_date


def compute_logreg_probability(
    model,
    features_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp
) -> tuple[float, dict]:
    """
    Compute home win probability using the logistic regression model.
    
    Parameters
    ----------
    model : sklearn model
        Trained logistic regression model
    features_df : pd.DataFrame
        Historical features data
    home_team : str
        Home team abbreviation
    away_team : str
        Away team abbreviation
    game_date : pd.Timestamp
        Game date (only data before this date is used)
        
    Returns
    -------
    tuple[float, dict]
        - p_home_win probability
        - info dict with computation details
    """
    info = {
        "method": None,
        "home_source": None,
        "away_source": None,
        "is_fallback": False,
    }
    
    # Get latest stats for each team (strictly before game date)
    home_stats, home_date = get_team_latest_stats(features_df, home_team, game_date)
    away_stats, away_date = get_team_latest_stats(features_df, away_team, game_date)
    
    # Check if we have valid stats
    if home_stats is None or away_stats is None:
        info["method"] = "fallback"
        info["is_fallback"] = True
        if home_stats is None:
            info["home_source"] = f"No data for {home_team} before {game_date.strftime('%Y-%m-%d')}"
        if away_stats is None:
            info["away_source"] = f"No data for {away_team} before {game_date.strftime('%Y-%m-%d')}"
        return 0.5, info
    
    # Check for NaN values
    if home_stats.isna().any() or away_stats.isna().any():
        info["method"] = "fallback"
        info["is_fallback"] = True
        info["home_source"] = "Stats contain NaN values"
        info["away_source"] = "Stats contain NaN values"
        return 0.5, info
    
    info["method"] = "computed_from_teams"
    info["home_source"] = f"{home_team}'s game on {home_date}"
    info["away_source"] = f"{away_team}'s game on {away_date}"
    
    # Compute diff features (home - away)
    features = {
        "pf_roll1_diff": home_stats["pf_roll1"] - away_stats["pf_roll1"],
        "pa_roll1_diff": home_stats["pa_roll1"] - away_stats["pa_roll1"],
        "pd_roll1_diff": home_stats["pd_roll1"] - away_stats["pd_roll1"],
        "pf_roll2_diff": home_stats["pf_roll2"] - away_stats["pf_roll2"],
        "pa_roll2_diff": home_stats["pa_roll2"] - away_stats["pa_roll2"],
        "pd_roll2_diff": home_stats["pd_roll2"] - away_stats["pd_roll2"],
    }
    
    X = pd.DataFrame([features])[REQUIRED_DIFF_COLS]
    p_home_win = model.predict_proba(X)[0, 1]
    
    return p_home_win, info


# ============================================================================
# Monte Carlo Season Context
# ============================================================================

def fetch_schedule_for_mc(
    season: int,
    before_date: pd.Timestamp,
    n_games: int = 100
) -> pd.DataFrame:
    """
    Fetch schedule for Monte Carlo simulation.
    
    Only includes games strictly before the specified date to avoid leakage.
    
    Parameters
    ----------
    season : int
        Season year (e.g., 2024 for 2024-25)
    before_date : pd.Timestamp
        Only use games before this date
    n_games : int
        Target number of recent games to include
        
    Returns
    -------
    pd.DataFrame
        Schedule DataFrame with date, home_team, away_team
    """
    try:
        from src.data.nba_api_client import fetch_season_games
    except ImportError:
        # Try to use cached data
        games_path = "data/raw/games.csv"
        if os.path.exists(games_path):
            df = pd.read_csv(games_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["season"] == season]
            df = df[df["date"] < before_date]
            df = df.sort_values("date", ascending=False).head(n_games)
            return df[["date", "home_team", "away_team"]].sort_values("date")
        raise ImportError("nba_api not available and no cached games found.")
    
    # Fetch from API
    games_df = fetch_season_games(season, verbose=False)
    
    if games_df.empty:
        raise ValueError(f"No games found for {season} season")
    
    games_df["date"] = pd.to_datetime(games_df["date"])
    
    # Filter to games before the target date
    games_df = games_df[games_df["date"] < before_date]
    
    # Take the most recent n_games
    games_df = games_df.sort_values("date", ascending=False).head(n_games)
    games_df = games_df.sort_values("date")  # Re-sort chronologically
    
    return games_df[["date", "home_team", "away_team"]]


def add_game_probs_for_schedule(
    schedule_df: pd.DataFrame,
    model,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add p_home_win to schedule using the trained model.
    
    Uses only data before each game's date to avoid leakage.
    """
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    schedule_df = schedule_df.sort_values("date").reset_index(drop=True)
    
    probabilities = []
    
    for _, row in schedule_df.iterrows():
        p_home, _ = compute_logreg_probability(
            model, features_df, 
            row["home_team"], row["away_team"], 
            row["date"]
        )
        probabilities.append(p_home)
    
    schedule_df["p_home_win"] = probabilities
    return schedule_df


def compute_mc_probability(
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    season: int,
    model,
    features_df: pd.DataFrame,
    n_sims: int = 10000,
    n_games: int = 100,
    seed: int = 42
) -> tuple[float, dict]:
    """
    Compute Monte Carlo season-context win probability.
    
    Uses projected win means from MC simulation to derive team strength,
    then converts to a head-to-head probability.
    
    Parameters
    ----------
    home_team : str
        Home team abbreviation
    away_team : str  
        Away team abbreviation
    game_date : pd.Timestamp
        Game date (only data before this is used)
    season : int
        Season year
    model : sklearn model
        Trained model for computing game probabilities
    features_df : pd.DataFrame
        Historical features
    n_sims : int
        Number of Monte Carlo simulations
    n_games : int
        Number of recent games to simulate
    seed : int
        Random seed
        
    Returns
    -------
    tuple[float, dict]
        - p_home_win from MC context
        - info dict with computation details
    """
    from src.sim.win_distribution import simulate_win_distribution, summarize_distributions
    
    info = {
        "n_games_simulated": 0,
        "n_sims": n_sims,
        "home_mean_wins": None,
        "away_mean_wins": None,
        "method": "mc_projected_strength",
    }
    
    try:
        # Fetch schedule for MC simulation (only games before target date)
        schedule_df = fetch_schedule_for_mc(season, game_date, n_games)
        
        if len(schedule_df) < 10:
            info["method"] = "insufficient_data"
            return 0.5, info
        
        info["n_games_simulated"] = len(schedule_df)
        
        # Add probabilities to schedule
        schedule_df = add_game_probs_for_schedule(schedule_df, model, features_df)
        
        # Run Monte Carlo simulation
        wins_df = simulate_win_distribution(schedule_df, n_sims=n_sims, seed=seed)
        
        # Get summary statistics
        summary_df = summarize_distributions(wins_df, win_threshold=50)
        
        # Extract mean wins for each team
        home_row = summary_df[summary_df["team"] == home_team]
        away_row = summary_df[summary_df["team"] == away_team]
        
        if len(home_row) == 0 or len(away_row) == 0:
            # One of the teams wasn't in the simulated schedule
            info["method"] = "team_not_in_schedule"
            return 0.5, info
        
        home_mean = home_row["mean_wins"].values[0]
        away_mean = away_row["mean_wins"].values[0]
        
        info["home_mean_wins"] = home_mean
        info["away_mean_wins"] = away_mean
        
        # Convert projected wins to probability
        # Using a simple ratio-based approach with home court adjustment
        total_projected = home_mean + away_mean
        
        if total_projected == 0:
            p_home = 0.5
        else:
            # Base probability from relative strength
            p_home_base = home_mean / total_projected
            
            # Apply small home court adjustment (typical NBA home advantage ~3-4%)
            home_court_boost = 0.03
            p_home = min(0.95, max(0.05, p_home_base + home_court_boost))
        
        return p_home, info
        
    except Exception as e:
        info["method"] = f"error: {str(e)}"
        return 0.5, info


# ============================================================================
# Combined Prediction
# ============================================================================

def predict_combined(
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    alpha: float = 0.7,
    season: Optional[int] = None,
    n_sims: int = 10000,
    n_mc_games: int = 100,
    seed: int = 42
) -> dict:
    """
    Generate combined prediction using logistic regression and Monte Carlo.
    
    Parameters
    ----------
    home_team : str
        Home team abbreviation
    away_team : str
        Away team abbreviation  
    game_date : pd.Timestamp
        Game date
    alpha : float
        Weight for logistic regression (1-alpha for MC), default 0.7
    season : int, optional
        Season year, inferred from date if not provided
    n_sims : int
        Number of Monte Carlo simulations
    n_mc_games : int
        Number of games to include in MC simulation
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Prediction results with all probabilities and metadata
    """
    # Normalize team names
    home_team = home_team.strip().upper()
    away_team = away_team.strip().upper()
    
    # Infer season from date if not provided
    if season is None:
        # NBA season starts in October
        if game_date.month >= 10:
            season = game_date.year
        else:
            season = game_date.year - 1
    
    # Load model and features
    model = load_model()
    features_df = load_features()
    
    # Validate teams
    all_teams = get_all_teams(features_df)
    if home_team not in all_teams:
        raise ValueError(f"Home team '{home_team}' not found. Valid teams: {sorted(all_teams)}")
    if away_team not in all_teams:
        raise ValueError(f"Away team '{away_team}' not found. Valid teams: {sorted(all_teams)}")
    if home_team == away_team:
        raise ValueError("Home and away teams cannot be the same.")
    
    # Compute logistic regression probability
    p_logreg, logreg_info = compute_logreg_probability(
        model, features_df, home_team, away_team, game_date
    )
    
    # Compute Monte Carlo probability
    p_mc, mc_info = compute_mc_probability(
        home_team, away_team, game_date, season,
        model, features_df,
        n_sims=n_sims, n_games=n_mc_games, seed=seed
    )
    
    # Combine probabilities
    p_final = alpha * p_logreg + (1 - alpha) * p_mc
    
    # Build result
    result = {
        "home_team": home_team,
        "away_team": away_team,
        "date": game_date.strftime("%Y-%m-%d"),
        "season": f"{season}-{str(season + 1)[-2:]}",
        "p_logreg": p_logreg,
        "p_mc": p_mc,
        "p_final": p_final,
        "alpha": alpha,
        "logreg_info": logreg_info,
        "mc_info": mc_info,
        "predicted_winner": home_team if p_final > 0.5 else away_team,
        "confidence": abs(p_final - 0.5) * 2,  # 0-1 scale
    }
    
    return result


# ============================================================================
# Display Functions
# ============================================================================

def print_prediction(result: dict) -> None:
    """Print prediction results in a clean, professional format."""
    
    home = result["home_team"]
    away = result["away_team"]
    date = result["date"]
    p_logreg = result["p_logreg"]
    p_mc = result["p_mc"]
    p_final = result["p_final"]
    alpha = result["alpha"]
    
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 NBA COMBINED PREDICTION{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()
    
    # Matchup header
    print(f"  {Colors.BOLD}Matchup:{Colors.RESET}  {away} @ {home}")
    print(f"  {Colors.BOLD}Date:{Colors.RESET}     {date}")
    print(f"  {Colors.BOLD}Season:{Colors.RESET}   {result['season']}")
    print()
    
    # Probabilities table
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📊 WIN PROBABILITIES{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    # Header row
    print(f"  {'Method':<25}{'Home Win':>15}{'Away Win':>15}")
    print(f"  {'-' * 55}")
    
    # Logistic Regression row
    logreg_home = f"{p_logreg:.1%}"
    logreg_away = f"{1 - p_logreg:.1%}"
    is_fallback = result["logreg_info"].get("is_fallback", False)
    fallback_note = f" {Colors.YELLOW}(fallback){Colors.RESET}" if is_fallback else ""
    print(f"  {'Logistic Regression':<25}{logreg_home:>15}{logreg_away:>15}{fallback_note}")
    
    # Monte Carlo row
    mc_home = f"{p_mc:.1%}"
    mc_away = f"{1 - p_mc:.1%}"
    mc_method = result["mc_info"].get("method", "")
    mc_note = ""
    if "error" in mc_method or "insufficient" in mc_method:
        mc_note = f" {Colors.YELLOW}(limited data){Colors.RESET}"
    print(f"  {'Monte Carlo Context':<25}{mc_home:>15}{mc_away:>15}{mc_note}")
    
    print(f"  {'-' * 55}")
    
    # Final combined row
    final_home = f"{p_final:.1%}"
    final_away = f"{1 - p_final:.1%}"
    
    # Color code based on strength
    if p_final >= 0.6:
        home_color = Colors.GREEN
        away_color = Colors.DIM
    elif p_final <= 0.4:
        home_color = Colors.DIM
        away_color = Colors.GREEN
    else:
        home_color = Colors.CYAN
        away_color = Colors.CYAN
    
    print(f"  {Colors.BOLD}{'COMBINED (Final)':<25}{Colors.RESET}"
          f"{home_color}{final_home:>15}{Colors.RESET}"
          f"{away_color}{final_away:>15}{Colors.RESET}")
    print()
    
    # Prediction result
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  🎯 PREDICTION{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    winner = result["predicted_winner"]
    confidence = result["confidence"]
    
    if confidence >= 0.3:
        conf_label = "High confidence"
        conf_color = Colors.GREEN
    elif confidence >= 0.15:
        conf_label = "Moderate confidence"
        conf_color = Colors.CYAN
    else:
        conf_label = "Low confidence (close game)"
        conf_color = Colors.YELLOW
    
    role = "home" if winner == home else "away"
    print(f"  Predicted winner: {Colors.BOLD}{winner}{Colors.RESET} ({role})")
    print(f"  Confidence: {conf_color}{conf_label} ({confidence:.0%}){Colors.RESET}")
    print()
    
    # Explanation
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📝 HOW THIS WAS COMPUTED{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    print(f"  The final probability combines two models:")
    print()
    print(f"    p_final = {alpha:.2f} × p_logreg + {1-alpha:.2f} × p_mc")
    print(f"            = {alpha:.2f} × {p_logreg:.3f} + {1-alpha:.2f} × {p_mc:.3f}")
    print(f"            = {alpha * p_logreg:.3f} + {(1-alpha) * p_mc:.3f}")
    print(f"            = {Colors.BOLD}{p_final:.3f}{Colors.RESET}")
    print()
    
    # Model details
    print(f"  {Colors.DIM}Logistic Regression:{Colors.RESET}")
    print(f"    • Uses rolling stats (points for/against) from each team's recent games")
    print(f"    • Data source: {result['logreg_info'].get('home_source', 'N/A')}")
    print()
    
    print(f"  {Colors.DIM}Monte Carlo Context:{Colors.RESET}")
    mc_info = result["mc_info"]
    n_games = mc_info.get("n_games_simulated", 0)
    n_sims = mc_info.get("n_sims", 0)
    home_mean = mc_info.get("home_mean_wins")
    away_mean = mc_info.get("away_mean_wins")
    
    print(f"    • Simulated {n_games} games × {n_sims:,} iterations")
    if home_mean is not None and away_mean is not None:
        print(f"    • Projected wins: {home} = {home_mean:.1f}, {away} = {away_mean:.1f}")
    print()
    
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()


# ============================================================================
# CLI
# ============================================================================

def validate_team_code(team: str) -> tuple[bool, str]:
    """
    Validate team code format (3 uppercase letters).
    
    Returns (is_valid, normalized_code or error_message)
    """
    team = team.strip().upper()
    if len(team) != 3:
        return False, "Team code must be exactly 3 letters"
    if not team.isalpha():
        return False, "Team code must contain only letters"
    return True, team


def validate_date_format(date_str: str) -> tuple[bool, pd.Timestamp | str]:
    """
    Validate date format (YYYY-MM-DD).
    
    Returns (is_valid, parsed_date or error_message)
    """
    import re
    date_str = date_str.strip()
    
    # Check format with regex
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return False, "Date must be in YYYY-MM-DD format (e.g., 2024-11-05)"
    
    try:
        parsed = pd.to_datetime(date_str)
        return True, parsed
    except Exception:
        return False, "Invalid date. Use format YYYY-MM-DD (e.g., 2024-11-05)"


def prompt_team(prompt_text: str, role: str) -> str:
    """Prompt user for team code with validation."""
    while True:
        team = input(f"  {prompt_text}: ").strip()
        if not team:
            print(f"    {Colors.YELLOW}⚠ Please enter a team code{Colors.RESET}")
            continue
        
        is_valid, result = validate_team_code(team)
        if is_valid:
            return result
        else:
            print(f"    {Colors.YELLOW}⚠ {result}{Colors.RESET}")


def prompt_date() -> pd.Timestamp:
    """Prompt user for date with validation."""
    while True:
        date_str = input("  Enter game date (YYYY-MM-DD): ").strip()
        if not date_str:
            print(f"    {Colors.YELLOW}⚠ Please enter a date{Colors.RESET}")
            continue
        
        is_valid, result = validate_date_format(date_str)
        if is_valid:
            return result
        else:
            print(f"    {Colors.YELLOW}⚠ {result}{Colors.RESET}")


def print_interactive_header() -> None:
    """Print header for interactive mode."""
    print()
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 Interactive NBA Game Prediction{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}Enter matchup details below. Team codes are 3 letters (e.g., BOS, LAL).{Colors.RESET}")
    print()


def main() -> None:
    """CLI entry point for combined prediction."""
    parser = argparse.ArgumentParser(
        description="NBA Combined Prediction (Logistic Regression + Monte Carlo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With all arguments (non-interactive):
  python -m src.predict_combined --home BOS --away NYK --date 2024-11-05
  python -m src.predict_combined --home LAL --away GSW --date 2024-12-01 --alpha 0.8
  
  # Interactive mode (prompts for missing arguments):
  python -m src.predict_combined
  python -m src.predict_combined --alpha 0.8 --n_sims 20000
        """
    )
    
    parser.add_argument(
        "--home",
        type=str,
        default=None,
        help="Home team abbreviation (e.g., BOS). Prompted if not provided."
    )
    parser.add_argument(
        "--away",
        type=str,
        default=None,
        help="Away team abbreviation (e.g., NYK). Prompted if not provided."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date in YYYY-MM-DD format. Prompted if not provided."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for logistic regression (default: 0.7, MC gets 1-alpha)"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (e.g., 2024 for 2024-25). Inferred from date if not provided."
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)"
    )
    parser.add_argument(
        "--n_mc_games",
        type=int,
        default=100,
        help="Number of games to include in MC simulation (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Check if we need interactive mode
    needs_interactive = args.home is None or args.away is None or args.date is None
    
    if needs_interactive:
        print_interactive_header()
    
    # Get home team
    if args.home is not None:
        is_valid, result = validate_team_code(args.home)
        if not is_valid:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid home team: {result}")
            sys.exit(1)
        home_team = result
    else:
        home_team = prompt_team("Enter HOME team code (e.g., BOS)", "home")
    
    # Get away team
    if args.away is not None:
        is_valid, result = validate_team_code(args.away)
        if not is_valid:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid away team: {result}")
            sys.exit(1)
        away_team = result
    else:
        away_team = prompt_team("Enter AWAY team code (e.g., NYK)", "away")
    
    # Get date
    if args.date is not None:
        is_valid, result = validate_date_format(args.date)
        if not is_valid:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {result}")
            sys.exit(1)
        game_date = result
    else:
        game_date = prompt_date()
    
    if needs_interactive:
        print()
        print(f"  {Colors.DIM}Processing: {away_team} @ {home_team} on {game_date.strftime('%Y-%m-%d')}...{Colors.RESET}")
    
    # Validate alpha
    if not 0 <= args.alpha <= 1:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Alpha must be between 0 and 1")
        sys.exit(1)
    
    try:
        # Run prediction
        result = predict_combined(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            alpha=args.alpha,
            season=args.season,
            n_sims=args.n_sims,
            n_mc_games=args.n_mc_games,
            seed=args.seed
        )
        
        # Print results
        print_prediction(result)
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

