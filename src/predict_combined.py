"""
Combined NBA Game Prediction Tool.

This module combines logistic regression model predictions with Monte Carlo
simulation-derived season context to produce ensemble win probabilities.

NOW USES PLAYER-BASED FEATURES:
- Aggregates player rolling stats (PTS, REB, AST, shooting %) to team level
- Minutes-weighted averaging for more accurate team strength

The combined probability is:
    p_final = alpha * p_logreg + (1 - alpha) * p_mc

Where:
    - p_logreg: Direct model prediction based on player rolling stats
    - p_mc: Season-context probability derived from Monte Carlo projected wins

Usage:
    python -m src.predict_combined --home BOS --away NYK --date 2025-01-15
    python -m src.predict_combined --home LAL --away GSW --date 2025-01-20 --alpha 0.8
    python -m src.predict_combined  # Interactive mode

Author: NBA Prediction Project
"""

import argparse
import json
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
FEATURE_COLS_PATH = "models/feature_cols.json"
PLAYER_BOX_PATH = "data/raw/player_boxscores.csv"
GAMES_PATH = "data/raw/games.csv"
LEGACY_FEATURES_PATH = "data/processed/games_features.csv"

# Rolling windows (must match training)
ROLLING_WINDOWS = [5, 10]

# Minimum minutes threshold for active players
MIN_MINUTES_ACTIVE = 10.0

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


# ============================================================================
# Data Loading
# ============================================================================

def load_model(path: str = MODEL_PATH):
    """Load the trained logistic regression model."""
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
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_legacy_features(path: str = LEGACY_FEATURES_PATH) -> pd.DataFrame:
    """Load legacy feature table for fallback."""
    if not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_available_teams(player_box_df: pd.DataFrame, games_df: pd.DataFrame) -> set:
    """Get set of available team abbreviations."""
    teams = set()
    
    if not player_box_df.empty and "team_abbreviation" in player_box_df.columns:
        teams.update(player_box_df["team_abbreviation"].unique())
    
    if not games_df.empty:
        if "home_team" in games_df.columns:
            teams.update(games_df["home_team"].unique())
        if "away_team" in games_df.columns:
            teams.update(games_df["away_team"].unique())
    
    return teams


# ============================================================================
# Player-Based Feature Building (from predict_cli.py)
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
    team_games = player_box_df[
        (player_box_df["team_abbreviation"] == team_abbrev) &
        (player_box_df["game_date"] < before_date)
    ].copy()
    
    if team_games.empty:
        return pd.DataFrame()
    
    recent_game_dates = team_games["game_date"].drop_duplicates().nlargest(n_games)
    recent_games = team_games[team_games["game_date"].isin(recent_game_dates)]
    
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
    """Compute rolling stats for a single player up to the target date."""
    player_games = player_box_df[
        (player_box_df["player_id"] == player_id) &
        (player_box_df["game_date"] < before_date)
    ].sort_values("game_date", ascending=False)
    
    if player_games.empty:
        return {}
    
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
    
    stats["latest_MIN"] = player_games.iloc[0]["MIN"] if len(player_games) > 0 else 0
    
    return stats


def build_team_features_for_prediction(
    player_box_df: pd.DataFrame,
    team_abbrev: str,
    game_date: pd.Timestamp,
    windows: list[int] = ROLLING_WINDOWS
) -> dict:
    """Build team-level features by aggregating player rolling stats."""
    active_players = get_active_players(player_box_df, team_abbrev, game_date)
    
    if active_players.empty:
        return {}
    
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
        return {}
    
    stats_df = pd.DataFrame(player_stats)
    
    team_features = {}
    total_min = stats_df["latest_MIN"].sum()
    
    if total_min > 0:
        weights = stats_df["latest_MIN"].values / total_min
    else:
        weights = np.ones(len(stats_df)) / len(stats_df)
    
    for window in windows:
        suffix = f"_L{window}"
        
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
    """Build feature vector for a matchup prediction using player stats."""
    info = {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "method": "player_based",
        "home_n_players": 0,
        "away_n_players": 0,
    }
    
    home_features = build_team_features_for_prediction(
        player_box_df, home_team, game_date, windows
    )
    info["home_n_players"] = home_features.get("n_active_players", 0)
    
    away_features = build_team_features_for_prediction(
        player_box_df, away_team, game_date, windows
    )
    info["away_n_players"] = away_features.get("n_active_players", 0)
    
    features = {}
    
    feature_keys = set(home_features.keys()) | set(away_features.keys())
    feature_keys = {k for k in feature_keys if k != "n_active_players"}
    
    for key in feature_keys:
        home_val = home_features.get(key, 0)
        away_val = away_features.get(key, 0)
        features[f"home_{key}"] = home_val
        features[f"away_{key}"] = away_val
        features[f"diff_{key}"] = home_val - away_val
    
    if expected_features:
        final_features = {}
        for col in expected_features:
            if col in features:
                final_features[col] = features[col]
            else:
                final_features[col] = 0.0
        features = final_features
    
    feature_df = pd.DataFrame([features])
    
    return feature_df, info


# ============================================================================
# Legacy Feature Building (Fallback)
# ============================================================================

def build_legacy_prediction_features(
    features_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    """Build features using legacy team-level rolling stats."""
    info = {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "method": "legacy_team_level",
        "home_n_players": "N/A",
        "away_n_players": "N/A",
    }
    
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
            other_col = "away_team" if is_home else "home_team"
            team_games = features_df[
                (features_df[other_col] == team) &
                (features_df["date"] < game_date)
            ].sort_values("date", ascending=False)
        
        return team_games.iloc[0] if len(team_games) > 0 else None
    
    home_row = get_latest_team_stats(home_team, True)
    away_row = get_latest_team_stats(away_team, False)
    
    if home_row is None or away_row is None:
        return pd.DataFrame(), info
    
    features = {}
    for window in [1, 2]:
        for stat in ["pf", "pa", "pd"]:
            home_col = f"home_{stat}_roll{window}"
            away_col = f"away_{stat}_roll{window}"
            diff_col = f"{stat}_roll{window}_diff"
            
            home_val = home_row.get(home_col, 0) if home_row is not None else 0
            away_val = away_row.get(away_col, 0) if away_row is not None else 0
            
            features[diff_col] = home_val - away_val
    
    return pd.DataFrame([features]), info


# ============================================================================
# Logistic Regression Probability
# ============================================================================

def compute_logreg_probability(
    model,
    player_box_df: pd.DataFrame,
    legacy_features_df: pd.DataFrame,
    expected_cols: list[str],
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp
) -> tuple[float, dict]:
    """
    Compute home win probability using the logistic regression model.
    
    Uses player-based features if available, falls back to legacy features.
    """
    info = {
        "method": None,
        "home_source": None,
        "away_source": None,
        "is_fallback": False,
        "home_n_players": 0,
        "away_n_players": 0,
    }
    
    # Try player-based features first
    if not player_box_df.empty:
        feature_df, build_info = build_prediction_features(
            player_box_df, home_team, away_team, game_date, expected_cols
        )
        
        if not feature_df.empty and build_info["home_n_players"] > 0 and build_info["away_n_players"] > 0:
            info["method"] = "player_based"
            info["home_n_players"] = build_info["home_n_players"]
            info["away_n_players"] = build_info["away_n_players"]
            info["home_source"] = f"{home_team}: {build_info['home_n_players']} active players"
            info["away_source"] = f"{away_team}: {build_info['away_n_players']} active players"
            
            # Ensure columns match
            if expected_cols:
                for col in expected_cols:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0
                feature_df = feature_df[expected_cols]
            
            p_home_win = model.predict_proba(feature_df.fillna(0))[0, 1]
            return p_home_win, info
    
    # Fallback to legacy features
    if not legacy_features_df.empty:
        feature_df, build_info = build_legacy_prediction_features(
            legacy_features_df, home_team, away_team, game_date
        )
        
        if not feature_df.empty:
            info["method"] = "legacy_team_level"
            info["is_fallback"] = True
            info["home_source"] = "Team-level rolling stats"
            info["away_source"] = "Team-level rolling stats"
            
            p_home_win = model.predict_proba(feature_df.fillna(0))[0, 1]
            return p_home_win, info
    
    # No data available
    info["method"] = "fallback"
    info["is_fallback"] = True
    info["home_source"] = "No historical data available"
    info["away_source"] = "No historical data available"
    
    return 0.5, info


# ============================================================================
# Monte Carlo Season Context
# ============================================================================

def fetch_schedule_for_mc(
    games_df: pd.DataFrame,
    season: int,
    before_date: pd.Timestamp,
    n_games: int = 100
) -> pd.DataFrame:
    """Fetch schedule for Monte Carlo simulation from cached games."""
    if games_df.empty:
        return pd.DataFrame()
    
    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    if "season" in df.columns:
        df = df[df["season"] == season]
    
    df = df[df["date"] < before_date]
    df = df.sort_values("date", ascending=False).head(n_games)
    
    return df[["date", "home_team", "away_team"]].sort_values("date")


def add_game_probs_for_schedule(
    schedule_df: pd.DataFrame,
    model,
    player_box_df: pd.DataFrame,
    legacy_features_df: pd.DataFrame,
    expected_cols: list[str]
) -> pd.DataFrame:
    """Add p_home_win to schedule using the trained model."""
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    schedule_df = schedule_df.sort_values("date").reset_index(drop=True)
    
    probabilities = []
    
    for _, row in schedule_df.iterrows():
        p_home, _ = compute_logreg_probability(
            model, player_box_df, legacy_features_df, expected_cols,
            row["home_team"], row["away_team"], row["date"]
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
    games_df: pd.DataFrame,
    player_box_df: pd.DataFrame,
    legacy_features_df: pd.DataFrame,
    expected_cols: list[str],
    n_sims: int = 10000,
    n_games: int = 100,
    seed: int = 42
) -> tuple[float, dict]:
    """Compute Monte Carlo season-context win probability."""
    from src.sim.win_distribution import simulate_win_distribution, summarize_distributions
    
    info = {
        "n_games_simulated": 0,
        "n_sims": n_sims,
        "home_mean_wins": None,
        "away_mean_wins": None,
        "method": "mc_projected_strength",
    }
    
    try:
        schedule_df = fetch_schedule_for_mc(games_df, season, game_date, n_games)
        
        if len(schedule_df) < 10:
            info["method"] = "insufficient_data"
            return 0.5, info
        
        info["n_games_simulated"] = len(schedule_df)
        
        schedule_df = add_game_probs_for_schedule(
            schedule_df, model, player_box_df, legacy_features_df, expected_cols
        )
        
        wins_df = simulate_win_distribution(schedule_df, n_sims=n_sims, seed=seed)
        summary_df = summarize_distributions(wins_df, win_threshold=50)
        
        home_row = summary_df[summary_df["team"] == home_team]
        away_row = summary_df[summary_df["team"] == away_team]
        
        if len(home_row) == 0 or len(away_row) == 0:
            info["method"] = "team_not_in_schedule"
            return 0.5, info
        
        home_mean = home_row["mean_wins"].values[0]
        away_mean = away_row["mean_wins"].values[0]
        
        info["home_mean_wins"] = home_mean
        info["away_mean_wins"] = away_mean
        
        total_projected = home_mean + away_mean
        
        if total_projected == 0:
            p_home = 0.5
        else:
            p_home_base = home_mean / total_projected
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
    
    NOW USES PLAYER-BASED FEATURES with fallback to legacy team features.
    """
    home_team = home_team.strip().upper()
    away_team = away_team.strip().upper()
    
    if season is None:
        if game_date.month >= 10:
            season = game_date.year
        else:
            season = game_date.year - 1
    
    # Load model and data
    model = load_model()
    expected_cols = load_feature_columns()
    player_box_df = load_player_boxscores()
    games_df = load_games_data()
    legacy_features_df = load_legacy_features()
    
    # Validate teams
    all_teams = get_available_teams(player_box_df, games_df)
    if not all_teams:
        all_teams = set(NBA_TEAM_ABBREVS.keys())
    
    if home_team not in all_teams:
        raise ValueError(f"Home team '{home_team}' not found. Valid teams: {sorted(all_teams)[:10]}...")
    if away_team not in all_teams:
        raise ValueError(f"Away team '{away_team}' not found. Valid teams: {sorted(all_teams)[:10]}...")
    if home_team == away_team:
        raise ValueError("Home and away teams cannot be the same.")
    
    # Compute logistic regression probability
    p_logreg, logreg_info = compute_logreg_probability(
        model, player_box_df, legacy_features_df, expected_cols,
        home_team, away_team, game_date
    )
    
    # Compute Monte Carlo probability
    p_mc, mc_info = compute_mc_probability(
        home_team, away_team, game_date, season,
        model, games_df, player_box_df, legacy_features_df, expected_cols,
        n_sims=n_sims, n_games=n_mc_games, seed=seed
    )
    
    # Combine probabilities
    p_final = alpha * p_logreg + (1 - alpha) * p_mc
    
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
        "confidence": abs(p_final - 0.5) * 2,
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
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 NBA COMBINED PREDICTION (Player-Based){Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()
    
    print(f"  {Colors.BOLD}Matchup:{Colors.RESET}  {away} @ {home}")
    print(f"  {Colors.BOLD}Date:{Colors.RESET}     {date}")
    print(f"  {Colors.BOLD}Season:{Colors.RESET}   {result['season']}")
    print()
    
    # Probabilities table
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📊 WIN PROBABILITIES{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    print(f"  {'Method':<30}{'Home Win':>15}{'Away Win':>15}")
    print(f"  {'-' * 60}")
    
    # Logistic Regression row
    logreg_home = f"{p_logreg:.1%}"
    logreg_away = f"{1 - p_logreg:.1%}"
    method = result["logreg_info"].get("method", "unknown")
    is_fallback = result["logreg_info"].get("is_fallback", False)
    
    method_label = "Player Rolling Stats" if method == "player_based" else "Team Rolling Stats"
    fallback_note = f" {Colors.YELLOW}(fallback){Colors.RESET}" if is_fallback else ""
    print(f"  {method_label:<30}{logreg_home:>15}{logreg_away:>15}{fallback_note}")
    
    # Monte Carlo row
    mc_home = f"{p_mc:.1%}"
    mc_away = f"{1 - p_mc:.1%}"
    mc_method = result["mc_info"].get("method", "")
    mc_note = ""
    if "error" in mc_method or "insufficient" in mc_method:
        mc_note = f" {Colors.YELLOW}(limited data){Colors.RESET}"
    print(f"  {'Monte Carlo Context':<30}{mc_home:>15}{mc_away:>15}{mc_note}")
    
    print(f"  {'-' * 60}")
    
    # Final combined row
    final_home = f"{p_final:.1%}"
    final_away = f"{1 - p_final:.1%}"
    
    if p_final >= 0.6:
        home_color = Colors.GREEN
        away_color = Colors.DIM
    elif p_final <= 0.4:
        home_color = Colors.DIM
        away_color = Colors.GREEN
    else:
        home_color = Colors.CYAN
        away_color = Colors.CYAN
    
    print(f"  {Colors.BOLD}{'COMBINED (Final)':<30}{Colors.RESET}"
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
    logreg_info = result["logreg_info"]
    print(f"  {Colors.DIM}Logistic Regression ({logreg_info.get('method', 'unknown')}):{Colors.RESET}")
    
    if logreg_info.get("method") == "player_based":
        print(f"    • {home}: {logreg_info.get('home_n_players', 'N/A')} active players")
        print(f"    • {away}: {logreg_info.get('away_n_players', 'N/A')} active players")
        print(f"    • Features: PTS, REB, AST, PRA, FG%, 3P%, FT% (L5, L10 windows)")
    else:
        print(f"    • Uses team-level rolling stats (legacy mode)")
        print(f"    • Data source: {logreg_info.get('home_source', 'N/A')}")
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
    """Validate team code format."""
    team = team.strip().upper()
    if len(team) != 3:
        return False, "Team code must be exactly 3 letters"
    if not team.isalpha():
        return False, "Team code must contain only letters"
    return True, team


def validate_date_format(date_str: str) -> tuple[bool, pd.Timestamp | str]:
    """Validate date format."""
    import re
    date_str = date_str.strip()
    
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return False, "Date must be in YYYY-MM-DD format"
    
    try:
        parsed = pd.to_datetime(date_str)
        return True, parsed
    except Exception:
        return False, "Invalid date"


def prompt_team(prompt_text: str) -> str:
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
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 Interactive NBA Game Prediction (Player-Based){Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}Enter matchup details. Team codes: BOS, LAL, NYK, etc.{Colors.RESET}")
    print()


def main() -> None:
    """CLI entry point for combined prediction."""
    parser = argparse.ArgumentParser(
        description="NBA Combined Prediction (Player-Based + Monte Carlo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.predict_combined --home BOS --away NYK --date 2025-01-15
  python -m src.predict_combined --home LAL --away GSW --date 2025-01-20 --alpha 0.8
  python -m src.predict_combined  # Interactive mode
        """
    )
    
    parser.add_argument("--home", type=str, default=None, help="Home team (e.g., BOS)")
    parser.add_argument("--away", type=str, default=None, help="Away team (e.g., NYK)")
    parser.add_argument("--date", type=str, default=None, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for logreg (default: 0.7)")
    parser.add_argument("--season", type=int, default=None, help="Season year (e.g., 2025)")
    parser.add_argument("--n_sims", type=int, default=10000, help="MC simulations (default: 10000)")
    parser.add_argument("--n_mc_games", type=int, default=100, help="Games for MC (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
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
        home_team = prompt_team("Enter HOME team code (e.g., BOS)")
    
    # Get away team
    if args.away is not None:
        is_valid, result = validate_team_code(args.away)
        if not is_valid:
            print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid away team: {result}")
            sys.exit(1)
        away_team = result
    else:
        away_team = prompt_team("Enter AWAY team code (e.g., NYK)")
    
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
    
    if not 0 <= args.alpha <= 1:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Alpha must be between 0 and 1")
        sys.exit(1)
    
    try:
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
