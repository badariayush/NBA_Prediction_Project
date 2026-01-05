"""
Monte Carlo Win Distribution Simulator for NBA Season Projections.

This module simulates season outcomes to generate win distributions for each team
using per-game win probabilities. It supports fetching schedules from the NBA API
or loading from CSV files.

Usage Examples:
    # Pull from NBA API and simulate:
    python -m src.sim.win_distribution --source nba_api --season 2024 --n_sims 20000
    
    # Use a CSV schedule:
    python -m src.sim.win_distribution --source csv --schedule data/schedules/2024_remaining.csv
    
    # Show team histogram:
    python -m src.sim.win_distribution --source csv --schedule data/schedules/sample.csv --team BOS
    
    # Print matchup probabilities for next 25 games:
    python -m src.sim.win_distribution --source nba_api --season 2024 --print_matchups
    
    # Print more matchups and save to custom path:
    python -m src.sim.win_distribution --source nba_api --season 2024 --print_matchups --matchups_n 50 --matchups_out data/my_matchups.csv
    
    # Quiet mode (minimal output):
    python -m src.sim.win_distribution --source nba_api --season 2024 --quiet
    
    # Verbose mode (detailed output):
    python -m src.sim.win_distribution --source nba_api --season 2024 --verbose
    
    # Force refresh from NBA API:
    python -m src.sim.win_distribution --source nba_api --season 2024 --refresh

Programmatic usage:
    from src.sim.win_distribution import simulate_win_distribution, summarize_distributions
    
    # schedule_df must have: home_team, away_team, p_home_win
    wins_df = simulate_win_distribution(schedule_df, n_sims=10000)
    summary_df = summarize_distributions(wins_df)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "models/logreg.pkl"
FEATURES_PATH = "data/processed/games_features.csv"
GAMES_CSV_PATH = "data/raw/games.csv"
MATCHUPS_CSV_PATH = "data/processed/matchup_probs.csv"

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


# Verbosity levels
QUIET = 0
NORMAL = 1
VERBOSE = 2


# ============================================================================
# Core Simulation Functions
# ============================================================================

def simulate_win_distribution(
    schedule_df: pd.DataFrame,
    n_sims: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation of season outcomes.
    
    For each simulation, samples each game outcome based on p_home_win probability,
    then aggregates total wins per team. Fully vectorized for performance.
    
    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with columns: home_team, away_team, p_home_win
    n_sims : int
        Number of simulations to run (default: 10000)
    seed : int
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    pd.DataFrame
        Shape (n_sims, n_teams) where each column is a team and each row 
        is that simulation's total wins for the team.
    """
    required_cols = {"home_team", "away_team", "p_home_win"}
    missing_cols = required_cols - set(schedule_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"schedule_df must contain: {required_cols}"
        )

    rng = np.random.default_rng(seed)

    home_teams = schedule_df["home_team"].to_numpy()
    away_teams = schedule_df["away_team"].to_numpy()
    p_home_win = schedule_df["p_home_win"].to_numpy(dtype=float)

    n_games = len(schedule_df)

    all_teams = sorted(set(home_teams) | set(away_teams))
    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = np.fromiter((team_to_idx[t] for t in home_teams), dtype=np.int32, count=n_games)
    away_idx = np.fromiter((team_to_idx[t] for t in away_teams), dtype=np.int32, count=n_games)

    # Sample outcomes: shape (n_sims, n_games)
    home_wins = rng.random((n_sims, n_games)) < p_home_win

    wins_matrix = np.zeros((n_sims, n_teams), dtype=np.int32)

    # Winning team indices per sim/game - fully vectorized
    winners = np.where(home_wins, home_idx[None, :], away_idx[None, :])

    # Flatten and accumulate with np.add.at for O(n_sims * n_games) performance
    sim_ids = np.repeat(np.arange(n_sims, dtype=np.int32), n_games)
    winner_ids = winners.reshape(-1)

    np.add.at(wins_matrix, (sim_ids, winner_ids), 1)

    return pd.DataFrame(wins_matrix, columns=all_teams)


def summarize_distributions(
    wins_df: pd.DataFrame,
    win_threshold: int = 50
) -> pd.DataFrame:
    """
    Compute summary statistics for win distributions.
    
    Parameters
    ----------
    wins_df : pd.DataFrame
        Output from simulate_win_distribution, shape (n_sims, n_teams)
    win_threshold : int
        Threshold for computing prob_ge_X (default: 50)
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per team, sorted by mean_wins descending.
    """
    # Vectorized computation for all teams at once
    data = wins_df.to_numpy()
    teams = wins_df.columns.tolist()
    
    summaries = {
        "team": teams,
        "mean_wins": np.mean(data, axis=0),
        "std_wins": np.std(data, axis=0),
        "min_wins": np.min(data, axis=0),
        "max_wins": np.max(data, axis=0),
        "p10": np.percentile(data, 10, axis=0),
        "p25": np.percentile(data, 25, axis=0),
        "p50": np.percentile(data, 50, axis=0),
        "p75": np.percentile(data, 75, axis=0),
        "p90": np.percentile(data, 90, axis=0),
        f"prob_ge_{win_threshold}": np.mean(data >= win_threshold, axis=0),
    }
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("mean_wins", ascending=False).reset_index(drop=True)
    
    return summary_df


# ============================================================================
# Output Functions
# ============================================================================

def save_outputs(
    wins_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: str = "data/processed",
    prefix: str = "mc",
    save_wins_matrix: bool = True,
    verbosity: int = NORMAL
) -> dict:
    """Save simulation outputs to CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save summary (always)
    summary_path = os.path.join(out_dir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    saved_files["summary"] = summary_path
    
    # Save wins matrix (optional)
    if save_wins_matrix:
        wins_path = os.path.join(out_dir, f"{prefix}_wins_matrix.csv")
        wins_df.to_csv(wins_path, index=False)
        saved_files["wins_matrix"] = wins_path
    
    return saved_files


def save_matchup_probs(
    schedule_df: pd.DataFrame,
    output_path: str = MATCHUPS_CSV_PATH,
    verbosity: int = NORMAL
) -> str:
    """
    Save matchup probabilities to CSV.
    
    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with p_home_win and optionally is_fallback columns
    output_path : str
        Path to save the CSV
    verbosity : int
        Verbosity level
        
    Returns
    -------
    str
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Prepare output DataFrame
    output_df = schedule_df.copy()
    
    # Ensure we have the required columns
    output_df["p_away_win"] = 1 - output_df["p_home_win"]
    
    # Select and order columns
    cols = ["date", "away_team", "home_team", "p_home_win", "p_away_win"]
    if "is_fallback" in output_df.columns:
        cols.append("is_fallback")
    
    # Keep only available columns
    cols = [c for c in cols if c in output_df.columns]
    output_df = output_df[cols]
    
    # Sort by date
    output_df = output_df.sort_values("date").reset_index(drop=True)
    
    # Round probabilities for readability
    output_df["p_home_win"] = output_df["p_home_win"].round(4)
    output_df["p_away_win"] = output_df["p_away_win"].round(4)
    
    output_df.to_csv(output_path, index=False)
    
    return output_path


# ============================================================================
# Probability Computation (for schedules without p_home_win)
# ============================================================================

def get_team_latest_stats(
    features_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp
) -> pd.Series | None:
    """Get the most recent rolling stats for a team before a given date."""
    team_games = features_df[
        ((features_df["home_team"] == team) | (features_df["away_team"] == team)) &
        (features_df["date"] < before_date)
    ].sort_values("date", ascending=False)
    
    if len(team_games) == 0:
        return None
    
    latest_game = team_games.iloc[0]
    
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
    
    return pd.Series(stats)


def add_game_probs(
    schedule_df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    verbosity: int = NORMAL,
    track_fallbacks: bool = False
) -> tuple[pd.DataFrame, int]:
    """
    Add p_home_win probabilities to a schedule using the trained model.
    Uses "latest prior rolling stats" method to avoid data leakage.
    
    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with columns: date, home_team, away_team
    model_path : str
        Path to trained model
    features_path : str
        Path to historical features CSV
    verbosity : int
        Verbosity level
    track_fallbacks : bool
        If True, add 'is_fallback' column to mark games with fallback probability
        
    Returns
    -------
    tuple[pd.DataFrame, int]
        - Schedule DataFrame with p_home_win (and optionally is_fallback) column added
        - Count of games that used fallback probability (0.5)
    """
    # Validate model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run 'python -m src.train' first."
        )
    model = joblib.load(model_path)
    
    # Validate features file exists
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Features file not found at '{features_path}'. "
            "Run 'python -m src.features' first."
        )
    features_df = pd.read_csv(features_path)
    features_df["date"] = pd.to_datetime(features_df["date"])
    
    # Ensure schedule dates are parsed
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    
    # Sort schedule by date
    schedule_df = schedule_df.sort_values("date").reset_index(drop=True)
    
    fallback_count = 0
    probabilities = []
    is_fallback = []
    
    for idx, row in schedule_df.iterrows():
        game_date = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        # Get latest stats for each team
        home_stats = get_team_latest_stats(features_df, home_team, game_date)
        away_stats = get_team_latest_stats(features_df, away_team, game_date)
        
        # Check if we have valid stats for both teams
        if home_stats is None or away_stats is None:
            probabilities.append(0.5)
            is_fallback.append(True)
            fallback_count += 1
            continue
        
        # Check for NaN values in stats
        if home_stats.isna().any() or away_stats.isna().any():
            probabilities.append(0.5)
            is_fallback.append(True)
            fallback_count += 1
            continue
        
        # Compute diff features
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
        probabilities.append(p_home_win)
        is_fallback.append(False)
    
    schedule_df["p_home_win"] = probabilities
    
    if track_fallbacks:
        schedule_df["is_fallback"] = is_fallback
    
    return schedule_df, fallback_count


# ============================================================================
# Display Functions - Enhanced Dashboard Output
# ============================================================================

def print_header(
    season: Optional[int],
    n_games: int,
    n_teams: int,
    n_sims: int,
    seed: int,
    source: str,
    verbosity: int = NORMAL
) -> None:
    """Print dashboard header with simulation parameters."""
    if verbosity == QUIET:
        return
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 NBA MONTE CARLO WIN DISTRIBUTION SIMULATOR{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}Run timestamp:{Colors.RESET}  {now}")
    if season:
        print(f"  {Colors.DIM}Season:{Colors.RESET}         {season}-{str(season + 1)[-2:]}")
    print(f"  {Colors.DIM}Data source:{Colors.RESET}    {source}")
    print(f"  {Colors.DIM}Games:{Colors.RESET}          {n_games:,}")
    print(f"  {Colors.DIM}Teams:{Colors.RESET}          {n_teams}")
    print(f"  {Colors.DIM}Simulations:{Colors.RESET}    {n_sims:,}")
    print(f"  {Colors.DIM}Random seed:{Colors.RESET}    {seed}")
    print()


def print_warning(message: str, verbosity: int = NORMAL) -> None:
    """Print a formatted warning message."""
    if verbosity == QUIET:
        return
    print(f"  {Colors.YELLOW}⚠ WARNING:{Colors.RESET} {message}")


def print_info(message: str, verbosity: int = NORMAL) -> None:
    """Print a formatted info message."""
    if verbosity == QUIET:
        return
    print(f"  {Colors.DIM}ℹ{Colors.RESET} {message}")


def print_success(message: str, verbosity: int = NORMAL) -> None:
    """Print a formatted success message."""
    if verbosity == QUIET:
        return
    print(f"  {Colors.GREEN}✓{Colors.RESET} {message}")


def print_error(message: str) -> None:
    """Print a formatted error message (always shown)."""
    print(f"  {Colors.RED}✗ ERROR:{Colors.RESET} {message}")


def print_top_teams(
    summary_df: pd.DataFrame,
    n: int = 10,
    win_threshold: int = 50,
    verbosity: int = NORMAL
) -> None:
    """Print top N teams by mean wins in a formatted table."""
    if verbosity == QUIET:
        return
    
    prob_col = [c for c in summary_df.columns if c.startswith("prob_ge_")][0]
    threshold = int(prob_col.split("_")[-1])
    
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📊 TOP {n} TEAMS BY PROJECTED WINS{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    # Header row
    header = (
        f"  {'#':<4}{'Team':<7}{'Mean':>8}{'Std':>7}"
        f"{'Range':>12}{'Median':>8}{f'P(≥{threshold})':>10}"
    )
    print(f"{Colors.DIM}{header}{Colors.RESET}")
    print(f"  {'-' * 62}")
    
    for idx, row in summary_df.head(n).iterrows():
        rank = idx + 1
        team = row['team']
        mean_wins = row['mean_wins']
        std_wins = row['std_wins']
        min_wins = int(row['min_wins'])
        max_wins = int(row['max_wins'])
        median = int(row['p50'])
        prob = row[prob_col]
        
        # Color code based on probability
        if prob >= 0.75:
            prob_color = Colors.GREEN
        elif prob >= 0.5:
            prob_color = Colors.CYAN
        elif prob >= 0.25:
            prob_color = Colors.YELLOW
        else:
            prob_color = Colors.DIM
        
        win_range = f"{min_wins}-{max_wins}"
        
        print(
            f"  {rank:<4}{Colors.BOLD}{team:<7}{Colors.RESET}"
            f"{mean_wins:>8.1f}{std_wins:>7.1f}"
            f"{win_range:>12}{median:>8}"
            f"{prob_color}{prob:>10.1%}{Colors.RESET}"
        )
    
    print()


def print_interpretation(
    summary_df: pd.DataFrame,
    n_games: int,
    win_threshold: int = 50,
    verbosity: int = NORMAL
) -> None:
    """Print a brief interpretation of the results."""
    if verbosity == QUIET:
        return
    
    prob_col = [c for c in summary_df.columns if c.startswith("prob_ge_")][0]
    
    top_team = summary_df.iloc[0]
    top_team_name = top_team["team"]
    top_mean = top_team["mean_wins"]
    top_prob = top_team[prob_col]
    
    # Count teams likely to reach threshold
    likely_teams = (summary_df[prob_col] >= 0.5).sum()
    
    print(f"  {Colors.DIM}{'─' * 66}{Colors.RESET}")
    print(f"  {Colors.BOLD}What this means:{Colors.RESET}")
    print(f"  • {Colors.BOLD}{top_team_name}{Colors.RESET} projects highest with ~{top_mean:.0f} wins "
          f"({top_prob:.0%} chance of ≥{int(prob_col.split('_')[-1])} wins)")
    print(f"  • {likely_teams} team(s) have >50% probability of reaching the win threshold")
    print()


def print_matchups(
    schedule_df: pd.DataFrame,
    n: int = 25,
    verbosity: int = NORMAL
) -> None:
    """
    Print game-by-game matchup probabilities.
    
    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with p_home_win column and optionally is_fallback
    n : int
        Number of games to display (default: 25)
    verbosity : int
        Verbosity level
    """
    if verbosity == QUIET:
        return
    
    # Sort by date and take first n
    df = schedule_df.sort_values("date").head(n).copy()
    
    print()
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  🎯 GAME-BY-GAME MATCHUP PROBABILITIES (Next {len(df)} Games){Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    # Header
    header = f"  {'Date':<12}{'Matchup':<18}{'Home Win':>12}{'Away Win':>12}{'Note':>10}"
    print(f"{Colors.DIM}{header}{Colors.RESET}")
    print(f"  {'-' * 64}")
    
    has_fallback_col = "is_fallback" in df.columns
    
    for _, row in df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        away = row["away_team"]
        home = row["home_team"]
        p_home = row["p_home_win"]
        p_away = 1 - p_home
        
        # Check if fallback was used
        is_fallback = has_fallback_col and row.get("is_fallback", False)
        
        # Format matchup as "AWAY @ HOME"
        matchup = f"{away} @ {home}"
        
        # Color code the favorite
        if p_home > 0.5:
            home_color = Colors.GREEN
            away_color = Colors.DIM
        elif p_away > 0.5:
            home_color = Colors.DIM
            away_color = Colors.GREEN
        else:
            home_color = Colors.CYAN
            away_color = Colors.CYAN
        
        # Note column
        if is_fallback:
            note = f"{Colors.YELLOW}(fallback){Colors.RESET}"
        else:
            note = ""
        
        print(
            f"  {date_str:<12}{matchup:<18}"
            f"{home_color}p_home={p_home:.2f}{Colors.RESET}  "
            f"{away_color}p_away={p_away:.2f}{Colors.RESET}"
            f"  {note}"
        )
    
    print()
    
    # Summary stats
    avg_home_win = schedule_df["p_home_win"].mean()
    n_home_favored = (schedule_df["p_home_win"] > 0.5).sum()
    n_away_favored = (schedule_df["p_home_win"] < 0.5).sum()
    n_tossup = (schedule_df["p_home_win"] == 0.5).sum()
    
    print(f"  {Colors.DIM}Summary:{Colors.RESET}")
    print(f"    Average home win probability: {avg_home_win:.1%}")
    print(f"    Home favored: {n_home_favored} | Away favored: {n_away_favored} | Toss-up: {n_tossup}")
    
    if has_fallback_col:
        n_fallback = schedule_df["is_fallback"].sum()
        if n_fallback > 0:
            print(f"    {Colors.YELLOW}⚠ {n_fallback} games using fallback probability (0.5){Colors.RESET}")
    
    print()


def print_team_histogram(
    wins_df: pd.DataFrame,
    team: str,
    win_threshold: int = 50,
    verbosity: int = NORMAL
) -> None:
    """Print an enhanced text-based histogram for a team's win distribution."""
    if team not in wins_df.columns:
        print_error(f"Team '{team}' not found in simulation results.")
        available = ", ".join(sorted(wins_df.columns[:10]))
        print(f"  Available teams include: {available}...")
        return
    
    wins = wins_df[team].values
    n_sims = len(wins)
    
    min_wins = int(np.min(wins))
    max_wins = int(np.max(wins))
    mean_wins = np.mean(wins)
    median_wins = np.median(wins)
    std_wins = np.std(wins)
    
    # Dynamic bin sizing
    win_range = max_wins - min_wins
    if win_range <= 10:
        n_bins = win_range + 1
    elif win_range <= 30:
        n_bins = min(15, win_range)
    else:
        n_bins = 20
    
    n_bins = max(3, n_bins)  # At least 3 bins
    counts, bin_edges = np.histogram(wins, bins=n_bins)
    
    max_count = np.max(counts)
    bar_width = 35
    
    print()
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📈 WIN DISTRIBUTION: {team}{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}Simulations:{Colors.RESET} {n_sims:,}")
    print(f"  {Colors.DIM}Win range:{Colors.RESET}   {min_wins} — {max_wins}")
    print(f"  {Colors.DIM}Mean:{Colors.RESET}        {mean_wins:.1f} ± {std_wins:.1f}")
    print(f"  {Colors.DIM}Median:{Colors.RESET}      {median_wins:.0f}")
    print()
    
    # Histogram bars
    print(f"  {'Wins':<10}{'Distribution':^40}{'Freq':>8}")
    print(f"  {'─' * 60}")
    
    for i in range(len(counts)):
        lo = int(bin_edges[i])
        hi = int(bin_edges[i + 1]) - 1 if i < len(counts) - 1 else int(bin_edges[i + 1])
        pct = counts[i] / n_sims
        bar_len = int((counts[i] / max_count) * bar_width) if max_count > 0 else 0
        
        # Color bars based on position relative to threshold
        mid_bin = (lo + hi) / 2
        if mid_bin >= win_threshold:
            bar_char = f"{Colors.GREEN}█{Colors.RESET}"
        else:
            bar_char = f"{Colors.CYAN}█{Colors.RESET}"
        
        bar = bar_char * bar_len
        bin_label = f"{lo}-{hi}" if lo != hi else str(lo)
        
        print(f"  {bin_label:<10}{bar:<{bar_width + 20}}{pct:>8.1%}")
    
    print()
    
    # Percentiles box
    print(f"  {Colors.BOLD}Percentiles:{Colors.RESET}")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = [np.percentile(wins, p) for p in percentiles]
    pct_str = "  " + "  ".join(f"p{p}={int(v)}" for p, v in zip(percentiles, pct_values))
    print(f"  {Colors.DIM}{pct_str}{Colors.RESET}")
    print()
    
    # Threshold probabilities
    print(f"  {Colors.BOLD}Probability of reaching win thresholds:{Colors.RESET}")
    thresholds = [30, 40, 50, 55, 60]
    for t in thresholds:
        prob = np.mean(wins >= t)
        bar_len = int(prob * 20)
        bar = "▓" * bar_len + "░" * (20 - bar_len)
        color = Colors.GREEN if prob >= 0.5 else (Colors.YELLOW if prob >= 0.25 else Colors.DIM)
        print(f"    ≥{t} wins: {color}{bar} {prob:>6.1%}{Colors.RESET}")
    print()


def print_saved_files(
    saved_files: dict,
    wins_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    matchups_path: Optional[str] = None,
    schedule_df: Optional[pd.DataFrame] = None,
    verbosity: int = NORMAL
) -> None:
    """Print information about saved output files."""
    if verbosity == QUIET:
        return
    
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  💾 OUTPUT FILES{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    if "summary" in saved_files:
        path = saved_files["summary"]
        shape = f"({len(summary_df)} teams × {len(summary_df.columns)} columns)"
        print(f"  {Colors.GREEN}✓{Colors.RESET} Summary:      {path}")
        print(f"    {Colors.DIM}{shape}{Colors.RESET}")
    
    if "wins_matrix" in saved_files:
        path = saved_files["wins_matrix"]
        shape = f"({len(wins_df)} sims × {len(wins_df.columns)} teams)"
        print(f"  {Colors.GREEN}✓{Colors.RESET} Wins matrix:  {path}")
        print(f"    {Colors.DIM}{shape}{Colors.RESET}")
    
    if matchups_path and schedule_df is not None:
        shape = f"({len(schedule_df)} games × 5 columns)"
        print(f"  {Colors.GREEN}✓{Colors.RESET} Matchups:     {matchups_path}")
        print(f"    {Colors.DIM}{shape}{Colors.RESET}")
    
    print()


# ============================================================================
# NBA API Integration
# ============================================================================

def check_nba_api_installed() -> bool:
    """Check if nba_api package is installed."""
    try:
        import nba_api  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_schedule_from_api(
    season: int,
    refresh: bool = False,
    verbosity: int = NORMAL
) -> pd.DataFrame:
    """
    Fetch season schedule from NBA API.
    
    Parameters
    ----------
    season : int
        Season year (e.g., 2024 for 2024-25 season)
    refresh : bool
        Force re-download even if cached
    verbosity : int
        Output verbosity level
        
    Returns
    -------
    pd.DataFrame
        Schedule DataFrame with date, home_team, away_team columns
    """
    # Check nba_api is installed
    if not check_nba_api_installed():
        print_error("nba_api package is not installed.")
        print(f"  {Colors.DIM}Install with: pip install nba_api{Colors.RESET}")
        sys.exit(1)
    
    try:
        from src.data.nba_api_client import (
            fetch_season_games,
            upsert_games_csv,
            normalize_team_codes,
        )
    except ImportError as e:
        print_error(f"Failed to import nba_api_client: {e}")
        print(f"  {Colors.DIM}Ensure src/data/nba_api_client.py exists{Colors.RESET}")
        sys.exit(1)
    
    if verbosity >= NORMAL:
        print_info(f"Fetching {season}-{str(season+1)[-2:]} season from NBA API...")
    
    try:
        games_df = fetch_season_games(
            season, 
            verbose=(verbosity >= VERBOSE)
        )
    except Exception as e:
        print_error(f"Failed to fetch from NBA API: {e}")
        sys.exit(1)
    
    if games_df.empty:
        print_error(f"No games found for {season} season.")
        sys.exit(1)
    
    # Update local cache
    if verbosity >= NORMAL:
        print_info(f"Updating local cache: {GAMES_CSV_PATH}")
    
    try:
        n_new = upsert_games_csv(
            games_df, 
            path=GAMES_CSV_PATH,
            verbose=(verbosity >= VERBOSE)
        )
        if verbosity >= NORMAL and n_new > 0:
            print_success(f"Added {n_new} new games to cache")
    except Exception as e:
        if verbosity >= NORMAL:
            print_warning(f"Could not update cache: {e}")
    
    # Prepare schedule DataFrame
    schedule_df = games_df[["date", "home_team", "away_team"]].copy()
    
    if verbosity >= NORMAL:
        print_success(f"Loaded {len(schedule_df)} games from NBA API")
    
    return schedule_df


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """CLI entry point for Monte Carlo win distribution simulation."""
    parser = argparse.ArgumentParser(
        description="NBA Monte Carlo Win Distribution Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pull from NBA API and simulate:
  python -m src.sim.win_distribution --source nba_api --season 2024 --n_sims 20000

  # Use a CSV schedule:
  python -m src.sim.win_distribution --source csv --schedule data/schedules/sample.csv

  # Show detailed team histogram:
  python -m src.sim.win_distribution --source csv --schedule data/schedules/sample.csv --team BOS

  # Print matchup probabilities:
  python -m src.sim.win_distribution --source nba_api --season 2024 --print_matchups

  # Print more matchups and save to custom path:
  python -m src.sim.win_distribution --source nba_api --season 2024 --print_matchups --matchups_n 50

  # Quiet mode:
  python -m src.sim.win_distribution --source nba_api --season 2024 --quiet

  # Verbose mode with refresh:
  python -m src.sim.win_distribution --source nba_api --season 2024 --verbose --refresh
        """
    )
    
    # Source arguments
    parser.add_argument(
        "--source",
        type=str,
        choices=["csv", "nba_api"],
        default="nba_api",
        help="Data source: 'csv' for local file, 'nba_api' for NBA API (default: nba_api)"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        help="Path to schedule CSV (required if source=csv)"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="NBA season year, e.g., 2024 for 2024-25 (default: 2024)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh from NBA API (ignore cache)"
    )
    
    # Simulation arguments
    parser.add_argument(
        "--n_sims",
        type=int,
        default=10000,
        help="Number of simulations (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--win_threshold",
        type=int,
        default=50,
        help="Win threshold for probability calculation (default: 50)"
    )
    
    # Matchup probability arguments
    parser.add_argument(
        "--print_matchups",
        action="store_true",
        help="Print game-by-game matchup probabilities"
    )
    parser.add_argument(
        "--matchups_n",
        type=int,
        default=25,
        help="Number of matchups to display (default: 25)"
    )
    parser.add_argument(
        "--matchups_out",
        type=str,
        default=MATCHUPS_CSV_PATH,
        help=f"Path to save matchup probabilities CSV (default: {MATCHUPS_CSV_PATH})"
    )
    
    # Output arguments
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mc",
        help="Output file prefix (default: mc)"
    )
    parser.add_argument(
        "--skip_wins_matrix",
        action="store_true",
        help="Skip saving the large wins matrix CSV"
    )
    
    # Display arguments
    parser.add_argument(
        "--team",
        type=str,
        help="Show detailed histogram for this team (e.g., BOS)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top teams to display (default: 10)"
    )
    
    # Verbosity
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (only errors and final results)"
    )
    verbosity_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Detailed output with progress info"
    )
    
    args = parser.parse_args()
    
    # Set verbosity level
    if args.quiet:
        verbosity = QUIET
    elif args.verbose:
        verbosity = VERBOSE
    else:
        verbosity = NORMAL
    
    # Validate arguments
    if args.source == "csv" and not args.schedule:
        print_error("--schedule is required when --source=csv")
        sys.exit(1)
    
    # Load schedule based on source
    season = args.season
    
    if args.source == "nba_api":
        schedule_df = fetch_schedule_from_api(
            season=args.season,
            refresh=args.refresh,
            verbosity=verbosity
        )
        source_desc = "NBA API"
    else:
        # CSV source
        if not os.path.exists(args.schedule):
            print_error(f"Schedule file not found: {args.schedule}")
            sys.exit(1)
        
        if verbosity >= NORMAL:
            print_info(f"Loading schedule from: {args.schedule}")
        
        schedule_df = pd.read_csv(args.schedule)
        
        # Try to extract season from data
        if "season" in schedule_df.columns:
            season = schedule_df["season"].mode().iloc[0] if len(schedule_df) > 0 else None
        
        source_desc = f"CSV ({args.schedule})"
    
    # Validate schedule columns
    required_cols = {"home_team", "away_team"}
    missing = required_cols - set(schedule_df.columns)
    if missing:
        print_error(f"Schedule missing required columns: {missing}")
        sys.exit(1)
    
    n_games = len(schedule_df)
    n_teams = len(set(schedule_df["home_team"]) | set(schedule_df["away_team"]))
    
    # Add probabilities if not present
    fallback_count = 0
    if "p_home_win" not in schedule_df.columns:
        if verbosity >= NORMAL:
            print_info("Computing game probabilities using trained model...")
        
        if "date" not in schedule_df.columns:
            print_error("'date' column required to compute probabilities.")
            sys.exit(1)
        
        try:
            schedule_df, fallback_count = add_game_probs(
                schedule_df, 
                verbosity=verbosity,
                track_fallbacks=True  # Track fallbacks for matchup display
            )
            if verbosity >= NORMAL:
                print_success(f"Computed probabilities for {len(schedule_df)} games")
        except FileNotFoundError as e:
            print_error(str(e))
            sys.exit(1)
    else:
        if verbosity >= NORMAL:
            print_info("Using existing p_home_win from schedule")
    
    # Print header
    print_header(
        season=season,
        n_games=n_games,
        n_teams=n_teams,
        n_sims=args.n_sims,
        seed=args.seed,
        source=source_desc,
        verbosity=verbosity
    )
    
    # Print warning about fallback probabilities
    if fallback_count > 0:
        print_warning(
            f"{fallback_count} games used fallback probability (0.5) due to missing team history"
        )
        print()
    
    # Print matchups if requested
    matchups_path = None
    if args.print_matchups:
        print_matchups(
            schedule_df,
            n=args.matchups_n,
            verbosity=verbosity
        )
        
        # Save matchup probabilities
        matchups_path = save_matchup_probs(
            schedule_df,
            output_path=args.matchups_out,
            verbosity=verbosity
        )
        if verbosity >= NORMAL:
            print_success(f"Saved matchup probabilities to: {matchups_path}")
            print()
    
    # Run simulation
    if verbosity >= NORMAL:
        print_info(f"Running {args.n_sims:,} simulations...")
    
    import time
    start_time = time.time()
    
    wins_df = simulate_win_distribution(
        schedule_df,
        n_sims=args.n_sims,
        seed=args.seed
    )
    
    elapsed = time.time() - start_time
    
    if verbosity >= NORMAL:
        print_success(f"Simulation complete in {elapsed:.2f}s")
        print()
    
    # Compute summary
    summary_df = summarize_distributions(
        wins_df, 
        win_threshold=args.win_threshold
    )
    
    # Print results
    print_top_teams(
        summary_df, 
        n=args.top_n,
        win_threshold=args.win_threshold,
        verbosity=verbosity
    )
    
    # Print interpretation
    print_interpretation(
        summary_df,
        n_games=n_games,
        win_threshold=args.win_threshold,
        verbosity=verbosity
    )
    
    # Print team histogram if requested
    if args.team:
        team = args.team.upper()
        print_team_histogram(
            wins_df, 
            team,
            win_threshold=args.win_threshold,
            verbosity=verbosity
        )
    
    # Save outputs
    saved_files = save_outputs(
        wins_df,
        summary_df,
        out_dir=args.out_dir,
        prefix=args.prefix,
        save_wins_matrix=not args.skip_wins_matrix,
        verbosity=verbosity
    )
    
    print_saved_files(
        saved_files,
        wins_df,
        summary_df,
        matchups_path=matchups_path,
        schedule_df=schedule_df if args.print_matchups else None,
        verbosity=verbosity
    )
    
    if verbosity >= NORMAL:
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.GREEN}  ✓ Done!{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print()


if __name__ == "__main__":
    main()
