"""
CLI tool for predicting NBA game outcomes.

Usage:
    python -m src.predict_cli
"""

import os
import sys
from datetime import datetime

import joblib
import pandas as pd

# Configuration
MODEL_PATH = "models/logreg.pkl"
FEATURES_PATH = "data/processed/games_features.csv"

# Required feature columns (diffs)
REQUIRED_DIFF_COLS = [
    "pf_roll1_diff",
    "pa_roll1_diff",
    "pd_roll1_diff",
    "pf_roll2_diff",
    "pa_roll2_diff",
    "pd_roll2_diff",
]

# Optional feature columns
OPTIONAL_DIFF_COLS = [
    "rest_days_diff",
    "b2b_diff",
]

# Rolling stat columns (for computing diffs from individual team stats)
ROLLING_STAT_COLS = [
    "home_pf_roll1", "home_pa_roll1", "home_pd_roll1",
    "home_pf_roll2", "home_pa_roll2", "home_pd_roll2",
    "away_pf_roll1", "away_pa_roll1", "away_pd_roll1",
    "away_pf_roll2", "away_pa_roll2", "away_pd_roll2",
]


def load_model(path: str):
    """Load the trained model pipeline."""
    if not os.path.exists(path):
        print(f"ERROR: Model file not found at '{path}'")
        print("FIX: Run 'python -m src.train' first to train and save the model.")
        sys.exit(1)
    return joblib.load(path)


def load_features(path: str) -> pd.DataFrame:
    """Load the feature table."""
    if not os.path.exists(path):
        print(f"ERROR: Feature file not found at '{path}'")
        print("FIX: Run 'python -m src.features' first to build the feature table.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_all_teams(df: pd.DataFrame) -> set:
    """Get all unique team abbreviations from the dataset."""
    home_teams = set(df["home_team"].unique())
    away_teams = set(df["away_team"].unique())
    return home_teams | away_teams


def validate_team(team: str, all_teams: set, role: str) -> str:
    """Validate that a team exists in the dataset."""
    team = team.strip().upper()
    if team not in all_teams:
        print(f"ERROR: {role} team '{team}' not found in dataset.")
        print(f"FIX: Valid teams are: {sorted(all_teams)}")
        sys.exit(1)
    return team


def parse_date(date_str: str, df: pd.DataFrame) -> pd.Timestamp:
    """Parse date string or return default (most recent date in data)."""
    if not date_str.strip():
        default_date = df["date"].max()
        print(f"No date provided. Using most recent date: {default_date.strftime('%Y-%m-%d')}")
        return default_date

    try:
        return pd.to_datetime(date_str.strip())
    except Exception:
        print(f"ERROR: Cannot parse date '{date_str}'")
        print("FIX: Use format YYYY-MM-DD (e.g., 2024-01-15)")
        sys.exit(1)


def get_team_latest_stats(
    df: pd.DataFrame, team: str, before_date: pd.Timestamp, is_home: bool
) -> tuple[pd.Series | None, str | None]:
    """
    Get the most recent rolling stats for a team before a given date.

    Returns the row and the date it was found, or (None, None) if not found.
    """
    # Find games where this team played (as home or away)
    team_games = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].sort_values("date", ascending=False)

    if len(team_games) == 0:
        return None, None

    latest_game = team_games.iloc[0]
    game_date = latest_game["date"].strftime("%Y-%m-%d")

    # Extract rolling stats depending on whether team was home or away in that game
    if latest_game["home_team"] == team:
        # Team was home in the latest game - use home stats
        stats = {
            "pf_roll1": latest_game.get("home_pf_roll1"),
            "pa_roll1": latest_game.get("home_pa_roll1"),
            "pd_roll1": latest_game.get("home_pd_roll1"),
            "pf_roll2": latest_game.get("home_pf_roll2"),
            "pa_roll2": latest_game.get("home_pa_roll2"),
            "pd_roll2": latest_game.get("home_pd_roll2"),
        }
    else:
        # Team was away in the latest game - use away stats
        stats = {
            "pf_roll1": latest_game.get("away_pf_roll1"),
            "pa_roll1": latest_game.get("away_pa_roll1"),
            "pd_roll1": latest_game.get("away_pd_roll1"),
            "pf_roll2": latest_game.get("away_pf_roll2"),
            "pa_roll2": latest_game.get("away_pa_roll2"),
            "pd_roll2": latest_game.get("away_pd_roll2"),
        }

    return pd.Series(stats), game_date


def build_features_from_matchup(
    df: pd.DataFrame, home_team: str, away_team: str, game_date: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    """
    Build feature vector for a matchup.

    First tries to find an exact historical matchup. If not found,
    computes features from each team's most recent rolling stats.

    Returns (feature_df, info_dict) where info_dict has transparency info.
    """
    info = {"method": None, "home_source": None, "away_source": None}

    # Try to find exact matchup before the date
    exact_matchup = df[
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team) &
        (df["date"] < game_date)
    ].sort_values("date", ascending=False)

    if len(exact_matchup) > 0:
        # Use the most recent exact matchup
        row = exact_matchup.iloc[0]
        info["method"] = "exact_matchup"
        info["home_source"] = f"{home_team} vs {away_team} on {row['date'].strftime('%Y-%m-%d')}"
        info["away_source"] = info["home_source"]

        # Build feature dict from existing diff columns
        features = {}
        for col in REQUIRED_DIFF_COLS:
            if col in row:
                features[col] = row[col]
            else:
                print(f"WARNING: Required feature '{col}' not found in data.")
                features[col] = 0.0

        for col in OPTIONAL_DIFF_COLS:
            if col in row and pd.notna(row[col]):
                features[col] = row[col]

        return pd.DataFrame([features]), info

    # No exact matchup - compute from individual team stats
    info["method"] = "computed_from_teams"

    home_stats, home_date = get_team_latest_stats(df, home_team, game_date, is_home=True)
    away_stats, away_date = get_team_latest_stats(df, away_team, game_date, is_home=False)

    if home_stats is None:
        print(f"ERROR: No historical data found for {home_team} before {game_date.strftime('%Y-%m-%d')}")
        print("FIX: Choose a later date or a team with more history.")
        sys.exit(1)

    if away_stats is None:
        print(f"ERROR: No historical data found for {away_team} before {game_date.strftime('%Y-%m-%d')}")
        print("FIX: Choose a later date or a team with more history.")
        sys.exit(1)

    info["home_source"] = f"{home_team}'s last game on {home_date}"
    info["away_source"] = f"{away_team}'s last game on {away_date}"

    # Compute diffs: home - away
    features = {
        "pf_roll1_diff": home_stats["pf_roll1"] - away_stats["pf_roll1"],
        "pa_roll1_diff": home_stats["pa_roll1"] - away_stats["pa_roll1"],
        "pd_roll1_diff": home_stats["pd_roll1"] - away_stats["pd_roll1"],
        "pf_roll2_diff": home_stats["pf_roll2"] - away_stats["pf_roll2"],
        "pa_roll2_diff": home_stats["pa_roll2"] - away_stats["pa_roll2"],
        "pd_roll2_diff": home_stats["pd_roll2"] - away_stats["pd_roll2"],
    }

    return pd.DataFrame([features]), info


def get_model_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Determine which feature columns to use based on what's available.

    Returns list of feature column names.
    """
    available = set(df.columns)
    feature_cols = []

    for col in REQUIRED_DIFF_COLS:
        if col in available:
            feature_cols.append(col)
        else:
            print(f"WARNING: Required feature '{col}' not in data. Using 0.")

    for col in OPTIONAL_DIFF_COLS:
        if col in available:
            feature_cols.append(col)

    return feature_cols


def main() -> None:
    """Main CLI entry point."""
    print("=" * 60)
    print("NBA Game Prediction CLI")
    print("=" * 60)
    print()

    # Load model and data
    model = load_model(MODEL_PATH)
    df = load_features(FEATURES_PATH)
    all_teams = get_all_teams(df)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Loaded {len(df)} games from {FEATURES_PATH}")
    print(f"Available teams: {len(all_teams)}")
    print()

    # Get user input
    home_team_input = input("Enter home team abbreviation (e.g., BOS): ")
    away_team_input = input("Enter away team abbreviation (e.g., LAL): ")
    date_input = input("Enter game date (YYYY-MM-DD) or press Enter for most recent: ")
    print()

    # Validate inputs
    home_team = validate_team(home_team_input, all_teams, "Home")
    away_team = validate_team(away_team_input, all_teams, "Away")
    game_date = parse_date(date_input, df)

    if home_team == away_team:
        print("ERROR: Home and away teams cannot be the same.")
        sys.exit(1)

    print()
    print("-" * 60)
    print(f"Predicting: {home_team} (home) vs {away_team} (away)")
    print(f"Game date: {game_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    print()

    # Build features
    X, info = build_features_from_matchup(df, home_team, away_team, game_date)

    # Ensure we have the right columns in the right order
    feature_cols = REQUIRED_DIFF_COLS.copy()
    for col in OPTIONAL_DIFF_COLS:
        if col in X.columns:
            feature_cols.append(col)

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    # Check for NaN values
    if X[feature_cols].isnull().any().any():
        print("WARNING: Some features have NaN values. Filling with 0.")
        X = X.fillna(0.0)

    # Select only the columns the model expects
    # The model was trained on REQUIRED_DIFF_COLS based on train.py
    model_features = REQUIRED_DIFF_COLS
    X_pred = X[model_features]

    # Make prediction
    home_win_prob = model.predict_proba(X_pred)[0, 1]
    away_win_prob = 1 - home_win_prob

    # Print transparency info
    print("Feature computation method:", info["method"])
    print(f"  Home team stats from: {info['home_source']}")
    print(f"  Away team stats from: {info['away_source']}")
    print()

    # Print feature values
    print("Feature values used:")
    for col in model_features:
        val = X_pred[col].iloc[0]
        print(f"  {col}: {val:.3f}" if pd.notna(val) else f"  {col}: N/A")
    print()

    # Print results
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"  {home_team} (home) win probability: {home_win_prob * 100:.2f}%")
    print(f"  {away_team} (away) win probability: {away_win_prob * 100:.2f}%")
    print("=" * 60)

    # Indicate favorite
    if home_win_prob > 0.5:
        print(f"\nPredicted winner: {home_team} (home)")
    elif away_win_prob > 0.5:
        print(f"\nPredicted winner: {away_team} (away)")
    else:
        print("\nToss-up game (50/50)")


if __name__ == "__main__":
    main()

