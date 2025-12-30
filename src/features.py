"""
Feature engineering module for NBA game prediction.

Builds leakage-free rolling features for each team using only
games that occurred BEFORE the current game date.
"""

import os

import numpy as np
import pandas as pd


def load_games(path: str) -> pd.DataFrame:
    """
    Load games from CSV and parse columns.

    Args:
        path: Path to the games CSV file.

    Returns:
        DataFrame with columns: date, season, home_team, away_team,
        home_score, away_score. Date is parsed and season is int.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(int)
    return df


def _games_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert games DataFrame to long format with one row per team per game.

    Each game produces two rows: one for the home team, one for the away team.

    Args:
        df: Games DataFrame with home/away columns.

    Returns:
        Long-format DataFrame with columns: game_idx, date, season, team,
        opponent, is_home, points_for, points_against, point_diff.
    """
    # Create home team rows
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

    # Create away team rows
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

    # Combine and compute point differential
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df["point_diff"] = long_df["points_for"] - long_df["points_against"]

    # Sort by date for proper rolling computation
    long_df = long_df.sort_values(["date", "game_idx"]).reset_index(drop=True)

    return long_df


def _compute_team_rolling_features(
    long_df: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """
    Compute rolling features for each team using only prior games.

    Args:
        long_df: Long-format DataFrame from _games_to_long_format.
        windows: List of rolling window sizes (e.g., [5, 10]).

    Returns:
        DataFrame with rolling features added for each window.
    """
    result = long_df.copy()

    # Sort by team and date for rolling computation
    result = result.sort_values(["team", "season", "date"]).reset_index(drop=True)

    for window in windows:
        suffix = f"_roll{window}"

        # Group by team and season, compute rolling mean with shift(1) to avoid leakage
        for col in ["points_for", "points_against", "point_diff"]:
            col_abbrev = {"points_for": "pf", "points_against": "pa", "point_diff": "pd"}[col]
            new_col = f"{col_abbrev}{suffix}"

            # Rolling mean of previous games (shift ensures we don't include current game)
            result[new_col] = (
                result.groupby(["team", "season"])[col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

    return result


def _merge_rolling_to_games(
    games_df: pd.DataFrame, long_df: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """
    Merge rolling features from long format back to game-level format.

    Args:
        games_df: Original games DataFrame.
        long_df: Long-format DataFrame with rolling features.
        windows: List of window sizes used.

    Returns:
        Games DataFrame with home/away rolling features added.
    """
    result = games_df.copy()

    # Build list of rolling feature columns
    roll_cols = []
    for window in windows:
        for stat in ["pf", "pa", "pd"]:
            roll_cols.append(f"{stat}_roll{window}")

    # Extract home team features
    home_features = long_df[long_df["is_home"] == 1][["game_idx"] + roll_cols].copy()
    home_features = home_features.rename(
        columns={col: f"home_{col}" for col in roll_cols}
    )
    home_features = home_features.set_index("game_idx")

    # Extract away team features
    away_features = long_df[long_df["is_home"] == 0][["game_idx"] + roll_cols].copy()
    away_features = away_features.rename(
        columns={col: f"away_{col}" for col in roll_cols}
    )
    away_features = away_features.set_index("game_idx")

    # Merge features back to games
    result = result.join(home_features)
    result = result.join(away_features)

    return result


def _add_diff_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add difference features between home and away rolling stats.

    Args:
        df: Games DataFrame with home/away rolling features.
        windows: List of window sizes.

    Returns:
        DataFrame with diff features added.
    """
    result = df.copy()

    for window in windows:
        for stat in ["pf", "pa", "pd"]:
            home_col = f"home_{stat}_roll{window}"
            away_col = f"away_{stat}_roll{window}"
            diff_col = f"{stat}_roll{window}_diff"
            result[diff_col] = result[home_col] - result[away_col]

    return result


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add leakage-free rolling features to a games DataFrame.

    For each team and window size, computes rolling means of:
    - points_for (PF)
    - points_against (PA)
    - point_diff (PD)

    Rolling stats use only games that occurred BEFORE the current game.

    Args:
        df: Games DataFrame with columns: date, season, home_team, away_team,
            home_score, away_score.
        windows: List of rolling window sizes (e.g., [5, 10]).

    Returns:
        DataFrame with original columns plus:
        - home_win: binary label (1 if home team won)
        - home_pf_roll{N}, home_pa_roll{N}, home_pd_roll{N} for each window N
        - away_pf_roll{N}, away_pa_roll{N}, away_pd_roll{N} for each window N
        - pf_roll{N}_diff, pa_roll{N}_diff, pd_roll{N}_diff for each window N
    """
    # Add label
    result = df.copy()
    result["home_win"] = (result["home_score"] > result["away_score"]).astype(int)

    # Convert to long format
    long_df = _games_to_long_format(df)

    # Compute rolling features per team
    long_df = _compute_team_rolling_features(long_df, windows)

    # Merge back to game level
    result = _merge_rolling_to_games(result, long_df, windows)

    # Add diff features
    result = _add_diff_features(result, windows)

    return result


def build_feature_table(raw_path: str = "data/raw/games.csv") -> pd.DataFrame:
    """
    Build the complete feature table from raw games data.

    Loads games, adds rolling features with windows [5, 10], and returns
    a DataFrame ready for modeling.

    Args:
        raw_path: Path to raw games CSV.

    Returns:
        DataFrame with all features and the home_win label.
    """
    print("Building feature table...")

    # Load raw games
    df = load_games(raw_path)
    print(f"Loaded {len(df)} games from {raw_path}")

    # Add rolling features
    windows = [1, 2]
    df = add_rolling_features(df, windows)
    print(f"Added rolling features with windows {windows}")

    # Report missing values (early season games)
    n_missing = df.isnull().any(axis=1).sum()
    print(f"Rows with missing values (early season): {n_missing}")

    return df


def save_feature_table(
    df: pd.DataFrame, out_path: str = "data/processed/games_features.csv"
) -> None:
    """
    Save the feature table to CSV.

    Args:
        df: Feature DataFrame to save.
        out_path: Output path for the CSV file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Feature table saved to {out_path}")


if __name__ == "__main__":
    # Build and save feature table
    feature_df = build_feature_table()
    save_feature_table(feature_df)

    # Display sample
    print("\nSample of feature table:")
    print(feature_df.head(10).to_string())

