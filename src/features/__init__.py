# Features package

import os
import pandas as pd

from src.features.builder import FeatureBuilder, get_feature_builder
from .aggregator import PlayerToTeamAggregator


def build_training_dataset(
    raw_dir: str = "data/raw",
    out_path: str | None = None,
    use_player_features: bool = True,
    n_recent_games: int = 10,
    **kwargs,
):
    """
    Build training dataset from raw CSVs on disk.

    Matches sanity_check.py call:
      build_training_dataset(raw_dir=..., out_path=..., use_player_features=True)
    """
    games_path = os.path.join(raw_dir, "games.csv")
    box_path = os.path.join(raw_dir, "player_boxscores.csv")

    games_df = pd.read_csv(games_path)
    player_box_df = pd.read_csv(box_path)

    # Parse datetimes
    if "date" in games_df.columns:
        games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce")
    if "game_date" in player_box_df.columns:
        player_box_df["game_date"] = pd.to_datetime(player_box_df["game_date"], errors="coerce")

    feature_df = get_feature_builder().build_training_features(
        games_df=games_df,
        player_box_df=player_box_df,
        n_recent_games=n_recent_games,
    )

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        feature_df.to_csv(out_path, index=False)

    return feature_df


def build_matchup_features(home_team, away_team, game_date, n_recent_games=10):
    """Convenience wrapper for prediction-time features."""
    return get_feature_builder().build_matchup_features(
        home_team=home_team,
        away_team=away_team,
        game_date=game_date,
        n_recent_games=n_recent_games,
    )
