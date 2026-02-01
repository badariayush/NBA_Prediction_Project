"""
Batch Prediction Script for NBA Games.

Loads upcoming games from a CSV and predicts outcomes using the trained model.

Usage:
    python -m src.predict
    python -m src.predict --input data/schedules/upcoming.csv --output results/predictions.csv
"""

import argparse
import json
import logging
import os

import joblib
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Configuration
MODEL_PATH = "models/logreg.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"
INPUT_PATH = "data/raw/upcoming_games.csv"
OUTPUT_PATH = "results/predictions.csv"
PLAYER_BOX_PATH = "data/raw/player_boxscores.csv"


def load_model(path: str = MODEL_PATH):
    """Load trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at '{path}'. Run 'python -m src.train' first.")
    return joblib.load(path)


def load_feature_columns(path: str = FEATURE_COLS_PATH) -> list[str]:
    """Load expected feature columns."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Batch NBA game predictions")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Input games CSV")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output predictions CSV")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained model")
    
    args = parser.parse_args()
    
    logger.info("NBA Batch Prediction")
    
    # Load model
    try:
        model = load_model(args.model)
        logger.info(f"Loaded model from {args.model}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Load expected columns
    expected_cols = load_feature_columns()
    
    # Load upcoming games
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    games_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(games_df)} games from {args.input}")
    
    if games_df.empty:
        logger.warning("No games to predict")
        return
    
    # Check if we have player data for feature building
    player_box_df = None
    if os.path.exists(PLAYER_BOX_PATH):
        player_box_df = pd.read_csv(PLAYER_BOX_PATH)
        player_box_df["game_date"] = pd.to_datetime(player_box_df["game_date"])
    
    predictions = []
    
    for _, game in games_df.iterrows():
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        game_date = pd.to_datetime(game.get("date", game.get("game_date")))
        
        if player_box_df is not None and not player_box_df.empty:
            # Use player-based features
            from src.predict_cli import build_prediction_features
            
            feature_df, info = build_prediction_features(
                player_box_df, home_team, away_team, game_date, expected_cols
            )
            
            if feature_df.empty:
                logger.warning(f"Could not build features for {away_team} @ {home_team}")
                home_prob = 0.5
            else:
                if expected_cols:
                    for col in expected_cols:
                        if col not in feature_df.columns:
                            feature_df[col] = 0.0
                    feature_df = feature_df[expected_cols]
                
                home_prob = model.predict_proba(feature_df.fillna(0))[0, 1]
        else:
            # Fallback: neutral prediction
            logger.warning("No player data available, using neutral prediction")
            home_prob = 0.5
        
        predictions.append({
            "date": game_date.strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_prob,
            "away_win_prob": 1 - home_prob,
            "predicted_winner": home_team if home_prob > 0.5 else away_team
        })
    
    # Save predictions
    output_df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} predictions to {args.output}")
    
    # Print results
    print()
    print("=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    for _, row in output_df.iterrows():
        print(f"  {row['date']}  {row['away_team']:>3s} @ {row['home_team']:<3s}  "
              f"→  {row['predicted_winner']} ({row['home_win_prob']:.1%} home)")
    print("=" * 80)


if __name__ == "__main__":
    main()
