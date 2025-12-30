import os
import pickle

import pandas as pd

# Configuration
MODEL_PATH = "models/logreg.pkl"
INPUT_PATH = "data/raw/upcoming_games.csv"
OUTPUT_PATH = "results/predictions.csv"


def main():
    print("Predict script running")
    
    # Load trained model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {MODEL_PATH}")
    
    # Load upcoming games
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} upcoming games from {INPUT_PATH}")
    
    # Build feature (same as training, but neutral value since no scores available)
    df["feature"] = 0  # Neutral placeholder for prediction
    
    # Predict probabilities
    X = df[["feature"]]
    home_win_proba = model.predict_proba(X)[:, 1]
    
    # Build output dataframe
    output_df = df[["date", "season", "home_team", "away_team"]].copy()
    output_df["home_win_proba"] = home_win_proba
    output_df["home_win_pct"] = (home_win_proba * 100).round(1).astype(str) + "%"
    
    # Save predictions
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    
    # Print table to console
    print("\n" + "=" * 80)
    print("Predictions:")
    print("=" * 80)
    print(output_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

