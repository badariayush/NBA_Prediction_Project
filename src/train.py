import os
import joblib

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configuration
FEATURES_PATH = "data/processed/games_features.csv"
RAW_PATH = "data/raw/games.csv"
MODEL_PATH = "models/logreg.pkl"
VALID_SEASON = 2024

# Features to use
FEATURE_COLS = [
    "pf_roll1_diff",
    "pa_roll1_diff",
    "pd_roll1_diff",
    "pf_roll2_diff",
    "pa_roll2_diff",
    "pd_roll2_diff",
]


def main() -> None:
    print("Train script running")

    # Build feature table if it doesn't exist
    if not os.path.exists(FEATURES_PATH):
        print(f"{FEATURES_PATH} not found. Building feature table...")
        from src.features import build_feature_table, save_feature_table

        feature_df = build_feature_table(RAW_PATH)
        save_feature_table(feature_df, FEATURES_PATH)

    # Load feature table
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} games from {FEATURES_PATH}")

    # Ensure season is numeric (prevents weird split bugs)
    df["season"] = df["season"].astype(int)

    # Drop rows with NaN in required features (early season games)
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    n_dropped = len(df) - len(df_clean)
    print(f"Dropped {n_dropped} rows with missing features")
    print(f"Using {len(df_clean)} games for training/validation")

    if len(df_clean) == 0:
        raise ValueError(
            "No usable rows after dropping NaNs. "
            "With a tiny dataset, use smaller rolling windows or add more games."
        )

    # Define target and features
    y = df_clean["home_win"]
    X = df_clean[FEATURE_COLS]

   # TEMP: tiny dataset fallback - train on all usable rows
    train_mask = df_clean["season"].notna()
    valid_mask = df_clean["season"].notna()


    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    print(f"\nTrain set: {len(X_train)} games (seasons < {VALID_SEASON})")
    print(f"Validation set: {len(X_valid)} games (season == {VALID_SEASON})")

    # Guard: logistic regression needs at least 2 classes in training
    if len(y_train) == 0:
        raise ValueError(
            f"Training split is empty. VALID_SEASON={VALID_SEASON}, "
            f"seasons present={sorted(df_clean['season'].unique().tolist())}"
        )

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training split has only one class: {y_train.unique().tolist()}. "
            f"Train rows={len(y_train)}. Add more data or reduce rolling windows."
        )

    # Build pipeline with StandardScaler + LogisticRegression
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)
    print("\nModel trained successfully")

    # Evaluate on validation set (only if validation exists)
    if len(y_valid) == 0:
        print("\nNo validation rows found for VALID_SEASON; skipping validation metrics.")
    else:
        y_pred = pipeline.predict(X_valid)
        y_prob = pipeline.predict_proba(X_valid)[:, 1]

        accuracy = accuracy_score(y_valid, y_pred)
        logloss = log_loss(y_valid, y_prob)

        print("\n" + "=" * 40)
        print("Validation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print("=" * 40)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()h
