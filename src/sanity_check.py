"""
Sanity Check Script for NBA Prediction Pipeline.

Runs through the complete pipeline to verify everything works:
1. Data pulling (optional)
2. Feature building
3. Model training
4. Prediction

Usage:
    python -m src.sanity_check
    python -m src.sanity_check --full  # Force re-pull data
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd


def print_header(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def print_success(text: str):
    print(f"  ✓ {text}")


def print_error(text: str):
    print(f"  ✗ ERROR: {text}")


def print_info(text: str):
    print(f"    {text}")


def check_data_files(data_dir: str = "data/raw"):
    print_header("Checking Data Files")

    required_files = [
        os.path.join(data_dir, "games.csv"),
        os.path.join(data_dir, "player_boxscores.csv"),
    ]

    all_exist = True
    for f in required_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print_success(f"{f} exists ({len(df)} rows)")

            date_col = "date" if "date" in df.columns else "game_date"
            if date_col in df.columns:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                dates = dates.dropna()
                if not dates.empty:
                    print_info(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")

            if "season" in df.columns:
                seasons = sorted(df["season"].dropna().unique())
                print_info(f"Seasons: {seasons}")
                if 2025 in seasons:
                    print_success("2025-26 season data found!")
        else:
            print_error(f"{f} not found")
            all_exist = False

    return all_exist


def run_feature_building():
    print_header("Building Features")

    try:
        from src.features import build_training_dataset

        feature_df = build_training_dataset(
            raw_dir="data/raw",
            out_path="data/processed/training_features.csv",
            use_player_features=True,
            n_recent_games=10,
        )

        if feature_df is None or feature_df.empty:
            print_error("Features built but dataframe is empty.")
            return False

        print_success(f"Features built: {feature_df.shape}")

        if "date" in feature_df.columns:
            print_info(f"Date range: {feature_df['date'].min()} to {feature_df['date'].max()}")
        if "season" in feature_df.columns:
            print_info(f"Seasons: {sorted(feature_df['season'].dropna().unique())}")

        diff_cols = [c for c in feature_df.columns if c.startswith("diff_")]
        print_info(f"Features (diff_ cols): {len(diff_cols)}")

        return True

    except Exception as e:
        print_error(f"Feature building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training():
    print_header("Training Model")

    try:
        from src.train import run_training as train_fn

        model, train_metrics, val_metrics = train_fn(
            features_path="data/processed/training_features.csv",
            model_path="models/logreg.pkl",
            val_fraction=0.15,
            use_recency_weights=True
        )

        print_success("Model trained successfully")
        if isinstance(val_metrics, dict):
            if "accuracy" in val_metrics:
                print_info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            if "roc_auc" in val_metrics:
                print_info(f"Validation ROC AUC:  {val_metrics['roc_auc']:.4f}")
            if "log_loss" in val_metrics:
                print_info(f"Validation log loss: {val_metrics['log_loss']:.4f}")

        return True

    except Exception as e:
        print_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_prediction():
    print_header("Running Sample Prediction")

    try:
        import joblib
        from src.predict_cli import (
            load_player_boxscores,
            build_prediction_features,
            load_feature_columns
        )

        model = joblib.load("models/logreg.pkl")
        expected_cols = load_feature_columns()

        player_box_df = load_player_boxscores()
        if player_box_df is None or player_box_df.empty:
            print_info("No player boxscore data found, skipping prediction test")
            return True

        # Ensure datetime
        if "game_date" in player_box_df.columns:
            player_box_df["game_date"] = pd.to_datetime(player_box_df["game_date"], errors="coerce")

        max_date = player_box_df["game_date"].dropna().max() + pd.Timedelta(days=1)

        team_col = "team_abbreviation" if "team_abbreviation" in player_box_df.columns else "team_abbr"
        teams = player_box_df[team_col].dropna().unique()
        if len(teams) < 2:
            print_error("Not enough teams in data for prediction")
            return False

        home_team, away_team = teams[0], teams[1]
        print_info(f"Test matchup: {away_team} @ {home_team} on {max_date.strftime('%Y-%m-%d')}")

        feature_df, info = build_prediction_features(
            player_box_df, home_team, away_team, max_date, expected_cols
        )

        if feature_df is None or feature_df.empty:
            print_error("Could not build prediction features")
            return False

        # Align to expected columns
        if expected_cols:
            for col in expected_cols:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            feature_df = feature_df[expected_cols]

        home_prob = model.predict_proba(feature_df.fillna(0))[0, 1]

        print_success("Prediction successful!")
        print_info(f"{home_team} home win probability: {home_prob:.1%}")
        print_info(f"{away_team} away win probability: {(1-home_prob):.1%}")

        if isinstance(info, dict):
            if "home_n_players" in info and "away_n_players" in info:
                print_info(f"Active players: {home_team}={info['home_n_players']}, {away_team}={info['away_n_players']}")

        return True

    except Exception as e:
        print_error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_pull(seasons=None, force=False):
    print_header("Pulling Data")

    if seasons is None:
        seasons = ["2023-24", "2024-25", "2025-26"]

    try:
        from src.pull_nba_data import pull_all_seasons

        print_info(f"Seasons: {seasons}")
        print_info(f"Force refresh: {force}")

        games_df, player_box_df = pull_all_seasons(
            seasons=seasons,
            outdir="data/raw",
            force_refresh=force
        )

        print_success(f"Data pulled: {len(games_df)} games, {len(player_box_df)} player records")
        return True

    except Exception as e:
        print_error(f"Data pull failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Pipeline Sanity Check")
    parser.add_argument("--full", action="store_true", help="Run full pipeline including data pull")
    parser.add_argument("--pull-only", action="store_true", help="Only pull data")
    parser.add_argument("--force", action="store_true", help="Force re-pull data even if cached")
    parser.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25", "2025-26"],
                        help="Seasons to pull")

    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  NBA PREDICTION PIPELINE SANITY CHECK".center(68) + "║")
    print("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    results = {}

    if args.full or args.pull_only or args.force:
        results["data_pull"] = run_data_pull(args.seasons, args.force)
        if args.pull_only:
            sys.exit(0 if results["data_pull"] else 1)

    results["data_check"] = check_data_files()
    if not results["data_check"]:
        print()
        print_error("Data files missing. Run with --full to pull data first:")
        print("  python -m src.sanity_check --full")
        print()
        sys.exit(1)

    results["features"] = run_feature_building()
    if not results["features"]:
        print()
        print_error("Feature building failed. Cannot continue.")
        sys.exit(1)

    results["training"] = run_training()
    if not results["training"]:
        print()
        print_error("Training failed. Cannot continue.")
        sys.exit(1)

    results["prediction"] = run_prediction()

    print_header("Summary")
    all_passed = all(results.values())

    for step, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {step:20s} {status}")

    print()
    if all_passed:
        print("  🎉 All checks passed! Pipeline is working correctly.")
    else:
        print("  ⚠️  Some checks failed. Review errors above.")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
