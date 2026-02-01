"""
Model Training Module for NBA Game Prediction.

Trains a classifier on player-based features with:
- Time-based train/validation split (no future leakage)
- Recency weighting (recent seasons matter more)
- Evaluation metrics saved to results/

Usage:
    python -m src.train
    python -m src.train --features data/processed/training_features.csv
    python -m src.train --val-fraction 0.15
"""

import argparse
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

FEATURES_PATH = "data/processed/training_features.csv"
LEGACY_FEATURES_PATH = "data/processed/games_features.csv"
MODEL_PATH = "models/logreg.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"
RESULTS_DIR = "results"

# Default recency weights by season
# More recent seasons get higher weights
RECENCY_WEIGHTS = {
    2025: 1.0,   # 2025-26 season
    2024: 0.85,  # 2024-25 season
    2023: 0.7,   # 2023-24 season
    2022: 0.5,
    2021: 0.35,
    2020: 0.25,
}
DEFAULT_WEIGHT = 0.15  # For seasons not in the dict

# Feature column patterns for new player-based features
PLAYER_FEATURE_PATTERNS = [
    "diff_team_pts",
    "diff_team_reb",
    "diff_team_ast",
    "diff_team_pra",
    "diff_team_fg_pct",
    "diff_team_3p_pct",
    "diff_team_ft_pct",
]

# Legacy feature columns (for backwards compatibility)
LEGACY_FEATURE_COLS = [
    "pf_roll1_diff",
    "pa_roll1_diff",
    "pd_roll1_diff",
    "pf_roll2_diff",
    "pa_roll2_diff",
    "pd_roll2_diff",
]


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_training_data(path: str) -> pd.DataFrame:
    """Load training data and parse columns."""
    logger.info(f"Loading training data from {path}")
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    
    logger.info(f"Loaded {len(df)} games")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Seasons: {sorted(df['season'].unique())}")
    
    return df


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Automatically detect available feature columns.
    
    Prefers player-based features if available, falls back to legacy.
    """
    columns = set(df.columns)
    
    # Check for player-based features
    player_features = []
    for pattern in PLAYER_FEATURE_PATTERNS:
        matching = [c for c in columns if c.startswith(pattern)]
        player_features.extend(matching)
    
    if player_features:
        logger.info(f"Detected {len(player_features)} player-based feature columns")
        return sorted(player_features)
    
    # Fall back to legacy features
    legacy_available = [c for c in LEGACY_FEATURE_COLS if c in columns]
    if legacy_available:
        logger.info(f"Using {len(legacy_available)} legacy feature columns")
        return legacy_available
    
    raise ValueError("No suitable feature columns found in data")


def compute_sample_weights(df: pd.DataFrame, weight_dict: dict = RECENCY_WEIGHTS) -> np.ndarray:
    """
    Compute sample weights based on season recency.
    
    More recent seasons get higher weights to emphasize current form.
    """
    weights = df["season"].map(lambda s: weight_dict.get(s, DEFAULT_WEIGHT))
    
    # Log weight distribution
    for season in sorted(df["season"].unique()):
        w = weight_dict.get(season, DEFAULT_WEIGHT)
        n = (df["season"] == season).sum()
        logger.info(f"  Season {season}: {n} games, weight={w:.2f}")
    
    return weights.values


def time_based_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (most recent games for validation).
    
    This ensures no future data leakage - we always train on past
    and validate on more recent games.
    """
    df = df.sort_values("date").reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - val_fraction))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train_df)} games ({train_df['date'].min()} to {train_df['date'].max()})")
    logger.info(f"  Val:   {len(val_df)} games ({val_df['date'].min()} to {val_df['date'].max()})")
    
    return train_df, val_df


# ============================================================================
# Model Training
# ============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: np.ndarray = None
) -> Pipeline:
    """
    Train logistic regression pipeline with optional sample weights.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )),
    ])
    
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    
    return pipeline


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str = "Validation"
) -> dict:
    """
    Evaluate model and return metrics dictionary.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_prob),
        "roc_auc": roc_auc_score(y, y_prob),
        "n_samples": len(y),
    }
    
    logger.info(f"\n{set_name} Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
    
    return metrics


# ============================================================================
# Save Outputs
# ============================================================================

def save_model(model: Pipeline, path: str = MODEL_PATH) -> None:
    """Save trained model to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def save_feature_columns(columns: list[str], path: str = FEATURE_COLS_PATH) -> None:
    """Save feature column list to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(columns, f, indent=2)
    logger.info(f"Feature columns saved to {path}")


def save_results(
    train_metrics: dict,
    val_metrics: dict,
    feature_cols: list[str],
    results_dir: str = RESULTS_DIR
) -> str:
    """Save training results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    
    results_path = os.path.join(results_dir, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results_path


# ============================================================================
# Main Training Function
# ============================================================================

def run_training(
    features_path: str = FEATURES_PATH,
    model_path: str = MODEL_PATH,
    val_fraction: float = 0.15,
    use_recency_weights: bool = True
) -> tuple[Pipeline, dict, dict]:
    """
    Run the complete training pipeline.
    
    Args:
        features_path: Path to training features CSV
        model_path: Output path for trained model
        val_fraction: Fraction of data for validation
        use_recency_weights: Whether to use recency-based sample weights
        
    Returns:
        Tuple of (model, train_metrics, val_metrics)
    """
    # Load data
    df = load_training_data(features_path)
    
    # Detect feature columns
    feature_cols = detect_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    
    # Drop rows with missing features
    df_clean = df.dropna(subset=feature_cols + ["home_win"]).copy()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} rows with missing values")
    
    if len(df_clean) == 0:
        raise ValueError("No valid rows after dropping NaNs")
    
    # Check for sufficient class balance
    y_all = df_clean["home_win"]
    if y_all.nunique() < 2:
        raise ValueError(f"Only one class present: {y_all.unique()}")
    
    logger.info(f"\nClass balance: {y_all.value_counts().to_dict()}")
    
    # Time-based split
    train_df, val_df = time_based_split(df_clean, val_fraction)
    
    # Prepare features and labels
    X_train = train_df[feature_cols]
    y_train = train_df["home_win"]
    X_val = val_df[feature_cols]
    y_val = val_df["home_win"]
    
    # Compute sample weights
    sample_weights = None
    if use_recency_weights:
        logger.info("\nComputing recency-based sample weights:")
        sample_weights = compute_sample_weights(train_df)
    
    # Train model
    logger.info("\nTraining model...")
    model = train_model(X_train, y_train, sample_weights)
    logger.info("Model trained successfully")
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    # Save outputs
    save_model(model, model_path)
    save_feature_columns(feature_cols)
    save_results(train_metrics, val_metrics, feature_cols)
    
    return model, train_metrics, val_metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train NBA game prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.train
  python -m src.train --features data/processed/training_features.csv
  python -m src.train --val-fraction 0.2 --no-recency-weights
        """
    )
    
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=f"Path to training features (default: {FEATURES_PATH} or {LEGACY_FEATURES_PATH})"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"Output path for model (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)"
    )
    parser.add_argument(
        "--no-recency-weights",
        action="store_true",
        help="Disable recency-based sample weighting"
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild feature table before training"
    )
    
    args = parser.parse_args()
    
    # Determine features path
    features_path = args.features
    if features_path is None:
        if os.path.exists(FEATURES_PATH):
            features_path = FEATURES_PATH
        elif os.path.exists(LEGACY_FEATURES_PATH):
            features_path = LEGACY_FEATURES_PATH
        else:
            logger.error(f"No features file found. Run 'python -m src.features' first.")
            return
    
    # Rebuild features if requested
    if args.rebuild_features:
        logger.info("Rebuilding feature table...")
        from src.features import build_training_dataset
        build_training_dataset(out_path=FEATURES_PATH)
        features_path = FEATURES_PATH
    
    # Check features file exists
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        logger.error("Run 'python -m src.features' first to build features.")
        return
    
    # Run training
    try:
        model, train_metrics, val_metrics = run_training(
            features_path=features_path,
            model_path=args.model_path,
            val_fraction=args.val_fraction,
            use_recency_weights=not args.no_recency_weights
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final Validation Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Final Validation ROC AUC:  {val_metrics['roc_auc']:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
