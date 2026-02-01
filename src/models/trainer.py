"""
Model Training Module.

Trains Gradient Boosting classifier with:
- Time-based train/validation split
- Recency-based sample weighting
- Feature importance analysis
- Multiple baseline comparisons
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# Feature columns for the new player-based model
FEATURE_COLS = [
    # Home team features
    "home_team_pts_per48",
    "home_team_reb_per48",
    "home_team_ast_per48",
    "home_team_fg_pct",
    "home_team_fg3_pct",
    "home_team_exp_minutes_sum",
    "home_top5_minutes_share",
    "home_top3_usage_share",
    "home_n_active_players",
    # Away team features
    "away_team_pts_per48",
    "away_team_reb_per48",
    "away_team_ast_per48",
    "away_team_fg_pct",
    "away_team_fg3_pct",
    "away_team_exp_minutes_sum",
    "away_top5_minutes_share",
    "away_top3_usage_share",
    "away_n_active_players",
    # Differentials
    "diff_team_pts_per48",
    "diff_team_reb_per48",
    "diff_team_ast_per48",
    "diff_team_fg_pct",
    "diff_team_fg3_pct",
    "diff_team_exp_minutes_sum",
    "diff_top5_minutes_share",
    "diff_top3_usage_share",
]

# Recency weights by season
RECENCY_WEIGHTS = {
    2025: 1.0,   # 2025-26 season
    2024: 0.85,  # 2024-25 season
    2023: 0.7,   # 2023-24 season
    2022: 0.5,
    2021: 0.35,
}
DEFAULT_WEIGHT = 0.2


class ModelTrainer:
    """
    Trains and evaluates NBA prediction models.
    
    Primary: HistGradientBoostingClassifier
    Baselines: Logistic Regression, Naive (home team wins)
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        results_dir: str = "results"
    ):
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Prepare feature matrix.
        
        Handles missing columns and NaN values.
        """
        if feature_cols is None:
            feature_cols = FEATURE_COLS
        
        # Use only available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if not available_cols:
            logger.warning("No feature columns found in data")
            # Try to detect feature columns dynamically
            available_cols = [c for c in df.columns if (
                c.startswith("home_") or 
                c.startswith("away_") or 
                c.startswith("diff_")
            ) and c not in ["home_team", "away_team", "home_win"]]
        
        X = df[available_cols].copy()
        X = X.fillna(0)
        
        return X, available_cols
    
    def compute_sample_weights(
        self,
        df: pd.DataFrame,
        weight_dict: dict = None
    ) -> np.ndarray:
        """Compute recency-based sample weights."""
        if weight_dict is None:
            weight_dict = RECENCY_WEIGHTS
        
        weights = df["season"].map(lambda s: weight_dict.get(s, DEFAULT_WEIGHT))
        return weights.values
    
    def train_gbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: np.ndarray = None
    ) -> HistGradientBoostingClassifier:
        """
        Train Gradient Boosting classifier.
        
        Uses HistGradientBoostingClassifier (sklearn native, no external deps).
        """
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    def train_logistic(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: np.ndarray = None
    ) -> Pipeline:
        """Train logistic regression baseline with scaling."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        return pipeline
    
    def evaluate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        name: str = "Model"
    ) -> dict:
        """Evaluate model performance."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        metrics = {
            "name": name,
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_prob),
            "roc_auc": roc_auc_score(y, y_prob),
            "brier_score": brier_score_loss(y, y_prob),
            "n_samples": len(y),
        }
        
        logger.info(f"{name} Metrics:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Log Loss:    {metrics['log_loss']:.4f}")
        logger.info(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        return metrics
    
    def train(
        self,
        df: pd.DataFrame,
        val_fraction: float = 0.15,
        use_recency_weights: bool = True
    ) -> dict:
        """
        Full training pipeline.
        
        Returns dict with models and metrics.
        """
        logger.info(f"Training on {len(df)} games...")
        
        # Prepare features
        X, feature_cols = self.prepare_features(df)
        y = df["home_win"]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        # Drop rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        df = df[valid_mask]
        
        # Time-based split
        df = df.sort_values("game_date").reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_fraction))
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        df_train = df.iloc[:split_idx]
        
        logger.info(f"Train: {len(X_train)} games, Val: {len(X_val)} games")
        
        # Sample weights
        sample_weights = None
        if use_recency_weights:
            sample_weights = self.compute_sample_weights(df_train)
        
        # Train models
        logger.info("Training Gradient Boosting...")
        gbm_model = self.train_gbm(X_train, y_train, sample_weights)
        
        logger.info("Training Logistic Regression baseline...")
        lr_model = self.train_logistic(X_train, y_train, sample_weights)
        
        # Evaluate
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING METRICS")
        gbm_train_metrics = self.evaluate_model(gbm_model, X_train, y_train, "GBM (Train)")
        lr_train_metrics = self.evaluate_model(lr_model, X_train, y_train, "LR (Train)")
        
        logger.info("\n" + "=" * 50)
        logger.info("VALIDATION METRICS")
        gbm_val_metrics = self.evaluate_model(gbm_model, X_val, y_val, "GBM (Val)")
        lr_val_metrics = self.evaluate_model(lr_model, X_val, y_val, "LR (Val)")
        
        # Naive baseline
        naive_accuracy = max(y_val.mean(), 1 - y_val.mean())
        logger.info(f"\nNaive baseline accuracy: {naive_accuracy:.4f}")
        
        # Save models
        gbm_path = os.path.join(self.model_dir, "gbm_model.pkl")
        lr_path = os.path.join(self.model_dir, "lr_model.pkl")
        
        joblib.dump(gbm_model, gbm_path)
        joblib.dump(lr_model, lr_path)
        logger.info(f"Saved GBM model to {gbm_path}")
        logger.info(f"Saved LR model to {lr_path}")
        
        # Save feature columns
        feature_cols_path = os.path.join(self.model_dir, "feature_cols_gbm.json")
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "gbm_train": gbm_train_metrics,
            "gbm_val": gbm_val_metrics,
            "lr_train": lr_train_metrics,
            "lr_val": lr_val_metrics,
            "naive_accuracy": naive_accuracy,
        }
        
        results_path = os.path.join(self.results_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return {
            "gbm_model": gbm_model,
            "lr_model": lr_model,
            "feature_cols": feature_cols,
            "results": results,
        }
    
    def get_feature_importance(
        self,
        model,
        feature_cols: list[str]
    ) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier", None), "coef_"):
            importance = np.abs(model.named_steps["classifier"].coef_[0])
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return df

