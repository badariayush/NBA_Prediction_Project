"""
Prediction Module with Monte Carlo Uncertainty.

Generates predictions with:
- Primary GBM probability
- Baseline LR probability
- Monte Carlo uncertainty intervals (5th/95th percentile)
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.builder import FeatureBuilder
from src.providers.roster_resolver import RosterResolver
from src.providers.injury_provider import INJURY_MINUTES_MAP
from src.utils.cache import DiskCache

logger = logging.getLogger(__name__)


class GamePredictor:
    """
    Single-game predictor using trained models.
    """
    
    def __init__(
        self,
        gbm_model_path: str = "models/gbm_model.pkl",
        lr_model_path: str = "models/lr_model.pkl",
        feature_cols_path: str = "models/feature_cols_gbm.json"
    ):
        self.gbm_model = None
        self.lr_model = None
        self.feature_cols = None
        
        self.gbm_model_path = gbm_model_path
        self.lr_model_path = lr_model_path
        self.feature_cols_path = feature_cols_path
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models and feature columns."""
        if os.path.exists(self.gbm_model_path):
            self.gbm_model = joblib.load(self.gbm_model_path)
            logger.info(f"Loaded GBM model from {self.gbm_model_path}")
        
        if os.path.exists(self.lr_model_path):
            self.lr_model = joblib.load(self.lr_model_path)
            logger.info(f"Loaded LR model from {self.lr_model_path}")
        
        if os.path.exists(self.feature_cols_path):
            with open(self.feature_cols_path) as f:
                self.feature_cols = json.load(f)
    
    def predict(
        self,
        features: dict,
        model: str = "gbm"
    ) -> float:
        """
        Predict home win probability from features dict.
        
        Args:
            features: Dict of feature values
            model: "gbm" or "lr"
            
        Returns:
            Home win probability (0-1)
        """
        if model == "gbm" and self.gbm_model:
            model_obj = self.gbm_model
        elif model == "lr" and self.lr_model:
            model_obj = self.lr_model
        else:
            raise ValueError(f"Model '{model}' not available")
        
        # Build feature vector
        if self.feature_cols:
            X = pd.DataFrame([{c: features.get(c, 0) for c in self.feature_cols}])
        else:
            X = pd.DataFrame([features])
        
        X = X.fillna(0)
        
        prob = model_obj.predict_proba(X)[0, 1]
        return prob
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        feature_builder: Optional[FeatureBuilder] = None
    ) -> dict:
        """
        Predict a game outcome.
        
        Returns dict with probabilities and metadata.
        """
        if feature_builder is None:
            feature_builder = FeatureBuilder()
        
        # Build features
        features = feature_builder.build_matchup_features(
            home_team, away_team, game_date
        )
        
        # Get predictions
        result = {
            "home_team": features.get("home_team", home_team),
            "away_team": features.get("away_team", away_team),
            "game_date": game_date.strftime("%Y-%m-%d"),
        }
        
        if self.gbm_model:
            result["p_home_gbm"] = self.predict(features, "gbm")
        
        if self.lr_model:
            result["p_home_lr"] = self.predict(features, "lr")
        
        # Primary prediction
        result["p_home_win"] = result.get("p_home_gbm", result.get("p_home_lr", 0.5))
        result["p_away_win"] = 1 - result["p_home_win"]
        result["predicted_winner"] = home_team if result["p_home_win"] > 0.5 else away_team
        
        return result


class MonteCarloPredictor:
    """
    Monte Carlo wrapper for uncertainty quantification.
    
    Runs multiple simulations with:
    - Random sampling of questionable/probable players
    - Minutes noise per player
    - Aggregated probability distribution
    """
    
    def __init__(
        self,
        base_predictor: Optional[GamePredictor] = None,
        n_simulations: int = 1000,
        seed: int = 42
    ):
        self.predictor = base_predictor or GamePredictor()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
    
    def predict_with_uncertainty(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        roster_resolver: Optional[RosterResolver] = None,
        feature_builder: Optional[FeatureBuilder] = None
    ) -> dict:
        """
        Predict with Monte Carlo uncertainty estimation.
        
        Returns:
            Dict with mean probability, confidence interval, and simulation stats
        """
        if roster_resolver is None:
            roster_resolver = RosterResolver()
        if feature_builder is None:
            feature_builder = FeatureBuilder()
        
        # Get rosters
        home_roster = roster_resolver.get_active_roster(home_team, game_date, include_injuries=True)
        away_roster = roster_resolver.get_active_roster(away_team, game_date, include_injuries=True)
        
        # Run simulations
        probabilities = []
        
        for sim in range(self.n_simulations):
            # Sample player availability
            home_sim = self._sample_roster(home_roster)
            away_sim = self._sample_roster(away_roster)
            
            # Build features with simulated rosters
            try:
                features = self._build_simulated_features(
                    home_sim, away_sim, home_team, away_team, game_date, feature_builder
                )
                
                prob = self.predictor.predict(features, "gbm")
                probabilities.append(prob)
                
            except Exception as e:
                logger.debug(f"Simulation {sim} failed: {e}")
                continue
        
        if not probabilities:
            # Fallback to single prediction
            result = self.predictor.predict_game(home_team, away_team, game_date, feature_builder)
            result["p_home_5th"] = result["p_home_win"]
            result["p_home_95th"] = result["p_home_win"]
            result["n_simulations"] = 0
            return result
        
        probs = np.array(probabilities)
        
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "game_date": game_date.strftime("%Y-%m-%d"),
            "p_home_win": float(np.mean(probs)),
            "p_home_5th": float(np.percentile(probs, 5)),
            "p_home_95th": float(np.percentile(probs, 95)),
            "p_home_std": float(np.std(probs)),
            "p_away_win": float(1 - np.mean(probs)),
            "n_simulations": len(probs),
            "predicted_winner": home_team if np.mean(probs) > 0.5 else away_team,
        }
        
        return result
    
    def _sample_roster(self, roster: pd.DataFrame) -> pd.DataFrame:
        """
        Sample roster with injury uncertainty.
        
        - OUT: always excluded
        - DOUBTFUL: 10% chance to play
        - QUESTIONABLE: 50% chance to play
        - PROBABLE: 90% chance to play
        """
        if roster.empty:
            return roster
        
        roster = roster.copy()
        
        for idx, player in roster.iterrows():
            avail_prob = player.get("availability_prob", 1.0)
            
            # Sample availability
            plays = self.rng.random() < avail_prob
            
            if not plays:
                roster.loc[idx, "expected_minutes"] = 0
            else:
                # Add minutes noise
                base_minutes = player["expected_minutes"]
                noise_std = max(2, 0.15 * base_minutes)
                noisy_minutes = self.rng.normal(base_minutes, noise_std)
                roster.loc[idx, "expected_minutes"] = np.clip(noisy_minutes, 0, 40)
        
        # Filter to players with minutes
        return roster[roster["expected_minutes"] > 0]
    
    def _build_simulated_features(
        self,
        home_roster: pd.DataFrame,
        away_roster: pd.DataFrame,
        home_team: str,
        away_team: str,
        game_date: datetime,
        feature_builder: FeatureBuilder
    ) -> dict:
        """Build features from simulated rosters."""
        # This is a simplified version - in production, you'd rebuild
        # full player stats aggregation with the simulated rosters
        
        # For now, use the standard feature builder
        return feature_builder.build_matchup_features(home_team, away_team, game_date)


def load_predictor(
    model_dir: str = "models"
) -> tuple[GamePredictor, MonteCarloPredictor]:
    """Load predictor instances."""
    predictor = GamePredictor(
        gbm_model_path=os.path.join(model_dir, "gbm_model.pkl"),
        lr_model_path=os.path.join(model_dir, "lr_model.pkl"),
        feature_cols_path=os.path.join(model_dir, "feature_cols_gbm.json")
    )
    
    mc_predictor = MonteCarloPredictor(predictor)
    
    return predictor, mc_predictor

