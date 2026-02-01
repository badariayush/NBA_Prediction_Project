"""
Player-to-Team Feature Aggregator.

Aggregates individual player stats to team-level features using:
- Minutes-weighted averaging
- Shooting percentages from makes/attempts (not rounded %)
- Recent form vs season-to-date blending
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PlayerToTeamAggregator:
    """
    Aggregates player-level stats to team-level matchup features.
    
    Features created:
    - team_exp_minutes_sum: Total expected team minutes
    - Weighted stats per 48 minutes: PTS/REB/AST
    - Shooting percentages: FG%, 3P%
    - Concentration metrics: top-5 minutes share, top-3 usage
    """
    
    # Blending weights
    RECENT_WEIGHT = 0.7  # Recent form (last N games)
    SEASON_WEIGHT = 0.3  # Season-to-date
    
    def __init__(self, recent_weight: float = 0.7):
        self.recent_weight = recent_weight
        self.season_weight = 1.0 - recent_weight
    
    def aggregate_team_features(
        self,
        roster_with_stats: pd.DataFrame,
        team_name: str = None
    ) -> dict:
        """
        Aggregate player stats to team features.
        
        Args:
            roster_with_stats: DataFrame with player stats including:
                - expected_minutes
                - pts_recent, reb_recent, ast_recent (recent form)
                - fgm_recent, fga_recent, fg3m_recent, fg3a_recent
                - pts_season, reb_season, etc. (season stats)
                - availability_prob
                
        Returns:
            Dict of team-level features
        """
        if roster_with_stats.empty:
            return self._empty_features(team_name)
        
        df = roster_with_stats.copy()
        
        # Filter to available players
        df = df[df.get("availability_prob", 1.0) > 0]
        
        if df.empty:
            return self._empty_features(team_name)
        
        features = {"team": team_name} if team_name else {}
        
        # Total expected minutes
        total_minutes = df["expected_minutes"].sum()
        features["team_exp_minutes_sum"] = total_minutes
        features["n_active_players"] = len(df)
        
        if total_minutes == 0:
            return features
        
        # Minutes weights
        weights = df["expected_minutes"].values / total_minutes
        
        # =====================================================================
        # Weighted Stats per 48 minutes
        # =====================================================================
        
        for stat in ["pts", "reb", "ast"]:
            recent_col = f"{stat}_recent"
            season_col = f"{stat}_season"
            min_recent_col = "minutes_recent"
            min_season_col = "minutes_season"
            
            # Recent per-minute rate
            if recent_col in df.columns and min_recent_col in df.columns:
                per_min_recent = df[recent_col] / df[min_recent_col].replace(0, np.nan)
                per_min_recent = per_min_recent.fillna(0)
            else:
                per_min_recent = pd.Series([0] * len(df))
            
            # Season per-minute rate
            if season_col in df.columns and min_season_col in df.columns:
                per_min_season = df[season_col] / df[min_season_col].replace(0, np.nan)
                per_min_season = per_min_season.fillna(0)
            else:
                per_min_season = per_min_recent
            
            # Blend recent and season
            blended_rate = (
                self.recent_weight * per_min_recent +
                self.season_weight * per_min_season
            )
            
            # Weighted team average per 48 minutes
            weighted_rate = np.average(blended_rate.fillna(0), weights=weights)
            features[f"team_{stat}_per48"] = weighted_rate * 48
        
        # =====================================================================
        # Shooting Percentages (from makes/attempts, not rounded %)
        # =====================================================================
        
        for made_col, att_col, pct_name in [
            ("fgm_recent", "fga_recent", "fg_pct"),
            ("fg3m_recent", "fg3a_recent", "fg3_pct"),
            ("ftm_recent", "fta_recent", "ft_pct"),
        ]:
            # Weight by expected minutes and attempts
            if made_col in df.columns and att_col in df.columns:
                # Scale attempts by expected minutes / actual minutes played
                exp_makes = (df[made_col] * weights).sum()
                exp_attempts = (df[att_col] * weights).sum()
                
                if exp_attempts > 0:
                    features[f"team_{pct_name}"] = exp_makes / exp_attempts
                else:
                    features[f"team_{pct_name}"] = 0.0
            else:
                features[f"team_{pct_name}"] = 0.0
        
        # Total projected makes/attempts
        for stat in ["fgm", "fga", "fg3m", "fg3a"]:
            col = f"{stat}_recent"
            if col in df.columns:
                # Scale by expected minutes ratio
                features[f"team_{stat}_proj"] = (df[col] * weights * len(df)).sum()
        
        # =====================================================================
        # Concentration Metrics
        # =====================================================================
        
        # Top 5 minutes share
        top_5_minutes = df.nlargest(5, "expected_minutes")["expected_minutes"].sum()
        features["top5_minutes_share"] = top_5_minutes / max(total_minutes, 1)
        
        # Top 3 scoring share (usage proxy)
        if "pts_recent" in df.columns:
            df_sorted = df.sort_values("pts_recent", ascending=False)
            top_3_pts = df_sorted.head(3)["pts_recent"].sum()
            total_pts = df["pts_recent"].sum()
            features["top3_usage_share"] = top_3_pts / max(total_pts, 1)
        else:
            features["top3_usage_share"] = 0.0
        
        # =====================================================================
        # Raw totals (for additional features)
        # =====================================================================
        
        for stat in ["pts", "reb", "ast"]:
            recent_col = f"{stat}_recent"
            if recent_col in df.columns:
                features[f"team_{stat}_total_recent"] = (df[recent_col] * weights).sum() * len(df)
        
        return features
    
    def create_matchup_features(
        self,
        home_features: dict,
        away_features: dict
    ) -> dict:
        """
        Create matchup features from home and away team features.
        
        Computes differentials and combined features.
        """
        features = {}
        
        # Home features with prefix
        for key, val in home_features.items():
            if key != "team":
                features[f"home_{key}"] = val
        
        # Away features with prefix
        for key, val in away_features.items():
            if key != "team":
                features[f"away_{key}"] = val
        
        # Differentials (home - away)
        diff_features = [
            "team_pts_per48",
            "team_reb_per48",
            "team_ast_per48",
            "team_fg_pct",
            "team_fg3_pct",
            "team_exp_minutes_sum",
            "top5_minutes_share",
            "top3_usage_share",
        ]
        
        for feat in diff_features:
            home_val = home_features.get(feat, 0)
            away_val = away_features.get(feat, 0)
            features[f"diff_{feat}"] = home_val - away_val
        
        # Home indicator
        features["is_home"] = 1  # Always 1 for home team perspective
        
        return features
    
    def _empty_features(self, team_name: str = None) -> dict:
        """Return empty feature dict when no data available."""
        features = {"team": team_name} if team_name else {}
        features.update({
            "team_exp_minutes_sum": 0,
            "n_active_players": 0,
            "team_pts_per48": 0,
            "team_reb_per48": 0,
            "team_ast_per48": 0,
            "team_fg_pct": 0,
            "team_fg3_pct": 0,
            "team_ft_pct": 0,
            "top5_minutes_share": 0,
            "top3_usage_share": 0,
        })
        return features

