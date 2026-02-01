"""
Tests for player-to-team feature aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.aggregator import PlayerToTeamAggregator


class TestPlayerToTeamAggregator:
    """Test feature aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        return PlayerToTeamAggregator(recent_weight=0.7)
    
    @pytest.fixture
    def sample_roster(self):
        """Sample roster with player stats."""
        return pd.DataFrame({
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["Player A", "Player B", "Player C", "Player D", "Player E"],
            "expected_minutes": [35.0, 32.0, 28.0, 22.0, 15.0],
            "pts_recent": [25.0, 20.0, 15.0, 10.0, 8.0],
            "reb_recent": [8.0, 6.0, 4.0, 3.0, 2.0],
            "ast_recent": [5.0, 4.0, 3.0, 2.0, 1.0],
            "fgm_recent": [9.0, 7.0, 5.0, 4.0, 3.0],
            "fga_recent": [18.0, 15.0, 12.0, 10.0, 8.0],
            "fg3m_recent": [2.0, 3.0, 2.0, 1.0, 1.0],
            "fg3a_recent": [5.0, 8.0, 6.0, 4.0, 3.0],
            "minutes_recent": [35.0, 32.0, 28.0, 22.0, 15.0],
            "minutes_season": [34.0, 31.0, 27.0, 21.0, 14.0],
            "availability_prob": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
    
    def test_aggregate_empty_roster(self, aggregator):
        """Test handling of empty roster."""
        empty_df = pd.DataFrame()
        result = aggregator.aggregate_team_features(empty_df, "TEST")
        
        assert result["team"] == "TEST"
        assert result["team_exp_minutes_sum"] == 0
        assert result["n_active_players"] == 0
    
    def test_total_minutes(self, aggregator, sample_roster):
        """Test total expected minutes calculation."""
        result = aggregator.aggregate_team_features(sample_roster, "TEST")
        
        expected_total = 35.0 + 32.0 + 28.0 + 22.0 + 15.0
        assert result["team_exp_minutes_sum"] == expected_total
        assert result["n_active_players"] == 5
    
    def test_weighted_stats(self, aggregator, sample_roster):
        """Test minutes-weighted stat calculation."""
        result = aggregator.aggregate_team_features(sample_roster, "TEST")
        
        # Should have per-48 stats
        assert "team_pts_per48" in result
        assert "team_reb_per48" in result
        assert "team_ast_per48" in result
        
        # Per-48 stats should be positive
        assert result["team_pts_per48"] > 0
    
    def test_shooting_percentages(self, aggregator, sample_roster):
        """Test shooting percentage calculation."""
        result = aggregator.aggregate_team_features(sample_roster, "TEST")
        
        # FG% should be between 0 and 1
        assert 0 <= result["team_fg_pct"] <= 1
        assert 0 <= result["team_fg3_pct"] <= 1
    
    def test_concentration_metrics(self, aggregator, sample_roster):
        """Test top-5 minutes share and usage metrics."""
        result = aggregator.aggregate_team_features(sample_roster, "TEST")
        
        # Top-5 should be 100% with only 5 players
        assert result["top5_minutes_share"] == pytest.approx(1.0)
        
        # Top-3 usage should be meaningful
        assert 0 <= result["top3_usage_share"] <= 1
    
    def test_unavailable_players_excluded(self, aggregator, sample_roster):
        """Test that unavailable players are excluded."""
        roster = sample_roster.copy()
        roster.loc[0, "availability_prob"] = 0  # Player A is out
        
        result = aggregator.aggregate_team_features(roster, "TEST")
        
        # Should only count 4 available players
        assert result["n_active_players"] == 4
        
        # Total minutes should exclude Player A
        expected_total = 32.0 + 28.0 + 22.0 + 15.0
        assert result["team_exp_minutes_sum"] == expected_total


class TestMatchupFeatures:
    """Test matchup feature creation."""
    
    @pytest.fixture
    def aggregator(self):
        return PlayerToTeamAggregator()
    
    def test_matchup_has_home_away(self, aggregator):
        """Test matchup features have home/away prefixes."""
        home_features = {"team_pts_per48": 120.0, "team_fg_pct": 0.48}
        away_features = {"team_pts_per48": 115.0, "team_fg_pct": 0.46}
        
        result = aggregator.create_matchup_features(home_features, away_features)
        
        assert "home_team_pts_per48" in result
        assert "away_team_pts_per48" in result
    
    def test_matchup_has_differentials(self, aggregator):
        """Test matchup features have differential columns."""
        home_features = {"team_pts_per48": 120.0, "team_fg_pct": 0.48}
        away_features = {"team_pts_per48": 115.0, "team_fg_pct": 0.46}
        
        result = aggregator.create_matchup_features(home_features, away_features)
        
        assert "diff_team_pts_per48" in result
        assert result["diff_team_pts_per48"] == pytest.approx(5.0)
        
        assert "diff_team_fg_pct" in result
        assert result["diff_team_fg_pct"] == pytest.approx(0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

