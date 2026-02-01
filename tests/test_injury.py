"""
Tests for injury status parsing.
"""

import pytest
from src.providers.injury_provider import (
    BasketballReferenceInjuryProvider,
    INJURY_MINUTES_MAP
)


class TestInjuryStatusMapping:
    """Test injury status to minutes mapping."""
    
    def test_out_status(self):
        """OUT players get 0 minutes."""
        assert INJURY_MINUTES_MAP["out"] == 0.0
    
    def test_doubtful_status(self):
        """DOUBTFUL players get 10% minutes."""
        assert INJURY_MINUTES_MAP["doubtful"] == 0.10
    
    def test_questionable_status(self):
        """QUESTIONABLE is scenario-based (50%)."""
        assert INJURY_MINUTES_MAP["questionable"] == 0.50
    
    def test_probable_status(self):
        """PROBABLE players get 90% minutes."""
        assert INJURY_MINUTES_MAP["probable"] == 0.90
    
    def test_active_status(self):
        """ACTIVE players get full minutes."""
        assert INJURY_MINUTES_MAP["active"] == 1.0


class TestInjuryProvider:
    """Test injury provider functionality."""
    
    def test_status_normalization(self):
        """Test injury status normalization."""
        provider = BasketballReferenceInjuryProvider()
        
        assert provider._normalize_status("Out") == "OUT"
        assert provider._normalize_status("out") == "OUT"
        assert provider._normalize_status("OUT (knee)") == "OUT"
        assert provider._normalize_status("Questionable") == "QUESTIONABLE"
        assert provider._normalize_status("Day-To-Day") == "DAY-TO-DAY"
    
    def test_status_note_parsing(self):
        """Test parsing combined status and note."""
        provider = BasketballReferenceInjuryProvider()
        
        status, note = provider._parse_status_note("Out (knee)")
        assert status == "Out"
        assert note == "knee"
        
        status, note = provider._parse_status_note("Questionable (ankle)")
        assert status == "Questionable"
        assert note == "ankle"


class TestMinutesCalculation:
    """Test expected minutes calculations."""
    
    def test_healthy_player(self):
        """Healthy players get full expected minutes."""
        # A player with 30 min avg should get ~30 expected
        base_minutes = 30.0
        factor = INJURY_MINUTES_MAP["active"]
        assert base_minutes * factor == 30.0
    
    def test_questionable_player(self):
        """Questionable players get reduced expected minutes."""
        base_minutes = 30.0
        factor = INJURY_MINUTES_MAP["questionable"]
        assert base_minutes * factor == 15.0
    
    def test_out_player(self):
        """OUT players get 0 expected minutes."""
        base_minutes = 30.0
        factor = INJURY_MINUTES_MAP["out"]
        assert base_minutes * factor == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

