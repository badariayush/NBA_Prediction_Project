"""
Tests for player name normalization.
"""

import pytest
from src.utils.names import normalize_player_name, fuzzy_match_name


class TestNormalization:
    """Test player name normalization."""
    
    def test_basic_normalization(self):
        """Test basic name cleaning."""
        assert normalize_player_name("LeBron James") == "lebron james"
        assert normalize_player_name("  Jayson Tatum  ") == "jayson tatum"
    
    def test_suffix_removal(self):
        """Test removal of name suffixes."""
        assert normalize_player_name("Tim Hardaway Jr.") == "tim hardaway"
        assert normalize_player_name("Gary Payton II") == "gary payton"
        assert normalize_player_name("Wendell Carter Jr") == "wendell carter"
        assert normalize_player_name("Robert Williams III") == "robert williams"
    
    def test_accent_removal(self):
        """Test removal of accents/diacritics."""
        assert normalize_player_name("Nikola Jokić") == "nikola jokic"
        assert normalize_player_name("José Calderón") == "jose calderon"
        assert normalize_player_name("Luka Dončić") == "luka doncic"
    
    def test_aliases(self):
        """Test common name aliases."""
        assert normalize_player_name("PJ Washington") == "p.j. washington"
        assert normalize_player_name("OG Anunoby") == "o.g. anunoby"
    
    def test_empty_input(self):
        """Test empty/None input handling."""
        assert normalize_player_name("") == ""
        assert normalize_player_name(None) == ""


class TestFuzzyMatching:
    """Test fuzzy name matching."""
    
    def test_exact_match(self):
        """Test exact match after normalization."""
        candidates = ["LeBron James", "Jayson Tatum", "Luka Doncic"]
        assert fuzzy_match_name("lebron james", candidates) == "LeBron James"
    
    def test_similar_names(self):
        """Test matching similar names."""
        candidates = ["Stephen Curry", "Seth Curry", "Steph Curry"]
        # Should find close match
        result = fuzzy_match_name("Stephen Curry", candidates)
        assert result is not None
    
    def test_no_match(self):
        """Test no match below threshold."""
        candidates = ["LeBron James", "Jayson Tatum"]
        result = fuzzy_match_name("xyz abc", candidates, threshold=0.8)
        assert result is None
    
    def test_suffix_mismatch(self):
        """Test matching names with different suffixes."""
        candidates = ["Tim Hardaway Jr.", "Tim Hardaway"]
        result = fuzzy_match_name("Tim Hardaway", candidates)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

