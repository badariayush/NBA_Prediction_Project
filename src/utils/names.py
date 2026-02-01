"""
Player name normalization and fuzzy matching.

Handles:
- Suffixes (Jr., Sr., III, IV)
- Accents and special characters
- Name variations
"""

import re
import unicodedata
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Common name variations
NAME_ALIASES = {
    "nic claxton": "nicolas claxton",
    "nick claxton": "nicolas claxton",
    "pj washington": "p.j. washington",
    "og anunoby": "o.g. anunoby",
    "cj mccollum": "c.j. mccollum",
    "tj mcconnell": "t.j. mcconnell",
    "rj barrett": "r.j. barrett",
    "svi mykhailiuk": "sviatoslav mykhailiuk",
    "nene": "nene hilario",
    "moe wagner": "moritz wagner",
    "ish smith": "ishmael smith",
    "herb jones": "herbert jones",
    "jd davison": "j.d. davison",
    "aj green": "a.j. green",
    "giannis": "giannis antetokounmpo",
    "luka": "luka doncic",
}


def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for consistent matching.
    
    Transformations:
    - Lowercase
    - Remove accents
    - Remove suffixes (Jr., Sr., III, IV, II)
    - Remove extra whitespace
    - Apply known aliases
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized name string
    """
    if not name:
        return ""
    
    # Lowercase
    name = name.lower().strip()
    
    # Remove accents (é -> e, ć -> c, etc.)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    
    # Remove suffixes
    suffixes = [
        r"\s+jr\.?$",
        r"\s+sr\.?$",
        r"\s+iii$",
        r"\s+iv$",
        r"\s+ii$",
        r"\s+v$",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    
    # Remove periods from initials but keep them separate
    # "P.J." -> "p.j." (keep the dots for now, they help with matching)
    
    # Normalize whitespace
    name = " ".join(name.split())
    
    # Apply aliases
    if name in NAME_ALIASES:
        name = NAME_ALIASES[name]
    
    return name


def fuzzy_match_name(
    target: str,
    candidates: list[str],
    threshold: float = 0.8
) -> Optional[str]:
    """
    Find the best matching name from candidates.
    
    Uses simple character-based similarity (no external deps).
    
    Args:
        target: Name to match
        candidates: List of possible matches
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Best matching name or None if below threshold
    """
    target_norm = normalize_player_name(target)
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        cand_norm = normalize_player_name(candidate)
        
        # Exact match after normalization
        if target_norm == cand_norm:
            return candidate
        
        # Calculate similarity
        score = _string_similarity(target_norm, cand_norm)
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match
    
    return None


def _string_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity using character bigrams.
    
    Returns a score between 0 (no match) and 1 (identical).
    """
    if not s1 or not s2:
        return 0.0
    
    if s1 == s2:
        return 1.0
    
    # Create bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    
    b1 = bigrams(s1)
    b2 = bigrams(s2)
    
    if not b1 or not b2:
        return 0.0
    
    # Dice coefficient
    intersection = len(b1 & b2)
    return 2.0 * intersection / (len(b1) + len(b2))


def extract_first_last(name: str) -> tuple[str, str]:
    """
    Extract first and last name from full name.
    
    Returns (first_name, last_name)
    """
    parts = name.strip().split()
    
    if len(parts) == 0:
        return "", ""
    elif len(parts) == 1:
        return "", parts[0]
    else:
        return parts[0], " ".join(parts[1:])


def match_name_variations(
    name: str,
    candidates: dict[str, any],
    return_value: bool = True
) -> Optional[any]:
    """
    Match a name against a dictionary with various matching strategies.
    
    Args:
        name: Name to look up
        candidates: Dict of name -> value
        return_value: If True, return the dict value; else return the matched key
        
    Returns:
        Matched value/key or None
    """
    name_norm = normalize_player_name(name)
    
    # Normalize all candidate keys
    norm_to_orig = {normalize_player_name(k): k for k in candidates}
    
    # Exact match
    if name_norm in norm_to_orig:
        key = norm_to_orig[name_norm]
        return candidates[key] if return_value else key
    
    # Try fuzzy match
    matched_key = fuzzy_match_name(name_norm, list(norm_to_orig.keys()))
    if matched_key:
        orig_key = norm_to_orig[matched_key]
        return candidates[orig_key] if return_value else orig_key
    
    return None

