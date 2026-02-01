"""
NBA team mapping utilities.

Maps between team names, abbreviations, and IDs.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Comprehensive team mapping
# Format: abbreviation -> (team_id, full_name, city, nickname, [aliases])
TEAM_DATA = {
    "ATL": (1610612737, "Atlanta Hawks", "Atlanta", "Hawks", ["hawks", "atl"]),
    "BOS": (1610612738, "Boston Celtics", "Boston", "Celtics", ["celtics", "boston"]),
    "BKN": (1610612751, "Brooklyn Nets", "Brooklyn", "Nets", ["nets", "brooklyn", "nj", "njn"]),
    "CHA": (1610612766, "Charlotte Hornets", "Charlotte", "Hornets", ["hornets", "charlotte", "cha", "cho"]),
    "CHI": (1610612741, "Chicago Bulls", "Chicago", "Bulls", ["bulls", "chicago"]),
    "CLE": (1610612739, "Cleveland Cavaliers", "Cleveland", "Cavaliers", ["cavaliers", "cavs", "cleveland"]),
    "DAL": (1610612742, "Dallas Mavericks", "Dallas", "Mavericks", ["mavericks", "mavs", "dallas"]),
    "DEN": (1610612743, "Denver Nuggets", "Denver", "Nuggets", ["nuggets", "denver"]),
    "DET": (1610612765, "Detroit Pistons", "Detroit", "Pistons", ["pistons", "detroit"]),
    "GSW": (1610612744, "Golden State Warriors", "Golden State", "Warriors", ["warriors", "golden state", "gs"]),
    "HOU": (1610612745, "Houston Rockets", "Houston", "Rockets", ["rockets", "houston"]),
    "IND": (1610612754, "Indiana Pacers", "Indiana", "Pacers", ["pacers", "indiana"]),
    "LAC": (1610612746, "LA Clippers", "Los Angeles", "Clippers", ["clippers", "lac", "la clippers"]),
    "LAL": (1610612747, "Los Angeles Lakers", "Los Angeles", "Lakers", ["lakers", "lal", "la lakers"]),
    "MEM": (1610612763, "Memphis Grizzlies", "Memphis", "Grizzlies", ["grizzlies", "memphis", "grizz"]),
    "MIA": (1610612748, "Miami Heat", "Miami", "Heat", ["heat", "miami"]),
    "MIL": (1610612749, "Milwaukee Bucks", "Milwaukee", "Bucks", ["bucks", "milwaukee"]),
    "MIN": (1610612750, "Minnesota Timberwolves", "Minnesota", "Timberwolves", ["timberwolves", "wolves", "minnesota"]),
    "NOP": (1610612740, "New Orleans Pelicans", "New Orleans", "Pelicans", ["pelicans", "new orleans", "nola", "no"]),
    "NYK": (1610612752, "New York Knicks", "New York", "Knicks", ["knicks", "new york", "ny"]),
    "OKC": (1610612760, "Oklahoma City Thunder", "Oklahoma City", "Thunder", ["thunder", "oklahoma city", "okc"]),
    "ORL": (1610612753, "Orlando Magic", "Orlando", "Magic", ["magic", "orlando"]),
    "PHI": (1610612755, "Philadelphia 76ers", "Philadelphia", "76ers", ["76ers", "sixers", "philadelphia", "philly"]),
    "PHX": (1610612756, "Phoenix Suns", "Phoenix", "Suns", ["suns", "phoenix", "phx"]),
    "POR": (1610612757, "Portland Trail Blazers", "Portland", "Trail Blazers", ["trail blazers", "blazers", "portland"]),
    "SAC": (1610612758, "Sacramento Kings", "Sacramento", "Kings", ["kings", "sacramento"]),
    "SAS": (1610612759, "San Antonio Spurs", "San Antonio", "Spurs", ["spurs", "san antonio", "sa"]),
    "TOR": (1610612761, "Toronto Raptors", "Toronto", "Raptors", ["raptors", "toronto"]),
    "UTA": (1610612762, "Utah Jazz", "Utah", "Jazz", ["jazz", "utah"]),
    "WAS": (1610612764, "Washington Wizards", "Washington", "Wizards", ["wizards", "washington", "dc"]),
}

# Build lookup dictionaries
TEAM_MAP = {abbr: data[1] for abbr, data in TEAM_DATA.items()}  # abbr -> full name
TEAM_IDS = {abbr: data[0] for abbr, data in TEAM_DATA.items()}  # abbr -> team_id
ID_TO_ABBR = {data[0]: abbr for abbr, data in TEAM_DATA.items()}  # team_id -> abbr

# Build alias lookup
_ALIAS_TO_ABBR = {}
for abbr, data in TEAM_DATA.items():
    _ALIAS_TO_ABBR[abbr.lower()] = abbr
    _ALIAS_TO_ABBR[data[1].lower()] = abbr  # full name
    _ALIAS_TO_ABBR[data[2].lower()] = abbr  # city
    _ALIAS_TO_ABBR[data[3].lower()] = abbr  # nickname
    for alias in data[4]:
        _ALIAS_TO_ABBR[alias.lower()] = abbr


def resolve_team_name(team_input: str) -> Optional[str]:
    """
    Resolve any team identifier to standard 3-letter abbreviation.
    
    Accepts:
    - Abbreviation (BOS, LAL)
    - Full name (Boston Celtics)
    - City (Boston)
    - Nickname (Celtics)
    - Common aliases (Cavs, Sixers)
    
    Returns:
        3-letter abbreviation or None if not found
    """
    if not team_input:
        return None
    
    normalized = team_input.strip().lower()
    
    # Direct lookup
    if normalized in _ALIAS_TO_ABBR:
        return _ALIAS_TO_ABBR[normalized]
    
    # Try partial match
    for alias, abbr in _ALIAS_TO_ABBR.items():
        if normalized in alias or alias in normalized:
            return abbr
    
    return None


def get_team_id(team_input: str) -> Optional[int]:
    """
    Get NBA.com team ID from any team identifier.
    
    Args:
        team_input: Team name, abbreviation, or alias
        
    Returns:
        Team ID or None
    """
    abbr = resolve_team_name(team_input)
    if abbr:
        return TEAM_IDS.get(abbr)
    return None


def get_team_full_name(team_input: str) -> Optional[str]:
    """Get full team name from any identifier."""
    abbr = resolve_team_name(team_input)
    if abbr:
        return TEAM_MAP.get(abbr)
    return None


def abbr_from_id(team_id: int) -> Optional[str]:
    """Get team abbreviation from team ID."""
    return ID_TO_ABBR.get(team_id)


def get_all_teams() -> list[dict]:
    """Get list of all teams with their info."""
    return [
        {
            "abbreviation": abbr,
            "team_id": data[0],
            "full_name": data[1],
            "city": data[2],
            "nickname": data[3],
        }
        for abbr, data in TEAM_DATA.items()
    ]

