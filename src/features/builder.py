"""
Feature Builder - End-to-end feature construction for predictions.

Combines:
- Roster resolution
- Player stats fetching
- Team aggregation
- Matchup feature creation
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.providers.nba_api_provider import NbaApiProvider
from src.providers.roster_resolver import RosterResolver
from src.providers.injury_provider import BasketballReferenceInjuryProvider
from src.features.aggregator import PlayerToTeamAggregator
from src.utils.cache import DiskCache
from src.utils.team_map import resolve_team_name

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds prediction features for NBA matchups.

    End-to-end pipeline:
    1. Resolve rosters for both teams
    2. Fetch player stats (recent + season)
    3. Apply injury adjustments
    4. Aggregate to team features
    5. Create matchup differentials
    """

    # Default parameters
    RECENT_GAMES = 10  # Games for recent form
    MIN_GAMES_RECENT = 3  # Minimum games for recent form

    def __init__(
        self,
        nba_provider: Optional[NbaApiProvider] = None,
        roster_resolver: Optional[RosterResolver] = None,
        injury_provider: Optional[BasketballReferenceInjuryProvider] = None,
        cache: Optional[DiskCache] = None,
    ):
        self.nba = nba_provider or NbaApiProvider()
        self.roster = roster_resolver or RosterResolver(nba_provider=self.nba)
        self.injury = injury_provider or BasketballReferenceInjuryProvider()
        self.aggregator = PlayerToTeamAggregator()
        self.cache = cache or DiskCache(cache_dir="data/cache/features")

    def build_matchup_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        n_recent_games: int = None,
    ) -> dict:
        """
        Build features for a home vs away matchup.
        """
        n_recent = n_recent_games or self.RECENT_GAMES

        home_abbr = resolve_team_name(home_team)
        away_abbr = resolve_team_name(away_team)

        if not home_abbr or not away_abbr:
            raise ValueError(f"Could not resolve teams: {home_team}, {away_team}")

        logger.info(f"Building features for {away_abbr} @ {home_abbr} on {game_date.date()}")

        home_features = self._build_team_features(home_abbr, game_date, n_recent)
        away_features = self._build_team_features(away_abbr, game_date, n_recent)

        matchup_features = self.aggregator.create_matchup_features(home_features, away_features)

        matchup_features["home_team"] = home_abbr
        matchup_features["away_team"] = away_abbr
        matchup_features["game_date"] = game_date.strftime("%Y-%m-%d")

        return matchup_features

    def _build_team_features(
        self,
        team: str,
        game_date: datetime,
        n_recent_games: int,
    ) -> dict:
        """Build aggregated features for a single team using live roster + API stats."""
        try:
            roster = self.roster.get_active_roster(team, game_date)
        except Exception as e:
            logger.warning(f"Could not get roster for {team}: {e}")
            return self.aggregator._empty_features(team)

        if roster.empty:
            logger.warning(f"Empty roster for {team}")
            return self.aggregator._empty_features(team)

        season = game_date.year if game_date.month >= 10 else game_date.year - 1

        roster = self._add_player_stats(roster, game_date, season, n_recent_games)

        team_features = self.aggregator.aggregate_team_features(roster, team)
        return team_features

    def _add_player_stats(
        self,
        roster: pd.DataFrame,
        game_date: datetime,
        season: int,
        n_recent: int,
    ) -> pd.DataFrame:
        """Add player stats (recent form + season) to roster."""
        roster = roster.copy()

        stat_cols = ["pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "minutes"]
        for col in stat_cols:
            roster[f"{col}_recent"] = 0.0
            roster[f"{col}_season"] = 0.0

        roster["games_recent"] = 0
        roster["games_season"] = 0

        for idx, player in roster.iterrows():
            player_id = player["player_id"]

            try:
                recent_stats = self.nba.get_player_recent_form(
                    player_id, game_date, n_games=n_recent, season=season
                )
                if recent_stats:
                    for stat in stat_cols:
                        if stat in recent_stats:
                            roster.loc[idx, f"{stat}_recent"] = recent_stats[stat]
                    roster.loc[idx, "games_recent"] = recent_stats.get("n_games", 0)

                season_stats = self.nba.get_player_season_stats(player_id, season)
                if season_stats:
                    for stat in stat_cols:
                        if stat in season_stats:
                            roster.loc[idx, f"{stat}_season"] = season_stats[stat]
                    roster.loc[idx, "games_season"] = season_stats.get("games_played", 0)

                if roster.loc[idx, "games_recent"] == 0 and roster.loc[idx, "games_season"] > 0:
                    for stat in stat_cols:
                        roster.loc[idx, f"{stat}_recent"] = roster.loc[idx, f"{stat}_season"]
                    roster.loc[idx, "games_recent"] = roster.loc[idx, "games_season"]

            except Exception as e:
                logger.debug(f"Could not get stats for player {player_id}: {e}")

        return roster

    def build_training_features(
        self,
        games_df: pd.DataFrame,
        player_box_df: pd.DataFrame,
        n_recent_games: int = 10,
    ) -> pd.DataFrame:
        """
        Build features for a batch of historical games (for training) using pre-loaded boxscores.
        """
        logger.info(f"Building training features for {len(games_df)} games...")

        all_features = []

        for idx, game in games_df.iterrows():
            try:
                game_date = pd.to_datetime(game["date"])

                features = self._build_matchup_from_boxscores(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    game_date=game_date,
                    player_box_df=player_box_df,
                    n_recent=n_recent_games,
                )

                features["date"] = game_date.strftime("%Y-%m-%d")
                features["game_id"] = game.get("game_id", idx)
                features["home_win"] = game.get(
                    "home_win",
                    int(game.get("home_score", 0) > game.get("away_score", 0)),
                )
                features["season"] = game.get("season")

                all_features.append(features)

            except Exception as e:
                logger.warning(f"Could not build features for game {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} games...")

        if not all_features:
            return pd.DataFrame()

        return pd.DataFrame(all_features)

    def _build_matchup_from_boxscores(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        player_box_df: pd.DataFrame,
        n_recent: int,
    ) -> dict:
        """Build matchup features from pre-loaded boxscore data."""
        home_features = self._build_team_from_boxscores(home_team, game_date, player_box_df, n_recent)
        away_features = self._build_team_from_boxscores(away_team, game_date, player_box_df, n_recent)
        return self.aggregator.create_matchup_features(home_features, away_features)

    def _build_team_from_boxscores(
        self,
        team: str,
        game_date: datetime,
        player_box_df: pd.DataFrame,
        n_recent: int,
    ) -> dict:
        """Build team features from pre-loaded boxscore data."""
        team_abbr = resolve_team_name(team) or team

        # Normalize dates
        if "game_date" in player_box_df.columns:
            df = player_box_df.copy()
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        else:
            df = player_box_df.copy()

        team_col = "team_abbr" if "team_abbr" in df.columns else "team_abbreviation"
        if team_col not in df.columns:
            # Fall back to older column name if needed
            team_col = "team_abbreviation" if "team_abbreviation" in df.columns else None

        if team_col is None:
            return self.aggregator._empty_features(team_abbr)

        team_games = df[(df[team_col] == team_abbr) & (df["game_date"] < game_date)].copy()

        if team_games.empty:
            return self.aggregator._empty_features(team_abbr)

        recent_dates = team_games["game_date"].drop_duplicates().nlargest(n_recent)
        recent_games = team_games[team_games["game_date"].isin(recent_dates)]

        player_stats = recent_games.groupby("player_id").agg(
            {
                "player_name": "first",
                "minutes": "mean",
                "pts": "mean",
                "reb": "mean",
                "ast": "mean",
                "fgm": "mean",
                "fga": "mean",
                "fg3m": "mean",
                "fg3a": "mean",
                "game_date": "count",
            }
        ).reset_index()

        player_stats = player_stats.rename(
            columns={
                "minutes": "expected_minutes",
                "pts": "pts_recent",
                "reb": "reb_recent",
                "ast": "ast_recent",
                "fgm": "fgm_recent",
                "fga": "fga_recent",
                "fg3m": "fg3m_recent",
                "fg3a": "fg3a_recent",
                "game_date": "games_recent",
            }
        )

        player_stats["minutes_recent"] = player_stats["expected_minutes"]
        player_stats["minutes_season"] = player_stats["expected_minutes"]
        player_stats["availability_prob"] = 1.0

        player_stats = player_stats[player_stats["expected_minutes"] >= 8]

        if player_stats.empty:
            return self.aggregator._empty_features(team_abbr)

        return self.aggregator.aggregate_team_features(player_stats, team_abbr)


def get_feature_builder() -> FeatureBuilder:
    """Get singleton feature builder instance."""
    global _feature_builder
    if "_feature_builder" not in globals():
        _feature_builder = FeatureBuilder()
    return _feature_builder
