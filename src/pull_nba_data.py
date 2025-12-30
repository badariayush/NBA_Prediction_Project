import time
import argparse
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def season_str(year: int) -> str:
    """Convert 2024 -> '2024-25' (NBA season string format)"""
    return f"{year}-{str(year + 1)[-2:]}"


def fetch_season_games(year: int, season_type: str) -> pd.DataFrame:
    """
    Returns raw leaguegamefinder dataframe (one row per TEAM per game).
    season_type: 'Regular Season' or 'Playoffs'
    """
    season = season_str(year)
    gf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    df = gf.get_data_frames()[0]
    return df


def to_game_level(df_team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level rows (2 rows per game) into game-level rows:
    date, season, season_type, home_team, away_team, home_score, away_score
    """
    # Keep only columns we need
    keep = ["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ABBREVIATION", "MATCHUP", "PTS"]
    df = df_team_rows[keep].copy()

    # Parse season year from SEASON_ID like 22024 (season 2024-25)
    # We'll store season as 2024 (start year)
    df["season"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)

    # HOME/AWAY from MATCHUP string:
    # "LAL vs. BOS" -> LAL is home
    # "LAL @ BOS"   -> LAL is away
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    # Build home side
    home = df[df["is_home"]].rename(
        columns={
            "TEAM_ABBREVIATION": "home_team",
            "PTS": "home_score",
        }
    )[["GAME_ID", "GAME_DATE", "season", "home_team", "home_score"]]

    # Build away side
    away = df[~df["is_home"]].rename(
        columns={
            "TEAM_ABBREVIATION": "away_team",
            "PTS": "away_score",
        }
    )[["GAME_ID", "away_team", "away_score"]]

    games = home.merge(away, on="GAME_ID", how="inner")

    # Clean date
    games = games.rename(columns={"GAME_DATE": "date"})
    games["date"] = pd.to_datetime(games["date"]).dt.date.astype(str)

    # Compute label
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Order columns
    games = games[
        ["date", "season", "home_team", "away_team", "home_score", "away_score", "home_win", "GAME_ID"]
    ].sort_values(["season", "date", "GAME_ID"])

    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2010, help="NBA season start year (e.g., 2010 -> 2010-11)")
    ap.add_argument("--end_year", type=int, default=2024, help="Inclusive end year (e.g., 2024 -> 2024-25)")
    ap.add_argument("--include_playoffs", action="store_true", help="Also fetch playoff games")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests (rate limit friendly)")
    ap.add_argument("--out", type=str, default="data/raw/games.csv")
    args = ap.parse_args()

    all_games = []

    for year in range(args.start_year, args.end_year + 1):
        print(f"Fetching {season_str(year)} Regular Season...")
        df_reg = fetch_season_games(year, "Regular Season")
        games_reg = to_game_level(df_reg)
        games_reg["season_type"] = "regular"
        all_games.append(games_reg)

        time.sleep(args.sleep)

        if args.include_playoffs:
            print(f"Fetching {season_str(year)} Playoffs...")
            df_po = fetch_season_games(year, "Playoffs")
            games_po = to_game_level(df_po)
            games_po["season_type"] = "playoffs"
            all_games.append(games_po)

            time.sleep(args.sleep)

    out_df = pd.concat(all_games, ignore_index=True)

    # Drop duplicates just in case (sometimes endpoints can return repeats)
    out_df = out_df.drop_duplicates(subset=["GAME_ID"])

    # Save
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} games to {args.out}")
    print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
