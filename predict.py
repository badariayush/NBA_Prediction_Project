#!/usr/bin/env python
"""
NBA Game Prediction CLI.

Predicts game outcomes with uncertainty intervals using:
- Gradient Boosting classifier (primary)
- Monte Carlo simulation for uncertainty
- Injury and rotation inference

Usage:
    python predict.py --home "Boston Celtics" --away "New York Knicks" --date "2026-01-15"
    python -m predict --home BOS --away NYK --date 2026-01-15
    
Author: NBA Prediction Project
"""

import argparse
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ANSI colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header():
    """Print CLI header."""
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏀 NBA GAME PREDICTION (GBM + Monte Carlo){Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()


def print_prediction(result: dict, home_rotation: dict = None, away_rotation: dict = None):
    """Print prediction results."""
    home = result["home_team"]
    away = result["away_team"]
    date = result["game_date"]
    
    # Matchup header
    print(f"  {Colors.BOLD}Matchup:{Colors.RESET}  {away} @ {home}")
    print(f"  {Colors.BOLD}Date:{Colors.RESET}     {date}")
    print()
    
    # Rotation info
    if home_rotation or away_rotation:
        print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}  📋 INFERRED ROTATIONS{Colors.RESET}")
        print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
        print()
        
        for team, rotation in [(home, home_rotation), (away, away_rotation)]:
            if rotation:
                print(f"  {Colors.BOLD}{team}{Colors.RESET}")
                
                # Starters
                starters = rotation.get("starters", [])
                if starters:
                    print(f"    Starters:")
                    for p in starters[:5]:
                        mins = p.get("expected_minutes", 0)
                        status = p.get("injury_status", "")
                        status_str = f" {Colors.YELLOW}({status}){Colors.RESET}" if status != "ACTIVE" else ""
                        print(f"      • {p['player_name']:<25} {mins:>5.1f} min{status_str}")
                
                # Key bench
                bench = rotation.get("bench", [])
                if bench:
                    print(f"    Key Bench:")
                    for p in bench[:3]:
                        mins = p.get("expected_minutes", 0)
                        print(f"      • {p['player_name']:<25} {mins:>5.1f} min")
                
                # Injuries
                injuries = rotation.get("injuries", [])
                if injuries:
                    print(f"    {Colors.RED}Injuries:{Colors.RESET}")
                    for inj in injuries:
                        print(f"      • {inj['player_name']}: {inj['injury_status']} - {inj.get('injury_note', '')}")
                
                print()
    
    # Probabilities
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  📊 WIN PROBABILITIES{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    p_home = result["p_home_win"]
    p_away = result["p_away_win"]
    p_5th = result.get("p_home_5th", p_home)
    p_95th = result.get("p_home_95th", p_home)
    
    # Main prediction
    home_color = Colors.GREEN if p_home > 0.5 else Colors.DIM
    away_color = Colors.GREEN if p_away > 0.5 else Colors.DIM
    
    print(f"  {'Team':<25}{'Win Prob':>15}{'95% CI':>20}")
    print(f"  {'-' * 60}")
    print(f"  {Colors.BOLD}{home:<25}{Colors.RESET}{home_color}{p_home:>14.1%}{Colors.RESET}    [{p_5th:.1%} - {p_95th:.1%}]")
    print(f"  {Colors.BOLD}{away:<25}{Colors.RESET}{away_color}{p_away:>14.1%}{Colors.RESET}")
    print()
    
    # Baseline comparison
    if "p_home_gbm" in result and "p_home_lr" in result:
        print(f"  {Colors.DIM}Model Breakdown:{Colors.RESET}")
        print(f"    GBM:                 {result['p_home_gbm']:.1%}")
        print(f"    Logistic Regression: {result['p_home_lr']:.1%}")
        print()
    
    # Prediction result
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  🎯 PREDICTION{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print()
    
    winner = result["predicted_winner"]
    confidence = abs(p_home - 0.5) * 2
    
    if confidence >= 0.3:
        conf_label = "High confidence"
        conf_color = Colors.GREEN
    elif confidence >= 0.15:
        conf_label = "Moderate confidence"
        conf_color = Colors.CYAN
    else:
        conf_label = "Low confidence (toss-up)"
        conf_color = Colors.YELLOW
    
    role = "home" if winner == home else "away"
    print(f"  Predicted winner: {Colors.BOLD}{winner}{Colors.RESET} ({role})")
    print(f"  Confidence: {conf_color}{conf_label} ({confidence:.0%}){Colors.RESET}")
    
    # Uncertainty info
    n_sims = result.get("n_simulations", 0)
    if n_sims > 0:
        print(f"  {Colors.DIM}Based on {n_sims:,} Monte Carlo simulations{Colors.RESET}")
    
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Game Prediction with Uncertainty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --home "Boston Celtics" --away "New York Knicks" --date "2026-01-15"
  python predict.py --home BOS --away NYK --date 2026-01-15
  python predict.py --home LAL --away GSW --date 2026-01-20 --n-sims 2000
        """
    )
    
    parser.add_argument(
        "--home",
        type=str,
        required=True,
        help="Home team (name or abbreviation)"
    )
    parser.add_argument(
        "--away",
        type=str,
        required=True,
        help="Away team (name or abbreviation)"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)"
    )
    parser.add_argument(
        "--no-mc",
        action="store_true",
        help="Skip Monte Carlo simulation (faster, no uncertainty)"
    )
    parser.add_argument(
        "--show-rotation",
        action="store_true",
        help="Show inferred rotation details"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Parse date
    try:
        game_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    if not args.quiet:
        print_header()
    
    # Import modules (late import to show header quickly)
    try:
        from src.utils.team_map import resolve_team_name
        from src.models.predictor import GamePredictor, MonteCarloPredictor
        from src.features.builder import FeatureBuilder
        from src.providers.roster_resolver import RosterResolver
    except ImportError as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Could not import modules: {e}")
        print(f"  Make sure all dependencies are installed.")
        sys.exit(1)
    
    # Resolve teams
    home_team = resolve_team_name(args.home)
    away_team = resolve_team_name(args.away)
    
    if not home_team:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Unknown home team: {args.home}")
        sys.exit(1)
    if not away_team:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Unknown away team: {args.away}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"  {Colors.DIM}Processing {away_team} @ {home_team} on {game_date.date()}...{Colors.RESET}")
        print()
    
    # Initialize components
    try:
        feature_builder = FeatureBuilder()
        predictor = GamePredictor()
        roster_resolver = RosterResolver()
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Warning:{Colors.RESET} Could not initialize all components: {e}")
        print(f"  Some features may be limited.")
        feature_builder = None
        predictor = None
        roster_resolver = None
    
    # Get rotation info
    home_rotation = None
    away_rotation = None
    
    if args.show_rotation and roster_resolver:
        try:
            home_rotation = roster_resolver.get_rotation_summary(home_team, game_date)
            away_rotation = roster_resolver.get_rotation_summary(away_team, game_date)
        except Exception as e:
            logger.warning(f"Could not get rotation info: {e}")
    
    # Run prediction
    try:
        if args.no_mc or predictor is None:
            # Simple prediction without Monte Carlo
            if predictor and feature_builder:
                result = predictor.predict_game(home_team, away_team, game_date, feature_builder)
                result["p_home_5th"] = result["p_home_win"]
                result["p_home_95th"] = result["p_home_win"]
                result["n_simulations"] = 0
            else:
                # Fallback to basic prediction
                result = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "game_date": game_date.strftime("%Y-%m-%d"),
                    "p_home_win": 0.55,  # Slight home advantage
                    "p_away_win": 0.45,
                    "p_home_5th": 0.55,
                    "p_home_95th": 0.55,
                    "predicted_winner": home_team,
                    "n_simulations": 0,
                }
                print(f"{Colors.YELLOW}⚠ Warning:{Colors.RESET} Using fallback prediction (models not loaded)")
        else:
            # Full Monte Carlo prediction
            mc_predictor = MonteCarloPredictor(predictor, n_simulations=args.n_sims)
            result = mc_predictor.predict_with_uncertainty(
                home_team, away_team, game_date, roster_resolver, feature_builder
            )
            
            # Add baseline predictions
            try:
                base_result = predictor.predict_game(home_team, away_team, game_date, feature_builder)
                result["p_home_gbm"] = base_result.get("p_home_gbm", result["p_home_win"])
                result["p_home_lr"] = base_result.get("p_home_lr", result["p_home_win"])
            except:
                pass
        
        # Print results
        if not args.quiet:
            print_prediction(result, home_rotation, away_rotation)
        else:
            # Minimal output
            print(f"{result['predicted_winner']} ({result['p_home_win']:.1%} home win)")
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} {e}")
        print(f"  Run training first: python -m src.models.trainer")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR:{Colors.RESET} Prediction failed: {e}")
        logger.exception("Prediction error")
        sys.exit(1)


if __name__ == "__main__":
    main()

