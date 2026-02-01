# NBA Game Prediction Project

Advanced NBA game winner prediction using player-level stats, injury data, and Monte Carlo uncertainty estimation.

## Features

- **Player-Based Features**: Aggregates individual player rolling stats to team level
- **Multiple Models**: Gradient Boosting (primary) + Logistic Regression baseline
- **Injury Integration**: Scrapes Basketball-Reference for current injury reports
- **Monte Carlo Uncertainty**: 1000+ simulations with player availability sampling
- **Free Data Sources Only**: Uses nba_api and Basketball-Reference (no paid APIs)

## Quick Start

### Installation

```bash
# Clone the repo
cd NBA_Prediction_Project

# Install dependencies
pip install -r requirements.txt
```

### Pull Data

```bash
# Pull game and player data for recent seasons
python -m src.pull_nba_data --seasons 2023-24 2024-25 2025-26
```

### Build Features & Train Model

```bash
# Build training features
python -m src.features

# Train the model
python -m src.train
```

### Make Predictions

```bash
# Basic prediction
python predict.py --home "Boston Celtics" --away "New York Knicks" --date "2026-01-15"

# With rotation details
python predict.py --home BOS --away NYK --date 2026-01-15 --show-rotation

# More simulations for tighter confidence intervals
python predict.py --home LAL --away GSW --date 2026-01-20 --n-sims 2000
```

## Data Sources (All Free)

| Source | Used For | Rate Limits |
|--------|----------|-------------|
| **nba_api** | Player stats, rosters, game logs | ~1 req/sec |
| **BallDontLie** | Backup for nba_api failures | 60 req/min (free tier) |
| **Basketball-Reference** | Injury reports | Scraping (be respectful) |

## Project Structure

```
NBA_Prediction_Project/
├── predict.py              # Main CLI
├── requirements.txt
├── README.md
│
├── src/
│   ├── providers/          # Data providers
│   │   ├── nba_api_provider.py
│   │   ├── balldontlie_provider.py
│   │   ├── injury_provider.py
│   │   └── roster_resolver.py
│   │
│   ├── features/           # Feature engineering
│   │   ├── aggregator.py   # Player -> team aggregation
│   │   └── builder.py      # End-to-end feature building
│   │
│   ├── models/             # ML models
│   │   ├── trainer.py      # Model training
│   │   └── predictor.py    # Prediction + Monte Carlo
│   │
│   └── utils/              # Utilities
│       ├── cache.py        # Disk caching
│       ├── names.py        # Name normalization
│       └── team_map.py     # Team ID mapping
│
├── data/
│   ├── raw/                # Raw API data
│   ├── processed/          # Training features
│   ├── cache/              # API response cache
│   └── schedules/          # Game schedules
│
├── models/                 # Trained models
│   ├── gbm_model.pkl
│   └── feature_cols_gbm.json
│
├── results/                # Evaluation results
│
└── tests/                  # Unit tests
```

## Feature Engineering

### Player Rolling Stats

For each player, we compute rolling averages over their last N games:
- **PTS, REB, AST**: Basic counting stats
- **FGM/FGA, FG3M/FG3A**: Shooting makes/attempts
- **Minutes**: Play time

### Team Aggregation

Player stats are aggregated to team level using **minutes-weighted averaging**:

```
team_pts_per48 = Σ(player_pts/player_min × player_weight) × 48
```

Where weights = expected_minutes / total_team_minutes

### Matchup Features

Final features are differentials: `home_stat - away_stat`

## Injury Handling

| Status | Availability | Minutes Factor |
|--------|--------------|----------------|
| OUT | 0% | 0.0 |
| DOUBTFUL | 10% | 0.1 |
| QUESTIONABLE | 50% (sampled) | 0.5 |
| PROBABLE | 90% | 0.9 |
| ACTIVE | 100% | 1.0 |

## Monte Carlo Uncertainty

For each prediction, we run 1000 simulations:

1. **Sample player availability** based on injury status
2. **Add minutes noise**: `N(expected_min, 0.15 × expected_min)`
3. **Recompute features** with sampled roster
4. **Get prediction** from GBM model

Output: Mean probability + 5th/95th percentile interval

## Example Output

```
══════════════════════════════════════════════════════════════════════
  🏀 NBA GAME PREDICTION (GBM + Monte Carlo)
══════════════════════════════════════════════════════════════════════

  Matchup:  NYK @ BOS
  Date:     2026-01-15

──────────────────────────────────────────────────────────────────────
  📊 WIN PROBABILITIES
──────────────────────────────────────────────────────────────────────

  Team                        Win Prob             95% CI
  ------------------------------------------------------------
  BOS                           62.3%    [58.1% - 66.8%]
  NYK                           37.7%

──────────────────────────────────────────────────────────────────────
  🎯 PREDICTION
──────────────────────────────────────────────────────────────────────

  Predicted winner: BOS (home)
  Confidence: Moderate confidence (25%)
  Based on 1,000 Monte Carlo simulations

══════════════════════════════════════════════════════════════════════
```

## Limitations

1. **No live lineup data**: We infer rotations from recent minutes (no DraftKings/paid sources)
2. **Injury uncertainty**: Questionable/probable players are sampled probabilistically
3. **No rest/travel adjustments**: Schedule factors not included (free data limitation)
4. **API rate limits**: Heavy usage may trigger temporary blocks

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License - See LICENSE file

