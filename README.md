# The Data Diamond -- Hierarchical Bayesian MLB Projection System

A three-layer Bayesian projection system for MLB player performance. Built to produce **calibrated probability distributions**, not just point estimates -- every projection comes with a full range of outcomes and a confidence level.

This system is designed to answer the questions that matter: *How good is this player really? How confident are we? And what happens when this pitcher faces this lineup?*

Built by [Kekoa Santana](https://www.linkedin.com/in/kekoa-santana) | [@TheDataDiamond](https://x.com/TheDataDiamond)

---

## What Makes This Different

Most public projection systems (ZiPS, Steamer, Marcel) give you a single number -- "this pitcher will have a 24.3% K rate." That's a best guess, but it hides everything interesting: How sure are we? Could he break out? Is he a regression risk?

This system is built on **hierarchical Bayesian modeling**, which means:

- **Full posterior distributions, not point estimates.** Every projection is a probability distribution. You get the most likely outcome *and* the range of realistic possibilities.
- **Partial pooling instead of sample size cutoffs.** Most systems throw out players with fewer than 200 PA. This model borrows strength across similar players, so a rookie with 80 PA still gets a meaningful (if uncertain) projection -- the model knows how much to trust small samples.
- **Statcast data informs the priors.** A pitcher's raw K% is just the starting point. Whiff rate, barrel rate, and exit velocity feed into the *prior belief* about true talent, not just a regression feature. This is how you identify breakouts before they show up in traditional stats.

### Understanding the Uncertainty Ranges

When the model says a pitcher's K% is projected at 24.8% with a 95% credible interval of [18.2%, 31.8%], it means:

> *"We believe there's a 95% chance his true-talent K rate next season falls between 18.2% and 31.8%, with 24.8% being most likely."*

These ranges might look wide -- but they're honest. Real-world pitcher K% changes by about 2pp per year on average, and 95% of pitchers shift by less than 4.5pp. The model's interval captures both *how much talent actually changes* and *how well we can estimate current talent*. Public systems have the same underlying uncertainty -- they just don't show it to you.

**The width of the interval is itself informative.** A tight range means the model is confident (lots of data, stable track record). A wide range means genuine uncertainty (young player, volatile history, small sample). That's the whole point.

---

## Architecture

The system has three layers, each building on the last:

### Layer 1: Season-Level Talent Projections

Hierarchical Bayesian models (PyMC) estimate true-talent rates for every player with proper uncertainty quantification.

| | Hitters | Pitchers |
|---|---|---|
| **Stats** | K%, BB%, HR/PA, xwOBA | K%, BB%, HR/BF |
| **Structure** | Partial pooling across players, Gaussian random walk for year-to-year evolution, age-bucket priors, Statcast skill covariates | Same + starter/reliever role adjustment |
| **Output** | Full posterior per player per stat | Full posterior per player per stat |

### Layer 2: Pitch-Type Matchup Model

Quantifies how a pitcher's specific arsenal maps onto a hitter's specific vulnerabilities.

- **Hitter vulnerability profiles:** Whiff rate, chase rate, and CSW% broken down by pitch type
- **Pitcher arsenal profiles:** Usage, pitch-level whiff rate, and barrel rate against per pitch type
- **Matchup scoring:** Log-odds additive method combining pitch-type interactions, with reliability-weighted fallback chains for small samples
- Supports both raw pitch types and KMeans-clustered pitch archetypes (k=8 on velocity, movement, spin, extension)

### Layer 3: Game-Level K Predictions

Combines Layer 1 posteriors with Layer 2 matchup adjustments and a workload model to produce a full distribution over a pitcher's strikeout total for a specific game.

- Monte Carlo simulation (10K draws) from the joint posterior
- Outputs P(over X.5) for any K line
- Walk-forward backtested: calibration at 50/80/90% confidence = 48/79/89%

---

## Dashboard

An interactive Streamlit dashboard for exploring projections, player profiles, and game-level simulations.

**Projections** -- Sortable, filterable tables for all projected hitters and pitchers with composite breakout/regression scoring.

**Player Profiles** -- Deep dives with plain English scouting reports that translate the Bayesian posteriors into accessible insights, percentile rankings (Baseball Savant style), and posterior distribution charts.

**Game K Simulator** -- Select any pitcher, adjust expected workload, and get a full K distribution with probability breakdowns.

### Running the Dashboard

```bash
# 1. Pre-compute projections (fits all Bayesian models, ~20 min)
python scripts/precompute_dashboard_data.py

# 2. Launch
streamlit run app.py
```

Use `--quick` for faster iteration (fewer MCMC draws):
```bash
python scripts/precompute_dashboard_data.py --quick
```

---

## Backtest Results

Every model is evaluated with strict walk-forward backtesting -- train on data through season N, predict season N+1, roll forward. No future data leakage.

**Season Projections vs. Marcel (industry baseline):**
- Hitter K%/BB%: Beats Marcel on MAE in 2 of 3 folds
- All stats (hitter + pitcher): Beats Marcel on Brier score across every fold -- better calibrated probability estimates
- 95% credible interval coverage: 84-94% across all stats (target: 95%, slight undercover is expected with forward projection variance)
- All folds converge cleanly (r_hat < 1.05, zero divergences)

**Game K Predictions (11,517 games, 3 walk-forward folds):**
- RMSE: 2.28 strikeouts
- Brier score: 0.187
- Calibration: 50/80/90% confidence levels hit 48/79/89% (near-perfect)
- At 70%+ model confidence on the 5.5 K line: **76.4% win rate** (3,340 of 4,371 games)

---

## Data Foundation

Built on a PostgreSQL database with comprehensive MLB data (2018-2025):

| Source | Records | Description |
|---|---|---|
| Statcast pitches | 5.4M | Every pitch: velocity, movement, spin, location, outcome |
| Plate appearances | 1.4M | PA-level results with full context |
| Batted balls | 483K | Exit velo, launch angle, xwOBA, barrel flags |
| Game lineups | 458K | Full batting orders with positions |
| Umpire assignments | 19.5K | Home plate umpire for every game |
| Weather conditions | 19.5K | Temperature, wind, dome flag |
| Park factors | 494 | HR park factors by batter hand (season + 3yr smoothed) |
| Sprint speed | 3K | Baserunning athleticism metrics |
| Fantasy scoring | 631K | DraftKings + ESPN game-level scoring |

---

## Tech Stack

- **Bayesian Modeling:** PyMC 5.x, ArviZ
- **Data:** PostgreSQL, SQLAlchemy, pandas
- **ML:** scikit-learn (calibration, metrics), KMeans (pitch archetypes)
- **Visualization:** matplotlib with custom brand theme ([tdd_theme](https://github.com/TheDataDiamond))
- **Dashboard:** Streamlit
- **Python 3.11+**

---

## Roadmap

### v1.0 -- Season Projections & Dashboard (Current)
- Multi-stat Bayesian projections for hitters (K%, BB%, HR/PA, xwOBA) and pitchers (K%, BB%, HR/BF)
- Pitch-type matchup model with archetype clustering
- Game-level K posterior simulator
- Interactive dashboard with scouting reports and percentile rankings

### v1.1 -- Statcast Skill Tier Priors
- Cluster players into skill tiers (elite, above-average, average, below-average) using Statcast indicators (exit velo, hard-hit rate, whiff rate)
- Replace single population prior with tier-specific priors in the hierarchical model -- young elite hitters regress toward the elite mean, not the league mean
- Biggest impact for players with 1-2 seasons of data where shrinkage is heaviest
- Show skill tier in dashboard player profiles

### v1.2 -- Matchup Explorer
- Surface the Layer 2 matchup model in the dashboard
- Pitcher vs. hitter pitch-type breakdown visualizations
- Arsenal and vulnerability profile charts

### v1.3 -- Season Counting Stat Projections (was v1.2)
- Combine K% posteriors with season-level workload projections to produce full-season K totals (e.g., "projected 158-204 Ks with 95% confidence")
- Extend to wins, IP, and other counting stats
- More tangible output format for non-technical audiences

### v1.4 -- Game Context Integration
- **Park factors** -- HR park adjustments by batter handedness into HR and xwOBA projections
- **Umpire tendencies** -- Home plate umpire K-rate adjustments into the Game K model
- **Weather** -- Temperature and wind effects on batted ball outcomes

### v1.5 -- Lineup-Aware Game Simulator
- Real lineup data feeds the Game K simulator instead of league-average assumptions
- Per-batter matchup adjustments using Layer 2 profiles
- Batting order position weighting for plate appearance probability

### v1.6 -- In-Season Updating
- Bayesian posterior updating as 2026 games are played
- Separate rolling form from true-talent shifts
- Spring training signal integration

### v1.7 -- Probability Calibration & Decision Support
- Systematic tracking of model calibration across all prediction types
- Optimal decision-making framework under uncertainty (Kelly criterion)
- Historical accuracy reporting and model comparison

### v1.8 -- Fantasy Scoring Layer
- Translate player projections into DraftKings and ESPN fantasy point distributions
- Uncertainty-aware fantasy valuations

### v1.9 -- Content Pipeline
- Auto-generated daily matchup cards using brand theme
- Biggest posterior shift alerts (breakout/regression movers)
- Streamlined social content workflow

---

## Project Structure

```
player_profiles/
├── app.py                           # Streamlit dashboard
├── config/model.yaml                # Sampling & model hyperparameters
├── src/
│   ├── data/
│   │   ├── db.py                    # Database connection
│   │   ├── queries.py               # SQL query functions
│   │   ├── feature_eng.py           # Pitch-type profiles & caching
│   │   ├── league_baselines.py      # Per-pitch-type population baselines
│   │   ├── pitch_archetypes.py      # KMeans pitch shape clustering
│   │   └── data_qa.py              # Data quality reports
│   ├── models/
│   │   ├── hitter_model.py          # Generalized hitter Bayesian model
│   │   ├── pitcher_model.py         # Generalized pitcher Bayesian model
│   │   ├── hitter_projections.py    # Composite hitter projections
│   │   ├── pitcher_projections.py   # Composite pitcher projections
│   │   ├── matchup.py               # Pitch-type matchup scoring (Layer 2)
│   │   ├── bf_model.py              # Batters-faced workload model
│   │   └── game_k_model.py          # Game-level K posterior (Layer 3)
│   ├── evaluation/
│   │   ├── hitter_backtest.py       # Walk-forward hitter backtest
│   │   ├── pitcher_backtest.py      # Walk-forward pitcher backtest
│   │   ├── matchup_validation.py    # Matchup lift validation
│   │   └── game_k_validation.py     # Game K calibration backtest
│   └── viz/
│       ├── theme.py                 # Brand theme (wraps tdd_theme)
│       ├── projections.py           # Projection content cards
│       └── composite_cards.py       # Breakout/regression cards
├── scripts/                         # CLI runners for backtests & content
├── tests/                           # 17 test files, 119 test functions
├── data/cached/                     # Parquet cache (~60 files)
├── data/dashboard/                  # Pre-computed dashboard data
└── outputs/                         # Backtest CSVs & content PNGs
```

---

## License

This project is proprietary. All rights reserved.

---

*Built with PyMC, PostgreSQL, and an unhealthy amount of Statcast data.*
