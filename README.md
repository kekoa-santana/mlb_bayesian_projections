# The Data Diamond -- Hierarchical Bayesian MLB Projection System

A three-layer Bayesian projection system for MLB player performance. Built to produce **calibrated probability distributions**, not just point estimates -- every projection comes with a full range of outcomes and a confidence level.

This system is designed to answer the questions that matter: *How good is this player really? How confident are we? And what happens when this pitcher faces this lineup?*

Built by [Kekoa Santana](https://www.linkedin.com/in/kekoa-santana) | [@TheDataDiamond](https://x.com/TheDataDiamond)

---

## Project Scope

**This repository contains the projection engine only.** The interactive Streamlit dashboard has been separated into its own repository: [`tdd-dashboard`](https://github.com/TheDataDiamond/tdd-dashboard).

This repo focuses on:
- Hierarchical Bayesian model training and inference
- Walk-forward backtesting and validation
- Feature engineering and data pipeline
- Game-level prop simulation framework
- Pre-computation of dashboard data

---

## What Makes This Different

Most public projection systems (ZiPS, Steamer, Marcel) give you a single number -- "this pitcher will have a 24.3% K rate." That's a best guess, but it hides everything interesting: How sure are we? Could he break out? Is he a regression risk?

This system is built on **hierarchical Bayesian modeling**, which means:

- **Full posterior distributions, not point estimates.** Every projection is a probability distribution. You get the most likely outcome *and* the range of realistic possibilities.
- **Partial pooling instead of sample size cutoffs.** Most systems throw out players with fewer than 200 PA. This model borrows strength across similar players, so a rookie with 80 PA still gets a meaningful (if uncertain) projection -- the model knows how much to trust small samples.
- **Statcast data informs the priors.** Whiff rate, barrel rate, chase rate, and exit velocity feed into the *prior belief* about true talent, not just regression features. This is how you identify breakouts before they show up in traditional stats.

### Understanding the Uncertainty Ranges

When the model projects a hitter's BB% at 9.2% with a 95% credible interval of [6.8%, 12.1%], it means:

> *"We believe there's a 95% chance his true-talent walk rate next season falls between 6.8% and 12.1%, with 9.2% being most likely."*

These ranges might look wide -- but they're honest. The model's interval captures both *how much talent actually changes* and *how well we can estimate current talent*. Public systems have the same underlying uncertainty -- they just don't show it to you.

**The width of the interval is itself informative.** A tight range means the model is confident (lots of data, stable track record). A wide range means genuine uncertainty (young player, volatile history, small sample). That's the whole point.

---

## Architecture

The system has three layers, each building on the last:

### Layer 1: Season-Level Talent Projections

Hierarchical Bayesian models (PyMC) estimate true-talent rates for every player with proper uncertainty quantification.

| | Hitters | Pitchers |
|---|---|---|
| **Stats** | K%, BB%, HR/PA, wOBA | K%, BB%, HR/BF |
| **Structure** | Partial pooling across players, AR(1) process for year-to-year evolution, age-bucket priors, Statcast skill covariates | Same + starter/reliever role adjustment |
| **Output** | Full posterior per player per stat | Full posterior per player per stat |

### Layer 2: Pitch-Type Matchup Model

Quantifies how a pitcher's specific arsenal maps onto a hitter's specific vulnerabilities.

- **Hitter vulnerability profiles:** Whiff rate, chase rate, and CSW% broken down by pitch type
- **Pitcher arsenal profiles:** Usage, pitch-level whiff rate, and barrel rate against per pitch type
- **Matchup scoring:** Log-odds additive method combining pitch-type interactions, with reliability-weighted fallback chains for small samples
- Supports both raw pitch types and KMeans-clustered pitch archetypes (k=8 on velocity, movement, spin, extension)

### Layer 3: Game-Level Prop Predictions

Combines Layer 1 posteriors with Layer 2 matchup adjustments and a workload model to produce full distributions for game-level props.

**Supported Props:** K, BB, HR, H, Outs (both pitchers and batters)
- Monte Carlo simulation (4K draws) from the joint posterior
- Outputs P(over X.5) for any prop line
- Confidence tier system (HIGH/MEDIUM/LOW) with automated explanations
- Advanced metrics: CRPS, ECE, temperature scaling beyond Brier score

---

## Game Prop Framework

Complete validation framework for all game-level props:

### Prop Types
- **Pitcher Props:** K, BB, HR, H, Outs
- **Batter Props:** K, BB, HR, H
- **Lines:** Any half-point line (5.5 K, 7.5 K, etc.)

### Confidence System
- **HIGH:** Strong historical performance, well-calibrated
- **MEDIUM:** Solid performance with room for improvement  
- **LOW:** Underperforming, needs investigation

### Advanced Validation
- **CRPS:** Full-distribution forecast quality
- **ECE:** Calibration error across probability bins
- **Temperature Scaling:** Posterior calibration optimization
- **Ensemble Methods:** Optimal Bayes-Marcel blending

---

## Dashboard Integration

This repo pre-computes all data needed by the dashboard:

```bash
# Pre-compute projections (fits all Bayesian models, ~20 min)
python scripts/precompute_dashboard_data.py

# Quick iteration (fewer MCMC draws)
python scripts/precompute_dashboard_data.py --quick
```

**Output Location:** `tdd-dashboard/data/dashboard/*.parquet`

The dashboard reads these pre-computed files - no runtime dependency on this repo.

---

## Validation Results

Every model is evaluated with strict walk-forward backtesting -- train on data through season N, predict season N+1, roll forward. No future data leakage. See [`docs/reliability_matrix.md`](docs/reliability_matrix.md) for the full stat-by-stat breakdown.

### Season-Level Projections
- **All core rate stats** (hitter K%/BB%/GB%, pitcher K%/BB%): Bayes beats Marcel on Brier score across every fold
- **95% credible interval coverage:** 55-73% at 80% CI across stats (slight undercoverage expected with forward projection)
- **All folds converge cleanly** (r_hat < 1.05, zero divergences)
- **Known gaps:** Pitcher counting stats (K, BB, Outs) lose to Marcel on MAPE. SB era adjustment too blunt.

### Game-Level Props (11,517 pitcher games, 131K+ batter games)

**Well-calibrated (Tier 1):**
- Batter H: ECE=0.018, Brier=0.148 (best-calibrated prop in the system)
- Batter K: ECE=0.035, Brier=0.147
- Batter BB: ECE=0.031, Brier=0.120
- Pitcher K: ECE=0.020, Brier=0.188

**Partially calibrated (Tier 2):** Pitcher BB (ECE=0.072), Pitcher H (slope < 0.30 -- overconfident)

**Unreliable (Tier 3):** HR (both sides), Pitcher Outs

### Evaluation Metrics
- **Brier score:** Binary outcome calibration
- **ECE:** Calibration error across probability bins
- **CRPS:** Full-distribution forecast quality
- **Calibration slope:** 1.0 = perfect; < 0.5 = overconfident; > 1.5 = underconfident

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
| **Minor league data** | 2.1M | AAA-AA-A+ seasons with translation factors |

---

## Tech Stack

- **Bayesian Modeling:** PyMC 5.x, ArviZ
- **Data:** PostgreSQL, SQLAlchemy, pandas
- **ML:** scikit-learn (calibration, metrics), KMeans (pitch archetypes)
- **Visualization:** matplotlib with custom brand theme ([tdd_theme](https://github.com/TheDataDiamond))
- **Advanced Metrics:** scipy (CRPS, optimization)
- **Python 3.11+**

---

## Status

### Complete
- Hierarchical Bayesian models for hitter (K%, BB%, GB%, FB%, HR/FB) and pitcher (K%, BB%, HR/BF)
- AR(1) talent evolution with age-bucket x Statcast skill tier priors
- PA-by-PA game simulator (pitcher + batter) with matchup, TTO, umpire, park, weather adjustments
- MiLB translation factors, prospect rankings, MLB readiness model
- TDD 20-80 scouting grades with Bayesian grade confidence intervals
- In-season conjugate updating (daily) + rankings refresh (weekly)
- Team ELO, profiles, power rankings, full-league season simulation
- Market edge / Kelly sizing framework

### Active
- Fixing pitcher H calibration (slope << 1.0)
- Layer 1 K-rate calibration (2pp under-prediction)
- Game-level BB bias correction
- Legacy module retirement (game_k_model.py, k_rate_model.py)

---

## Project Structure

```
player_profiles/                          # Projection engine only
├── config/model.yaml                    # Sampling & model hyperparameters
├── src/
│   ├── data/
│   │   ├── db.py                         # Database connection
│   │   ├── queries.py                    # SQL query functions
│   │   ├── feature_eng.py                # Pitch-type profiles & caching
│   │   ├── milb_translation.py           # MiLB-to-MLB translation factors
│   │   ├── league_baselines.py          # Per-pitch-type population baselines
│   │   ├── pitch_archetypes.py           # KMeans pitch shape clustering
│   │   ├── schedule.py                   # MLB API integration
│   │   └── data_qa.py                    # Data quality reports
│   ├── models/
│   │   ├── hitter_model.py               # Generalized hitter Bayesian model
│   │   ├── pitcher_model.py              # Generalized pitcher Bayesian model
│   │   ├── k_rate_model.py               # Original hitter K% model
│   │   ├── pitcher_k_rate_model.py       # Original pitcher K% model
│   │   ├── hitter_projections.py         # Composite hitter projections
│   │   ├── pitcher_projections.py        # Composite pitcher projections
│   │   ├── matchup.py                    # Pitch-type matchup scoring (Layer 2)
│   │   ├── bf_model.py                   # Batters-faced workload model
│   │   ├── game_k_model.py               # Game-level K posterior (legacy Layer 3 engine)
│   │   ├── counting_projections.py       # Counting stat projections
│   │   ├── pa_model.py                   # PA/BF shrinkage models
│   │   └── in_season_updater.py          # Conjugate updating
│   ├── evaluation/
│   │   ├── backtesting.py                # Original backtesting framework
│   │   ├── hitter_backtest.py            # Walk-forward hitter backtest
│   │   ├── pitcher_backtest.py           # Walk-forward pitcher backtest
│   │   ├── matchup_validation.py         # Matchup lift validation
│   │   ├── game_k_validation.py          # Game-level K backtest (legacy engine)
│   │   ├── confidence_tiers.py           # Prop confidence scoring
│   │   ├── ensemble.py                   # Bayes-Marcel blending
│   │   ├── metrics.py                    # CRPS, ECE, calibration
│   │   ├── counting_backtest.py          # Counting stat validation
│   │   └── game_prop_validation.py       # Complete prop framework
│   ├── utils/
│   │   └── constants.py                  # Pitch maps, league averages, etc.
│   └── viz/
│       ├── theme.py                      # Brand theme (wraps tdd_theme)
│       ├── projections.py                # Projection content cards
│       └── composite_cards.py            # Breakout/regression cards
├── scripts/                              # CLI runners
│   ├── precompute_dashboard_data.py      # Generate all dashboard data
│   ├── update_in_season.py               # Daily conjugate updates
│   ├── run_*_backtest.py                 # Various backtest runners
│   ├── build_milb_translations.py        # Build MiLB factors
│   ├── run_game_prop_backtest.py         # Game prop validation
│   ├── generate_*_cards.py               # Content generation
│   └── apply_injury_adjustments.py       # Injury adjustments
├── tests/                                # 31 test files, 291 tests (278 fast + 13 integration)
├── data/cached/                          # Parquet cache (~60 files)
├── outputs/                              # Backtest CSVs & content PNGs
├── docs/                                 # Documentation
├── pyproject.toml                        # Dependencies & config
└── swe_plan.md                           # Enhancement roadmap
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database (`mlb_fantasy` on `localhost:5433`)
- PyMC, ArviZ, pandas, numpy (see `pyproject.toml`)

### Installation
```bash
# Clone this repo
git clone https://github.com/TheDataDiamond/player_profiles
cd player_profiles

# Install dependencies
pip install -e .

# Set up database connection (copy .env.example to .env)
cp .env.example .env
# Edit .env with your database credentials
```

### Run Projections
```bash
# Full pre-computation (20+ minutes)
python scripts/precompute_dashboard_data.py

# Quick development mode (5 minutes)
python scripts/precompute_dashboard_data.py --quick
```

### Run Tests
```bash
# Fast unit tests (~40s)
pytest -m "not integration"

# Full suite including PyMC sampling (~70s)
pytest
```

### Run Backtests
```bash
# Season projection backtests
python scripts/run_hitter_backtest.py
python scripts/run_pitcher_backtest.py

# Game prop backtests  
python scripts/run_game_prop_backtest.py --side pitcher --stat k bb
```

---

## License

This project is proprietary. All rights reserved.

---

*Built with PyMC, PostgreSQL, and an unhealthy amount of Statcast data.*
