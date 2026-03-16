# CLAUDE.md — MLB Bayesian Projection & Matchup System

## Project Overview
A hierarchical Bayesian projection system for MLB player performance, with a game-level pitcher-batter matchup model that powers both front-office-grade analytics and betting-actionable K prop predictions. Built by The Data Diamond (Koa).

**This is the projection engine.** Model training, backtesting, feature engineering, and precomputation live here. The Streamlit dashboard has been split into a separate repo (`tdd-dashboard`).

## Related Repos
- **Dashboard:** `C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/` — Streamlit app, daily updates, live game coverage
- **Theme package:** `tdd_theme` — shared brand colors/utilities (pip-installed)

## Tech Stack
- **Language:** Python 3.11+
- **Bayesian Modeling:** PyMC 5.x, ArviZ for diagnostics
- **Data:** PostgreSQL (existing database with Statcast pitch-level + boxscore data, 2018-2025 complete seasons)
- **Data Processing:** pandas, numpy, sqlalchemy
- **ML/Evaluation:** scikit-learn (calibration curves, metrics), XGBoost (optional Stuff+ layer)
- **Visualization:** matplotlib, seaborn
- **API:** MLB Stats API (pybaseball or custom queries for supplemental data)
- **Environment:** Docker (existing setup), conda or venv

## Database Context
Koa has an existing PostgreSQL database (`mlb_fantasy` on `localhost:5433`) containing:
- **Statcast pitch-level data** (2018-2025, 5.4M pitches): Every pitch with velocity, movement (pfx_x, pfx_z), release point, spin rate, spin axis, plate location (plate_x, plate_z), extension, pitch type, outcome (description field), batter/pitcher IDs
- **Batted ball data** (483K): Exit velo, launch angle, xwOBA, xBA, xSLG, spray, hard_hit, barrel flags
- **Boxscore data** (2018-2025): Game-level stats, plate appearances, strikeouts, walks, innings pitched, pitch counts
- **Star schema layout:** `production.fact_pitch`, `production.fact_pa`, `production.sat_batted_balls`, `production.sat_pitch_shape`, `production.dim_player`, `production.dim_game`, `production.dim_team`, plus `staging.*` and `raw.*` upstream tables
- Key column names: `release_speed`, `pfx_x`, `pfx_z`, `plate_x`, `plate_z`, `release_spin_rate`, `pitch_type`, `events`, `description`, `batter_id`, `pitcher_id`, `game_pk`, `game_counter`
- Pre-computed boolean flags on `fact_pitch`: `is_whiff`, `is_swing`, `is_called_strike`, `is_bip`, `is_foul`

**IMPORTANT:** Before writing any SQL or data loading code, inspect the actual database schema first:
```bash
psql -U <user> -d <dbname> -c "\dt"  # list tables
psql -U <user> -d <dbname> -c "\d <tablename>"  # describe columns
```
Do NOT assume table or column names — verify them.

## Project Structure
```
player_profiles/
├── .env                         # DB credentials (env vars, fallback for db.py)
├── config/
│   └── model.yaml               # Sampling, season config, game_k params
├── src/
│   ├── data/
│   │   ├── db.py                # SQLAlchemy engine + read_sql helper
│   │   ├── queries.py           # Query functions (season totals, profiles, game logs, trad stats, etc.)
│   │   ├── feature_eng.py       # Vulnerability/strength profiles + caching
│   │   ├── league_baselines.py  # Per (pitch_type/archetype, batter_stand) baselines
│   │   ├── pitch_archetypes.py  # KMeans k=8 pitch shape clustering
│   │   ├── milb_translation.py  # MiLB-to-MLB translation factors
│   │   ├── schedule.py          # MLB Stats API schedule/lineup fetcher
│   │   └── data_qa.py           # Data quality / sanity reports
│   ├── models/
│   │   ├── k_rate_model.py          # Hitter K% hierarchical Bayesian (PyMC)
│   │   ├── pitcher_k_rate_model.py  # Pitcher K% hierarchical Bayesian
│   │   ├── hitter_model.py          # Generalized hitter model (K%, BB%)
│   │   ├── pitcher_model.py         # Generalized pitcher model (K%, BB%)
│   │   ├── hitter_projections.py    # Composite hitter projections + breakout scoring
│   │   ├── pitcher_projections.py   # Composite pitcher projections + breakout scoring
│   │   ├── pa_model.py              # PA/BF shrinkage priors for playing time
│   │   ├── counting_projections.py  # Rate × playing time Monte Carlo counting stats
│   │   ├── matchup.py               # Pitch-type & archetype matchup scoring
│   │   ├── bf_model.py              # Batters-faced workload model
│   │   ├── game_k_model.py          # Game-level K posterior (Layer 3)
│   │   └── in_season_updater.py     # Beta-Binomial conjugate updating
│   ├── utils/
│   │   └── constants.py         # Pitch maps, whiff defs, zone boundaries, league avgs
│   ├── evaluation/
│   │   ├── backtesting.py           # Original K%-only hitter + pitcher backtest
│   │   ├── hitter_backtest.py       # Generalized hitter walk-forward backtest
│   │   ├── pitcher_backtest.py      # Generalized pitcher walk-forward backtest
│   │   ├── matchup_validation.py    # In-season matchup lift validation
│   │   ├── game_k_validation.py     # Full game-level K backtest
│   │   ├── confidence_tiers.py      # Prop confidence scoring system
│   │   ├── ensemble.py              # Bayes-Marcel ensemble blending
│   │   ├── metrics.py               # CRPS, ECE, temperature scaling
│   │   ├── counting_backtest.py     # Counting stat validation
│   │   └── game_prop_validation.py  # Complete game prop framework
│   └── viz/
│       ├── theme.py                 # The Data Diamond brand theme
│       ├── projections.py           # K% mover cards, individual pitcher cards
│       └── composite_cards.py       # Composite breakout/regression cards
├── scripts/
│   ├── precompute_dashboard_data.py   # Generates all dashboard parquets/npz
│   ├── update_in_season.py            # Daily conjugate update pipeline
│   ├── run_season_backtest.py         # K%-only backtest runner
│   ├── run_hitter_backtest.py         # Multi-stat hitter backtest runner
│   ├── run_pitcher_backtest.py        # Multi-stat pitcher backtest runner
│   ├── run_counting_backtest.py       # Counting stat backtest runner
│   ├── run_game_k_backtest.py         # Game-level K backtest runner
│   ├── run_game_prop_backtest.py      # Game prop validation runner
│   ├── generate_preseason_content.py  # 2026 K% mover cards
│   ├── generate_composite_cards.py    # Composite breakout/regression cards
│   ├── build_preseason_injuries.py    # Preseason injury data
│   ├── build_milb_translations.py     # Build MiLB translation factors
│   ├── apply_injury_adjustments.py    # Adjust counting projections for injuries
│   ├── backfill_dim_player.py         # One-time dim_player rebuild
│   ├── cross_year_corr.py             # Year-to-year correlation analysis
│   ├── full_eda.py                    # Exploratory data analysis
│   └── yty_correlation.py            # Year-to-year correlation utilities
├── data/
│   └── cached/                  # Parquet cache (~60 files, all seasons)
├── tests/                       # 17 test files, 119 test functions
├── notebooks/
├── outputs/                     # Backtest CSVs + content PNGs
├── docs/
│   ├── style_guide.md
│   ├── advanced_projection_features.md
│   └── failed_hypotheses.md
└── pyproject.toml
```

## Dashboard Relationship

This repo produces pre-computed data that the dashboard consumes:

```
player_profiles/                         tdd-dashboard/
  precompute_dashboard_data.py ──────►     data/dashboard/*.parquet
                                              app.py (reads parquets)
                                              update_in_season.py (conjugate updates)
```

### Workflow:
1. Run `python scripts/precompute_dashboard_data.py` — writes directly to `tdd-dashboard/data/dashboard/`
2. Dashboard reads parquets — no dependency on this repo at runtime

### Files shared with dashboard (`tdd-dashboard/lib/`):
These are copied (not linked) to the dashboard repo. Sync when function signatures change:
- `src/utils/constants.py` → `lib/constants.py`
- `src/viz/theme.py` → `lib/theme.py`
- `src/models/matchup.py` → `lib/matchup.py`
- `src/models/bf_model.py` → `lib/bf_model.py`
- `src/models/game_k_model.py` → `lib/game_k_model.py`
- `src/models/in_season_updater.py` → `lib/in_season_updater.py`
- `src/data/schedule.py` → `lib/schedule.py`
- `src/data/db.py` → `lib/db.py` (subset)

**To sync:** copy file, replace `from src.` with `from lib.` in imports.

## Architecture: Three-Layer System

### Layer 1: Season-Level Bayesian Projections (PyMC)
**Purpose:** Estimate true-talent rates for pitchers and hitters with proper uncertainty.

**Target stats (hitters):** K%, BB%, wOBA
**Target stats (pitchers):** K%, BB%
**Observed stats used as priors:** wOBA, HR/PA, barrel rate, hard-hit rate, sprint speed, whiff rate, chase rate, etc.

**Model structure:**
- Hierarchical partial pooling across players (shrink small samples toward population)
- AR(1) process for year-to-year talent evolution (mean-reverting, rho ~ Beta(8,2))
- Age-bucket population priors (3 buckets) × Statcast skill tier (4 tiers)
- Binomial/Beta observation model for rate stats
- LogNormal sigma_season with floor for forward projection uncertainty
- Output: Full posterior distributions per player per stat, not just point estimates

### Layer 2: Pitch-Type Matchup Model
**Purpose:** Quantify how a pitcher's arsenal maps onto a hitter's vulnerabilities.

**Method:** Log-odds additive scoring with reliability-weighted fallback chains:
1. Direct pitch-type match (reliability = min(swings, 50) / 50)
2. Pitch-family fallback (reliability × 0.5)
3. League baseline (reliability = 0)

Both pitch_type and pitch_archetype (KMeans k=8) scoring available.

### Layer 3: Game-Level K Prediction
**Purpose:** Produce full posterior over pitcher's K total for a specific game.

**Inputs:** K% posterior (Layer 1) + matchup lifts (Layer 2) + BF distribution + umpire/weather adjustments
**Output:** P(over X.5) for K prop lines, posterior distribution over total Ks

## Game Prop Framework

Complete validation system for game-level props:

### Supported Props
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

### Key Files Added Recently
```
src/evaluation/
├── confidence_tiers.py     # Prop confidence scoring
├── ensemble.py             # Bayes-Marcel blending
├── metrics.py              # CRPS, ECE, calibration
├── counting_backtest.py    # Counting stat validation
└── game_prop_validation.py # Complete prop framework

src/data/
├── milb_translation.py      # MiLB-to-MLB factors
└── schedule.py              # MLB API integration

scripts/
├── run_game_prop_backtest.py # Game prop validation
└── build_milb_translations.py # Build translation factors
```

## Key Design Principles

1. **Bayesian everything.** Use posterior distributions, not point estimates. Uncertainty quantification is the differentiator.
2. **Partial pooling over arbitrary filters.** Never use hard minimum PA cutoffs. Let the hierarchical model shrink small samples toward priors.
3. **Statcast informs priors, not just features.** Barrel rate and exit velo inform the PRIOR on talent, not just features in a regression.
4. **Separation of concerns.** Season projections, matchup profiles, and game predictions are separate modules that compose together.
5. **Cache aggressively.** Pitch-level queries are expensive. Feature-engineered tables cached as Parquet with clear date stamps.
6. **Branding consistency.** All visualizations use The Data Diamond color palette via `tdd_theme`.

## Coding Standards

- **Python 3.11+**, type hints on all function signatures
- **Docstrings** on all public functions (numpy style)
- **No notebooks for production code.** Notebooks are for EDA and model development only. All reusable code goes in `src/`.
- **SQL queries** live in `src/data/queries.py` as functions that return DataFrames. No raw SQL strings scattered through the codebase.
- **PyMC models** are built inside functions that return the model and trace, making them testable and reproducible. Always set `random_seed` for reproducibility.
- **Logging** over print statements. Use `logging` module.
- **Config** via YAML files, not hardcoded values. DB credentials, league average constants, model hyperparameters all go in config.

## Evaluation Requirements

Every model must be evaluated with:
1. **Walk-forward backtesting:** Train on data through season N, predict season N+1, roll forward. Never leak future data.
2. **Calibration curves:** Do 80% credible intervals contain the truth ~80% of the time?
3. **Brier score:** For binary outcomes derived from continuous projections.
4. **Benchmark vs Marcel:** Season projections must beat the Marcel system (weighted 5/4/3 recent seasons, regressed to mean, age-adjusted). If they don't, something is wrong.
5. **Betting ROI tracking:** Simulated and live bet tracking with Kelly sizing.

## Common Pitfalls to Avoid

- **Do NOT use `description` field for whiff detection without checking all values.** Use pre-computed boolean flags (`is_whiff`, `is_swing`, etc.) or the explicit mappings in `constants.py`.
- **Do NOT treat pitch_type as static.** Pitchers add/drop pitches across seasons.
- **Do NOT ignore platoon splits.** Build L/R handedness into matchup profiles.
- **Do NOT run PyMC on the full dataset during development.** Sample on a subset first, verify convergence (r_hat, ESS, divergences), then scale up.
- **Do NOT forget to regress.** Raw observed rates are not true talent. The entire point of the Bayesian model is regression to the mean with proper uncertainty.

## Implementation Status

### Phase 1: Data Foundation — COMPLETE
1. [x] Database connection + schema inspection
2. [x] Pitch-type aggregation tables (hitter/pitcher profiles) with caching
3. [x] Data QA sanity reports
4. [x] League baselines v2 — per (pitch_type, batter_stand) and (pitch_archetype, batter_stand)
5. [x] Pitch archetype clustering — KMeans k=8, 5 shape features

### Phase 2: Layer 1 — Season Talent Models — COMPLETE
6. [x] Hitter K% hierarchical Bayesian model (platoon splits, Statcast skill tier priors)
7. [x] Pitcher K% model (starter/reliever role covariate)
8. [x] Generalized hitter model (K%, BB%) + composite projections
9. [x] Generalized pitcher model (K%, BB%) + composite projections
10. [x] Walk-forward backtests — all beat Marcel on Brier, coverage 84-94%

### Phase 3: Layer 2 — Matchup Model — COMPLETE
11. [x] Matchup scoring: pitch_type + pitch_archetype, log-odds additive, fallback chains
12. [x] Matchup validation: lift over no-matchup baseline on game Ks

### Phase 4: Layer 3 — Game K Posterior — COMPLETE
13. [x] BF distribution model (shrinkage estimator, P/PA adjustment)
14. [x] Game K posterior (Monte Carlo, matchup + umpire + weather adjustments)
15. [x] Walk-forward backtest: 11,517 games, RMSE=2.280, Brier=0.1872

### Phase 5: Expansion — COMPLETE
16. [x] Counting stat projections (rate × playing time Monte Carlo)
17. [x] Content visualizations (K% movers, composite breakout/regression cards)
18. [x] In-season conjugate updating (Beta-Binomial)
19. [x] Park factors, umpire tendencies, weather effects

### Phase 6: Dashboard — MIGRATED TO tdd-dashboard
20. [x] Dashboard split complete — `tdd-dashboard/` repo with `lib/` synced modules
21. [x] `precompute_dashboard_data.py` generates all 43+ parquets/npz for dashboard

### Phase 7: Advanced Validation & Game Props — COMPLETE
22. [x] Game prop validation framework (all stats, both sides)
23. [x] Confidence tier system with automated explanations
24. [x] Advanced metrics (CRPS, ECE, temperature scaling)
25. [x] Bayes-Marcel ensemble optimization
26. [x] Counting stat backtests (hitter + pitcher)
27. [x] Minor league translation system

### Phase 8: Model Enhancement — IN PROGRESS
28. [x] AR(1) process replacement for random walk (rho ~ Beta(8,2) in hitter_model, pitcher_model, pitcher_k_rate_model)
29. [ ] wOBA projection (replacing xwOBA)
30. [ ] Enhanced season evolution modeling

### Remaining
- [ ] Betting edge finder and tracker (Kelly sizing)
- [ ] Fix SB projections (Bayes loses to Marcel — era adjustment too blunt)
- [ ] Complete AR(1) process implementation
- [ ] wOBA model integration

## Projection Target
- **Full seasons available:** 2018–2025
- **Projection target:** 2026 season (train on 2018-2025, project forward)
- Pre-season content uses 2025 posteriors projected into 2026

## Advanced Feature Triage

### Defer (real signal, but handled by model structure or needed later)
- **Stabilized contact quality** — partial pooling + random walk already captures this. Revisit for in-season.
- **Plate discipline stability** — already computed per pitch type. Revisit for in-season.
- **Velocity trend acceleration** — real signal for in-season and injury risk.

### Skip (low incremental value for this architecture)
- **Aging curve delta** — the AR(1) process IS the aging adjustment.
- **Pitch sequencing entropy** — thin evidence at season level.
- **Player similarity embeddings** — hierarchical model already borrows strength.

## Key Findings
- The sat_batted_balls.xwoba column has IEEE NaN float values (not SQL NULL), which poison PostgreSQL's AVG(). All queries use CASE WHEN xwoba != 'NaN' to handle this.
- Hitter K% and BB% are the only stats stable enough for Bayesian projection (r=0.795, 0.706 YoY). wOBA and HR/PA used as observed/informative priors only.
- Pitcher HR/BF (r=0.267) too noisy to project — replaced by GB% (r=0.619) as batted-ball indicator.
- All queries filter `game_type = 'R'` (regular season only) — no spring training/postseason leakage.
- **Counting stat performance:** Bayes beats Marcel on pitcher K (13.3% MAE improvement), BB (6.2% improvement), and Outs (15.4% improvement) with proper calibration.
- **Game prop framework:** Complete validation system with confidence tiers and advanced metrics beyond Brier score.

## AI Assistant Best Practices

### When Analyzing Code
1. **Always verify file existence** before referencing functions
2. **Check actual parameter names** in function signatures  
3. **Use grep_search** for finding function usage patterns
4. **Read the actual implementation** before suggesting changes
5. **Check model convergence** (r_hat < 1.05, ESS > 400, zero divergences)

### When Making Recommendations  
1. **Prioritize minimal changes** that address root causes
2. **Explain the "why"** behind each suggestion
3. **Consider the Bayesian framework** constraints
4. **Test assumptions** against actual data when possible
5. **Validate against backtest results** - Bayes should beat Marcel

### Common Context Needed
- Database schema verification (use psql commands)
- Model convergence diagnostics (r_hat, ESS, divergences)
- Backtest performance metrics (MAE improvement vs Marcel)
- Current season targets (2026)
- Counting stat validation results
- Game prop confidence tier assignments
