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
│   ├── model.yaml               # Sampling, season config, game_k params
│   └── change_safety.yaml       # Schema change safety validation rules
├── src/
│   ├── data/
│   │   ├── db.py                # SQLAlchemy engine + read_sql helper
│   │   ├── queries.py           # Query functions (season totals, profiles, game logs, trad stats, etc.)
│   │   ├── feature_eng.py       # Vulnerability/strength profiles + caching
│   │   ├── league_baselines.py  # Per (pitch_type/archetype, batter_stand) baselines
│   │   ├── pitch_archetypes.py  # KMeans k=8 pitch shape clustering
│   │   ├── player_clustering.py # Player archetype clustering (hitter/pitcher)
│   │   ├── archetype_matchups.py# Archetype-based matchup matrix + scoring
│   │   ├── park_factors.py      # Park factor queries + adjustments
│   │   ├── milb_translation.py  # MiLB-to-MLB translation factors
│   │   ├── team_queries.py      # Team-level SQL queries (game results, park factors, roster comp)
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
│   │   ├── player_rankings.py       # Positional rankings (hitter + pitcher composites)
│   │   ├── scouting_grades.py       # TDD 20-80 scouting grade system
│   │   ├── statcast_projections.py  # AR(1) Bayesian Statcast metric projections
│   │   ├── breakout_model.py        # XGBoost hitter breakout model
│   │   ├── pitcher_breakout_model.py# Pitcher breakout model
│   │   ├── breakout_engine.py       # Breakout scoring + walk-forward validation
│   │   ├── player_glicko.py         # Glicko-2 player ratings from PA outcomes
│   │   ├── prospect_comps.py        # Prospect-to-MLB player comparables
│   │   ├── prospect_ranking.py      # TDD prospect rankings (batter + pitcher)
│   │   ├── mlb_readiness.py         # MLB readiness model (logistic regression)
│   │   ├── reliever_roles.py        # CL/SU/MR classification
│   │   ├── reliever_rankings.py     # Role-specific reliever composite
│   │   ├── season_simulator.py      # Starter season sim via game sim
│   │   ├── health_score.py          # IL-based health/injury risk scores
│   │   ├── derived_stats.py         # Derived stat calculations (pWOBA, etc.)
│   │   ├── rest_adjustment.py       # Rest-day performance adjustments
│   │   ├── lineup_adjustments.py    # Lineup-based game adjustments
│   │   ├── lineup_context.py        # Lineup context features
│   │   ├── xgb_priors.py            # XGBoost Stuff+ priors
│   │   ├── team_elo.py              # Component ELO engine (offense/pitching/SP)
│   │   ├── team_profiles.py         # Team profile builder (6 dimensions)
│   │   ├── team_rankings.py         # Composite team rankings + tiers
│   │   ├── series_elo.py            # Series-level ELO ratings
│   │   ├── in_season_updater.py     # Beta-Binomial conjugate updating
│   │   ├── game_sim/               # Game-level PA-by-PA simulator
│   │   │   ├── simulator.py         # Pitcher game sim (sequential PA MC)
│   │   │   ├── batter_simulator.py  # Batter game sim
│   │   │   ├── pa_outcome_model.py  # PA outcome multinomial model
│   │   │   ├── bip_model.py         # Ball-in-play outcome model
│   │   │   ├── exit_model.py        # Pitcher exit model (leverage-aware)
│   │   │   ├── pitch_count_model.py # Pitch count per PA model
│   │   │   ├── tto_model.py         # Times-through-order adjustments
│   │   │   ├── batter_pa_model.py   # Batter PA count by lineup position
│   │   │   ├── lineup_simulator.py  # Full lineup simulation
│   │   │   └── fantasy_scoring.py   # DK + ESPN fantasy point distributions
│   │   └── team_sim/               # Team/league-level simulation
│   │       ├── league_season_sim.py  # Full-league Bernoulli season sim (2,430 games)
│   │       ├── team_season_sim.py    # Per-team season simulation
│   │       ├── injury_model.py       # Season injury/IL model
│   │       └── depth_cascade.py      # Roster depth cascade model
│   ├── utils/
│   │   └── constants.py         # Pitch maps, whiff defs, zone boundaries, league avgs
│   ├── evaluation/
│   │   ├── backtesting.py           # Original K%-only hitter + pitcher backtest
│   │   ├── hitter_backtest.py       # Generalized hitter walk-forward backtest
│   │   ├── pitcher_backtest.py      # Generalized pitcher walk-forward backtest
│   │   ├── matchup_validation.py    # Matchup lift validation (kept for future use)
│   │   ├── game_k_validation.py     # Full game-level K backtest
│   │   ├── game_sim_validation.py   # Pitcher game sim backtest
│   │   ├── batter_sim_validation.py # Batter game sim backtest
│   │   ├── season_sim_backtest.py   # Season sim validation
│   │   ├── ranking_validation.py    # Player ranking validation
│   │   ├── team_ranking_validation.py # Team ranking validation
│   │   ├── confidence_tiers.py      # Prop confidence scoring system
│   │   ├── ensemble.py              # Bayes-Marcel ensemble blending
│   │   ├── metrics.py               # ECE, temperature scaling, calibration
│   │   ├── counting_backtest.py     # Counting stat validation
│   │   ├── game_prop_validation.py  # Complete game prop framework
│   │   └── team_elo_validation.py   # Walk-forward ELO validation
│   └── viz/
│       ├── theme.py                 # The Data Diamond brand theme
│       ├── projections.py           # K% mover cards, individual pitcher cards
│       ├── composite_cards.py       # Composite breakout/regression cards
│       └── zone_charts.py           # Strike zone heatmap visualizations
├── scripts/
│   ├── precompute_dashboard_data.py   # Generates all dashboard parquets/npz
│   ├── precompute/                    # Precompute submodules
│   │   ├── confident_picks.py         # Game props, DK/PP prop lines
│   │   ├── game_data.py              # BF priors, game context data
│   │   ├── game_sim.py               # Daily game simulations
│   │   ├── glicko.py                 # Glicko-2 ratings computation
│   │   ├── models.py                 # Bayesian model fitting (K%, BB%)
│   │   ├── profiles.py               # Player archetype profiles
│   │   ├── projections.py            # Counting stat projections + health/parks
│   │   ├── prospects.py              # Prospect rankings + comps
│   │   ├── rankings.py               # Player rankings generation
│   │   ├── samples.py                # Posterior sample extraction
│   │   ├── snapshots.py              # Preseason snapshot + backtest parquets
│   │   ├── team.py                   # Team ELO, profiles, rankings, league sim
│   │   ├── traditional.py            # Traditional stat aggregations
│   │   ├── backtest_head_to_head.py   # Head-to-head backtest comparison
│   │   ├── backtest_ld_babip.py       # LD rate BABIP backtest
│   │   ├── backtest_lineup_sim.py     # Lineup sim backtest
│   │   ├── backtest_park_factors.py   # Park factor backtest
│   │   ├── backtest_sprint_speed.py   # Sprint speed backtest
│   │   ├── backtest_umpire_bb.py      # Umpire BB rate backtest
│   │   ├── compare_old_vs_new.py      # Old vs new model comparison
│   │   ├── feature_collinearity.py    # Feature collinearity analysis
│   │   ├── precompute_ld_rate.py      # Line drive rate precompute
│   │   ├── precompute_park_factors.py # Park factor precompute
│   │   ├── precompute_sprint_speed.py # Sprint speed precompute
│   │   └── validate_lineup_sim.py     # Lineup sim validation
│   ├── update_in_season.py            # Daily conjugate update pipeline
│   ├── run_statcast_projections.py    # AR(1) Statcast metric projections
│   ├── run_season_backtest.py         # K%-only backtest runner
│   ├── run_hitter_backtest.py         # Multi-stat hitter backtest runner
│   ├── run_pitcher_backtest.py        # Multi-stat pitcher backtest runner
│   ├── run_counting_backtest.py       # Counting stat backtest runner
│   ├── run_game_k_backtest.py         # Game-level K backtest runner
│   ├── run_game_sim_backtest.py       # Game sim (pitcher) backtest runner
│   ├── run_batter_sim_backtest.py     # Batter sim backtest runner
│   ├── run_season_sim_backtest.py     # Season sim backtest runner
│   ├── run_breakout_backtest.py       # Breakout model backtest runner
│   ├── run_team_elo_backtest.py       # Team ELO calibration runner
│   ├── run_game_prop_backtest.py      # Game prop validation runner
│   ├── generate_preseason_content.py  # 2026 K% mover cards
│   ├── generate_composite_cards.py    # Composite breakout/regression cards
│   ├── build_preseason_injuries.py    # Preseason injury data
│   ├── build_milb_translations.py     # Build MiLB translation factors
│   ├── apply_injury_adjustments.py    # Adjust counting projections for injuries
│   ├── backfill_dim_player.py         # One-time dim_player rebuild
│   ├── grade_prior_analysis.py        # Grade prior distribution analysis
│   ├── grade_yoy_stability.py         # Grade year-over-year stability analysis
│   ├── validate_change_safety.py      # Change safety validation runner
│   └── validate_daily_props.py        # Day-forward prop validation (reads dashboard game_props)
├── data/
│   └── cached/                  # Parquet cache (~60 files, all seasons)
├── tests/
├── notebooks/
├── outputs/                     # Backtest CSVs + content PNGs
├── docs/
│   ├── style_guide.md
│   ├── advanced_projection_features.md
│   ├── failed_hypotheses.md
│   ├── SCHEMA_REFERENCE.md
│   ├── model_performance_dashboard_spec.md
│   ├── change_safety_checklist.md
│   └── recommended_improvements.md
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
- `src/viz/zone_charts.py` → `lib/zone_charts.py`
- `src/models/matchup.py` → `lib/matchup.py`
- `src/models/bf_model.py` → `lib/bf_model.py`
- `src/models/game_k_model.py` → `lib/game_k_model.py`
- `src/models/in_season_updater.py` → `lib/in_season_updater.py`
- `src/models/rest_adjustment.py` → `lib/rest_adjustment.py`
- `src/models/game_sim/*.py` → `lib/game_sim/*.py` (9 files: simulator, batter_simulator, pa_outcome_model, bip_model, exit_model, pitch_count_model, tto_model, batter_pa_model, fantasy_scoring)
- `src/data/schedule.py` → `lib/schedule.py`
- `src/data/db.py` → `lib/db.py` (subset)

Dashboard-only files (no source equivalent): `lib/bovada.py`, `lib/draftkings.py`, `lib/prizepicks.py`, `lib/diamond_rating.py`

**To sync:** copy file, replace `from src.` with `from lib.` in imports.

## Architecture: Three-Layer System

### Layer 1: Season-Level Bayesian Projections (PyMC)
**Purpose:** Estimate true-talent rates for pitchers and hitters with proper uncertainty.

**Target stats (hitters):** K%, BB%, GB%, FB%, HR/FB
**Target stats (pitchers):** K%, BB%, HR/BF
**Observed stats used as priors:** wOBA, HR/PA, barrel rate, hard-hit rate, sprint speed, whiff rate, chase rate, etc.
**Statcast projections:** AR(1) on 10 individual metrics (whiff%, barrel%, CSW%, etc.) via `statcast_projections.py`

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
- Database connection + schema inspection, pitch-type aggregation with caching
- Data QA, league baselines v2, pitch archetype clustering (KMeans k=8)

### Phase 2: Layer 1 — Season Talent Models — COMPLETE
- Hitter + Pitcher K%/BB% hierarchical Bayesian models (PyMC)
- AR(1) talent evolution (rho ~ Beta(8,2)), age-bucket × Statcast skill tier priors
- Walk-forward backtests: all beat Marcel on Brier, coverage 84-94%

### Phase 3: Layer 2 — Matchup Model — COMPLETE
- Pitch-type + pitch-archetype log-odds scoring with reliability-weighted fallback chains

### Phase 4: Layer 3 — Game Prediction — COMPLETE
- Game K posterior (BF distribution + matchup + umpire/weather), backtest: 11,517 games, RMSE=2.280
- PA-by-PA game simulator (pitcher + batter), exit model (AUC=0.921), DK/ESPN fantasy scoring
- Game prop framework (pitcher K/BB/HR/H/Outs, batter K/BB/HR/H) with confidence tiers

### Phase 5: Expansion — COMPLETE
- Counting stat projections (rate × playing time MC), in-season conjugate updating
- Park factors, umpire tendencies, weather effects, content visualizations

### Phase 6: Dashboard — MIGRATED TO tdd-dashboard
- Precompute bridge generates 70+ parquets for dashboard consumption

### Phase 7: Rankings & Prospects — COMPLETE
- Positional player rankings (hitter 8-dim composite, pitcher 6-dim composite)
- TDD 20-80 scouting grades (6 hitter tools + 3 pitcher tools + Diamond Rating)
- AR(1) Statcast metric projections with grade confidence intervals
- XGBoost breakout model (hitter + pitcher), prospect rankings + MLB readiness
- Prospect-to-MLB player comparables, MiLB translation factors
- Reliever role classification (CL/SU/MR), reliever-specific rankings

### Phase 8: Team Intelligence — COMPLETE
- Component ELO (offense/pitching/SP, 61K games, 55.2% accuracy)
- Team profiles (6 dimensions), composite rankings with tier labels
- Full-league Bernoulli season sim (2,430 games × 1,000 sims)
- Power rankings (70% wins + 12% trajectory + 10% depth + 8% form)
- Glicko-2 player ratings from PA outcomes

### Remaining
- [ ] Betting edge finder and tracker (Kelly sizing)
- [ ] Fix SB projections (Bayes loses to Marcel — era adjustment too blunt)
- [ ] Wire capability-gap dashboard views (prospect comps, Glicko trends, game prop summary)
- [x] In-memory precompute intermediates (eliminate parquet round-trips for internal cache files)
- [x] Grade confidence intervals wired into dashboard (schedule, profile, rankings views)

## Projection Target
- **Full seasons available:** 2018–2025
- **Projection target:** 2026 season (train on 2018-2025, project forward)
- Pre-season content uses 2025 posteriors projected into 2026

## Advanced Feature Triage

### Defer (revisit for in-season)
- **Velocity trend acceleration** — real signal for in-season and injury risk.
- **Stabilized contact quality / plate discipline stability** — partial pooling + AR(1) handles at season level; revisit for in-season updating.

### Skip (low incremental value for this architecture)
- **Aging curve delta** — the AR(1) process IS the aging adjustment.
- **Pitch sequencing entropy** — thin evidence at season level.
- **Player similarity embeddings** — hierarchical model already borrows strength.

## Key Findings
- The sat_batted_balls.xwoba column has IEEE NaN float values (not SQL NULL), which poison PostgreSQL's AVG(). All queries use CASE WHEN xwoba != 'NaN' to handle this.
- Hitter K% and BB% are the most stable stats for Bayesian projection (r=0.795, 0.706 YoY). Batted ball decomposition (GB%, FB%, HR/FB) added for hitter model v2.
- Pitcher HR/BF (r=0.267) too noisy to project — replaced by GB% (r=0.619) as batted-ball indicator.
- All queries filter `game_type = 'R'` (regular season only) — no spring training/postseason leakage.
- **Counting stat performance:** Bayes beats Marcel on pitcher K (+13.3% MAE), BB (+6.2%), Outs (+15.4%). SB loses to Marcel (era adjustment too blunt).
- **Game sim performance:** Pitcher K RMSE=2.305, calibration near-perfect (50.8/81.2/90.9%).
- **Hitter chase rate** is the most stable count metric (r=0.84 YoY, 87% between-player variance). Aggressiveness is value-neutral (Q4 vs Q1 wOBA = .328 vs .326).

## AI Assistant Best Practices

### When Analyzing Code
1. **Always verify file existence** before referencing functions
2. **Check actual parameter names** in function signatures  
3. **Search for usage patterns** before modifying shared functions
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
