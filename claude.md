# CLAUDE.md — MLB Bayesian Projection & Matchup System

## Project Overview
A hierarchical Bayesian projection system for MLB player performance, with a game-level pitcher-batter matchup model that powers both front-office-grade analytics and betting-actionable K prop predictions. Built by The Data Diamond (Koa).

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
│   │   ├── queries.py           # 13 query functions (season totals, profiles, game logs, etc.)
│   │   ├── feature_eng.py       # Vulnerability/strength profiles + caching
│   │   ├── league_baselines.py  # Per (pitch_type/archetype, batter_stand) baselines
│   │   ├── pitch_archetypes.py  # KMeans k=8 pitch shape clustering
│   │   └── data_qa.py           # Data quality / sanity reports
│   ├── models/
│   │   ├── k_rate_model.py          # Hitter K% hierarchical Bayesian (PyMC)
│   │   ├── pitcher_k_rate_model.py  # Pitcher K% hierarchical Bayesian
│   │   ├── hitter_model.py          # Generalized hitter model (K%, BB%, HR/PA, xwOBA)
│   │   ├── pitcher_model.py         # Generalized pitcher model (K%, BB%, HR/BF)
│   │   ├── hitter_projections.py    # Composite hitter projections + breakout scoring
│   │   ├── pitcher_projections.py   # Composite pitcher projections + breakout scoring
│   │   ├── matchup.py               # Pitch-type & archetype matchup scoring
│   │   ├── bf_model.py              # Batters-faced workload model
│   │   └── game_k_model.py          # Game-level K posterior (Layer 3)
│   ├── utils/
│   │   └── constants.py         # Pitch maps, whiff defs, zone boundaries, league avgs
│   ├── evaluation/
│   │   ├── backtesting.py           # Original K%-only hitter + pitcher backtest
│   │   ├── hitter_backtest.py       # Generalized hitter walk-forward backtest
│   │   ├── pitcher_backtest.py      # Generalized pitcher walk-forward backtest
│   │   ├── matchup_validation.py    # In-season matchup lift validation
│   │   └── game_k_validation.py     # Full game-level K backtest
│   └── viz/
│       ├── theme.py                 # The Data Diamond brand theme
│       ├── projections.py           # K% mover cards, individual pitcher cards
│       └── composite_cards.py       # Composite breakout/regression cards
├── scripts/
│   ├── backfill_dim_player.py       # One-time dim_player rebuild
│   ├── run_season_backtest.py       # K%-only backtest runner
│   ├── run_hitter_backtest.py       # Multi-stat hitter backtest runner
│   ├── run_pitcher_backtest.py      # Multi-stat pitcher backtest runner
│   ├── generate_preseason_content.py  # 2026 K% mover cards
│   └── generate_composite_cards.py    # Composite breakout/regression cards
├── data/cached/                 # Parquet cache (~60 files, all seasons)
├── tests/                       # 17 test files, 119 test functions
├── notebooks/
├── outputs/                     # Backtest CSVs + content PNGs
├── docs/
│   ├── style_guide.md
│   └── advanced_projection_features.md
└── pyproject.toml
```

## Architecture: Three-Layer System

### Layer 1: Season-Level Bayesian Projections (PyMC)
**Purpose:** Estimate true-talent rates for pitchers and hitters with proper uncertainty.

**Target stats (hitters):** K%, BB%, HR/PA, xwOBA
**Target stats (pitchers):** K%, BB%, HR/BF

**Model structure:**
- Hierarchical partial pooling across players (shrink small samples toward population)
- Gaussian random walk for year-to-year talent evolution
- Aging curve as population-level covariate (polynomial or spline in age)
- Statcast skill indicators as informative priors (exit velo, barrel rate, whiff rate inform the talent prior beyond just observed outcomes)
- Binomial/Beta observation model for rate stats
- Output: Full posterior distributions per player per stat, not just point estimates

### Layer 2: Pitch-Type Matchup Model
**Purpose:** Quantify how a pitcher's arsenal maps onto a hitter's vulnerabilities AND strengths.

**Hitter vulnerability profile (per batter, per pitch type):**
- Whiff rate (swings & misses / total swings)
- Chase rate (swings at pitches outside zone / pitches outside zone)
- CSW% (called strikes + whiffs / total pitches)
- These are partially pooled via the Bayesian model to handle small samples

**Hitter strength profile (per batter, per pitch type):**
- Barrel rate on contact
- xwOBA on contact
- Hard-hit rate (exit velo >= 95 mph)
- Pull% and spray angle distribution (shows WHERE they crush it)
- These feed content (hitter strength cards) and can inform pitcher avoidance strategy

**Pitcher arsenal profile (per pitcher, per pitch type):**
- Usage rate
- Pitch-level whiff rate
- Pitch-level barrel rate against
- Typical location heatmap (zone distribution)

**Matchup calculation:**
```
For each (pitcher, batter) pair:
    For each pitch_type in pitcher.arsenal:
        k_contribution = pitcher.usage[pt] * hitter.whiff_vuln[pt] * pitcher.whiff_quality[pt]
        damage_risk = pitcher.usage[pt] * hitter.strength[pt] * (1 - pitcher.whiff_quality[pt])
    matchup_k_score = sum(k_contributions) vs baseline
    matchup_damage_score = sum(damage_risks) vs baseline
```

**Sample size handling:** Hierarchical partial pooling. Pitch types with few observations shrink toward:
1. The hitter's overall whiff/damage tendency
2. The population average for that pitch type
3. Correlated pitch types (sliders inform cutters, changeups inform splitters)

### Layer 3: Game-Level K Prediction
**Purpose:** Produce a full posterior distribution over a pitcher's K total for a specific game.

**Inputs:**
- Pitcher's true-talent K% (from Layer 1 posterior)
- Each lineup batter's pitch-type vulnerability profile (from Layer 2)
- Projected innings / pitch count (from historical workload patterns in boxscores)
- Park factor (minor adjustment)
- Handedness composition of lineup vs pitcher

**Output:**
- Posterior distribution over total Ks
- P(over X.5) for direct comparison to sportsbook lines
- Matchup-specific K probabilities per batter (for hitter K prop bets)

## Key Design Principles

1. **Bayesian everything.** Use posterior distributions, not point estimates. Uncertainty quantification is the differentiator for front offices AND for bet sizing (Kelly criterion needs probability estimates).

2. **Partial pooling over arbitrary filters.** Never use hard minimum PA cutoffs to filter data. Let the hierarchical model shrink small samples toward priors. This uses ALL available data.

3. **Statcast informs priors, not just features.** A hitter's barrel rate and exit velo should inform the PRIOR on their xwOBA talent, not just be a feature in a regression. This is the edge over public projection systems.

4. **Separation of concerns.** Season projections, matchup profiles, and game predictions are separate modules that compose together. Each is independently testable and valuable.

5. **Cache aggressively.** Pitch-level queries are expensive. Feature-engineered tables should be cached as Parquet files with clear date stamps. Rebuild only when new data arrives.

6. **Branding consistency.** All visualizations use The Data Diamond color palette:
   - Gold: #C8A96E (primary accent, highlights)
   - Teal: #4FC3C8 (secondary accent, positive/improvement)
   - Slate: #7B8FA6 (neutral, axes, secondary text)
   - Cream: #F5F2EE (light background)
   - Dark: #0F1117 (dark background, primary text on light)
   - Font: Clean sans-serif, no clutter
   - Every chart gets a "The Data Diamond" watermark

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

- **Do NOT use `description` field for whiff detection without checking all values.** Statcast's `description` column has many possible values (swinging_strike, swinging_strike_blocked, foul_tip, etc.). Define whiff/chase/called_strike mappings explicitly in `constants.py`.
- **Do NOT treat pitch_type as static.** Pitchers add/drop pitches across seasons. The model must handle pitchers whose arsenal changes.
- **Do NOT ignore platoon splits.** Build L/R handedness into matchup profiles from the start.
- **Do NOT run PyMC on the full dataset during development.** Sample on a subset of players first, verify convergence (r_hat, ESS, divergences), then scale up.
- **Do NOT forget to regress.** Raw observed rates are not true talent. The entire point of the Bayesian model is regression to the mean with proper uncertainty.

## Implementation Checklist

### Phase 1: Data Foundation
1. [x] Verify database connection and inspect schema
2. [x] Build and cache pitch-type aggregation tables (hitter and pitcher profiles)
3. [x] Data QA sanity reports (denominator consistency, row count validation, anomaly flags)
4. [x] League baselines v2: `src/data/league_baselines.py` — per (pitch_type, batter_stand) and (pitch_archetype, batter_stand), multi-season pooled with volume weighting, fallback chain lookup
5. [x] Pitch archetype clustering: `src/data/pitch_archetypes.py` — KMeans k=8 on 5 shape features, pooled multi-season fit, LHP normalization, per-season assignment with caching

### Phase 2: Layer 1 — Season Talent Models
6. [x] Hitter K% hierarchical Bayesian model (PyMC, binomial, random walk, Statcast covariates)
7. [x] Add **platoon split** (batter_stand) as hierarchical factor to hitter K% model
8. [x] Pitcher K% model (mirror hitter model using `pitcher_season_totals`; separate whiff skill vs contact suppression as covariates)
9. [x] Add **starter/reliever role** flag to pitcher model (derive from IP/game in boxscores)
10. [x] Walk-forward backtest (`scripts/run_season_backtest.py`, results in `outputs/`):
    - Hitter: beats Marcel 2/3 folds (MAE +5.1%, +3.0%, -2.0%), 95% coverage 86-89%
    - Pitcher: Bayes loses to Marcel on MAE/RMSE but wins on Brier (calibration edge: 0.171-0.209 vs 0.205-0.251)
    - All folds converge (r_hat < 1.05, 0 divergences)

### Phase 3: Layer 2 — Matchup Model
11. [x] Matchup scoring module: both pitch_type AND pitch_archetype scoring (`matchup.py`), log-odds additive method, reliability-weighted fallback chains
12. [x] Validate matchup lift over "no-matchup" baseline on game Ks

### Phase 4: Layer 3 — Game K Posterior
13. [x] Workload / BF distribution model (predict BF by role, team, season)
14. [x] Game K posterior: combine K% posterior + matchup adjustment + BF distribution
15. [x] Produce P(over X.5) and calibration — walk-forward backtest (11,517 games): RMSE=2.280, Brier=0.1872, calibration 50/80/90% = 48/79/89%

### Phase 5: Expansion & Content
16. [x] Expand hitter targets: BB%, HR/PA, xwOBA — generalized hitter model (`src/models/hitter_model.py`), composite projections (`src/models/hitter_projections.py`), age-bucket priors, LogNormal sigma_season + floor, backtest: K%/BB% beat Marcel MAE 2/3, all stats beat Brier, coverage 84-94%
16b. [x] Expand pitcher targets: K%, BB%, HR/BF — generalized pitcher model (`src/models/pitcher_model.py`), composite projections (`src/models/pitcher_projections.py`), age-bucket priors, starter/reliever covariate, backtest: HR/BF beats Marcel MAE all 3, all stats beat Brier, coverage 88-93%
17. [ ] Betting edge finder and tracker (Kelly sizing)
18. [x] Content visualizations: `src/viz/theme.py`, `src/viz/projections.py`, `scripts/generate_preseason_content.py`, `docs/style_guide.md` — pitcher/hitter K% mover cards + individual pitcher cards generated for 2026
19. [x] Counting stat projections: `src/models/pa_model.py` (PA shrinkage), `src/models/counting_projections.py` (rate × playing time Monte Carlo). Hitters: K, BB, HR (beat Marcel 14-17%, 9-17%, 5-11%). Pitchers: K, BB, Outs (beat Marcel 10-18%, 2-13%, 13-16%). Dashboard integrated.
20. [ ] Fix SB projections — Bayes loses to Marcel (era adjustment too blunt). Retune SB_ERA_FACTOR or use cross-validated era factor, or fall back to Marcel for SB.

### Phase 6: v1.4 — Game Context Integration
21. [x] Park factors — HR park adjustments by batter handedness into HR counting projections. `queries.get_park_factors()` + `queries.get_hitter_team_venue()`, half-weighted PF (50% home, 50% neutral), switch hitter L/R averaging, dashboard display in header + scouting report
22. [x] Umpire tendencies — `queries.get_umpire_k_tendencies()` (multi-season shrinkage, k=80 games), logit-scale lift in `simulate_game_ks()` and `predict_game()`, dashboard Game K Simulator umpire selector with K-rate delta display
23. [x] Weather effects — `queries.get_weather_effects()` (temp bucket × wind category → K/HR multipliers from 2018-2025 outdoor games). Logit-scale K lift in `simulate_game_ks()`, dashboard temp/wind/dome controls in Game K Simulator with K-rate and HR-rate impact display

### Projection Target
- **Full seasons available:** 2018–2025
- **Projection target:** 2026 season (train on 2018-2025, project forward)
- Pre-season content uses 2025 posteriors projected into 2026

## Advanced Feature Triage

Features from `docs/advanced_projection_features.md` evaluated for signal vs complexity.
Only features that add incremental value beyond what the hierarchical Bayesian model already provides are prioritized.

### Build (integrated into checklist above)
- **Pitch archetype clustering** (Phase 1, step 5) — clusters pitches by shape (velo, movement, spin, extension) rather than just pitch_type label. Feeds league baselines, Layer 2 matchup model, and content. High value.
- **Contact suppression vs whiff skill separation** (Phase 2, step 8) — already implicit in pitcher profiles (whiff_rate vs barrel_rate_against). Make explicit as separate covariates in pitcher K% model.
- **Role adjustment** (Phase 2, step 9) — starter/reliever distinction. Critical for workload model and prevents overestimating reliever talent.

### Defer (real signal, but handled by model structure or needed later)
- **Stabilized contact quality** (rolling xwOBA windows, variance) — partial pooling + multi-season random walk already captures this. Revisit for in-season updating.
- **Plate discipline stability** (chase rate deltas) — already computed per pitch type. Rolling stability metrics add value for in-season, not season-level.
- **Velocity trend acceleration** — real signal for in-season and injury risk. Not needed for season-level projections where random walk handles year-to-year shifts.

### Skip (low incremental value for this architecture)
- **Aging curve delta** — the random walk IS the aging adjustment. Computing a population aging curve to derive deltas feeds derived noise into the model.
- **Pitch sequencing entropy** — thin evidence of predictive power at season level. Observed whiff/chase rates already capture the downstream effect.
- **Park/environment normalization** — HR park factors now integrated (v1.4). K park effects skipped (too small, 0.95-1.05 range).
- **Player similarity embeddings** — hierarchical model already borrows strength across players via population prior. Nearest-neighbor comps are better for content than projections.

## Key Findings
- The sat_batted_balls.xwoba column has IEEE NaN float values (not SQL NULL), which poison PostgreSQL's AVG(). All queries use CASE WHEN xwoba != 'NaN' to handle this.