# AGENTS.md

## Purpose
Session bootstrap notes for AI agents working in `e:\data_analytics\player_profiles`.
Read this at the start of every new session before making changes.

## Project Summary
- Domain: MLB analytics and betting-focused strikeout projections.
- Core architecture: 3 layers from `docs/claude.md`.
- Layer 1 status: hitter K% model implemented (`src/models/k_rate_model.py`, `src/evaluation/backtesting.py`). Pitcher K% model not yet built. Platoon splits not yet added.
- Layer 2 status: partially prepared via feature engineering and query outputs; matchup model not yet implemented. Pitch archetype clustering (from `sat_pitch_shape`) is the key prerequisite.
- Layer 3 status: not implemented. Workload/BF model needed first.

## Current Code Map
- Config
  - `.env`: Postgres connection settings (with optional legacy YAML fallback in code).
  - `config/model.yaml`: season split, sampling settings, cache settings.
- Data
  - `src/data/db.py`: SQLAlchemy engine and `read_sql`.
  - `src/data/queries.py`: all SQL query functions.
  - `src/data/feature_eng.py`: cache management and feature builders (includes `compute_league_baselines`, `build_multi_season_k_data`).
- Modeling
  - `src/models/k_rate_model.py`: hierarchical Bayesian hitter K% model in PyMC (binomial likelihood, random walk, Statcast covariates).
- Evaluation
  - `src/evaluation/backtesting.py`: walk-forward, Marcel baseline, metrics/calibration.
- Constants
  - `src/utils/constants.py`: pitch mappings, event definitions, league prior constants, pitch families.
- Docs
  - `docs/claude.md`: product intent, architecture, standards, implementation checklist, and advanced feature triage.
  - `docs/advanced_projection_features.md`: reference list of advanced features (triaged — see claude.md for build/defer/skip decisions).

## Startup Workflow (Every Session)
1. Read `docs/claude.md` and this file.
2. Inspect current file tree for new/changed files.
3. Confirm task scope against the implementation checklist phases (1-5) in claude.md.
4. Keep SQL in `src/data/queries.py`; keep reusable logic in `src/`.
5. Add or update tests when behavior changes.
6. Report what changed, what was verified, and what remains unverified.

## Engineering Rules For This Repo
- Use type hints and docstrings on public functions.
- Use logging, not print.
- Use YAML-driven config values when possible.
- Cache expensive query outputs to `data/cached` as Parquet.
- Avoid introducing notebook-only production logic.
- The hierarchical Bayesian model handles small samples via partial pooling — never use hard PA cutoffs to exclude data from the model. Filter only ghost rows (0 PA).

## Known Risks / Gaps
- Secrets exposure risk: DB credentials are committed in config/local settings.
- No real test coverage yet (`tests/` is effectively empty).
- Backtest has not been run end-to-end yet — convergence and Marcel comparison unverified.
- Player 444482 exists in `fact_pitch` but not `dim_player` (data gap — queries joining on `dim_player` silently drop this player).
- `sat_batted_balls.xwoba` contains IEEE NaN floats (not SQL NULL). All aggregations must filter with `WHERE xwoba != 'NaN'` or `CASE WHEN`.

## What's Next (in order)
See `docs/claude.md` implementation checklist for the full sequence. Immediate next steps:

1. **Data QA sanity reports** — validate denominator consistency (swings >= whiffs, pitches >= swings, etc.), row counts, anomaly flags.
2. **League baselines v2** — extend with batter_stand (handedness) splits and pitch archetype granularity.
3. **Pitch archetype clustering** — cluster pitches from `sat_pitch_shape` by shape characteristics. This feeds baselines, Layer 2 matchup model, and content.
4. **Platoon split in hitter K% model** — add `batter_stand` as hierarchical factor.
5. **Pitcher K% model** — mirror hitter model for pitchers, separate whiff skill vs contact suppression covariates.
6. **Run the backtest** — execute `run_full_backtest()`, verify convergence, confirm beats Marcel.

## Advanced Features Decision
Detailed triage is in `docs/claude.md` under "Advanced Feature Triage". Summary:
- **Build:** pitch archetype clustering, contact suppression vs whiff skill separation, role adjustment
- **Defer:** stabilized contact quality, plate discipline stability, velocity trends (all better suited for in-season updating, not season-level projections)
- **Skip:** aging curve delta (random walk handles it), sequencing entropy (thin evidence), park factors (minimal K effect), player embeddings (population prior already borrows strength)
