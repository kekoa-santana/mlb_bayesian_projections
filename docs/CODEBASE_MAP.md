# Codebase Map — player_profiles + tdd-dashboard

Complete inventory of every file, script, artifact, and data flow in the system.
Generated 2026-04-20 after three rounds of overlap cleanup.

---

## Table of Contents

1. [Data Flow Overview](#data-flow-overview)
2. [Pipeline Entry Points](#pipeline-entry-points)
3. [Dashboard Artifacts (parquet/npz)](#dashboard-artifacts)
4. [Cache Artifacts (data/cached/)](#cache-artifacts)
5. [Backtest Output Artifacts (outputs/)](#backtest-outputs)
6. [Source Modules (src/)](#source-modules)
7. [Scripts (scripts/)](#scripts)
8. [Precompute Modules (scripts/precompute/)](#precompute-modules)
9. [Dashboard Repo (tdd-dashboard)](#dashboard-repo)

---

## Data Flow Overview

```
                    PRESEASON (once)                          IN-SEASON (daily)
                    ================                          =================

    precompute_dashboard_data.py                      update_in_season.py
            |                                                 |
            v                                                 v
    +------------------+                              +------------------+
    | 15 precompute    |                              | Beta-Binomial    |
    | stages           |                              | conjugate update |
    +------------------+                              +------------------+
            |                                                 |
            v                                                 v
    tdd-dashboard/data/dashboard/                     tdd-dashboard/data/dashboard/
    (70+ parquets, 15+ npz)                           (overwrite projections,
                                                       write todays_*.parquet)
            |                                                 |
            +------------------+----------------------------->+
                               |
                               v
                    tdd-dashboard/app.py (Streamlit)
```

---

## Pipeline Entry Points

### 1. Preseason Pipeline
**Script:** `scripts/precompute_dashboard_data.py`
**When:** Once before season starts (train on 2018-2025, project 2026)
**What:** Runs 15 gated stages via `should_run()`:

| Stage | Section Name | Module | What It Produces |
|-------|-------------|--------|------------------|
| 1 | `hitter_models` | `precompute/models.py` | Fit hitter K%, BB%, GB%, FB%, HR/FB Bayesian models |
| 2 | `pitcher_models` | `precompute/models.py` | Fit pitcher K%, BB%, HR/BF Bayesian models |
| 3 | `projections` | `precompute/projections.py` | Composite projections, counting stats, health scores, park factors |
| 4 | `pitcher_samples` | `precompute/samples.py` | Pitcher K/BB/HR posterior npz files |
| 5 | `game_data` | `precompute/game_data.py` | BF priors, umpire tendencies, weather effects, catcher framing |
| 6 | `profiles` | `precompute/profiles.py` | Arsenal, vulnerability, archetypes, zone grids |
| 7 | `projection_snapshots` | `precompute/snapshots.py` | Frozen preseason snapshots |
| 8 | `backtest_summaries` | `precompute/snapshots.py` | Backtest result parquets for dashboard |
| 9 | `hitter_samples` | `precompute/samples.py` | Hitter K/BB posterior npz files |
| 10 | `game_sim_data` | `precompute/game_sim.py` | Exit model, pitch count, TTO, bullpen, BIP profiles |
| 11 | `traditional` | `precompute/traditional.py` | Traditional stats, aggressiveness, efficiency |
| 12 | `prospects` | `precompute/prospects.py` | Prospect readiness, rankings, comps |
| 13 | `glicko` | `precompute/glicko.py` | Glicko-2 player ratings |
| 14 | `rankings` | `precompute/rankings.py` | Positional rankings, breakout candidates, weekly form, standouts |
| 15 | `team` | `precompute/team.py` | Team ELO, profiles, power rankings, league sim, depth charts |
| 16 | `confident_picks` | `precompute/confident_picks.py` | Game props, DK/PP prop lines, market legs |

### 2. Daily In-Season Pipeline
**Script:** `scripts/update_in_season.py`
**When:** Daily during season (via `daily_update.bat`)
**What:**
1. Conjugate Beta-Binomial update of hitter/pitcher rate projections with observed 2026 data
2. Rebuild posterior samples (K%, BB%, HR%)
3. Fetch today's schedule + lineups from MLB Stats API
4. Run PA-by-PA game simulations for all starters (10K sims each)
5. Generate game predictions (moneyline, spread, O/U)
6. Write updated projections + daily game files

### 3. Weekly Update (flag)
**Script:** `scripts/update_in_season.py --weekly`
**When:** Weekly during season
**What:** Additionally refreshes player rankings, team ELO, team profiles/rankings/power rankings

---

## Dashboard Artifacts

All written to `tdd-dashboard/data/dashboard/` (resolved via `TDD_DASHBOARD_DIR` env var).

### Posterior Samples (npz)

| File | Producer | Contents |
|------|----------|----------|
| `pitcher_k_samples.npz` | `precompute/samples.py` | Dict[pitcher_id → array(1000)] K% posteriors |
| `pitcher_bb_samples.npz` | `precompute/samples.py` | Dict[pitcher_id → array(1000)] BB% posteriors |
| `pitcher_hr_samples.npz` | `precompute/samples.py` | Dict[pitcher_id → array(1000)] HR/BF posteriors |
| `hitter_k_samples.npz` | `precompute/samples.py` | Dict[batter_id → array(1000)] K% posteriors |
| `hitter_bb_samples.npz` | `precompute/samples.py` | Dict[batter_id → array(1000)] BB% posteriors |
| `hitter_hr_samples.npz` | `precompute/game_sim.py` | Dict[batter_id → array(1000)] HR rate synthetic Beta |
| `pitcher_game_sim_samples.npz` | `update_in_season.py` / `confident_picks.py` | Dict[key → array(10000)] game-level K/BB/H/HR/outs/runs draws |
| `batter_game_sim_samples.npz` | `confident_picks.py` | Dict[key → array(10000)] batter game-level draws |

### Preseason Snapshots (snapshots/)

| File | Producer | Contents |
|------|----------|----------|
| `pitcher_k_samples_preseason.npz` | `precompute/samples.py` | Frozen preseason pitcher K% samples |
| `pitcher_bb_samples_preseason.npz` | `precompute/samples.py` | Frozen preseason pitcher BB% samples |
| `pitcher_hr_samples_preseason.npz` | `precompute/samples.py` | Frozen preseason pitcher HR samples |
| `hitter_k_samples_preseason.npz` | `precompute/samples.py` | Frozen preseason hitter K% samples |
| `hitter_bb_samples_preseason.npz` | `precompute/samples.py` | Frozen preseason hitter BB% samples |
| `hitter_projections_2026_preseason.parquet` | `precompute/snapshots.py` | Frozen preseason hitter projections |
| `pitcher_projections_2026_preseason.parquet` | `precompute/snapshots.py` | Frozen preseason pitcher projections |

### Projection Parquets

| File | Producer | Contents |
|------|----------|----------|
| `hitter_projections.parquet` | `precompute/models.py` + `update_in_season.py` (daily) | Season projections: K%, BB%, GB%, FB%, HR/FB + composite scores |
| `pitcher_projections.parquet` | `precompute/models.py` + `update_in_season.py` (daily) | Season projections: K%, BB%, HR/BF + composite scores |
| `hitter_counting.parquet` | `precompute/projections.py` | Counting stats: PA, H, HR, BB, K, SB, R, RBI |
| `pitcher_counting.parquet` | `precompute/projections.py` | Counting stats: IP, K, BB, HR, W |
| `pitcher_counting_sim.parquet` | `precompute/projections.py` | MC season sim counting stats |
| `hitter_counting_sim.parquet` | `precompute/projections.py` | MC season sim hitter counting stats |
| `health_scores.parquet` | `precompute/projections.py` | Player health/durability scores (0-100) |
| `hr_park_factors.parquet` | `precompute/projections.py` | Per-venue HR park factors |
| `hitter_venues.parquet` | `precompute/projections.py` | Hitter-venue cross-tabulation |
| `pitcher_roles.parquet` | `precompute/projections.py` | SP/RP/CL/SU/MR role classification |

### Game Context Parquets

| File | Producer | Contents |
|------|----------|----------|
| `bf_priors.parquet` | `precompute/game_data.py` | Pitcher BF distribution priors (mu, sigma) |
| `umpire_tendencies.parquet` | `precompute/game_data.py` | Umpire K/BB logit lifts |
| `weather_effects.parquet` | `precompute/game_data.py` | Weather K/HR multipliers by temp+wind bucket |
| `catcher_framing.parquet` | `precompute/game_data.py` | Catcher framing logit lifts |
| `park_factor_lifts.parquet` | `precompute/precompute_park_factors.py` | Park K/BB/HR/BABIP logit lifts per venue |
| `team_defense_lifts.parquet` | `precompute/game_data.py` | Team OAA-based BABIP adjustment |
| `player_teams.parquet` | `precompute/game_data.py` + `update_in_season.py` | Player-to-team mapping |
| `game_lineups.parquet` | `precompute/game_data.py` | Historical game lineups |
| `pitcher_game_logs.parquet` | `precompute/game_data.py` | Pitcher game logs (starters, BF>=9) |
| `game_info.parquet` | `precompute/game_data.py` | Game metadata (date, teams, venue) |

### Game Simulator Data

| File | Producer | Contents |
|------|----------|----------|
| `exit_model.pkl` | `precompute/game_sim.py` | Trained HistGBM exit model |
| `game_bb_adj_model.pkl` | `precompute/game_sim.py` | Trained XGBoost BB adjustment model |
| `pitcher_exit_tendencies.parquet` | `precompute/game_sim.py` | Pitcher exit velocity/angle tendencies |
| `pitcher_pitch_count_features.parquet` | `precompute/game_sim.py` | Pitcher P/PA adjustments |
| `batter_pitch_count_features.parquet` | `precompute/game_sim.py` | Batter P/PA adjustments |
| `tto_profiles.parquet` | `precompute/game_sim.py` | TTO adjustment profiles per pitcher |
| `team_bullpen_rates.parquet` | `precompute/game_sim.py` | Team bullpen K/BB/HR rates |
| `batter_bip_profiles.parquet` | `precompute/game_sim.py` | Per-batter BIP outcome profiles |

### Profile & Matchup Parquets

| File | Producer | Contents |
|------|----------|----------|
| `pitcher_arsenal.parquet` | `precompute/profiles.py` | Pitch arsenal (type, velo, movement, usage) |
| `hitter_vuln.parquet` / `hitter_vuln_career.parquet` | `precompute/profiles.py` | Hitter vulnerability vs pitch types |
| `hitter_str.parquet` / `hitter_str_career.parquet` | `precompute/profiles.py` | Hitter strength profiles |
| `pitcher_location_grid.parquet` | `precompute/profiles.py` | Pitcher location heatmaps |
| `pitcher_pitch_locations.parquet` | `precompute/profiles.py` | Per-pitch-type location clusters |
| `hitter_zone_grid.parquet` / `hitter_zone_grid_career.parquet` | `precompute/profiles.py` | Hitter zone performance |
| `pitcher_offerings.parquet` | `precompute/profiles.py` | Pitch archetype cluster assignments |
| `baselines_arch.parquet` | `precompute/profiles.py` | League baselines by archetype |
| `hitter_vuln_arch.parquet` / `hitter_vuln_arch_career.parquet` | `precompute/profiles.py` | Hitter-vs-archetype vulnerability |

### Rankings & Scouting

| File | Producer | Contents |
|------|----------|----------|
| `hitters_rankings.parquet` | `precompute/rankings.py` | Positional hitter rankings (8-dim composite) |
| `pitchers_rankings.parquet` | `precompute/rankings.py` | Pitcher rankings (6-dim composite) |
| `reliever_rankings.parquet` | `precompute/projections.py` | Role-specific reliever rankings |
| `hitter_breakout_candidates.parquet` | `precompute/rankings.py` | XGBoost breakout scores |
| `pitcher_breakout_candidates.parquet` | `precompute/rankings.py` | Pitcher breakout scores |
| `hitter_grade_ci.parquet` | `precompute/rankings.py` | Scouting grade confidence intervals |
| `pitcher_grade_ci.parquet` | `precompute/rankings.py` | Pitcher grade CIs |
| `hitters_weekly_form.parquet` | `precompute/rankings.py` | Rolling 14-day form leaderboard |
| `pitchers_weekly_form.parquet` | `precompute/rankings.py` | Pitcher rolling form |
| `hitters_daily_standouts.parquet` | `precompute/rankings.py` | Single-game outstanding performances |
| `pitchers_daily_standouts.parquet` | `precompute/rankings.py` | Pitcher standout games |
| `hitter_position_eligibility.parquet` | `precompute/rankings.py` | Position eligibility mapping |

### Traditional Stats

| File | Producer | Contents |
|------|----------|----------|
| `hitter_traditional.parquet` | `precompute/traditional.py` | BA, OBP, SLG, wOBA, etc. |
| `pitcher_traditional.parquet` | `precompute/traditional.py` | ERA, WHIP, K/9, BB/9, etc. |
| `hitter_aggressiveness.parquet` | `precompute/traditional.py` | Swing%, contact%, z-contact% |
| `pitcher_efficiency.parquet` | `precompute/traditional.py` | K/9, BB/9, P/PA |
| `*_all.parquet` variants | `precompute/traditional.py` | Multi-season stacks (2018-2025) |

### Team Data

| File | Producer | Contents |
|------|----------|----------|
| `team_elo.parquet` | `precompute/team.py` | Current team ELO ratings |
| `team_elo_history.parquet` | `precompute/team.py` | ELO over time |
| `team_profiles.parquet` | `precompute/team.py` | 6-dimension team profiles |
| `team_rankings.parquet` | `precompute/team.py` | Composite team rankings |
| `team_power_rankings.parquet` | `precompute/team.py` | Power rankings (70% wins + trajectory + depth + form) |
| `league_sim.parquet` | `precompute/team.py` | Full-league Bernoulli sim results |
| `team_sim_wins.parquet` | `precompute/team.py` | Team MC season win distributions |

### Prospects

| File | Producer | Contents |
|------|----------|----------|
| `prospect_readiness.parquet` | `precompute/prospects.py` | MLB readiness scores |
| `prospect_rankings.parquet` | `precompute/prospects.py` | TDD hitter prospect rankings |
| `pitching_prospect_rankings.parquet` | `precompute/prospects.py` | Pitcher prospect rankings |
| `prospect_comps_batters.parquet` | `precompute/prospects.py` | Prospect-to-MLB player comps |
| `prospect_comps_pitchers.parquet` | `precompute/prospects.py` | Pitcher prospect comps |

### Daily Game Files (written by update_in_season.py)

| File | Contents |
|------|----------|
| `todays_sims.parquet` | Game simulation results for today's starters |
| `todays_game_predictions.parquet` | Moneyline, spread, O/U probabilities |
| `todays_games.parquet` | Today's schedule + matchup data |
| `todays_lineups.parquet` | Announced lineups |
| `game_props.parquet` | Player prop projections (P(over) at various lines) |

### Betting/Market

| File | Producer | Contents |
|------|----------|----------|
| `game_props.parquet` | `confident_picks.py` | Prop projections with confidence tiers |
| `dk_props.parquet` | `confident_picks.py` | DraftKings prop line overlay |
| `pp_props.parquet` | `confident_picks.py` | PrizePicks prop line overlay |
| `market_legs.parquet` | `confident_picks.py` | Correlated parlay legs |
| `market_parlays.parquet` | `confident_picks.py` | Parlay recommendations |

---

## Cache Artifacts

Written to `data/cached/` within player_profiles repo. Internal computation caches.

| Pattern | Producer | Contents |
|---------|----------|----------|
| `season_totals_{season}.parquet` | `feature_eng.py` | Hitter season totals per year |
| `pitcher_season_totals_{season}.parquet` | `feature_eng.py` | Pitcher season totals per year |
| `hitter_vuln_{season}.parquet` | `feature_eng.py` | Hitter vulnerability profiles per year |
| `pitcher_arsenal_{season}.parquet` | `feature_eng.py` | Pitcher arsenal profiles per year |
| `game_lineups_{season}.parquet` | `feature_eng.py` | Game lineups per year |
| `pitcher_game_logs_{season}.parquet` | `feature_eng.py` | Pitcher game logs per year |
| `milb_translated_batters.parquet` | `feature_eng.py` | MiLB translated batter stats |
| `milb_translated_pitchers.parquet` | `feature_eng.py` | MiLB translated pitcher stats |
| `matchup_shrinkage_coefs.parquet` | `fit_matchup_shrinkage.py` | Calibrated matchup dampening slopes |
| `pitch_archetypes_*_{season}.parquet` | `pitch_archetypes.py` | Pitch archetype clusters per year |
| `hitter_clusters_{season}.parquet` | `player_clustering.py` | Player archetype clusters per year |
| `pitcher_clusters_{season}.parquet` | `player_clustering.py` | Pitcher archetype clusters per year |

---

## Backtest Outputs

Written to `outputs/` within player_profiles repo. Generated by `run_*_backtest.py` scripts.

| File | Producer Script | Contents |
|------|----------------|----------|
| `hitter_k_backtest.csv` | `run_hitter_backtest.py` | Hitter K% walk-forward metrics |
| `pitcher_k_backtest.csv` | `run_pitcher_backtest.py` | Pitcher K% walk-forward metrics |
| `game_k_backtest.csv` | `run_game_k_backtest.py` | Game-level K prediction metrics |
| `game_prop_backtest_summary.parquet` | `run_game_prop_backtest.py` | All game prop backtest results |
| `game_sim_backtest.csv` | `run_game_sim_backtest.py` | PA-by-PA sim backtest metrics |
| `batter_sim_backtest.csv` | `run_batter_sim_backtest.py` | Batter sim backtest metrics |
| `counting_backtest.csv` | `run_counting_backtest.py` | Counting stat backtest metrics |
| `team_elo_backtest.csv` | `run_team_elo_backtest.py` | Team ELO validation metrics |
| `breakout_backtest.csv` | `run_breakout_backtest.py` | Breakout model AUC/precision |

---

## Source Modules

### src/data/ — Data Access & Feature Engineering

| Module | Description | Key Exports |
|--------|-------------|-------------|
| `db.py` | PostgreSQL engine + `read_sql()` helper | `get_engine()`, `read_sql()` |
| `paths.py` | Dashboard/output path resolution | `dashboard_dir()`, `dashboard_repo()` |
| `schedule.py` | MLB Stats API schedule/lineup fetcher | `fetch_todays_schedule()`, `fetch_all_lineups()` |
| `feature_eng.py` | Cached pitch-level profiles + multi-season data builders | `build_multi_season_hitter_data()`, `build_multi_season_pitcher_data()`, `load_milb_translated()` |
| `league_baselines.py` | Per (pitch_type, hand) baselines | `get_baselines_dict()` |
| `pitch_archetypes.py` | KMeans k=8 pitch shape clustering | `fit_pitch_archetype_model()` |
| `player_clustering.py` | Player archetype clustering | `fit_hitter_clustering()`, `fit_pitcher_clustering()` |
| `archetype_matchups.py` | Archetype-based matchup matrix | `get_archetype_matchup_matrix()` |
| `catcher_framing.py` | Catcher framing logit lifts | `get_catcher_framing_lift()` |
| `park_factors.py` | Park factor computation | `compute_multi_stat_park_factors()` |
| `milb_translation.py` | MiLB-to-MLB translation factors | `build_milb_translations()` |
| `fangraphs.py` | FanGraphs CSV loader | `load_fangraphs_projections()` |
| `team_queries.py` | Team-level SQL queries | `get_team_game_results()` |
| `live_boxscores.py` | Live boxscore fetcher | `fetch_boxscores()` |
| `data_qa.py` | Data quality reports | `run_data_qa()` |

### src/data/queries/ — SQL Query Functions

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `hitter.py` | Hitter profiles, season totals, zone grids | `get_hitter_observed_profile()`, `get_season_totals()` |
| `pitcher.py` | Pitcher arsenal, season totals, efficiency | `get_pitcher_arsenal_profile()`, `get_pitcher_season_totals()` |
| `game.py` | Game logs, lineups, batter game stats | `get_game_lineups()`, `get_game_pitcher_stats()` |
| `environment.py` | Park factors, umpire, weather, catcher, rest | `get_umpire_tendencies()`, `get_days_rest()`, `get_catcher_framing_effects()` |
| `traditional.py` | Traditional stats, sprint speed, spray charts | `get_season_totals_extended()`, `get_sprint_speed()` |
| `breakout.py` | Breakout features, rolling form, postseason | `get_hitter_breakout_features()`, `get_rolling_form()` |
| `prospect.py` | Prospect snapshots, FG rankings, debut rates | `get_prospect_snapshots_for_org_depth()`, `get_fg_prospect_rankings()` |
| `simulator.py` | Exit model data, pitch count features, TTO, bullpen | `get_exit_model_training_data()`, `get_tto_adjustment_profiles()` |

### src/models/ — Projection & Simulation Models

#### Layer 1: Season-Level Bayesian Projections

| Module | Description |
|--------|-------------|
| `hitter_model.py` | Generalized hierarchical Bayesian hitter (K%, BB%, GB%, FB%, HR/FB) |
| `pitcher_model.py` | Generalized hierarchical Bayesian pitcher (K%, BB%, HR/BF) |
| `hitter_projections.py` | Composite hitter projections + breakout scoring |
| `pitcher_projections.py` | Composite pitcher projections + breakout scoring |
| `counting_projections.py` | Rate x playing time Monte Carlo counting stats |
| `statcast_projections.py` | AR(1) Bayesian Statcast metric projections |
| `pa_model.py` | Hitter PA/playing time shrinkage priors |
| `bf_model.py` | Pitcher BF distribution priors |
| `in_season_updater.py` | Beta-Binomial conjugate updating |
| `derived_stats.py` | Derived stat calculations (pWOBA, etc.) |

#### Layer 2: Matchup Model

| Module | Description |
|--------|-------------|
| `matchup.py` | Pitch-type + archetype log-odds scoring with reliability fallback chains |
| `lineup_adjustments.py` | Lineup proneness aggregation for pitcher props |
| `lineup_context.py` | Lineup context features for R/RBI |

#### Layer 3: Game Simulation (src/models/game_sim/)

| Module | Description |
|--------|-------------|
| `simulator.py` | PA-by-PA Monte Carlo pitcher game sim (production engine) |
| `batter_simulator.py` | Batter game sim |
| `lineup_simulator.py` | Full lineup sim with base-state tracking |
| `pa_outcome_model.py` | Multinomial PA outcome model + GameContext |
| `bip_model.py` | Ball-in-play outcome model |
| `exit_model.py` | Pitcher exit model (leverage-aware) |
| `pitch_count_model.py` | Pitch count per PA |
| `tto_model.py` | Times-through-order adjustments |
| `batter_pa_model.py` | Batter PA count by lineup position |
| `form_model.py` | Rolling form lift calculations |
| `bullpen_model.py` | Reliever succession model |
| `fantasy_scoring.py` | DK + ESPN fantasy point distributions |
| `_sim_utils.py` | Shared sim utilities |

#### Layer 4: Market Edge

| Module | Description |
|--------|-------------|
| `market_edge.py` | Kelly sizing, correlation-aware portfolio |
| `game_predictions.py` | Game-level moneyline/spread/O-U from sim distributions |
| `game_stat_model.py` | BF-draw MC simulations (validation engine) |
| `game_bb_adj.py` | XGBoost game-level BB rate adjustment |
| `rest_adjustment.py` | Days-rest K/BB/BF adjustments |
| `posterior_utils.py` | Posterior extraction + prop-line utilities |

#### Rankings & Scouting

| Module | Description |
|--------|-------------|
| `player_rankings.py` | Positional player rankings (8-dim hitter, 6-dim pitcher) |
| `scouting_grades.py` | TDD 20-80 scouting grade system |
| `reliever_rankings.py` | Role-specific reliever composite |
| `reliever_roles.py` | CL/SU/MR classification |

#### Breakout Detection

| Module | Description |
|--------|-------------|
| `breakout_model.py` | XGBoost hitter breakout model |
| `pitcher_breakout_model.py` | Pitcher breakout model |
| `breakout_engine.py` | Breakout scoring + walk-forward validation |
| `breakout_utils.py` | Shared config/fold utilities |

#### Team Intelligence (src/models/team_sim/)

| Module | Description |
|--------|-------------|
| `team_elo.py` | Component ELO (offense/pitching/SP) |
| `series_elo.py` | Series-level ELO ratings |
| `team_profiles.py` | Team profiles (6 dimensions) |
| `team_rankings.py` | Composite team rankings + tiers |
| `season_simulator.py` | Starter season sim via game sim |
| `league_season_sim.py` | Full-league Bernoulli sim (2,430 games) |
| `team_season_sim.py` | Per-team season simulation |
| `injury_model.py` | Season injury/IL model |
| `depth_cascade.py` | Roster depth cascade |

#### Prospects

| Module | Description |
|--------|-------------|
| `prospect_ranking.py` | TDD prospect rankings |
| `prospect_bridge.py` | Prospect-to-MLB projection bridge |
| `prospect_comps.py` | Prospect-to-MLB player comparables |
| `mlb_readiness.py` | MLB readiness model |

#### Ratings

| Module | Description |
|--------|-------------|
| `player_glicko.py` | Glicko-2 player ratings from PA outcomes |
| `xgb_priors.py` | XGBoost Stuff+ priors |
| `health_score.py` | IL-based health/injury risk scores |

### src/evaluation/ — Backtesting & Validation

| Module | Description | Called By |
|--------|-------------|----------|
| `runner.py` | Shared CLI/folds/sampling defaults | All `run_*_backtest.py` scripts |
| `metrics.py` | ECE, CRPS, log loss, temperature scaling, calibration | All backtest modules |
| `baselines.py` | Marcel baseline implementation | Rate backtests |
| `ensemble.py` | Bayes-Marcel ensemble blending | Rate backtests |
| `context_lifts.py` | Umpire + weather lift construction | Game-level backtests |
| `hitter_backtest.py` | Hitter rate stat walk-forward backtest | `run_hitter_backtest.py` |
| `pitcher_backtest.py` | Pitcher rate stat walk-forward backtest | `run_pitcher_backtest.py` |
| `counting_backtest.py` | Counting stat validation | `run_counting_backtest.py` |
| `game_sim_validation.py` | PA-by-PA game sim backtest | `run_game_sim_backtest.py` |
| `game_prop_validation.py` | Unified game prop framework (canonical) | `run_game_prop_backtest.py` |
| `game_k_validation.py` | K-only wrapper (delegates to game_prop) | `run_game_k_backtest.py` |
| `batter_sim_validation.py` | Batter game sim backtest | `run_batter_sim_backtest.py` |
| `season_sim_backtest.py` | Season sim validation | `run_season_sim_backtest.py` |
| `team_elo_validation.py` | Team ELO walk-forward validation | `run_team_elo_backtest.py` |
| `confidence_tiers.py` | Prop confidence scoring | `confident_picks.py` |
| `fangraphs_comparison.py` | FanGraphs Depth Charts benchmark | `run_fangraphs_comparison.py` |
| `pa_sim_prop_adapter.py` | PA-sim to legacy prop schema bridge | `game_prop_validation.py` |

### src/utils/ — Shared Utilities

| Module | Description |
|--------|-------------|
| `constants.py` | Pitch maps, league averages, zone boundaries, BABIP coefficients |
| `math_helpers.py` | safe_logit, safe_log, credible intervals |
| `weather.py` | Temperature bucket + wind category helpers |

### src/viz/ — Visualization

| Module | Description |
|--------|-------------|
| `theme.py` | TDD brand theme (synced to dashboard) |
| `projections.py` | Projection content cards |
| `composite_cards.py` | Breakout/regression cards |
| `zone_charts.py` | Strike zone heatmaps |

---

## Scripts

### Backtest Scripts

| Script | What It Validates | Frequency |
|--------|-------------------|-----------|
| `run_hitter_backtest.py` | Hitter K%, BB%, HR/PA, xwOBA rate projections | On demand |
| `run_pitcher_backtest.py` | Pitcher K%, BB%, HR/BF rate projections | On demand |
| `run_counting_backtest.py` | Counting stat projections (rate x PA) | On demand |
| `run_game_sim_backtest.py` | PA-by-PA pitcher game simulator | On demand |
| `run_batter_sim_backtest.py` | Batter game simulator | On demand |
| `run_game_prop_backtest.py` | All game props (K/BB/HR/H/Outs, pitcher+batter) | On demand |
| `run_game_k_backtest.py` | Game K only (thin wrapper) | On demand |
| `run_season_sim_backtest.py` | Season simulator | On demand |
| `run_team_elo_backtest.py` | Team ELO calibration | On demand |
| `run_breakout_backtest.py` | XGBoost breakout models | On demand |
| `run_sim_vs_odds_backtest.py` | Sims vs sportsbook odds | On demand |

### Validation Scripts

| Script | Purpose |
|--------|---------|
| `validate_change_safety.py` | Pre-merge safety gate (schema checks) |
| `validate_core_weekly_form.py` | Contract check for core + weekly form files |
| `validate_daily_props.py` | Day-forward prop accuracy (no re-run needed) |

### One-Off / Research

| Script | Purpose |
|--------|---------|
| `generate_preseason_content.py` | Preseason PNG content cards |
| `generate_composite_cards.py` | Breakout/regression visual cards |
| `build_hitter_descriptions.py` | Ranking description text |
| `build_milb_translations.py` | MiLB translation factor derivation |
| `build_preseason_injuries.py` | Preseason injury data |
| `apply_injury_adjustments.py` | Adjust counting projections for injuries |
| `backfill_dim_player.py` | One-time dim_player rebuild |
| `run_statcast_projections.py` | AR(1) Statcast metric projections |
| `run_fangraphs_comparison.py` | TDD vs FanGraphs benchmark |
| `grade_prior_analysis.py` | Grade prior distribution research |
| `grade_yoy_stability.py` | Grade year-over-year stability research |

---

## Dashboard Repo (tdd-dashboard)

### Files Synced from player_profiles

These are copied (not linked) from `src/` to `tdd-dashboard/lib/`. Sync when signatures change:

| Source (player_profiles) | Destination (tdd-dashboard) |
|--------------------------|-----------------------------|
| `src/utils/constants.py` | `lib/constants.py` |
| `src/viz/theme.py` | `lib/theme.py` |
| `src/viz/zone_charts.py` | `lib/zone_charts.py` |
| `src/models/matchup.py` | `lib/matchup.py` |
| `src/models/bf_model.py` | `lib/bf_model.py` |
| `src/models/in_season_updater.py` | `lib/in_season_updater.py` |
| `src/models/rest_adjustment.py` | `lib/rest_adjustment.py` |
| `src/models/game_sim/*.py` (13 files) | `lib/game_sim/*.py` |
| `src/models/game_predictions.py` | `lib/game_predictions.py` |
| `src/data/schedule.py` | `lib/schedule.py` |
| `src/data/db.py` (subset) | `lib/db.py` |

### Dashboard-Only Files (no source equivalent)

| File | Purpose |
|------|---------|
| `lib/bovada.py` | Bovada odds scraper |
| `lib/draftkings.py` | DraftKings API client |
| `lib/prizepicks.py` | PrizePicks API client |
| `lib/diamond_rating.py` | Diamond Rating display logic |

### Data Flow: player_profiles → tdd-dashboard

```
player_profiles/                              tdd-dashboard/
  precompute_dashboard_data.py ──writes──►      data/dashboard/*.parquet
  update_in_season.py          ──writes──►      data/dashboard/todays_*.parquet
                                                    |
                                                    v
                                                  app.py (Streamlit reads parquets)
```

---

## Test Files

| Test | What It Validates |
|------|-------------------|
| `test_sim_consistency.py` (13 tests) | Legacy model deletion, rest/catcher/TTO/fatigue wiring, reliever symmetry, BABIP variance |
| `test_overlap_dedup.py` (10 tests) | Weather helpers, TTO dedup, precompute gating, breakout utils, game_k wrapper, fold centralization |
| `test_daily_standouts.py` | Daily standout computation |
| `test_weekly_form.py` | Weekly form model |
| `test_model_integration.py` | End-to-end model fit + posterior extraction (integration, slow) |
| `test_precompute_rankings_wiring.py` | Rankings precompute wiring |
| `test_stat_config_merge.py` | Stat config consistency |
| `test_update_rate_samples_dispatch.py` | Rate sample dispatch logic |
| `test_rest_adjustment.py` | Rest adjustment calculations |
| `test_matchup_*.py` | Matchup scoring |
