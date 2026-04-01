# Change Safety Checklist

Use this checklist after modifying any high-risk modeling, simulation, or calibration logic.

## 0) Pre-change setup (always)
- Record exactly what changed (files + function names).
- Pick one deterministic run mode (`--quick` or full) and keep it fixed for before/after comparison.
- Snapshot baseline artifacts before changes:
  - `backtest_game_prop_summary.parquet`
  - `backtest_confidence_tiers.parquet`
  - `game_props.parquet`
  - `pitcher_k_samples.npz`, `pitcher_bb_samples.npz`, `pitcher_hr_samples.npz`

## 1) If you changed logit math or matchup-lift logic
- Files: `src/models/matchup.py`, `src/models/game_k_model.py`, `src/models/lineup_adjustments.py`
- Run:
  - `python scripts/run_game_prop_backtest.py`
- Verify:
  - Mean `ECE` does not materially worsen (target near prior baseline).
  - Temperature stays near 1.0 (no new systematic over/under-confidence).
  - No sign inversion symptoms:
    - High K-prone lineups should not reduce pitcher K `P(over)`.
    - High chase batters should not increase BB lift.

## 2) If you changed BF/PA distributions or sampling bounds
- Files: `src/models/bf_model.py`
- Run:
  - `python scripts/run_game_k_backtest.py`
  - `python scripts/run_game_prop_backtest.py`
- Verify:
  - Expected totals and variance stay realistic:
    - Pitcher K RMSE not materially worse.
    - Coverage (50/80/90/95) does not collapse.
  - Tail behavior sanity:
    - `P(over)` curves are smooth/monotonic by line.
    - No sudden clipping near 0 or 1 across many games.

## 3) If you changed TTO/slot indexing or lineup-weighting logic
- Files: `src/models/game_k_model.py`, `src/models/game_sim/simulator.py`, `src/models/lineup_adjustments.py`
- Run:
  - `python scripts/precompute/validate_lineup_sim.py`
  - `python scripts/run_game_prop_backtest.py`
- Verify:
  - No off-by-one artifacts:
    - Similar lineup orders should have similar aggregate lift.
    - TTO effects should show expected directional pattern by stat.
  - Batter slot sensitivity remains plausible (1-3 > 8-9 in opportunity impact).

## 4) If you changed rest adjustments
- Files: `src/models/rest_adjustment.py`
- Run:
  - `python scripts/run_game_k_backtest.py`
  - `python scripts/run_game_prop_backtest.py`
- Verify:
  - Short-rest starts show lower BF on average, not higher.
  - K/BB rate shifts remain small (unless intentionally redesigned).
  - Global calibration unchanged outside short-rest cohort.

## 5) If you changed in-season conjugate updating
- Files: `src/models/in_season_updater.py`, `scripts/update_in_season.py`
- Run:
  - `python scripts/update_in_season.py --skip-schedule`
- Verify:
  - Posterior means move in expected direction with observed outcomes.
  - Posterior SD narrows with larger trial counts.
  - New-player behavior is stable (no extreme 0/1 rate spikes).
  - BB/HR update path maps to correct observed columns.

## 6) If you changed pitcher exit model, exit calibration, or PA caps
- Files: `src/models/game_sim/simulator.py`, `src/models/game_sim/exit_model.py`
- Run:
  - `python scripts/run_game_sim_backtest.py`
  - `python scripts/run_batter_sim_backtest.py`
- Verify:
  - Simulated IP/BF distributions align with historical range.
  - No mass pileups at hard caps (`MAX_PA_PER_GAME`, pitch cap).
  - Fantasy and counting tails remain realistic.

## 7) If you changed fallback priors / synthetic sample generation
- Files: `scripts/precompute/projections.py`, `scripts/precompute/game_sim.py`, `scripts/precompute/confident_picks.py`
- Run:
  - `python scripts/precompute_dashboard_data.py --include samples,game_sim,picks --quick`
- Verify:
  - Fallback players do not dominate edges.
  - League-average fallback constants are consistent across modules.
  - Sample count consistency (8k/10k) does not create unstable rank ordering.

## 8) If you changed validation schema or metric wiring
- Files: `src/evaluation/game_prop_validation.py`, `src/evaluation/metrics.py`
- Run:
  - `python scripts/run_game_prop_backtest.py`
  - `python scripts/precompute_dashboard_data.py --include snapshots --quick`
- Verify:
  - All expected `p_over_*` columns are present.
  - No silently skipped lines in Brier/ECE/MCE/temperature.
  - Summary parquets load cleanly in downstream dashboard pages.

## 9) Minimal acceptance gates before merging
- Calibration:
  - No major degradation in `ECE`, `MCE`, temperature, or coverage.
- Accuracy:
  - RMSE/MAE not materially worse on core props (at least pitcher K, BB, HR).
- Stability:
  - Confidence tier assignments do not churn wildly without intentional reason.
- Operational:
  - `scripts/precompute_dashboard_data.py --quick` completes successfully.
  - No schema breaks in generated dashboard parquets.

## 10) Optional deeper diagnostics (when something looks off)
- Slice results by:
  - rest bucket (short/normal/extended)
  - lineup strength deciles
  - high/low park-factor games
  - confirmed vs projected lineups
- Compare before/after:
  - distribution means, SDs, and 90th percentiles
  - count of edges above 60/65/70% thresholds
  - calibration by line, not just aggregate

## 11) (Optional) Roadmap to reduce manual risk
- Automate these checks in `scripts/validate_change_safety.py` (thresholds in `config/change_safety.yaml`).
- Add parquet schema/column contract validation for the dashboard outputs.
- Add small invariant tests for matchup/TTO/rest/logit monotonicity.
- Centralize clip bounds and default constants used across modules.