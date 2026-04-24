# Batter hit & multi-hit signal -- execution plan

Status as of 2026-04-23.

## Goals

1. Surface realistic hit distributions in sims, including non-trivial probability at 1.5+ / 2.5+ hit lines.
2. Remove or document silent gaps where environmental or park signal exists but is not passed into the sim.
3. Keep scope tight: wire existing data, validate, then consider deeper model changes.

## Completed work

### Phase 1 -- Wire `park_h_babip_adj` (DONE)

All items complete:
- `park_factor_lifts.parquet` `h_babip_adj` column loaded in `confident_picks.py` park lift lookup
- `park_h_babip_adj` passed to all 3 sim call sites:
  - `simulate_lineup_game()` (batter H/K/BB/R/RBI/HRR props)
  - `simulate_batter_game()` (batter TB props)
  - `simulate_full_game_both_teams()` (game-level ML/spread/O-U)
- `park_k_lift`, `park_bb_lift`, `park_hr_lift` also added to batter sim (were previously missing)
- Pitcher sim `GameContext` now receives `park_k_lift`, `park_bb_lift`, `park_hr_lift`, `park_h_babip_adj`
- Stale comment at line 331 ("K and HR only; H/BB skipped") updated
- Impact verified: Coors +1.78 H/game, Oracle -0.75 H/game

### Phase 2 -- H vs TB path consistency (DONE -- Option A)

Both paths now get identical environment (ump, weather, park, h_babip). The batter sim for TB receives the same `park_k/bb/hr_lift` and `park_h_babip_adj` as the lineup sim for H.

### BIP model differentiation (DONE)

- Quality-metric `_BABIP_COEFS` scaled ~2x, intercept recalculated for league-avg pred ~0.295
- `shrinkage_k` 300 -> 150 (faster convergence to observed splits)
- `pred_babip` clamp widened from [0.15, 0.45] to [0.12, 0.55]
- Post-adjustment hit-probability clamps widened from [0.05, 0.50] to [0.03, 0.55]
- `BABIP_LD_COEFF` 0.25 -> 0.60 (in `constants.py`)
- BIP profiles regenerated with new coefficients (std 0.019 -> 0.032, 1.7x wider)

### Correlated K/BB/HR resampling (DONE -- Phase 2 Arch 2)

`resample_posterior_joint()` in `_sim_utils.py` preserves joint K-HR correlation (-0.23) through the sim via shared index vector. Wired into both `lineup_simulator.py` and `simulator.py`.

### BatterHModel -- direct H count model (DONE)

- Poisson regression for per-batter H counts (`src/models/game_sim/batter_h_model.py`)
- Features: batter H/PA, K%, BABIP, sprint speed, pitcher K%/BB%, batting order, platoon, park BABIP, home
- Trained on 2022-2024, validated on 2025 (r=0.165, well-calibrated P(2+H) by decile)
- Blended 50/50 with sim H samples in `confident_picks.py`
- Net: expected H std +17% (0.149 -> 0.175), P(>1.5) max +8pp

## Remaining work

### Phase 3 -- Multi-hit visibility in outputs

- `p_over_1.5` and `p_over_2.5` columns already exist in `game_props.parquet`
- Dashboard may need UI changes to highlight these lines for batter H
- Book merge: match 1.5/2.5 hit markets when offered

### Phase 4 -- Deeper model signal (deferred)

1. HR matchup (Path B): re-enable only with backtest and shrinkage (disabled for -0.43 logit bias)
2. BIP/BABIP from hitter model: export wOBA/contact-quality posteriors to npz
3. H form lift: add rolling 15g vs 30g H/PA deviation to `BatterFormLifts`

### Phase 5 -- Calibration re-check

After all BIP + LD + clamp changes, ECE for batter H should be re-checked on walk-forward holdout. The `docs/paper_v1_calibrated_game_props.md` cites strong batter-H calibration from the pre-change model.

## File map

| Area | Files |
|------|-------|
| Park loading & pass-through | `scripts/precompute/confident_picks.py` |
| BIP model | `src/models/game_sim/bip_model.py` |
| Constants | `src/utils/constants.py` |
| Batter H model | `src/models/game_sim/batter_h_model.py`, `scripts/fit_batter_h_model.py` |
| Sim APIs | `src/models/game_sim/lineup_simulator.py`, `src/models/game_sim/batter_simulator.py` |
| Correlated posteriors | `src/models/multinomial_logistic_normal.py`, `src/models/game_sim/_sim_utils.py` |
| Context object | `src/models/game_sim/pa_outcome_model.py` (`GameContext`) |
