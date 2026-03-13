# SWE Plan: MLB Projection System

## Current State (2026-03-13)

### Completed
- **AR(1) process:** Replaced random walk in both hitter and pitcher models. rho ~ Beta(8,2), mean-reverting season evolution.
- **Batted ball decomposition:** Hitter projections use GB%, FB%, HR/FB instead of HR/PA + wOBA.
- **Stat-specific covariates:** Hitter K% uses whiff_rate + chase_rate. Pitcher BB% uses zone_pct. Pitcher HR/BF uses gb_pct.
- **Counting stat framework:** Rate x playing time Monte Carlo. Beats Marcel on pitcher K (+13%), BB (+6%), Outs (+15%). Hitter K (+14-17%), BB (+9-17%), HR (+5-11%).
- **Game prop framework:** Full validation system with confidence tiers, CRPS, ECE, temperature scaling, Bayes-Marcel ensemble.
- **MiLB translations:** Translation factors built, integrated into ensemble.
- **Dashboard split:** `tdd-dashboard/` repo with precompute bridge.

### Confirmed Limitations
- **Pitcher K% vs Marcel on MAE:** Covariates (whiff_rate, avg_velo, both) tested under AR(1) with tight priors (2026-03-13). None improve MAE — partial pooling already absorbs the signal. Bayes wins on calibration (Brier), not point accuracy. See `docs/failed_hypotheses.md` §1a.

---

## Remaining Work

### Priority 2: SB Projections Fix
- Bayes currently loses to Marcel on SB (-2% to -12%)
- Root cause: era adjustment too blunt (stolen base rule changes 2023+)
- Possible fix: era-specific priors or separate pre/post-2023 models

### Priority 3: Position-Based Rankings
- Composite player value from projected components
- Position scarcity adjustments
- Fantasy and front-office ranking outputs

### Priority 4: Betting Edge Finder
- Kelly criterion bankroll management
- Automated edge detection from game prop posteriors
- ROI tracking system

### Priority 5: Fielding Metrics
- Statcast OAA via Baseball Savant leaderboard API
- Integrate into composite player value

---

## Disproven / Skipped Ideas
- **Pitcher K% covariates (whiff_rate, avg_velo):** Tested 3 configurations under AR(1) with tight priors. None improve MAE vs Marcel — hierarchical partial pooling already absorbs the signal.
- **barrel_rate_against for pitcher K%:** r=-0.109 with K%, essentially uncorrelated. Contact quality metric, not K predictor.
- **wOBA as projection target:** wOBA YoY correlation too low for direct Bayesian projection. Used as informative prior instead.
- **Pitch sequencing entropy:** Thin evidence at season level.
- **Player similarity embeddings:** Hierarchical model already borrows strength via partial pooling.
