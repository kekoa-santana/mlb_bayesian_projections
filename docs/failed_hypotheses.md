# Rejected Hypotheses, Failed Approaches, and Skip Decisions

A comprehensive record of every feature, covariate, projection target, and modeling approach that was tested, triaged, or abandoned during the development of the Data Diamond MLB Bayesian Projection System. This document exists so we do not re-test ideas that have already been evaluated, and so the reasoning behind each decision is preserved.

Last updated: 2026-03-13

---

## 1. Rejected Model Covariates

Features that were tested (or rigorously evaluated via EDA) and found to add zero or negative incremental value to the projection system.

### 1a. Pitcher K% Statcast covariates (whiff_rate, avg_velo)

- **Hypothesis:** Whiff rate (r=0.822 with K%) and/or avg_velo (r=0.336, YoY r=0.908) should improve pitcher K% projections, similar to how whiff_rate + chase_rate improve hitter K%.
- **Round 1 (random walk, wide priors):** At ANY level (observation, player-level prior), ANY parameterization (centered, non-centered), ANY prior width (sigma 0.05 to 0.30), estimating beta_whiff collapsed a variance component. Player-level covariates: sigma_player ESS of 37-86 (needs >400). Observation-level: sigma_season ESS of 33. Fixed-coefficient fallback (beta_whiff at 0.00-0.12) gave identical Brier scores.
- **Round 2 (AR(1), tight priors, 2026-03-13):** Retested under AR(1) model with tight priors (sigma=0.10-0.15) and reduced sigma_player_prior (0.4). Three configurations tested:
  - **whiff_rate + avg_velo** (sigma=0.15 each): avg Brier improvement -1.4%, ESS 53-138
  - **whiff_rate only** (sigma=0.10): avg Brier improvement -2.3%, ESS 91-127
  - **avg_velo only** (sigma=0.15): avg Brier improvement -2.7%, ESS 136-153
  - All three worse than no-covariate baseline AND worse than Marcel
- **Why it fails:** The hierarchical structure (age-bucket x skill-tier population means + partial pooling) already absorbs the whiff/velo signal through the observed K rate itself. Adding explicit covariates doesn't provide incremental information — it just competes with the variance components.
- **Decision:** Pitcher K% model uses NO Statcast covariates. The model's edge over Marcel comes from calibration (Brier scores), not point-estimate accuracy. This is a confirmed structural limitation, not a fixable bug.

### 1b. Pitcher first-strike% as covariate

- **Hypothesis:** Pitchers who get ahead in the count more often should have higher K rates. First-strike% might capture a "command/aggression" dimension independent of pure stuff.
- **EDA results:** Year-over-year r=0.52. Variance decomposition: 51% between-pitcher (half noise). Partial correlation with next-year K rate after controlling for current K rate: r=-0.07. **Zero incremental predictive power.**
- **Why it fails:** First-strike% is a blend of pitcher command (real skill) and batter aggressiveness (not pitcher-controlled). After accounting for the pitcher's whiff ability, knowing whether they get ahead tells you nothing additional about their K rate.
- **Decision:** Not worth modeling as a separate covariate.

### 1c. Pitcher putaway rate as covariate

- **Hypothesis:** Putaway rate (K% in two-strike counts) might capture a "finishing ability" dimension — some pitchers are better at converting two-strike counts into strikeouts.
- **EDA results:** Concurrent correlation with K rate: r=0.92. Year-over-year r=0.65. Totally collinear with K rate. Putaway rate is a mathematical decomposition of K rate, not an independent skill.
- **Decision:** Rejected as redundant. Would add multicollinearity without new information.

### 1d. First-pitch contact quality (hitters) as projectable metric

- **Hypothesis:** Hitters who make better contact on the first pitch might have a projectable skill advantage — they are "ready to hit" from pitch one.
- **EDA results:** Year-over-year r=0.15. Essentially random. First-pitch batted ball outcomes are dominated by BABIP variance and the small sample of first-pitch contact events per hitter per season (~50-80 BIP).
- **Decision:** Rejected. No stable signal exists. Cannot be projected.

### 1e. Two-strike approach delta (hitters) as covariate

- **Hypothesis:** The magnitude of a hitter's behavioral change with two strikes (whiff rate delta between early counts and 2-strike counts) might be a projectable skill informing K rate.
- **EDA results:** Year-over-year r=0.38. Moderate signal but very noisy. Average delta is ~-1pp (hitters actually whiff slightly LESS with 2 strikes due to protective swing), SD ~4pp.
- **Decision:** Not stable enough to project independently. Would need extremely heavy partial pooling. The existing model's random walk already absorbs this through the player's aggregate K rate.

### 1f. Hitter aggressiveness as quality indicator

- **Hypothesis:** More aggressive hitters (higher first-pitch swing rates) might produce better or worse outcomes. Aggressiveness style could predict offensive production.
- **EDA results:** Q4 aggressive hitters (42.6% FP swing) vs Q1 passive (21.1%) produce **identical wOBA: .328 vs .326**. K% rises slightly with aggressiveness (18.2% → 22.3%), BB% drops (9.5% → 7.8%). Aggressive hitters make better first-pitch contact (.414 vs .372 xwOBA on FP BIP) but that's selection bias — they swing at better pitches.
- **Decision:** Aggressiveness is value-neutral. It's an approach archetype (useful for content/scouting), not a quality indicator. No predictive value for offensive production.

### 1g. Observation-level vs player-level Statcast covariates (hitter model)

- **Hypothesis:** Moving barrel_z and hard_hit_z from observation-level covariates to player-level priors might improve the hitter K% model.
- **Backtest results:** Observation-level beat Marcel on 2/3 folds (+5.8%, +3.2%, -1.6%). Player-level degraded to (+5.1%, +2.3%, -2.1%) with convergence issues.
- **Decision:** Reverted to observation-level. Statcast covariates work better as season-level adjustments than as priors on latent talent.

---

## 2. Dropped Projection Targets

Stats we initially tried to project with the full Bayesian machinery but pulled back.

### 2a. Hitter xwOBA as Bayesian-projected stat

- **Year-over-year r:** 0.746 (decent stability).
- **Why dropped:** xwOBA is a derived outcome aggregating contact quality, walk rate, and power — all better captured by component stats. Projecting it separately introduced redundancy without improving the composite system.
- **Current use:** Observed stat in composite profile (batted-ball dimension) and informative prior. Not Bayesian-projected forward.

### 2b. Hitter HR/PA as Bayesian-projected stat

- **Year-over-year r:** 0.569 (unstable for a rate stat).
- **Why dropped:** Heavily influenced by BABIP-like variance on fly balls, park factors, and weather. Posteriors were too wide to be useful. Signal-to-noise ratio too low for meaningful Bayesian shrinkage.
- **Current use:** Observed stat and informative prior only. Power captured via barrel_rate, hard_hit_pct, avg_exit_velo.

### 2c. Pitcher HR/BF as Bayesian-projected stat

- **Year-over-year r:** 0.267 (essentially unprojectable at individual level).
- **Why dropped:** HR allowed is dominated by fly ball rate (somewhat projectable) interacting with HR/FB rate (largely random). Model could not find stable player-level signals.
- **Replacement:** GB% (r=0.619) substituted as batted-ball quality indicator. Far more stable, captures the same underlying skill.

### 2d. Stolen base projections (Bayesian approach)

- **Status:** Step 20, deferred.
- **Backtest results:** Bayes SB projections lost to Marcel by -2% to -12% on Brier score across all 4 folds.
- **Why it fails:** The era adjustment (SB_ERA_FACTOR_PRE_2023=1.8) for 2023 rule changes is too blunt — a single multiplicative factor applied uniformly cannot capture heterogeneous impact across player speed profiles.
- **Fix needed:** Cross-validated era factors, speed-dependent adjustments, or fall back to Marcel for SB entirely.

---

## 3. Skipped Approaches

Methods triaged as low incremental value before extensive testing, based on architectural reasoning.

### 3a. Aging curve delta

- **What:** Compute population aging curve, measure each player's deviation. Delta = "aging faster/slower than expected."
- **Why skipped:** The Gaussian random walk in the hierarchical model IS the aging adjustment. It captures year-to-year talent evolution directly from data with proper uncertainty. Computing a population curve to derive deltas feeds derived noise into a model that already has the raw information.

### 3b. Pitch sequencing entropy

- **What:** Shannon entropy of pitch-type transition probabilities. Low entropy = predictable pitcher.
- **Why skipped:** Thin evidence of season-level predictive power. Downstream effects of sequencing (whiff rates, chase rates) are already captured in pitch-type-level outcome data. Measuring an input when we already measure the outputs.

### 3c. Park/environment K normalization

- **What:** Adjust K rates for park effects like HR park factors.
- **Why skipped:** K park factors are extremely compressed (0.95-1.05 range). Signal too small relative to estimation noise. Only HR park factors integrated (range ~0.80-1.30).

### 3d. Player similarity embeddings

- **What:** Embed players in feature space, use nearest-neighbor comps for projections (a la PECOTA).
- **Why skipped:** Hierarchical model's partial pooling IS the similarity mechanism — players with small samples are pulled toward the population mean informed by all similar players. Explicit embeddings are redundant. Better suited for content (comp cards) than projections.

### 3e. Stabilized contact quality (rolling xwOBA windows)

- **What:** Rolling 200/400/800 BIP xwOBA windows with variance measurement.
- **Why deferred:** Multi-season random walk with partial pooling already handles stability through the data itself. Consistent performers get tighter posteriors automatically. Rolling windows add value for in-season updating, not pre-season projections.

### 3f. Plate discipline stability (chase rate deltas, rolling)

- **What:** Rolling chase rate and delta from baseline as leading indicators.
- **Why deferred:** Already computed per pitch type in vulnerability profiles. Season-level model doesn't need leading indicators. Real value is in-season updating.

### 3g. Velocity trend acceleration

- **What:** First and second derivatives of velocity as injury/decline indicators.
- **Why deferred:** For season-level projections, random walk handles year-to-year velocity shifts. Velocity trends are most valuable for in-season monitoring and injury risk (a separate system). Downstream effects already captured through observed K rates and whiff rates.

---

## 4. Known Limitations

### 4a. Pitcher Bayes model wins on calibration, not point estimates

The pitcher K% model wins on calibration (Brier: 0.171-0.209 vs Marcel's 0.205-0.251) and CRPS. For betting (Kelly criterion), calibrated probabilities matter more than point estimates. This is the expected outcome of a Bayesian approach.

### 4b. Batted-ball data coverage poor before 2022

`sat_batted_balls` coverage: 21% (2018) → 89% (2025). xwOBA/barrel unreliable before 2022. Partial pooling mitigates but doesn't eliminate the impact on pre-2022 skill tier assignments.

### 4c. IEEE NaN values in PostgreSQL

`sat_batted_balls.xwoba` stores IEEE NaN as float, not SQL NULL. Poisons `AVG()`. All queries must filter with `CASE WHEN xwoba != 'NaN'`. Same issue in `fact_pitch` for release_speed, pfx_x/z, spin_rate, plate_z.

### 4d. Pitcher-hitter asymmetry in count metrics

Hitter count management reflects stable true-talent approach (79-87% between-player variance). Pitcher count metrics are much noisier (51-64% between-player). Hitters control their approach; pitchers control outcomes much less. This means count-based features are far more useful on the hitter side than the pitcher side.

---

## 5. Key Principles Learned

### 5a. High concurrent correlation with the target is a red flag, not a green flag

When a covariate correlates r>0.70 with the target, it's likely a definitional proxy. Adding it to a hierarchical Bayesian model collapses variance components. Whiff rate (r=0.71 with K%) is the canonical example. **Moderate correlations (r=0.3-0.5) are more useful** because they carry independent information.

### 5b. The hierarchical model already does what many "advanced features" promise

Partial pooling = similarity mechanism. Random walk = aging adjustment. Multi-season shrinkage = stability measure. Before adding a feature, ask: "Does the model architecture already capture this?" Usually yes.

### 5c. Year-over-year correlation is the gatekeeper

YoY r < 0.50 → dominated by noise, extremely hard to project. Sweet spot for Bayesian advantage: r=0.50-0.80 (real signal + enough noise that shrinkage adds value). Projectable: K% (0.80), BB% (0.71), GB% (0.62). Not projectable: HR/BF (0.27), FP contact quality (0.15).

### 5d. Calibration edge > point-estimate edge for betting

A model that says "70% confident" and is right 70% of the time beats one that's right 72% but claims 85%. Design evaluation around the actual use case.

### 5e. Decomposing a rate stat into components doesn't create new information

Putaway rate is the last step of K rate. First-strike% feeds K rate through count progression. The model already sees the final outcome (K rate) and primary mechanism (whiff rate). Intermediate causal steps don't add incremental signal.

### 5f. Era/regime changes need player-type-specific adjustments

A single multiplicative era factor for the 2023 SB rule changes is too blunt. Fast players benefited differently from average players. Heterogeneous impacts require heterogeneous adjustments.

### 5g. Trust the backtest over the theory

Observation-level covariates seemed theoretically inferior to player-level priors but performed better empirically. Always run the walk-forward backtest before committing to a parameterization change.

---

## 6. Rejected Ranking/Rating Critiques (2026-03-23)

Suggestions evaluated during team ranking and Glicko review. Each was investigated and found to be either incorrect, already handled, or not worth the engineering cost.

### 6a. Glicko wOBA scaling cap is lossy

- **Critique:** The `min(woba_value / 2.0, 1.0)` cap in Glicko treats doubles and HRs the same.
- **Why rejected:** Wrong premise. `woba_value` is a per-PA linear weight (double ~1.24, HR ~2.0), so they score 0.62 and 1.00 respectively after the /2.0 scaling. The cap only clips values above 2.0, which is negligible information loss (essentially capping grand slams). The ordinal ranking of PA outcomes is fully preserved.

### 6b. Age-based Glicko volatility initialization

- **Critique:** Initialize Glicko sigma (phi) differently by age — young players should have higher uncertainty.
- **Why rejected:** Already handled by the system. New players enter with phi=350 (maximum uncertainty), which naturally decays as they accumulate rated games. A 22-year-old rookie starts at phi=350 and converges after ~20 games. An age-based heuristic would be strictly worse than this data-driven approach.

### 6c. Component double-counting in power rankings

- **Critique:** `tdd_value_score` appears in both the Projection component and the Profile component of power rankings, causing double-counting.
- **Why rejected:** Technically true but low impact. Each component is percentile-ranked independently across 30 teams, and the Profile component is ~60% non-tdd_value_score signals (Glicko ratings, observed ERA, K-rate, park-adjusted runs). The effective overweight is ~30% vs 20% intended — not worth restructuring the entire power ranking framework to fix.

### 6d. Arbitrary reliever role weights in bullpen scoring

- **Critique:** CL 1.5x, SU 1.0x, MR 0.5x weights are arbitrary — should use pLI (leverage index).
- **Why rejected:** Valid in principle, but bullpen is only 15% of total team ranking weight. We lack leverage index data in the current database. The marginal improvement from pLI-based weighting within a 15% weight component doesn't justify the engineering effort of sourcing and integrating leverage data.

### 6e. Clutch/high-leverage performance as a player ranking factor

- **Hypothesis:** Players who perform well in high-leverage situations (RISP, close games, late innings) have a repeatable skill that should be modeled. Similarly, teams that perform well in high-stakes games (pennant races, elimination scenarios) should get credit.
- **Why rejected:** Overwhelming research consensus (109 years of data) shows clutch hitting is not a repeatable skill.
  - YoY correlation of FanGraphs Clutch stat: **r = 0.06** (essentially zero). Source: FanGraphs Clutch Library.
  - WPA year-to-year R² = 0.27, but **all persistence is from "being a good hitter"**, not the clutch component. Source: FanGraphs "Is WPA Predictive for Batters?"
  - Team BA/RISP YoY correlation: **21%**. Regresses to overall BA. Source: SABR.
  - LOB% (pitcher strand rate) YoY R² = **0.048**. Driven by K rate, not a separate "pressure" skill. Source: FanGraphs LOB% Library, Hardball Times xLOB%.
  - One-run game record: R² = **0.02** between halves of season. Source: Baseball Prospectus.
  - 109-year study: no meaningful clutch ability detected; elite performers actually show *largest drop-off* in postseason. Source: ScienceDirect (2013).
  - SABR conclusion: "Clutch hitting is not predictable from year to year."
- **Exception noted:** High-K pitchers strand more runners because strikeouts are sequencing-independent — but our K% projections already capture this.
- **Decision:** Do not model clutch/pressure/leverage as a player or team ranking factor. It would add noise, not signal.

### 6f. Platoon advantage profiling for team rankings

- **Critique:** Team rankings should account for platoon advantages in the lineup.
- **Why rejected:** Platoon effects are already embedded in aggregate wOBA projections (the Bayesian model uses platoon splits) and in the Layer 2 matchup model. Adding explicit platoon profiling at the team level would be marginal incremental value on top of what's already captured in the player-level projections that feed the team aggregation.
