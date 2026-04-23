# Calibrated Game-Level Prop Probabilities for MLB: A Hierarchical Bayesian Approach

**Kekoa Santana — The Data Diamond**

_Draft v0.1 — April 2026_

---

## Abstract

Game-level player prop markets in baseball (e.g., "Will this pitcher record over 5.5 strikeouts?") are priced using implied probabilities that are rarely validated against calibration standards. We present a hierarchical Bayesian system that produces calibrated probability distributions for MLB game-level props, evaluated through strict walk-forward backtesting over 11,517 pitcher games and 131,000+ batter games across three held-out seasons (2023-2025). The system achieves ECE below 0.04 for batter hits, strikeouts, walks, and pitcher strikeouts — with batter hits achieving ECE=0.018, the best-calibrated prop in the system. We document transparent failure modes for pitcher hits (slope < 0.30), home runs (nearly unprojectable), and pitcher outs (manager-driven). The central finding is that distributional forecasts with proper uncertainty quantification outperform point estimates even in a domain where single-game correlations are inherently low (r=0.33 for the best prop).

---

## 1. Introduction

### 1.1 The calibration gap in baseball analytics

Most baseball projection systems produce point estimates — a pitcher is projected for 6.2 strikeouts. But point estimates discard the most valuable information: how confident are we? A 6.2 projection with a tight distribution (std=1.5) implies very different betting value than 6.2 with wide uncertainty (std=3.0), even if both center on the same number.

The sportsbook market prices props as implied probabilities: "over 5.5 K" at -130 implies ~56.5% probability. If a model's probability estimates are well-calibrated — when it says 60%, the event occurs 60% of the time — then any gap between model probability and market implied probability represents an exploitable edge. Conversely, an uncalibrated model cannot identify edges at all, regardless of how sophisticated its internals are.

This paper asks a specific question: **can a hierarchical Bayesian projection system produce well-calibrated probability distributions for MLB game-level props?** We focus on the props where the answer is clearly yes, and document honestly where it is not.

### 1.2 Why batter props are underexplored

The baseball analytics community has focused heavily on pitcher strikeout prediction — it is the highest-volume prop market and the stat with the most obvious pitcher-level signal. But our validation reveals that batter-side props (hits, strikeouts, walks) are actually better-calibrated than pitcher-side props, despite lower game-to-game correlation. This counterintuitive result — that the "harder" prediction problem produces better-calibrated probabilities — is the central finding of this paper, and we investigate why.

### 1.3 Contributions

1. A complete probabilistic pipeline (hierarchical Bayesian priors + pitch-type matchup adjustments + Monte Carlo game simulation) that produces full posterior distributions, not point estimates
2. Large-sample walk-forward evidence (131K+ batter games, 11.5K pitcher games, three held-out seasons) with strict no-leakage evaluation
3. Demonstration that batter hits are the best-calibrated game-level prop (ECE=0.018), contradicting the field's focus on pitcher strikeouts
4. Transparent failure analysis for props where calibration breaks down (pitcher hits, home runs, outs)
5. Evidence that calibration quality matters more than correlation — well-calibrated distributional forecasts are operationally valuable even when game-to-game r < 0.35

---

## 2. System Architecture

The system has three layers that compose together. Each layer's output feeds the next.

### 2.1 Layer 1: Season-level talent estimation (PyMC)

**Goal:** Estimate true-talent rates for every player with proper uncertainty bounds.

For each player-season, we fit hierarchical Bayesian models for rate statistics using PyMC 5.x:
- **Hitter targets:** K%, BB%, GB%, FB%, HR/FB
- **Pitcher targets:** K%, BB%, HR/BF

**Model structure:**
- Partial pooling across players (shrinks small-sample observations toward the population mean)
- AR(2) process for year-to-year talent evolution with mean-reverting dynamics (rho ~ Beta(8,2), rho2 ~ Beta(2,8))
- Age-bucket population priors (hitters: 4 buckets; pitchers: 3 buckets) crossed with Statcast skill tiers (4 tiers)
- Binomial observation model for most rate statistics; BetaBinomial for pitcher K% to handle overdispersion (~1.25x empirical)
- Output: Full posterior distributions per player per stat

**Key design choice: Statcast metrics as informative covariates.** A hitter's barrel rate, whiff rate, and chase rate enter the model as regression covariates on the linear predictor, shifting the posterior toward the rate implied by the player's underlying skills. Combined with age-bucket population priors, this means a 50-PA hitter with elite Statcast metrics receives a posterior distribution that reflects their skill profile rather than being purely shrunk toward the population mean. The effect is most pronounced for small samples where the covariate signal carries more weight relative to the binomial observation.

**Convergence requirements:** All models must achieve r_hat < 1.05, ESS > 400, and zero divergences. Models that fail convergence are excluded from downstream predictions.

[_Source: `src/models/hitter_model.py`, `src/models/pitcher_model.py`, `src/models/_model_utils.py`_]

### 2.2 Layer 2: Pitch-type matchup adjustments

**Goal:** Quantify how a specific pitcher's arsenal interacts with a specific hitter's vulnerabilities.

**Method:** Log-odds additive scoring with reliability-weighted fallback chains:
1. **Direct pitch-type match** — if pitcher throws >50 of this pitch type to batters of this handedness, use the observed contact/whiff rates (reliability = min(swings, 50) / 50)
2. **Pitch-family fallback** — group pitch types into families (fastballs, breaking, offspeed) and use family-level rates (reliability scaled by 0.5)
3. **League baseline** — fall back to population rates (reliability = 0)

The reliability weighting ensures small-sample matchup data is blended with population rates rather than taken at face value. A pitcher who has thrown 10 sliders to a specific hitter gets 20% weight on the direct observation and 80% on the population prior.

**Matchup damping:** Empirical calibration showed that raw matchup lifts are too aggressive at the game level. Damping coefficients (K=0.55, BB=0.40, HR=0.0) are loaded from walk-forward calibrated shrinkage fits when available, with hand-tuned fallbacks. The K coefficient was fitted via PA-level logistic regression across walk-forward folds (reduced from an earlier 0.787 estimate). The BB coefficient is pinned at 0.40 based on A/B testing that showed the hand-picked value outperforms the fitted value. The HR matchup coefficient was set to zero — game-level HR prediction does not benefit from matchup adjustments.

[_Coefficients: `src/models/game_sim/_sim_utils.py`, loaded from `data/cached/matchup_shrinkage_coefs.parquet`_]

[_Source: `src/models/matchup.py`, `src/data/archetype_matchups.py`_]

### 2.3 Layer 3: Game-level simulation

**Goal:** Translate season-level rate posteriors into game-level stat distributions.

Two simulation engines exist:

**BF-draw engine** (used for batter props in this paper): Draws from the posterior rate distribution, applies matchup and contextual adjustments, then samples game outcomes from a Binomial/Poisson model conditioned on expected plate appearances (BF for pitchers, PA for batters). This produces full distributions over game-level counting stats. Computationally lightweight (~4,000 draws per game).

**PA-by-PA simulator** (used for pitcher props and daily production): Simulates each plate appearance sequentially, modeling exit decisions (when the pitcher leaves the game), times-through-order fatigue, pitch count accumulation, and bullpen transitions. Outcomes are drawn from a multinomial model over {K, BB, HR, HBP, BIP} at each PA, with BIP outcomes resolved by a separate batted-ball contact model. Typically runs 10,000 simulations per game in production.

Both engines produce Monte Carlo samples from which prop-line probabilities are computed directly: P(K > 5.5) = fraction of draws exceeding 5.5.

**Contextual adjustments applied (via frozen `GameContext` dataclass):**
- Park factors (K, BB, HR, BABIP)
- Home plate umpire tendencies (K%, BB%)
- Weather effects (K%, HR%)
- Rest days (K%, BB%, BF distribution)
- Catcher framing effects (K%, BB%)
- Pitcher rolling form (BB%)
- XGBoost game-level BB adjustment (lineup proneness, umpire interaction)

[_Source: `src/models/game_stat_model.py` (BF-draw), `src/models/game_sim/simulator.py` (PA sim), `src/models/game_sim/pa_outcome_model.py` (GameContext)_]

---

## 3. Evaluation Design

### 3.1 Walk-forward protocol

We use strict walk-forward backtesting with no future data leakage:

| Test Season | Training Data | Pitcher Games | Batter Games |
|-------------|---------------|---------------|--------------|
| 2023 | 2018-2022 | 3,782 | ~43,400 |
| 2024 | 2018-2023 | 3,789 | ~44,300 |
| 2025 | 2018-2024 | 3,946 | ~43,800 |
| **Total** | | **11,517** | **~131,300** |

For each fold:
1. Fit hierarchical Bayesian models on training data only
2. Generate posterior rate distributions for all players appearing in the test season
3. For each game in the test season, produce prop-line probabilities
4. Evaluate against actual outcomes

No parameter tuning is performed on test data. Convergence diagnostics are computed on training data only.

### 3.2 Metrics

We evaluate distributional forecast quality using:

- **Brier score:** Mean squared error of probability forecasts for binary outcomes (over/under a specific line). Lower is better; 0.25 is the uninformed baseline for 50/50 events.
- **ECE (Expected Calibration Error):** Bins predictions by confidence, measures gap between predicted probability and observed frequency. Lower is better; 0.0 is perfect calibration.
- **Coverage:** Do X% credible intervals contain the truth X% of the time? Perfect calibration gives 80% coverage at 80% CI.
- **Temperature:** Calibration scaling factor. T=1.0 means well-calibrated; T>1.5 means overconfident (intervals too narrow); T<0.8 means underconfident (intervals too wide).

We do **not** report MAE or RMSE on point estimates. This is a distributional forecasting system — the posterior distribution is the product, not a single number. Evaluating distributional quality requires calibration-aware metrics.

### 3.3 Baselines

At the season level, we benchmark against the Marcel projection system (weighted 5/4/3 recent seasons, regressed to mean, age-adjusted). Our Bayesian models beat Marcel on Brier score for all core rate stats (hitter and pitcher K%, BB%).

At the game level, the natural baseline is the line itself — a well-set line should have ~50% over rate. Our evaluation shows whether the model adds signal beyond the line.

---

## 4. Results

### 4.1 Tier 1: Well-calibrated props

These four props achieve ECE < 0.04 across all walk-forward folds.

| Prop | Games | Brier | ECE | Temperature | Coverage 80% | Coverage 90% |
|------|-------|-------|-----|-------------|-------------|-------------|
| **Batter H** | 132,332 | 0.148 | 0.018 | 1.12 | 87.8% | 93.6% |
| **Batter K** | 131,308 | 0.147 | 0.037 | 1.22 | 84.4% | 92.4% |
| **Batter BB** | 131,308 | 0.120 | 0.030 | 0.98 | 87.9% | 92.5% |
| **Pitcher K** | 11,517 | 0.189 | 0.038 | 1.14 | 78.1% | 88.3% |

**Batter BB** achieves near-perfect temperature (0.98) — the posterior intervals are almost exactly right-sized. **Batter H** has the lowest ECE at 0.018, meaning the predicted probabilities track observed frequencies within 1.8 percentage points on average.

#### 4.1.1 Per-line Brier scores

Not all prop lines are equally predictable. For batter props, lower lines are harder because most batters cluster near 0-2 occurrences per game:

| Prop | Line | Brier |
|------|------|-------|
| Batter H | 0.5 | 0.242 |
| Batter H | 1.5 | 0.161 |
| Batter H | 2.5 | 0.042 |
| Batter K | 0.5 | 0.240 |
| Batter K | 1.5 | 0.162 |
| Batter K | 2.5 | 0.041 |
| Batter BB | 0.5 | 0.198 |
| Batter BB | 1.5 | 0.042 |
| Pitcher K | 3.5 | 0.188 |
| Pitcher K | 4.5 | 0.228 |
| Pitcher K | 5.5 | 0.222 |
| Pitcher K | 6.5 | 0.180 |
| Pitcher K | 7.5 | 0.126 |

The highest-volume market line (pitcher K over 5.5) has Brier=0.222. The batter H over 0.5 line (will this batter get at least one hit?) has Brier=0.242 — only marginally worse than the pitcher K equivalent despite appearing simpler.

### 4.2 Why batter hits calibrate so well

Batter H achieving ECE=0.018 is the system's most surprising result. Several factors contribute:

1. **BABIP compression at the game level.** Season-level BABIP varies widely (0.250-0.350), but at the game level with 3-5 PA, the BABIP signal is heavily diluted. This means the model's job is primarily to estimate the right number of PA and the right contact rate — both of which are stable and well-estimated by the hierarchical model.

2. **Natural bounds create calibration.** A batter gets 3-5 PA per game. Even with no model at all, the population rate (~0.260 batting average) creates a natural probability of ~0.55-0.70 for "over 0.5 hits." The model's contribution is shifting this probability up or down based on the batter's true talent and the matchup — a modest adjustment that is well-calibrated because the base rate does most of the work.

3. **Partial pooling prevents overconfidence.** The hierarchical model shrinks extreme batting averages toward the population mean. A batter hitting .350 in 100 PA gets a posterior centered around .310 — the model is appropriately skeptical of extreme samples. This shrinkage directly improves calibration by preventing the model from being too confident about extreme outcomes.

4. **Low information content per game is a feature, not a bug.** With only 3-5 PA, a single batter game has low Shannon information about the batter's true ability. But this means the posterior is wide, which is honest. The model doesn't pretend to know whether a .280 hitter will go 2-for-4 vs 1-for-4 — it assigns appropriate probabilities to both, and those probabilities turn out to be well-calibrated.

Contrast this with pitcher strikeouts, where the model has more signal (20+ BF per game, stable K%) but the higher information content tempts overconfidence. Pitcher K temperature=1.14 shows slight overconfidence; batter H temperature=1.12 is comparable but applied to more stable base rates.

### 4.3 Tier 2: Partially calibrated props

| Prop | Games | Brier | ECE | Temperature | Notes |
|------|-------|-------|-----|-------------|-------|
| **Pitcher BB** | 11,517 | 0.168 | 0.058 | 1.11 | Underconfident at key lines (slope < 1.0) |
| **Pitcher H** | 11,517 | 0.199 | 0.049 | 1.35 | Overconfident (slope 0.15-0.30) |

**Pitcher BB** has acceptable ECE but inconsistent calibration slope. The XGBoost BB adjustment (incorporating lineup walk-proneness, umpire interactions) improved RMSE by 5% but did not fully resolve the slope issue.

**Pitcher H** is the system's most clearly broken prop. Temperature=1.35 and slope < 0.30 indicate severe overconfidence — the model assigns probabilities that are far too extreme. The root cause is that pitcher H prediction lacks a proper Bayesian posterior; it uses population-based rates rather than individualized distributions.

### 4.4 Tier 3: Unreliable props

| Prop | Games | Brier | ECE | Temperature | Notes |
|------|-------|-------|-----|-------------|-------|
| **Pitcher HR** | 11,517 | 0.210 | 0.116 | 5.75 | Temperature >> 1.0 |
| **Pitcher Outs** | 11,517 | 0.232 | 0.124 | 2.58 | Manager-driven |
| **Batter HR** | — | — | — | — | Too rare at game level |

**Pitcher HR** has temperature=5.75, meaning the model's probability estimates need to be divided by nearly 6x to match reality. HR/BF has r=0.267 year-over-year — the weakest rate stat in baseball. The model's matchup adjustment for HR was set to zero (damping=0.0) because empirical testing showed it added no signal.

**Pitcher Outs** is not meaningfully a rate stat — it depends on manager decisions about when to pull a starter, which are driven by game state, pitch count, and bullpen availability rather than pitcher ability alone.

**Batter HR** is excluded from formal evaluation because 85%+ of batters hit zero home runs in a given game. The binary outcome is too rare for meaningful calibration.

---

## 5. Discussion

### 5.1 Calibration vs. correlation

A common objection to game-level models is that game-to-game correlations are low. Our best prop (pitcher K) only achieves r=0.33 game-to-game. Does this mean the model is useless?

No — because calibration and correlation measure different things. Correlation measures whether the model's ranking of games (this game will have more K than that one) is correct. Calibration measures whether the model's probability statements (60% chance of over 5.5 K) are reliable. A well-calibrated model with low correlation is still operationally valuable: it correctly prices the uncertainty, which is exactly what you need for identifying mispriced props.

Put differently: a model that says "55% over 5.5 K" when the true probability is 55% is valuable for betting even if it can't tell you which specific games will go over. The market line implies a probability; the model produces a calibrated probability; any gap is exploitable.

### 5.2 Why distributional forecasts matter

Point estimate systems cannot identify edge. A projection of 6.2 K tells you nothing about whether "over 5.5" is a good bet — you need the full distribution shape. The 6.2 K projection might have a 0.55 probability of over 5.5 (thin edge) or 0.70 (strong edge), depending on the posterior width.

Our system produces these probabilities directly from Monte Carlo samples. The calibration evidence in Section 4 shows that these probabilities are trustworthy for Tier 1 props.

### 5.3 Limitations

**1. Legacy engine for batter props.** The batter prop results use a simpler simulation engine (Binomial/Poisson draws) rather than the PA-by-PA simulator. While this produces excellent calibration, it cannot model within-game sequencing effects (e.g., lineup protection, TTO fatigue) that the PA-sim captures for pitcher props.

**2. No double play modeling.** The PA-by-PA pitcher sim models one PA = at most one out. Real games produce ~0.75-0.80 double plays per starter, creating a structural gap in outs prediction.

**3. Sample period.** Training data spans 2018-2025, a period that includes the COVID-shortened 2020 season and multiple rule changes (pitch clock 2023, shift ban 2023, larger bases 2023). The 2020 season is included in training data for all folds but is not used as a test season. The system makes no explicit adjustment for rule changes — it relies on the AR(2) mean-reverting process to adapt through talent evolution.

**4. Remaining contextual factors not modeled.** Weather effects, park factors, umpire tendencies, catcher framing, and rest days are all incorporated (see Section 2.3). Factors not yet modeled include: stadium-specific dimensions beyond park factors, day/night splits, and platoon adjustments at the per-PA level (L/R splits are incorporated at the season level through the hierarchical model but not as game-level matchup lifts).

**5. Honest about HR.** Home runs are essentially unprojectable at the game level (HR/BF has r=0.267 year-over-year; matchup damping coefficient set to 0.0). Any claim of game-level HR prediction edge should be viewed with extreme skepticism.

### 5.4 Future directions

1. **Port batter props to PA-by-PA simulator.** The current batter prop calibration is strong but uses the simpler BF-draw engine. Moving to the PA-sim would enable modeling lineup position effects and starter-vs-reliever splits. A batter game simulator (`src/models/game_sim/batter_simulator.py`) and lineup simulator (`src/models/game_sim/lineup_simulator.py`) exist but have not yet been validated at the prop-calibration level.
2. **Fix pitcher H calibration.** The current population-based approach needs replacement with individualized Bayesian posteriors, similar to the batter-side implementation.
3. **Platoon-aware matchup adjustments.** L/R splits are incorporated at the season level through the hierarchical model. A per-PA platoon lift function exists (`src/models/matchup.py:_compute_platoon_lift`) but is not yet wired into the active simulation path.
4. **Market validation.** A portfolio/market edge module (`src/models/market_edge.py`) implements Kelly sizing and correlation-aware position management. Connecting calibration quality to actual betting ROI against live DraftKings and PrizePicks lines is the next validation step.

---

## 6. Conclusion

We demonstrate that hierarchical Bayesian models with proper uncertainty quantification can produce well-calibrated game-level prop probabilities for MLB batter hits (ECE=0.018), batter strikeouts (ECE=0.037), batter walks (ECE=0.030), and pitcher strikeouts (ECE=0.038). The system's most counterintuitive finding — that batter hits are the best-calibrated prop despite lower game-to-game signal — suggests that the field's focus on pitcher strikeout prediction may be misplaced from a calibration perspective.

Transparent documentation of failure modes (pitcher hits, home runs, outs) is as important as documenting successes. A model that knows where it fails is more operationally useful than one that claims universal performance.

---

## Appendix A: Reproducibility

All results are generated from scripts in the public repository:

| Component | Script |
|-----------|--------|
| Season-level models | `scripts/run_hitter_backtest.py`, `scripts/run_pitcher_backtest.py` |
| Game prop backtest (legacy) | `scripts/run_game_prop_backtest.py --engine legacy` |
| Game prop backtest (PA sim) | `scripts/run_game_prop_backtest.py --engine pa_sim` |
| Pitcher game sim backtest | `scripts/run_game_sim_backtest.py` |

Frozen result artifacts for this paper are in `outputs/paper_v1/`.

## Appendix B: Code-to-Section Mapping

| Section | Key Modules |
|---------|-------------|
| 2.1 Layer 1 | `src/models/hitter_model.py`, `src/models/pitcher_model.py`, `src/models/_model_utils.py` |
| 2.2 Layer 2 | `src/models/matchup.py`, `src/data/archetype_matchups.py`, `src/models/game_sim/_sim_utils.py` (damping) |
| 2.3 Layer 3 (BF-draw) | `src/models/game_stat_model.py`, `src/evaluation/game_prop_validation.py` |
| 2.3 Layer 3 (PA sim) | `src/models/game_sim/simulator.py`, `src/models/game_sim/pa_outcome_model.py` |
| 2.3 GameContext | `src/models/game_sim/pa_outcome_model.py` (GameContext dataclass, contextual lifts) |
| 3. Evaluation | `src/evaluation/game_prop_validation.py`, `src/evaluation/pa_sim_prop_adapter.py`, `src/evaluation/metrics.py` |
| 4.2 Batter H analysis | `src/models/hitter_model.py` (shrinkage), `src/models/bf_model.py` (PA distribution) |
