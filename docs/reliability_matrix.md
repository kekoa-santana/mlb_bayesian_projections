# Stat Reliability Matrix

_Last updated: 2026-04-17. Based on walk-forward backtesting (2018-2025 training, 2023-2025 test folds, 11,517 pitcher games, 131K+ batter games). All metrics produced by the **legacy** engine (`run_game_prop_backtest.py --engine legacy`)._

This document is the canonical source of truth for what this system does well and where it falls short. Every claim here is backed by validation artifacts in `outputs/`. When documentation or dashboard messaging conflicts with this matrix, the matrix wins.

---

## Game-Level Props

### Tier 1 -- Well-Calibrated (actionable)

| Prop | Brier | ECE | Calib Slope | Notes |
|------|-------|-----|-------------|-------|
| **Batter H** | 0.148 | 0.018 | 1.12 | Best-calibrated prop in the system. BABIP-dominated at game level but shrinkage captures signal. |
| **Batter K** | 0.147 | 0.037 | 1.22 | Hitter K% is r=0.795 YoY stable. Whiff-rate matchup scoring adds signal even with 3-5 PA/game. |
| **Batter BB** | 0.120 | 0.030 | 0.98 | Near-perfect calibration. Chase rate (r=0.84 YoY) drives the signal. Walks are rare (0-1/game) so binary outcomes are inherently noisy. |
| **Pitcher K** | 0.189 | 0.038 | 1.14 | ECE varies by fold (0.024-0.048). Well-calibrated at the distribution level; game-to-game correlation is only r=0.33. |

### Tier 2 -- Partially Calibrated (use with caution)

| Prop | Brier | ECE | Calib Slope | Notes |
|------|-------|-----|-------------|-------|
| **Pitcher BB** | 0.168 | 0.058 | 0.68-0.73 | ECE varies by fold (0.037-0.078). Slope < 1.0 at key lines (1.5, 2.5) means the model is overconfident (probability estimates too extreme). XGB BB adjustment helps (+5% RMSE) but bias remains. |
| **Pitcher H** | 0.199 | 0.049 | 0.15-0.30 | **Severely overconfident.** Slope << 1.0 across all lines. The model's probability estimates are far too narrow. Population-based approach, no Bayesian posterior. Needs fundamental rework. |

### Tier 3 -- Unreliable (do not action)

| Prop | Brier | ECE | Calib Slope | Notes |
|------|-------|-----|-------------|-------|
| **Pitcher HR** | 0.210 | 0.116 | 0.25-0.50 | Rare event (~3% of BF). HR/BF has r=0.267 YoY (weakest rate stat). Calibration is broken. |
| **Pitcher Outs** | 0.232 | 0.124 | 2.58 | Driven by manager decisions, not statistical patterns. Not meaningfully a rate stat. |
| **Batter HR** | -- | -- | -- | Too rare at game level (85% of batters hit 0 HR/game). Essentially a binary coin flip. |

---

## Season-Level Projections

### Rate Stats (Bayesian posterior quality)

| Stat | Side | YoY Stability | 80% Coverage | Brier | vs Marcel | Assessment |
|------|------|--------------|-------------|-------|-----------|------------|
| **K%** | Hitter | r=0.795 | 65.5% | 0.152 | Beats | Core strength. Most stable hitter rate. |
| **BB%** | Hitter | r=0.706 | 65.2% | 0.145 | Beats | Strong. Chase rate (r=0.84) is the best underlying signal in the system. |
| **GB%** | Hitter | r=0.72 | 67.6% | 0.176 | Beats | Solid. Batted ball profile is stable. |
| **K%** | Pitcher | r=0.74 | 64.1% | 0.208 | Beats | Reliable rate, but pitcher K counting stats are weak (see below). |
| **BB%** | Pitcher | r=0.55 | 73.4% | 0.190 | Beats | Wider posteriors correctly reflect higher variance. |
| **HR/FB** | Hitter | r=0.45 | 55.6% | 0.207 | Mixed | Under-covers at 80% CI. HR is inherently noisy. |
| **HR/BF** | Pitcher | r=0.267 | 65.2% | 0.246 | Mixed | Weakest rate stat. Park factors help but noise dominates. |

### Counting Stats (rate x playing time)

| Stat | Side | Bayes Corr | vs Marcel MAPE | Assessment |
|------|------|-----------|---------------|------------|
| **BB** | Hitter | 0.719 | -2.4% (beats) | Best counting stat projection. |
| **K** | Hitter | 0.611 | +3.0% (loses) | Reliable rate undermined by PA uncertainty. |
| **HR** | Hitter | 0.593 | -7.5% (beats) | Moderate. Park factors and barrel rate help. |
| **K** | Pitcher | 0.558 | +6.2% (loses) | Weak. Layer 1 K-rate is 2pp under-calibrated, causing -0.66 K/game bias. Known issue. |
| **BB** | Pitcher | 0.488 | +2.0% (loses) | Worst counting correlation. Walks are inherently unstable. |
| **Outs** | Pitcher | 0.548 | -6.7% (beats) | Workload-driven. Beats Marcel because BF model captures starter usage patterns. |
| **SB** | Hitter | 0.709 | +6.8% (loses) | Era adjustment too blunt. Bayes loses to Marcel. Known issue. |

---

## Underlying Signal Stability (YoY correlations)

These are the raw materials the models work with. Higher stability = more projectable.

| Signal | r (YoY) | Between-Player Var | Notes |
|--------|---------|-------------------|-------|
| **Chase rate** | 0.84 | 87% | Most stable count metric. Value-neutral (Q4 vs Q1 wOBA = .328 vs .326). |
| **Hitter K%** | 0.795 | -- | Core projection anchor. |
| **Hitter BB%** | 0.706 | -- | Strong. |
| **Pitcher K%** | 0.74 | -- | Stable rate, but counting stat projection is weak. |
| **GB%** | 0.72 | -- | Batted ball profile. |
| **Pitcher BB%** | 0.55 | -- | Moderate. More variance than hitter BB%. |
| **HR/FB** | 0.45 | -- | Noisy. |
| **HR/BF (pitcher)** | 0.267 | -- | Nearly unprojectable. Park factors are the best lever. |
| **BABIP** | 0.35 | -- | Mostly luck at season level. |

---

## Known Issues and Calibration Gaps

1. **Pitcher K-rate under-prediction:** Per-PA K rate ~0.205 vs actual 0.225 (2pp gap). Causes -0.66 K/game bias. Fix is upstream in Layer 1 (`pitcher_model.py`), not the simulator.

2. **Pitcher H overconfidence:** Calibration slope 0.15-0.30 across all lines. Model probability estimates are far too narrow. Needs fundamental rework of the H prediction pathway.

3. **Game-level BB bias:** +0.1 BB/game systematic bias and 0.5 calibration slope. Root cause is upstream of matchup slope.

4. **SB era adjustment:** Bayes loses to Marcel on stolen bases. The rule-change era adjustment is too blunt.

5. **Pitcher HR calibration:** Slope < 0.5 at all lines. Confidence intervals are too narrow for a rare, noisy event.

---

## What This Means for Research

The system's genuine strengths are:
- **Batter-side game props** (H, K, BB) -- all well-calibrated, ECE < 0.04
- **Season-level rate projections** -- Bayes beats Marcel on Brier/calibration for all core rates
- **Uncertainty quantification** -- the posterior distributions are the product, not point estimates
- **Chase rate as a projection anchor** -- r=0.84 YoY, most stable signal in baseball

The system's honest weaknesses are:
- **Pitcher counting stats** -- K, BB, Outs all lose to Marcel on MAPE
- **HR at any level** -- r=0.267 YoY makes this nearly unprojectable
- **Game-level pitcher H** -- calibration is broken (slope << 1.0)
- **Game-to-game correlation** -- even "good" props like pitcher K only correlate r=0.34 game-to-game
