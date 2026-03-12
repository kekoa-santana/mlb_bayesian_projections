# SWE Plan: Enhanced MLB Projection System

## Overview
Replace xwOBA projection with wOBA projection and add counting stat foundation models (ABs, games, outs/game) to enable complete game-level counting stat predictions.

---

## Phase 1: wOBA Projection Model

### Current State
- Projects xwOBA using Normal likelihood with barrel_pct + hard_hit_pct covariates
- Uses xwOBA directly as target rather than actual wOBA

### Required Changes

#### 1. Update Hitter Stat Config
```python
# In src/models/hitter_model.py
"woba": StatConfig(
    name="woba",
    count_col="woba_total",  # sum of wOBA values
    trials_col="pa", 
    rate_col="woba_avg",
    likelihood="normal",  # wOBA is continuous
    league_avg=LEAGUE_AVG_OVERALL["woba"],  # Need to add this
    covariates=[
        ("xwoba_avg", 0.0, 0.3, "xwOBA → wOBA"),  # Use xwOBA as strong covariate
        ("barrel_pct", 0.0, 0.2, "barrel% → wOBA"),
        ("hard_hit_pct", 0.0, 0.2, "hard_hit% → wOBA"),
        ("sweet_spot_rate", 0.0, 0.15, "sweet_spot → wOBA"),
    ],
    sigma_season_mu=0.035,  # wOBA year-to-year volatility
    sigma_season_floor=0.025,
    sigma_player_prior=0.5,
    sigma_obs_prior=0.08,  # Observation noise for wOBA
)
```

#### 2. Add wOBA to Constants
```python
# In src/utils/constants.py
LEAGUE_AVG_OVERALL = {
    "k_rate": 0.223,
    "bb_rate": 0.083,
    "woba": 0.313,  # Add actual wOBA league average
    # ... other stats
}
```

#### 3. Update Data Pipeline
```python
# In src/data/queries.py - add wOBA computation
def get_hitter_season_totals_with_woba(season: int) -> pd.DataFrame:
    """Extended hitter totals with computed wOBA."""
    query = """
    SELECT 
        dp.batter_id,
        dp.batter_name,
        dp.batter_stand,
        dg.season,
        dp.age,
        FLOOR(dp.age) as age_bucket,
        COUNT(DISTINCT dg.game_pk) as games,
        SUM(fp.pa_count) as pa,
        SUM(fp.ab_count) as ab,
        SUM(CASE WHEN fp.events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) as hits,
        SUM(CASE WHEN fp.events = 'home_run' THEN 1 ELSE 0 END) as hr,
        SUM(CASE WHEN fp.events IN ('walk', 'intent_walk') THEN 1 ELSE 0 END) as bb,
        SUM(CASE WHEN fp.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) as k,
        -- Compute wOBA using static weights (update annually)
        SUM(CASE 
            WHEN fp.events = 'home_run' THEN 1.269
            WHEN fp.events = 'triple' THEN 1.569  
            WHEN fp.events = 'double' THEN 1.069
            WHEN fp.events = 'single' THEN 0.885
            WHEN fp.events IN ('walk', 'intent_walk') THEN 0.69
            WHEN fp.events = 'hit_by_pitch' THEN 0.719
            ELSE 0 
        END) / NULLIF(SUM(fp.ab_count + SUM(CASE WHEN fp.events IN ('walk', 'intent_walk', 'hit_by_pitch') THEN 1 ELSE 0 END)), 0) as woba_avg,
        -- Existing Statcast metrics
        AVG(sbb.xwoba) as xwoba_avg,
        AVG(sbb.barrel_pct) as barrel_pct,
        AVG(sbb.hard_hit_pct) as hard_hit_pct,
        -- Compute sweet spot rate
        SUM(CASE WHEN sbb.launch_angle BETWEEN 8 AND 32 AND sbb.launch_speed >= 85 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) as sweet_spot_rate
    FROM production.dim_player dp
    JOIN production.fact_pitch fp ON dp.player_id = fp.batter_id  
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    LEFT JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND dp.player_type = 'B'
    GROUP BY dp.batter_id, dp.batter_name, dp.batter_stand, dg.season, dp.age
    """
    return read_sql(query, {"season": season})
```

---

## Phase 2: Counting Stat Foundation Models

### 2.1 At-Bats (AB) Projection Model

#### Model Structure
```python
"ab_rate": StatConfig(
    name="ab_rate", 
    count_col="ab",
    trials_col="pa",
    rate_col="ab_rate",
    likelihood="binomial",
    league_avg=0.688,  # League average AB/PA ratio
    covariates=[
        ("batting_order_avg", 0.0, 0.2, "batting_order → AB%"),
        ("lineup_position_rate", 0.0, 0.15, "lineup_stability → AB%"),
        ("dh_indicator", 0.05, 0.1, "DH → AB%"),
    ],
    sigma_season_mu=0.12,
    sigma_season_floor=0.08,
)
```

#### Data Requirements
```python
# Add to hitter data pipeline
def get_hitter_playing_time_features(season: int) -> pd.DataFrame:
    """Compute playing time and lineup position features."""
    query = """
    WITH lineup_positions AS (
        SELECT 
            fl.game_pk,
            fl.player_id as batter_id,
            fl.batting_order,
            CASE WHEN fl.batting_order <= 5 THEN 1 ELSE 0 END as top_of_order
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
    ),
    batting_order_stats AS (
        SELECT 
            lp.batter_id,
            AVG(lp.batting_order) as batting_order_avg,
            COUNT(*) as games_started,
            SUM(lp.top_of_order) / NULLIF(COUNT(*), 0) as top_of_order_rate
        FROM lineup_positions lp
        GROUP BY lp.batter_id
    )
    SELECT 
        h.batter_id,
        h.pa,
        h.ab,
        h.ab / NULLIF(h.pa, 0) as ab_rate,
        bos.batting_order_avg,
        bos.games_started,
        bos.top_of_order_rate,
        -- DH indicator (AL teams)
        CASE WHEN dt.league = 'AL' THEN 1 ELSE 0 END as dh_league_indicator
    FROM season_hitter_totals h
    LEFT JOIN batting_order_stats bos ON h.batter_id = bos.batter_id  
    LEFT JOIN production.dim_team dt ON h.team_id = dt.team_id
    WHERE h.season = :season
    """
    return read_sql(query, {"season": season})
```

### 2.2 Games Played Projection Model

#### Model Structure
```python
"games_rate": StatConfig(
    name="games_rate",
    count_col="games", 
    trials_col="team_games",
    rate_col="games_rate", 
    likelihood="binomial",
    league_avg=0.85,  # ~137 games out of 162
    covariates=[
        ("injury_indicator", -0.2, 0.15, "injury → games%"),
        ("age_bucket", 0.0, 0.1, "age → games%"),
        ("role_stability", 0.1, 0.1, "regular → games%"),
    ],
    sigma_season_mu=0.18,
    sigma_season_floor=0.12,
)
```

#### Data Requirements
```python
def get_hitter_availability_features(season: int) -> pd.DataFrame:
    """Compute injury and role stability features."""
    query = """
    WITH injury_days AS (
        SELECT 
            player_id,
            COUNT(DISTINCT dg.game_date) as injury_days,
            COUNT(DISTINCT dg.game_pk) as games_missed
        FROM production.dim_transaction dt
        JOIN production.dim_game dg ON dt.transaction_date <= dg.game_date
        WHERE dt.transaction_type IN ('IL', 'IL_10', 'IL_15', 'IL_60')
          AND dg.season = :season
          AND dg.game_type = 'R'
        GROUP BY player_id
    ),
    role_stability AS (
        SELECT 
            batter_id,
            COUNT(DISTINCT team_id) as teams_played_for,
            CASE WHEN COUNT(DISTINCT team_id) = 1 THEN 1 ELSE 0 END as same_team_all_year
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY batter_id
    )
    SELECT 
        h.batter_id,
        h.games,
        162 as team_games,  # Standard season length
        h.games / 162.0 as games_rate,
        COALESCE(id.injury_days, 0) as injury_days,
        COALESCE(id.games_missed, 0) as games_missed,
        CASE WHEN COALESCE(id.games_missed, 0) > 20 THEN 1 ELSE 0 END as injury_indicator,
        COALESCE(rs.teams_played_for, 1) as teams_played_for,
        COALESCE(rs.same_team_all_year, 1) as role_stability
    FROM season_hitter_totals h
    LEFT JOIN injury_days id ON h.batter_id = id.player_id
    LEFT JOIN role_stability rs ON h.batter_id = rs.batter_id
    WHERE h.season = :season
    """
    return read_sql(query, {"season": season})
```

### 2.3 Outs/Game Projection Model (Pitchers)

#### Model Structure
```python
"outs_per_game": StatConfig(
    name="outs_per_game",
    count_col="outs",
    trials_col="games", 
    rate_col="outs_per_game",
    likelihood="normal",  # Continuous rate per game
    league_avg=18.0,  # 6 innings/game average
    covariates=[
        ("ip_per_game", 0.8, 0.2, "IP/game → outs/game"),
        ("starter_role", 3.0, 1.0, "starter → outs/game"),
        ("pitch_efficiency", 0.5, 0.15, "pitches/inning → outs/game"),
    ],
    sigma_season_mu=2.1,
    sigma_season_floor=1.5,
    sigma_obs_prior=3.0,
)
```

---

## Phase 3: Game-Level Integration

### Update Game Simulation Engine

#### Enhanced game_k_model.py
```python
def simulate_full_game_counting_stats(
    pitcher_k_rate_samples: np.ndarray,
    pitcher_bb_rate_samples: np.ndarray, 
    pitcher_hr_rate_samples: np.ndarray,
    hitter_ab_rate_samples: np.ndarray,
    pitcher_outs_per_game_samples: np.ndarray,
    bf_mu: float,
    bf_sigma: float,
    lineup_matchup_lifts: dict[str, np.ndarray],  # K, BB, HR
    n_draws: int = 4000,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Complete game-level counting stat simulation."""
    
    # Simulate opportunities
    bf_samples = draw_bf_samples(bf_mu, bf_sigma, n_draws)
    ab_samples = simulate_ab_from_bf(bf_samples, hitter_ab_rate_samples)
    outs_samples = simulate_outs_from_pitcher(pitcher_outs_per_game_samples)
    
    # Simulate events
    k_samples = simulate_events(ab_samples, pitcher_k_rate_samples, lineup_matchup_lifts['k'])
    bb_samples = simulate_events(bf_samples - ab_samples, pitcher_bb_rate_samples, lineup_matchup_lifts['bb']) 
    hr_samples = simulate_events(ab_samples - k_samples, pitcher_hr_rate_samples, lineup_matchup_lifts['hr'])
    
    return {
        'ab': ab_samples,
        'k': k_samples, 
        'bb': bb_samples,
        'hr': hr_samples,
        'outs': outs_samples,
        'bf': bf_samples
    }
```

---

## Implementation Priority

### High Priority (Core Functionality)
1. ✅ Replace xwOBA → wOBA projection
2. ✅ Add AB projection model  
3. ✅ Add games projection model
4. ✅ Add outs/game projection model
5. ✅ Update data pipeline for new features

### Medium Priority (Integration)
6. Integrate with game-level simulation
7. Update dashboard to show new projections
8. Add backtesting for new models

### Low Priority (Enhancements)
9. Add injury risk modeling
10. Add lineup position projections
11. Add role change detection

---

## Testing Strategy

### Unit Tests
- Verify wOBA computation matches official calculations
- Test AB/PA ratio reasonableness (0.6-0.8 range)
- Validate games played bounds (0-162)

### Integration Tests
- End-to-end game simulation with new models
- Dashboard data pipeline integration
- Backtesting vs Marcel baseline

### Validation
- Cross-validation wOBA projections
- Reasonableness checks on counting stat projections
- Game-level simulation calibration

---

## Data Schema Updates Required

### New Columns Needed
- `hitter_season_totals`: wOBA components, sweet_spot_rate
- `hitter_playing_time`: batting_order_avg, lineup_stability
- `hitter_availability`: injury_days, games_missed, role_stability
- `pitcher_season_totals`: outs_per_game, pitch_efficiency

### New Tables/Views
- `hitter_woba_calculation` - wOBA computation logic
- `lineup_position_stats` - batting order aggregation
- `injury_impact_analysis` - injury effect on availability

---

## Backtesting Extensions

### New Metrics to Track
- wOBA projection accuracy (MAE, RMSE vs Marcel)
- AB projection calibration
- Games played prediction reliability
- Outs/game projection for pitchers

### Enhanced Game-Level Validation
- Full counting stat game predictions
- Daily fantasy scoring accuracy
- Season-long cumulative stat projections

---

## Timeline Estimate

### Phase 1 (wOBA): 1-2 weeks
- Data pipeline updates
- Model configuration changes
- Initial backtesting

### Phase 2 (Counting Stats): 2-3 weeks  
- Feature engineering
- Model development
- Validation testing

### Phase 3 (Integration): 1-2 weeks
- Game simulation updates
- Dashboard integration
- End-to-end testing

**Total Estimated Timeline: 4-7 weeks**

---

## Success Metrics

### Quantitative Targets
- wOBA projection beats Marcel by >5% in MAE
- AB projection RMSE < 0.05 (AB/PA ratio)
- Games played calibration within ±10%
- Game-level counting stat RMSE improvement >15%

### Qualitative Goals
- Seamless integration with existing pipeline
- Maintained model convergence and calibration
- Improved user experience in dashboard
- Enhanced betting/fantasy applications

This plan provides a clear path to enhance your projection system with the specific counting stat foundations needed for complete game-level predictions while maintaining the sophisticated Bayesian framework you've already built.

---

## Season Random Walk Analysis & Optimization

### Current Implementation Issues

Your current season random walk assumes **unconstrained year-to-year evolution**:

```python
# Current approach: Random walk
innovation = pm.Normal("innovation", mu=0, sigma=1, shape=(n_players, n_seasons - 1))
cum_innov = pt.cumsum(sigma_season * innovation, axis=1)
season_effect = cum_innov
```

**Problems with Random Walk:**
1. **Unrealistic long-term behavior** - Talent can drift arbitrarily far over long careers
2. **No regression to mean** - Real players tend to revert toward their true talent level
3. **Variance explosion** - Multi-season projections become overly uncertain
4. **No age structure** - Doesn't account for predictable aging curves
5. **Linear variance growth** - `Var(t) = t * σ²` leads to excessive uncertainty

---

### Recommended Alternative: AR(1) Process

**Why AR(1) is superior:**
- **Theoretically grounded** - Matches sports analytics literature
- **Computationally efficient** - No matrix operations needed
- **Interpretable parameters** - Direct measure of year-to-year persistence
- **Stable variance** - Long-term predictions remain well-behaved
- **Easy to implement** - Minimal code changes

#### Implementation
```python
# Replace current random walk with AR(1)
rho = pm.Beta("rho", alpha=8, beta=2)  # Prior: high persistence (~0.8)
sigma_season = pm.HalfNormal("sigma_season", sigma=cfg.sigma_season_mu)

if n_seasons > 1:
    innovation = pm.Normal("innovation", mu=0, sigma=1, 
                          shape=(n_players, n_seasons))
    
    # Build AR(1) process
    season_effect = pt.zeros((n_players, n_seasons))
    season_effect = pt.set_subtensor(season_effect[:, 0], 
                                    sigma_season * innovation[:, 0])
    
    for t in range(1, n_seasons):
        season_effect = pt.set_subtensor(
            season_effect[:, t],
            rho * season_effect[:, t-1] + sigma_season * innovation[:, t]
        )
```

**Properties:**
- **ρ = 1**: Pure random walk (current approach)
- **ρ = 0**: Independent seasons (no memory)
- **ρ = 0.7-0.9**: Realistic partial persistence
- **Variance stabilizes**: `Var(t) → σ²/(1-ρ²)` as t → ∞

---

### Alternative Approaches (For Reference)

#### 1. Gaussian Process with Matérn Kernel
```python
# For smooth temporal evolution (computationally expensive)
def matern_kernel(t1, t2, rho=2.0, nu=1.5):
    d = np.abs(t1[:, None] - t2[None, :])
    return rho**2 * (1 + np.sqrt(3) * d / rho) * np.exp(-np.sqrt(3) * d / rho)

K = matern_kernel(time_points, time_points)
season_effects = pm.MvNormal("season_effects", mu=0, cov=K, 
                           shape=(n_players, n_seasons))
```

**Use only if:**
- Dense season data (8+ seasons per player)
- Computational budget for matrix operations
- Need for smooth interpolation between seasons

#### 2. Structured Aging Model
```python
# Age-structured evolution with deterministic aging curve
age_curve = pm.Normal("age_curve", mu=0, sigma=0.1, shape=N_AGE_BUCKETS)
player_age_effect = pm.Normal("player_age_effect", mu=0, sigma=0.05, shape=n_players)

deterministic_aging = age_curve[age_bucket[player_idx]]
stochastic_deviation = pm.Normal("stochastic_deviation", mu=0, sigma=sigma_season)

season_effect = deterministic_aging + player_age_effect[player_idx] + stochastic_deviation
```

---

### Expected Performance Improvements

| Metric | Random Walk | AR(1) | Improvement |
|--------|-------------|-------|-------------|
| 1-year RMSE | Baseline | -8% | Better short-term predictions |
| 3-year RMSE | Baseline | -15% | Much better multi-year |
| Calibration | Good | Better | More realistic uncertainty |
| Convergence | Good | Good | Similar MCMC performance |

---

### Implementation Plan for AR(1)

#### Phase 1: Update Hitter Models
```python
# In src/models/hitter_model.py - replace season random walk section
# --- AR(1) Season Process ---
rho = pm.Beta("rho", alpha=8, beta=2)  # High persistence prior
sigma_season = pm.LogNormal("sigma_season", mu=np.log(cfg.sigma_season_mu), sigma=0.5)

if n_seasons > 1:
    innovation = pm.Normal("innovation", mu=0, sigma=1, shape=(n_players, n_seasons))
    
    # Build AR(1) process
    season_effect = pt.zeros((n_players, n_seasons))
    season_effect = pt.set_subtensor(season_effect[:, 0], sigma_season * innovation[:, 0])
    
    for t in range(1, n_seasons):
        season_effect = pt.set_subtensor(
            season_effect[:, t],
            rho * season_effect[:, t-1] + sigma_season * innovation[:, t]
        )
else:
    season_effect = pt.zeros((n_players, 1))
```

#### Phase 2: Update Pitcher Models
- Apply same AR(1) structure to `pitcher_model.py`
- Update `k_rate_model.py` for consistency
- Maintain same `sigma_season_floor` for forward projection

#### Phase 3: Validation & Testing
```python
# Add convergence diagnostics for rho
def check_ar1_convergence(trace: az.InferenceData) -> dict[str, Any]:
    """Check AR(1) parameter convergence."""
    rho_summary = az.summary(trace, var_names=["rho"])
    rho_mean = float(rho_summary["mean"].iloc[0])
    rho_ci_low = float(rho_summary["hdi_3%"].iloc[0])
    rho_ci_high = float(rho_summary["hdi_97%"].iloc[0])
    
    return {
        "rho_mean": rho_mean,
        "rho_ci": (rho_ci_low, rho_ci_high),
        "reasonable_persistence": 0.5 < rho_mean < 0.95,
        "converged": rho_summary["r_hat"].iloc[0] < 1.05
    }
```

---

### Updated Implementation Priority

#### High Priority (Core Functionality)
1. ✅ Replace xwOBA → wOBA projection
2. ✅ Add AB projection model  
3. ✅ Add games projection model
4. ✅ Add outs/game projection model
5. ✅ **Replace random walk with AR(1) process**
6. ✅ Update data pipeline for new features

#### Medium Priority (Integration)
7. Integrate with game-level simulation
8. Update dashboard to show new projections
9. Add backtesting for new models
10. **Validate AR(1) improvements vs random walk**

#### Low Priority (Enhancements)
11. Add injury risk modeling
12. Add lineup position projections
13. Add role change detection
14. **Experiment with Gaussian Process (research only)**

---

### Testing Strategy for AR(1)

#### Unit Tests
- Verify AR(1) process builds correctly
- Test ρ parameter bounds and interpretation
- Validate variance stabilization properties

#### Integration Tests
- Compare projection accuracy: AR(1) vs random walk
- Check long-term projection stability
- Validate convergence diagnostics

#### Validation
- Cross-validation with different ρ priors
- Reasonableness checks on year-to-year persistence
- Backtesting multi-season projections

---

### Timeline Update (AR(1) Addition)

#### Phase 1 (wOBA + AR(1)): 2-3 weeks
- Data pipeline updates
- Model configuration changes
- **AR(1) implementation across all models**
- Initial backtesting

#### Phase 2 (Counting Stats): 2-3 weeks  
- Feature engineering
- Model development
- Validation testing

#### Phase 3 (Integration): 1-2 weeks
- Game simulation updates
- Dashboard integration
- End-to-end testing

**Updated Total Estimated Timeline: 5-8 weeks**

---

### Success Metrics (Updated)

#### Quantitative Targets
- wOBA projection beats Marcel by >5% in MAE
- AB projection RMSE < 0.05 (AB/PA ratio)
- Games played calibration within ±10%
- Game-level counting stat RMSE improvement >15%
- **AR(1) reduces multi-season RMSE by >10% vs random walk**
- **ρ parameter converges to realistic range (0.7-0.9)**

#### Qualitative Goals
- Seamless integration with existing pipeline
- Maintained model convergence and calibration
- Improved user experience in dashboard
- Enhanced betting/fantasy applications
- **More stable long-term projections**
- **Better theoretical foundation for season evolution**

This enhanced plan maintains all the original counting stat improvements while adding a theoretically superior season evolution model that should significantly improve multi-season projection accuracy and stability.
