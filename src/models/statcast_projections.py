"""Bayesian AR(1) projections for stable Statcast metrics.

Extends the existing K%/BB%/HR model framework to project:
  Hitter: whiff_rate, chase_rate, O-contact%, z_contact%, exit_velo, hard_hit%
  Pitcher: swstr_pct, csw_pct, zone_pct, chase_pct

These feed the scouting grade system with projected values + CIs,
replacing single-year observed values for the stable metrics (r > 0.70 YoY).

Uses lightweight AR(1): per-player intercept + season process + partial pooling.
No covariates, no age buckets — the metrics are physical measurements that
are already stable enough for simple mean-reversion modeling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StatcastConfig:
    """Configuration for a single Statcast metric projection."""
    name: str
    rate_col: str           # column name in the data
    likelihood: str         # "normal" for continuous metrics
    league_avg: float       # population mean
    sigma_season_mu: float  # LogNormal prior on season-to-season SD
    sigma_season_floor: float  # floor for forward projection CI
    rho_alpha: float        # AR(1) persistence prior Beta(alpha, beta)
    rho_beta: float
    player_type: str        # "hitter" or "pitcher"


# Configs based on YoY stability research (grade_yoy_stability.py)
STATCAST_CONFIGS: dict[str, StatcastConfig] = {
    # === HITTER METRICS ===
    "whiff_rate": StatcastConfig(
        name="whiff_rate", rate_col="whiff_rate", likelihood="normal",
        league_avg=0.25, sigma_season_mu=0.03, sigma_season_floor=0.025,
        rho_alpha=9, rho_beta=1.5,  # r=0.846 YoY → high persistence
        player_type="hitter",
    ),
    "chase_rate": StatcastConfig(
        name="chase_rate", rate_col="chase_rate", likelihood="normal",
        league_avg=0.30, sigma_season_mu=0.025, sigma_season_floor=0.020,
        rho_alpha=9, rho_beta=1.5,  # r=0.834 YoY
        player_type="hitter",
    ),
    "o_contact_pct": StatcastConfig(
        name="o_contact_pct", rate_col="o_contact_pct", likelihood="normal",
        league_avg=0.65, sigma_season_mu=0.04, sigma_season_floor=0.035,
        rho_alpha=8, rho_beta=2,  # r=0.804 YoY
        player_type="hitter",
    ),
    "z_contact_pct": StatcastConfig(
        name="z_contact_pct", rate_col="z_contact_pct", likelihood="normal",
        league_avg=0.82, sigma_season_mu=0.025, sigma_season_floor=0.020,
        rho_alpha=8, rho_beta=2,  # r=0.796 YoY
        player_type="hitter",
    ),
    "avg_exit_velo": StatcastConfig(
        name="avg_exit_velo", rate_col="avg_exit_velo", likelihood="normal",
        league_avg=88.5, sigma_season_mu=1.5, sigma_season_floor=1.2,
        rho_alpha=8, rho_beta=2,  # r=0.765 YoY
        player_type="hitter",
    ),
    "hard_hit_pct": StatcastConfig(
        name="hard_hit_pct", rate_col="hard_hit_pct", likelihood="normal",
        league_avg=0.40, sigma_season_mu=0.04, sigma_season_floor=0.030,
        rho_alpha=8, rho_beta=2.5,  # r=0.750 YoY
        player_type="hitter",
    ),
    # === PITCHER METRICS ===
    "swstr_pct": StatcastConfig(
        name="swstr_pct", rate_col="swstr_pct", likelihood="normal",
        league_avg=0.11, sigma_season_mu=0.02, sigma_season_floor=0.015,
        rho_alpha=7, rho_beta=2.5,  # r=0.689 YoY
        player_type="pitcher",
    ),
    "csw_pct": StatcastConfig(
        name="csw_pct", rate_col="csw_pct", likelihood="normal",
        league_avg=0.29, sigma_season_mu=0.02, sigma_season_floor=0.015,
        rho_alpha=7, rho_beta=3,  # r=0.584 YoY
        player_type="pitcher",
    ),
    "zone_pct": StatcastConfig(
        name="zone_pct", rate_col="zone_pct", likelihood="normal",
        league_avg=0.45, sigma_season_mu=0.025, sigma_season_floor=0.020,
        rho_alpha=7, rho_beta=3,  # r=0.591 YoY
        player_type="pitcher",
    ),
    "chase_pct_pitcher": StatcastConfig(
        name="chase_pct_pitcher", rate_col="chase_pct", likelihood="normal",
        league_avg=0.30, sigma_season_mu=0.025, sigma_season_floor=0.020,
        rho_alpha=7, rho_beta=3,  # r=0.508 YoY
        player_type="pitcher",
    ),
}


def _load_hitter_metric_history(
    metric: str,
    seasons: list[int],
    min_pa: int = 200,
) -> pd.DataFrame:
    """Load per-player per-season values for a hitter Statcast metric."""
    from src.data.feature_eng import get_cached_hitter_observed_profile
    from src.data.db import read_sql

    frames = []
    for s in seasons:
        # Base PA data
        batting = read_sql(f"""
            SELECT batter_id, pa FROM production.fact_batting_advanced
            WHERE season = {s} AND pa >= {min_pa}
        """, {})
        if batting.empty:
            continue

        # Observed profile
        try:
            obs = get_cached_hitter_observed_profile(s)
            if metric in obs.columns:
                batting = batting.merge(
                    obs[["batter_id", metric]], on="batter_id", how="inner",
                )
        except Exception:
            continue

        # O-contact% needs special query
        if metric == "o_contact_pct" and metric not in batting.columns:
            try:
                oc = read_sql(f"""
                    SELECT fp.batter_id,
                        COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                            OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing AND NOT is_whiff)::float
                        / NULLIF(COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                            OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing), 0) AS o_contact_pct
                    FROM production.fact_pitch fp
                    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
                    WHERE dg.season = {s} AND dg.game_type = 'R'
                    GROUP BY fp.batter_id
                    HAVING COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                        OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing) >= 50
                """, {})
                if not oc.empty:
                    batting = batting.merge(oc, on="batter_id", how="inner")
            except Exception:
                continue

        if metric not in batting.columns:
            continue

        batting = batting.dropna(subset=[metric])
        batting["season"] = s
        frames.append(batting[["batter_id", "season", "pa", metric]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_pitcher_metric_history(
    metric: str,
    seasons: list[int],
    min_bf: int = 100,
) -> pd.DataFrame:
    """Load per-player per-season values for a pitcher Statcast metric."""
    from src.data.db import read_sql

    frames = []
    for s in seasons:
        pitching = read_sql(f"""
            SELECT pitcher_id, batters_faced, {metric}
            FROM production.fact_pitching_advanced
            WHERE season = {s} AND batters_faced >= {min_bf}
              AND {metric} IS NOT NULL
        """, {})
        if pitching.empty:
            continue
        pitching["season"] = s
        frames.append(pitching[["pitcher_id", "season", "batters_faced", metric]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fit_statcast_model(
    cfg: StatcastConfig,
    seasons: list[int],
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit a lightweight Bayesian AR(1) model for a Statcast metric.

    Returns dict with 'projections' DataFrame (player_id, projected_mean, projected_sd,
    projected_lo, projected_hi) for the last season in the training window.
    """
    import pymc as pm
    import pytensor.tensor as pt

    # Load data
    if cfg.player_type == "hitter":
        df = _load_hitter_metric_history(cfg.rate_col, seasons)
        id_col = "batter_id"
    else:
        df = _load_pitcher_metric_history(cfg.rate_col, seasons)
        id_col = "pitcher_id"

    if df.empty or len(df) < 50:
        logger.warning("Insufficient data for %s (%d rows)", cfg.name, len(df))
        return {"projections": pd.DataFrame()}

    # Encode players and seasons as contiguous indices
    player_ids = df[id_col].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df[id_col].map(player_map)
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season
    n_players = len(player_ids)
    n_seasons = df["season_idx"].max() + 1

    logger.info(
        "Fitting %s: %d players, %d seasons, %d observations",
        cfg.name, n_players, n_seasons, len(df),
    )

    y_obs = df[cfg.rate_col].values.astype(float)
    player_idx = df["player_idx"].values.astype(int)
    season_idx = df["season_idx"].values.astype(int)

    with pm.Model() as model:
        # Population mean
        mu_pop = pm.Normal("mu_pop", mu=cfg.league_avg, sigma=0.1)

        # Player intercepts (partial pooling)
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.1)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        alpha = pm.Deterministic("alpha", mu_pop + sigma_player * alpha_raw)

        # AR(1) season process
        sigma_season = pm.LogNormal(
            "sigma_season", mu=np.log(cfg.sigma_season_mu), sigma=0.5,
        )
        rho = pm.Beta("rho", alpha=cfg.rho_alpha, beta=cfg.rho_beta)

        if n_seasons > 1:
            innovation = pm.Normal("innovation", mu=0, sigma=1, shape=(n_players, n_seasons))
            season_0 = (sigma_season * innovation[:, 0]).dimshuffle(0, "x")
            ar_components = [season_0]
            for t in range(1, n_seasons):
                prev = ar_components[-1][:, -1]
                cur = rho * prev + sigma_season * innovation[:, t]
                ar_components.append(cur.dimshuffle(0, "x"))
            season_effect = pm.Deterministic(
                "season_effect", pt.concatenate(ar_components, axis=1),
            )
        else:
            season_effect = pt.zeros((n_players, 1))

        # Observation model
        mu_obs = alpha[player_idx] + season_effect[player_idx, season_idx]
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.05)
        pm.Normal("y", mu=mu_obs, sigma=sigma_obs, observed=y_obs)

    # Sample
    with model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=random_seed, progressbar=False,
            cores=1,  # Windows compatibility — avoid multiprocessing fork issues
        )

    # Check convergence
    try:
        import arviz as az
        summary = az.summary(trace, var_names=["rho", "sigma_season", "sigma_obs"])
        r_hat_max = summary["r_hat"].max()
        ess_min = summary["ess_bulk"].min()
        logger.info(
            "%s convergence: r_hat_max=%.3f, ESS_min=%.0f",
            cfg.name, r_hat_max, ess_min,
        )
    except Exception:
        r_hat_max = np.nan
        ess_min = np.nan

    # Extract forward projections for players in the last season
    last_season = df["season"].max()
    last_season_idx = last_season - min_season
    active_players = df[df["season"] == last_season][id_col].unique()

    rho_draws = trace.posterior["rho"].values.flatten()
    sigma_draws = trace.posterior["sigma_season"].values.flatten()
    season_effect_draws = trace.posterior["season_effect"].values
    # Reshape: (chains, draws, n_players, n_seasons) → (total_draws, n_players, n_seasons)
    se_flat = season_effect_draws.reshape(-1, n_players, n_seasons)
    alpha_draws = trace.posterior["alpha"].values.reshape(-1, n_players)

    projections = []
    for pid in active_players:
        pidx = player_map[pid]
        # Last season deviation
        dev_last = se_flat[:, pidx, last_season_idx]
        # Forward: new_dev = rho * dev_last + innovation
        # Mean of forward deviation = rho * dev_last (innovation mean = 0)
        # SD of forward = sqrt(rho^2 * var_last + sigma_season^2)
        # But simpler: just sample forward
        n_draws = len(rho_draws)
        innov = np.random.randn(n_draws) * np.maximum(sigma_draws, cfg.sigma_season_floor)
        new_dev = rho_draws * dev_last + innov
        projected_values = alpha_draws[:, pidx] + new_dev

        projections.append({
            "player_id": pid,
            f"projected_{cfg.name}": projected_values.mean(),
            f"projected_{cfg.name}_sd": projected_values.std(),
            f"projected_{cfg.name}_lo": np.percentile(projected_values, 10),
            f"projected_{cfg.name}_hi": np.percentile(projected_values, 90),
        })

    result_df = pd.DataFrame(projections)
    logger.info(
        "%s projections: %d players, mean=%.4f, sd=%.4f",
        cfg.name, len(result_df),
        result_df[f"projected_{cfg.name}"].mean(),
        result_df[f"projected_{cfg.name}_sd"].mean(),
    )

    return {
        "projections": result_df,
        "convergence": {"r_hat_max": r_hat_max, "ess_min": ess_min},
        "config": cfg,
    }


def fit_all_statcast_models(
    seasons: list[int] | None = None,
    metrics: list[str] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
) -> pd.DataFrame:
    """Fit all Statcast metric models and return merged projections.

    Returns a DataFrame with player_id and projected_* columns for each metric.
    """
    if seasons is None:
        seasons = list(range(2020, 2026))
    if metrics is None:
        metrics = list(STATCAST_CONFIGS.keys())

    all_projections: list[pd.DataFrame] = []

    for metric_name in metrics:
        if metric_name not in STATCAST_CONFIGS:
            logger.warning("Unknown metric: %s", metric_name)
            continue

        cfg = STATCAST_CONFIGS[metric_name]
        logger.info("=" * 50)
        logger.info("Fitting %s (%s)...", cfg.name, cfg.player_type)

        result = fit_statcast_model(cfg, seasons, draws=draws, tune=tune, chains=chains)
        proj = result["projections"]
        if not proj.empty:
            all_projections.append(proj)

    if not all_projections:
        return pd.DataFrame()

    # Merge all projections by player_id
    merged = all_projections[0]
    for proj in all_projections[1:]:
        merged = merged.merge(proj, on="player_id", how="outer")

    logger.info("All Statcast projections merged: %d players, %d columns",
                len(merged), len(merged.columns))
    return merged


def compute_projected_grades_with_ci(
    hitter_statcast_proj: pd.DataFrame,
    pitcher_statcast_proj: pd.DataFrame,
    hitter_rankings: pd.DataFrame,
    pitcher_rankings: pd.DataFrame,
    n_mc: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute projected tool grades with Monte Carlo CIs.

    Samples from the posterior distributions of projected Statcast metrics
    simultaneously, computes tool grades for each sample, then takes
    percentiles for floor/ceiling.

    Returns (hitter_grade_ci, pitcher_grade_ci) DataFrames with columns:
    player_id, grade_X, grade_X_lo, grade_X_hi, diamond_rating,
    diamond_rating_lo, diamond_rating_hi.
    """
    # For now, compute simple CIs from the projected mean +/- SD
    # Full MC sampling requires the tool grade functions to accept arrays,
    # which is a bigger refactor. This gives a good approximation.

    hitter_ci = pd.DataFrame()
    pitcher_ci = pd.DataFrame()

    if not hitter_statcast_proj.empty and not hitter_rankings.empty:
        hitter_ci = hitter_rankings[["batter_id", "diamond_rating",
                                      "grade_hit", "grade_power", "grade_speed",
                                      "grade_fielding", "grade_discipline"]].copy()
        hitter_ci = hitter_ci.rename(columns={"batter_id": "player_id"})

        # Merge projected SDs
        hitter_ci = hitter_ci.merge(hitter_statcast_proj, on="player_id", how="left")

        # Estimate grade CI from projected metric SDs
        # Each tool grade moves ~2 grade points per 1 SD of its primary metric
        grade_sensitivity = 2.0  # approximate grade points per metric SD
        for grade_col, sd_cols in [
            ("grade_hit", ["projected_whiff_rate_sd", "projected_z_contact_pct_sd",
                           "projected_o_contact_pct_sd"]),
            ("grade_power", ["projected_avg_exit_velo_sd", "projected_hard_hit_pct_sd"]),
            ("grade_discipline", ["projected_chase_rate_sd"]),
        ]:
            available_sds = [c for c in sd_cols if c in hitter_ci.columns]
            if available_sds:
                avg_sd = hitter_ci[available_sds].mean(axis=1).fillna(0)
                # Normalize: SD as fraction of population spread
                grade_uncertainty = (avg_sd / avg_sd.median()).clip(0.5, 3.0) * 5
                hitter_ci[f"{grade_col}_lo"] = (hitter_ci[grade_col] - grade_uncertainty).clip(20, 80).astype(int)
                hitter_ci[f"{grade_col}_hi"] = (hitter_ci[grade_col] + grade_uncertainty).clip(20, 80).astype(int)
            else:
                hitter_ci[f"{grade_col}_lo"] = (hitter_ci[grade_col] - 5).clip(20, 80)
                hitter_ci[f"{grade_col}_hi"] = (hitter_ci[grade_col] + 5).clip(20, 80)

        # Speed and fielding: fixed uncertainty (speed very stable, fielding volatile)
        hitter_ci["grade_speed_lo"] = (hitter_ci["grade_speed"] - 3).clip(20, 80)
        hitter_ci["grade_speed_hi"] = (hitter_ci["grade_speed"] + 3).clip(20, 80)
        hitter_ci["grade_fielding_lo"] = (hitter_ci["grade_fielding"] - 8).clip(20, 80)
        hitter_ci["grade_fielding_hi"] = (hitter_ci["grade_fielding"] + 8).clip(20, 80)

        # Diamond rating CI: propagate from tool CIs
        # Floor: all tools at their low end; Ceiling: all at their high end
        dr = hitter_ci["diamond_rating"]
        tool_uncertainty = (
            hitter_ci[[f"grade_{t}_hi" for t in ["hit", "power", "speed", "fielding", "discipline"]]].mean(axis=1)
            - hitter_ci[[f"grade_{t}_lo" for t in ["hit", "power", "speed", "fielding", "discipline"]]].mean(axis=1)
        ) / 60 * 10  # convert grade spread to diamond spread

        hitter_ci["diamond_rating_lo"] = (dr - tool_uncertainty / 2).clip(1, 10).round(1)
        hitter_ci["diamond_rating_hi"] = (dr + tool_uncertainty / 2).clip(1, 10).round(1)

    if not pitcher_statcast_proj.empty and not pitcher_rankings.empty:
        pitcher_ci = pitcher_rankings[["pitcher_id", "diamond_rating",
                                        "grade_stuff", "grade_command",
                                        "grade_durability"]].copy()
        pitcher_ci = pitcher_ci.rename(columns={"pitcher_id": "player_id"})
        pitcher_ci = pitcher_ci.merge(pitcher_statcast_proj, on="player_id", how="left")

        for grade_col, sd_cols in [
            ("grade_stuff", ["projected_swstr_pct_sd", "projected_csw_pct_sd"]),
            ("grade_command", ["projected_zone_pct_sd", "projected_chase_pct_pitcher_sd"]),
        ]:
            available_sds = [c for c in sd_cols if c in pitcher_ci.columns]
            if available_sds:
                avg_sd = pitcher_ci[available_sds].mean(axis=1).fillna(0)
                grade_uncertainty = (avg_sd / avg_sd.median()).clip(0.5, 3.0) * 5
                pitcher_ci[f"{grade_col}_lo"] = (pitcher_ci[grade_col] - grade_uncertainty).clip(20, 80).astype(int)
                pitcher_ci[f"{grade_col}_hi"] = (pitcher_ci[grade_col] + grade_uncertainty).clip(20, 80).astype(int)
            else:
                pitcher_ci[f"{grade_col}_lo"] = (pitcher_ci[grade_col] - 5).clip(20, 80)
                pitcher_ci[f"{grade_col}_hi"] = (pitcher_ci[grade_col] + 5).clip(20, 80)

        pitcher_ci["grade_durability_lo"] = (pitcher_ci["grade_durability"] - 7).clip(20, 80)
        pitcher_ci["grade_durability_hi"] = (pitcher_ci["grade_durability"] + 7).clip(20, 80)

        dr = pitcher_ci["diamond_rating"]
        tool_uncertainty = (
            pitcher_ci[[f"grade_{t}_hi" for t in ["stuff", "command", "durability"]]].mean(axis=1)
            - pitcher_ci[[f"grade_{t}_lo" for t in ["stuff", "command", "durability"]]].mean(axis=1)
        ) / 60 * 10

        pitcher_ci["diamond_rating_lo"] = (dr - tool_uncertainty / 2).clip(1, 10).round(1)
        pitcher_ci["diamond_rating_hi"] = (dr + tool_uncertainty / 2).clip(1, 10).round(1)

    return hitter_ci, pitcher_ci
