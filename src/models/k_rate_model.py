"""
Layer 1: Hitter K%-only hierarchical Bayesian model.

Used by content generation scripts and the original K%-only backtest.
For multi-stat composite projections (K%, BB%, HR/PA, xwOBA), see
``hitter_model.py`` and ``hitter_projections.py`` instead.

Estimates true-talent strikeout rate for hitters using:
- Hierarchical partial pooling across players (no hard PA cutoffs)
- Multi-season data with year-to-year talent evolution
- Statcast quality metrics (barrel_pct, hard_hit_pct) as
  observation-level covariates in the linear predictor
- Binomial observation model

The model produces full posterior distributions, not point estimates.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.utils.constants import LEAGUE_AVG_OVERALL
from src.utils.math_helpers import flatten_posterior, posterior_point_summary

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def prepare_model_data(
    df: pd.DataFrame,
    platoon: bool = False,
) -> dict[str, Any]:
    """Prepare raw multi-season data for the PyMC model.

    Encodes player IDs as integer indices, computes season offsets,
    and extracts Statcast covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_k_data`` (or
        ``build_multi_season_k_data_platoon`` when ``platoon=True``).
        Must contain: batter_id, season, pa, k, k_rate, barrel_pct,
        hard_hit_pct.  When ``platoon=True``, must also contain
        ``same_side``.
    platoon : bool
        If True, include same_side array for the platoon model.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.

    Raises
    ------
    ValueError
        If ``platoon=True`` but ``same_side`` column is missing.
    """
    if platoon and "same_side" not in df.columns:
        raise ValueError(
            "platoon=True requires a 'same_side' column in the DataFrame. "
            "Use build_multi_season_k_data_platoon() to produce the input."
        )

    df = df.copy()

    # --- encode player IDs as contiguous ints ---
    player_ids = df["batter_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["batter_id"].map(player_map)

    # --- season offsets (0-based from earliest) ---
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    # --- Statcast covariates (observation-level z-scores) ---
    n_players = len(player_ids)

    for col in ["barrel_pct", "hard_hit_pct"]:
        if col not in df.columns:
            df[col] = np.nan

    # Z-score each covariate across the full training set
    barrel_vals = df["barrel_pct"].values.astype(float)
    barrel_vals = np.nan_to_num(barrel_vals, nan=0.0)
    mu_b, sd_b = barrel_vals.mean(), barrel_vals.std()
    barrel_z = (barrel_vals - mu_b) / sd_b if not np.isclose(sd_b, 0.0) else np.zeros_like(barrel_vals)

    hard_hit_vals = df["hard_hit_pct"].values.astype(float)
    hard_hit_vals = np.nan_to_num(hard_hit_vals, nan=0.0)
    mu_h, sd_h = hard_hit_vals.mean(), hard_hit_vals.std()
    hard_hit_z = (hard_hit_vals - mu_h) / sd_h if not np.isclose(sd_h, 0.0) else np.zeros_like(hard_hit_vals)

    result = {
        "player_idx": df["player_idx"].values.astype(int),
        "season_idx": df["season_idx"].values.astype(int),
        "pa": df["pa"].values.astype(int),
        "k": df["k"].values.astype(int),
        "barrel_z": barrel_z,
        "hard_hit_z": hard_hit_z,
        "n_players": n_players,
        "n_seasons": df["season_idx"].max() + 1,
        "player_map": player_map,
        "player_ids": player_ids,
        "min_season": min_season,
        "platoon": platoon,
        "df": df,
    }

    if platoon:
        result["same_side"] = df["same_side"].values.astype(int)

    return result


def build_k_rate_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build the hierarchical Bayesian K% model.

    Model structure
    ---------------
    - Population mean K% on logit scale ~ Normal(logit(0.22), 0.3)
    - Player-level random intercepts ~ Normal(0, sigma_player)
    - Season random walk: talent[t] = talent[t-1] + innovation
    - Statcast covariates shift the player intercept
    - Binomial likelihood: K ~ Binomial(PA, inv_logit(theta))

    Parameters
    ----------
    data : dict
        Output of ``prepare_model_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    import pytensor.tensor as pt

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    pa = data["pa"]
    k_obs = data["k"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    is_platoon = data.get("platoon", False)

    # Observation-level z-scored covariates
    barrel_z = data["barrel_z"]
    hard_hit_z = data["hard_hit_z"]

    # League-average K% on logit scale
    league_k_logit = np.log(
        LEAGUE_AVG_OVERALL["k_rate"] / (1 - LEAGUE_AVG_OVERALL["k_rate"])
    )

    with pm.Model() as model:
        # --- Population-level ---
        mu_pop = pm.Normal("mu_pop", mu=league_k_logit, sigma=0.3)
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.5)

        # --- Statcast covariate effects (observation-level) ---
        # Higher barrel% and hard-hit% → lower K rate (better contact ability)
        beta_barrel = pm.Normal("beta_barrel", mu=0, sigma=0.2)
        beta_hard_hit = pm.Normal("beta_hard_hit", mu=0, sigma=0.2)

        # --- Player-level intercepts (non-centered) ---
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        alpha = pm.Deterministic(
            "alpha",
            mu_pop + sigma_player * alpha_raw,
        )

        # --- Season-to-season innovation (random walk) ---
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.15)

        if n_seasons > 1:
            # Innovation for seasons 1..T-1
            innovation = pm.Normal(
                "innovation", mu=0, sigma=1,
                shape=(n_players, n_seasons - 1),
            )
            # Build cumulative walk: season 0 = 0, season t = sum of innovations
            cum_innov = pt.concatenate(
                [
                    pt.zeros((n_players, 1)),
                    pt.cumsum(sigma_season * innovation, axis=1),
                ],
                axis=1,
            )
            season_effect = pm.Deterministic("season_effect", cum_innov)
        else:
            season_effect = pt.zeros((n_players, 1))

        # --- Linear predictor ---
        theta = (
            alpha[player_idx]
            + season_effect[player_idx, season_idx]
            + beta_barrel * barrel_z
            + beta_hard_hit * hard_hit_z
        )

        # --- Platoon effect (same-side matchup shift) ---
        if is_platoon:
            same_side = data["same_side"]
            # Population-level same-side effect (+0.05 logit ≈ +1.2 pp at 22%)
            gamma_pop = pm.Normal("gamma_pop", mu=0.05, sigma=0.1)
            sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.15)
            # Non-centered player-level platoon sensitivity
            gamma_raw = pm.Normal("gamma_raw", mu=0, sigma=1, shape=n_players)
            gamma = pm.Deterministic(
                "gamma", gamma_pop + sigma_gamma * gamma_raw
            )
            theta = theta + gamma[player_idx] * same_side

        # --- K rate on probability scale ---
        k_rate = pm.Deterministic("k_rate", pm.math.invlogit(theta))

        # --- Likelihood ---
        pm.Binomial("k_obs", n=pa, p=k_rate, observed=k_obs)

    return model


def fit_k_rate_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the K% model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_model_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
        The PyMC model and the ArviZ InferenceData with posterior + posterior_predictive.
    """
    model = build_k_rate_model(data, random_seed=random_seed)

    with model:
        logger.info(
            "Sampling K-rate model: %d draws, %d tune, %d chains, "
            "%d players, %d seasons",
            draws, tune, chains, data["n_players"], data["n_seasons"],
        )

        # Use nutpie (Rust-based sampler) if available — much faster
        # on Windows where g++ is often unavailable for PyTensor C compilation
        try:
            import nutpie
            logger.info("Using nutpie sampler (Rust backend)")
            compiled = nutpie.compile_pymc_model(model)
            trace = nutpie.sample(
                compiled,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                seed=random_seed,
            )
        except (ImportError, Exception) as e:
            logger.info("nutpie unavailable (%s), falling back to PyMC NUTS", e)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        # Posterior predictive for calibration checks
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return model, trace


def extract_player_posteriors(
    trace: az.InferenceData,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Extract posterior K% summaries per player per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace from ``fit_k_rate_model``.
    data : dict
        Model data dict (for player_map, seasons, etc.).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, k_rate_mean, k_rate_sd,
        k_rate_2_5, k_rate_25, k_rate_50, k_rate_75, k_rate_97_5,
        observed_k_rate, pa.
    """
    df = data["df"]
    is_platoon = data.get("platoon", False)
    k_rate_flat = flatten_posterior(trace, "k_rate")

    # Platoon gamma posteriors (per player)
    if is_platoon and "gamma" in trace.posterior:
        gamma_flat = flatten_posterior(trace, "gamma")
    else:
        gamma_flat = None

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = k_rate_flat[:, pos]
        rec = {
            "batter_id": row["batter_id"],
            "batter_name": row.get("batter_name", ""),
            "season": row["season"],
            "pa": row["pa"],
            "observed_k_rate": row["k_rate"],
            **posterior_point_summary(samples, prefix="k_rate"),
        }
        if is_platoon:
            rec["pitch_hand"] = row.get("pitch_hand", "")
            rec["same_side"] = row.get("same_side", 0)
            if gamma_flat is not None:
                pidx = int(row["player_idx"])
                rec["gamma_mean"] = float(np.mean(gamma_flat[:, pidx]))
                rec["gamma_sd"] = float(np.std(gamma_flat[:, pidx]))
        records.append(rec)

    return pd.DataFrame(records)


def check_convergence(trace: az.InferenceData) -> dict[str, Any]:
    """Run standard convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData

    Returns
    -------
    dict
        Summary with r_hat, ESS, and divergence counts.
    """
    var_names = ["mu_pop", "sigma_player", "sigma_season",
                  "beta_barrel", "beta_hard_hit"]
    # Auto-detect platoon parameters
    for v in ["gamma_pop", "sigma_gamma"]:
        if v in trace.posterior:
            var_names.append(v)
    summary = az.summary(trace, var_names=var_names)
    n_divergences = int(trace.sample_stats["diverging"].sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())

    result = {
        "n_divergences": n_divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "converged": max_rhat < 1.05 and n_divergences == 0 and min_ess_bulk > 400,
        "summary": summary,
    }

    logger.info(
        "Convergence: r_hat_max=%.4f, ESS_min=%d, divergences=%d → %s",
        max_rhat, min_ess_bulk, n_divergences,
        "OK" if result["converged"] else "ISSUES DETECTED",
    )
    return result
