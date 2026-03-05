"""
Layer 1: Hierarchical Bayesian K% projection model for pitchers.

Mirrors the hitter K% model (k_rate_model.py) with pitcher-specific
covariates:
- whiff_rate (arsenal-level whiff skill) → positive effect on K%
- barrel_rate_against (contact suppression) → negative effect on K%
- is_starter (role flag from IP/game) → population-level shift

Uses batters_faced (bf) as the Binomial trial count, not PA.
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

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def prepare_pitcher_model_data(df: pd.DataFrame) -> dict[str, Any]:
    """Prepare pitcher multi-season data for the PyMC model.

    Encodes pitcher IDs as integer indices, computes season offsets,
    and z-scores arsenal covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_pitcher_k_data``.
        Must contain: pitcher_id, season, batters_faced, k, k_rate,
        whiff_rate, barrel_rate_against.  Optionally ``is_starter``
        (1=starter, 0=reliever); if present it is passed through as-is.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.

    Raises
    ------
    ValueError
        If ``whiff_rate`` or ``barrel_rate_against`` columns are missing.
    """
    for col in ["whiff_rate", "barrel_rate_against"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                "Use build_multi_season_pitcher_k_data() to produce the input."
            )

    df = df.copy()

    # --- encode player IDs as contiguous ints ---
    player_ids = df["pitcher_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["pitcher_id"].map(player_map)

    # --- season offsets (0-based from earliest) ---
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    # --- Statcast covariates (z-scored, NaN → 0 for missing/constant) ---
    for col in ["whiff_rate", "barrel_rate_against"]:
        mu = df[col].mean()
        sd = df[col].std()
        if pd.isna(sd) or np.isclose(sd, 0.0):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = ((df[col] - mu) / sd).fillna(0.0)

    result = {
        "player_idx": df["player_idx"].values.astype(int),
        "season_idx": df["season_idx"].values.astype(int),
        "bf": df["batters_faced"].values.astype(int),
        "k": df["k"].values.astype(int),
        "whiff_z": df["whiff_rate_z"].values.astype(float),
        "barrel_against_z": df["barrel_rate_against_z"].values.astype(float),
        "n_players": len(player_ids),
        "n_seasons": df["season_idx"].max() + 1,
        "player_map": player_map,
        "player_ids": player_ids,
        "min_season": min_season,
        "df": df,
    }

    if "is_starter" in df.columns:
        result["is_starter"] = df["is_starter"].values.astype(int)

    return result


def build_pitcher_k_rate_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build the hierarchical Bayesian pitcher K% model.

    Model structure
    ---------------
    - Population mean K% on logit scale ~ Normal(logit(0.224), 0.3)
    - Player-level random intercepts (non-centered)
    - Season random walk for talent evolution
    - Arsenal covariates: whiff_rate (+K%), barrel_rate_against
    - Starter/reliever role flag (optional)
    - Binomial likelihood: K ~ Binomial(BF, inv_logit(theta))

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_model_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    import pytensor.tensor as pt

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    bf = data["bf"]
    k_obs = data["k"]
    whiff_z = data["whiff_z"]
    barrel_against_z = data["barrel_against_z"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    has_role = "is_starter" in data

    # League-average K% on logit scale
    league_k_logit = np.log(
        LEAGUE_AVG_OVERALL["k_rate"] / (1 - LEAGUE_AVG_OVERALL["k_rate"])
    )

    with pm.Model() as model:
        # --- Population-level ---
        mu_pop = pm.Normal("mu_pop", mu=league_k_logit, sigma=0.3)
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.5)

        # --- Arsenal covariate effects ---
        # Wider prior (0.3) than hitter model (0.2): whiff_rate is a more
        # direct causal predictor of pitcher K%.
        beta_whiff = pm.Normal("beta_whiff", mu=0, sigma=0.3)
        beta_barrel_against = pm.Normal("beta_barrel_against", mu=0, sigma=0.3)

        # --- Player-level intercepts (non-centered) ---
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        alpha = pm.Deterministic(
            "alpha", mu_pop + sigma_player * alpha_raw
        )

        # --- Season-to-season innovation (random walk) ---
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.15)

        if n_seasons > 1:
            innovation = pm.Normal(
                "innovation", mu=0, sigma=1,
                shape=(n_players, n_seasons - 1),
            )
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

        # --- Starter/reliever role effect ---
        if has_role:
            is_starter = data["is_starter"]
            beta_starter = pm.Normal("beta_starter", mu=0, sigma=0.2)

        # --- Linear predictor ---
        theta = (
            alpha[player_idx]
            + season_effect[player_idx, season_idx]
            + beta_whiff * whiff_z
            + beta_barrel_against * barrel_against_z
        )
        if has_role:
            theta = theta + beta_starter * is_starter

        # --- K rate on probability scale ---
        k_rate = pm.Deterministic("k_rate", pm.math.invlogit(theta))

        # --- Likelihood ---
        pm.Binomial("k_obs", n=bf, p=k_rate, observed=k_obs)

    return model


def fit_pitcher_k_rate_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the pitcher K% model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_model_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
    """
    model = build_pitcher_k_rate_model(data, random_seed=random_seed)

    with model:
        logger.info(
            "Sampling pitcher K-rate model: %d draws, %d tune, %d chains, "
            "%d players, %d seasons",
            draws, tune, chains, data["n_players"], data["n_seasons"],
        )

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

        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return model, trace


def extract_pitcher_posteriors(
    trace: az.InferenceData,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Extract posterior K% summaries per pitcher per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace from ``fit_pitcher_k_rate_model``.
    data : dict
        Model data dict from ``prepare_pitcher_model_data``.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season,
        batters_faced, is_starter, observed_k_rate, k_rate_mean, k_rate_sd,
        k_rate_2_5, k_rate_25, k_rate_50, k_rate_75, k_rate_97_5.
    """
    df = data["df"]
    k_rate_post = trace.posterior["k_rate"].values  # (chains, draws, obs)
    k_rate_flat = k_rate_post.reshape(-1, k_rate_post.shape[-1])

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = k_rate_flat[:, pos]
        rec = {
            "pitcher_id": row["pitcher_id"],
            "pitcher_name": row.get("pitcher_name", ""),
            "pitch_hand": row.get("pitch_hand", ""),
            "season": row["season"],
            "batters_faced": row["batters_faced"],
            "observed_k_rate": row["k_rate"],
            "k_rate_mean": np.mean(samples),
            "k_rate_sd": np.std(samples),
            "k_rate_2_5": np.percentile(samples, 2.5),
            "k_rate_25": np.percentile(samples, 25),
            "k_rate_50": np.percentile(samples, 50),
            "k_rate_75": np.percentile(samples, 75),
            "k_rate_97_5": np.percentile(samples, 97.5),
        }
        if "is_starter" in row.index:
            rec["is_starter"] = int(row["is_starter"])
        records.append(rec)

    return pd.DataFrame(records)


def check_pitcher_convergence(trace: az.InferenceData) -> dict[str, Any]:
    """Run standard convergence diagnostics on the pitcher trace.

    Parameters
    ----------
    trace : az.InferenceData

    Returns
    -------
    dict
        Summary with r_hat, ESS, and divergence counts.
    """
    var_names = [
        "mu_pop", "sigma_player", "sigma_season",
        "beta_whiff", "beta_barrel_against",
    ]
    if "beta_starter" in trace.posterior:
        var_names.append("beta_starter")
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
