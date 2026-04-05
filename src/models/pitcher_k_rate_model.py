"""
Layer 1: Pitcher K%-only hierarchical Bayesian model.

Used by the Game K prediction pipeline (``game_k_model.py``) to extract
per-pitcher K% posterior samples, and by content generation scripts.
For multi-stat composite projections (K%, BB%, HR/BF), see
``pitcher_model.py`` and ``pitcher_projections.py`` instead.

Uses hierarchical partial pooling + AR(1) process for talent evolution.
Whiff/barrel covariates are intentionally excluded from the model:
whiff_rate correlates r=0.71 with K%, which collapses sigma_player
regardless of parameterization (observation-level, player-level,
centered, non-centered, any prior width). The model's edge over
Marcel comes from calibration (Brier scores) and coverage, not point estimates.

- is_starter (role flag from IP/game) → observation-level shift

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


def prepare_pitcher_model_data(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Prepare pitcher multi-season data for the PyMC model.

    Encodes pitcher IDs as integer indices, computes season offsets,
    and computes player-level covariate means for informational use.
    The covariates are NOT used in the model (whiff_rate's r=0.71
    correlation with K% collapses variance components), but are
    retained in the output dict for data exploration.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_pitcher_k_data``.
        Must contain: pitcher_id, season, batters_faced, k, k_rate.
        Optionally whiff_rate, barrel_rate_against, is_starter.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    df = df.copy()

    # --- encode player IDs as contiguous ints ---
    player_ids = df["pitcher_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["pitcher_id"].map(player_map)

    # --- season offsets (0-based from earliest) ---
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    # --- Player-level covariate means (averaged across all seasons) ---
    # These inform the prior on where a player's K% talent is centered,
    # NOT the observation-level theta.  This avoids the r=0.71 correlation
    # issue that caused overconfident posteriors when covariates were at
    # the observation level.
    n_players = len(player_ids)
    whiff_player_mean = np.zeros(n_players, dtype=float)
    barrel_player_mean = np.zeros(n_players, dtype=float)

    for col in ["whiff_rate", "barrel_rate_against"]:
        if col not in df.columns:
            df[col] = np.nan

    player_means = df.groupby("player_idx").agg(
        whiff_mean=("whiff_rate", "mean"),
        barrel_mean=("barrel_rate_against", "mean"),
    )

    for pidx in range(n_players):
        if pidx in player_means.index:
            whiff_player_mean[pidx] = player_means.loc[pidx, "whiff_mean"]
            barrel_player_mean[pidx] = player_means.loc[pidx, "barrel_mean"]

    # Z-score the player-level means
    for arr, name in [
        (whiff_player_mean, "whiff"),
        (barrel_player_mean, "barrel"),
    ]:
        arr[:] = np.nan_to_num(arr, nan=0.0)
        mu = arr.mean()
        sd = arr.std()
        if np.isclose(sd, 0.0):
            arr[:] = 0.0
        else:
            arr[:] = (arr - mu) / sd

    result = {
        "player_idx": df["player_idx"].values.astype(int),
        "season_idx": df["season_idx"].values.astype(int),
        "bf": df["batters_faced"].values.astype(int),
        "k": df["k"].values.astype(int),
        "n_players": n_players,
        "n_seasons": df["season_idx"].max() + 1,
        "player_map": player_map,
        "player_ids": player_ids,
        "min_season": min_season,
        "whiff_player_mean": whiff_player_mean,
        "barrel_player_mean": barrel_player_mean,
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
    - AR(1) season process for talent evolution
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

        # --- Player-level intercepts (non-centered) ---
        # No Statcast covariates: whiff_rate (r=0.71 with K%) collapses
        # sigma_player regardless of parameterization. The hierarchical
        # structure + random walk provide the model's calibration edge.
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        alpha = pm.Deterministic(
            "alpha",
            mu_pop + sigma_player * alpha_raw,
        )

        # --- AR(1) season process ---
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.2)
        rho = pm.Beta("rho", alpha=8, beta=2)  # high persistence prior (~0.8)

        if n_seasons > 1:
            innovation = pm.Normal(
                "innovation", mu=0, sigma=1,
                shape=(n_players, n_seasons),
            )
            # Build AR(1) process iteratively
            season_0 = (sigma_season * innovation[:, 0]).dimshuffle(0, "x")
            ar_components = [season_0]
            for t in range(1, n_seasons):
                prev = ar_components[-1][:, -1]
                cur = rho * prev + sigma_season * innovation[:, t]
                ar_components.append(cur.dimshuffle(0, "x"))
            season_effect = pm.Deterministic(
                "season_effect", pt.concatenate(ar_components, axis=1)
            )
        else:
            season_effect = pt.zeros((n_players, 1))

        # --- Linear predictor ---
        theta = (
            alpha[player_idx]
            + season_effect[player_idx, season_idx]
        )

        # --- Starter/reliever role effect (role, not skill metric) ---
        if has_role:
            is_starter = data["is_starter"]
            beta_starter = pm.Normal("beta_starter", mu=0, sigma=0.2)
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
    var_names = ["mu_pop", "sigma_player", "sigma_season"]
    if "rho" in trace.posterior:
        var_names.append("rho")
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
