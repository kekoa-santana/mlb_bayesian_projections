"""
Generalized hierarchical Bayesian pitcher projection model.

Supports multiple target stats with appropriate likelihoods:
- K%, BB%: Binomial(BF, inv_logit(theta))
- HR/BF: Binomial(BF, inv_logit(theta))

All share the same model structure:
- Age-bucket population priors (young/prime/veteran)
- Player-level random intercepts (non-centered)
- AR(1) season process for talent evolution
- Starter/reliever role covariate
- Full posterior distributions per player per season

Age buckets: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)

Covariates tested for K%: whiff_rate (r=0.822), avg_velo (r=0.336),
and both combined. None improve MAE vs Marcel — partial pooling already
captures the signal. See docs/failed_hypotheses.md for details.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.feature_eng import N_SKILL_TIERS, SKILL_TIER_LABELS
from src.utils.constants import LEAGUE_AVG_OVERALL

logger = logging.getLogger(__name__)

N_AGE_BUCKETS = 3
AGE_BUCKET_LABELS = {0: "young (<=25)", 1: "prime (26-30)", 2: "veteran (31+)"}


@dataclass
class PitcherStatConfig:
    """Configuration for a single pitcher target stat."""

    name: str
    count_col: str         # numerator (e.g. "k", "bb", "hr")
    trials_col: str        # denominator ("batters_faced")
    rate_col: str          # pre-computed rate (e.g. "k_rate")
    likelihood: str        # "binomial"
    league_avg: float
    # Covariate config: list of (col_name, prior_mu, prior_sigma, direction_label)
    covariates: list[tuple[str, float, float, str]] = None
    # Recency weighting: weight = decay^(max_season - season)
    # decay=0.8 → 1.0, 0.8, 0.64, 0.51, 0.41 (≈ Marcel 5/4/3)
    # decay=1.0 → no weighting (all seasons equal)
    season_decay: float = 1.0
    # sigma_season prior: LogNormal(mu, 0.5)
    sigma_season_mu: float = 0.15
    sigma_season_floor: float = 0.0
    sigma_player_prior: float = 0.5


# Empirical year-to-year volatility (logit scale) from 2018-2025 pitcher data:
# k_rate: ~0.28, bb_rate: ~0.35, hr_per_bf: ~0.55
PITCHER_STAT_CONFIGS: dict[str, PitcherStatConfig] = {
    "k_rate": PitcherStatConfig(
        name="k_rate",
        count_col="k",
        trials_col="batters_faced",
        rate_col="k_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["k_rate"],
        covariates=[],
        season_decay=1.0,  # no decay (tested 0.6, 0.8 — neither improves MAE)
        sigma_season_mu=0.22,
        sigma_season_floor=0.18,
    ),
    "bb_rate": PitcherStatConfig(
        name="bb_rate",
        count_col="bb",
        trials_col="batters_faced",
        rate_col="bb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["bb_rate"],
        covariates=[
            ("zone_pct", 0.0, 0.2, "zone% → BB%"),
        ],
        sigma_season_mu=0.28,
        sigma_season_floor=0.22,
    ),
    "hr_per_bf": PitcherStatConfig(
        name="hr_per_bf",
        count_col="hr",
        trials_col="batters_faced",
        rate_col="hr_per_bf",
        likelihood="binomial",
        league_avg=0.030,  # ~3.0% HR/BF league avg
        covariates=[
            ("gb_pct", 0.0, 0.2, "gb% → HR/BF"),
        ],
        sigma_player_prior=0.8,
        sigma_season_mu=0.40,
        sigma_season_floor=0.35,
    ),
}


def prepare_pitcher_data(
    df: pd.DataFrame,
    stat: str,
) -> dict[str, Any]:
    """Prepare multi-season pitcher data for a specific stat model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_pitcher_data``.
        Must contain: pitcher_id, season, age_bucket, batters_faced,
        and the stat-specific columns.
    stat : str
        Stat key from PITCHER_STAT_CONFIGS.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    cfg = PITCHER_STAT_CONFIGS[stat]
    df = df.copy()

    # Encode player IDs
    player_ids = df["pitcher_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["pitcher_id"].map(player_map)

    # Season offsets
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    n_players = len(player_ids)
    n_seasons = df["season_idx"].max() + 1

    # Age bucket + skill tier per player (most recent season)
    player_age_bucket = np.zeros(n_players, dtype=int)
    player_skill_tier = np.ones(n_players, dtype=int)  # default: average (1)
    latest_season = df.groupby("player_idx")["season"].max()
    for pidx in range(n_players):
        if pidx in latest_season.index:
            latest_s = latest_season[pidx]
            rows = df[(df["player_idx"] == pidx) & (df["season"] == latest_s)]
            if len(rows) > 0:
                player_age_bucket[pidx] = int(rows.iloc[0]["age_bucket"])
                if "skill_tier" in rows.columns:
                    player_skill_tier[pidx] = int(rows.iloc[0].get(
                        "skill_tier", 1,
                    ))

    # Z-score covariates
    cov_arrays = {}
    if cfg.covariates:
        for col_name, _, _, _ in cfg.covariates:
            if col_name not in df.columns:
                df[col_name] = np.nan
            vals = df[col_name].values.astype(float)
            vals = np.nan_to_num(vals, nan=0.0)
            mu, sd = vals.mean(), vals.std()
            if np.isclose(sd, 0.0):
                cov_arrays[col_name] = np.zeros_like(vals)
            else:
                cov_arrays[col_name] = (vals - mu) / sd

    # --- Recency weighting via effective sample size ---
    raw_trials = df[cfg.trials_col].values.astype(float)
    raw_counts = df[cfg.count_col].values.astype(float)

    if cfg.season_decay < 1.0:
        max_season_idx = df["season_idx"].max()
        recency = max_season_idx - df["season_idx"].values
        weights = cfg.season_decay ** recency
        eff_trials = np.round(raw_trials * weights).astype(int)
        eff_counts = np.round(raw_counts * weights).astype(int)
        # Ensure counts <= trials after rounding
        eff_counts = np.minimum(eff_counts, eff_trials)
        logger.info(
            "Season decay=%.2f: weights range [%.3f, %.3f] over %d seasons",
            cfg.season_decay, weights.min(), weights.max(), n_seasons,
        )
    else:
        eff_trials = raw_trials.astype(int)
        eff_counts = raw_counts.astype(int)

    result: dict[str, Any] = {
        "player_idx": df["player_idx"].values.astype(int),
        "season_idx": df["season_idx"].values.astype(int),
        "n_players": n_players,
        "n_seasons": n_seasons,
        "player_map": player_map,
        "player_ids": player_ids,
        "min_season": min_season,
        "player_age_bucket": player_age_bucket,
        "player_skill_tier": player_skill_tier,
        "covariates": cov_arrays,
        "stat": stat,
        "df": df,
        "trials": eff_trials,
        "counts": eff_counts,
    }

    if "is_starter" in df.columns:
        result["is_starter"] = df["is_starter"].values.astype(int)

    return result


def build_pitcher_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build a hierarchical Bayesian model for the specified pitcher stat.

    Model structure
    ---------------
    - Age-bucket population means on logit scale
    - Player-level random intercepts (non-centered)
    - AR(1) season process for talent evolution
    - Starter/reliever role shift (optional)
    - Binomial likelihood: count ~ Binomial(BF, inv_logit(theta))

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    import pytensor.tensor as pt

    stat = data["stat"]
    cfg = PITCHER_STAT_CONFIGS[stat]

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    age_bucket = data["player_age_bucket"]
    skill_tier = data["player_skill_tier"]
    has_role = "is_starter" in data

    league_logit = np.log(cfg.league_avg / (1 - cfg.league_avg))

    with pm.Model() as model:
        # --- Age-bucket x skill-tier population means ---
        mu_pop = pm.Normal(
            "mu_pop",
            mu=league_logit,
            sigma=0.3,
            shape=(N_AGE_BUCKETS, N_SKILL_TIERS),
        )

        sigma_player = pm.HalfNormal("sigma_player", sigma=cfg.sigma_player_prior)

        # --- Covariate effects ---
        betas = {}
        if cfg.covariates:
            for col_name, prior_mu, prior_sigma, label in cfg.covariates:
                betas[col_name] = pm.Normal(
                    f"beta_{col_name}", mu=prior_mu, sigma=prior_sigma
                )

        # --- Player-level intercepts (non-centered) ---
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        alpha = pm.Deterministic(
            "alpha",
            mu_pop[age_bucket, skill_tier] + sigma_player * alpha_raw,
        )

        # --- AR(1) season process ---
        sigma_season = pm.LogNormal(
            "sigma_season",
            mu=np.log(cfg.sigma_season_mu),
            sigma=0.5,
        )
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

        # Add covariate effects
        if cfg.covariates:
            for col_name, _, _, _ in cfg.covariates:
                theta = theta + betas[col_name] * data["covariates"][col_name]

        # --- Starter/reliever role effect ---
        if has_role:
            beta_starter = pm.Normal("beta_starter", mu=0, sigma=0.2)
            theta = theta + beta_starter * data["is_starter"]

        # --- Likelihood ---
        rate = pm.Deterministic("rate", pm.math.invlogit(theta))
        pm.Binomial(
            "obs",
            n=data["trials"],
            p=rate,
            observed=data["counts"],
        )

    return model


def fit_pitcher_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the pitcher projection model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
    """
    stat = data["stat"]
    model = build_pitcher_model(data, random_seed=random_seed)

    with model:
        logger.info(
            "Sampling pitcher %s model: %d draws, %d tune, %d chains, "
            "%d players, %d seasons",
            stat, draws, tune, chains, data["n_players"], data["n_seasons"],
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


def extract_posteriors(
    trace: az.InferenceData,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Extract posterior summaries per pitcher per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, age,
        age_bucket, batters_faced, is_starter, observed_{stat},
        {stat}_mean, {stat}_sd, {stat}_2_5, {stat}_50, {stat}_97_5.
    """
    stat = data["stat"]
    cfg = PITCHER_STAT_CONFIGS[stat]
    df = data["df"]

    rate_post = trace.posterior["rate"].values
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = rate_flat[:, pos]
        rec = {
            "pitcher_id": row["pitcher_id"],
            "pitcher_name": row.get("pitcher_name", ""),
            "pitch_hand": row.get("pitch_hand", ""),
            "season": row["season"],
            "age": row.get("age", None),
            "age_bucket": row.get("age_bucket", None),
            "skill_tier": row.get("skill_tier", 1),
            "batters_faced": row["batters_faced"],
            f"observed_{stat}": row[cfg.rate_col],
            f"{stat}_mean": float(np.mean(samples)),
            f"{stat}_sd": float(np.std(samples)),
            f"{stat}_2_5": float(np.percentile(samples, 2.5)),
            f"{stat}_25": float(np.percentile(samples, 25)),
            f"{stat}_50": float(np.percentile(samples, 50)),
            f"{stat}_75": float(np.percentile(samples, 75)),
            f"{stat}_97_5": float(np.percentile(samples, 97.5)),
        }
        if "is_starter" in row.index:
            rec["is_starter"] = int(row["is_starter"])
        records.append(rec)

    return pd.DataFrame(records)


def extract_rate_samples(
    trace: az.InferenceData,
    data: dict[str, Any],
    pitcher_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw posterior samples for one pitcher-season.

    Parameters
    ----------
    trace : az.InferenceData
    data : dict
    pitcher_id : int
    season : int
    project_forward : bool
        If True, add one random walk step for out-of-sample projection.
    random_seed : int

    Returns
    -------
    np.ndarray
        Posterior samples (1D).
    """
    stat = data["stat"]
    cfg = PITCHER_STAT_CONFIGS[stat]
    df = data["df"]

    mask = (df["pitcher_id"] == pitcher_id) & (df["season"] == season)
    positions = df.index[mask].tolist()
    if not positions:
        raise ValueError(f"Pitcher {pitcher_id} not found in season {season}")

    pos = positions[0]
    iloc_pos = df.index.get_loc(pos)

    rate_post = trace.posterior["rate"].values
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])
    samples = rate_flat[:, iloc_pos].copy()

    if project_forward and "sigma_season" in trace.posterior:
        rng = np.random.default_rng(random_seed)
        sigma_samples = trace.posterior["sigma_season"].values.flatten()
        # Apply floor
        sigma_samples = np.maximum(sigma_samples, cfg.sigma_season_floor)
        if len(sigma_samples) != len(samples):
            sigma_draws = rng.choice(sigma_samples, size=len(samples), replace=True)
        else:
            sigma_draws = sigma_samples

        # Get rho for AR(1) dampening
        if "rho" in trace.posterior:
            rho_samples = trace.posterior["rho"].values.flatten()
            if len(rho_samples) != len(samples):
                rho_draws = rng.choice(rho_samples, size=len(samples), replace=True)
            else:
                rho_draws = rho_samples
        else:
            rho_draws = np.ones(len(samples))  # fallback: pure random walk

        # Project on logit scale with AR(1): next = rho * current_effect + innovation
        # The season_effect at the last observed season is already in the samples,
        # so we apply rho dampening + new innovation
        eps = np.clip(samples, 1e-6, 1 - 1e-6)
        logit_samples = np.log(eps / (1 - eps))
        innovation = rng.normal(0, sigma_draws)

        # Extract alpha (player intercept) to compute season effect
        alpha_post = trace.posterior["alpha"].values
        alpha_flat = alpha_post.reshape(-1, alpha_post.shape[-1])
        pidx = data["player_map"][pitcher_id]
        alpha_draws = alpha_flat[:, pidx]
        if len(alpha_draws) != len(samples):
            alpha_draws = rng.choice(alpha_draws, size=len(samples), replace=True)

        # Treat logit(rate) - alpha as the total deviation (season_effect
        # + covariate contributions).  Applying rho to the whole deviation
        # implicitly regresses both the season effect and covariates,
        # which is correct — covariate values also regress toward the mean.
        deviation_last = logit_samples - alpha_draws
        new_deviation = rho_draws * deviation_last + innovation
        samples = 1.0 / (1.0 + np.exp(-(alpha_draws + new_deviation)))

    return samples


def check_convergence(trace: az.InferenceData, stat: str) -> dict[str, Any]:
    """Run convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData
    stat : str

    Returns
    -------
    dict
    """
    cfg = PITCHER_STAT_CONFIGS[stat]
    var_names = ["mu_pop", "sigma_player", "sigma_season"]
    if "rho" in trace.posterior:
        var_names.append("rho")
    if cfg.covariates:
        for col_name, _, _, _ in cfg.covariates:
            beta_name = f"beta_{col_name}"
            if beta_name in trace.posterior:
                var_names.append(beta_name)
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
        "pitcher %s convergence: r_hat_max=%.4f, ESS_min=%d, divergences=%d",
        stat, max_rhat, min_ess_bulk, n_divergences,
    )
    return result
