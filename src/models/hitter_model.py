"""
Generalized hierarchical Bayesian hitter projection model.

Supports multiple target stats with appropriate likelihoods:
- K%, BB%, HR/PA: Binomial(PA, inv_logit(theta))
- xwOBA: Normal(theta, sigma_obs)

All share the same model structure:
- Age-bucket population priors (young/prime/veteran)
- Player-level random intercepts (non-centered)
- Season random walk for talent evolution
- Statcast covariates (barrel_pct, hard_hit_pct)
- Full posterior distributions per player per season

Age buckets: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)
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
class StatConfig:
    """Configuration for a single target stat."""

    name: str
    # Column names in the DataFrame
    count_col: str         # numerator (e.g. "k", "bb", "hr") or value col for continuous
    trials_col: str        # denominator (e.g. "pa") — unused for continuous
    rate_col: str          # pre-computed rate (e.g. "k_rate")
    # Likelihood type
    likelihood: str        # "binomial" or "normal"
    # Prior location (league average on natural scale)
    league_avg: float
    # Covariate config: list of (col_name, prior_mu, prior_sigma, direction_label)
    # direction_label is for logging only
    covariates: list[tuple[str, float, float, str]]
    # sigma_season prior: LogNormal(mu, 0.5) — mode on logit/natural scale
    # derived from empirical year-to-year volatility
    sigma_season_mu: float = 0.15
    # Floor for sigma_season in forward projection — ensures CIs reflect
    # known empirical year-to-year volatility even when the model under-
    # estimates it due to short training windows
    sigma_season_floor: float = 0.0
    # sigma prior for player intercepts
    sigma_player_prior: float = 0.5
    # sigma prior for observation noise (normal likelihood only)
    sigma_obs_prior: float = 0.05


# Pre-defined stat configs
STAT_CONFIGS: dict[str, StatConfig] = {
    "k_rate": StatConfig(
        name="k_rate",
        count_col="k",
        trials_col="pa",
        rate_col="k_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["k_rate"],
        covariates=[
            ("barrel_pct", 0.0, 0.2, "barrel% → K%"),
            ("hard_hit_pct", 0.0, 0.2, "hard_hit% → K%"),
        ],
        # empirical logit-scale yr-to-yr SD ≈ 0.24
        sigma_season_mu=0.20,
        sigma_season_floor=0.18,
    ),
    "bb_rate": StatConfig(
        name="bb_rate",
        count_col="bb",
        trials_col="pa",
        rate_col="bb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["bb_rate"],
        covariates=[
            ("barrel_pct", 0.0, 0.2, "barrel% → BB%"),
            ("hard_hit_pct", 0.0, 0.2, "hard_hit% → BB%"),
        ],
        # empirical logit-scale yr-to-yr SD ≈ 0.33
        sigma_season_mu=0.25,
        sigma_season_floor=0.20,
    ),
    "hr_rate": StatConfig(
        name="hr_rate",
        count_col="hr",
        trials_col="pa",
        rate_col="hr_rate",
        likelihood="binomial",
        league_avg=0.033,  # ~3.3% HR/PA league avg
        covariates=[
            ("barrel_pct", 0.0, 0.3, "barrel% → HR/PA"),
            ("hard_hit_pct", 0.0, 0.2, "hard_hit% → HR/PA"),
        ],
        sigma_player_prior=0.8,
        # empirical logit-scale yr-to-yr SD ≈ 0.54 — HR power is volatile
        sigma_season_mu=0.40,
        sigma_season_floor=0.35,
    ),
    "xwoba": StatConfig(
        name="xwoba",
        count_col="xwoba_avg",
        trials_col="pa",  # used as weight, not binomial n
        rate_col="xwoba_avg",
        likelihood="normal",
        league_avg=LEAGUE_AVG_OVERALL["xwoba"],
        covariates=[
            ("barrel_pct", 0.0, 0.3, "barrel% → xwOBA"),
            ("hard_hit_pct", 0.0, 0.3, "hard_hit% → xwOBA"),
        ],
        sigma_player_prior=0.5,
        # empirical natural-scale yr-to-yr SD ≈ 0.042
        sigma_season_mu=0.035,
        sigma_season_floor=0.025,
        sigma_obs_prior=0.10,
    ),
    "woba": StatConfig(
        name="woba",
        count_col="woba",
        trials_col="pa",  # used as weight, not binomial n
        rate_col="woba",
        likelihood="normal",
        league_avg=LEAGUE_AVG_OVERALL["woba"],
        covariates=[
            ("xwoba_avg", 0.0, 0.3, "xwOBA → wOBA"),
            ("barrel_pct", 0.0, 0.2, "barrel% → wOBA"),
        ],
        sigma_player_prior=0.5,
        # wOBA is noisier than xwOBA (r≈0.45 vs 0.75 YoY)
        # Natural-scale yr-to-yr SD ≈ 0.050
        sigma_season_mu=0.040,
        sigma_season_floor=0.030,
        sigma_obs_prior=0.10,
    ),
}


def prepare_hitter_data(
    df: pd.DataFrame,
    stat: str,
) -> dict[str, Any]:
    """Prepare multi-season hitter data for a specific stat model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_hitter_data``.
        Must contain: batter_id, season, age_bucket, pa, and the
        stat-specific columns.
    stat : str
        Stat key from STAT_CONFIGS (e.g. "k_rate", "bb_rate").

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    cfg = STAT_CONFIGS[stat]
    df = df.copy()

    # For xwOBA, drop rows with NaN values
    if cfg.likelihood == "normal":
        df = df.dropna(subset=[cfg.rate_col])

    # Encode player IDs as contiguous ints
    player_ids = df["batter_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["batter_id"].map(player_map)

    # Season offsets
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    n_players = len(player_ids)
    n_seasons = df["season_idx"].max() + 1

    # Age bucket + skill tier per player (use most recent season)
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
    }

    if cfg.likelihood == "binomial":
        result["trials"] = df[cfg.trials_col].values.astype(int)
        result["counts"] = df[cfg.count_col].values.astype(int)
    else:
        result["y_obs"] = df[cfg.rate_col].values.astype(float)
        # PA as precision weight for xwOBA
        result["pa_weight"] = df[cfg.trials_col].values.astype(float)

    return result


def build_hitter_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build a hierarchical Bayesian model for the specified stat.

    Model structure
    ---------------
    - Age-bucket population means on logit/natural scale
    - Player-level random intercepts (non-centered)
    - Season random walk for talent evolution
    - Statcast covariates shift the linear predictor
    - Binomial or Normal likelihood depending on stat type

    Parameters
    ----------
    data : dict
        Output of ``prepare_hitter_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    import pytensor.tensor as pt

    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    age_bucket = data["player_age_bucket"]
    skill_tier = data["player_skill_tier"]

    if cfg.likelihood == "binomial":
        league_logit = np.log(cfg.league_avg / (1 - cfg.league_avg))
    else:
        league_logit = cfg.league_avg  # not actually logit for normal

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

        # --- Season random walk ---
        # LogNormal prior resists collapsing to zero; centered on empirical
        # year-to-year volatility (logit scale for binomial, natural for normal)
        sigma_season = pm.LogNormal(
            "sigma_season",
            mu=np.log(cfg.sigma_season_mu),
            sigma=0.5,
        )

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

        # --- Linear predictor ---
        theta = (
            alpha[player_idx]
            + season_effect[player_idx, season_idx]
        )

        # Add covariate effects
        for col_name, _, _, _ in cfg.covariates:
            theta = theta + betas[col_name] * data["covariates"][col_name]

        # --- Likelihood ---
        if cfg.likelihood == "binomial":
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            pm.Binomial(
                "obs",
                n=data["trials"],
                p=rate,
                observed=data["counts"],
            )
        else:
            # Normal likelihood for xwOBA
            # theta is on the natural scale (not logit)
            rate = pm.Deterministic("rate", theta)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=cfg.sigma_obs_prior)
            pm.Normal(
                "obs",
                mu=rate,
                sigma=sigma_obs,
                observed=data["y_obs"],
            )

    return model


def fit_hitter_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the hitter projection model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_hitter_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
    """
    stat = data["stat"]
    model = build_hitter_model(data, random_seed=random_seed)

    with model:
        logger.info(
            "Sampling %s model: %d draws, %d tune, %d chains, "
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
    """Extract posterior summaries per player per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, season, age, age_bucket, pa,
        observed_{stat}, {stat}_mean, {stat}_sd, {stat}_2_5, {stat}_50,
        {stat}_97_5.
    """
    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]
    df = data["df"]

    rate_post = trace.posterior["rate"].values  # (chains, draws, obs)
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = rate_flat[:, pos]
        rec = {
            "batter_id": row["batter_id"],
            "batter_name": row.get("batter_name", ""),
            "batter_stand": row.get("batter_stand", ""),
            "season": row["season"],
            "age": row.get("age", None),
            "age_bucket": row.get("age_bucket", None),
            "skill_tier": row.get("skill_tier", 1),
            "pa": row["pa"],
            f"observed_{stat}": row[cfg.rate_col],
            f"{stat}_mean": float(np.mean(samples)),
            f"{stat}_sd": float(np.std(samples)),
            f"{stat}_2_5": float(np.percentile(samples, 2.5)),
            f"{stat}_25": float(np.percentile(samples, 25)),
            f"{stat}_50": float(np.percentile(samples, 50)),
            f"{stat}_75": float(np.percentile(samples, 75)),
            f"{stat}_97_5": float(np.percentile(samples, 97.5)),
        }
        records.append(rec)

    return pd.DataFrame(records)


def extract_rate_samples(
    trace: az.InferenceData,
    data: dict[str, Any],
    batter_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw posterior samples for one player-season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.
    batter_id : int
        Target batter.
    season : int
        Season whose posterior to extract.
    project_forward : bool
        If True, add one random walk step for out-of-sample projection.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Posterior samples (1D array).
    """
    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]
    df = data["df"]

    mask = (df["batter_id"] == batter_id) & (df["season"] == season)
    positions = df.index[mask].tolist()
    if not positions:
        raise ValueError(f"Batter {batter_id} not found in season {season}")

    pos = positions[0]
    iloc_pos = df.index.get_loc(pos)

    rate_post = trace.posterior["rate"].values
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])
    samples = rate_flat[:, iloc_pos].copy()

    if project_forward and "sigma_season" in trace.posterior:
        rng = np.random.default_rng(random_seed)
        sigma_samples = trace.posterior["sigma_season"].values.flatten()
        # Apply floor — ensures CIs reflect known empirical volatility
        sigma_samples = np.maximum(sigma_samples, cfg.sigma_season_floor)
        if len(sigma_samples) != len(samples):
            sigma_draws = rng.choice(sigma_samples, size=len(samples), replace=True)
        else:
            sigma_draws = sigma_samples

        if cfg.likelihood == "binomial":
            # Project on logit scale
            eps = np.clip(samples, 1e-6, 1 - 1e-6)
            logit_samples = np.log(eps / (1 - eps))
            innovation = rng.normal(0, sigma_draws)
            samples = 1.0 / (1.0 + np.exp(-(logit_samples + innovation)))
        else:
            # Project on natural scale
            innovation = rng.normal(0, sigma_draws)
            samples = samples + innovation

    return samples


def check_convergence(trace: az.InferenceData, stat: str) -> dict[str, Any]:
    """Run convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData
    stat : str
        Stat key for covariate naming.

    Returns
    -------
    dict
        Summary with r_hat, ESS, and divergence counts.
    """
    cfg = STAT_CONFIGS[stat]
    var_names = ["mu_pop", "sigma_player", "sigma_season"]
    for col_name, _, _, _ in cfg.covariates:
        var_names.append(f"beta_{col_name}")
    if cfg.likelihood == "normal":
        var_names.append("sigma_obs")

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
        "%s convergence: r_hat_max=%.4f, ESS_min=%d, divergences=%d → %s",
        stat, max_rhat, min_ess_bulk, n_divergences,
        "OK" if result["converged"] else "ISSUES DETECTED",
    )
    return result
