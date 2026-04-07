"""
Shared utilities for hierarchical Bayesian player projection models.

Extracted from hitter_model.py and pitcher_model.py to eliminate ~350 lines
of duplicated logic. Both model files delegate to these functions internally
while preserving their original public APIs.

The functions accept any stat-config object that has the required fields
(duck-typed), so there's no need for a shared base class between
StatConfig and PitcherStatConfig.
"""
from __future__ import annotations

import logging
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.feature_eng import N_SKILL_TIERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prepare_player_data
# ---------------------------------------------------------------------------

def prepare_player_data(
    df: pd.DataFrame,
    cfg: Any,
    stat: str,
    id_col: str,
    n_age_buckets: int,
    player_type: str,
) -> dict[str, Any]:
    """Prepare multi-season player data for a hierarchical Bayesian model.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-season player data with columns for the target stat.
    cfg : StatConfig or PitcherStatConfig
        Configuration for the target stat.
    stat : str
        Stat key (e.g. "k_rate").
    id_col : str
        Player ID column name ("batter_id" or "pitcher_id").
    n_age_buckets : int
        Number of age buckets (hitter=4, pitcher=3).
    player_type : str
        "hitter" or "pitcher" — used for MiLB prior offset lookup.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    df = df.copy()

    # For continuous likelihoods, drop rows with NaN values
    likelihood = cfg.likelihood
    if likelihood in ("normal", "logit_normal"):
        df = df.dropna(subset=[cfg.rate_col])

    # Encode player IDs as contiguous ints
    player_ids = df[id_col].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df[id_col].map(player_map)

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

    # Recency weighting handled at projection step (extract_rate_samples),
    # not in the likelihood.
    recency_wt = np.ones(len(df), dtype=np.float64)

    # Z-score covariates
    cov_arrays = {}
    covariates = cfg.covariates if cfg.covariates else []
    for col_name, _, _, _ in covariates:
        if col_name not in df.columns:
            df[col_name] = np.nan
        vals = df[col_name].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0)
        mu, sd = vals.mean(), vals.std()
        if np.isclose(sd, 0.0):
            cov_arrays[col_name] = np.zeros_like(vals)
        else:
            cov_arrays[col_name] = (vals - mu) / sd

    # --- MiLB-informed prior offsets for early-career players ---
    milb_prior_offset = np.zeros(n_players, dtype=np.float64)
    has_milb_prior = np.zeros(n_players, dtype=np.float64)
    try:
        from src.data.feature_eng import compute_milb_prior_offsets
        max_season = int(df["season"].max())
        offsets = compute_milb_prior_offsets(
            df, player_id_col=id_col, player_type=player_type,
            projection_season=max_season + 1,
        )
        if offsets:
            for pid, stat_offsets in offsets.items():
                if pid in player_map and stat in stat_offsets:
                    pidx = player_map[pid]
                    milb_prior_offset[pidx] = stat_offsets[stat]
                    has_milb_prior[pidx] = 1.0
            n_with_prior = int(has_milb_prior.sum())
            if n_with_prior > 0:
                logger.info(
                    "MiLB prior offsets: %d/%d %ss for %s",
                    n_with_prior, n_players, player_type, stat,
                )
    except Exception as exc:
        logger.debug("MiLB prior offsets unavailable: %s", exc)

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
        "milb_prior_offset": milb_prior_offset,
        "has_milb_prior": has_milb_prior,
        "recency_weight": recency_wt,
        "stat": stat,
        "df": df,
    }

    # Likelihood-specific observed data
    if likelihood == "binomial":
        result["trials"] = df[cfg.trials_col].values.astype(int)
        result["counts"] = df[cfg.count_col].values.astype(int)
    elif likelihood == "logit_normal":
        raw_vals = df[cfg.rate_col].values.astype(float)
        clipped = np.clip(raw_vals, 1e-4, 1 - 1e-4)
        result["y_obs"] = np.log(clipped / (1 - clipped))
        result["pa_weight"] = df[cfg.trials_col].values.astype(float)
    elif likelihood == "normal":
        result["y_obs"] = df[cfg.rate_col].values.astype(float)
        result["pa_weight"] = df[cfg.trials_col].values.astype(float)

    # Role covariate (pitcher-specific: is_starter)
    if "is_starter" in df.columns:
        result["is_starter"] = df["is_starter"].values.astype(int)

    return result


# ---------------------------------------------------------------------------
# build_hierarchical_model
# ---------------------------------------------------------------------------

def build_hierarchical_model(
    data: dict[str, Any],
    cfg: Any,
    n_age_buckets: int,
    random_seed: int = 42,
) -> pm.Model:
    """Build a hierarchical Bayesian model for the specified stat.

    Supports four likelihood types:
    - "binomial": Binomial(n, inv_logit(theta))
    - "beta_binomial": BetaBinomial(n, alpha=rate*phi, beta=(1-rate)*phi)
      Set cfg.use_beta_binomial = True on a binomial stat to enable.
    - "logit_normal": Normal on logit scale, rate = inv_logit(theta)
    - "normal": Normal on natural scale, rate = theta

    Parameters
    ----------
    data : dict
        Output of ``prepare_player_data``.
    cfg : StatConfig or PitcherStatConfig
        Stat configuration.
    n_age_buckets : int
        Number of age buckets (hitter=4, pitcher=3).
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    import pytensor.tensor as pt

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    age_bucket = data["player_age_bucket"]
    skill_tier = data["player_skill_tier"]
    has_role = "is_starter" in data

    if cfg.likelihood in ("binomial", "logit_normal"):
        league_logit = np.log(cfg.league_avg / (1 - cfg.league_avg))
    else:
        league_logit = cfg.league_avg  # not actually logit for normal

    with pm.Model() as model:
        # --- Age-bucket x skill-tier population means ---
        mu_pop = pm.Normal(
            "mu_pop",
            mu=league_logit,
            sigma=cfg.mu_pop_sigma,
            shape=(n_age_buckets, N_SKILL_TIERS),
        )

        sigma_player = pm.HalfNormal("sigma_player", sigma=cfg.sigma_player_prior)

        # --- Covariate effects ---
        betas = {}
        covariates = cfg.covariates if cfg.covariates else []
        for col_name, prior_mu, prior_sigma, label in covariates:
            betas[col_name] = pm.Normal(
                f"beta_{col_name}", mu=prior_mu, sigma=prior_sigma
            )

        # --- Player-level intercepts (non-centered) ---
        # StudentT(nu) has 2-3x heavier tails than Normal, letting elite/
        # replacement players sit further from the population mean without
        # over-shrinkage.  Used for composite metrics (wOBA) where the tails
        # are genuine talent, not noise.  (Gelman 2006; Jensen et al. 2009)
        _nu = getattr(cfg, "alpha_prior_nu", None)
        if _nu is not None:
            alpha_raw = pm.StudentT(
                "alpha_raw", nu=_nu, mu=0, sigma=1, shape=n_players,
            )
        else:
            alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        milb_offset = data.get("milb_prior_offset")
        has_milb = data.get("has_milb_prior")
        if milb_offset is not None and has_milb is not None and has_milb.sum() > 0:
            milb_offset_t = pt.as_tensor_variable(milb_offset.astype(np.float64))
            has_milb_t = pt.as_tensor_variable(has_milb.astype(np.float64))
            alpha = pm.Deterministic(
                "alpha",
                mu_pop[age_bucket, skill_tier]
                + sigma_player * alpha_raw
                + milb_offset_t * has_milb_t,
            )
        else:
            alpha = pm.Deterministic(
                "alpha",
                mu_pop[age_bucket, skill_tier] + sigma_player * alpha_raw,
            )

        # --- AR(2) season process ---
        sigma_season = pm.LogNormal(
            "sigma_season",
            mu=np.log(cfg.sigma_season_mu),
            sigma=0.35,
        )
        rho = pm.Beta("rho", alpha=cfg.rho_alpha, beta=cfg.rho_beta)
        rho2 = pm.Beta("rho2", alpha=cfg.rho2_alpha, beta=cfg.rho2_beta)

        if n_seasons > 1:
            innovation = pm.Normal(
                "innovation", mu=0, sigma=1,
                shape=(n_players, n_seasons),
            )
            # Season 0: just innovation
            s0 = (sigma_season * innovation[:, 0]).dimshuffle(0, "x")
            ar_components = [s0]

            if n_seasons >= 2:
                # Season 1: AR(1) only (no lag-2 available yet)
                s1 = rho * s0[:, -1] + sigma_season * innovation[:, 1]
                ar_components.append(s1.dimshuffle(0, "x"))

            # Seasons 2+: full AR(2)
            for t in range(2, n_seasons):
                prev1 = ar_components[-1][:, -1]   # t-1
                prev2 = ar_components[-2][:, -1]   # t-2
                cur = (rho * prev1 + rho2 * prev2
                       + sigma_season * innovation[:, t])
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
        for col_name, _, _, _ in covariates:
            theta = theta + betas[col_name] * data["covariates"][col_name]

        # --- Starter/reliever role effect (pitcher only) ---
        if has_role:
            beta_starter = pm.Normal("beta_starter", mu=0, sigma=0.2)
            theta = theta + beta_starter * data["is_starter"]

        # --- Likelihood ---
        # Determine effective likelihood: check for beta_binomial override
        use_beta_binomial = getattr(cfg, "use_beta_binomial", False)

        if cfg.likelihood == "binomial" and not use_beta_binomial:
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            pm.Binomial(
                "obs",
                n=data["trials"],
                p=rate,
                observed=data["counts"],
            )
        elif cfg.likelihood == "binomial" and use_beta_binomial:
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            phi = pm.HalfNormal("phi", sigma=50)
            alpha_bb = rate * phi
            beta_bb = (1.0 - rate) * phi
            pm.BetaBinomial(
                "obs",
                n=data["trials"],
                alpha=alpha_bb,
                beta=beta_bb,
                observed=data["counts"],
            )
        elif cfg.likelihood == "logit_normal":
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=cfg.sigma_obs_prior)
            pm.Normal(
                "obs",
                mu=theta,
                sigma=sigma_obs,
                observed=data["y_obs"],
            )
        else:
            # Normal likelihood (unbounded, for stats not in [0,1])
            rate = pm.Deterministic("rate", theta)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=cfg.sigma_obs_prior)
            pm.Normal(
                "obs",
                mu=rate,
                sigma=sigma_obs,
                observed=data["y_obs"],
            )

    return model


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------

def fit_model(
    model: pm.Model,
    stat: str,
    label: str,
    n_players: int,
    n_seasons: int,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> az.InferenceData:
    """Sample a PyMC model with nutpie, falling back to pm.sample.

    Parameters
    ----------
    model : pm.Model
        Compiled PyMC model.
    stat : str
        Stat name for logging.
    label : str
        Log prefix (e.g. "hitter" or "pitcher").
    n_players, n_seasons : int
        For logging.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    az.InferenceData
    """
    with model:
        logger.info(
            "Sampling %s %s model: %d draws, %d tune, %d chains, "
            "%d players, %d seasons",
            label, stat, draws, tune, chains, n_players, n_seasons,
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

    return trace


# ---------------------------------------------------------------------------
# extract_rate_samples_forward
# ---------------------------------------------------------------------------

def extract_rate_samples_forward(
    trace: az.InferenceData,
    data: dict[str, Any],
    cfg: Any,
    player_id: int,
    season: int,
    id_col: str,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw posterior samples for one player-season, optionally
    forward-projecting one AR(2) step.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.
    cfg : StatConfig or PitcherStatConfig
        Stat configuration.
    player_id : int
        Target player ID.
    season : int
        Season whose posterior to extract.
    id_col : str
        Player ID column ("batter_id" or "pitcher_id").
    project_forward : bool
        If True, add one AR(2) random walk step for out-of-sample projection.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Posterior samples (1D array).
    """
    df = data["df"]

    mask = (df[id_col] == player_id) & (df["season"] == season)
    positions = df.index[mask].tolist()
    if not positions:
        raise ValueError(f"Player {player_id} not found in season {season}")

    pos = positions[0]
    iloc_pos = df.index.get_loc(pos)

    rate_post = trace.posterior["rate"].values
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])
    samples = rate_flat[:, iloc_pos].copy()

    if not (project_forward and "sigma_season" in trace.posterior):
        return samples

    rng = np.random.default_rng(random_seed)

    # --- Joint posterior sampling ---
    n_target = len(samples)
    sigma_samples = trace.posterior["sigma_season"].values.flatten()
    sigma_samples = np.maximum(sigma_samples, cfg.sigma_season_floor)
    n_posterior = len(sigma_samples)

    if n_posterior != n_target:
        shared_idx = rng.choice(n_posterior, size=n_target, replace=True)
        sigma_draws = sigma_samples[shared_idx]
    else:
        shared_idx = None
        sigma_draws = sigma_samples

    # Get rho for AR(2) dampening
    if "rho" in trace.posterior:
        rho_samples = trace.posterior["rho"].values.flatten()
        rho_draws = rho_samples[shared_idx] if shared_idx is not None else rho_samples
    else:
        rho_draws = np.ones(n_target)

    # Get rho2 (AR(2) lag-2 coefficient)
    if "rho2" in trace.posterior:
        rho2_samples = trace.posterior["rho2"].values.flatten()
        rho2_draws = rho2_samples[shared_idx] if shared_idx is not None else rho2_samples
    else:
        rho2_draws = np.zeros(n_target)

    # Extract alpha (player intercept) — aligned with rate samples
    alpha_post = trace.posterior["alpha"].values
    alpha_flat = alpha_post.reshape(-1, alpha_post.shape[-1])
    pidx = data["player_map"][player_id]
    alpha_draws = alpha_flat[:, pidx]
    if shared_idx is not None:
        alpha_draws = alpha_draws[shared_idx]

    innovation = rng.normal(0, sigma_draws)

    # Age-dependent persistence
    age = df.loc[pos, "age"] if "age" in df.columns else None
    if age is not None:
        if age <= 27:
            age_rho_mult = 1.0 + (27 - age) * 0.065
        elif age <= 32:
            age_rho_mult = 1.0
        else:
            age_rho_mult = max(0.70, 1.0 - (age - 32) * 0.05)
        effective_rho = np.clip(rho_draws * age_rho_mult, 0, 0.99)
    else:
        effective_rho = rho_draws

    # Enforce AR(2) stationarity: rho + rho2 < 1
    rho2_draws = np.minimum(rho2_draws, 0.98 - effective_rho)

    # --- Compute deviation_prev (lag-2) for AR(2) forward projection ---
    prev_season = season - 1
    prev_mask = (df[id_col] == player_id) & (df["season"] == prev_season)
    prev_positions = df.index[prev_mask].tolist()

    # Determine if we operate on logit scale or natural scale
    uses_logit = cfg.likelihood in ("binomial", "logit_normal")

    if uses_logit:
        eps = np.clip(samples, 1e-6, 1 - 1e-6)
        logit_samples = np.log(eps / (1 - eps))
        deviation_last = logit_samples - alpha_draws

        if prev_positions:
            prev_iloc = df.index.get_loc(prev_positions[0])
            prev_samples = rate_flat[:, prev_iloc].copy()
            if shared_idx is not None:
                prev_samples = prev_samples[shared_idx]
            eps_prev = np.clip(prev_samples, 1e-6, 1 - 1e-6)
            logit_prev = np.log(eps_prev / (1 - eps_prev))
            deviation_prev = logit_prev - alpha_draws
        else:
            deviation_prev = np.zeros_like(deviation_last)
            rho2_draws = np.zeros_like(rho2_draws)

        new_deviation = (effective_rho * deviation_last
                         + rho2_draws * deviation_prev + innovation)
        samples = 1.0 / (1.0 + np.exp(-(alpha_draws + new_deviation)))
    else:
        # Normal likelihood: natural scale
        deviation_last = samples - alpha_draws

        if prev_positions:
            prev_iloc = df.index.get_loc(prev_positions[0])
            prev_samples = rate_flat[:, prev_iloc].copy()
            if shared_idx is not None:
                prev_samples = prev_samples[shared_idx]
            deviation_prev = prev_samples - alpha_draws
        else:
            rho2_draws = np.zeros_like(rho2_draws)
            deviation_prev = np.zeros_like(deviation_last)

        new_deviation = (effective_rho * deviation_last
                         + rho2_draws * deviation_prev + innovation)
        samples = alpha_draws + new_deviation

    return samples


# ---------------------------------------------------------------------------
# check_convergence_generic
# ---------------------------------------------------------------------------

def check_convergence_generic(
    trace: az.InferenceData,
    cfg: Any,
    stat: str,
    label: str = "model",
) -> dict[str, Any]:
    """Run convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData
    cfg : StatConfig or PitcherStatConfig
        Stat configuration (used to find covariate beta names).
    stat : str
        Stat name for logging.
    label : str
        Log prefix (e.g. "hitter k_rate" or "pitcher k_rate").

    Returns
    -------
    dict
        Summary with r_hat, ESS, and divergence counts.
    """
    var_names = ["mu_pop", "sigma_player", "sigma_season"]
    if "rho" in trace.posterior:
        var_names.append("rho")
    if "rho2" in trace.posterior:
        var_names.append("rho2")

    covariates = cfg.covariates if cfg.covariates else []
    for col_name, _, _, _ in covariates:
        beta_name = f"beta_{col_name}"
        if beta_name in trace.posterior:
            var_names.append(beta_name)

    # Likelihood-specific variables
    if "sigma_obs" in trace.posterior:
        var_names.append("sigma_obs")
    if "phi" in trace.posterior:
        var_names.append("phi")
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
        "%s convergence: r_hat_max=%.4f, ESS_min=%d, divergences=%d%s",
        label, max_rhat, min_ess_bulk, n_divergences,
        " -> OK" if result["converged"] else " -> ISSUES DETECTED",
    )
    return result
