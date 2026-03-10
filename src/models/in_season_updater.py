"""
In-season Bayesian conjugate updating for rate projections.

Uses Beta-Binomial conjugate updating: the preseason posterior
(approximated as Beta via moment-matching) is updated with observed
2026 binomial outcomes to produce an exact Beta posterior.

This is instantaneous compared to re-running full MCMC, and gives
proper posterior uncertainty that narrows as the season progresses.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def moment_match_to_beta(
    samples: np.ndarray,
    floor_total: float = 5.0,
) -> tuple[float, float]:
    """Convert posterior samples to Beta(alpha, beta) via moment-matching.

    Parameters
    ----------
    samples : np.ndarray
        Posterior rate samples in [0, 1].
    floor_total : float
        Minimum effective sample size (alpha + beta).

    Returns
    -------
    tuple[float, float]
        (alpha, beta) parameters.
    """
    mu = float(np.mean(samples))
    var = float(np.var(samples))

    # Clip to valid range
    mu = np.clip(mu, 1e-4, 1 - 1e-4)
    var = np.clip(var, 1e-8, mu * (1 - mu) - 1e-8)

    total = mu * (1 - mu) / var - 1
    total = max(total, floor_total)

    alpha = mu * total
    beta = (1 - mu) * total

    return max(alpha, 0.5), max(beta, 0.5)


def conjugate_update(
    alpha_0: float,
    beta_0: float,
    k_obs: int,
    n_obs: int,
) -> dict[str, float]:
    """Beta-Binomial conjugate update.

    Parameters
    ----------
    alpha_0, beta_0 : float
        Prior Beta parameters (from preseason posterior).
    k_obs : int
        Observed successes (e.g., strikeouts) in 2026.
    n_obs : int
        Observed trials (PA or BF) in 2026.

    Returns
    -------
    dict
        alpha, beta, mean, sd, ci_2_5, ci_97_5.
    """
    alpha_post = alpha_0 + k_obs
    beta_post = beta_0 + (n_obs - k_obs)

    dist = stats.beta(alpha_post, beta_post)
    return {
        "alpha": alpha_post,
        "beta": beta_post,
        "mean": dist.mean(),
        "sd": dist.std(),
        "ci_2_5": float(dist.ppf(0.025)),
        "ci_97_5": float(dist.ppf(0.975)),
    }


def draw_updated_samples(
    alpha: float,
    beta: float,
    n_samples: int = 8000,
    random_seed: int = 42,
) -> np.ndarray:
    """Draw posterior samples from an updated Beta distribution."""
    rng = np.random.default_rng(random_seed)
    return rng.beta(alpha, beta, size=n_samples)


def update_projections(
    preseason_proj: pd.DataFrame,
    observed_2026: pd.DataFrame,
    id_col: str,
    rate_stats: list[dict],
    min_trials: int = 10,
) -> pd.DataFrame:
    """Update rate projections with observed 2026 data via conjugate updating.

    Parameters
    ----------
    preseason_proj : pd.DataFrame
        Preseason projection DataFrame (frozen snapshot).
    observed_2026 : pd.DataFrame
        Current 2026 season totals with columns: id_col, trials_col, successes
        for each rate stat.
    id_col : str
        Player ID column name ('batter_id' or 'pitcher_id').
    rate_stats : list[dict]
        List of dicts describing each rate stat to update:
        [{"name": "k_rate", "trials": "pa", "successes": "k"}, ...]
    min_trials : int
        Minimum trials before updating (below this, keep preseason).

    Returns
    -------
    pd.DataFrame
        Updated projections with new projected_* columns.
    """
    proj = preseason_proj.copy()

    # Merge observed data
    obs = observed_2026.copy()
    obs = obs.rename(columns={c: f"obs_{c}" for c in obs.columns if c != id_col})

    proj = proj.merge(obs, on=id_col, how="left")

    for stat_cfg in rate_stats:
        stat = stat_cfg["name"]
        trials_col = f"obs_{stat_cfg['trials']}"
        success_col = f"obs_{stat_cfg['successes']}"

        proj_col = f"projected_{stat}"
        sd_col = f"projected_{stat}_sd"
        ci_lo = f"projected_{stat}_2_5"
        ci_hi = f"projected_{stat}_97_5"
        delta_col = f"delta_{stat}"

        for idx, row in proj.iterrows():
            n_obs = int(row.get(trials_col, 0) or 0)
            k_obs = int(row.get(success_col, 0) or 0)

            if n_obs < min_trials:
                continue

            # Moment-match preseason to Beta
            pre_mean = row[proj_col]
            pre_sd = row.get(sd_col, 0.03)
            if pd.isna(pre_mean) or pd.isna(pre_sd) or pre_sd <= 0:
                continue

            # Reconstruct Beta params from mean/sd
            mu = np.clip(pre_mean, 1e-4, 1 - 1e-4)
            var = pre_sd ** 2
            var = np.clip(var, 1e-8, mu * (1 - mu) - 1e-8)
            total = mu * (1 - mu) / var - 1
            total = max(total, 5.0)
            alpha_0 = max(mu * total, 0.5)
            beta_0 = max((1 - mu) * total, 0.5)

            # Conjugate update
            updated = conjugate_update(alpha_0, beta_0, k_obs, n_obs)

            proj.at[idx, proj_col] = updated["mean"]
            proj.at[idx, sd_col] = updated["sd"]
            proj.at[idx, ci_lo] = updated["ci_2_5"]
            proj.at[idx, ci_hi] = updated["ci_97_5"]

            # Delta vs observed career
            career_col = f"career_{stat}"
            if career_col in proj.columns and pd.notna(row.get(career_col)):
                proj.at[idx, delta_col] = updated["mean"] - row[career_col]

    # Add observed 2026 columns for display
    for stat_cfg in rate_stats:
        trials_col = f"obs_{stat_cfg['trials']}"
        success_col = f"obs_{stat_cfg['successes']}"
        rate_name = stat_cfg["name"]

        if trials_col in proj.columns and success_col in proj.columns:
            proj[f"observed_2026_{rate_name}"] = np.where(
                proj[trials_col].fillna(0) >= min_trials,
                proj[success_col].fillna(0) / proj[trials_col].replace(0, np.nan),
                np.nan,
            )
            proj[f"obs_2026_{stat_cfg['trials']}"] = proj[trials_col].fillna(0).astype(int)

    # Drop intermediate obs_ columns
    drop_cols = [c for c in proj.columns if c.startswith("obs_") and not c.startswith("obs_2026_")]
    proj = proj.drop(columns=drop_cols, errors="ignore")

    return proj


def update_pitcher_k_samples(
    preseason_samples: dict[str, np.ndarray],
    observed_2026: pd.DataFrame,
    min_bf: int = 10,
    n_samples: int = 8000,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Regenerate pitcher K% posterior samples with conjugate updating.

    Parameters
    ----------
    preseason_samples : dict[str, np.ndarray]
        Preseason K% samples keyed by str(pitcher_id).
    observed_2026 : pd.DataFrame
        Columns: pitcher_id, batters_faced, strike_outs.
    min_bf : int
        Minimum BF before updating.
    n_samples : int
        Number of posterior samples to draw.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Updated K% samples per pitcher.
    """
    updated = {}
    obs_lookup = {}
    for _, row in observed_2026.iterrows():
        pid = str(int(row["pitcher_id"]))
        obs_lookup[pid] = {
            "bf": int(row.get("batters_faced", 0) or 0),
            "k": int(row.get("strike_outs", 0) or 0),
        }

    rng = np.random.default_rng(random_seed)

    for pid_str, samples in preseason_samples.items():
        obs = obs_lookup.get(pid_str, {"bf": 0, "k": 0})

        if obs["bf"] < min_bf:
            updated[pid_str] = samples
            continue

        # Moment-match to Beta
        alpha_0, beta_0 = moment_match_to_beta(samples)

        # Conjugate update
        alpha_post = alpha_0 + obs["k"]
        beta_post = beta_0 + (obs["bf"] - obs["k"])

        updated[pid_str] = rng.beta(alpha_post, beta_post, size=n_samples)

    # Handle new pitchers (in observed but not in preseason)
    league_k_rate = 0.22  # ~league average
    new_prior_strength = 50

    for pid_str, obs in obs_lookup.items():
        if pid_str not in updated and obs["bf"] >= min_bf:
            alpha_0 = league_k_rate * new_prior_strength
            beta_0 = (1 - league_k_rate) * new_prior_strength
            alpha_post = alpha_0 + obs["k"]
            beta_post = beta_0 + (obs["bf"] - obs["k"])
            updated[pid_str] = rng.beta(alpha_post, beta_post, size=n_samples)
            logger.info("New pitcher %s: %d BF, %d K → K%% = %.1f%%",
                        pid_str, obs["bf"], obs["k"],
                        alpha_post / (alpha_post + beta_post) * 100)

    return updated
