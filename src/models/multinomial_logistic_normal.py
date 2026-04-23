"""
Multinomial logistic-normal joint sampler for pitcher K%/BB%/HR%.

Replaces independent Beta posterior sampling with correlated draws that
respect the empirical correlation structure between outcomes. Uses a
logistic-normal approximation: draws on the logit scale from a
multivariate normal, then transforms back with expit.

The population-level correlation matrix is learned from observed
pitcher-season data (2022-2025, 932 pitcher-seasons with >= 100 BF).
Per-pitcher marginal means and variances come from their existing
Beta posteriors (moment-matched to logistic-normal via delta method).

Key correlation: K% and HR/BF are negatively correlated (-0.23 on
logit scale) -- high-K pitchers allow fewer HR. This is the dominant
signal; K-BB and BB-HR correlations are small.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

# Population logit-scale correlation matrix (2022-2025 starters, BF >= 100).
# Order: [K%, BB%, HR/BF]
_POP_LOGIT_CORR = np.array([
    [1.000, -0.082, -0.233],
    [-0.082, 1.000, -0.023],
    [-0.233, -0.023, 1.000],
])

# Cholesky of the correlation matrix (precomputed for speed)
_POP_LOGIT_CHOL = np.linalg.cholesky(_POP_LOGIT_CORR)


def correlated_rate_samples(
    k_samples: np.ndarray,
    bb_samples: np.ndarray,
    hr_samples: np.ndarray,
    n_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw correlated K%/BB%/HR% samples from a logistic-normal.

    Takes existing independent posterior samples (from Beta posteriors),
    estimates per-pitcher marginal moments, and draws new samples with
    the population correlation structure imposed.

    Parameters
    ----------
    k_samples, bb_samples, hr_samples : np.ndarray
        Independent posterior samples for a single pitcher (1-D arrays).
    n_draws : int, optional
        Number of correlated draws to produce. Defaults to len(k_samples).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    (k_corr, bb_corr, hr_corr) : tuple of np.ndarray
        Correlated posterior samples, each shape (n_draws,).
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_draws is None:
        n_draws = len(k_samples)

    # Marginal means on probability scale
    mu_k = float(np.clip(np.mean(k_samples), 0.01, 0.99))
    mu_bb = float(np.clip(np.mean(bb_samples), 0.01, 0.99))
    mu_hr = float(np.clip(np.mean(hr_samples), 0.001, 0.99))

    # Marginal stds on probability scale
    sd_k = float(np.clip(np.std(k_samples), 1e-4, 0.5))
    sd_bb = float(np.clip(np.std(bb_samples), 1e-4, 0.5))
    sd_hr = float(np.clip(np.std(hr_samples), 1e-6, 0.5))

    # Convert to logit scale via delta method:
    # If X ~ Beta with mean mu, std sigma, then
    # logit(X) has approximate mean logit(mu) and std sigma / (mu*(1-mu))
    logit_mu = np.array([logit(mu_k), logit(mu_bb), logit(mu_hr)])
    logit_sd = np.array([
        sd_k / (mu_k * (1 - mu_k)),
        sd_bb / (mu_bb * (1 - mu_bb)),
        sd_hr / (mu_hr * (1 - mu_hr)),
    ])
    # Floor the logit SD to avoid degenerate MVN
    logit_sd = np.clip(logit_sd, 0.01, 3.0)

    # Build covariance: Sigma_ij = corr_ij * sd_i * sd_j
    # Using Cholesky of correlation * diagonal scaling
    scaled_chol = _POP_LOGIT_CHOL * logit_sd[:, None]

    # Draw from MVN via Cholesky: X = mu + L @ Z
    z = rng.standard_normal((n_draws, 3))
    logit_draws = logit_mu + z @ scaled_chol.T

    # Transform back to probability scale
    k_corr = expit(logit_draws[:, 0])
    bb_corr = expit(logit_draws[:, 1])
    hr_corr = expit(logit_draws[:, 2])

    return k_corr, bb_corr, hr_corr


def build_correlated_npz(
    k_npz: dict[str, np.ndarray],
    bb_npz: dict[str, np.ndarray],
    hr_npz: dict[str, np.ndarray],
    n_draws: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Convert independent NPZ posterior samples to correlated versions.

    For each pitcher present in all three NPZs, produces correlated
    samples. Pitchers missing from any NPZ are skipped (they keep
    their original independent samples).

    Parameters
    ----------
    k_npz, bb_npz, hr_npz : dict[str, np.ndarray]
        Independent posterior sample dicts (pitcher_id str -> samples).
    n_draws : int
        Number of correlated draws per pitcher.
    seed : int
        Base random seed.

    Returns
    -------
    (k_corr, bb_corr, hr_corr) : dicts with same keys as inputs
    """
    rng = np.random.default_rng(seed)
    k_out: dict[str, np.ndarray] = {}
    bb_out: dict[str, np.ndarray] = {}
    hr_out: dict[str, np.ndarray] = {}

    common_ids_set = set(k_npz.keys()) & set(bb_npz.keys()) & set(hr_npz.keys())
    common_ids = sorted(common_ids_set)
    n_correlated = 0

    for pid in common_ids:
        k_s = k_npz[pid]
        bb_s = bb_npz[pid]
        hr_s = hr_npz[pid]

        if len(k_s) == 0 or len(bb_s) == 0 or len(hr_s) == 0:
            continue

        k_c, bb_c, hr_c = correlated_rate_samples(
            k_s, bb_s, hr_s,
            n_draws=n_draws,
            rng=rng,
        )
        k_out[pid] = k_c.astype(np.float32)
        bb_out[pid] = bb_c.astype(np.float32)
        hr_out[pid] = hr_c.astype(np.float32)
        n_correlated += 1

    # Copy any pitchers that were in only some NPZs (keep independent)
    for pid in set(k_npz.keys()) - common_ids_set:
        k_out[pid] = k_npz[pid]
    for pid in set(bb_npz.keys()) - common_ids_set:
        bb_out[pid] = bb_npz[pid]
    for pid in set(hr_npz.keys()) - common_ids_set:
        hr_out[pid] = hr_npz[pid]

    logger.info(
        "Correlated %d pitchers (%.0f%% of K NPZ), %d independent fallback",
        n_correlated,
        100 * n_correlated / max(len(k_npz), 1),
        len(k_npz) - n_correlated,
    )
    return k_out, bb_out, hr_out
