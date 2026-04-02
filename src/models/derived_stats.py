"""
Derived pitcher/batter rates from Bayesian posteriors.

Instead of using crude Beta-shrinkage priors for H/BF and Outs/BF,
derive them as residuals from the Bayesian-projected K%, BB%, HR%
posteriors.  This propagates calibrated uncertainty from the hierarchical
models into H and Outs predictions.

Identity used
-------------
    BIP/BF = 1 - K% - BB% - HR% - HBP%
    H/BF   = BIP/BF  *  BABIP
    Outs/BF = K% + BIP/BF * (1 - BABIP)

BABIP is extremely noisy year-to-year (r = 0.105 for qualified pitchers),
so we shrink each pitcher's observed BABIP heavily toward the league
average (~0.292) using a Beta conjugate prior.
"""
from __future__ import annotations

import logging

import numpy as np

from src.utils.constants import (
    LEAGUE_BABIP_PITCHER,
    LEAGUE_BABIP_BATTER,
    LEAGUE_HBP_RATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League-average constants (2018-2025 pooled)
# ---------------------------------------------------------------------------
LEAGUE_BABIP: float = LEAGUE_BABIP_PITCHER
LEAGUE_H_PER_BF: float = 0.220
LEAGUE_OUTS_PER_BF: float = 0.703

# Batter-side league averages
BATTER_LEAGUE_BABIP: float = LEAGUE_BABIP_BATTER
BATTER_LEAGUE_HBP_RATE: float = LEAGUE_HBP_RATE


def _shrink_babip(
    observed_babip: float | None,
    bip: float,
    league_babip: float = LEAGUE_BABIP,
    effective_n: float = 30.0,
) -> tuple[float, float]:
    """Shrink observed BABIP toward league average using Beta conjugate.

    Parameters
    ----------
    observed_babip : float or None
        Pitcher's observed BABIP in the training period.
        If None, returns league average with prior uncertainty.
    bip : float
        Number of balls in play (reliability weight).
    league_babip : float
        League-average BABIP to shrink toward.
    effective_n : float
        Prior strength (number of pseudo-BIP).  Higher = more shrinkage.
        Default 30 reflects BABIP's very low YoY correlation (~0.10).

    Returns
    -------
    tuple[float, float]
        (alpha_posterior, beta_posterior) for a Beta distribution.
    """
    alpha_prior = league_babip * effective_n
    beta_prior = (1 - league_babip) * effective_n

    if observed_babip is None or np.isnan(observed_babip) or bip < 1:
        return alpha_prior, beta_prior

    hits_on_bip = observed_babip * bip
    outs_on_bip = bip - hits_on_bip

    alpha_post = alpha_prior + hits_on_bip
    beta_post = beta_prior + outs_on_bip

    return max(alpha_post, 0.1), max(beta_post, 0.1)


def derive_pitcher_h_rate(
    k_rate_posterior: np.ndarray,
    bb_rate_posterior: np.ndarray,
    hr_rate_posterior: np.ndarray,
    observed_babip: float | None = None,
    bip: float = 0.0,
    hbp_rate: float = LEAGUE_HBP_RATE,
    league_babip: float = LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Derive pitcher H/BF rate posterior from Bayesian K%, BB%, HR% posteriors.

    The decomposition is:
        BIP/BF = 1 - K% - BB% - HR% - HBP%
        H/BF   = BIP/BF * BABIP + HR%
    (because Hits includes home runs, but BABIP excludes HR from numerator)

    BABIP is drawn from a Beta posterior (heavy shrinkage toward league avg).

    Parameters
    ----------
    k_rate_posterior : np.ndarray
        Posterior samples of pitcher K% (shape: [n_samples]).
    bb_rate_posterior : np.ndarray
        Posterior samples of pitcher BB%.
    hr_rate_posterior : np.ndarray
        Posterior samples of pitcher HR/BF.
    observed_babip : float or None
        Pitcher's observed BABIP.  If None, uses league average.
    bip : float
        Number of balls in play in the training data (for BABIP reliability).
    hbp_rate : float
        HBP rate (treated as fixed — very stable).
    league_babip : float
        League-average BABIP.
    babip_prior_n : float
        Effective prior sample size for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray
        Posterior samples of H/BF rate.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(k_rate_posterior)

    # BIP rate per BF
    bip_rate = 1.0 - k_rate_posterior - bb_rate_posterior - hr_rate_posterior - hbp_rate
    bip_rate = np.clip(bip_rate, 0.0, 1.0)

    # BABIP posterior draws
    alpha, beta = _shrink_babip(observed_babip, bip, league_babip, babip_prior_n)
    babip_samples = rng.beta(alpha, beta, size=n_samples)

    # H/BF = hits on BIP + home runs (since "hits" includes HR in baseball)
    h_rate = bip_rate * babip_samples + hr_rate_posterior
    return np.clip(h_rate, 0.0, 1.0)


def derive_pitcher_outs_rate(
    k_rate_posterior: np.ndarray,
    bb_rate_posterior: np.ndarray,
    hr_rate_posterior: np.ndarray,
    observed_babip: float | None = None,
    bip: float = 0.0,
    hbp_rate: float = LEAGUE_HBP_RATE,
    league_babip: float = LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Derive pitcher Outs/BF rate posterior from Bayesian K%, BB%, HR% posteriors.

    The decomposition is:
        BIP/BF = 1 - K% - BB% - HR% - HBP%
        Outs/BF = K% + BIP/BF * (1 - BABIP)

    Parameters
    ----------
    k_rate_posterior : np.ndarray
        Posterior samples of pitcher K% (shape: [n_samples]).
    bb_rate_posterior : np.ndarray
        Posterior samples of pitcher BB%.
    hr_rate_posterior : np.ndarray
        Posterior samples of pitcher HR/BF.
    observed_babip : float or None
        Pitcher's observed BABIP.
    bip : float
        Number of balls in play in the training data.
    hbp_rate : float
        HBP rate (treated as fixed).
    league_babip : float
        League-average BABIP.
    babip_prior_n : float
        Effective prior sample size for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray
        Posterior samples of Outs/BF rate.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(k_rate_posterior)

    bip_rate = 1.0 - k_rate_posterior - bb_rate_posterior - hr_rate_posterior - hbp_rate
    bip_rate = np.clip(bip_rate, 0.0, 1.0)

    alpha, beta = _shrink_babip(observed_babip, bip, league_babip, babip_prior_n)
    babip_samples = rng.beta(alpha, beta, size=n_samples)

    # Outs = strikeouts + outs on BIP
    outs_rate = k_rate_posterior + bip_rate * (1.0 - babip_samples)
    return np.clip(outs_rate, 0.0, 1.0)


def derive_batter_hr_rate(
    hr_per_fb_posterior: np.ndarray,
    fb_rate_posterior: np.ndarray,
    k_rate_posterior: np.ndarray,
    bb_rate_posterior: np.ndarray,
    hbp_rate: float = BATTER_LEAGUE_HBP_RATE,
) -> np.ndarray:
    """Derive batter HR/PA from HR/FB x FB% x BIP% component posteriors.

    Identity: HR/PA = (HR/FB) x (FB/BIP) x (BIP/PA)
    where BIP/PA = 1 - K% - BB% - HBP%.

    Small approximation: FB/BIP_with_LA ≈ FB/BIP (>95% coverage
    in Statcast era, dwarfed by posterior uncertainty).

    Parameters
    ----------
    hr_per_fb_posterior : np.ndarray
        Posterior samples of HR/FB rate.
    fb_rate_posterior : np.ndarray
        Posterior samples of FB% (FB / BIP_with_LA).
    k_rate_posterior : np.ndarray
        Posterior samples of batter K%.
    bb_rate_posterior : np.ndarray
        Posterior samples of batter BB%.
    hbp_rate : float
        HBP rate (constant).

    Returns
    -------
    np.ndarray
        Posterior samples of HR/PA rate.
    """
    bip_rate = np.clip(
        1.0 - k_rate_posterior - bb_rate_posterior - hbp_rate, 0.0, 1.0,
    )
    hr_rate = hr_per_fb_posterior * fb_rate_posterior * bip_rate
    return np.clip(hr_rate, 0.0, 1.0)


def derive_batter_h_rate(
    k_rate_posterior: np.ndarray,
    bb_rate_posterior: np.ndarray,
    hr_rate_posterior: np.ndarray,
    observed_babip: float | None = None,
    bip: float = 0.0,
    hbp_rate: float = BATTER_LEAGUE_HBP_RATE,
    league_babip: float = BATTER_LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Derive batter H/PA rate posterior from Bayesian K%, BB%, HR% posteriors.

    Same decomposition as the pitcher side but from the batter perspective:
        BIP/PA = 1 - K% - BB% - HR% - HBP%
        H/PA   = BIP/PA * BABIP + HR%
    (because Hits includes home runs, but BABIP excludes HR from numerator)

    Parameters
    ----------
    k_rate_posterior : np.ndarray
        Posterior samples of batter K%.
    bb_rate_posterior : np.ndarray
        Posterior samples of batter BB%.
    hr_rate_posterior : np.ndarray
        Posterior samples of batter HR/PA.
    observed_babip : float or None
        Batter's observed BABIP.
    bip : float
        Balls in play for reliability weighting.
    hbp_rate : float
        HBP rate.
    league_babip : float
        League-average BABIP (batter perspective).
    babip_prior_n : float
        Effective prior sample size for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray
        Posterior samples of H/PA rate.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(k_rate_posterior)

    bip_rate = 1.0 - k_rate_posterior - bb_rate_posterior - hr_rate_posterior - hbp_rate
    bip_rate = np.clip(bip_rate, 0.0, 1.0)

    alpha, beta = _shrink_babip(observed_babip, bip, league_babip, babip_prior_n)
    babip_samples = rng.beta(alpha, beta, size=n_samples)

    # H/PA = hits on BIP + home runs
    h_rate = bip_rate * babip_samples + hr_rate_posterior
    return np.clip(h_rate, 0.0, 1.0)


def derive_pitcher_rates_batch(
    pitcher_posteriors: dict[str, dict[int, np.ndarray]],
    pitcher_babip_data: dict[int, tuple[float | None, float]],
    stat: str,
    hbp_rate: float = LEAGUE_HBP_RATE,
    league_babip: float = LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Derive H/BF or Outs/BF posteriors for a batch of pitchers.

    Requires K%, BB%, and HR/BF Bayesian posteriors to be available.

    Parameters
    ----------
    pitcher_posteriors : dict
        {rate_col: {pitcher_id: samples}}.
        Must contain "k_rate", "bb_rate", "hr_per_bf".
    pitcher_babip_data : dict
        {pitcher_id: (observed_babip, bip_count)}.
    stat : str
        "h_per_bf" or "outs_per_bf".
    hbp_rate : float
        League HBP rate.
    league_babip : float
        League BABIP.
    babip_prior_n : float
        Prior strength for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    dict[int, np.ndarray]
        {pitcher_id: rate_samples}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    k_posteriors = pitcher_posteriors.get("k_rate", {})
    bb_posteriors = pitcher_posteriors.get("bb_rate", {})
    hr_posteriors = pitcher_posteriors.get("hr_per_bf", {})

    if not k_posteriors or not bb_posteriors or not hr_posteriors:
        logger.warning(
            "Missing Bayesian posteriors for derived %s — need k_rate, "
            "bb_rate, hr_per_bf. Available: %s",
            stat, list(pitcher_posteriors.keys()),
        )
        return {}

    # Find pitchers with all three posteriors
    common_ids = (
        set(k_posteriors.keys())
        & set(bb_posteriors.keys())
        & set(hr_posteriors.keys())
    )

    derive_fn = derive_pitcher_h_rate if stat == "h_per_bf" else derive_pitcher_outs_rate
    results: dict[int, np.ndarray] = {}

    for pid in common_ids:
        k_samples = k_posteriors[pid]
        bb_samples = bb_posteriors[pid]
        hr_samples = hr_posteriors[pid]

        # Align sample sizes (take minimum)
        n = min(len(k_samples), len(bb_samples), len(hr_samples))
        k_samples = k_samples[:n]
        bb_samples = bb_samples[:n]
        hr_samples = hr_samples[:n]

        obs_babip, bip_count = pitcher_babip_data.get(pid, (None, 0.0))

        results[pid] = derive_fn(
            k_rate_posterior=k_samples,
            bb_rate_posterior=bb_samples,
            hr_rate_posterior=hr_samples,
            observed_babip=obs_babip,
            bip=bip_count,
            hbp_rate=hbp_rate,
            league_babip=league_babip,
            babip_prior_n=babip_prior_n,
            rng=rng,
        )

    logger.info(
        "Derived %s posteriors for %d/%d pitchers (from Bayesian K%%/BB%%/HR%%)",
        stat, len(results), len(common_ids),
    )
    return results


def derive_batter_rates_batch(
    batter_posteriors: dict[str, dict[int, np.ndarray]],
    batter_babip_data: dict[int, tuple[float | None, float]],
    hbp_rate: float = BATTER_LEAGUE_HBP_RATE,
    league_babip: float = BATTER_LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Derive H/PA posteriors for a batch of batters.

    Parameters
    ----------
    batter_posteriors : dict
        {rate_col: {batter_id: samples}}.
        Must contain "k_rate", "bb_rate", and one of "hr_rate"/"hr_per_pa".
    batter_babip_data : dict
        {batter_id: (observed_babip, bip_count)}.
    hbp_rate : float
        League HBP rate.
    league_babip : float
        League BABIP.
    babip_prior_n : float
        Prior strength for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    dict[int, np.ndarray]
        {batter_id: h_rate_samples}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    k_posteriors = batter_posteriors.get("k_rate", {})
    bb_posteriors = batter_posteriors.get("bb_rate", {})
    # Batter HR rate may be keyed as "hr_rate" or "hr_per_pa"
    hr_posteriors = batter_posteriors.get("hr_rate", {})
    if not hr_posteriors:
        hr_posteriors = batter_posteriors.get("hr_per_pa", {})

    if not k_posteriors or not bb_posteriors or not hr_posteriors:
        logger.warning(
            "Missing Bayesian posteriors for derived batter H rate — "
            "need k_rate, bb_rate, hr_rate. Available: %s",
            list(batter_posteriors.keys()),
        )
        return {}

    common_ids = (
        set(k_posteriors.keys())
        & set(bb_posteriors.keys())
        & set(hr_posteriors.keys())
    )

    results: dict[int, np.ndarray] = {}
    for bid in common_ids:
        k_samples = k_posteriors[bid]
        bb_samples = bb_posteriors[bid]
        hr_samples = hr_posteriors[bid]

        n = min(len(k_samples), len(bb_samples), len(hr_samples))
        k_samples = k_samples[:n]
        bb_samples = bb_samples[:n]
        hr_samples = hr_samples[:n]

        obs_babip, bip_count = batter_babip_data.get(bid, (None, 0.0))

        results[bid] = derive_batter_h_rate(
            k_rate_posterior=k_samples,
            bb_rate_posterior=bb_samples,
            hr_rate_posterior=hr_samples,
            observed_babip=obs_babip,
            bip=bip_count,
            hbp_rate=hbp_rate,
            league_babip=league_babip,
            babip_prior_n=babip_prior_n,
            rng=rng,
        )

    logger.info(
        "Derived batter H/PA posteriors for %d/%d batters",
        len(results), len(common_ids),
    )
    return results


def derive_batter_hr_rate_batch(
    batter_posteriors: dict[str, dict[int, np.ndarray]],
    hbp_rate: float = BATTER_LEAGUE_HBP_RATE,
) -> dict[int, np.ndarray]:
    """Derive HR/PA posteriors from HR/FB x FB% x BIP% for a batch of batters.

    Requires ``hr_per_fb``, ``fb_rate``, ``k_rate``, and ``bb_rate``
    posterior samples in *batter_posteriors*.

    Parameters
    ----------
    batter_posteriors : dict
        {rate_col: {batter_id: samples}}.
    hbp_rate : float
        League HBP rate.

    Returns
    -------
    dict[int, np.ndarray]
        {batter_id: hr_rate_samples}.
    """
    hr_fb = batter_posteriors.get("hr_per_fb", {})
    fb = batter_posteriors.get("fb_rate", {})
    k = batter_posteriors.get("k_rate", {})
    bb = batter_posteriors.get("bb_rate", {})

    if not hr_fb or not fb or not k or not bb:
        logger.warning(
            "Missing posteriors for composed batter HR rate — need "
            "hr_per_fb, fb_rate, k_rate, bb_rate. Available: %s",
            list(batter_posteriors.keys()),
        )
        return {}

    common_ids = set(hr_fb) & set(fb) & set(k) & set(bb)

    results: dict[int, np.ndarray] = {}
    for bid in common_ids:
        n = min(len(hr_fb[bid]), len(fb[bid]), len(k[bid]), len(bb[bid]))
        results[bid] = derive_batter_hr_rate(
            hr_per_fb_posterior=hr_fb[bid][:n],
            fb_rate_posterior=fb[bid][:n],
            k_rate_posterior=k[bid][:n],
            bb_rate_posterior=bb[bid][:n],
            hbp_rate=hbp_rate,
        )

    logger.info(
        "Derived batter HR/PA posteriors for %d/%d batters "
        "(HR/FB x FB%% x BIP%% composition)",
        len(results), len(common_ids),
    )
    return results


# ---------------------------------------------------------------------------
# ERA/FIP derivation constants (2018-2025 pooled)
# ---------------------------------------------------------------------------
LEAGUE_ERA: float = 4.17
LEAGUE_FIP_CONSTANT: float = 3.20
LEAGUE_GB_PCT: float = 0.446
ERA_FIP_GAP_EFFECTIVE_N: float = 15.0
GB_ERA_COEFFICIENT: float = -0.035  # per 1pp GB% above league


def derive_pitcher_fip(
    k_rate_posterior: np.ndarray,
    bb_rate_posterior: np.ndarray,
    hr_rate_posterior: np.ndarray,
    outs_rate_posterior: np.ndarray,
    hbp_rate: float = LEAGUE_HBP_RATE,
    fip_constant: float = LEAGUE_FIP_CONSTANT,
) -> np.ndarray:
    """Derive FIP posterior from Bayesian rate posteriors.

    FIP = (13*HR/BF + 3*(BB/BF + HBP/BF) - 2*K/BF) / (IP/BF) + cFIP

    where IP/BF = Outs/BF / 3.

    Parameters
    ----------
    k_rate_posterior : np.ndarray
        Posterior samples of K%.
    bb_rate_posterior : np.ndarray
        Posterior samples of BB%.
    hr_rate_posterior : np.ndarray
        Posterior samples of HR/BF.
    outs_rate_posterior : np.ndarray
        Posterior samples of Outs/BF.
    hbp_rate : float
        HBP rate (fixed — very stable).
    fip_constant : float
        League FIP constant.

    Returns
    -------
    np.ndarray
        Posterior samples of FIP.
    """
    ip_per_bf = outs_rate_posterior / 3.0
    ip_per_bf = np.clip(ip_per_bf, 0.01, None)

    numerator = (
        13.0 * hr_rate_posterior
        + 3.0 * (bb_rate_posterior + hbp_rate)
        - 2.0 * k_rate_posterior
    )
    fip = numerator / ip_per_bf + fip_constant

    return np.clip(fip, 0.0, 15.0)


def _shrink_era_fip_gap(
    observed_gap: float | None,
    ip: float,
    gb_pct: float | None,
    league_gb: float = LEAGUE_GB_PCT,
    xgb_prior_mean: float | None = None,
) -> tuple[float, float]:
    """Conjugate Normal shrinkage of ERA-FIP gap toward informed prior.

    ERA-FIP gap has YoY r~0.15, warranting heavy shrinkage.  The prior
    mean comes from either an XGBoost model (if available) or a simple
    GB%-linear formula (high GB% → negative gap).

    Parameters
    ----------
    observed_gap : float or None
        Pitcher's observed ERA minus FIP.
    ip : float
        Innings pitched (reliability weight).
    gb_pct : float or None
        Pitcher's ground ball percentage (fraction, e.g. 0.50).
    league_gb : float
        League average GB%.
    xgb_prior_mean : float or None
        XGBoost-predicted ERA-FIP gap prior.  When provided, overrides
        the simple GB%-linear formula.

    Returns
    -------
    tuple[float, float]
        (gap_mean, gap_sd) for the shrunken ERA-FIP gap.
    """
    # Prior mean: XGBoost prediction > GB%-linear fallback
    if xgb_prior_mean is not None and not np.isnan(xgb_prior_mean):
        prior_mean = xgb_prior_mean
    elif gb_pct is not None and not np.isnan(gb_pct):
        pp_above = (gb_pct - league_gb) * 100.0
        prior_mean = GB_ERA_COEFFICIENT * pp_above
    else:
        prior_mean = 0.0

    prior_sd = 0.50  # population SD of ERA-FIP gap

    if observed_gap is None or np.isnan(observed_gap) or ip < 10:
        return prior_mean, prior_sd

    # Conjugate Normal: prior weight = ERA_FIP_GAP_EFFECTIVE_N (IP units),
    # observation weight = ip.
    total = ERA_FIP_GAP_EFFECTIVE_N + ip
    gap_mean = (
        ERA_FIP_GAP_EFFECTIVE_N * prior_mean + ip * observed_gap
    ) / total
    gap_sd = prior_sd / np.sqrt(1.0 + ip / ERA_FIP_GAP_EFFECTIVE_N)

    return gap_mean, gap_sd


def derive_pitcher_era(
    fip_posterior: np.ndarray,
    observed_era_fip_gap: float | None,
    ip: float,
    gb_pct: float | None,
    rng: np.random.Generator | None = None,
    xgb_gap_prior: float | None = None,
) -> np.ndarray:
    """Derive ERA posterior by adding shrunken ERA-FIP gap to FIP posterior.

    ERA = FIP + shrunken(observed_ERA - observed_FIP)

    Parameters
    ----------
    fip_posterior : np.ndarray
        Posterior samples of FIP.
    observed_era_fip_gap : float or None
        Pitcher's observed ERA minus FIP.
    ip : float
        Innings pitched (for gap shrinkage reliability).
    gb_pct : float or None
        Pitcher's ground ball percentage.
    rng : np.random.Generator or None
    xgb_gap_prior : float or None
        XGBoost-predicted ERA-FIP gap prior.  Overrides GB%-linear formula.

    Returns
    -------
    np.ndarray
        Posterior samples of ERA.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gap_mean, gap_sd = _shrink_era_fip_gap(
        observed_era_fip_gap, ip, gb_pct,
        xgb_prior_mean=xgb_gap_prior,
    )

    n_samples = len(fip_posterior)
    gap_samples = rng.normal(gap_mean, max(gap_sd, 0.01), size=n_samples)

    era = fip_posterior + gap_samples
    return np.clip(era, 0.0, 15.0)


def derive_pitcher_fip_batch(
    pitcher_posteriors: dict[str, dict[int, np.ndarray]],
    pitcher_babip_data: dict[int, tuple[float | None, float]],
    hbp_rate: float = LEAGUE_HBP_RATE,
    fip_constant: float = LEAGUE_FIP_CONSTANT,
    league_babip: float = LEAGUE_BABIP,
    babip_prior_n: float = 30.0,
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Derive FIP posteriors for a batch of pitchers.

    Internally derives Outs/BF posteriors from K/BB/HR posteriors
    (using league BABIP), then computes FIP from all four rate posteriors.

    Parameters
    ----------
    pitcher_posteriors : dict
        {rate_col: {pitcher_id: samples}}.
        Must contain "k_rate", "bb_rate", "hr_per_bf".
    pitcher_babip_data : dict
        {pitcher_id: (observed_babip, bip_count)}.
    hbp_rate : float
        League HBP rate.
    fip_constant : float
        League FIP constant.
    league_babip : float
        League BABIP.
    babip_prior_n : float
        Prior strength for BABIP shrinkage.
    rng : np.random.Generator or None

    Returns
    -------
    dict[int, np.ndarray]
        {pitcher_id: fip_samples}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Derive outs_per_bf posteriors
    outs_posteriors = derive_pitcher_rates_batch(
        pitcher_posteriors, pitcher_babip_data, "outs_per_bf",
        hbp_rate=hbp_rate, league_babip=league_babip,
        babip_prior_n=babip_prior_n, rng=rng,
    )

    k_posteriors = pitcher_posteriors.get("k_rate", {})
    bb_posteriors = pitcher_posteriors.get("bb_rate", {})
    hr_posteriors = pitcher_posteriors.get("hr_per_bf", {})

    common_ids = (
        set(k_posteriors.keys())
        & set(bb_posteriors.keys())
        & set(hr_posteriors.keys())
        & set(outs_posteriors.keys())
    )

    results: dict[int, np.ndarray] = {}
    for pid in common_ids:
        k_s = k_posteriors[pid]
        bb_s = bb_posteriors[pid]
        hr_s = hr_posteriors[pid]
        outs_s = outs_posteriors[pid]

        n = min(len(k_s), len(bb_s), len(hr_s), len(outs_s))
        results[pid] = derive_pitcher_fip(
            k_s[:n], bb_s[:n], hr_s[:n], outs_s[:n],
            hbp_rate=hbp_rate, fip_constant=fip_constant,
        )

    logger.info("Derived FIP posteriors for %d pitchers", len(results))
    return results


def derive_pitcher_era_batch(
    fip_posteriors: dict[int, np.ndarray],
    pitcher_era_fip_data: dict[int, tuple[float | None, float, float | None]],
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Derive ERA posteriors for a batch of pitchers.

    Parameters
    ----------
    fip_posteriors : dict
        {pitcher_id: fip_samples} from ``derive_pitcher_fip_batch``.
    pitcher_era_fip_data : dict
        {pitcher_id: (observed_era_fip_gap, ip, gb_pct)}.
    rng : np.random.Generator or None

    Returns
    -------
    dict[int, np.ndarray]
        {pitcher_id: era_samples}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results: dict[int, np.ndarray] = {}
    for pid, fip_post in fip_posteriors.items():
        gap, ip, gb_pct = pitcher_era_fip_data.get(pid, (None, 0.0, None))
        results[pid] = derive_pitcher_era(fip_post, gap, ip, gb_pct, rng=rng)

    logger.info("Derived ERA posteriors for %d pitchers", len(results))
    return results
