"""
Enhanced evaluation metrics for game-level prop predictions.

Provides log loss, CRPS, ECE, MCE, and temperature scaling diagnostics
that go beyond Brier score for full-distribution evaluation.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import kstest

logger = logging.getLogger(__name__)


def compute_log_loss(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Binary log loss (negative log-likelihood).

    Penalizes confidently wrong predictions much harder than Brier score,
    making it more relevant for betting applications where a confident
    wrong bet costs more.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (n,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (n,).
    eps : float
        Clipping bound to avoid log(0). Probabilities are clipped
        to [eps, 1 - eps].

    Returns
    -------
    float
        Log loss value. Lower is better. Perfect = 0, random = ln(2) ~ 0.693.
    """
    n = len(predicted_probs)
    if n == 0:
        return np.nan

    p = np.clip(np.asarray(predicted_probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(actual_outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_crps_single(observed: float, samples: np.ndarray) -> float:
    """CRPS for a single observation against MC samples.

    Parameters
    ----------
    observed : float
        Single actual outcome.
    samples : np.ndarray
        MC samples, shape (n_draws,).

    Returns
    -------
    float
        CRPS value.
    """
    abs_diff = np.mean(np.abs(samples - observed))
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    weights = 2 * np.arange(1, n + 1) - n - 1
    mean_abs_diff = np.sum(weights * sorted_s) / (n * n)
    return float(abs_diff - 0.5 * mean_abs_diff)


def compute_ece(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Weighted average of |accuracy - confidence| across probability bins.
    Lower is better. A perfectly calibrated model has ECE = 0.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (n,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (n,).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    n = len(predicted_probs)
    if n == 0:
        return np.nan

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for j in range(n_bins):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        if j == n_bins - 1:
            mask = (predicted_probs >= lo) & (predicted_probs <= hi)
        else:
            mask = (predicted_probs >= lo) & (predicted_probs < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = float(np.mean(predicted_probs[mask]))
        avg_accuracy = float(np.mean(actual_outcomes[mask]))
        ece += (n_in_bin / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_mce(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error.

    Maximum |accuracy - confidence| across any bin. Identifies the
    worst-calibrated region.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (n,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (n,).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        MCE value in [0, 1].
    """
    n = len(predicted_probs)
    if n == 0:
        return np.nan

    bin_edges = np.linspace(0, 1, n_bins + 1)
    max_ce = 0.0

    for j in range(n_bins):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        if j == n_bins - 1:
            mask = (predicted_probs >= lo) & (predicted_probs <= hi)
        else:
            mask = (predicted_probs >= lo) & (predicted_probs < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = float(np.mean(predicted_probs[mask]))
        avg_accuracy = float(np.mean(actual_outcomes[mask]))
        max_ce = max(max_ce, abs(avg_accuracy - avg_confidence))

    return float(max_ce)


def compute_temperature(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> float:
    """Fit temperature scaling parameter T.

    T > 1 means the model is overconfident (probabilities too extreme).
    T < 1 means the model is underconfident (probabilities too moderate).
    T = 1 means well-calibrated.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (n,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (n,).

    Returns
    -------
    float
        Fitted temperature T.
    """
    n = len(predicted_probs)
    if n == 0:
        return np.nan

    # Clip to avoid infinities in logit
    clipped = np.clip(predicted_probs, 1e-6, 1 - 1e-6)
    raw_logits = logit(clipped)

    def neg_log_likelihood(T: float) -> float:
        scaled_probs = expit(raw_logits / T)
        scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
        ll = actual_outcomes * np.log(scaled_probs) + \
             (1 - actual_outcomes) * np.log(1 - scaled_probs)
        return -np.mean(ll)

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def calibrate_posterior_samples(
    samples: np.ndarray,
    calibration_t: float,
) -> np.ndarray:
    """Scale posterior sample spread by calibration_t on logit scale.

    Preserves the posterior mean while adjusting the spread of the
    distribution.  Used to correct systematic over- or under-confidence
    identified in walk-forward backtesting.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples on rate scale [0, 1].
    calibration_t : float
        Scale factor for posterior spread.
        T < 1.0 narrows intervals (posterior was too wide / over-coverage).
        T > 1.0 widens intervals (posterior was too narrow / under-coverage).
        T = 1.0 no change.

    Returns
    -------
    np.ndarray
        Calibrated posterior samples.
    """
    if abs(calibration_t - 1.0) < 1e-6 or len(samples) < 2:
        return samples

    eps = np.clip(samples, 1e-6, 1 - 1e-6)
    logit_s = logit(eps)
    mu = np.mean(logit_s)
    logit_scaled = mu + (logit_s - mu) * calibration_t
    return expit(logit_scaled)


def compute_posterior_calibration_t(
    coverage_80: float,
    target_80: float = 0.80,
) -> float:
    """Derive calibration T from observed 80% CI coverage.

    If the 80% credible interval has empirical coverage > target (too wide),
    T < 1.0 will narrow the intervals.  If coverage < target (too narrow),
    T > 1.0 will widen them.

    Uses the analytical z-ratio for approximate-Normal posteriors:
        T = z(target_coverage) / z(actual_coverage)

    Parameters
    ----------
    coverage_80 : float
        Observed empirical coverage of the 80% CI (e.g., 0.88).
    target_80 : float
        Target coverage (default 0.80).

    Returns
    -------
    float
        Calibration T value.
    """
    from scipy.stats import norm

    if coverage_80 <= 0.01 or coverage_80 >= 0.99:
        return 1.0

    z_target = norm.ppf(0.5 + target_80 / 2)  # 1.282 for 80%
    z_actual = norm.ppf(0.5 + coverage_80 / 2)

    if z_actual <= 0.01:
        return 1.0

    return float(z_target / z_actual)


def compute_sharpness(predicted_probs: np.ndarray) -> dict[str, float]:
    """Measure how decisive/actionable a model's probability predictions are.

    A model outputting P(over)=0.50 on everything is useless for betting
    even if perfectly calibrated. Sharpness quantifies how often the model
    produces strong signals away from 50/50.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities in [0, 1], shape (n,).

    Returns
    -------
    dict[str, float]
        mean_confidence : Mean of |p - 0.5| (higher = sharper).
        pct_actionable_60 : % of predictions with p > 0.60 or p < 0.40.
        pct_actionable_65 : % of predictions with p > 0.65 or p < 0.35.
        pct_actionable_70 : % of predictions with p > 0.70 or p < 0.30.
        entropy : Average binary entropy (lower = sharper).
    """
    p = np.asarray(predicted_probs, dtype=float)
    n = len(p)
    if n == 0:
        return {
            "mean_confidence": np.nan,
            "pct_actionable_60": np.nan,
            "pct_actionable_65": np.nan,
            "pct_actionable_70": np.nan,
            "entropy": np.nan,
        }

    # Mean confidence: average distance from 0.5
    mean_confidence = float(np.mean(np.abs(p - 0.5)))

    # Actionable percentages at various thresholds
    pct_actionable_60 = float(np.mean((p > 0.60) | (p < 0.40)) * 100)
    pct_actionable_65 = float(np.mean((p > 0.65) | (p < 0.35)) * 100)
    pct_actionable_70 = float(np.mean((p > 0.70) | (p < 0.30)) * 100)

    # Average binary entropy: -[p*log(p) + (1-p)*log(1-p)]
    p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
    entropy = float(np.mean(
        -(p_clipped * np.log2(p_clipped)
          + (1 - p_clipped) * np.log2(1 - p_clipped))
    ))

    return {
        "mean_confidence": mean_confidence,
        "pct_actionable_60": pct_actionable_60,
        "pct_actionable_65": pct_actionable_65,
        "pct_actionable_70": pct_actionable_70,
        "entropy": entropy,
    }


def compute_ppc_pvalues(
    trace_ppc: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Compute posterior predictive p-values per observation.

    Parameters
    ----------
    trace_ppc : np.ndarray
        Posterior predictive samples, shape (n_draws, n_obs).
        Flattened from chains × draws.
    observed : np.ndarray
        Actual observed values, shape (n_obs,).

    Returns
    -------
    np.ndarray
        P-values, shape (n_obs,). Under a well-calibrated model,
        these should be Uniform(0, 1).
    """
    observed = np.asarray(observed)
    trace_ppc = np.asarray(trace_ppc)
    # p_value[i] = fraction of draws >= observed[i]
    return np.mean(trace_ppc >= observed[np.newaxis, :], axis=0)


def summarize_ppc_calibration(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """Summarize PPC calibration from p-values.

    Parameters
    ----------
    pvalues : np.ndarray
        Per-observation p-values from ``compute_ppc_pvalues``.
    alpha : float
        Significance level for KS test.

    Returns
    -------
    dict
        ks_stat, ks_pvalue, n_outliers_low, n_outliers_high,
        pct_outliers, uniform_consistent.
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return {
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "n_outliers_low": 0,
            "n_outliers_high": 0,
            "pct_outliers": np.nan,
            "uniform_consistent": False,
        }

    ks_result = kstest(pvalues, "uniform")
    n_low = int(np.sum(pvalues < 0.025))
    n_high = int(np.sum(pvalues > 0.975))
    pct = (n_low + n_high) / n * 100

    return {
        "ks_stat": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "n_outliers_low": n_low,
        "n_outliers_high": n_high,
        "pct_outliers": float(pct),
        "uniform_consistent": bool(ks_result.pvalue >= alpha),
    }
