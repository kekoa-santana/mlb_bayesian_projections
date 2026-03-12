"""
Enhanced evaluation metrics for game-level prop predictions.

Provides CRPS, ECE, MCE, and temperature scaling diagnostics
that go beyond Brier score for full-distribution evaluation.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

logger = logging.getLogger(__name__)


def compute_crps(observed: np.ndarray, posterior_samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score for count data.

    CRPS measures the full-distribution forecast quality, not just
    a binary threshold like Brier. Lower is better.

    Parameters
    ----------
    observed : np.ndarray
        Actual outcomes, shape (n_games,).
    posterior_samples : np.ndarray
        MC samples, shape (n_games, n_draws).

    Returns
    -------
    float
        Mean CRPS across all games.
    """
    n_games = len(observed)
    if n_games == 0:
        return np.nan

    crps_values = np.empty(n_games)
    for i in range(n_games):
        samples = posterior_samples[i]
        y = observed[i]
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        abs_diff = np.mean(np.abs(samples - y))
        # For the E|X-X'| term, use sorted samples for efficiency
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        # E|X-X'| = (2/n^2) * sum_{i=1}^{n} (2i - n - 1) * x_{(i)}
        weights = 2 * np.arange(1, n + 1) - n - 1
        mean_abs_diff = np.sum(weights * sorted_s) / (n * n)
        crps_values[i] = abs_diff - 0.5 * mean_abs_diff

    return float(np.mean(crps_values))


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
