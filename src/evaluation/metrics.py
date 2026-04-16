"""
Enhanced evaluation metrics for game-level prop predictions.

Provides log loss, CRPS, ECE, MCE, and temperature scaling diagnostics
that go beyond Brier score for full-distribution evaluation.

Also includes shared coverage, PPC extraction, and CRPS comparison
utilities used by hitter/pitcher/counting/season-sim backtests.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Shared coverage utilities
# ---------------------------------------------------------------------------


def compute_coverage(
    actual: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    """Fraction of actuals within [lo, hi] interval.

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    lo : np.ndarray
        Lower bounds of credible/confidence interval.
    hi : np.ndarray
        Upper bounds of credible/confidence interval.

    Returns
    -------
    float
        Empirical coverage fraction in [0, 1].
    """
    return float(np.mean((actual >= lo) & (actual <= hi)))


def compute_coverage_from_sd(
    actual: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    level: float = 0.80,
) -> float:
    """Coverage using normal approximation at given confidence level.

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    mean : np.ndarray
        Predicted means.
    sd : np.ndarray
        Predicted standard deviations.
    level : float
        Confidence level (e.g. 0.80 for 80% CI).

    Returns
    -------
    float
        Empirical coverage fraction in [0, 1].
    """
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    lo = mean - z * sd
    hi = mean + z * sd
    return compute_coverage(actual, lo, hi)


def compute_coverage_levels(
    actual: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90),
) -> dict[str, float]:
    """Multi-level normal-approx coverage.

    Returns ``{percent: coverage}`` where ``percent`` is an integer-percent
    string (e.g. ``"50"``, ``"80"``, ``"90"``) matching the column-naming
    convention used by the game-level backtests.
    """
    return {
        str(int(round(level * 100))): compute_coverage_from_sd(actual, mean, sd, level=level)
        for level in levels
    }


# ---------------------------------------------------------------------------
# Per-line binary prop metrics
# ---------------------------------------------------------------------------


def _default_prob_col(line: float) -> str:
    """Default ``p_over_<line>`` column template used by game prop backtests."""
    return f"p_over_{line:.1f}".replace(".", "_")


def compute_per_line_binary_metrics(
    predictions: pd.DataFrame,
    actual_col: str,
    lines: list[float],
    prob_col_fn=_default_prob_col,
    clip: bool = False,
    skip_degenerate: bool = False,
) -> dict[str, Any]:
    """Per-line Brier + log loss for game-prop style predictions.

    For each line, reads ``prob_col_fn(line)`` from ``predictions`` as the
    model's P(actual > line) and ``predictions[actual_col] > line`` as the
    binary outcome, then computes Brier score and log loss.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain ``actual_col`` and one prob column per line.
    actual_col : str
        Column name holding the observed count (e.g. ``"actual_k"``).
    lines : list[float]
        Prop lines to evaluate (e.g. ``[3.5, 4.5, 5.5]``).
    prob_col_fn : callable(line) -> str
        Maps a line to its prob column name. Defaults to
        ``"p_over_<line>"`` with a ``.`` → ``_`` substitution.
    clip : bool
        If True, clip probability values to ``[0, 1]`` before scoring.
        Preserves ``game_prop_validation`` defensive behavior.
    skip_degenerate : bool
        If True, skip lines where all observed outcomes are the same
        class (``y_true.std() == 0``), matching ``game_prop_validation``.

    Returns
    -------
    dict with keys ``brier_scores`` (per-line), ``log_losses`` (per-line),
    ``avg_brier``, ``avg_log_loss``.
    """
    from sklearn.metrics import brier_score_loss

    brier_scores: dict[float, float] = {}
    log_losses: dict[float, float] = {}

    for line in lines:
        col = prob_col_fn(line)
        if col not in predictions.columns:
            continue
        y_true = (predictions[actual_col] > line).astype(float).values
        y_prob = predictions[col].values
        if clip:
            y_prob = np.clip(y_prob, 0, 1)
        if skip_degenerate and y_true.std() == 0:
            continue
        brier_scores[line] = float(brier_score_loss(y_true, y_prob))
        log_losses[line] = compute_log_loss(y_prob, y_true)

    avg_brier = (
        float(np.mean(list(brier_scores.values()))) if brier_scores else np.nan
    )
    avg_log_loss = (
        float(np.mean(list(log_losses.values()))) if log_losses else np.nan
    )
    return {
        "brier_scores": brier_scores,
        "log_losses": log_losses,
        "avg_brier": avg_brier,
        "avg_log_loss": avg_log_loss,
    }


def compute_per_line_calibration(
    predictions: pd.DataFrame,
    actual_col: str,
    lines: list[float],
    prob_col_fn=_default_prob_col,
    clip: bool = False,
    skip_degenerate: bool = False,
    include_mce: bool = False,
) -> dict[str, Any]:
    """Per-line ECE / MCE / temperature for game-prop style predictions.

    Parameters mirror ``compute_per_line_binary_metrics``.

    Returns
    -------
    dict with keys ``ece_per_line``, ``temperature_per_line``, plus
    ``mce_per_line`` when ``include_mce=True``. ``all_probs`` and
    ``all_outcomes`` arrays are also returned so callers can compute
    pooled diagnostics.
    """
    ece_per_line: dict[float, float] = {}
    mce_per_line: dict[float, float] = {}
    temp_per_line: dict[float, float] = {}
    all_probs: list[np.ndarray] = []
    all_outcomes: list[np.ndarray] = []

    for line in lines:
        col = prob_col_fn(line)
        if col not in predictions.columns:
            continue
        y_true = (predictions[actual_col] > line).astype(float).values
        y_prob = predictions[col].values
        if clip:
            y_prob = np.clip(y_prob, 0, 1)
        if skip_degenerate and y_true.std() == 0:
            continue
        ece_per_line[line] = compute_ece(y_prob, y_true)
        if include_mce:
            mce_per_line[line] = compute_mce(y_prob, y_true)
        temp_per_line[line] = compute_temperature(y_prob, y_true)
        all_probs.append(y_prob)
        all_outcomes.append(y_true)

    result: dict[str, Any] = {
        "ece_per_line": ece_per_line,
        "temperature_per_line": temp_per_line,
        "all_probs": all_probs,
        "all_outcomes": all_outcomes,
    }
    if include_mce:
        result["mce_per_line"] = mce_per_line
    return result


# ---------------------------------------------------------------------------
# PPC extraction
# ---------------------------------------------------------------------------


def extract_ppc_summary(
    trace: Any,
    data: dict[str, Any],
    cfg: Any,
    label: str = "",
) -> dict[str, Any] | None:
    """Extract and summarize posterior predictive checks from a fitted trace.

    Parameters
    ----------
    trace : InferenceData
        Fitted ArviZ trace with optional ``posterior_predictive`` group.
    data : dict
        Model data dict. Must contain ``"counts"`` (binomial) or
        ``"y_obs"`` (normal) for comparison.
    cfg : object
        Stat config with a ``likelihood`` attribute (``"binomial"`` or
        ``"normal"``).
    label : str
        Descriptive label for log messages.

    Returns
    -------
    dict | None
        PPC calibration summary, or None if PPC is unavailable.
    """
    try:
        if not (hasattr(trace, "posterior_predictive")
                and "obs" in trace.posterior_predictive):
            return None

        ppc_obs = trace.posterior_predictive["obs"].values  # (chains, draws, n_obs)
        n_chains, n_draws_per, n_obs = ppc_obs.shape
        ppc_flat = ppc_obs.reshape(n_chains * n_draws_per, n_obs)

        if cfg.likelihood == "binomial":
            observed_for_ppc = data["counts"]
        else:
            observed_for_ppc = data["y_obs"]

        pvalues = compute_ppc_pvalues(ppc_flat, observed_for_ppc)
        summary = summarize_ppc_calibration(pvalues)
        if label:
            logger.info(
                "%s PPC: KS stat=%.3f (p=%.3f), outliers=%.1f%%",
                label, summary["ks_stat"], summary["ks_pvalue"],
                summary["pct_outliers"],
            )
        return summary
    except Exception as e:
        logger.warning("PPC computation failed for %s: %s", label, e)
        return None


# ---------------------------------------------------------------------------
# CRPS comparison (Bayes vs Marcel)
# ---------------------------------------------------------------------------


def compute_bayes_vs_marcel_crps(
    comp: pd.DataFrame,
    stat: str,
    proj_samples: dict[int, np.ndarray],
    id_col: str,
    weight_col: str,
    rng: np.random.Generator,
    compute_crps_single_fn: Any,
    likelihood: str = "binomial",
) -> tuple[float, float]:
    """Compute Bayes and Marcel CRPS for a comparison DataFrame.

    Parameters
    ----------
    comp : pd.DataFrame
        Comparison DataFrame with ``actual_{stat}``, ``marcel_{stat}``,
        ``{weight_col}``, and ``{id_col}`` columns.
    stat : str
        Stat key (e.g. ``"k_rate"``).
    proj_samples : dict[int, np.ndarray]
        Mapping of player ID -> posterior samples.
    id_col : str
        Player ID column name (e.g. ``"batter_id"``).
    weight_col : str
        Weight column name (e.g. ``"weighted_pa"`` or ``"weighted_bf"``).
    rng : np.random.Generator
        Random number generator for Marcel draws.
    compute_crps_single_fn : callable
        Function ``(observed, samples) -> float``.
    likelihood : str
        ``"binomial"`` uses Beta draws for Marcel; ``"normal"`` uses
        Normal draws.

    Returns
    -------
    tuple[float, float]
        ``(bayes_crps, marcel_crps)`` mean values.
    """
    bayes_crps_vals: list[float] = []
    marcel_crps_vals: list[float] = []

    for _, row in comp.iterrows():
        pid = int(row[id_col])
        act = float(row[f"actual_{stat}"])

        # Bayes CRPS
        if pid in proj_samples:
            bayes_crps_vals.append(compute_crps_single_fn(act, proj_samples[pid]))

        # Marcel CRPS via Beta or Normal draws
        m_rate = float(row[f"marcel_{stat}"])
        w = float(row[weight_col])
        if likelihood == "binomial":
            a = 1.0 + m_rate * max(w, 1.0)
            b = 1.0 + (1.0 - m_rate) * max(w, 1.0)
            marcel_draws = rng.beta(a, b, size=4000)
        else:
            marcel_draws = rng.normal(m_rate, max(0.01, abs(m_rate) * 0.1), size=4000)
        marcel_crps_vals.append(compute_crps_single_fn(act, marcel_draws))

    bayes_crps = float(np.mean(bayes_crps_vals)) if bayes_crps_vals else np.nan
    marcel_crps = float(np.mean(marcel_crps_vals)) if marcel_crps_vals else np.nan
    return bayes_crps, marcel_crps


# ---------------------------------------------------------------------------
# Regression metrics for counting stat backtests
# ---------------------------------------------------------------------------


def compute_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lo_80: np.ndarray | None = None,
    hi_80: np.ndarray | None = None,
    lo_95: np.ndarray | None = None,
    hi_95: np.ndarray | None = None,
    include_mape: bool = False,
) -> dict[str, float]:
    """Compute regression metrics for counting stat predictions.

    Shared implementation for ``counting_backtest._compute_metrics``
    (with ``include_mape=True``) and ``season_sim_backtest._compute_metrics``
    (with ``include_mape=False``, computes bias instead).

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    predicted : np.ndarray
        Predicted values.
    lo_80, hi_80 : np.ndarray | None
        Optional 80% interval bounds.
    lo_95, hi_95 : np.ndarray | None
        Optional 95% interval bounds.
    include_mape : bool
        If True, compute MAPE. If False, compute bias.

    Returns
    -------
    dict[str, float]
        Keys always include ``n`` and ``correlation``.
        If ``include_mape``, includes ``mape``.
        Otherwise includes ``bias``.
        Optionally includes ``coverage_80`` and ``coverage_95``.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[valid], predicted[valid]
    n = len(a)
    if n == 0:
        return {"n": 0} if not include_mape else {}

    # Correlation
    if np.std(a) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(a, p)[0, 1])
    else:
        corr = 0.0

    metrics: dict[str, float] = {"n": n, "correlation": corr}

    if include_mape:
        nonzero = a > 0
        if nonzero.sum() > 0:
            metrics["mape"] = float(np.mean(np.abs((a - p)[nonzero]) / a[nonzero]))
        else:
            metrics["mape"] = float("nan")
    else:
        metrics["bias"] = float(np.mean(p - a))

    # Coverage
    if lo_80 is not None and hi_80 is not None:
        lo_v, hi_v = np.asarray(lo_80)[valid], np.asarray(hi_80)[valid]
        metrics["coverage_80"] = compute_coverage(a, lo_v, hi_v)
    if lo_95 is not None and hi_95 is not None:
        lo_v, hi_v = np.asarray(lo_95)[valid], np.asarray(hi_95)[valid]
        metrics["coverage_95"] = compute_coverage(a, lo_v, hi_v)

    return metrics
