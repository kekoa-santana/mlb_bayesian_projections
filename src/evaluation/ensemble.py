"""
Bayes-Marcel ensemble for season-level projections.

Fits an optimal weight w that blends Bayesian and Marcel point estimates,
while shifting Bayesian posteriors to preserve calibration advantages
(Brier, coverage).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def fit_ensemble_weight(
    actual: np.ndarray,
    bayes_pred: np.ndarray,
    marcel_pred: np.ndarray,
) -> float:
    """Find optimal ensemble weight via grid search.

    Parameters
    ----------
    actual : np.ndarray
        Observed rates, shape (n,).
    bayes_pred : np.ndarray
        Bayesian point estimates, shape (n,).
    marcel_pred : np.ndarray
        Marcel point estimates, shape (n,).

    Returns
    -------
    float
        Optimal w in [0, 1]. w=1 → all Bayes, w=0 → all Marcel.
    """
    actual = np.asarray(actual, dtype=float)
    bayes_pred = np.asarray(bayes_pred, dtype=float)
    marcel_pred = np.asarray(marcel_pred, dtype=float)

    grid = np.arange(0.0, 1.01, 0.01)
    best_w = 0.5
    best_loss = np.inf

    # Use Brier score (not MAE) — this system produces distributional
    # forecasts, so the ensemble weight should optimize the probabilistic
    # metric we actually evaluate on.
    league_avg = float(np.median(actual))
    actual_above = (actual > league_avg).astype(float)

    for w in grid:
        blend = w * bayes_pred + (1 - w) * marcel_pred
        # P(above league avg) ≈ fraction of blend > league_avg per player
        # For point estimates, use a soft sigmoid approximation
        spread = np.std(actual) if np.std(actual) > 1e-9 else 0.01
        blend_prob = 1.0 / (1.0 + np.exp(-(blend - league_avg) / (spread * 0.4)))
        loss = float(brier_score_loss(actual_above, blend_prob))

        if loss < best_loss:
            best_loss = loss
            best_w = float(w)

    return best_w


def apply_ensemble(
    comp: pd.DataFrame,
    w: float,
    stat: str,
) -> pd.DataFrame:
    """Apply ensemble blend to a comparison DataFrame.

    Adds columns: ensemble_{stat}, ensemble_ci_95_lo, ensemble_ci_95_hi.
    CIs are shifted by the difference between ensemble and Bayes mean.

    Parameters
    ----------
    comp : pd.DataFrame
        Must contain bayes_{stat}, marcel_{stat}, ci_95_lo, ci_95_hi.
    w : float
        Ensemble weight (1 = all Bayes, 0 = all Marcel).
    stat : str
        Stat key.

    Returns
    -------
    pd.DataFrame
        Copy with ensemble columns added.
    """
    comp = comp.copy()
    bayes_col = f"bayes_{stat}"
    marcel_col = f"marcel_{stat}"

    comp[f"ensemble_{stat}"] = w * comp[bayes_col] + (1 - w) * comp[marcel_col]

    # Shift CIs: move the Bayesian interval by (ensemble - bayes) offset
    shift = comp[f"ensemble_{stat}"] - comp[bayes_col]
    comp["ensemble_ci_95_lo"] = comp["ci_95_lo"] + shift
    comp["ensemble_ci_95_hi"] = comp["ci_95_hi"] + shift

    return comp


def compute_ensemble_metrics(
    comp: pd.DataFrame,
    stat: str,
    league_avg: float,
    proj_samples: dict[int, np.ndarray],
    id_col: str = "batter_id",
) -> dict[str, float]:
    """Compute evaluation metrics for the ensemble projection.

    Parameters
    ----------
    comp : pd.DataFrame
        Must contain ensemble_{stat}, actual_{stat}, ensemble_ci_95_lo/hi.
    stat : str
        Stat key.
    league_avg : float
        League average for Brier threshold.
    proj_samples : dict[int, np.ndarray]
        Bayesian posterior samples per player, used for shifted Brier.
    id_col : str
        Player ID column name.

    Returns
    -------
    dict
        ensemble_coverage_95, ensemble_brier.
    """
    actual = comp[f"actual_{stat}"].values
    ensemble = comp[f"ensemble_{stat}"].values
    bayes = comp[f"bayes_{stat}"].values

    lo = comp["ensemble_ci_95_lo"].values
    hi = comp["ensemble_ci_95_hi"].values
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))

    # Shifted Brier: shift posterior samples by (ensemble - bayes) then
    # compute P(shifted_samples > league_avg)
    actual_above = (actual > league_avg).astype(float)
    probs = []
    for _, row in comp.iterrows():
        pid = int(row[id_col])
        if pid in proj_samples:
            shift = float(row[f"ensemble_{stat}"] - row[f"bayes_{stat}"])
            shifted = proj_samples[pid] + shift
            probs.append(float(np.mean(shifted > league_avg)))
        else:
            probs.append(0.5)

    brier = float(brier_score_loss(actual_above, np.array(probs)))

    return {
        "ensemble_coverage_95": coverage,
        "ensemble_brier": brier,
    }
