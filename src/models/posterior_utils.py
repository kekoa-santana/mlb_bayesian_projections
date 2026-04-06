"""Posterior sample extraction and prop-line utilities.

Reusable functions for extracting posterior samples from PyMC traces
and computing P(over X.5) prop-line probabilities from MC samples.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.utils.constants import CLIP_LO, CLIP_HI

logger = logging.getLogger(__name__)


def _safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    """Logit with clipping."""
    return logit(np.clip(p, CLIP_LO, CLIP_HI))


def extract_pitcher_k_rate_samples(
    trace: Any,
    data: dict[str, Any],
    pitcher_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw K% posterior samples for one pitcher.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted pitcher K% model trace.
    data : dict
        Model data dict from ``prepare_pitcher_model_data``.
    pitcher_id : int
        Target pitcher.
    season : int
        Season whose posterior to extract.
    project_forward : bool
        If True, add AR(1) forward projection noise (for out-of-sample
        prediction). Uses rho dampening on the season effect.
    random_seed : int
        For reproducibility of forward projection noise.

    Returns
    -------
    np.ndarray
        K% posterior samples (1D array, values in [0, 1]).

    Raises
    ------
    ValueError
        If pitcher not found in the data for the given season.
    """
    df = data["df"]
    mask = (df["pitcher_id"] == pitcher_id) & (df["season"] == season)
    positions = df.index[mask].tolist()

    if not positions:
        raise ValueError(
            f"Pitcher {pitcher_id} not found in season {season}"
        )

    pos = positions[0]
    iloc_pos = df.index.get_loc(pos)

    # Extract posterior samples: (chains, draws, n_obs)
    k_rate_post = trace.posterior["k_rate"].values
    k_rate_flat = k_rate_post.reshape(-1, k_rate_post.shape[-1])
    samples = k_rate_flat[:, iloc_pos].copy()

    if project_forward and "sigma_season" in trace.posterior:
        rng = np.random.default_rng(random_seed)
        sigma_samples = trace.posterior["sigma_season"].values.flatten()
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
            rho_draws = np.ones(len(samples))

        # AR(1) forward projection on logit scale
        logit_samples = _safe_logit(samples)
        innovation = rng.normal(0, sigma_draws)

        alpha_post = trace.posterior["alpha"].values
        alpha_flat = alpha_post.reshape(-1, alpha_post.shape[-1])
        pidx = data["player_map"][pitcher_id]
        alpha_draws = alpha_flat[:, pidx]
        if len(alpha_draws) != len(samples):
            alpha_draws = rng.choice(alpha_draws, size=len(samples), replace=True)

        season_effect_last = logit_samples - alpha_draws
        new_effect = rho_draws * season_effect_last + innovation
        samples = expit(alpha_draws + new_effect)

    return samples


def compute_over_probs(
    stat_samples: np.ndarray,
    lines: list[float] | None = None,
    stat_name: str = "stat",
) -> pd.DataFrame:
    """Compute P(over X.5) for prop lines.

    Parameters
    ----------
    stat_samples : np.ndarray
        MC samples of stat totals.
    lines : list[float] or None
        Lines to evaluate. If None, auto-generate based on stat range.
    stat_name : str
        Name for column labeling.

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected_{stat_name}, std_{stat_name}.
    """
    if lines is None:
        max_val = int(np.max(stat_samples)) if len(stat_samples) > 0 else 5
        upper = min(max_val + 1, 20)
        lines = [x + 0.5 for x in range(upper)]

    expected = float(np.mean(stat_samples))
    std = float(np.std(stat_samples))

    records = []
    for line in lines:
        p_over = float(np.mean(stat_samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            f"expected_{stat_name}": expected,
            f"std_{stat_name}": std,
        })

    return pd.DataFrame(records)


def compute_k_over_probs(
    k_samples: np.ndarray,
    lines: list[float] | None = None,
) -> pd.DataFrame:
    """Compute P(over X.5) for standard K prop lines.

    Parameters
    ----------
    k_samples : np.ndarray
        Monte Carlo K total samples.
    lines : list[float] or None
        Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected_k, std_k.
    """
    if lines is None:
        lines = [x + 0.5 for x in range(13)]

    expected_k = float(np.mean(k_samples))
    std_k = float(np.std(k_samples))

    records = []
    for line in lines:
        p_over = float(np.mean(k_samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "expected_k": expected_k,
            "std_k": std_k,
        })

    return pd.DataFrame(records)
