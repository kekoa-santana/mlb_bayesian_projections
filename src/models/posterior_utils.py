"""Posterior sample extraction and prop-line utilities.

Reusable functions for extracting posterior samples from PyMC traces
and computing P(over X.5) prop-line probabilities from MC samples.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger(__name__)


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


# Re-exports from game_stat_model for backward compatibility.
# Validation scripts (game_k_validation, game_prop_validation) import these
# from posterior_utils after the game_k_model.py deletion.
from src.models.game_stat_model import (  # noqa: F401, E402
    predict_batter_game as predict_batter_game,
    predict_game_batch as predict_game_batch,
    predict_game_batch_stat as predict_game_batch_stat,
    simulate_game_ks as simulate_game_ks,
)
