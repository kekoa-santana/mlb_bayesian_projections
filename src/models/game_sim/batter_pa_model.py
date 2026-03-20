"""
Batter PA count model.

Determines how many plate appearances a batter gets in a game,
based on lineup position, and splits them into PAs vs starter
and PAs vs reliever based on the starter's workload draw.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Mean and std of total PAs by batting order position (1-9)
# From 2022-2025 fact_player_game_mlb joined with fact_lineup
_PA_MEAN_BY_SLOT = np.array([
    4.49, 4.40, 4.30, 4.20, 4.06, 3.94, 3.79, 3.63, 3.46,
])
_PA_STD_BY_SLOT = np.array([
    0.71, 0.68, 0.67, 0.65, 0.68, 0.70, 0.74, 0.77, 0.79,
])

# Clamp range for total PAs
MIN_PA = 1
MAX_PA = 7

# Default starter BF (used when no pitcher-specific data available)
DEFAULT_STARTER_BF_MU = 22.0
DEFAULT_STARTER_BF_STD = 4.5


def draw_total_pa(
    batting_order: int,
    rng: np.random.Generator,
    n_draws: int = 1,
) -> np.ndarray:
    """Draw total PAs for a batter from lineup-position distribution.

    Parameters
    ----------
    batting_order : int
        Batting order position (1-9).
    rng : np.random.Generator
        Random number generator.
    n_draws : int
        Number of draws.

    Returns
    -------
    np.ndarray
        Integer PA counts, shape (n_draws,).
    """
    idx = batting_order - 1  # 0-indexed
    idx = min(max(idx, 0), 8)

    mean = _PA_MEAN_BY_SLOT[idx]
    std = _PA_STD_BY_SLOT[idx]

    raw = rng.normal(mean, std, n_draws)
    return np.clip(np.round(raw), MIN_PA, MAX_PA).astype(np.int32)


def split_pa_starter_reliever(
    total_pa: np.ndarray,
    batting_order: int,
    starter_bf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split total PAs into PAs vs starter and PAs vs reliever.

    Uses the rule: PA k for a batter in slot s occurs at approximately
    global BF position s + 9*(k-1). If that position is before the
    starter exits, it's a PA vs starter.

    Parameters
    ----------
    total_pa : np.ndarray
        Total PAs per simulation, shape (n_sims,).
    batting_order : int
        Batting order position (1-9).
    starter_bf : np.ndarray
        Starter's total BF drawn per simulation, shape (n_sims,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pa_vs_starter, pa_vs_reliever), each shape (n_sims,).
    """
    n_sims = len(total_pa)
    pa_vs_starter = np.zeros(n_sims, dtype=np.int32)

    for k in range(int(total_pa.max())):
        # Global BF position when this batter comes up for PA k+1
        global_pos = batting_order + 9 * k
        # This PA is vs starter if the starter hasn't exited yet
        still_batting = k < total_pa
        vs_starter = (global_pos <= starter_bf) & still_batting
        pa_vs_starter += vs_starter.astype(np.int32)

    pa_vs_reliever = total_pa - pa_vs_starter
    return pa_vs_starter, pa_vs_reliever
