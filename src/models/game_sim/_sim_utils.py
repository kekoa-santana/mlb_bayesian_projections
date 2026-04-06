"""
Shared simulation utilities for game_sim package.

Extracted from duplicated code across simulator.py, batter_simulator.py,
lineup_simulator.py, and pa_outcome_model.py.  Zero behavioral changes —
pure deduplication.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from src.utils.constants import CLIP_LO, CLIP_HI


# ---------------------------------------------------------------------------
# Logit helper
# ---------------------------------------------------------------------------

def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    """Logit with clipping to avoid infinities."""
    return logit(np.clip(p, CLIP_LO, CLIP_HI))


# ---------------------------------------------------------------------------
# Posterior resampling
# ---------------------------------------------------------------------------

def resample_posterior(
    arr: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample posterior array to exactly n_sims samples."""
    if len(arr) == n_sims:
        return arr.copy()
    return rng.choice(arr, size=n_sims, replace=True)


# ---------------------------------------------------------------------------
# Matchup dampening constants
# ---------------------------------------------------------------------------

# Empirical calibration from 11,517 game walk-forward backtest (2023-2025).
# Raw pitch-type matchup scoring over-applies lifts by ~2x for K/BB.
# HR signal is near-zero.
MATCHUP_DAMPEN: dict[str, float] = {"k": 0.55, "bb": 0.40, "hr": 0.20}


# ---------------------------------------------------------------------------
# Pitcher quality lift computation
# ---------------------------------------------------------------------------

def compute_pitcher_quality_lifts(
    k_rate: float,
    bb_rate: float,
    hr_rate: float,
    league_k: float,
    league_bb: float,
    league_hr: float,
) -> tuple[float, float, float]:
    """Compute logit-scale pitcher quality lifts vs league average."""
    return (
        safe_logit(k_rate) - safe_logit(league_k),
        safe_logit(bb_rate) - safe_logit(league_bb),
        safe_logit(hr_rate) - safe_logit(league_hr),
    )


# ---------------------------------------------------------------------------
# None-to-zero array defaulting
# ---------------------------------------------------------------------------

def default_lift_array(
    arr: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Return arr if not None, else zeros of given size."""
    return arr if arr is not None else np.zeros(size)
