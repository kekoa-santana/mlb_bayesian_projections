"""Shared math helpers used across models, simulators, and evaluation.

Kept small and dependency-light so any module can import from here
without creating cycles.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from src.utils.constants import CLIP_LO, CLIP_HI


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    """Logit with clipping to avoid infinities at 0 and 1."""
    return logit(np.clip(p, CLIP_LO, CLIP_HI))


def _fmt_quantile_key(q: float) -> str:
    if float(q).is_integer():
        return str(int(q))
    return str(float(q)).replace(".", "_")


def percentile_summary(
    samples: np.ndarray,
    prefix: str,
    quantiles: tuple[float, ...] = (2.5, 25, 50, 75, 97.5),
) -> dict[str, float]:
    """Return ``{prefix}_{q}`` → percentile mapping.

    Fractional quantile keys use underscore-for-decimal notation
    (e.g. ``2.5`` → ``"2_5"``, ``97.5`` → ``"97_5"``) to match the
    existing parquet column conventions.
    """
    values = np.percentile(samples, list(quantiles))
    return {
        f"{prefix}_{_fmt_quantile_key(q)}": float(v)
        for q, v in zip(quantiles, values)
    }


def posterior_point_summary(
    samples: np.ndarray,
    prefix: str,
    quantiles: tuple[float, ...] = (2.5, 25, 50, 75, 97.5),
) -> dict[str, float]:
    """Return mean, sd, and percentile summary for a single player's samples."""
    return {
        f"{prefix}_mean": float(np.mean(samples)),
        f"{prefix}_sd": float(np.std(samples)),
        **percentile_summary(samples, prefix=prefix, quantiles=quantiles),
    }


def flatten_posterior(trace, var_name: str) -> np.ndarray:
    """Flatten an arviz posterior variable from (chains, draws, obs) to (draws_flat, obs)."""
    arr = trace.posterior[var_name].values
    return arr.reshape(-1, arr.shape[-1])
