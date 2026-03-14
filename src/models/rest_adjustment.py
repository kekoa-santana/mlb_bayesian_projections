"""
Days-rest adjustments for starting pitcher game predictions.

Computes logit-scale K/BB rate adjustments and BF distribution modifiers
based on the number of days since a pitcher's previous start.

Rest buckets:
- **short** (<=4 days): Rare in modern MLB. BF significantly lower
  (managers pull earlier), rates roughly unchanged.
- **normal** (5 days): Baseline — no adjustment.
- **extended** (6+ days): Slightly lower BF on long rest (8+ days),
  K/BB rates essentially unchanged.

Empirical evidence (2018-2025, 30,486 starts with BF >= 9):
    Bucket     N       K%      BB%     Avg BF   Std BF
    short      242     .2195   .0787   20.77    4.79
    normal     11,631  .2209   .0774   23.03    3.75
    extended   18,613  .2208   .0797   22.80    3.66

The main signal is in BF (workload management), not K/BB rates.
Rate adjustments are kept conservative given noisy small-sample evidence.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---- Empirical baselines (2018-2025, BF >= 9 filter) ----
# K% and BB% differences are tiny and not statistically significant.
# We use conservative adjustments that are smaller than raw differences.

# Logit-scale K rate adjustments by rest bucket
# Empirical K%: short=.2195, normal=.2209, extended=.2208
# Difference is <0.002 in probability — within noise.
_K_LOGIT_LIFT: dict[str, float] = {
    "short": -0.01,     # Slight negative (conservative, raw diff ~ -0.001)
    "normal": 0.0,      # Baseline
    "extended": 0.0,    # No meaningful difference from normal
}

# Logit-scale BB rate adjustments by rest bucket
# Empirical BB%: short=.0787, normal=.0774, extended=.0797
# Short rest and extended rest both slightly higher, but noisy.
_BB_LOGIT_LIFT: dict[str, float] = {
    "short": 0.02,      # Slight increase (raw diff ~ +0.001)
    "normal": 0.0,      # Baseline
    "extended": 0.01,   # Very slight increase on long rest
}

# BF distribution modifiers by rest bucket
# This is where the real signal is:
# Short rest: avg BF = 20.77 vs normal 23.03 (−9.8%), higher variance
# Extended: avg BF = 22.80 vs normal 23.03 (−1.0%), lower variance
_BF_MEAN_MULT: dict[str, float] = {
    "short": 0.90,      # ~10% fewer BF on short rest
    "normal": 1.0,      # Baseline
    "extended": 0.99,   # Barely any change
}

_BF_VAR_MULT: dict[str, float] = {
    "short": 1.30,      # More variable outcomes on short rest
    "normal": 1.0,      # Baseline
    "extended": 0.98,   # Slightly tighter on extended rest
}


def classify_rest_bucket(days_rest: int | None) -> str:
    """Classify days of rest into a bucket.

    Parameters
    ----------
    days_rest : int or None
        Days since previous start.  None or negative returns ``'normal'``.

    Returns
    -------
    str
        One of ``'short'``, ``'normal'``, ``'extended'``.
    """
    if days_rest is None or days_rest < 0:
        return "normal"
    if days_rest <= 4:
        return "short"
    if days_rest == 5:
        return "normal"
    return "extended"


def get_rest_adjustment(days_rest: int | None) -> dict[str, float]:
    """Get logit-scale rate adjustments for a given days-rest value.

    Parameters
    ----------
    days_rest : int or None
        Days since previous start.  None returns neutral adjustments.

    Returns
    -------
    dict
        Keys: ``rest_bucket``, ``k_lift`` (logit-scale), ``bb_lift``
        (logit-scale), ``days_rest``.
    """
    bucket = classify_rest_bucket(days_rest)
    return {
        "rest_bucket": bucket,
        "k_lift": _K_LOGIT_LIFT[bucket],
        "bb_lift": _BB_LOGIT_LIFT[bucket],
        "days_rest": days_rest,
    }


def get_rest_bf_modifier(days_rest: int | None) -> dict[str, float]:
    """Get BF distribution modifiers for a given days-rest value.

    Applied as multipliers to the BF mean and variance from the BF model.
    This is where the main rest effect lives — managers pull pitchers
    earlier on short rest.

    Parameters
    ----------
    days_rest : int or None
        Days since previous start.  None returns neutral modifiers.

    Returns
    -------
    dict
        Keys: ``rest_bucket``, ``bf_mean_multiplier``,
        ``bf_var_multiplier``, ``days_rest``.
    """
    bucket = classify_rest_bucket(days_rest)
    return {
        "rest_bucket": bucket,
        "bf_mean_multiplier": _BF_MEAN_MULT[bucket],
        "bf_var_multiplier": _BF_VAR_MULT[bucket],
        "days_rest": days_rest,
    }


def apply_rest_to_bf(
    bf_mu: float,
    bf_sigma: float,
    days_rest: int | None,
) -> tuple[float, float]:
    """Apply rest-day modifier to BF distribution parameters.

    Convenience wrapper that returns adjusted (mu, sigma) values.

    Parameters
    ----------
    bf_mu : float
        Original BF mean from the BF model.
    bf_sigma : float
        Original BF std from the BF model.
    days_rest : int or None
        Days since previous start.

    Returns
    -------
    tuple[float, float]
        (adjusted_mu, adjusted_sigma)
    """
    mod = get_rest_bf_modifier(days_rest)
    adj_mu = bf_mu * mod["bf_mean_multiplier"]
    adj_sigma = bf_sigma * mod["bf_var_multiplier"]
    return adj_mu, adj_sigma


def compute_rest_for_game(
    pitcher_id: int,
    game_pk: int,
    rest_df: "pd.DataFrame | None",
) -> int | None:
    """Look up days rest for a specific pitcher-game from pre-computed data.

    Parameters
    ----------
    pitcher_id : int
        MLB pitcher ID.
    game_pk : int
        Game primary key.
    rest_df : pd.DataFrame or None
        Output of ``queries.get_days_rest()``.  Must have columns:
        ``pitcher_id``, ``game_pk``, ``days_rest``.

    Returns
    -------
    int or None
        Days since previous start, or None if not found.
    """
    if rest_df is None or rest_df.empty:
        return None

    mask = (rest_df["pitcher_id"] == pitcher_id) & (rest_df["game_pk"] == game_pk)
    rows = rest_df[mask]
    if rows.empty:
        return None
    return int(rows.iloc[0]["days_rest"])
