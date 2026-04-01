"""
Through-the-order (TTO) adjustment model.

Provides logit-scale lifts for K, BB, HR, and hit rates by TTO bucket
(1st, 2nd, 3rd+). Pitcher-specific lifts are reliability-weighted
toward league averages.

Extracted from game_k_model.py and extended with hit-rate TTO lifts.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import logit

from src.utils.constants import CLIP_LO, CLIP_HI

logger = logging.getLogger(__name__)

# League-average TTO rates (computed from 2018-2025 fact_pa)
# Used to derive logit lifts relative to overall rates.
_LEAGUE_TTO_RATES: dict[str, dict[str, np.ndarray]] = {
    "k": {
        "tto": np.array([0.23782, 0.20952, 0.19421]),
        "overall": 0.22557,
    },
    "bb": {
        "tto": np.array([0.08483, 0.07479, 0.07470]),
        "overall": 0.08115,
    },
    "hr": {
        "tto": np.array([0.02979, 0.03385, 0.03658]),
        "overall": 0.03162,
    },
    "h": {
        "tto": np.array([0.13922, 0.14439, 0.15014]),
        "overall": 0.14157,
    },
}

# Pre-computed league-average logit lifts
LEAGUE_TTO_LOGIT_LIFTS: dict[str, np.ndarray] = {}
for _stat, _rates in _LEAGUE_TTO_RATES.items():
    _overall_logit = logit(np.clip(_rates["overall"], CLIP_LO, CLIP_HI))
    _tto_logits = logit(np.clip(_rates["tto"], CLIP_LO, CLIP_HI))
    LEAGUE_TTO_LOGIT_LIFTS[_stat] = _tto_logits - _overall_logit

# BF per TTO block
BF_PER_TTO = 9


def build_tto_logit_lifts(
    tto_profiles: pd.DataFrame | None,
    pitcher_id: int,
    season: int,
    stat_name: str = "k",
    min_reliability_pa: float = 100.0,
) -> np.ndarray:
    """Get TTO logit lifts for a pitcher, blended toward league average.

    Parameters
    ----------
    tto_profiles : pd.DataFrame or None
        Output of ``get_tto_adjustment_profiles()``. Must contain columns:
        pitcher_id, season, tto, {stat}_rate, overall_{stat}_rate, pa_count.
        If None, returns league-average lifts.
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season to look up.
    stat_name : str
        One of 'k', 'bb', 'hr', 'h'.
    min_reliability_pa : float
        PA count at which pitcher-specific lifts get full weight.

    Returns
    -------
    np.ndarray
        Shape (3,) logit lifts for TTO 1, 2, 3.
    """
    sn = stat_name.lower()
    league_lifts = LEAGUE_TTO_LOGIT_LIFTS.get(sn)
    if league_lifts is None:
        return np.zeros(3)

    if tto_profiles is None or tto_profiles.empty:
        return league_lifts.copy()

    mask = (
        (tto_profiles["pitcher_id"] == pitcher_id)
        & (tto_profiles["season"] == season)
    )
    pitcher_data = tto_profiles[mask]

    if len(pitcher_data) < 3:
        return league_lifts.copy()

    rate_col = f"{sn}_rate"
    overall_col = f"overall_{sn}_rate"

    if rate_col not in pitcher_data.columns or overall_col not in pitcher_data.columns:
        return league_lifts.copy()

    pitcher_data = pitcher_data.sort_values("tto")
    tto_rates = pitcher_data[rate_col].values.astype(float)
    overall_rate = pitcher_data[overall_col].values[0].astype(float)

    if overall_rate < CLIP_LO or overall_rate > CLIP_HI:
        return league_lifts.copy()

    overall_logit = logit(np.clip(overall_rate, CLIP_LO, CLIP_HI))
    tto_logits = logit(np.clip(tto_rates, CLIP_LO, CLIP_HI))
    pitcher_lifts = tto_logits - overall_logit

    # Reliability-weight toward league average
    pa_counts = pitcher_data["pa_count"].values.astype(float)
    reliability = np.clip(pa_counts / min_reliability_pa, 0.0, 1.0)
    blended = reliability * pitcher_lifts + (1.0 - reliability) * league_lifts

    return blended


def build_all_tto_lifts(
    tto_profiles: pd.DataFrame | None,
    pitcher_id: int,
    season: int,
) -> dict[str, np.ndarray]:
    """Build TTO lifts for all stats at once.

    Parameters
    ----------
    tto_profiles : pd.DataFrame or None
        TTO profiles data.
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season to look up.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: 'k', 'bb', 'hr', 'h'. Values: shape (3,) logit lifts.
    """
    return {
        stat: build_tto_logit_lifts(tto_profiles, pitcher_id, season, stat)
        for stat in ("k", "bb", "hr", "h")
    }


def get_tto_for_bf(bf_number: int) -> int:
    """Map a 0-indexed BF number to TTO block (0, 1, or 2).

    Parameters
    ----------
    bf_number : int
        0-indexed batter faced number within the game.

    Returns
    -------
    int
        TTO block: 0 (1st time), 1 (2nd time), 2 (3rd+ time).
    """
    return min(bf_number // BF_PER_TTO, 2)
