"""
Rolling momentum / form lift model for game simulation.

Computes logit-scale form adjustments from rolling 15-game and 30-game
player stats.  These adjust the Bayesian posterior rates to reflect
short-term form deviations.

Calibrated dampening factors from walk-forward backtest (2022-2025,
163K batter-games, 72K pitcher-games):

    Batter K%  : 0.025 logit  (rate-space slope 0.070, R²=0.001)
    Batter BB% : 0.010 logit  (rate-space slope 0.050, R²=0.0006)
    Batter HR  : ±0.012 logit (15g vs 30g direction, +11.4% relative)
    Batter HH% : ±0.010 logit (hard-hit% direction, +3.1% relative)
    Pitcher BB%: 0.015 logit  (rate-space slope 0.020, R²=0.0001)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.game_sim._sim_utils import safe_logit

logger = logging.getLogger(__name__)

# --- Calibrated dampening factors (logit scale) ---
# From walk-forward regression: next-game outcome ~ logit(roll_15g) - logit(roll_30g)
BATTER_K_DAMPENING = 0.025
BATTER_BB_DAMPENING = 0.010
# HR uses direction-based binary lift (not continuous dampening)
BATTER_HR_ACCEL_LIFT = 0.012   # applied when 15g HR/PA > 30g HR/PA
BATTER_HR_DECEL_LIFT = -0.011  # applied when 15g HR/PA < 30g HR/PA
# Hard-hit% direction-based lift
BATTER_HH_UP_LIFT = 0.010
BATTER_HH_DOWN_LIFT = -0.009
# Pitcher form (starters only)
PITCHER_BB_DAMPENING = 0.015
# Platoon × K deviation interaction (super-additive residual)
PLATOON_K_INTERACTION_LIFT = -0.010

# Minimum rolling PA/BF for reliable rate estimation
MIN_BATTER_PA_15 = 30
MIN_BATTER_PA_30 = 50
MIN_PITCHER_BF_15 = 30
MIN_PITCHER_BF_30 = 50
# Minimum BIP for hard-hit% reliability
MIN_HH_BIP = 15

# League-average hard-hit% (2020-2025 Statcast)
LEAGUE_HH_PCT = 0.395


_safe_logit = safe_logit  # alias for backward compat


@dataclass
class BatterFormLifts:
    """Logit-scale form lifts for a single batter."""
    k_lift: float = 0.0
    bb_lift: float = 0.0
    hr_lift: float = 0.0
    hh_lift: float = 0.0


@dataclass
class PitcherFormLifts:
    """Logit-scale form lifts for a single pitcher (starter)."""
    bb_lift: float = 0.0


def compute_batter_form_lifts(
    row: pd.Series,
    hard_hit_pct: float | None = None,
    hard_hit_bip: int = 0,
) -> BatterFormLifts:
    """Compute logit-scale form lifts for a batter from rolling data.

    Parameters
    ----------
    row : pd.Series
        Row from fact_player_form_rolling with bat_k_15, bat_pa_15, etc.
    hard_hit_pct : float, optional
        Rolling 15-game hard-hit percentage (0-1).
    hard_hit_bip : int
        Number of BIP in the hard-hit rolling window.

    Returns
    -------
    BatterFormLifts
        Logit-scale lifts for K, BB, HR, and hard-hit.
    """
    lifts = BatterFormLifts()

    pa_15 = int(row.get("bat_pa_15", 0) or 0)
    pa_30 = int(row.get("bat_pa_30", 0) or 0)

    if pa_15 < MIN_BATTER_PA_15 or pa_30 < MIN_BATTER_PA_30:
        return lifts

    # Derive rates
    k_15 = int(row.get("bat_k_15", 0) or 0)
    bb_15 = int(row.get("bat_bb_15", 0) or 0)
    hr_15 = int(row.get("bat_hr_15", 0) or 0)
    k_30 = int(row.get("bat_k_30", 0) or 0)
    bb_30 = int(row.get("bat_bb_30", 0) or 0)
    hr_30 = int(row.get("bat_hr_30", 0) or 0)

    k_pct_15 = k_15 / pa_15
    bb_pct_15 = bb_15 / pa_15
    hr_pct_15 = hr_15 / pa_15
    k_pct_30 = k_30 / pa_30
    bb_pct_30 = bb_30 / pa_30
    hr_pct_30 = hr_30 / pa_30

    # K% form lift: dampened logit deviation from 30g baseline
    k_dev = _safe_logit(k_pct_15) - _safe_logit(k_pct_30)
    lifts.k_lift = float(BATTER_K_DAMPENING * k_dev)

    # BB% form lift
    bb_dev = _safe_logit(bb_pct_15) - _safe_logit(bb_pct_30)
    lifts.bb_lift = float(BATTER_BB_DAMPENING * bb_dev)

    # HR/PA acceleration: binary direction-based lift
    if hr_pct_15 > hr_pct_30:
        lifts.hr_lift = BATTER_HR_ACCEL_LIFT
    elif hr_pct_15 < hr_pct_30:
        lifts.hr_lift = BATTER_HR_DECEL_LIFT
    # else: no lift when equal

    # Hard-hit% direction-based lift
    if hard_hit_pct is not None and hard_hit_bip >= MIN_HH_BIP:
        if hard_hit_pct > LEAGUE_HH_PCT:
            lifts.hh_lift = BATTER_HH_UP_LIFT
        elif hard_hit_pct < LEAGUE_HH_PCT:
            lifts.hh_lift = BATTER_HH_DOWN_LIFT

    return lifts


def compute_pitcher_form_lifts(
    row: pd.Series,
) -> PitcherFormLifts:
    """Compute logit-scale form lifts for a starting pitcher.

    Only BB% form is applied — pitcher K% form has near-zero signal.

    Parameters
    ----------
    row : pd.Series
        Row from fact_player_form_rolling with pit_bb_15, pit_bf_15, etc.

    Returns
    -------
    PitcherFormLifts
        Logit-scale BB lift.
    """
    lifts = PitcherFormLifts()

    bf_15 = int(row.get("pit_bf_15", 0) or 0)
    bf_30 = int(row.get("pit_bf_30", 0) or 0)

    if bf_15 < MIN_PITCHER_BF_15 or bf_30 < MIN_PITCHER_BF_30:
        return lifts

    bb_15 = int(row.get("pit_bb_15", 0) or 0)
    bb_30 = int(row.get("pit_bb_30", 0) or 0)

    bb_pct_15 = bb_15 / bf_15
    bb_pct_30 = bb_30 / bf_30

    bb_dev = _safe_logit(bb_pct_15) - _safe_logit(bb_pct_30)
    lifts.bb_lift = float(PITCHER_BB_DAMPENING * bb_dev)

    return lifts


def build_batter_form_lifts_batch(
    rolling_df: pd.DataFrame,
    hard_hit_df: pd.DataFrame | None = None,
) -> dict[int, BatterFormLifts]:
    """Compute form lifts for all batters in a rolling DataFrame.

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Output of get_rolling_form(..., player_role='batter').
        Must have player_id column.
    hard_hit_df : pd.DataFrame, optional
        Output of get_rolling_hard_hit(). Must have batter_id,
        hard_hit_pct, bip_count columns.

    Returns
    -------
    dict[int, BatterFormLifts]
        Keyed by player_id.
    """
    result: dict[int, BatterFormLifts] = {}

    hh_lookup: dict[int, tuple[float, int]] = {}
    if hard_hit_df is not None and not hard_hit_df.empty:
        for _, hh_row in hard_hit_df.iterrows():
            bid = int(hh_row["batter_id"])
            hh_pct = float(hh_row.get("hard_hit_pct", 0) or 0)
            hh_bip = int(hh_row.get("bip_count", 0) or 0)
            hh_lookup[bid] = (hh_pct, hh_bip)

    for _, row in rolling_df.iterrows():
        pid = int(row["player_id"])
        hh_pct, hh_bip = hh_lookup.get(pid, (None, 0))
        result[pid] = compute_batter_form_lifts(
            row, hard_hit_pct=hh_pct, hard_hit_bip=hh_bip,
        )

    return result


def build_pitcher_form_lifts_batch(
    rolling_df: pd.DataFrame,
) -> dict[int, PitcherFormLifts]:
    """Compute form lifts for all pitchers in a rolling DataFrame.

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Output of get_rolling_form(..., player_role='pitcher').

    Returns
    -------
    dict[int, PitcherFormLifts]
        Keyed by player_id.
    """
    result: dict[int, PitcherFormLifts] = {}
    for _, row in rolling_df.iterrows():
        pid = int(row["player_id"])
        result[pid] = compute_pitcher_form_lifts(row)
    return result
