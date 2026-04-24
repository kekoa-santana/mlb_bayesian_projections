"""
Batted-in-play (BIP) outcome model.

Given that a ball is put in play (not K, BB, HBP, or HR), determines
whether the outcome is an out, single, double, or triple.

Uses league-average conditional splits with a pitcher BABIP adjustment.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.constants import LEAGUE_BABIP_BATTER

logger = logging.getLogger(__name__)

# League-average BIP outcome splits
# Conditional on BIP (excludes K, BB, HBP, HR)
# Historical BABIP (2022-2025): 0.290.  Out rate = 1 - BABIP = 0.710.
# Regenerated per-batter profiles have mean p_out = 0.709.
# Default used as fallback for batters without individual profiles.
_DEFAULT_BIP_PROBS = {
    "out": 0.710,
    "single": 0.215,
    "double": 0.063,
    "triple": 0.005,
}

# Population BABIP for shrinkage
POP_BABIP = LEAGUE_BABIP_BATTER

# Shrinkage constant
_SHRINKAGE_K = 500  # BIP needed for full weight on pitcher-specific BABIP


# Coefficients for quality-metric BIP split prediction
# Trained on 2021-2025 BIP data (337 year-pairs, walk-forward validated)
# then scaled ~2x to match actual BABIP differentiation (std 0.036 vs
# original model std 0.017). Intercept recalculated so league-avg inputs
# (EV=88, LA=12, GB=0.44, sprint=27) produce pred_babip=0.295.
_BABIP_COEFS = {
    "avg_ev": 0.00070,
    "avg_la": -0.00790,
    "gb_pct": -0.17000,
    "sprint_speed": 0.00870,
    "intercept": 0.16810,
}

# Per-hit-type share adjustments from quality metrics
# Derived from observed splits by player archetype (2021-2025)
# Higher EV -> more doubles, fewer singles
# Higher speed -> more triples, more singles (infield hits)
# Higher GB% -> more singles, fewer doubles


def compute_player_bip_probs(
    avg_ev: float = 88.0,
    avg_la: float = 12.0,
    gb_pct: float = 0.44,
    sprint_speed: float = 27.0,
    observed_bip_splits: dict[str, float] | None = None,
    bip_count: int = 0,
    shrinkage_k: int = 150,
) -> np.ndarray:
    """Compute personalized BIP outcome probabilities.

    Blends quality-metric predictions with observed BIP splits,
    shrinkage-regressed toward league average.

    Parameters
    ----------
    avg_ev : float
        Average exit velocity (mph).
    avg_la : float
        Average launch angle (degrees).
    gb_pct : float
        Ground ball percentage (0-1).
    sprint_speed : float
        Sprint speed (ft/s).
    observed_bip_splits : dict, optional
        Observed {out, single, double, triple} rates from recent data.
    bip_count : int
        Total BIP for reliability weighting.
    shrinkage_k : int
        BIP needed for full weight on observed splits.

    Returns
    -------
    np.ndarray
        Shape (4,) probabilities for [out, single, double, triple].
    """
    league = np.array([
        _DEFAULT_BIP_PROBS["out"],
        _DEFAULT_BIP_PROBS["single"],
        _DEFAULT_BIP_PROBS["double"],
        _DEFAULT_BIP_PROBS["triple"],
    ])

    # Quality-metric predicted BABIP
    pred_babip = (
        _BABIP_COEFS["avg_ev"] * avg_ev
        + _BABIP_COEFS["avg_la"] * avg_la
        + _BABIP_COEFS["gb_pct"] * gb_pct
        + _BABIP_COEFS["sprint_speed"] * sprint_speed
        + _BABIP_COEFS["intercept"]
    )
    pred_babip = np.clip(pred_babip, 0.12, 0.55)

    # Adjust hit distribution based on player profile
    # High EV -> more extra-base hits (doubles especially)
    ev_factor = (avg_ev - 88.0) / 10.0  # normalized, 0 = average
    # High speed -> more triples and infield singles
    speed_factor = (sprint_speed - 27.0) / 3.0  # normalized
    # High GB% -> more singles, fewer doubles
    gb_factor = (gb_pct - 0.44) / 0.15  # normalized

    # Start from league hit distribution, adjust
    base_single_share = 0.755  # singles as share of all hits (league avg)
    base_double_share = 0.222
    base_triple_share = 0.023

    # Adjust shares
    double_share = base_double_share + 0.03 * ev_factor - 0.02 * gb_factor
    triple_share = base_triple_share + 0.01 * speed_factor
    single_share = 1.0 - double_share - triple_share

    # Clip to valid ranges
    double_share = np.clip(double_share, 0.05, 0.45)
    triple_share = np.clip(triple_share, 0.0, 0.08)
    single_share = np.clip(single_share, 0.40, 0.90)
    total = single_share + double_share + triple_share
    single_share /= total
    double_share /= total
    triple_share /= total

    # Build quality-predicted probs
    quality_probs = np.array([
        1.0 - pred_babip,
        pred_babip * single_share,
        pred_babip * double_share,
        pred_babip * triple_share,
    ])

    # Blend with observed splits if available
    if observed_bip_splits is not None and bip_count > 0:
        obs = np.array([
            observed_bip_splits.get("out", league[0]),
            observed_bip_splits.get("single", league[1]),
            observed_bip_splits.get("double", league[2]),
            observed_bip_splits.get("triple", league[3]),
        ])
        reliability = min(bip_count / shrinkage_k, 1.0)
        # Blend: reliability-weighted observed + (1-rel) quality prediction
        player_probs = reliability * obs + (1 - reliability) * quality_probs
    else:
        player_probs = quality_probs

    # Ensure valid probabilities
    player_probs = np.clip(player_probs, 0.001, 0.999)
    player_probs /= player_probs.sum()

    return player_probs


class BIPOutcomeModel:
    """Model for batted-in-play outcomes.

    Parameters
    ----------
    base_probs : dict[str, float], optional
        Base BIP outcome probabilities. Keys: 'out', 'single', 'double', 'triple'.
    """

    def __init__(
        self, base_probs: dict[str, float] | None = None
    ) -> None:
        self.base_probs = base_probs or _DEFAULT_BIP_PROBS.copy()
        self._outcome_names = ["out", "single", "double", "triple"]
        self._base_prob_array = np.array(
            [self.base_probs[k] for k in self._outcome_names]
        )

    def compute_pitcher_babip_adj(
        self,
        pitcher_babip: float,
        bip_count: int,
    ) -> float:
        """Compute shrinkage-adjusted BABIP deviation for a pitcher.

        Parameters
        ----------
        pitcher_babip : float
            Observed pitcher BABIP.
        bip_count : int
            Number of BIP (for reliability weighting).

        Returns
        -------
        float
            BABIP adjustment (positive = more hits on BIP than average).
        """
        reliability = min(bip_count / _SHRINKAGE_K, 1.0)
        raw_dev = pitcher_babip - POP_BABIP
        return reliability * raw_dev

    def get_adjusted_probs(self, babip_adj: float = 0.0) -> np.ndarray:
        """Get BIP outcome probabilities adjusted for pitcher BABIP.

        Parameters
        ----------
        babip_adj : float
            BABIP adjustment from compute_pitcher_babip_adj().

        Returns
        -------
        np.ndarray
            Shape (4,) probabilities for [out, single, double, triple].
        """
        probs = self._base_prob_array.copy()

        if abs(babip_adj) < 0.001:
            return probs

        # Shift hit probability (single + double + triple) by BABIP adj
        hit_prob = probs[1:].sum()
        new_hit_prob = np.clip(hit_prob + babip_adj, 0.03, 0.55)
        scale = new_hit_prob / hit_prob if hit_prob > 0 else 1.0

        probs[1:] *= scale
        probs[0] = 1.0 - probs[1:].sum()
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        return probs

    def draw_outcomes(
        self,
        rng: np.random.Generator,
        n_draws: int,
        babip_adj: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Draw BIP outcomes.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        n_draws : int
            Number of draws.
        babip_adj : float or np.ndarray
            BABIP adjustment. Scalar applies uniformly; array shape
            ``(n_draws,)`` gives per-sim values.

        Returns
        -------
        np.ndarray
            Integer outcome codes, shape (n_draws,).
            0 = out, 1 = single, 2 = double, 3 = triple.
        """
        if isinstance(babip_adj, np.ndarray) and babip_adj.ndim >= 1:
            # Per-sim BABIP: build per-sample probs and use inverse CDF
            base = self.get_adjusted_probs(0.0)  # (4,)
            p = np.broadcast_to(base, (n_draws, 4)).copy()
            nonzero = np.abs(babip_adj) >= 0.001
            if nonzero.any():
                hit_p = p[nonzero, 1:].sum(axis=1)
                safe_hit = np.where(hit_p > 1e-9, hit_p, 1e-9)
                new_hit = np.clip(hit_p + babip_adj[nonzero], 0.03, 0.55)
                scale = new_hit / safe_hit
                p[nonzero, 1:] = p[nonzero, 1:] * scale[:, None]
                p[nonzero, 0] = 1.0 - p[nonzero, 1:].sum(axis=1)
            p = np.clip(p, 1e-9, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            # Inverse CDF sampling
            u = rng.random(n_draws)
            cum = np.cumsum(p[:, :-1], axis=1)
            outcomes = (u[:, None] >= cum).sum(axis=1).astype(np.int8)
            return outcomes
        else:
            probs = self.get_adjusted_probs(babip_adj)
            probs = probs / probs.sum()
            return rng.choice(4, size=n_draws, p=probs)

    def draw_outcomes_per_sample(
        self,
        rng: np.random.Generator,
        probs: np.ndarray,
        babip_adj: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Draw BIP outcomes where each sample has its own probability vector.

        Vectorized inverse-CDF sampling: each row of ``probs`` defines that
        sample's [out, single, double, triple] distribution. The pitcher
        BABIP adjustment (if any) is applied as a multiplicative shift on
        hit mass, preserving the relative hit-type distribution.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        probs : np.ndarray
            Shape (n, 4) per-sample [out, single, double, triple] probs.
            Rows need not sum to exactly 1.0; they are renormalized.
        babip_adj : float
            Pitcher BABIP adjustment (positive = more hits on BIP).

        Returns
        -------
        np.ndarray
            Integer outcome codes, shape (n,). 0=out, 1=single, 2=double, 3=triple.
        """
        p = np.asarray(probs, dtype=np.float64).copy()
        # Apply pitcher BABIP adjustment by shifting hit mass
        babip_adj = np.asarray(babip_adj)
        if np.any(np.abs(babip_adj) >= 0.001):
            hit_p = p[:, 1:].sum(axis=1)
            # Avoid divide-by-zero on pathological rows
            safe_hit = np.where(hit_p > 1e-9, hit_p, 1e-9)
            new_hit = np.clip(hit_p + babip_adj, 0.03, 0.55)
            scale = new_hit / safe_hit
            p[:, 1:] = p[:, 1:] * scale[:, None]
            p[:, 0] = 1.0 - p[:, 1:].sum(axis=1)
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        # Inverse-CDF sampling using a single uniform per sample
        cdf = np.cumsum(p, axis=1)
        u = rng.random(p.shape[0])
        # argmax returns first True index along axis
        outcomes = (u[:, None] < cdf).argmax(axis=1).astype(np.int8)
        return outcomes


# Outcome code mapping
BIP_OUT = 0
BIP_SINGLE = 1
BIP_DOUBLE = 2
BIP_TRIPLE = 3
