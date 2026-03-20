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

logger = logging.getLogger(__name__)

# League-average BIP outcome splits (2022-2025)
# Conditional on BIP (excludes K, BB, HBP, HR)
# Derived from: field_out / (field_out + single + double + triple)
# where field_out includes force_out, GIDP, FC, sac_fly, etc.
_DEFAULT_BIP_PROBS = {
    "out": 0.700,
    "single": 0.222,
    "double": 0.065,
    "triple": 0.005,
}

# Population BABIP for shrinkage
POP_BABIP = 0.300

# Shrinkage constant
_SHRINKAGE_K = 500  # BIP needed for full weight on pitcher-specific BABIP


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

    def get_league_probs(self) -> np.ndarray:
        """Get league-average BIP outcome probabilities.

        Returns
        -------
        np.ndarray
            Shape (4,) probabilities for [out, single, double, triple].
        """
        return self._base_prob_array.copy()

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
        new_hit_prob = np.clip(hit_prob + babip_adj, 0.05, 0.50)
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
        babip_adj: float = 0.0,
    ) -> np.ndarray:
        """Draw BIP outcomes.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        n_draws : int
            Number of draws.
        babip_adj : float
            BABIP adjustment.

        Returns
        -------
        np.ndarray
            Integer outcome codes, shape (n_draws,).
            0 = out, 1 = single, 2 = double, 3 = triple.
        """
        probs = self.get_adjusted_probs(babip_adj)
        # Ensure exact sum to 1.0 for numpy multinomial
        probs = probs / probs.sum()
        return rng.choice(4, size=n_draws, p=probs)


# Outcome code mapping
BIP_OUT = 0
BIP_SINGLE = 1
BIP_DOUBLE = 2
BIP_TRIPLE = 3
