"""
Pitch count per PA model.

Predicts how many pitches a plate appearance will consume, using an
empirical PMF shifted by pitcher efficiency (putaway rate) and batter
patience (contact rate, foul rate) adjustments.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Empirical pitch count PMF (2022-2025, all PAs)
# Index 0 = 1 pitch, index 1 = 2 pitches, etc.
_DEFAULT_PMF = np.array([
    0.1148,  # 1 pitch
    0.1492,  # 2 pitches
    0.1762,  # 3 pitches
    0.1879,  # 4 pitches
    0.1669,  # 5 pitches
    0.1201,  # 6 pitches
    0.0518,  # 7 pitches
    0.0207,  # 8 pitches
    0.0077,  # 9 pitches
    0.0030,  # 10 pitches
    0.0010,  # 11 pitches
    0.0004,  # 12 pitches
    0.0002,  # 13 pitches
    0.0001,  # 14+ pitches
])

# Population means for shrinkage
POP_PITCHES_PER_PA = 3.89
POP_PUTAWAY_RATE = 0.230
POP_CONTACT_RATE = 0.734
POP_FOUL_RATE = 0.370

# Shrinkage constant for pitcher/batter adjustments
_SHRINKAGE_K_PITCHER = 150  # PA needed for full weight on pitcher adj
_SHRINKAGE_K_BATTER = 150   # PA needed for full weight on batter adj

# Sensitivity: how much a 1-unit deviation in the feature shifts mean P/PA
# These are calibrated from the data: pitchers at 25th vs 75th percentile
# of putaway rate differ by ~0.3 P/PA.
_PUTAWAY_SENSITIVITY = -3.0   # Higher putaway → fewer pitches
_FOUL_RATE_SENSITIVITY = 2.5  # Higher foul rate → more pitches
_CONTACT_SENSITIVITY = -1.0   # Higher contact rate → fewer pitches (fewer deep counts)


class PitchCountModel:
    """Model for drawing pitch counts per PA.

    Parameters
    ----------
    base_pmf : np.ndarray, optional
        Empirical PMF of pitch counts (1-indexed).
    """

    def __init__(self, base_pmf: np.ndarray | None = None) -> None:
        self.base_pmf = base_pmf if base_pmf is not None else _DEFAULT_PMF.copy()
        # Cumulative for sampling
        self._base_cdf = np.cumsum(self.base_pmf)

    @staticmethod
    def compute_pitcher_adjustment(
        pitches_per_pa: float,
        putaway_rate: float,
        foul_rate: float,
        total_pa: int,
    ) -> float:
        """Compute pitcher-specific P/PA adjustment (shrunk toward 0).

        Parameters
        ----------
        pitches_per_pa : float
            Observed pitcher pitches per PA.
        putaway_rate : float
            Pitcher putaway rate (whiff% on 2-strike counts).
        foul_rate : float
            Pitcher foul ball rate (fouls / swings).
        total_pa : int
            Sample size for reliability weighting.

        Returns
        -------
        float
            Adjustment to mean P/PA (positive = more pitches than average).
        """
        reliability = min(total_pa / _SHRINKAGE_K_PITCHER, 1.0)

        # Raw deviation from population averages
        putaway_dev = putaway_rate - POP_PUTAWAY_RATE
        foul_dev = foul_rate - POP_FOUL_RATE

        # Feature-based predicted adjustment
        predicted_adj = (
            _PUTAWAY_SENSITIVITY * putaway_dev
            + _FOUL_RATE_SENSITIVITY * foul_dev
        )

        # Blend with direct observed deviation
        observed_adj = pitches_per_pa - POP_PITCHES_PER_PA
        raw_adj = 0.5 * predicted_adj + 0.5 * observed_adj

        return reliability * raw_adj

    @staticmethod
    def compute_batter_adjustment(
        pitches_per_pa: float,
        contact_rate: float,
        foul_rate: float,
        total_pa: int,
    ) -> float:
        """Compute batter-specific P/PA adjustment (shrunk toward 0).

        Parameters
        ----------
        pitches_per_pa : float
            Observed batter pitches per PA.
        contact_rate : float
            Batter contact rate (contact / swings).
        foul_rate : float
            Batter foul rate (fouls / swings).
        total_pa : int
            Sample size for reliability weighting.

        Returns
        -------
        float
            Adjustment to mean P/PA (positive = more pitches than average).
        """
        reliability = min(total_pa / _SHRINKAGE_K_BATTER, 1.0)

        contact_dev = contact_rate - POP_CONTACT_RATE
        foul_dev = foul_rate - POP_FOUL_RATE

        predicted_adj = (
            _CONTACT_SENSITIVITY * contact_dev
            + _FOUL_RATE_SENSITIVITY * foul_dev
        )

        observed_adj = pitches_per_pa - POP_PITCHES_PER_PA
        raw_adj = 0.5 * predicted_adj + 0.5 * observed_adj

        return reliability * raw_adj

    def draw_pitches(
        self,
        pitcher_adj: float,
        batter_adj: float | np.ndarray,
        rng: np.random.Generator,
        n_draws: int = 1,
    ) -> np.ndarray:
        """Draw pitch counts from adjusted PMF.

        The adjustment shifts the PMF by modifying the mean while
        preserving the general shape.

        Parameters
        ----------
        pitcher_adj : float
            Pitcher P/PA adjustment (scalar).
        batter_adj : float or np.ndarray
            Batter P/PA adjustment (scalar or per-draw array).
        rng : np.random.Generator
            Random number generator.
        n_draws : int
            Number of draws.

        Returns
        -------
        np.ndarray
            Integer pitch counts (1-indexed), shape (n_draws,).
        """
        total_adj = np.asarray(pitcher_adj + batter_adj)

        # Draw base pitch counts from empirical PMF
        u = rng.random(n_draws)
        base_pitches = np.searchsorted(self._base_cdf, u) + 1  # 1-indexed

        # If no meaningful adjustment, return base draws
        if np.all(np.abs(total_adj) < 0.05):
            return base_pitches

        # Broadcast total_adj to n_draws
        total_adj = np.broadcast_to(total_adj, (n_draws,)).copy()

        # Scale each draw by the adjustment ratio
        pitch_values = np.arange(1, len(self.base_pmf) + 1)
        base_mean = np.sum(pitch_values * self.base_pmf)
        target_mean = np.clip(base_mean + total_adj, 2.5, 6.0)
        scale = target_mean / base_mean

        raw_pitches = base_pitches.astype(float) * scale

        # Add small jitter and round
        jittered = raw_pitches + rng.normal(0, 0.3, n_draws)
        result = np.clip(np.round(jittered), 1, 15).astype(int)

        return result


def build_pitch_count_features(
    pitcher_features: pd.DataFrame,
    batter_features: pd.DataFrame,
    pitcher_id: int,
    batter_ids: list[int],
    season: int,
) -> tuple[float, np.ndarray]:
    """Build pitcher and batter P/PA adjustments for a game.

    Parameters
    ----------
    pitcher_features : pd.DataFrame
        From get_pitcher_pitch_count_features().
    batter_features : pd.DataFrame
        From get_batter_pitch_count_features().
    pitcher_id : int
        Pitcher MLB ID.
    batter_ids : list[int]
        List of 9 batter IDs in lineup order.
    season : int
        Season for lookup (use most recent available).

    Returns
    -------
    tuple[float, np.ndarray]
        (pitcher_adj, batter_adjs) where batter_adjs is shape (9,).
    """
    model = PitchCountModel()

    # Pitcher adjustment
    pmask = (
        (pitcher_features["pitcher_id"] == pitcher_id)
        & (pitcher_features["season"] == season)
    )
    p_row = pitcher_features[pmask]

    if len(p_row) > 0:
        p_row = p_row.iloc[0]
        pitcher_adj = model.compute_pitcher_adjustment(
            pitches_per_pa=float(p_row.get("pitches_per_pa", POP_PITCHES_PER_PA)),
            putaway_rate=float(p_row.get("putaway_rate", POP_PUTAWAY_RATE)),
            foul_rate=float(p_row.get("foul_rate", POP_FOUL_RATE)),
            total_pa=int(p_row.get("total_pa", 0)),
        )
    else:
        pitcher_adj = 0.0

    # Batter adjustments
    batter_adjs = np.zeros(len(batter_ids))
    for i, bid in enumerate(batter_ids):
        bmask = (
            (batter_features["batter_id"] == bid)
            & (batter_features["season"] == season)
        )
        b_row = batter_features[bmask]

        if len(b_row) > 0:
            b_row = b_row.iloc[0]
            batter_adjs[i] = model.compute_batter_adjustment(
                pitches_per_pa=float(b_row.get("pitches_per_pa", POP_PITCHES_PER_PA)),
                contact_rate=float(b_row.get("contact_rate", POP_CONTACT_RATE)),
                foul_rate=float(b_row.get("foul_rate", POP_FOUL_RATE)),
                total_pa=int(b_row.get("total_pa", 0)),
            )

    return pitcher_adj, batter_adjs
