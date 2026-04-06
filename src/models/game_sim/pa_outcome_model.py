"""
PA outcome model — multinomial over 8 outcome types.

For each plate appearance, computes adjusted probabilities for:
K, BB, HBP, HR (modeled individually on logit scale), then allocates
remaining probability to BIP outcomes via the BIP model.

Integrates:
- Pitcher rate posteriors (Layer 1)
- Matchup logit lifts (Layer 2)
- TTO adjustments
- Fatigue adjustments (pitch count)
- Game context (umpire/park/weather/catcher framing)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from src.models.game_sim.bip_model import (
    BIP_DOUBLE,
    BIP_OUT,
    BIP_SINGLE,
    BIP_TRIPLE,
    BIPOutcomeModel,
)
from src.models.game_sim._sim_utils import safe_logit
from src.utils.constants import LEAGUE_HBP_RATE

logger = logging.getLogger(__name__)

# PA outcome codes
PA_STRIKEOUT = 0
PA_WALK = 1
PA_HBP = 2
PA_SINGLE = 3
PA_DOUBLE = 4
PA_TRIPLE = 5
PA_HOME_RUN = 6
PA_OUT = 7

# Sim calibration offsets (logit scale).
# After exit model calibration fix (-0.35 logit offset), residual biases are:
#   K: ~-0.18/game (still slightly under-predicted by stacked logit lifts)
#   BB: ~+0.33/game (over-predicted)
# These offsets fine-tune the per-PA rates.
_CALIBRATION_K_OFFSET = -0.02   # slight K logit reduction (was -0.06, reduced after exit fix)
_CALIBRATION_BB_OFFSET = 0.01   # slight BB logit increase (was 0.03, reduced after exit fix)

# Fatigue adjustment thresholds and slopes (logit scale)
_FATIGUE_PITCH_THRESHOLD = 90   # Research (Bradbury 2007, Statcast velocity studies): meaningful
                                # fatigue effects begin ~90-95 pitches, steepest past 100
_FATIGUE_K_SLOPE = -0.003       # K logit drops per pitch above threshold
_FATIGUE_BB_SLOPE = 0.00239     # BB logit increases per pitch above threshold
_FATIGUE_HR_SLOPE = 0.003       # HR logit increases per pitch above threshold (was 0.001;
                                # Perez & Sherwood 2020: HR rate increases 15-20% relative
                                # in high-pitch-count regime → ~0.003 logit/pitch)


@dataclass(frozen=True)
class GameContext:
    """Per-game environmental logit lifts (constant throughout game).

    Bundles all context signals that are known at lineup lock and do not
    change PA-to-PA. New environmental signals (e.g., platoon) should be
    added here rather than as new parameters to compute_pa_probs.
    """

    umpire_k_lift: float = 0.0
    umpire_bb_lift: float = 0.0
    park_k_lift: float = 0.0
    park_bb_lift: float = 0.0
    park_hr_lift: float = 0.0
    park_h_babip_adj: float = 0.0
    weather_k_lift: float = 0.0
    form_bb_lift: float = 0.0
    catcher_k_lift: float = 0.0
    xgb_bb_lift: float = 0.0


_EMPTY_CONTEXT = GameContext()


def compute_fatigue_adjustments(
    cumulative_pitches: int | np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute fatigue logit adjustments based on cumulative pitch count.

    Parameters
    ----------
    cumulative_pitches : int or np.ndarray
        Total pitches thrown so far in the game.

    Returns
    -------
    dict[str, float | np.ndarray]
        Keys: 'k', 'bb', 'hr'. Values: logit-scale adjustments.
    """
    excess = np.maximum(cumulative_pitches - _FATIGUE_PITCH_THRESHOLD, 0)
    return {
        "k": _FATIGUE_K_SLOPE * excess,
        "bb": _FATIGUE_BB_SLOPE * excess,
        "hr": _FATIGUE_HR_SLOPE * excess,
    }


class PAOutcomeModel:
    """Multinomial PA outcome model with logit-additive adjustments.

    Parameters
    ----------
    bip_model : BIPOutcomeModel, optional
        Model for batted-in-play outcomes. Uses default if not provided.
    hbp_rate : float
        League-average HBP rate.
    """

    def __init__(
        self,
        bip_model: BIPOutcomeModel | None = None,
        hbp_rate: float = LEAGUE_HBP_RATE,
    ) -> None:
        self.bip_model = bip_model or BIPOutcomeModel()
        self.hbp_rate = hbp_rate

    @staticmethod
    def _safe_logit(p: np.ndarray | float) -> np.ndarray | float:
        """Logit with clipping."""
        return safe_logit(p)

    def compute_pa_probs(
        self,
        pitcher_k_rate: float | np.ndarray,
        pitcher_bb_rate: float | np.ndarray,
        pitcher_hr_rate: float | np.ndarray,
        matchup_k_lift: float = 0.0,
        matchup_bb_lift: float = 0.0,
        matchup_hr_lift: float = 0.0,
        tto_k_lift: float = 0.0,
        tto_bb_lift: float = 0.0,
        tto_hr_lift: float = 0.0,
        fatigue_k_lift: float = 0.0,
        fatigue_bb_lift: float = 0.0,
        fatigue_hr_lift: float = 0.0,
        ctx: GameContext | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Compute adjusted PA outcome probabilities.

        All lifts are on the logit scale. Probabilities are computed
        independently for K, BB, HR, then renormalized to ensure they
        don't exceed 1.0 (with HBP).

        Parameters
        ----------
        pitcher_k_rate : float or np.ndarray
            Base K rate from posterior.
        pitcher_bb_rate : float or np.ndarray
            Base BB rate from posterior.
        pitcher_hr_rate : float or np.ndarray
            Base HR rate from posterior.
        matchup_*_lift : float
            Per-batter matchup logit lifts (from Layer 2).
        tto_*_lift : float
            Through-the-order logit lifts.
        fatigue_*_lift : float
            Pitch count fatigue logit lifts.
        ctx : GameContext, optional
            Per-game environmental lifts (umpire, park, weather, catcher
            framing, pitcher form). Defaults to zero lifts.

        Returns
        -------
        dict[str, float | np.ndarray]
            Keys: 'k', 'bb', 'hbp', 'hr', 'bip'. Values: probabilities.
        """
        _ctx = ctx or _EMPTY_CONTEXT

        # K probability (with calibration offset to correct sim bias)
        k_logit = (
            self._safe_logit(pitcher_k_rate)
            + matchup_k_lift + tto_k_lift + fatigue_k_lift
            + _ctx.umpire_k_lift + _ctx.park_k_lift + _ctx.weather_k_lift
            + _ctx.catcher_k_lift
            + _CALIBRATION_K_OFFSET
        )
        k_prob = expit(k_logit)

        # BB probability (with calibration offset + pitcher form + XGB adjustment)
        bb_logit = (
            self._safe_logit(pitcher_bb_rate)
            + matchup_bb_lift + tto_bb_lift + fatigue_bb_lift
            + _ctx.umpire_bb_lift + _ctx.park_bb_lift
            + _ctx.form_bb_lift + _ctx.xgb_bb_lift
            + _CALIBRATION_BB_OFFSET
        )
        bb_prob = expit(bb_logit)

        # HR probability
        hr_logit = (
            self._safe_logit(pitcher_hr_rate)
            + matchup_hr_lift + tto_hr_lift + fatigue_hr_lift
            + _ctx.park_hr_lift
        )
        hr_prob = expit(hr_logit)

        # HBP probability (constant)
        hbp_prob = self.hbp_rate

        # Renormalize if sum exceeds 1.0
        total = k_prob + bb_prob + hr_prob + hbp_prob
        if np.any(total > 0.95):
            # Scale down proportionally, preserving HBP
            scale = np.where(
                total > 0.95,
                (0.95 - hbp_prob) / (k_prob + bb_prob + hr_prob),
                1.0,
            )
            k_prob = k_prob * scale
            bb_prob = bb_prob * scale
            hr_prob = hr_prob * scale

        bip_prob = np.maximum(1.0 - k_prob - bb_prob - hr_prob - hbp_prob, 0.01)

        return {
            "k": k_prob,
            "bb": bb_prob,
            "hbp": hbp_prob,
            "hr": hr_prob,
            "bip": bip_prob,
        }

    def draw_outcomes(
        self,
        probs: dict[str, float | np.ndarray],
        rng: np.random.Generator,
        n_draws: int = 1,
        babip_adj: float = 0.0,
    ) -> np.ndarray:
        """Draw PA outcomes from computed probabilities.

        Parameters
        ----------
        probs : dict
            Output of compute_pa_probs().
        rng : np.random.Generator
            Random number generator.
        n_draws : int
            Number of draws.
        babip_adj : float
            Pitcher BABIP adjustment for BIP outcomes.

        Returns
        -------
        np.ndarray
            Integer outcome codes, shape (n_draws,).
            See PA_* constants for mapping.
        """
        # Build multinomial probabilities
        k_p = np.broadcast_to(np.asarray(probs["k"]), (n_draws,))
        bb_p = np.broadcast_to(np.asarray(probs["bb"]), (n_draws,))
        hbp_p = np.broadcast_to(np.asarray(probs["hbp"]), (n_draws,))
        hr_p = np.broadcast_to(np.asarray(probs["hr"]), (n_draws,))
        bip_p = np.broadcast_to(np.asarray(probs["bip"]), (n_draws,))

        # Use inverse CDF sampling for speed
        u = rng.random(n_draws)
        outcomes = np.full(n_draws, PA_OUT, dtype=np.int8)

        cum = np.zeros(n_draws)
        cum += k_p
        k_mask = u < cum
        outcomes[k_mask] = PA_STRIKEOUT

        prev_cum = cum.copy()
        cum += bb_p
        bb_mask = (u >= prev_cum) & (u < cum)
        outcomes[bb_mask] = PA_WALK

        prev_cum = cum.copy()
        cum += hbp_p
        hbp_mask = (u >= prev_cum) & (u < cum)
        outcomes[hbp_mask] = PA_HBP

        prev_cum = cum.copy()
        cum += hr_p
        hr_mask = (u >= prev_cum) & (u < cum)
        outcomes[hr_mask] = PA_HOME_RUN

        # Remaining are BIP — resolve into out/single/double/triple
        bip_mask = ~(k_mask | bb_mask | hbp_mask | hr_mask)
        n_bip = bip_mask.sum()

        if n_bip > 0:
            bip_outcomes = self.bip_model.draw_outcomes(
                rng=rng, n_draws=n_bip, babip_adj=babip_adj
            )
            # Map BIP codes to PA codes
            bip_to_pa = {
                BIP_OUT: PA_OUT,
                BIP_SINGLE: PA_SINGLE,
                BIP_DOUBLE: PA_DOUBLE,
                BIP_TRIPLE: PA_TRIPLE,
            }
            for bip_code, pa_code in bip_to_pa.items():
                outcomes[bip_mask] = np.where(
                    bip_outcomes == bip_code,
                    pa_code,
                    outcomes[bip_mask],
                )

        return outcomes
