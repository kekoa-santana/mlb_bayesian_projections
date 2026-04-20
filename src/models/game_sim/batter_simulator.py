"""
Batter game simulator.

Simulates a batter's plate appearances across a game, determining
for each PA whether the batter faces the starter (with specific
matchup adjustments) or a reliever (team bullpen aggregate rates),
then resolving outcomes.

Produces joint distributions over K, BB, H, HR, TB, R, RBI per game.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from src.models.game_sim.bip_model import (
    BIP_DOUBLE,
    BIP_SINGLE,
    BIP_TRIPLE,
    BIPOutcomeModel,
)
from src.models.game_sim.pa_outcome_model import (
    PA_DOUBLE,
    PA_HBP,
    PA_HOME_RUN,
    PA_OUT,
    PA_SINGLE,
    PA_STRIKEOUT,
    PA_TRIPLE,
    PA_WALK,
    GameContext,
    PAOutcomeModel,
)
from src.models.game_sim.batter_pa_model import (
    draw_total_pa,
    split_pa_starter_reliever,
)
from src.models.bf_model import draw_bf_samples
from src.models.game_sim._sim_utils import (
    safe_logit,
    resample_posterior,
    compute_pitcher_quality_lifts,
    pitcher_rate_to_lift_array,
)
from src.utils.constants import (
    SIM_LEAGUE_K_RATE,
    SIM_LEAGUE_BB_RATE,
    SIM_LEAGUE_HR_RATE,
    BULLPEN_K_RATE,
    BULLPEN_BB_RATE,
    BULLPEN_HR_RATE,
)

logger = logging.getLogger(__name__)

# Max PAs a batter can get in a game
MAX_BATTER_PA = 7


@dataclass
class BatterSimulationResult:
    """Results from a batter game simulation.

    All arrays have shape (n_sims,).
    """

    k_samples: np.ndarray
    bb_samples: np.ndarray
    h_samples: np.ndarray
    hr_samples: np.ndarray
    single_samples: np.ndarray
    double_samples: np.ndarray
    triple_samples: np.ndarray
    tb_samples: np.ndarray
    r_samples: np.ndarray
    rbi_samples: np.ndarray
    hbp_samples: np.ndarray
    pa_samples: np.ndarray
    pa_vs_starter_samples: np.ndarray
    pa_vs_reliever_samples: np.ndarray
    n_sims: int = 0

    @property
    def hrr_samples(self) -> np.ndarray:
        """Hits + Runs + RBIs combined samples."""
        return self.h_samples + self.r_samples + self.rbi_samples

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all stats."""
        stats = {}
        for name in [
            "k", "bb", "h", "hr", "single", "double", "triple",
            "tb", "r", "rbi", "hbp", "pa", "hrr",
        ]:
            samples = getattr(self, f"{name}_samples")
            stats[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "q10": float(np.percentile(samples, 10)),
                "q90": float(np.percentile(samples, 90)),
            }
        return stats

    def over_probs(
        self,
        stat: str,
        lines: list[float] | None = None,
    ) -> pd.DataFrame:
        """Compute P(over X.5) for prop lines."""
        samples = getattr(self, f"{stat}_samples")
        if lines is None:
            max_val = int(np.percentile(samples, 99)) + 2
            lines = [x + 0.5 for x in range(max_val)]

        expected = float(np.mean(samples))
        std = float(np.std(samples))

        records = []
        for line in lines:
            p_over = float(np.mean(samples > line))
            records.append({
                "line": line,
                "p_over": p_over,
                "p_under": 1.0 - p_over,
                "expected": expected,
                "std": std,
            })
        return pd.DataFrame(records)


def simulate_batter_game(
    batter_k_rate_samples: np.ndarray,
    batter_bb_rate_samples: np.ndarray,
    batter_hr_rate_samples: np.ndarray,
    batting_order: int,
    starter_k_rate: float | np.ndarray,
    starter_bb_rate: float | np.ndarray,
    starter_hr_rate: float | np.ndarray,
    starter_bf_mu: float,
    starter_bf_sigma: float,
    matchup_k_lift: float = 0.0,
    matchup_bb_lift: float = 0.0,
    matchup_hr_lift: float = 0.0,
    bullpen_k_rate: float = BULLPEN_K_RATE,
    bullpen_bb_rate: float = BULLPEN_BB_RATE,
    bullpen_hr_rate: float = BULLPEN_HR_RATE,
    bullpen_matchup_k_lift: float = 0.0,
    bullpen_matchup_bb_lift: float = 0.0,
    bullpen_matchup_hr_lift: float = 0.0,
    batter_babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_k_lift: float = 0.0,
    park_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    park_h_babip_adj: float = 0.0,
    weather_k_lift: float = 0.0,
    form_k_lift: float = 0.0,
    form_bb_lift: float = 0.0,
    form_hr_lift: float = 0.0,
    game_context: GameContext | None = None,
    batter_bip_probs: np.ndarray | None = None,
    n_sims: int = 50_000,
    random_seed: int = 42,
) -> BatterSimulationResult:
    """Run vectorized Monte Carlo batter game simulation.

    Parameters
    ----------
    batter_k_rate_samples : np.ndarray
        Batter K% posterior samples from Layer 1.
    batter_bb_rate_samples : np.ndarray
        Batter BB% posterior samples.
    batter_hr_rate_samples : np.ndarray
        Batter HR/PA posterior samples.
    batting_order : int
        Batting order position (1-9).
    starter_k_rate : float or np.ndarray
        Opposing starter's K% rate (scalar mean or posterior samples).
    starter_bb_rate : float or np.ndarray
        Opposing starter's BB% rate (scalar mean or posterior samples).
    starter_hr_rate : float or np.ndarray
        Opposing starter's HR/BF rate (scalar mean or posterior samples).
    starter_bf_mu : float
        Starter's expected BF for this game.
    starter_bf_sigma : float
        Starter's BF std.
    matchup_k_lift : float
        Batter-vs-starter matchup logit lift for K.
    matchup_bb_lift : float
        Batter-vs-starter matchup logit lift for BB.
    matchup_hr_lift : float
        Batter-vs-starter matchup logit lift for HR.
    bullpen_k_rate : float
        Opposing team bullpen aggregate K rate.
    bullpen_bb_rate : float
        Opposing team bullpen aggregate BB rate.
    bullpen_hr_rate : float
        Opposing team bullpen aggregate HR rate.
    batter_babip_adj : float
        Batter BABIP adjustment for BIP outcomes.
    umpire_k_lift, umpire_bb_lift : float
        Umpire tendency adjustments on logit scale.
    park_k_lift, park_bb_lift, park_hr_lift : float
        Park factor adjustments on logit scale.
    park_h_babip_adj : float
        Park factor BABIP adjustment for hits.
    weather_k_lift : float
        Weather K adjustment on logit scale.
    form_k_lift : float
        Batter rolling form K% logit lift (from form_model).
    form_bb_lift : float
        Batter rolling form BB% logit lift.
    form_hr_lift : float
        Batter rolling form HR logit lift (includes HR/PA accel + hard-hit).
    game_context : GameContext, optional
        Per-game environmental lifts. When provided, its fields supply
        defaults for any environmental lift parameter left at 0.0.
        Explicit keyword arguments always take priority over the context.
    n_sims : int
        Number of simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    BatterSimulationResult
        Joint distributions over batter counting stats.
    """
    # Resolve environmental lifts: explicit kwargs beat GameContext
    if game_context is not None:
        if umpire_k_lift == 0.0:
            umpire_k_lift = game_context.umpire_k_lift
        if umpire_bb_lift == 0.0:
            umpire_bb_lift = game_context.umpire_bb_lift
        if park_k_lift == 0.0:
            park_k_lift = game_context.park_k_lift
        if park_bb_lift == 0.0:
            park_bb_lift = game_context.park_bb_lift
        if park_hr_lift == 0.0:
            park_hr_lift = game_context.park_hr_lift
        if park_h_babip_adj == 0.0:
            park_h_babip_adj = game_context.park_h_babip_adj
        if weather_k_lift == 0.0:
            weather_k_lift = game_context.weather_k_lift

    rng = np.random.default_rng(random_seed)
    pa_model = PAOutcomeModel()

    # Resample batter posteriors to n_sims
    batter_k = resample_posterior(batter_k_rate_samples, n_sims, rng)
    batter_bb = resample_posterior(batter_bb_rate_samples, n_sims, rng)
    batter_hr = resample_posterior(batter_hr_rate_samples, n_sims, rng)

    # --- 1. Draw total PAs and starter BF ---
    total_pa = draw_total_pa(batting_order, rng, n_sims)

    starter_bf = draw_bf_samples(
        mu_bf=starter_bf_mu,
        sigma_bf=starter_bf_sigma,
        n_draws=n_sims,
        bf_min=9,
        bf_max=35,
        rng=rng,
    )

    # Split PAs into vs-starter and vs-reliever
    pa_vs_starter, pa_vs_reliever = split_pa_starter_reliever(
        total_pa, batting_order, starter_bf,
    )

    # --- 2. Compute per-sim pitcher quality lifts ---
    starter_k_lift = pitcher_rate_to_lift_array(starter_k_rate, SIM_LEAGUE_K_RATE, n_sims, rng)
    starter_bb_lift = pitcher_rate_to_lift_array(starter_bb_rate, SIM_LEAGUE_BB_RATE, n_sims, rng)
    starter_hr_lift = pitcher_rate_to_lift_array(starter_hr_rate, SIM_LEAGUE_HR_RATE, n_sims, rng)

    bullpen_k_lift = pitcher_rate_to_lift_array(bullpen_k_rate, SIM_LEAGUE_K_RATE, n_sims, rng)
    bullpen_bb_lift = pitcher_rate_to_lift_array(bullpen_bb_rate, SIM_LEAGUE_BB_RATE, n_sims, rng)
    bullpen_hr_lift = pitcher_rate_to_lift_array(bullpen_hr_rate, SIM_LEAGUE_HR_RATE, n_sims, rng)

    # --- 3. Accumulators ---
    k_total = np.zeros(n_sims, dtype=np.int32)
    bb_total = np.zeros(n_sims, dtype=np.int32)
    h_total = np.zeros(n_sims, dtype=np.int32)
    hr_total = np.zeros(n_sims, dtype=np.int32)
    single_total = np.zeros(n_sims, dtype=np.int32)
    double_total = np.zeros(n_sims, dtype=np.int32)
    triple_total = np.zeros(n_sims, dtype=np.int32)
    hbp_total = np.zeros(n_sims, dtype=np.int32)
    r_total = np.zeros(n_sims, dtype=np.int32)
    rbi_total = np.zeros(n_sims, dtype=np.int32)

    # --- 4. Simulate each PA ---
    for pa_num in range(MAX_BATTER_PA):
        active = pa_num < total_pa
        n_active = active.sum()
        if n_active == 0:
            break

        # Determine if facing starter or reliever for this PA
        # PA pa_num happens at global BF = batting_order + 9 * pa_num
        global_pos = batting_order + 9 * pa_num
        vs_starter = active & (global_pos <= starter_bf)

        # Build rates: use batter posteriors as base,
        # add pitcher quality lift + matchup lift
        k_logit_base = safe_logit(batter_k[active])
        bb_logit_base = safe_logit(batter_bb[active])
        hr_logit_base = safe_logit(batter_hr[active])

        # Pitcher quality + matchup lifts
        vs_starter_active = vs_starter[active]

        k_pitcher_lift = np.where(
            vs_starter_active,
            starter_k_lift[active] + matchup_k_lift,
            bullpen_k_lift[active],  # K matchup lift skipped — adds noise for relievers
        )
        bb_pitcher_lift = np.where(
            vs_starter_active,
            starter_bb_lift[active] + matchup_bb_lift,
            bullpen_bb_lift[active],  # BB matchup lift skipped — adds noise for relievers
        )
        hr_pitcher_lift = np.where(
            vs_starter_active,
            starter_hr_lift[active] + matchup_hr_lift,
            bullpen_hr_lift[active] + bullpen_matchup_hr_lift,
        )

        # Final adjusted rates (including rolling form lifts)
        k_rate_adj = expit(
            k_logit_base + k_pitcher_lift
            + umpire_k_lift + park_k_lift + weather_k_lift
            + form_k_lift
        )
        bb_rate_adj = expit(
            bb_logit_base + bb_pitcher_lift + umpire_bb_lift + park_bb_lift
            + form_bb_lift
        )
        hr_rate_adj = expit(
            hr_logit_base + hr_pitcher_lift + park_hr_lift
            + form_hr_lift
        )

        # Draw outcomes
        probs = pa_model.compute_pa_probs(
            pitcher_k_rate=k_rate_adj,
            pitcher_bb_rate=bb_rate_adj,
            pitcher_hr_rate=hr_rate_adj,
        )
        # Per-sim batter BIP probability array (tile the single (4,) vector)
        batter_bip_arr = None
        if batter_bip_probs is not None:
            batter_bip_arr = np.broadcast_to(
                np.asarray(batter_bip_probs, dtype=np.float64), (n_active, 4),
            )
        outcomes = pa_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active,
            babip_adj=batter_babip_adj + park_h_babip_adj,
            batter_bip_probs=batter_bip_arr,
        )

        # Accumulate
        k_total[active] += (outcomes == PA_STRIKEOUT).astype(np.int32)
        bb_total[active] += (outcomes == PA_WALK).astype(np.int32)
        hbp_total[active] += (outcomes == PA_HBP).astype(np.int32)
        hr_total[active] += (outcomes == PA_HOME_RUN).astype(np.int32)
        single_total[active] += (outcomes == PA_SINGLE).astype(np.int32)
        double_total[active] += (outcomes == PA_DOUBLE).astype(np.int32)
        triple_total[active] += (outcomes == PA_TRIPLE).astype(np.int32)

        is_hit = np.isin(outcomes, [PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN])
        h_total[active] += is_hit.astype(np.int32)

        # R/RBI scoring (calibrated to 2019-2025 league averages)
        # HR: batter always scores + drives in ~0.4 runners on avg
        is_hr = (outcomes == PA_HOME_RUN)
        r_total[active] += is_hr.astype(np.int32)
        rbi_total[active] += is_hr.astype(np.int32)
        # HR extra RBI: ~40% chance of driving in an additional runner
        rbi_total[active] += (is_hr & (rng.random(n_active) < 0.40)).astype(np.int32)

        # Non-HR hits: calibrated from actual R/PA and RBI/PA rates
        non_hr_hit = is_hit & ~is_hr
        n_non_hr = non_hr_hit.sum()
        if n_non_hr > 0:
            r_total[active] += (non_hr_hit & (rng.random(n_active) < 0.36)).astype(np.int32)
            rbi_total[active] += (non_hr_hit & (rng.random(n_active) < 0.35)).astype(np.int32)

        # BB/HBP: ~18% chance of eventually scoring
        on_base_no_hit = np.isin(outcomes, [PA_WALK, PA_HBP])
        n_ob = on_base_no_hit.sum()
        if n_ob > 0:
            r_total[active] += (on_base_no_hit & (rng.random(n_active) < 0.18)).astype(np.int32)
            # RBI on BB/HBP (bases loaded walk): ~3%
            rbi_total[active] += (on_base_no_hit & (rng.random(n_active) < 0.03)).astype(np.int32)

    # Compute total bases
    tb_total = (
        single_total
        + 2 * double_total
        + 3 * triple_total
        + 4 * hr_total
    )

    return BatterSimulationResult(
        k_samples=k_total,
        bb_samples=bb_total,
        h_samples=h_total,
        hr_samples=hr_total,
        single_samples=single_total,
        double_samples=double_total,
        triple_samples=triple_total,
        tb_samples=tb_total,
        r_samples=r_total,
        rbi_samples=rbi_total,
        hbp_samples=hbp_total,
        pa_samples=total_pa,
        pa_vs_starter_samples=pa_vs_starter,
        pa_vs_reliever_samples=pa_vs_reliever,
        n_sims=n_sims,
    )
