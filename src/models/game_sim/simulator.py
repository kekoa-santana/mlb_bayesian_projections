"""
Vectorized Monte Carlo game simulator.

Simulates plate appearances sequentially through the batting order,
advancing all N simulations in lockstep. Each PA draws a pitch count
and outcome, updates game state, and checks for pitcher exit.

Produces joint distributions over all pitcher counting stats from
a single simulation run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.models.game_sim.bip_model import BIPOutcomeModel
from src.models.game_sim.exit_model import ExitModel
from src.models.game_sim.pa_outcome_model import (
    PA_DOUBLE,
    PA_HBP,
    PA_HOME_RUN,
    PA_OUT,
    PA_SINGLE,
    PA_STRIKEOUT,
    PA_TRIPLE,
    PA_WALK,
    PAOutcomeModel,
    compute_fatigue_adjustments,
)
from src.models.game_sim.pitch_count_model import PitchCountModel
from src.models.game_sim.tto_model import BF_PER_TTO, get_tto_for_bf

logger = logging.getLogger(__name__)

_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6

# Maximum PA per game (safety valve)
MAX_PA_PER_GAME = 45


@dataclass
class SimulationResult:
    """Results from a game simulation run.

    All arrays have shape (n_sims,) — one value per simulated game.
    """

    k_samples: np.ndarray
    bb_samples: np.ndarray
    h_samples: np.ndarray
    hr_samples: np.ndarray
    hbp_samples: np.ndarray
    bf_samples: np.ndarray
    pitch_count_samples: np.ndarray
    outs_samples: np.ndarray
    runs_samples: np.ndarray
    n_sims: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all stats.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict with mean, std, median, q10, q90 per stat.
        """
        stats = {}
        for name in [
            "k", "bb", "h", "hr", "hbp", "bf", "pitch_count", "outs", "runs",
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
        """Compute P(over X.5) for prop lines.

        Parameters
        ----------
        stat : str
            Stat name (e.g., 'k', 'bb', 'h', 'hr').
        lines : list[float], optional
            Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

        Returns
        -------
        pd.DataFrame
            Columns: line, p_over, p_under, expected, std.
        """
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

    def ip_samples(self) -> np.ndarray:
        """Compute innings pitched from outs."""
        full_innings = self.outs_samples // 3
        partial = self.outs_samples % 3
        return full_innings + partial / 10.0  # Baseball IP notation


def simulate_game(
    pitcher_k_rate_samples: np.ndarray,
    pitcher_bb_rate_samples: np.ndarray,
    pitcher_hr_rate_samples: np.ndarray,
    lineup_matchup_lifts: dict[str, np.ndarray],
    tto_lifts: dict[str, np.ndarray],
    pitcher_ppa_adj: float,
    batter_ppa_adjs: np.ndarray,
    exit_model: ExitModel,
    pitcher_avg_pitches: float = 88.0,
    babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> SimulationResult:
    """Run vectorized Monte Carlo game simulation.

    Advances all simulations one PA at a time through the lineup order.
    Simulations where the pitcher has exited are masked out.

    Parameters
    ----------
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples from Layer 1.
    pitcher_bb_rate_samples : np.ndarray
        BB% posterior samples from Layer 1.
    pitcher_hr_rate_samples : np.ndarray
        HR/BF posterior samples from Layer 1.
    lineup_matchup_lifts : dict[str, np.ndarray]
        Per-stat matchup logit lifts. Keys: 'k', 'bb', 'hr'.
        Each value shape (9,) for 9 batting order slots.
    tto_lifts : dict[str, np.ndarray]
        TTO logit lifts. Keys: 'k', 'bb', 'hr'.
        Each value shape (3,) for TTO 1, 2, 3.
    pitcher_ppa_adj : float
        Pitcher pitches-per-PA adjustment.
    batter_ppa_adjs : np.ndarray
        Shape (9,) batter P/PA adjustments.
    exit_model : ExitModel
        Trained pitcher exit model.
    pitcher_avg_pitches : float
        Pitcher's historical average exit pitch count.
    babip_adj : float
        Pitcher BABIP adjustment for BIP outcomes.
    umpire_k_lift : float
        Umpire K-rate logit lift.
    umpire_bb_lift : float
        Umpire BB-rate logit lift.
    park_hr_lift : float
        Park HR logit lift.
    weather_k_lift : float
        Weather K logit lift.
    n_sims : int
        Number of Monte Carlo simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    SimulationResult
        Joint distributions over all pitcher counting stats.
    """
    rng = np.random.default_rng(random_seed)

    # Initialize component models
    pitch_count_model = PitchCountModel()
    pa_outcome_model = PAOutcomeModel()

    # Resample posterior draws to n_sims
    def _resample(arr: np.ndarray) -> np.ndarray:
        if len(arr) == n_sims:
            return arr.copy()
        idx = rng.choice(len(arr), size=n_sims, replace=True)
        return arr[idx]

    k_rates = _resample(pitcher_k_rate_samples)
    bb_rates = _resample(pitcher_bb_rate_samples)
    hr_rates = _resample(pitcher_hr_rate_samples)

    # Default matchup lifts to zeros if missing stat keys
    for stat in ("k", "bb", "hr"):
        if stat not in lineup_matchup_lifts:
            lineup_matchup_lifts[stat] = np.zeros(9)
        if stat not in tto_lifts:
            tto_lifts[stat] = np.zeros(3)

    # --- Game state arrays (all shape n_sims) ---
    pitches = np.zeros(n_sims, dtype=np.int32)
    outs = np.zeros(n_sims, dtype=np.int32)
    inning = np.ones(n_sims, dtype=np.int32)
    inning_outs = np.zeros(n_sims, dtype=np.int32)  # 0, 1, 2 within inning
    lineup_pos = np.zeros(n_sims, dtype=np.int32)    # 0-8
    bf_count = np.zeros(n_sims, dtype=np.int32)
    runners = np.zeros(n_sims, dtype=np.int32)        # simplified count
    runs = np.zeros(n_sims, dtype=np.int32)
    score_diff = np.zeros(n_sims, dtype=np.int32)     # pitcher team perspective

    # Accumulators
    k_total = np.zeros(n_sims, dtype=np.int32)
    bb_total = np.zeros(n_sims, dtype=np.int32)
    h_total = np.zeros(n_sims, dtype=np.int32)
    hr_total = np.zeros(n_sims, dtype=np.int32)
    hbp_total = np.zeros(n_sims, dtype=np.int32)

    # Recent trouble tracker (last 2 PA: BB, H, HBP)
    recent_trouble = np.zeros(n_sims, dtype=np.int32)
    prev_trouble = np.zeros(n_sims, dtype=np.int32)

    # Active mask — simulations where pitcher is still in the game
    active = np.ones(n_sims, dtype=bool)

    # --- Main simulation loop ---
    for pa_num in range(MAX_PA_PER_GAME):
        n_active = active.sum()
        if n_active == 0:
            break

        # Current batter slot (0-8)
        slot = lineup_pos[active] % 9
        # Current TTO
        tto = np.minimum(bf_count[active] // BF_PER_TTO, 2)

        # --- 1. Draw pitch count for this PA ---
        # Get per-batter P/PA adjustments for active sims
        batter_adj_active = batter_ppa_adjs[slot]
        pa_pitches = pitch_count_model.draw_pitches(
            pitcher_adj=pitcher_ppa_adj,
            batter_adj=batter_adj_active,
            rng=rng,
            n_draws=n_active,
        )
        pitches[active] += pa_pitches

        # --- 2. Compute PA outcome probabilities ---
        fatigue = compute_fatigue_adjustments(pitches[active])

        # Gather per-batter matchup lifts
        k_matchup = np.array([lineup_matchup_lifts["k"][s] for s in slot])
        bb_matchup = np.array([lineup_matchup_lifts["bb"][s] for s in slot])
        hr_matchup = np.array([lineup_matchup_lifts["hr"][s] for s in slot])

        # Gather TTO lifts
        k_tto = np.array([tto_lifts["k"][t] for t in tto])
        bb_tto = np.array([tto_lifts["bb"][t] for t in tto])
        hr_tto = np.array([tto_lifts["hr"][t] for t in tto])

        probs = pa_outcome_model.compute_pa_probs(
            pitcher_k_rate=k_rates[active],
            pitcher_bb_rate=bb_rates[active],
            pitcher_hr_rate=hr_rates[active],
            matchup_k_lift=k_matchup,
            matchup_bb_lift=bb_matchup,
            matchup_hr_lift=hr_matchup,
            tto_k_lift=k_tto,
            tto_bb_lift=bb_tto,
            tto_hr_lift=hr_tto,
            fatigue_k_lift=fatigue["k"],
            fatigue_bb_lift=fatigue["bb"],
            fatigue_hr_lift=fatigue["hr"],
            umpire_k_lift=umpire_k_lift,
            umpire_bb_lift=umpire_bb_lift,
            park_hr_lift=park_hr_lift,
            weather_k_lift=weather_k_lift,
        )

        # --- 3. Draw PA outcomes ---
        outcomes = pa_outcome_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active, babip_adj=babip_adj,
        )

        # --- 4. Update game state ---
        # Update accumulators
        k_total[active] += (outcomes == PA_STRIKEOUT).astype(np.int32)
        bb_total[active] += (outcomes == PA_WALK).astype(np.int32)
        hbp_total[active] += (outcomes == PA_HBP).astype(np.int32)
        hr_total[active] += (outcomes == PA_HOME_RUN).astype(np.int32)

        is_hit = np.isin(outcomes, [PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN])
        h_total[active] += is_hit.astype(np.int32)

        # Update outs
        is_out = np.isin(outcomes, [PA_STRIKEOUT, PA_OUT])
        outs[active] += is_out.astype(np.int32)
        inning_outs[active] += is_out.astype(np.int32)

        # Update runners (simplified model)
        # Outs clear no runners; walks/HBP/singles add 1; doubles add 1
        # (and may score a runner); HR clears bases + scores all + batter
        is_on_base = np.isin(
            outcomes, [PA_WALK, PA_HBP, PA_SINGLE, PA_DOUBLE, PA_TRIPLE]
        )

        # Score runs on HR: all runners + batter
        hr_mask_active = (outcomes == PA_HOME_RUN)
        runs_scored = np.where(hr_mask_active, runners[active] + 1, 0)

        # Score some runners on doubles/triples (simplified)
        double_mask = (outcomes == PA_DOUBLE)
        triple_mask = (outcomes == PA_TRIPLE)
        runs_scored += np.where(
            double_mask, np.minimum(runners[active], 2), 0
        )
        runs_scored += np.where(
            triple_mask, runners[active], 0
        )

        runs[active] += runs_scored.astype(np.int32)
        score_diff[active] -= runs_scored.astype(np.int32)

        # Update runners
        # HR: clear bases
        runners[active] = np.where(hr_mask_active, 0, runners[active])
        # Doubles/triples: some runners scored, batter on base
        runners[active] = np.where(
            double_mask,
            np.minimum(runners[active] - np.minimum(runners[active], 2) + 1, 3),
            runners[active],
        )
        runners[active] = np.where(
            triple_mask, 1, runners[active]  # batter on 3rd
        )
        # Single/walk/HBP: add batter, keep existing (capped at 3)
        runners[active] = np.where(
            is_on_base & ~double_mask & ~triple_mask,
            np.minimum(runners[active] + 1, 3),
            runners[active],
        )
        # Outs: don't change runner count (simplified)

        # Check for inning change (3 outs in inning)
        inning_over = inning_outs[active] >= 3
        inning[active] = np.where(inning_over, inning[active] + 1, inning[active])
        inning_outs[active] = np.where(inning_over, 0, inning_outs[active])
        runners[active] = np.where(inning_over, 0, runners[active])

        # Update recent trouble
        is_trouble = np.isin(
            outcomes,
            [PA_WALK, PA_HBP, PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN],
        ).astype(np.int32)
        new_recent = prev_trouble[active] + is_trouble
        prev_trouble[active] = is_trouble
        recent_trouble[active] = new_recent

        # Advance lineup position and BF count
        bf_count[active] += 1
        lineup_pos[active] += 1

        # --- 5. Exit check ---
        # Force exit on complete game (27 outs)
        force_exit = outs[active] >= 27
        # Force exit on pitch count hard cap
        force_exit |= pitches[active] >= 130

        # Model-based exit probability
        current_tto = np.minimum(bf_count[active] // BF_PER_TTO + 1, 3)
        exit_prob = exit_model.predict_exit_prob(
            cumulative_pitches=pitches[active],
            inning=inning[active],
            inning_outs=inning_outs[active],
            score_diff=score_diff[active],
            runners=runners[active],
            tto=current_tto,
            recent_trouble=recent_trouble[active],
            pitcher_avg_pitches=pitcher_avg_pitches,
        )

        # Draw exit decisions
        exit_draw = rng.random(n_active) < exit_prob
        exits = force_exit | exit_draw

        # Update active mask
        active_indices = np.where(active)[0]
        active[active_indices[exits]] = False

    return SimulationResult(
        k_samples=k_total,
        bb_samples=bb_total,
        h_samples=h_total,
        hr_samples=hr_total,
        hbp_samples=hbp_total,
        bf_samples=bf_count,
        pitch_count_samples=pitches,
        outs_samples=outs,
        runs_samples=runs,
        n_sims=n_sims,
    )


def predict_game(
    pitcher_id: int,
    season: int,
    lineup_batter_ids: list[int],
    pitcher_k_rate_samples: np.ndarray,
    pitcher_bb_rate_samples: np.ndarray,
    pitcher_hr_rate_samples: np.ndarray,
    lineup_matchup_lifts: dict[str, np.ndarray],
    tto_lifts: dict[str, np.ndarray],
    pitcher_features: pd.DataFrame,
    batter_features: pd.DataFrame,
    exit_model: ExitModel,
    pitcher_avg_pitches: float = 88.0,
    babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    park_k_lift: float = 0.0,
    park_bb_lift: float = 0.0,
    catcher_k_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """High-level game prediction interface.

    Computes pitch count features, runs simulation, and returns
    comprehensive results including prop line probabilities.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season for feature lookup.
    lineup_batter_ids : list[int]
        9 batter IDs in batting order.
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples.
    pitcher_bb_rate_samples : np.ndarray
        BB% posterior samples.
    pitcher_hr_rate_samples : np.ndarray
        HR/BF posterior samples.
    lineup_matchup_lifts : dict[str, np.ndarray]
        Per-stat matchup lifts, shape (9,) each.
    tto_lifts : dict[str, np.ndarray]
        TTO lifts, shape (3,) each.
    pitcher_features : pd.DataFrame
        Pitcher pitch count features.
    batter_features : pd.DataFrame
        Batter pitch count features.
    exit_model : ExitModel
        Trained exit model.
    pitcher_avg_pitches : float
        Average exit pitch count.
    babip_adj : float
        Pitcher BABIP adjustment.
    umpire_k_lift, umpire_bb_lift, park_hr_lift, weather_k_lift : float
        Context adjustments.
    n_sims : int
        Number of simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict[str, Any]
        Keys: 'result' (SimulationResult), 'summary', 'k_over_probs',
        'bb_over_probs', 'h_over_probs', 'hr_over_probs'.
    """
    from src.models.game_sim.pitch_count_model import build_pitch_count_features

    # Build pitch count adjustments
    pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
        pitcher_features=pitcher_features,
        batter_features=batter_features,
        pitcher_id=pitcher_id,
        batter_ids=lineup_batter_ids,
        season=season,
    )

    # Run simulation — combine park + catcher + umpire + weather lifts
    total_k_lift = umpire_k_lift + park_k_lift + catcher_k_lift + weather_k_lift
    total_bb_lift = umpire_bb_lift + park_bb_lift

    result = simulate_game(
        pitcher_k_rate_samples=pitcher_k_rate_samples,
        pitcher_bb_rate_samples=pitcher_bb_rate_samples,
        pitcher_hr_rate_samples=pitcher_hr_rate_samples,
        lineup_matchup_lifts=lineup_matchup_lifts,
        tto_lifts=tto_lifts,
        pitcher_ppa_adj=pitcher_ppa_adj,
        batter_ppa_adjs=batter_ppa_adjs,
        exit_model=exit_model,
        pitcher_avg_pitches=pitcher_avg_pitches,
        babip_adj=babip_adj,
        umpire_k_lift=total_k_lift,
        umpire_bb_lift=total_bb_lift,
        park_hr_lift=park_hr_lift,
        weather_k_lift=0.0,  # already folded into total_k_lift
        n_sims=n_sims,
        random_seed=random_seed,
    )

    return {
        "result": result,
        "summary": result.summary(),
        "k_over_probs": result.over_probs("k"),
        "bb_over_probs": result.over_probs("bb"),
        "h_over_probs": result.over_probs("h"),
        "hr_over_probs": result.over_probs("hr"),
    }
