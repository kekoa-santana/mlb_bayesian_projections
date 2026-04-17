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

from dataclasses import replace as _dc_replace

from src.models.game_sim.bip_model import BIPOutcomeModel
from src.models.game_sim.bullpen_model import TeamBullpenProfile
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
    GameContext,
    PAOutcomeModel,
    compute_fatigue_adjustments,
)
from src.models.game_sim.pitch_count_model import PitchCountModel
from src.models.game_sim.tto_model import BF_PER_TTO, get_tto_for_bf
from src.models.game_sim._sim_utils import resample_posterior, MATCHUP_DAMPEN
from src.utils.constants import (
    CLIP_LO,
    CLIP_HI,
    BULLPEN_K_RATE,
    BULLPEN_BB_RATE,
    BULLPEN_HR_RATE,
)

logger = logging.getLogger(__name__)

# Maximum PA per game (safety valve)
MAX_PA_PER_GAME = 45

# Default exit model calibration offset (logit scale).
# The exit model over-predicts exit probability, pulling pitchers ~1.4 BF early.
# This offset reduces exit probability to match observed BF distribution.
_DEFAULT_EXIT_CALIBRATION_OFFSET = -0.35

# Stamina offset parameters.
# Workhorses (high avg IP) get more negative offset → stay longer.
# Short-leash pitchers get less negative / positive offset → exit sooner.
# Scale chosen so output IP spread matches observed 2025 distribution
# (real std ≈ 0.49 IP across starters, 3.7–6.4 range).
_STAMINA_POP_MEAN_IP = 5.22  # 2022-2025 population mean IP for starters (10+ starts, BF >= 9)
_STAMINA_POP_STD_IP = 0.50   # 2022-2025 population std
_STAMINA_LOGIT_SCALE = 0.45  # logit shift per z-score of avg IP

# BF prior bridge: population BF mean for z-scoring
# Revalidated 2022-2025 with BF >= 9 filter (excludes openers/bullpen games)
_POP_BF_MU = 22.4
_POP_BF_STD = 1.8   # Between-pitcher std of mean BF (not within-game)
_BF_LOGIT_SCALE = 0.25  # logit shift per z-score of pitcher BF mean (tuned from 0.35 to recenter BF bias)

# BF-anchored exit: logit shift per BF deviation from target
# At target_bf: no shift.  3 BF past target → shift = 0.3 * 3 = 0.9 logit.
# Calibrate via backtest so mean pred_bf ≈ mean actual_bf.
_BF_ANCHOR_K = 0.3

# Mid-inning blowup threshold: yank pitcher if runs_this_inning >= this value
_BLOWUP_RUNS_THRESHOLD = 4

# Bullpen state adjustment parameters
_BULLPEN_WORKLOAD_POP_MEAN = 5.0   # Mean relief IP over trailing 3 days
_BULLPEN_WORKLOAD_POP_STD = 2.5    # Std of trailing 3-day bullpen IP
_BULLPEN_LOGIT_SCALE = -0.12       # Negative → taxed bullpen keeps starter in longer

# Lineup patience adjustment parameters
_LINEUP_PPA_POP_MEAN = 0.0   # Population mean lineup aggregate P/PA adj
_LINEUP_PPA_SENSITIVITY = 0.10  # Exit offset per unit of lineup P/PA aggregate


def compute_stamina_offset(
    pitcher_avg_ip: float,
    base_offset: float = _DEFAULT_EXIT_CALIBRATION_OFFSET,
) -> float:
    """Compute per-pitcher exit calibration offset based on stamina.

    Workhorses (high avg IP) get a more negative offset (lower exit prob),
    short-leash pitchers get a less negative offset (higher exit prob).

    Parameters
    ----------
    pitcher_avg_ip : float
        Pitcher's historical average innings pitched per start.
    base_offset : float
        Population-level calibration offset (logit scale).

    Returns
    -------
    float
        Per-pitcher exit calibration offset (logit scale).
    """
    z = (pitcher_avg_ip - _STAMINA_POP_MEAN_IP) / _STAMINA_POP_STD_IP
    # Negative direction: higher avg IP → more negative offset → stays longer
    return base_offset - _STAMINA_LOGIT_SCALE * z


def compute_bf_calibration_offset(
    mu_bf: float,
    base_offset: float = _DEFAULT_EXIT_CALIBRATION_OFFSET,
) -> float:
    """Compute exit offset anchored to pitcher's known BF distribution.

    Uses the bf_model prior (empirical Bayes from historical starts) to
    set the exit calibration so the simulator targets the correct BF mean.
    Replaces compute_stamina_offset when bf_model priors are available.

    Parameters
    ----------
    mu_bf : float
        Pitcher's shrinkage-estimated mean BF per start (from bf_model).
    base_offset : float
        Population-level calibration offset (logit scale).

    Returns
    -------
    float
        Per-pitcher exit calibration offset (logit scale).
    """
    z = (mu_bf - _POP_BF_MU) / _POP_BF_STD
    # Higher BF mean → more negative offset → stays longer
    return base_offset - _BF_LOGIT_SCALE * z


def compute_bullpen_workload_offset(
    bullpen_trailing_ip: float,
) -> float:
    """Compute exit offset adjustment for bullpen fatigue state.

    When the bullpen is taxed (high recent IP), managers let starters
    go deeper. When the bullpen is fresh, managers pull starters earlier.

    Parameters
    ----------
    bullpen_trailing_ip : float
        Team's total bullpen IP over the trailing 3 days.

    Returns
    -------
    float
        Additive exit calibration offset (logit scale).
        Negative values reduce exit probability (starter stays longer).
    """
    z = (bullpen_trailing_ip - _BULLPEN_WORKLOAD_POP_MEAN) / _BULLPEN_WORKLOAD_POP_STD
    # Positive z = taxed bullpen → negative offset → starter stays longer
    return _BULLPEN_LOGIT_SCALE * z


def compute_lineup_patience_offset(
    lineup_ppa_aggregate: float,
) -> float:
    """Compute exit offset for opposing lineup patience.

    Patient lineups (high aggregate P/PA adjustment) drive up pitch counts
    faster, leading to earlier pitcher exits independent of outcome rates.

    Parameters
    ----------
    lineup_ppa_aggregate : float
        Mean of the 9 batter P/PA adjustments for the opposing lineup.

    Returns
    -------
    float
        Additive exit calibration offset (logit scale).
        Positive values increase exit probability (pitcher pulled earlier).
    """
    return _LINEUP_PPA_SENSITIVITY * lineup_ppa_aggregate


def compute_exit_offset(
    *,
    mu_bf: float | None = None,
    pitcher_avg_ip: float | None = None,
    bullpen_trailing_ip: float | None = None,
    lineup_ppa_aggregate: float | None = None,
    base_offset: float = _DEFAULT_EXIT_CALIBRATION_OFFSET,
) -> float:
    """Unified exit calibration offset combining all BF-related signals.

    Priority for pitcher stamina:
    1. mu_bf from bf_model prior (if available) — best signal
    2. pitcher_avg_ip stamina offset (fallback)

    Additional additive adjustments:
    - Bullpen workload state
    - Opposing lineup patience

    Parameters
    ----------
    mu_bf : float, optional
        Pitcher's shrinkage BF mean from bf_model.
    pitcher_avg_ip : float, optional
        Pitcher's avg IP per start (fallback for stamina).
    bullpen_trailing_ip : float, optional
        Team bullpen IP over trailing 3 days.
    lineup_ppa_aggregate : float, optional
        Mean lineup P/PA adjustment for opposing batters.
    base_offset : float
        Population-level calibration offset.

    Returns
    -------
    float
        Combined exit calibration offset (logit scale).
    """
    # Core pitcher-specific offset: prefer BF prior, fall back to avg IP
    if mu_bf is not None:
        offset = compute_bf_calibration_offset(mu_bf, base_offset)
    elif pitcher_avg_ip is not None:
        offset = compute_stamina_offset(pitcher_avg_ip, base_offset)
    else:
        offset = base_offset

    # Additive adjustments
    if bullpen_trailing_ip is not None:
        offset += compute_bullpen_workload_offset(bullpen_trailing_ip)

    if lineup_ppa_aggregate is not None:
        offset += compute_lineup_patience_offset(lineup_ppa_aggregate)

    return offset


@dataclass
class SimulationResult:
    """Results from a game simulation run.

    All arrays have shape (n_sims,) — one value per simulated game.

    Counting stats (k/bb/h/hr/hbp/bf/pitch_count/outs) reflect the
    **starter only** and back pitcher prop probabilities.

    ``runs_samples`` is the **full game** runs allowed by the pitching
    team (starter innings + bullpen tail). For diagnostics, the
    component arrays ``starter_runs_samples`` and ``bullpen_runs_samples``
    are also exposed and always sum to ``runs_samples``.
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
    starter_runs_samples: np.ndarray | None = None
    bullpen_runs_samples: np.ndarray | None = None
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


# Safety cap on bullpen tail PAs. A full game has 27 outs; after a typical
# starter exit there are ~11 outs left → ~13 bullpen PAs. 80 leaves huge
# headroom for blow-up innings without runaway loops.
_MAX_BULLPEN_PA = 80

# Runner advancement probabilities (league-average, see CLAUDE discussion
# 2026-04-10). These recover the ~30% run under-estimate that the count-only
# runner model was producing.
_SAC_FLY_PROB_0OUT = 0.55  # R3 scores on non-K out with 0 outs
_SAC_FLY_PROB_1OUT = 0.35  # R3 scores on non-K out with 1 out
_R2_SCORES_ON_SINGLE = 0.65             # empirical league ~0.65-0.70
_R1_SCORES_ON_DOUBLE = 0.40
_R1_TO_3B_ON_SINGLE = 0.27              # first-to-third when 2B is empty
_R2_TO_3B_ON_NONK_OUT_0OUT = 0.25       # productive out to right side
_R2_TO_3B_ON_NONK_OUT_1OUT = 0.15


def _advance_runners(
    outcomes: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    inning_outs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply PA outcomes to base state and return runs + new base occupancy.

    Vectorized over ``n`` active sim lanes. ``r1/r2/r3`` are int32 0/1 flags
    for base occupancy; ``inning_outs`` is the PRE-PA out count in the
    current inning (so sac fly eligibility can be checked before the out is
    recorded).

    Advancement rules (approved 2026-04-10):
    - **Strikeout** / **non-K out**: no movement, except a runner on 3B
      scores with prob 0.50 (0 outs) / 0.30 (1 out) / 0.0 (2 outs) to model
      sac flies and productive outs.
    - **Walk / HBP**: forced advancement only. Bases-loaded walk scores 1.
    - **Single**: batter → 1B, r3 scores, r2 scores (p=0.60) else → 3B,
      r1 → 2B.
    - **Double**: batter → 2B, r3 scores, r2 scores, r1 scores (p=0.40)
      else → 3B.
    - **Triple**: all on-base runners score, batter → 3B.
    - **HR**: batter + all on-base runners score, bases clear.

    No GIDPs, errors, passed balls, or 1B→3B hit-and-run advancement —
    these cut both ways and are deferred.

    Returns
    -------
    (runs, new_r1, new_r2, new_r3)
        Each shape ``(n,)``. ``runs`` is int32 count, base flags are int32
        0/1.
    """
    n = outcomes.shape[0]
    runs = np.zeros(n, dtype=np.int32)
    new_r1 = r1.copy()
    new_r2 = r2.copy()
    new_r3 = r3.copy()

    # --- Non-K out: sac fly with runner on 3B, then R2 productive-out adv ---
    out_mask = outcomes == PA_OUT
    sf_prob = np.where(
        out_mask & (r3 == 1) & (inning_outs == 0),
        _SAC_FLY_PROB_0OUT,
        np.where(
            out_mask & (r3 == 1) & (inning_outs == 1),
            _SAC_FLY_PROB_1OUT,
            0.0,
        ),
    )
    sf_draw = rng.random(n) < sf_prob
    runs += sf_draw.astype(np.int32)
    new_r3 = np.where(sf_draw, 0, new_r3)

    # R2 → 3B on productive out (groundout to right side). Only fires when
    # 3B will be empty after sac fly handling and there are fewer than 2
    # outs. Approximates the productive-out advancement rate that is
    # otherwise invisible to a three-true-outcome sim.
    r2_adv_prob = np.where(
        inning_outs == 0, _R2_TO_3B_ON_NONK_OUT_0OUT,
        np.where(inning_outs == 1, _R2_TO_3B_ON_NONK_OUT_1OUT, 0.0),
    )
    r2_adv_draw = rng.random(n) < r2_adv_prob
    r2_advances_on_out = out_mask & (r2 == 1) & (new_r3 == 0) & r2_adv_draw
    new_r3 = np.where(r2_advances_on_out, 1, new_r3)
    new_r2 = np.where(r2_advances_on_out, 0, new_r2)

    # --- Walk / HBP: forced advancement ---
    walk_mask = (outcomes == PA_WALK) | (outcomes == PA_HBP)
    bases_loaded_walk = walk_mask & (r1 == 1) & (r2 == 1) & (r3 == 1)
    runs += bases_loaded_walk.astype(np.int32)
    # r3 becomes 1 when r1 AND r2 are already on (the runner on 2B gets
    # pushed to 3B). r3=1 sims where r3 was already on also keep r3=1.
    walk_push_r3 = walk_mask & (r1 == 1) & (r2 == 1)
    # r2 becomes 1 when r1 was on (r1 pushed) — or if r2 was already on.
    walk_push_r2 = walk_mask & (r1 == 1)
    new_r3 = np.where(walk_push_r3, 1, new_r3)
    new_r2 = np.where(walk_push_r2, 1, new_r2)
    new_r1 = np.where(walk_mask, 1, new_r1)

    # --- Single ---
    single_mask = outcomes == PA_SINGLE
    r3_scores_single = single_mask & (r3 == 1)
    runs += r3_scores_single.astype(np.int32)

    r2_on_single = single_mask & (r2 == 1)
    r2_score_draw = rng.random(n) < _R2_SCORES_ON_SINGLE
    r2_scores_single = r2_on_single & r2_score_draw
    r2_holds_at_3_single = r2_on_single & ~r2_score_draw
    runs += r2_scores_single.astype(np.int32)

    # R1 → 3B first-to-third is only physically possible when 2B is empty
    # before the PA (otherwise traffic forces R1 to 2B).
    r1_on_single = single_mask & (r1 == 1)
    r1_to_3b_draw = rng.random(n) < _R1_TO_3B_ON_SINGLE
    r1_advances_to_3b = r1_on_single & (r2 == 0) & r1_to_3b_draw

    # new_r3: either R2 held at 3B (didn't score), or R1 advanced to 3B
    new_r3 = np.where(
        single_mask,
        (r2_holds_at_3_single | r1_advances_to_3b).astype(np.int32),
        new_r3,
    )
    # new_r2: R1 goes to 2B unless they took the extra base to 3B
    new_r2 = np.where(
        single_mask,
        (r1_on_single & ~r1_advances_to_3b).astype(np.int32),
        new_r2,
    )
    new_r1 = np.where(single_mask, 1, new_r1)

    # --- Double ---
    double_mask = outcomes == PA_DOUBLE
    runs += (double_mask & (r3 == 1)).astype(np.int32)
    runs += (double_mask & (r2 == 1)).astype(np.int32)

    r1_on_double = double_mask & (r1 == 1)
    r1_score_draw = rng.random(n) < _R1_SCORES_ON_DOUBLE
    r1_scores_double = r1_on_double & r1_score_draw
    r1_holds_at_3_double = r1_on_double & ~r1_score_draw
    runs += r1_scores_double.astype(np.int32)

    new_r3 = np.where(double_mask, r1_holds_at_3_double.astype(np.int32), new_r3)
    new_r2 = np.where(double_mask, 1, new_r2)
    new_r1 = np.where(double_mask, 0, new_r1)

    # --- Triple ---
    triple_mask = outcomes == PA_TRIPLE
    runs += (triple_mask & (r3 == 1)).astype(np.int32)
    runs += (triple_mask & (r2 == 1)).astype(np.int32)
    runs += (triple_mask & (r1 == 1)).astype(np.int32)
    new_r3 = np.where(triple_mask, 1, new_r3)
    new_r2 = np.where(triple_mask, 0, new_r2)
    new_r1 = np.where(triple_mask, 0, new_r1)

    # --- Home run ---
    hr_mask = outcomes == PA_HOME_RUN
    hr_runs = (
        hr_mask.astype(np.int32)
        + (hr_mask & (r1 == 1)).astype(np.int32)
        + (hr_mask & (r2 == 1)).astype(np.int32)
        + (hr_mask & (r3 == 1)).astype(np.int32)
    )
    runs += hr_runs
    new_r3 = np.where(hr_mask, 0, new_r3)
    new_r2 = np.where(hr_mask, 0, new_r2)
    new_r1 = np.where(hr_mask, 0, new_r1)

    return runs, new_r1, new_r2, new_r3


def simulate_bullpen_tail(
    *,
    outs: np.ndarray,
    inning: np.ndarray,
    inning_outs: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    runs_this_inning: np.ndarray,
    bullpen_k_rate: float,
    bullpen_bb_rate: float,
    bullpen_hr_rate: float,
    game_context: GameContext | None,
    babip_adj: float,
    rng: np.random.Generator,
    n_sims: int,
    bullpen_profile: TeamBullpenProfile | None = None,
    starter_runs: np.ndarray | None = None,
    opposing_runs: np.ndarray | None = None,
) -> np.ndarray:
    """Continue a game simulation after the starter exits using team bullpen rates.

    Runs PA-by-PA with team-aggregate bullpen rates until every sim reaches
    27 outs (or ``_MAX_BULLPEN_PA`` is hit as a safety valve). Mutates the
    passed-in state vectors (``outs``, ``inning``, ``inning_outs``,
    ``r1``, ``r2``, ``r3``, ``runs_this_inning``) in place -- they represent
    the same sim lanes the starter loop was tracking.

    When a ``bullpen_profile`` is provided, rates vary by game state:
    high-leverage arms (CL/SU) in close games, low-leverage (MR) in blowouts.

    Parameters
    ----------
    outs, inning, inning_outs, r1, r2, r3, runs_this_inning : np.ndarray
        Per-sim game state at starter exit. Shape ``(n_sims,)``. Mutated.
    bullpen_k_rate, bullpen_bb_rate, bullpen_hr_rate : float
        Flat fallback rates (used when ``bullpen_profile`` is None).
    game_context : GameContext, optional
        Environmental context. Starter-specific fields zeroed for bullpen.
    babip_adj : float
        BABIP adjustment for BIP outcomes.
    rng : np.random.Generator
        Shared RNG from the parent simulation.
    n_sims : int
        Total number of simulation lanes.
    bullpen_profile : TeamBullpenProfile, optional
        Two-tier bullpen profile. When provided, rates are selected per-sim
        based on the current score differential.
    starter_runs : np.ndarray, optional
        Runs allowed by the starter (this pitching team's perspective).
        Required when ``bullpen_profile`` is set to compute score_diff.
    opposing_runs : np.ndarray, optional
        Runs scored by the opposing team's starter against our lineup.
        Required when ``bullpen_profile`` is set.

    Returns
    -------
    np.ndarray
        ``bullpen_runs`` of shape ``(n_sims,)``
    """
    pa_outcome_model = PAOutcomeModel()

    # Drop starter-specific context for the bullpen phase.
    if game_context is not None:
        bullpen_ctx = _dc_replace(game_context, form_bb_lift=0.0, xgb_bb_lift=0.0)
    else:
        bullpen_ctx = None

    # Determine whether we can do leverage-tier selection
    use_tiers = (
        bullpen_profile is not None
        and starter_runs is not None
        and opposing_runs is not None
    )

    bullpen_runs = np.zeros(n_sims, dtype=np.int32)

    # Active = sims that still need more outs (skip complete games)
    active = outs < 27

    for _ in range(_MAX_BULLPEN_PA):
        n_active = int(active.sum())
        if n_active == 0:
            break

        zeros_active = np.zeros(n_active, dtype=np.float64)

        if use_tiers:
            # Score diff from pitching team's perspective (positive = winning)
            # Our team's runs = opposing_runs (scored against opposing pitcher)
            # Opponent's runs = starter_runs + bullpen_runs (scored against us)
            our_runs = opposing_runs[active] if opposing_runs is not None else np.zeros(n_active)
            their_runs = (
                (starter_runs[active] if starter_runs is not None else np.zeros(n_active))
                + bullpen_runs[active]
            )
            score_diff = (our_runs - their_runs).astype(np.int32)
            k_rates, bb_rates, hr_rates = bullpen_profile.select_rates(score_diff)  # type: ignore[union-attr]
            k_rates = np.asarray(k_rates, dtype=np.float64)
            bb_rates = np.asarray(bb_rates, dtype=np.float64)
            hr_rates = np.asarray(hr_rates, dtype=np.float64)
        else:
            k_rates = np.full(n_active, bullpen_k_rate, dtype=np.float64)
            bb_rates = np.full(n_active, bullpen_bb_rate, dtype=np.float64)
            hr_rates = np.full(n_active, bullpen_hr_rate, dtype=np.float64)

        probs = pa_outcome_model.compute_pa_probs(
            pitcher_k_rate=k_rates,
            pitcher_bb_rate=bb_rates,
            pitcher_hr_rate=hr_rates,
            matchup_k_lift=zeros_active,
            matchup_bb_lift=zeros_active,
            matchup_hr_lift=zeros_active,
            tto_k_lift=zeros_active,
            tto_bb_lift=zeros_active,
            tto_hr_lift=zeros_active,
            fatigue_k_lift=zeros_active,
            fatigue_bb_lift=zeros_active,
            fatigue_hr_lift=zeros_active,
            ctx=bullpen_ctx,
        )

        outcomes = pa_outcome_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active, babip_adj=babip_adj,
        )

        # --- Runner advancement (pre-out-increment so sac flies can read
        #     inning_outs as 0 or 1 and score the runner on 3B).
        runs_scored, new_r1, new_r2, new_r3 = _advance_runners(
            outcomes=outcomes,
            r1=r1[active],
            r2=r2[active],
            r3=r3[active],
            inning_outs=inning_outs[active],
            rng=rng,
        )

        bullpen_runs[active] += runs_scored
        r1[active] = new_r1
        r2[active] = new_r2
        r3[active] = new_r3

        # --- Outs (after the sac fly check) ---
        is_out = np.isin(outcomes, [PA_STRIKEOUT, PA_OUT])
        outs[active] += is_out.astype(np.int32)
        inning_outs[active] += is_out.astype(np.int32)

        # Inning rollover
        inning_over = inning_outs[active] >= 3
        inning[active] = np.where(inning_over, inning[active] + 1, inning[active])
        inning_outs[active] = np.where(inning_over, 0, inning_outs[active])
        r1[active] = np.where(inning_over, 0, r1[active])
        r2[active] = np.where(inning_over, 0, r2[active])
        r3[active] = np.where(inning_over, 0, r3[active])
        runs_this_inning[active] = np.where(
            inning_over, 0, runs_this_inning[active],
        )
        runs_this_inning[active] += runs_scored

        # Recompute active: sims that have reached 27 outs drop out
        active = outs < 27

    return bullpen_runs


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
    game_context: GameContext | None = None,
    exit_calibration_offset: float = _DEFAULT_EXIT_CALIBRATION_OFFSET,
    manager_pull_tendency: float = 88.0,
    mu_bf: float | None = None,
    sigma_bf: float | None = None,
    lineup_matchup_reliabilities: dict[str, np.ndarray] | None = None,
    bullpen_k_rate: float = BULLPEN_K_RATE,
    bullpen_bb_rate: float = BULLPEN_BB_RATE,
    bullpen_hr_rate: float = BULLPEN_HR_RATE,
    bullpen_profile: TeamBullpenProfile | None = None,
    opposing_runs_estimate: np.ndarray | None = None,
    lineup_bip_probs: np.ndarray | None = None,
    n_sims: int = 50_000,
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
    game_context : GameContext, optional
        Per-game environmental lifts (umpire, park, weather, catcher
        framing, pitcher form). Defaults to zero lifts.
    exit_calibration_offset : float
        Logit-scale offset for exit model probabilities. Negative values
        reduce exit probability (pitcher stays in longer).
    manager_pull_tendency : float
        Team's avg starter exit pitch count (manager proxy).
    mu_bf : float, optional
        Pitcher's shrinkage-estimated mean BF per start (from bf_model).
        When provided alongside sigma_bf, enables BF-anchored exit logic:
        target_bf is drawn per-sim from Normal(mu_bf, sigma_bf) and the
        exit decision only fires at inning boundaries, anchored to this
        target. This makes total BF sample-source-invariant (not affected
        by K/BB/HR posterior distribution changes). When None, falls back
        to the legacy per-PA exit model path (DEPRECATED — sensitive to
        K sample distribution, see exit model documentation).
    sigma_bf : float, optional
        Within-pitcher std of BF per start. Used with mu_bf.
    lineup_matchup_reliabilities : dict[str, np.ndarray], optional
        Per-stat reliability per batter slot. Keys: 'k', 'bb', 'hr'.
        Each value shape (9,), range [0, 1]. When provided, adds per-sim
        noise to matchup lifts: high reliability → tight, low → wide.
    bullpen_k_rate, bullpen_bb_rate, bullpen_hr_rate : float
        Opposing team's aggregate bullpen rates, used after the starter
        exits to complete the game to 27 outs. Defaults to league-average
        constants if team-specific rates are unavailable.
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
    k_rates = resample_posterior(pitcher_k_rate_samples, n_sims, rng)
    bb_rates = resample_posterior(pitcher_bb_rate_samples, n_sims, rng)
    hr_rates = resample_posterior(pitcher_hr_rate_samples, n_sims, rng)

    # Default matchup lifts to zeros if missing stat keys
    for stat in ("k", "bb", "hr"):
        if stat not in lineup_matchup_lifts:
            lineup_matchup_lifts[stat] = np.zeros(9)
        if stat not in tto_lifts:
            tto_lifts[stat] = np.zeros(3)

    # Dampen matchup lifts — empirical calibration from 11,517 game
    # walk-forward backtest (2023-2025). Raw pitch-type matchup scoring
    # over-applies lifts by ~2x for K/BB. HR signal is near-zero.
    for stat, damp in MATCHUP_DAMPEN.items():
        lineup_matchup_lifts[stat] = lineup_matchup_lifts[stat] * damp

    # Reliability-based per-sim noise: when reliability is low the true
    # matchup effect is uncertain, so we draw per-sim perturbations.
    # At reliability=1.0 sigma→0 (point mass at computed lift);
    # at reliability=0.0 sigma→full spread (essentially random).
    # Drawn once per simulation path (not per PA) — this represents
    # epistemic uncertainty about the matchup, not within-game noise.
    # Shape: (n_sims, 9) per stat — indexed by batter slot in the PA loop.
    _MATCHUP_NOISE_SIGMA = {"k": 0.30, "bb": 0.25, "hr": 0.15}
    matchup_noise: dict[str, np.ndarray] = {}
    if lineup_matchup_reliabilities is not None:
        for stat in ("k", "bb", "hr"):
            rel = lineup_matchup_reliabilities.get(stat, np.ones(9))
            sigma_per_slot = _MATCHUP_NOISE_SIGMA[stat] * (1.0 - rel)  # (9,)
            # Draw (n_sims, 9) noise, then add to base lifts per sim
            matchup_noise[stat] = rng.normal(
                loc=0.0, scale=sigma_per_slot[np.newaxis, :], size=(n_sims, 9),
            )
    else:
        for stat in ("k", "bb", "hr"):
            matchup_noise[stat] = np.zeros((n_sims, 9))

    # --- Game state arrays (all shape n_sims) ---
    pitches = np.zeros(n_sims, dtype=np.int32)
    outs = np.zeros(n_sims, dtype=np.int32)
    inning = np.ones(n_sims, dtype=np.int32)
    inning_outs = np.zeros(n_sims, dtype=np.int32)  # 0, 1, 2 within inning
    lineup_pos = np.zeros(n_sims, dtype=np.int32)    # 0-8
    bf_count = np.zeros(n_sims, dtype=np.int32)
    # Base occupancy: 0/1 flags. The exit model still takes a total count,
    # which we compute on the fly as r1+r2+r3.
    r1 = np.zeros(n_sims, dtype=np.int32)
    r2 = np.zeros(n_sims, dtype=np.int32)
    r3 = np.zeros(n_sims, dtype=np.int32)
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

    # 3-PA trouble tracker (blow-up detection)
    prev_trouble_2 = np.zeros(n_sims, dtype=np.int32)

    # Runs scored in current inning (blow-up indicator)
    runs_this_inning = np.zeros(n_sims, dtype=np.int32)

    # Active mask — simulations where pitcher is still in the game
    active = np.ones(n_sims, dtype=bool)

    # BF-anchored exit: draw per-sim target BF from pitcher's BF prior.
    # When mu_bf/sigma_bf are provided, the exit decision fires only at
    # inning boundaries, anchored to this target. Mid-inning exits are
    # limited to hard caps and blow-up conditions. This eliminates the
    # per-PA K/BB/HR → game-state → exit-model feedback loop that made
    # total BF sensitive to the posterior sample source.
    use_bf_anchor = mu_bf is not None and sigma_bf is not None
    if use_bf_anchor:
        target_bf = rng.normal(mu_bf, max(sigma_bf, 0.5), size=n_sims)
        target_bf = np.clip(target_bf, 9, 35).astype(float)
    else:
        target_bf = np.full(n_sims, _POP_BF_MU, dtype=float)

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

        # Gather per-batter matchup lifts + per-sim reliability noise
        active_idx = np.where(active)[0]
        k_matchup = np.array([lineup_matchup_lifts["k"][s] for s in slot]) + \
            matchup_noise["k"][active_idx, slot]
        bb_matchup = np.array([lineup_matchup_lifts["bb"][s] for s in slot]) + \
            matchup_noise["bb"][active_idx, slot]
        hr_matchup = np.array([lineup_matchup_lifts["hr"][s] for s in slot]) + \
            matchup_noise["hr"][active_idx, slot]

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
            ctx=game_context,
        )

        # --- 3. Draw PA outcomes ---
        # If per-batter BIP probs are available, slice by slot for active sims
        batter_bip_active: np.ndarray | None = None
        if lineup_bip_probs is not None:
            batter_bip_active = lineup_bip_probs[slot]
        outcomes = pa_outcome_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active, babip_adj=babip_adj,
            batter_bip_probs=batter_bip_active,
        )

        # --- 4. Update game state ---
        # Update accumulators
        k_total[active] += (outcomes == PA_STRIKEOUT).astype(np.int32)
        bb_total[active] += (outcomes == PA_WALK).astype(np.int32)
        hbp_total[active] += (outcomes == PA_HBP).astype(np.int32)
        hr_total[active] += (outcomes == PA_HOME_RUN).astype(np.int32)

        is_hit = np.isin(outcomes, [PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN])
        h_total[active] += is_hit.astype(np.int32)

        # --- Runner advancement (pre-out-increment so sac flies see the
        #     current inning_outs as 0 or 1).
        runs_scored, new_r1, new_r2, new_r3 = _advance_runners(
            outcomes=outcomes,
            r1=r1[active],
            r2=r2[active],
            r3=r3[active],
            inning_outs=inning_outs[active],
            rng=rng,
        )

        runs[active] += runs_scored
        score_diff[active] -= runs_scored
        r1[active] = new_r1
        r2[active] = new_r2
        r3[active] = new_r3

        # Update outs (after sac fly check)
        is_out = np.isin(outcomes, [PA_STRIKEOUT, PA_OUT])
        outs[active] += is_out.astype(np.int32)
        inning_outs[active] += is_out.astype(np.int32)

        # Check for inning change (3 outs in inning)
        inning_over = inning_outs[active] >= 3
        inning[active] = np.where(inning_over, inning[active] + 1, inning[active])
        inning_outs[active] = np.where(inning_over, 0, inning_outs[active])
        r1[active] = np.where(inning_over, 0, r1[active])
        r2[active] = np.where(inning_over, 0, r2[active])
        r3[active] = np.where(inning_over, 0, r3[active])
        runs_this_inning[active] = np.where(inning_over, 0, runs_this_inning[active])

        # Update runs_this_inning with runs scored this PA
        runs_this_inning[active] += runs_scored

        # Update recent trouble (2-PA and 3-PA windows)
        is_trouble = np.isin(
            outcomes,
            [PA_WALK, PA_HBP, PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN],
        ).astype(np.int32)
        new_recent = prev_trouble[active] + is_trouble
        blowup_3pa = prev_trouble_2[active] + prev_trouble[active] + is_trouble
        prev_trouble_2[active] = prev_trouble[active]
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

        if use_bf_anchor:
            # ---- BF-ANCHORED EXIT LOGIC (preferred) ----
            # Mid-inning: only force exits (hard caps + blow-up).
            # Between-inning: exit model + BF anchor term.
            # This eliminates the per-PA K/BB/HR → game-state feedback
            # loop that inflated BF when K samples shifted.

            # Mid-inning blow-up: yank pitcher if >= 4 runs this inning
            mid_inning_blowup = ~inning_over & (
                runs_this_inning[active] >= _BLOWUP_RUNS_THRESHOLD
            )
            force_exit |= mid_inning_blowup

            # Between-inning exit: only at inning boundaries
            exit_prob = np.zeros(n_active)
            # inning_over was set above (line ~927): True for sims where
            # this PA made the 3rd out. At this point inning_outs is
            # already 0 and inning is incremented.
            at_boundary = inning_over & (bf_count[active] >= 3)
            at_boundary &= ~force_exit

            if at_boundary.any():
                # Pure BF-target sigmoid: exit probability is a
                # function of how far past the target BF we are.
                # At target: p = 0.5. Before: lower. After: higher.
                # This pins mean BF to the per-pitcher prior by
                # construction, independent of K/BB/HR sample source.
                bf_dev = (
                    bf_count[active].astype(float) - target_bf[active]
                )
                exit_prob[at_boundary] = expit(
                    (_BF_ANCHOR_K * bf_dev)[at_boundary]
                )
        else:
            # ---- DEPRECATED: Per-PA exit model (legacy) ----
            # WARNING: This path evaluates exit probability after every
            # PA, using game-state features (runners, recent_trouble)
            # that are downstream of the K/BB/HR sample distribution.
            # This creates a feedback loop: different K samples →
            # different game states → different exit probabilities →
            # different total BF. The resulting BF is NOT invariant to
            # the posterior sample source. When K samples over-predict
            # K rate, this path inflates BF by ~15-20%.
            #
            # Use BF-anchored exit (pass mu_bf/sigma_bf) instead.
            # This path is kept only for backward compatibility with
            # callers that do not yet supply BF priors.
            current_tto = np.minimum(
                bf_count[active] // BF_PER_TTO + 1, 3
            )
            exit_prob = exit_model.predict_exit_prob(
                cumulative_pitches=pitches[active],
                inning=inning[active],
                inning_outs=inning_outs[active],
                score_diff=score_diff[active],
                runners=r1[active] + r2[active] + r3[active],
                tto=current_tto,
                recent_trouble=recent_trouble[active],
                pitcher_avg_pitches=pitcher_avg_pitches,
                runs_this_inning=runs_this_inning[active],
                blowup_recent_3pa=blowup_3pa,
                manager_pull_tendency=manager_pull_tendency,
            )
            if exit_calibration_offset != 0.0:
                exit_prob = np.clip(exit_prob, CLIP_LO, CLIP_HI)
                exit_logit = logit(exit_prob) + exit_calibration_offset
                exit_prob = expit(exit_logit)

        # Draw exit decisions
        exit_draw = rng.random(n_active) < exit_prob
        exits = force_exit | exit_draw

        # Update active mask
        active_indices = np.where(active)[0]
        active[active_indices[exits]] = False

    # --- Bullpen tail: finish the game to 27 outs using team bullpen rates ---
    # `runs` and `outs` up to this point belong to the starter. Snapshot the
    # starter counting stats before the tail mutates shared state arrays.
    starter_runs = runs.copy()
    starter_outs = outs.copy()

    bullpen_runs = simulate_bullpen_tail(
        outs=outs,
        inning=inning,
        inning_outs=inning_outs,
        r1=r1,
        r2=r2,
        r3=r3,
        runs_this_inning=runs_this_inning,
        bullpen_k_rate=bullpen_k_rate,
        bullpen_bb_rate=bullpen_bb_rate,
        bullpen_hr_rate=bullpen_hr_rate,
        game_context=game_context,
        babip_adj=babip_adj,
        rng=rng,
        n_sims=n_sims,
        bullpen_profile=bullpen_profile,
        starter_runs=starter_runs,
        opposing_runs=opposing_runs_estimate,
    )

    total_runs = starter_runs + bullpen_runs

    return SimulationResult(
        k_samples=k_total,
        bb_samples=bb_total,
        h_samples=h_total,
        hr_samples=hr_total,
        hbp_samples=hbp_total,
        bf_samples=bf_count,
        pitch_count_samples=pitches,
        outs_samples=starter_outs,
        runs_samples=total_runs,
        starter_runs_samples=starter_runs,
        bullpen_runs_samples=bullpen_runs,
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
    game_context: GameContext | None = None,
    manager_pull_tendency: float = 88.0,
    bullpen_k_rate: float = BULLPEN_K_RATE,
    bullpen_bb_rate: float = BULLPEN_BB_RATE,
    bullpen_hr_rate: float = BULLPEN_HR_RATE,
    mu_bf: float | None = None,
    sigma_bf: float | None = None,
    lineup_bip_probs: np.ndarray | None = None,
    n_sims: int = 50_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """High-level game prediction interface.

    Computes pitch count features, runs simulation, and returns
    comprehensive results including prop line probabilities.
    """
    from src.models.game_sim.pitch_count_model import build_pitch_count_features

    pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
        pitcher_features=pitcher_features,
        batter_features=batter_features,
        pitcher_id=pitcher_id,
        batter_ids=lineup_batter_ids,
        season=season,
    )

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
        game_context=game_context,
        mu_bf=mu_bf,
        sigma_bf=sigma_bf,
        manager_pull_tendency=manager_pull_tendency,
        bullpen_k_rate=bullpen_k_rate,
        bullpen_bb_rate=bullpen_bb_rate,
        bullpen_hr_rate=bullpen_hr_rate,
        lineup_bip_probs=lineup_bip_probs,
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
        "outs_over_probs": result.over_probs("outs"),
    }
