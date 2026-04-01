"""
Lineup game simulator with full base-state tracking.

Simulates an entire 9-man batting lineup through ~9 innings of a
half-game (one team batting), tracking runners on all three bases.
Produces per-batter joint distributions over PA, H, 1B, 2B, 3B, HR,
BB, K, R, RBI, and TB.

Unlike the per-batter simulator (batter_simulator.py) which uses flat
Bernoulli draws for R and RBI, this engine tracks the 3-element base
state across all simulations and resolves runner advancement with
calibrated probabilities on every PA outcome.

Runner identity (which batter slot occupies each base) is tracked so
that R credit goes to the correct batter when they score.

The simulation is fully vectorized across n_sims using numpy. The main
loop iterates over PAs (not sims), and all n_sims advance in lockstep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

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
)
from src.models.bf_model import draw_bf_samples
from src.utils.constants import (
    CLIP_LO,
    CLIP_HI,
    SIM_LEAGUE_K_RATE,
    SIM_LEAGUE_BB_RATE,
    SIM_LEAGUE_HR_RATE,
    BULLPEN_K_RATE,
    BULLPEN_BB_RATE,
    BULLPEN_HR_RATE,
)

logger = logging.getLogger(__name__)

# Maximum PAs in a game across the full lineup (safety valve)
MAX_LINEUP_PA = 55

# Number of lineup slots
LINEUP_SIZE = 9

# Sentinel value for empty base (no runner)
_EMPTY: np.int8 = np.int8(-1)

# ---------------------------------------------------------------------------
# Runner advancement probabilities
# Calibrated from 2021-2025 Statcast EDA (914K PAs with base state).
# ---------------------------------------------------------------------------

# -- Single: runner on 3B --
P_SCORE_FROM_3B_ON_1B = 0.97

# -- Single: runner on 2B (outs-dependent) --
P_SCORE_FROM_2B_ON_1B_0OUT = 0.39
P_SCORE_FROM_2B_ON_1B_1OUT = 0.54
P_SCORE_FROM_2B_ON_1B_2OUT = 0.79

# -- Single: runner on 1B --
P_1B_TO_3B_ON_1B = 0.27        # goes to 3B (vs stays at 2B)
P_SCORE_FROM_1B_ON_1B_0OUT = 0.026
P_SCORE_FROM_1B_ON_1B_1OUT = 0.036
P_SCORE_FROM_1B_ON_1B_2OUT = 0.070

# -- Double --
P_SCORE_FROM_1B_ON_2B = 0.42   # EDA: 0.417 (was 0.65)
P_1B_TO_3B_ON_2B = 0.55        # if doesn't score, goes to 3B

# -- Out (non-K): runner on 3B scoring (sac fly / productive out) --
P_SCORE_ON_OUT_3B_0OUT = 0.233
P_SCORE_ON_OUT_3B_1OUT = 0.284
P_SCORE_ON_OUT_3B_2OUT = 0.014  # almost never with 2 outs

# -- Out: runner advancement on ground outs --
P_ADVANCE_2B_TO_3B_ON_OUT = 0.41  # runner on 2B goes to 3B on out
P_ADVANCE_1B_TO_2B_ON_OUT = 0.14  # runner on 1B goes to 2B on out


def _safe_logit(p: float | np.ndarray) -> float | np.ndarray:
    """Logit with clipping to avoid infinities."""
    return logit(np.clip(p, CLIP_LO, CLIP_HI))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LineupSimulationResult:
    """Results from a full-lineup game simulation.

    Per-batter arrays have shape (9, n_sims) where index 0 = batting
    order slot 1 (leadoff).
    """

    # Per-batter counting stat arrays, shape (9, n_sims)
    pa: np.ndarray
    k: np.ndarray
    bb: np.ndarray
    hbp: np.ndarray
    h: np.ndarray
    singles: np.ndarray
    doubles: np.ndarray
    triples: np.ndarray
    hr: np.ndarray
    r: np.ndarray
    rbi: np.ndarray
    tb: np.ndarray

    # Game-level totals, shape (n_sims,)
    team_runs: np.ndarray

    n_sims: int = 0

    def batter_result(self, slot: int) -> dict[str, np.ndarray]:
        """Extract per-batter arrays for a single lineup slot (0-indexed).

        Returns a dict with keys matching BatterSimulationResult field
        names so it can be used as a drop-in replacement.
        """
        result = {
            "pa_samples": self.pa[slot],
            "k_samples": self.k[slot],
            "bb_samples": self.bb[slot],
            "hbp_samples": self.hbp[slot],
            "h_samples": self.h[slot],
            "single_samples": self.singles[slot],
            "double_samples": self.doubles[slot],
            "triple_samples": self.triples[slot],
            "hr_samples": self.hr[slot],
            "r_samples": self.r[slot],
            "rbi_samples": self.rbi[slot],
            "tb_samples": self.tb[slot],
        }
        result["hrr_samples"] = (
            result["h_samples"] + result["r_samples"] + result["rbi_samples"]
        )
        return result

    def batter_summary(self, slot: int) -> dict[str, dict[str, float]]:
        """Summary statistics for one batter (0-indexed slot)."""
        result = self.batter_result(slot)
        stats = {}
        for name in [
            "k", "bb", "h", "hr", "single", "double", "triple",
            "tb", "r", "rbi", "hbp", "pa",
        ]:
            samples = result[f"{name}_samples"]
            stats[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "q10": float(np.percentile(samples, 10)),
                "q90": float(np.percentile(samples, 90)),
            }
        # HRR = H + R + RBI
        hrr = result["h_samples"] + result["r_samples"] + result["rbi_samples"]
        stats["hrr"] = {
            "mean": float(np.mean(hrr)),
            "std": float(np.std(hrr)),
            "median": float(np.median(hrr)),
            "q10": float(np.percentile(hrr, 10)),
            "q90": float(np.percentile(hrr, 90)),
        }
        return stats

    def batter_over_probs(
        self,
        slot: int,
        stat: str,
        lines: list[float] | None = None,
    ) -> pd.DataFrame:
        """Compute P(over X.5) for prop lines for one batter."""
        result = self.batter_result(slot)
        samples = result[f"{stat}_samples"]
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

    def team_summary(self) -> dict[str, dict[str, float]]:
        """Summary statistics for team-level totals."""
        stats = {}
        for name in [
            "pa", "k", "bb", "h", "hr", "r", "rbi", "tb",
        ]:
            arr = getattr(self, name)
            team_total = arr.sum(axis=0)  # sum across 9 batters
            stats[name] = {
                "mean": float(np.mean(team_total)),
                "std": float(np.std(team_total)),
                "median": float(np.median(team_total)),
                "q10": float(np.percentile(team_total, 10)),
                "q90": float(np.percentile(team_total, 90)),
            }
        stats["team_runs"] = {
            "mean": float(np.mean(self.team_runs)),
            "std": float(np.std(self.team_runs)),
            "median": float(np.median(self.team_runs)),
            "q10": float(np.percentile(self.team_runs, 10)),
            "q90": float(np.percentile(self.team_runs, 90)),
        }
        return stats


# ---------------------------------------------------------------------------
# Vectorized runner scoring helper
# ---------------------------------------------------------------------------

def _score_runners(
    r_total: np.ndarray,
    base_runner_ids: np.ndarray,
    mask: np.ndarray,
    global_idx: np.ndarray,
) -> None:
    """Credit a run to each runner identified by base_runner_ids where mask is True.

    Parameters
    ----------
    r_total : np.ndarray
        Shape (9, n_sims_total). Modified in place.
    base_runner_ids : np.ndarray
        Shape (n_active,) int8. Batter slot IDs on a particular base.
    mask : np.ndarray
        Shape (n_active,) bool. Which sims have a runner scoring.
    global_idx : np.ndarray
        Shape (n_active,) int. Maps active-local index to global sim index.
    """
    scored = mask & (base_runner_ids != _EMPTY)
    if not scored.any():
        return
    s = np.where(scored)[0]
    slots = base_runner_ids[s]
    sims = global_idx[s]
    np.add.at(r_total, (slots, sims), 1)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_lineup_game(
    batter_k_rate_samples: list[np.ndarray],
    batter_bb_rate_samples: list[np.ndarray],
    batter_hr_rate_samples: list[np.ndarray],
    starter_k_rate: float,
    starter_bb_rate: float,
    starter_hr_rate: float,
    starter_bf_mu: float,
    starter_bf_sigma: float,
    matchup_k_lifts: np.ndarray | None = None,
    matchup_bb_lifts: np.ndarray | None = None,
    matchup_hr_lifts: np.ndarray | None = None,
    bullpen_k_rate: float = BULLPEN_K_RATE,
    bullpen_bb_rate: float = BULLPEN_BB_RATE,
    bullpen_hr_rate: float = BULLPEN_HR_RATE,
    bullpen_matchup_k_lifts: np.ndarray | None = None,
    bullpen_matchup_bb_lifts: np.ndarray | None = None,
    bullpen_matchup_hr_lifts: np.ndarray | None = None,
    batter_babip_adjs: np.ndarray | None = None,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_k_lift: float = 0.0,
    park_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    park_h_babip_adj: float = 0.0,
    weather_k_lift: float = 0.0,
    n_sims: int = 50_000,
    random_seed: int = 42,
) -> LineupSimulationResult:
    """Simulate a full 9-man lineup through a half-game with base-state tracking.

    Loops through the batting order one PA at a time, resolving each PA
    with PAOutcomeModel, then advancing runners on all three bases with
    calibrated probabilities. Tracks outs per inning (3 outs clears the
    bases). Accumulates per-batter counting stats including R and RBI
    derived from actual base-state transitions.

    The runner identity on each base is tracked so that when a runner
    scores, the R credit goes to the correct batter.

    Parameters
    ----------
    batter_k_rate_samples : list of np.ndarray
        Length-9 list of K% posterior sample arrays, one per lineup slot.
    batter_bb_rate_samples : list of np.ndarray
        Length-9 list of BB% posterior sample arrays.
    batter_hr_rate_samples : list of np.ndarray
        Length-9 list of HR/PA posterior sample arrays.
    starter_k_rate, starter_bb_rate, starter_hr_rate : float
        Opposing starter's posterior mean rates.
    starter_bf_mu, starter_bf_sigma : float
        Starter's batters-faced distribution parameters.
    matchup_k_lifts, matchup_bb_lifts, matchup_hr_lifts : np.ndarray, optional
        Shape (9,) logit lifts per batter vs the starter.
    bullpen_k_rate, bullpen_bb_rate, bullpen_hr_rate : float
        Opposing bullpen aggregate rates.
    bullpen_matchup_k_lifts, bullpen_matchup_bb_lifts, bullpen_matchup_hr_lifts :
        np.ndarray, optional. Shape (9,) logit lifts per batter vs bullpen.
    batter_babip_adjs : np.ndarray, optional
        Shape (9,) BABIP adjustments per batter.
    umpire_k_lift, umpire_bb_lift : float
        Umpire tendency adjustments on logit scale.
    park_k_lift, park_bb_lift, park_hr_lift : float
        Park factor adjustments on logit scale (from park_factor_lifts.parquet).
    park_h_babip_adj : float
        Park factor BABIP adjustment for hits (stacks with batter_babip_adjs).
    weather_k_lift : float
        Weather-based K adjustment on logit scale.
    n_sims : int
        Number of Monte Carlo simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    LineupSimulationResult
        Per-batter and team-level counting stat distributions.
    """
    rng = np.random.default_rng(random_seed)
    pa_model = PAOutcomeModel()

    # --- Default optional arrays ---
    if matchup_k_lifts is None:
        matchup_k_lifts = np.zeros(LINEUP_SIZE)
    if matchup_bb_lifts is None:
        matchup_bb_lifts = np.zeros(LINEUP_SIZE)
    if matchup_hr_lifts is None:
        matchup_hr_lifts = np.zeros(LINEUP_SIZE)
    if bullpen_matchup_k_lifts is None:
        bullpen_matchup_k_lifts = np.zeros(LINEUP_SIZE)
    if bullpen_matchup_bb_lifts is None:
        bullpen_matchup_bb_lifts = np.zeros(LINEUP_SIZE)
    if bullpen_matchup_hr_lifts is None:
        bullpen_matchup_hr_lifts = np.zeros(LINEUP_SIZE)
    if batter_babip_adjs is None:
        batter_babip_adjs = np.zeros(LINEUP_SIZE)

    # --- Resample batter posteriors to n_sims ---
    def _resample(arr: np.ndarray) -> np.ndarray:
        if len(arr) == n_sims:
            return arr.copy()
        idx = rng.choice(len(arr), size=n_sims, replace=True)
        return arr[idx]

    bat_k = np.stack([_resample(s) for s in batter_k_rate_samples])   # (9, n_sims)
    bat_bb = np.stack([_resample(s) for s in batter_bb_rate_samples])
    bat_hr = np.stack([_resample(s) for s in batter_hr_rate_samples])

    # --- Pitcher quality lifts (logit scale) ---
    starter_k_lift = _safe_logit(starter_k_rate) - _safe_logit(SIM_LEAGUE_K_RATE)
    starter_bb_lift = _safe_logit(starter_bb_rate) - _safe_logit(SIM_LEAGUE_BB_RATE)
    starter_hr_lift = _safe_logit(starter_hr_rate) - _safe_logit(SIM_LEAGUE_HR_RATE)

    bp_k_lift = _safe_logit(bullpen_k_rate) - _safe_logit(SIM_LEAGUE_K_RATE)
    bp_bb_lift = _safe_logit(bullpen_bb_rate) - _safe_logit(SIM_LEAGUE_BB_RATE)
    bp_hr_lift = _safe_logit(bullpen_hr_rate) - _safe_logit(SIM_LEAGUE_HR_RATE)

    # --- Draw starter BF for each sim ---
    starter_bf = draw_bf_samples(
        mu_bf=starter_bf_mu,
        sigma_bf=starter_bf_sigma,
        n_draws=n_sims,
        bf_min=3,
        bf_max=35,
        rng=rng,
    )

    # --- Per-batter accumulators, shape (9, n_sims) ---
    pa_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    k_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    bb_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    hbp_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    h_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    single_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    double_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    triple_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    hr_total_acc = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    r_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)
    rbi_total = np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32)

    # --- Game state arrays ---
    # Base occupancy: shape (n_sims, 3) bool. Columns: [1B, 2B, 3B].
    bases_occ = np.zeros((n_sims, 3), dtype=bool)
    # Runner identity: shape (n_sims, 3) int8. Batter slot (0-8) or _EMPTY.
    runner_id = np.full((n_sims, 3), _EMPTY, dtype=np.int8)

    outs_in_inning = np.zeros(n_sims, dtype=np.int32)
    inning = np.ones(n_sims, dtype=np.int32)
    global_bf = np.zeros(n_sims, dtype=np.int32)
    team_runs = np.zeros(n_sims, dtype=np.int32)

    # Active mask: sims still in progress (haven't completed 9 innings)
    active = np.ones(n_sims, dtype=bool)

    lineup_pos = 0

    for _pa_num in range(MAX_LINEUP_PA):
        n_active = active.sum()
        if n_active == 0:
            break

        slot = lineup_pos % LINEUP_SIZE
        aidx = np.where(active)[0]  # global indices of active sims

        # --- Track PA ---
        pa_total[slot, aidx] += 1

        # --- Starter vs bullpen ---
        global_bf[aidx] += 1
        vs_starter = global_bf[aidx] <= starter_bf[aidx]

        # --- Compute adjusted rates ---
        k_logit = _safe_logit(bat_k[slot, aidx])
        bb_logit = _safe_logit(bat_bb[slot, aidx])
        hr_logit = _safe_logit(bat_hr[slot, aidx])

        k_pitch = np.where(
            vs_starter,
            starter_k_lift + matchup_k_lifts[slot],
            bp_k_lift + bullpen_matchup_k_lifts[slot],
        )
        bb_pitch = np.where(
            vs_starter,
            starter_bb_lift + matchup_bb_lifts[slot],
            bp_bb_lift + bullpen_matchup_bb_lifts[slot],
        )
        hr_pitch = np.where(
            vs_starter,
            starter_hr_lift + matchup_hr_lifts[slot],
            bp_hr_lift + bullpen_matchup_hr_lifts[slot],
        )

        k_rate = expit(
            k_logit + k_pitch + umpire_k_lift + park_k_lift + weather_k_lift
        )
        bb_rate = expit(bb_logit + bb_pitch + umpire_bb_lift + park_bb_lift)
        hr_rate = expit(hr_logit + hr_pitch + park_hr_lift)

        # --- Draw PA outcomes ---
        probs = pa_model.compute_pa_probs(
            pitcher_k_rate=k_rate,
            pitcher_bb_rate=bb_rate,
            pitcher_hr_rate=hr_rate,
        )
        outcomes = pa_model.draw_outcomes(
            probs=probs,
            rng=rng,
            n_draws=n_active,
            babip_adj=float(batter_babip_adjs[slot]) + park_h_babip_adj,
        )

        # --- Classify ---
        is_single = outcomes == PA_SINGLE
        is_double = outcomes == PA_DOUBLE
        is_triple = outcomes == PA_TRIPLE
        is_hr = outcomes == PA_HOME_RUN
        is_walk = outcomes == PA_WALK
        is_hbp = outcomes == PA_HBP
        is_k = outcomes == PA_STRIKEOUT
        is_bip_out = outcomes == PA_OUT
        is_hit = is_single | is_double | is_triple | is_hr
        is_out = is_k | is_bip_out

        # --- Accumulate counting stats (R/RBI handled below) ---
        k_total[slot, aidx] += is_k.astype(np.int32)
        bb_total[slot, aidx] += is_walk.astype(np.int32)
        hbp_total[slot, aidx] += is_hbp.astype(np.int32)
        hr_total_acc[slot, aidx] += is_hr.astype(np.int32)
        single_total[slot, aidx] += is_single.astype(np.int32)
        double_total[slot, aidx] += is_double.astype(np.int32)
        triple_total[slot, aidx] += is_triple.astype(np.int32)
        h_total[slot, aidx] += is_hit.astype(np.int32)

        # =================================================================
        # RUNNER ADVANCEMENT
        # All logic operates directly on bases_occ[aidx] and runner_id[aidx].
        # Since aidx is a fancy index, we extract copies, modify, and
        # write back at the end of the PA.
        # =================================================================
        occ = bases_occ[aidx].copy()       # (n_active, 3) bool
        rid = runner_id[aidx].copy()       # (n_active, 3) int8
        cur_outs = outs_in_inning[aidx]    # (n_active,)

        # Pre-draw all random numbers for this PA at once for efficiency
        rand_block = rng.random((n_active, 6))

        # Accumulators for this PA
        runs_this_pa = np.zeros(n_active, dtype=np.int32)
        rbi_this_pa = np.zeros(n_active, dtype=np.int32)

        # Outs-dependent probability selectors
        p_2b_score_on_1b = np.where(
            cur_outs == 0, P_SCORE_FROM_2B_ON_1B_0OUT,
            np.where(cur_outs == 1, P_SCORE_FROM_2B_ON_1B_1OUT,
                     P_SCORE_FROM_2B_ON_1B_2OUT),
        )
        p_1b_score_on_1b = np.where(
            cur_outs == 0, P_SCORE_FROM_1B_ON_1B_0OUT,
            np.where(cur_outs == 1, P_SCORE_FROM_1B_ON_1B_1OUT,
                     P_SCORE_FROM_1B_ON_1B_2OUT),
        )

        # -----------------------------------------------------------
        # SINGLE
        # -----------------------------------------------------------
        if is_single.any():
            m = is_single  # (n_active,) bool

            # 3B runner
            on3 = m & occ[:, 2]
            scores3 = on3 & (rand_block[:, 0] < P_SCORE_FROM_3B_ON_1B)
            stays3 = on3 & ~scores3
            _score_runners(r_total, rid[:, 2], scores3, aidx)
            runs_this_pa += scores3.astype(np.int32)
            rbi_this_pa += scores3.astype(np.int32)

            # 2B runner (outs-dependent)
            on2 = m & occ[:, 1]
            scores2 = on2 & (rand_block[:, 1] < p_2b_score_on_1b)
            to3_from2 = on2 & ~scores2
            _score_runners(r_total, rid[:, 1], scores2, aidx)
            runs_this_pa += scores2.astype(np.int32)
            rbi_this_pa += scores2.astype(np.int32)

            # 1B runner (outs-dependent scoring + advancement)
            on1 = m & occ[:, 0]
            scores1 = on1 & (rand_block[:, 2] < p_1b_score_on_1b)
            _score_runners(r_total, rid[:, 0], scores1, aidx)
            runs_this_pa += scores1.astype(np.int32)
            rbi_this_pa += scores1.astype(np.int32)
            still_on = on1 & ~scores1
            to3_from1 = still_on & (rand_block[:, 3] < P_1B_TO_3B_ON_1B)
            to2_from1 = still_on & ~to3_from1

            # Save runner IDs before overwriting
            rid0_save = rid[:, 0].copy()
            rid1_save = rid[:, 1].copy()

            # New base state (only modify sims with a single)
            # 1B: batter
            occ[:, 0] = np.where(m, True, occ[:, 0])
            rid[:, 0] = np.where(m, np.int8(slot), rid[:, 0])

            # 2B: runner from 1B who stayed at 2B (and clear if no runner)
            occ[:, 1] = np.where(m, to2_from1, occ[:, 1])
            rid[:, 1] = np.where(to2_from1, rid0_save, np.where(m, _EMPTY, rid[:, 1]))

            # 3B: stays from 3B OR from 2B (didn't score) OR from 1B (fast)
            new3 = stays3 | to3_from2 | to3_from1
            occ[:, 2] = np.where(m, new3, occ[:, 2])
            # Determine the runner ID for 3B (priority: from1 > from2 > stays3)
            new_rid3 = np.where(to3_from1, rid0_save,
                       np.where(to3_from2, rid1_save,
                       np.where(stays3, rid[:, 2], _EMPTY)))
            rid[:, 2] = np.where(m, new_rid3, rid[:, 2])

        # -----------------------------------------------------------
        # DOUBLE
        # -----------------------------------------------------------
        if is_double.any():
            m = is_double

            # 3B runner scores
            on3 = m & occ[:, 2]
            _score_runners(r_total, rid[:, 2], on3, aidx)
            runs_this_pa += on3.astype(np.int32)
            rbi_this_pa += on3.astype(np.int32)

            # 2B runner scores
            on2 = m & occ[:, 1]
            _score_runners(r_total, rid[:, 1], on2, aidx)
            runs_this_pa += on2.astype(np.int32)
            rbi_this_pa += on2.astype(np.int32)

            # 1B runner: score or to 3B
            on1 = m & occ[:, 0]
            scores1 = on1 & (rand_block[:, 4] < P_SCORE_FROM_1B_ON_2B)
            to3_from1 = on1 & ~scores1
            _score_runners(r_total, rid[:, 0], scores1, aidx)
            runs_this_pa += scores1.astype(np.int32)
            rbi_this_pa += scores1.astype(np.int32)

            rid0_save = rid[:, 0].copy()

            # New state: batter on 2B, possibly 1B runner on 3B
            occ[:, 0] = np.where(m, False, occ[:, 0])
            rid[:, 0] = np.where(m, _EMPTY, rid[:, 0])

            occ[:, 1] = np.where(m, True, occ[:, 1])
            rid[:, 1] = np.where(m, np.int8(slot), rid[:, 1])

            occ[:, 2] = np.where(m, to3_from1, occ[:, 2])
            rid[:, 2] = np.where(m & to3_from1, rid0_save, np.where(m, _EMPTY, rid[:, 2]))

        # -----------------------------------------------------------
        # TRIPLE
        # -----------------------------------------------------------
        if is_triple.any():
            m = is_triple

            # All runners score
            for bc in range(3):
                on = m & occ[:, bc]
                _score_runners(r_total, rid[:, bc], on, aidx)
                runs_this_pa += on.astype(np.int32)
                rbi_this_pa += on.astype(np.int32)

            # Batter to 3B, rest empty
            occ[:, 0] = np.where(m, False, occ[:, 0])
            occ[:, 1] = np.where(m, False, occ[:, 1])
            occ[:, 2] = np.where(m, True, occ[:, 2])
            rid[:, 0] = np.where(m, _EMPTY, rid[:, 0])
            rid[:, 1] = np.where(m, _EMPTY, rid[:, 1])
            rid[:, 2] = np.where(m, np.int8(slot), rid[:, 2])

        # -----------------------------------------------------------
        # HOME RUN
        # -----------------------------------------------------------
        if is_hr.any():
            m = is_hr

            # All runners score
            for bc in range(3):
                on = m & occ[:, bc]
                _score_runners(r_total, rid[:, bc], on, aidx)
                runs_this_pa += on.astype(np.int32)
                rbi_this_pa += on.astype(np.int32)

            # Batter scores
            batter_ids = np.full(n_active, slot, dtype=np.int8)
            _score_runners(r_total, batter_ids, m, aidx)
            runs_this_pa += m.astype(np.int32)
            rbi_this_pa += m.astype(np.int32)  # batter RBI for themselves

            # Bases empty
            for bc in range(3):
                occ[:, bc] = np.where(m, False, occ[:, bc])
                rid[:, bc] = np.where(m, _EMPTY, rid[:, bc])

        # -----------------------------------------------------------
        # WALK / HBP  (force advancement only)
        # -----------------------------------------------------------
        wb = is_walk | is_hbp
        if wb.any():
            m = wb
            has1 = m & occ[:, 0]
            has2 = m & occ[:, 1]
            has3 = m & occ[:, 2]

            # Bases loaded: 3B scores
            loaded = has1 & has2 & has3
            _score_runners(r_total, rid[:, 2], loaded, aidx)
            runs_this_pa += loaded.astype(np.int32)
            rbi_this_pa += loaded.astype(np.int32)

            rid0_save = rid[:, 0].copy()
            rid1_save = rid[:, 1].copy()

            # Force chain (resolve from 3B down to 1B):
            # 3B: 2B runner pushed here if 1B+2B were both occupied,
            #      else stays if already there (and not loaded/scored)
            force_2_to_3 = has1 & has2
            new_occ3 = np.where(m, force_2_to_3 | (has3 & ~loaded), occ[:, 2])
            new_rid3 = np.where(
                m & force_2_to_3, rid1_save,
                np.where(m & has3 & ~loaded, rid[:, 2],
                np.where(m, _EMPTY, rid[:, 2])),
            )

            # 2B: 1B runner pushed here if 1B was occupied,
            #      else stays if already there and not pushed
            new_occ2 = np.where(m, has1 | (has2 & ~has1), occ[:, 1])
            new_rid2 = np.where(
                m & has1, rid0_save,
                np.where(m & has2 & ~has1, rid[:, 1],
                np.where(m, _EMPTY, rid[:, 1])),
            )

            # 1B: batter
            new_occ1 = np.where(m, True, occ[:, 0])
            new_rid1 = np.where(m, np.int8(slot), rid[:, 0])

            occ[:, 0] = new_occ1
            occ[:, 1] = new_occ2
            occ[:, 2] = new_occ3
            rid[:, 0] = new_rid1
            rid[:, 1] = new_rid2
            rid[:, 2] = new_rid3

        # -----------------------------------------------------------
        # OUTS (K and BIP out)
        # -----------------------------------------------------------
        if is_out.any():
            m = is_out
            is_bip = m & ~is_k

            # Runner on 3B scoring (sac fly / productive out, outs-dependent)
            p_score_3b_on_out = np.where(
                cur_outs == 0, P_SCORE_ON_OUT_3B_0OUT,
                np.where(cur_outs == 1, P_SCORE_ON_OUT_3B_1OUT,
                         P_SCORE_ON_OUT_3B_2OUT),
            )
            on3_out = is_bip & occ[:, 2]
            scores3 = on3_out & (rand_block[:, 5] < p_score_3b_on_out)

            _score_runners(r_total, rid[:, 2], scores3, aidx)
            runs_this_pa += scores3.astype(np.int32)
            rbi_this_pa += scores3.astype(np.int32)

            # Clear 3B if scored
            occ[:, 2] = np.where(scores3, False, occ[:, 2])
            rid[:, 2] = np.where(scores3, _EMPTY, rid[:, 2])

            # Runner advancement on BIP outs (not K)
            # 2B -> 3B
            adv_2to3 = is_bip & occ[:, 1] & ~occ[:, 2] & (rand_block[:, 4] < P_ADVANCE_2B_TO_3B_ON_OUT)
            rid_1_save = rid[:, 1].copy()
            occ[:, 2] = np.where(adv_2to3, True, occ[:, 2])
            rid[:, 2] = np.where(adv_2to3, rid_1_save, rid[:, 2])
            occ[:, 1] = np.where(adv_2to3, False, occ[:, 1])
            rid[:, 1] = np.where(adv_2to3, _EMPTY, rid[:, 1])

            # 1B -> 2B
            adv_1to2 = is_bip & occ[:, 0] & ~occ[:, 1] & (rand_block[:, 3] < P_ADVANCE_1B_TO_2B_ON_OUT)
            rid_0_save = rid[:, 0].copy()
            occ[:, 1] = np.where(adv_1to2, True, occ[:, 1])
            rid[:, 1] = np.where(adv_1to2, rid_0_save, rid[:, 1])
            occ[:, 0] = np.where(adv_1to2, False, occ[:, 0])
            rid[:, 0] = np.where(adv_1to2, _EMPTY, rid[:, 0])

        # --- Write back base state ---
        bases_occ[aidx] = occ
        runner_id[aidx] = rid

        # --- Credit RBI and team runs ---
        rbi_total[slot, aidx] += rbi_this_pa
        team_runs[aidx] += runs_this_pa

        # --- Update outs ---
        outs_in_inning[aidx] += is_out.astype(np.int32)

        # --- End of inning check ---
        inn_over = outs_in_inning[aidx] >= 3
        if inn_over.any():
            over_global = aidx[inn_over]
            inning[over_global] += 1
            outs_in_inning[over_global] = 0
            bases_occ[over_global] = False
            runner_id[over_global] = _EMPTY

            # Game over after 9 complete half-innings
            done = inning[over_global] > 9
            active[over_global[done]] = False

        lineup_pos += 1

    # --- Compute TB ---
    tb_total = (
        single_total
        + 2 * double_total
        + 3 * triple_total
        + 4 * hr_total_acc
    )

    return LineupSimulationResult(
        pa=pa_total,
        k=k_total,
        bb=bb_total,
        hbp=hbp_total,
        h=h_total,
        singles=single_total,
        doubles=double_total,
        triples=triple_total,
        hr=hr_total_acc,
        r=r_total,
        rbi=rbi_total,
        tb=tb_total,
        team_runs=team_runs,
        n_sims=n_sims,
    )
