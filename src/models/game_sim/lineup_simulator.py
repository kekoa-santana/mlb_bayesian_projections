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

Also provides ``simulate_full_game_both_teams()`` which alternates
half-innings for two lineups with walk-off logic in the 9th inning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

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
from src.models.bf_model import draw_bf_samples
from src.models.game_sim._sim_utils import (
    safe_logit,
    resample_posterior,
    MATCHUP_DAMPEN,
    compute_pitcher_quality_lifts,
    default_lift_array,
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
    return safe_logit(p)


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


@dataclass
class FullGameSimulationResult:
    """Results from a full two-team game simulation.

    Contains separate ``LineupSimulationResult`` objects for the away
    and home batting sides, plus game-level run totals and win outcomes.
    """

    away: LineupSimulationResult
    home: LineupSimulationResult

    # Per-sim game-level arrays, shape (n_sims,)
    away_runs: np.ndarray
    home_runs: np.ndarray
    home_win: np.ndarray  # bool

    n_sims: int = 0

    @property
    def home_win_prob(self) -> float:
        """Fraction of simulations won by the home team."""
        return float(np.mean(self.home_win))

    @property
    def away_win_prob(self) -> float:
        """Fraction of simulations won by the away team."""
        return 1.0 - self.home_win_prob

    def total_runs_summary(self) -> dict[str, float]:
        """Summary statistics for total runs (away + home).

        Returns
        -------
        dict[str, float]
            Keys: mean, std, median, q10, q90.
        """
        total = self.away_runs + self.home_runs
        return {
            "mean": float(np.mean(total)),
            "std": float(np.std(total)),
            "median": float(np.median(total)),
            "q10": float(np.percentile(total, 10)),
            "q90": float(np.percentile(total, 90)),
        }

    def run_line_probs(self, line: float = 0.5) -> dict[str, float]:
        """Probability of covering the run line (spread).

        Parameters
        ----------
        line : float
            Run line spread applied to the away team.  A negative value
            means the away team is favoured (e.g. ``-1.5``).

        Returns
        -------
        dict[str, float]
            ``away_cover`` and ``home_cover`` probabilities.
        """
        margin = self.away_runs.astype(np.float64) - self.home_runs.astype(np.float64)
        away_cover = float(np.mean(margin > line))
        home_cover = float(np.mean(margin < line))
        return {"away_cover": away_cover, "home_cover": home_cover}

    def over_under_probs(
        self,
        lines: list[float] | None = None,
    ) -> pd.DataFrame:
        """Over/under probabilities for total runs.

        Parameters
        ----------
        lines : list[float], optional
            Total-run lines to evaluate.  Defaults to half-point lines
            from 5.5 to 13.5.

        Returns
        -------
        pd.DataFrame
            Columns: line, p_over, p_under, expected, std.
        """
        total = self.away_runs + self.home_runs
        if lines is None:
            lines = [x + 0.5 for x in range(5, 14)]

        expected = float(np.mean(total))
        std = float(np.std(total))
        records = []
        for line in lines:
            p_over = float(np.mean(total > line))
            records.append({
                "line": line,
                "p_over": p_over,
                "p_under": 1.0 - p_over,
                "expected": expected,
                "std": std,
            })
        return pd.DataFrame(records)


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
# Half-inning simulation kernel
# ---------------------------------------------------------------------------

def _simulate_half_inning(
    # Per-batter accumulators (9, n_sims) -- modified in-place
    pa_total: np.ndarray,
    k_total: np.ndarray,
    bb_total: np.ndarray,
    hbp_total: np.ndarray,
    h_total: np.ndarray,
    single_total: np.ndarray,
    double_total: np.ndarray,
    triple_total: np.ndarray,
    hr_total_acc: np.ndarray,
    r_total: np.ndarray,
    rbi_total: np.ndarray,
    # Game state -- modified in-place
    bases_occ: np.ndarray,
    runner_id: np.ndarray,
    outs_in_inning: np.ndarray,
    lineup_pos: np.ndarray,
    global_bf: np.ndarray,
    team_runs: np.ndarray,
    # Which sims are active for this half
    active: np.ndarray,
    # Batter rates (9, n_sims)
    bat_k: np.ndarray,
    bat_bb: np.ndarray,
    bat_hr: np.ndarray,
    # Pitching context
    starter_bf: np.ndarray,
    starter_k_lift: float,
    starter_bb_lift: float,
    starter_hr_lift: float,
    bp_k_lift: float,
    bp_bb_lift: float,
    bp_hr_lift: float,
    # Matchup lifts (9,) arrays
    matchup_k_lifts: np.ndarray,
    matchup_bb_lifts: np.ndarray,
    matchup_hr_lifts: np.ndarray,
    bullpen_matchup_k_lifts: np.ndarray,
    bullpen_matchup_bb_lifts: np.ndarray,
    bullpen_matchup_hr_lifts: np.ndarray,
    # BABIP and context lifts
    batter_babip_adjs: np.ndarray,
    park_h_babip_adj: float,
    umpire_k_lift: float,
    umpire_bb_lift: float,
    park_k_lift: float,
    park_bb_lift: float,
    park_hr_lift: float,
    weather_k_lift: float,
    # Per-batter form lifts (9,) arrays
    form_k_lifts: np.ndarray,
    form_bb_lifts: np.ndarray,
    form_hr_lifts: np.ndarray,
    # Models and RNG
    pa_model: PAOutcomeModel,
    rng: np.random.Generator,
    max_pa: int = MAX_LINEUP_PA,
) -> np.ndarray:
    """Simulate one half-inning for all active sims.

    Loops through PAs until every active sim has recorded 3 outs (or
    the ``max_pa`` safety valve is reached). All mutable state arrays
    are updated **in-place**.

    Parameters
    ----------
    pa_total ... rbi_total : np.ndarray
        Per-batter counting stat accumulators, shape ``(9, n_sims)``.
    bases_occ : np.ndarray
        Base occupancy, shape ``(n_sims, 3)`` bool.
    runner_id : np.ndarray
        Runner identity on each base, shape ``(n_sims, 3)`` int8.
    outs_in_inning : np.ndarray
        Outs recorded so far this half-inning, shape ``(n_sims,)``.
    lineup_pos : np.ndarray
        Per-sim lineup position counter, shape ``(n_sims,)`` int32.
    global_bf : np.ndarray
        Cumulative batters faced by the opposing starter, per sim.
    team_runs : np.ndarray
        Cumulative team runs per sim.
    active : np.ndarray
        Bool mask of sims that should participate in this half-inning.
    bat_k, bat_bb, bat_hr : np.ndarray
        Batter rate posteriors, shape ``(9, n_sims)``.
    starter_bf : np.ndarray
        Starter batters-faced draw per sim, shape ``(n_sims,)``.
    starter_k_lift ... bp_hr_lift : float
        Pitcher quality lifts on the logit scale.
    matchup_k_lifts ... bullpen_matchup_hr_lifts : np.ndarray
        Per-batter matchup logit lifts, shape ``(9,)``.
    batter_babip_adjs : np.ndarray
        Per-batter BABIP adjustments, shape ``(9,)``.
    park_h_babip_adj : float
        Park factor BABIP adjustment.
    umpire_k_lift ... weather_k_lift : float
        Context logit lifts.
    pa_model : PAOutcomeModel
        PA outcome draw model.
    rng : np.random.Generator
        Random number generator.
    max_pa : int
        Safety valve for maximum PAs in a half-inning.

    Returns
    -------
    np.ndarray
        Runs scored this half-inning per sim, shape ``(n_sims,)``.
    """
    n_sims = len(active)
    runs_this_half = np.zeros(n_sims, dtype=np.int32)

    for _pa in range(max_pa):
        # Only process sims that are active AND haven't completed 3 outs
        still_batting = active & (outs_in_inning < 3)
        if not still_batting.any():
            break

        aidx = np.where(still_batting)[0]
        n_act = len(aidx)
        slot = lineup_pos[aidx] % LINEUP_SIZE  # per-sim slot

        # --- Track PA (per-sim batter slot) ---
        np.add.at(pa_total, (slot, aidx), 1)

        # --- Starter vs bullpen ---
        global_bf[aidx] += 1
        vs_starter = global_bf[aidx] <= starter_bf[aidx]

        # --- Compute adjusted rates ---
        # bat_k has shape (9, n_sims_total); index by (per-sim slot, global idx)
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
            + form_k_lifts[slot]
        )
        bb_rate = expit(
            bb_logit + bb_pitch + umpire_bb_lift + park_bb_lift
            + form_bb_lifts[slot]
        )
        hr_rate = expit(hr_logit + hr_pitch + park_hr_lift + form_hr_lifts[slot])

        # --- Draw PA outcomes ---
        probs = pa_model.compute_pa_probs(
            pitcher_k_rate=k_rate,
            pitcher_bb_rate=bb_rate,
            pitcher_hr_rate=hr_rate,
        )
        # BABIP adj: per-sim slot -> per-sim float; draw_outcomes expects a
        # scalar, so use the mean across active sims this PA.
        babip_vals = batter_babip_adjs[slot] + park_h_babip_adj
        babip_for_draw = float(np.mean(babip_vals))
        outcomes = pa_model.draw_outcomes(
            probs=probs,
            rng=rng,
            n_draws=n_act,
            babip_adj=babip_for_draw,
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
        np.add.at(k_total, (slot, aidx), is_k.astype(np.int32))
        np.add.at(bb_total, (slot, aidx), is_walk.astype(np.int32))
        np.add.at(hbp_total, (slot, aidx), is_hbp.astype(np.int32))
        np.add.at(hr_total_acc, (slot, aidx), is_hr.astype(np.int32))
        np.add.at(single_total, (slot, aidx), is_single.astype(np.int32))
        np.add.at(double_total, (slot, aidx), is_double.astype(np.int32))
        np.add.at(triple_total, (slot, aidx), is_triple.astype(np.int32))
        np.add.at(h_total, (slot, aidx), is_hit.astype(np.int32))

        # =================================================================
        # RUNNER ADVANCEMENT
        # =================================================================
        occ = bases_occ[aidx].copy()       # (n_act, 3) bool
        rid = runner_id[aidx].copy()       # (n_act, 3) int8
        cur_outs = outs_in_inning[aidx]    # (n_act,)

        # Pre-draw all random numbers for this PA at once
        rand_block = rng.random((n_act, 6))

        # Accumulators for this PA
        runs_this_pa = np.zeros(n_act, dtype=np.int32)
        rbi_this_pa = np.zeros(n_act, dtype=np.int32)

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

        # Per-sim slot as int8 for runner placement
        slot_i8 = slot.astype(np.int8)

        # -----------------------------------------------------------
        # SINGLE
        # -----------------------------------------------------------
        if is_single.any():
            m = is_single

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
            rid[:, 0] = np.where(m, slot_i8, rid[:, 0])

            # 2B: runner from 1B who stayed at 2B
            occ[:, 1] = np.where(m, to2_from1, occ[:, 1])
            rid[:, 1] = np.where(to2_from1, rid0_save, np.where(m, _EMPTY, rid[:, 1]))

            # 3B: stays from 3B OR from 2B (didn't score) OR from 1B (fast)
            new3 = stays3 | to3_from2 | to3_from1
            occ[:, 2] = np.where(m, new3, occ[:, 2])
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
            rid[:, 1] = np.where(m, slot_i8, rid[:, 1])

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
            rid[:, 2] = np.where(m, slot_i8, rid[:, 2])

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
            _score_runners(r_total, slot_i8, m, aidx)
            runs_this_pa += m.astype(np.int32)
            rbi_this_pa += m.astype(np.int32)

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

            # Force chain
            force_2_to_3 = has1 & has2
            new_occ3 = np.where(m, force_2_to_3 | (has3 & ~loaded), occ[:, 2])
            new_rid3 = np.where(
                m & force_2_to_3, rid1_save,
                np.where(m & has3 & ~loaded, rid[:, 2],
                np.where(m, _EMPTY, rid[:, 2])),
            )

            new_occ2 = np.where(m, has1 | (has2 & ~has1), occ[:, 1])
            new_rid2 = np.where(
                m & has1, rid0_save,
                np.where(m & has2 & ~has1, rid[:, 1],
                np.where(m, _EMPTY, rid[:, 1])),
            )

            new_occ1 = np.where(m, True, occ[:, 0])
            new_rid1 = np.where(m, slot_i8, rid[:, 0])

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

            # Runner on 3B scoring (sac fly / productive out)
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
        np.add.at(rbi_total, (slot, aidx), rbi_this_pa)
        team_runs[aidx] += runs_this_pa
        runs_this_half[aidx] += runs_this_pa

        # --- Update outs ---
        outs_in_inning[aidx] += is_out.astype(np.int32)

        # --- Advance lineup position ---
        lineup_pos[aidx] += 1

    return runs_this_half


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
    form_k_lifts: np.ndarray | None = None,
    form_bb_lifts: np.ndarray | None = None,
    form_hr_lifts: np.ndarray | None = None,
    game_context: GameContext | None = None,
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
    form_k_lifts : np.ndarray, optional
        Shape (9,) per-batter rolling form K% logit lifts.
    form_bb_lifts : np.ndarray, optional
        Shape (9,) per-batter rolling form BB% logit lifts.
    form_hr_lifts : np.ndarray, optional
        Shape (9,) per-batter rolling form HR logit lifts (HR/PA accel + hard-hit).
    game_context : GameContext, optional
        Per-game environmental lifts. When provided, its fields supply
        defaults for any environmental lift parameter left at 0.0.
        Explicit keyword arguments always take priority over the context.
    n_sims : int
        Number of Monte Carlo simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    LineupSimulationResult
        Per-batter and team-level counting stat distributions.
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

    # --- Default optional arrays ---
    matchup_k_lifts = default_lift_array(matchup_k_lifts, LINEUP_SIZE)
    matchup_bb_lifts = default_lift_array(matchup_bb_lifts, LINEUP_SIZE)
    matchup_hr_lifts = default_lift_array(matchup_hr_lifts, LINEUP_SIZE)
    bullpen_matchup_k_lifts = default_lift_array(bullpen_matchup_k_lifts, LINEUP_SIZE)
    bullpen_matchup_bb_lifts = default_lift_array(bullpen_matchup_bb_lifts, LINEUP_SIZE)
    bullpen_matchup_hr_lifts = default_lift_array(bullpen_matchup_hr_lifts, LINEUP_SIZE)
    batter_babip_adjs = default_lift_array(batter_babip_adjs, LINEUP_SIZE)
    form_k_lifts = default_lift_array(form_k_lifts, LINEUP_SIZE)
    form_bb_lifts = default_lift_array(form_bb_lifts, LINEUP_SIZE)
    form_hr_lifts = default_lift_array(form_hr_lifts, LINEUP_SIZE)

    # Dampen matchup lifts — empirical calibration from 11,517 game
    # walk-forward backtest (2023-2025). Raw pitch-type matchup scoring
    # over-applies lifts by ~2x for K/BB. HR signal is near-zero.
    matchup_k_lifts = matchup_k_lifts * MATCHUP_DAMPEN["k"]
    matchup_bb_lifts = matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    matchup_hr_lifts = matchup_hr_lifts * MATCHUP_DAMPEN["hr"]
    bullpen_matchup_k_lifts = bullpen_matchup_k_lifts * MATCHUP_DAMPEN["k"]
    bullpen_matchup_bb_lifts = bullpen_matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    bullpen_matchup_hr_lifts = bullpen_matchup_hr_lifts * MATCHUP_DAMPEN["hr"]

    # --- Resample batter posteriors to n_sims ---
    bat_k = np.stack([resample_posterior(s, n_sims, rng) for s in batter_k_rate_samples])   # (9, n_sims)
    bat_bb = np.stack([resample_posterior(s, n_sims, rng) for s in batter_bb_rate_samples])
    bat_hr = np.stack([resample_posterior(s, n_sims, rng) for s in batter_hr_rate_samples])

    # --- Pitcher quality lifts (logit scale) ---
    starter_k_lift, starter_bb_lift, starter_hr_lift = compute_pitcher_quality_lifts(
        starter_k_rate, starter_bb_rate, starter_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    bp_k_lift, bp_bb_lift, bp_hr_lift = compute_pitcher_quality_lifts(
        bullpen_k_rate, bullpen_bb_rate, bullpen_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    # --- Draw starter BF for each sim ---
    starter_bf = draw_bf_samples(
        mu_bf=starter_bf_mu,
        sigma_bf=starter_bf_sigma,
        n_draws=n_sims,
        bf_min=9,
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
    bases_occ = np.zeros((n_sims, 3), dtype=bool)
    runner_id = np.full((n_sims, 3), _EMPTY, dtype=np.int8)

    outs_in_inning = np.zeros(n_sims, dtype=np.int32)
    global_bf = np.zeros(n_sims, dtype=np.int32)
    team_runs = np.zeros(n_sims, dtype=np.int32)

    # Per-sim lineup position counter
    lineup_pos = np.zeros(n_sims, dtype=np.int32)

    # Active mask: sims still in progress
    active = np.ones(n_sims, dtype=bool)

    # --- 9-inning loop ---
    for _inn in range(9):
        # Reset half-inning state for active sims
        outs_in_inning[active] = 0
        bases_occ[active] = False
        runner_id[active] = _EMPTY

        _simulate_half_inning(
            pa_total, k_total, bb_total, hbp_total, h_total,
            single_total, double_total, triple_total, hr_total_acc,
            r_total, rbi_total,
            bases_occ, runner_id, outs_in_inning, lineup_pos,
            global_bf, team_runs,
            active,
            bat_k, bat_bb, bat_hr,
            starter_bf, starter_k_lift, starter_bb_lift, starter_hr_lift,
            bp_k_lift, bp_bb_lift, bp_hr_lift,
            matchup_k_lifts, matchup_bb_lifts, matchup_hr_lifts,
            bullpen_matchup_k_lifts, bullpen_matchup_bb_lifts,
            bullpen_matchup_hr_lifts,
            batter_babip_adjs, park_h_babip_adj,
            umpire_k_lift, umpire_bb_lift,
            park_k_lift, park_bb_lift, park_hr_lift,
            weather_k_lift,
            form_k_lifts, form_bb_lifts, form_hr_lifts,
            pa_model, rng,
        )

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


# ---------------------------------------------------------------------------
# Batting state helpers for full-game simulation
# ---------------------------------------------------------------------------

def _init_batting_state(n_sims: int) -> dict[str, np.ndarray]:
    """Allocate per-batter accumulators and game-state arrays.

    Returns a dict whose keys are used as keyword arguments to
    ``_simulate_half_inning``.
    """
    return {
        "pa_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "k_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "bb_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "hbp_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "h_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "single_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "double_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "triple_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "hr_total_acc": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "r_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "rbi_total": np.zeros((LINEUP_SIZE, n_sims), dtype=np.int32),
        "bases_occ": np.zeros((n_sims, 3), dtype=bool),
        "runner_id": np.full((n_sims, 3), _EMPTY, dtype=np.int8),
        "outs_in_inning": np.zeros(n_sims, dtype=np.int32),
        "global_bf": np.zeros(n_sims, dtype=np.int32),
        "team_runs": np.zeros(n_sims, dtype=np.int32),
        "lineup_pos": np.zeros(n_sims, dtype=np.int32),
    }


def _build_lineup_result(st: dict[str, np.ndarray], n_sims: int) -> LineupSimulationResult:
    """Build a ``LineupSimulationResult`` from a batting state dict."""
    tb = (
        st["single_total"]
        + 2 * st["double_total"]
        + 3 * st["triple_total"]
        + 4 * st["hr_total_acc"]
    )
    return LineupSimulationResult(
        pa=st["pa_total"],
        k=st["k_total"],
        bb=st["bb_total"],
        hbp=st["hbp_total"],
        h=st["h_total"],
        singles=st["single_total"],
        doubles=st["double_total"],
        triples=st["triple_total"],
        hr=st["hr_total_acc"],
        r=st["r_total"],
        rbi=st["rbi_total"],
        tb=tb,
        team_runs=st["team_runs"],
        n_sims=n_sims,
    )


# ---------------------------------------------------------------------------
# Full game (both teams) simulation
# ---------------------------------------------------------------------------

def simulate_full_game_both_teams(
    # Away team batting inputs
    away_batter_k_rate_samples: list[np.ndarray],
    away_batter_bb_rate_samples: list[np.ndarray],
    away_batter_hr_rate_samples: list[np.ndarray],
    # Home starter that away batters face
    home_starter_k_rate: float,
    home_starter_bb_rate: float,
    home_starter_hr_rate: float,
    home_starter_bf_mu: float,
    home_starter_bf_sigma: float,
    # Home team batting inputs
    home_batter_k_rate_samples: list[np.ndarray],
    home_batter_bb_rate_samples: list[np.ndarray],
    home_batter_hr_rate_samples: list[np.ndarray],
    # Away starter that home batters face
    away_starter_k_rate: float,
    away_starter_bb_rate: float,
    away_starter_hr_rate: float,
    away_starter_bf_mu: float,
    away_starter_bf_sigma: float,
    # Per-team matchup lifts (optional)
    away_matchup_k_lifts: np.ndarray | None = None,
    away_matchup_bb_lifts: np.ndarray | None = None,
    away_matchup_hr_lifts: np.ndarray | None = None,
    home_matchup_k_lifts: np.ndarray | None = None,
    home_matchup_bb_lifts: np.ndarray | None = None,
    home_matchup_hr_lifts: np.ndarray | None = None,
    # Per-team bullpen (optional)
    away_bullpen_k_rate: float = BULLPEN_K_RATE,
    away_bullpen_bb_rate: float = BULLPEN_BB_RATE,
    away_bullpen_hr_rate: float = BULLPEN_HR_RATE,
    home_bullpen_k_rate: float = BULLPEN_K_RATE,
    home_bullpen_bb_rate: float = BULLPEN_BB_RATE,
    home_bullpen_hr_rate: float = BULLPEN_HR_RATE,
    # Per-team bullpen matchup lifts (optional)
    away_bullpen_matchup_k_lifts: np.ndarray | None = None,
    away_bullpen_matchup_bb_lifts: np.ndarray | None = None,
    away_bullpen_matchup_hr_lifts: np.ndarray | None = None,
    home_bullpen_matchup_k_lifts: np.ndarray | None = None,
    home_bullpen_matchup_bb_lifts: np.ndarray | None = None,
    home_bullpen_matchup_hr_lifts: np.ndarray | None = None,
    # Per-team BABIP
    away_batter_babip_adjs: np.ndarray | None = None,
    home_batter_babip_adjs: np.ndarray | None = None,
    # Shared context (individual kwargs OR GameContext)
    park_k_lift: float = 0.0,
    park_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    park_h_babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    game_context: GameContext | None = None,
    # Per-team batter form lifts (optional)
    away_form_k_lifts: np.ndarray | None = None,
    away_form_bb_lifts: np.ndarray | None = None,
    away_form_hr_lifts: np.ndarray | None = None,
    home_form_k_lifts: np.ndarray | None = None,
    home_form_bb_lifts: np.ndarray | None = None,
    home_form_hr_lifts: np.ndarray | None = None,
    # Sim params
    n_sims: int = 50_000,
    random_seed: int = 42,
) -> FullGameSimulationResult:
    """Simulate a full 9-inning game with alternating half-innings.

    Runs both the away and home lineups through a shared 9-inning
    structure.  The top of each inning simulates the away team batting
    against the home pitching staff; the bottom simulates the home team
    batting against the away pitching staff.

    Walk-off logic is applied in the bottom of the 9th: if the home
    team already leads after the top of the 9th, the bottom half is
    skipped for those sims.  Tied sims after 9 innings are resolved
    with a coin flip (extra-inning simulation deferred).

    Parameters
    ----------
    away_batter_k_rate_samples, away_batter_bb_rate_samples,
    away_batter_hr_rate_samples : list of np.ndarray
        Length-9 lists of posterior sample arrays for the away lineup.
    home_starter_k_rate, home_starter_bb_rate, home_starter_hr_rate : float
        Home starting pitcher's posterior mean rates.
    home_starter_bf_mu, home_starter_bf_sigma : float
        Home starter batters-faced distribution parameters.
    home_batter_k_rate_samples, home_batter_bb_rate_samples,
    home_batter_hr_rate_samples : list of np.ndarray
        Length-9 lists of posterior sample arrays for the home lineup.
    away_starter_k_rate, away_starter_bb_rate, away_starter_hr_rate : float
        Away starting pitcher's posterior mean rates.
    away_starter_bf_mu, away_starter_bf_sigma : float
        Away starter batters-faced distribution parameters.
    away_matchup_*_lifts, home_matchup_*_lifts : np.ndarray, optional
        Shape (9,) logit lifts per batter vs the opposing starter.
    away_bullpen_*_rate, home_bullpen_*_rate : float
        Opposing bullpen aggregate rates for each side.
    away_bullpen_matchup_*_lifts, home_bullpen_matchup_*_lifts :
        np.ndarray, optional. Shape (9,) logit lifts per batter vs
        opposing bullpen.
    away_batter_babip_adjs, home_batter_babip_adjs : np.ndarray, optional
        Shape (9,) BABIP adjustments per batter for each side.
    park_k_lift ... weather_k_lift : float
        Shared park/umpire/weather context lifts.
    game_context : GameContext, optional
        Per-game environmental lifts. When provided, its fields supply
        defaults for any environmental lift parameter left at 0.0.
        Explicit keyword arguments always take priority over the context.
    away_form_k_lifts ... home_form_hr_lifts : np.ndarray, optional
        Shape (9,) per-batter rolling form logit lifts for each side.
    n_sims : int
        Number of Monte Carlo simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    FullGameSimulationResult
        Per-team lineup results plus game-level run totals and win
        outcomes.
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

    # --- Default optional arrays ---
    away_matchup_k_lifts = default_lift_array(away_matchup_k_lifts, LINEUP_SIZE)
    away_matchup_bb_lifts = default_lift_array(away_matchup_bb_lifts, LINEUP_SIZE)
    away_matchup_hr_lifts = default_lift_array(away_matchup_hr_lifts, LINEUP_SIZE)
    home_matchup_k_lifts = default_lift_array(home_matchup_k_lifts, LINEUP_SIZE)
    home_matchup_bb_lifts = default_lift_array(home_matchup_bb_lifts, LINEUP_SIZE)
    home_matchup_hr_lifts = default_lift_array(home_matchup_hr_lifts, LINEUP_SIZE)
    away_bullpen_matchup_k_lifts = default_lift_array(away_bullpen_matchup_k_lifts, LINEUP_SIZE)
    away_bullpen_matchup_bb_lifts = default_lift_array(away_bullpen_matchup_bb_lifts, LINEUP_SIZE)
    away_bullpen_matchup_hr_lifts = default_lift_array(away_bullpen_matchup_hr_lifts, LINEUP_SIZE)
    home_bullpen_matchup_k_lifts = default_lift_array(home_bullpen_matchup_k_lifts, LINEUP_SIZE)
    home_bullpen_matchup_bb_lifts = default_lift_array(home_bullpen_matchup_bb_lifts, LINEUP_SIZE)
    home_bullpen_matchup_hr_lifts = default_lift_array(home_bullpen_matchup_hr_lifts, LINEUP_SIZE)

    # Default form lift arrays
    away_form_k_lifts = default_lift_array(away_form_k_lifts, LINEUP_SIZE)
    away_form_bb_lifts = default_lift_array(away_form_bb_lifts, LINEUP_SIZE)
    away_form_hr_lifts = default_lift_array(away_form_hr_lifts, LINEUP_SIZE)
    home_form_k_lifts = default_lift_array(home_form_k_lifts, LINEUP_SIZE)
    home_form_bb_lifts = default_lift_array(home_form_bb_lifts, LINEUP_SIZE)
    home_form_hr_lifts = default_lift_array(home_form_hr_lifts, LINEUP_SIZE)

    # Dampen matchup lifts (same calibration as simulate_lineup_game)
    away_matchup_k_lifts = away_matchup_k_lifts * MATCHUP_DAMPEN["k"]
    away_matchup_bb_lifts = away_matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    away_matchup_hr_lifts = away_matchup_hr_lifts * MATCHUP_DAMPEN["hr"]
    home_matchup_k_lifts = home_matchup_k_lifts * MATCHUP_DAMPEN["k"]
    home_matchup_bb_lifts = home_matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    home_matchup_hr_lifts = home_matchup_hr_lifts * MATCHUP_DAMPEN["hr"]
    away_bullpen_matchup_k_lifts = away_bullpen_matchup_k_lifts * MATCHUP_DAMPEN["k"]
    away_bullpen_matchup_bb_lifts = away_bullpen_matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    away_bullpen_matchup_hr_lifts = away_bullpen_matchup_hr_lifts * MATCHUP_DAMPEN["hr"]
    home_bullpen_matchup_k_lifts = home_bullpen_matchup_k_lifts * MATCHUP_DAMPEN["k"]
    home_bullpen_matchup_bb_lifts = home_bullpen_matchup_bb_lifts * MATCHUP_DAMPEN["bb"]
    home_bullpen_matchup_hr_lifts = home_bullpen_matchup_hr_lifts * MATCHUP_DAMPEN["hr"]
    away_batter_babip_adjs = default_lift_array(away_batter_babip_adjs, LINEUP_SIZE)
    home_batter_babip_adjs = default_lift_array(home_batter_babip_adjs, LINEUP_SIZE)

    # --- Resample batter posteriors to n_sims ---
    away_bat_k = np.stack([resample_posterior(s, n_sims, rng) for s in away_batter_k_rate_samples])
    away_bat_bb = np.stack([resample_posterior(s, n_sims, rng) for s in away_batter_bb_rate_samples])
    away_bat_hr = np.stack([resample_posterior(s, n_sims, rng) for s in away_batter_hr_rate_samples])

    home_bat_k = np.stack([resample_posterior(s, n_sims, rng) for s in home_batter_k_rate_samples])
    home_bat_bb = np.stack([resample_posterior(s, n_sims, rng) for s in home_batter_bb_rate_samples])
    home_bat_hr = np.stack([resample_posterior(s, n_sims, rng) for s in home_batter_hr_rate_samples])

    # --- Pitcher quality lifts (logit scale) ---
    # Home pitching staff (away batters face these)
    home_starter_k_lift, home_starter_bb_lift, home_starter_hr_lift = compute_pitcher_quality_lifts(
        home_starter_k_rate, home_starter_bb_rate, home_starter_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    home_bp_k_lift, home_bp_bb_lift, home_bp_hr_lift = compute_pitcher_quality_lifts(
        home_bullpen_k_rate, home_bullpen_bb_rate, home_bullpen_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    # Away pitching staff (home batters face these)
    away_starter_k_lift, away_starter_bb_lift, away_starter_hr_lift = compute_pitcher_quality_lifts(
        away_starter_k_rate, away_starter_bb_rate, away_starter_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    away_bp_k_lift, away_bp_bb_lift, away_bp_hr_lift = compute_pitcher_quality_lifts(
        away_bullpen_k_rate, away_bullpen_bb_rate, away_bullpen_hr_rate,
        SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE,
    )

    # --- Draw starter BF for each side ---
    home_starter_bf = draw_bf_samples(
        mu_bf=home_starter_bf_mu, sigma_bf=home_starter_bf_sigma,
        n_draws=n_sims, bf_min=9, bf_max=35, rng=rng,
    )
    away_starter_bf = draw_bf_samples(
        mu_bf=away_starter_bf_mu, sigma_bf=away_starter_bf_sigma,
        n_draws=n_sims, bf_min=9, bf_max=35, rng=rng,
    )

    # --- Initialize batting state for each side ---
    away_st = _init_batting_state(n_sims)
    home_st = _init_batting_state(n_sims)

    # Active masks (all sims start active)
    away_active = np.ones(n_sims, dtype=bool)
    home_active = np.ones(n_sims, dtype=bool)

    # --- 9-inning alternating loop ---
    for inn in range(9):
        # ---- TOP: Away bats vs home pitching ----
        away_st["outs_in_inning"][away_active] = 0
        away_st["bases_occ"][away_active] = False
        away_st["runner_id"][away_active] = _EMPTY

        _simulate_half_inning(
            away_st["pa_total"], away_st["k_total"], away_st["bb_total"],
            away_st["hbp_total"], away_st["h_total"],
            away_st["single_total"], away_st["double_total"],
            away_st["triple_total"], away_st["hr_total_acc"],
            away_st["r_total"], away_st["rbi_total"],
            away_st["bases_occ"], away_st["runner_id"],
            away_st["outs_in_inning"], away_st["lineup_pos"],
            away_st["global_bf"], away_st["team_runs"],
            away_active,
            away_bat_k, away_bat_bb, away_bat_hr,
            home_starter_bf,
            home_starter_k_lift, home_starter_bb_lift, home_starter_hr_lift,
            home_bp_k_lift, home_bp_bb_lift, home_bp_hr_lift,
            away_matchup_k_lifts, away_matchup_bb_lifts, away_matchup_hr_lifts,
            home_bullpen_matchup_k_lifts, home_bullpen_matchup_bb_lifts,
            home_bullpen_matchup_hr_lifts,
            away_batter_babip_adjs, park_h_babip_adj,
            umpire_k_lift, umpire_bb_lift,
            park_k_lift, park_bb_lift, park_hr_lift,
            weather_k_lift,
            away_form_k_lifts, away_form_bb_lifts, away_form_hr_lifts,
            pa_model, rng,
        )

        # ---- Walk-off check: bottom of 9th ----
        # If home already leads after the top of the 9th, skip bottom
        if inn == 8:
            home_leads = home_st["team_runs"] > away_st["team_runs"]
            home_active = home_active & ~home_leads

        # ---- BOTTOM: Home bats vs away pitching ----
        home_st["outs_in_inning"][home_active] = 0
        home_st["bases_occ"][home_active] = False
        home_st["runner_id"][home_active] = _EMPTY

        _simulate_half_inning(
            home_st["pa_total"], home_st["k_total"], home_st["bb_total"],
            home_st["hbp_total"], home_st["h_total"],
            home_st["single_total"], home_st["double_total"],
            home_st["triple_total"], home_st["hr_total_acc"],
            home_st["r_total"], home_st["rbi_total"],
            home_st["bases_occ"], home_st["runner_id"],
            home_st["outs_in_inning"], home_st["lineup_pos"],
            home_st["global_bf"], home_st["team_runs"],
            home_active,
            home_bat_k, home_bat_bb, home_bat_hr,
            away_starter_bf,
            away_starter_k_lift, away_starter_bb_lift, away_starter_hr_lift,
            away_bp_k_lift, away_bp_bb_lift, away_bp_hr_lift,
            home_matchup_k_lifts, home_matchup_bb_lifts, home_matchup_hr_lifts,
            away_bullpen_matchup_k_lifts, away_bullpen_matchup_bb_lifts,
            away_bullpen_matchup_hr_lifts,
            home_batter_babip_adjs, park_h_babip_adj,
            umpire_k_lift, umpire_bb_lift,
            park_k_lift, park_bb_lift, park_hr_lift,
            weather_k_lift,
            home_form_k_lifts, home_form_bb_lifts, home_form_hr_lifts,
            pa_model, rng,
        )

    # --- Resolve ties with coin flip (extras deferred) ---
    away_runs = away_st["team_runs"]
    home_runs = home_st["team_runs"]
    tied = away_runs == home_runs
    n_tied = tied.sum()
    if n_tied > 0:
        coin = rng.random(n_tied) < 0.5
        home_runs[tied] += coin.astype(np.int32)
        away_runs[tied] += (~coin).astype(np.int32)

    home_win = home_runs > away_runs

    return FullGameSimulationResult(
        away=_build_lineup_result(away_st, n_sims),
        home=_build_lineup_result(home_st, n_sims),
        away_runs=away_runs,
        home_runs=home_runs,
        home_win=home_win,
        n_sims=n_sims,
    )
