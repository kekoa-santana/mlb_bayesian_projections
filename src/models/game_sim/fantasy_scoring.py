"""
Fantasy point distributions from game simulation results.

Computes DraftKings and ESPN fantasy point distributions as a linear
transform of the simulation output arrays. Since all stats come from
the same simulation, the fantasy point distribution captures the
correct correlations between stats.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.game_sim.batter_simulator import BatterSimulationResult
from src.models.game_sim.simulator import SimulationResult

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# DraftKings scoring rules (classic)
# -----------------------------------------------------------------------
# Batter
DK_BAT_SINGLE = 3.0
DK_BAT_DOUBLE = 5.0
DK_BAT_TRIPLE = 8.0
DK_BAT_HR = 10.0
DK_BAT_RBI = 2.0
DK_BAT_R = 2.0
DK_BAT_BB = 2.0
DK_BAT_HBP = 2.0
DK_BAT_SB = 5.0  # not modeled in sim

# Pitcher
DK_PIT_IP = 2.25   # per out (0.75 per out = 2.25 per IP)
DK_PIT_K = 2.0
DK_PIT_W = 4.0     # not modeled in sim
DK_PIT_ER = -2.0
DK_PIT_H = -0.6
DK_PIT_BB = -0.6
DK_PIT_HBP = -0.6
DK_PIT_CG = 2.5    # bonus, rare
DK_PIT_CGSO = 2.5  # bonus on top of CG
DK_PIT_NH = 5.0    # bonus, extremely rare

# -----------------------------------------------------------------------
# ESPN standard points (H2H categories often, but points leagues use):
# -----------------------------------------------------------------------
ESPN_BAT_SINGLE = 1.0
ESPN_BAT_DOUBLE = 2.0
ESPN_BAT_TRIPLE = 3.0
ESPN_BAT_HR = 4.0
ESPN_BAT_RBI = 1.0
ESPN_BAT_R = 1.0
ESPN_BAT_BB = 1.0
ESPN_BAT_HBP = 1.0
ESPN_BAT_SB = 1.0
ESPN_BAT_K = -1.0  # batters penalized for K

ESPN_PIT_IP = 3.0   # per IP
ESPN_PIT_K = 1.0
ESPN_PIT_W = 5.0
ESPN_PIT_ER = -2.0
ESPN_PIT_H = -1.0
ESPN_PIT_BB = -1.0


@dataclass
class FantasyResult:
    """Fantasy point distribution from a simulation."""

    dk_points: np.ndarray
    espn_points: np.ndarray
    n_sims: int

    def dk_summary(self) -> dict[str, float]:
        return {
            "mean": float(np.mean(self.dk_points)),
            "std": float(np.std(self.dk_points)),
            "median": float(np.median(self.dk_points)),
            "q10": float(np.percentile(self.dk_points, 10)),
            "q25": float(np.percentile(self.dk_points, 25)),
            "q75": float(np.percentile(self.dk_points, 75)),
            "q90": float(np.percentile(self.dk_points, 90)),
        }

    def espn_summary(self) -> dict[str, float]:
        return {
            "mean": float(np.mean(self.espn_points)),
            "std": float(np.std(self.espn_points)),
            "median": float(np.median(self.espn_points)),
            "q10": float(np.percentile(self.espn_points, 10)),
            "q25": float(np.percentile(self.espn_points, 25)),
            "q75": float(np.percentile(self.espn_points, 75)),
            "q90": float(np.percentile(self.espn_points, 90)),
        }

    def dk_over_probs(
        self, lines: list[float] | None = None,
    ) -> pd.DataFrame:
        """P(DK points over X) for given lines."""
        if lines is None:
            lines = [5, 10, 15, 20, 25, 30]
        records = []
        for line in lines:
            records.append({
                "line": line,
                "p_over": float(np.mean(self.dk_points > line)),
            })
        return pd.DataFrame(records)


def compute_batter_fantasy(
    result: BatterSimulationResult,
) -> FantasyResult:
    """Compute fantasy point distributions for a batter game simulation.

    Parameters
    ----------
    result : BatterSimulationResult
        Output of simulate_batter_game().

    Returns
    -------
    FantasyResult
        DK and ESPN point distributions.
    """
    dk = (
        DK_BAT_SINGLE * result.single_samples
        + DK_BAT_DOUBLE * result.double_samples
        + DK_BAT_TRIPLE * result.triple_samples
        + DK_BAT_HR * result.hr_samples
        + DK_BAT_RBI * result.rbi_samples
        + DK_BAT_R * result.r_samples
        + DK_BAT_BB * result.bb_samples
        + DK_BAT_HBP * result.hbp_samples
    ).astype(float)

    espn = (
        ESPN_BAT_SINGLE * result.single_samples
        + ESPN_BAT_DOUBLE * result.double_samples
        + ESPN_BAT_TRIPLE * result.triple_samples
        + ESPN_BAT_HR * result.hr_samples
        + ESPN_BAT_RBI * result.rbi_samples
        + ESPN_BAT_R * result.r_samples
        + ESPN_BAT_BB * result.bb_samples
        + ESPN_BAT_HBP * result.hbp_samples
        + ESPN_BAT_K * result.k_samples
    ).astype(float)

    return FantasyResult(
        dk_points=dk,
        espn_points=espn,
        n_sims=result.n_sims,
    )


def compute_pitcher_fantasy(
    result: SimulationResult,
) -> FantasyResult:
    """Compute fantasy point distributions for a pitcher game simulation.

    Parameters
    ----------
    result : SimulationResult
        Output of simulate_game().

    Returns
    -------
    FantasyResult
        DK and ESPN point distributions.

    Notes
    -----
    Win probability is not modeled, so DK_PIT_W and ESPN_PIT_W are
    excluded. CG/CGSO/NH bonuses are calculated from the sim
    (27 outs = CG, 27 outs + 0 runs = CGSO).
    """
    outs = result.outs_samples.astype(float)
    ip = outs / 3.0  # continuous IP for scoring

    dk = (
        DK_PIT_IP * ip * 3.0  # DK scores per out: 0.75 per out = 2.25 per IP
        + DK_PIT_K * result.k_samples
        + DK_PIT_ER * result.runs_samples  # using runs as ER proxy
        + DK_PIT_H * result.h_samples
        + DK_PIT_BB * result.bb_samples
        + DK_PIT_HBP * result.hbp_samples
    ).astype(float)

    # CG bonus (27 outs)
    is_cg = result.outs_samples >= 27
    dk += DK_PIT_CG * is_cg
    # CGSO bonus (CG + 0 runs)
    is_cgso = is_cg & (result.runs_samples == 0)
    dk += DK_PIT_CGSO * is_cgso

    espn = (
        ESPN_PIT_IP * ip
        + ESPN_PIT_K * result.k_samples
        + ESPN_PIT_ER * result.runs_samples
        + ESPN_PIT_H * result.h_samples
        + ESPN_PIT_BB * result.bb_samples
    ).astype(float)

    return FantasyResult(
        dk_points=dk,
        espn_points=espn,
        n_sims=result.n_sims,
    )


# -----------------------------------------------------------------------
# Season-level scoring (ESPN saves/blown saves from DB: +2 SV, -2 BS)
# -----------------------------------------------------------------------
ESPN_PIT_SV = 2.0    # verified from fantasy.espn_pitcher_game_scores
ESPN_PIT_BS = -2.0   # verified from fantasy.espn_pitcher_game_scores


def compute_season_pitcher_fantasy(
    k: np.ndarray,
    bb: np.ndarray,
    h: np.ndarray,
    hr: np.ndarray,
    hbp: np.ndarray,
    outs: np.ndarray,
    runs: np.ndarray,
    sv: np.ndarray,
    hld: np.ndarray,
) -> FantasyResult:
    """Compute season-level fantasy points from counting stat arrays.

    All arrays shape (n_seasons,). DK has no save scoring (confirmed from
    DB schema — dk_pitcher_game_scores has no SV column). ESPN adds save
    and blown save points.

    Parameters
    ----------
    k, bb, h, hr, hbp, outs, runs, sv, hld : np.ndarray
        Season counting stat totals.

    Returns
    -------
    FantasyResult
        Season-level DK and ESPN point distributions.
    """
    ip = outs.astype(float) / 3.0
    er = runs.astype(float) * 0.92  # league-average earned run fraction

    # DK: no save scoring — closer DK value comes from stuff
    dk = (
        DK_PIT_IP * ip * 3.0  # 0.75 per out = 2.25 per IP, scored per out
        + DK_PIT_K * k
        + DK_PIT_ER * er
        + DK_PIT_H * h
        + DK_PIT_BB * bb
        + DK_PIT_HBP * hbp
    ).astype(float)

    # ESPN: includes save points
    espn = (
        ESPN_PIT_IP * ip
        + ESPN_PIT_K * k
        + ESPN_PIT_ER * er
        + ESPN_PIT_H * h
        + ESPN_PIT_BB * bb
        + ESPN_PIT_SV * sv
    ).astype(float)

    return FantasyResult(
        dk_points=dk,
        espn_points=espn,
        n_sims=len(k),
    )
