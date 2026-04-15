"""
Two-tier bullpen model for game simulation.

Replaces flat league-average bullpen rates with team-specific rates
split by leverage tier:
- High leverage (CL + SU): deployed in close games (|score_diff| <= 3)
- Low leverage (MR): deployed in blowouts

Builds TeamBullpenProfile from reliever role classification and
per-reliever season stats already in the system.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.constants import BULLPEN_BB_RATE, BULLPEN_HR_RATE, BULLPEN_K_RATE

logger = logging.getLogger(__name__)

# Score differential threshold for leverage tier selection.
# |diff| <= this value -> high leverage, else low leverage.
CLOSE_GAME_THRESHOLD = 3


@dataclass(frozen=True)
class TeamBullpenProfile:
    """Two-tier bullpen rates for a single team.

    High-leverage arms (CL + SU) are deployed in close games,
    low-leverage arms (MR) in blowouts.
    """

    team_id: int
    # High leverage (closer + setup)
    high_lev_k_rate: float
    high_lev_bb_rate: float
    high_lev_hr_rate: float
    high_lev_bf: int  # total BF for reliability
    # Low leverage (middle relief)
    low_lev_k_rate: float
    low_lev_bb_rate: float
    low_lev_hr_rate: float
    low_lev_bf: int

    def select_rates(
        self, score_diff: int | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Select bullpen rates based on game state.

        Parameters
        ----------
        score_diff : int or np.ndarray
            Score differential from the pitching team's perspective.
            Positive = winning, negative = losing.

        Returns
        -------
        tuple of (k_rate, bb_rate, hr_rate)
            Rates for the appropriate leverage tier.
        """
        close = np.abs(score_diff) <= CLOSE_GAME_THRESHOLD
        if isinstance(close, np.ndarray):
            k = np.where(close, self.high_lev_k_rate, self.low_lev_k_rate)
            bb = np.where(close, self.high_lev_bb_rate, self.low_lev_bb_rate)
            hr = np.where(close, self.high_lev_hr_rate, self.low_lev_hr_rate)
            return k, bb, hr
        if close:
            return self.high_lev_k_rate, self.high_lev_bb_rate, self.high_lev_hr_rate
        return self.low_lev_k_rate, self.low_lev_bb_rate, self.low_lev_hr_rate


# Minimum BF to trust tier-specific rates; below this, fall back to team aggregate
_MIN_TIER_BF = 200


def build_team_bullpen_profiles(
    role_history: pd.DataFrame,
    roles: pd.DataFrame,
    team_aggregate: pd.DataFrame,
    season: int,
) -> dict[int, TeamBullpenProfile]:
    """Build per-team two-tier bullpen profiles.

    Parameters
    ----------
    role_history : pd.DataFrame
        Per-reliever season stats from ``get_reliever_role_history()``.
        Columns: pitcher_id, season, bf, k, bb, hr, ...
    roles : pd.DataFrame
        Role classification from ``classify_reliever_roles()``.
        Columns: pitcher_id, role (CL/SU/MR), ...
    team_aggregate : pd.DataFrame
        Team-level aggregate bullpen rates from ``get_team_bullpen_rates()``.
        Columns: team_id, season, k_rate, bb_rate, hr_rate, total_bf.
    season : int
        Season to build profiles for.

    Returns
    -------
    dict[int, TeamBullpenProfile]
        Mapping of team_id -> profile. Teams without sufficient data
        get profiles with league-average rates.
    """
    # Get current-season reliever stats
    rh = role_history[role_history["season"] == season].copy()
    if rh.empty:
        logger.warning("No reliever history for season %d", season)
        return {}

    # Map pitcher_id -> team_id from the role history
    # Need team info - join from fact_player_game_mlb
    # role_history doesn't have team_id, so we need to get it
    # Use the team_aggregate to know which teams exist
    agg = team_aggregate[team_aggregate["season"] == season]

    # We need pitcher -> team mapping. Get from role_history by looking up
    # team assignment. For now, use a separate query approach:
    # role_history has pitcher_id + season stats; roles has pitcher_id + role.
    # We need to know which team each pitcher is on.
    # The role_history query doesn't include team_id, so we'll need to
    # build this from the roster data.

    # Merge roles onto stats
    if roles.empty or rh.empty:
        return {}

    merged = rh.merge(roles[["pitcher_id", "role"]], on="pitcher_id", how="left")
    merged["role"] = merged["role"].fillna("MR")

    # We need team_id for each pitcher. This should come from the caller.
    # For now, check if team_id is already in role_history
    if "team_id" not in merged.columns:
        logger.warning("role_history missing team_id; cannot build tier profiles")
        return {}

    profiles = {}
    for team_id in merged["team_id"].unique():
        team_df = merged[merged["team_id"] == team_id]

        # High leverage: CL + SU
        hi = team_df[team_df["role"].isin(["CL", "SU"])]
        hi_bf = int(hi["bf"].sum()) if len(hi) > 0 else 0
        hi_k = float(hi["k"].sum() / hi["bf"].sum()) if hi_bf > 0 else BULLPEN_K_RATE
        hi_bb = float(hi["bb"].sum() / hi["bf"].sum()) if hi_bf > 0 else BULLPEN_BB_RATE
        hi_hr = float(hi["hr"].sum() / hi["bf"].sum()) if hi_bf > 0 else BULLPEN_HR_RATE

        # Low leverage: MR
        lo = team_df[team_df["role"] == "MR"]
        lo_bf = int(lo["bf"].sum()) if len(lo) > 0 else 0
        lo_k = float(lo["k"].sum() / lo["bf"].sum()) if lo_bf > 0 else BULLPEN_K_RATE
        lo_bb = float(lo["bb"].sum() / lo["bf"].sum()) if lo_bf > 0 else BULLPEN_BB_RATE
        lo_hr = float(lo["hr"].sum() / lo["bf"].sum()) if lo_bf > 0 else BULLPEN_HR_RATE

        # If either tier has insufficient data, blend toward team aggregate
        team_agg = agg[agg["team_id"] == team_id]
        if len(team_agg) > 0:
            ta = team_agg.iloc[0]
            if hi_bf < _MIN_TIER_BF:
                blend = hi_bf / _MIN_TIER_BF
                hi_k = blend * hi_k + (1 - blend) * float(ta["k_rate"])
                hi_bb = blend * hi_bb + (1 - blend) * float(ta["bb_rate"])
                hi_hr = blend * hi_hr + (1 - blend) * float(ta["hr_rate"])
            if lo_bf < _MIN_TIER_BF:
                blend = lo_bf / _MIN_TIER_BF
                lo_k = blend * lo_k + (1 - blend) * float(ta["k_rate"])
                lo_bb = blend * lo_bb + (1 - blend) * float(ta["bb_rate"])
                lo_hr = blend * lo_hr + (1 - blend) * float(ta["hr_rate"])

        profiles[int(team_id)] = TeamBullpenProfile(
            team_id=int(team_id),
            high_lev_k_rate=hi_k, high_lev_bb_rate=hi_bb, high_lev_hr_rate=hi_hr,
            high_lev_bf=hi_bf,
            low_lev_k_rate=lo_k, low_lev_bb_rate=lo_bb, low_lev_hr_rate=lo_hr,
            low_lev_bf=lo_bf,
        )

    logger.info(
        "Built %d team bullpen profiles for %d. "
        "Avg high-lev K%%=%.3f BB%%=%.3f, low-lev K%%=%.3f BB%%=%.3f",
        len(profiles), season,
        np.mean([p.high_lev_k_rate for p in profiles.values()]),
        np.mean([p.high_lev_bb_rate for p in profiles.values()]),
        np.mean([p.low_lev_k_rate for p in profiles.values()]),
        np.mean([p.low_lev_bb_rate for p in profiles.values()]),
    )
    return profiles


def get_default_profile(team_id: int = 0) -> TeamBullpenProfile:
    """Return a profile with league-average rates for both tiers."""
    return TeamBullpenProfile(
        team_id=team_id,
        high_lev_k_rate=BULLPEN_K_RATE, high_lev_bb_rate=BULLPEN_BB_RATE,
        high_lev_hr_rate=BULLPEN_HR_RATE, high_lev_bf=0,
        low_lev_k_rate=BULLPEN_K_RATE, low_lev_bb_rate=BULLPEN_BB_RATE,
        low_lev_hr_rate=BULLPEN_HR_RATE, low_lev_bf=0,
    )
