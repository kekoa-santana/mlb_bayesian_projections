"""
Reliever role classification: Closer (CL), Setup (SU), Middle (MR).

Classifies relievers by recency-weighted saves/holds and produces per-role
priors for the season simulator (games, BF/app, save/hold opportunities).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RelieverRolePriors:
    """Per-role population priors for season simulation."""

    role: str
    games_mu: float
    games_sigma: float
    bf_per_app_mu: float
    bf_per_app_sigma: float
    save_opp_pct: float     # fraction of appearances that are save opportunities
    save_conversion: float  # P(save | save opportunity)
    hold_opp_pct: float     # fraction of appearances that are hold opportunities
    hold_conversion: float  # P(hold | hold opportunity)


# Empirical priors from 2021-2025 DB actuals (validated against query)
# Empirical priors calibrated from DB actuals (2021-2025 verified)
_DEFAULT_ROLE_PRIORS: dict[str, RelieverRolePriors] = {
    "CL": RelieverRolePriors(
        role="CL",
        games_mu=63.0, games_sigma=12.0,
        bf_per_app_mu=4.0, bf_per_app_sigma=1.5,
        save_opp_pct=0.40, save_conversion=0.80,  # DB: ~40% opp, ~80% conv
        hold_opp_pct=0.12, hold_conversion=0.80,
    ),
    "SU": RelieverRolePriors(
        role="SU",
        games_mu=60.0, games_sigma=14.0,
        bf_per_app_mu=4.1, bf_per_app_sigma=1.5,
        save_opp_pct=0.05, save_conversion=0.80,
        hold_opp_pct=0.26, hold_conversion=1.00,  # DB: ~26% hold rate (hold = opp*conv)
    ),
    "MR": RelieverRolePriors(
        role="MR",
        games_mu=45.0, games_sigma=18.0,
        bf_per_app_mu=5.1, bf_per_app_sigma=2.5,
        save_opp_pct=0.02, save_conversion=0.80,
        hold_opp_pct=0.06, hold_conversion=0.80,
    ),
}


def classify_reliever_roles(
    seasons: list[int],
    current_season: int,
    min_games: int = 10,
) -> pd.DataFrame:
    """Classify relievers into CL / SU / MR roles.

    Uses recency-weighted saves and holds (current season 3x, prior 1x).
    Threshold: weighted_saves >= 24 -> CL, weighted_holds >= 24 -> SU, else MR.
    Calibrated to match DB actuals (~40 CL, ~110 SU, ~250 MR).

    Parameters
    ----------
    seasons : list[int]
        Seasons to consider (usually last 2-3).
    current_season : int
        Most recent season (gets 3x weight).
    min_games : int
        Minimum games in current season to be classified.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, role, confidence, games, saves, holds,
        weighted_saves, weighted_holds, k_rate, bb_rate, bf_per_app.
    """
    from src.data.queries import get_reliever_role_history

    history = get_reliever_role_history(seasons, min_games=1)
    if history.empty:
        logger.warning("No reliever role history found")
        return pd.DataFrame()

    # Recency weights: current season 3x, prior seasons 1x
    history["weight"] = np.where(
        history["season"] == current_season, 3.0, 1.0,
    )
    history["weighted_saves"] = history["saves"] * history["weight"]
    history["weighted_holds"] = history["holds"] * history["weight"]

    # Aggregate per pitcher
    agg = (
        history.groupby("pitcher_id")
        .agg(
            total_games=("games", "sum"),
            current_games=("games", lambda x: x[history.loc[x.index, "season"] == current_season].sum()),
            saves=("saves", "sum"),
            holds=("holds", "sum"),
            blown_saves=("blown_saves", "sum"),
            weighted_saves=("weighted_saves", "sum"),
            weighted_holds=("weighted_holds", "sum"),
            total_bf=("bf", "sum"),
            total_k=("k", "sum"),
            total_bb=("bb", "sum"),
        )
        .reset_index()
    )

    # Filter to pitchers active in current season with enough games
    current_active = set(
        history[
            (history["season"] == current_season)
            & (history["games"] >= min_games)
        ]["pitcher_id"]
    )
    agg = agg[agg["pitcher_id"].isin(current_active)].copy()

    if agg.empty:
        logger.warning("No relievers met min_games=%d in season %d", min_games, current_season)
        return pd.DataFrame()

    # Classify
    agg["role"] = "MR"
    agg.loc[agg["weighted_saves"] >= 24, "role"] = "CL"
    # SU only if not already CL
    su_mask = (agg["role"] == "MR") & (agg["weighted_holds"] >= 24)
    agg.loc[su_mask, "role"] = "SU"

    # Confidence: how clearly the pitcher fits the role
    # CL: confidence = min(weighted_saves / 20, 1.0)
    # SU: confidence = min(weighted_holds / 20, 1.0)
    # MR: confidence = 0.5 + 0.5 * min(current_games / 40, 1.0)
    agg["confidence"] = 0.5
    cl_mask = agg["role"] == "CL"
    agg.loc[cl_mask, "confidence"] = (agg.loc[cl_mask, "weighted_saves"] / 20.0).clip(0, 1)
    su_mask = agg["role"] == "SU"
    agg.loc[su_mask, "confidence"] = (agg.loc[su_mask, "weighted_holds"] / 20.0).clip(0, 1)
    mr_mask = agg["role"] == "MR"
    agg.loc[mr_mask, "confidence"] = 0.5 + 0.5 * (agg.loc[mr_mask, "current_games"] / 40.0).clip(0, 1)

    # Derived rates
    agg["k_rate"] = agg["total_k"] / agg["total_bf"].replace(0, np.nan)
    agg["bb_rate"] = agg["total_bb"] / agg["total_bf"].replace(0, np.nan)
    agg["bf_per_app"] = agg["total_bf"] / agg["total_games"].replace(0, np.nan)

    result = agg[[
        "pitcher_id", "role", "confidence", "current_games",
        "saves", "holds", "blown_saves",
        "weighted_saves", "weighted_holds",
        "k_rate", "bb_rate", "bf_per_app",
    ]].rename(columns={"current_games": "games"})

    role_counts = result["role"].value_counts().to_dict()
    logger.info(
        "Classified %d relievers: %s",
        len(result), role_counts,
    )
    return result


def get_role_priors(role: str) -> RelieverRolePriors:
    """Get population priors for a reliever role.

    Parameters
    ----------
    role : str
        One of 'CL', 'SU', 'MR'.

    Returns
    -------
    RelieverRolePriors
    """
    return _DEFAULT_ROLE_PRIORS.get(role, _DEFAULT_ROLE_PRIORS["MR"])
