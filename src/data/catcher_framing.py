"""Catcher framing lookup utilities.

Builds per-game catcher framing logit lifts for K and BB rates.
Catcher framing affects called-strike rates on borderline pitches,
which translates to ±0.5-1.0 pp on K% for elite/poor framers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Scaling factor: called strikes are only a fraction of the pathways
# to K or BB outcomes, so the raw framing logit lift is attenuated.
_FRAMING_WEIGHT: float = 0.3


def get_catcher_framing_lift(
    catcher_id: int,
    season: int,
    framing_data: pd.DataFrame,
    weight: float = _FRAMING_WEIGHT,
) -> dict[str, float]:
    """Return logit lifts for K and BB from catcher framing effects.

    Parameters
    ----------
    catcher_id : int
        Catcher MLB ID.
    season : int
        Season to look up (uses most recent available if exact season
        is missing).
    framing_data : pd.DataFrame
        Output of ``get_catcher_framing_effects()``.  Must contain
        columns: catcher_id, season, logit_lift.
    weight : float
        Scaling factor applied to the raw framing logit lift.

    Returns
    -------
    dict[str, float]
        Keys: ``k_logit_lift`` (positive = more Ks),
        ``bb_logit_lift`` (negative = fewer BBs when framing is good).
        Both are 0.0 if catcher not found.
    """
    if framing_data is None or framing_data.empty:
        return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}

    mask = framing_data["catcher_id"] == catcher_id
    catcher_rows = framing_data.loc[mask]

    if catcher_rows.empty:
        return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}

    exact = catcher_rows.loc[catcher_rows["season"] == season]
    if not exact.empty:
        raw_lift = float(exact.iloc[0]["logit_lift"])
    else:
        prior = catcher_rows.loc[catcher_rows["season"] <= season]
        if prior.empty:
            return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}
        raw_lift = float(prior.sort_values("season").iloc[-1]["logit_lift"])

    weighted_lift = raw_lift * weight

    # Good framing (positive lift) -> more called strikes -> more K, fewer BB
    return {
        "k_logit_lift": weighted_lift,
        "bb_logit_lift": -weighted_lift,
    }


def build_catcher_framing_lookup(
    train_seasons: list[int],
    test_season: int,
) -> dict[str, dict[tuple[int, int], float]]:
    """Build (game_pk, pitcher_id) -> catcher framing logit lifts.

    Computes framing effects from training seasons, identifies the starting
    catcher for each starting pitcher's game in the test season, and returns
    per-(game_pk, pitcher_id) logit lifts for K and BB.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to compute framing effects from.
    test_season : int
        Season whose games to map.

    Returns
    -------
    dict[str, dict[tuple[int, int], float]]
        ``{"k": {(game_pk, pitcher_id): lift}, "bb": {(game_pk, pitcher_id): lift}}``.
    """
    from src.data.queries import (
        get_catcher_framing_effects,
        get_catcher_game_assignments,
        get_game_starter_teams,
    )

    framing_data = get_catcher_framing_effects(seasons=train_seasons)
    if framing_data.empty:
        logger.warning("No catcher framing data for seasons %s", train_seasons)
        return {"k": {}, "bb": {}}

    catcher_assignments = get_catcher_game_assignments(int(test_season))

    if catcher_assignments.empty:
        logger.warning("No catcher lineup data for season %d", test_season)
        return {"k": {}, "bb": {}}

    pitcher_teams = get_game_starter_teams(int(test_season))

    if pitcher_teams.empty:
        logger.warning("No starter data for season %d", test_season)
        return {"k": {}, "bb": {}}

    # Build (game_pk, team_id) -> catcher framing lift
    catcher_lift_by_team: dict[tuple[int, int], dict[str, float]] = {}
    last_train = max(train_seasons)
    for _, row in catcher_assignments.iterrows():
        gpk = int(row["game_pk"])
        catcher_id_val = int(row["catcher_id"])
        team_id = int(row["team_id"])

        lifts = get_catcher_framing_lift(
            catcher_id=catcher_id_val,
            season=last_train,
            framing_data=framing_data,
        )
        catcher_lift_by_team[(gpk, team_id)] = lifts

    # Map each starter game to the catcher on the SAME team
    k_lifts: dict[tuple[int, int], float] = {}
    bb_lifts: dict[tuple[int, int], float] = {}

    for _, row in pitcher_teams.iterrows():
        gpk = int(row["game_pk"])
        pid = int(row["pitcher_id"])
        team_id = int(row["team_id"])
        team_key = (gpk, team_id)

        if team_key in catcher_lift_by_team:
            lifts = catcher_lift_by_team[team_key]
            k_lifts[(gpk, pid)] = lifts["k_logit_lift"]
            bb_lifts[(gpk, pid)] = lifts["bb_logit_lift"]

    n_entries = len(k_lifts)
    non_zero_k = sum(1 for v in k_lifts.values() if abs(v) > 0.001)
    logger.info(
        "Catcher framing lookup: %d pitcher-games, %d non-zero K lifts",
        n_entries, non_zero_k,
    )
    return {"k": k_lifts, "bb": bb_lifts}
