"""
Lineup context model for R/RBI adjustments.

A hitter's R and RBI depend heavily on:
- Batting order position (leadoff = more R, cleanup = more RBI)
- Team offensive quality (better teammates = more opportunities)

This module computes per-player R/RBI multipliers that adjust the
sim's league-average run-scoring probabilities to account for context.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# League-average R and RBI per PA by batting order position (2021-2025)
# Computed from fact_player_game_mlb
_DEFAULT_R_PER_PA_BY_SLOT = {
    1: 0.135, 2: 0.125, 3: 0.120, 4: 0.112, 5: 0.108,
    6: 0.103, 7: 0.098, 8: 0.095, 9: 0.090,
}
_DEFAULT_RBI_PER_PA_BY_SLOT = {
    1: 0.090, 2: 0.115, 3: 0.128, 4: 0.130, 5: 0.120,
    6: 0.112, 7: 0.105, 8: 0.098, 9: 0.085,
}
_LEAGUE_AVG_R_PER_PA = 0.110
_LEAGUE_AVG_RBI_PER_PA = 0.109


def compute_lineup_context(
    season: int,
    min_games: int = 50,
) -> pd.DataFrame:
    """Compute per-player R/RBI context multipliers.

    Uses actual batting order and team OBP to determine how much
    more/less R and RBI a player gets compared to league average.

    Parameters
    ----------
    season : int
        Season to compute context from.
    min_games : int
        Minimum games for reliable context estimate.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, avg_order, team_obp, r_multiplier, rbi_multiplier.
        Multipliers are centered on 1.0 (e.g., 1.15 = 15% more than average).
    """
    # Get batting order and team for each player
    player_context = read_sql("""
        SELECT fl.player_id as batter_id,
               ROUND(AVG(fl.batting_order))::int as avg_order,
               fg.team_id,
               COUNT(DISTINCT fg.game_pk) as games,
               SUM(fg.bat_r)::float / NULLIF(SUM(fg.bat_pa), 0) as r_per_pa,
               SUM(fg.bat_rbi)::float / NULLIF(SUM(fg.bat_pa), 0) as rbi_per_pa
        FROM production.fact_lineup fl
        JOIN production.fact_player_game_mlb fg
            ON fl.player_id = fg.player_id AND fl.game_pk = fg.game_pk
        WHERE fl.season = :season AND fl.is_starter = true
              AND fl.batting_order BETWEEN 1 AND 9
              AND fg.player_role = 'batter'
        GROUP BY fl.player_id, fg.team_id
        HAVING COUNT(DISTINCT fg.game_pk) >= :min_games
    """, {"season": season, "min_games": min_games})

    if player_context.empty:
        return pd.DataFrame()

    # Get team offensive quality (OBP)
    team_obp = read_sql("""
        SELECT team_id,
               SUM(bat_h + bat_bb + bat_hbp)::float /
                   NULLIF(SUM(bat_pa), 0) as team_obp
        FROM production.fact_player_game_mlb
        WHERE player_role = 'batter' AND season = :season
        GROUP BY team_id
    """, {"season": season})

    lg_obp = team_obp["team_obp"].mean() if not team_obp.empty else 0.315

    if not team_obp.empty:
        player_context = player_context.merge(team_obp, on="team_id", how="left")
    else:
        player_context["team_obp"] = lg_obp

    player_context["team_obp"] = player_context["team_obp"].fillna(lg_obp)

    # R multiplier: batting order effect + team quality
    # Leadoff hitters score ~23% more R/PA than league avg
    # Team OBP above average means more baserunners = more R for everyone
    r_order_mult = player_context["avg_order"].map(_DEFAULT_R_PER_PA_BY_SLOT).fillna(
        _LEAGUE_AVG_R_PER_PA
    ) / _LEAGUE_AVG_R_PER_PA
    team_r_mult = player_context["team_obp"] / lg_obp

    # RBI multiplier: batting order + team quality
    rbi_order_mult = player_context["avg_order"].map(_DEFAULT_RBI_PER_PA_BY_SLOT).fillna(
        _LEAGUE_AVG_RBI_PER_PA
    ) / _LEAGUE_AVG_RBI_PER_PA
    team_rbi_mult = team_r_mult  # same team effect

    # Combined: order effect * team effect, regressed toward 1.0
    # Use 0.7 * (combined) + 0.3 * 1.0 to avoid extreme adjustments
    raw_r = r_order_mult * team_r_mult
    raw_rbi = rbi_order_mult * team_rbi_mult
    player_context["r_multiplier"] = (0.70 * raw_r + 0.30).round(3)
    player_context["rbi_multiplier"] = (0.70 * raw_rbi + 0.30).round(3)

    logger.info(
        "Lineup context: %d hitters, R mult range [%.2f-%.2f], RBI mult range [%.2f-%.2f]",
        len(player_context),
        player_context["r_multiplier"].min(), player_context["r_multiplier"].max(),
        player_context["rbi_multiplier"].min(), player_context["rbi_multiplier"].max(),
    )

    return player_context[["batter_id", "avg_order", "team_obp",
                            "r_multiplier", "rbi_multiplier"]]
