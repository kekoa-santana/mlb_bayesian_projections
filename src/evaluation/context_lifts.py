"""Shared umpire and weather logit-lift construction for evaluation backtests.

Provides unified functions that build per-game logit lifts from umpire
tendencies and weather conditions.  Previously duplicated across
``game_k_validation.py`` (4 functions) and ``game_sim_validation.py``
(2 functions).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Umpire lifts
# ---------------------------------------------------------------------------

def build_umpire_logit_lifts(
    train_seasons: list[int],
    test_season: int,
    stats: tuple[str, ...] = ("k", "bb", "hr"),
) -> dict[str, dict[int, float]]:
    """Build per-umpire logit lifts for given stats.

    Computes umpire tendencies from training seasons, then maps each
    test-season game to its HP umpire's stat-specific logit lifts.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons used to estimate umpire tendencies.
    test_season : int
        Season whose games receive lift assignments.
    stats : tuple[str, ...]
        Which stats to build lifts for.  Default ``("k", "bb", "hr")``.

    Returns
    -------
    dict[str, dict[int, float]]
        ``{stat: {game_pk: logit_lift}}`` for each requested stat.
    """
    from src.data.db import read_sql
    from src.data.queries import get_umpire_tendencies

    ump_tendencies = get_umpire_tendencies(
        seasons=train_seasons, min_games=30,
    )

    # Initialise empty result for every requested stat
    result: dict[str, dict[int, float]] = {s: {} for s in stats}

    if ump_tendencies.empty:
        return result

    # Map stat name → column in the tendencies DataFrame
    stat_cols = {"k": "k_logit_lift", "bb": "bb_logit_lift", "hr": "hr_logit_lift"}
    ump_lift_maps: dict[str, dict[str, float]] = {}
    for stat in stats:
        col = stat_cols[stat]
        ump_lift_maps[stat] = dict(zip(
            ump_tendencies["hp_umpire_name"],
            ump_tendencies[col],
        ))

    # Get umpire assignments for test season
    ump_assignments = read_sql(f"""
        SELECT du.game_pk, du.hp_umpire_name
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    for _, row in ump_assignments.iterrows():
        gpk = int(row["game_pk"])
        name = row["hp_umpire_name"]
        for stat in stats:
            lift = ump_lift_maps[stat].get(name, 0.0)
            if pd.notna(lift):
                result[stat][gpk] = float(lift)

    logger.info(
        "Umpire lifts: %d games, stats=%s, non-zero%%=%s",
        len(ump_assignments),
        stats,
        {
            s: f"{100 * sum(1 for v in result[s].values() if abs(v) > 0.001) / max(len(result[s]), 1):.1f}"
            for s in stats
        },
    )
    return result


# ---------------------------------------------------------------------------
# Weather lifts
# ---------------------------------------------------------------------------

def build_weather_logit_lifts(
    train_seasons: list[int],
    test_season: int,
    stats: tuple[str, ...] = ("k", "hr"),
) -> dict[str, dict[int, float]]:
    """Build per-weather-bucket logit lifts for given stats.

    Computes weather multipliers from training seasons, converts to logit
    lifts, then maps each test-season outdoor game to its weather lift.
    Dome games receive a lift of 0.0.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons used to estimate weather effects.
    test_season : int
        Season whose games receive lift assignments.
    stats : tuple[str, ...]
        Which stats to build lifts for.  Default ``("k", "hr")``.
        BB is excluded by default because weather has negligible effect
        on BB rates.

    Returns
    -------
    dict[str, dict[int, float]]
        ``{stat: {game_pk: logit_lift}}`` for each requested stat.
    """
    from scipy.special import logit as _logit_fn

    from src.data.db import read_sql
    from src.data.queries import get_weather_effects

    wx_effects = get_weather_effects(seasons=train_seasons)

    # Initialise empty result for every requested stat
    result: dict[str, dict[int, float]] = {s: {} for s in stats}

    if wx_effects.empty:
        return result

    # Map stat → (overall_rate_col, multiplier_col)
    stat_meta = {
        "k":  ("overall_k_rate",  "k_multiplier"),
        "hr": ("overall_hr_rate", "hr_multiplier"),
    }

    # Build (temp_bucket, wind_category) → logit lift per stat
    wx_maps: dict[str, dict[tuple[str, str], float]] = {}
    for stat in stats:
        overall_col, mult_col = stat_meta[stat]
        overall_rate = float(wx_effects.iloc[0][overall_col])
        wx_map: dict[tuple[str, str], float] = {}
        for _, row in wx_effects.iterrows():
            key = (row["temp_bucket"], row["wind_category"])
            mult = float(row[mult_col])
            adj_rate = overall_rate * mult
            wx_map[key] = float(
                _logit_fn(np.clip(adj_rate, 1e-6, 1 - 1e-6))
                - _logit_fn(np.clip(overall_rate, 1e-6, 1 - 1e-6))
            )
        wx_maps[stat] = wx_map

    # Get weather for test season games
    weather_data = read_sql(f"""
        SELECT dw.game_pk, dw.is_dome, dw.wind_category,
            CASE
                WHEN dw.temperature < 55 THEN 'cold'
                WHEN dw.temperature BETWEEN 55 AND 69 THEN 'cool'
                WHEN dw.temperature BETWEEN 70 AND 84 THEN 'warm'
                ELSE 'hot'
            END AS temp_bucket
        FROM production.dim_weather dw
        JOIN production.dim_game dg ON dw.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    for _, row in weather_data.iterrows():
        gpk = int(row["game_pk"])
        if row.get("is_dome"):
            for stat in stats:
                result[stat][gpk] = 0.0
        else:
            key = (row.get("temp_bucket", ""), row.get("wind_category", ""))
            for stat in stats:
                result[stat][gpk] = wx_maps[stat].get(key, 0.0)

    logger.info(
        "Weather lifts: %d games, stats=%s, non-zero%%=%s",
        len(weather_data),
        stats,
        {
            s: f"{100 * sum(1 for v in result[s].values() if abs(v) > 0.001) / max(len(result[s]), 1):.1f}"
            for s in stats
        },
    )
    return result
