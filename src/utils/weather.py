"""Shared weather helper functions for temperature and wind bucketing.

Thresholds match the SQL spec in src/data/queries/environment.py (lines 291-296)
and the categories in production.dim_weather.
"""

from __future__ import annotations

import math


def parse_temp_bucket(temp) -> str:
    """Convert temperature to dim_weather bucket string.

    Parameters
    ----------
    temp : float, int, str, or None
        Temperature value (Fahrenheit).

    Returns
    -------
    str
        One of "cold", "cool", "warm", "hot", or "unknown".

    Bucket cutoffs match the SQL in src/data/queries/environment.py:
    cold < 55, cool <= 69, warm <= 84, hot > 84.
    """
    if temp is None:
        return "unknown"
    if isinstance(temp, float) and (math.isnan(temp) or math.isinf(temp)):
        return "unknown"
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(t):
        return "unknown"
    if t < 55:
        return "cold"
    if t <= 69:
        return "cool"
    if t <= 84:
        return "warm"
    return "hot"


def wind_category(game) -> str:
    """Return the weather wind direction bucket for a game row.

    Parameters
    ----------
    game : dict-like
        Game row with a ``weather_wind_direction`` field (populated by
        ``fetch_todays_schedule`` via ``parse_wind_string``).

    Returns
    -------
    str
        One of "in", "out", "cross", "none", or "unknown".
        Matches categories in production.dim_weather.
    """
    direction = game.get("weather_wind_direction") if hasattr(game, "get") else None
    if direction is None:
        return "unknown"
    if isinstance(direction, float) and (math.isnan(direction) or math.isinf(direction)):
        return "unknown"
    direction = str(direction).strip().lower()
    if direction in ("in", "out", "cross", "none"):
        return direction
    return "unknown"
