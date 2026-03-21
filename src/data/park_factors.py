"""
Multi-stat park factors from game-level data.

Computes park factor indices for R, H, HR, K, BB by venue, using the
standard home/away ratio method with regression toward 1.0 for small
samples. A park factor of 1.10 means 10% more of that stat at that venue.

Uses 3-year rolling windows for stability.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# Shrinkage: regress toward 1.0 with this many games for full weight
_SHRINKAGE_GAMES = 200  # ~2.5 seasons of home games

# Stats to compute park factors for
PARK_FACTOR_STATS = ["r", "h", "hr", "k", "bb"]


def compute_multi_stat_park_factors(
    seasons: list[int] | None = None,
    min_games: int = 50,
) -> pd.DataFrame:
    """Compute park factors for R, H, HR, K, BB per venue.

    Uses home vs away comparison: PF = (home_rate / away_rate).
    Regressed toward 1.0 based on sample size.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to last 3 complete seasons.
    min_games : int
        Minimum home games for a venue to be included.

    Returns
    -------
    pd.DataFrame
        Columns: venue_id, venue_name, games, pf_r, pf_h, pf_hr, pf_k, pf_bb.
        All factors indexed to 1.0 = neutral.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    season_list = ", ".join(str(s) for s in seasons)

    raw = read_sql(f"""
        WITH game_team AS (
            SELECT dg.venue_id, dg.game_pk, dg.season,
                   fg.team_id,
                   CASE WHEN fg.team_id = dg.home_team_id THEN 'home'
                        ELSE 'away' END as loc,
                   SUM(fg.bat_h) as h, SUM(fg.bat_hr) as hr,
                   SUM(fg.bat_r) as r, SUM(fg.bat_k) as k,
                   SUM(fg.bat_bb) as bb, SUM(fg.bat_pa) as pa
            FROM production.fact_player_game_mlb fg
            JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
            WHERE fg.player_role = 'batter' AND dg.game_type = 'R'
                  AND dg.season IN ({season_list})
            GROUP BY dg.venue_id, dg.game_pk, dg.season,
                     fg.team_id, dg.home_team_id
        )
        SELECT venue_id,
               COUNT(DISTINCT game_pk) as games,
               -- Per-PA rates by location
               SUM(CASE WHEN loc='home' THEN r END)::float /
                   NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_r_rate,
               SUM(CASE WHEN loc='away' THEN r END)::float /
                   NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_r_rate,
               SUM(CASE WHEN loc='home' THEN h END)::float /
                   NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_h_rate,
               SUM(CASE WHEN loc='away' THEN h END)::float /
                   NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_h_rate,
               SUM(CASE WHEN loc='home' THEN hr END)::float /
                   NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_hr_rate,
               SUM(CASE WHEN loc='away' THEN hr END)::float /
                   NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_hr_rate,
               SUM(CASE WHEN loc='home' THEN k END)::float /
                   NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_k_rate,
               SUM(CASE WHEN loc='away' THEN k END)::float /
                   NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_k_rate,
               SUM(CASE WHEN loc='home' THEN bb END)::float /
                   NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_bb_rate,
               SUM(CASE WHEN loc='away' THEN bb END)::float /
                   NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_bb_rate
        FROM game_team
        GROUP BY venue_id
        HAVING COUNT(DISTINCT game_pk) >= {min_games}
    """, {})

    if raw.empty:
        return pd.DataFrame()

    # Compute raw park factors (home / away ratio)
    reliability = (raw["games"] / _SHRINKAGE_GAMES).clip(0, 1)

    for stat in PARK_FACTOR_STATS:
        home_col = f"home_{stat}_rate"
        away_col = f"away_{stat}_rate"
        raw_pf = raw[home_col] / raw[away_col].replace(0, np.nan)
        # Regress toward 1.0
        raw[f"pf_{stat}"] = reliability * raw_pf + (1 - reliability) * 1.0

    # Get venue names
    venues = read_sql("""
        SELECT DISTINCT venue_id,
               FIRST_VALUE(home_team_name) OVER (PARTITION BY venue_id ORDER BY game_date DESC) as venue_name
        FROM production.dim_game
        WHERE game_type = 'R' AND season >= 2023
    """, {})
    if not venues.empty:
        raw = raw.merge(venues.drop_duplicates("venue_id"), on="venue_id", how="left")

    result = raw[["venue_id"] + [f"pf_{s}" for s in PARK_FACTOR_STATS] + ["games"]].copy()
    if "venue_name" in raw.columns:
        result["venue_name"] = raw["venue_name"]

    # Round for readability
    for col in [f"pf_{s}" for s in PARK_FACTOR_STATS]:
        result[col] = result[col].round(3)

    logger.info(
        "Multi-stat park factors: %d venues, seasons %s",
        len(result), seasons,
    )
    return result


def get_player_park_adjustments(
    park_factors: pd.DataFrame,
    player_venues: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Compute per-player park adjustments (half-weight: ~half home, half road).

    Parameters
    ----------
    park_factors : pd.DataFrame
        From ``compute_multi_stat_park_factors``.
    player_venues : pd.DataFrame
        Columns: player_id, venue_id (home venue).

    Returns
    -------
    dict[int, dict[str, float]]
        player_id -> {pf_r, pf_h, pf_hr, pf_k, pf_bb}.
        Half-weighted: 0.5 * home_pf + 0.5 * 1.0.
    """
    if park_factors.empty or player_venues.empty:
        return {}

    merged = player_venues.merge(park_factors, on="venue_id", how="left")

    result: dict[int, dict[str, float]] = {}
    for _, row in merged.iterrows():
        pid = int(row["player_id"])
        adjs = {}
        for stat in PARK_FACTOR_STATS:
            pf = row.get(f"pf_{stat}", 1.0)
            if pd.isna(pf):
                pf = 1.0
            # Half-weight: half games at home, half on the road
            adjs[f"pf_{stat}"] = 0.5 * pf + 0.5 * 1.0
        result[pid] = adjs

    logger.info("Park adjustments for %d players", len(result))
    return result
