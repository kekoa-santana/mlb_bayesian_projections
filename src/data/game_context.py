"""
Game-level context data for the PA-by-PA simulator.

Provides park factor logit lifts (K, BB, HR, H) and catcher framing
effects that feed directly into simulate_game() and simulate_batter_game().
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"

# Shrinkage for park factors (games needed for full weight)
_PARK_SHRINKAGE_GAMES = 200


def get_park_logit_lifts(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-venue logit-scale lifts for K, BB, HR.

    These plug directly into the game sim's logit-additive model:
    ``park_k_lift``, ``park_hr_lift``, etc.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to compute from. Defaults to last 3.

    Returns
    -------
    pd.DataFrame
        Columns: venue_id, park_k_lift, park_bb_lift, park_hr_lift, games.
        Lifts are on logit scale (positive = more of that stat at this park).
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]
    season_list = ", ".join(str(s) for s in seasons)

    raw = read_sql(f"""
        WITH venue_stats AS (
            SELECT dg.venue_id,
                   CASE WHEN fg.team_id = dg.home_team_id THEN 'home' ELSE 'away' END as loc,
                   SUM(fg.pit_k)::float / NULLIF(SUM(fg.pit_bf), 0) as k_rate,
                   SUM(fg.pit_bb)::float / NULLIF(SUM(fg.pit_bf), 0) as bb_rate,
                   SUM(fg.pit_hr)::float / NULLIF(SUM(fg.pit_bf), 0) as hr_rate,
                   COUNT(DISTINCT dg.game_pk) as games
            FROM production.fact_player_game_mlb fg
            JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
            WHERE fg.player_role = 'pitcher' AND dg.game_type = 'R'
                  AND dg.season IN ({season_list})
            GROUP BY dg.venue_id,
                     CASE WHEN fg.team_id = dg.home_team_id THEN 'home' ELSE 'away' END
        )
        SELECT h.venue_id,
               h.games,
               h.k_rate as home_k, a.k_rate as away_k,
               h.bb_rate as home_bb, a.bb_rate as away_bb,
               h.hr_rate as home_hr, a.hr_rate as away_hr
        FROM venue_stats h
        JOIN venue_stats a ON h.venue_id = a.venue_id AND h.loc = 'home' AND a.loc = 'away'
        WHERE h.games >= 50
    """, {})

    if raw.empty:
        return pd.DataFrame()

    def _safe_logit(p: float) -> float:
        p = np.clip(p, 0.001, 0.999)
        return float(np.log(p / (1 - p)))

    reliability = (raw["games"] / _PARK_SHRINKAGE_GAMES).clip(0, 1)

    for stat in ["k", "bb", "hr"]:
        home = raw[f"home_{stat}"]
        away = raw[f"away_{stat}"]
        raw_lift = home.apply(_safe_logit) - away.apply(_safe_logit)
        # Regress toward 0 (neutral)
        raw[f"park_{stat}_lift"] = (reliability * raw_lift).round(4)

    result = raw[["venue_id", "park_k_lift", "park_bb_lift", "park_hr_lift", "games"]].copy()
    logger.info("Park logit lifts: %d venues", len(result))
    return result


def get_catcher_framing_lifts(
    season: int | None = None,
    min_games: int = 20,
) -> pd.DataFrame:
    """Get catcher framing K-rate logit lifts.

    Converts ``strike_rate_diff`` from fact_catcher_framing into a
    logit-scale K lift that can be added to the pitcher's K logit
    in the game sim.

    Parameters
    ----------
    season : int, optional
        Season for framing data. Defaults to most recent.
    min_games : int
        Minimum games caught.

    Returns
    -------
    pd.DataFrame
        Columns: catcher_id, catcher_name, framing_k_lift, runs_extra_strikes.
    """
    season_filter = f"AND cf.season = {season}" if season else ""

    framing = read_sql(f"""
        SELECT cf.player_id as catcher_id,
               dp.player_name as catcher_name,
               cf.strike_rate_diff,
               cf.runs_extra_strikes,
               cf.shadow_zone_strike_rate
        FROM production.fact_catcher_framing cf
        JOIN production.dim_player dp ON cf.player_id = dp.player_id
        WHERE 1=1 {season_filter}
    """, {})

    if framing.empty:
        return pd.DataFrame()

    # Convert strike_rate_diff to K logit lift
    # strike_rate_diff = catcher's called strike rate - league avg
    # On ~35% of pitches that are borderline (shadow zone), this shifts
    # the probability of a called strike. Net effect on K rate:
    # ~35% of PAs have a borderline pitch, and the catcher shifts
    # the called strike prob by strike_rate_diff on those pitches.
    # Approximate K logit lift ≈ strike_rate_diff * 2.5 (empirical scaling)
    framing["framing_k_lift"] = (framing["strike_rate_diff"] * 2.5).round(4)

    # Shrinkage: regress toward 0 based on games caught
    # (runs_extra_strikes magnitude is a proxy for sample size)
    abs_res = framing["runs_extra_strikes"].abs()
    reliability = (abs_res / abs_res.quantile(0.75)).clip(0, 1)
    framing["framing_k_lift"] = framing["framing_k_lift"] * reliability

    logger.info(
        "Catcher framing: %d catchers, K lift range [%.4f, %.4f]",
        len(framing), framing["framing_k_lift"].min(), framing["framing_k_lift"].max(),
    )

    return framing[["catcher_id", "catcher_name", "framing_k_lift", "runs_extra_strikes"]]


def get_game_context(
    venue_id: int,
    catcher_id: int | None = None,
    park_lifts: pd.DataFrame | None = None,
    catcher_lifts: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Get combined context lifts for a specific game.

    Parameters
    ----------
    venue_id : int
        Game venue.
    catcher_id : int, optional
        Catcher behind the plate.
    park_lifts : pd.DataFrame, optional
        Pre-computed park lifts (avoids re-querying).
    catcher_lifts : pd.DataFrame, optional
        Pre-computed catcher lifts.

    Returns
    -------
    dict[str, float]
        Keys: park_k_lift, park_bb_lift, park_hr_lift, catcher_k_lift.
        All on logit scale, ready for simulate_game().
    """
    context = {
        "park_k_lift": 0.0,
        "park_bb_lift": 0.0,
        "park_hr_lift": 0.0,
        "catcher_k_lift": 0.0,
    }

    # Park lifts
    if park_lifts is not None and not park_lifts.empty:
        venue_row = park_lifts[park_lifts["venue_id"] == venue_id]
        if not venue_row.empty:
            r = venue_row.iloc[0]
            context["park_k_lift"] = float(r.get("park_k_lift", 0))
            context["park_bb_lift"] = float(r.get("park_bb_lift", 0))
            context["park_hr_lift"] = float(r.get("park_hr_lift", 0))

    # Catcher framing
    if catcher_id is not None and catcher_lifts is not None and not catcher_lifts.empty:
        catcher_row = catcher_lifts[catcher_lifts["catcher_id"] == catcher_id]
        if not catcher_row.empty:
            context["catcher_k_lift"] = float(catcher_row.iloc[0]["framing_k_lift"])

    return context
