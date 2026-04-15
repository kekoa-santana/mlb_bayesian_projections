"""Hitter pitch-type profiles, season totals, observed profiles, aggressiveness, zone grids."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.data.queries._common import _WOBA_WEIGHTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hitter pitch-type profile
# ---------------------------------------------------------------------------
def get_hitter_pitch_type_profile(season: int) -> pd.DataFrame:
    """Per batter, per pitch type aggregations for a season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_stand, pitch_type, pitches, swings, whiffs,
        chase_pitches, chase_swings, called_strikes, csw,
        bip, barrels_proxy, xwoba_contact, hard_hits.
    """
    query = """
    WITH pitch_agg AS (
        SELECT
            fp.batter_id,
            fp.batter_stand,
            fp.pitch_type,
            COUNT(*)                                          AS pitches,
            SUM(fp.is_swing::int)                             AS swings,
            SUM(fp.is_whiff::int)                             AS whiffs,
            SUM(CASE WHEN fp.zone NOT IN (1,2,3,4,5,6,7,8,9) THEN 1 ELSE 0 END)
                                                              AS out_of_zone_pitches,
            SUM(CASE WHEN fp.zone NOT IN (1,2,3,4,5,6,7,8,9) AND fp.is_swing THEN 1 ELSE 0 END)
                                                              AS chase_swings,
            SUM(fp.is_called_strike::int)                     AS called_strikes,
            SUM(CASE WHEN fp.is_whiff OR fp.is_called_strike THEN 1 ELSE 0 END)
                                                              AS csw,
            SUM(fp.is_bip::int)                               AS bip
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.batter_id, fp.batter_stand, fp.pitch_type
    ),
    batted_agg AS (
        SELECT
            fp.batter_id,
            fp.pitch_type,
            COUNT(*)                                          AS contacts,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)    AS hard_hits,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END)
                                                              AS xwoba_contact,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)                      AS barrels_proxy
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.is_bip
        GROUP BY fp.batter_id, fp.pitch_type
    )
    SELECT
        pa.batter_id,
        pa.batter_stand,
        pa.pitch_type,
        pa.pitches,
        pa.swings,
        pa.whiffs,
        pa.out_of_zone_pitches,
        pa.chase_swings,
        pa.called_strikes,
        pa.csw,
        pa.bip,
        COALESCE(ba.hard_hits, 0)        AS hard_hits,
        ba.xwoba_contact,
        COALESCE(ba.barrels_proxy, 0)    AS barrels_proxy
    FROM pitch_agg pa
    LEFT JOIN batted_agg ba
        ON pa.batter_id = ba.batter_id
       AND pa.pitch_type = ba.pitch_type
    ORDER BY pa.batter_id, pa.pitch_type
    """
    logger.info("Fetching hitter pitch-type profiles for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Season totals (player season lines)
# ---------------------------------------------------------------------------
def get_season_totals(season: int) -> pd.DataFrame:
    """Per-player season lines: PA, K, BB, barrel%, xwOBA, wOBA, etc.

    Combines plate-appearance events with batted-ball quality metrics.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, season, pa, k, bb,
        ibb, hbp, sf, hits, hr, xwoba_avg, barrel_pct, hard_hit_pct,
        k_rate, bb_rate, woba.
    """
    query = """
    WITH stand_agg AS (
        -- Get batter_stand from fact_pitch (not in fact_pa or dim_player).
        -- dim_player only has ~2025 active players so an INNER JOIN would
        -- silently drop the majority of historical batters.
        SELECT
            fp.batter_id,
            MODE() WITHIN GROUP (ORDER BY fp.batter_stand) AS batter_stand
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season    = :season
          AND dg.game_type = 'R'
          AND fp.batter_stand IS NOT NULL
        GROUP BY fp.batter_id
    ),
    pa_agg AS (
        SELECT
            fpa.batter_id,
            dp.player_name                                           AS batter_name,
            sa.batter_stand,
            dg.season,
            COUNT(*)                                                 AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END)                             AS k,
            SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                     THEN 1 ELSE 0 END)                             AS bb,
            SUM(CASE WHEN fpa.events = 'intent_walk'
                     THEN 1 ELSE 0 END)                             AS ibb,
            SUM(CASE WHEN fpa.events = 'hit_by_pitch'
                     THEN 1 ELSE 0 END)                             AS hbp,
            SUM(CASE WHEN fpa.events IN ('sac_fly','sac_fly_double_play')
                     THEN 1 ELSE 0 END)                             AS sf,
            SUM(CASE WHEN fpa.events IN ('single','double','triple','home_run')
                     THEN 1 ELSE 0 END)                             AS hits,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END)                             AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg    ON fpa.game_pk  = dg.game_pk
        LEFT JOIN production.dim_player dp ON fpa.batter_id = dp.player_id
        LEFT JOIN stand_agg sa         ON fpa.batter_id = sa.batter_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
        GROUP BY fpa.batter_id, dp.player_name, sa.batter_stand, dg.season
    ),
    batted_agg AS (
        SELECT
            fpa.batter_id,
            COUNT(*)                                                 AS bip,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END)   AS xwoba_avg,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS barrel_pct,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS hard_hit_pct,
            SUM(sbb.woba_value)                                      AS woba_bip_sum
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND sbb.woba_value IS NOT NULL
        GROUP BY fpa.batter_id
    )
    SELECT
        pa.batter_id,
        pa.batter_name,
        pa.batter_stand,
        pa.season,
        pa.pa,
        pa.k,
        pa.bb,
        pa.ibb,
        pa.hbp,
        pa.sf,
        pa.hits,
        pa.hr,
        ROUND(ba.xwoba_avg::numeric, 3)     AS xwoba_avg,
        ROUND(ba.barrel_pct::numeric, 4)     AS barrel_pct,
        ROUND(ba.hard_hit_pct::numeric, 4)   AS hard_hit_pct,
        ba.woba_bip_sum,
        ROUND((pa.k::numeric / pa.pa), 4)    AS k_rate,
        ROUND((pa.bb::numeric / pa.pa), 4)   AS bb_rate
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.pa >= 1
    ORDER BY pa.pa DESC
    """
    logger.info("Fetching season totals for %d", season)
    df = read_sql(query, {"season": season})

    # Compute wOBA: BIP contribution + non-BIP weights
    non_ibb_bb = df["bb"] - df["ibb"]
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp"] * _WOBA_WEIGHTS["hbp"]
    )
    woba_den = (df["pa"] - df["ibb"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).round(3)

    # Drop intermediate column
    df.drop(columns=["woba_bip_sum"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Season totals split by pitcher hand (for platoon model)
# ---------------------------------------------------------------------------
def get_season_totals_by_pitcher_hand(season: int) -> pd.DataFrame:
    """Per (batter_id, pitch_hand) season K/PA with same_side flag.

    Batted-ball quality metrics (barrel_pct, hard_hit_pct) are NOT split by
    pitch_hand -- contact quality is stable across pitcher handedness, and
    splitting would create dangerously thin samples.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, pitch_hand, season,
        pa, k, bb, hits, hr, xwoba_avg, barrel_pct, hard_hit_pct,
        k_rate, bb_rate, same_side.
    """
    query = """
    WITH stand_agg AS (
        -- Get batter_stand per PA from fact_pitch (actual side chosen).
        SELECT DISTINCT ON (fp.pa_id)
            fp.pa_id,
            fp.batter_stand
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season    = :season
          AND dg.game_type = 'R'
          AND fp.batter_stand IS NOT NULL
        ORDER BY fp.pa_id, fp.pitch_number
    ),
    pa_agg AS (
        SELECT
            fpa.batter_id,
            dp_b.player_name                                         AS batter_name,
            sa.batter_stand,
            dp_p.pitch_hand,
            dg.season,
            COUNT(*)                                                 AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END)                             AS k,
            SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                     THEN 1 ELSE 0 END)                             AS bb,
            SUM(CASE WHEN fpa.events IN ('single','double','triple','home_run')
                     THEN 1 ELSE 0 END)                             AS hits,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END)                             AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg    ON fpa.game_pk  = dg.game_pk
        LEFT JOIN production.dim_player dp_b ON fpa.batter_id  = dp_b.player_id
        LEFT JOIN production.dim_player dp_p ON fpa.pitcher_id = dp_p.player_id
        LEFT JOIN stand_agg sa               ON fpa.pa_id      = sa.pa_id
        WHERE dg.season    = :season
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
          AND dp_p.pitch_hand IN ('L', 'R')
        GROUP BY fpa.batter_id, dp_b.player_name, sa.batter_stand,
                 dp_p.pitch_hand, dg.season
    ),
    batted_agg AS (
        -- Batted-ball quality at player-season level (NOT split by pitch_hand)
        SELECT
            fpa.batter_id,
            COUNT(*)                                                 AS bip,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END)   AS xwoba_avg,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS barrel_pct,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS hard_hit_pct
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
        GROUP BY fpa.batter_id
    )
    SELECT
        pa.batter_id,
        pa.batter_name,
        pa.batter_stand,
        pa.pitch_hand,
        pa.season,
        pa.pa,
        pa.k,
        pa.bb,
        pa.hits,
        pa.hr,
        ROUND(ba.xwoba_avg::numeric, 3)     AS xwoba_avg,
        ROUND(ba.barrel_pct::numeric, 4)     AS barrel_pct,
        ROUND(ba.hard_hit_pct::numeric, 4)   AS hard_hit_pct,
        ROUND((pa.k::numeric / pa.pa), 4)    AS k_rate,
        ROUND((pa.bb::numeric / pa.pa), 4)   AS bb_rate,
        CASE WHEN pa.batter_stand = pa.pitch_hand THEN 1 ELSE 0 END AS same_side
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.pa >= 1
    ORDER BY pa.pa DESC
    """
    logger.info("Fetching season totals by pitcher hand for %d", season)
    return read_sql(query, {"season": season})


def get_league_platoon_baselines(
    seasons: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute league-average platoon K% and BB% deltas (logit-scale).

    Pools PA across *seasons* and computes the difference between
    same-side / opposite-side rates and overall rates in logit space.

    Parameters
    ----------
    seasons : list[int] | None
        Seasons to pool.  Defaults to [2022, 2023, 2024, 2025].

    Returns
    -------
    dict
        ``{"platoon_k_logit": {"same": float, "opposite": float},
           "platoon_bb_logit": {"same": float, "opposite": float},
           "lg_k_rate": float, "lg_bb_rate": float,
           "lg_gb_rate": float, "lg_fb_rate": float}``
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    query = """
    WITH stand_agg AS (
        SELECT DISTINCT ON (fp.pa_id)
            fp.pa_id,
            fp.batter_stand
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season    = ANY(:seasons)
          AND dg.game_type = 'R'
          AND fp.batter_stand IS NOT NULL
        ORDER BY fp.pa_id, fp.pitch_number
    ),
    pa_level AS (
        SELECT
            sa.batter_stand,
            dp_p.pitch_hand,
            CASE WHEN sa.batter_stand = dp_p.pitch_hand
                 THEN 'same' ELSE 'opposite' END          AS platoon_side,
            CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                 THEN 1 ELSE 0 END                        AS is_k,
            CASE WHEN fpa.events IN ('walk','intent_walk')
                 THEN 1 ELSE 0 END                        AS is_bb
        FROM production.fact_pa fpa
        JOIN production.dim_game dg    ON fpa.game_pk  = dg.game_pk
        LEFT JOIN production.dim_player dp_p ON fpa.pitcher_id = dp_p.player_id
        LEFT JOIN stand_agg sa               ON fpa.pa_id      = sa.pa_id
        WHERE dg.season    = ANY(:seasons)
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
          AND dp_p.pitch_hand IN ('L', 'R')
    )
    SELECT
        platoon_side,
        COUNT(*)                                   AS pa,
        SUM(is_k)::float / COUNT(*)                AS k_rate,
        SUM(is_bb)::float / COUNT(*)               AS bb_rate
    FROM pa_level
    GROUP BY platoon_side
    ORDER BY platoon_side
    """
    df = read_sql(query, {"seasons": seasons})

    # Also get league-average GB% and FB% from batted balls
    bb_query = """
    SELECT
        COUNT(*)                                                              AS total_bip,
        SUM(CASE WHEN sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                             AS lg_gb_rate,
        SUM(CASE WHEN sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                             AS lg_fb_rate
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
    WHERE dg.season    = ANY(:seasons)
      AND dg.game_type = 'R'
      AND sbb.launch_angle IS NOT NULL
    """
    bb_df = read_sql(bb_query, {"seasons": seasons})

    def _logit(p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return float(np.log(p / (1.0 - p)))

    overall_k = float(df["k_rate"].mean())  # simple avg of same/opposite
    overall_bb = float(df["bb_rate"].mean())

    # Weight by PA for true overall
    total_pa = df["pa"].sum()
    overall_k = float((df["k_rate"] * df["pa"]).sum() / total_pa)
    overall_bb = float((df["bb_rate"] * df["pa"]).sum() / total_pa)

    result: dict[str, dict[str, float] | float] = {
        "platoon_k_logit": {},
        "platoon_bb_logit": {},
        "lg_k_rate": overall_k,
        "lg_bb_rate": overall_bb,
    }
    for _, row in df.iterrows():
        side = row["platoon_side"]
        result["platoon_k_logit"][side] = _logit(row["k_rate"]) - _logit(overall_k)
        result["platoon_bb_logit"][side] = _logit(row["bb_rate"]) - _logit(overall_bb)

    if not bb_df.empty:
        result["lg_gb_rate"] = float(bb_df["lg_gb_rate"].iloc[0])
        result["lg_fb_rate"] = float(bb_df["lg_fb_rate"].iloc[0])
    else:
        result["lg_gb_rate"] = 0.446
        result["lg_fb_rate"] = 0.321

    logger.info(
        "League platoon baselines (seasons %s): K logit same=%.3f opp=%.3f, "
        "BB logit same=%.3f opp=%.3f, GB=%.3f FB=%.3f",
        seasons,
        result["platoon_k_logit"].get("same", 0),
        result["platoon_k_logit"].get("opposite", 0),
        result["platoon_bb_logit"].get("same", 0),
        result["platoon_bb_logit"].get("opposite", 0),
        result["lg_gb_rate"],
        result["lg_fb_rate"],
    )
    return result


def get_season_totals_with_age(season: int) -> pd.DataFrame:
    """Per-player season lines enriched with age and age_bucket.

    Age is computed as of July 1 of the season (midpoint).

    Returns
    -------
    pd.DataFrame
        Same columns as ``get_season_totals`` plus: birth_date, age, age_bucket.
        age_bucket: 0 = young (<=25), 1 = prime (26-30), 2 = veteran (31+).
    """
    query = """
    WITH stand_agg AS (
        SELECT
            fp.batter_id,
            MODE() WITHIN GROUP (ORDER BY fp.batter_stand) AS batter_stand
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season    = :season
          AND dg.game_type = 'R'
          AND fp.batter_stand IS NOT NULL
        GROUP BY fp.batter_id
    ),
    pa_agg AS (
        SELECT
            fpa.batter_id,
            dp.player_name                                           AS batter_name,
            sa.batter_stand,
            dg.season,
            dp.birth_date,
            COUNT(*)                                                 AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END)                             AS k,
            SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                     THEN 1 ELSE 0 END)                             AS bb,
            SUM(CASE WHEN fpa.events = 'intent_walk'
                     THEN 1 ELSE 0 END)                             AS ibb,
            SUM(CASE WHEN fpa.events = 'hit_by_pitch'
                     THEN 1 ELSE 0 END)                             AS hbp,
            SUM(CASE WHEN fpa.events IN ('sac_fly','sac_fly_double_play')
                     THEN 1 ELSE 0 END)                             AS sf,
            SUM(CASE WHEN fpa.events IN ('single','double','triple','home_run')
                     THEN 1 ELSE 0 END)                             AS hits,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END)                             AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg    ON fpa.game_pk  = dg.game_pk
        LEFT JOIN production.dim_player dp ON fpa.batter_id = dp.player_id
        LEFT JOIN stand_agg sa         ON fpa.batter_id = sa.batter_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
        GROUP BY fpa.batter_id, dp.player_name, sa.batter_stand,
                 dg.season, dp.birth_date
    ),
    batted_agg AS (
        SELECT
            fpa.batter_id,
            COUNT(*)                                                 AS bip,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END)   AS xwoba_avg,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS barrel_pct,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)
                                                                     AS hard_hit_pct,
            SUM(sbb.woba_value)                                      AS woba_bip_sum,
            -- Batted ball type counts (for projection models)
            SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle < 10
                     THEN 1 ELSE 0 END)                             AS gb,
            SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle > 25
                     THEN 1 ELSE 0 END)                             AS fb,
            SUM(CASE WHEN sbb.launch_angle != 'NaN'
                     THEN 1 ELSE 0 END)                             AS bip_with_la,
            -- HR on fly balls
            SUM(CASE WHEN fpa.events = 'home_run'
                      AND sbb.launch_angle != 'NaN'
                      AND sbb.launch_angle > 25
                     THEN 1 ELSE 0 END)                             AS hr_fb
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND sbb.woba_value IS NOT NULL
        GROUP BY fpa.batter_id
    )
    SELECT
        pa.batter_id,
        pa.batter_name,
        pa.batter_stand,
        pa.season,
        pa.birth_date,
        EXTRACT(YEAR FROM AGE(DATE(pa.season || '-07-01'), pa.birth_date))::int AS age,
        pa.pa,
        pa.k,
        pa.bb,
        pa.ibb,
        pa.hbp,
        pa.sf,
        pa.hits,
        pa.hr,
        ROUND(ba.xwoba_avg::numeric, 3)     AS xwoba_avg,
        ROUND(ba.barrel_pct::numeric, 4)     AS barrel_pct,
        ROUND(ba.hard_hit_pct::numeric, 4)   AS hard_hit_pct,
        ba.woba_bip_sum,
        ba.bip,
        ba.gb,
        ba.fb,
        ba.bip_with_la,
        ba.hr_fb,
        ROUND((pa.k::numeric / pa.pa), 4)    AS k_rate,
        ROUND((pa.bb::numeric / pa.pa), 4)   AS bb_rate,
        ROUND((pa.hr::numeric / pa.pa), 4)   AS hr_rate,
        ROUND((ba.gb::numeric / NULLIF(ba.bip_with_la, 0)), 4)   AS gb_rate,
        ROUND((ba.fb::numeric / NULLIF(ba.bip_with_la, 0)), 4)   AS fb_rate,
        ROUND((ba.hr_fb::numeric / NULLIF(ba.fb, 0)), 4)         AS hr_per_fb
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.pa >= 1
    ORDER BY pa.pa DESC
    """
    logger.info("Fetching season totals with age for %d", season)
    df = read_sql(query, {"season": season})

    # Compute wOBA: BIP contribution + non-BIP weights
    non_ibb_bb = df["bb"] - df["ibb"]
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp"] * _WOBA_WEIGHTS["hbp"]
    )
    woba_den = (df["pa"] - df["ibb"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).round(3)

    # Drop intermediate column
    df.drop(columns=["woba_bip_sum"], inplace=True)

    # Compute age_bucket: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 28, 32, 99],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Hitter observed profile (pitch-level aggregates)
# ---------------------------------------------------------------------------
def get_hitter_observed_profile(season: int) -> pd.DataFrame:
    """Per-batter pitch-level and batted-ball aggregates for composite scoring.

    Returns whiff rate, chase rate, zone contact %, avg exit velocity,
    fly ball %, and hard-hit % -- all aggregated to the batter level.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, whiff_rate, chase_rate, z_contact_pct,
        avg_exit_velo, fb_pct, hard_hit_pct, bip.
    """
    query = """
    WITH pitch_agg AS (
        SELECT
            fp.batter_id,
            SUM(fp.is_swing::int)                           AS swings,
            SUM(fp.is_whiff::int)                           AS whiffs,
            -- Zone: zones 1-9
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                     THEN fp.is_swing::int ELSE 0 END)     AS z_swings,
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                     THEN fp.is_whiff::int ELSE 0 END)     AS z_whiffs,
            -- Out of zone: zone > 9 or zone IS NULL
            SUM(CASE WHEN fp.zone > 9 OR fp.zone IS NULL
                     THEN 1 ELSE 0 END)                    AS ooz_pitches,
            SUM(CASE WHEN (fp.zone > 9 OR fp.zone IS NULL)
                      AND fp.is_swing
                     THEN 1 ELSE 0 END)                    AS chase_swings
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.pitch_type NOT IN ('PO', 'UN', 'SC', 'FA')
        GROUP BY fp.batter_id
    ),
    batted_agg AS (
        SELECT
            fpa.batter_id,
            COUNT(*)                                                    AS bip,
            AVG(sbb.launch_speed)                                       AS avg_exit_velo,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0)                                   AS hard_hit_pct,
            SUM(CASE WHEN sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0)                                   AS fb_pct
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND sbb.launch_speed != 'NaN'
        GROUP BY fpa.batter_id
    )
    SELECT
        pa.batter_id,
        ROUND((pa.whiffs::numeric / NULLIF(pa.swings, 0)), 4)     AS whiff_rate,
        ROUND((pa.chase_swings::numeric / NULLIF(pa.ooz_pitches, 0)), 4)
                                                                    AS chase_rate,
        ROUND(((pa.z_swings - pa.z_whiffs)::numeric
               / NULLIF(pa.z_swings, 0)), 4)                       AS z_contact_pct,
        ROUND(ba.avg_exit_velo::numeric, 1)                         AS avg_exit_velo,
        ROUND(ba.fb_pct::numeric, 4)                                AS fb_pct,
        ROUND(ba.hard_hit_pct::numeric, 4)                          AS hard_hit_pct,
        ba.bip,
        pa.chase_swings,
        pa.ooz_pitches                                              AS out_of_zone_pitches
    FROM pitch_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.swings >= 50
    ORDER BY pa.swings DESC
    """
    logger.info("Fetching hitter observed profile for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Hitter BIP profile (for batter-specific BIP outcome resolution in game sim)
# ---------------------------------------------------------------------------
def get_batter_bip_profile(seasons: list[int]) -> pd.DataFrame:
    """Per-batter BIP quality metrics and observed BIP outcome splits.

    Used by the game sim to resolve batted-in-play outcomes into
    out/single/double/triple using batter-specific probabilities rather
    than league averages.

    Parameters
    ----------
    seasons : list[int]
        One or more MLB seasons.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, bip, bip_outs, bip_singles,
        bip_doubles, bip_triples, avg_ev, avg_la, gb_pct.
        Restricted to batters with >= 25 BIP.
    """
    query = """
    SELECT
        fpa.batter_id,
        dg.season,
        COUNT(*)                                                           AS bip,
        SUM(CASE WHEN fpa.events IN (
                'field_out','force_out','grounded_into_double_play',
                'fielders_choice','fielders_choice_out','sac_fly',
                'sac_fly_double_play','double_play','triple_play',
                'sac_bunt','sac_bunt_double_play','field_error')
                 THEN 1 ELSE 0 END)                                        AS bip_outs,
        SUM(CASE WHEN fpa.events = 'single' THEN 1 ELSE 0 END)             AS bip_singles,
        SUM(CASE WHEN fpa.events = 'double' THEN 1 ELSE 0 END)             AS bip_doubles,
        SUM(CASE WHEN fpa.events = 'triple' THEN 1 ELSE 0 END)             AS bip_triples,
        AVG(sbb.launch_speed)                                              AS avg_ev,
        AVG(sbb.launch_angle)                                              AS avg_la,
        SUM(CASE WHEN sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                          AS gb_pct
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
    WHERE dg.season = ANY(:seasons)
      AND dg.game_type = 'R'
      AND sbb.launch_speed IS NOT NULL
      AND sbb.launch_speed != 'NaN'
      AND sbb.launch_angle IS NOT NULL
      AND sbb.launch_angle != 'NaN'
      AND fpa.events IS NOT NULL
      AND fpa.events NOT IN ('home_run','strikeout','strikeout_double_play',
                             'walk','intent_walk','hit_by_pitch','catcher_interf')
    GROUP BY fpa.batter_id, dg.season
    HAVING COUNT(*) >= 25
    ORDER BY dg.season, COUNT(*) DESC
    """
    logger.info("Fetching batter BIP profile for seasons %s", seasons)
    return read_sql(query, {"seasons": seasons})


# ---------------------------------------------------------------------------
# Hitter aggressiveness profile
# ---------------------------------------------------------------------------
def get_hitter_aggressiveness(season: int) -> pd.DataFrame:
    """Per-batter approach / aggressiveness metrics for a season.

    Combines pitch-level aggression indicators (first-pitch swing rate,
    chase rate, two-strike discipline) with plate-appearance-level depth
    (pitches per PA).

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, season, pa,
        first_pitch_swing_pct, first_pitch_contact_pct,
        chase_rate, two_strike_chase_rate, two_strike_whiff_rate,
        zone_swing_pct, pitches_per_pa.
    """
    query = """
    WITH pa_counts AS (
        SELECT
            fpa.batter_id,
            COUNT(DISTINCT fpa.pa_id) AS pa,
            ROUND(AVG(fpa.last_pitch_number)::numeric, 2) AS pitches_per_pa
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY fpa.batter_id
    ),
    first_pitches AS (
        SELECT
            fp.batter_id,
            COUNT(*)                        AS total_first,
            SUM(fp.is_swing::int)           AS first_swings,
            SUM(CASE WHEN fp.is_swing AND fp.is_bip THEN 1 ELSE 0 END) AS first_contact
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_number = 1 AND fp.balls = 0 AND fp.strikes = 0
        GROUP BY fp.batter_id
    ),
    chase AS (
        SELECT
            fp.batter_id,
            COUNT(*)                AS ooz_pitches,
            SUM(fp.is_swing::int)   AS ooz_swings
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.zone IN (11, 12, 13, 14)
        GROUP BY fp.batter_id
    ),
    two_strike AS (
        SELECT
            fp.batter_id,
            SUM(CASE WHEN fp.zone IN (11, 12, 13, 14) THEN 1 ELSE 0 END)          AS ts_ooz_pitches,
            SUM(CASE WHEN fp.zone IN (11, 12, 13, 14) AND fp.is_swing THEN 1 ELSE 0 END) AS ts_ooz_swings,
            SUM(fp.is_swing::int)       AS ts_swings,
            SUM(fp.is_whiff::int)       AS ts_whiffs
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.strikes = 2
        GROUP BY fp.batter_id
    ),
    zone_agg AS (
        SELECT
            fp.batter_id,
            COUNT(*)                AS zone_pitches,
            SUM(fp.is_swing::int)   AS zone_swings
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.zone BETWEEN 1 AND 9
        GROUP BY fp.batter_id
    )
    SELECT
        pc.batter_id,
        COALESCE(dp.player_name, 'Unknown') AS batter_name,
        :season AS season,
        pc.pa,
        ROUND((fpch.first_swings::numeric / NULLIF(fpch.total_first, 0)), 3) AS first_pitch_swing_pct,
        ROUND((fpch.first_contact::numeric / NULLIF(fpch.first_swings, 0)), 3) AS first_pitch_contact_pct,
        ROUND((ch.ooz_swings::numeric / NULLIF(ch.ooz_pitches, 0)), 3) AS chase_rate,
        ROUND((ts.ts_ooz_swings::numeric / NULLIF(ts.ts_ooz_pitches, 0)), 3) AS two_strike_chase_rate,
        ROUND((ts.ts_whiffs::numeric / NULLIF(ts.ts_swings, 0)), 3) AS two_strike_whiff_rate,
        ROUND((za.zone_swings::numeric / NULLIF(za.zone_pitches, 0)), 3) AS zone_swing_pct,
        pc.pitches_per_pa
    FROM pa_counts pc
    LEFT JOIN production.dim_player dp ON pc.batter_id = dp.player_id
    LEFT JOIN first_pitches fpch ON pc.batter_id = fpch.batter_id
    LEFT JOIN chase ch ON pc.batter_id = ch.batter_id
    LEFT JOIN two_strike ts ON pc.batter_id = ts.batter_id
    LEFT JOIN zone_agg za ON pc.batter_id = za.batter_id
    ORDER BY pc.pa DESC
    """
    logger.info("Fetching hitter aggressiveness for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Hitter zone grid
# ---------------------------------------------------------------------------
def get_hitter_zone_grid(season: int) -> pd.DataFrame:
    """Hitter zone grid with whiff and batted-ball metrics per cell.

    Returns one row per (batter_id, batter_stand, grid_row, grid_col)
    with pitch/swing/whiff counts and batted-ball quality metrics.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, grid_row, grid_col,
        pitches, swings, whiffs, called_strikes, bip, xwoba_sum, xwoba_count,
        hard_hits, barrels.
    """
    query = """
    WITH batter_filter AS (
        SELECT fp.batter_id
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.batter_id
        HAVING COUNT(*) >= 200
    ),
    pitch_grid AS (
        SELECT
            fp.batter_id,
            fp.batter_stand,
            fp.pa_id,
            LEAST(GREATEST(FLOOR((fp.plate_x + 1.33) / 0.532)::int, 0), 4) AS grid_col,
            LEAST(GREATEST(FLOOR((fp.plate_z - 1.0) / 0.6)::int, 0), 4) AS grid_row,
            fp.pitch_type,
            fp.is_swing,
            fp.is_whiff,
            fp.is_called_strike,
            fp.is_bip
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN batter_filter bf ON fp.batter_id = bf.batter_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.plate_x IS NOT NULL AND fp.plate_z IS NOT NULL
          AND CAST(fp.plate_x AS text) != 'NaN'
          AND CAST(fp.plate_z AS text) != 'NaN'
    )
    SELECT
        pg.batter_id,
        COALESCE(dp.player_name, 'Unknown') AS batter_name,
        pg.batter_stand,
        pg.pitch_type,
        pg.grid_row,
        pg.grid_col,
        COUNT(*) AS pitches,
        SUM(pg.is_swing::int) AS swings,
        SUM(pg.is_whiff::int) AS whiffs,
        SUM(pg.is_called_strike::int) AS called_strikes,
        SUM(pg.is_bip::int) AS bip,
        COALESCE(SUM(
            CASE WHEN pg.is_bip AND sbb.xwoba IS NOT NULL
                      AND CAST(sbb.xwoba AS text) != 'NaN'
                 THEN sbb.xwoba ELSE 0 END
        ), 0) AS xwoba_sum,
        COALESCE(SUM(
            CASE WHEN pg.is_bip AND sbb.xwoba IS NOT NULL
                      AND CAST(sbb.xwoba AS text) != 'NaN'
                 THEN 1 ELSE 0 END
        ), 0) AS xwoba_count,
        COALESCE(SUM(
            CASE WHEN pg.is_bip AND sbb.launch_speed >= 95 THEN 1 ELSE 0 END
        ), 0) AS hard_hits,
        COALESCE(SUM(
            CASE WHEN pg.is_bip AND sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                 THEN 1 ELSE 0 END
        ), 0) AS barrels
    FROM pitch_grid pg
    LEFT JOIN production.dim_player dp ON pg.batter_id = dp.player_id
    LEFT JOIN production.sat_batted_balls sbb ON pg.pa_id = sbb.pa_id
    GROUP BY pg.batter_id, dp.player_name, pg.batter_stand,
             pg.pitch_type, pg.grid_row, pg.grid_col
    ORDER BY pg.batter_id, pg.pitch_type, pg.grid_row, pg.grid_col
    """
    logger.info("Fetching hitter zone grid for %d", season)
    return read_sql(query, {"season": season})
