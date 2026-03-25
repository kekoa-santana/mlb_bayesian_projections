"""
Core SQL queries for the Bayesian projection system.

Every function returns a pandas DataFrame.  All SQL lives here —
no raw query strings elsewhere in the codebase.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Pitch-level data
# ---------------------------------------------------------------------------
def get_pitch_level_data(season: int) -> pd.DataFrame:
    """All pitches for a season with batter/pitcher IDs, pitch type,
    movement, velocity, location, and outcome flags.

    Parameters
    ----------
    season : int
        MLB season year (e.g. 2024).

    Returns
    -------
    pd.DataFrame
        One row per pitch with columns from fact_pitch + dim_game.season.
    """
    query = """
    SELECT
        fp.pitch_id,
        fp.pa_id,
        fp.game_pk,
        fp.pitcher_id,
        fp.batter_id,
        fp.pitch_number,
        fp.pitch_type,
        fp.pitch_name,
        fp.description,
        fp.release_speed,
        fp.effective_speed,
        fp.release_spin_rate,
        fp.release_extension,
        fp.spin_axis,
        fp.pfx_x,
        fp.pfx_z,
        fp.zone,
        fp.plate_x,
        fp.plate_z,
        fp.balls,
        fp.strikes,
        fp.outs_when_up,
        fp.bat_score_diff,
        fp.is_whiff,
        fp.is_called_strike,
        fp.is_bip,
        fp.is_swing,
        fp.is_foul,
        fp.batter_stand,
        dg.game_date,
        dg.season
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
    ORDER BY fp.game_pk, fp.game_counter, fp.pitch_number
    """
    logger.info("Fetching pitch-level data for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 2. Hitter pitch-type profile
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
# 3. Pitcher arsenal profile
# ---------------------------------------------------------------------------
def get_pitcher_arsenal_profile(season: int) -> pd.DataFrame:
    """Per pitcher, per pitch type: usage, whiff rate, barrel rate against,
    avg velocity, avg horizontal/vertical movement.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitch_hand, pitch_type, pitches, total_pitches,
        usage_pct, swings, whiffs, bip, barrels_proxy, hard_hits,
        xwoba_against, avg_velo, avg_pfx_x, avg_pfx_z.
    """
    query = """
    WITH pitcher_totals AS (
        SELECT
            fp.pitcher_id,
            COUNT(*) AS total_pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id
    ),
    pitch_agg AS (
        SELECT
            fp.pitcher_id,
            dp.pitch_hand,
            fp.pitch_type,
            COUNT(*)                          AS pitches,
            SUM(fp.is_swing::int)             AS swings,
            SUM(fp.is_whiff::int)             AS whiffs,
            SUM(fp.is_called_strike::int)     AS called_strikes,
            SUM(CASE WHEN fp.is_whiff OR fp.is_called_strike THEN 1 ELSE 0 END)
                                              AS csw,
            SUM(fp.is_bip::int)               AS bip,
            AVG(fp.release_speed)             AS avg_velo,
            AVG(fp.pfx_x)                     AS avg_pfx_x,
            AVG(fp.pfx_z)                     AS avg_pfx_z
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.dim_player dp ON fp.pitcher_id = dp.player_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id, dp.pitch_hand, fp.pitch_type
    ),
    batted_agg AS (
        SELECT
            fp.pitcher_id,
            fp.pitch_type,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END) AS hard_hits,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)                   AS barrels_proxy,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba_against
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.is_bip
        GROUP BY fp.pitcher_id, fp.pitch_type
    )
    SELECT
        pa.pitcher_id,
        pa.pitch_hand,
        pa.pitch_type,
        pa.pitches,
        pt.total_pitches,
        ROUND((pa.pitches::numeric / pt.total_pitches), 4) AS usage_pct,
        pa.swings,
        pa.whiffs,
        pa.called_strikes,
        pa.csw,
        pa.bip,
        COALESCE(ba.barrels_proxy, 0)  AS barrels_proxy,
        COALESCE(ba.hard_hits, 0)      AS hard_hits,
        ROUND(ba.xwoba_against::numeric, 3) AS xwoba_against,
        ROUND(pa.avg_velo::numeric, 1) AS avg_velo,
        ROUND(pa.avg_pfx_x::numeric, 2) AS avg_pfx_x,
        ROUND(pa.avg_pfx_z::numeric, 2) AS avg_pfx_z
    FROM pitch_agg pa
    JOIN pitcher_totals pt ON pa.pitcher_id = pt.pitcher_id
    LEFT JOIN batted_agg ba
        ON pa.pitcher_id = ba.pitcher_id
       AND pa.pitch_type = ba.pitch_type
    ORDER BY pa.pitcher_id, pa.pitches DESC
    """
    logger.info("Fetching pitcher arsenal profiles for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 4. Season totals (player season lines)
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

    # Compute wOBA: BIP contribution + non-BIP weights (uBB=0.69, HBP=0.72)
    woba_bb_weight = 0.69
    woba_hbp_weight = 0.72
    non_ibb_bb = df["bb"] - df["ibb"]
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * woba_bb_weight
        + df["hbp"] * woba_hbp_weight
    )
    woba_den = (df["pa"] - df["ibb"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).round(3)

    # Drop intermediate column
    df.drop(columns=["woba_bip_sum"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# 4b. Season totals split by pitcher hand (for platoon model)
# ---------------------------------------------------------------------------
def get_season_totals_by_pitcher_hand(season: int) -> pd.DataFrame:
    """Per (batter_id, pitch_hand) season K/PA with same_side flag.

    Batted-ball quality metrics (barrel_pct, hard_hit_pct) are NOT split by
    pitch_hand — contact quality is stable across pitcher handedness, and
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


# ---------------------------------------------------------------------------
# 5. Pitcher season totals (from boxscores)
# ---------------------------------------------------------------------------
def get_pitcher_season_totals(season: int) -> pd.DataFrame:
    """Per-pitcher season aggregates from staging boxscores.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, games, ip,
        k, bb, hr, batters_faced, k_rate, bb_rate, hr_per_9.
    """
    query = """
    SELECT
        pb.pitcher_id,
        dp.player_name   AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        COUNT(DISTINCT pb.game_pk) AS games,
        SUM(pb.innings_pitched)    AS ip,
        SUM(pb.strike_outs)        AS k,
        SUM(pb.walks)              AS bb,
        SUM(pb.home_runs)          AS hr,
        SUM(pb.batters_faced)      AS batters_faced,
        ROUND(SUM(pb.strike_outs)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4)
                                   AS k_rate,
        ROUND(SUM(pb.walks)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4)
                                   AS bb_rate,
        ROUND(SUM(pb.home_runs)::numeric / NULLIF(SUM(pb.innings_pitched)::numeric, 0) * 9, 2)
                                   AS hr_per_9
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg   ON pb.game_pk = dg.game_pk
    JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand, dg.season
    ORDER BY SUM(pb.batters_faced) DESC
    """
    logger.info("Fetching pitcher season totals for %d", season)
    return read_sql(query, {"season": season})


def get_pitcher_season_totals_with_age(season: int) -> pd.DataFrame:
    """Per-pitcher season aggregates enriched with age and age_bucket.

    Age is computed as of July 1 of the season (midpoint).

    Returns
    -------
    pd.DataFrame
        Same columns as ``get_pitcher_season_totals`` plus: birth_date,
        age, age_bucket, bb_per_bf, hr_per_bf.
        age_bucket: 0 = young (<=25), 1 = prime (26-30), 2 = veteran (31+).
    """
    query = """
    SELECT
        pb.pitcher_id,
        dp.player_name   AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        dp.birth_date,
        EXTRACT(YEAR FROM AGE(DATE(dg.season || '-07-01'), dp.birth_date))::int AS age,
        COUNT(DISTINCT pb.game_pk) AS games,
        SUM(pb.innings_pitched)    AS ip,
        SUM(pb.strike_outs)        AS k,
        SUM(pb.walks)              AS bb,
        SUM(pb.home_runs)          AS hr,
        SUM(pb.batters_faced)      AS batters_faced,
        ROUND(SUM(pb.strike_outs)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4)
                                   AS k_rate,
        ROUND(SUM(pb.walks)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4)
                                   AS bb_rate,
        ROUND(SUM(pb.home_runs)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4)
                                   AS hr_per_bf,
        ROUND(SUM(pb.home_runs)::numeric / NULLIF(SUM(pb.innings_pitched)::numeric, 0) * 9, 2)
                                   AS hr_per_9
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg   ON pb.game_pk = dg.game_pk
    JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand, dg.season, dp.birth_date
    ORDER BY SUM(pb.batters_faced) DESC
    """
    logger.info("Fetching pitcher season totals with age for %d", season)
    df = read_sql(query, {"season": season})

    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 30, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# 6. Pitch shape data (for pitch archetype clustering)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 8. Pitcher outcomes by batter stand (for league baselines v2)
# ---------------------------------------------------------------------------
def get_pitcher_outcomes_by_stand(season: int) -> pd.DataFrame:
    """Per (pitcher_id, pitch_type, batter_stand) outcome counts for one season.

    xwOBA is returned as sum + count (not AVG) so Python can correctly compute
    weighted averages when aggregating across pitchers.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitch_type, batter_stand, pitches, swings, whiffs,
        out_of_zone_pitches, chase_swings, called_strikes, csw, bip,
        hard_hits, barrels_proxy, xwoba_contact_sum, xwoba_contact_n.
    """
    query = """
    WITH pitch_agg AS (
        SELECT
            fp.pitcher_id,
            fp.pitch_type,
            fp.batter_stand,
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
          AND fp.pitch_type NOT IN ('PO','UN','SC','FA')
        GROUP BY fp.pitcher_id, fp.pitch_type, fp.batter_stand
    ),
    batted_agg AS (
        SELECT
            fp.pitcher_id,
            fp.pitch_type,
            fp.batter_stand,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)    AS hard_hits,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)                      AS barrels_proxy,
            SUM(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba ELSE 0 END)
                                                              AS xwoba_contact_sum,
            SUM(CASE WHEN sbb.xwoba != 'NaN' THEN 1 ELSE 0 END)
                                                              AS xwoba_contact_n
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.pitch_type NOT IN ('PO','UN','SC','FA')
          AND fp.is_bip
        GROUP BY fp.pitcher_id, fp.pitch_type, fp.batter_stand
    )
    SELECT
        pa.pitcher_id,
        pa.pitch_type,
        pa.batter_stand,
        pa.pitches,
        pa.swings,
        pa.whiffs,
        pa.out_of_zone_pitches,
        pa.chase_swings,
        pa.called_strikes,
        pa.csw,
        pa.bip,
        COALESCE(ba.hard_hits, 0)            AS hard_hits,
        COALESCE(ba.barrels_proxy, 0)        AS barrels_proxy,
        COALESCE(ba.xwoba_contact_sum, 0)    AS xwoba_contact_sum,
        COALESCE(ba.xwoba_contact_n, 0)      AS xwoba_contact_n
    FROM pitch_agg pa
    LEFT JOIN batted_agg ba
        ON pa.pitcher_id = ba.pitcher_id
       AND pa.pitch_type = ba.pitch_type
       AND pa.batter_stand = ba.batter_stand
    ORDER BY pa.pitcher_id, pa.pitch_type, pa.batter_stand
    """
    logger.info("Fetching pitcher outcomes by stand for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 9. Pitcher game logs (from pitching boxscores)
# ---------------------------------------------------------------------------
def get_pitcher_game_logs(season: int) -> pd.DataFrame:
    """Per-game pitcher lines from staging boxscores.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, pitcher_name, pitch_hand, season,
        strike_outs, batters_faced, innings_pitched, number_of_pitches,
        is_starter, walks, hits, home_runs, outs, earned_runs.
    """
    query = """
    SELECT
        pb.game_pk,
        pb.pitcher_id,
        dp.player_name   AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        pb.strike_outs,
        pb.batters_faced,
        pb.innings_pitched,
        pb.number_of_pitches,
        pb.is_starter,
        pb.walks,
        pb.hits,
        pb.home_runs,
        pb.outs,
        pb.earned_runs
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg   ON pb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    ORDER BY pb.game_pk, pb.pitcher_id
    """
    logger.info("Fetching pitcher game logs for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 10. Game-level batter Ks (per pitcher-batter within a game)
# ---------------------------------------------------------------------------
def get_game_batter_ks(season: int) -> pd.DataFrame:
    """Per (game_pk, pitcher_id, batter_id) PA and K counts.

    Only completed PAs (events IS NOT NULL) in regular-season games.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, batter_id, pa, k.
    """
    query = """
    SELECT
        fpa.game_pk,
        fpa.pitcher_id,
        fpa.batter_id,
        COUNT(*)  AS pa,
        SUM(CASE WHEN fpa.events IN ('strikeout', 'strikeout_double_play')
                 THEN 1 ELSE 0 END) AS k
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fpa.events IS NOT NULL
    GROUP BY fpa.game_pk, fpa.pitcher_id, fpa.batter_id
    ORDER BY fpa.game_pk, fpa.pitcher_id, fpa.batter_id
    """
    logger.info("Fetching game batter Ks for %d", season)
    return read_sql(query, {"season": season})


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

    # Compute wOBA: BIP contribution + non-BIP weights (uBB=0.69, HBP=0.72)
    woba_bb_weight = 0.69
    woba_hbp_weight = 0.72
    non_ibb_bb = df["bb"] - df["ibb"]
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * woba_bb_weight
        + df["hbp"] * woba_hbp_weight
    )
    woba_den = (df["pa"] - df["ibb"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).round(3)

    # Drop intermediate column
    df.drop(columns=["woba_bip_sum"], inplace=True)

    # Compute age_bucket: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 30, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype("Int64")

    return df


def get_pitch_shape_data(season: int) -> pd.DataFrame:
    """Pitch-shape records for one season from sat_pitch_shape.

    Parameters
    ----------
    season : int
        MLB season year (e.g. 2025).

    Returns
    -------
    pd.DataFrame
        One row per pitch with shape metrics plus pitcher and pitch metadata.
    """
    query = """
    SELECT
        fp.pitch_id,
        fp.pitcher_id,
        fp.pitch_type,
        fp.pitch_name,
        dp.pitch_hand,
        dg.season,
        sps.release_speed,
        sps.pfx_x,
        sps.pfx_z,
        sps.release_spin_rate,
        sps.release_extension,
        sps.release_pos_x,
        sps.release_pos_z
    FROM production.sat_pitch_shape sps
    JOIN production.fact_pitch fp ON sps.pitch_id = fp.pitch_id
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON fp.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
    ORDER BY fp.pitch_id
    """
    logger.info("Fetching pitch shape data for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 7. Pitch shape offerings (pre-aggregated for archetype clustering)
# ---------------------------------------------------------------------------
def get_pitch_shape_offerings(season: int) -> pd.DataFrame:
    """One row per (pitcher_id, pitch_hand, pitch_type, pitch_name) with
    averaged shape features, pre-filtered in SQL.

    This replaces the pattern of fetching ~600K raw rows and aggregating in
    Python.  Result is ~3,800 rows/season.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: season, pitcher_id, pitch_hand, pitch_type, pitch_name,
        pitches, release_speed, pfx_x, pfx_z, release_spin_rate,
        release_extension, release_pos_x, release_pos_z, pitch_family.
    """
    query = """
    SELECT
        dg.season,
        fp.pitcher_id,
        dp.pitch_hand,
        fp.pitch_type,
        fp.pitch_name,
        COUNT(*)                          AS pitches,
        AVG(sps.release_speed)            AS release_speed,
        AVG(sps.pfx_x)                    AS pfx_x,
        AVG(sps.pfx_z)                    AS pfx_z,
        AVG(sps.release_spin_rate)        AS release_spin_rate,
        AVG(sps.release_extension)        AS release_extension,
        AVG(sps.release_pos_x)            AS release_pos_x,
        AVG(sps.release_pos_z)            AS release_pos_z
    FROM production.sat_pitch_shape sps
    JOIN production.fact_pitch fp ON sps.pitch_id = fp.pitch_id
    JOIN production.dim_game dg   ON fp.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON fp.pitcher_id = dp.player_id
    WHERE dg.season    = :season
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
      AND fp.pitch_type NOT IN ('PO', 'UN', 'SC', 'FA')
      AND sps.release_speed     != 'NaN'
      AND sps.pfx_x             != 'NaN'
      AND sps.pfx_z             != 'NaN'
      AND sps.release_spin_rate != 'NaN'
      AND sps.release_extension != 'NaN'
    GROUP BY dg.season, fp.pitcher_id, dp.pitch_hand,
             fp.pitch_type, fp.pitch_name
    ORDER BY fp.pitcher_id, pitches DESC
    """
    logger.info("Fetching pitch shape offerings for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 14. Hitter observed profile (pitch-level aggregates)
# ---------------------------------------------------------------------------
def get_hitter_observed_profile(season: int) -> pd.DataFrame:
    """Per-batter pitch-level and batted-ball aggregates for composite scoring.

    Returns whiff rate, chase rate, zone contact %, avg exit velocity,
    fly ball %, and hard-hit % — all aggregated to the batter level.

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
# 14b. Batted-ball spray distribution
# ---------------------------------------------------------------------------
def get_batted_ball_spray(season: int) -> pd.DataFrame:
    """Per-batter batted ball spray distribution (pull / middle / oppo).

    Uses pre-computed ``spray_bucket`` from ``sat_batted_balls``, which is
    already relative to batter handedness ('pull' = pulled side for that hitter).

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, bip, pull_pct, middle_pct, oppo_pct.
    """
    query = """
    SELECT
        fpa.batter_id,
        COUNT(*)                                                    AS bip,
        SUM(CASE WHEN sbb.spray_bucket = 'pull' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                   AS pull_pct,
        SUM(CASE WHEN sbb.spray_bucket = 'middle' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                   AS middle_pct,
        SUM(CASE WHEN sbb.spray_bucket = 'oppo' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)                                   AS oppo_pct
    FROM production.sat_batted_balls sbb
    JOIN production.fact_pa fpa ON sbb.pa_id = fpa.pa_id
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND sbb.spray_bucket IS NOT NULL
    GROUP BY fpa.batter_id
    HAVING COUNT(*) >= 50
    """
    logger.info("Fetching batted ball spray distribution for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 14c. Pitcher fly-ball / HR data (for xFIP computation)
# ---------------------------------------------------------------------------
def get_pitcher_fly_ball_data(season: int) -> pd.DataFrame:
    """Per-pitcher fly ball counts and HR/FB rate.

    Uses batted-ball data to count fly balls (launch_angle > 25) and
    home runs.  Required for xFIP derivation.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, total_bip, fly_balls, home_runs, hr_per_fb.
    """
    query = """
    SELECT
        fp.pitcher_id,
        COUNT(*)                                                    AS total_bip,
        SUM(CASE WHEN sbb.launch_angle > 25 THEN 1 ELSE 0 END)    AS fly_balls,
        SUM(CASE WHEN sbb.is_homerun THEN 1 ELSE 0 END)            AS home_runs,
        SUM(CASE WHEN sbb.is_homerun THEN 1 ELSE 0 END)::float
            / NULLIF(SUM(CASE WHEN sbb.launch_angle > 25
                         THEN 1 ELSE 0 END), 0)                    AS hr_per_fb
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fp.is_bip = true
      AND sbb.launch_angle IS NOT NULL
    GROUP BY fp.pitcher_id
    HAVING COUNT(*) >= 50
    """
    logger.info("Fetching pitcher fly ball data for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 15. Sprint speed
# ---------------------------------------------------------------------------
def get_sprint_speed(season: int) -> pd.DataFrame:
    """Sprint speed from Statcast sprint speed table.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, sprint_speed, hp_to_1b, bolts.
    """
    query = """
    SELECT
        player_id,
        sprint_speed,
        hp_to_1b,
        bolts
    FROM staging.statcast_sprint_speed
    WHERE season = :season
      AND sprint_speed IS NOT NULL
    """
    logger.info("Fetching sprint speed for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 16. Pitcher observed profile (pitch-level aggregates)
# ---------------------------------------------------------------------------
def get_pitcher_observed_profile(season: int) -> pd.DataFrame:
    """Per-pitcher pitch-level aggregates for composite scoring.

    Returns whiff rate, avg velo, release extension, zone %, and GB%.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, whiff_rate, avg_velo, release_extension,
        zone_pct, gb_pct.
    """
    query = """
    WITH pitch_agg AS (
        SELECT
            fp.pitcher_id,
            SUM(fp.is_swing::int)                           AS swings,
            SUM(fp.is_whiff::int)                           AS whiffs,
            COUNT(*)                                        AS pitches,
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                     THEN 1 ELSE 0 END)                    AS zone_pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.pitch_type NOT IN ('PO', 'UN', 'SC', 'FA')
        GROUP BY fp.pitcher_id
    ),
    shape_agg AS (
        SELECT
            fp.pitcher_id,
            AVG(sps.release_speed)       AS avg_velo,
            AVG(sps.release_extension)   AS release_extension
        FROM production.sat_pitch_shape sps
        JOIN production.fact_pitch fp ON sps.pitch_id = fp.pitch_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND sps.release_speed     != 'NaN'
          AND sps.release_extension != 'NaN'
        GROUP BY fp.pitcher_id
    ),
    batted_agg AS (
        SELECT
            fpa.pitcher_id,
            COUNT(*)                                                    AS bip,
            SUM(CASE WHEN sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0)                                   AS gb_pct
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND sbb.launch_angle != 'NaN'
        GROUP BY fpa.pitcher_id
    )
    SELECT
        pa.pitcher_id,
        ROUND((pa.whiffs::numeric / NULLIF(pa.swings, 0)), 4)   AS whiff_rate,
        ROUND(sa.avg_velo::numeric, 1)                           AS avg_velo,
        ROUND(sa.release_extension::numeric, 2)                  AS release_extension,
        ROUND((pa.zone_pitches::numeric / NULLIF(pa.pitches, 0)), 4)
                                                                  AS zone_pct,
        ROUND(ba.gb_pct::numeric, 4)                              AS gb_pct
    FROM pitch_agg pa
    LEFT JOIN shape_agg sa ON pa.pitcher_id = sa.pitcher_id
    LEFT JOIN batted_agg ba ON pa.pitcher_id = ba.pitcher_id
    WHERE pa.swings >= 50
    ORDER BY pa.pitches DESC
    """
    logger.info("Fetching pitcher observed profile for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 17. Extended hitter season totals (from batting boxscores — includes games, SB)
# ---------------------------------------------------------------------------
def get_hitter_season_totals_extended(season: int) -> pd.DataFrame:
    """Per-batter season totals from batting boxscores.

    Includes games played, stolen bases, caught stealing, total bases,
    runs, RBI — columns not available from fact_pa.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, season, age, games, pa, k, bb,
        hits, hr, sb, cs, total_bases, runs, rbi, doubles, triples,
        hit_by_pitch, age_bucket.
    """
    query = """
    SELECT
        bb.batter_id,
        dp.player_name                  AS batter_name,
        dg.season,
        dp.birth_date,
        EXTRACT(YEAR FROM AGE(
            DATE(dg.season || '-07-01'), dp.birth_date
        ))::int                         AS age,
        COUNT(DISTINCT bb.game_pk)      AS games,
        SUM(bb.plate_appearances)       AS pa,
        SUM(bb.strikeouts)              AS k,
        SUM(bb.walks + bb.intentional_walks) AS bb,
        SUM(bb.hits)                    AS hits,
        SUM(bb.home_runs)              AS hr,
        SUM(bb.sb)                     AS sb,
        SUM(bb.caught_stealing)        AS cs,
        SUM(bb.total_bases)            AS total_bases,
        SUM(bb.runs)                   AS runs,
        SUM(bb.rbi)                    AS rbi,
        SUM(bb.doubles)                AS doubles,
        SUM(bb.triples)                AS triples,
        SUM(bb.hit_by_pitch)           AS hit_by_pitch
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON bb.batter_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    GROUP BY bb.batter_id, dp.player_name, dg.season, dp.birth_date
    HAVING SUM(bb.plate_appearances) >= 1
    ORDER BY SUM(bb.plate_appearances) DESC
    """
    logger.info("Fetching extended hitter season totals for %d", season)
    df = read_sql(query, {"season": season})

    # Compute age_bucket: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)
    import pandas as _pd
    df["age_bucket"] = _pd.cut(
        df["age"],
        bins=[0, 25, 30, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype("Int64")

    # Derived rates
    df["k_rate"] = (df["k"] / df["pa"]).round(4)
    df["bb_rate"] = (df["bb"] / df["pa"]).round(4)
    df["hr_rate"] = (df["hr"] / df["pa"]).round(4)
    df["hit_rate"] = (df["hits"] / df["pa"]).round(4)
    df["sb_per_game"] = (df["sb"] / df["games"].replace(0, float("nan"))).round(4)

    return df


# ---------------------------------------------------------------------------
# 18. Extended pitcher season totals (from pitching boxscores — includes outs)
# ---------------------------------------------------------------------------
def get_pitcher_season_totals_extended(season: int) -> pd.DataFrame:
    """Per-pitcher season totals from pitching boxscores.

    Includes outs recorded directly, plus games, IP, BF.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, age, games,
        ip, batters_faced, k, bb, hr, outs, hits_allowed, runs_allowed,
        is_starter, age_bucket.
    """
    query = """
    SELECT
        pb.pitcher_id,
        dp.player_name                  AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        dp.birth_date,
        EXTRACT(YEAR FROM AGE(
            DATE(dg.season || '-07-01'), dp.birth_date
        ))::int                         AS age,
        COUNT(DISTINCT pb.game_pk)      AS games,
        SUM(pb.innings_pitched)         AS ip,
        SUM(pb.batters_faced)           AS batters_faced,
        SUM(pb.strike_outs)             AS k,
        SUM(pb.outs)                    AS outs,
        SUM(pb.is_starter::int)         AS starts,
        SUM(pb.home_runs)               AS hr,
        SUM(pb.walks + pb.intentional_walks) AS bb,
        SUM(pb.hits)                    AS hits_allowed,
        SUM(pb.earned_runs)             AS earned_runs
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand,
             dg.season, dp.birth_date
    HAVING SUM(pb.batters_faced) >= 1
    ORDER BY SUM(pb.batters_faced) DESC
    """
    logger.info("Fetching extended pitcher season totals for %d", season)
    df = read_sql(query, {"season": season})

    import pandas as _pd
    df["age_bucket"] = _pd.cut(
        df["age"],
        bins=[0, 25, 30, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype("Int64")

    # Derive starter flag and rates
    df["is_starter"] = (df["starts"] >= 3).astype(int)
    df["k_rate"] = (df["k"] / df["batters_faced"].replace(0, float("nan"))).round(4)
    df["bb_rate"] = (df["bb"] / df["batters_faced"].replace(0, float("nan"))).round(4)
    df["outs_per_bf"] = (df["outs"] / df["batters_faced"].replace(0, float("nan"))).round(4)

    return df


# ---------------------------------------------------------------------------
# 19. Park factors (HR by batter handedness)
# ---------------------------------------------------------------------------
def get_park_factors(season: int) -> pd.DataFrame:
    """HR park factors by venue and batter handedness.

    Uses 3-year smoothed park factor (more stable than single-season).
    Falls back to single-season if 3yr is missing.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: venue_id, venue_name, batter_stand, hr_pf.
    """
    query = """
    SELECT
        venue_id,
        venue_name,
        batter_stand,
        COALESCE(hr_pf_3yr, hr_pf_season, 1.0) AS hr_pf
    FROM production.dim_park_factor
    WHERE season = :season
    ORDER BY venue_id, batter_stand
    """
    logger.info("Fetching park factors for %d", season)
    return read_sql(query, {"season": season})


def get_hitter_team_venue(season: int) -> pd.DataFrame:
    """Map each hitter to their primary team and home venue for a season.

    Uses batting boxscores to find the team where each hitter played the
    most games, then maps that team to its home venue via dim_game.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, team_id, team_name, venue_id, bat_side.
    """
    query = """
    WITH batter_teams AS (
        SELECT
            bb.batter_id,
            bb.team_id,
            bb.team_name,
            COUNT(*) AS games,
            ROW_NUMBER() OVER (
                PARTITION BY bb.batter_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
        GROUP BY bb.batter_id, bb.team_id, bb.team_name
    ),
    team_venue AS (
        SELECT
            home_team_id AS team_id,
            venue_id,
            ROW_NUMBER() OVER (
                PARTITION BY home_team_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM production.dim_game
        WHERE season = :season
          AND game_type = 'R'
        GROUP BY home_team_id, venue_id
    )
    SELECT
        bt.batter_id,
        bt.team_id,
        bt.team_name,
        tv.venue_id,
        COALESCE(dp.bat_side, 'R') AS bat_side
    FROM batter_teams bt
    JOIN team_venue tv ON bt.team_id = tv.team_id AND tv.rn = 1
    LEFT JOIN production.dim_player dp ON bt.batter_id = dp.player_id
    WHERE bt.rn = 1
    ORDER BY bt.batter_id
    """
    logger.info("Fetching hitter team-venue mapping for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 20. Umpire tendencies (K, BB, HR rate — derived from pitch/PA data)
# ---------------------------------------------------------------------------
def get_umpire_tendencies(
    seasons: list[int] | None = None,
    min_games: int = 30,
) -> pd.DataFrame:
    """Compute per-umpire K, BB, and HR rate tendencies with shrinkage.

    Uses multi-season PA-level data joined to dim_umpire to compute each
    HP umpire's K-rate, BB-rate, and HR-rate, then shrinks toward the
    league mean based on games umpired (more games = more trust).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2021-2025 (5 recent seasons).
    min_games : int
        Minimum games umpired to be included.

    Returns
    -------
    pd.DataFrame
        Columns: hp_umpire_name, games, total_pa, k_rate, bb_rate, hr_rate,
        league_k_rate, league_bb_rate, league_hr_rate,
        k_rate_shrunk, bb_rate_shrunk, hr_rate_shrunk,
        k_logit_lift, bb_logit_lift, hr_logit_lift.
    """
    if seasons is None:
        seasons = list(range(2021, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH ump_stats AS (
        SELECT
            du.hp_umpire_name,
            COUNT(DISTINCT du.game_pk) AS games,
            COUNT(fpa.pa_id) AS total_pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS total_k,
            SUM(CASE WHEN fpa.events = 'walk'
                     THEN 1 ELSE 0 END) AS total_bb,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END) AS total_hr
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        JOIN production.fact_pa fpa ON du.game_pk = fpa.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY du.hp_umpire_name
        HAVING COUNT(DISTINCT du.game_pk) >= {int(min_games)}
    )
    SELECT
        hp_umpire_name,
        games,
        total_pa,
        total_k,
        total_bb,
        total_hr,
        ROUND((total_k::numeric / NULLIF(total_pa, 0)), 5) AS k_rate,
        ROUND((total_bb::numeric / NULLIF(total_pa, 0)), 5) AS bb_rate,
        ROUND((total_hr::numeric / NULLIF(total_pa, 0)), 5) AS hr_rate
    FROM ump_stats
    ORDER BY games DESC
    """
    logger.info("Fetching umpire tendencies for seasons %s (min_games=%d)",
                seasons, min_games)
    df = read_sql(query, {})

    if df.empty:
        return df

    import numpy as _np
    from scipy.special import logit as _logit
    _clip = lambda x: _np.clip(x, 1e-6, 1 - 1e-6)

    # Shrinkage: trust umpires with more games
    # k = 80 games (~1 full season) as the shrinkage constant
    shrinkage_k = 80.0
    df["reliability"] = df["games"] / (df["games"] + shrinkage_k)

    # --- K-rate ---
    league_k_rate = float(df["total_k"].sum() / df["total_pa"].sum())
    df["league_k_rate"] = league_k_rate
    df["k_rate_shrunk"] = (
        df["reliability"] * df["k_rate"]
        + (1 - df["reliability"]) * league_k_rate
    )
    df["k_logit_lift"] = (
        _logit(_clip(df["k_rate_shrunk"].values))
        - _logit(_clip(league_k_rate))
    ).round(4)

    # --- BB-rate ---
    league_bb_rate = float(df["total_bb"].sum() / df["total_pa"].sum())
    df["league_bb_rate"] = league_bb_rate
    df["bb_rate_shrunk"] = (
        df["reliability"] * df["bb_rate"]
        + (1 - df["reliability"]) * league_bb_rate
    )
    df["bb_logit_lift"] = (
        _logit(_clip(df["bb_rate_shrunk"].values))
        - _logit(_clip(league_bb_rate))
    ).round(4)

    # --- HR-rate ---
    league_hr_rate = float(df["total_hr"].sum() / df["total_pa"].sum())
    df["league_hr_rate"] = league_hr_rate
    df["hr_rate_shrunk"] = (
        df["reliability"] * df["hr_rate"]
        + (1 - df["reliability"]) * league_hr_rate
    )
    df["hr_logit_lift"] = (
        _logit(_clip(df["hr_rate_shrunk"].values))
        - _logit(_clip(league_hr_rate))
    ).round(4)

    df = df.drop(columns=["reliability"])
    logger.info(
        "Umpire tendencies: %d umpires, league K%%=%.4f BB%%=%.4f HR%%=%.5f, "
        "K lift range=[%.4f, %.4f], BB lift range=[%.4f, %.4f], "
        "HR lift range=[%.4f, %.4f]",
        len(df), league_k_rate, league_bb_rate, league_hr_rate,
        df["k_logit_lift"].min(), df["k_logit_lift"].max(),
        df["bb_logit_lift"].min(), df["bb_logit_lift"].max(),
        df["hr_logit_lift"].min(), df["hr_logit_lift"].max(),
    )
    return df


def get_umpire_k_tendencies(
    seasons: list[int] | None = None,
    min_games: int = 30,
) -> pd.DataFrame:
    """Backward-compatible alias for ``get_umpire_tendencies()``.

    Returns the same DataFrame (which now includes bb/hr columns too).
    """
    return get_umpire_tendencies(seasons=seasons, min_games=min_games)


# ---------------------------------------------------------------------------
# 21. Weather effects on K and HR rates
# ---------------------------------------------------------------------------
def get_weather_effects(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute weather-adjusted K and HR rate multipliers.

    Groups outdoor games by temperature bucket and wind category, computes
    K-rate and HR-rate for each combination, and expresses as a multiplier
    relative to the overall outdoor average.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2018-2025.

    Returns
    -------
    pd.DataFrame
        Columns: temp_bucket, wind_category, games, k_rate, hr_rate,
        k_multiplier, hr_multiplier.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH game_stats AS (
        SELECT
            fpa.game_pk,
            COUNT(*) AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN fpa.events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY fpa.game_pk
    )
    SELECT
        CASE
            WHEN dw.temperature < 55 THEN 'cold'
            WHEN dw.temperature BETWEEN 55 AND 69 THEN 'cool'
            WHEN dw.temperature BETWEEN 70 AND 84 THEN 'warm'
            ELSE 'hot'
        END AS temp_bucket,
        dw.wind_category,
        COUNT(*) AS games,
        SUM(gs.pa) AS total_pa,
        SUM(gs.k) AS total_k,
        SUM(gs.hr) AS total_hr
    FROM game_stats gs
    JOIN production.dim_weather dw ON gs.game_pk = dw.game_pk
    WHERE NOT dw.is_dome
    GROUP BY 1, dw.wind_category
    HAVING COUNT(*) >= 50
    ORDER BY 1, dw.wind_category
    """
    logger.info("Fetching weather effects for seasons %s", seasons)
    df = read_sql(query, {})

    if df.empty:
        return df

    df["k_rate"] = (df["total_k"] / df["total_pa"]).round(5)
    df["hr_rate"] = (df["total_hr"] / df["total_pa"]).round(5)

    # Overall outdoor averages
    overall_k = float(df["total_k"].sum() / df["total_pa"].sum())
    overall_hr = float(df["total_hr"].sum() / df["total_pa"].sum())

    df["k_multiplier"] = (df["k_rate"] / overall_k).round(4)
    df["hr_multiplier"] = (df["hr_rate"] / overall_hr).round(4)
    df["overall_k_rate"] = overall_k
    df["overall_hr_rate"] = overall_hr

    df = df.drop(columns=["total_pa", "total_k", "total_hr"])

    logger.info("Weather effects: %d combinations, K mult range=[%.3f, %.3f], HR mult range=[%.3f, %.3f]",
                len(df),
                df["k_multiplier"].min(), df["k_multiplier"].max(),
                df["hr_multiplier"].min(), df["hr_multiplier"].max())
    return df


# ---------------------------------------------------------------------------
# 22. Player team mapping (player_id → team abbreviation)
# ---------------------------------------------------------------------------
def get_player_teams(season: int) -> pd.DataFrame:
    """Map each player to their primary team abbreviation for a given season.

    Uses batting + pitching boxscores to find the team each player appeared
    with most often, then joins to dim_team for the abbreviation.

    Parameters
    ----------
    season : int
        Season to look up.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, team_abbr, team_name.
    """
    query = """
    WITH player_team_games AS (
        -- Hitter appearances
        SELECT bb.batter_id AS player_id, bb.team_id,
               COUNT(*) AS games
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY bb.batter_id, bb.team_id

        UNION ALL

        -- Pitcher appearances
        SELECT pb.pitcher_id AS player_id, pb.team_id,
               COUNT(*) AS games
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id, pb.team_id
    ),
    primary_team AS (
        SELECT player_id, team_id,
               SUM(games) AS total_games,
               ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY SUM(games) DESC) AS rn
        FROM player_team_games
        GROUP BY player_id, team_id
    )
    SELECT pt.player_id,
           COALESCE(dt.abbreviation, '') AS team_abbr,
           COALESCE(dt.team_name, '') AS team_name,
           COALESCE(dt.league, '') AS league,
           COALESCE(dt.division, '') AS division
    FROM primary_team pt
    LEFT JOIN production.dim_team dt ON pt.team_id = dt.team_id
    WHERE pt.rn = 1
    ORDER BY pt.player_id
    """
    logger.info("Fetching player-team mapping for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 23. Game lineups from fact_lineup
# ---------------------------------------------------------------------------
def get_game_lineups(season: int) -> pd.DataFrame:
    """Fetch starting lineups (batting order 1-9) for every regular-season game.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, player_id, batting_order, team_id, batter_name.
    """
    query = """
    SELECT
        fl.game_pk,
        fl.player_id,
        fl.batting_order,
        fl.team_id,
        COALESCE(dp.player_name, 'Unknown') AS batter_name
    FROM production.fact_lineup fl
    JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON fl.player_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fl.batting_order BETWEEN 1 AND 9
      AND fl.is_starter = true
    ORDER BY fl.game_pk, fl.batting_order
    """
    logger.info("Fetching game lineups for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# 23b. Lineup for a single game
# ---------------------------------------------------------------------------
def get_lineup_for_game(game_pk: int) -> pd.DataFrame:
    """Fetch the starting batting lineup for a single game.

    Parameters
    ----------
    game_pk : int
        MLB game identifier.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, player_id, batting_order, team_id, batter_name,
        home_away.  Up to 18 rows (9 per team).
    """
    query = """
    SELECT
        fl.game_pk,
        fl.player_id,
        fl.batting_order,
        fl.team_id,
        fl.home_away,
        COALESCE(dp.player_name, 'Unknown') AS batter_name
    FROM production.fact_lineup fl
    LEFT JOIN production.dim_player dp ON fl.player_id = dp.player_id
    WHERE fl.game_pk = :game_pk
      AND fl.batting_order BETWEEN 1 AND 9
      AND fl.is_starter = true
    ORDER BY fl.team_id, fl.batting_order
    """
    logger.info("Fetching lineup for game_pk=%d", game_pk)
    return read_sql(query, {"game_pk": game_pk})


# ---------------------------------------------------------------------------
# 23c. Opposing starting pitcher for a game
# ---------------------------------------------------------------------------
def get_opposing_pitcher_for_game(
    game_pk: int, team_id: int
) -> pd.DataFrame:
    """Fetch the opposing starting pitcher for a given team in a game.

    Parameters
    ----------
    game_pk : int
        MLB game identifier.
    team_id : int
        Team ID for which we want the *opponent's* starter.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, pitcher_name, team_id.
        One row (the opposing starter), or empty if not found.
    """
    query = """
    SELECT
        pb.game_pk,
        pb.pitcher_id,
        COALESCE(dp.player_name, 'Unknown') AS pitcher_name,
        pb.team_id
    FROM staging.pitching_boxscores pb
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE pb.game_pk = :game_pk
      AND pb.is_starter = true
      AND pb.team_id != :team_id
    LIMIT 1
    """
    logger.info(
        "Fetching opposing pitcher for game_pk=%d, team_id=%d",
        game_pk, team_id,
    )
    return read_sql(query, {"game_pk": game_pk, "team_id": team_id})


# ---------------------------------------------------------------------------
# 23d. Starting pitchers for a full season (batch)
# ---------------------------------------------------------------------------
def get_game_starting_pitchers(season: int) -> pd.DataFrame:
    """Fetch starting pitchers for all regular-season games in a season.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, team_id.
        Two rows per game (home and away starters).
    """
    query = """
    SELECT
        pb.game_pk,
        pb.pitcher_id,
        pb.team_id
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND pb.is_starter = true
    ORDER BY pb.game_pk, pb.team_id
    """
    logger.info("Fetching starting pitchers for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Location grid queries (5x5 zone grid)
# ---------------------------------------------------------------------------
def get_pitcher_location_grid(season: int) -> pd.DataFrame:
    """Pitch location grid for all qualified pitchers in a season.

    Returns one row per (pitcher_id, pitch_type, batter_stand, grid_row, grid_col)
    with pitch counts and outcome tallies in each 5x5 zone cell.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_type, batter_stand,
        grid_row, grid_col, pitches, swings, whiffs, called_strikes, bip.
    """
    query = """
    WITH pitcher_filter AS (
        SELECT fp.pitcher_id
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id
        HAVING COUNT(*) >= 500
    )
    SELECT
        fp.pitcher_id,
        COALESCE(dp.player_name, 'Unknown') AS pitcher_name,
        fp.pitch_type,
        fp.batter_stand,
        LEAST(GREATEST(FLOOR((fp.plate_x + 1.33) / 0.532)::int, 0), 4) AS grid_col,
        LEAST(GREATEST(FLOOR((fp.plate_z - 1.0) / 0.6)::int, 0), 4) AS grid_row,
        COUNT(*) AS pitches,
        SUM(fp.is_swing::int) AS swings,
        SUM(fp.is_whiff::int) AS whiffs,
        SUM(fp.is_called_strike::int) AS called_strikes,
        SUM(fp.is_bip::int) AS bip
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    JOIN pitcher_filter pf ON fp.pitcher_id = pf.pitcher_id
    LEFT JOIN production.dim_player dp ON fp.pitcher_id = dp.player_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
      AND fp.plate_x IS NOT NULL AND fp.plate_z IS NOT NULL
      AND CAST(fp.plate_x AS text) != 'NaN'
      AND CAST(fp.plate_z AS text) != 'NaN'
    GROUP BY fp.pitcher_id, dp.player_name, fp.pitch_type, fp.batter_stand,
             grid_col, grid_row
    ORDER BY fp.pitcher_id, fp.pitch_type, grid_row, grid_col
    """
    logger.info("Fetching pitcher location grid for %d", season)
    return read_sql(query, {"season": season})


def get_pitcher_pitch_locations(season: int) -> pd.DataFrame:
    """Raw pitch coordinates for all qualified pitchers in a season.

    Returns one row per pitch with plate_x, plate_z — no grid binning.
    Used for KDE pitch density visualizations.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitch_type, batter_stand, plate_x, plate_z.
    """
    query = """
    WITH pitcher_filter AS (
        SELECT fp.pitcher_id
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id
        HAVING COUNT(*) >= 500
    )
    SELECT
        fp.pitcher_id,
        fp.pitch_type,
        fp.batter_stand,
        fp.plate_x,
        fp.plate_z
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    JOIN pitcher_filter pf ON fp.pitcher_id = pf.pitcher_id
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
      AND fp.plate_x IS NOT NULL AND fp.plate_z IS NOT NULL
      AND CAST(fp.plate_x AS text) != 'NaN'
      AND CAST(fp.plate_z AS text) != 'NaN'
    ORDER BY fp.pitcher_id, fp.pitch_type
    """
    logger.info("Fetching raw pitcher pitch locations for %d", season)
    return read_sql(query, {"season": season})


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


# ---------------------------------------------------------------------------
# 27. Hitter traditional stats (AVG, OBP, SLG, OPS, ISO, wOBA, BABIP)
# ---------------------------------------------------------------------------
def get_hitter_traditional_stats(season: int) -> pd.DataFrame:
    """Season-level traditional batting stats from boxscores + Statcast.

    Combines counting stats from ``staging.batting_boxscores``,
    sac-fly counts from ``production.fact_pa``, and wOBA linear weights
    from ``production.sat_batted_balls`` (BIP) plus standard non-BIP
    weights for BB/HBP.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, season, games, pa, ab, hits,
        doubles, triples, hr, runs, rbi, bb, ibb, hbp, k, sb, cs,
        total_bases, sf, avg, obp, slg, ops, iso, babip, woba.
    """
    query = """
    WITH box AS (
        SELECT
            bb.batter_id,
            dp.player_name              AS batter_name,
            dg.season,
            COUNT(DISTINCT bb.game_pk)  AS games,
            SUM(bb.plate_appearances)   AS pa,
            SUM(bb.at_bats)             AS ab,
            SUM(bb.hits)                AS hits,
            SUM(bb.doubles)             AS doubles,
            SUM(bb.triples)             AS triples,
            SUM(bb.home_runs)           AS hr,
            SUM(bb.runs)                AS runs,
            SUM(bb.rbi)                 AS rbi,
            SUM(bb.walks)               AS bb,
            SUM(bb.intentional_walks)   AS ibb,
            SUM(bb.hit_by_pitch)        AS hbp,
            SUM(bb.strikeouts)          AS k,
            SUM(bb.sb)                  AS sb,
            SUM(bb.caught_stealing)     AS cs,
            SUM(bb.total_bases)         AS total_bases
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON bb.batter_id = dp.player_id
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY bb.batter_id, dp.player_name, dg.season
        HAVING SUM(bb.plate_appearances) >= 1
    ),
    sac_flies AS (
        SELECT
            fp.batter_id,
            SUM(CASE WHEN fp.events = 'sac_fly' THEN 1
                     WHEN fp.events = 'sac_fly_double_play' THEN 1
                     ELSE 0 END) AS sf
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY fp.batter_id
    ),
    woba_bip AS (
        -- wOBA linear weights on batted balls
        SELECT
            fp.batter_id,
            SUM(sbb.woba_value)  AS woba_bip_sum,
            COUNT(*)             AS bip_with_woba
        FROM production.sat_batted_balls sbb
        JOIN production.fact_pa fp ON sbb.pa_id = fp.pa_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND sbb.woba_value IS NOT NULL
        GROUP BY fp.batter_id
    ),
    babip_stat AS (
        SELECT
            fp.batter_id,
            AVG(sbb.babip_value) AS babip
        FROM production.sat_batted_balls sbb
        JOIN production.fact_pa fp ON sbb.pa_id = fp.pa_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND sbb.babip_value IS NOT NULL
          AND fp.events NOT IN ('home_run', 'sac_fly', 'sac_fly_double_play',
                                'sac_bunt', 'sac_bunt_double_play',
                                'catcher_interf')
        GROUP BY fp.batter_id
    )
    SELECT
        b.batter_id,
        b.batter_name,
        b.season,
        b.games,
        b.pa,
        b.ab,
        b.hits,
        b.doubles,
        b.triples,
        b.hr,
        b.runs,
        b.rbi,
        b.bb,
        b.ibb,
        b.hbp,
        b.k,
        b.sb,
        b.cs,
        b.total_bases,
        COALESCE(sf.sf, 0)              AS sf,
        w.woba_bip_sum,
        w.bip_with_woba,
        bs.babip
    FROM box b
    LEFT JOIN sac_flies sf ON b.batter_id = sf.batter_id
    LEFT JOIN woba_bip w ON b.batter_id = w.batter_id
    LEFT JOIN babip_stat bs ON b.batter_id = bs.batter_id
    ORDER BY b.pa DESC
    """
    logger.info("Fetching hitter traditional stats for %d", season)
    df = read_sql(query, {"season": season})

    # -- Compute rate stats in Python (cleaner than SQL for NaN handling) --
    ab = df["ab"].replace(0, float("nan"))
    pa = df["pa"].replace(0, float("nan"))

    df["avg"] = (df["hits"] / ab).round(3)
    df["slg"] = (df["total_bases"] / ab).round(3)
    df["iso"] = (df["slg"] - df["avg"]).round(3)

    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_num = df["hits"] + df["bb"] + df["hbp"]
    obp_den = (df["ab"] + df["bb"] + df["hbp"] + df["sf"]).replace(0, float("nan"))
    df["obp"] = (obp_num / obp_den).round(3)
    df["ops"] = (df["obp"] + df["slg"]).round(3)

    # wOBA: BIP contribution + non-BIP weights (uBB=0.69, HBP=0.72, K=0)
    # Standard wOBA weights (2024 FanGraphs scale, close enough for 2018-2025)
    woba_bb_weight = 0.69
    woba_hbp_weight = 0.72
    non_ibb_bb = df["bb"] - df["ibb"]  # IBB excluded from wOBA
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * woba_bb_weight
        + df["hbp"] * woba_hbp_weight
    )
    # wOBA denominator = AB + non-IBB BB + SF + HBP
    woba_den = (df["ab"] + non_ibb_bb + df["sf"] + df["hbp"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).round(3)

    # BABIP already computed in SQL (from sat_batted_balls, excludes HR/sac)
    df["babip"] = df["babip"].round(3)

    # Drop intermediate columns
    df.drop(columns=["woba_bip_sum", "bip_with_woba"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# 28. Pitcher traditional stats (ERA, WHIP, K/9, BB/9, FIP, W/L/SV)
# ---------------------------------------------------------------------------
def get_pitcher_traditional_stats(season: int) -> pd.DataFrame:
    """Season-level traditional pitching stats from boxscores.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, games,
        starts, w, l, sv, hld, ip, bf, hits_allowed, er, hr_allowed,
        k, bb, hbp, number_of_pitches, era, whip, k_per_9, bb_per_9,
        hr_per_9, k_per_bb, fip.
    """
    query = """
    SELECT
        pb.pitcher_id,
        dp.player_name                  AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        COUNT(DISTINCT pb.game_pk)      AS games,
        SUM(pb.is_starter::int)         AS starts,
        SUM(pb.wins)                    AS w,
        SUM(pb.losses)                  AS l,
        SUM(pb.saves)                   AS sv,
        SUM(pb.holds)                   AS hld,
        SUM(pb.innings_pitched)         AS ip,
        SUM(pb.batters_faced)           AS bf,
        SUM(pb.hits)                    AS hits_allowed,
        SUM(pb.earned_runs)             AS er,
        SUM(pb.home_runs)               AS hr_allowed,
        SUM(pb.strike_outs)             AS k,
        SUM(pb.walks + pb.intentional_walks) AS bb,
        SUM(pb.hit_by_pitch)            AS hbp,
        SUM(pb.number_of_pitches)       AS number_of_pitches,
        SUM(pb.wild_pitches)            AS wild_pitches,
        SUM(pb.ground_outs)             AS ground_outs,
        SUM(pb.air_outs)                AS air_outs
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season = :season AND dg.game_type = 'R'
    GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand, dg.season
    HAVING SUM(pb.batters_faced) >= 1
    ORDER BY SUM(pb.innings_pitched) DESC
    """
    logger.info("Fetching pitcher traditional stats for %d", season)
    df = read_sql(query, {"season": season})

    ip = df["ip"].replace(0, float("nan"))

    df["era"] = ((df["er"] / ip) * 9).round(2)
    df["whip"] = ((df["hits_allowed"] + df["bb"]) / ip).round(2)
    df["k_per_9"] = ((df["k"] / ip) * 9).round(2)
    df["bb_per_9"] = ((df["bb"] / ip) * 9).round(2)
    df["hr_per_9"] = ((df["hr_allowed"] / ip) * 9).round(2)
    df["k_per_bb"] = (df["k"] / df["bb"].replace(0, float("nan"))).round(2)

    # FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + cFIP
    # cFIP ≈ 3.20 (league constant, varies slightly by year)
    cfip = 3.20
    df["fip"] = (
        ((13 * df["hr_allowed"] + 3 * (df["bb"] + df["hbp"]) - 2 * df["k"]) / ip)
        + cfip
    ).round(2)

    df["go_ao"] = (
        df["ground_outs"] / df["air_outs"].replace(0, float("nan"))
    ).round(2)

    return df


# ---------------------------------------------------------------------------
# 29. Hitter aggressiveness profile
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
# 30. Pitcher efficiency profile
# ---------------------------------------------------------------------------
def get_pitcher_efficiency(season: int) -> pd.DataFrame:
    """Per-pitcher efficiency metrics for a season.

    Combines first-pitch strike rate, zone rate, ahead-after-one rate,
    putaway rate (K% on 2-strike counts), and pitches per PA.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, bf,
        pitches_per_pa, first_strike_pct, zone_pct, ahead_after_1_pct,
        putaway_rate.
    """
    query = """
    WITH pa_counts AS (
        SELECT
            fpa.pitcher_id,
            COUNT(DISTINCT fpa.pa_id) AS bf,
            ROUND(AVG(fpa.last_pitch_number)::numeric, 2) AS pitches_per_pa
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY fpa.pitcher_id
    ),
    first_pitches AS (
        SELECT
            fp.pitcher_id,
            COUNT(*)  AS total_first,
            SUM(CASE WHEN fp.is_called_strike OR fp.is_whiff OR fp.is_foul
                     THEN 1 ELSE 0 END) AS first_strikes
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_number = 1 AND fp.balls = 0 AND fp.strikes = 0
        GROUP BY fp.pitcher_id
    ),
    zone_agg AS (
        SELECT
            fp.pitcher_id,
            COUNT(*)                                        AS total_with_zone,
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS in_zone
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.zone IS NOT NULL
        GROUP BY fp.pitcher_id
    ),
    ahead_after_1 AS (
        SELECT
            fp.pitcher_id,
            COUNT(DISTINCT fp.pa_id) AS total_pa,
            COUNT(DISTINCT CASE WHEN fp.is_called_strike OR fp.is_whiff OR fp.is_foul
                                THEN fp.pa_id END) AS ahead_pa
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.pitch_number = 1 AND fp.balls = 0 AND fp.strikes = 0
        GROUP BY fp.pitcher_id
    ),
    putaway AS (
        SELECT
            fp.pitcher_id,
            COUNT(DISTINCT fp.pa_id) AS pa_reached_2strikes,
            COUNT(DISTINCT CASE WHEN fpa.events IN ('strikeout', 'strikeout_double_play')
                                THEN fp.pa_id END) AS pa_k
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.fact_pa fpa ON fp.pa_id = fpa.pa_id
        WHERE dg.season = :season AND dg.game_type = 'R'
          AND fp.strikes = 2
        GROUP BY fp.pitcher_id
    )
    SELECT
        pc.pitcher_id,
        COALESCE(dp.player_name, 'Unknown') AS pitcher_name,
        dp.pitch_hand,
        :season AS season,
        pc.bf,
        pc.pitches_per_pa,
        ROUND((fpch.first_strikes::numeric / NULLIF(fpch.total_first, 0)), 3) AS first_strike_pct,
        ROUND((za.in_zone::numeric / NULLIF(za.total_with_zone, 0)), 3) AS zone_pct,
        ROUND((aa.ahead_pa::numeric / NULLIF(aa.total_pa, 0)), 3) AS ahead_after_1_pct,
        ROUND((pu.pa_k::numeric / NULLIF(pu.pa_reached_2strikes, 0)), 3) AS putaway_rate
    FROM pa_counts pc
    LEFT JOIN production.dim_player dp ON pc.pitcher_id = dp.player_id
    LEFT JOIN first_pitches fpch ON pc.pitcher_id = fpch.pitcher_id
    LEFT JOIN zone_agg za ON pc.pitcher_id = za.pitcher_id
    LEFT JOIN ahead_after_1 aa ON pc.pitcher_id = aa.pitcher_id
    LEFT JOIN putaway pu ON pc.pitcher_id = pu.pitcher_id
    ORDER BY pc.bf DESC
    """
    logger.info("Fetching pitcher efficiency for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Batter game logs (per-game boxscore lines)
# ---------------------------------------------------------------------------
def get_batter_game_logs(season: int) -> pd.DataFrame:
    """Per-game batter lines from staging boxscores.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, batter_id, batter_name, season,
        strikeouts, walks, hits, home_runs, plate_appearances,
        at_bats, team_id.
    """
    query = """
    SELECT
        bb.game_pk,
        bb.batter_id,
        bb.batter_name,
        dg.season,
        bb.strikeouts,
        bb.walks,
        bb.hits,
        bb.home_runs,
        bb.plate_appearances,
        bb.at_bats,
        bb.team_id
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
    ORDER BY bb.game_pk, bb.batter_id
    """
    logger.info("Fetching batter game logs for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Game-level batter stats (per pitcher-batter within a game) — extended
# ---------------------------------------------------------------------------
def get_game_batter_stats(season: int) -> pd.DataFrame:
    """Per (game_pk, pitcher_id, batter_id) PA and outcome counts.

    Extends get_game_batter_ks() with walks, hits, and home runs.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, batter_id, pa, k, bb, h, hr.
    """
    query = """
    SELECT
        fpa.game_pk,
        fpa.pitcher_id,
        fpa.batter_id,
        COUNT(*) AS pa,
        SUM(CASE WHEN fpa.events IN ('strikeout', 'strikeout_double_play')
                 THEN 1 ELSE 0 END) AS k,
        SUM(CASE WHEN fpa.events = 'walk'
                 THEN 1 ELSE 0 END) AS bb,
        SUM(CASE WHEN fpa.events IN ('single', 'double', 'triple', 'home_run')
                 THEN 1 ELSE 0 END) AS h,
        SUM(CASE WHEN fpa.events = 'home_run'
                 THEN 1 ELSE 0 END) AS hr
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    WHERE dg.season = :season
      AND dg.game_type = 'R'
      AND fpa.events IS NOT NULL
    GROUP BY fpa.game_pk, fpa.pitcher_id, fpa.batter_id
    ORDER BY fpa.game_pk, fpa.pitcher_id, fpa.batter_id
    """
    logger.info("Fetching game batter stats for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# MiLB data queries
# ---------------------------------------------------------------------------
def get_milb_batter_season_totals(season: int) -> pd.DataFrame:
    """Aggregate MiLB batting game logs into season totals per player per level.

    Parameters
    ----------
    season : int
        MiLB season year.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, level) with counting stats and rates.
    """
    query = """
    SELECT
        batter_id      AS player_id,
        batter_name    AS player_name,
        level,
        sport_id,
        season,
        SUM(plate_appearances) AS pa,
        SUM(at_bats)           AS ab,
        SUM(hits)              AS h,
        SUM(doubles)           AS "2b",
        SUM(triples)           AS "3b",
        SUM(home_runs)         AS hr,
        SUM(strikeouts)        AS k,
        SUM(walks)             AS bb,
        SUM(hit_by_pitch)      AS hbp,
        SUM(sb)                AS sb,
        SUM(caught_stealing)   AS cs,
        SUM(total_bases)       AS tb,
        SUM(runs)              AS r,
        SUM(rbi)               AS rbi,
        SUM(ground_outs)       AS ground_outs,
        SUM(air_outs)          AS air_outs,
        COUNT(DISTINCT game_pk) AS games
    FROM staging.milb_batting_game_logs
    WHERE season = :season
      AND level IN ('AAA', 'AA', 'A+', 'A')
    GROUP BY batter_id, batter_name, level, sport_id, season
    ORDER BY SUM(plate_appearances) DESC
    """
    logger.info("Fetching MiLB batter season totals for %d", season)
    return read_sql(query, {"season": season})


def get_milb_pitcher_season_totals(season: int) -> pd.DataFrame:
    """Aggregate MiLB pitching game logs into season totals per player per level.

    Parameters
    ----------
    season : int
        MiLB season year.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, level) with counting stats and rates.
    """
    query = """
    SELECT
        pitcher_id      AS player_id,
        pitcher_name    AS player_name,
        level,
        sport_id,
        season,
        SUM(batters_faced)    AS bf,
        SUM(strike_outs)      AS k,
        SUM(walks)            AS bb,
        SUM(hits)             AS h,
        SUM(home_runs)        AS hr,
        SUM(earned_runs)      AS er,
        SUM(runs)             AS r,
        SUM(innings_pitched)  AS ip,
        SUM(number_of_pitches) AS pitches,
        SUM(wins)             AS w,
        SUM(losses)           AS l,
        SUM(saves)            AS sv,
        SUM(holds)            AS hld,
        COUNT(DISTINCT game_pk) AS games,
        SUM(CASE WHEN is_starter THEN 1 ELSE 0 END) AS games_started,
        SUM(ground_outs)       AS ground_outs,
        SUM(air_outs)          AS air_outs,
        SUM(fly_outs)          AS fly_outs
    FROM staging.milb_pitching_game_logs
    WHERE season = :season
      AND level IN ('AAA', 'AA', 'A+', 'A')
    GROUP BY pitcher_id, pitcher_name, level, sport_id, season
    ORDER BY SUM(batters_faced) DESC
    """
    logger.info("Fetching MiLB pitcher season totals for %d", season)
    return read_sql(query, {"season": season})


def get_tto_adjustment_profiles(
    seasons: list[int],
    min_pa_per_tto: int = 30,
) -> pd.DataFrame:
    """Per-pitcher TTO (times-through-order) adjustment factors.

    For each pitcher-season with sufficient PA in each TTO bucket (1st, 2nd,
    3rd+), compute the logit-scale lift relative to their own overall K/BB/HR
    rate.  The result is used inside the MC game simulator to apply different
    rates for each TTO block.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include (e.g. [2018, 2019, ..., 2025]).
    min_pa_per_tto : int
        Minimum PA in *each* TTO bucket for a pitcher-season to be included.
        Default 30 ensures reasonable reliability.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, tto (1/2/3),
        k_rate, bb_rate, hr_rate, overall_k_rate, overall_bb_rate,
        overall_hr_rate, pa_count.
        One row per (pitcher_id, season, tto).
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH pitcher_tto AS (
        SELECT
            fp.pitcher_id,
            dg.season,
            LEAST(fp.times_through_order, 3) AS tto,
            COUNT(*)                           AS pa_count,
            AVG(CASE WHEN fp.events = 'strikeout' THEN 1.0 ELSE 0.0 END) AS k_rate,
            AVG(CASE WHEN fp.events = 'walk'      THEN 1.0 ELSE 0.0 END) AS bb_rate,
            AVG(CASE WHEN fp.events = 'home_run'  THEN 1.0 ELSE 0.0 END) AS hr_rate
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fp.events IS NOT NULL
        GROUP BY fp.pitcher_id, dg.season, LEAST(fp.times_through_order, 3)
    ),
    pitcher_overall AS (
        SELECT
            fp.pitcher_id,
            dg.season,
            AVG(CASE WHEN fp.events = 'strikeout' THEN 1.0 ELSE 0.0 END) AS overall_k_rate,
            AVG(CASE WHEN fp.events = 'walk'      THEN 1.0 ELSE 0.0 END) AS overall_bb_rate,
            AVG(CASE WHEN fp.events = 'home_run'  THEN 1.0 ELSE 0.0 END) AS overall_hr_rate
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fp.events IS NOT NULL
        GROUP BY fp.pitcher_id, dg.season
    ),
    qualified AS (
        -- Only keep pitcher-seasons where all 3 TTO buckets meet min PA
        SELECT pitcher_id, season
        FROM pitcher_tto
        GROUP BY pitcher_id, season
        HAVING COUNT(DISTINCT tto) = 3
           AND MIN(pa_count) >= {min_pa_per_tto}
    )
    SELECT
        pt.pitcher_id,
        pt.season,
        pt.tto,
        pt.k_rate,
        pt.bb_rate,
        pt.hr_rate,
        po.overall_k_rate,
        po.overall_bb_rate,
        po.overall_hr_rate,
        pt.pa_count
    FROM pitcher_tto pt
    JOIN pitcher_overall po
      ON pt.pitcher_id = po.pitcher_id AND pt.season = po.season
    JOIN qualified q
      ON pt.pitcher_id = q.pitcher_id AND pt.season = q.season
    ORDER BY pt.pitcher_id, pt.season, pt.tto
    """
    logger.info(
        "Fetching TTO adjustment profiles for seasons %s (min_pa=%d)",
        seasons, min_pa_per_tto,
    )
    return read_sql(query)


def get_prospect_info() -> pd.DataFrame:
    """Fetch enriched prospect metadata from dim_prospects.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, full_name, primary_position, bat_side, pitch_hand,
        birth_date, current_age, parent_org_name, level, mlb_debut_date,
        draft_year, season
    """
    query = """
    SELECT
        player_id,
        full_name,
        primary_position,
        bat_side,
        pitch_hand,
        birth_date,
        current_age,
        parent_org_name,
        level,
        mlb_debut_date,
        draft_year,
        season
    FROM production.dim_prospects
    ORDER BY player_id, season
    """
    logger.info("Fetching prospect info from dim_prospects")
    return read_sql(query)


# ---------------------------------------------------------------------------
# 30. Days rest for starting pitchers
# ---------------------------------------------------------------------------
def get_days_rest(seasons: list[int] | None = None) -> pd.DataFrame:
    """Compute days since previous start for each pitcher start.

    Uses ``fact_player_game_mlb`` with ``pit_is_starter = true`` and
    ``pit_bf >= 9`` to filter out openers / bullpen games.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.  Defaults to 2018-2025.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, game_pk, game_date, season, days_rest,
        rest_bucket, pit_k, pit_bb, pit_bf, pit_ip.
        ``rest_bucket`` is one of ``'short'``, ``'normal'``, ``'extended'``.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH starter_games AS (
        SELECT
            fpg.player_id  AS pitcher_id,
            fpg.game_pk,
            fpg.game_date,
            fpg.season,
            fpg.pit_k,
            fpg.pit_bb,
            fpg.pit_bf,
            fpg.pit_ip,
            LAG(fpg.game_date) OVER (
                PARTITION BY fpg.player_id ORDER BY fpg.game_date
            ) AS prev_start_date
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND fpg.pit_bf >= 9
          AND fpg.pit_ip > 0
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
    )
    SELECT
        pitcher_id,
        game_pk,
        game_date,
        season,
        (game_date - prev_start_date) AS days_rest,
        CASE
            WHEN (game_date - prev_start_date) <= 4 THEN 'short'
            WHEN (game_date - prev_start_date) = 5  THEN 'normal'
            ELSE 'extended'
        END AS rest_bucket,
        pit_k,
        pit_bb,
        pit_bf,
        pit_ip
    FROM starter_games
    WHERE prev_start_date IS NOT NULL
      AND (game_date - prev_start_date) BETWEEN 3 AND 14
    ORDER BY pitcher_id, game_date
    """
    logger.info("Fetching days rest for seasons %s", seasons)
    return read_sql(query)


def get_rest_bucket_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """League-average K%, BB%, and BF by rest bucket.

    Empirical baseline for computing rest-based adjustments.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.  Defaults to 2018-2025.

    Returns
    -------
    pd.DataFrame
        Columns: rest_bucket, n_starts, avg_k_rate, avg_bb_rate,
        avg_bf, std_bf, avg_ip.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH starter_games AS (
        SELECT
            fpg.player_id  AS pitcher_id,
            fpg.game_pk,
            fpg.game_date,
            fpg.season,
            fpg.pit_k,
            fpg.pit_bb,
            fpg.pit_bf,
            fpg.pit_ip,
            LAG(fpg.game_date) OVER (
                PARTITION BY fpg.player_id ORDER BY fpg.game_date
            ) AS prev_start_date
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND fpg.pit_bf >= 9
          AND fpg.pit_ip > 0
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
    ),
    with_rest AS (
        SELECT *,
            (game_date - prev_start_date) AS days_rest
        FROM starter_games
        WHERE prev_start_date IS NOT NULL
    ),
    bucketed AS (
        SELECT *,
            CASE
                WHEN days_rest <= 4 THEN 'short'
                WHEN days_rest = 5  THEN 'normal'
                ELSE 'extended'
            END AS rest_bucket
        FROM with_rest
        WHERE days_rest BETWEEN 3 AND 14
    )
    SELECT
        rest_bucket,
        COUNT(*)                                              AS n_starts,
        ROUND(AVG(pit_k::numeric / NULLIF(pit_bf, 0)), 4)    AS avg_k_rate,
        ROUND(AVG(pit_bb::numeric / NULLIF(pit_bf, 0)), 4)   AS avg_bb_rate,
        ROUND(AVG(pit_bf::numeric), 2)                        AS avg_bf,
        ROUND(STDDEV(pit_bf::numeric), 2)                     AS std_bf,
        ROUND(AVG(pit_ip::numeric), 2)                        AS avg_ip
    FROM bucketed
    GROUP BY rest_bucket
    ORDER BY rest_bucket
    """
    logger.info("Fetching rest bucket stats for seasons %s", seasons)
    return read_sql(query)


# ---------------------------------------------------------------------------
# Catcher framing effects
# ---------------------------------------------------------------------------
def get_catcher_framing_effects(
    seasons: list[int] | None = None,
    shrinkage_k: int = 500,
) -> pd.DataFrame:
    """Compute per-catcher called-strike-rate lift on taken pitches.

    For each catcher-season, computes the called-strike rate on non-swing
    pitches, applies empirical Bayes shrinkage, and returns a logit-scale
    lift relative to the season league average.

    Join path: fact_lineup (position='C', starter) -> fact_player_game_mlb
    (pitcher team_id) -> fact_pitch (is_swing=false).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2018-2025.
    shrinkage_k : int
        Shrinkage constant in taken-pitch units.  ``shrunk_rate =
        (n * observed + k * league_avg) / (n + k)``.  Default 500
        provides heavy shrinkage for low-sample catchers.

    Returns
    -------
    pd.DataFrame
        Columns: catcher_id, season, pitches_received, called_strikes,
        called_strike_rate_raw, league_cs_rate, called_strike_rate_shrunk,
        logit_lift.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH catcher_game AS (
        SELECT fl.game_pk, fl.player_id AS catcher_id,
               fl.team_id, fl.season
        FROM production.fact_lineup fl
        WHERE fl.position = 'C'
          AND fl.is_starter = true
          AND fl.season IN ({season_list})
    ),
    catcher_pitch AS (
        SELECT
            cg.catcher_id,
            cg.season,
            COUNT(*)                       AS pitches_received,
            SUM(fp.is_called_strike::int)  AS called_strikes
        FROM catcher_game cg
        JOIN production.fact_player_game_mlb pg
            ON pg.game_pk = cg.game_pk
           AND pg.team_id = cg.team_id
        JOIN production.fact_pitch fp
            ON fp.game_pk = cg.game_pk
           AND fp.pitcher_id = pg.player_id
        JOIN production.dim_game dg
            ON fp.game_pk = dg.game_pk
        WHERE fp.is_swing = false
          AND dg.game_type = 'R'
          AND pg.player_role = 'pitcher'
        GROUP BY cg.catcher_id, cg.season
    )
    SELECT
        catcher_id,
        season,
        pitches_received,
        called_strikes,
        ROUND(called_strikes::numeric
              / NULLIF(pitches_received, 0), 5) AS called_strike_rate_raw
    FROM catcher_pitch
    ORDER BY season, pitches_received DESC
    """
    logger.info(
        "Fetching catcher framing effects for seasons %s (shrinkage_k=%d)",
        seasons, shrinkage_k,
    )
    df = read_sql(query, {})

    if df.empty:
        return df

    import numpy as _np
    from scipy.special import logit as _logit

    _clip = lambda x: _np.clip(x, 1e-6, 1 - 1e-6)  # noqa: E731

    # League-average called-strike rate per season (weighted by pitches)
    season_league = (
        df.groupby("season")
        .apply(
            lambda g: g["called_strikes"].sum() / g["pitches_received"].sum(),
            include_groups=False,
        )
        .rename("league_cs_rate")
    )
    df = df.merge(season_league, on="season", how="left")

    # Shrinkage toward season league average
    n = df["pitches_received"].values.astype(float)
    obs = df["called_strike_rate_raw"].values.astype(float)
    lg = df["league_cs_rate"].values.astype(float)
    df["called_strike_rate_shrunk"] = (
        (n * obs + shrinkage_k * lg) / (n + shrinkage_k)
    ).round(5)

    # Logit lift relative to league average
    df["logit_lift"] = (
        _logit(_clip(df["called_strike_rate_shrunk"].values))
        - _logit(_clip(lg))
    ).round(5)

    logger.info(
        "Catcher framing: %d catcher-seasons, lift range=[%.4f, %.4f]",
        len(df), df["logit_lift"].min(), df["logit_lift"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Pitcher pitch-type run values
# ---------------------------------------------------------------------------
def get_pitcher_run_values(season: int) -> pd.DataFrame:
    """Per-pitcher pitch-type run values with a weighted composite.

    Returns individual pitch-type rows plus a per-pitcher
    ``weighted_rv_per_100`` column (usage-weighted run value per 100
    pitches across all pitch types).

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitch_type, run_value_per_100, usage_pct,
        weighted_rv_per_100.
    """
    query = """
    WITH raw AS (
        SELECT
            pitcher_id,
            pitch_type,
            run_value_per_100,
            usage_pct
        FROM production.fact_pitch_type_run_value
        WHERE season = :season
    ),
    weighted AS (
        SELECT
            pitcher_id,
            SUM(run_value_per_100 * usage_pct) AS weighted_rv_per_100
        FROM raw
        GROUP BY pitcher_id
    )
    SELECT
        r.pitcher_id,
        r.pitch_type,
        r.run_value_per_100,
        r.usage_pct,
        w.weighted_rv_per_100
    FROM raw r
    JOIN weighted w ON r.pitcher_id = w.pitcher_id
    ORDER BY r.pitcher_id, r.usage_pct DESC
    """
    logger.info("Fetching pitcher run values for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Prospect transitions
# ---------------------------------------------------------------------------
def get_prospect_transitions(player_ids: list[int]) -> pd.DataFrame:
    """Fetch prospect transition records (promotions, demotions, etc.).

    Parameters
    ----------
    player_ids : list[int]
        Player IDs to retrieve transitions for.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, event_date, from_level, to_level,
        from_sport_id, to_sport_id, from_team_name, to_team_name,
        transition_type, season.
    """
    if not player_ids:
        return pd.DataFrame()

    placeholders = ", ".join(str(int(pid)) for pid in player_ids)
    query = f"""
    SELECT
        player_id,
        event_date,
        from_level,
        to_level,
        from_sport_id,
        to_sport_id,
        from_team_name,
        to_team_name,
        transition_type,
        season
    FROM production.fact_prospect_transition
    WHERE player_id IN ({placeholders})
    ORDER BY player_id, event_date
    """
    logger.info("Fetching prospect transitions for %d players", len(player_ids))
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# Game Simulator — pitch count features
# ---------------------------------------------------------------------------
def get_pitcher_pitch_count_features(seasons: list[int]) -> pd.DataFrame:
    """Per-pitcher-season pitch count efficiency features.

    Computes putaway rate (whiff% on 2-strike counts), pitches per PA,
    and foul ball rate (fouls / swings) for use in the game simulator's
    pitch count model.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, pitches_per_pa, putaway_rate,
        foul_rate, total_pa.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH pitcher_pa_pitches AS (
        SELECT
            fpa.pitcher_id,
            dg.season,
            COUNT(*)                  AS total_pa,
            AVG(fpa.last_pitch_number) AS pitches_per_pa
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY fpa.pitcher_id, dg.season
    ),
    pitcher_pitch_agg AS (
        SELECT
            fp.pitcher_id,
            dg.season,
            SUM(CASE WHEN fp.strikes = 2 AND fp.is_swing THEN 1 ELSE 0 END)
                AS two_strike_swings,
            SUM(CASE WHEN fp.strikes = 2 AND fp.is_whiff THEN 1 ELSE 0 END)
                AS two_strike_whiffs,
            SUM(fp.is_swing::int)  AS total_swings,
            SUM(fp.is_foul::int)   AS total_fouls
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id, dg.season
    )
    SELECT
        pp.pitcher_id,
        pp.season,
        ROUND(pp.pitches_per_pa::numeric, 3)       AS pitches_per_pa,
        CASE WHEN pa.two_strike_swings > 0
             THEN ROUND(pa.two_strike_whiffs::numeric
                         / pa.two_strike_swings, 4)
             ELSE NULL END                           AS putaway_rate,
        CASE WHEN pa.total_swings > 0
             THEN ROUND(pa.total_fouls::numeric
                         / pa.total_swings, 4)
             ELSE NULL END                           AS foul_rate,
        pp.total_pa
    FROM pitcher_pa_pitches pp
    LEFT JOIN pitcher_pitch_agg pa
      ON pp.pitcher_id = pa.pitcher_id AND pp.season = pa.season
    ORDER BY pp.pitcher_id, pp.season
    """
    logger.info(
        "Fetching pitcher pitch count features for seasons %s", seasons
    )
    return read_sql(query)


def get_batter_pitch_count_features(seasons: list[int]) -> pd.DataFrame:
    """Per-batter-season plate discipline features for pitch count model.

    Computes contact rate, foul rate, Z-contact%, O-contact%, and
    pitches per PA to model how many pitches each batter consumes.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, pitches_per_pa, contact_rate,
        foul_rate, z_contact_rate, o_contact_rate, total_pa.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH batter_pa_pitches AS (
        SELECT
            fpa.batter_id,
            dg.season,
            COUNT(*)                  AS total_pa,
            AVG(fpa.last_pitch_number) AS pitches_per_pa
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY fpa.batter_id, dg.season
    ),
    batter_pitch_agg AS (
        SELECT
            fp.batter_id,
            dg.season,
            SUM(fp.is_swing::int)      AS total_swings,
            SUM(fp.is_foul::int)       AS total_fouls,
            SUM(CASE WHEN fp.is_swing AND NOT fp.is_whiff
                     THEN 1 ELSE 0 END) AS contacts,
            -- Zone swings and contacts (zones 1-9 = strike zone)
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing
                     THEN 1 ELSE 0 END) AS z_swings,
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing
                          AND NOT fp.is_whiff
                     THEN 1 ELSE 0 END) AS z_contacts,
            -- Outside zone swings and contacts
            SUM(CASE WHEN (fp.zone < 1 OR fp.zone > 9) AND fp.is_swing
                     THEN 1 ELSE 0 END) AS o_swings,
            SUM(CASE WHEN (fp.zone < 1 OR fp.zone > 9) AND fp.is_swing
                          AND NOT fp.is_whiff
                     THEN 1 ELSE 0 END) AS o_contacts
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.batter_id, dg.season
    )
    SELECT
        bp.batter_id,
        bp.season,
        ROUND(bp.pitches_per_pa::numeric, 3) AS pitches_per_pa,
        CASE WHEN pa.total_swings > 0
             THEN ROUND(pa.contacts::numeric / pa.total_swings, 4)
             ELSE NULL END                    AS contact_rate,
        CASE WHEN pa.total_swings > 0
             THEN ROUND(pa.total_fouls::numeric / pa.total_swings, 4)
             ELSE NULL END                    AS foul_rate,
        CASE WHEN pa.z_swings > 0
             THEN ROUND(pa.z_contacts::numeric / pa.z_swings, 4)
             ELSE NULL END                    AS z_contact_rate,
        CASE WHEN pa.o_swings > 0
             THEN ROUND(pa.o_contacts::numeric / pa.o_swings, 4)
             ELSE NULL END                    AS o_contact_rate,
        bp.total_pa
    FROM batter_pa_pitches bp
    LEFT JOIN batter_pitch_agg pa
      ON bp.batter_id = pa.batter_id AND bp.season = pa.season
    ORDER BY bp.batter_id, bp.season
    """
    logger.info(
        "Fetching batter pitch count features for seasons %s", seasons
    )
    return read_sql(query)


# ---------------------------------------------------------------------------
# Game Simulator — exit model training data
# ---------------------------------------------------------------------------
def get_exit_model_training_data(seasons: list[int]) -> pd.DataFrame:
    """Build training data for the pitcher exit model.

    One row per PA for every starting pitcher, with cumulative game state
    features and a binary label indicating if this was the pitcher's
    last PA in the game.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, game_pk, season, team_id,
        pitcher_pa_number, cumulative_pitches, inning, outs_when_up,
        score_diff, events, is_last_pa.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH starter_games AS (
        -- Identify starter games and their total PA count
        SELECT
            player_id AS pitcher_id,
            game_pk,
            season,
            team_id,
            pit_bf AS total_bf,
            pit_pitches AS total_pitches
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = TRUE
          AND season IN ({season_list})
          AND pit_bf >= 3
    ),
    pa_with_cumulative AS (
        SELECT
            fpa.pitcher_id,
            fpa.game_pk,
            sg.season,
            sg.team_id,
            fpa.pitcher_pa_number,
            -- Cumulative pitch count: sum of last_pitch_number up to
            -- and including this PA
            SUM(fpa.last_pitch_number) OVER (
                PARTITION BY fpa.pitcher_id, fpa.game_pk
                ORDER BY fpa.game_counter
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_pitches,
            fpa.inning,
            fpa.outs_when_up,
            fpa.bat_score_diff,
            fpa.events,
            -- Is this the pitcher's last PA in the game?
            CASE WHEN fpa.pitcher_pa_number = sg.total_bf
                 THEN 1 ELSE 0 END AS is_last_pa
        FROM production.fact_pa fpa
        JOIN starter_games sg
          ON fpa.pitcher_id = sg.pitcher_id
         AND fpa.game_pk = sg.game_pk
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND fpa.events IS NOT NULL
    )
    SELECT
        pitcher_id,
        game_pk,
        season,
        team_id,
        pitcher_pa_number,
        cumulative_pitches,
        inning,
        outs_when_up,
        -- Flip sign: bat_score_diff is from batter perspective,
        -- we want pitcher's team perspective
        -bat_score_diff AS score_diff,
        events,
        is_last_pa
    FROM pa_with_cumulative
    ORDER BY pitcher_id, game_pk, pitcher_pa_number
    """
    logger.info(
        "Fetching exit model training data for seasons %s", seasons
    )
    return read_sql(query)


def get_pitcher_exit_tendencies(seasons: list[int]) -> pd.DataFrame:
    """Per-pitcher and per-team average exit pitch counts.

    Used as features in the exit model to capture pitcher stamina
    and manager pull tendencies.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, team_id, n_starts, avg_pitches,
        std_pitches, avg_bf, team_avg_pitches.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH pitcher_starts AS (
        SELECT
            player_id AS pitcher_id,
            season,
            team_id,
            pit_pitches,
            pit_bf
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = TRUE
          AND season IN ({season_list})
          AND pit_bf >= 3
    ),
    pitcher_agg AS (
        SELECT
            pitcher_id,
            season,
            MODE() WITHIN GROUP (ORDER BY team_id) AS team_id,
            COUNT(*)            AS n_starts,
            AVG(pit_pitches)    AS avg_pitches,
            STDDEV(pit_pitches) AS std_pitches,
            AVG(pit_bf)         AS avg_bf
        FROM pitcher_starts
        GROUP BY pitcher_id, season
    ),
    team_agg AS (
        SELECT
            team_id,
            season,
            AVG(pit_pitches) AS team_avg_pitches
        FROM pitcher_starts
        GROUP BY team_id, season
    )
    SELECT
        pa.pitcher_id,
        pa.season,
        pa.team_id,
        pa.n_starts,
        ROUND(pa.avg_pitches::numeric, 1)  AS avg_pitches,
        ROUND(pa.std_pitches::numeric, 1)  AS std_pitches,
        ROUND(pa.avg_bf::numeric, 1)       AS avg_bf,
        ROUND(ta.team_avg_pitches::numeric, 1) AS team_avg_pitches
    FROM pitcher_agg pa
    LEFT JOIN team_agg ta
      ON pa.team_id = ta.team_id AND pa.season = ta.season
    ORDER BY pa.pitcher_id, pa.season
    """
    logger.info(
        "Fetching pitcher exit tendencies for seasons %s", seasons
    )
    return read_sql(query)


def get_pa_outcome_distribution(seasons: list[int]) -> pd.DataFrame:
    """League-wide PA outcome distribution by TTO.

    Returns the probability of each outcome type (K, BB, HBP, 1B, 2B,
    3B, HR, out) for each TTO bucket, used as baseline rates in the
    PA outcome model.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: tto, total_pa, k, bb, hbp, single, double, triple,
        hr, field_out, k_rate, bb_rate, hbp_rate, single_rate,
        double_rate, triple_rate, hr_rate, out_rate.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    SELECT
        LEAST(fpa.times_through_order, 3)  AS tto,
        COUNT(*)                           AS total_pa,
        SUM(CASE WHEN fpa.events IN ('strikeout', 'strikeout_double_play')
                 THEN 1 ELSE 0 END)        AS k,
        SUM(CASE WHEN fpa.events IN ('walk', 'intent_walk')
                 THEN 1 ELSE 0 END)        AS bb,
        SUM(CASE WHEN fpa.events = 'hit_by_pitch'
                 THEN 1 ELSE 0 END)        AS hbp,
        SUM(CASE WHEN fpa.events = 'single'
                 THEN 1 ELSE 0 END)        AS single,
        SUM(CASE WHEN fpa.events = 'double'
                 THEN 1 ELSE 0 END)        AS double,
        SUM(CASE WHEN fpa.events = 'triple'
                 THEN 1 ELSE 0 END)        AS triple,
        SUM(CASE WHEN fpa.events = 'home_run'
                 THEN 1 ELSE 0 END)        AS hr,
        SUM(CASE WHEN fpa.events IN (
                 'field_out', 'force_out', 'grounded_into_double_play',
                 'double_play', 'fielders_choice', 'fielders_choice_out',
                 'sac_fly', 'sac_bunt', 'field_error',
                 'sac_fly_double_play', 'sac_bunt_double_play',
                 'triple_play', 'catcher_interf', 'truncated_pa')
                 THEN 1 ELSE 0 END)        AS field_out
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      AND dg.season IN ({season_list})
      AND fpa.events IS NOT NULL
    GROUP BY LEAST(fpa.times_through_order, 3)
    ORDER BY tto
    """
    df = read_sql(query)

    # Compute rates
    for col in ["k", "bb", "hbp", "single", "double", "triple", "hr", "field_out"]:
        df[f"{col}_rate"] = df[col] / df["total_pa"]

    # Combine into "out" rate (field_out captures everything not K/BB/HBP/hit)
    df["out_rate"] = df["field_out_rate"]

    logger.info(
        "Fetching PA outcome distributions for seasons %s", seasons
    )
    return df


def get_pitch_count_distribution(seasons: list[int]) -> pd.DataFrame:
    """Empirical distribution of pitches per PA.

    Returns the fraction of PAs that consumed each pitch count (1-15+),
    split by starter vs reliever context.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitch_count, total_pa, fraction.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    SELECT
        LEAST(fpa.last_pitch_number, 15) AS pitch_count,
        COUNT(*)                          AS total_pa
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      AND dg.season IN ({season_list})
      AND fpa.events IS NOT NULL
      AND fpa.last_pitch_number IS NOT NULL
    GROUP BY LEAST(fpa.last_pitch_number, 15)
    ORDER BY pitch_count
    """
    df = read_sql(query)
    df["fraction"] = df["total_pa"] / df["total_pa"].sum()
    logger.info(
        "Fetching pitch count distribution for seasons %s", seasons
    )
    return df


# ---------------------------------------------------------------------------
# Batter Game Simulator queries
# ---------------------------------------------------------------------------
def get_team_bullpen_rates(seasons: list[int]) -> pd.DataFrame:
    """Per-team-season aggregate reliever rates.

    Used as the "unknown pitcher" rates when a batter faces the bullpen.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, k_rate, bb_rate, hr_rate, total_bf.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    SELECT
        team_id,
        season,
        SUM(pit_k)::float   / NULLIF(SUM(pit_bf), 0) AS k_rate,
        SUM(pit_bb)::float  / NULLIF(SUM(pit_bf), 0) AS bb_rate,
        SUM(pit_hr)::float  / NULLIF(SUM(pit_bf), 0) AS hr_rate,
        SUM(pit_bf) AS total_bf
    FROM production.fact_player_game_mlb
    WHERE pit_is_starter = FALSE
      AND pit_bf >= 1
      AND season IN ({season_list})
    GROUP BY team_id, season
    ORDER BY team_id, season
    """
    logger.info("Fetching team bullpen rates for seasons %s", seasons)
    return read_sql(query)


def get_reliever_role_history(
    seasons: list[int],
    min_games: int = 10,
) -> pd.DataFrame:
    """Per-reliever-season aggregates for role classification.

    Aggregates saves, holds, blown saves, BF, K, BB, HR, HBP, H, runs,
    outs, pitches from ``fact_player_game_mlb`` for non-starters.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_games : int
        Minimum relief appearances per season to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, games, saves, holds, blown_saves,
        bf, k, bb, hr, h, runs, outs, pitches, k_rate, bb_rate.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    SELECT
        player_id AS pitcher_id,
        season,
        COUNT(*)                     AS games,
        SUM(pit_sv)                  AS saves,
        SUM(pit_hld)                 AS holds,
        SUM(pit_bs)                  AS blown_saves,
        SUM(pit_bf)                  AS bf,
        SUM(pit_k)                   AS k,
        SUM(pit_bb)                  AS bb,
        SUM(pit_hr)                  AS hr,
        SUM(pit_h)                   AS h,
        SUM(pit_r)                   AS runs,
        ROUND(SUM(pit_ip) * 3)::int  AS outs,
        SUM(pit_pitches)             AS pitches,
        SUM(pit_k)::float / NULLIF(SUM(pit_bf), 0) AS k_rate,
        SUM(pit_bb)::float / NULLIF(SUM(pit_bf), 0) AS bb_rate
    FROM production.fact_player_game_mlb
    WHERE pit_is_starter = FALSE
      AND pit_bf >= 1
      AND season IN ({season_list})
    GROUP BY player_id, season
    HAVING COUNT(*) >= {min_games}
    ORDER BY player_id, season
    """
    logger.info("Fetching reliever role history for seasons %s (min_games=%d)", seasons, min_games)
    return read_sql(query)


def get_batter_game_actuals(season: int) -> pd.DataFrame:
    """Per-batter game actuals with batting order and opposing starter.

    Used for batter simulator backtesting.

    Parameters
    ----------
    season : int
        Season to fetch.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, game_pk, game_date, season, team_id,
        batting_order, bat_pa, bat_k, bat_bb, bat_h, bat_hr,
        bat_2b, bat_3b, bat_tb, bat_r, bat_rbi, bat_hbp,
        opp_starter_id, opp_team_id.
    """
    query = """
    WITH starters AS (
        SELECT player_id AS pitcher_id, game_pk, team_id AS pitcher_team_id
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = TRUE
          AND season = :season
    )
    SELECT
        fg.player_id  AS batter_id,
        fg.game_pk,
        fg.game_date,
        fg.season,
        fg.team_id,
        fl.batting_order,
        fg.bat_pa,
        fg.bat_k,
        fg.bat_bb,
        fg.bat_h,
        fg.bat_hr,
        fg.bat_2b,
        fg.bat_3b,
        fg.bat_tb,
        fg.bat_r,
        fg.bat_rbi,
        fg.bat_hbp,
        st.pitcher_id AS opp_starter_id,
        st.pitcher_team_id AS opp_team_id
    FROM production.fact_player_game_mlb fg
    JOIN production.fact_lineup fl
      ON fg.player_id = fl.player_id AND fg.game_pk = fl.game_pk
    LEFT JOIN starters st
      ON fg.game_pk = st.game_pk AND st.pitcher_team_id != fg.team_id
    WHERE fg.player_role = 'batter'
      AND fl.is_starter = TRUE
      AND fg.season = :season
      AND fg.bat_pa >= 1
    ORDER BY fg.game_pk, fl.batting_order
    """
    logger.info("Fetching batter game actuals for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Lineup priors (position + batting order history)
# ---------------------------------------------------------------------------
def get_lineup_priors(season: int) -> pd.DataFrame:
    """Get each player's position and batting order history from fact_lineup.

    Returns one row per player-position with start counts and most common
    batting order at that position.  Team-agnostic — a player's position
    history carries over when they change teams.

    Parameters
    ----------
    season : int
        Season to pull lineup data from.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, starts, pct, is_primary,
        batting_order_mode, bo_games.
    """
    query = """
    WITH pos_starts AS (
        SELECT
            fl.player_id,
            fl.position,
            COUNT(*) AS starts
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fl.is_starter = true
          AND fl.position NOT IN ('P', 'PR', 'PH')
        GROUP BY fl.player_id, fl.position
    ),
    bo_starts AS (
        SELECT
            fl.player_id,
            fl.position,
            fl.batting_order,
            COUNT(*) AS bo_games,
            ROW_NUMBER() OVER (
                PARTITION BY fl.player_id, fl.position
                ORDER BY COUNT(*) DESC
            ) AS rn
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fl.is_starter = true
          AND fl.position NOT IN ('P', 'PR', 'PH')
        GROUP BY fl.player_id, fl.position, fl.batting_order
    )
    SELECT
        ps.player_id,
        ps.position,
        ps.starts,
        bs.batting_order AS batting_order_mode,
        bs.bo_games
    FROM pos_starts ps
    JOIN bo_starts bs
      ON ps.player_id = bs.player_id
     AND ps.position = bs.position
     AND bs.rn = 1
    ORDER BY ps.player_id, ps.starts DESC
    """
    logger.info("Fetching lineup priors for %d", season)
    df = read_sql(query, {"season": season})

    if df.empty:
        return df

    # Compute pct and is_primary in Python (same pattern as position eligibility)
    totals = df.groupby("player_id")["starts"].transform("sum")
    df["pct"] = df["starts"] / totals
    idx = df.groupby("player_id")["starts"].idxmax()
    df["is_primary"] = False
    df.loc[idx, "is_primary"] = True

    return df


def get_lineup_priors_by_hand(season: int) -> pd.DataFrame:
    """Get per-player position/batting-order history split by opposing pitcher hand.

    Same structure as ``get_lineup_priors`` but with an additional ``vs_hand``
    column ('L' or 'R') indicating the opposing starting pitcher's throwing hand.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, vs_hand, starts, batting_order_mode,
        bo_games, pct, is_primary.
    """
    query = """
    WITH opp_pitcher AS (
        -- Starting pitcher for each team in each game
        SELECT fl.game_pk, fl.team_id, dp.pitch_hand
        FROM production.fact_lineup fl
        JOIN production.dim_player dp ON fl.player_id = dp.player_id
        WHERE fl.position = 'P'
          AND fl.is_starter = true
    ),
    pos_starts AS (
        SELECT
            fl.player_id,
            fl.position,
            op.pitch_hand AS vs_hand,
            COUNT(*) AS starts
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        -- Join to opposing team's starting pitcher
        JOIN opp_pitcher op
          ON op.game_pk = fl.game_pk
         AND op.team_id != fl.team_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fl.is_starter = true
          AND fl.position NOT IN ('P', 'PR', 'PH')
          AND op.pitch_hand IN ('L', 'R')
        GROUP BY fl.player_id, fl.position, op.pitch_hand
    ),
    bo_starts AS (
        SELECT
            fl.player_id,
            fl.position,
            op.pitch_hand AS vs_hand,
            fl.batting_order,
            COUNT(*) AS bo_games,
            ROW_NUMBER() OVER (
                PARTITION BY fl.player_id, fl.position, op.pitch_hand
                ORDER BY COUNT(*) DESC
            ) AS rn
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        JOIN opp_pitcher op
          ON op.game_pk = fl.game_pk
         AND op.team_id != fl.team_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fl.is_starter = true
          AND fl.position NOT IN ('P', 'PR', 'PH')
          AND op.pitch_hand IN ('L', 'R')
        GROUP BY fl.player_id, fl.position, op.pitch_hand, fl.batting_order
    )
    SELECT
        ps.player_id,
        ps.position,
        ps.vs_hand,
        ps.starts,
        bs.batting_order AS batting_order_mode,
        bs.bo_games
    FROM pos_starts ps
    JOIN bo_starts bs
      ON ps.player_id = bs.player_id
     AND ps.position = bs.position
     AND ps.vs_hand = bs.vs_hand
     AND bs.rn = 1
    ORDER BY ps.player_id, ps.vs_hand, ps.starts DESC
    """
    logger.info("Fetching lineup priors by hand for %d", season)
    df = read_sql(query, {"season": season})

    if df.empty:
        return df

    # Compute pct and is_primary within each hand split
    totals = df.groupby(["player_id", "vs_hand"])["starts"].transform("sum")
    df["pct"] = df["starts"] / totals
    idx = df.groupby(["player_id", "vs_hand"])["starts"].idxmax()
    df["is_primary"] = False
    df.loc[idx, "is_primary"] = True

    return df


# ---------------------------------------------------------------------------
# PA-level outcomes (for Glicko-2 player ratings)
# ---------------------------------------------------------------------------

def get_pa_outcomes(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """PA-level data with woba_value for Glicko-2 rating computation.

    Joins ``fact_pa`` with ``sat_batted_balls`` (for BIP woba_value)
    and ``dim_game`` (for chronological ordering).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.  Defaults to all available.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, game_date, season, batter_id, pitcher_id,
        events, woba_value (NULL for non-BIP).
        Sorted by game_date, game_pk, at_bat_number.
    """
    where = "dg.game_type = 'R' AND fp.events IS NOT NULL"
    params: dict = {}
    if seasons:
        where += " AND dg.season = ANY(:seasons)"
        params["seasons"] = seasons

    query = f"""
    SELECT fp.game_pk, dg.game_date, dg.season,
           fp.batter_id, fp.pitcher_id, fp.events,
           sbb.woba_value
    FROM production.fact_pa fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    LEFT JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
    WHERE {where}
    ORDER BY dg.game_date, fp.game_pk, fp.game_counter
    """

    logger.info("Fetching PA outcomes for Glicko-2%s",
                f" (seasons {seasons})" if seasons else "")
    df = read_sql(query, params)
    logger.info("PA outcomes: %d rows, %d batters, %d pitchers",
                len(df),
                df["batter_id"].nunique() if not df.empty else 0,
                df["pitcher_id"].nunique() if not df.empty else 0)
    return df


# ---------------------------------------------------------------------------
# 42. Hitter breakout features (tiered: T1 boxscore 2000+, T2 Statcast 2018+,
#     T3 MiLB for young players)
# ---------------------------------------------------------------------------
def get_hitter_breakout_features(
    season: int,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Consolidated hitter feature set for breakout model training/scoring.

    Returns one row per qualified batter with three feature tiers:
    - **T1 (2000+):** boxscore rates (K%, BB%, ISO, BABIP, OPS, etc.)
    - **T2 (2018+):** Statcast advanced (barrel%, xwOBA, chase, EV, etc.)
    - **T3 (young players):** translated MiLB rates, prospect pedigree

    Tier 2/3 columns are NaN when data is unavailable.

    Parameters
    ----------
    season : int
        MLB season year.
    min_pa : int
        Minimum plate appearances to qualify.

    Returns
    -------
    pd.DataFrame
        Columns documented inline (30 features + metadata).
    """
    # --- T1: Boxscore base (2000+) ---
    box = read_sql("""
        SELECT
            bb.batter_id,
            dp.player_name  AS batter_name,
            dg.season,
            COUNT(DISTINCT bb.game_pk) AS games,
            SUM(bb.plate_appearances) AS pa,
            SUM(bb.at_bats)           AS ab,
            SUM(bb.hits)              AS hits,
            SUM(bb.doubles)           AS doubles,
            SUM(bb.triples)           AS triples,
            SUM(bb.home_runs)         AS hr,
            SUM(bb.walks)             AS bb,
            SUM(bb.intentional_walks) AS ibb,
            SUM(bb.hit_by_pitch)      AS hbp,
            SUM(bb.strikeouts)        AS k,
            SUM(bb.total_bases)       AS total_bases,
            SUM(bb.sb)                AS sb,
            SUM(bb.caught_stealing)   AS cs,
            SUM(bb.ground_outs)       AS ground_outs,
            SUM(bb.air_outs)          AS air_outs
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON bb.batter_id = dp.player_id
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY bb.batter_id, dp.player_name, dg.season
        HAVING SUM(bb.plate_appearances) >= :min_pa
    """, {"season": season, "min_pa": min_pa})

    if box.empty:
        logger.warning("No qualified hitters for breakout features (season=%d)", season)
        return pd.DataFrame()

    # Derived T1 rates
    ab = box["ab"].replace(0, np.nan)
    pa = box["pa"].replace(0, np.nan)
    box["k_pct"] = box["k"] / pa
    box["bb_pct"] = box["bb"] / pa
    box["avg"] = box["hits"] / ab
    box["slg"] = box["total_bases"] / ab
    box["iso"] = box["slg"] - box["avg"]
    box["hr_ab"] = box["hr"] / ab
    obp_den = (box["ab"] + box["bb"] + box["hbp"]).replace(0, np.nan)
    box["obp"] = (box["hits"] + box["bb"] + box["hbp"]) / obp_den
    box["ops"] = box["obp"] + box["slg"]
    box["go_ao"] = box["ground_outs"] / box["air_outs"].replace(0, np.nan)

    # BABIP from boxscores: (H - HR) / (AB - K - HR)
    babip_den = (box["ab"] - box["k"] - box["hr"]).replace(0, np.nan)
    box["babip"] = (box["hits"] - box["hr"]) / babip_den

    # est_wOBA from linear weights (works for all eras)
    non_ibb = box["bb"] - box["ibb"]
    singles = box["hits"] - box["doubles"] - box["triples"] - box["hr"]
    woba_num = (
        0.69 * non_ibb + 0.72 * box["hbp"]
        + 0.88 * singles + 1.27 * box["doubles"]
        + 1.62 * box["triples"] + 2.10 * box["hr"]
    )
    woba_den = (box["ab"] + non_ibb + box["hbp"]).replace(0, np.nan)
    box["est_woba"] = woba_num / woba_den

    # Demographics (age)
    demo = read_sql("""
        SELECT player_id AS batter_id,
               (:season - EXTRACT(YEAR FROM birth_date))::int AS age,
               bat_side AS batter_stand
        FROM production.dim_player
    """, {"season": season})
    box = box.merge(demo, on="batter_id", how="left")
    box["age"] = box["age"].fillna(28)

    # MLB service time (seasons with 50+ PA)
    svc = read_sql("""
        SELECT bb.batter_id, COUNT(DISTINCT dg.season) AS mlb_seasons
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.game_type = 'R' AND dg.season < :season
        GROUP BY bb.batter_id
        HAVING SUM(bb.plate_appearances) >= 50
    """, {"season": season})
    box = box.merge(svc, on="batter_id", how="left")
    box["mlb_seasons"] = box["mlb_seasons"].fillna(0).astype(int)

    # YoY deltas from prior season
    prior = season - 1 if season != 2021 else 2019  # skip 2020
    prior_df = read_sql("""
        SELECT bb.batter_id,
            SUM(bb.strikeouts)::float / NULLIF(SUM(bb.plate_appearances), 0)
                AS k_prior,
            SUM(bb.walks)::float / NULLIF(SUM(bb.plate_appearances), 0)
                AS bb_prior,
            (SUM(bb.total_bases)::float / NULLIF(SUM(bb.at_bats), 0))
            - (SUM(bb.hits)::float / NULLIF(SUM(bb.at_bats), 0))
                AS iso_prior
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.season = :prior AND dg.game_type = 'R'
        GROUP BY bb.batter_id
        HAVING SUM(bb.plate_appearances) >= 100
    """, {"prior": prior})
    box = box.merge(prior_df, on="batter_id", how="left")
    box["delta_k_pct"] = box["k_pct"] - box["k_prior"]
    box["delta_bb_pct"] = box["bb_pct"] - box["bb_prior"]
    box["delta_iso"] = box["iso"] - box["iso_prior"]
    box.drop(columns=["k_prior", "bb_prior", "iso_prior"],
             inplace=True, errors="ignore")

    # --- T2: Statcast advanced (2018+, NaN otherwise) ---
    try:
        adv = read_sql("""
            SELECT batter_id, woba, xwoba, wrc_plus, barrel_pct, hard_hit_pct
            FROM production.fact_batting_advanced
            WHERE season = :season AND pa >= :min_pa
        """, {"season": season, "min_pa": min_pa})
        if not adv.empty:
            box = box.merge(adv, on="batter_id", how="left")
            box["xwoba_minus_woba"] = box["xwoba"] - box["woba"]
    except Exception:
        logger.debug("No fact_batting_advanced for %d (pre-Statcast)", season)

    for col in ["woba", "xwoba", "wrc_plus", "barrel_pct", "hard_hit_pct",
                "xwoba_minus_woba"]:
        if col not in box.columns:
            box[col] = np.nan

    # Observed profile: whiff, chase, z_contact, exit velo
    try:
        obs = get_hitter_observed_profile(season)
        if not obs.empty:
            obs_cols = ["batter_id", "whiff_rate", "chase_rate",
                        "z_contact_pct", "avg_exit_velo"]
            obs_avail = [c for c in obs_cols if c in obs.columns]
            box = box.merge(obs[obs_avail], on="batter_id", how="left")
    except Exception:
        logger.debug("No observed profile for %d", season)

    for col in ["whiff_rate", "chase_rate", "z_contact_pct", "avg_exit_velo"]:
        if col not in box.columns:
            box[col] = np.nan

    # Sprint speed
    try:
        sprint = get_sprint_speed(season)
        if not sprint.empty:
            box = box.merge(
                sprint[["player_id", "sprint_speed"]].rename(
                    columns={"player_id": "batter_id"}),
                on="batter_id", how="left",
            )
    except Exception:
        pass
    if "sprint_speed" not in box.columns:
        box["sprint_speed"] = np.nan

    # OAA
    try:
        oaa = read_sql("""
            SELECT player_id AS batter_id,
                   SUM(outs_above_average) AS oaa
            FROM production.fact_fielding_oaa
            WHERE season = :season
            GROUP BY player_id
        """, {"season": season})
        if not oaa.empty:
            box = box.merge(oaa, on="batter_id", how="left")
    except Exception:
        pass
    if "oaa" not in box.columns:
        box["oaa"] = np.nan
    box["oaa"] = box["oaa"].fillna(0).astype(float)

    # --- T3: MiLB features (young/new players) ---
    # Only attach for players with <= 3 MLB seasons or age <= 26
    from pathlib import Path
    cache_dir = Path("data/cached")
    milb_path = cache_dir / "milb_translated_batters.parquet"
    if milb_path.exists():
        try:
            milb = pd.read_parquet(milb_path)
            # Take the most recent MiLB season per player
            milb_latest = (
                milb.sort_values("season", ascending=False)
                .drop_duplicates("player_id", keep="first")
                .rename(columns={
                    "player_id": "batter_id",
                    "translated_k_pct": "milb_translated_k_pct",
                    "translated_bb_pct": "milb_translated_bb_pct",
                    "translated_iso": "milb_translated_iso",
                })
            )
            milb_cols = ["batter_id", "milb_translated_k_pct",
                         "milb_translated_bb_pct", "milb_translated_iso"]
            milb_avail = [c for c in milb_cols if c in milb_latest.columns]
            # Age-relative-to-level if available
            if "age_relative_to_level_avg" in milb_latest.columns:
                milb_latest = milb_latest.rename(
                    columns={"age_relative_to_level_avg": "milb_age_relative"})
                milb_avail.append("milb_age_relative")

            eligible = (box["mlb_seasons"] <= 3) | (box["age"] <= 26)
            eligible_ids = set(box.loc[eligible, "batter_id"])
            milb_sub = milb_latest[
                milb_latest["batter_id"].isin(eligible_ids)
            ][milb_avail]
            box = box.merge(milb_sub, on="batter_id", how="left")
        except Exception as e:
            logger.debug("Could not load MiLB translations: %s", e)

    for col in ["milb_translated_k_pct", "milb_translated_bb_pct",
                "milb_translated_iso", "milb_age_relative"]:
        if col not in box.columns:
            box[col] = np.nan

    # FG Future Value
    try:
        fv = read_sql("""
            SELECT player_id AS batter_id,
                   MAX(future_value) AS fg_future_value
            FROM production.dim_prospect_ranking
            GROUP BY player_id
        """, {})
        if not fv.empty:
            eligible = (box["mlb_seasons"] <= 3) | (box["age"] <= 26)
            eligible_ids = set(box.loc[eligible, "batter_id"])
            fv_sub = fv[fv["batter_id"].isin(eligible_ids)]
            box = box.merge(fv_sub, on="batter_id", how="left")
    except Exception:
        pass
    if "fg_future_value" not in box.columns:
        box["fg_future_value"] = np.nan

    logger.info(
        "Hitter breakout features for %d: %d players, T2=%d, T3=%d",
        season, len(box),
        box["barrel_pct"].notna().sum(),
        box["milb_translated_k_pct"].notna().sum(),
    )
    return box


# ---------------------------------------------------------------------------
# 43. Pitcher breakout features (tiered)
# ---------------------------------------------------------------------------
def get_pitcher_breakout_features(
    season: int,
    min_bf: int = 200,
) -> pd.DataFrame:
    """Consolidated pitcher feature set for breakout model training/scoring.

    Returns one row per qualified pitcher with three feature tiers:
    - **T1 (2000+):** boxscore rates (K%, BB%, ERA, FIP, WHIP, etc.)
    - **T2 (2018+):** Statcast advanced (whiff%, SwStr%, CSW%, velo, etc.)
    - **T3 (young pitchers):** translated MiLB rates

    Parameters
    ----------
    season : int
        MLB season year.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        Columns documented inline (27 features + metadata).
    """
    # --- T1: Boxscore base (2000+) ---
    box = read_sql("""
        SELECT
            pb.pitcher_id,
            dp.player_name  AS pitcher_name,
            dp.pitch_hand,
            dg.season,
            COUNT(DISTINCT pb.game_pk) AS games,
            SUM(pb.is_starter::int)    AS starts,
            SUM(pb.innings_pitched)    AS ip,
            SUM(pb.batters_faced)      AS bf,
            SUM(pb.strike_outs)        AS k,
            SUM(pb.walks + pb.intentional_walks) AS bb,
            SUM(pb.home_runs)          AS hr_allowed,
            SUM(pb.earned_runs)        AS er,
            SUM(pb.hits)               AS hits_allowed,
            SUM(pb.hit_by_pitch)       AS hbp,
            SUM(pb.ground_outs)        AS ground_outs,
            SUM(pb.air_outs)           AS air_outs
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand, dg.season
        HAVING SUM(pb.batters_faced) >= :min_bf
    """, {"season": season, "min_bf": min_bf})

    if box.empty:
        logger.warning("No qualified pitchers for breakout features (season=%d)", season)
        return pd.DataFrame()

    ip = box["ip"].replace(0, np.nan)
    bf = box["bf"].replace(0, np.nan)

    box["k_pct"] = box["k"] / bf
    box["bb_pct"] = box["bb"] / bf
    box["hr_bf"] = box["hr_allowed"] / bf
    box["era"] = (box["er"] / ip) * 9
    box["whip"] = (box["hits_allowed"] + box["bb"]) / ip
    box["k_per_9"] = (box["k"] / ip) * 9
    box["bb_per_9"] = (box["bb"] / ip) * 9
    box["go_ao"] = box["ground_outs"] / box["air_outs"].replace(0, np.nan)

    # FIP
    cfip = 3.20
    box["fip"] = (
        (13 * box["hr_allowed"] + 3 * (box["bb"] + box["hbp"]) - 2 * box["k"])
        / ip
    ) + cfip

    box["is_starter"] = (box["starts"] >= 3).astype(int)

    # Demographics
    demo = read_sql("""
        SELECT player_id AS pitcher_id,
               (:season - EXTRACT(YEAR FROM birth_date))::int AS age
        FROM production.dim_player
    """, {"season": season})
    box = box.merge(demo, on="pitcher_id", how="left")
    box["age"] = box["age"].fillna(28)

    # MLB service time
    svc = read_sql("""
        SELECT pb.pitcher_id, COUNT(DISTINCT dg.season) AS mlb_seasons
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.game_type = 'R' AND dg.season < :season
        GROUP BY pb.pitcher_id
        HAVING SUM(pb.batters_faced) >= 50
    """, {"season": season})
    box = box.merge(svc, on="pitcher_id", how="left")
    box["mlb_seasons"] = box["mlb_seasons"].fillna(0).astype(int)

    # YoY deltas
    prior = season - 1 if season != 2021 else 2019
    prior_df = read_sql("""
        SELECT pb.pitcher_id,
            SUM(pb.strike_outs)::float / NULLIF(SUM(pb.batters_faced), 0)
                AS k_prior,
            (SUM(pb.walks + pb.intentional_walks))::float
                / NULLIF(SUM(pb.batters_faced), 0) AS bb_prior,
            SUM(pb.earned_runs) * 9.0
                / NULLIF(SUM(pb.innings_pitched), 0) AS era_prior
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :prior AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id
        HAVING SUM(pb.batters_faced) >= 100
    """, {"prior": prior})
    box = box.merge(prior_df, on="pitcher_id", how="left")
    box["delta_k_pct"] = box["k_pct"] - box["k_prior"]
    box["delta_bb_pct"] = box["bb_pct"] - box["bb_prior"]
    box["delta_era"] = box["era"] - box["era_prior"]
    box.drop(columns=["k_prior", "bb_prior", "era_prior"],
             inplace=True, errors="ignore")

    # --- T2: Statcast advanced (2018+) ---
    try:
        adv = read_sql("""
            SELECT pitcher_id, swstr_pct, csw_pct, zone_pct, chase_pct,
                   contact_pct, xwoba_against, barrel_pct_against
            FROM production.fact_pitching_advanced
            WHERE season = :season AND batters_faced >= :min_bf
        """, {"season": season, "min_bf": min_bf})
        if not adv.empty:
            box = box.merge(adv, on="pitcher_id", how="left")
    except Exception:
        logger.debug("No fact_pitching_advanced for %d", season)

    # Observed profile: whiff_rate, avg_velo
    try:
        obs = get_pitcher_observed_profile(season)
        if not obs.empty:
            obs_cols = ["pitcher_id", "whiff_rate", "avg_velo"]
            obs_avail = [c for c in obs_cols if c in obs.columns]
            box = box.merge(obs[obs_avail], on="pitcher_id", how="left")
    except Exception:
        pass

    # HR/FB
    try:
        fb = get_pitcher_fly_ball_data(season)
        if not fb.empty and "hr_per_fb" in fb.columns:
            box = box.merge(
                fb[["pitcher_id", "hr_per_fb"]], on="pitcher_id", how="left")
    except Exception:
        pass

    # Efficiency: first_strike_pct, putaway_rate
    try:
        eff = get_pitcher_efficiency(season)
        if not eff.empty:
            eff_cols = [c for c in ["pitcher_id", "first_strike_pct", "putaway_rate"]
                        if c in eff.columns]
            box = box.merge(eff[eff_cols], on="pitcher_id", how="left")
    except Exception:
        pass

    # ERA - xFIP
    if "fip" in box.columns:
        # Approximate xFIP by adjusting HR component to league avg HR/FB
        fb_col = box.get("hr_per_fb")
        if fb_col is not None and fb_col.notna().any():
            lg_hr_fb = 0.10  # league average fallback
            box["era_minus_xfip"] = box["era"] - box["fip"]  # approximate
        else:
            box["era_minus_xfip"] = np.nan

    for col in ["whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
                "xwoba_against", "barrel_pct_against", "zone_pct",
                "chase_pct", "hr_per_fb", "era_minus_xfip",
                "first_strike_pct", "putaway_rate"]:
        if col not in box.columns:
            box[col] = np.nan

    # --- T3: MiLB features (young pitchers) ---
    from pathlib import Path
    cache_dir = Path("data/cached")
    milb_path = cache_dir / "milb_translated_pitchers.parquet"
    if milb_path.exists():
        try:
            milb = pd.read_parquet(milb_path)
            milb_latest = (
                milb.sort_values("season", ascending=False)
                .drop_duplicates("player_id", keep="first")
                .rename(columns={
                    "player_id": "pitcher_id",
                    "translated_k_pct": "milb_translated_k_pct",
                    "translated_bb_pct": "milb_translated_bb_pct",
                    "translated_hr_bf": "milb_translated_hr_bf",
                })
            )
            milb_cols = ["pitcher_id", "milb_translated_k_pct",
                         "milb_translated_bb_pct", "milb_translated_hr_bf"]
            milb_avail = [c for c in milb_cols if c in milb_latest.columns]
            if "age_relative_to_level_avg" in milb_latest.columns:
                milb_latest = milb_latest.rename(
                    columns={"age_relative_to_level_avg": "milb_age_relative"})
                milb_avail.append("milb_age_relative")

            eligible = (box["mlb_seasons"] <= 3) | (box["age"] <= 26)
            eligible_ids = set(box.loc[eligible, "pitcher_id"])
            milb_sub = milb_latest[
                milb_latest["pitcher_id"].isin(eligible_ids)
            ][milb_avail]
            box = box.merge(milb_sub, on="pitcher_id", how="left")
        except Exception as e:
            logger.debug("Could not load MiLB pitcher translations: %s", e)

    for col in ["milb_translated_k_pct", "milb_translated_bb_pct",
                "milb_translated_hr_bf", "milb_age_relative"]:
        if col not in box.columns:
            box[col] = np.nan

    logger.info(
        "Pitcher breakout features for %d: %d players, T2=%d, T3=%d",
        season, len(box),
        box["whiff_rate"].notna().sum(),
        box["milb_translated_k_pct"].notna().sum(),
    )
    return box
