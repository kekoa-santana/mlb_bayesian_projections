"""Pitcher arsenal profiles, season totals, observed profiles, efficiency, fly-ball data."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitcher arsenal profile
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
# Pitcher season totals (from boxscores)
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
        bins=[0, 25, 28, 32, 99],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Pitcher outcomes by batter stand (for league baselines v2)
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
# Pitch shape offerings (pre-aggregated for archetype clustering)
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
# Pitcher fly-ball / HR data (for xFIP computation)
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
# Pitcher observed profile (pitch-level aggregates)
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
# Extended pitcher season totals (from pitching boxscores -- includes outs)
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

    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 28, 32, 99],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype("Int64")

    # Derive starter flag and rates
    df["is_starter"] = (df["starts"] >= 3).astype(int)
    df["k_rate"] = (df["k"] / df["batters_faced"].replace(0, float("nan"))).round(4)
    df["bb_rate"] = (df["bb"] / df["batters_faced"].replace(0, float("nan"))).round(4)
    df["outs_per_bf"] = (df["outs"] / df["batters_faced"].replace(0, float("nan"))).round(4)

    return df


# ---------------------------------------------------------------------------
# Pitcher efficiency profile
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
