"""
Core SQL queries for the Bayesian projection system.

Every function returns a pandas DataFrame.  All SQL lives here —
no raw query strings elsewhere in the codebase.
"""
from __future__ import annotations

import logging

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
        JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
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
        JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
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
    """Per-player season lines: PA, K, BB, barrel%, xwOBA, etc.

    Combines plate-appearance events with batted-ball quality metrics.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, season, pa, k, bb,
        hits, hr, xwoba_avg, barrel_pct, hard_hit_pct, k_rate, bb_rate.
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
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END)                                           AS xwoba_avg,
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
        ROUND((pa.bb::numeric / pa.pa), 4)   AS bb_rate
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.pa >= 1
    ORDER BY pa.pa DESC
    """
    logger.info("Fetching season totals for %d", season)
    return read_sql(query, {"season": season})


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
        JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
        WHERE dg.season = :season
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
          AND fp.pitch_type NOT IN ('PO','UN','SC','FA')
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
        strike_outs, batters_faced, innings_pitched, is_starter.
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
        pb.is_starter
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
        pa.season,
        pa.birth_date,
        EXTRACT(YEAR FROM AGE(DATE(pa.season || '-07-01'), pa.birth_date))::int AS age,
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
        ROUND((pa.hr::numeric / pa.pa), 4)   AS hr_rate
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id
    WHERE pa.pa >= 1
    ORDER BY pa.pa DESC
    """
    logger.info("Fetching season totals with age for %d", season)
    df = read_sql(query, {"season": season})

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
