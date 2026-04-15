"""Exit model training data, pitch count features, bullpen rates, reliever roles, TTO profiles."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTO adjustment profiles
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Game Simulator -- pitch count features
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
# Game Simulator -- exit model training data
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
            fpg.player_id AS pitcher_id,
            fpg.game_pk,
            fpg.season,
            fpg.team_id,
            fpg.pit_bf AS total_bf,
            fpg.pit_pitches AS total_pitches
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = TRUE
          AND dg.game_type = 'R'
          AND fpg.season IN ({season_list})
          AND fpg.pit_bf >= 9
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
            fpg.player_id AS pitcher_id,
            fpg.season,
            fpg.team_id,
            fpg.pit_pitches,
            fpg.pit_bf,
            -- Convert baseball IP notation (.1=1/3, .2=2/3) to true outs
            FLOOR(fpg.pit_ip) * 3 + ROUND((fpg.pit_ip - FLOOR(fpg.pit_ip)) * 10) AS outs
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = TRUE
          AND dg.game_type = 'R'
          AND fpg.season IN ({season_list})
          AND fpg.pit_bf >= 9
    ),
    pitcher_agg AS (
        SELECT
            pitcher_id,
            season,
            MODE() WITHIN GROUP (ORDER BY team_id) AS team_id,
            COUNT(*)            AS n_starts,
            AVG(pit_pitches)    AS avg_pitches,
            STDDEV(pit_pitches) AS std_pitches,
            AVG(pit_bf)         AS avg_bf,
            AVG(outs) / 3.0     AS avg_ip
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
        ROUND(pa.avg_ip::numeric, 2)      AS avg_ip,
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
        fpg.team_id,
        fpg.season,
        SUM(fpg.pit_k)::float   / NULLIF(SUM(fpg.pit_bf), 0) AS k_rate,
        SUM(fpg.pit_bb)::float  / NULLIF(SUM(fpg.pit_bf), 0) AS bb_rate,
        SUM(fpg.pit_hr)::float  / NULLIF(SUM(fpg.pit_bf), 0) AS hr_rate,
        SUM(fpg.pit_bf) AS total_bf
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.pit_is_starter = FALSE
      AND dg.game_type = 'R'
      AND fpg.pit_bf >= 1
      AND fpg.season IN ({season_list})
    GROUP BY fpg.team_id, fpg.season
    ORDER BY fpg.team_id, fpg.season
    """
    logger.info("Fetching team bullpen rates for seasons %s", seasons)
    return read_sql(query)


def get_team_reliever_roster(
    seasons: list[int],
    min_bf: int = 10,
) -> pd.DataFrame:
    """Per-team reliever roster with BF shares for bullpen matchup weighting.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum total BF to include a reliever (filters mop-up / position players).

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, pitcher_id, bf, bf_share.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    SELECT
        fpg.team_id,
        fpg.season,
        fpg.player_id AS pitcher_id,
        SUM(fpg.pit_bf) AS bf,
        SUM(fpg.pit_bf)::float / SUM(SUM(fpg.pit_bf)) OVER (
            PARTITION BY fpg.team_id, fpg.season
        ) AS bf_share
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.pit_is_starter = FALSE
      AND dg.game_type = 'R'
      AND fpg.pit_bf >= 1
      AND fpg.season IN ({season_list})
    GROUP BY fpg.team_id, fpg.season, fpg.player_id
    HAVING SUM(fpg.pit_bf) >= {min_bf}
    ORDER BY fpg.team_id, fpg.season, bf DESC
    """
    logger.info("Fetching team reliever rosters for seasons %s", seasons)
    return read_sql(query)


def get_bullpen_trailing_workload(
    seasons: list[int],
    trailing_days: int = 3,
) -> pd.DataFrame:
    """Per-team-game trailing bullpen IP over the last N days.

    Used to detect bullpen fatigue: high trailing IP -> manager stretches
    the starter deeper.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    trailing_days : int
        Number of trailing calendar days to sum bullpen IP (default 3).

    Returns
    -------
    pd.DataFrame
        Columns: team_id, game_pk, game_date, bullpen_trailing_ip.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH reliever_ip AS (
        SELECT
            fpg.team_id,
            dg.game_pk,
            dg.game_date,
            SUM(fpg.pit_ip) AS bullpen_ip
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = FALSE
          AND dg.game_type = 'R'
          AND fpg.pit_bf >= 1
          AND fpg.season IN ({season_list})
        GROUP BY fpg.team_id, dg.game_pk, dg.game_date
    )
    SELECT
        r.team_id,
        r.game_pk,
        r.game_date,
        COALESCE(SUM(prev.bullpen_ip), 0.0) AS bullpen_trailing_ip
    FROM reliever_ip r
    LEFT JOIN reliever_ip prev
        ON prev.team_id = r.team_id
       AND prev.game_date >= r.game_date - INTERVAL '{trailing_days} days'
       AND prev.game_date < r.game_date
    GROUP BY r.team_id, r.game_pk, r.game_date
    ORDER BY r.team_id, r.game_date
    """
    logger.info(
        "Fetching bullpen trailing workload (%d-day) for seasons %s",
        trailing_days, seasons,
    )
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
        fpg.player_id AS pitcher_id,
        fpg.season,
        COUNT(*)                     AS games,
        SUM(fpg.pit_sv)                  AS saves,
        SUM(fpg.pit_hld)                 AS holds,
        SUM(fpg.pit_bs)                  AS blown_saves,
        SUM(fpg.pit_bf)                  AS bf,
        SUM(fpg.pit_k)                   AS k,
        SUM(fpg.pit_bb)                  AS bb,
        SUM(fpg.pit_hr)                  AS hr,
        SUM(fpg.pit_h)                   AS h,
        SUM(fpg.pit_r)                   AS runs,
        ROUND(SUM(fpg.pit_ip) * 3)::int  AS outs,
        SUM(fpg.pit_pitches)             AS pitches,
        SUM(fpg.pit_k)::float / NULLIF(SUM(fpg.pit_bf), 0) AS k_rate,
        SUM(fpg.pit_bb)::float / NULLIF(SUM(fpg.pit_bf), 0) AS bb_rate
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.pit_is_starter = FALSE
      AND dg.game_type = 'R'
      AND fpg.pit_bf >= 1
      AND fpg.season IN ({season_list})
    GROUP BY fpg.player_id, fpg.season
    HAVING COUNT(*) >= {min_games}
    ORDER BY fpg.player_id, fpg.season
    """
    logger.info("Fetching reliever role history for seasons %s (min_games=%d)", seasons, min_games)
    return read_sql(query)


def get_reliever_stats_by_team(
    seasons: list[int],
    min_bf: int = 20,
) -> pd.DataFrame:
    """Per-reliever-season stats with team assignment for bullpen tier profiles.

    Like ``get_reliever_role_history`` but includes ``team_id`` (most-used
    team) and filters by BF instead of games.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum total BF to include a reliever.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, team_id, season, games, bf, k, bb, hr.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH reliever_team AS (
        SELECT
            fpg.player_id AS pitcher_id,
            fpg.team_id,
            fpg.season,
            COUNT(*)         AS games,
            SUM(fpg.pit_bf)  AS bf,
            SUM(fpg.pit_k)   AS k,
            SUM(fpg.pit_bb)  AS bb,
            SUM(fpg.pit_hr)  AS hr,
            ROW_NUMBER() OVER (
                PARTITION BY fpg.player_id, fpg.season
                ORDER BY SUM(fpg.pit_bf) DESC
            ) AS team_rank
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = FALSE
          AND dg.game_type = 'R'
          AND fpg.pit_bf >= 1
          AND fpg.season IN ({season_list})
        GROUP BY fpg.player_id, fpg.team_id, fpg.season
    )
    SELECT pitcher_id, team_id, season, games, bf, k, bb, hr
    FROM reliever_team
    WHERE team_rank = 1 AND bf >= {min_bf}
    ORDER BY team_id, season, bf DESC
    """
    logger.info("Fetching reliever stats by team for seasons %s", seasons)
    return read_sql(query)
