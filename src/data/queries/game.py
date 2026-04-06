"""Game-level queries (game logs, game batter stats, lineups, actuals)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitcher game logs (from pitching boxscores)
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
# Game-level batter Ks (per pitcher-batter within a game)
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
# Game-level batter stats (per pitcher-batter within a game) -- extended
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
# Game lineups from fact_lineup
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
# Batter game actuals (for batter sim backtesting)
# ---------------------------------------------------------------------------
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
        SELECT fpg.player_id AS pitcher_id, fpg.game_pk, fpg.team_id AS pitcher_team_id
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = TRUE
          AND dg.game_type = 'R'
          AND fpg.season = :season
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
    JOIN production.dim_game dg2 ON fg.game_pk = dg2.game_pk
    JOIN production.fact_lineup fl
      ON fg.player_id = fl.player_id AND fg.game_pk = fl.game_pk
    LEFT JOIN starters st
      ON fg.game_pk = st.game_pk AND st.pitcher_team_id != fg.team_id
    WHERE fg.player_role = 'batter'
      AND dg2.game_type = 'R'
      AND fl.is_starter = TRUE
      AND fg.season = :season
      AND fg.bat_pa >= 1
    ORDER BY fg.game_pk, fl.batting_order
    """
    logger.info("Fetching batter game actuals for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Catcher game assignments (for framing lookup)
# ---------------------------------------------------------------------------
def get_catcher_game_assignments(season: int) -> pd.DataFrame:
    """Starting catchers per game for a season.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, catcher_id, team_id.
    """
    query = """
        SELECT fl.game_pk, fl.player_id AS catcher_id,
               fl.team_id
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE fl.position = 'C'
          AND fl.is_starter = true
          AND dg.season = :season
          AND dg.game_type = 'R'
    """
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Game starter teams (for framing lookup)
# ---------------------------------------------------------------------------
def get_game_starter_teams(season: int) -> pd.DataFrame:
    """Starting pitchers and their teams per game for a season.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, team_id.
    """
    query = """
        SELECT fpg.game_pk, fpg.player_id AS pitcher_id, fpg.team_id
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND dg.season = :season
          AND dg.game_type = 'R'
    """
    return read_sql(query, {"season": season})
