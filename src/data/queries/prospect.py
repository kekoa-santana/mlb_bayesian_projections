"""Prospect-related SQL queries.

Queries for prospect snapshots, FanGraphs rankings, MLB debut data,
and eligibility filtering used by ``mlb_readiness.py`` and
``prospect_ranking.py``.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Org depth from prospect snapshots
# ---------------------------------------------------------------------------

def get_prospect_snapshots_for_org_depth() -> pd.DataFrame:
    """Fetch prospect snapshots for organizational depth calculation.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, primary_position, level,
        parent_org_id, parent_org_name.
    """
    query = """
    SELECT player_id, season, primary_position, level,
           parent_org_id, parent_org_name
    FROM production.fact_prospect_snapshot
    WHERE primary_position NOT IN ('P', 'PH', 'PR')
    """
    logger.info("Fetching prospect snapshots for org depth")
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# MLB first seasons (batter)
# ---------------------------------------------------------------------------

def get_mlb_batter_first_seasons() -> pd.DataFrame:
    """Get first MLB regular-season appearance per batter.

    Uses ``staging.batting_boxscores`` (data back to 2000) for full
    historical coverage.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, first_season.
    """
    query = """
    SELECT bb.batter_id, MIN(dg.season) AS first_season
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      AND bb.plate_appearances > 0
    GROUP BY bb.batter_id
    """
    logger.info("Fetching MLB batter first seasons")
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# MLB stuck IDs (batters with 200+ PA season)
# ---------------------------------------------------------------------------

def get_mlb_batters_with_min_pa_season(min_pa: int = 200) -> pd.DataFrame:
    """Get batter IDs who had at least one MLB season with *min_pa* PA.

    Uses ``staging.batting_boxscores`` (data back to 2000) for full
    historical coverage.

    Parameters
    ----------
    min_pa : int
        Minimum plate appearances in a single season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id (one row per qualifying batter, deduplicated).
    """
    query = """
    SELECT bb.batter_id
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
    GROUP BY bb.batter_id, dg.season
    HAVING SUM(bb.plate_appearances) >= :min_pa
    """
    logger.info("Fetching MLB batters with >= %d PA season", min_pa)
    return read_sql(query, {"min_pa": min_pa})


# ---------------------------------------------------------------------------
# MLB pitcher stuck IDs (pitchers with min BF)
# ---------------------------------------------------------------------------

def get_mlb_pitchers_with_min_bf(min_bf: int = 100) -> pd.DataFrame:
    """Get pitcher IDs with at least *min_bf* batters faced in a season.

    Queries ``production.fact_pitching_advanced``.

    Parameters
    ----------
    min_bf : int
        Minimum batters faced in a single season row.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id.
    """
    query = """
    SELECT DISTINCT pitcher_id
    FROM production.fact_pitching_advanced
    WHERE batters_faced >= :min_bf
    """
    logger.info("Fetching MLB pitchers with >= %d BF", min_bf)
    return read_sql(query, {"min_bf": min_bf})


# ---------------------------------------------------------------------------
# MLB established pitchers (for prospect exclusion)
# ---------------------------------------------------------------------------

def get_established_mlb_pitcher_ids(min_bf: int = 400) -> pd.DataFrame:
    """Get pitcher IDs with cumulative MLB BF above threshold.

    Used to exclude established MLB pitchers from prospect rankings.
    Queries ``production.fact_pitching_advanced`` for single-season rows
    where BF >= *min_bf*.

    Parameters
    ----------
    min_bf : int
        Minimum batters faced in a single season.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id.
    """
    query = """
    SELECT DISTINCT pitcher_id
    FROM production.fact_pitching_advanced
    WHERE batters_faced >= :min_bf
    """
    logger.info("Fetching established MLB pitchers (>= %d BF)", min_bf)
    return read_sql(query, {"min_bf": min_bf})


# ---------------------------------------------------------------------------
# FanGraphs prospect rankings
# ---------------------------------------------------------------------------

def get_fg_prospect_rankings(season: int) -> pd.DataFrame:
    """Load FanGraphs prospect rankings for a given season.

    Parameters
    ----------
    season : int
        The prospect ranking season.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, future_value, overall_rank, org_rank,
        risk, eta, source.
    """
    query = f"""
    SELECT player_id, future_value, overall_rank, org_rank,
           risk, eta, source
    FROM production.dim_prospect_ranking
    WHERE season = {int(season)}
    """
    logger.info("Fetching FG prospect rankings for season %d", season)
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# FanGraphs FV map (most recent per player)
# ---------------------------------------------------------------------------

def get_prospect_fv_grades() -> pd.DataFrame:
    """Look up most recent FanGraphs Future Value grade for each player.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, future_value.
        One row per player (most recent season, preferring earlier source).
    """
    query = """
    SELECT DISTINCT ON (player_id)
           player_id, future_value
    FROM production.dim_prospect_ranking
    WHERE future_value IS NOT NULL
    ORDER BY player_id, season DESC, source
    """
    logger.info("Fetching prospect FV grades")
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# Pitcher player IDs (position guard)
# ---------------------------------------------------------------------------

def get_pitcher_player_ids() -> pd.DataFrame:
    """Get player IDs whose primary position is pitcher.

    Used to exclude pitchers from batter prospect rankings.

    Returns
    -------
    pd.DataFrame
        Columns: player_id.
    """
    query = """
    SELECT player_id
    FROM production.dim_player
    WHERE primary_position = 'P'
    """
    logger.info("Fetching pitcher player IDs from dim_player")
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# MLB debut rates (pitchers)
# ---------------------------------------------------------------------------

def get_mlb_debut_pitcher_rates(
    min_bf: int = 20,
    max_bf: int = 400,
) -> pd.DataFrame:
    """Fetch MLB rates for pitching prospects with small debut samples.

    Parameters
    ----------
    min_bf, max_bf : int
        Batters-faced bounds for inclusion.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, mlb_bf, mlb_k, mlb_bb, mlb_hr.
    """
    query = """
    SELECT
        pb.pitcher_id  AS player_id,
        SUM(pb.batters_faced) AS mlb_bf,
        SUM(pb.strike_outs)   AS mlb_k,
        SUM(pb.walks)         AS mlb_bb,
        SUM(pb.home_runs)     AS mlb_hr
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
    GROUP BY pb.pitcher_id
    HAVING SUM(pb.batters_faced) BETWEEN :min_bf AND :max_bf
    """
    logger.info("Fetching MLB debut pitcher rates (BF %d-%d)", min_bf, max_bf)
    return read_sql(query, {"min_bf": min_bf, "max_bf": max_bf})


# ---------------------------------------------------------------------------
# MLB debut rates (batters)
# ---------------------------------------------------------------------------

def get_mlb_debut_batter_rates(
    min_pa: int = 20,
    max_pa: int = 200,
) -> pd.DataFrame:
    """Fetch MLB rates for batting prospects with small debut samples.

    Parameters
    ----------
    min_pa, max_pa : int
        Plate appearance bounds for inclusion.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, mlb_pa, mlb_k, mlb_bb, mlb_hr, mlb_h,
        mlb_ab, mlb_tb.
    """
    query = """
    SELECT
        bb.batter_id  AS player_id,
        SUM(bb.plate_appearances) AS mlb_pa,
        SUM(bb.strikeouts)        AS mlb_k,
        SUM(bb.walks)             AS mlb_bb,
        SUM(bb.home_runs)         AS mlb_hr,
        SUM(bb.hits)              AS mlb_h,
        SUM(bb.at_bats)           AS mlb_ab,
        SUM(bb.total_bases)       AS mlb_tb
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
    GROUP BY bb.batter_id
    HAVING SUM(bb.plate_appearances) BETWEEN :min_pa AND :max_pa
    """
    logger.info("Fetching MLB debut batter rates (PA %d-%d)", min_pa, max_pa)
    return read_sql(query, {"min_pa": min_pa, "max_pa": max_pa})
