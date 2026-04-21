"""Traditional batting/pitching stats, spray charts, lineup priors, run values, sprint speed."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.data.queries._common import _WOBA_WEIGHTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hitter traditional stats (AVG, OBP, SLG, OPS, ISO, wOBA, BABIP)
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

    # wOBA: BIP contribution + non-BIP weights (K=0, IBB excluded)
    non_ibb_bb = df["bb"] - df["ibb"]
    woba_num = (
        df["woba_bip_sum"].fillna(0)
        + non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp"] * _WOBA_WEIGHTS["hbp"]
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
# Pitcher traditional stats (ERA, WHIP, K/9, BB/9, FIP, W/L/SV)
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
    # cFIP ~ 3.20 (league constant, varies slightly by year)
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
# Recent form (rolling window)
# ---------------------------------------------------------------------------
def get_hitter_recent_form(days: int = 14, as_of_date: str | None = None) -> pd.DataFrame:
    """Rolling-window hitter form from boxscores + Statcast quality.

    Parameters
    ----------
    days : int
        Number of trailing calendar days (inclusive) in the window.
    as_of_date : str or None
        Anchor date (YYYY-MM-DD). If None, uses CURRENT_DATE.

    Returns
    -------
    pd.DataFrame
        Columns include batter_id, batter_name, pa_14d, games_14d,
        k_rate_14d, bb_rate_14d, woba_14d, ops_14d, iso_14d,
        hard_hit_pct_14d, barrel_pct_14d, xwoba_14d, bip_14d.
    """
    query = """
    WITH params AS (
        SELECT
            COALESCE(CAST(:as_of_date AS date), CURRENT_DATE) AS end_date,
            CAST(:days AS int) AS n_days
    ),
    box AS (
        SELECT
            bb.batter_id,
            dp.player_name AS batter_name,
            COUNT(DISTINCT bb.game_pk) AS games_14d,
            SUM(bb.plate_appearances) AS pa_14d,
            SUM(bb.at_bats) AS ab_14d,
            SUM(bb.hits) AS hits_14d,
            SUM(bb.doubles) AS doubles_14d,
            SUM(bb.triples) AS triples_14d,
            SUM(bb.total_bases) AS total_bases_14d,
            SUM(bb.walks) AS bb_14d,
            SUM(bb.intentional_walks) AS ibb_14d,
            SUM(bb.hit_by_pitch) AS hbp_14d,
            SUM(bb.strikeouts) AS k_14d,
            SUM(bb.home_runs) AS hr_14d,
            SUM(bb.runs) AS runs_14d,
            SUM(bb.rbi) AS rbi_14d,
            SUM(bb.sb) AS sb_14d,
            SUM(bb.caught_stealing) AS cs_14d
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON bb.batter_id = dp.player_id
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date BETWEEN (p.end_date - (p.n_days - 1) * INTERVAL '1 day') AND p.end_date
        GROUP BY bb.batter_id, dp.player_name
        HAVING SUM(bb.plate_appearances) >= 1
    ),
    sac_flies AS (
        SELECT
            fp.batter_id,
            SUM(CASE WHEN fp.events IN ('sac_fly', 'sac_fly_double_play') THEN 1 ELSE 0 END) AS sf_14d
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date BETWEEN (p.end_date - (p.n_days - 1) * INTERVAL '1 day') AND p.end_date
        GROUP BY fp.batter_id
    ),
    batted AS (
        SELECT
            fp.batter_id,
            COUNT(*) AS bip_14d,
            AVG(CASE WHEN sbb.hard_hit THEN 1.0 ELSE 0.0 END) AS hard_hit_pct_14d,
            AVG(
                CASE
                    WHEN sbb.launch_speed >= 98 AND sbb.launch_angle BETWEEN 26 AND 30 THEN 1.0
                    ELSE 0.0
                END
            ) AS barrel_pct_14d,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba_14d
        FROM production.sat_batted_balls sbb
        JOIN production.fact_pa fp ON sbb.pa_id = fp.pa_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date BETWEEN (p.end_date - (p.n_days - 1) * INTERVAL '1 day') AND p.end_date
        GROUP BY fp.batter_id
    )
    SELECT
        b.batter_id,
        b.batter_name,
        b.games_14d,
        b.pa_14d,
        b.ab_14d,
        b.hits_14d,
        b.doubles_14d,
        b.triples_14d,
        b.total_bases_14d,
        b.bb_14d,
        b.ibb_14d,
        b.hbp_14d,
        b.k_14d,
        b.hr_14d,
        b.runs_14d,
        b.rbi_14d,
        b.sb_14d,
        b.cs_14d,
        COALESCE(sf.sf_14d, 0) AS sf_14d,
        bt.bip_14d,
        bt.hard_hit_pct_14d,
        bt.barrel_pct_14d,
        bt.xwoba_14d
    FROM box b
    LEFT JOIN sac_flies sf ON b.batter_id = sf.batter_id
    LEFT JOIN batted bt ON b.batter_id = bt.batter_id
    ORDER BY b.pa_14d DESC
    """
    logger.info("Fetching hitter recent form (%d-day)", days)
    df = read_sql(query, {"days": days, "as_of_date": as_of_date})
    if df.empty:
        return df

    pa = df["pa_14d"].replace(0, float("nan"))
    ab = df["ab_14d"].replace(0, float("nan"))
    non_ibb_bb = (df["bb_14d"] - df["ibb_14d"]).clip(lower=0)

    df["k_rate_14d"] = (df["k_14d"] / pa).fillna(0.0)
    df["bb_rate_14d"] = (df["bb_14d"] / pa).fillna(0.0)
    df["avg_14d"] = (df["hits_14d"] / ab).fillna(0.0)
    df["slg_14d"] = (df["total_bases_14d"] / ab).fillna(0.0)
    obp_num = df["hits_14d"] + df["bb_14d"] + df["hbp_14d"]
    obp_den = (df["ab_14d"] + df["bb_14d"] + df["hbp_14d"] + df["sf_14d"]).replace(0, float("nan"))
    df["obp_14d"] = (obp_num / obp_den).fillna(0.0)
    df["ops_14d"] = df["obp_14d"] + df["slg_14d"]
    df["iso_14d"] = (df["slg_14d"] - df["avg_14d"]).fillna(0.0)

    woba_num = (
        non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp_14d"] * _WOBA_WEIGHTS["hbp"]
        + (df["hits_14d"] - df["doubles_14d"] - df["triples_14d"] - df["hr_14d"]) * _WOBA_WEIGHTS["single"]
        + df["doubles_14d"] * _WOBA_WEIGHTS["double"]
        + df["triples_14d"] * _WOBA_WEIGHTS["triple"]
        + df["hr_14d"] * _WOBA_WEIGHTS["hr"]
    )
    woba_den = (df["ab_14d"] + non_ibb_bb + df["sf_14d"] + df["hbp_14d"]).replace(0, float("nan"))
    df["woba_14d"] = (woba_num / woba_den).fillna(0.0)

    for c in ("hard_hit_pct_14d", "barrel_pct_14d", "xwoba_14d"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    if "bip_14d" in df.columns:
        df["bip_14d"] = df["bip_14d"].fillna(0).astype(int)

    return df


def get_pitcher_recent_form(days: int = 14, as_of_date: str | None = None) -> pd.DataFrame:
    """Rolling-window pitcher form from pitching boxscores.

    Parameters
    ----------
    days : int
        Number of trailing calendar days (inclusive) in the window.
    as_of_date : str or None
        Anchor date (YYYY-MM-DD). If None, uses CURRENT_DATE.

    Returns
    -------
    pd.DataFrame
        Columns include pitcher_id, pitcher_name, bf_14d, games_14d,
        starts_14d, k_rate_14d, bb_rate_14d, era_14d, whip_14d, hr_per_9_14d.
    """
    query = """
    WITH params AS (
        SELECT
            COALESCE(CAST(:as_of_date AS date), CURRENT_DATE) AS end_date,
            CAST(:days AS int) AS n_days
    )
    SELECT
        pb.pitcher_id,
        dp.player_name AS pitcher_name,
        COUNT(DISTINCT pb.game_pk) AS games_14d,
        SUM(pb.is_starter::int) AS starts_14d,
        SUM(pb.batters_faced) AS bf_14d,
        SUM(pb.innings_pitched) AS ip_14d,
        SUM(pb.strike_outs) AS k_14d,
        SUM(pb.walks + pb.intentional_walks) AS bb_14d,
        SUM(pb.home_runs) AS hr_14d,
        SUM(pb.hits) AS hits_14d,
        SUM(pb.earned_runs) AS er_14d,
        SUM(pb.saves) AS sv_14d,
        SUM(pb.holds) AS hld_14d
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    JOIN params p ON TRUE
    WHERE dg.game_type = 'R'
      AND dg.game_date::date BETWEEN (p.end_date - (p.n_days - 1) * INTERVAL '1 day') AND p.end_date
    GROUP BY pb.pitcher_id, dp.player_name
    HAVING SUM(pb.batters_faced) >= 1
    ORDER BY SUM(pb.batters_faced) DESC
    """
    logger.info("Fetching pitcher recent form (%d-day)", days)
    df = read_sql(query, {"days": days, "as_of_date": as_of_date})
    if df.empty:
        return df

    bf = df["bf_14d"].replace(0, float("nan"))
    ip = df["ip_14d"].replace(0, float("nan"))
    df["k_rate_14d"] = (df["k_14d"] / bf).fillna(0.0)
    df["bb_rate_14d"] = (df["bb_14d"] / bf).fillna(0.0)
    df["k_minus_bb_14d"] = df["k_rate_14d"] - df["bb_rate_14d"]
    df["era_14d"] = ((df["er_14d"] / ip) * 9).fillna(0.0)
    df["whip_14d"] = ((df["hits_14d"] + df["bb_14d"]) / ip).fillna(0.0)
    df["hr_per_9_14d"] = ((df["hr_14d"] / ip) * 9).fillna(0.0)
    df["role_14d"] = np.where(df["starts_14d"] >= 1, "SP", "RP")
    return df


# ---------------------------------------------------------------------------
# Extended hitter season totals (from batting boxscores -- includes games, SB)
# ---------------------------------------------------------------------------
def get_hitter_season_totals_extended(season: int) -> pd.DataFrame:
    """Per-batter season totals from batting boxscores.

    Includes games played, stolen bases, caught stealing, total bases,
    runs, RBI -- columns not available from fact_pa.

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
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 28, 32, 99],
        labels=[0, 1, 2, 3],
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
# Batted-ball spray distribution
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
# Sprint speed
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
# Pitcher location grid queries (5x5 zone grid)
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

    Returns one row per pitch with plate_x, plate_z -- no grid binning.
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


# ---------------------------------------------------------------------------
# Player team mapping (player_id -> team abbreviation)
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
# Prospect info
# ---------------------------------------------------------------------------
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
# Lineup priors (position + batting order history)
# ---------------------------------------------------------------------------
def get_lineup_priors(season: int) -> pd.DataFrame:
    """Get each player's position and batting order history from fact_lineup.

    Returns one row per player-position with start counts and most common
    batting order at that position.  Team-agnostic -- a player's position
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


# ---------------------------------------------------------------------------
# Daily standouts (single-game performances)
# ---------------------------------------------------------------------------
def get_hitter_daily_standouts(game_date: str | None = None) -> pd.DataFrame:
    """Single-game hitter box-score lines + Statcast quality for one date.

    Parameters
    ----------
    game_date : str or None
        Target date (YYYY-MM-DD).  Defaults to yesterday.

    Returns
    -------
    pd.DataFrame
        One row per player-game with counting stats, rates, wOBA, and
        optional Statcast quality columns.
    """
    query = """
    WITH params AS (
        SELECT COALESCE(CAST(:game_date AS date), CURRENT_DATE - INTERVAL '1 day') AS target_date
    ),
    box AS (
        SELECT
            bb.batter_id,
            dp.player_name AS batter_name,
            bb.game_pk,
            dg.game_date,
            dt_away.abbreviation AS away_team,
            dt_home.abbreviation AS home_team,
            bb.plate_appearances AS pa,
            bb.at_bats          AS ab,
            bb.hits,
            bb.doubles,
            bb.triples,
            bb.home_runs        AS hr,
            bb.total_bases,
            bb.runs,
            bb.rbi,
            bb.walks            AS bb,
            bb.intentional_walks AS ibb,
            bb.hit_by_pitch     AS hbp,
            bb.strikeouts       AS k,
            bb.sb,
            bb.caught_stealing  AS cs
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON bb.batter_id = dp.player_id
        LEFT JOIN production.dim_team dt_away ON dg.away_team_id = dt_away.team_id
        LEFT JOIN production.dim_team dt_home ON dg.home_team_id = dt_home.team_id
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date = p.target_date
          AND bb.plate_appearances >= 1
    ),
    sac_flies AS (
        SELECT
            fp.batter_id,
            fp.game_pk,
            SUM(CASE WHEN fp.events IN ('sac_fly', 'sac_fly_double_play') THEN 1 ELSE 0 END) AS sf
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date = p.target_date
        GROUP BY fp.batter_id, fp.game_pk
    ),
    batted AS (
        SELECT
            fp.batter_id,
            fp.game_pk,
            COUNT(*) AS bip,
            AVG(CASE WHEN sbb.hard_hit THEN 1.0 ELSE 0.0 END) AS hard_hit_pct,
            AVG(
                CASE
                    WHEN sbb.launch_speed >= 98 AND sbb.launch_angle BETWEEN 26 AND 30 THEN 1.0
                    ELSE 0.0
                END
            ) AS barrel_pct,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba,
            MAX(CASE WHEN sbb.is_homerun THEN sbb.hit_distance_sc END) AS max_hr_distance
        FROM production.sat_batted_balls sbb
        JOIN production.fact_pa fp ON sbb.pa_id = fp.pa_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN params p ON TRUE
        WHERE dg.game_type = 'R'
          AND dg.game_date::date = p.target_date
        GROUP BY fp.batter_id, fp.game_pk
    )
    SELECT
        b.*,
        COALESCE(sf.sf, 0) AS sf,
        bt.bip,
        bt.hard_hit_pct,
        bt.barrel_pct,
        bt.xwoba,
        bt.max_hr_distance
    FROM box b
    LEFT JOIN sac_flies sf ON b.batter_id = sf.batter_id AND b.game_pk = sf.game_pk
    LEFT JOIN batted bt ON b.batter_id = bt.batter_id AND b.game_pk = bt.game_pk
    ORDER BY b.pa DESC
    """
    logger.info("Fetching hitter daily standouts for %s", game_date or "yesterday")
    df = read_sql(query, {"game_date": game_date})
    if df.empty:
        return df

    ab = df["ab"].replace(0, float("nan"))
    pa = df["pa"].replace(0, float("nan"))
    non_ibb_bb = (df["bb"] - df["ibb"]).clip(lower=0)

    df["avg"] = (df["hits"] / ab).fillna(0.0)
    df["slg"] = (df["total_bases"] / ab).fillna(0.0)
    obp_num = df["hits"] + df["bb"] + df["hbp"]
    obp_den = (df["ab"] + df["bb"] + df["hbp"] + df["sf"]).replace(0, float("nan"))
    df["obp"] = (obp_num / obp_den).fillna(0.0)
    df["ops"] = df["obp"] + df["slg"]
    df["iso"] = (df["slg"] - df["avg"]).fillna(0.0)

    singles = df["hits"] - df["doubles"] - df["triples"] - df["hr"]
    woba_num = (
        non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp"] * _WOBA_WEIGHTS["hbp"]
        + singles * _WOBA_WEIGHTS["single"]
        + df["doubles"] * _WOBA_WEIGHTS["double"]
        + df["triples"] * _WOBA_WEIGHTS["triple"]
        + df["hr"] * _WOBA_WEIGHTS["hr"]
    )
    woba_den = (df["ab"] + non_ibb_bb + df["sf"] + df["hbp"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).fillna(0.0)

    df["k_rate"] = (df["k"] / pa).fillna(0.0)
    df["bb_rate"] = (df["bb"] / pa).fillna(0.0)

    for c in ("hard_hit_pct", "barrel_pct", "xwoba"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "bip" in df.columns:
        df["bip"] = df["bip"].fillna(0).astype(int)
    # Keep NaN for players who didn't hit a HR -- downstream (dashboard) uses
    # "first non-null" semantics so missing distance shouldn't look like 0 ft.
    if "max_hr_distance" in df.columns:
        df["max_hr_distance"] = pd.to_numeric(df["max_hr_distance"], errors="coerce")

    return df


def get_pitcher_daily_standouts(game_date: str | None = None) -> pd.DataFrame:
    """Single-game pitcher lines for one date.

    Parameters
    ----------
    game_date : str or None
        Target date (YYYY-MM-DD).  Defaults to yesterday.

    Returns
    -------
    pd.DataFrame
        One row per pitcher-game with counting stats and derived rates.
    """
    query = """
    WITH params AS (
        SELECT COALESCE(CAST(:game_date AS date), CURRENT_DATE - INTERVAL '1 day') AS target_date
    )
    SELECT
        pb.pitcher_id,
        dp.player_name AS pitcher_name,
        pb.game_pk,
        dg.game_date,
        dt_away.abbreviation AS away_team,
        dt_home.abbreviation AS home_team,
        pb.is_starter,
        pb.batters_faced AS bf,
        pb.innings_pitched AS ip,
        pb.strike_outs AS k,
        pb.walks + pb.intentional_walks AS bb,
        pb.home_runs AS hr,
        pb.hits,
        pb.earned_runs AS er,
        pb.runs,
        pb.wins AS w,
        pb.losses AS l,
        pb.saves AS sv,
        pb.holds AS hld,
        pb.number_of_pitches AS pitches
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    LEFT JOIN production.dim_team dt_away ON dg.away_team_id = dt_away.team_id
    LEFT JOIN production.dim_team dt_home ON dg.home_team_id = dt_home.team_id
    JOIN params p ON TRUE
    WHERE dg.game_type = 'R'
      AND dg.game_date::date = p.target_date
      AND pb.batters_faced >= 1
    ORDER BY pb.innings_pitched DESC, pb.strike_outs DESC
    """
    logger.info("Fetching pitcher daily standouts for %s", game_date or "yesterday")
    df = read_sql(query, {"game_date": game_date})
    if df.empty:
        return df

    bf = df["bf"].replace(0, float("nan"))
    ip = df["ip"].replace(0, float("nan"))
    df["k_rate"] = (df["k"] / bf).fillna(0.0)
    df["bb_rate"] = (df["bb"] / bf).fillna(0.0)
    df["k_minus_bb"] = df["k_rate"] - df["bb_rate"]
    df["era"] = ((df["er"] / ip) * 9).fillna(0.0)
    df["whip"] = ((df["hits"] + df["bb"]) / ip).fillna(0.0)
    df["hr_per_9"] = ((df["hr"] / ip) * 9).fillna(0.0)
    df["role"] = np.where(df["is_starter"], "SP", "RP")

    # Quality start flag: 6+ IP and <= 3 ER
    df["qs"] = ((df["ip"] >= 6.0) & (df["er"] <= 3)).astype(int)

    return df
