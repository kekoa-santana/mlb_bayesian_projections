"""Breakout features (hitter + pitcher), postseason stats, PA outcomes, rolling form, Glicko-related."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.data.queries._common import _WOBA_WEIGHTS, season_in_clause

logger = logging.getLogger(__name__)

_PS_ROUND_ORDER = {"F": 1, "D": 2, "L": 3, "W": 4}


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
# Hitter breakout features (tiered)
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
    # Avoid circular import — these are in sibling modules
    from src.data.queries.hitter import get_hitter_observed_profile
    from src.data.queries.pitcher import get_pitcher_fly_ball_data
    from src.data.queries.traditional import get_sprint_speed

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
    w = _WOBA_WEIGHTS
    woba_num = (
        w["ubb"] * non_ibb + w["hbp"] * box["hbp"]
        + w["single"] * singles + w["double"] * box["doubles"]
        + w["triple"] * box["triples"] + w["hr"] * box["hr"]
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
# Pitcher breakout features (tiered)
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
    # Avoid circular import
    from src.data.queries.pitcher import (
        get_pitcher_observed_profile,
        get_pitcher_fly_ball_data,
        get_pitcher_efficiency,
    )

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


# ---------------------------------------------------------------------------
# Postseason stats (player-specific)
# ---------------------------------------------------------------------------
def get_postseason_batter_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-batter postseason stats aggregated by season.

    Queries ``fact_player_game_mlb`` joined with ``dim_game`` for
    postseason game types (F=Wild Card, D=Division, L=LCS, W=World Series).

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, ps_pa, ps_k, ps_bb, ps_hr, ps_h,
        ps_tb, ps_k_rate, ps_bb_rate, ps_hr_rate, ps_iso,
        best_round, num_rounds_played, ps_games.
    """
    if not seasons:
        return pd.DataFrame()

    season_list = season_in_clause(seasons)
    query = f"""
    WITH game_rounds AS (
        SELECT
            fpg.player_id                       AS batter_id,
            fpg.season,
            dg.game_type                        AS round,
            fpg.bat_pa,
            fpg.bat_k,
            fpg.bat_bb,
            fpg.bat_hr,
            fpg.bat_h,
            fpg.bat_tb
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.player_role = 'batter'
          AND dg.game_type IN ('F','D','L','W')
          AND fpg.season IN ({season_list})
          AND fpg.bat_pa > 0
    ),
    season_agg AS (
        SELECT
            batter_id,
            season,
            SUM(bat_pa)                                  AS ps_pa,
            SUM(bat_k)                                   AS ps_k,
            SUM(bat_bb)                                  AS ps_bb,
            SUM(bat_hr)                                  AS ps_hr,
            SUM(bat_h)                                   AS ps_h,
            SUM(bat_tb)                                  AS ps_tb,
            COUNT(*)                                     AS ps_games,
            COUNT(DISTINCT round)                        AS num_rounds_played,
            MAX(CASE round
                WHEN 'W' THEN 4
                WHEN 'L' THEN 3
                WHEN 'D' THEN 2
                WHEN 'F' THEN 1
            END)                                         AS best_round_ord
        FROM game_rounds
        GROUP BY batter_id, season
    )
    SELECT
        batter_id,
        season,
        ps_pa,
        ps_k,
        ps_bb,
        ps_hr,
        ps_h,
        ps_tb,
        ROUND(ps_k::numeric / NULLIF(ps_pa, 0), 4)     AS ps_k_rate,
        ROUND(ps_bb::numeric / NULLIF(ps_pa, 0), 4)     AS ps_bb_rate,
        ROUND(ps_hr::numeric / NULLIF(ps_pa, 0), 4)     AS ps_hr_rate,
        ROUND((ps_tb - ps_h)::numeric / NULLIF(ps_pa - ps_bb, 0), 4)
                                                          AS ps_iso,
        CASE best_round_ord
            WHEN 4 THEN 'W'
            WHEN 3 THEN 'L'
            WHEN 2 THEN 'D'
            WHEN 1 THEN 'F'
        END                                               AS best_round,
        num_rounds_played,
        ps_games
    FROM season_agg
    ORDER BY batter_id, season
    """
    logger.info("Fetching postseason batter stats for seasons %s", seasons)
    return read_sql(query, {})


def get_postseason_pitcher_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-pitcher postseason stats aggregated by season.

    Queries ``fact_player_game_mlb`` joined with ``dim_game`` for
    postseason game types (F=Wild Card, D=Division, L=LCS, W=World Series).

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, ps_bf, ps_k, ps_bb, ps_hr, ps_h,
        ps_k_rate, ps_bb_rate, ps_hr_rate, ps_ip, ps_games,
        is_starter, best_round, num_rounds_played.
    """
    if not seasons:
        return pd.DataFrame()

    season_list = season_in_clause(seasons)
    query = f"""
    WITH game_rounds AS (
        SELECT
            fpg.player_id                       AS pitcher_id,
            fpg.season,
            dg.game_type                        AS round,
            fpg.pit_bf,
            fpg.pit_k,
            fpg.pit_bb,
            fpg.pit_hr,
            fpg.pit_h,
            fpg.pit_ip,
            fpg.pit_is_starter
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.player_role = 'pitcher'
          AND dg.game_type IN ('F','D','L','W')
          AND fpg.season IN ({season_list})
          AND fpg.pit_bf > 0
    ),
    season_agg AS (
        SELECT
            pitcher_id,
            season,
            SUM(pit_bf)                                  AS ps_bf,
            SUM(pit_k)                                   AS ps_k,
            SUM(pit_bb)                                  AS ps_bb,
            SUM(pit_hr)                                  AS ps_hr,
            SUM(pit_h)                                   AS ps_h,
            SUM(pit_ip)                                  AS ps_ip,
            COUNT(*)                                     AS ps_games,
            COUNT(DISTINCT round)                        AS num_rounds_played,
            BOOL_OR(pit_is_starter)                      AS has_started,
            SUM(CASE WHEN pit_is_starter THEN 1 ELSE 0 END)
                                                          AS start_count,
            MAX(CASE round
                WHEN 'W' THEN 4
                WHEN 'L' THEN 3
                WHEN 'D' THEN 2
                WHEN 'F' THEN 1
            END)                                         AS best_round_ord
        FROM game_rounds
        GROUP BY pitcher_id, season
    )
    SELECT
        pitcher_id,
        season,
        ps_bf,
        ps_k,
        ps_bb,
        ps_hr,
        ps_h,
        ps_ip,
        ps_games,
        ROUND(ps_k::numeric / NULLIF(ps_bf, 0), 4)      AS ps_k_rate,
        ROUND(ps_bb::numeric / NULLIF(ps_bf, 0), 4)      AS ps_bb_rate,
        ROUND(ps_hr::numeric / NULLIF(ps_bf, 0), 4)      AS ps_hr_rate,
        (start_count > ps_games / 2)                      AS is_starter,
        CASE best_round_ord
            WHEN 4 THEN 'W'
            WHEN 3 THEN 'L'
            WHEN 2 THEN 'D'
            WHEN 1 THEN 'F'
        END                                               AS best_round,
        num_rounds_played
    FROM season_agg
    ORDER BY pitcher_id, season
    """
    logger.info("Fetching postseason pitcher stats for seasons %s", seasons)
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# Rolling momentum / form features
# ---------------------------------------------------------------------------
def get_rolling_form(
    player_ids: list[int],
    player_role: str = "batter",
    season: int | None = None,
) -> pd.DataFrame:
    """Fetch most recent rolling 15g/30g form data for a set of players.

    Returns the latest row per player from fact_player_form_rolling,
    giving rolling counts and rates over the prior 15 and 30 games.

    Parameters
    ----------
    player_ids : list[int]
        Player IDs to fetch.
    player_role : str
        'batter' or 'pitcher'.
    season : int, optional
        If provided, restricts to this season's data only.

    Returns
    -------
    pd.DataFrame
        One row per player with rolling 15g/30g counts.
    """
    if not player_ids:
        return pd.DataFrame()

    id_list = ", ".join(str(int(pid)) for pid in player_ids)
    season_filter = f"AND season = {int(season)}" if season else ""

    query = f"""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY player_id
                   ORDER BY game_date DESC, game_pk DESC
               ) AS rn
        FROM production.fact_player_form_rolling
        WHERE player_id IN ({id_list})
          AND player_role = :role
          {season_filter}
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    """
    logger.info(
        "Fetching rolling form for %d %ss", len(player_ids), player_role,
    )
    df = read_sql(query, {"role": player_role})
    if "rn" in df.columns:
        df = df.drop(columns=["rn"])
    return df


def get_rolling_hard_hit(
    batter_ids: list[int],
    season: int | None = None,
    window: int = 15,
) -> pd.DataFrame:
    """Compute rolling hard-hit% for batters from Statcast BIP data.

    Uses the most recent N games with BIP data per batter.

    Parameters
    ----------
    batter_ids : list[int]
        Batter IDs to fetch.
    season : int, optional
        Restrict to this season.
    window : int
        Number of recent games to include (default 15).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, bip_count, hard_hit_count, hard_hit_pct.
    """
    if not batter_ids:
        return pd.DataFrame()

    id_list = ", ".join(str(int(bid)) for bid in batter_ids)
    season_filter = f"AND dg.season = {int(season)}" if season else ""

    query = f"""
    WITH game_bip AS (
        SELECT
            fp.batter_id,
            fp.game_pk,
            dg.game_date,
            COUNT(*)                                   AS bip,
            COUNT(*) FILTER (WHERE bb.hard_hit = TRUE) AS hard_hits
        FROM production.sat_batted_balls bb
        JOIN production.fact_pa fp ON bb.pa_id = fp.pa_id
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE fp.batter_id IN ({id_list})
          AND dg.game_type = 'R'
          {season_filter}
        GROUP BY fp.batter_id, fp.game_pk, dg.game_date
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY batter_id
                   ORDER BY game_date DESC, game_pk DESC
               ) AS rn
        FROM game_bip
    )
    SELECT
        batter_id,
        SUM(bip)       AS bip_count,
        SUM(hard_hits)  AS hard_hit_count,
        SUM(hard_hits)::float / NULLIF(SUM(bip), 0) AS hard_hit_pct
    FROM ranked
    WHERE rn <= :window
    GROUP BY batter_id
    """
    logger.info(
        "Fetching rolling hard-hit%% for %d batters (window=%d)",
        len(batter_ids), window,
    )
    return read_sql(query, {"window": window})
