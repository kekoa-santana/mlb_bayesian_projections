"""
Feature collinearity analysis: proposed new features vs existing model features.

Tests whether proposed new features (K park factor, platoon splits, component
park factors, sprint speed, umpire BB zone, LD%, team defense OAA, Stuff+)
add independent signal beyond what the model already captures.

Computes:
  1. Pearson correlation matrix (all features)
  2. Partial correlations (controlling for existing features)
  3. VIF (Variance Inflation Factor)
  4. Incremental R-squared from simple regressions
  5. Summary recommendation: ADD / SKIP / MAYBE
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEASONS = [2023, 2024, 2025]
MIN_BATTER_PA = 300
MIN_PITCHER_IP = 50

# Safe logit to avoid inf
_CLIP = lambda x: np.clip(x, 1e-5, 1 - 1e-5)


# ============================================================================
# 1. DATA PULLS
# ============================================================================

def pull_batter_season_data() -> pd.DataFrame:
    """Pull qualified batter season-level data for 2023-2025."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    WITH stand_agg AS (
        SELECT
            fp.batter_id,
            dg.season,
            MODE() WITHIN GROUP (ORDER BY fp.batter_stand) AS batter_stand
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND fp.batter_stand IS NOT NULL
        GROUP BY fp.batter_id, dg.season
    ),
    pa_agg AS (
        SELECT
            fpa.batter_id,
            dp.player_name AS batter_name,
            sa.batter_stand,
            dg.season,
            COUNT(*) AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                     THEN 1 ELSE 0 END) AS bb,
            SUM(CASE WHEN fpa.events IN ('single','double','triple','home_run')
                     THEN 1 ELSE 0 END) AS hits,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END) AS hr,
            SUM(CASE WHEN fpa.events = 'hit_by_pitch'
                     THEN 1 ELSE 0 END) AS hbp
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON fpa.batter_id = dp.player_id
        LEFT JOIN stand_agg sa ON fpa.batter_id = sa.batter_id AND dg.season = sa.season
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
        GROUP BY fpa.batter_id, dp.player_name, sa.batter_stand, dg.season
    ),
    batted_agg AS (
        SELECT
            fpa.batter_id,
            dg.season,
            COUNT(*) AS bip,
            AVG(sbb.launch_speed) AS avg_exit_velo,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS hard_hit_pct,
            SUM(CASE WHEN sbb.launch_speed >= 98
                      AND sbb.launch_angle BETWEEN 26 AND 30
                      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS barrel_pct,
            AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba_avg,
            -- Batted ball types
            SUM(CASE WHEN sbb.launch_angle BETWEEN 10 AND 25 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS ld_pct,
            SUM(CASE WHEN sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS gb_pct,
            SUM(CASE WHEN sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS fb_pct
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND sbb.launch_speed IS NOT NULL
        GROUP BY fpa.batter_id, dg.season
    )
    SELECT
        pa.batter_id, pa.batter_name, pa.batter_stand, pa.season,
        pa.pa, pa.k, pa.bb, pa.hits, pa.hr, pa.hbp,
        ba.bip, ba.avg_exit_velo, ba.hard_hit_pct, ba.barrel_pct,
        ba.xwoba_avg, ba.ld_pct, ba.gb_pct, ba.fb_pct,
        ROUND((pa.k::numeric / pa.pa), 4) AS k_rate,
        ROUND((pa.bb::numeric / pa.pa), 4) AS bb_rate,
        ROUND((pa.hr::numeric / pa.pa), 4) AS hr_rate,
        ROUND(((pa.hits - pa.hr)::numeric
               / NULLIF(pa.pa - pa.k - pa.bb - pa.hbp - pa.hr, 0)), 4) AS babip
    FROM pa_agg pa
    LEFT JOIN batted_agg ba ON pa.batter_id = ba.batter_id AND pa.season = ba.season
    WHERE pa.pa >= {MIN_BATTER_PA}
    ORDER BY pa.pa DESC
    """
    print(f"  Pulling batter season data ({', '.join(str(s) for s in SEASONS)}, min {MIN_BATTER_PA} PA)...")
    df = read_sql(query, {})
    print(f"  -> {len(df)} batter-seasons")
    return df


def pull_batter_approach_data() -> pd.DataFrame:
    """Pull batter approach metrics: chase rate, whiff rate, zone contact."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT
        fp.batter_id,
        dg.season,
        SUM(fp.is_swing::int) AS swings,
        SUM(fp.is_whiff::int) AS whiffs,
        SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                 THEN fp.is_swing::int ELSE 0 END) AS z_swings,
        SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                 THEN fp.is_whiff::int ELSE 0 END) AS z_whiffs,
        SUM(CASE WHEN fp.zone > 9 OR fp.zone IS NULL
                 THEN 1 ELSE 0 END) AS ooz_pitches,
        SUM(CASE WHEN (fp.zone > 9 OR fp.zone IS NULL)
                  AND fp.is_swing
                 THEN 1 ELSE 0 END) AS chase_swings
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
      AND fp.pitch_type NOT IN ('PO', 'UN', 'SC', 'FA')
    GROUP BY fp.batter_id, dg.season
    HAVING SUM(fp.is_swing::int) >= 100
    """
    print("  Pulling batter approach data (chase, whiff, z_contact)...")
    df = read_sql(query, {})
    df["whiff_rate"] = (df["whiffs"] / df["swings"].replace(0, np.nan)).round(4)
    df["chase_rate"] = (df["chase_swings"] / df["ooz_pitches"].replace(0, np.nan)).round(4)
    df["z_contact_pct"] = ((df["z_swings"] - df["z_whiffs"]) / df["z_swings"].replace(0, np.nan)).round(4)
    print(f"  -> {len(df)} batter-seasons with approach data")
    return df[["batter_id", "season", "whiff_rate", "chase_rate", "z_contact_pct"]]


def pull_platoon_splits() -> pd.DataFrame:
    """Pull batter K% vs LHP and vs RHP separately."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT
        fpa.batter_id,
        dg.season,
        dp_p.pitch_hand,
        COUNT(*) AS pa,
        SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                 THEN 1 ELSE 0 END) AS k,
        SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                 THEN 1 ELSE 0 END) AS bb,
        SUM(CASE WHEN fpa.events IN ('single','double','triple','home_run')
                 THEN 1 ELSE 0 END) AS hits
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    LEFT JOIN production.dim_player dp_p ON fpa.pitcher_id = dp_p.player_id
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
      AND fpa.events IS NOT NULL
      AND dp_p.pitch_hand IN ('L', 'R')
    GROUP BY fpa.batter_id, dg.season, dp_p.pitch_hand
    """
    print("  Pulling platoon splits (K% vs LHP, K% vs RHP)...")
    df = read_sql(query, {})
    df["k_rate"] = (df["k"] / df["pa"].replace(0, np.nan)).round(4)
    df["bb_rate"] = (df["bb"] / df["pa"].replace(0, np.nan)).round(4)

    # Pivot to get vs_LHP and vs_RHP columns
    pivoted = df.pivot_table(
        index=["batter_id", "season"],
        columns="pitch_hand",
        values=["k_rate", "bb_rate", "pa"],
        aggfunc="first",
    )
    pivoted.columns = [f"{stat}_vs_{hand}HP" for stat, hand in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Compute platoon gap
    if "k_rate_vs_LHP" in pivoted.columns and "k_rate_vs_RHP" in pivoted.columns:
        pivoted["k_platoon_gap"] = pivoted["k_rate_vs_LHP"] - pivoted["k_rate_vs_RHP"]
    else:
        pivoted["k_platoon_gap"] = np.nan

    # Filter: need at least 50 PA vs each hand
    if "pa_vs_LHP" in pivoted.columns and "pa_vs_RHP" in pivoted.columns:
        pivoted = pivoted[
            (pivoted["pa_vs_LHP"] >= 50) & (pivoted["pa_vs_RHP"] >= 50)
        ].copy()

    print(f"  -> {len(pivoted)} batter-seasons with platoon splits")
    return pivoted


def pull_pitcher_season_data() -> pd.DataFrame:
    """Pull qualified pitcher season-level data."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT
        pb.pitcher_id,
        dp.player_name AS pitcher_name,
        dp.pitch_hand,
        dg.season,
        COUNT(DISTINCT pb.game_pk) AS games,
        SUM(pb.innings_pitched) AS ip,
        SUM(pb.strike_outs) AS k,
        SUM(pb.walks) AS bb,
        SUM(pb.home_runs) AS hr,
        SUM(pb.hits) AS hits_allowed,
        SUM(pb.batters_faced) AS batters_faced,
        SUM(pb.is_starter::int) AS starts,
        ROUND(SUM(pb.strike_outs)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4) AS k_rate,
        ROUND(SUM(pb.walks)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4) AS bb_rate,
        ROUND(SUM(pb.home_runs)::numeric / NULLIF(SUM(pb.batters_faced), 0), 4) AS hr_per_bf
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    JOIN production.dim_player dp ON pb.pitcher_id = dp.player_id
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
    GROUP BY pb.pitcher_id, dp.player_name, dp.pitch_hand, dg.season
    HAVING SUM(pb.innings_pitched) >= {MIN_PITCHER_IP}
    ORDER BY SUM(pb.batters_faced) DESC
    """
    print(f"  Pulling pitcher season data (min {MIN_PITCHER_IP} IP)...")
    df = read_sql(query, {})
    df["is_starter"] = (df["starts"] >= 3).astype(int)
    print(f"  -> {len(df)} pitcher-seasons")
    return df


def pull_pitcher_arsenal_summary() -> pd.DataFrame:
    """Pull pitcher aggregate pitch metrics: whiff rate, avg velo, gb%."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    WITH pitch_agg AS (
        SELECT
            fp.pitcher_id,
            dg.season,
            SUM(fp.is_swing::int) AS swings,
            SUM(fp.is_whiff::int) AS whiffs,
            COUNT(*) AS pitches,
            AVG(fp.release_speed) AS avg_velo,
            SUM(CASE WHEN fp.zone BETWEEN 1 AND 9
                     THEN 1 ELSE 0 END) AS zone_pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND fp.pitch_type IS NOT NULL
        GROUP BY fp.pitcher_id, dg.season
    ),
    batted_agg AS (
        SELECT
            fp.pitcher_id,
            dg.season,
            COUNT(*) AS bip,
            SUM(CASE WHEN sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS gb_pct,
            SUM(CASE WHEN sbb.launch_angle BETWEEN 10 AND 25 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS ld_pct_against,
            SUM(CASE WHEN sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS fb_pct_against,
            SUM(CASE WHEN sbb.hard_hit THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS hard_hit_pct_against
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.sat_batted_balls sbb ON fp.pitch_id = sbb.pitch_id
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND fp.is_bip = true
          AND sbb.launch_angle IS NOT NULL
        GROUP BY fp.pitcher_id, dg.season
    )
    SELECT
        pa.pitcher_id,
        pa.season,
        ROUND((pa.whiffs::numeric / NULLIF(pa.swings, 0)), 4) AS p_whiff_rate,
        ROUND(pa.avg_velo::numeric, 1) AS avg_velo,
        ROUND((pa.zone_pitches::numeric / NULLIF(pa.pitches, 0)), 4) AS zone_pct,
        ba.gb_pct AS p_gb_pct,
        ba.ld_pct_against AS p_ld_pct,
        ba.fb_pct_against AS p_fb_pct,
        ba.hard_hit_pct_against AS p_hard_hit_pct
    FROM pitch_agg pa
    LEFT JOIN batted_agg ba ON pa.pitcher_id = ba.pitcher_id AND pa.season = ba.season
    WHERE pa.swings >= 100
    """
    print("  Pulling pitcher arsenal summary (velo, whiff, gb%)...")
    df = read_sql(query, {})
    print(f"  -> {len(df)} pitcher-seasons with arsenal data")
    return df


def pull_park_factors() -> pd.DataFrame:
    """Compute multi-stat park factors (K, BB, H, HR) from home/away splits."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    WITH game_team AS (
        SELECT dg.venue_id, dg.game_pk, dg.season,
               fg.team_id,
               CASE WHEN fg.team_id = dg.home_team_id THEN 'home'
                    ELSE 'away' END as loc,
               SUM(fg.bat_h) as h, SUM(fg.bat_hr) as hr,
               SUM(fg.bat_r) as r, SUM(fg.bat_k) as k,
               SUM(fg.bat_bb) as bb, SUM(fg.bat_pa) as pa
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
        WHERE fg.player_role = 'batter' AND dg.game_type = 'R'
              AND dg.season IN ({season_list})
        GROUP BY dg.venue_id, dg.game_pk, dg.season,
                 fg.team_id, dg.home_team_id
    )
    SELECT venue_id,
           COUNT(DISTINCT game_pk) as games,
           SUM(CASE WHEN loc='home' THEN k END)::float /
               NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_k_rate,
           SUM(CASE WHEN loc='away' THEN k END)::float /
               NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_k_rate,
           SUM(CASE WHEN loc='home' THEN bb END)::float /
               NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_bb_rate,
           SUM(CASE WHEN loc='away' THEN bb END)::float /
               NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_bb_rate,
           SUM(CASE WHEN loc='home' THEN h END)::float /
               NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_h_rate,
           SUM(CASE WHEN loc='away' THEN h END)::float /
               NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_h_rate,
           SUM(CASE WHEN loc='home' THEN hr END)::float /
               NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) as home_hr_rate,
           SUM(CASE WHEN loc='away' THEN hr END)::float /
               NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) as away_hr_rate
    FROM game_team
    GROUP BY venue_id
    HAVING COUNT(DISTINCT game_pk) >= 50
    """
    print("  Computing park factors (K, BB, H, HR by venue)...")
    raw = read_sql(query, {})

    # Shrink toward 1.0 based on sample size
    shrinkage_games = 200
    reliability = (raw["games"] / shrinkage_games).clip(0, 1)

    for stat in ["k", "bb", "h", "hr"]:
        home_col = f"home_{stat}_rate"
        away_col = f"away_{stat}_rate"
        raw_pf = raw[home_col] / raw[away_col].replace(0, np.nan)
        raw[f"pf_{stat}"] = reliability * raw_pf + (1 - reliability) * 1.0

    print(f"  -> {len(raw)} venues with park factors")
    return raw[["venue_id", "games", "pf_k", "pf_bb", "pf_h", "pf_hr"]]


def pull_player_venues() -> pd.DataFrame:
    """Map players to their home venue (team)."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT DISTINCT ON (fg.player_id, dg.season)
        fg.player_id,
        dg.season,
        fg.team_id,
        dg.venue_id
    FROM production.fact_player_game_mlb fg
    JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
      AND fg.team_id = dg.home_team_id
    ORDER BY fg.player_id, dg.season, dg.game_date DESC
    """
    print("  Pulling player home venues...")
    df = read_sql(query, {})
    print(f"  -> {len(df)} player-venue mappings")
    return df


def pull_sprint_speed() -> pd.DataFrame:
    """Pull sprint speed from statcast."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT player_id, season, sprint_speed
    FROM staging.statcast_sprint_speed
    WHERE season IN ({season_list})
      AND sprint_speed IS NOT NULL
    """
    print("  Pulling sprint speed data...")
    df = read_sql(query, {})
    print(f"  -> {len(df)} sprint speed records")
    return df


def pull_umpire_tendencies() -> pd.DataFrame:
    """Pull per-umpire K and BB tendencies (shrunk toward league mean)."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    WITH ump_stats AS (
        SELECT
            du.hp_umpire_name,
            COUNT(DISTINCT du.game_pk) AS games,
            COUNT(fpa.pa_id) AS total_pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS total_k,
            SUM(CASE WHEN fpa.events = 'walk'
                     THEN 1 ELSE 0 END) AS total_bb
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        JOIN production.fact_pa fpa ON du.game_pk = fpa.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY du.hp_umpire_name
        HAVING COUNT(DISTINCT du.game_pk) >= 30
    )
    SELECT
        hp_umpire_name, games, total_pa, total_k, total_bb,
        ROUND((total_k::numeric / NULLIF(total_pa, 0)), 5) AS k_rate,
        ROUND((total_bb::numeric / NULLIF(total_pa, 0)), 5) AS bb_rate
    FROM ump_stats
    ORDER BY games DESC
    """
    print("  Pulling umpire tendencies...")
    df = read_sql(query, {})

    if df.empty:
        return df

    # Shrinkage
    shrinkage_k = 80.0
    df["reliability"] = df["games"] / (df["games"] + shrinkage_k)

    league_k = float(df["total_k"].sum() / df["total_pa"].sum())
    league_bb = float(df["total_bb"].sum() / df["total_pa"].sum())

    df["k_rate_shrunk"] = df["reliability"] * df["k_rate"] + (1 - df["reliability"]) * league_k
    df["bb_rate_shrunk"] = df["reliability"] * df["bb_rate"] + (1 - df["reliability"]) * league_bb

    df["ump_k_logit_lift"] = (
        logit(_CLIP(df["k_rate_shrunk"].values)) - logit(_CLIP(league_k))
    ).round(4)
    df["ump_bb_logit_lift"] = (
        logit(_CLIP(df["bb_rate_shrunk"].values)) - logit(_CLIP(league_bb))
    ).round(4)

    print(f"  -> {len(df)} umpires with tendencies")
    return df


def pull_team_defense() -> pd.DataFrame:
    """Pull team-level defense proxy: team BABIP against, and OAA if available."""
    season_list = ",".join(str(s) for s in SEASONS)

    # Try OAA first
    try:
        oaa_query = f"""
        SELECT
            team_id,
            season,
            SUM(outs_above_average) AS team_oaa
        FROM production.fact_fielding_oaa
        WHERE season IN ({season_list})
        GROUP BY team_id, season
        """
        oaa_df = read_sql(oaa_query, {})
        if not oaa_df.empty:
            print(f"  -> {len(oaa_df)} team-season OAA records from fact_fielding_oaa")
            return oaa_df
    except Exception:
        pass

    # Fallback: team BABIP against as defense proxy
    query = f"""
    WITH team_defense AS (
        SELECT
            dg.home_team_id AS team_id,
            dg.season,
            COUNT(*) AS bip_against,
            SUM(CASE WHEN fpa.events IN ('single','double','triple') THEN 1 ELSE 0 END) AS hits_on_bip
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.season IN ({season_list})
          AND dg.game_type = 'R'
          AND fpa.events IS NOT NULL
          AND fpa.events NOT IN ('strikeout','strikeout_double_play','walk',
                                  'intent_walk','hit_by_pitch','home_run',
                                  'catcher_interf','sac_bunt','sac_fly',
                                  'sac_fly_double_play')
          -- Only PA where defensive team is the home team
          AND fpa.batter_id != ANY(
              SELECT fg.player_id FROM production.fact_player_game_mlb fg
              WHERE fg.game_pk = dg.game_pk AND fg.team_id = dg.home_team_id
              LIMIT 1
          )
        GROUP BY dg.home_team_id, dg.season
    )
    SELECT team_id, season, bip_against,
           ROUND((hits_on_bip::numeric / NULLIF(bip_against, 0)), 4) AS team_babip_against
    FROM team_defense
    WHERE bip_against >= 500
    """
    print("  Pulling team defense proxy (team BABIP against)...")
    try:
        df = read_sql(query, {})
        # Invert: lower BABIP = better defense, so proxy OAA ~ -(BABIP - league_avg)
        if not df.empty:
            league_babip = df["team_babip_against"].mean()
            df["team_oaa_proxy"] = -(df["team_babip_against"] - league_babip) * 1000
            print(f"  -> {len(df)} team-seasons with BABIP defense proxy")
        return df
    except Exception as e:
        print(f"  -> Team defense query failed: {e}")
        return pd.DataFrame()


def pull_pitcher_game_logs() -> pd.DataFrame:
    """Pull game-level pitcher logs for regression analysis."""
    season_list = ",".join(str(s) for s in SEASONS)
    query = f"""
    SELECT
        pb.game_pk,
        pb.pitcher_id,
        dg.season,
        dg.venue_id,
        dg.home_team_id,
        pb.strike_outs AS game_k,
        pb.walks AS game_bb,
        pb.hits AS game_h,
        pb.home_runs AS game_hr,
        pb.batters_faced AS game_bf,
        pb.innings_pitched AS game_ip,
        pb.is_starter
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
      AND pb.is_starter = true
      AND pb.batters_faced >= 10
    ORDER BY pb.game_pk
    """
    print("  Pulling pitcher game logs for regression...")
    df = read_sql(query, {})
    print(f"  -> {len(df)} starter game logs")
    return df


# ============================================================================
# 2. ANALYSIS FUNCTIONS
# ============================================================================

def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each column."""
    from numpy.linalg import LinAlgError

    vif_data = {}
    X_clean = X.dropna()
    if len(X_clean) < X_clean.shape[1] + 2:
        return pd.Series(dtype=float)

    for col in X_clean.columns:
        y = X_clean[col].values
        others = X_clean.drop(columns=[col]).values
        # Add intercept
        others = np.column_stack([np.ones(len(others)), others])
        try:
            beta = np.linalg.lstsq(others, y, rcond=None)[0]
            y_hat = others @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = 1 - ss_res / max(ss_tot, 1e-10)
            vif_data[col] = 1.0 / max(1 - r_sq, 1e-10)
        except (LinAlgError, ValueError):
            vif_data[col] = np.nan

    return pd.Series(vif_data)


def partial_correlation(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> tuple[float, float]:
    """Compute partial correlation of x and y, controlling for Z.

    Returns (partial_r, p_value).
    """
    # Ensure arrays are float
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Build a clean mask: no NaN in x, y, or any column of Z
    mask = np.isfinite(x) & np.isfinite(y)
    for col_idx in range(Z.shape[1]):
        mask &= np.isfinite(Z[:, col_idx])
    x_c, y_c, Z_c = x[mask], y[mask], Z[mask]

    n = len(x_c)
    k = Z_c.shape[1]
    if n < k + 5:
        return np.nan, np.nan

    # Regress x on Z
    Z_int = np.column_stack([np.ones(n), Z_c])
    beta_x, res_x, _, _ = np.linalg.lstsq(Z_int, x_c, rcond=None)
    resid_x = x_c - Z_int @ beta_x

    # Regress y on Z
    beta_y, res_y, _, _ = np.linalg.lstsq(Z_int, y_c, rcond=None)
    resid_y = y_c - Z_int @ beta_y

    std_x = np.std(resid_x)
    std_y = np.std(resid_y)
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0, 1.0

    r, p = stats.pearsonr(resid_x, resid_y)
    return round(float(r), 4), round(float(p), 6)


def incremental_r_squared(
    y: np.ndarray, X_base: np.ndarray, x_new: np.ndarray
) -> tuple[float, float, float]:
    """Compute incremental R-squared when adding x_new to X_base.

    Returns (r2_base, r2_full, delta_r2).
    """
    mask = ~(np.isnan(y) | np.any(np.isnan(X_base), axis=1) | np.isnan(x_new))
    y, X_base, x_new = y[mask], X_base[mask], x_new[mask]

    if len(y) < X_base.shape[1] + 3:
        return np.nan, np.nan, np.nan

    # Baseline model
    X_b = np.column_stack([np.ones(len(X_base)), X_base])
    beta_b = np.linalg.lstsq(X_b, y, rcond=None)[0]
    y_hat_b = X_b @ beta_b
    ss_res_b = np.sum((y - y_hat_b) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_base = 1 - ss_res_b / max(ss_tot, 1e-10)

    # Full model (add new feature)
    X_f = np.column_stack([X_b, x_new])
    beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
    y_hat_f = X_f @ beta_f
    ss_res_f = np.sum((y - y_hat_f) ** 2)
    r2_full = 1 - ss_res_f / max(ss_tot, 1e-10)

    delta = r2_full - r2_base
    return round(r2_base, 4), round(r2_full, 4), round(delta, 4)


# ============================================================================
# 3. MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("FEATURE COLLINEARITY ANALYSIS")
    print("Proposed new features vs existing model features")
    print("=" * 80)
    print()

    # ----- Pull all data -----
    print("STEP 1: Pulling data from database...")
    print("-" * 50)
    batters = pull_batter_season_data()
    approach = pull_batter_approach_data()
    platoon = pull_platoon_splits()
    pitchers = pull_pitcher_season_data()
    arsenal = pull_pitcher_arsenal_summary()
    park_factors = pull_park_factors()
    player_venues = pull_player_venues()
    sprint = pull_sprint_speed()
    umpires = pull_umpire_tendencies()
    team_def = pull_team_defense()
    game_logs = pull_pitcher_game_logs()
    print()

    # ----- Merge batter features -----
    print("STEP 2: Building feature matrices...")
    print("-" * 50)

    # Batter feature matrix
    batter_df = batters.merge(approach, on=["batter_id", "season"], how="left")
    batter_df = batter_df.merge(
        platoon[["batter_id", "season", "k_rate_vs_LHP", "k_rate_vs_RHP", "k_platoon_gap"]],
        on=["batter_id", "season"], how="left"
    )

    # Add sprint speed
    batter_df = batter_df.merge(
        sprint.rename(columns={"player_id": "batter_id"}),
        on=["batter_id", "season"], how="left"
    )

    # Add park factors via home venue
    batter_venues = player_venues.rename(columns={"player_id": "batter_id"})
    batter_df = batter_df.merge(
        batter_venues[["batter_id", "season", "venue_id"]],
        on=["batter_id", "season"], how="left"
    )
    batter_df = batter_df.merge(
        park_factors[["venue_id", "pf_k", "pf_bb", "pf_h", "pf_hr"]],
        on="venue_id", how="left"
    )

    # Add team defense proxy
    if not team_def.empty:
        batter_venues_team = batter_venues[["batter_id", "season", "venue_id"]].merge(
            player_venues[["player_id", "season", "team_id"]].rename(
                columns={"player_id": "batter_id"}
            ),
            on=["batter_id", "season"], how="left"
        )
        # For batters, the relevant defense is opposing team's. We approximate
        # by attaching their own team's defense (it is still useful for cross-
        # correlation testing at the season level).
        if "team_oaa" in team_def.columns:
            def_col = "team_oaa"
        elif "team_oaa_proxy" in team_def.columns:
            def_col = "team_oaa_proxy"
        else:
            def_col = None

        if def_col:
            batter_df = batter_df.merge(
                batter_venues_team[["batter_id", "season", "team_id"]],
                on=["batter_id", "season"], how="left"
            )
            batter_df = batter_df.merge(
                team_def[["team_id", "season", def_col]],
                on=["team_id", "season"], how="left"
            )

    # Pitcher feature matrix
    pitcher_df = pitchers.merge(arsenal, on=["pitcher_id", "season"], how="left")

    pitcher_venues = player_venues.rename(columns={"player_id": "pitcher_id"})
    pitcher_df = pitcher_df.merge(
        pitcher_venues[["pitcher_id", "season", "venue_id"]],
        on=["pitcher_id", "season"], how="left"
    )
    pitcher_df = pitcher_df.merge(
        park_factors[["venue_id", "pf_k", "pf_bb", "pf_h", "pf_hr"]],
        on="venue_id", how="left"
    )

    print(f"  Batter feature matrix: {batter_df.shape}")
    print(f"  Pitcher feature matrix: {pitcher_df.shape}")
    print()

    # ===================================================================
    # SECTION A: BATTER-SIDE COLLINEARITY
    # ===================================================================
    print("=" * 80)
    print("SECTION A: BATTER-SIDE FEATURE COLLINEARITY")
    print("=" * 80)
    print()

    # Define existing and proposed features for batters
    existing_batter = ["k_rate", "bb_rate", "hr_rate", "chase_rate", "whiff_rate",
                       "z_contact_pct", "barrel_pct", "hard_hit_pct", "avg_exit_velo"]
    proposed_batter = {
        "pf_k": "K Park Factor",
        "pf_bb": "BB Park Factor",
        "pf_h": "H Park Factor",
        "k_rate_vs_LHP": "Platoon K% vs LHP",
        "k_rate_vs_RHP": "Platoon K% vs RHP",
        "k_platoon_gap": "Platoon K% Gap (LHP-RHP)",
        "sprint_speed": "Sprint Speed",
        "ld_pct": "Line Drive %",
        "babip": "BABIP",
    }

    # Add defense proxy if available
    if "team_oaa" in batter_df.columns:
        proposed_batter["team_oaa"] = "Team OAA"
    elif "team_oaa_proxy" in batter_df.columns:
        proposed_batter["team_oaa_proxy"] = "Team OAA Proxy (BABIP)"

    # Filter to columns that exist
    available_existing_b = [c for c in existing_batter if c in batter_df.columns]
    available_proposed_b = {k: v for k, v in proposed_batter.items() if k in batter_df.columns}

    all_batter_features = available_existing_b + list(available_proposed_b.keys())
    batter_corr = batter_df[all_batter_features].corr()

    # Print correlation matrix (proposed vs existing)
    print("A1. Correlation Matrix: Proposed vs Existing Batter Features")
    print("-" * 70)
    print()

    # Print header
    header_existing = [f"{c[:12]:>12}" for c in available_existing_b]
    print(f"{'PROPOSED':<22} " + " ".join(header_existing))
    print("-" * (22 + 13 * len(available_existing_b)))

    for feat, label in available_proposed_b.items():
        row_vals = []
        for ex in available_existing_b:
            r = batter_corr.loc[feat, ex] if feat in batter_corr.index and ex in batter_corr.columns else np.nan
            if abs(r) > 0.70:
                flag = "***"
            elif abs(r) > 0.50:
                flag = " **"
            elif abs(r) > 0.30:
                flag = "  *"
            else:
                flag = "   "
            row_vals.append(f"{r:>8.3f}{flag}")
        print(f"{label:<22} " + " ".join(row_vals))

    print()
    print("  *** = |r| > 0.70 (HIGH collinearity)")
    print("   ** = |r| > 0.50 (MODERATE collinearity)")
    print("    * = |r| > 0.30 (low-moderate)")
    print()

    # A2. Partial correlations and VIF
    print("A2. Partial Correlations (Independent Signal After Controlling for Existing)")
    print("-" * 70)
    print()

    targets = {"k_rate": "K%", "bb_rate": "BB%", "hr_rate": "HR/PA"}

    results_batter = []
    for feat, label in available_proposed_b.items():
        row = {"Feature": label, "Column": feat}

        # Max absolute correlation with existing
        corrs_with_existing = []
        for ex in available_existing_b:
            r = batter_corr.loc[feat, ex] if feat in batter_corr.index else np.nan
            corrs_with_existing.append((ex, r))

        max_corr_pair = max(corrs_with_existing, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
        row["Max |r| with existing"] = f"{max_corr_pair[1]:.3f} ({max_corr_pair[0]})"

        # For partial correlations, create per-feature clean subset
        needed_cols = available_existing_b + [feat]
        needed_cols_present = [c for c in needed_cols if c in batter_df.columns]
        batter_sub = batter_df.dropna(subset=needed_cols_present).copy()

        if len(batter_sub) < 30:
            for target_col, target_label in targets.items():
                row[f"Partial r ({target_label})"] = np.nan
                row[f"p-value ({target_label})"] = np.nan
            row["n_clean"] = len(batter_sub)
            results_batter.append(row)
            continue

        x_sub = batter_sub[feat].values
        row["n_clean"] = len(batter_sub)

        # Partial correlations with targets
        # IMPORTANT: exclude the target itself from the control set Z
        for target_col, target_label in targets.items():
            controls = [c for c in available_existing_b if c != target_col]
            Z_sub = batter_sub[controls].values
            y_sub = batter_sub[target_col].values
            pr, pp = partial_correlation(x_sub, y_sub, Z_sub)
            row[f"Partial r ({target_label})"] = pr
            row[f"p-value ({target_label})"] = pp

        results_batter.append(row)

    results_df = pd.DataFrame(results_batter)
    for _, row in results_df.iterrows():
        n_clean = row.get("n_clean", "?")
        print(f"  {row['Feature']:<28} (n={n_clean})")
        print(f"    Max |r| with existing: {row['Max |r| with existing']}")
        for target_col, target_label in targets.items():
            pr = row[f"Partial r ({target_label})"]
            pp = row[f"p-value ({target_label})"]
            if np.isnan(pr):
                print(f"    Partial r with {target_label}: insufficient data")
            else:
                sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
                print(f"    Partial r with {target_label}: {pr:>7.4f}  (p={pp:.6f} {sig})")
        print()

    # A3. VIF for proposed features when added to existing
    print("A3. VIF (Variance Inflation Factor) - Batter Features")
    print("-" * 70)
    print()
    print("  VIF > 10 = severe multicollinearity, VIF > 5 = concerning")
    print()

    for feat, label in available_proposed_b.items():
        cols = available_existing_b + [feat]
        cols_present = [c for c in cols if c in batter_df.columns]
        sub = batter_df[cols_present].dropna()
        if len(sub) > len(cols_present) + 2 and feat in sub.columns:
            vifs = compute_vif(sub)
            feat_vif = vifs.get(feat, np.nan)
            flag = " << SEVERE" if feat_vif > 10 else " << HIGH" if feat_vif > 5 else ""
            print(f"  {label:<28} VIF = {feat_vif:>6.2f}{flag}")
        else:
            print(f"  {label:<28} VIF = insufficient data")

    print()

    # ===================================================================
    # SECTION B: PITCHER-SIDE COLLINEARITY
    # ===================================================================
    print("=" * 80)
    print("SECTION B: PITCHER-SIDE FEATURE COLLINEARITY")
    print("=" * 80)
    print()

    existing_pitcher = ["k_rate", "bb_rate", "hr_per_bf", "p_whiff_rate", "avg_velo",
                        "zone_pct", "p_gb_pct"]
    proposed_pitcher = {
        "pf_k": "K Park Factor",
        "pf_bb": "BB Park Factor",
        "pf_h": "H Park Factor",
        "p_ld_pct": "LD% Against",
        "p_hard_hit_pct": "Hard Hit% Against",
    }

    available_existing_p = [c for c in existing_pitcher if c in pitcher_df.columns]
    available_proposed_p = {k: v for k, v in proposed_pitcher.items() if k in pitcher_df.columns}

    all_pitcher_features = available_existing_p + list(available_proposed_p.keys())
    pitcher_corr = pitcher_df[all_pitcher_features].corr()

    print("B1. Correlation Matrix: Proposed vs Existing Pitcher Features")
    print("-" * 70)
    print()

    header_existing_p = [f"{c[:12]:>12}" for c in available_existing_p]
    print(f"{'PROPOSED':<22} " + " ".join(header_existing_p))
    print("-" * (22 + 13 * len(available_existing_p)))

    for feat, label in available_proposed_p.items():
        row_vals = []
        for ex in available_existing_p:
            r = pitcher_corr.loc[feat, ex] if feat in pitcher_corr.index and ex in pitcher_corr.columns else np.nan
            if abs(r) > 0.70:
                flag = "***"
            elif abs(r) > 0.50:
                flag = " **"
            elif abs(r) > 0.30:
                flag = "  *"
            else:
                flag = "   "
            row_vals.append(f"{r:>8.3f}{flag}")
        print(f"{label:<22} " + " ".join(row_vals))

    print()
    print("  *** = |r| > 0.70 (HIGH collinearity)")
    print("   ** = |r| > 0.50 (MODERATE collinearity)")
    print("    * = |r| > 0.30 (low-moderate)")
    print()

    # B2. Partial correlations for pitcher features
    print("B2. Partial Correlations (Independent Signal After Controlling for Existing)")
    print("-" * 70)
    print()

    targets_p = {"k_rate": "K%", "bb_rate": "BB%", "hr_per_bf": "HR/BF"}

    for feat, label in available_proposed_p.items():
        needed_cols = available_existing_p + [feat]
        needed_present = [c for c in needed_cols if c in pitcher_df.columns]
        p_sub = pitcher_df.dropna(subset=needed_present).copy()

        print(f"  {label:<28} (n={len(p_sub)})")
        if len(p_sub) < 30:
            print("    Insufficient data for partial correlations")
            print()
            continue

        x_sub = p_sub[feat].values

        for target_col, target_label in targets_p.items():
            # Exclude the target from controls to avoid perfect collinearity
            controls = [c for c in available_existing_p if c != target_col]
            Z_p_sub = p_sub[controls].values
            y_sub = p_sub[target_col].values
            pr, pp = partial_correlation(x_sub, y_sub, Z_p_sub)
            if np.isnan(pr):
                print(f"    Partial r with {target_label}: insufficient data")
            else:
                sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
                print(f"    Partial r with {target_label}: {pr:>7.4f}  (p={pp:.6f} {sig})")
        print()

    # B3. VIF for pitcher features
    print("B3. VIF - Pitcher Features")
    print("-" * 70)
    print()

    for feat, label in available_proposed_p.items():
        cols = available_existing_p + [feat]
        sub = pitcher_df[cols].dropna()
        if len(sub) > len(cols) + 2:
            vifs = compute_vif(sub)
            feat_vif = vifs.get(feat, np.nan)
            flag = " << SEVERE" if feat_vif > 10 else " << HIGH" if feat_vif > 5 else ""
            print(f"  {label:<28} VIF = {feat_vif:>6.2f}{flag}")
        else:
            print(f"  {label:<28} VIF = insufficient data")

    print()

    # ===================================================================
    # SECTION C: UMPIRE BB ZONE ANALYSIS
    # ===================================================================
    print("=" * 80)
    print("SECTION C: UMPIRE BB ZONE ANALYSIS")
    print("=" * 80)
    print()

    if not umpires.empty:
        print("Umpire K logit lift vs BB logit lift correlation:")
        k_lift = umpires["ump_k_logit_lift"]
        bb_lift = umpires["ump_bb_logit_lift"]
        r, p = stats.pearsonr(k_lift.dropna(), bb_lift.dropna())
        print(f"  Pearson r = {r:.4f}  (p = {p:.4f})")
        print()
        print("  If highly correlated, umpire BB zone adds NO independent signal")
        print("  beyond the K logit lift already in the model.")
        print()

        print("Umpire BB lift distribution:")
        print(f"  Mean:  {bb_lift.mean():.4f}")
        print(f"  Std:   {bb_lift.std():.4f}")
        print(f"  Range: [{bb_lift.min():.4f}, {bb_lift.max():.4f}]")
        print(f"  IQR:   [{bb_lift.quantile(0.25):.4f}, {bb_lift.quantile(0.75):.4f}]")
        print()

        # Variance explained
        unique_var = 1 - r**2
        print(f"  Unique variance in ump BB lift not explained by ump K lift: {unique_var:.4f}")
        print(f"  -> {'Meaningful independent signal' if unique_var > 0.3 else 'Mostly redundant with ump K lift'}")
    print()

    # ===================================================================
    # SECTION D: GAME-LEVEL INCREMENTAL R-SQUARED
    # ===================================================================
    print("=" * 80)
    print("SECTION D: GAME-LEVEL INCREMENTAL R-SQUARED (Starter Games)")
    print("=" * 80)
    print()

    # Merge pitcher season stats onto game logs
    gl = game_logs.merge(
        pitcher_df[["pitcher_id", "season", "k_rate", "bb_rate", "hr_per_bf",
                     "p_whiff_rate", "avg_velo", "zone_pct", "p_gb_pct",
                     "pf_k", "pf_bb", "pf_h",
                     "p_ld_pct", "p_hard_hit_pct"]],
        on=["pitcher_id", "season"], how="left"
    )

    # Game-level targets: game K, game BB, game H (per BF for rates)
    gl["game_k_rate"] = gl["game_k"] / gl["game_bf"].replace(0, np.nan)
    gl["game_bb_rate"] = gl["game_bb"] / gl["game_bf"].replace(0, np.nan)
    gl["game_h_rate"] = gl["game_h"] / gl["game_bf"].replace(0, np.nan)

    game_existing = ["k_rate", "bb_rate", "hr_per_bf", "p_whiff_rate",
                     "avg_velo", "zone_pct", "p_gb_pct"]
    game_proposed = {
        "pf_k": "K Park Factor",
        "pf_bb": "BB Park Factor",
        "pf_h": "H Park Factor",
        "p_ld_pct": "LD% Against",
        "p_hard_hit_pct": "Hard Hit% Against",
    }

    game_targets = {
        "game_k_rate": "Game K Rate",
        "game_bb_rate": "Game BB Rate",
        "game_h_rate": "Game H Rate",
    }

    available_game_existing = [c for c in game_existing if c in gl.columns]
    available_game_proposed = {k: v for k, v in game_proposed.items() if k in gl.columns}

    print(f"  Game logs with features: {len(gl.dropna(subset=available_game_existing))} games")
    print()

    for target_col, target_label in game_targets.items():
        print(f"  Target: {target_label}")
        print(f"  {'Feature':<24} {'R2 base':>8} {'R2 full':>8} {'Delta R2':>9}  Sig")
        print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*9}  ---")

        y = gl[target_col].values
        X_base = gl[available_game_existing].values

        for feat, label in available_game_proposed.items():
            x_new = gl[feat].values
            r2_b, r2_f, delta = incremental_r_squared(y, X_base, x_new)

            if not np.isnan(delta):
                # F-test for significance
                n = np.sum(~(np.isnan(y) | np.any(np.isnan(X_base), axis=1) | np.isnan(x_new)))
                p_base = len(available_game_existing)
                f_stat = (delta / 1) / ((1 - r2_f) / max(n - p_base - 2, 1))
                p_val = 1 - stats.f.cdf(f_stat, 1, max(n - p_base - 2, 1))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                print(f"  {label:<24} {r2_b:>8.4f} {r2_f:>8.4f} {delta:>+9.4f}  {sig}")
            else:
                print(f"  {label:<24} {'N/A':>8} {'N/A':>8} {'N/A':>9}")

        print()

    # ===================================================================
    # SECTION E: SUMMARY RECOMMENDATIONS
    # ===================================================================
    print("=" * 80)
    print("SECTION E: SUMMARY RECOMMENDATIONS")
    print("=" * 80)
    print()

    recommendations = []

    # 1. K Park Factor
    rec = {"Feature": "K Park Factor"}
    pf_k_corrs = []
    for ex in available_existing_b:
        if "pf_k" in batter_corr.index and ex in batter_corr.columns:
            pf_k_corrs.append(abs(batter_corr.loc["pf_k", ex]))
    rec["Max |r| existing"] = max(pf_k_corrs) if pf_k_corrs else np.nan

    # Partial correlation with K%
    if "pf_k" in batter_df.columns:
        needed = available_existing_b + ["pf_k"]
        sub = batter_df.dropna(subset=[c for c in needed if c in batter_df.columns])
        if len(sub) >= 30:
            controls = [c for c in available_existing_b if c != "k_rate"]
            pr, pp = partial_correlation(
                sub["pf_k"].values, sub["k_rate"].values,
                sub[controls].values
            )
            rec["Partial r (K%)"] = pr
            rec["p-value"] = pp
        else:
            rec["Partial r (K%)"] = np.nan
            rec["p-value"] = np.nan
    else:
        rec["Partial r (K%)"] = np.nan
        rec["p-value"] = np.nan

    # Determine recommendation based on effect size
    max_r = rec["Max |r| existing"]
    pr_val = rec.get("Partial r (K%)", np.nan)
    if not np.isnan(max_r) and max_r < 0.30 and not np.isnan(pr_val) and abs(pr_val) > 0.02:
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Low collinearity with existing features, park effects are venue-level context"
    elif not np.isnan(max_r) and max_r < 0.50:
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Moderate collinearity but captures venue-specific signal not in player-level features"
    else:
        rec["Verdict"] = "MAYBE"
        rec["Reason"] = "Check if park context is already implicit in player posteriors"
    recommendations.append(rec)

    # 2. Platoon Splits
    rec = {"Feature": "Platoon Splits (K% vs LHP/RHP)"}
    plat_corrs = []
    for feat in ["k_rate_vs_LHP", "k_rate_vs_RHP", "k_platoon_gap"]:
        if feat in batter_corr.index:
            for ex in available_existing_b:
                if ex in batter_corr.columns:
                    plat_corrs.append(abs(batter_corr.loc[feat, ex]))
    rec["Max |r| existing"] = max(plat_corrs) if plat_corrs else np.nan

    # K_vs_LHP and K_vs_RHP are essentially splits of K%, so high correlation expected
    if "k_rate_vs_LHP" in batter_corr.index and "k_rate" in batter_corr.columns:
        r_lhp = batter_corr.loc["k_rate_vs_LHP", "k_rate"]
        r_rhp = batter_corr.loc["k_rate_vs_RHP", "k_rate"]
        gap_corr = batter_corr.loc["k_platoon_gap", "k_rate"] if "k_platoon_gap" in batter_corr.index else np.nan
        rec["r(K_vs_LHP, K%)"] = round(r_lhp, 3)
        rec["r(K_vs_RHP, K%)"] = round(r_rhp, 3)
        rec["r(K_gap, K%)"] = round(gap_corr, 3) if not np.isnan(gap_corr) else np.nan

        if "k_platoon_gap" in batter_df.columns:
            needed = available_existing_b + ["k_platoon_gap"]
            sub = batter_df.dropna(subset=[c for c in needed if c in batter_df.columns])
            if len(sub) >= 30:
                x = sub["k_platoon_gap"].values
                y = sub["k_rate"].values
                controls = [c for c in available_existing_b if c != "k_rate"]
                pr, pp = partial_correlation(x, y, sub[controls].values)
                rec["Partial r (gap->K%)"] = pr
                rec["p-value"] = pp

    if not np.isnan(rec["Max |r| existing"]) and rec["Max |r| existing"] > 0.70:
        rec["Verdict"] = "SKIP (splits) / MAYBE (gap)"
        rec["Reason"] = "K% vs LHP/RHP highly correlated with overall K%. The platoon gap may add signal."
    else:
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Splits provide context beyond overall rate"
    recommendations.append(rec)

    # 3. Component Park Factors (H, BB)
    rec = {"Feature": "Component Park Factors (H, BB)"}
    comp_corrs = []
    for feat in ["pf_h", "pf_bb"]:
        if feat in batter_corr.index:
            for ex in available_existing_b:
                if ex in batter_corr.columns:
                    comp_corrs.append(abs(batter_corr.loc[feat, ex]))
    rec["Max |r| existing"] = max(comp_corrs) if comp_corrs else np.nan

    # Check pf_h vs pf_hr, pf_bb vs pf_k correlations
    if "pf_h" in batter_corr.index and "pf_hr" in batter_corr.columns:
        rec["r(pf_h, pf_hr)"] = round(batter_corr.loc["pf_h", "pf_hr"] if "pf_hr" in batter_corr.index else np.nan, 3)
    if "pf_bb" in batter_corr.index and "pf_k" in batter_corr.columns:
        rec["r(pf_bb, pf_k)"] = round(batter_corr.loc["pf_bb", "pf_k"] if "pf_k" in batter_corr.index else np.nan, 3)

    rec["Verdict"] = "ADD"
    rec["Reason"] = "Park factors are venue-level context, orthogonal to player-level features"
    recommendations.append(rec)

    # 4. Sprint Speed
    rec = {"Feature": "Sprint Speed"}
    sp_corrs = []
    if "sprint_speed" in batter_corr.index:
        for ex in available_existing_b:
            if ex in batter_corr.columns:
                sp_corrs.append(abs(batter_corr.loc["sprint_speed", ex]))
    rec["Max |r| existing"] = max(sp_corrs) if sp_corrs else np.nan

    if "sprint_speed" in batter_df.columns:
        needed = available_existing_b + ["sprint_speed", "babip"]
        sub = batter_df.dropna(subset=[c for c in needed if c in batter_df.columns])
        if len(sub) >= 30:
            x = sub["sprint_speed"].values
            y = sub["babip"].values if "babip" in sub.columns else sub["k_rate"].values
            # BABIP not in existing controls, so use all existing as controls
            pr, pp = partial_correlation(x, y, sub[available_existing_b].values)
        rec["Partial r (BABIP)"] = pr
        rec["p-value"] = pp

    if not np.isnan(rec["Max |r| existing"]) and rec["Max |r| existing"] < 0.30:
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Low collinearity, sprint speed captures infield hit/BABIP signal not in approach metrics"
    elif not np.isnan(rec["Max |r| existing"]) and rec["Max |r| existing"] < 0.50:
        rec["Verdict"] = "MAYBE"
        rec["Reason"] = "Moderate correlation; may partially overlap with batted ball profile"
    else:
        rec["Verdict"] = "SKIP"
        rec["Reason"] = "High correlation with existing features"
    recommendations.append(rec)

    # 5. Umpire BB Zone
    rec = {"Feature": "Umpire BB Zone"}
    if not umpires.empty:
        r_ump, _ = stats.pearsonr(
            umpires["ump_k_logit_lift"].dropna(),
            umpires["ump_bb_logit_lift"].dropna()
        )
        rec["r(ump_K, ump_BB)"] = round(r_ump, 3)
        unique_var = 1 - r_ump**2
        rec["Unique variance"] = round(unique_var, 3)

        if unique_var > 0.30:
            rec["Verdict"] = "ADD"
            rec["Reason"] = f"Umpire BB zone has {unique_var:.0%} unique variance beyond K zone"
        elif unique_var > 0.15:
            rec["Verdict"] = "MAYBE"
            rec["Reason"] = f"Only {unique_var:.0%} unique variance; marginal over ump K lift"
        else:
            rec["Verdict"] = "SKIP"
            rec["Reason"] = f"Only {unique_var:.0%} unique variance; redundant with ump K lift"
    else:
        rec["Verdict"] = "N/A"
        rec["Reason"] = "No umpire data available"
    recommendations.append(rec)

    # 6. LD%
    rec = {"Feature": "Line Drive % (LD%)"}
    ld_corrs = []
    if "ld_pct" in batter_corr.index:
        for ex in available_existing_b:
            if ex in batter_corr.columns:
                ld_corrs.append(abs(batter_corr.loc["ld_pct", ex]))
    rec["Max |r| existing"] = max(ld_corrs) if ld_corrs else np.nan

    if "ld_pct" in batter_df.columns:
        needed = available_existing_b + ["ld_pct"]
        sub = batter_df.dropna(subset=[c for c in needed if c in batter_df.columns])
        x = sub["ld_pct"].values if len(sub) >= 30 else np.array([])
        for target_col, target_label in [("babip", "BABIP"), ("hr_rate", "HR/PA")]:
            if target_col in sub.columns and len(x) > 0:
                y = sub[target_col].values
                # Exclude target from controls if it happens to be in existing set
                controls = [c for c in available_existing_b if c != target_col]
                pr, pp = partial_correlation(x, y, sub[controls].values)
                rec[f"Partial r ({target_label})"] = pr
                rec[f"p-value ({target_label})"] = pp

    if not np.isnan(rec["Max |r| existing"]) and rec["Max |r| existing"] > 0.50:
        rec["Verdict"] = "SKIP"
        rec["Reason"] = "Highly correlated with existing batted ball features (barrel%, hard_hit%)"
    elif not np.isnan(rec["Max |r| existing"]) and rec["Max |r| existing"] > 0.30:
        rec["Verdict"] = "MAYBE"
        rec["Reason"] = "Moderate overlap with barrel%/hard_hit%; check partial correlation with BABIP"
    else:
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Independent of existing features"
    recommendations.append(rec)

    # 7. Team Defense OAA
    def_feat = "team_oaa" if "team_oaa" in batter_df.columns else ("team_oaa_proxy" if "team_oaa_proxy" in batter_df.columns else None)
    rec = {"Feature": "Team Defense OAA"}
    if def_feat:
        def_corrs = []
        if def_feat in batter_corr.index:
            for ex in available_existing_b:
                if ex in batter_corr.columns:
                    def_corrs.append(abs(batter_corr.loc[def_feat, ex]))
        rec["Max |r| existing"] = max(def_corrs) if def_corrs else np.nan
        rec["Verdict"] = "ADD"
        rec["Reason"] = "Team-level context, fundamentally different from player-level approach/quality metrics"
    else:
        rec["Max |r| existing"] = np.nan
        rec["Verdict"] = "ADD (when available)"
        rec["Reason"] = "Team defense is orthogonal to all player-level features; high a priori value"
    recommendations.append(rec)

    # 8. Stuff+
    rec = {"Feature": "Stuff+"}
    rec["Max |r| existing"] = np.nan
    rec["Verdict"] = "ADD (when available)"
    rec["Reason"] = "Not in DB. Expected to correlate with whiff_rate and avg_velo but captures additional pitch quality."
    recommendations.append(rec)

    # Print summary table
    print(f"{'Feature':<32} {'Max |r|':>8} {'Verdict':<12} Reason")
    print(f"{'-'*32} {'-'*8} {'-'*12} {'-'*50}")

    for rec in recommendations:
        max_r = rec.get("Max |r| existing", np.nan)
        max_r_str = f"{max_r:.3f}" if not np.isnan(max_r) else "N/A"
        print(f"  {rec['Feature']:<30} {max_r_str:>8} {rec['Verdict']:<12} {rec['Reason']}")

    print()

    # Print detailed notes
    print("=" * 80)
    print("DETAILED NOTES")
    print("=" * 80)
    print()

    for rec in recommendations:
        print(f"  {rec['Feature']}")
        print(f"    Verdict: {rec['Verdict']}")
        for k, v in rec.items():
            if k not in ("Feature", "Verdict", "Reason", "Column"):
                if isinstance(v, float) and not np.isnan(v):
                    print(f"    {k}: {v:.4f}" if abs(v) < 10 else f"    {k}: {v:.2f}")
                elif isinstance(v, str):
                    print(f"    {k}: {v}")
        print(f"    Rationale: {rec['Reason']}")
        print()

    # Additional: Proposed-to-proposed correlations
    print("=" * 80)
    print("BONUS: PROPOSED-TO-PROPOSED CORRELATIONS (Batter Side)")
    print("=" * 80)
    print()

    proposed_cols = [c for c in available_proposed_b.keys() if c in batter_df.columns]
    if len(proposed_cols) > 1:
        pp_corr = batter_df[proposed_cols].corr()
        print("  Checking for redundancy AMONG proposed features:")
        print()
        for i, c1 in enumerate(proposed_cols):
            for c2 in proposed_cols[i+1:]:
                r = pp_corr.loc[c1, c2]
                label1 = available_proposed_b.get(c1, c1)
                label2 = available_proposed_b.get(c2, c2)
                if abs(r) > 0.30:
                    flag = "***" if abs(r) > 0.70 else " **" if abs(r) > 0.50 else "  *"
                    print(f"  {label1} <-> {label2}: r = {r:.3f} {flag}")
        print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
