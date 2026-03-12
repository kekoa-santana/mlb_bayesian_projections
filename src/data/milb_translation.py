"""
MiLB-to-MLB translation factor derivation and application.

Derives empirical translation multipliers by comparing players who appeared
at a given MiLB level and in MLB within a ±1-season window.  Applies those
factors to raw MiLB season aggregates to produce MLB-equivalent rate estimates.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"

# Levels ordered from closest to MLB → furthest
MILB_LEVELS = ["AAA", "AA", "A+", "A"]

# sport_id → level mapping
SPORT_ID_TO_LEVEL = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "ROK"}

# Minimum PA/BF thresholds for overlap players
MIN_MILB_PA = 50
MIN_MLB_PA = 50
MIN_MILB_BF = 50
MIN_MLB_BF = 50


# ---------------------------------------------------------------------------
# Step 1: Get overlap player-seasons (MiLB level X → MLB within ±1 season)
# ---------------------------------------------------------------------------
def _get_batter_overlap_data() -> pd.DataFrame:
    """Find batters who played MiLB level X in season N and MLB in N or N+1.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, milb_season, level, milb_pa, milb_ab, milb_k,
        milb_bb, milb_hr, milb_h, milb_tb, milb_sb, mlb_season, mlb_pa,
        mlb_k, mlb_bb, mlb_hr, mlb_h, mlb_ab, mlb_tb, mlb_sb
    """
    query = """
    WITH milb_agg AS (
        SELECT
            batter_id        AS player_id,
            season            AS milb_season,
            level,
            SUM(plate_appearances) AS milb_pa,
            SUM(at_bats)           AS milb_ab,
            SUM(strikeouts)        AS milb_k,
            SUM(walks)             AS milb_bb,
            SUM(home_runs)         AS milb_hr,
            SUM(hits)              AS milb_h,
            SUM(total_bases)       AS milb_tb,
            SUM(sb)                AS milb_sb
        FROM staging.milb_batting_game_logs
        WHERE level IN ('AAA', 'AA', 'A+', 'A')
        GROUP BY batter_id, season, level
        HAVING SUM(plate_appearances) >= :min_milb_pa
    ),
    mlb_agg AS (
        SELECT
            fp.batter_id             AS player_id,
            dg.season                AS mlb_season,
            COUNT(*)                 AS mlb_pa,
            SUM(CASE WHEN fp.events = 'strikeout' THEN 1 ELSE 0 END) AS mlb_k,
            SUM(CASE WHEN fp.events = 'walk' THEN 1 ELSE 0 END)      AS mlb_bb,
            SUM(CASE WHEN fp.events = 'home_run' THEN 1 ELSE 0 END)  AS mlb_hr,
            SUM(CASE WHEN fp.events IN (
                'single','double','triple','home_run'
            ) THEN 1 ELSE 0 END) AS mlb_h,
            SUM(CASE WHEN fp.events NOT IN (
                'walk','hit_by_pitch','sac_fly','sac_bunt',
                'sac_fly_double_play','catcher_interf'
            ) THEN 1 ELSE 0 END) AS mlb_ab,
            SUM(CASE WHEN fp.events = 'single' THEN 1
                      WHEN fp.events = 'double' THEN 2
                      WHEN fp.events = 'triple' THEN 3
                      WHEN fp.events = 'home_run' THEN 4
                      ELSE 0 END) AS mlb_tb,
            SUM(CASE WHEN fp.events = 'stolen_base' THEN 1 ELSE 0 END) AS mlb_sb
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY fp.batter_id, dg.season
        HAVING COUNT(*) >= :min_mlb_pa
    )
    SELECT
        m.player_id,
        m.milb_season,
        m.level,
        m.milb_pa, m.milb_ab, m.milb_k, m.milb_bb,
        m.milb_hr, m.milb_h, m.milb_tb, m.milb_sb,
        b.mlb_season,
        b.mlb_pa, b.mlb_k, b.mlb_bb, b.mlb_hr,
        b.mlb_h, b.mlb_ab, b.mlb_tb, b.mlb_sb
    FROM milb_agg m
    JOIN mlb_agg b
      ON m.player_id = b.player_id
     AND b.mlb_season BETWEEN m.milb_season AND m.milb_season + 1
    ORDER BY m.level, m.milb_season, m.player_id
    """
    logger.info("Fetching batter MiLB↔MLB overlap data")
    return read_sql(query, {
        "min_milb_pa": MIN_MILB_PA,
        "min_mlb_pa": MIN_MLB_PA,
    })


def _get_pitcher_overlap_data() -> pd.DataFrame:
    """Find pitchers who played MiLB level X in season N and MLB in N or N+1.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, milb_season, level, milb_bf, milb_k, milb_bb,
        milb_hr, milb_ip, mlb_season, mlb_bf, mlb_k, mlb_bb, mlb_hr, mlb_ip
    """
    query = """
    WITH milb_agg AS (
        SELECT
            pitcher_id        AS player_id,
            season            AS milb_season,
            level,
            SUM(batters_faced)  AS milb_bf,
            SUM(strike_outs)    AS milb_k,
            SUM(walks)          AS milb_bb,
            SUM(home_runs)      AS milb_hr,
            SUM(innings_pitched) AS milb_ip
        FROM staging.milb_pitching_game_logs
        WHERE level IN ('AAA', 'AA', 'A+', 'A')
        GROUP BY pitcher_id, season, level
        HAVING SUM(batters_faced) >= :min_milb_bf
    ),
    mlb_agg AS (
        SELECT
            sb.pitcher_id     AS player_id,
            dg.season         AS mlb_season,
            SUM(sb.batters_faced) AS mlb_bf,
            SUM(sb.strike_outs)   AS mlb_k,
            SUM(sb.walks)         AS mlb_bb,
            SUM(sb.home_runs)     AS mlb_hr,
            SUM(sb.innings_pitched) AS mlb_ip
        FROM staging.pitching_boxscores sb
        JOIN production.dim_game dg ON sb.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY sb.pitcher_id, dg.season
        HAVING SUM(sb.batters_faced) >= :min_mlb_bf
    )
    SELECT
        m.player_id,
        m.milb_season,
        m.level,
        m.milb_bf, m.milb_k, m.milb_bb, m.milb_hr, m.milb_ip,
        b.mlb_season,
        b.mlb_bf, b.mlb_k, b.mlb_bb, b.mlb_hr, b.mlb_ip
    FROM milb_agg m
    JOIN mlb_agg b
      ON m.player_id = b.player_id
     AND b.mlb_season BETWEEN m.milb_season AND m.milb_season + 1
    ORDER BY m.level, m.milb_season, m.player_id
    """
    logger.info("Fetching pitcher MiLB↔MLB overlap data")
    return read_sql(query, {
        "min_milb_bf": MIN_MILB_BF,
        "min_mlb_bf": MIN_MLB_BF,
    })


# ---------------------------------------------------------------------------
# Step 2: Derive translation factors
# ---------------------------------------------------------------------------
def _compute_rate_factor(
    milb_values: pd.Series,
    mlb_values: pd.Series,
    *,
    clip_lo: float = 0.3,
    clip_hi: float = 3.0,
) -> dict:
    """Compute median ratio of MLB rate / MiLB rate for a single stat.

    Parameters
    ----------
    milb_values, mlb_values : pd.Series
        Rate values (e.g. K%) for each overlap player-season.
    clip_lo, clip_hi : float
        Clip individual ratios to avoid extreme outliers.

    Returns
    -------
    dict
        Keys: factor, median, mean, std, n, p25, p75
    """
    mask = (milb_values > 0) & (mlb_values > 0) & milb_values.notna() & mlb_values.notna()
    if mask.sum() < 5:
        return {"factor": np.nan, "n": int(mask.sum())}

    ratios = (mlb_values[mask] / milb_values[mask]).clip(clip_lo, clip_hi)
    return {
        "factor": float(ratios.median()),
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "n": int(len(ratios)),
        "p25": float(ratios.quantile(0.25)),
        "p75": float(ratios.quantile(0.75)),
    }


def derive_batter_translation_factors(
    overlap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive batter translation factors by level, pooled across all years.

    Parameters
    ----------
    overlap_df : pd.DataFrame, optional
        Pre-fetched overlap data. If None, fetches from DB.

    Returns
    -------
    pd.DataFrame
        One row per (level, stat). Columns: level, stat, factor, n, std, p25, p75
    """
    if overlap_df is None:
        overlap_df = _get_batter_overlap_data()

    # Compute rates
    df = overlap_df.copy()
    df["milb_k_pct"] = df["milb_k"] / df["milb_pa"]
    df["mlb_k_pct"] = df["mlb_k"] / df["mlb_pa"]
    df["milb_bb_pct"] = df["milb_bb"] / df["milb_pa"]
    df["mlb_bb_pct"] = df["mlb_bb"] / df["mlb_pa"]
    df["milb_iso"] = (df["milb_tb"] - df["milb_h"]) / df["milb_ab"].replace(0, np.nan)
    df["mlb_iso"] = (df["mlb_tb"] - df["mlb_h"]) / df["mlb_ab"].replace(0, np.nan)
    df["milb_hr_pa"] = df["milb_hr"] / df["milb_pa"]
    df["mlb_hr_pa"] = df["mlb_hr"] / df["mlb_pa"]

    stats = ["k_pct", "bb_pct", "iso", "hr_pa"]
    rows = []

    for level in MILB_LEVELS:
        ldf = df[df["level"] == level]
        if len(ldf) == 0:
            continue

        for stat in stats:
            result = _compute_rate_factor(ldf[f"milb_{stat}"], ldf[f"mlb_{stat}"])
            rows.append({"level": level, "stat": stat, **result})

        # Per-year breakdown for trend detection
        for season in sorted(ldf["milb_season"].unique()):
            sdf = ldf[ldf["milb_season"] == season]
            for stat in stats:
                result = _compute_rate_factor(sdf[f"milb_{stat}"], sdf[f"mlb_{stat}"])
                rows.append({
                    "level": level,
                    "stat": stat,
                    "season": int(season),
                    **result,
                })

    factors_df = pd.DataFrame(rows)
    # Separate pooled (no season) from per-year (has season)
    factors_df["pooled"] = factors_df.get("season", pd.Series(dtype=float)).isna()

    logger.info(
        "Derived batter translation factors: %d pooled, %d per-year",
        factors_df["pooled"].sum(),
        (~factors_df["pooled"]).sum(),
    )
    return factors_df


def derive_pitcher_translation_factors(
    overlap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive pitcher translation factors by level, pooled across all years.

    Parameters
    ----------
    overlap_df : pd.DataFrame, optional
        Pre-fetched overlap data. If None, fetches from DB.

    Returns
    -------
    pd.DataFrame
        One row per (level, stat). Columns: level, stat, factor, n, std, p25, p75
    """
    if overlap_df is None:
        overlap_df = _get_pitcher_overlap_data()

    df = overlap_df.copy()
    df["milb_k_pct"] = df["milb_k"] / df["milb_bf"]
    df["mlb_k_pct"] = df["mlb_k"] / df["mlb_bf"]
    df["milb_bb_pct"] = df["milb_bb"] / df["milb_bf"]
    df["mlb_bb_pct"] = df["mlb_bb"] / df["mlb_bf"]
    df["milb_hr_bf"] = df["milb_hr"] / df["milb_bf"]
    df["mlb_hr_bf"] = df["mlb_hr"] / df["mlb_bf"]

    stats = ["k_pct", "bb_pct", "hr_bf"]
    rows = []

    for level in MILB_LEVELS:
        ldf = df[df["level"] == level]
        if len(ldf) == 0:
            continue

        for stat in stats:
            result = _compute_rate_factor(ldf[f"milb_{stat}"], ldf[f"mlb_{stat}"])
            rows.append({"level": level, "stat": stat, **result})

        for season in sorted(ldf["milb_season"].unique()):
            sdf = ldf[ldf["milb_season"] == season]
            for stat in stats:
                result = _compute_rate_factor(sdf[f"milb_{stat}"], sdf[f"mlb_{stat}"])
                rows.append({
                    "level": level,
                    "stat": stat,
                    "season": int(season),
                    **result,
                })

    factors_df = pd.DataFrame(rows)
    factors_df["pooled"] = factors_df.get("season", pd.Series(dtype=float)).isna()

    logger.info(
        "Derived pitcher translation factors: %d pooled, %d per-year",
        factors_df["pooled"].sum(),
        (~factors_df["pooled"]).sum(),
    )
    return factors_df


# ---------------------------------------------------------------------------
# Step 3: Apply factors to raw MiLB season aggregates
# ---------------------------------------------------------------------------
def _get_pooled_factors(factors_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Extract pooled factors as {(level, stat): factor} dict."""
    pooled = factors_df[factors_df["pooled"]].copy()
    return {
        (row["level"], row["stat"]): row["factor"]
        for _, row in pooled.iterrows()
        if pd.notna(row["factor"])
    }


def _interpolate_factor(
    factors: dict[tuple[str, str], float],
    level: str,
    stat: str,
) -> float:
    """Get factor for level+stat, interpolating if missing.

    Falls back up the chain: A → A+ → AA → AAA.
    If a lower level is missing, use the next available level's factor
    with a 10% additional penalty per level gap.
    """
    if (level, stat) in factors:
        return factors[(level, stat)]

    # Fallback chain
    level_order = ["A", "A+", "AA", "AAA"]
    if level not in level_order:
        return np.nan

    idx = level_order.index(level)
    # Try levels above
    for step, upper_idx in enumerate(range(idx + 1, len(level_order)), 1):
        upper_level = level_order[upper_idx]
        if (upper_level, stat) in factors:
            base = factors[(upper_level, stat)]
            # Move further from 1.0 by 10% per level gap
            deviation = base - 1.0
            return 1.0 + deviation * (1.0 + 0.10 * step)

    return np.nan


def translate_batter_season(
    raw_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply translation factors to raw MiLB batter season aggregates.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw MiLB batter season totals with columns: player_id, season, level,
        pa, ab, k, bb, hr, h, tb, sb
    factors_df : pd.DataFrame
        Output of derive_batter_translation_factors().

    Returns
    -------
    pd.DataFrame
        Original columns plus translated rates: translated_k_pct, translated_bb_pct,
        translated_iso, translated_hr_pa, translation_confidence
    """
    factors = _get_pooled_factors(factors_df)
    df = raw_df.copy()

    # Raw rates
    df["raw_k_pct"] = df["k"] / df["pa"]
    df["raw_bb_pct"] = df["bb"] / df["pa"]
    df["raw_iso"] = (df["tb"] - df["h"]) / df["ab"].replace(0, np.nan)
    df["raw_hr_pa"] = df["hr"] / df["pa"]

    # Apply factors
    stat_map = {
        "k_pct": "translated_k_pct",
        "bb_pct": "translated_bb_pct",
        "iso": "translated_iso",
        "hr_pa": "translated_hr_pa",
    }

    for stat, col in stat_map.items():
        df[col] = df.apply(
            lambda row: row[f"raw_{stat}"] * _interpolate_factor(factors, row["level"], stat),
            axis=1,
        )

    # Translation confidence: based on PA and level reliability
    level_reliability = {"AAA": 0.90, "AA": 0.70, "A+": 0.50, "A": 0.35}
    df["level_reliability"] = df["level"].map(level_reliability).fillna(0.2)
    # PA reliability: ramps from 0 at 0 PA to 1.0 at 200 PA
    df["pa_reliability"] = (df["pa"] / 200.0).clip(0.0, 1.0)
    df["translation_confidence"] = df["level_reliability"] * df["pa_reliability"]

    return df


def translate_pitcher_season(
    raw_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply translation factors to raw MiLB pitcher season aggregates.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw MiLB pitcher season totals with columns: player_id, season, level,
        bf, k, bb, hr, ip
    factors_df : pd.DataFrame
        Output of derive_pitcher_translation_factors().

    Returns
    -------
    pd.DataFrame
        Original columns plus translated rates and translation_confidence.
    """
    factors = _get_pooled_factors(factors_df)
    df = raw_df.copy()

    df["raw_k_pct"] = df["k"] / df["bf"]
    df["raw_bb_pct"] = df["bb"] / df["bf"]
    df["raw_hr_bf"] = df["hr"] / df["bf"]

    stat_map = {
        "k_pct": "translated_k_pct",
        "bb_pct": "translated_bb_pct",
        "hr_bf": "translated_hr_bf",
    }

    for stat, col in stat_map.items():
        df[col] = df.apply(
            lambda row: row[f"raw_{stat}"] * _interpolate_factor(factors, row["level"], stat),
            axis=1,
        )

    level_reliability = {"AAA": 0.90, "AA": 0.70, "A+": 0.50, "A": 0.35}
    df["level_reliability"] = df["level"].map(level_reliability).fillna(0.2)
    df["pa_reliability"] = (df["bf"] / 200.0).clip(0.0, 1.0)
    df["translation_confidence"] = df["level_reliability"] * df["pa_reliability"]

    return df


# ---------------------------------------------------------------------------
# Step 4: Age-relative-to-level context
# ---------------------------------------------------------------------------
# Average age at each level (approximate MLB norms)
LEVEL_AVG_AGE = {"A": 20.5, "A+": 21.5, "AA": 23.0, "AAA": 25.5, "ROK": 19.0}


def add_age_context(
    translated_df: pd.DataFrame,
    prospect_info: pd.DataFrame,
) -> pd.DataFrame:
    """Add age_at_level and age_relative_to_level_avg columns.

    Parameters
    ----------
    translated_df : pd.DataFrame
        Translated season aggregates with player_id, season, level.
    prospect_info : pd.DataFrame
        From get_prospect_info() — must have player_id, birth_date.

    Returns
    -------
    pd.DataFrame
        Input plus age_at_level, age_relative_to_level_avg columns.
    """
    df = translated_df.merge(
        prospect_info[["player_id", "birth_date"]].drop_duplicates("player_id"),
        on="player_id",
        how="left",
    )

    # Age at midseason (July 1) of the MiLB season
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    midseason = pd.to_datetime(df["season"].astype(str) + "-07-01")
    df["age_at_level"] = (midseason - df["birth_date"]).dt.days / 365.25
    df["age_at_level"] = df["age_at_level"].round(1)

    df["level_avg_age"] = df["level"].map(LEVEL_AVG_AGE)
    df["age_relative_to_level_avg"] = df["age_at_level"] - df["level_avg_age"]

    df.drop(columns=["birth_date", "level_avg_age"], inplace=True)
    return df
