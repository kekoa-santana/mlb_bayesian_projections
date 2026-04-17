"""
MiLB-to-MLB translation factor derivation and application.

Derives empirical translation multipliers by comparing players who appeared
at a given MiLB level and in MLB within a ±1-season window.  Applies those
factors to raw MiLB season aggregates to produce MLB-equivalent rate estimates.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.db import read_sql

from src.data.paths import CACHE_DIR

logger = logging.getLogger(__name__)


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

    Uses ``staging.batting_boxscores`` for the MLB side (data back to 2000)
    instead of ``production.fact_pa`` (2018+ only), giving ~2.7x more
    overlap player-seasons when MiLB data starts from 2005.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, milb_season, level, milb_pa, milb_ab, milb_k,
        milb_bb, milb_hr, milb_h, milb_tb, milb_sb, milb_ground_outs,
        milb_air_outs, mlb_season, mlb_pa, mlb_k, mlb_bb, mlb_hr, mlb_h,
        mlb_ab, mlb_tb, mlb_sb, mlb_ground_outs, mlb_air_outs
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
            SUM(sb)                AS milb_sb,
            SUM(ground_outs)       AS milb_ground_outs,
            SUM(air_outs)          AS milb_air_outs
        FROM staging.milb_batting_game_logs
        WHERE level IN ('AAA', 'AA', 'A+', 'A')
        GROUP BY batter_id, season, level
        HAVING SUM(plate_appearances) >= :min_milb_pa
    ),
    mlb_agg AS (
        SELECT
            bb.batter_id            AS player_id,
            dg.season                AS mlb_season,
            SUM(bb.plate_appearances) AS mlb_pa,
            SUM(bb.strikeouts)       AS mlb_k,
            SUM(bb.walks)            AS mlb_bb,
            SUM(bb.home_runs)        AS mlb_hr,
            SUM(bb.hits)             AS mlb_h,
            SUM(bb.at_bats)          AS mlb_ab,
            SUM(bb.total_bases)      AS mlb_tb,
            SUM(bb.sb)               AS mlb_sb,
            SUM(bb.ground_outs)      AS mlb_ground_outs,
            SUM(bb.air_outs)         AS mlb_air_outs
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY bb.batter_id, dg.season
        HAVING SUM(bb.plate_appearances) >= :min_mlb_pa
    )
    SELECT
        m.player_id,
        m.milb_season,
        m.level,
        m.milb_pa, m.milb_ab, m.milb_k, m.milb_bb,
        m.milb_hr, m.milb_h, m.milb_tb, m.milb_sb,
        m.milb_ground_outs, m.milb_air_outs,
        b.mlb_season,
        b.mlb_pa, b.mlb_k, b.mlb_bb, b.mlb_hr,
        b.mlb_h, b.mlb_ab, b.mlb_tb, b.mlb_sb,
        b.mlb_ground_outs, b.mlb_air_outs
    FROM milb_agg m
    JOIN mlb_agg b
      ON m.player_id = b.player_id
     AND b.mlb_season BETWEEN m.milb_season AND m.milb_season + 1
    ORDER BY m.level, m.milb_season, m.player_id
    """
    logger.info("Fetching batter MiLB↔MLB overlap data (2005+)")
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
_TIME_WEIGHT_HALF_LIFE: float = 5.0  # years; recent data weighted ~2x vs 5yr-old data


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


def _compute_time_weighted_factor(
    milb_values: pd.Series,
    mlb_values: pd.Series,
    seasons: pd.Series,
    *,
    reference_season: int | None = None,
    half_life: float = _TIME_WEIGHT_HALF_LIFE,
    clip_lo: float = 0.3,
    clip_hi: float = 3.0,
) -> dict:
    """Compute time-weighted median ratio of MLB rate / MiLB rate.

    More recent overlap seasons contribute more via exponential decay.
    EDA showed BB% and ISO translation ratios shifted significantly
    post-2021 (e.g. AAA BB% ratio: 0.81 → 0.69), so flat pooling
    across 20 years misweights stale data.

    Parameters
    ----------
    milb_values, mlb_values : pd.Series
        Rate values for each overlap player-season.
    seasons : pd.Series
        MiLB season for each overlap row (used for time weighting).
    reference_season : int or None
        Anchor year for decay.  Defaults to ``seasons.max()``.
    half_life : float
        Half-life in years for exponential decay (default 5).
    clip_lo, clip_hi : float
        Clip individual ratios to avoid extreme outliers.

    Returns
    -------
    dict
        Keys: factor, n, std, p25, p75 (factor is weighted median).
    """
    mask = (
        (milb_values > 0)
        & (mlb_values > 0)
        & milb_values.notna()
        & mlb_values.notna()
        & seasons.notna()
    )
    if mask.sum() < 5:
        return {"factor": np.nan, "n": int(mask.sum())}

    ratios = (mlb_values[mask] / milb_values[mask]).clip(clip_lo, clip_hi)
    s = seasons[mask].astype(float)
    if reference_season is None:
        reference_season = int(s.max())
    decay = np.exp(-np.log(2) * (reference_season - s) / half_life)

    # Weighted median via sorted cumulative weights
    order = ratios.argsort()
    sorted_ratios = ratios.iloc[order].values
    sorted_weights = decay.iloc[order].values
    cum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cum, cum[-1] / 2.0)
    w_median = float(sorted_ratios[min(median_idx, len(sorted_ratios) - 1)])

    # Weighted std (for diagnostics)
    w_total = decay.sum()
    w_mean = float((ratios * decay).sum() / w_total)
    w_var = float(((ratios - w_mean) ** 2 * decay).sum() / w_total)

    return {
        "factor": w_median,
        "mean": w_mean,
        "std": float(np.sqrt(w_var)),
        "n": int(mask.sum()),
        "p25": float(ratios.quantile(0.25)),
        "p75": float(ratios.quantile(0.75)),
    }


def _add_batter_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MiLB and MLB rate columns on overlap DataFrame."""
    df["milb_k_pct"] = df["milb_k"] / df["milb_pa"]
    df["mlb_k_pct"] = df["mlb_k"] / df["mlb_pa"]
    df["milb_bb_pct"] = df["milb_bb"] / df["milb_pa"]
    df["mlb_bb_pct"] = df["mlb_bb"] / df["mlb_pa"]
    df["milb_iso"] = (df["milb_tb"] - df["milb_h"]) / df["milb_ab"].replace(0, np.nan)
    df["mlb_iso"] = (df["mlb_tb"] - df["mlb_h"]) / df["mlb_ab"].replace(0, np.nan)
    df["milb_hr_pa"] = df["milb_hr"] / df["milb_pa"]
    df["mlb_hr_pa"] = df["mlb_hr"] / df["mlb_pa"]

    milb_bip = df["milb_ground_outs"] + df["milb_air_outs"]
    mlb_bip = df["mlb_ground_outs"] + df["mlb_air_outs"]
    df["milb_gb_pct"] = df["milb_ground_outs"] / milb_bip.replace(0, np.nan)
    df["mlb_gb_pct"] = df["mlb_ground_outs"] / mlb_bip.replace(0, np.nan)
    return df


def derive_batter_translation_factors(
    overlap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive batter translation factors by level.

    Produces both **time-weighted pooled** factors (used for application)
    and **per-year** factors (for trend diagnostics).  Time-weighting uses
    exponential decay with a 5-year half-life so that recent overlap
    seasons contribute more.  EDA showed BB% and ISO ratios shifted
    significantly post-2021 while K% remained stable.

    Parameters
    ----------
    overlap_df : pd.DataFrame, optional
        Pre-fetched overlap data. If None, fetches from DB.

    Returns
    -------
    pd.DataFrame
        Rows with ``pooled=True`` are the time-weighted factors to use.
        Per-year rows (``pooled=False``) included for diagnostics.
    """
    if overlap_df is None:
        overlap_df = _get_batter_overlap_data()

    df = _add_batter_rates(overlap_df.copy())

    stats = ["k_pct", "bb_pct", "iso", "hr_pa", "gb_pct"]
    rows = []

    for level in MILB_LEVELS:
        ldf = df[df["level"] == level]
        if len(ldf) == 0:
            continue

        # Time-weighted pooled factor (primary)
        for stat in stats:
            result = _compute_time_weighted_factor(
                ldf[f"milb_{stat}"],
                ldf[f"mlb_{stat}"],
                ldf["milb_season"],
            )
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
    factors_df["pooled"] = factors_df.get("season", pd.Series(dtype=float)).isna()

    logger.info(
        "Derived batter translation factors: %d pooled (time-weighted), %d per-year",
        factors_df["pooled"].sum(),
        (~factors_df["pooled"]).sum(),
    )
    return factors_df


def derive_pitcher_translation_factors(
    overlap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive pitcher translation factors by level (time-weighted).

    Parameters
    ----------
    overlap_df : pd.DataFrame, optional
        Pre-fetched overlap data. If None, fetches from DB.

    Returns
    -------
    pd.DataFrame
        Rows with ``pooled=True`` are the time-weighted factors to use.
        Per-year rows (``pooled=False``) included for diagnostics.
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

        # Time-weighted pooled factor (primary)
        for stat in stats:
            result = _compute_time_weighted_factor(
                ldf[f"milb_{stat}"],
                ldf[f"mlb_{stat}"],
                ldf["milb_season"],
            )
            rows.append({"level": level, "stat": stat, **result})

        # Per-year for diagnostics
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
        "Derived pitcher translation factors: %d pooled (time-weighted), %d per-year",
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


def _get_pooled_factor_n(factors_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Extract overlap sample sizes as {(level, stat): n} dict."""
    pooled = factors_df[factors_df["pooled"]].copy()
    return {
        (row["level"], row["stat"]): int(row.get("n", 0))
        for _, row in pooled.iterrows()
        if pd.notna(row.get("n", 0))
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
    # Build overlap-N lookup for factor reliability
    factor_n = _get_pooled_factor_n(factors_df)
    df = raw_df.copy()

    # Raw rates
    df["raw_k_pct"] = df["k"] / df["pa"]
    df["raw_bb_pct"] = df["bb"] / df["pa"]
    df["raw_iso"] = (df["tb"] - df["h"]) / df["ab"].replace(0, np.nan)
    df["raw_hr_pa"] = df["hr"] / df["pa"]

    # GB rate from ground_outs/air_outs (if available)
    if "ground_outs" in df.columns and "air_outs" in df.columns:
        bip_total = df["ground_outs"] + df["air_outs"]
        df["raw_gb_pct"] = df["ground_outs"] / bip_total.replace(0, np.nan)
    else:
        df["raw_gb_pct"] = np.nan

    # Apply factors
    stat_map = {
        "k_pct": "translated_k_pct",
        "bb_pct": "translated_bb_pct",
        "iso": "translated_iso",
        "hr_pa": "translated_hr_pa",
        "gb_pct": "translated_gb_pct",
    }

    for stat, col in stat_map.items():
        df[col] = df.apply(
            lambda row: row.get(f"raw_{stat}", np.nan) * _interpolate_factor(factors, row["level"], stat)
            if pd.notna(row.get(f"raw_{stat}", np.nan)) else np.nan,
            axis=1,
        )

    # Translation confidence: level × PA × factor reliability
    level_reliability = {"AAA": 0.90, "AA": 0.70, "A+": 0.50, "A": 0.35}
    df["level_reliability"] = df["level"].map(level_reliability).fillna(0.2)
    df["pa_reliability"] = (df["pa"] / 200.0).clip(0.0, 1.0)

    # Factor reliability: penalize levels with few overlap seasons
    # Uses K% factor N as proxy (most important stat)
    df["factor_reliability"] = df["level"].map(
        lambda lvl: min(factor_n.get((lvl, "k_pct"), 10) / 100.0, 1.0)
    )
    df["translation_confidence"] = (
        df["level_reliability"] * df["pa_reliability"] * df["factor_reliability"]
    )

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
    factor_n = _get_pooled_factor_n(factors_df)
    df = raw_df.copy()

    df["raw_k_pct"] = df["k"] / df["bf"]
    df["raw_bb_pct"] = df["bb"] / df["bf"]
    df["raw_hr_bf"] = df["hr"] / df["bf"]

    # GB rate from ground_outs/air_outs (if available)
    if "ground_outs" in df.columns and "air_outs" in df.columns:
        bip_total = df["ground_outs"] + df["air_outs"]
        df["raw_gb_pct"] = df["ground_outs"] / bip_total.replace(0, np.nan)
    else:
        df["raw_gb_pct"] = np.nan

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

    # GB rate: pass through raw (no translation needed — GB% is level-stable)
    df["translated_gb_pct"] = df["raw_gb_pct"]

    level_reliability = {"AAA": 0.90, "AA": 0.70, "A+": 0.50, "A": 0.35}
    df["level_reliability"] = df["level"].map(level_reliability).fillna(0.2)
    df["pa_reliability"] = (df["bf"] / 200.0).clip(0.0, 1.0)

    # Factor reliability from overlap sample size
    df["factor_reliability"] = df["level"].map(
        lambda lvl: min(factor_n.get((lvl, "k_pct"), 10) / 100.0, 1.0)
    )
    df["translation_confidence"] = (
        df["level_reliability"] * df["pa_reliability"] * df["factor_reliability"]
    )

    return df


# ---------------------------------------------------------------------------
# Step 4: Age-relative-to-level context
# ---------------------------------------------------------------------------
# Typical prospect age at each level — P25 of the empirical age
# distribution (61K player-level-seasons, 2005-2025).  P25 is used
# instead of the median because the median is inflated by veterans and
# AAAA players who are not the prospect population of interest.
# Empirical medians for reference: ROK 19.2, A 22.1, A+ 23.2, AA 24.6, AAA 27.3
LEVEL_AVG_AGE = {"ROK": 18.2, "A": 20.8, "A+": 22.2, "AA": 23.4, "AAA": 25.2}


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


# ---------------------------------------------------------------------------
# Step 5: FV-conditioned translation adjustments
# ---------------------------------------------------------------------------
# Population-level translation factors assume all prospects translate the same
# way.  Empirical analysis shows that FanGraphs Future Value (FV) grade
# predicts differential translation for certain stat/level combos:
#
# Batters:
#   - ISO at AAA: FV 60+ retains 83% of power vs 72% for FV 50-55 (+14%)
#   - ISO at AA:  FV 60+ retains 81% vs 77% pooled (+5%)
#   - K% at AA:   FV 60+ K ratio 1.207 vs 1.258 pooled (-4%, elite K less)
#
# Pitchers:
#   - K% at AA:  FV 60+ retains 82% of K% vs 78% pooled (+6%)
#   - K% at A+:  FV 60+ retains 83% vs 73% pooled (+7%, small n=12)
#   - K% at AAA: NOT conditioned — selection bias reverses signal
#
# Multipliers are applied on top of the pooled factor:
#   adjusted_rate = translated_rate * fv_multiplier
#
# Stats/levels without a clear signal keep multiplier = 1.0 (no adjustment).

_FV_PITCHER_K_MULT: dict[str, dict[str, float]] = {
    # level: {fv_tier: multiplier vs pooled}
    "AA":  {"high": 1.06, "mid": 0.96, "low": 1.00},
    "A+":  {"high": 1.07, "mid": 0.96, "low": 1.00},
    # AAA omitted: selection bias reverses the signal (n=30, ratio 0.741)
    # A omitted: insufficient FV 60+ sample
}

_FV_BATTER_ISO_MULT: dict[str, dict[str, float]] = {
    "AAA": {"high": 1.14, "mid": 0.99, "low": 1.00},
    "AA":  {"high": 1.05, "mid": 1.02, "low": 1.00},
    "A+":  {"high": 1.04, "mid": 0.89, "low": 1.00},
}

_FV_BATTER_K_MULT: dict[str, dict[str, float]] = {
    "AA":  {"high": 0.96, "mid": 0.99, "low": 1.00},
    "AAA": {"high": 0.99, "mid": 1.02, "low": 1.00},
}


def _fv_to_tier(fv: float | int | None) -> str:
    """Map FV grade to tier for multiplier lookup.

    high: FV >= 60  (elite prospects)
    mid:  FV 50–55  (solid prospects)
    low:  FV <= 45 or unknown  (population baseline, multiplier = 1.0)
    """
    if fv is None or (isinstance(fv, float) and np.isnan(fv)):
        return "low"
    if fv >= 60:
        return "high"
    if fv >= 50:
        return "mid"
    return "low"


def apply_fv_adjustments(
    df: pd.DataFrame,
    fv_map: dict[int, float],
    player_type: str = "pitcher",
) -> pd.DataFrame:
    """Apply FV-conditioned multipliers to translated MiLB rates.

    Adjusts the ``translated_*`` columns in-place based on each prospect's
    FanGraphs Future Value grade.  Only stat/level combos with empirically
    validated signal get adjusted; all others keep multiplier = 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        Translated MiLB data (per-level rows).  Must have ``player_id``,
        ``level``, and the relevant ``translated_*`` columns.
    fv_map : dict[int, float]
        ``{player_id: FV_grade}`` lookup.
    player_type : {'pitcher', 'batter'}
        Selects which multiplier tables to apply.

    Returns
    -------
    pd.DataFrame
        Same shape, translated rates adjusted where applicable.
    """
    df = df.copy()
    df["_fv_tier"] = df["player_id"].map(fv_map).apply(_fv_to_tier)

    n_adjusted = 0

    if player_type == "pitcher":
        col = "translated_k_pct"
        for level, mults in _FV_PITCHER_K_MULT.items():
            for tier, mult in mults.items():
                if mult == 1.0:
                    continue
                mask = (df["level"] == level) & (df["_fv_tier"] == tier)
                count = mask.sum()
                if count > 0:
                    df.loc[mask, col] *= mult
                    n_adjusted += count

    elif player_type == "batter":
        # ISO adjustment
        if "translated_iso" in df.columns:
            for level, mults in _FV_BATTER_ISO_MULT.items():
                for tier, mult in mults.items():
                    if mult == 1.0:
                        continue
                    mask = (df["level"] == level) & (df["_fv_tier"] == tier)
                    count = mask.sum()
                    if count > 0:
                        df.loc[mask, "translated_iso"] *= mult
                        n_adjusted += count

        # K% adjustment
        for level, mults in _FV_BATTER_K_MULT.items():
            for tier, mult in mults.items():
                if mult == 1.0:
                    continue
                mask = (df["level"] == level) & (df["_fv_tier"] == tier)
                count = mask.sum()
                if count > 0:
                    df.loc[mask, "translated_k_pct"] *= mult
                    n_adjusted += count

        # Update hr_pa from ISO if available (ISO drives HR translation)
        if "translated_iso" in df.columns and "translated_hr_pa" in df.columns:
            # HR/PA scales roughly proportionally with ISO adjustments
            for level, mults in _FV_BATTER_ISO_MULT.items():
                for tier, mult in mults.items():
                    if mult == 1.0:
                        continue
                    mask = (df["level"] == level) & (df["_fv_tier"] == tier)
                    if mask.any():
                        df.loc[mask, "translated_hr_pa"] *= mult

    if n_adjusted > 0:
        tier_counts = df["_fv_tier"].value_counts()
        logger.info(
            "Applied FV adjustments to %d %s rows "
            "(high=%d, mid=%d, low=%d [unchanged])",
            n_adjusted,
            player_type,
            tier_counts.get("high", 0),
            tier_counts.get("mid", 0),
            tier_counts.get("low", 0),
        )

    df.drop(columns=["_fv_tier"], inplace=True)
    return df
