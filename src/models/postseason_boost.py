"""
Player-specific postseason performance boost for rankings.

Computes individual performance scores from postseason stats with
heavy sample-size dampening to prevent noise from small samples.
A mediocre reliever on a World Series team gets nearly zero boost;
a dominant starter gets a meaningful adjustment.

The scores are applied as additive adjustments to offense_score
(hitters) and stuff_score (pitchers) in the ranking pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Round multiplier: deeper runs amplify the deviation from neutral
_ROUND_MULTIPLIER = {
    "F": 1.0,   # Wild Card
    "D": 1.1,   # Division Series
    "L": 1.2,   # LCS
    "W": 1.3,   # World Series
}

# Recency weights for multi-season aggregation (most recent = highest)
_RECENCY_WEIGHTS = {0: 3, 1: 2, 2: 1}  # offset from latest season


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank values on 0-1 scale (higher = better)."""
    return series.rank(pct=True, method="average")


def _shrinkage_hitter(ps_pa: pd.Series) -> pd.Series:
    """Sample-size shrinkage for hitter postseason scores.

    Parameters
    ----------
    ps_pa : pd.Series
        Total postseason plate appearances.

    Returns
    -------
    pd.Series
        Shrinkage factor (0.0 = full signal, 0.85 = barely moves).
    """
    return pd.Series(
        np.where(
            ps_pa >= 50, 0.0,
            np.where(
                ps_pa >= 30, 0.3,
                np.where(ps_pa >= 15, 0.6, 0.85),
            ),
        ),
        index=ps_pa.index,
    )


def _shrinkage_pitcher_starter(ps_bf: pd.Series) -> pd.Series:
    """Sample-size shrinkage for starter postseason scores.

    Parameters
    ----------
    ps_bf : pd.Series
        Total postseason batters faced.

    Returns
    -------
    pd.Series
        Shrinkage factor.
    """
    return pd.Series(
        np.where(
            ps_bf >= 60, 0.0,
            np.where(
                ps_bf >= 30, 0.3,
                np.where(ps_bf >= 15, 0.5, 0.85),
            ),
        ),
        index=ps_bf.index,
    )


def _shrinkage_pitcher_reliever(ps_bf: pd.Series) -> pd.Series:
    """Sample-size shrinkage for reliever postseason scores.

    Relievers need fewer BF for signal since their role is shorter.

    Parameters
    ----------
    ps_bf : pd.Series
        Total postseason batters faced.

    Returns
    -------
    pd.Series
        Shrinkage factor.
    """
    return pd.Series(
        np.where(
            ps_bf >= 25, 0.0,
            np.where(
                ps_bf >= 15, 0.3,
                np.where(ps_bf >= 8, 0.6, 0.90),
            ),
        ),
        index=ps_bf.index,
    )


def _apply_dampening(
    raw_score: pd.Series,
    shrinkage: pd.Series,
    best_round: pd.Series,
) -> pd.Series:
    """Apply sample-size dampening and round bonus to raw scores.

    Dampened toward 0.50 (neutral), then deviation from 0.50 is
    multiplied by the round factor for the player's best round.

    Parameters
    ----------
    raw_score : pd.Series
        Raw percentile-based score (0-1).
    shrinkage : pd.Series
        Shrinkage factor per player (0 = keep full, 1 = shrink to 0.5).
    best_round : pd.Series
        Best postseason round reached ('F', 'D', 'L', 'W').

    Returns
    -------
    pd.Series
        Dampened and round-adjusted score (0-1).
    """
    # Dampen toward neutral (0.50)
    dampened = 0.50 + (raw_score - 0.50) * (1 - shrinkage)

    # Round bonus: amplify deviation from neutral by round multiplier
    round_mult = best_round.map(_ROUND_MULTIPLIER).fillna(1.0)
    deviation = dampened - 0.50
    adjusted = 0.50 + deviation * round_mult

    return adjusted.clip(0, 1)


def _aggregate_multi_season(
    df: pd.DataFrame,
    id_col: str,
    count_col: str,
    target_season: int,
) -> pd.DataFrame:
    """Recency-weight multi-season postseason stats.

    Aggregates across seasons with 3/2/1 weighting (most recent first).
    This prevents a single hot postseason from dominating while still
    rewarding sustained October performance.

    Parameters
    ----------
    df : pd.DataFrame
        Per-player-per-season postseason stats.
    id_col : str
        Player ID column ('batter_id' or 'pitcher_id').
    count_col : str
        Counting column for weighting ('ps_pa' or 'ps_bf').
    target_season : int
        The most recent season to weight from.

    Returns
    -------
    pd.DataFrame
        Single row per player with recency-weighted stats.
    """
    if df.empty:
        return df

    df = df.copy()
    df["season_offset"] = target_season - df["season"]
    # Only keep seasons within recency window (0, 1, 2 = last 3 years)
    df = df[df["season_offset"].between(0, 2)]
    if df.empty:
        return df

    df["recency_wt"] = df["season_offset"].map(_RECENCY_WEIGHTS).fillna(0)
    df["total_wt"] = df["recency_wt"] * df[count_col]

    # Keep best_round from most recent season
    best_rounds = (
        df.sort_values("season", ascending=False)
        .drop_duplicates(id_col, keep="first")[[id_col, "best_round"]]
    )

    # Total exposure (unweighted — for shrinkage calculation)
    total_exposure = df.groupby(id_col)[count_col].sum().reset_index(
        name=f"total_{count_col}"
    )

    return df, best_rounds, total_exposure


def compute_postseason_scores(
    batter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute player-specific postseason performance scores.

    Scores are on a 0-1 scale centered at 0.50 (neutral). Players
    with no postseason data are not included in the output; callers
    should default missing players to 0.50.

    Parameters
    ----------
    batter_stats : pd.DataFrame
        Output of ``get_postseason_batter_stats()`` with columns:
        batter_id, season, ps_pa, ps_k, ps_bb, ps_hr, ps_h, ps_tb,
        ps_k_rate, ps_bb_rate, ps_hr_rate, ps_iso, best_round.
    pitcher_stats : pd.DataFrame
        Output of ``get_postseason_pitcher_stats()`` with columns:
        pitcher_id, season, ps_bf, ps_k, ps_bb, ps_hr, ps_h,
        ps_k_rate, ps_bb_rate, ps_hr_rate, ps_ip, is_starter, best_round.
    season : int
        Most recent completed season (recency weighting anchors here).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (hitter_scores, pitcher_scores) with columns:
        - player_id (batter_id or pitcher_id)
        - postseason_raw_score
        - postseason_score (after dampening + round bonus)
        - postseason_pa or postseason_bf (total exposure)
        - best_round
    """
    hitter_scores = _compute_hitter_postseason(batter_stats, season)
    pitcher_scores = _compute_pitcher_postseason(pitcher_stats, season)
    return hitter_scores, pitcher_scores


def _compute_hitter_postseason(
    batter_stats: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute hitter postseason scores with dampening.

    Raw score: 40% (1 - K rate pctl) + 30% ISO pctl + 30% BB rate pctl
    within the postseason population.

    Parameters
    ----------
    batter_stats : pd.DataFrame
        Per-season postseason batting stats.
    season : int
        Most recent completed season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, postseason_raw_score, postseason_score,
        postseason_pa, best_round.
    """
    if batter_stats.empty:
        logger.info("No postseason batter stats — returning empty")
        return pd.DataFrame(
            columns=["batter_id", "postseason_raw_score",
                      "postseason_score", "postseason_pa", "best_round"]
        )

    df = batter_stats.copy()

    # Filter to recent seasons (last 3 years)
    df = df[df["season"].between(season - 2, season)]
    if df.empty:
        return pd.DataFrame(
            columns=["batter_id", "postseason_raw_score",
                      "postseason_score", "postseason_pa", "best_round"]
        )

    # Recency weighting
    df["season_offset"] = season - df["season"]
    df["recency_wt"] = df["season_offset"].map(_RECENCY_WEIGHTS).fillna(0)
    df["total_wt"] = df["recency_wt"] * df["ps_pa"]

    # Best round from most recent season
    best_rounds = (
        df.sort_values("season", ascending=False)
        .drop_duplicates("batter_id", keep="first")[["batter_id", "best_round"]]
    )

    # Total PA (unweighted — for shrinkage)
    total_pa = (
        df.groupby("batter_id")["ps_pa"].sum()
        .reset_index(name="postseason_pa")
    )

    # Weighted-average rates across seasons
    def _wtd_rate(group: pd.DataFrame, num_col: str, denom_col: str) -> float:
        valid = group[group[denom_col] > 0]
        if valid.empty or valid["total_wt"].sum() == 0:
            return np.nan
        num = (valid[num_col] * valid["recency_wt"]).sum()
        denom = (valid[denom_col] * valid["recency_wt"]).sum()
        return num / denom if denom > 0 else np.nan

    agg = df.groupby("batter_id").apply(
        lambda g: pd.Series({
            "wtd_k_rate": _wtd_rate(g, "ps_k", "ps_pa"),
            "wtd_bb_rate": _wtd_rate(g, "ps_bb", "ps_pa"),
            "wtd_iso": _wtd_rate(g, "ps_tb", "ps_pa") - _wtd_rate(g, "ps_h", "ps_pa")
            if _wtd_rate(g, "ps_pa", "ps_pa") else np.nan,
        }),
        include_groups=False,
    ).reset_index()

    # Fix ISO: (TB - H) / (PA - BB) approximation, but use simple TB/PA - H/PA
    # for simplicity since it's being percentile-ranked anyway
    agg = agg.merge(total_pa, on="batter_id")
    agg = agg.merge(best_rounds, on="batter_id")

    # Need minimum of 5 players to compute meaningful percentiles
    if len(agg) < 5:
        logger.warning("Only %d batters with postseason data — too few for pctl", len(agg))
        agg["postseason_raw_score"] = 0.50
        agg["postseason_score"] = 0.50
        return agg[["batter_id", "postseason_raw_score", "postseason_score",
                     "postseason_pa", "best_round"]]

    # Percentile ranks within postseason population
    k_rate_pctl = 1.0 - _percentile_rank(agg["wtd_k_rate"].fillna(agg["wtd_k_rate"].median()))
    iso_pctl = _percentile_rank(agg["wtd_iso"].fillna(agg["wtd_iso"].median()))
    bb_rate_pctl = _percentile_rank(agg["wtd_bb_rate"].fillna(agg["wtd_bb_rate"].median()))

    # Raw score
    agg["postseason_raw_score"] = (
        0.40 * k_rate_pctl + 0.30 * iso_pctl + 0.30 * bb_rate_pctl
    )

    # Dampening
    shrinkage = _shrinkage_hitter(agg["postseason_pa"])
    agg["postseason_score"] = _apply_dampening(
        agg["postseason_raw_score"], shrinkage, agg["best_round"],
    )

    n_boosted = (agg["postseason_score"] > 0.52).sum()
    n_penalized = (agg["postseason_score"] < 0.48).sum()
    n_neutral = len(agg) - n_boosted - n_penalized
    logger.info(
        "Hitter postseason scores: %d players — %d boosted, %d penalized, %d neutral",
        len(agg), n_boosted, n_penalized, n_neutral,
    )

    return agg[["batter_id", "postseason_raw_score", "postseason_score",
                "postseason_pa", "best_round"]]


def _compute_pitcher_postseason(
    pitcher_stats: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute pitcher postseason scores with dampening.

    Raw score: 50% K rate pctl + 30% (1 - BB rate pctl) + 20% (1 - HR rate pctl)
    within the postseason population.

    Uses separate shrinkage curves for starters vs relievers.

    Parameters
    ----------
    pitcher_stats : pd.DataFrame
        Per-season postseason pitching stats.
    season : int
        Most recent completed season.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, postseason_raw_score, postseason_score,
        postseason_bf, best_round.
    """
    if pitcher_stats.empty:
        logger.info("No postseason pitcher stats — returning empty")
        return pd.DataFrame(
            columns=["pitcher_id", "postseason_raw_score",
                      "postseason_score", "postseason_bf", "best_round"]
        )

    df = pitcher_stats.copy()

    # Filter to recent seasons (last 3 years)
    df = df[df["season"].between(season - 2, season)]
    if df.empty:
        return pd.DataFrame(
            columns=["pitcher_id", "postseason_raw_score",
                      "postseason_score", "postseason_bf", "best_round"]
        )

    # Recency weighting
    df["season_offset"] = season - df["season"]
    df["recency_wt"] = df["season_offset"].map(_RECENCY_WEIGHTS).fillna(0)
    df["total_wt"] = df["recency_wt"] * df["ps_bf"]

    # Best round from most recent season
    best_rounds = (
        df.sort_values("season", ascending=False)
        .drop_duplicates("pitcher_id", keep="first")[["pitcher_id", "best_round"]]
    )

    # Total BF (unweighted — for shrinkage)
    total_bf = (
        df.groupby("pitcher_id")["ps_bf"].sum()
        .reset_index(name="postseason_bf")
    )

    # Determine if starter (majority of appearances across seasons)
    starter_flag = (
        df.groupby("pitcher_id")["is_starter"]
        .apply(lambda g: g.sum() > len(g) / 2)
        .reset_index(name="is_ps_starter")
    )

    # Weighted-average rates
    def _wtd_rate(group: pd.DataFrame, num_col: str, denom_col: str) -> float:
        valid = group[group[denom_col] > 0]
        if valid.empty or valid["total_wt"].sum() == 0:
            return np.nan
        num = (valid[num_col] * valid["recency_wt"]).sum()
        denom = (valid[denom_col] * valid["recency_wt"]).sum()
        return num / denom if denom > 0 else np.nan

    agg = df.groupby("pitcher_id").apply(
        lambda g: pd.Series({
            "wtd_k_rate": _wtd_rate(g, "ps_k", "ps_bf"),
            "wtd_bb_rate": _wtd_rate(g, "ps_bb", "ps_bf"),
            "wtd_hr_rate": _wtd_rate(g, "ps_hr", "ps_bf"),
        }),
        include_groups=False,
    ).reset_index()

    agg = agg.merge(total_bf, on="pitcher_id")
    agg = agg.merge(best_rounds, on="pitcher_id")
    agg = agg.merge(starter_flag, on="pitcher_id")

    # Need minimum players for percentiles
    if len(agg) < 5:
        logger.warning("Only %d pitchers with postseason data — too few for pctl", len(agg))
        agg["postseason_raw_score"] = 0.50
        agg["postseason_score"] = 0.50
        return agg[["pitcher_id", "postseason_raw_score", "postseason_score",
                     "postseason_bf", "best_round"]]

    # Percentile ranks within postseason population
    k_rate_pctl = _percentile_rank(agg["wtd_k_rate"].fillna(agg["wtd_k_rate"].median()))
    bb_rate_pctl = _percentile_rank(agg["wtd_bb_rate"].fillna(agg["wtd_bb_rate"].median()))
    hr_rate_pctl = _percentile_rank(agg["wtd_hr_rate"].fillna(agg["wtd_hr_rate"].median()))

    # Raw score
    agg["postseason_raw_score"] = (
        0.50 * k_rate_pctl + 0.30 * (1.0 - bb_rate_pctl) + 0.20 * (1.0 - hr_rate_pctl)
    )

    # Dampening: separate curves for starters vs relievers
    starter_shrinkage = _shrinkage_pitcher_starter(agg["postseason_bf"])
    reliever_shrinkage = _shrinkage_pitcher_reliever(agg["postseason_bf"])
    shrinkage = np.where(agg["is_ps_starter"], starter_shrinkage, reliever_shrinkage)
    shrinkage = pd.Series(shrinkage, index=agg.index)

    agg["postseason_score"] = _apply_dampening(
        agg["postseason_raw_score"], shrinkage, agg["best_round"],
    )

    n_boosted = (agg["postseason_score"] > 0.52).sum()
    n_penalized = (agg["postseason_score"] < 0.48).sum()
    n_neutral = len(agg) - n_boosted - n_penalized
    logger.info(
        "Pitcher postseason scores: %d players — %d boosted, %d penalized, %d neutral "
        "(%d starters, %d relievers)",
        len(agg), n_boosted, n_penalized, n_neutral,
        agg["is_ps_starter"].sum(), (~agg["is_ps_starter"]).sum(),
    )

    return agg[["pitcher_id", "postseason_raw_score", "postseason_score",
                "postseason_bf", "best_round"]]
