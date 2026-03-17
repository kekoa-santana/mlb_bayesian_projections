"""
TDD Prospect Ranking — composite prospect score blending performance,
readiness, age, trajectory, and positional scarcity.

Produces a proprietary TDD Prospect Score distinct from FanGraphs rankings.
FanGraphs Future Value (FV) is displayed alongside for context but labeled
explicitly as ``fg_future_value``.

Components
----------
1. **Readiness** (~25%): P(sticks in MLB) from ``mlb_readiness.py``
2. **Translated rate quality** (~30%): Percentile rank of translated
   K%, BB%, ISO vs MLB population — how good will they be?
3. **Age-relative-to-level** (~15%): Younger-for-level bonus
4. **Trajectory** (~15%): Promotion speed + stat improvement YoY
5. **Positional scarcity** (~15%): Defensive spectrum weighting
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"

# ---------------------------------------------------------------------------
# Component weights (sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHTS = {
    "readiness": 0.25,
    "rate_quality": 0.30,
    "age_rel": 0.15,
    "trajectory": 0.15,
    "positional": 0.15,
}

# ---------------------------------------------------------------------------
# Positional scarcity multipliers (defensive spectrum)
# Higher = scarcer / more valuable position
# ---------------------------------------------------------------------------
_POS_SCARCITY = {
    "C": 1.00,
    "SS": 0.90,
    "CF": 0.80,
    "2B": 0.70,
    "3B": 0.65,
    "RF": 0.55,
    "LF": 0.45,
    "1B": 0.30,
    "DH": 0.20,
}

# Level numeric mapping (for trajectory calc)
_LEVEL_NUM = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4, "MLB": 5}


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a series to 0-1 percentile ranks (higher = better)."""
    return series.rank(pct=True, method="average")


def _compute_rate_quality(df: pd.DataFrame) -> pd.Series:
    """Score translated rates against MLB-population percentiles.

    Lower K% is better, higher BB% and ISO are better.
    """
    # Invert K% so lower is better
    k_pctl = 1.0 - _percentile_rank(df["wtd_k_pct"])
    bb_pctl = _percentile_rank(df["wtd_bb_pct"])
    iso_pctl = _percentile_rank(df["wtd_iso"])

    # Weight: K-BB approach matters most, ISO secondary
    return 0.40 * k_pctl + 0.30 * bb_pctl + 0.30 * iso_pctl


def _compute_age_score(df: pd.DataFrame) -> pd.Series:
    """Score age-relative-to-level (younger = better).

    Uses youngest_age_rel (negative = younger than avg for level).
    Clip and invert so younger prospects score higher.
    """
    clipped = df["youngest_age_rel"].clip(-5, 5)
    # Invert: -5 (very young) → 1.0, +5 (old) → 0.0
    return (5.0 - clipped) / 10.0


def _compute_trajectory(
    milb_df: pd.DataFrame,
    prospect_ids: set[int],
) -> dict[int, float]:
    """Score promotion speed and stat improvement.

    Combines:
    - Levels climbed per MiLB season (faster = better)
    - K% improvement trend (decreasing translated K% = better)
    - BB% improvement trend (increasing translated BB% = better)

    Parameters
    ----------
    milb_df : pd.DataFrame
        Full translated MiLB batter data.
    prospect_ids : set[int]
        Player IDs to score.

    Returns
    -------
    dict[int, float]
        Player ID → trajectory score (0–1).
    """
    scores: dict[int, float] = {}
    sub = milb_df[milb_df["player_id"].isin(prospect_ids)].copy()
    sub["level_num"] = sub["level"].map(_LEVEL_NUM).fillna(0)

    for pid, grp in sub.groupby("player_id"):
        grp = grp.sort_values("season")

        # Promotion speed: levels climbed / seasons played
        seasons_played = grp["season"].nunique()
        level_range = grp["level_num"].max() - grp["level_num"].min()
        promo_speed = level_range / max(seasons_played, 1)
        # Normalize: 0 levels/yr → 0, 2+ levels/yr → 1
        promo_score = min(promo_speed / 2.0, 1.0)

        # Stat trend: need 2+ season-level combos
        if len(grp) >= 2:
            # Weighted by recency
            yearly = grp.groupby("season").agg(
                k_pct=("translated_k_pct", "mean"),
                bb_pct=("translated_bb_pct", "mean"),
                pa=("pa", "sum"),
            ).sort_index()

            if len(yearly) >= 2:
                # K% decreasing is good (negative slope → positive score)
                k_diff = yearly["k_pct"].iloc[-1] - yearly["k_pct"].iloc[0]
                k_trend = np.clip(-k_diff / 0.10, -1, 1) * 0.5 + 0.5

                # BB% increasing is good
                bb_diff = yearly["bb_pct"].iloc[-1] - yearly["bb_pct"].iloc[0]
                bb_trend = np.clip(bb_diff / 0.05, -1, 1) * 0.5 + 0.5

                stat_score = 0.5 * k_trend + 0.5 * bb_trend
            else:
                stat_score = 0.5
        else:
            stat_score = 0.5

        scores[pid] = 0.50 * promo_score + 0.50 * stat_score

    return scores


def _compute_positional_score(df: pd.DataFrame) -> pd.Series:
    """Score based on defensive spectrum scarcity."""
    return df["primary_position"].map(_POS_SCARCITY).fillna(0.40)


def _load_fg_rankings(season: int) -> pd.DataFrame:
    """Load FanGraphs prospect rankings for display (not model input).

    Returns DataFrame with player_id, fg_future_value, fg_overall_rank,
    fg_org_rank, fg_risk, fg_eta.
    """
    try:
        from src.data.db import read_sql
        rankings = read_sql(
            "SELECT player_id, future_value, overall_rank, org_rank, "
            "risk, eta, source "
            "FROM production.dim_prospect_ranking "
            f"WHERE season = {season}",
            {},
        )
    except Exception:
        logger.warning("Could not load FG rankings for season %d", season)
        return pd.DataFrame()

    if rankings.empty:
        return pd.DataFrame()

    # Prefer fg_report (preseason), fall back to fg_updated
    rankings = (
        rankings.sort_values("source", ascending=True)
        .drop_duplicates("player_id", keep="first")
    )
    return rankings.rename(columns={
        "future_value": "fg_future_value",
        "overall_rank": "fg_overall_rank",
        "org_rank": "fg_org_rank",
        "risk": "fg_risk",
        "eta": "fg_eta",
    })[["player_id", "fg_future_value", "fg_overall_rank",
        "fg_org_rank", "fg_risk", "fg_eta"]]


def rank_prospects(
    milb_df: pd.DataFrame | None = None,
    readiness_df: pd.DataFrame | None = None,
    projection_season: int = 2026,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build composite TDD prospect rankings.

    Parameters
    ----------
    milb_df : pd.DataFrame or None
        Translated MiLB batter data. If None, loads from cache.
    readiness_df : pd.DataFrame or None
        Output of ``score_prospects()``. If None, runs readiness model.
    projection_season : int
        Target season for rankings.
    weights : dict or None
        Override component weights. Keys: readiness, rate_quality,
        age_rel, trajectory, positional.

    Returns
    -------
    pd.DataFrame
        Prospect rankings sorted by ``tdd_prospect_score`` descending.
        Includes component sub-scores and FanGraphs FV for reference.
    """
    if weights is None:
        weights = _WEIGHTS

    # Load data
    if milb_df is None:
        milb_df = pd.read_parquet(CACHE_DIR / "milb_translated_batters.parquet")

    if readiness_df is None:
        from src.models.mlb_readiness import score_prospects
        readiness_df = score_prospects(
            milb_df=milb_df, projection_season=projection_season,
        )

    if readiness_df.empty:
        logger.warning("No readiness scores available, cannot rank prospects")
        return pd.DataFrame()

    df = readiness_df.copy()

    # -------------------------------------------------------------------
    # Component 1: Readiness (already 0-1 probability)
    # -------------------------------------------------------------------
    df["comp_readiness"] = df["readiness_score"]

    # -------------------------------------------------------------------
    # Component 2: Translated rate quality (percentile within cohort)
    # -------------------------------------------------------------------
    df["comp_rate_quality"] = _compute_rate_quality(df)

    # -------------------------------------------------------------------
    # Component 3: Age-relative-to-level
    # -------------------------------------------------------------------
    df["comp_age"] = _compute_age_score(df)

    # -------------------------------------------------------------------
    # Component 4: Trajectory (promotion speed + stat trends)
    # -------------------------------------------------------------------
    prospect_ids = set(df["player_id"].unique())
    traj_scores = _compute_trajectory(milb_df, prospect_ids)
    df["comp_trajectory"] = df["player_id"].map(traj_scores).fillna(0.5)

    # -------------------------------------------------------------------
    # Component 5: Positional scarcity
    # -------------------------------------------------------------------
    df["comp_positional"] = _compute_positional_score(df)

    # -------------------------------------------------------------------
    # Composite score (weighted sum → 0-1 scale)
    # -------------------------------------------------------------------
    df["tdd_prospect_score"] = (
        weights["readiness"] * df["comp_readiness"]
        + weights["rate_quality"] * df["comp_rate_quality"]
        + weights["age_rel"] * df["comp_age"]
        + weights["trajectory"] * df["comp_trajectory"]
        + weights["positional"] * df["comp_positional"]
    )

    # Rank
    df = df.sort_values("tdd_prospect_score", ascending=False).reset_index(drop=True)
    df["tdd_rank"] = range(1, len(df) + 1)

    # Tier labels
    df["tdd_tier"] = pd.cut(
        df["tdd_prospect_score"],
        bins=[0, 0.25, 0.40, 0.55, 0.70, 1.0],
        labels=["Org Filler", "Developing", "Solid", "Impact", "Elite"],
    )

    # -------------------------------------------------------------------
    # Merge FanGraphs FV for display (clearly labeled)
    # -------------------------------------------------------------------
    fg = _load_fg_rankings(projection_season)
    if not fg.empty:
        df = df.merge(fg, on="player_id", how="left")
        logger.info("Merged FG rankings for %d prospects", df["fg_future_value"].notna().sum())
    else:
        df["fg_future_value"] = np.nan
        df["fg_overall_rank"] = np.nan
        df["fg_org_rank"] = np.nan
        df["fg_risk"] = np.nan
        df["fg_eta"] = np.nan

    # -------------------------------------------------------------------
    # Select output columns
    # -------------------------------------------------------------------
    output_cols = [
        # Identity
        "tdd_rank", "player_id", "name", "primary_position", "pos_group",
        "max_level", "min_age",
        # TDD scores
        "tdd_prospect_score", "tdd_tier",
        "comp_readiness", "comp_rate_quality", "comp_age",
        "comp_trajectory", "comp_positional",
        # Translated stats
        "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "k_bb_diff", "sb_rate",
        "career_milb_pa", "youngest_age_rel",
        # Readiness
        "readiness_score", "readiness_tier",
        # Org depth
        "n_above", "total_at_pos_in_org",
        # FanGraphs (display only)
        "fg_future_value", "fg_overall_rank", "fg_org_rank",
        "fg_risk", "fg_eta",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].copy()

    logger.info(
        "TDD Prospect Rankings: %d prospects — %d Elite, %d Impact, %d Solid",
        len(result),
        (result["tdd_tier"] == "Elite").sum(),
        (result["tdd_tier"] == "Impact").sum(),
        (result["tdd_tier"] == "Solid").sum(),
    )
    return result
