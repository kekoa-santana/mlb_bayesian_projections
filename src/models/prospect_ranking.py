"""
TDD Prospect Ranking — composite prospect score blending performance,
readiness, age, trajectory, and positional scarcity.

Produces a proprietary TDD Prospect Score distinct from FanGraphs rankings.
FanGraphs Future Value (FV) is displayed alongside for context but labeled
explicitly as ``fg_future_value``.

Supports both batting and pitching prospects with role-appropriate metrics.

Dual ranking system:
- **tdd_prospect_score**: Balanced composite (default ranking)
- **impact_score**: Upside-focused (higher rate quality + trajectory weight)
- **floor_eta_score**: Floor/ETA-focused (higher readiness + age weight)

Components
----------
1. **Readiness** (~25%): P(sticks in MLB) from ``mlb_readiness.py`` (batters)
   or heuristic model (pitchers)
2. **Translated rate quality** (~30%): Percentile rank of translated stats
   vs MLB population — includes contact metrics for batters
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

# Impact score: upside-focused (lower readiness, higher quality + trajectory)
_IMPACT_WEIGHTS = {
    "readiness": 0.10,
    "rate_quality": 0.35,
    "age_rel": 0.15,
    "trajectory": 0.25,
    "positional": 0.15,
}

# Floor/ETA score: readiness-focused (higher readiness + age, lower trajectory)
_FLOOR_WEIGHTS = {
    "readiness": 0.40,
    "rate_quality": 0.20,
    "age_rel": 0.20,
    "trajectory": 0.10,
    "positional": 0.10,
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

# Pitcher positional scarcity (SP > RP)
_PITCHER_POS_SCARCITY = {
    "SP": 0.80,
    "RP": 0.50,
}

# Level numeric mapping (for trajectory calc)
_LEVEL_NUM = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4, "MLB": 5}


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a series to 0-1 percentile ranks (higher = better)."""
    return series.rank(pct=True, method="average")


def _compute_batter_rate_quality(df: pd.DataFrame) -> pd.Series:
    """Score translated batter rates against population percentiles.

    Includes contact quality via K-BB spread, power via ISO, and
    speed via SB rate. Rewards well-rounded offensive tools.

    Lower K% is better, higher BB%, ISO, and SB rate are better.
    A wide K-BB gap (high K, low BB) is penalized.
    """
    # Plate discipline: lower K% = better contact/bat-to-ball
    k_pctl = 1.0 - _percentile_rank(df["wtd_k_pct"])
    bb_pctl = _percentile_rank(df["wtd_bb_pct"])
    iso_pctl = _percentile_rank(df["wtd_iso"])

    # Contact quality: K-BB spread (lower = better plate approach)
    kbb_pctl = 1.0 - _percentile_rank(df["k_bb_diff"])

    # Speed tool: SB rate (higher = faster, more baserunning value)
    sb_pctl = _percentile_rank(df["sb_rate"].fillna(0))

    # HR power tool
    hr_pctl = _percentile_rank(df["wtd_hr_pa"].fillna(0))

    # Weight: K% (contact), BB% (discipline), ISO (power), HR/PA (power),
    # K-BB spread (hit tool), SB rate (speed)
    return (
        0.22 * k_pctl + 0.18 * bb_pctl + 0.15 * iso_pctl
        + 0.10 * hr_pctl + 0.20 * kbb_pctl + 0.15 * sb_pctl
    )


def _compute_pitcher_rate_quality(df: pd.DataFrame) -> pd.Series:
    """Score translated pitcher rates against population percentiles.

    Higher K% is better, lower BB% and HR rate are better.
    """
    k_pctl = _percentile_rank(df["wtd_k_pct"])
    bb_pctl = 1.0 - _percentile_rank(df["wtd_bb_pct"])
    hr_pctl = 1.0 - _percentile_rank(df["wtd_hr_bf"])

    # K-BB spread (higher = better for pitchers — more K, fewer BB)
    kbb_pctl = _percentile_rank(df["k_bb_diff"])

    return 0.35 * k_pctl + 0.25 * bb_pctl + 0.20 * hr_pctl + 0.20 * kbb_pctl


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
    k_col: str = "translated_k_pct",
    bb_col: str = "translated_bb_pct",
    pa_col: str = "pa",
) -> dict[int, float]:
    """Score promotion speed and stat improvement.

    Combines:
    - Levels climbed per MiLB season (faster = better)
    - K% trend (for batters: decreasing = better; for pitchers: increasing = better)
    - BB% trend (for batters: increasing = better; for pitchers: decreasing = better)

    Parameters
    ----------
    milb_df : pd.DataFrame
        Full translated MiLB data.
    prospect_ids : set[int]
        Player IDs to score.
    k_col, bb_col : str
        Column names for K% and BB%.
    pa_col : str
        Column for weighting (PA for batters, BF for pitchers).

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
            yearly = grp.groupby("season").agg(
                k_pct=(k_col, "mean"),
                bb_pct=(bb_col, "mean"),
                pa=(pa_col, "sum"),
            ).sort_index()

            if len(yearly) >= 2:
                # K% decreasing is good for batters (negative slope → positive score)
                k_diff = yearly["k_pct"].iloc[-1] - yearly["k_pct"].iloc[0]
                k_trend = np.clip(-k_diff / 0.10, -1, 1) * 0.5 + 0.5

                # BB% increasing is good for batters
                bb_diff = yearly["bb_pct"].iloc[-1] - yearly["bb_pct"].iloc[0]
                bb_trend = np.clip(bb_diff / 0.05, -1, 1) * 0.5 + 0.5

                stat_score = 0.5 * k_trend + 0.5 * bb_trend
            else:
                stat_score = 0.5
        else:
            stat_score = 0.5

        scores[pid] = 0.50 * promo_score + 0.50 * stat_score

    return scores


def _compute_transition_velocity(
    prospect_ids: set[int],
) -> dict[int, float]:
    """Score promotion velocity from fact_prospect_transition.

    Faster promotions (fewer days between levels) = better score.
    Demotions apply a mild penalty.

    Returns
    -------
    dict[int, float]
        Player ID -> velocity score (0-1).
    """
    try:
        from src.data.queries import get_prospect_transitions
        transitions = get_prospect_transitions(list(prospect_ids))
    except Exception:
        return {}

    if transitions.empty:
        return {}

    scores: dict[int, float] = {}

    for pid, grp in transitions.groupby("player_id"):
        promos = grp[grp["transition_type"] == "promotion"].sort_values("event_date")
        demos = grp[grp["transition_type"] == "demotion"]

        if promos.empty:
            scores[pid] = 0.3  # no promotions = low velocity
            continue

        # Average days between promotions
        if len(promos) >= 2:
            promo_dates = pd.to_datetime(promos["event_date"])
            avg_days = promo_dates.diff().dropna().dt.days.mean()
            # Normalize: 180 days between promos = 1.0, 720+ days = 0.0
            speed_score = float(np.clip(1.0 - (avg_days - 180) / 540, 0, 1))
        else:
            speed_score = 0.5

        # Demotion penalty: mild — single demotion is fine, multiple is concerning
        n_demos = len(demos)
        demo_penalty = min(n_demos * 0.10, 0.30)  # cap at 0.30

        scores[pid] = float(np.clip(speed_score - demo_penalty, 0, 1))

    return scores


def _compute_pitcher_trajectory(
    milb_df: pd.DataFrame,
    prospect_ids: set[int],
) -> dict[int, float]:
    """Score pitcher trajectory — K% increasing and BB% decreasing are good."""
    scores: dict[int, float] = {}
    sub = milb_df[milb_df["player_id"].isin(prospect_ids)].copy()
    sub["level_num"] = sub["level"].map(_LEVEL_NUM).fillna(0)

    for pid, grp in sub.groupby("player_id"):
        grp = grp.sort_values("season")

        seasons_played = grp["season"].nunique()
        level_range = grp["level_num"].max() - grp["level_num"].min()
        promo_speed = level_range / max(seasons_played, 1)
        promo_score = min(promo_speed / 2.0, 1.0)

        if len(grp) >= 2:
            yearly = grp.groupby("season").agg(
                k_pct=("translated_k_pct", "mean"),
                bb_pct=("translated_bb_pct", "mean"),
                bf=("bf", "sum"),
            ).sort_index()

            if len(yearly) >= 2:
                # K% increasing is good for pitchers (positive slope → positive)
                k_diff = yearly["k_pct"].iloc[-1] - yearly["k_pct"].iloc[0]
                k_trend = np.clip(k_diff / 0.10, -1, 1) * 0.5 + 0.5

                # BB% decreasing is good for pitchers
                bb_diff = yearly["bb_pct"].iloc[-1] - yearly["bb_pct"].iloc[0]
                bb_trend = np.clip(-bb_diff / 0.05, -1, 1) * 0.5 + 0.5

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


def _compute_promotion_resilience(
    milb_df: pd.DataFrame,
    prospect_ids: set[int],
) -> dict[int, float]:
    """Score how well a batter's K% held up across level promotions.

    A prospect whose K% barely moves on promotion is a safer bet than
    one whose K% spikes +10% at each new level.

    Returns
    -------
    dict[int, float]
        Player ID → resilience score (0–1, higher = more resilient).
    """
    scores: dict[int, float] = {}
    sub = milb_df[milb_df["player_id"].isin(prospect_ids)].copy()
    sub["level_num"] = sub["level"].map(_LEVEL_NUM).fillna(0)

    for pid, grp in sub.groupby("player_id"):
        # Need data at 2+ levels
        if grp["level"].nunique() < 2:
            scores[pid] = 0.5
            continue

        # Get K% by level (PA-weighted within level)
        by_level = (
            grp.groupby("level_num")
            .apply(
                lambda g: np.average(g["translated_k_pct"], weights=g["pa"])
                if g["pa"].sum() > 0 else np.nan,
                include_groups=False,
            )
            .dropna()
            .sort_index()
        )

        if len(by_level) < 2:
            scores[pid] = 0.5
            continue

        # Average K% increase per level jump
        level_diffs = by_level.diff().dropna()
        avg_k_spike = level_diffs.mean()

        # Convert: -0.05 (K% dropped) → 1.0, 0 (held steady) → 0.75,
        # +0.05 (5% spike) → 0.5, +0.10 → 0.25, +0.15+ → 0.0
        scores[pid] = float(np.clip(0.75 - avg_k_spike * 5.0, 0, 1))

    return scores


def _compute_games_played_ratio(
    milb_df: pd.DataFrame,
    prospect_ids: set[int],
    full_season_games: int = 130,
) -> dict[int, float]:
    """Score prospect availability/durability from games played.

    A prospect who plays 130+ games shows durability. One who plays
    60 games (injury/demotion) is riskier.

    Returns
    -------
    dict[int, float]
        Player ID → availability score (0–1).
    """
    scores: dict[int, float] = {}
    sub = milb_df[milb_df["player_id"].isin(prospect_ids)].copy()

    for pid, grp in sub.groupby("player_id"):
        # Games per season (sum across levels within a season)
        by_season = grp.groupby("season")["games"].sum()
        if by_season.empty:
            scores[pid] = 0.5
            continue

        # Use most recent 2 seasons, recency-weighted
        recent = by_season.sort_index().tail(2)
        if len(recent) == 2:
            wtd_games = 0.4 * recent.iloc[0] + 0.6 * recent.iloc[1]
        else:
            wtd_games = recent.iloc[0]

        # Normalize: 130+ games = 1.0, 65 games = 0.5, 0 = 0.0
        scores[pid] = float(np.clip(wtd_games / full_season_games, 0, 1))

    return scores


def _compute_pitcher_positional_score(df: pd.DataFrame) -> pd.Series:
    """Score pitcher positional scarcity (SP > RP)."""
    return df["pitcher_role"].map(_PITCHER_POS_SCARCITY).fillna(0.50)


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


def _compute_composite_scores(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute weighted composite from component scores.

    Rate quality is age-adjusted: young + good scores higher than
    treating them independently.
    """
    # Age-adjusted rate quality: multiply rate quality by a mild age bonus
    # Young (age score ~1.0): rate quality boosted by ~15%
    # Old (age score ~0.0): rate quality reduced by ~15%
    age_adj = 0.85 + 0.30 * df["comp_age"]  # 0.85 to 1.15
    adj_rate_quality = (df["comp_rate_quality"] * age_adj).clip(0, 1)

    return (
        weights["readiness"] * df["comp_readiness"]
        + weights["rate_quality"] * adj_rate_quality
        + weights["age_rel"] * df["comp_age"]
        + weights["trajectory"] * df["comp_trajectory"]
        + weights["positional"] * df["comp_positional"]
    )


# ===================================================================
# Pitching prospect feature builder
# ===================================================================

_PITCHER_LEVEL_MAP = {"A": 1, "A+": 2, "AA": 3, "AAA": 4}


def _build_pitcher_prospect_features(
    milb_df: pd.DataFrame,
    max_prospect_age: int = 27,
    min_bf: int = 50,
) -> pd.DataFrame:
    """Build feature matrix from translated MiLB pitcher data.

    Parameters
    ----------
    milb_df : pd.DataFrame
        Translated MiLB pitcher data.
    max_prospect_age : int
        Maximum age to be considered a prospect.
    min_bf : int
        Minimum career MiLB batters faced.

    Returns
    -------
    pd.DataFrame
        One row per pitching prospect with features + metadata.
    """
    records: list[dict[str, Any]] = []
    for pid, grp in milb_df.groupby("player_id"):
        if grp["age_at_level"].min() > max_prospect_age:
            continue

        grp = grp.sort_values(["season", "level"])
        grp["_lvl_num"] = grp["level"].map(_PITCHER_LEVEL_MAP).fillna(0)
        best = grp.sort_values(
            ["_lvl_num", "season"], ascending=[False, False]
        ).iloc[0]

        total_bf = grp["bf"].sum()
        if total_bf < min_bf:
            continue

        conf = grp["translation_confidence"].fillna(0.5) if "translation_confidence" in grp.columns else pd.Series(0.5, index=grp.index)
        conf_bf = conf * grp["bf"]
        conf_bf_sum = conf_bf.sum()
        if conf_bf_sum > 0:
            wtd_k = (grp["translated_k_pct"] * conf_bf).sum() / conf_bf_sum
            wtd_bb = (grp["translated_bb_pct"] * conf_bf).sum() / conf_bf_sum
            wtd_hr = (grp["translated_hr_bf"] * conf_bf).sum() / conf_bf_sum
        else:
            wtd_k = (grp["translated_k_pct"] * grp["bf"]).sum() / total_bf
            wtd_bb = (grp["translated_bb_pct"] * grp["bf"]).sum() / total_bf
            wtd_hr = (grp["translated_hr_bf"] * grp["bf"]).sum() / total_bf

        # Determine SP vs RP from games_started ratio
        total_games = grp["games"].sum()
        total_gs = grp["games_started"].sum()
        sp_pct = total_gs / max(total_games, 1)
        role = "SP" if sp_pct > 0.5 else "RP"

        records.append({
            "player_id": pid,
            "name": best["player_name"],
            "latest_season": int(grp["season"].max()),
            "max_level": best["level"],
            "max_level_num": grp["_lvl_num"].max(),
            "levels_played": grp["level"].nunique(),
            "career_milb_bf": total_bf,
            "career_seasons": grp["season"].nunique(),
            "wtd_k_pct": wtd_k,
            "wtd_bb_pct": wtd_bb,
            "wtd_hr_bf": wtd_hr,
            "k_bb_diff": wtd_k - wtd_bb,
            "youngest_age_rel": grp["age_relative_to_level_avg"].min(),
            "avg_age_rel": grp["age_relative_to_level_avg"].mean(),
            "min_age": grp["age_at_level"].min(),
            "pitcher_role": role,
            "sp_pct": sp_pct,
            "primary_position": "P",
            "prospect_type": "pitcher",
        })

    return pd.DataFrame(records)


def _compute_pitcher_readiness_heuristic(df: pd.DataFrame) -> pd.Series:
    """Heuristic readiness score for pitching prospects.

    Uses level attained, age, K-BB differential, and BF sample size
    as proxy for MLB readiness since we don't have a trained LogReg
    for pitchers.

    Returns 0-1 score.
    """
    # Level component: higher level = more ready
    level_score = df["max_level_num"].clip(0, 4) / 4.0

    # Age component: younger at same level = better prospect but less "ready"
    # Invert for readiness: older + high level = more ready NOW
    age_norm = ((df["min_age"] - 18) / 10.0).clip(0, 1)

    # K-BB component: wider positive gap = better stuff
    kbb_score = _percentile_rank(df["k_bb_diff"])

    # Sample size: more BF = more confidence
    bf_score = (df["career_milb_bf"] / 500.0).clip(0, 1)

    return 0.40 * level_score + 0.20 * age_norm + 0.25 * kbb_score + 0.15 * bf_score


def rank_prospects(
    milb_df: pd.DataFrame | None = None,
    readiness_df: pd.DataFrame | None = None,
    projection_season: int = 2026,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build composite TDD prospect rankings for batting prospects.

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
        Includes component sub-scores, dual rankings, and FanGraphs FV.
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
    df["prospect_type"] = "batter"

    # -------------------------------------------------------------------
    # Component 1: Readiness (already 0-1 probability)
    # -------------------------------------------------------------------
    df["comp_readiness"] = df["readiness_score"]

    # -------------------------------------------------------------------
    # Component 2: Translated rate quality (with contact metrics)
    # -------------------------------------------------------------------
    df["comp_rate_quality"] = _compute_batter_rate_quality(df)

    # -------------------------------------------------------------------
    # Component 3: Age-relative-to-level
    # -------------------------------------------------------------------
    df["comp_age"] = _compute_age_score(df)

    # -------------------------------------------------------------------
    # Component 4: Trajectory (promotion speed + stat trends +
    #              promotion resilience + availability)
    # -------------------------------------------------------------------
    prospect_ids = set(df["player_id"].unique())
    traj_scores = _compute_trajectory(milb_df, prospect_ids)
    resilience_scores = _compute_promotion_resilience(milb_df, prospect_ids)
    availability_scores = _compute_games_played_ratio(milb_df, prospect_ids)

    base_traj = df["player_id"].map(traj_scores).fillna(0.5)
    resilience = df["player_id"].map(resilience_scores).fillna(0.5)
    availability = df["player_id"].map(availability_scores).fillna(0.5)

    # Transition velocity from fact_prospect_transition
    transition_scores = _compute_transition_velocity(prospect_ids)
    transition_vel = df["player_id"].map(transition_scores).fillna(0.5)

    # Blend: transition velocity available -> use it; otherwise fall back to prior blend
    has_transition = df["player_id"].map(lambda x: x in transition_scores)
    df["comp_trajectory"] = np.where(
        has_transition,
        0.35 * base_traj + 0.25 * transition_vel + 0.20 * resilience + 0.20 * availability,
        0.50 * base_traj + 0.25 * resilience + 0.25 * availability,
    )
    df["promotion_resilience"] = resilience
    df["availability_score"] = availability

    # -------------------------------------------------------------------
    # Component 5: Positional scarcity
    # -------------------------------------------------------------------
    df["comp_positional"] = _compute_positional_score(df)

    # -------------------------------------------------------------------
    # Composite scores: balanced, impact, and floor/ETA
    # -------------------------------------------------------------------
    df["tdd_prospect_score"] = _compute_composite_scores(df, weights)
    df["impact_score"] = _compute_composite_scores(df, _IMPACT_WEIGHTS)
    df["floor_eta_score"] = _compute_composite_scores(df, _FLOOR_WEIGHTS)

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
        "max_level", "min_age", "prospect_type",
        # TDD scores
        "tdd_prospect_score", "tdd_tier",
        "impact_score", "floor_eta_score",
        "comp_readiness", "comp_rate_quality", "comp_age",
        "comp_trajectory", "comp_positional",
        # Translated stats
        "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "wtd_hr_pa", "k_bb_diff", "sb_rate",
        "career_milb_pa", "youngest_age_rel",
        # Sub-trajectory components
        "promotion_resilience", "availability_score",
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
        "TDD Prospect Rankings (batters): %d prospects — %d Elite, %d Impact, %d Solid",
        len(result),
        (result["tdd_tier"] == "Elite").sum(),
        (result["tdd_tier"] == "Impact").sum(),
        (result["tdd_tier"] == "Solid").sum(),
    )
    return result


def rank_pitching_prospects(
    milb_df: pd.DataFrame | None = None,
    projection_season: int = 2026,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build composite TDD prospect rankings for pitching prospects.

    Parameters
    ----------
    milb_df : pd.DataFrame or None
        Translated MiLB pitcher data. If None, loads from cache.
    projection_season : int
        Target season for rankings.
    weights : dict or None
        Override component weights.

    Returns
    -------
    pd.DataFrame
        Pitching prospect rankings sorted by ``tdd_prospect_score``.
    """
    if weights is None:
        weights = _WEIGHTS

    if milb_df is None:
        milb_df = pd.read_parquet(CACHE_DIR / "milb_translated_pitchers.parquet")

    # Filter to recent prospects (active in last 2 seasons)
    recent_seasons = [projection_season - 2, projection_season - 1]
    recent_ids = set(
        milb_df[milb_df["season"].isin(recent_seasons)]["player_id"].unique()
    )
    milb_recent = milb_df[milb_df["player_id"].isin(recent_ids)].copy()

    # Exclude players who already have significant MLB time
    try:
        from src.data.db import read_sql
        mlb_pitchers = read_sql("""
            SELECT DISTINCT pitcher_id FROM production.fact_pitching_advanced
            WHERE batters_faced >= 100
        """, {})
        mlb_ids = set(mlb_pitchers["pitcher_id"].astype(int))
    except Exception:
        mlb_ids = set()

    milb_recent = milb_recent[~milb_recent["player_id"].isin(mlb_ids)]

    # Build features
    df = _build_pitcher_prospect_features(milb_recent)
    if df.empty:
        logger.warning("No pitching prospects found")
        return pd.DataFrame()

    logger.info("Building pitcher prospect rankings for %d prospects", len(df))

    # -------------------------------------------------------------------
    # Component 1: Readiness (ML model with heuristic fallback)
    # -------------------------------------------------------------------
    # Try ML model first, fall back to heuristic
    try:
        from src.models.mlb_readiness import train_pitcher_readiness_model
        pitcher_model = train_pitcher_readiness_model(milb_df=milb_recent)
        _pitcher_features = pitcher_model["features"]
        X = df[_pitcher_features].fillna(0).values
        X_s = pitcher_model["scaler"].transform(X)
        df["comp_readiness"] = pitcher_model["model"].predict_proba(X_s)[:, 1]
        logger.info("Pitcher readiness: using ML model (AUC=%.3f)", pitcher_model["train_auc"])
    except Exception as e:
        logger.warning("Pitcher readiness ML failed (%s), using heuristic", e)
        df["comp_readiness"] = _compute_pitcher_readiness_heuristic(df)
    df["readiness_score"] = df["comp_readiness"]
    df["readiness_tier"] = pd.cut(
        df["readiness_score"],
        bins=[0, 0.15, 0.25, 0.40, 0.55, 1.0],
        labels=["Long Shot", "Fringe", "Developing", "Strong", "Elite"],
    )

    # -------------------------------------------------------------------
    # Component 2: Translated rate quality
    # -------------------------------------------------------------------
    df["comp_rate_quality"] = _compute_pitcher_rate_quality(df)

    # -------------------------------------------------------------------
    # Component 3: Age-relative-to-level
    # -------------------------------------------------------------------
    df["comp_age"] = _compute_age_score(df)

    # -------------------------------------------------------------------
    # Component 4: Trajectory
    # -------------------------------------------------------------------
    prospect_ids = set(df["player_id"].unique())
    traj_scores = _compute_pitcher_trajectory(milb_recent, prospect_ids)
    base_traj = df["player_id"].map(traj_scores).fillna(0.5)

    # Transition velocity
    transition_scores = _compute_transition_velocity(prospect_ids)
    transition_vel = df["player_id"].map(transition_scores).fillna(0.5)

    has_transition = df["player_id"].map(lambda x: x in transition_scores)
    df["comp_trajectory"] = np.where(
        has_transition,
        0.50 * base_traj + 0.50 * transition_vel,
        base_traj,
    )

    # -------------------------------------------------------------------
    # Component 5: Positional scarcity (SP > RP)
    # -------------------------------------------------------------------
    df["comp_positional"] = _compute_pitcher_positional_score(df)

    # -------------------------------------------------------------------
    # Composite scores: balanced, impact, and floor/ETA
    # -------------------------------------------------------------------
    df["tdd_prospect_score"] = _compute_composite_scores(df, weights)
    df["impact_score"] = _compute_composite_scores(df, _IMPACT_WEIGHTS)
    df["floor_eta_score"] = _compute_composite_scores(df, _FLOOR_WEIGHTS)

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
    # Merge FanGraphs FV for display
    # -------------------------------------------------------------------
    fg = _load_fg_rankings(projection_season)
    if not fg.empty:
        df = df.merge(fg, on="player_id", how="left")
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
        "tdd_rank", "player_id", "name", "pitcher_role", "max_level",
        "min_age", "prospect_type",
        # TDD scores
        "tdd_prospect_score", "tdd_tier",
        "impact_score", "floor_eta_score",
        "comp_readiness", "comp_rate_quality", "comp_age",
        "comp_trajectory", "comp_positional",
        # Translated stats
        "wtd_k_pct", "wtd_bb_pct", "wtd_hr_bf", "k_bb_diff",
        "career_milb_bf", "youngest_age_rel", "sp_pct",
        # Readiness
        "readiness_score", "readiness_tier",
        # FanGraphs (display only)
        "fg_future_value", "fg_overall_rank", "fg_org_rank",
        "fg_risk", "fg_eta",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].copy()

    logger.info(
        "TDD Prospect Rankings (pitchers): %d prospects — %d Elite, %d Impact, %d Solid",
        len(result),
        (result["tdd_tier"] == "Elite").sum(),
        (result["tdd_tier"] == "Impact").sum(),
        (result["tdd_tier"] == "Solid").sum(),
    )
    return result


def rank_all_prospects(
    projection_season: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Run both batter and pitcher prospect rankings.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'batters', 'pitchers'.
    """
    batters = rank_prospects(projection_season=projection_season)
    pitchers = rank_pitching_prospects(projection_season=projection_season)
    logger.info(
        "All prospect rankings: %d batters, %d pitchers",
        len(batters), len(pitchers),
    )
    return {"batters": batters, "pitchers": pitchers}
