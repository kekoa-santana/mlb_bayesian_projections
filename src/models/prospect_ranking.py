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
# Batter component weights (sum to 1.0)
# ---------------------------------------------------------------------------
# Fully data-driven: no external scouting (FG FV) or readiness.
# pwOBA is the core BABIP-stripped quality signal (R²=0.790).
# Diamond Rating is internal tool grades (position-aware 20-80).
# Age-for-level is the key differentiator — doing it young matters.
_WEIGHTS = {
    "pwoba": 0.35,
    "diamond_rating": 0.15,
    "age_rel": 0.20,
    "trajectory": 0.15,
    "rate_quality": 0.10,
    "positional": 0.05,
}

# Impact score: upside-focused (max ceiling)
_IMPACT_WEIGHTS = {
    "pwoba": 0.30,
    "diamond_rating": 0.15,
    "age_rel": 0.25,
    "trajectory": 0.20,
    "rate_quality": 0.10,
    "positional": 0.00,
}

# Floor/ETA score: proven production at scarce positions
_FLOOR_WEIGHTS = {
    "pwoba": 0.35,
    "diamond_rating": 0.10,
    "age_rel": 0.10,
    "trajectory": 0.10,
    "rate_quality": 0.10,
    "positional": 0.25,
}

# Pitcher weights (no pwOBA — pitcher_quality serves same role)
_PITCHER_WEIGHTS = {
    "pitcher_quality": 0.40,
    "diamond_rating": 0.15,
    "age_rel": 0.20,
    "trajectory": 0.15,
    "positional": 0.10,
}

_PITCHER_IMPACT_WEIGHTS = {
    "pitcher_quality": 0.30,
    "diamond_rating": 0.15,
    "age_rel": 0.25,
    "trajectory": 0.25,
    "positional": 0.05,
}

_PITCHER_FLOOR_WEIGHTS = {
    "pitcher_quality": 0.40,
    "diamond_rating": 0.10,
    "age_rel": 0.10,
    "trajectory": 0.10,
    "positional": 0.30,
}

# ---------------------------------------------------------------------------
# Positional scarcity multipliers (defensive spectrum)
# Compressed range (0.40-0.80) so position matters but doesn't dominate.
# A great hitter at 1B should still rank well — the best bat wins.
# ---------------------------------------------------------------------------
_POS_SCARCITY = {
    "C": 0.80,
    "SS": 0.75,
    "CF": 0.70,
    "2B": 0.65,
    "3B": 0.60,
    "RF": 0.55,
    "LF": 0.50,
    "1B": 0.45,
    "DH": 0.40,
}

# Pitcher positional scarcity (SP > RP)
_PITCHER_POS_SCARCITY = {
    "SP": 0.80,
    "RP": 0.50,
}

# Level numeric mapping (for trajectory calc)
_LEVEL_NUM = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4, "MLB": 5}

# ---------------------------------------------------------------------------
# MLB prospect eligibility thresholds
# ---------------------------------------------------------------------------
# Prospects with brief MLB callups should remain in prospect rankings.
# ~400 BF ≈ 100 IP (roughly one full season of pitching).
# ~200 PA ≈ 130 AB (close to MLB prospect eligibility rules).
_MAX_MLB_BF_PROSPECT = 400
_MAX_MLB_PA_PROSPECT = 200

# MLB debut data is more informative per unit than translated MiLB.
# 1 MLB BF/PA counts as this many translated MiLB BF/PA for blending.
_MLB_RELIABILITY_MULTIPLIER = 2.0

# Recency decay half-life for MiLB aggregation (years).
# A developing prospect's most recent data is more predictive than older data.
# 2-year half-life: 1yr ago = 71% weight, 2yr ago = 50%, 3yr ago = 35%.
_RECENCY_HALF_LIFE = 2.0


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

    # Batted ball profile: GB% (lower = more fly ball, more power upside)
    if "wtd_gb_pct" in df.columns and df["wtd_gb_pct"].notna().any():
        gb_pctl = 1.0 - _percentile_rank(df["wtd_gb_pct"].fillna(0.5))
    else:
        gb_pctl = 0.5  # neutral if not available

    # Weight: K% (contact), BB% (discipline), ISO (power), HR/PA (power),
    # K-BB spread (hit tool), SB rate (speed), GB% (batted ball profile)
    return (
        0.20 * k_pctl + 0.16 * bb_pctl + 0.13 * iso_pctl
        + 0.09 * hr_pctl + 0.18 * kbb_pctl + 0.13 * sb_pctl
        + 0.11 * gb_pctl
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

        # Drop in-progress seasons (<30 games) — a 5-game AAA stint
        # in early April shouldn't tank a prospect's availability score.
        by_season = by_season[by_season >= 30]
        if by_season.empty:
            scores[pid] = 0.5
            continue

        # Use most recent 2 completed seasons, recency-weighted
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


# ---------------------------------------------------------------------------
# pwOBA — Peripheral wOBA (BABIP-stripped quality signal)
# ---------------------------------------------------------------------------
# Fit on 8,873 MLB player-seasons (2000-2025, 200+ PA), R²=0.790
_PWOBA_COEFS = {
    "intercept": 0.199,
    "k_pct": -0.228,
    "bb_pct": 0.347,
    "iso": 0.575,
    "gb_pct": 0.078,
    "sb_rate": 0.160,
    "hbp_pct": 0.400,
}
_LEAGUE_AVG_HBP_RATE = 0.012


def _compute_pwoba(df: pd.DataFrame) -> pd.Series:
    """Compute peripheral wOBA from translated MiLB rates.

    Returns raw pwOBA value (typically 0.250-0.400 range).
    Uses league-average HBP rate when prospect HBP data is missing.
    """
    hbp = df["hbp_rate"].fillna(_LEAGUE_AVG_HBP_RATE) if "hbp_rate" in df.columns else _LEAGUE_AVG_HBP_RATE
    gb = df["wtd_gb_pct"].fillna(0.44) if "wtd_gb_pct" in df.columns else 0.44
    sb = df["sb_rate"].fillna(0.0) if "sb_rate" in df.columns else 0.0

    return (
        _PWOBA_COEFS["intercept"]
        + _PWOBA_COEFS["k_pct"] * df["wtd_k_pct"]
        + _PWOBA_COEFS["bb_pct"] * df["wtd_bb_pct"]
        + _PWOBA_COEFS["iso"] * df["wtd_iso"]
        + _PWOBA_COEFS["gb_pct"] * gb
        + _PWOBA_COEFS["sb_rate"] * sb
        + _PWOBA_COEFS["hbp_pct"] * hbp
    )


def _compute_composite_scores(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute weighted composite from component scores.

    Fully data-driven: uses pwOBA (batters) or pitcher_quality (pitchers)
    as the core quality signal, internal Diamond Rating for tool grades,
    and age/trajectory/positional components.

    The concave age adjustment amplifies the quality signal for young
    prospects: a 19yo posting .330 pwOBA at AA is far more valuable
    than a 24yo doing the same.
    """
    score = pd.Series(0.0, index=df.index)

    # Concave age adjustment: applied to the primary quality signal
    # age_score ~0.0 (old): adj = 0.80, ~0.5 (avg): adj = 0.99, ~1.0 (young): adj = 1.20
    age_adj = 0.80 + 0.40 * df["comp_age"] ** 0.7

    # Batter quality: pwOBA (age-adjusted)
    if "pwoba" in weights and "comp_pwoba" in df.columns:
        adj_pwoba = (df["comp_pwoba"] * age_adj).clip(0, 1)
        score += weights["pwoba"] * adj_pwoba

    # Pitcher quality: rate quality (age-adjusted) — same role as pwOBA for batters
    if "pitcher_quality" in weights and "comp_rate_quality" in df.columns:
        adj_quality = (df["comp_rate_quality"] * age_adj).clip(0, 1)
        score += weights["pitcher_quality"] * adj_quality

    # Profile balance (rate quality for batters — NOT age-adjusted)
    if "rate_quality" in weights and "comp_rate_quality" in df.columns:
        score += weights["rate_quality"] * df["comp_rate_quality"]

    # Internal tool grades (Diamond Rating 0-10 → 0-1)
    if "diamond_rating" in weights:
        comp_diamond = df.get("comp_diamond", pd.Series(0.50, index=df.index))
        score += weights["diamond_rating"] * comp_diamond

    # Age-for-level
    if "age_rel" in weights:
        score += weights["age_rel"] * df["comp_age"]

    # Trajectory
    if "trajectory" in weights:
        score += weights["trajectory"] * df["comp_trajectory"]

    # Positional scarcity
    if "positional" in weights:
        score += weights["positional"] * df["comp_positional"]

    return score


# ===================================================================
# FV grade lookup — used for FV-conditioned translation adjustments
# ===================================================================

def _get_prospect_fv_map(player_ids: set[int]) -> dict[int, float]:
    """Look up most recent FanGraphs Future Value grade for each player.

    Returns ``{player_id: fv_grade}`` for players that have FG rankings.
    Players without FG data are absent from the dict (will default to
    population-level translation, i.e. no FV adjustment).
    """
    if not player_ids:
        return {}
    try:
        from src.data.db import read_sql
        df = read_sql("""
            SELECT DISTINCT ON (player_id)
                   player_id, future_value
            FROM production.dim_prospect_ranking
            WHERE future_value IS NOT NULL
            ORDER BY player_id, season DESC, source
        """, {})
        df = df[df["player_id"].isin(player_ids)]
        return dict(zip(df["player_id"].astype(int), df["future_value"]))
    except Exception:
        logger.warning("Could not fetch FV grades from dim_prospect_ranking")
        return {}


# ===================================================================
# Fix 2: Position guard — exclude pitchers from batter prospect pipeline
# ===================================================================

def _get_pitcher_player_ids() -> set[int]:
    """Get player IDs whose primary position is pitcher.

    Used to exclude converted pitchers (e.g. Nolan McLean) from
    batter prospect rankings when they still have MiLB batting data.
    """
    try:
        from src.data.db import read_sql
        df = read_sql(
            "SELECT player_id FROM production.dim_player "
            "WHERE primary_position = 'P'",
            {},
        )
        return set(df["player_id"].astype(int))
    except Exception:
        logger.warning("Could not query pitcher IDs from dim_player")
        return set()


# ===================================================================
# Fix 3: MLB debut blending for prospects with small MLB samples
# ===================================================================

def _get_mlb_debut_pitcher_rates(
    prospect_ids: set[int],
    max_bf: int = _MAX_MLB_BF_PROSPECT,
    min_bf: int = 20,
) -> pd.DataFrame:
    """Fetch MLB rates for pitching prospects with small debut samples.

    Parameters
    ----------
    prospect_ids : set[int]
        Player IDs to look up.
    max_bf : int
        Upper bound on MLB BF (above this they're excluded from prospects).
    min_bf : int
        Minimum MLB BF to be worth blending (very small samples add noise).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, mlb_bf, mlb_k_pct, mlb_bb_pct, mlb_hr_bf.
        Empty if no matching data found.
    """
    if not prospect_ids:
        return pd.DataFrame()

    try:
        from src.data.db import read_sql
        df = read_sql("""
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
        """, {"min_bf": min_bf, "max_bf": max_bf})
    except Exception:
        logger.warning("Could not fetch MLB debut pitcher rates")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df[df["player_id"].isin(prospect_ids)].copy()
    if df.empty:
        return df

    df["mlb_k_pct"] = df["mlb_k"] / df["mlb_bf"]
    df["mlb_bb_pct"] = df["mlb_bb"] / df["mlb_bf"]
    df["mlb_hr_bf"] = df["mlb_hr"] / df["mlb_bf"]
    return df[["player_id", "mlb_bf", "mlb_k_pct", "mlb_bb_pct", "mlb_hr_bf"]]


def _blend_mlb_debut_pitcher_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Blend actual MLB debut rates into translated MiLB rates for pitching prospects.

    For prospects who have a small MLB sample (20–400 BF), their actual MLB
    performance is blended with their translated MiLB rates using a
    reliability-weighted average.  MLB data counts as 2x per BF since it's
    real MLB performance, not translated.

    Parameters
    ----------
    df : pd.DataFrame
        Pitcher prospect features from ``_build_pitcher_prospect_features()``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with wtd_k_pct, wtd_bb_pct, wtd_hr_bf, k_bb_diff
        updated for prospects who have MLB debut data.
    """
    prospect_ids = set(df["player_id"].unique())
    mlb_rates = _get_mlb_debut_pitcher_rates(prospect_ids)

    if mlb_rates.empty:
        return df

    df = df.merge(mlb_rates, on="player_id", how="left")
    has_mlb = df["mlb_bf"].notna()

    if not has_mlb.any():
        df.drop(columns=["mlb_bf", "mlb_k_pct", "mlb_bb_pct", "mlb_hr_bf"],
                inplace=True)
        return df

    logger.info(
        "Blending MLB debut data for %d pitching prospects (avg %.0f BF)",
        has_mlb.sum(),
        df.loc[has_mlb, "mlb_bf"].mean(),
    )

    mlb_weight = df["mlb_bf"] * _MLB_RELIABILITY_MULTIPLIER
    milb_weight = df["career_milb_bf"] * 0.7  # avg translation confidence
    total_weight = mlb_weight + milb_weight

    for stat_milb, stat_mlb in [
        ("wtd_k_pct", "mlb_k_pct"),
        ("wtd_bb_pct", "mlb_bb_pct"),
        ("wtd_hr_bf", "mlb_hr_bf"),
    ]:
        blended = (
            mlb_weight * df[stat_mlb] + milb_weight * df[stat_milb]
        ) / total_weight
        df.loc[has_mlb, stat_milb] = blended[has_mlb]

    df.loc[has_mlb, "k_bb_diff"] = (
        df.loc[has_mlb, "wtd_k_pct"] - df.loc[has_mlb, "wtd_bb_pct"]
    )

    # Track that this prospect has MLB data (useful for downstream)
    df["has_mlb_debut"] = has_mlb
    df["mlb_debut_bf"] = df["mlb_bf"].fillna(0).astype(int)

    df.drop(columns=["mlb_bf", "mlb_k_pct", "mlb_bb_pct", "mlb_hr_bf"],
            inplace=True)
    return df


def _get_mlb_debut_batter_rates(
    prospect_ids: set[int],
    max_pa: int = _MAX_MLB_PA_PROSPECT,
    min_pa: int = 20,
) -> pd.DataFrame:
    """Fetch MLB rates for batting prospects with small debut samples.

    Parameters
    ----------
    prospect_ids : set[int]
        Player IDs to look up.
    max_pa, min_pa : int
        PA bounds for inclusion.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, mlb_pa, mlb_k_pct, mlb_bb_pct, mlb_iso.
        Empty if no matching data found.
    """
    if not prospect_ids:
        return pd.DataFrame()

    try:
        from src.data.db import read_sql
        df = read_sql("""
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
        """, {"min_pa": min_pa, "max_pa": max_pa})
    except Exception:
        logger.warning("Could not fetch MLB debut batter rates")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df[df["player_id"].isin(prospect_ids)].copy()
    if df.empty:
        return df

    df["mlb_k_pct"] = df["mlb_k"] / df["mlb_pa"]
    df["mlb_bb_pct"] = df["mlb_bb"] / df["mlb_pa"]
    df["mlb_iso"] = (df["mlb_tb"] - df["mlb_h"]) / df["mlb_ab"].replace(0, np.nan)
    df["mlb_hr_pa"] = df["mlb_hr"] / df["mlb_pa"]
    return df[["player_id", "mlb_pa", "mlb_k_pct", "mlb_bb_pct",
               "mlb_iso", "mlb_hr_pa"]]


def _blend_mlb_debut_batter_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Blend actual MLB debut rates into translated MiLB rates for batting prospects.

    Same approach as pitcher blending — MLB data counts as 2x per PA.

    Parameters
    ----------
    df : pd.DataFrame
        Batter prospect features with wtd_ columns and career_milb_pa.

    Returns
    -------
    pd.DataFrame
        Updated with blended rates where MLB debut data exists.
    """
    prospect_ids = set(df["player_id"].unique())
    mlb_rates = _get_mlb_debut_batter_rates(prospect_ids)

    if mlb_rates.empty:
        return df

    df = df.merge(mlb_rates, on="player_id", how="left")
    has_mlb = df["mlb_pa"].notna()

    if not has_mlb.any():
        df.drop(columns=["mlb_pa", "mlb_k_pct", "mlb_bb_pct",
                          "mlb_iso", "mlb_hr_pa"], inplace=True)
        return df

    logger.info(
        "Blending MLB debut data for %d batting prospects (avg %.0f PA)",
        has_mlb.sum(),
        df.loc[has_mlb, "mlb_pa"].mean(),
    )

    mlb_weight = df["mlb_pa"] * _MLB_RELIABILITY_MULTIPLIER
    milb_weight = df["career_milb_pa"] * 0.7
    total_weight = mlb_weight + milb_weight

    for stat_milb, stat_mlb in [
        ("wtd_k_pct", "mlb_k_pct"),
        ("wtd_bb_pct", "mlb_bb_pct"),
        ("wtd_iso", "mlb_iso"),
        ("wtd_hr_pa", "mlb_hr_pa"),
    ]:
        if stat_milb in df.columns:
            blended = (
                mlb_weight * df[stat_mlb] + milb_weight * df[stat_milb]
            ) / total_weight
            df.loc[has_mlb, stat_milb] = blended[has_mlb]

    if "k_bb_diff" in df.columns:
        df.loc[has_mlb, "k_bb_diff"] = (
            df.loc[has_mlb, "wtd_k_pct"] - df.loc[has_mlb, "wtd_bb_pct"]
        )

    df["has_mlb_debut"] = has_mlb
    df["mlb_debut_pa"] = df["mlb_pa"].fillna(0).astype(int)

    df.drop(columns=["mlb_pa", "mlb_k_pct", "mlb_bb_pct",
                      "mlb_iso", "mlb_hr_pa"], inplace=True)
    return df


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

        # Recency decay: recent seasons contribute more than stale data.
        # A developing pitcher's 2025 AA dominance is more informative
        # than their 2023 A-ball numbers.
        max_season = grp["season"].max()
        recency = np.exp(
            -np.log(2) * (max_season - grp["season"]) / _RECENCY_HALF_LIFE
        )

        conf_bf = conf * grp["bf"] * recency
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

    # No FV-conditioned adjustments — translations stand on their own stats.

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
    # Fix 2: Exclude pitchers from batter prospect rankings
    # -------------------------------------------------------------------
    # Converted players (e.g. McLean) may still have MiLB batting data
    # but their primary position is now P.
    pitcher_ids = _get_pitcher_player_ids()
    n_before = len(df)
    df = df[~df["player_id"].isin(pitcher_ids)].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Excluded %d pitchers from batter prospect rankings", n_dropped)

    if df.empty:
        logger.warning("No batter prospects remain after pitcher exclusion")
        return pd.DataFrame()

    # -------------------------------------------------------------------
    # Fix 3: Blend MLB debut data for prospects with small MLB samples
    # -------------------------------------------------------------------
    df = _blend_mlb_debut_batter_rates(df)

    # -------------------------------------------------------------------
    # Readiness (kept for display / future callup prediction, NOT scored)
    # -------------------------------------------------------------------
    df["comp_readiness"] = df["readiness_score"]

    # -------------------------------------------------------------------
    # Component 1: pwOBA quality (BABIP-stripped offensive production)
    # -------------------------------------------------------------------
    df["pwoba"] = _compute_pwoba(df)
    df["comp_pwoba"] = df["pwoba"].rank(pct=True, method="average")

    # -------------------------------------------------------------------
    # Component 2: Rate quality / profile balance
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
    # Component 6: Diamond Rating (internal tool grades, position-aware)
    # Must be computed BEFORE composite so it feeds into scoring.
    # -------------------------------------------------------------------
    try:
        from src.models.scouting_grades import grade_prospect_hitter
        prospect_grades = grade_prospect_hitter(df, season=projection_season - 1)
        if not prospect_grades.empty:
            df = df.merge(prospect_grades, on="player_id", how="left")
            logger.info("Scouting grades computed for %d batting prospects",
                        prospect_grades["tools_rating"].notna().sum())
    except Exception:
        logger.warning("Could not compute prospect scouting grades", exc_info=True)

    # Convert Diamond Rating (0-10) to 0-1 component score
    if "tools_rating" in df.columns:
        df["comp_diamond"] = (df["tools_rating"].fillna(5.0) / 10.0).clip(0, 1)
    else:
        df["comp_diamond"] = 0.50

    # -------------------------------------------------------------------
    # Merge FanGraphs FV for display only (NOT used in scoring)
    # -------------------------------------------------------------------
    fg = _load_fg_rankings(projection_season)
    if not fg.empty:
        df = df.merge(fg, on="player_id", how="left")
        logger.info("Merged FG rankings for %d prospects (display only)",
                     df["fg_future_value"].notna().sum())
    else:
        df["fg_future_value"] = np.nan
        df["fg_overall_rank"] = np.nan
        df["fg_org_rank"] = np.nan
        df["fg_risk"] = np.nan
        df["fg_eta"] = np.nan

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
        bins=[0, 0.35, 0.50, 0.65, 0.78, 1.0],
        labels=["Org Filler", "Developing", "Solid", "Impact", "Elite"],
    )

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
        # Component scores
        "comp_pwoba", "pwoba", "comp_diamond",
        "comp_rate_quality", "comp_age",
        "comp_trajectory", "comp_positional",
        "comp_readiness",
        # Translated stats (blended with MLB debut data when available)
        "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "wtd_hr_pa", "wtd_gb_pct",
        "k_bb_diff", "sb_rate",
        "career_milb_pa", "youngest_age_rel",
        # MLB debut blend metadata
        "has_mlb_debut", "mlb_debut_pa",
        # Sub-trajectory components
        "promotion_resilience", "availability_score",
        # Readiness
        "readiness_score", "readiness_tier",
        # Org depth
        "n_above", "total_at_pos_in_org",
        # FanGraphs (display only)
        "fg_future_value", "fg_overall_rank", "fg_org_rank",
        "fg_risk", "fg_eta",
        # Scouting grades (20-80) — present + future + diamond rating
        "grade_hit", "grade_power", "grade_speed", "grade_fielding",
        "grade_discipline", "tools_rating",
        "future_hit", "future_power", "future_speed", "future_fielding",
        "future_discipline",
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
    # Fix 1: Raised from 100 BF to _MAX_MLB_BF_PROSPECT (400 BF ≈ 100 IP).
    # Prospects with brief callups (e.g. McLean's 8-start debut) should
    # remain in prospect rankings.
    try:
        from src.data.db import read_sql
        mlb_pitchers = read_sql(f"""
            SELECT DISTINCT pitcher_id FROM production.fact_pitching_advanced
            WHERE batters_faced >= {_MAX_MLB_BF_PROSPECT}
        """, {})
        mlb_ids = set(mlb_pitchers["pitcher_id"].astype(int))
    except Exception:
        mlb_ids = set()

    n_before = len(milb_recent["player_id"].unique())
    milb_recent = milb_recent[~milb_recent["player_id"].isin(mlb_ids)]
    n_after = len(milb_recent["player_id"].unique())
    if n_before > n_after:
        logger.info("Excluded %d pitchers with >= %d MLB BF from prospect pool",
                     n_before - n_after, _MAX_MLB_BF_PROSPECT)

    # No FV-conditioned adjustments — translations stand on their own stats.

    # Build features
    df = _build_pitcher_prospect_features(milb_recent)
    if df.empty:
        logger.warning("No pitching prospects found")
        return pd.DataFrame()

    # Fix 3: Blend MLB debut data for prospects with small samples
    df = _blend_mlb_debut_pitcher_rates(df)

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
    # Component 6: Diamond Rating (internal pitcher tool grades)
    # -------------------------------------------------------------------
    try:
        from src.models.scouting_grades import grade_prospect_pitcher
        prospect_grades = grade_prospect_pitcher(df, season=projection_season - 1)
        if not prospect_grades.empty:
            df = df.merge(prospect_grades, on="player_id", how="left")
            logger.info("Scouting grades computed for %d pitching prospects",
                        prospect_grades["tools_rating"].notna().sum())
    except Exception:
        logger.warning("Could not compute pitching prospect scouting grades", exc_info=True)

    if "tools_rating" in df.columns:
        df["comp_diamond"] = (df["tools_rating"].fillna(5.0) / 10.0).clip(0, 1)
    else:
        df["comp_diamond"] = 0.50

    # -------------------------------------------------------------------
    # Merge FanGraphs FV for display only (NOT used in scoring)
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
    # Composite scores: pitcher-specific weights
    # -------------------------------------------------------------------
    p_weights = weights if "pitcher_quality" in weights else _PITCHER_WEIGHTS
    df["tdd_prospect_score"] = _compute_composite_scores(df, p_weights)
    df["impact_score"] = _compute_composite_scores(df, _PITCHER_IMPACT_WEIGHTS)
    df["floor_eta_score"] = _compute_composite_scores(df, _PITCHER_FLOOR_WEIGHTS)

    # Rank
    df = df.sort_values("tdd_prospect_score", ascending=False).reset_index(drop=True)
    df["tdd_rank"] = range(1, len(df) + 1)

    # Tier labels
    df["tdd_tier"] = pd.cut(
        df["tdd_prospect_score"],
        bins=[0, 0.35, 0.50, 0.65, 0.78, 1.0],
        labels=["Org Filler", "Developing", "Solid", "Impact", "Elite"],
    )

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
        # Component scores
        "comp_diamond", "comp_rate_quality", "comp_age",
        "comp_trajectory", "comp_positional",
        "comp_readiness",
        # Translated stats (blended with MLB debut data when available)
        "wtd_k_pct", "wtd_bb_pct", "wtd_hr_bf", "k_bb_diff",
        "career_milb_bf", "youngest_age_rel", "sp_pct",
        # MLB debut blend metadata
        "has_mlb_debut", "mlb_debut_bf",
        # Readiness
        "readiness_score", "readiness_tier",
        # FanGraphs (display only)
        "fg_future_value", "fg_overall_rank", "fg_org_rank",
        "fg_risk", "fg_eta",
        # Scouting grades (20-80) — present + future + diamond rating
        "grade_stuff", "grade_command", "grade_durability", "tools_rating",
        "future_stuff", "future_command", "future_durability",
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
