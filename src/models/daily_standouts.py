"""Daily standout leaderboards (single-game box-score celebrations)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.queries import get_hitter_daily_standouts, get_pitcher_daily_standouts
from src.models._ranking_utils import _inv_zscore_pctl, _pctl, _zscore_pctl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hitter daily standouts
# ---------------------------------------------------------------------------
def score_hitter_daily(
    daily_df: pd.DataFrame,
    min_pa: int = 3,
) -> pd.DataFrame:
    """Score single-game hitter performances.

    Weights
    -------
    45% production (wOBA + OPS)
    30% counting impact (hits, HR, RBI, R, SB)
    15% extra-base power (ISO + HR)
    10% plate discipline (BB rate, low K)

    Statcast quality (xwoba, barrel%, hard-hit%) is included as bonus
    columns for display but does NOT drive the score.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of ``get_hitter_daily_standouts()``.
    min_pa : int
        Minimum PA to qualify.  Pinch-hit 1-PA games are excluded.
    """
    if daily_df.empty:
        return pd.DataFrame(
            columns=[
                "batter_id", "batter_name", "game_pk", "game_date",
                "pa", "daily_standout_score", "daily_standout_rank",
            ],
        )

    df = daily_df.loc[daily_df["pa"] >= min_pa].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "batter_id", "batter_name", "game_pk", "game_date",
                "pa", "daily_standout_score", "daily_standout_rank",
            ],
        )

    # --- Production (45%): the offensive bottom line ---
    production = (
        0.60 * _zscore_pctl(df["woba"].fillna(0))
        + 0.40 * _zscore_pctl(df["ops"].fillna(0))
    )

    # --- Counting impact (30%): showed up in the box score ---
    counting = (
        0.30 * _pctl(df["hits"].fillna(0))
        + 0.25 * _pctl(df["hr"].fillna(0))
        + 0.20 * _pctl(df["rbi"].fillna(0))
        + 0.15 * _pctl(df["runs"].fillna(0))
        + 0.10 * _pctl(df["sb"].fillna(0))
    )

    # --- Extra-base power (15%): the wow plays ---
    power = (
        0.60 * _zscore_pctl(df["iso"].fillna(0))
        + 0.40 * _pctl(df["hr"].fillna(0))
    )

    # --- Plate discipline (10%): controlled the at-bats ---
    discipline = (
        0.50 * _zscore_pctl(df["bb_rate"].fillna(0))
        + 0.50 * _inv_zscore_pctl(df["k_rate"].fillna(df["k_rate"].median()))
    )

    df["daily_production"] = production
    df["daily_counting"] = counting
    df["daily_power"] = power
    df["daily_discipline"] = discipline

    df["daily_standout_score"] = (
        0.45 * production
        + 0.30 * counting
        + 0.15 * power
        + 0.10 * discipline
    )

    df["daily_standout_rank"] = (
        df["daily_standout_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    # Build a readable game label
    if "away_team" in df.columns and "home_team" in df.columns:
        df["game_label"] = df["away_team"].fillna("") + " @ " + df["home_team"].fillna("")

    cols = [
        "batter_id", "batter_name", "game_pk", "game_date",
        "game_label",
        "pa", "ab", "hits", "doubles", "triples", "hr",
        "runs", "rbi", "bb", "k", "sb",
        "avg", "obp", "slg", "ops", "iso", "woba",
        "daily_standout_score", "daily_standout_rank",
        "daily_production", "daily_counting", "daily_power", "daily_discipline",
        # Statcast bonus columns (display only)
        "xwoba", "hard_hit_pct", "barrel_pct", "bip",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("daily_standout_rank")


# ---------------------------------------------------------------------------
# Pitcher daily standouts
# ---------------------------------------------------------------------------
def score_pitcher_daily(
    daily_df: pd.DataFrame,
    min_bf: int = 6,
) -> pd.DataFrame:
    """Score single-game pitcher performances.

    Weights
    -------
    35% run prevention (ERA + WHIP)
    30% K dominance (raw K count + K rate)
    20% innings depth (IP -- going deep matters)
    15% win contribution (QS, W, SV, HLD)

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of ``get_pitcher_daily_standouts()``.
    min_bf : int
        Minimum BF to qualify.  Removes 1-batter mop-up appearances.
    """
    if daily_df.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id", "pitcher_name", "role", "game_pk", "game_date",
                "bf", "daily_standout_score", "daily_standout_rank",
            ],
        )

    df = daily_df.loc[daily_df["bf"] >= min_bf].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id", "pitcher_name", "role", "game_pk", "game_date",
                "bf", "daily_standout_score", "daily_standout_rank",
            ],
        )

    # --- Run prevention (35%): kept runs off the board ---
    era_safe = df["era"].replace(0, np.nan).fillna(df["era"].median())
    whip_safe = df["whip"].replace(0, np.nan).fillna(df["whip"].median())
    run_prevention = (
        0.60 * _inv_zscore_pctl(era_safe)
        + 0.40 * _inv_zscore_pctl(whip_safe)
    )

    # --- K dominance (30%): raw Ks + rate ---
    k_dominance = (
        0.55 * _pctl(df["k"].fillna(0))
        + 0.45 * _zscore_pctl(df["k_rate"].fillna(0))
    )

    # --- Innings depth (20%): went deep into the game ---
    innings_depth = _pctl(df["ip"].fillna(0))

    # --- Win contribution (15%): impacted the outcome ---
    # QS (6+ IP, <= 3 ER), W, SV, HLD — binary bonuses scored via pctl
    win_events = (
        df["qs"].fillna(0) * 3
        + df["w"].fillna(0) * 2
        + df["sv"].fillna(0) * 2
        + df["hld"].fillna(0) * 1
    )
    win_contribution = _pctl(win_events)

    df["daily_run_prevention"] = run_prevention
    df["daily_k_dominance"] = k_dominance
    df["daily_innings_depth"] = innings_depth
    df["daily_win_contribution"] = win_contribution

    df["daily_standout_score"] = (
        0.35 * run_prevention
        + 0.30 * k_dominance
        + 0.20 * innings_depth
        + 0.15 * win_contribution
    )

    # Rank within role so SP and RP each have their own leaderboard
    df["daily_standout_rank"] = (
        df.groupby("role")["daily_standout_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    if "away_team" in df.columns and "home_team" in df.columns:
        df["game_label"] = df["away_team"].fillna("") + " @ " + df["home_team"].fillna("")

    cols = [
        "pitcher_id", "pitcher_name", "role", "game_pk", "game_date",
        "game_label",
        "bf", "ip", "k", "bb", "hr", "hits", "er", "runs",
        "w", "l", "sv", "hld", "qs", "pitches",
        "k_rate", "bb_rate", "k_minus_bb", "era", "whip",
        "daily_standout_score", "daily_standout_rank",
        "daily_run_prevention", "daily_k_dominance",
        "daily_innings_depth", "daily_win_contribution",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["role", "daily_standout_rank"])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_daily_standout_boards(
    *,
    game_date: str | None = None,
    core_hitters: pd.DataFrame | None = None,
    core_pitchers: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build hitter + pitcher daily standout leaderboards.

    Optionally merges core (power) rankings to show ``delta_vs_core``
    so the dashboard can highlight unexpected standouts.
    """
    hitter_raw = get_hitter_daily_standouts(game_date=game_date)
    pitcher_raw = get_pitcher_daily_standouts(game_date=game_date)

    hitters = score_hitter_daily(hitter_raw)
    pitchers = score_pitcher_daily(pitcher_raw)

    if core_hitters is not None and not core_hitters.empty and not hitters.empty:
        c = core_hitters[["batter_id", "overall_rank"]].drop_duplicates("batter_id")
        c = c.rename(columns={"overall_rank": "core_rank"})
        hitters = hitters.merge(c, on="batter_id", how="left")
        hitters["delta_vs_core"] = hitters["core_rank"] - hitters["daily_standout_rank"]

    if core_pitchers is not None and not core_pitchers.empty and not pitchers.empty:
        c = core_pitchers[["pitcher_id", "overall_rank"]].drop_duplicates("pitcher_id")
        c = c.rename(columns={"overall_rank": "core_rank"})
        pitchers = pitchers.merge(c, on="pitcher_id", how="left")
        pitchers["delta_vs_core"] = pitchers["core_rank"] - pitchers["daily_standout_rank"]

    logger.info(
        "Daily standout boards built: %d hitters, %d pitchers",
        len(hitters), len(pitchers),
    )
    return {"hitters": hitters, "pitchers": pitchers}
