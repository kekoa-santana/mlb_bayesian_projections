"""Weekly form leaderboards (short-horizon, separate from core rankings)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.queries import get_hitter_recent_form, get_pitcher_recent_form
from src.models._ranking_utils import _inv_zscore_pctl, _pctl, _zscore_pctl

logger = logging.getLogger(__name__)


def compute_hitter_weekly_form(
    recent_df: pd.DataFrame,
    min_pa: int = 15,
    full_pa: int = 80,
) -> pd.DataFrame:
    """Compute hitter weekly-form scores from recent-window aggregates."""
    if recent_df.empty:
        return pd.DataFrame(
            columns=[
                "batter_id", "batter_name", "games_14d", "pa_14d",
                "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
                "weekly_contact", "weekly_discipline", "weekly_damage",
                "weekly_production", "weekly_opportunity",
            ],
        )

    df = recent_df.copy()
    pa = df["pa_14d"].fillna(0)
    reliability = ((pa - min_pa) / (full_pa - min_pa)).clip(0, 1)

    contact = _inv_zscore_pctl(df["k_rate_14d"].fillna(df["k_rate_14d"].median()))
    discipline = _zscore_pctl(df["bb_rate_14d"].fillna(df["bb_rate_14d"].median()))
    damage = (
        0.50 * _zscore_pctl(df["xwoba_14d"].fillna(df["woba_14d"].median()))
        + 0.30 * _zscore_pctl(df["hard_hit_pct_14d"].fillna(0))
        + 0.20 * _zscore_pctl(df["barrel_pct_14d"].fillna(0))
    )
    production = (
        0.60 * _zscore_pctl(df["woba_14d"].fillna(df["woba_14d"].median()))
        + 0.40 * _zscore_pctl(df["ops_14d"].fillna(df["ops_14d"].median()))
    )
    opportunity = _pctl(pa)

    raw = (
        0.25 * contact
        + 0.10 * discipline
        + 0.35 * damage
        + 0.25 * production
        + 0.05 * opportunity
    )
    score = reliability * raw + (1 - reliability) * 0.50

    df["weekly_contact"] = contact
    df["weekly_discipline"] = discipline
    df["weekly_damage"] = damage
    df["weekly_production"] = production
    df["weekly_opportunity"] = opportunity
    df["weekly_form_reliability"] = reliability
    df["weekly_form_score"] = score
    df["weekly_form_rank"] = (
        df["weekly_form_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    cols = [
        "batter_id", "batter_name", "games_14d", "pa_14d",
        "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
        "weekly_contact", "weekly_discipline", "weekly_damage",
        "weekly_production", "weekly_opportunity",
        "k_rate_14d", "bb_rate_14d", "woba_14d", "xwoba_14d",
        "ops_14d", "iso_14d", "hard_hit_pct_14d", "barrel_pct_14d", "bip_14d",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("weekly_form_rank")


def compute_pitcher_weekly_form(
    recent_df: pd.DataFrame,
    min_bf: int = 20,
    full_bf: int = 100,
) -> pd.DataFrame:
    """Compute pitcher weekly-form scores from recent-window aggregates."""
    if recent_df.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id", "pitcher_name", "role_14d", "games_14d", "bf_14d",
                "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
                "weekly_kbb", "weekly_run_prevention", "weekly_contact_suppression",
                "weekly_opportunity",
            ],
        )

    df = recent_df.copy()
    bf = df["bf_14d"].fillna(0)
    reliability = ((bf - min_bf) / (full_bf - min_bf)).clip(0, 1)

    kbb = _zscore_pctl(df["k_minus_bb_14d"].fillna(df["k_minus_bb_14d"].median()))
    run_prevention = (
        0.60 * _inv_zscore_pctl(df["era_14d"].replace(0, np.nan).fillna(df["era_14d"].median()))
        + 0.40 * _inv_zscore_pctl(df["whip_14d"].replace(0, np.nan).fillna(df["whip_14d"].median()))
    )
    contact_supp = _inv_zscore_pctl(df["hr_per_9_14d"].replace(0, np.nan).fillna(df["hr_per_9_14d"].median()))
    role_usage = _pctl(
        3 * df["starts_14d"].fillna(0) + df["sv_14d"].fillna(0) + df["hld_14d"].fillna(0),
    )

    raw = 0.45 * kbb + 0.30 * run_prevention + 0.15 * contact_supp + 0.10 * role_usage
    score = reliability * raw + (1 - reliability) * 0.50

    df["weekly_kbb"] = kbb
    df["weekly_run_prevention"] = run_prevention
    df["weekly_contact_suppression"] = contact_supp
    df["weekly_opportunity"] = role_usage
    df["weekly_form_reliability"] = reliability
    df["weekly_form_score"] = score

    # Rank within role to keep SP/RP comparable for dashboard usage.
    df["weekly_form_rank"] = (
        df.groupby("role_14d")["weekly_form_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    cols = [
        "pitcher_id", "pitcher_name", "role_14d", "games_14d", "starts_14d", "bf_14d",
        "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
        "weekly_kbb", "weekly_run_prevention", "weekly_contact_suppression",
        "weekly_opportunity",
        "k_rate_14d", "bb_rate_14d", "k_minus_bb_14d", "era_14d",
        "whip_14d", "hr_per_9_14d", "sv_14d", "hld_14d",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["role_14d", "weekly_form_rank"])


def build_weekly_form_boards(
    *,
    days: int = 14,
    as_of_date: str | None = None,
    core_hitters: pd.DataFrame | None = None,
    core_pitchers: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build hitter/pitcher weekly form boards and optional core deltas."""
    hitter_recent = get_hitter_recent_form(days=days, as_of_date=as_of_date)
    pitcher_recent = get_pitcher_recent_form(days=days, as_of_date=as_of_date)

    hitters = compute_hitter_weekly_form(hitter_recent)
    pitchers = compute_pitcher_weekly_form(pitcher_recent)

    if core_hitters is not None and not core_hitters.empty and not hitters.empty:
        c = core_hitters[["batter_id", "overall_rank"]].drop_duplicates("batter_id")
        c = c.rename(columns={"overall_rank": "core_rank"})
        hitters = hitters.merge(c, on="batter_id", how="left")
        hitters["delta_vs_core"] = hitters["core_rank"] - hitters["weekly_form_rank"]

    if core_pitchers is not None and not core_pitchers.empty and not pitchers.empty:
        c = core_pitchers[["pitcher_id", "overall_rank"]].drop_duplicates("pitcher_id")
        c = c.rename(columns={"overall_rank": "core_rank"})
        pitchers = pitchers.merge(c, on="pitcher_id", how="left")
        pitchers["delta_vs_core"] = pitchers["core_rank"] - pitchers["weekly_form_rank"]

    logger.info(
        "Weekly form boards built: %d hitters, %d pitchers",
        len(hitters), len(pitchers),
    )
    return {"hitters": hitters, "pitchers": pitchers}
