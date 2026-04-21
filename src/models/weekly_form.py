"""Weekly form leaderboards (short-horizon, separate from core rankings)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.queries import get_hitter_recent_form, get_pitcher_recent_form
from src.models._ranking_utils import _inv_zscore_pctl, _pctl, _zscore_pctl

logger = logging.getLogger(__name__)


# Dashboard-facing schemas: columns the "The Diamond Daily" page reads directly.
# Keep in sync with views/diamond_daily.py in tdd-dashboard.
_HITTER_WEEKLY_EMPTY_COLUMNS = [
    "batter_id", "batter_name", "games_14d", "pa_14d",
    "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
    "weekly_contact", "weekly_discipline", "weekly_damage",
    "weekly_production", "weekly_opportunity",
    "hits_14d", "hr_14d",
    "k_rate_14d", "bb_rate_14d", "woba_14d", "xwoba_14d",
    "ops_14d", "iso_14d", "hard_hit_pct_14d", "barrel_pct_14d", "bip_14d",
]

_PITCHER_WEEKLY_EMPTY_COLUMNS = [
    "pitcher_id", "pitcher_name", "role_14d", "games_14d", "starts_14d", "bf_14d",
    "weekly_form_score", "weekly_form_rank", "weekly_form_reliability",
    "weekly_kbb", "weekly_run_prevention", "weekly_contact_suppression",
    "weekly_opportunity",
    "k_rate_14d", "bb_rate_14d", "k_minus_bb_14d", "era_14d",
    "whip_14d", "hr_per_9_14d", "sv_14d", "hld_14d",
]


def compute_hitter_weekly_form(
    recent_df: pd.DataFrame,
    min_pa: int = 15,
    full_pa: int = 80,
) -> pd.DataFrame:
    """Compute hitter weekly-form scores from recent-window aggregates."""
    if recent_df.empty:
        return pd.DataFrame(columns=_HITTER_WEEKLY_EMPTY_COLUMNS)

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
        "hits_14d", "hr_14d",
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
        return pd.DataFrame(columns=_PITCHER_WEEKLY_EMPTY_COLUMNS)

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


def _merge_live_hitter_form(
    db_recent: pd.DataFrame,
    live_today: pd.DataFrame,
) -> pd.DataFrame:
    """Merge today's live hitter boxscores into the DB rolling-window aggregate."""
    if live_today.empty:
        return db_recent

    agg = live_today.groupby("batter_id").agg(
        batter_name=("batter_name", "first"),
        games_today=("game_pk", "nunique"),
        pa_today=("pa", "sum"),
        ab_today=("ab", "sum"),
        hits_today=("hits", "sum"),
        doubles_today=("doubles", "sum"),
        triples_today=("triples", "sum"),
        hr_today=("hr", "sum"),
        total_bases_today=("total_bases", "sum"),
        bb_today=("bb", "sum"),
        ibb_today=("ibb", "sum"),
        hbp_today=("hbp", "sum"),
        k_today=("k", "sum"),
        runs_today=("runs", "sum"),
        rbi_today=("rbi", "sum"),
        sb_today=("sb", "sum"),
        cs_today=("cs", "sum"),
        sf_today=("sf", "sum"),
    ).reset_index()

    if db_recent.empty:
        merged = agg.rename(columns={c: c.replace("_today", "_14d") for c in agg.columns if c.endswith("_today")})
        merged = merged.rename(columns={"games_today": "games_14d"})
    else:
        merged = db_recent.merge(agg, on="batter_id", how="outer", suffixes=("", "_live"))
        if "batter_name_live" in merged.columns:
            merged["batter_name"] = merged["batter_name"].fillna(merged["batter_name_live"])
            merged.drop(columns=["batter_name_live"], inplace=True)

        _sum_cols = [
            ("games_14d", "games_today"), ("pa_14d", "pa_today"),
            ("ab_14d", "ab_today"), ("hits_14d", "hits_today"),
            ("doubles_14d", "doubles_today"), ("triples_14d", "triples_today"),
            ("hr_14d", "hr_today"), ("bb_14d", "bb_today"),
            ("ibb_14d", "ibb_today"), ("hbp_14d", "hbp_today"),
            ("k_14d", "k_today"), ("runs_14d", "runs_today"),
            ("rbi_14d", "rbi_today"), ("sb_14d", "sb_today"),
            ("cs_14d", "cs_today"), ("sf_14d", "sf_today"),
        ]
        if "total_bases_14d" in merged.columns:
            _sum_cols.append(("total_bases_14d", "total_bases_today"))

        for db_col, live_col in _sum_cols:
            if live_col in merged.columns:
                merged[db_col] = merged[db_col].fillna(0) + merged[live_col].fillna(0)
                merged.drop(columns=[live_col], inplace=True)

    # Recompute rates
    pa = merged["pa_14d"].replace(0, float("nan"))
    ab = merged.get("ab_14d", merged["pa_14d"]).replace(0, float("nan"))
    non_ibb_bb = (merged["bb_14d"] - merged.get("ibb_14d", 0)).clip(lower=0)

    merged["k_rate_14d"] = (merged["k_14d"] / pa).fillna(0.0)
    merged["bb_rate_14d"] = (merged["bb_14d"] / pa).fillna(0.0)
    if "total_bases_14d" in merged.columns:
        merged["slg_14d"] = (merged["total_bases_14d"] / ab).fillna(0.0)
    merged["avg_14d"] = (merged["hits_14d"] / ab).fillna(0.0)
    if "slg_14d" in merged.columns:
        merged["iso_14d"] = (merged["slg_14d"] - merged["avg_14d"]).fillna(0.0)

    obp_num = merged["hits_14d"] + merged["bb_14d"] + merged.get("hbp_14d", 0)
    obp_den = (ab + merged["bb_14d"] + merged.get("hbp_14d", 0) + merged.get("sf_14d", 0)).replace(0, float("nan"))
    merged["obp_14d"] = (obp_num / obp_den).fillna(0.0)
    merged["ops_14d"] = merged["obp_14d"] + merged.get("slg_14d", 0.0)

    from src.data.queries._common import _WOBA_WEIGHTS
    singles = merged["hits_14d"] - merged.get("doubles_14d", 0) - merged.get("triples_14d", 0) - merged["hr_14d"]
    woba_num = (
        non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + merged.get("hbp_14d", 0) * _WOBA_WEIGHTS["hbp"]
        + singles * _WOBA_WEIGHTS["single"]
        + merged.get("doubles_14d", 0) * _WOBA_WEIGHTS["double"]
        + merged.get("triples_14d", 0) * _WOBA_WEIGHTS["triple"]
        + merged["hr_14d"] * _WOBA_WEIGHTS["hr"]
    )
    woba_den = (ab + non_ibb_bb + merged.get("sf_14d", 0) + merged.get("hbp_14d", 0)).replace(0, float("nan"))
    merged["woba_14d"] = (woba_num / woba_den).fillna(0.0)

    # Statcast columns: keep DB values where available, NaN for live-only players
    for c in ("xwoba_14d", "hard_hit_pct_14d", "barrel_pct_14d", "bip_14d"):
        if c not in merged.columns:
            merged[c] = 0.0 if "bip" in c else np.nan

    return merged


def _merge_live_pitcher_form(
    db_recent: pd.DataFrame,
    live_today: pd.DataFrame,
) -> pd.DataFrame:
    """Merge today's live pitcher boxscores into the DB rolling-window aggregate."""
    if live_today.empty:
        return db_recent

    agg = live_today.groupby("pitcher_id").agg(
        pitcher_name=("pitcher_name", "first"),
        games_today=("game_pk", "nunique"),
        starts_today=("is_starter", "sum"),
        bf_today=("bf", "sum"),
        ip_today=("ip", "sum"),
        k_today=("k", "sum"),
        bb_today=("bb", "sum"),
        hr_today=("hr", "sum"),
        hits_today=("hits", "sum"),
        er_today=("er", "sum"),
        sv_today=("sv", "sum"),
        hld_today=("hld", "sum"),
    ).reset_index()

    if db_recent.empty:
        merged = agg.rename(columns={c: c.replace("_today", "_14d") for c in agg.columns if c.endswith("_today")})
        merged = merged.rename(columns={"games_today": "games_14d"})
    else:
        merged = db_recent.merge(agg, on="pitcher_id", how="outer", suffixes=("", "_live"))
        if "pitcher_name_live" in merged.columns:
            merged["pitcher_name"] = merged["pitcher_name"].fillna(merged["pitcher_name_live"])
            merged.drop(columns=["pitcher_name_live"], inplace=True)

        _sum_cols = [
            ("games_14d", "games_today"), ("starts_14d", "starts_today"),
            ("bf_14d", "bf_today"), ("ip_14d", "ip_today"),
            ("k_14d", "k_today"), ("bb_14d", "bb_today"),
            ("hr_14d", "hr_today"), ("hits_14d", "hits_today"),
            ("er_14d", "er_today"), ("sv_14d", "sv_today"),
            ("hld_14d", "hld_today"),
        ]
        for db_col, live_col in _sum_cols:
            if live_col in merged.columns:
                merged[db_col] = merged[db_col].fillna(0) + merged[live_col].fillna(0)
                merged.drop(columns=[live_col], inplace=True)

    # Recompute rates
    bf = merged["bf_14d"].replace(0, float("nan"))
    ip = merged["ip_14d"].replace(0, float("nan"))
    merged["k_rate_14d"] = (merged["k_14d"] / bf).fillna(0.0)
    merged["bb_rate_14d"] = (merged["bb_14d"] / bf).fillna(0.0)
    merged["k_minus_bb_14d"] = merged["k_rate_14d"] - merged["bb_rate_14d"]
    merged["era_14d"] = ((merged["er_14d"] / ip) * 9).fillna(0.0)
    merged["whip_14d"] = ((merged["hits_14d"] + merged["bb_14d"]) / ip).fillna(0.0)
    merged["hr_per_9_14d"] = ((merged["hr_14d"] / ip) * 9).fillna(0.0)
    merged["role_14d"] = np.where(merged["starts_14d"].fillna(0) >= 1, "SP", "RP")

    return merged


def build_weekly_form_boards(
    *,
    days: int = 14,
    as_of_date: str | None = None,
    core_hitters: pd.DataFrame | None = None,
    core_pitchers: pd.DataFrame | None = None,
    live_hitters: pd.DataFrame | None = None,
    live_pitchers: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build hitter/pitcher weekly form boards and optional core deltas.

    Parameters
    ----------
    live_hitters, live_pitchers : pd.DataFrame or None
        Today's completed-game boxscores from the API. When provided,
        the DB query covers ``days - 1`` through yesterday, and live
        data is merged in for the full window.
    """
    if live_hitters is not None or live_pitchers is not None:
        from datetime import date as _date, timedelta
        yesterday = (as_of_date or _date.today().isoformat())
        if as_of_date is None:
            yesterday = (_date.today() - timedelta(days=1)).isoformat()
        hitter_recent = get_hitter_recent_form(days=days - 1, as_of_date=yesterday)
        pitcher_recent = get_pitcher_recent_form(days=days - 1, as_of_date=yesterday)
        hitter_recent = _merge_live_hitter_form(
            hitter_recent, live_hitters if live_hitters is not None else pd.DataFrame(),
        )
        pitcher_recent = _merge_live_pitcher_form(
            pitcher_recent, live_pitchers if live_pitchers is not None else pd.DataFrame(),
        )
    else:
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
