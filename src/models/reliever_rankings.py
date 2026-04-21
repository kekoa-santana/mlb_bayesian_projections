"""
Reliever rankings with role-specific composite weights.

Ranks CL, SU, and MR relievers by blending stuff quality, command,
role value (save/hold volume), stability, workload, and trajectory.
Uses sim-based projections for counting stats and fantasy scoring.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import dashboard_dir

logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()

# ---------------------------------------------------------------------------
# Role-specific composite weights (sum to 1.0 per role)
# ---------------------------------------------------------------------------
_CL_WEIGHTS = {
    "stuff": 0.30,
    "role_value": 0.25,
    "stability": 0.20,
    "command": 0.15,
    "trajectory": 0.10,
}
_SU_WEIGHTS = {
    "stuff": 0.35,
    "role_value": 0.20,
    "command": 0.20,
    "workload": 0.15,
    "trajectory": 0.10,
}
_MR_WEIGHTS = {
    "stuff": 0.30,
    "command": 0.25,
    "workload": 0.25,
    "trajectory": 0.20,
}


def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = better)."""
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    """Inverse percentile (lower raw = higher score)."""
    return 1.0 - series.rank(pct=True, method="average")


def rank_relievers(
    sim_df: pd.DataFrame,
    roles_df: pd.DataFrame,
    season: int = 2025,
) -> pd.DataFrame:
    """Rank relievers by role with composite scores.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Output of ``season_results_to_dataframe`` — sim projections.
    roles_df : pd.DataFrame
        Output of ``classify_reliever_roles`` — pitcher_id, role, etc.
    season : int
        Most recent observed season.

    Returns
    -------
    pd.DataFrame
        Reliever rankings with role_rank, overall_rp_rank, sub-scores,
        and projected stats.
    """
    from src.data.db import read_sql

    # Filter sim_df to relievers only
    rp_sims = sim_df[sim_df["role"].isin(["CL", "SU", "MR"])].copy()
    if rp_sims.empty:
        logger.warning("No reliever sim results to rank")
        return pd.DataFrame()

    # Merge role classification details
    rp_sims = rp_sims.merge(
        roles_df[["pitcher_id", "role", "confidence", "games",
                   "saves", "holds", "blown_saves"]],
        on=["pitcher_id", "role"],
        how="left",
        suffixes=("", "_cls"),
    )

    # Load observed pitching advanced stats
    observed = read_sql(f"""
        SELECT pitcher_id, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, xwoba_against,
               barrel_pct_against, batters_faced
        FROM production.fact_pitching_advanced
        WHERE season = {season} AND batters_faced >= 30
    """, {})

    if not observed.empty:
        rp_sims = rp_sims.merge(observed, on="pitcher_id", how="left")

    # Load pitcher projections for projected rates
    proj_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    if proj_path.exists():
        proj = pd.read_parquet(proj_path)
        proj_cols = ["pitcher_id", "pitcher_name", "age", "pitch_hand",
                     "projected_k_rate", "projected_bb_rate",
                     "projected_k_rate_sd", "projected_bb_rate_sd"]
        proj_cols = [c for c in proj_cols if c in proj.columns]
        rp_sims = rp_sims.merge(proj[proj_cols], on="pitcher_id", how="left")

    # --- Stuff score ---
    stuff_components = []
    if "projected_k_rate" in rp_sims.columns:
        stuff_components.append(("proj_k", _pctl(rp_sims["projected_k_rate"].fillna(0.20)), 0.35))
    if "swstr_pct" in rp_sims.columns:
        stuff_components.append(("swstr", _pctl(rp_sims["swstr_pct"].fillna(0)), 0.25))
    if "csw_pct" in rp_sims.columns:
        stuff_components.append(("csw", _pctl(rp_sims["csw_pct"].fillna(0)), 0.20))
    if "xwoba_against" in rp_sims.columns:
        stuff_components.append(("xwoba", _inv_pctl(rp_sims["xwoba_against"].fillna(0.320)), 0.20))

    if stuff_components:
        total_w = sum(w for _, _, w in stuff_components)
        rp_sims["stuff_score"] = sum(
            s * (w / total_w) for _, s, w in stuff_components
        )
    else:
        # Fallback: use sim K rate
        k_rate_sim = rp_sims["total_k_mean"] / rp_sims["total_bf_mean"].clip(lower=1)
        rp_sims["stuff_score"] = _pctl(k_rate_sim)

    # --- Command score ---
    cmd_components = []
    if "projected_bb_rate" in rp_sims.columns:
        cmd_components.append(("proj_bb", _inv_pctl(rp_sims["projected_bb_rate"].fillna(0.08)), 0.35))
    if "zone_pct" in rp_sims.columns:
        cmd_components.append(("zone", _pctl(rp_sims["zone_pct"].fillna(0.45)), 0.25))
    if "chase_pct" in rp_sims.columns:
        cmd_components.append(("chase", _pctl(rp_sims["chase_pct"].fillna(0.28)), 0.25))
    if "bb_pct" in rp_sims.columns:
        cmd_components.append(("obs_bb", _inv_pctl(rp_sims["bb_pct"].fillna(0.08)), 0.15))

    if cmd_components:
        total_w = sum(w for _, _, w in cmd_components)
        rp_sims["command_score"] = sum(
            s * (w / total_w) for _, s, w in cmd_components
        )
    else:
        bb_rate_sim = rp_sims["total_bb_mean"] / rp_sims["total_bf_mean"].clip(lower=1)
        rp_sims["command_score"] = _inv_pctl(bb_rate_sim)

    # --- Role value score (CL: saves, SU: holds) ---
    rp_sims["role_value_score"] = 0.50  # default neutral
    cl_mask = rp_sims["role"] == "CL"
    if cl_mask.any():
        sv_vol = _pctl(rp_sims.loc[cl_mask, "total_sv_mean"].fillna(0))
        sv_conversion = rp_sims.loc[cl_mask, "saves"].fillna(0) / (
            rp_sims.loc[cl_mask, "saves"].fillna(0)
            + rp_sims.loc[cl_mask, "blown_saves"].fillna(0)
        ).replace(0, np.nan).fillna(0.85)
        sv_conv_pctl = _pctl(sv_conversion)
        rp_sims.loc[cl_mask, "role_value_score"] = 0.60 * sv_vol + 0.40 * sv_conv_pctl

    su_mask = rp_sims["role"] == "SU"
    if su_mask.any():
        hld_vol = _pctl(rp_sims.loc[su_mask, "total_hld_mean"].fillna(0))
        rp_sims.loc[su_mask, "role_value_score"] = hld_vol

    # --- Stability score (low ERA variance, low blown saves) ---
    rp_sims["stability_score"] = 0.50
    if "projected_era_sd" in rp_sims.columns:
        era_consistency = _inv_pctl(rp_sims["projected_era_sd"].fillna(
            rp_sims["projected_era_sd"].median() if "projected_era_sd" in rp_sims.columns else 1.0
        ))
    else:
        era_consistency = _inv_pctl(rp_sims["total_runs_sd"].fillna(1.0))

    bs_rate = rp_sims["blown_saves"].fillna(0) / rp_sims["games"].fillna(1).clip(lower=1)
    low_bs = _inv_pctl(bs_rate)
    rp_sims["stability_score"] = 0.60 * era_consistency + 0.40 * low_bs

    # --- Workload score (games, IP, flexibility) ---
    games_pctl = _pctl(rp_sims["total_games_mean"].fillna(40))
    ip_pctl = _pctl(rp_sims["projected_ip_mean"].fillna(40))
    rp_sims["workload_score"] = 0.50 * games_pctl + 0.50 * ip_pctl

    # --- Trajectory score ---
    if "projected_k_rate_sd" in rp_sims.columns and "projected_k_rate" in rp_sims.columns:
        k_cv = rp_sims["projected_k_rate_sd"].fillna(0.03) / rp_sims["projected_k_rate"].clip(0.01)
        certainty = _inv_pctl(k_cv)
    else:
        certainty = 0.50

    if "age" in rp_sims.columns:
        age = rp_sims["age"].fillna(28)
        age_upside = _inv_pctl(age)
    else:
        age_upside = 0.50

    rp_sims["trajectory_score"] = 0.60 * certainty + 0.40 * age_upside

    # --- Composite score (role-specific weights) ---
    rp_sims["tdd_value_score"] = 0.0

    for role, weights in [("CL", _CL_WEIGHTS), ("SU", _SU_WEIGHTS), ("MR", _MR_WEIGHTS)]:
        mask = rp_sims["role"] == role
        if not mask.any():
            continue
        score = pd.Series(0.0, index=rp_sims.loc[mask].index)
        for dim, w in weights.items():
            col = f"{dim}_score"
            if col in rp_sims.columns:
                score += w * rp_sims.loc[mask, col].fillna(0.50)
        rp_sims.loc[mask, "tdd_value_score"] = score

    # Rank within role
    rp_sims = rp_sims.sort_values("tdd_value_score", ascending=False)
    rp_sims["role_rank"] = rp_sims.groupby("role").cumcount() + 1
    rp_sims["overall_rp_rank"] = rp_sims["tdd_value_score"].rank(
        ascending=False, method="min",
    ).astype(int)

    # Select output columns
    output_cols = [
        "role_rank", "overall_rp_rank", "pitcher_id", "pitcher_name", "role",
        "age", "pitch_hand", "confidence",
        # Composite
        "tdd_value_score",
        # Sub-scores
        "stuff_score", "command_score", "role_value_score",
        "stability_score", "workload_score", "trajectory_score",
        # Observed
        "k_pct", "bb_pct", "swstr_pct", "csw_pct", "xwoba_against",
        # Projected rates
        "projected_k_rate", "projected_bb_rate",
        # Sim projections
        "total_k_mean", "total_bb_mean", "total_sv_mean", "total_hld_mean",
        "projected_ip_mean", "projected_era_mean", "projected_whip_mean",
        "dk_season_mean", "espn_season_mean",
        "total_games_mean",
        # Role history
        "saves", "holds", "blown_saves", "games",
    ]
    available = [c for c in output_cols if c in rp_sims.columns]
    result = rp_sims[available].sort_values(["role", "role_rank"])

    for role in ["CL", "SU", "MR"]:
        role_df = result[result["role"] == role]
        if not role_df.empty:
            top = role_df.iloc[0]
            logger.info(
                "%s: %d ranked — #1 %s (%.3f)",
                role, len(role_df),
                top.get("pitcher_name", top["pitcher_id"]),
                top["tdd_value_score"],
            )

    return result
