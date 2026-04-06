"""Pitcher sub-score builders for the player rankings system.

Extracted from ``player_rankings.py`` -- zero behavioral changes.

Each ``_build_pitcher_*`` function computes one component of the pitcher
composite ranking score.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models._ranking_utils import (
    DASHBOARD_DIR,
    _OBS_WEIGHT_BASE,
    _inv_pctl,
    _pctl,
    _pitcher_age_factor,
    _stat_family_trust,
    _zscore_pctl,
)

logger = logging.getLogger(__name__)


def _compute_stuff_plus(run_values: pd.DataFrame) -> pd.DataFrame:
    """Compute arsenal-level Stuff+ from per-pitch-type run values.

    Stuff+ = 100 + (league_mean - pitcher_rv) / league_std * 10
    Inverted because negative run value = better for pitcher.
    Arsenal-level = usage-weighted average across pitch types.

    Parameters
    ----------
    run_values : pd.DataFrame
        From ``get_pitcher_run_values``. Must have pitcher_id, pitch_type,
        run_value_per_100, usage_pct columns.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, arsenal_stuff_plus.
    """
    if run_values is None or run_values.empty:
        return pd.DataFrame(columns=["pitcher_id", "arsenal_stuff_plus"])

    rv = run_values.copy()

    # League mean and std per pitch type
    lg_stats = rv.groupby("pitch_type")["run_value_per_100"].agg(["mean", "std"])
    lg_stats["std"] = lg_stats["std"].clip(lower=0.5)  # floor to avoid division by ~0

    # Merge league stats and compute per-pitch Stuff+
    rv = rv.merge(lg_stats, on="pitch_type", how="left", suffixes=("", "_lg"))
    rv["stuff_plus_pitch"] = 100 + (rv["mean"] - rv["run_value_per_100"]) / rv["std"] * 10

    # Arsenal-level: usage-weighted average
    arsenal = (
        rv.groupby("pitcher_id")
        .apply(
            lambda g: np.average(g["stuff_plus_pitch"], weights=g["usage_pct"])
            if g["usage_pct"].sum() > 0 else 100.0,
            include_groups=False,
        )
        .rename("arsenal_stuff_plus")
        .reset_index()
    )

    return arsenal


def _build_pitcher_stuff_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    run_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Blend projected and observed stuff/quality metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.
    run_values : pd.DataFrame, optional
        Pitcher run values from ``get_pitcher_run_values``.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, stuff_score.
    """
    p_sub = proj[
        ["pitcher_id", "projected_k_rate", "projected_hr_per_bf",
         "whiff_rate", "gb_pct"]
    ].drop_duplicates("pitcher_id", keep="first")
    o_cols = ["pitcher_id", "swstr_pct", "csw_pct", "xwoba_against",
              "barrel_pct_against", "hard_hit_pct_against"]
    if "batters_faced" in observed.columns:
        o_cols.append("batters_faced")
    o_sub = observed[o_cols].drop_duplicates("pitcher_id", keep="first")
    merged = p_sub.merge(o_sub, on="pitcher_id", how="left")

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "stuff_score"])

    bf = merged["batters_faced"].fillna(0) if "batters_faced" in merged.columns else pd.Series(0, index=merged.index)

    # Merge run values and compute Stuff+ if available
    has_stuff_plus = False
    if run_values is not None and not run_values.empty:
        rv_dedup = run_values[["pitcher_id", "weighted_rv_per_100"]].drop_duplicates("pitcher_id", keep="first")
        merged = merged.merge(rv_dedup, on="pitcher_id", how="left")

        # Compute arsenal Stuff+ (league-normalized, 100 = average)
        stuff_plus = _compute_stuff_plus(run_values)
        if not stuff_plus.empty:
            merged = merged.merge(stuff_plus, on="pitcher_id", how="left")
            has_stuff_plus = "arsenal_stuff_plus" in merged.columns and merged["arsenal_stuff_plus"].notna().any()

    # Projected: K rate (higher = better), HR/BF (lower = better),
    # GB% (higher = better -- values groundball pitchers)
    proj_k = _pctl(merged["projected_k_rate"])
    proj_hr = _inv_pctl(merged["projected_hr_per_bf"])
    proj_gb = _pctl(merged["gb_pct"].fillna(merged["gb_pct"].median()))
    projected = 0.55 * proj_k + 0.25 * proj_hr + 0.20 * proj_gb

    # Observed: SwStr%, CSW%, xwOBA-against, barrel%-against,
    # plus Stuff+ when available (league-normalized pitch quality)
    obs_swstr = _pctl(merged["swstr_pct"].fillna(0))
    obs_csw = _pctl(merged["csw_pct"].fillna(0))
    obs_xwoba = _inv_pctl(merged["xwoba_against"].fillna(merged["xwoba_against"].median()))
    obs_barrel = _inv_pctl(merged["barrel_pct_against"].fillna(0))

    if has_stuff_plus:
        # Stuff+ captures underlying pitch quality better than raw SwStr%/CSW%;
        # stabilizes in fewer pitches (~80) and predicts ROS better
        stuff_plus_score = _zscore_pctl(merged["arsenal_stuff_plus"].fillna(100))
        observed_score = (
            0.30 * stuff_plus_score + 0.20 * obs_swstr + 0.20 * obs_csw
            + 0.20 * obs_xwoba + 0.10 * obs_barrel
        )
    elif "weighted_rv_per_100" in merged.columns:
        obs_run_value = _inv_pctl(merged["weighted_rv_per_100"].fillna(merged["weighted_rv_per_100"].median()))
        observed_score = (
            0.25 * obs_swstr + 0.25 * obs_csw + 0.20 * obs_run_value
            + 0.20 * obs_xwoba + 0.10 * obs_barrel
        )
    else:
        observed_score = 0.30 * obs_swstr + 0.30 * obs_csw + 0.25 * obs_xwoba + 0.15 * obs_barrel

    # BF-based trust ramp: SwStr%/CSW% stabilize ~150 pitches (~40 BF),
    # but full stuff picture needs ~300 BF.  At low BF, lean on projections.
    stuff_trust = _stat_family_trust(bf, min_pa=75, full_pa=300)
    obs_weight = _OBS_WEIGHT_BASE * stuff_trust
    # Fill NaN observed (no in-season data) with projected so blend is clean
    observed_score = observed_score.fillna(projected)
    merged["stuff_score"] = (1 - obs_weight) * projected + obs_weight * observed_score

    # Preserve Stuff+ for output display
    cols = ["pitcher_id", "stuff_score"]
    if has_stuff_plus:
        cols.append("arsenal_stuff_plus")
    return merged[cols]


def _build_pitcher_command_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    efficiency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score command/control from projected BB% and observed metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.
    efficiency : pd.DataFrame, optional
        Pitcher efficiency data from ``get_pitcher_efficiency``.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, command_score.
    """
    p_sub = proj[["pitcher_id", "projected_bb_rate"]].drop_duplicates(
        "pitcher_id", keep="first",
    )
    o_cols = ["pitcher_id", "bb_pct", "zone_pct", "chase_pct"]
    if "batters_faced" in observed.columns:
        o_cols.append("batters_faced")
    o_sub = observed[o_cols].drop_duplicates("pitcher_id", keep="first")
    merged = p_sub.merge(o_sub, on="pitcher_id", how="left")

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "command_score"])

    bf = merged["batters_faced"].fillna(0) if "batters_faced" in merged.columns else pd.Series(0, index=merged.index)

    # Merge efficiency data if available
    if efficiency is not None and not efficiency.empty:
        eff_cols = ["pitcher_id", "first_strike_pct", "putaway_rate"]
        eff_subset = efficiency[[c for c in eff_cols if c in efficiency.columns]].copy()
        eff_subset = eff_subset.drop_duplicates("pitcher_id", keep="first")
        merged = merged.merge(eff_subset, on="pitcher_id", how="left")

    # Lower BB% = better command
    proj_bb = _inv_pctl(merged["projected_bb_rate"])
    obs_bb = _inv_pctl(merged["bb_pct"].fillna(merged["bb_pct"].median()))
    obs_zone = _pctl(merged["zone_pct"].fillna(0))
    obs_chase = _pctl(merged["chase_pct"].fillna(0))

    projected_cmd = proj_bb

    if "first_strike_pct" in merged.columns and "putaway_rate" in merged.columns:
        obs_first_strike = _pctl(merged["first_strike_pct"].fillna(merged["first_strike_pct"].median()))
        obs_putaway = _pctl(merged["putaway_rate"].fillna(merged["putaway_rate"].median()))
        observed_cmd = (
            0.30 * obs_bb + 0.20 * obs_zone + 0.20 * obs_chase
            + 0.15 * obs_first_strike + 0.15 * obs_putaway
        )
    else:
        observed_cmd = 0.40 * obs_bb + 0.30 * obs_zone + 0.30 * obs_chase

    # BF-based trust ramp: BB% less stable than whiff metrics, needs more BF.
    cmd_trust = _stat_family_trust(bf, min_pa=100, full_pa=400)
    obs_cmd_weight = _OBS_WEIGHT_BASE * cmd_trust
    observed_cmd = observed_cmd.fillna(projected_cmd)
    merged["command_score"] = (1 - obs_cmd_weight) * projected_cmd + obs_cmd_weight * observed_cmd
    return merged[["pitcher_id", "command_score"]]


def _build_pitcher_workload_score(
    health_df: pd.DataFrame | None = None,
    use_counting_sim: bool = True,
) -> pd.DataFrame:
    """Score projected workload from sim-based counting projections + health.

    Prefers ``pitcher_counting_sim.parquet`` (sim-based IP/games) over
    the old ``pitcher_counting.parquet`` (rate x BF). Falls back to old
    file if sim parquet doesn't exist.

    Parameters
    ----------
    health_df : pd.DataFrame or None
        Pre-loaded health scores (columns: player_id, health_score,
        health_label).  When provided the disk read of
        ``health_scores.parquet`` is skipped.
    use_counting_sim : bool
        If False, skip sim parquet and use ``pitcher_counting.parquet`` only
        (in-season weekly refresh does not rebuild sim).

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, workload_score, health_score, health_label,
        plus sim projection columns for display.
    """
    # Prefer sim-based projections (IP, games, ERA, fantasy)
    sim_path = DASHBOARD_DIR / "pitcher_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "pitcher_counting.parquet"

    if use_counting_sim and sim_path.exists():
        counting = pd.read_parquet(sim_path)
        counting = counting.drop_duplicates("pitcher_id", keep="first")
        # Sim has projected_ip_mean and total_games_mean
        ip_pctl = _pctl(counting["projected_ip_mean"].fillna(0))
        games_pctl = _pctl(counting["total_games_mean"].fillna(0))
        base_workload = 0.60 * ip_pctl + 0.40 * games_pctl
        logger.info("Workload score from sim parquet: %d pitchers", len(counting))
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        counting = counting.drop_duplicates("pitcher_id", keep="first")
        base_workload = _pctl(counting["projected_bf_mean"])
        logger.info(
            "Workload score from old counting parquet (%s)",
            "sim disabled (in-season)" if not use_counting_sim else "sim not found",
        )
    else:
        logger.warning("No counting parquet found for workload score")
        return pd.DataFrame(columns=["pitcher_id", "workload_score"])

    # Health scores: prefer columns on counting, then in-memory, then disk
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        if health_df is None:
            health_path = DASHBOARD_DIR / "health_scores.parquet"
            if health_path.exists():
                health_df = pd.read_parquet(health_path)
        if health_df is not None and not health_df.empty:
            _h = health_df[
                ["player_id", "health_score", "health_label"]
            ].rename(columns={"player_id": "pitcher_id"}).drop_duplicates(
                "pitcher_id", keep="first",
            )
            counting = counting.merge(_h, on="pitcher_id", how="left")
            has_health = True

    if has_health:
        counting["health_score"] = counting["health_score"].fillna(0.85)
        counting["health_label"] = counting["health_label"].fillna("Unknown")
        health_pctl = _pctl(counting["health_score"])
        counting["workload_score"] = 0.70 * base_workload + 0.30 * health_pctl
    else:
        counting["workload_score"] = base_workload
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["pitcher_id", "workload_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_pitcher_velo_trend(season: int = 2025) -> pd.DataFrame:
    """Compute YoY velocity trend from pitch-level data.

    Positive delta = velo gain (good), negative = velo loss (risk).

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, velo_delta (mph change from prior year).
    """
    from src.data.db import read_sql

    velo = read_sql(f"""
        SELECT fp.pitcher_id, dg.season,
               AVG(fp.release_speed) as avg_velo,
               COUNT(*) as pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season >= {season - 1} AND dg.game_type = 'R'
              AND fp.release_speed != 'NaN'
        GROUP BY fp.pitcher_id, dg.season
        HAVING COUNT(*) >= 300
    """, {})

    if velo.empty:
        return pd.DataFrame(columns=["pitcher_id", "velo_delta"])

    # Pivot to get prior and current season
    prior = velo[velo["season"] == season - 1][["pitcher_id", "avg_velo"]].rename(
        columns={"avg_velo": "velo_prior"},
    )
    current = velo[velo["season"] == season][["pitcher_id", "avg_velo"]].rename(
        columns={"avg_velo": "velo_current"},
    )
    merged = current.merge(prior, on="pitcher_id", how="inner")
    merged["velo_delta"] = merged["velo_current"] - merged["velo_prior"]

    return merged[["pitcher_id", "velo_delta"]]


def _build_pitcher_innings_durability(min_seasons: int = 2) -> pd.DataFrame:
    """Score SP innings durability from historical IP track record.

    Rewards starters who consistently log high innings. Separate from
    health score (which measures IL stints).

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, durability_score (0-1).
    """
    from src.data.db import read_sql

    ip_hist = read_sql("""
        SELECT player_id as pitcher_id, season,
               SUM(pit_ip) as total_ip,
               COUNT(*) as games,
               SUM(CASE WHEN pit_is_starter THEN 1 ELSE 0 END) as starts
        FROM production.fact_player_game_mlb
        WHERE player_role = 'pitcher' AND season >= 2020
        GROUP BY player_id, season
        HAVING SUM(pit_bf) >= 50
    """, {})

    if ip_hist.empty:
        return pd.DataFrame(columns=["pitcher_id", "durability_score"])

    # Only consider pitchers with enough seasons
    season_counts = ip_hist.groupby("pitcher_id")["season"].nunique()
    eligible = season_counts[season_counts >= min_seasons].index
    ip_hist = ip_hist[ip_hist["pitcher_id"].isin(eligible)].copy()

    if ip_hist.empty:
        return pd.DataFrame(columns=["pitcher_id", "durability_score"])

    # Recency-weighted average IP per season
    ip_hist = ip_hist.sort_values(["pitcher_id", "season"])
    ip_hist["weight"] = ip_hist.groupby("pitcher_id").cumcount() + 1  # older=1, newer=higher

    agg = (
        ip_hist.groupby("pitcher_id")
        .apply(
            lambda g: pd.Series({
                "wtd_ip": np.average(g["total_ip"], weights=g["weight"]),
                "max_ip": g["total_ip"].max(),
                "seasons": g["season"].nunique(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Score: 180+ IP average = elite durability, 100 IP = average, <60 = poor
    agg["durability_score"] = ((agg["wtd_ip"] - 40) / 160.0).clip(0, 1)

    return agg[["pitcher_id", "durability_score"]]


def _build_pitcher_trajectory_score(season: int = 2025) -> pd.DataFrame:
    """Score trajectory using posterior certainty, projected quality,
    age/decline risk, and velocity trend.

    Components
    ----------
    - **Projection certainty** (35%): tighter posterior SD = more proven
    - **Projected rate quality** (25%): high K%, low BB%
    - **Age factor** (20%): upside for youth, decline penalty for 32+
    - **Velocity trend** (20%): velo gain = upside, velo loss = risk

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, trajectory_score, velo_delta.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

    # Posterior certainty via CV (SD/mean) -- same fix as hitters
    k_mean = proj["projected_k_rate"].clip(0.01)
    bb_mean = proj["projected_bb_rate"].clip(0.01)
    k_cv = proj["projected_k_rate_sd"].fillna(k_mean.median() * 0.15) / k_mean
    bb_cv = proj["projected_bb_rate_sd"].fillna(bb_mean.median() * 0.15) / bb_mean
    k_certainty = _inv_pctl(k_cv)
    bb_certainty = _inv_pctl(bb_cv)
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # Projected rate quality: elite rates signal upside regardless of track
    # record length.  Without this, short-track-record aces (Crochet, Yamamoto)
    # get penalized purely for having wide posteriors.
    k_quality = _pctl(proj["projected_k_rate"])
    bb_quality = _inv_pctl(proj["projected_bb_rate"])
    rate_quality = 0.60 * k_quality + 0.40 * bb_quality

    # Age factor: research-aligned curve (peak 27-29, decline from 30, accel 35+)
    age = proj["age"].fillna(28)
    age_factor = _pitcher_age_factor(age)

    # Velocity trend
    velo_trend = _build_pitcher_velo_trend(season=season)
    proj = proj.merge(velo_trend, on="pitcher_id", how="left")
    proj["velo_delta"] = proj["velo_delta"].fillna(0)

    # Convert velo delta to 0-1 score: +2 mph gain = 1.0, -2 mph loss = 0.0
    velo_score = ((proj["velo_delta"] + 2.0) / 4.0).clip(0, 1)

    proj["trajectory_score"] = (
        0.35 * certainty_score
        + 0.25 * rate_quality
        + 0.20 * age_factor
        + 0.20 * velo_score
    )
    return proj[["pitcher_id", "trajectory_score", "velo_delta"]]
