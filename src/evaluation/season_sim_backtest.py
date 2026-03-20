"""
Walk-forward backtest for the season simulator pipeline.

Full end-to-end: PyMC hierarchical models -> posterior extraction ->
PA-by-PA game simulator (starters) / vectorized reliever sim ->
season counting stats + fantasy scoring.

Compares against: old rate x BF counting projections and Marcel baseline.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_pitcher_extended,
)
from src.data.queries import (
    get_exit_model_training_data,
    get_pitcher_exit_tendencies,
)
from src.models.counting_projections import (
    marcel_counting_pitcher,
    project_pitcher_counting,
    project_pitcher_counting_sim,
)
from src.models.game_sim.exit_model import ExitModel
from src.models.health_score import compute_health_scores
from src.models.reliever_roles import classify_reliever_roles

logger = logging.getLogger(__name__)

# Stats the sim produces that the old pipeline doesn't
SIM_ONLY_STATS = ["total_h", "total_hr", "total_runs", "total_sv", "total_hld",
                   "total_games", "total_pitches"]
DERIVED_STATS = ["projected_ip", "projected_era", "projected_whip"]
FANTASY_STATS = ["dk_season", "espn_season"]

# Map sim column prefixes to test actual columns
ACTUAL_MAP = {
    "total_k": "k",
    "total_bb": "bb",
    "total_outs": "outs",
    "total_h": "hits_allowed",
    "total_hr": "hr",
    "total_runs": "runs",
    "total_games": "games",
    "total_sv": "sv",
    "total_hld": "hld",
    "projected_ip": "ip",
}


def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lo_80: np.ndarray | None = None,
    hi_80: np.ndarray | None = None,
    lo_95: np.ndarray | None = None,
    hi_95: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute regression metrics."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[valid], predicted[valid]
    n = len(a)
    if n == 0:
        return {"n": 0}

    residuals = a - p
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(p - a))
    corr = float(np.corrcoef(a, p)[0, 1]) if np.std(a) > 0 and np.std(p) > 0 else 0.0

    metrics = {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "correlation": corr}

    if lo_80 is not None and hi_80 is not None:
        lo, hi = lo_80[valid], hi_80[valid]
        metrics["coverage_80"] = float(np.mean((a >= lo) & (a <= hi)))
    if lo_95 is not None and hi_95 is not None:
        lo, hi = lo_95[valid], hi_95[valid]
        metrics["coverage_95"] = float(np.mean((a >= lo) & (a <= hi)))

    return metrics


def _build_posteriors(
    pitcher_results: dict[str, dict[str, Any]],
    pitcher_ext: pd.DataFrame,
    from_season: int,
    min_bf: int = 50,
    random_seed: int = 42,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, str]]:
    """Extract rate posteriors for all active pitchers.

    Returns (posteriors_dict, name_lookup).
    """
    from src.models.pitcher_model import extract_rate_samples

    rng = np.random.default_rng(random_seed)

    active = pitcher_ext[
        (pitcher_ext["season"] == from_season)
        & (pitcher_ext["batters_faced"] >= min_bf)
    ][["pitcher_id", "pitcher_name", "batters_faced", "is_starter",
       "hr"]].drop_duplicates("pitcher_id")

    posteriors: dict[int, dict[str, np.ndarray]] = {}
    names: dict[int, str] = {}

    for _, row in active.iterrows():
        pid = int(row["pitcher_id"])
        names[pid] = row.get("pitcher_name", "")
        bf = int(row["batters_faced"])
        rates: dict[str, np.ndarray] = {}

        for stat_key, post_key in [("k_rate", "k_rate"), ("bb_rate", "bb_rate")]:
            if stat_key in pitcher_results:
                try:
                    rates[post_key] = extract_rate_samples(
                        pitcher_results[stat_key]["trace"],
                        pitcher_results[stat_key]["data"],
                        pitcher_id=pid, season=from_season,
                        project_forward=True,
                    )
                    continue
                except (ValueError, KeyError):
                    pass
            # Fallback
            if stat_key == "k_rate":
                rates[post_key] = rng.beta(5, 20, size=8000)
            else:
                rates[post_key] = rng.beta(3, 30, size=8000)

        # HR rate
        if "hr_per_bf" in pitcher_results:
            try:
                rates["hr_rate"] = extract_rate_samples(
                    pitcher_results["hr_per_bf"]["trace"],
                    pitcher_results["hr_per_bf"]["data"],
                    pitcher_id=pid, season=from_season,
                    project_forward=True,
                )
            except (ValueError, KeyError):
                hr_count = int(row.get("hr", bf * 0.03))
                rates["hr_rate"] = rng.beta(hr_count + 1, bf - hr_count + 1, size=8000)
        else:
            hr_count = int(row.get("hr", bf * 0.03))
            rates["hr_rate"] = rng.beta(hr_count + 1, bf - hr_count + 1, size=8000)

        posteriors[pid] = rates

    logger.info("Built posteriors for %d pitchers", len(posteriors))
    return posteriors, names


def _build_starter_priors(
    pitcher_ext: pd.DataFrame,
    from_season: int,
    starter_ids: set[int],
) -> pd.DataFrame:
    """Build per-starter n_starts priors from multi-season history."""
    rows = []
    for pid in starter_ids:
        hist = pitcher_ext[
            (pitcher_ext["pitcher_id"] == pid)
            & (pitcher_ext["season"] <= from_season)
            & (pitcher_ext["is_starter"] == 1)
        ].sort_values("season", ascending=False).head(3)

        if hist.empty:
            rows.append({"pitcher_id": pid, "n_starts_mu": 30.0,
                         "n_starts_sigma": 6.0, "avg_pitches": 88.0})
            continue

        weights = [5, 4, 3][:len(hist)]
        adj_g = hist["games"].values.copy().astype(float)
        for i, s in enumerate(hist["season"].values):
            if s == 2020:
                adj_g[i] = min(adj_g[i] * (162 / 60), 36)

        wtd = sum(g * w for g, w in zip(adj_g, weights)) / sum(weights)
        n = len(hist)
        rel = n / (n + 1.5)
        mu = rel * wtd + (1 - rel) * 30.0

        sigma = max(float(np.std(adj_g, ddof=1)), 3.0) if len(adj_g) >= 2 else 6.0

        rows.append({"pitcher_id": pid, "n_starts_mu": mu,
                      "n_starts_sigma": sigma, "avg_pitches": 88.0})

    df = pd.DataFrame(rows)
    logger.info("Built starter priors for %d pitchers (mean mu=%.1f, sigma=%.1f)",
                len(df), df["n_starts_mu"].mean(), df["n_starts_sigma"].mean())
    return df


def _build_babip_adjustments(
    pitcher_ext: pd.DataFrame,
    from_season: int,
    shrinkage_k: int = 500,
    pop_babip: float = 0.300,
) -> dict[int, float]:
    """Compute shrinkage-regressed BABIP adjustments for each pitcher.

    BABIP = (H - HR) / (BF - K - HR - BB). Uses recency-weighted average
    across recent seasons, regressed toward population BABIP.

    Parameters
    ----------
    pitcher_ext : pd.DataFrame
        Multi-season pitcher data with k, bb, hr, hits_allowed, batters_faced.
    from_season : int
        Last training season.
    shrinkage_k : int
        BIP needed for full weight on pitcher-specific BABIP.
    pop_babip : float
        Population BABIP to regress toward.

    Returns
    -------
    dict[int, float]
        pitcher_id -> BABIP adjustment (positive = more hits on BIP).
    """
    recent = pitcher_ext[
        (pitcher_ext["season"] >= from_season - 2)
        & (pitcher_ext["season"] <= from_season)
    ].copy()

    if recent.empty:
        return {}

    # Compute per-season BABIP components
    h_col = "hits_allowed" if "hits_allowed" in recent.columns else "h"
    if h_col not in recent.columns:
        logger.warning("No hits column found for BABIP; skipping adjustments")
        return {}

    recent["bip"] = (
        recent["batters_faced"] - recent["k"] - recent["hr"] - recent["bb"]
    ).clip(lower=1)
    recent["hits_on_bip"] = (recent[h_col] - recent["hr"]).clip(lower=0)
    recent["babip"] = recent["hits_on_bip"] / recent["bip"]

    # Recency-weighted
    recent["weight"] = np.where(recent["season"] == from_season, 3.0, 1.0)

    agg = (
        recent.groupby("pitcher_id")
        .apply(
            lambda g: pd.Series({
                "wtd_babip": np.average(g["babip"], weights=g["weight"] * g["bip"]),
                "total_bip": g["bip"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Shrinkage: reliability = min(total_bip / shrinkage_k, 1.0)
    agg["reliability"] = (agg["total_bip"] / shrinkage_k).clip(0, 1)
    agg["babip_adj"] = agg["reliability"] * (agg["wtd_babip"] - pop_babip)

    result = dict(zip(agg["pitcher_id"].astype(int), agg["babip_adj"]))
    logger.info(
        "BABIP adjustments: %d pitchers, mean=%.4f, range=[%.4f, %.4f]",
        len(result), np.mean(list(result.values())),
        min(result.values()), max(result.values()),
    )
    return result


def _get_test_actuals(
    pitcher_ext: pd.DataFrame,
    test_season: int,
    min_bf: int = 100,
) -> pd.DataFrame:
    """Load test-season actuals including saves/holds from DB."""
    from src.data.db import read_sql

    base = pitcher_ext[
        (pitcher_ext["season"] == test_season)
        & (pitcher_ext["batters_faced"] >= min_bf)
    ].copy()

    # Get saves, holds, ERA from fact_player_game_mlb
    extra = read_sql("""
        SELECT player_id as pitcher_id,
               SUM(pit_sv) as sv, SUM(pit_hld) as hld,
               SUM(pit_er) as er,
               SUM(pit_r) as runs,
               SUM(pit_h) as hits_allowed
        FROM production.fact_player_game_mlb
        WHERE season = :season AND player_role = 'pitcher'
        GROUP BY player_id
    """, {"season": test_season})

    if not extra.empty:
        base = base.merge(extra, on="pitcher_id", how="left", suffixes=("", "_db"))
        # Use DB values where available
        for col in ["sv", "hld", "runs", "hits_allowed"]:
            if f"{col}_db" in base.columns:
                base[col] = base[f"{col}_db"].fillna(base.get(col, 0))

    # Derive ERA and WHIP
    base["era"] = base.get("er", base.get("earned_runs", 0)).astype(float) / base["ip"].replace(0, np.nan) * 9.0
    base["whip"] = (base["bb"].astype(float) + base.get("hits_allowed", 0).astype(float)) / base["ip"].replace(0, np.nan)

    return base


def walk_forward_season_sim(
    train_seasons: list[int],
    test_season: int,
    min_bf_train: int = 100,
    min_bf_test: int = 100,
    n_seasons: int = 200,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Full walk-forward evaluation of the season simulator pipeline.

    Fits PyMC models on training data, extracts posteriors, runs season
    simulator, and compares against test actuals + baselines.

    Parameters
    ----------
    train_seasons : list[int]
        Training seasons.
    test_season : int
        Season to evaluate predictions against.
    min_bf_train, min_bf_test : int
        Minimum batters faced thresholds.
    n_seasons : int
        Monte Carlo seasons per pitcher in sim.
    draws, tune, chains : int
        MCMC sampling parameters.
    random_seed : int

    Returns
    -------
    dict
        Keys: 'summary_rows', 'predictions', 'test_season', 'timings'.
    """
    from src.models.pitcher_projections import fit_all_models as fit_pitcher_models

    timings: dict[str, float] = {}
    last_train = max(train_seasons)
    all_seasons = sorted(set(train_seasons) | {test_season})

    logger.info("=" * 60)
    logger.info("Season sim backtest: train=%s, test=%d", train_seasons, test_season)
    logger.info("=" * 60)

    # ---- 1. Build extended data ----
    t0 = time.time()
    pitcher_ext = build_multi_season_pitcher_extended(all_seasons, min_bf=1)
    train_ext = pitcher_ext[pitcher_ext["season"].isin(train_seasons)]
    timings["data_build"] = time.time() - t0
    logger.info("Data: %d pitcher-seasons (%.1fs)", len(pitcher_ext), timings["data_build"])

    # ---- 2. Fit PyMC pitcher models ----
    t0 = time.time()
    pitcher_results = fit_pitcher_models(
        seasons=train_seasons, min_bf=min_bf_train,
        draws=draws, tune=tune, chains=chains, random_seed=random_seed,
    )
    timings["mcmc_fit"] = time.time() - t0
    logger.info("MCMC fit: %.1fs", timings["mcmc_fit"])

    # ---- 3. Extract posteriors ----
    t0 = time.time()
    posteriors, names = _build_posteriors(
        pitcher_results, pitcher_ext, from_season=last_train,
        min_bf=min_bf_train, random_seed=random_seed,
    )
    timings["posteriors"] = time.time() - t0

    # ---- 4. Classify reliever roles ----
    t0 = time.time()
    role_seasons = [s for s in train_seasons if s >= max(2020, last_train - 2)]
    reliever_roles = classify_reliever_roles(
        seasons=role_seasons, current_season=last_train, min_games=10,
    )
    timings["roles"] = time.time() - t0
    logger.info("Roles: %s", reliever_roles["role"].value_counts().to_dict() if not reliever_roles.empty else "none")

    # ---- 5. Build starter priors ----
    starter_ids = set(
        pitcher_ext[
            (pitcher_ext["season"] == last_train)
            & (pitcher_ext["is_starter"] == 1)
            & (pitcher_ext["batters_faced"] >= min_bf_train)
        ]["pitcher_id"]
    )
    starter_priors = _build_starter_priors(pitcher_ext, last_train, starter_ids)

    # ---- 6. Train exit model ----
    t0 = time.time()
    exit_model = ExitModel()
    try:
        exit_data = get_exit_model_training_data(train_seasons)
        exit_tend = get_pitcher_exit_tendencies(train_seasons)
        exit_model.train(exit_data, exit_tend)
        logger.info("Exit model trained (AUC available)")
    except Exception as e:
        logger.warning("Exit model training failed, using fallback: %s", e)
    timings["exit_model"] = time.time() - t0

    # ---- 7. Health scores ----
    try:
        health_df = compute_health_scores(last_train)
    except Exception:
        health_df = None

    # ---- 7b. BABIP adjustments ----
    t0_babip = time.time()
    babip_adjs = _build_babip_adjustments(pitcher_ext, last_train)
    timings["babip"] = time.time() - t0_babip

    # ---- 8. Run season simulator ----
    t0 = time.time()
    sim_proj = project_pitcher_counting_sim(
        posteriors=posteriors,
        roles=reliever_roles,
        exit_model=exit_model,
        starter_priors=starter_priors,
        health_scores=health_df,
        babip_adjs=babip_adjs,
        pitcher_names=names,
        n_seasons=n_seasons,
        random_seed=random_seed,
    )
    timings["season_sim"] = time.time() - t0
    logger.info("Season sim: %d pitchers in %.1fs", len(sim_proj), timings["season_sim"])

    # ---- 9. Run old counting baseline (rate x BF) ----
    t0 = time.time()
    old_proj = project_pitcher_counting(
        rate_model_results=pitcher_results,
        pitcher_extended=train_ext,
        from_season=last_train,
        n_draws=4000, min_bf=min_bf_train,
        random_seed=random_seed,
        health_scores=health_df,
    )
    timings["old_counting"] = time.time() - t0

    # ---- 10. Marcel baseline ----
    marcel_proj = marcel_counting_pitcher(
        train_ext, from_season=last_train, min_bf=min_bf_train,
    )

    # ---- 11. Load test actuals ----
    test_df = _get_test_actuals(pitcher_ext, test_season, min_bf=min_bf_test)
    logger.info("Test actuals: %d pitchers with %d+ BF in %d",
                len(test_df), min_bf_test, test_season)

    # ---- 12. Merge and compute metrics ----
    common_ids = (
        set(sim_proj["pitcher_id"])
        & set(test_df["pitcher_id"])
    )
    logger.info("Common pitchers (sim & test): %d", len(common_ids))

    # Build role lookup for SP/RP split
    role_lookup = dict(zip(sim_proj["pitcher_id"], sim_proj["role"]))

    # Merge all projections with test actuals
    test_sub = test_df[test_df["pitcher_id"].isin(common_ids)].copy()
    sim_sub = sim_proj[sim_proj["pitcher_id"].isin(common_ids)].copy()
    old_sub = old_proj[old_proj["pitcher_id"].isin(common_ids)].copy() if not old_proj.empty else pd.DataFrame()
    marcel_sub = marcel_proj[marcel_proj["pitcher_id"].isin(common_ids)].copy() if not marcel_proj.empty else pd.DataFrame()

    summary_rows = []

    # Stats to evaluate
    eval_stats = [
        # (stat_prefix, actual_col, has_old_counting, has_marcel)
        ("total_k", "k", True, True),
        ("total_bb", "bb", True, True),
        ("total_outs", "outs", True, True),
        ("total_h", "hits_allowed", False, False),
        ("total_hr", "hr", False, False),
        ("total_games", "games", False, False),
        ("total_sv", "sv", False, False),
        ("total_hld", "hld", False, False),
        ("projected_ip", "ip", True, False),
    ]

    for stat_prefix, actual_col, has_old, has_marcel in eval_stats:
        if actual_col not in test_sub.columns:
            continue

        mean_col = f"{stat_prefix}_mean"
        if mean_col not in sim_sub.columns:
            continue

        actual_vals = test_sub.set_index("pitcher_id")[actual_col].reindex(
            sim_sub["pitcher_id"].values
        ).values.astype(float)
        sim_vals = sim_sub[mean_col].values

        lo_80 = sim_sub.get(f"{stat_prefix}_p10", pd.Series(dtype=float)).values
        hi_80 = sim_sub.get(f"{stat_prefix}_p90", pd.Series(dtype=float)).values
        lo_95 = sim_sub.get(f"{stat_prefix}_p2_5", pd.Series(dtype=float)).values
        hi_95 = sim_sub.get(f"{stat_prefix}_p97_5", pd.Series(dtype=float)).values

        # Sim metrics (all pitchers)
        sim_m = _compute_metrics(actual_vals, sim_vals, lo_80, hi_80, lo_95, hi_95)

        # Old counting baseline
        old_m: dict[str, float] = {}
        if has_old and not old_sub.empty:
            old_col_map = {"total_k": "total_k_mean", "total_bb": "total_bb_mean",
                           "total_outs": "total_outs_mean", "projected_ip": "projected_ip_mean"}
            old_mean_col = old_col_map.get(stat_prefix)
            if old_mean_col and old_mean_col in old_sub.columns:
                old_vals = old_sub.set_index("pitcher_id")[old_mean_col].reindex(
                    sim_sub["pitcher_id"].values
                ).values.astype(float)
                old_m = _compute_metrics(actual_vals, old_vals)

        # Marcel baseline
        marcel_m: dict[str, float] = {}
        if has_marcel and not marcel_sub.empty:
            marcel_col_map = {"total_k": "marcel_k", "total_bb": "marcel_bb",
                              "total_outs": "marcel_outs"}
            marcel_col = marcel_col_map.get(stat_prefix)
            if marcel_col and marcel_col in marcel_sub.columns:
                marcel_vals = marcel_sub.set_index("pitcher_id")[marcel_col].reindex(
                    sim_sub["pitcher_id"].values
                ).values.astype(float)
                marcel_m = _compute_metrics(actual_vals, marcel_vals)

        # Overall row
        mae_vs_marcel = 0.0
        if marcel_m.get("mae", 0) > 0:
            mae_vs_marcel = (marcel_m["mae"] - sim_m.get("mae", 0)) / marcel_m["mae"] * 100
        mae_vs_old = 0.0
        if old_m.get("mae", 0) > 0:
            mae_vs_old = (old_m["mae"] - sim_m.get("mae", 0)) / old_m["mae"] * 100

        summary_rows.append({
            "stat": stat_prefix, "role": "ALL", "test_season": test_season,
            "n": sim_m.get("n", 0),
            "sim_mae": sim_m.get("mae"), "sim_rmse": sim_m.get("rmse"),
            "sim_bias": sim_m.get("bias"), "sim_corr": sim_m.get("correlation"),
            "sim_cov80": sim_m.get("coverage_80"), "sim_cov95": sim_m.get("coverage_95"),
            "old_mae": old_m.get("mae"), "old_corr": old_m.get("correlation"),
            "marcel_mae": marcel_m.get("mae"), "marcel_corr": marcel_m.get("correlation"),
            "mae_vs_marcel_pct": mae_vs_marcel, "mae_vs_old_pct": mae_vs_old,
        })

        # SP / RP split
        for role_label, role_mask_fn in [
            ("SP", lambda pid: role_lookup.get(pid) == "SP"),
            ("RP", lambda pid: role_lookup.get(pid) in ("CL", "SU", "MR")),
        ]:
            mask = np.array([role_mask_fn(pid) for pid in sim_sub["pitcher_id"].values])
            if mask.sum() < 5:
                continue

            split_sim = _compute_metrics(
                actual_vals[mask], sim_vals[mask],
                lo_80[mask] if len(lo_80) > 0 else None,
                hi_80[mask] if len(hi_80) > 0 else None,
                lo_95[mask] if len(lo_95) > 0 else None,
                hi_95[mask] if len(hi_95) > 0 else None,
            )

            split_marcel: dict[str, float] = {}
            if has_marcel and marcel_m:
                split_marcel = _compute_metrics(actual_vals[mask], marcel_vals[mask])

            mae_vs_m = 0.0
            if split_marcel.get("mae", 0) > 0:
                mae_vs_m = (split_marcel["mae"] - split_sim.get("mae", 0)) / split_marcel["mae"] * 100

            summary_rows.append({
                "stat": stat_prefix, "role": role_label, "test_season": test_season,
                "n": split_sim.get("n", 0),
                "sim_mae": split_sim.get("mae"), "sim_rmse": split_sim.get("rmse"),
                "sim_bias": split_sim.get("bias"), "sim_corr": split_sim.get("correlation"),
                "sim_cov80": split_sim.get("coverage_80"), "sim_cov95": split_sim.get("coverage_95"),
                "old_mae": None, "old_corr": None,
                "marcel_mae": split_marcel.get("mae"), "marcel_corr": split_marcel.get("correlation"),
                "mae_vs_marcel_pct": mae_vs_m, "mae_vs_old_pct": None,
            })

    # ERA / WHIP (sim only, derived stats)
    if "projected_era_mean" in sim_sub.columns:
        test_era = test_sub.set_index("pitcher_id")["era"].reindex(
            sim_sub["pitcher_id"].values
        ).values.astype(float)
        sim_era = sim_sub["projected_era_mean"].values
        era_m = _compute_metrics(
            test_era, sim_era,
            sim_sub.get("projected_era_p10", pd.Series(dtype=float)).values,
            sim_sub.get("projected_era_p90", pd.Series(dtype=float)).values,
            sim_sub.get("projected_era_p2_5", pd.Series(dtype=float)).values,
            sim_sub.get("projected_era_p97_5", pd.Series(dtype=float)).values,
        )
        summary_rows.append({
            "stat": "ERA", "role": "ALL", "test_season": test_season,
            "n": era_m.get("n"), "sim_mae": era_m.get("mae"),
            "sim_rmse": era_m.get("rmse"), "sim_bias": era_m.get("bias"),
            "sim_corr": era_m.get("correlation"),
            "sim_cov80": era_m.get("coverage_80"), "sim_cov95": era_m.get("coverage_95"),
            "old_mae": None, "old_corr": None,
            "marcel_mae": None, "marcel_corr": None,
            "mae_vs_marcel_pct": None, "mae_vs_old_pct": None,
        })

    # FIP-derived ERA (should be more predictive than run-based ERA)
    if "projected_fip_era_mean" in sim_sub.columns:
        test_era_fip = test_sub.set_index("pitcher_id")["era"].reindex(
            sim_sub["pitcher_id"].values
        ).values.astype(float)
        sim_fip_era = sim_sub["projected_fip_era_mean"].values
        fip_era_m = _compute_metrics(
            test_era_fip, sim_fip_era,
            sim_sub.get("projected_fip_era_p10", pd.Series(dtype=float)).values,
            sim_sub.get("projected_fip_era_p90", pd.Series(dtype=float)).values,
            sim_sub.get("projected_fip_era_p2_5", pd.Series(dtype=float)).values,
            sim_sub.get("projected_fip_era_p97_5", pd.Series(dtype=float)).values,
        )
        summary_rows.append({
            "stat": "FIP-ERA", "role": "ALL", "test_season": test_season,
            "n": fip_era_m.get("n"), "sim_mae": fip_era_m.get("mae"),
            "sim_rmse": fip_era_m.get("rmse"), "sim_bias": fip_era_m.get("bias"),
            "sim_corr": fip_era_m.get("correlation"),
            "sim_cov80": fip_era_m.get("coverage_80"), "sim_cov95": fip_era_m.get("coverage_95"),
            "old_mae": None, "old_corr": None,
            "marcel_mae": None, "marcel_corr": None,
            "mae_vs_marcel_pct": None, "mae_vs_old_pct": None,
        })

    if "projected_whip_mean" in sim_sub.columns:
        test_whip = test_sub.set_index("pitcher_id")["whip"].reindex(
            sim_sub["pitcher_id"].values
        ).values.astype(float)
        sim_whip = sim_sub["projected_whip_mean"].values
        whip_m = _compute_metrics(test_whip, sim_whip)
        summary_rows.append({
            "stat": "WHIP", "role": "ALL", "test_season": test_season,
            "n": whip_m.get("n"), "sim_mae": whip_m.get("mae"),
            "sim_rmse": whip_m.get("rmse"), "sim_bias": whip_m.get("bias"),
            "sim_corr": whip_m.get("correlation"),
            "sim_cov80": None, "sim_cov95": None,
            "old_mae": None, "old_corr": None,
            "marcel_mae": None, "marcel_corr": None,
            "mae_vs_marcel_pct": None, "mae_vs_old_pct": None,
        })

    summary = pd.DataFrame(summary_rows)

    return {
        "summary": summary,
        "test_season": test_season,
        "timings": timings,
        "n_pitchers": len(common_ids),
    }
