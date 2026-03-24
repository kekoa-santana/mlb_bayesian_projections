"""Precompute: Posterior sample extraction (npz files)."""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.samples")


def run_pitcher_samples(
    *,
    pitcher_results: dict,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
) -> None:
    """Extract and save pitcher posterior samples (BB%, HR/BF, K%)."""
    from src.data.feature_eng import build_multi_season_pitcher_k_data
    from src.models.game_k_model import extract_pitcher_k_rate_samples
    from src.models.pitcher_k_rate_model import (
        check_pitcher_convergence,
        fit_pitcher_k_rate_model,
        prepare_pitcher_model_data,
    )

    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not pitcher_results:
        logger.warning("pitcher_results not available (pitcher_models skipped) -- skipping pitcher_samples")
        return

    # Save BB% and HR/BF samples from composite model
    for stat_name, npz_name in [("bb_rate", "pitcher_bb_samples"), ("hr_per_bf", "pitcher_hr_samples")]:
        pre = pitcher_results.get(stat_name, {}).get("rate_samples", {})
        if not pre:
            logger.warning("No %s pre-extracted samples -- skipping %s", stat_name, npz_name)
            continue

        samples_dict = {str(pid): arr for pid, arr in pre.items()}
        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **samples_dict)
        logger.info("Saved %s posterior samples for %d pitchers", stat_name, len(samples_dict))

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **samples_dict)
        logger.info("Saved preseason %s samples snapshot", stat_name)

    # Free pre-extracted samples -- already saved to disk above
    for stat_key in list(pitcher_results.keys()):
        pitcher_results[stat_key].pop("rate_samples", None)
    gc.collect()

    # -- Pitcher K% model (for posterior samples -> Game K sim) --
    logger.info("=" * 60)
    logger.info("Fitting pitcher K%% model for posterior samples...")
    df_pitcher = build_multi_season_pitcher_k_data(seasons, min_bf=1)
    pitcher_data = prepare_pitcher_model_data(df_pitcher)
    _model, pitcher_trace = fit_pitcher_k_rate_model(
        pitcher_data, draws=draws, tune=tune, chains=chains,
    )
    del _model
    gc.collect()
    conv = check_pitcher_convergence(pitcher_trace)
    logger.info("Pitcher K%% convergence: %s (r_hat=%.4f)",
                "OK" if conv["converged"] else "ISSUES", conv["max_rhat"])

    # Extract forward-projected K% samples for each pitcher active in from_season
    active = df_pitcher[
        (df_pitcher["season"] == from_season) & (df_pitcher["batters_faced"] >= 50)
    ]["pitcher_id"].unique()

    k_samples_dict: dict[str, np.ndarray] = {}
    for pid in active:
        try:
            samples = extract_pitcher_k_rate_samples(
                pitcher_trace, pitcher_data,
                pitcher_id=int(pid),
                season=from_season,
                project_forward=True,
            )
            k_samples_dict[str(int(pid))] = samples
        except ValueError:
            continue

    np.savez_compressed(
        DASHBOARD_DIR / "pitcher_k_samples.npz",
        **k_samples_dict,
    )
    logger.info("Saved K%% posterior samples for %d pitchers", len(k_samples_dict))

    # Save preseason K samples snapshot for in-season conjugate updating
    np.savez_compressed(
        snapshot_dir / "pitcher_k_samples_preseason.npz",
        **k_samples_dict,
    )
    logger.info("Saved preseason K%% samples snapshot")

    # Free standalone K% model trace
    del pitcher_trace, pitcher_data, df_pitcher
    gc.collect()


def run_hitter_samples(
    *,
    hitter_results: dict,
) -> None:
    """Extract and save hitter K% and BB% posterior samples."""
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not hitter_results:
        logger.warning("hitter_results not available (hitter_models skipped) -- skipping hitter_samples")
        return

    for stat_name, npz_name in [("k_rate", "hitter_k_samples"), ("bb_rate", "hitter_bb_samples")]:
        pre = hitter_results.get(stat_name, {}).get("rate_samples", {})
        if not pre:
            logger.warning("No %s pre-extracted samples -- skipping %s", stat_name, npz_name)
            continue

        h_samples_dict = {str(pid): arr for pid, arr in pre.items()}
        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **h_samples_dict)
        logger.info("Saved hitter %s posterior samples for %d batters", stat_name, len(h_samples_dict))

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **h_samples_dict)
        logger.info("Saved preseason hitter %s samples snapshot", stat_name)
