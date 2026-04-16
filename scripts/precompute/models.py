"""Precompute: Bayesian model fitting (hitter + pitcher)."""
from __future__ import annotations

import logging

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS, load_calibration_t, save_dashboard_parquet

logger = logging.getLogger("precompute.models")


def run(
    *,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    include_hitter: bool = True,
    include_pitcher: bool = True,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Fit hitter and/or pitcher composite models and produce projections.

    Returns
    -------
    hitter_results, pitcher_results, hitter_proj, pitcher_proj
    """
    from src.models.hitter_projections import (
        fit_all_models as fit_hitter_models,
        project_forward as project_hitter_forward,
    )
    from src.models.pitcher_projections import (
        fit_all_models as fit_pitcher_models,
        project_forward as project_pitcher_forward,
    )

    hitter_cal_t, pitcher_cal_t = load_calibration_t()

    hitter_results: dict = {}
    pitcher_results: dict = {}
    hitter_proj = pd.DataFrame()
    pitcher_proj = pd.DataFrame()

    # Try loading existing projections as fallbacks
    try:
        hitter_proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")
    except FileNotFoundError:
        pass
    try:
        pitcher_proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    except FileNotFoundError:
        pass

    # -- Hitter models --
    if include_hitter:
        logger.info("=" * 60)
        logger.info("Fitting hitter composite models (K%%, BB%%)...")

        if any(v != 1.0 for v in hitter_cal_t.values()):
            logger.info("Hitter calibration T: %s", hitter_cal_t)
        if any(v != 1.0 for v in pitcher_cal_t.values()):
            logger.info("Pitcher calibration T: %s", pitcher_cal_t)

        hitter_results = fit_hitter_models(
            seasons=seasons, min_pa=100,
            draws=draws, tune=tune, chains=chains, random_seed=42,
            extract_season=from_season,
        )
        hitter_proj = project_hitter_forward(
            hitter_results, from_season=from_season, min_pa=150,
            calibration_t=hitter_cal_t,
        )
        save_dashboard_parquet(hitter_proj, "hitter_projections.parquet")
        logger.info("Saved hitter projections: %d players", len(hitter_proj))

    # -- Pitcher models --
    if include_pitcher:
        logger.info("=" * 60)
        logger.info("Fitting pitcher composite models (K%%, BB%%)...")
        pitcher_results = fit_pitcher_models(
            seasons=seasons, min_bf=50,
            draws=draws, tune=tune, chains=chains, random_seed=42,
            extract_season=from_season,
        )
        # Get ERA-FIP gap data for ERA derivation
        from src.data.feature_eng import get_pitcher_era_fip_data
        era_fip_data = get_pitcher_era_fip_data(from_season)
        logger.info("Loaded ERA-FIP data for %d pitchers", len(era_fip_data))

        # Train XGBoost priors for ERA-FIP gap and BABIP
        from src.models.xgb_priors import train_all_pitcher_priors, predict_pitcher_priors
        xgb_models = train_all_pitcher_priors(seasons=seasons, min_ip=40.0)
        xgb_preds: dict[str, dict[int, float]] = {}
        for target, bundle in xgb_models.items():
            if bundle.get("model") is not None:
                xgb_preds[target] = predict_pitcher_priors(bundle, from_season)
                logger.info(
                    "XGBoost %s: %d predictions",
                    target, len(xgb_preds[target]),
                )

        pitcher_proj = project_pitcher_forward(
            pitcher_results, from_season=from_season, min_bf=150,
            era_fip_data=era_fip_data,
            calibration_t=pitcher_cal_t,
            xgb_priors=xgb_preds,
        )
        save_dashboard_parquet(pitcher_proj, "pitcher_projections.parquet")
        logger.info("Saved pitcher projections: %d players", len(pitcher_proj))

    return hitter_results, pitcher_results, hitter_proj, pitcher_proj
