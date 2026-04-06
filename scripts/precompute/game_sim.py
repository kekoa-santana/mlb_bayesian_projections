"""Precompute: Game simulator data (samples, exit model, etc.)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.game_sim")


def run(
    *,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
) -> None:
    """Generate game simulator supporting data."""
    from src.data.feature_eng import build_multi_season_hitter_extended
    from src.data.queries import (
        get_batter_pitch_count_features,
        get_exit_model_training_data,
        get_pitcher_exit_tendencies,
        get_pitcher_pitch_count_features,
        get_team_bullpen_rates,
        get_tto_adjustment_profiles,
    )
    from src.models.game_sim.exit_model import ExitModel

    logger.info("=" * 60)
    logger.info("Generating game simulator data...")

    # 9b. Hitter HR synthetic Beta samples (no Bayesian model -- use Beta posterior)
    logger.info("Generating synthetic hitter HR samples...")
    hitter_ext = build_multi_season_hitter_extended(seasons, min_pa=1)

    hr_season = hitter_ext[hitter_ext["season"] == from_season].copy()
    hr_samples_dict: dict[str, np.ndarray] = {}
    rng_hr = np.random.default_rng(42)
    for _, row in hr_season.iterrows():
        bid = int(row["batter_id"])
        pa = int(row.get("pa", row.get("plate_appearances", 0)))
        hr = int(row.get("hr", row.get("home_runs", 0)))
        if pa < 50:
            continue
        # Beta(hr + 1, pa - hr + 1) posterior
        hr_rate_samples = rng_hr.beta(hr + 1, pa - hr + 1, size=1_000)
        hr_samples_dict[str(bid)] = hr_rate_samples.astype(np.float32)
    np.savez_compressed(DASHBOARD_DIR / "hitter_hr_samples.npz", **hr_samples_dict)
    logger.info("Saved hitter HR Beta posterior samples for %d batters", len(hr_samples_dict))

    # 9c. Train + save exit model
    logger.info("Training pitcher exit model...")
    try:
        exit_training = get_exit_model_training_data(seasons)
        exit_tendencies = get_pitcher_exit_tendencies(seasons)
        exit_model = ExitModel()
        exit_metrics = exit_model.train(exit_training, exit_tendencies)
        logger.info(
            "Exit model: AUC=%.4f, n=%d",
            exit_metrics["auc"], exit_metrics["n_samples"],
        )
        exit_model.save(DASHBOARD_DIR / "exit_model.pkl")
        exit_tendencies.to_parquet(
            DASHBOARD_DIR / "pitcher_exit_tendencies.parquet", index=False,
        )
        logger.info("Saved exit model + tendencies")
    except Exception:
        logger.exception("Failed to train exit model")

    # 9d. Pitch count features
    logger.info("Computing pitch count features...")
    try:
        pitcher_pc = get_pitcher_pitch_count_features(seasons)
        pitcher_pc.to_parquet(
            DASHBOARD_DIR / "pitcher_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved pitcher pitch count features: %d rows", len(pitcher_pc))

        batter_pc = get_batter_pitch_count_features(seasons)
        batter_pc.to_parquet(
            DASHBOARD_DIR / "batter_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved batter pitch count features: %d rows", len(batter_pc))
    except Exception:
        logger.exception("Failed to compute pitch count features")

    # 9e. TTO profiles
    logger.info("Computing TTO adjustment profiles...")
    try:
        tto_profiles = get_tto_adjustment_profiles(seasons)
        tto_profiles.to_parquet(
            DASHBOARD_DIR / "tto_profiles.parquet", index=False,
        )
        logger.info("Saved TTO profiles: %d rows", len(tto_profiles))
    except Exception:
        logger.exception("Failed to compute TTO profiles")

    # 9f. Train + save game-level BB adjustment model
    logger.info("Training game-level BB adjustment model...")
    try:
        import pickle
        from src.models.game_bb_adj import train_game_bb_model

        bb_adj_bundle = train_game_bb_model(seasons)
        if bb_adj_bundle.get("model") is not None:
            with open(DASHBOARD_DIR / "game_bb_adj_model.pkl", "wb") as f:
                pickle.dump(bb_adj_bundle, f)
            logger.info(
                "Saved game BB adjustment model: n=%d, RMSE=%.4f",
                bb_adj_bundle["n_train"],
                bb_adj_bundle.get("rmse_logit", 0),
            )
        else:
            logger.warning("Game BB adjustment model training returned no model")
    except Exception:
        logger.exception("Failed to train game BB adjustment model")

    # 9g. Team bullpen rates
    logger.info("Computing team bullpen rates...")
    try:
        bullpen_rates = get_team_bullpen_rates(seasons)
        bullpen_rates.to_parquet(
            DASHBOARD_DIR / "team_bullpen_rates.parquet", index=False,
        )
        logger.info("Saved team bullpen rates: %d rows", len(bullpen_rates))
    except Exception:
        logger.exception("Failed to compute team bullpen rates")

    logger.info("Game simulator data complete.")
