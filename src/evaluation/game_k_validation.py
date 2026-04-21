"""Game-level K prediction backtest — thin wrapper over game_prop_validation.

This module preserves backward-compatible entry points (``run_full_game_k_backtest``,
``run_full_game_backtest``, ``build_game_k_predictions``, ``compute_game_k_metrics``)
but delegates to the canonical ``game_prop_validation`` implementation.

For new code, import from ``game_prop_validation`` directly.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.game_prop_validation import (
    PITCHER_PROP_CONFIGS,
    GamePropConfig,
    build_game_prop_predictions,
    compute_game_prop_metrics,
    run_full_game_prop_backtest,
)

logger = logging.getLogger(__name__)

# Re-export the K config for callers that reference it
K_CONFIG = PITCHER_PROP_CONFIGS["k"]

# Config for additional game-level stats beyond K
GAME_STAT_CONFIGS: dict[str, str] = {
    "bb": "bb",
    "hr": "hr",
    "outs": "outs",
}


def build_game_k_predictions(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    starters_only: bool = True,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build game-level K predictions for a single train→test fold.

    Delegates to ``game_prop_validation.build_game_prop_predictions``
    with K configuration.
    """
    return build_game_prop_predictions(
        config=K_CONFIG,
        train_seasons=train_seasons,
        test_season=test_season,
        draws=draws,
        tune=tune,
        chains=chains,
        n_mc_draws=n_mc_draws,
        random_seed=random_seed,
    )


def compute_game_k_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute calibration and accuracy metrics for K predictions.

    Delegates to ``game_prop_validation.compute_game_prop_metrics``.
    """
    return compute_game_prop_metrics(predictions, K_CONFIG)


def run_full_game_k_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run walk-forward K backtest across multiple folds.

    Returns a summary DataFrame with one row per fold.
    """
    summary_df, _ = run_full_game_prop_backtest(
        config=K_CONFIG,
        folds=folds,
        draws=draws,
        tune=tune,
        chains=chains,
        n_mc_draws=n_mc_draws,
        random_seed=random_seed,
    )
    return summary_df


def run_full_game_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    additional_stats: list[str] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest for K + additional game-level stats.

    Returns (k_results, stat_results).
    """
    if additional_stats is None:
        additional_stats = ["bb", "hr", "outs"]

    # K backtest
    k_results = run_full_game_k_backtest(
        folds=folds,
        draws=draws,
        tune=tune,
        chains=chains,
        n_mc_draws=n_mc_draws,
        min_bf_game=min_bf_game,
        random_seed=random_seed,
    )

    # Additional stats
    stat_rows = []
    for stat_name in additional_stats:
        config = PITCHER_PROP_CONFIGS.get(stat_name)
        if config is None:
            logger.warning("No config for stat '%s' — skipping", stat_name)
            continue

        try:
            summary_df, _ = run_full_game_prop_backtest(
                config=config,
                folds=folds,
                draws=draws,
                tune=tune,
                chains=chains,
                n_mc_draws=n_mc_draws,
                random_seed=random_seed,
            )
            if not summary_df.empty:
                summary_df["stat"] = stat_name
                stat_rows.append(summary_df)
        except Exception as e:
            logger.warning("Failed to backtest %s: %s", stat_name, e)

    stat_results = pd.concat(stat_rows, ignore_index=True) if stat_rows else pd.DataFrame()
    return k_results, stat_results
