"""
Walk-forward validation of the Component ELO system.

Train on seasons 2000-N, predict game outcomes in season N+1.
Measures accuracy, log loss, Brier score, and calibration.

Target: 55-58% accuracy (MLB ELO literature range).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.models.team_elo import (
    ELORatings,
    _expected_score,
    compute_elo_history,
    get_config,
)

logger = logging.getLogger(__name__)


def walk_forward_validation(
    game_results: pd.DataFrame,
    venue_factors: pd.DataFrame | None = None,
    train_end: int = 2020,
    test_seasons: list[int] | None = None,
    config: dict | None = None,
) -> dict:
    """Walk-forward ELO validation.

    Parameters
    ----------
    game_results : pd.DataFrame
        Full game results (2000-2025).
    venue_factors : pd.DataFrame, optional
        Park factors.
    train_end : int
        Last season in initial training window.
    test_seasons : list[int], optional
        Seasons to test on.  Defaults to [train_end+1, ..., max_season].
    config : dict, optional
        ELO config overrides.

    Returns
    -------
    dict
        ``{summary, per_season, predictions}`` with metrics and detail.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    max_season = int(game_results["season"].max())
    if test_seasons is None:
        test_seasons = list(range(train_end + 1, max_season + 1))

    all_preds: list[dict] = []
    per_season_metrics: list[dict] = []

    for test_year in test_seasons:
        # Train on everything up to (but not including) test_year
        train_data = game_results[game_results["season"] < test_year]
        if train_data.empty:
            continue

        # Compute ELO on training data
        ratings, _hist = compute_elo_history(train_data, venue_factors, cfg)

        # Test on test_year games
        test_data = game_results[game_results["season"] == test_year]
        if test_data.empty:
            continue

        # Use pre-season regression for prediction
        from src.models.team_elo import project_preseason_elo
        preseason = project_preseason_elo(ratings, cfg)
        ha = cfg["home_advantage"]
        sp_weight = cfg["sp_weight"]
        initial = cfg["initial_rating"]

        season_preds = []
        for _, row in test_data.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            home_runs = int(row["home_runs"])
            away_runs = int(row["away_runs"])

            home_sp = int(row["home_sp_id"]) if pd.notna(row.get("home_sp_id")) else None
            away_sp = int(row["away_sp_id"]) if pd.notna(row.get("away_sp_id")) else None

            # Predicted win probability
            home_eff = preseason.effective(home_id, home_sp, sp_weight, initial) + ha
            away_eff = preseason.effective(away_id, away_sp, sp_weight, initial)
            pred_home = _expected_score(home_eff, away_eff)

            # Actual outcome
            if home_runs == away_runs:
                continue  # skip ties
            actual_home = 1.0 if home_runs > away_runs else 0.0

            season_preds.append({
                "season": test_year,
                "game_pk": row["game_pk"],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "pred_home_win": pred_home,
                "actual_home_win": actual_home,
            })

        if not season_preds:
            continue

        sdf = pd.DataFrame(season_preds)
        all_preds.extend(season_preds)

        # Season metrics
        correct = ((sdf["pred_home_win"] > 0.5) == (sdf["actual_home_win"] == 1.0)).mean()
        brier = ((sdf["pred_home_win"] - sdf["actual_home_win"]) ** 2).mean()
        log_loss = -np.mean(
            sdf["actual_home_win"] * np.log(np.clip(sdf["pred_home_win"], 1e-10, 1))
            + (1 - sdf["actual_home_win"]) * np.log(np.clip(1 - sdf["pred_home_win"], 1e-10, 1))
        )

        per_season_metrics.append({
            "season": test_year,
            "games": len(sdf),
            "accuracy": correct,
            "brier": brier,
            "log_loss": log_loss,
        })
        logger.info(
            "Season %d: accuracy=%.3f, brier=%.4f, log_loss=%.4f (%d games)",
            test_year, correct, brier, log_loss, len(sdf),
        )

    # Overall metrics
    pred_df = pd.DataFrame(all_preds) if all_preds else pd.DataFrame()
    if pred_df.empty:
        return {"summary": {}, "per_season": [], "predictions": pred_df}

    overall_acc = ((pred_df["pred_home_win"] > 0.5) == (pred_df["actual_home_win"] == 1.0)).mean()
    overall_brier = ((pred_df["pred_home_win"] - pred_df["actual_home_win"]) ** 2).mean()
    overall_ll = -np.mean(
        pred_df["actual_home_win"] * np.log(np.clip(pred_df["pred_home_win"], 1e-10, 1))
        + (1 - pred_df["actual_home_win"]) * np.log(np.clip(1 - pred_df["pred_home_win"], 1e-10, 1))
    )

    # Baseline: always pick home team (historical ~53.2%)
    home_baseline_acc = pred_df["actual_home_win"].mean()

    # Calibration: bin predictions and compare to actual rates
    pred_df["pred_bin"] = pd.cut(pred_df["pred_home_win"], bins=10, labels=False)
    calibration = pred_df.groupby("pred_bin").agg(
        predicted=("pred_home_win", "mean"),
        actual=("actual_home_win", "mean"),
        count=("actual_home_win", "count"),
    ).reset_index()
    calibration["error"] = abs(calibration["predicted"] - calibration["actual"])
    cal_error = calibration["error"].mean()

    summary = {
        "total_games": len(pred_df),
        "test_seasons": test_seasons,
        "accuracy": overall_acc,
        "brier": overall_brier,
        "log_loss": overall_ll,
        "home_baseline_acc": home_baseline_acc,
        "improvement_over_baseline": overall_acc - home_baseline_acc,
        "calibration_error": cal_error,
    }

    logger.info("=" * 60)
    logger.info("OVERALL ELO VALIDATION")
    logger.info("  Games:      %d", summary["total_games"])
    logger.info("  Accuracy:   %.3f (baseline: %.3f, +%.3f)",
                summary["accuracy"], summary["home_baseline_acc"],
                summary["improvement_over_baseline"])
    logger.info("  Brier:      %.4f", summary["brier"])
    logger.info("  Log Loss:   %.4f", summary["log_loss"])
    logger.info("  Cal Error:  %.4f", summary["calibration_error"])

    return {
        "summary": summary,
        "per_season": per_season_metrics,
        "predictions": pred_df,
        "calibration": calibration,
    }
