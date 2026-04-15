#!/usr/bin/env python
"""
Run team ELO walk-forward backtest.

Usage
-----
    python scripts/run_team_elo_backtest.py                 # full (2000-2025)
    python scripts/run_team_elo_backtest.py --train-end 2015  # custom split
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.team_queries import get_game_results, get_team_info, get_venue_run_factors
from src.evaluation.runner import ensure_out_dir, save_csv, setup_logging
from src.evaluation.team_elo_validation import walk_forward_validation
from src.models.team_elo import compute_elo_history, get_current_ratings

logger = setup_logging("team_elo_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Team ELO backtest")
    parser.add_argument(
        "--train-end", type=int, default=2020,
        help="Last season in initial training window (default: 2020)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out_dir()

    # Load data
    logger.info("Loading game results...")
    games = get_game_results()
    logger.info("Loaded %d games (%d-%d)",
                len(games), games["season"].min(), games["season"].max())

    venue_factors = get_venue_run_factors()
    logger.info("Loaded %d venue factors", len(venue_factors))

    team_info = get_team_info()

    # Run full ELO computation (for current ratings display)
    logger.info("Computing full ELO history...")
    ratings, history = compute_elo_history(games, venue_factors)

    current = get_current_ratings(ratings, team_info)
    logger.info("\n=== Current ELO Rankings ===")
    for _, row in current.head(15).iterrows():
        logger.info(
            "  %2d. %-4s  Composite: %.0f  (Off: %.0f, Pit: %.0f)",
            row["composite_rank"], row["abbreviation"],
            row["composite_elo"], row["offense_elo"], row["pitching_elo"],
        )

    # Save current ratings
    save_csv(current, "team_elo_current.csv", logger)

    # Walk-forward validation
    logger.info("=" * 60)
    logger.info("Running walk-forward validation (train through %d)...", args.train_end)
    results = walk_forward_validation(
        games, venue_factors, train_end=args.train_end,
    )

    # Save results
    if not results["predictions"].empty:
        save_csv(results["predictions"], "team_elo_predictions.csv", logger)
    save_csv(
        pd.DataFrame(results["per_season"]),
        "team_elo_per_season.csv",
        logger,
    )

    # Print per-season summary
    logger.info("\n=== Per-Season Accuracy ===")
    for m in results["per_season"]:
        logger.info(
            "  %d: %.1f%% (%d games)",
            m["season"], m["accuracy"] * 100, m["games"],
        )

    # Print calibration
    if "calibration" in results and results["calibration"] is not None:
        logger.info("\n=== Calibration ===")
        for _, row in results["calibration"].iterrows():
            logger.info(
                "  Predicted: %.2f  Actual: %.2f  (n=%d)",
                row["predicted"], row["actual"], row["count"],
            )


if __name__ == "__main__":
    main()
