#!/usr/bin/env python
"""Run game-level prop backtests for all stat types.

Usage
-----
python scripts/run_game_prop_backtest.py                    # all props
python scripts/run_game_prop_backtest.py --side pitcher      # pitcher only
python scripts/run_game_prop_backtest.py --stat k bb         # K and BB only
python scripts/run_game_prop_backtest.py --side batter --stat k  # batter K only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.game_prop_validation import (
    BATTER_PROP_CONFIGS,
    PITCHER_PROP_CONFIGS,
    run_full_game_prop_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Game prop backtests")
    parser.add_argument("--side", choices=["pitcher", "batter", "all"], default="all")
    parser.add_argument("--stat", nargs="+", default=None,
                        help="Stats to backtest (k, bb, hr, h, outs)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs_to_run = []

    if args.side in ("pitcher", "all"):
        for name, config in PITCHER_PROP_CONFIGS.items():
            if args.stat is None or name in args.stat:
                configs_to_run.append((f"pitcher_{name}", config))

    if args.side in ("batter", "all"):
        for name, config in BATTER_PROP_CONFIGS.items():
            if args.stat is None or name in args.stat:
                configs_to_run.append((f"batter_{name}", config))

    all_summaries = []

    for label, config in configs_to_run:
        logger.info("=" * 60)
        logger.info("Running backtest: %s", label)
        logger.info("=" * 60)

        try:
            summary_df, predictions_df = run_full_game_prop_backtest(
                config=config,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                n_mc_draws=args.mc_draws,
                random_seed=args.seed,
            )

            # Save predictions
            pred_path = OUTPUT_DIR / f"game_prop_predictions_{label}.parquet"
            predictions_df.to_parquet(pred_path, index=False)
            logger.info("Saved predictions to %s", pred_path)

            # Add label to summary
            summary_df["prop"] = label
            summary_df["side"] = config.side
            summary_df["stat"] = config.stat_name
            all_summaries.append(summary_df)

        except Exception:
            logger.exception("Failed backtest for %s", label)
            continue

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        summary_path = OUTPUT_DIR / "game_prop_backtest_summary.parquet"
        combined.to_parquet(summary_path, index=False)
        logger.info("Saved combined summary to %s", summary_path)

        # Print summary table
        print("\n" + "=" * 80)
        print("GAME PROP BACKTEST SUMMARY")
        print("=" * 80)
        display_cols = ["prop", "test_season", "n_games", "rmse", "mae",
                        "avg_brier", "ece", "coverage_80", "coverage_90"]
        available = [c for c in display_cols if c in combined.columns]
        print(combined[available].to_string(index=False))
    else:
        logger.warning("No backtests completed successfully")


if __name__ == "__main__":
    main()
