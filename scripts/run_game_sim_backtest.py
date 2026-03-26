#!/usr/bin/env python
"""
Run walk-forward backtest for the sequential PA game simulator.

Evaluates the full PA-by-PA simulator against historical games,
comparing against the current Layer 3 (game_k_model.py) baseline.

Usage
-----
    python scripts/run_game_sim_backtest.py          # full quality
    python scripts/run_game_sim_backtest.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.game_sim_validation import (
    run_full_game_sim_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Game simulator walk-forward backtest"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws and sims for fast iteration",
    )
    parser.add_argument(
        "--single-fold", type=int, default=None,
        help="Run only a single test season (e.g., --single-fold 2024)",
    )
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_sims = 500, 250, 2, 2000
    else:
        draws, tune, chains, n_sims = 1000, 500, 2, 5000

    # Configure folds
    if args.single_fold:
        test_season = args.single_fold
        train_start = 2020
        train_seasons = list(range(train_start, test_season))
        folds = [(train_seasons, test_season)]
    else:
        folds = None  # default: 2023, 2024, 2025

    logger.info("=" * 70)
    logger.info("GAME SIMULATOR BACKTEST")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC sims per game: %d", n_sims)
    if args.single_fold:
        logger.info("  Single fold: test=%d", args.single_fold)
    logger.info("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    summary, predictions = run_full_game_sim_backtest(
        folds=folds,
        draws=draws,
        tune=tune,
        chains=chains,
        n_sims=n_sims,
        random_seed=42,
    )

    # Save results
    summary.to_csv(out_dir / "game_sim_backtest_summary.csv", index=False)
    if len(predictions) > 0:
        predictions.to_csv(
            out_dir / "game_sim_backtest_predictions.csv", index=False
        )

    # Print results
    _print_results(summary, predictions)


def _print_results(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("GAME SIMULATOR BACKTEST RESULTS")
    print("=" * 70)

    if len(summary) == 0:
        print("No results.")
        return

    # Per-fold summary
    print("\n--- Per-Fold Results ---")
    for _, row in summary.iterrows():
        print(f"\nTest Season {int(row['test_season'])} ({int(row['n_games'])} games):")
        for stat in ("k", "bb", "h", "hr", "outs"):
            rmse_key = f"{stat}_rmse"
            mae_key = f"{stat}_mae"
            corr_key = f"{stat}_corr"
            if rmse_key in row:
                label = stat.upper() if stat != "outs" else "OUTS"
                line = f"  {label:>4s}: RMSE={row[rmse_key]:.3f}"
                if mae_key in row:
                    line += f"  MAE={row[mae_key]:.3f}"
                if corr_key in row and pd.notna(row[corr_key]):
                    line += f"  corr={row[corr_key]:.3f}"
                print(line)

        if "k_avg_brier" in row and pd.notna(row["k_avg_brier"]):
            print(f"  K Brier: {row['k_avg_brier']:.4f}")
        if "outs_avg_brier" in row and pd.notna(row["outs_avg_brier"]):
            print(f"  Outs Brier: {row['outs_avg_brier']:.4f}")

        for ci in ("50", "80", "90"):
            key = f"k_coverage_{ci}"
            if key in row and pd.notna(row[key]):
                print(f"  K Coverage {ci}%: {row[key]:.1%}")

        for ci in ("50", "80", "90"):
            key = f"outs_coverage_{ci}"
            if key in row and pd.notna(row[key]):
                print(f"  Outs Coverage {ci}%: {row[key]:.1%}")

        if "ip_rmse" in row and pd.notna(row["ip_rmse"]):
            print(f"   IP: RMSE={row['ip_rmse']:.3f}  MAE={row.get('ip_mae', 0):.3f}")

        if "pitches_rmse" in row and pd.notna(row["pitches_rmse"]):
            print(f"  PIT: RMSE={row['pitches_rmse']:.3f}")

    # Overall averages
    print("\n--- Overall Averages ---")
    for stat in ("k", "bb", "h", "hr", "outs"):
        rmse_key = f"{stat}_rmse"
        if rmse_key in summary.columns:
            label = stat.upper() if stat != "outs" else "OUTS"
            print(f"  {label:>4s}: RMSE={summary[rmse_key].mean():.3f}  "
                  f"MAE={summary[f'{stat}_mae'].mean():.3f}")

    if "k_avg_brier" in summary.columns:
        print(f"  K Avg Brier: {summary['k_avg_brier'].mean():.4f}")
    if "outs_avg_brier" in summary.columns:
        print(f"  Outs Avg Brier: {summary['outs_avg_brier'].mean():.4f}")

    # Comparison targets
    print("\n--- vs Current Layer 3 Baseline ---")
    print("  Layer 3 K RMSE: 2.280  |  K Brier: 0.1872")
    if "k_rmse" in summary.columns:
        avg_rmse = summary["k_rmse"].mean()
        avg_brier = summary["k_avg_brier"].mean() if "k_avg_brier" in summary.columns else float("nan")
        rmse_delta = avg_rmse - 2.280
        brier_delta = avg_brier - 0.1872
        print(f"  Sim K RMSE:    {avg_rmse:.3f}  (delta: {rmse_delta:+.3f})")
        print(f"  Sim K Brier:   {avg_brier:.4f}  (delta: {brier_delta:+.4f})")

    print()


if __name__ == "__main__":
    main()
