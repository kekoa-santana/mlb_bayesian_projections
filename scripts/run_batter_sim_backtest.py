#!/usr/bin/env python
"""
Run walk-forward backtest for the batter game simulator.

Usage
-----
    python scripts/run_batter_sim_backtest.py          # full quality
    python scripts/run_batter_sim_backtest.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.batter_sim_validation import run_full_batter_sim_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batter game simulator backtest"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--single-fold", type=int, default=None)
    parser.add_argument("--max-batters", type=int, default=None,
                        help="Cap batter-games per fold for quick testing")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_sims = 500, 250, 2, 2000
        max_batters = args.max_batters or 5000
    else:
        draws, tune, chains, n_sims = 1000, 500, 2, 5000
        max_batters = args.max_batters

    if args.single_fold:
        test = args.single_fold
        folds = [(list(range(2020, test)), test)]
    else:
        folds = None

    logger.info("=" * 70)
    logger.info("BATTER GAME SIMULATOR BACKTEST")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC sims per game: %d", n_sims)
    if max_batters:
        logger.info("  Max batter-games per fold: %d", max_batters)
    logger.info("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    summary, predictions = run_full_batter_sim_backtest(
        folds=folds,
        draws=draws,
        tune=tune,
        chains=chains,
        n_sims=n_sims,
        max_batters_per_fold=max_batters,
        random_seed=42,
    )

    summary.to_csv(out_dir / "batter_sim_backtest_summary.csv", index=False)
    if len(predictions) > 0:
        predictions.to_csv(
            out_dir / "batter_sim_backtest_predictions.csv", index=False
        )

    print("\n" + "=" * 70)
    print("BATTER SIMULATOR BACKTEST RESULTS")
    print("=" * 70)

    for _, row in summary.iterrows():
        print(f"\nTest Season {int(row['test_season'])} ({int(row['n_games'])} games):")
        for stat, label in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR"),
                            ("double", "2B"), ("triple", "3B"), ("tb", "TB")]:
            rmse_key = f"{stat}_rmse"
            if rmse_key in row and not pd.isna(row[rmse_key]):
                line = f"  {label:>3s}: RMSE={row[rmse_key]:.3f}"
                mae_key = f"{stat}_mae"
                if mae_key in row:
                    line += f"  MAE={row[mae_key]:.3f}"
                corr_key = f"{stat}_corr"
                if corr_key in row and not pd.isna(row[corr_key]):
                    line += f"  corr={row[corr_key]:.3f}"
                print(line)

        # Brier scores
        brier_labels = [
            ("brier_tb_0.5", "TB>0.5"), ("brier_tb_1.5", "TB>1.5"),
            ("brier_tb_2.5", "TB>2.5"), ("brier_double_0.5", "2B>0.5"),
            ("brier_triple_0.5", "3B>0.5"),
        ]
        printed_brier = False
        for key, label in brier_labels:
            if key in row and not pd.isna(row[key]):
                if not printed_brier:
                    print("  Brier:")
                    printed_brier = True
                print(f"    {label}: {row[key]:.4f}")

    if len(summary) > 0:
        print("\n--- Averages ---")
        for stat, label in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR"),
                            ("double", "2B"), ("triple", "3B"), ("tb", "TB")]:
            rmse_key = f"{stat}_rmse"
            if rmse_key in summary.columns:
                print(f"  {label:>3s}: RMSE={summary[rmse_key].mean():.3f}  "
                      f"MAE={summary[f'{stat}_mae'].mean():.3f}")


if __name__ == "__main__":
    import pandas as pd
    main()
