#!/usr/bin/env python
"""
Run walk-forward game-level K prediction backtest.

Evaluates the full pipeline: Layer 1 (K% posteriors) + Layer 2 (matchup lifts)
+ BF model + umpire tendencies + weather effects + real lineups.

Usage
-----
    python scripts/run_game_k_backtest.py          # full quality
    python scripts/run_game_k_backtest.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.game_k_validation import (
    compare_to_baselines,
    compute_game_k_metrics,
    run_full_game_k_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Game K backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws for fast iteration")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_mc = 500, 250, 2, 1000
    else:
        draws, tune, chains, n_mc = 1000, 500, 2, 2000

    logger.info("=" * 70)
    logger.info("GAME K BACKTEST — Full Pipeline")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC draws per game: %d", n_mc)
    logger.info("=" * 70)

    results = run_full_game_k_backtest(
        draws=draws, tune=tune, chains=chains,
        n_mc_draws=n_mc, random_seed=42,
    )

    # Save results
    out_path = PROJECT_ROOT / "outputs" / "game_k_backtest.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved backtest results to %s", out_path)

    # Print summary
    print("\n" + "=" * 70)
    print("GAME K BACKTEST RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))
    print()

    # Averages across folds
    if len(results) > 0:
        print(f"Average RMSE:  {results['rmse'].mean():.3f}")
        print(f"Average MAE:   {results['mae'].mean():.3f}")
        print(f"Average Brier: {results['avg_brier'].mean():.4f}")
        print(f"Coverage 50/80/90: "
              f"{results['coverage_50'].mean():.0%} / "
              f"{results['coverage_80'].mean():.0%} / "
              f"{results['coverage_90'].mean():.0%}")


if __name__ == "__main__":
    main()
