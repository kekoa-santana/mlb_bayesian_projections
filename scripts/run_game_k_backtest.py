#!/usr/bin/env python
"""
Run walk-forward game-level prediction backtest.

Evaluates the full pipeline for K (Layer 1 + Layer 2 + BF + context),
plus game-level BB, HR, and Outs predictions.

Usage
-----
    python scripts/run_game_k_backtest.py          # full quality, all stats
    python scripts/run_game_k_backtest.py --quick   # fast iteration
    python scripts/run_game_k_backtest.py --k-only  # K only (skip BB/HR/Outs)
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
    run_full_game_backtest,
    run_full_game_k_backtest,
)
from src.evaluation.runner import save_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Game-level prediction backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws for fast iteration")
    parser.add_argument("--k-only", action="store_true",
                        help="Only run K predictions (skip BB/HR/Outs)")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_mc = 500, 250, 2, 1000
    else:
        draws, tune, chains, n_mc = 1000, 500, 2, 2000

    logger.info("=" * 70)
    logger.info("GAME-LEVEL PREDICTION BACKTEST")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC draws per game: %d", n_mc)
    logger.info("  Stats: %s", "K only" if args.k_only else "K + BB + HR + Outs")
    logger.info("=" * 70)

    if args.k_only:
        # Original K-only backtest
        results = run_full_game_k_backtest(
            draws=draws, tune=tune, chains=chains,
            n_mc_draws=n_mc, random_seed=42,
        )
        save_csv(results, "game_k_backtest.csv", logger)

        _print_k_results(results)
    else:
        # Full multi-stat backtest
        k_results, stat_results = run_full_game_backtest(
            draws=draws, tune=tune, chains=chains,
            n_mc_draws=n_mc, random_seed=42,
        )

        save_csv(k_results, "game_k_backtest.csv", logger)
        if not stat_results.empty:
            save_csv(stat_results, "game_stat_backtest.csv", logger)

        _print_k_results(k_results)
        _print_stat_results(stat_results)


def _print_k_results(results: pd.DataFrame) -> None:
    """Print K prediction results."""
    print("\n" + "=" * 70)
    print("GAME K BACKTEST RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))
    print()

    if len(results) > 0:
        print(f"Average Brier: {results['avg_brier'].mean():.4f}")
        if "avg_log_loss" in results.columns:
            print(f"Average Log Loss: {results['avg_log_loss'].mean():.4f}")
        print(f"Coverage 50/80/90: "
              f"{results['coverage_50'].mean():.0%} / "
              f"{results['coverage_80'].mean():.0%} / "
              f"{results['coverage_90'].mean():.0%}")
        if "pooled_ece" in results.columns:
            print(f"Average ECE:   {results['pooled_ece'].mean():.4f}")
            print(f"Average Temperature: {results['pooled_temperature'].mean():.3f}")


def _print_stat_results(stat_results: pd.DataFrame) -> None:
    """Print additional stat prediction results."""
    if stat_results.empty:
        return

    print("\n" + "=" * 70)
    print("GAME-LEVEL COUNTING STAT RESULTS (BB, HR, Outs)")
    print("=" * 70)
    print(stat_results.to_string(index=False, float_format="%.3f"))
    print()

    print("=== PER-STAT AVERAGES ===")
    for stat, group in stat_results.groupby("stat"):
        avg_corr = group["correlation"].mean()
        total_games = group["n_games"].sum()
        print(f"  {stat.upper()}: corr={avg_corr:.3f}, total games={total_games}")


if __name__ == "__main__":
    main()
