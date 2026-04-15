#!/usr/bin/env python
"""
Run walk-forward backtest for counting stat projections.

Tests Bayesian rate × playing time vs Marcel baseline for:
  Hitters: Total Ks, Total BBs, Total HRs, Total SBs
  Pitchers: Total Ks, Total BBs, Total Outs

Usage
-----
    # Quick mode (fewer MCMC draws)
    python scripts/run_counting_backtest.py --quick

    # Hitters only
    python scripts/run_counting_backtest.py --quick --type hitter

    # Pitchers only
    python scripts/run_counting_backtest.py --quick --type pitcher

    # Single stat
    python scripts/run_counting_backtest.py --quick --type hitter --stat total_k

    # Full quality
    python scripts/run_counting_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.counting_backtest import (
    run_hitter_counting_backtest,
    run_pitcher_counting_backtest,
)
from src.evaluation.runner import (
    add_common_args,
    quick_full_sampling,
    save_csv,
    setup_logging,
)

logger = setup_logging("counting_backtest")

FOLDS = [
    {"train_seasons": list(range(2018, 2022)), "test_season": 2022},
    {"train_seasons": list(range(2018, 2023)), "test_season": 2023},
    {"train_seasons": list(range(2018, 2024)), "test_season": 2024},
    {"train_seasons": list(range(2018, 2025)), "test_season": 2025},
]


def _print_summary(summary, player_type: str) -> None:
    """Print formatted summary table and per-stat verdicts."""
    print(f"\n{'=' * 60}")
    print(f"  {player_type.upper()} COUNTING STAT BACKTEST RESULTS")
    print(f"{'=' * 60}")

    # Display key columns
    display_cols = [
        "stat", "test_season", "n_players",
        "bayes_corr", "coverage_80", "coverage_95",
    ]
    available = [c for c in display_cols if c in summary.columns]
    print(summary[available].to_string(index=False, float_format="%.2f"))
    print()

    # Per-stat verdict
    print("=== PER-STAT VERDICT ===")
    for stat, group in summary.groupby("stat"):
        avg_corr = group["bayes_corr"].mean()
        avg_cov80 = group["coverage_80"].mean()
        avg_cov95 = group["coverage_95"].mean()
        print(
            f"  {stat}: "
            f"corr: {avg_corr:.3f}, "
            f"80% cov: {avg_cov80:.0%}, 95% cov: {avg_cov95:.0%}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Counting stat backtest")
    add_common_args(parser, stat=True)
    parser.add_argument("--type", choices=["hitter", "pitcher", "both"],
                        default="both", help="Player type to backtest")
    parser.add_argument("--n-draws", type=int, default=4000,
                        help="Monte Carlo draws for counting distributions")
    args = parser.parse_args()

    sampling = quick_full_sampling(args.quick)
    logger.info("%s mode: %s", "QUICK" if args.quick else "FULL", sampling)

    sampling["n_draws"] = args.n_draws

    # Hitter backtest
    if args.type in ("hitter", "both"):
        stats = [args.stat] if args.stat else None
        hitter_summary = run_hitter_counting_backtest(
            stats=stats, folds=FOLDS, **sampling,
        )
        _print_summary(hitter_summary, "hitter")
        save_csv(hitter_summary, "hitter_counting_backtest.csv", logger)

    # Pitcher backtest
    if args.type in ("pitcher", "both"):
        stats = [args.stat] if args.stat else None
        pitcher_summary = run_pitcher_counting_backtest(
            stats=stats, folds=FOLDS, **sampling,
        )
        _print_summary(pitcher_summary, "pitcher")
        save_csv(pitcher_summary, "pitcher_counting_backtest.csv", logger)


if __name__ == "__main__":
    main()
