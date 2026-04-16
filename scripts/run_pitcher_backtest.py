#!/usr/bin/env python
"""
Run walk-forward backtest for all pitcher stats.

Rate stats: K%, BB%, HR/BF (Bayesian vs Marcel, ensemble, CRPS, PPC)
Counting stats: Total Ks, Total BBs, Total Outs

Usage
-----
    # Quick (fewer draws, faster but noisier)
    python scripts/run_pitcher_backtest.py --quick

    # Single rate stat only
    python scripts/run_pitcher_backtest.py --quick --stat k_rate

    # Skip counting stats
    python scripts/run_pitcher_backtest.py --quick --skip-counting

    # Full quality
    python scripts/run_pitcher_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pitcher_backtest import run_pitcher_backtest
from src.evaluation.runner import (
    COUNTING_FOLDS,
    RATE_FOLDS,
    add_common_args,
    quick_full_sampling,
    save_csv,
    setup_logging,
)

logger = setup_logging("pitcher_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitcher multi-stat backtest")
    add_common_args(parser, stat=True, skip_counting=True)
    args = parser.parse_args()

    sampling = quick_full_sampling(args.quick)
    logger.info("%s mode: %s", "QUICK" if args.quick else "FULL", sampling)

    # ---- Rate stat backtest ----
    stats = [args.stat] if args.stat else None
    summary = run_pitcher_backtest(stats=stats, folds=RATE_FOLDS, **sampling)

    print("\n=== PITCHER RATE STAT BACKTEST RESULTS ===")
    print(summary.to_string(index=False))
    print()

    print("=== PER-STAT VERDICT (RATE) ===")
    for stat, group in summary.groupby("stat"):
        avg_cov = group["coverage_95"].mean()
        avg_ens_w = group["ensemble_w"].mean()
        avg_bayes_crps = group["bayes_crps"].mean()
        avg_marcel_crps = group["marcel_crps"].mean()
        crps_beats = avg_bayes_crps < avg_marcel_crps
        print(f"  {stat}: Bayes {'BEATS' if crps_beats else 'LOSES TO'} Marcel on CRPS "
              f"(95% coverage: {avg_cov:.1%})")
        print(f"    Ensemble: w={avg_ens_w:.2f}")
        print(f"    CRPS: Bayes={avg_bayes_crps:.4f}, Marcel={avg_marcel_crps:.4f}")

    save_csv(summary, "pitcher_multi_stat_backtest.csv", logger)

    # ---- Counting stat backtest ----
    if not args.skip_counting and args.stat is None:
        from src.evaluation.counting_backtest import run_pitcher_counting_backtest

        counting_sampling = dict(
            draws=sampling["draws"],
            tune=sampling["tune"],
            chains=sampling["chains"],
            random_seed=sampling["random_seed"],
            n_draws=4000,
        )

        counting_summary = run_pitcher_counting_backtest(
            folds=COUNTING_FOLDS, **counting_sampling,
        )

        print("\n=== PITCHER COUNTING STAT BACKTEST RESULTS ===")
        display_cols = [
            "stat", "test_season", "n_players",
            "bayes_corr", "coverage_80", "coverage_95",
        ]
        available = [c for c in display_cols if c in counting_summary.columns]
        print(counting_summary[available].to_string(index=False, float_format="%.2f"))
        print()

        print("=== PER-STAT VERDICT (COUNTING) ===")
        for stat, group in counting_summary.groupby("stat"):
            avg_corr = group["bayes_corr"].mean()
            avg_cov95 = group["coverage_95"].mean()
            print(
                f"  {stat}: "
                f"corr: {avg_corr:.3f}, 95% cov: {avg_cov95:.0%}"
            )
        print()

        save_csv(counting_summary, "pitcher_counting_backtest.csv", logger)


if __name__ == "__main__":
    main()
