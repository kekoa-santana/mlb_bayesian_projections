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
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pitcher_backtest import run_pitcher_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitcher_backtest")

RATE_FOLDS = [
    {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
]

COUNTING_FOLDS = [
    {"train_seasons": list(range(2018, 2022)), "test_season": 2022},
    {"train_seasons": list(range(2018, 2023)), "test_season": 2023},
    {"train_seasons": list(range(2018, 2024)), "test_season": 2024},
    {"train_seasons": list(range(2018, 2025)), "test_season": 2025},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitcher multi-stat backtest")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stat", type=str, default=None,
                        help="Single rate stat to backtest (e.g. k_rate, bb_rate)")
    parser.add_argument("--skip-counting", action="store_true",
                        help="Skip counting stat backtest")
    args = parser.parse_args()

    if args.quick:
        sampling = dict(draws=500, tune=250, chains=2, random_seed=42)
        logger.info("QUICK mode: %s", sampling)
    else:
        sampling = dict(draws=2000, tune=1000, chains=4, random_seed=42)
        logger.info("FULL mode: %s", sampling)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ---- Rate stat backtest ----
    stats = [args.stat] if args.stat else None
    summary = run_pitcher_backtest(stats=stats, folds=RATE_FOLDS, **sampling)

    print("\n=== PITCHER RATE STAT BACKTEST RESULTS ===")
    print(summary.to_string(index=False))
    print()

    print("=== PER-STAT VERDICT (RATE) ===")
    for stat, group in summary.groupby("stat"):
        avg_mae_imp = group["mae_improvement_pct"].mean()
        avg_cov = group["coverage_95"].mean()
        avg_ens_w = group["ensemble_w"].mean()
        avg_ens_mae = group["ensemble_mae"].mean()
        avg_bayes_crps = group["bayes_crps"].mean()
        avg_marcel_crps = group["marcel_crps"].mean()
        beats = avg_mae_imp > 0
        print(f"  {stat}: Bayes {'BEATS' if beats else 'LOSES TO'} Marcel "
              f"(MAE improvement: {avg_mae_imp:+.1f}%, "
              f"95% coverage: {avg_cov:.1%})")
        print(f"    Ensemble: w={avg_ens_w:.2f}, MAE={avg_ens_mae:.4f}")
        print(f"    CRPS: Bayes={avg_bayes_crps:.4f}, Marcel={avg_marcel_crps:.4f}")

    summary.to_csv(out_dir / "pitcher_multi_stat_backtest.csv", index=False)
    logger.info("Rate results saved to outputs/pitcher_multi_stat_backtest.csv")

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
            "bayes_mae", "marcel_mae", "mae_improvement_pct",
            "bayes_corr", "coverage_80", "coverage_95",
        ]
        available = [c for c in display_cols if c in counting_summary.columns]
        print(counting_summary[available].to_string(index=False, float_format="%.2f"))
        print()

        print("=== PER-STAT VERDICT (COUNTING) ===")
        for stat, group in counting_summary.groupby("stat"):
            avg_mae_imp = group["mae_improvement_pct"].mean()
            avg_corr = group["bayes_corr"].mean()
            avg_cov95 = group["coverage_95"].mean()
            beats = avg_mae_imp > 0
            print(
                f"  {stat}: Bayes {'BEATS' if beats else 'LOSES TO'} Marcel "
                f"(MAE improvement: {avg_mae_imp:+.1f}%, "
                f"corr: {avg_corr:.3f}, 95% cov: {avg_cov95:.0%})"
            )
        print()

        counting_summary.to_csv(out_dir / "pitcher_counting_backtest.csv", index=False)
        logger.info("Counting results saved to outputs/pitcher_counting_backtest.csv")


if __name__ == "__main__":
    main()
