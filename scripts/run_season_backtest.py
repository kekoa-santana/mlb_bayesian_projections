#!/usr/bin/env python
"""
Step 10: Run walk-forward season-level K% backtests for hitters and pitchers.

Confirms the Bayesian model beats Marcel on calibration and distributional metrics.

Usage
-----
    # Quick (fewer draws, faster but noisier)
    python scripts/run_season_backtest.py --quick

    # Full quality
    python scripts/run_season_backtest.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtesting import run_full_backtest, run_full_pitcher_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("season_backtest")

FOLDS = [
    {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Season-level K% backtest")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        sampling = dict(draws=500, tune=250, chains=2, random_seed=42)
        logger.info("QUICK mode: %s", sampling)
    else:
        sampling = dict(draws=2000, tune=1000, chains=4, random_seed=42)
        logger.info("FULL mode: %s", sampling)

    # --- Hitter K% backtest ---
    logger.info("=" * 60)
    logger.info("HITTER K%% BACKTEST")
    logger.info("=" * 60)
    hitter_summary = run_full_backtest(folds=FOLDS, **sampling)
    print("\n=== HITTER K% BACKTEST RESULTS ===")
    print(hitter_summary.to_string(index=False))
    print()

    # --- Pitcher K% backtest ---
    logger.info("=" * 60)
    logger.info("PITCHER K%% BACKTEST")
    logger.info("=" * 60)
    pitcher_summary = run_full_pitcher_backtest(folds=FOLDS, **sampling)
    print("\n=== PITCHER K% BACKTEST RESULTS ===")
    print(pitcher_summary.to_string(index=False))
    print()

    # --- Summary ---
    print("=== VERDICT ===")
    for label, df in [("Hitter", hitter_summary), ("Pitcher", pitcher_summary)]:
        avg_cov = df["coverage_95"].mean()
        avg_brier = df["brier_score"].mean() if "brier_score" in df.columns else float("nan")
        print(f"{label}: 95% coverage: {avg_cov:.1%}, "
              f"Brier: {avg_brier:.4f}")

    # Save results
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    hitter_summary.to_csv(out_dir / "hitter_k_backtest.csv", index=False)
    pitcher_summary.to_csv(out_dir / "pitcher_k_backtest.csv", index=False)
    logger.info("Results saved to outputs/")


if __name__ == "__main__":
    main()
