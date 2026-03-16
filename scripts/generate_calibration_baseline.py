#!/usr/bin/env python
"""Generate calibration baseline snapshot for all game prop types.

Runs the full walk-forward game prop backtest for every supported prop
(pitcher K/BB/HR/H/Outs, batter K/BB/HR/H), aggregates metrics across
folds, and saves a per-prop-type summary to outputs/baseline_calibration.csv.

Usage
-----
python scripts/generate_calibration_baseline.py
python scripts/generate_calibration_baseline.py --draws 500 --tune 250 --chains 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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

# Metrics to aggregate across folds (mean)
AGG_METRICS = [
    "n_games", "rmse", "mae", "avg_brier", "crps",
    "ece", "mce", "temperature",
    "coverage_50", "coverage_80", "coverage_90", "coverage_95",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration baseline for all game props"
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of all props to run
    configs_to_run: list[tuple[str, object]] = []
    for name, config in PITCHER_PROP_CONFIGS.items():
        configs_to_run.append((f"pitcher_{name}", config))
    for name, config in BATTER_PROP_CONFIGS.items():
        configs_to_run.append((f"batter_{name}", config))

    all_fold_summaries: list[pd.DataFrame] = []
    aggregated_rows: list[dict] = []

    for label, config in configs_to_run:
        logger.info("=" * 70)
        logger.info("BASELINE BACKTEST: %s", label)
        logger.info("=" * 70)

        try:
            summary_df, predictions_df = run_full_game_prop_backtest(
                config=config,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                n_mc_draws=args.mc_draws,
                random_seed=args.seed,
            )

            if summary_df.empty:
                logger.warning("No results for %s — skipping", label)
                continue

            # Tag with prop label
            summary_df["prop"] = label
            summary_df["side"] = config.side
            summary_df["stat"] = config.stat_name
            all_fold_summaries.append(summary_df)

            # Aggregate across folds: mean of each metric
            agg_row: dict = {
                "prop": label,
                "side": config.side,
                "stat": config.stat_name,
                "n_folds": len(summary_df),
            }
            for metric in AGG_METRICS:
                if metric in summary_df.columns:
                    if metric == "n_games":
                        agg_row[metric] = int(summary_df[metric].sum())
                    else:
                        agg_row[metric] = float(summary_df[metric].mean())
                else:
                    agg_row[metric] = np.nan

            aggregated_rows.append(agg_row)

            logger.info(
                "  %s aggregate: RMSE=%.3f  MAE=%.3f  Brier=%.4f  "
                "ECE=%.4f  Temp=%.3f  Cov50/80/90=%.2f/%.2f/%.2f  "
                "n_games=%d",
                label,
                agg_row["rmse"], agg_row["mae"], agg_row["avg_brier"],
                agg_row["ece"], agg_row["temperature"],
                agg_row["coverage_50"], agg_row["coverage_80"],
                agg_row["coverage_90"], agg_row["n_games"],
            )

        except Exception:
            logger.exception("FAILED backtest for %s", label)
            continue

    if not aggregated_rows:
        logger.error("No backtests completed — no baseline file written")
        return

    # Build output DataFrame
    baseline_df = pd.DataFrame(aggregated_rows)
    baseline_df["generated_at"] = datetime.utcnow().isoformat(timespec="seconds")

    # Column ordering
    col_order = [
        "prop", "side", "stat", "n_folds", "n_games",
        "rmse", "mae", "avg_brier", "crps",
        "ece", "mce", "temperature",
        "coverage_50", "coverage_80", "coverage_90", "coverage_95",
        "generated_at",
    ]
    col_order = [c for c in col_order if c in baseline_df.columns]
    baseline_df = baseline_df[col_order]

    # Save CSV
    csv_path = OUTPUT_DIR / "baseline_calibration.csv"
    baseline_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("Saved baseline calibration to %s", csv_path)

    # Also save per-fold detail
    if all_fold_summaries:
        fold_detail = pd.concat(all_fold_summaries, ignore_index=True)
        detail_path = OUTPUT_DIR / "baseline_calibration_folds.csv"
        fold_detail.to_csv(detail_path, index=False, float_format="%.6f")
        logger.info("Saved per-fold detail to %s", detail_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("CALIBRATION BASELINE SNAPSHOT")
    print("=" * 90)

    display_cols = [
        "prop", "n_folds", "n_games", "rmse", "mae", "avg_brier",
        "ece", "temperature", "coverage_50", "coverage_80", "coverage_90",
    ]
    available = [c for c in display_cols if c in baseline_df.columns]

    # Format for display
    display_df = baseline_df[available].copy()
    for col in display_df.columns:
        if display_df[col].dtype == float:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))
    print("=" * 90)
    print(f"\nBaseline saved to: {csv_path}")


if __name__ == "__main__":
    main()
