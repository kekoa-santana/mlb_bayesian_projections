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
            corr_key = f"{stat}_corr"
            if corr_key in row and pd.notna(row[corr_key]):
                label = stat.upper() if stat != "outs" else "OUTS"
                print(f"  {label:>4s}: corr={row[corr_key]:.3f}")

        if "k_avg_brier" in row and pd.notna(row["k_avg_brier"]):
            print(f"  K Brier: {row['k_avg_brier']:.4f}")
        if "k_avg_log_loss" in row and pd.notna(row["k_avg_log_loss"]):
            print(f"  K Log Loss: {row['k_avg_log_loss']:.4f}")
        if "outs_avg_brier" in row and pd.notna(row["outs_avg_brier"]):
            print(f"  Outs Brier: {row['outs_avg_brier']:.4f}")
        if "outs_avg_log_loss" in row and pd.notna(row["outs_avg_log_loss"]):
            print(f"  Outs Log Loss: {row['outs_avg_log_loss']:.4f}")

        for ci in ("50", "80", "90"):
            key = f"k_coverage_{ci}"
            if key in row and pd.notna(row[key]):
                print(f"  K Coverage {ci}%: {row[key]:.1%}")

        for ci in ("50", "80", "90"):
            key = f"outs_coverage_{ci}"
            if key in row and pd.notna(row[key]):
                print(f"  Outs Coverage {ci}%: {row[key]:.1%}")

        if "ip_corr" in row and pd.notna(row.get("ip_corr")):
            print(f"   IP: corr={row['ip_corr']:.3f}")

        # Run-level metrics
        if "runs_n" in row and pd.notna(row.get("runs_n")):
            print(f"  --- RUNS (n={int(row['runs_n'])}) ---")
            print(f"    bias={row.get('runs_bias', float('nan')):+.3f}  "
                  f"mae={row.get('runs_mae', float('nan')):.3f}  "
                  f"rmse={row.get('runs_rmse', float('nan')):.3f}  "
                  f"corr={row.get('runs_corr', float('nan')):.3f}")
            if pd.notna(row.get("runs_crps_mean")):
                print(f"    CRPS: mean={row['runs_crps_mean']:.3f}  "
                      f"median={row.get('runs_crps_median', float('nan')):.3f}")
            for ci in ("50", "80", "90"):
                k = f"runs_coverage_{ci}"
                if k in row and pd.notna(row[k]):
                    print(f"    Runs Coverage {ci}%: {row[k]:.1%}")
            if pd.notna(row.get("runs_coverage_q80_empirical")):
                print(f"    Runs Coverage (empirical q10-q90, ~80%): "
                      f"{row['runs_coverage_q80_empirical']:.1%}")
            sr = row.get("expected_starter_runs_mean")
            br = row.get("expected_bullpen_runs_mean")
            if pd.notna(sr) and pd.notna(br):
                print(f"    Decomp: starter={sr:.2f}  bullpen={br:.2f}  "
                      f"total={sr + br:.2f}")

        # Sharpness
        for prefix, label in [("k", "K"), ("outs", "Outs")]:
            conf_key = f"{prefix}_sharpness_mean_confidence"
            if conf_key in row and pd.notna(row[conf_key]):
                act60 = row.get(f"{prefix}_sharpness_pct_actionable_60", float("nan"))
                act65 = row.get(f"{prefix}_sharpness_pct_actionable_65", float("nan"))
                act70 = row.get(f"{prefix}_sharpness_pct_actionable_70", float("nan"))
                ent = row.get(f"{prefix}_sharpness_entropy", float("nan"))
                print(f"  {label} Sharpness: conf={row[conf_key]:.3f}  "
                      f"act60={act60:.1f}%  act65={act65:.1f}%  "
                      f"act70={act70:.1f}%  entropy={ent:.3f}")

    # Overall averages
    print("\n--- Overall Averages ---")
    for stat in ("k", "bb", "h", "hr", "outs"):
        corr_key = f"{stat}_corr"
        if corr_key in summary.columns:
            label = stat.upper() if stat != "outs" else "OUTS"
            print(f"  {label:>4s}: corr={summary[corr_key].mean():.3f}")

    if "k_avg_brier" in summary.columns:
        print(f"  K Avg Brier: {summary['k_avg_brier'].mean():.4f}")
    if "k_avg_log_loss" in summary.columns:
        print(f"  K Avg Log Loss: {summary['k_avg_log_loss'].mean():.4f}")
    if "outs_avg_brier" in summary.columns:
        print(f"  Outs Avg Brier: {summary['outs_avg_brier'].mean():.4f}")
    if "outs_avg_log_loss" in summary.columns:
        print(f"  Outs Avg Log Loss: {summary['outs_avg_log_loss'].mean():.4f}")

    # Sharpness averages
    for prefix, label in [("k", "K"), ("outs", "Outs")]:
        conf_col = f"{prefix}_sharpness_mean_confidence"
        if conf_col in summary.columns:
            print(f"  {label} Sharpness: conf={summary[conf_col].mean():.3f}  "
                  f"act60={summary[f'{prefix}_sharpness_pct_actionable_60'].mean():.1f}%  "
                  f"act65={summary[f'{prefix}_sharpness_pct_actionable_65'].mean():.1f}%  "
                  f"act70={summary[f'{prefix}_sharpness_pct_actionable_70'].mean():.1f}%  "
                  f"entropy={summary[f'{prefix}_sharpness_entropy'].mean():.3f}")

    # Comparison targets
    print("\n--- vs Current Layer 3 Baseline ---")
    print("  Layer 3 K Brier: 0.1872")
    if "k_avg_brier" in summary.columns:
        avg_brier = summary["k_avg_brier"].mean()
        brier_delta = avg_brier - 0.1872
        print(f"  Sim K Brier:   {avg_brier:.4f}  (delta: {brier_delta:+.4f})")

    print()


if __name__ == "__main__":
    main()
