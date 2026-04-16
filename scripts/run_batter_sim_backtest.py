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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.batter_sim_validation import run_full_batter_sim_backtest
from src.evaluation.runner import quick_full_game_mcmc, setup_logging

logger = setup_logging(__name__)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batter game simulator backtest"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--single-fold", type=int, default=None)
    parser.add_argument("--max-batters", type=int, default=None,
                        help="Cap batter-games per fold for quick testing")
    args = parser.parse_args()

    draws, tune, chains = quick_full_game_mcmc(args.quick)
    n_sims = 2000 if args.quick else 5000
    max_batters = (args.max_batters or 5000) if args.quick else args.max_batters

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

    _print_results(summary, predictions)


def _print_results(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("BATTER SIMULATOR BACKTEST RESULTS")
    print("=" * 70)

    if len(summary) == 0:
        print("No results.")
        return

    # Per-fold summary
    print("\n--- Per-Fold Results ---")
    for _, row in summary.iterrows():
        print(f"\nTest Season {int(row['test_season'])} ({int(row['n_games'])} games):")
        for stat, label in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR"),
                            ("double", "2B"), ("triple", "3B"), ("tb", "TB")]:
            corr_key = f"{stat}_corr"
            if corr_key in row and not pd.isna(row.get(corr_key)):
                print(f"  {label:>3s}: corr={row[corr_key]:.3f}")

        # Brier scores
        brier_labels = [
            ("brier_k_0.5", "K>0.5"), ("brier_k_1.5", "K>1.5"),
            ("brier_h_0.5", "H>0.5"), ("brier_h_1.5", "H>1.5"),
            ("brier_hr_0.5", "HR>0.5"),
            ("brier_tb_0.5", "TB>0.5"), ("brier_tb_1.5", "TB>1.5"),
            ("brier_tb_2.5", "TB>2.5"), ("brier_tb_3.5", "TB>3.5"),
            ("brier_double_0.5", "2B>0.5"),
            ("brier_triple_0.5", "3B>0.5"),
        ]
        printed_brier = False
        for key, label in brier_labels:
            if key in row and not pd.isna(row[key]):
                if not printed_brier:
                    print("  Brier:")
                    printed_brier = True
                print(f"    {label}: {row[key]:.4f}")

        # Log loss
        logloss_labels = [
            ("logloss_k_0.5", "K>0.5"), ("logloss_k_1.5", "K>1.5"),
            ("logloss_h_0.5", "H>0.5"), ("logloss_h_1.5", "H>1.5"),
            ("logloss_hr_0.5", "HR>0.5"),
            ("logloss_tb_0.5", "TB>0.5"), ("logloss_tb_1.5", "TB>1.5"),
            ("logloss_tb_2.5", "TB>2.5"), ("logloss_tb_3.5", "TB>3.5"),
            ("logloss_double_0.5", "2B>0.5"),
            ("logloss_triple_0.5", "3B>0.5"),
        ]
        printed_ll = False
        for key, label in logloss_labels:
            if key in row and not pd.isna(row[key]):
                if not printed_ll:
                    print("  Log Loss:")
                    printed_ll = True
                print(f"    {label}: {row[key]:.4f}")

        # Coverage
        for stat, label in [("k", "K"), ("h", "H")]:
            has_cov = False
            for ci in ("50", "80", "90"):
                key = f"{stat}_coverage_{ci}"
                if key in row and not pd.isna(row[key]):
                    if not has_cov:
                        has_cov = True
                    print(f"  {label} Coverage {ci}%: {row[key]:.1%}")

        # Sharpness
        for prefix, label in [("k", "K"), ("h", "H"), ("hr", "HR"), ("tb", "TB")]:
            conf_key = f"{prefix}_sharpness_mean_confidence"
            if conf_key in row and not pd.isna(row[conf_key]):
                act60 = row.get(f"{prefix}_sharpness_pct_actionable_60", float("nan"))
                act65 = row.get(f"{prefix}_sharpness_pct_actionable_65", float("nan"))
                act70 = row.get(f"{prefix}_sharpness_pct_actionable_70", float("nan"))
                ent = row.get(f"{prefix}_sharpness_entropy", float("nan"))
                print(f"  {label} Sharpness: conf={row[conf_key]:.3f}  "
                      f"act60={act60:.1f}%  act65={act65:.1f}%  "
                      f"act70={act70:.1f}%  entropy={ent:.3f}")

    # Overall averages
    if len(summary) == 0:
        return

    print("\n--- Overall Averages ---")
    for stat, label in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR"),
                        ("double", "2B"), ("triple", "3B"), ("tb", "TB")]:
        corr_key = f"{stat}_corr"
        if corr_key in summary.columns:
            print(f"  {label:>3s}: corr={summary[corr_key].mean():.3f}")

    # Avg Brier and log loss per stat
    for stat, label in [("k", "K"), ("h", "H"), ("hr", "HR"), ("tb", "TB")]:
        brier_key = f"{stat}_avg_brier"
        ll_key = f"{stat}_avg_log_loss"
        if brier_key in summary.columns:
            print(f"  {label} Avg Brier: {summary[brier_key].mean():.4f}", end="")
            if ll_key in summary.columns:
                print(f"  Log Loss: {summary[ll_key].mean():.4f}")
            else:
                print()

    # Sharpness averages
    for prefix, label in [("k", "K"), ("h", "H"), ("hr", "HR"), ("tb", "TB")]:
        conf_col = f"{prefix}_sharpness_mean_confidence"
        if conf_col in summary.columns:
            print(f"  {label} Sharpness: conf={summary[conf_col].mean():.3f}  "
                  f"act60={summary[f'{prefix}_sharpness_pct_actionable_60'].mean():.1f}%  "
                  f"act65={summary[f'{prefix}_sharpness_pct_actionable_65'].mean():.1f}%  "
                  f"act70={summary[f'{prefix}_sharpness_pct_actionable_70'].mean():.1f}%  "
                  f"entropy={summary[f'{prefix}_sharpness_entropy'].mean():.3f}")

    print()


if __name__ == "__main__":
    main()
