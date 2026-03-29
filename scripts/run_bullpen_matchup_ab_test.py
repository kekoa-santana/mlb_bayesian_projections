#!/usr/bin/env python
"""A/B backtest: flat bullpen rates vs per-batter bullpen matchup lifts.

Runs two variants of the batter game sim backtest and compares
H (hits) metrics: MAE, RMSE, Brier score, and bias.

Usage
-----
    python scripts/run_bullpen_matchup_ab_test.py
    python scripts/run_bullpen_matchup_ab_test.py --fold 2025      # single fold
    python scripts/run_bullpen_matchup_ab_test.py --max-games 2000  # quick test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bullpen_ab_test")

from src.evaluation.batter_sim_validation import (
    build_batter_sim_predictions,
    compute_batter_sim_metrics,
)


def run_ab_test(
    folds: list[tuple[list[int], int]],
    max_games: int | None = None,
    n_sims: int = 5000,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
) -> None:
    """Run A/B backtest comparing flat vs matchup-adjusted bullpen."""
    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info("FOLD: train=%s, test=%d", train_seasons, test_season)
        logger.info("=" * 60)

        # --- A: Flat bullpen (baseline) ---
        logger.info("Running FLAT bullpen (baseline)...")
        t0 = time.time()
        preds_flat = build_batter_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws, tune=tune, chains=chains,
            n_sims=n_sims,
            enable_bullpen_matchup=False,
        )
        if max_games and len(preds_flat) > max_games:
            preds_flat = preds_flat.head(max_games)
        t_flat = time.time() - t0
        logger.info("Flat: %d predictions in %.1fs", len(preds_flat), t_flat)

        # --- B: Bullpen matchup lifts ---
        logger.info("Running BULLPEN MATCHUP (new)...")
        t0 = time.time()
        preds_matchup = build_batter_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws, tune=tune, chains=chains,
            n_sims=n_sims,
            enable_bullpen_matchup=True,
        )
        if max_games and len(preds_matchup) > max_games:
            preds_matchup = preds_matchup.head(max_games)
        t_matchup = time.time() - t0
        logger.info("Matchup: %d predictions in %.1fs", len(preds_matchup), t_matchup)

        # --- Compare ---
        metrics_flat = compute_batter_sim_metrics(preds_flat)
        metrics_matchup = compute_batter_sim_metrics(preds_matchup)

        logger.info("")
        logger.info("=" * 60)
        logger.info("RESULTS: test=%d  (n=%d flat, n=%d matchup)",
                     test_season, metrics_flat["n_games"], metrics_matchup["n_games"])
        logger.info("=" * 60)

        # Print comparison table
        stats = ["h", "k", "bb", "hr", "tb"]
        metric_types = ["mae", "rmse", "bias", "corr"]

        header = f"{'Stat':<6} {'Metric':<8} {'Flat':>8} {'Matchup':>8} {'Delta':>8} {'%Chg':>7}"
        logger.info(header)
        logger.info("-" * len(header))

        for stat in stats:
            for mt in metric_types:
                key = f"{stat}_{mt}"
                v_flat = metrics_flat.get(key)
                v_matchup = metrics_matchup.get(key)
                if v_flat is None or v_matchup is None:
                    continue
                delta = v_matchup - v_flat
                pct = (delta / abs(v_flat) * 100) if v_flat != 0 else 0
                # For MAE/RMSE, negative delta = improvement
                logger.info(
                    f"{stat:<6} {mt:<8} {v_flat:>8.4f} {v_matchup:>8.4f} {delta:>+8.4f} {pct:>+6.1f}%"
                )

        # Brier scores
        logger.info("")
        logger.info("Brier Scores:")
        brier_flat = metrics_flat.get("brier_scores", {})
        brier_matchup = metrics_matchup.get("brier_scores", {})
        for label in sorted(set(brier_flat) | set(brier_matchup)):
            bf = brier_flat.get(label)
            bm = brier_matchup.get(label)
            if bf is None or bm is None:
                continue
            delta = bm - bf
            pct = (delta / abs(bf) * 100) if bf != 0 else 0
            logger.info(
                f"  {label:<12} Flat={bf:.4f}  Matchup={bm:.4f}  Delta={delta:+.4f} ({pct:+.1f}%)"
            )

        # Save predictions for further analysis
        out_flat = f"outputs/batter_sim_flat_{test_season}.csv"
        out_matchup = f"outputs/batter_sim_bp_matchup_{test_season}.csv"
        preds_flat.to_csv(out_flat, index=False)
        preds_matchup.to_csv(out_matchup, index=False)
        logger.info("Saved: %s, %s", out_flat, out_matchup)


def main():
    parser = argparse.ArgumentParser(description="Bullpen matchup A/B backtest")
    parser.add_argument("--fold", type=int, choices=[2024, 2025],
                        help="Run single fold (2024 or 2025)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap batter-games per fold for quick testing")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    if args.fold == 2024:
        folds = [([2020, 2021, 2022, 2023], 2024)]
    elif args.fold == 2025:
        folds = [([2020, 2021, 2022, 2023, 2024], 2025)]
    else:
        folds = [
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]

    run_ab_test(
        folds=folds,
        max_games=args.max_games,
        n_sims=args.n_sims,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
    )


if __name__ == "__main__":
    main()
