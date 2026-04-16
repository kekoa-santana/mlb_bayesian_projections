#!/usr/bin/env python
"""
Backtest the lineup simulator against actual batter game results.

Pulls historical game data (lineups, boxscores, opposing starters) from the
database, runs simulate_lineup_game for each game using pre-computed posterior
samples, and compares predicted vs actual per-batter stats.

Reports calibration tables and P(over) accuracy for H, R, RBI, HRR, K, BB.

Usage
-----
    python scripts/precompute/backtest_lineup_sim.py
    python scripts/precompute/backtest_lineup_sim.py --max-games 200
    python scripts/precompute/backtest_lineup_sim.py --season 2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.evaluation.runner import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute import DASHBOARD_DIR
from scripts.precompute.backtest_harness import (
    ALL_STATS,
    DEFAULT_BP_K_RATE as DEFAULT_BP_K,
    DEFAULT_BP_BB_RATE as DEFAULT_BP_BB,
    DEFAULT_BP_HR_RATE as DEFAULT_BP_HR,
    N_SIMS,
    PRIMARY_LINE,
    PROP_LINES,
    fetch_backtest_games,
    pool_predictions,
    report_calibration_buckets,
    report_per_stat_calibration,
    run_sides_loop,
    simulate_one_side as _harness_simulate_one_side,
)

logger = setup_logging(__name__)
def load_posteriors() -> dict:
    """Load all pre-computed posterior NPZ files and supporting data."""
    logger.info("Loading posterior samples and supporting data...")

    data = {}
    data["hitter_k"] = np.load(DASHBOARD_DIR / "hitter_k_samples.npz")
    data["hitter_bb"] = np.load(DASHBOARD_DIR / "hitter_bb_samples.npz")
    data["hitter_hr"] = np.load(DASHBOARD_DIR / "hitter_hr_samples.npz")
    data["pitcher_k"] = np.load(DASHBOARD_DIR / "pitcher_k_samples.npz")
    data["pitcher_bb"] = np.load(DASHBOARD_DIR / "pitcher_bb_samples.npz")
    data["pitcher_hr"] = np.load(DASHBOARD_DIR / "pitcher_hr_samples.npz")

    data["bf_priors"] = pd.read_parquet(DASHBOARD_DIR / "bf_priors.parquet")
    data["bullpen_rates"] = pd.read_parquet(
        DASHBOARD_DIR / "team_bullpen_rates.parquet"
    )

    # Index bullpen rates by team_id (latest season)
    bp = data["bullpen_rates"].sort_values("season").groupby("team_id").last().reset_index()
    data["bp_lookup"] = {
        int(row["team_id"]): {
            "k_rate": float(row["k_rate"]),
            "bb_rate": float(row["bb_rate"]),
            "hr_rate": float(row["hr_rate"]),
        }
        for _, row in bp.iterrows()
    }

    # Index BF priors by pitcher_id (latest season)
    bf = data["bf_priors"].sort_values("season").groupby("pitcher_id").last().reset_index()
    data["bf_lookup"] = {
        int(row["pitcher_id"]): {
            "mu_bf": float(row["mu_bf"]),
            "sigma_bf": float(row["sigma_bf"]),
        }
        for _, row in bf.iterrows()
    }

    hitter_ids = set(data["hitter_k"].files)
    pitcher_ids = set(data["pitcher_k"].files)
    logger.info(
        "Loaded posteriors: %d hitters, %d pitchers, %d BF priors, %d bullpen teams",
        len(hitter_ids), len(pitcher_ids), len(data["bf_lookup"]),
        len(data["bp_lookup"]),
    )
    return data


def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run the lineup simulator for one team-side of a game.

    Thin wrapper around the harness ``simulate_one_side``.
    """
    return _harness_simulate_one_side(
        side_df, posteriors, prop_lines=PROP_LINES, n_sims=n_sims,
    )


def run_backtest(
    seasons: list[int],
    max_games: int | None = None,
    n_sims: int = N_SIMS,
) -> pd.DataFrame:
    """Run the full backtest and return per-batter predictions."""

    posteriors = load_posteriors()
    games_df = fetch_backtest_games(seasons, max_games=max_games)

    if games_df.empty:
        logger.error("No games found for backtest")
        return pd.DataFrame()

    sides = list(games_df.groupby(["game_pk", "team_id"]))
    logger.info("Simulating %d team-game sides with %d sims each...", len(sides), n_sims)

    df, _skipped = run_sides_loop(
        label="Lineup",
        sides=sides,
        simulate_fn=lambda sdf: simulate_one_side(sdf, posteriors, n_sims=n_sims),
    )
    return df


def compute_metrics(df: pd.DataFrame) -> None:
    """Compute and print all backtest metrics."""

    # Filter to only batters with real posteriors for headline metrics
    df_real = df[df["has_posterior"]].copy()
    n_total = len(df)
    n_real = len(df_real)
    n_games = df["game_pk"].nunique()

    print()
    print("=" * 72)
    print("LINEUP SIMULATOR BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Total batter-games:       {n_total:,}")
    print(f"  With real posteriors:      {n_real:,}")
    print(f"  Unique games:             {n_games:,}")
    print(f"  Seasons:                  {sorted(df['season'].unique())}")
    print()

    # --- Bias per stat ---
    print("-" * 72)
    print("BIAS (batters with posteriors only)")
    print("-" * 72)
    stats = ["h", "r", "rbi", "hrr", "k", "bb"]
    for stat in stats:
        actual = df_real[f"actual_{stat}"]
        pred = df_real[f"pred_{stat}"]
        bias = float(np.mean(pred - actual))
        print(f"  {stat.upper():>4s}:  Bias = {bias:+.4f}   "
              f"Actual mean = {actual.mean():.3f}   Pred mean = {pred.mean():.3f}")

    print()

    # --- Calibration at primary line per stat ---
    print("-" * 72)
    print("CALIBRATION AT PRIMARY PROP LINE")
    print("-" * 72)
    print(f"  {'Stat':>4s}  {'Line':>5s}  {'P(over) mean':>13s}  "
          f"{'Actual hit%':>12s}  {'N bets':>7s}")
    print(f"  {'----':>4s}  {'-----':>5s}  {'-------------':>13s}  "
          f"{'------------':>12s}  {'------':>7s}")

    for stat in stats:
        line = PRIMARY_LINE[stat]
        col = f"p_over_{stat}_{line}"
        if col not in df_real.columns:
            continue
        p_over = df_real[col]
        actual_over = (df_real[f"actual_{stat}"] > line).astype(float)
        avg_p = float(p_over.mean())
        actual_rate = float(actual_over.mean())
        n = len(p_over)
        print(f"  {stat.upper():>4s}  {line:>5.1f}  {avg_p:>13.4f}  "
              f"{actual_rate:>12.4f}  {n:>7,}")

    print()

    # --- Calibration buckets ---
    print("-" * 72)
    print("CALIBRATION BY CONFIDENCE BUCKET (all stats pooled)")
    print("-" * 72)

    all_p, all_a = pool_predictions(df_real, stats, PROP_LINES)

    report_calibration_buckets(all_p, all_a, label="Over side")

    # Also do under side
    print()
    print("  (Under side)")
    all_p_under = 1 - all_p
    all_a_under = 1 - all_a
    report_calibration_buckets(all_p_under, all_a_under, label="Under side")

    print()

    # --- Headline: >= 63% confidence accuracy ---
    print("-" * 72)
    print("HEADLINE: ACCURACY AT >= 63% CONFIDENCE")
    print("-" * 72)

    mask_63 = all_p >= 0.63
    if mask_63.sum() > 0:
        hit_rate = float(all_a[mask_63].mean())
        n_63 = mask_63.sum()
        print(f"  When model says P(over) >= 63%%: hit rate = {hit_rate:.4f} "
              f"({hit_rate*100:.1f}%) on {n_63:,} predictions")
    else:
        print("  No predictions at >= 63% confidence")

    mask_63_under = all_p_under >= 0.63
    if mask_63_under.sum() > 0:
        hit_rate_u = float(all_a_under[mask_63_under].mean())
        n_63_u = mask_63_under.sum()
        print(f"  When model says P(under) >= 63%%: hit rate = {hit_rate_u:.4f} "
              f"({hit_rate_u*100:.1f}%) on {n_63_u:,} predictions")

    print()

    # --- Per-stat calibration at all lines ---
    print("-" * 72)
    print("PER-STAT CALIBRATION AT EACH PROP LINE")
    print("-" * 72)

    report_per_stat_calibration(df_real, stats, PROP_LINES, PRIMARY_LINE)

    print()
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest lineup simulator against actual game results"
    )
    parser.add_argument(
        "--season", type=int, nargs="+", default=[2024, 2025],
        help="Season(s) to backtest (default: 2024 2025)",
    )
    parser.add_argument(
        "--max-games", type=int, default=600,
        help="Max team-game sides to simulate (default: 600)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=N_SIMS,
        help="Monte Carlo sims per game (default: 10000)",
    )
    args = parser.parse_args()

    df = run_backtest(
        seasons=args.season,
        max_games=args.max_games,
        n_sims=args.n_sims,
    )

    if df.empty:
        logger.error("No results to report")
        return

    # Save detailed results
    out_path = DASHBOARD_DIR / "backtest_lineup_sim.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved detailed results to %s (%d rows)", out_path, len(df))

    compute_metrics(df)


if __name__ == "__main__":
    main()
