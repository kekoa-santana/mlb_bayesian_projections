#!/usr/bin/env python
"""
Head-to-head backtest: OLD batter sim vs NEW lineup sim.

Runs both simulators on the same regular-season games and compares
P(over) accuracy, calibration, and per-stat breakdowns.

Usage
-----
    python scripts/precompute/backtest_head_to_head.py
    python scripts/precompute/backtest_head_to_head.py --max-games 400
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.game_sim.lineup_simulator import simulate_lineup_game
from src.models.game_sim.batter_simulator import simulate_batter_game

# Reuse helpers from the existing backtest
from scripts.precompute.backtest_lineup_sim import (
    load_posteriors,
    fetch_backtest_games,
    DEFAULT_BP_K,
    DEFAULT_BP_BB,
    DEFAULT_BP_HR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

N_SIMS = 10_000
BATCH_SIZE = 25

# Stats to compare (excluding HRR per user request)
STATS = ["h", "k", "bb", "r", "rbi"]

# Primary prop line for each stat
PRIMARY_LINE = {"h": 0.5, "k": 0.5, "bb": 0.5, "r": 0.5, "rbi": 0.5}


def simulate_one_side_both(
    side_df: pd.DataFrame,
    posteriors: dict,
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run BOTH simulators for one team-side, returning per-batter records."""
    side_df = side_df.sort_values("batting_order")
    batter_ids = side_df["batter_id"].astype(int).tolist()
    batting_orders = side_df["batting_order"].astype(int).tolist()
    opp_starter_id = int(side_df.iloc[0]["opp_starter_id"])
    opp_team_id = int(side_df.iloc[0]["opp_team_id"])
    pid_str = str(opp_starter_id)

    # Check pitcher posteriors
    for npz_key in ["pitcher_k", "pitcher_bb", "pitcher_hr"]:
        if pid_str not in posteriors[npz_key].files:
            return None

    # Build batter sample lists
    batter_k_samples = []
    batter_bb_samples = []
    batter_hr_samples = []
    valid_count = 0

    for bid in batter_ids:
        bid_str = str(bid)
        has_k = bid_str in posteriors["hitter_k"].files
        has_bb = bid_str in posteriors["hitter_bb"].files
        has_hr = bid_str in posteriors["hitter_hr"].files

        if has_k and has_bb and has_hr:
            batter_k_samples.append(posteriors["hitter_k"][bid_str])
            batter_bb_samples.append(posteriors["hitter_bb"][bid_str])
            batter_hr_samples.append(posteriors["hitter_hr"][bid_str])
            valid_count += 1
        else:
            rng_fb = np.random.default_rng(bid % 100000)
            batter_k_samples.append(
                np.clip(rng_fb.normal(0.226, 0.03, 2000), 0.05, 0.50)
            )
            batter_bb_samples.append(
                np.clip(rng_fb.normal(0.082, 0.02, 2000), 0.02, 0.25)
            )
            batter_hr_samples.append(
                np.clip(rng_fb.normal(0.031, 0.01, 2000), 0.005, 0.10)
            )

    if valid_count < 5:
        return None

    # Pitcher rates (posterior means)
    starter_k = float(np.mean(posteriors["pitcher_k"][pid_str]))
    starter_bb = float(np.mean(posteriors["pitcher_bb"][pid_str]))
    starter_hr = float(np.mean(posteriors["pitcher_hr"][pid_str]))

    # BF priors
    bf_info = posteriors["bf_lookup"].get(opp_starter_id)
    mu_bf = bf_info["mu_bf"] if bf_info else 22.0
    sigma_bf = bf_info["sigma_bf"] if bf_info else 3.4

    # Bullpen rates
    bp_info = posteriors["bp_lookup"].get(opp_team_id)
    bp_k = bp_info["k_rate"] if bp_info else DEFAULT_BP_K
    bp_bb = bp_info["bb_rate"] if bp_info else DEFAULT_BP_BB
    bp_hr = bp_info["hr_rate"] if bp_info else DEFAULT_BP_HR

    game_pk = int(side_df.iloc[0]["game_pk"])
    seed = game_pk % (2**31)

    # --- NEW: lineup sim ---
    new_result = simulate_lineup_game(
        batter_k_rate_samples=batter_k_samples,
        batter_bb_rate_samples=batter_bb_samples,
        batter_hr_rate_samples=batter_hr_samples,
        starter_k_rate=starter_k,
        starter_bb_rate=starter_bb,
        starter_hr_rate=starter_hr,
        starter_bf_mu=mu_bf,
        starter_bf_sigma=sigma_bf,
        bullpen_k_rate=bp_k,
        bullpen_bb_rate=bp_bb,
        bullpen_hr_rate=bp_hr,
        n_sims=n_sims,
        random_seed=seed,
    )

    # --- OLD: per-batter sim (one call per batter) ---
    old_results = []
    for i in range(9):
        old_res = simulate_batter_game(
            batter_k_rate_samples=batter_k_samples[i],
            batter_bb_rate_samples=batter_bb_samples[i],
            batter_hr_rate_samples=batter_hr_samples[i],
            batting_order=batting_orders[i],
            starter_k_rate=starter_k,
            starter_bb_rate=starter_bb,
            starter_hr_rate=starter_hr,
            starter_bf_mu=mu_bf,
            starter_bf_sigma=sigma_bf,
            bullpen_k_rate=bp_k,
            bullpen_bb_rate=bp_bb,
            bullpen_hr_rate=bp_hr,
            n_sims=n_sims,
            random_seed=seed + i,
        )
        old_results.append(old_res)

    # --- Collect per-batter records ---
    records = []
    for i, (_, row) in enumerate(side_df.iterrows()):
        bid = int(row["batter_id"])
        bid_str = str(bid)
        has_posterior = (
            bid_str in posteriors["hitter_k"].files
            and bid_str in posteriors["hitter_bb"].files
            and bid_str in posteriors["hitter_hr"].files
        )

        new_br = new_result.batter_result(i)
        old_br = old_results[i]

        rec = {
            "game_pk": game_pk,
            "season": int(row["season"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "has_posterior": has_posterior,
        }

        for stat in STATS:
            line = PRIMARY_LINE[stat]
            actual_val = int(row[f"actual_{stat}"])
            rec[f"actual_{stat}"] = actual_val

            # NEW model P(over)
            new_samples = new_br[f"{stat}_samples"]
            new_p_over = float(np.mean(new_samples > line))
            rec[f"new_p_over_{stat}"] = new_p_over

            # OLD model P(over)
            old_samples = getattr(old_br, f"{stat}_samples")
            old_p_over = float(np.mean(old_samples > line))
            rec[f"old_p_over_{stat}"] = old_p_over

        records.append(rec)

    return records


def run_h2h_backtest(
    seasons: list[int],
    max_games: int = 800,
    n_sims: int = N_SIMS,
) -> pd.DataFrame:
    """Run both sims on the same games and collect results."""
    posteriors = load_posteriors()
    games_df = fetch_backtest_games(seasons, max_games=max_games)

    if games_df.empty:
        logger.error("No games found")
        return pd.DataFrame()

    sides = list(games_df.groupby(["game_pk", "team_id"]))
    n_sides = len(sides)
    logger.info(
        "Running head-to-head on %d team-game sides (%d sims each)...",
        n_sides, n_sims,
    )

    all_records: list[dict] = []
    n_skipped = 0
    t0 = time.perf_counter()

    for idx, ((game_pk, team_id), side_df) in enumerate(sides):
        recs = simulate_one_side_both(side_df, posteriors, n_sims=n_sims)
        if recs is None:
            n_skipped += 1
        else:
            all_records.extend(recs)

        if (idx + 1) % BATCH_SIZE == 0 or idx == n_sides - 1:
            elapsed = time.perf_counter() - t0
            pct = 100 * (idx + 1) / n_sides
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.1f sides/sec | "
                "%d skipped | %d batter-games",
                idx + 1, n_sides, pct, rate, n_skipped, len(all_records),
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: %d batter-games in %.1fs. Skipped %d sides.",
        len(all_records), elapsed, n_skipped,
    )
    return pd.DataFrame(all_records)


def print_results(df: pd.DataFrame) -> None:
    """Compute and print all head-to-head comparison metrics."""
    df_real = df[df["has_posterior"]].copy()
    n_total = len(df)
    n_real = len(df_real)

    print()
    print("=" * 76)
    print("HEAD-TO-HEAD BACKTEST: OLD BATTER SIM vs NEW LINEUP SIM")
    print("=" * 76)
    print(f"  Total batter-games:       {n_total:,}")
    print(f"  With real posteriors:      {n_real:,}")
    print(f"  Unique games:             {df['game_pk'].nunique():,}")
    print(f"  Seasons:                  {sorted(df['season'].unique())}")
    print(f"  Stats compared:           {', '.join(s.upper() for s in STATS)}")
    print(f"  (HRR excluded per request)")
    print()

    # ---------------------------------------------------------------
    # Pool all stat predictions for batters with real posteriors
    # ---------------------------------------------------------------
    old_p_list = []
    new_p_list = []
    actual_list = []
    stat_label_list = []

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        old_p = df_real[f"old_p_over_{stat}"].values
        new_p = df_real[f"new_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        old_p_list.append(old_p)
        new_p_list.append(new_p)
        actual_list.append(actual_over)
        stat_label_list.append(np.full(len(old_p), stat))

    all_old_p = np.concatenate(old_p_list)
    all_new_p = np.concatenate(new_p_list)
    all_actual = np.concatenate(actual_list)
    all_stat = np.concatenate(stat_label_list)

    # ---------------------------------------------------------------
    # Headline: >= 65% confidence hit rate
    # ---------------------------------------------------------------
    print("-" * 76)
    print("HEADLINE: HIT RATE AT >= 65% CONFIDENCE (excluding HRR)")
    print("-" * 76)

    for label, p_arr in [("OLD batter sim", all_old_p), ("NEW lineup sim", all_new_p)]:
        # Over side
        mask_over = p_arr >= 0.65
        n_over = mask_over.sum()
        if n_over > 0:
            hr_over = float(all_actual[mask_over].mean())
            print(f"  {label:18s}  P(over) >= 65%:  hit {hr_over:.4f} "
                  f"({hr_over*100:.1f}%)  on {n_over:,} picks")
        else:
            print(f"  {label:18s}  P(over) >= 65%:  no picks")

        # Under side
        p_under = 1 - p_arr
        a_under = 1 - all_actual
        mask_under = p_under >= 0.65
        n_under = mask_under.sum()
        if n_under > 0:
            hr_under = float(a_under[mask_under].mean())
            print(f"  {label:18s}  P(under) >= 65%: hit {hr_under:.4f} "
                  f"({hr_under*100:.1f}%)  on {n_under:,} picks")
        else:
            print(f"  {label:18s}  P(under) >= 65%: no picks")

    print()

    # ---------------------------------------------------------------
    # Per-stat breakdown at >= 65%
    # ---------------------------------------------------------------
    print("-" * 76)
    print("PER-STAT BREAKDOWN AT >= 65% CONFIDENCE")
    print("-" * 76)
    print(f"  {'Stat':>4s}  {'Side':>6s}  {'OLD hit%':>9s}  {'OLD N':>7s}  "
          f"{'NEW hit%':>9s}  {'NEW N':>7s}  {'Diff':>7s}")
    print(f"  {'----':>4s}  {'------':>6s}  {'---------':>9s}  {'-------':>7s}  "
          f"{'---------':>9s}  {'-------':>7s}  {'-------':>7s}")

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        old_p = df_real[f"old_p_over_{stat}"].values
        new_p = df_real[f"new_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        for side_label, flip in [("Over", False), ("Under", True)]:
            if flip:
                old_side = 1 - old_p
                new_side = 1 - new_p
                actual_side = 1 - actual_over
            else:
                old_side = old_p
                new_side = new_p
                actual_side = actual_over

            old_mask = old_side >= 0.65
            new_mask = new_side >= 0.65

            old_n = old_mask.sum()
            new_n = new_mask.sum()

            old_hr_str = f"{float(actual_side[old_mask].mean()):.4f}" if old_n > 0 else "--"
            new_hr_str = f"{float(actual_side[new_mask].mean()):.4f}" if new_n > 0 else "--"

            if old_n > 0 and new_n > 0:
                diff = float(actual_side[new_mask].mean()) - float(actual_side[old_mask].mean())
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "--"

            print(f"  {stat.upper():>4s}  {side_label:>6s}  {old_hr_str:>9s}  "
                  f"{old_n:>7,}  {new_hr_str:>9s}  {new_n:>7,}  {diff_str:>7s}")

    print()

    # ---------------------------------------------------------------
    # Calibration table for both models
    # ---------------------------------------------------------------
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    bucket_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

    print("-" * 76)
    print("CALIBRATION TABLE (all stats pooled, over side)")
    print("-" * 76)
    print(f"  {'Bucket':>8s}  {'OLD pred':>9s}  {'OLD actual':>11s}  {'OLD N':>7s}  "
          f"{'NEW pred':>9s}  {'NEW actual':>11s}  {'NEW N':>7s}")
    print(f"  {'--------':>8s}  {'---------':>9s}  {'-----------':>11s}  {'-------':>7s}  "
          f"{'---------':>9s}  {'-----------':>11s}  {'-------':>7s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]

        old_mask = (all_old_p >= lo) & (all_old_p < hi)
        new_mask = (all_new_p >= lo) & (all_new_p < hi)

        old_n = old_mask.sum()
        new_n = new_mask.sum()

        if old_n > 0:
            old_pred = float(all_old_p[old_mask].mean())
            old_actual = float(all_actual[old_mask].mean())
            old_str = f"{old_pred:>9.4f}  {old_actual:>11.4f}  {old_n:>7,}"
        else:
            old_str = f"{'--':>9s}  {'--':>11s}  {0:>7d}"

        if new_n > 0:
            new_pred = float(all_new_p[new_mask].mean())
            new_actual = float(all_actual[new_mask].mean())
            new_str = f"{new_pred:>9.4f}  {new_actual:>11.4f}  {new_n:>7,}"
        else:
            new_str = f"{'--':>9s}  {'--':>11s}  {0:>7d}"

        print(f"  {label:>8s}  {old_str}  {new_str}")

    print()

    # Under side calibration
    print("-" * 76)
    print("CALIBRATION TABLE (all stats pooled, under side)")
    print("-" * 76)
    all_old_p_under = 1 - all_old_p
    all_new_p_under = 1 - all_new_p
    all_actual_under = 1 - all_actual

    print(f"  {'Bucket':>8s}  {'OLD pred':>9s}  {'OLD actual':>11s}  {'OLD N':>7s}  "
          f"{'NEW pred':>9s}  {'NEW actual':>11s}  {'NEW N':>7s}")
    print(f"  {'--------':>8s}  {'---------':>9s}  {'-----------':>11s}  {'-------':>7s}  "
          f"{'---------':>9s}  {'-----------':>11s}  {'-------':>7s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]

        old_mask = (all_old_p_under >= lo) & (all_old_p_under < hi)
        new_mask = (all_new_p_under >= lo) & (all_new_p_under < hi)

        old_n = old_mask.sum()
        new_n = new_mask.sum()

        if old_n > 0:
            old_pred = float(all_old_p_under[old_mask].mean())
            old_actual = float(all_actual_under[old_mask].mean())
            old_str = f"{old_pred:>9.4f}  {old_actual:>11.4f}  {old_n:>7,}"
        else:
            old_str = f"{'--':>9s}  {'--':>11s}  {0:>7d}"

        if new_n > 0:
            new_pred = float(all_new_p_under[new_mask].mean())
            new_actual = float(all_actual_under[new_mask].mean())
            new_str = f"{new_pred:>9.4f}  {new_actual:>11.4f}  {new_n:>7,}"
        else:
            new_str = f"{'--':>9s}  {'--':>11s}  {0:>7d}"

        print(f"  {label:>8s}  {old_str}  {new_str}")

    print()

    # ---------------------------------------------------------------
    # Sample sizes summary
    # ---------------------------------------------------------------
    print("-" * 76)
    print("SAMPLE SIZES")
    print("-" * 76)
    print(f"  Total batter-game-stat predictions (pooled): {len(all_old_p):,}")
    for stat in STATS:
        n_stat = len(df_real)
        old_65 = (df_real[f"old_p_over_{stat}"].values >= 0.65).sum()
        old_65 += ((1 - df_real[f"old_p_over_{stat}"].values) >= 0.65).sum()
        new_65 = (df_real[f"new_p_over_{stat}"].values >= 0.65).sum()
        new_65 += ((1 - df_real[f"new_p_over_{stat}"].values) >= 0.65).sum()
        print(f"  {stat.upper():>4s}: {n_stat:,} predictions  "
              f"(OLD >= 65%: {old_65:,}  NEW >= 65%: {new_65:,})")

    print()
    print("=" * 76)


def main() -> None:
    seasons = [2024, 2025, 2026]
    max_games = 800
    n_sims = N_SIMS

    logger.info("Starting head-to-head backtest: seasons=%s, max_games=%d, n_sims=%d",
                seasons, max_games, n_sims)

    df = run_h2h_backtest(seasons=seasons, max_games=max_games, n_sims=n_sims)

    if df.empty:
        logger.error("No results to report")
        return

    print_results(df)


if __name__ == "__main__":
    main()
