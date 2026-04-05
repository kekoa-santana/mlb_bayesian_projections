#!/usr/bin/env python
"""
Backtest: lineup sim WITH vs WITHOUT LD%-based BABIP adjustments.

Runs the lineup simulator on the same set of historical games twice:
  1. BASELINE: batter_babip_adjs = zeros (current behavior)
  2. LD% ADJ:  batter_babip_adjs derived from regressed LD%

Compares hit rate at >= 65% confidence and calibration.

Usage
-----
    python scripts/precompute/backtest_ld_babip.py
    python scripts/precompute/backtest_ld_babip.py --max-games 200
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.game_sim.lineup_simulator import simulate_lineup_game
from scripts.precompute.backtest_lineup_sim import (
    load_posteriors,
    fetch_backtest_games,
    DEFAULT_BP_K,
    DEFAULT_BP_BB,
    DEFAULT_BP_HR,
)
from scripts.precompute.precompute_ld_rate import LEAGUE_LD_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")

N_SIMS = 10_000
BATCH_SIZE = 25

# Stats to compare (H is the primary target since LD% affects BABIP -> hits)
STATS = ["h", "k", "bb", "r", "rbi"]
PRIMARY_LINE = {"h": 0.5, "k": 0.5, "bb": 0.5, "r": 0.5, "rbi": 0.5}

# LD% -> BABIP adjustment coefficient
# Each +1% LD% above league average adds ~0.0025 to BABIP
# This is the scaling factor applied to (regressed_ld - league_avg)
LD_BABIP_COEFFICIENT = 0.25


def load_ld_lookup() -> dict[int, float]:
    """Load LD% parquet and build player_id -> babip_adj lookup.

    Uses the latest season available per player.
    Returns dict mapping player_id to BABIP adjustment (float).
    """
    ld_path = DASHBOARD_DIR / "batter_ld_rate.parquet"
    if not ld_path.exists():
        logger.warning("batter_ld_rate.parquet not found; run precompute_ld_rate.py first")
        return {}

    df = pd.read_parquet(ld_path)
    if df.empty:
        return {}

    # Use latest season per player
    latest = df.sort_values("season").groupby("player_id").last().reset_index()

    lookup: dict[int, float] = {}
    for _, row in latest.iterrows():
        ld_regressed = float(row["ld_rate_regressed"])
        # Convert LD% deviation to BABIP adjustment
        # Positive means higher LD% than average -> more hits on BIP
        babip_adj = (ld_regressed - LEAGUE_LD_RATE) * LD_BABIP_COEFFICIENT
        lookup[int(row["player_id"])] = babip_adj

    n_pos = sum(1 for v in lookup.values() if v > 0)
    n_neg = sum(1 for v in lookup.values() if v < 0)
    adj_vals = list(lookup.values())
    logger.info(
        "LD lookup: %d players (%d positive adj, %d negative adj), "
        "adj range [%.4f, %.4f], mean=%.4f",
        len(lookup), n_pos, n_neg,
        min(adj_vals), max(adj_vals), np.mean(adj_vals),
    )

    return lookup


def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    ld_lookup: dict[int, float],
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run lineup sim TWICE for one team-side: baseline and LD%-adjusted.

    Returns per-batter records with predictions from both runs, or None
    if posteriors are missing for too many batters.
    """
    side_df = side_df.sort_values("batting_order")
    batter_ids = side_df["batter_id"].astype(int).tolist()
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

    # Build LD%-based BABIP adjustments for this lineup
    babip_adjs = np.zeros(9)
    for i, bid in enumerate(batter_ids[:9]):
        babip_adjs[i] = ld_lookup.get(bid, 0.0)

    # Common sim kwargs
    common_kwargs = dict(
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

    # --- RUN 1: BASELINE (no BABIP adj, same as current production) ---
    baseline_result = simulate_lineup_game(
        **common_kwargs,
        batter_babip_adjs=np.zeros(9),
    )

    # --- RUN 2: LD%-adjusted BABIP ---
    ld_result = simulate_lineup_game(
        **common_kwargs,
        batter_babip_adjs=babip_adjs,
    )

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

        base_br = baseline_result.batter_result(i)
        ld_br = ld_result.batter_result(i)

        rec = {
            "game_pk": game_pk,
            "season": int(row["season"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "has_posterior": has_posterior,
            "babip_adj": float(babip_adjs[i]),
        }

        for stat in STATS:
            line = PRIMARY_LINE[stat]
            actual_val = int(row[f"actual_{stat}"])
            rec[f"actual_{stat}"] = actual_val

            # BASELINE P(over) and expected
            base_samples = base_br[f"{stat}_samples"]
            base_p_over = float(np.mean(base_samples > line))
            base_expected = float(np.mean(base_samples))
            rec[f"base_p_over_{stat}"] = base_p_over
            rec[f"base_expected_{stat}"] = base_expected

            # LD%-adjusted P(over) and expected
            ld_samples = ld_br[f"{stat}_samples"]
            ld_p_over = float(np.mean(ld_samples > line))
            ld_expected = float(np.mean(ld_samples))
            rec[f"ld_p_over_{stat}"] = ld_p_over
            rec[f"ld_expected_{stat}"] = ld_expected

        records.append(rec)

    return records


def run_backtest(
    seasons: list[int],
    max_games: int = 400,
    n_sims: int = N_SIMS,
) -> pd.DataFrame:
    """Run both sim variants on the same games and collect results."""
    posteriors = load_posteriors()
    ld_lookup = load_ld_lookup()

    if not ld_lookup:
        logger.error("No LD% data available. Run precompute_ld_rate.py first.")
        return pd.DataFrame()

    games_df = fetch_backtest_games(seasons, max_games=max_games)

    if games_df.empty:
        logger.error("No games found for backtest")
        return pd.DataFrame()

    sides = list(games_df.groupby(["game_pk", "team_id"]))
    n_sides = len(sides)
    logger.info(
        "Running LD%% BABIP backtest on %d team-game sides (%d sims each)...",
        n_sides, n_sims,
    )

    all_records: list[dict] = []
    n_skipped = 0
    t0 = time.perf_counter()

    for idx, ((game_pk, team_id), side_df) in enumerate(sides):
        recs = simulate_one_side(side_df, posteriors, ld_lookup, n_sims=n_sims)
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
    """Compute and print all comparison metrics."""
    df_real = df[df["has_posterior"]].copy()
    n_total = len(df)
    n_real = len(df_real)

    # Count how many batters had nonzero LD% adjustments
    n_with_adj = (df_real["babip_adj"].abs() > 0.001).sum()

    print()
    print("=" * 76)
    print("BACKTEST: BASELINE vs LD%-ADJUSTED BABIP")
    print("=" * 76)
    print(f"  Total batter-games:       {n_total:,}")
    print(f"  With real posteriors:      {n_real:,}")
    print(f"  With nonzero LD adj:      {n_with_adj:,} ({100*n_with_adj/max(n_real,1):.1f}%)")
    print(f"  Unique games:             {df['game_pk'].nunique():,}")
    print(f"  Seasons:                  {sorted(df['season'].unique())}")
    print(f"  Stats compared:           {', '.join(s.upper() for s in STATS)}")
    print()

    # Show BABIP adj distribution
    adj_vals = df_real["babip_adj"].values
    print(f"  BABIP adj distribution:")
    print(f"    mean={np.mean(adj_vals):+.4f}  std={np.std(adj_vals):.4f}")
    print(f"    min={np.min(adj_vals):+.4f}  max={np.max(adj_vals):+.4f}")
    pcts = np.percentile(adj_vals, [10, 25, 50, 75, 90])
    print(f"    P10={pcts[0]:+.4f}  P25={pcts[1]:+.4f}  P50={pcts[2]:+.4f}  "
          f"P75={pcts[3]:+.4f}  P90={pcts[4]:+.4f}")
    print()

    # Pool all stat predictions
    base_p_list = []
    ld_p_list = []
    actual_list = []

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        base_p = df_real[f"base_p_over_{stat}"].values
        ld_p = df_real[f"ld_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        base_p_list.append(base_p)
        ld_p_list.append(ld_p)
        actual_list.append(actual_over)
    all_base_p = np.concatenate(base_p_list)
    all_ld_p = np.concatenate(ld_p_list)
    all_actual = np.concatenate(actual_list)

    # ---------------------------------------------------------------
    # Headline: >= 65% confidence hit rate
    # ---------------------------------------------------------------
    print("-" * 76)
    print("HEADLINE: HIT RATE AT >= 65% CONFIDENCE (all stats pooled)")
    print("-" * 76)

    for label, p_arr in [("Baseline", all_base_p), ("LD-adjusted", all_ld_p)]:
        # Over side
        mask_over = p_arr >= 0.65
        n_over = mask_over.sum()
        if n_over > 0:
            hr_over = float(all_actual[mask_over].mean())
            print(f"  {label:14s}  P(over) >= 65%:  hit {hr_over:.4f} "
                  f"({hr_over*100:.1f}%)  on {n_over:,} picks")
        else:
            print(f"  {label:14s}  P(over) >= 65%:  no picks")

        # Under side
        p_under = 1 - p_arr
        a_under = 1 - all_actual
        mask_under = p_under >= 0.65
        n_under = mask_under.sum()
        if n_under > 0:
            hr_under = float(a_under[mask_under].mean())
            print(f"  {label:14s}  P(under) >= 65%: hit {hr_under:.4f} "
                  f"({hr_under*100:.1f}%)  on {n_under:,} picks")
        else:
            print(f"  {label:14s}  P(under) >= 65%: no picks")

    print()

    # ---------------------------------------------------------------
    # Per-stat breakdown at >= 65%
    # ---------------------------------------------------------------
    print("-" * 76)
    print("PER-STAT BREAKDOWN AT >= 65% CONFIDENCE")
    print("-" * 76)
    print(f"  {'Stat':>4s}  {'Side':>6s}  {'Base hit%':>10s}  {'Base N':>7s}  "
          f"{'LD hit%':>10s}  {'LD N':>7s}  {'Diff':>7s}")
    print(f"  {'----':>4s}  {'------':>6s}  {'----------':>10s}  {'-------':>7s}  "
          f"{'----------':>10s}  {'-------':>7s}  {'-------':>7s}")

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        base_p = df_real[f"base_p_over_{stat}"].values
        ld_p = df_real[f"ld_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        for side_label, flip in [("Over", False), ("Under", True)]:
            if flip:
                base_side = 1 - base_p
                ld_side = 1 - ld_p
                actual_side = 1 - actual_over
            else:
                base_side = base_p
                ld_side = ld_p
                actual_side = actual_over

            base_mask = base_side >= 0.65
            ld_mask = ld_side >= 0.65

            base_n = base_mask.sum()
            ld_n = ld_mask.sum()

            base_hr_str = f"{float(actual_side[base_mask].mean()):.4f}" if base_n > 0 else "--"
            ld_hr_str = f"{float(actual_side[ld_mask].mean()):.4f}" if ld_n > 0 else "--"

            if base_n > 0 and ld_n > 0:
                diff = float(actual_side[ld_mask].mean()) - float(actual_side[base_mask].mean())
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "--"

            print(f"  {stat.upper():>4s}  {side_label:>6s}  {base_hr_str:>10s}  "
                  f"{base_n:>7,}  {ld_hr_str:>10s}  {ld_n:>7,}  {diff_str:>7s}")

    print()

    # ---------------------------------------------------------------
    # H-only deep dive (since LD% primarily affects BABIP -> hits)
    # ---------------------------------------------------------------
    print("-" * 76)
    print("H (HITS) DEEP DIVE -- primary target of LD%% BABIP adjustment")
    print("-" * 76)

    h_base_p = df_real["base_p_over_h"].values
    h_ld_p = df_real["ld_p_over_h"].values
    h_actual = (df_real["actual_h"].values > 0.5).astype(float)

    # Show how many picks changed confidence bucket
    n_upgraded = ((h_ld_p >= 0.65) & (h_base_p < 0.65)).sum()
    n_downgraded = ((h_ld_p < 0.65) & (h_base_p >= 0.65)).sum()
    print(f"  Picks upgraded to >= 65%%:   {n_upgraded:,}")
    print(f"  Picks downgraded below 65%%: {n_downgraded:,}")

    # Hit rate on picks that LD% upgraded to >= 65%
    upgraded_mask = (h_ld_p >= 0.65) & (h_base_p < 0.65)
    if upgraded_mask.sum() > 0:
        hr_upgraded = float(h_actual[upgraded_mask].mean())
        print(f"  Hit rate on upgraded picks: {hr_upgraded:.4f} ({hr_upgraded*100:.1f}%)")

    print()

    # ---------------------------------------------------------------
    # Calibration table
    # ---------------------------------------------------------------
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    bucket_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

    print("-" * 76)
    print("CALIBRATION TABLE (all stats pooled, over side)")
    print("-" * 76)
    print(f"  {'Bucket':>8s}  {'Base pred':>10s}  {'Base actual':>12s}  {'Base N':>7s}  "
          f"{'LD pred':>10s}  {'LD actual':>12s}  {'LD N':>7s}")
    print(f"  {'--------':>8s}  {'----------':>10s}  {'------------':>12s}  {'-------':>7s}  "
          f"{'----------':>10s}  {'------------':>12s}  {'-------':>7s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]

        base_mask = (all_base_p >= lo) & (all_base_p < hi)
        ld_mask = (all_ld_p >= lo) & (all_ld_p < hi)

        base_n = base_mask.sum()
        ld_n = ld_mask.sum()

        if base_n > 0:
            base_pred = float(all_base_p[base_mask].mean())
            base_actual = float(all_actual[base_mask].mean())
            base_str = f"{base_pred:>10.4f}  {base_actual:>12.4f}  {base_n:>7,}"
        else:
            base_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        if ld_n > 0:
            ld_pred = float(all_ld_p[ld_mask].mean())
            ld_actual = float(all_actual[ld_mask].mean())
            ld_str = f"{ld_pred:>10.4f}  {ld_actual:>12.4f}  {ld_n:>7,}"
        else:
            ld_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        print(f"  {label:>8s}  {base_str}  {ld_str}")

    print()

    # Under-side calibration
    print("-" * 76)
    print("CALIBRATION TABLE (all stats pooled, under side)")
    print("-" * 76)
    all_base_under = 1 - all_base_p
    all_ld_under = 1 - all_ld_p
    all_actual_under = 1 - all_actual

    print(f"  {'Bucket':>8s}  {'Base pred':>10s}  {'Base actual':>12s}  {'Base N':>7s}  "
          f"{'LD pred':>10s}  {'LD actual':>12s}  {'LD N':>7s}")
    print(f"  {'--------':>8s}  {'----------':>10s}  {'------------':>12s}  {'-------':>7s}  "
          f"{'----------':>10s}  {'------------':>12s}  {'-------':>7s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]

        base_mask = (all_base_under >= lo) & (all_base_under < hi)
        ld_mask = (all_ld_under >= lo) & (all_ld_under < hi)

        base_n = base_mask.sum()
        ld_n = ld_mask.sum()

        if base_n > 0:
            base_pred = float(all_base_under[base_mask].mean())
            base_actual = float(all_actual_under[base_mask].mean())
            base_str = f"{base_pred:>10.4f}  {base_actual:>12.4f}  {base_n:>7,}"
        else:
            base_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        if ld_n > 0:
            ld_pred = float(all_ld_under[ld_mask].mean())
            ld_actual = float(all_actual_under[ld_mask].mean())
            ld_str = f"{ld_pred:>10.4f}  {ld_actual:>12.4f}  {ld_n:>7,}"
        else:
            ld_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        print(f"  {label:>8s}  {base_str}  {ld_str}")

    print()

    # ---------------------------------------------------------------
    # Summary verdict
    # ---------------------------------------------------------------
    print("-" * 76)
    print("SUMMARY")
    print("-" * 76)

    # Overall 65% hit rate change
    for label, p_arr in [("Baseline", all_base_p), ("LD-adj", all_ld_p)]:
        m65 = p_arr >= 0.65
        m65u = (1 - p_arr) >= 0.65
        combined = m65 | m65u
        if combined.sum() > 0:
            # Actual hit rate for combined over+under
            over_hit = all_actual[m65].mean() if m65.sum() > 0 else 0
            under_hit = (1 - all_actual)[m65u].mean() if m65u.sum() > 0 else 0
            total_picks = m65.sum() + m65u.sum()
            total_hits = (
                (all_actual[m65].sum() if m65.sum() > 0 else 0)
                + ((1 - all_actual)[m65u].sum() if m65u.sum() > 0 else 0)
            )
            combined_hr = total_hits / total_picks if total_picks > 0 else 0
            print(f"  {label:10s} >= 65%% combined hit rate: {combined_hr:.4f} "
                  f"({combined_hr*100:.1f}%) on {total_picks:,} picks")

    print()
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest LD%% BABIP adjustments"
    )
    parser.add_argument(
        "--max-games", type=int, default=400,
        help="Max team-game sides to test (default 400)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=N_SIMS,
        help="Monte Carlo sims per game (default 10000)",
    )
    args = parser.parse_args()

    seasons = [2024, 2025]
    logger.info(
        "Starting LD%% BABIP backtest: seasons=%s, max_games=%d, n_sims=%d",
        seasons, args.max_games, args.n_sims,
    )

    df = run_backtest(
        seasons=seasons,
        max_games=args.max_games,
        n_sims=args.n_sims,
    )

    if df.empty:
        logger.error("No results to report")
        return

    print_results(df)


if __name__ == "__main__":
    main()
