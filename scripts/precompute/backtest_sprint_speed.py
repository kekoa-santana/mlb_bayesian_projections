#!/usr/bin/env python
"""
Backtest: LD%-only BABIP adj vs LD% + sprint speed BABIP adj.

Runs the lineup simulator on the same set of historical games twice:
  1. LD-ONLY:  batter_babip_adjs from LD% only (current production)
  2. LD+SPEED: batter_babip_adjs from LD% + sprint speed (proposed)

Compares 65% confidence hit rate and breakdown by speed tier
(fast / average / slow runners).

Usage
-----
    python scripts/precompute/backtest_sprint_speed.py
    python scripts/precompute/backtest_sprint_speed.py --max-games 200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute import DASHBOARD_DIR
from scripts.precompute.backtest_harness import (
    DEFAULT_BP_K_RATE as DEFAULT_BP_K,
    DEFAULT_BP_BB_RATE as DEFAULT_BP_BB,
    DEFAULT_BP_HR_RATE as DEFAULT_BP_HR,
    N_SIMS,
    fetch_backtest_games,
    run_sides_loop,
)
from scripts.precompute.backtest_lineup_sim import load_posteriors
from src.models.game_sim.lineup_simulator import simulate_lineup_game
from scripts.precompute.precompute_ld_rate import LEAGUE_LD_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Stats to compare (H is the primary target since sprint speed affects BABIP)
STATS = ["h", "k", "bb", "r", "rbi"]
PRIMARY_LINE = {"h": 0.5, "k": 0.5, "bb": 0.5, "r": 0.5, "rbi": 0.5}

# LD% -> BABIP adjustment coefficient (matches confident_picks.py)
LD_BABIP_COEFFICIENT = 0.25

# Sprint speed -> BABIP adjustment coefficient
# Each +1 ft/sec above league average adds ~0.010 to BABIP
LEAGUE_SPRINT_SPEED = 27.0
SPEED_BABIP_COEFFICIENT = 0.010

# Speed tier thresholds (regressed ft/sec)
FAST_THRESHOLD = 28.0    # top ~25%
SLOW_THRESHOLD = 26.0    # bottom ~25%


def load_ld_lookup() -> dict[int, float]:
    """Load LD% parquet and build player_id -> babip_adj lookup."""
    ld_path = DASHBOARD_DIR / "batter_ld_rate.parquet"
    if not ld_path.exists():
        logger.warning("batter_ld_rate.parquet not found; run precompute_ld_rate.py first")
        return {}

    df = pd.read_parquet(ld_path)
    if df.empty:
        return {}

    latest = df.sort_values("season").groupby("player_id").last().reset_index()

    lookup: dict[int, float] = {}
    for _, row in latest.iterrows():
        ld_regressed = float(row["ld_rate_regressed"])
        babip_adj = (ld_regressed - LEAGUE_LD_RATE) * LD_BABIP_COEFFICIENT
        lookup[int(row["player_id"])] = babip_adj

    n_pos = sum(1 for v in lookup.values() if v > 0)
    n_neg = sum(1 for v in lookup.values() if v < 0)
    adj_vals = list(lookup.values())
    logger.info(
        "LD lookup: %d players (%d positive, %d negative), "
        "range [%.4f, %.4f], mean=%.4f",
        len(lookup), n_pos, n_neg,
        min(adj_vals), max(adj_vals), np.mean(adj_vals),
    )
    return lookup


def load_speed_lookup() -> tuple[dict[int, float], dict[int, float]]:
    """Load sprint speed parquet and build two lookups.

    Returns
    -------
    speed_babip_lookup : dict
        player_id -> BABIP adjustment from sprint speed
    speed_raw_lookup : dict
        player_id -> regressed sprint speed (for tier classification)
    """
    speed_path = DASHBOARD_DIR / "batter_sprint_speed.parquet"
    if not speed_path.exists():
        logger.warning(
            "batter_sprint_speed.parquet not found; "
            "run precompute_sprint_speed.py first"
        )
        return {}, {}

    df = pd.read_parquet(speed_path)
    if df.empty:
        return {}, {}

    latest = df.sort_values("season").groupby("player_id").last().reset_index()

    babip_lookup: dict[int, float] = {}
    raw_lookup: dict[int, float] = {}
    for _, row in latest.iterrows():
        pid = int(row["player_id"])
        speed_reg = float(row["sprint_speed_regressed"])
        raw_lookup[pid] = speed_reg
        babip_lookup[pid] = (speed_reg - LEAGUE_SPRINT_SPEED) * SPEED_BABIP_COEFFICIENT

    n_pos = sum(1 for v in babip_lookup.values() if v > 0)
    n_neg = sum(1 for v in babip_lookup.values() if v < 0)
    adj_vals = list(babip_lookup.values())
    logger.info(
        "Speed lookup: %d players (%d positive, %d negative), "
        "range [%.4f, %.4f], mean=%.4f",
        len(babip_lookup), n_pos, n_neg,
        min(adj_vals), max(adj_vals), np.mean(adj_vals),
    )
    return babip_lookup, raw_lookup


def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    ld_lookup: dict[int, float],
    speed_babip_lookup: dict[int, float],
    speed_raw_lookup: dict[int, float],
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run lineup sim TWICE for one team-side: LD-only and LD+speed.

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

    # Build LD%-only BABIP adjustments
    ld_adjs = np.zeros(9)
    for i, bid in enumerate(batter_ids[:9]):
        ld_adjs[i] = ld_lookup.get(bid, 0.0)

    # Build LD% + sprint speed BABIP adjustments (additive stack)
    combined_adjs = np.zeros(9)
    for i, bid in enumerate(batter_ids[:9]):
        combined_adjs[i] = ld_lookup.get(bid, 0.0) + speed_babip_lookup.get(bid, 0.0)

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

    # --- RUN 1: LD%-only (current production) ---
    ld_result = simulate_lineup_game(
        **common_kwargs,
        batter_babip_adjs=ld_adjs,
    )

    # --- RUN 2: LD% + sprint speed (proposed) ---
    combined_result = simulate_lineup_game(
        **common_kwargs,
        batter_babip_adjs=combined_adjs,
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

        ld_br = ld_result.batter_result(i)
        comb_br = combined_result.batter_result(i)

        rec = {
            "game_pk": game_pk,
            "season": int(row["season"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "has_posterior": has_posterior,
            "ld_babip_adj": float(ld_adjs[i]),
            "speed_babip_adj": float(speed_babip_lookup.get(bid, 0.0)),
            "combined_babip_adj": float(combined_adjs[i]),
            "sprint_speed": speed_raw_lookup.get(bid, np.nan),
        }

        for stat in STATS:
            line = PRIMARY_LINE[stat]
            actual_val = int(row[f"actual_{stat}"])
            rec[f"actual_{stat}"] = actual_val

            # LD-only P(over) and expected
            ld_samples = ld_br[f"{stat}_samples"]
            rec[f"ld_p_over_{stat}"] = float(np.mean(ld_samples > line))
            rec[f"ld_expected_{stat}"] = float(np.mean(ld_samples))

            # LD+speed P(over) and expected
            comb_samples = comb_br[f"{stat}_samples"]
            rec[f"comb_p_over_{stat}"] = float(np.mean(comb_samples > line))
            rec[f"comb_expected_{stat}"] = float(np.mean(comb_samples))

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
    speed_babip_lookup, speed_raw_lookup = load_speed_lookup()

    if not ld_lookup:
        logger.error("No LD%% data available. Run precompute_ld_rate.py first.")
        return pd.DataFrame()

    if not speed_babip_lookup:
        logger.error(
            "No sprint speed data available. Run precompute_sprint_speed.py first."
        )
        return pd.DataFrame()

    games_df = fetch_backtest_games(seasons, max_games=max_games)

    if games_df.empty:
        logger.error("No games found for backtest")
        return pd.DataFrame()

    sides = list(games_df.groupby(["game_pk", "team_id"]))
    logger.info(
        "Running sprint speed BABIP backtest on %d team-game sides (%d sims each)...",
        len(sides), n_sims,
    )

    df, _skipped = run_sides_loop(
        label="Sprint",
        sides=sides,
        simulate_fn=lambda sdf: simulate_one_side(
            sdf, posteriors, ld_lookup,
            speed_babip_lookup, speed_raw_lookup, n_sims=n_sims,
        ),
    )
    return df


def print_results(df: pd.DataFrame) -> None:
    """Compute and print all comparison metrics."""
    df_real = df[df["has_posterior"]].copy()
    n_total = len(df)
    n_real = len(df_real)

    # Count batters with nonzero adjustments
    n_with_ld = (df_real["ld_babip_adj"].abs() > 0.001).sum()
    n_with_speed = (df_real["speed_babip_adj"].abs() > 0.001).sum()
    n_with_both = (
        (df_real["ld_babip_adj"].abs() > 0.001)
        & (df_real["speed_babip_adj"].abs() > 0.001)
    ).sum()

    print()
    print("=" * 76)
    print("BACKTEST: LD%-ONLY vs LD% + SPRINT SPEED BABIP ADJUSTMENT")
    print("=" * 76)
    print(f"  Total batter-games:       {n_total:,}")
    print(f"  With real posteriors:      {n_real:,}")
    print(f"  With nonzero LD adj:      {n_with_ld:,} ({100*n_with_ld/max(n_real,1):.1f}%)")
    print(f"  With nonzero speed adj:   {n_with_speed:,} ({100*n_with_speed/max(n_real,1):.1f}%)")
    print(f"  With both adjustments:    {n_with_both:,} ({100*n_with_both/max(n_real,1):.1f}%)")
    print(f"  Unique games:             {df['game_pk'].nunique():,}")
    print(f"  Seasons:                  {sorted(df['season'].unique())}")
    print()

    # Show adjustment distributions
    for label, col in [
        ("LD BABIP adj", "ld_babip_adj"),
        ("Speed BABIP adj", "speed_babip_adj"),
        ("Combined BABIP adj", "combined_babip_adj"),
    ]:
        vals = df_real[col].values
        print(f"  {label}:")
        print(f"    mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}")
        pcts = np.percentile(vals, [10, 25, 50, 75, 90])
        print(f"    P10={pcts[0]:+.4f}  P25={pcts[1]:+.4f}  P50={pcts[2]:+.4f}  "
              f"P75={pcts[3]:+.4f}  P90={pcts[4]:+.4f}")
    print()

    # Pool all stat predictions
    ld_p_list, comb_p_list, actual_list = [], [], []

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        ld_p = df_real[f"ld_p_over_{stat}"].values
        comb_p = df_real[f"comb_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        ld_p_list.append(ld_p)
        comb_p_list.append(comb_p)
        actual_list.append(actual_over)
    all_ld_p = np.concatenate(ld_p_list)
    all_comb_p = np.concatenate(comb_p_list)
    all_actual = np.concatenate(actual_list)

    # ---------------------------------------------------------------
    # Headline: >= 65% confidence hit rate
    # ---------------------------------------------------------------
    print("-" * 76)
    print("HEADLINE: HIT RATE AT >= 65% CONFIDENCE (all stats pooled)")
    print("-" * 76)

    for label, p_arr in [("LD-only", all_ld_p), ("LD+Speed", all_comb_p)]:
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
    print(f"  {'Stat':>4s}  {'Side':>6s}  {'LD hit%':>10s}  {'LD N':>7s}  "
          f"{'Comb hit%':>10s}  {'Comb N':>7s}  {'Diff':>7s}")
    print(f"  {'----':>4s}  {'------':>6s}  {'----------':>10s}  {'-------':>7s}  "
          f"{'----------':>10s}  {'-------':>7s}  {'-------':>7s}")

    for stat in STATS:
        line = PRIMARY_LINE[stat]
        ld_p = df_real[f"ld_p_over_{stat}"].values
        comb_p = df_real[f"comb_p_over_{stat}"].values
        actual_over = (df_real[f"actual_{stat}"].values > line).astype(float)

        for side_label, flip in [("Over", False), ("Under", True)]:
            if flip:
                ld_side = 1 - ld_p
                comb_side = 1 - comb_p
                actual_side = 1 - actual_over
            else:
                ld_side = ld_p
                comb_side = comb_p
                actual_side = actual_over

            ld_mask = ld_side >= 0.65
            comb_mask = comb_side >= 0.65

            ld_n = ld_mask.sum()
            comb_n = comb_mask.sum()

            ld_hr_str = f"{float(actual_side[ld_mask].mean()):.4f}" if ld_n > 0 else "--"
            comb_hr_str = f"{float(actual_side[comb_mask].mean()):.4f}" if comb_n > 0 else "--"

            if ld_n > 0 and comb_n > 0:
                diff = float(actual_side[comb_mask].mean()) - float(actual_side[ld_mask].mean())
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "--"

            print(f"  {stat.upper():>4s}  {side_label:>6s}  {ld_hr_str:>10s}  "
                  f"{ld_n:>7,}  {comb_hr_str:>10s}  {comb_n:>7,}  {diff_str:>7s}")

    print()

    # ---------------------------------------------------------------
    # H (HITS) DEEP DIVE -- primary target of sprint speed adjustment
    # ---------------------------------------------------------------
    print("-" * 76)
    print("H (HITS) DEEP DIVE -- primary target of sprint speed BABIP adjustment")
    print("-" * 76)

    h_ld_p = df_real["ld_p_over_h"].values
    h_comb_p = df_real["comb_p_over_h"].values
    h_actual = (df_real["actual_h"].values > 0.5).astype(float)

    # Show how many picks changed confidence bucket
    n_upgraded = ((h_comb_p >= 0.65) & (h_ld_p < 0.65)).sum()
    n_downgraded = ((h_comb_p < 0.65) & (h_ld_p >= 0.65)).sum()
    print(f"  Picks upgraded to >= 65%%:   {n_upgraded:,}")
    print(f"  Picks downgraded below 65%%: {n_downgraded:,}")

    # Hit rate on picks that sprint speed upgraded to >= 65%
    upgraded_mask = (h_comb_p >= 0.65) & (h_ld_p < 0.65)
    if upgraded_mask.sum() > 0:
        hr_upgraded = float(h_actual[upgraded_mask].mean())
        print(f"  Hit rate on upgraded picks: {hr_upgraded:.4f} ({hr_upgraded*100:.1f}%)")

    # Downgraded hit rate
    downgraded_mask = (h_comb_p < 0.65) & (h_ld_p >= 0.65)
    if downgraded_mask.sum() > 0:
        hr_downgraded = float(h_actual[downgraded_mask].mean())
        print(f"  Hit rate on downgraded picks: {hr_downgraded:.4f} ({hr_downgraded*100:.1f}%)")

    print()

    # ---------------------------------------------------------------
    # SPEED TIER ANALYSIS -- the core question
    # ---------------------------------------------------------------
    print("-" * 76)
    print("SPEED TIER ANALYSIS (H bias by runner speed)")
    print("-" * 76)

    has_speed = df_real["sprint_speed"].notna()
    fast = df_real[has_speed & (df_real["sprint_speed"] >= FAST_THRESHOLD)]
    slow = df_real[has_speed & (df_real["sprint_speed"] <= SLOW_THRESHOLD)]
    avg_speed = df_real[
        has_speed
        & (df_real["sprint_speed"] > SLOW_THRESHOLD)
        & (df_real["sprint_speed"] < FAST_THRESHOLD)
    ]
    no_speed = df_real[~has_speed]

    print(f"  Speed coverage: {has_speed.sum():,} of {n_real:,} batter-games "
          f"({100*has_speed.sum()/max(n_real,1):.1f}%)")
    print()

    for tier_label, tier_df in [
        (f"FAST (>= {FAST_THRESHOLD} ft/s)", fast),
        (f"AVERAGE ({SLOW_THRESHOLD}-{FAST_THRESHOLD} ft/s)", avg_speed),
        (f"SLOW (<= {SLOW_THRESHOLD} ft/s)", slow),
        ("NO SPEED DATA", no_speed),
    ]:
        if len(tier_df) == 0:
            print(f"  {tier_label}: no data")
            continue

        ld_h_bias = float(np.mean(
            tier_df["ld_expected_h"] - tier_df["actual_h"]
        ))
        comb_h_bias = float(np.mean(
            tier_df["comb_expected_h"] - tier_df["actual_h"]
        ))

        # Mean speed adj in this tier
        mean_adj = tier_df["speed_babip_adj"].mean()

        # Mean actual H
        mean_h = tier_df["actual_h"].mean()

        print(f"  {tier_label} (N={len(tier_df):,}, avg actual H={mean_h:.2f}):")
        print(f"    LD-only H bias: {ld_h_bias:+.4f}  LD+Speed H bias: {comb_h_bias:+.4f}")
        print(f"    Mean speed adj: {mean_adj:+.4f}")

        # 65% confidence hit rate for H in this tier
        ld_h_p = tier_df["ld_p_over_h"].values
        comb_h_p = tier_df["comb_p_over_h"].values
        h_actual_tier = (tier_df["actual_h"].values > 0.5).astype(float)

        ld_conf = ld_h_p >= 0.65
        comb_conf = comb_h_p >= 0.65
        if ld_conf.sum() > 0:
            ld_hr65 = float(h_actual_tier[ld_conf].mean())
            print(f"    LD-only H 65%% picks: {ld_conf.sum():,} "
                  f"(hit rate {ld_hr65:.4f} / {ld_hr65*100:.1f}%)")
        if comb_conf.sum() > 0:
            comb_hr65 = float(h_actual_tier[comb_conf].mean())
            print(f"    LD+Speed H 65%% picks: {comb_conf.sum():,} "
                  f"(hit rate {comb_hr65:.4f} / {comb_hr65*100:.1f}%)")
        print()

    # ---------------------------------------------------------------
    # Calibration table
    # ---------------------------------------------------------------
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    bucket_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

    print("-" * 76)
    print("CALIBRATION TABLE (all stats pooled, over side)")
    print("-" * 76)
    print(f"  {'Bucket':>8s}  {'LD pred':>10s}  {'LD actual':>12s}  {'LD N':>7s}  "
          f"{'Comb pred':>10s}  {'Comb actual':>12s}  {'Comb N':>7s}")
    print(f"  {'--------':>8s}  {'----------':>10s}  {'------------':>12s}  {'-------':>7s}  "
          f"{'----------':>10s}  {'------------':>12s}  {'-------':>7s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]

        ld_mask = (all_ld_p >= lo) & (all_ld_p < hi)
        comb_mask = (all_comb_p >= lo) & (all_comb_p < hi)

        ld_n = ld_mask.sum()
        comb_n = comb_mask.sum()

        if ld_n > 0:
            ld_pred = float(all_ld_p[ld_mask].mean())
            ld_actual = float(all_actual[ld_mask].mean())
            ld_str = f"{ld_pred:>10.4f}  {ld_actual:>12.4f}  {ld_n:>7,}"
        else:
            ld_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        if comb_n > 0:
            comb_pred = float(all_comb_p[comb_mask].mean())
            comb_actual = float(all_actual[comb_mask].mean())
            comb_str = f"{comb_pred:>10.4f}  {comb_actual:>12.4f}  {comb_n:>7,}"
        else:
            comb_str = f"{'--':>10s}  {'--':>12s}  {0:>7d}"

        print(f"  {label:>8s}  {ld_str}  {comb_str}")

    print()

    # ---------------------------------------------------------------
    # Summary verdict
    # ---------------------------------------------------------------
    print("-" * 76)
    print("SUMMARY")
    print("-" * 76)

    # Overall 65% hit rate change
    for label, p_arr in [("LD-only", all_ld_p), ("LD+Speed", all_comb_p)]:
        m65 = p_arr >= 0.65
        m65u = (1 - p_arr) >= 0.65
        total_picks = m65.sum() + m65u.sum()
        if total_picks > 0:
            total_hits = (
                (all_actual[m65].sum() if m65.sum() > 0 else 0)
                + ((1 - all_actual)[m65u].sum() if m65u.sum() > 0 else 0)
            )
            combined_hr = total_hits / total_picks
            print(f"  {label:10s} >= 65%% combined hit rate: {combined_hr:.4f} "
                  f"({combined_hr*100:.1f}%) on {total_picks:,} picks")

    print()
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest sprint speed BABIP adjustments"
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
        "Starting sprint speed BABIP backtest: seasons=%s, max_games=%d, n_sims=%d",
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
