#!/usr/bin/env python
"""
Backtest the lineup simulator against actual batter game results.

Pulls historical game data (lineups, boxscores, opposing starters) from the
database, runs simulate_lineup_game for each game using pre-computed posterior
samples, and compares predicted vs actual per-batter stats.

Reports MAE, calibration tables, and P(over) accuracy for H, R, RBI, HRR, K, BB.

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
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.models.game_sim.lineup_simulator import simulate_lineup_game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")

# Default prop lines for calibration
PROP_LINES = {
    "h": [0.5, 1.5, 2.5],
    "r": [0.5, 1.5],
    "rbi": [0.5, 1.5],
    "hrr": [0.5, 1.5, 2.5, 3.5],
    "k": [0.5, 1.5],
    "bb": [0.5],
}

# Primary line per stat (used for the headline calibration check)
PRIMARY_LINE = {
    "h": 0.5,
    "r": 0.5,
    "rbi": 0.5,
    "hrr": 1.5,
    "k": 0.5,
    "bb": 0.5,
}

N_SIMS = 10_000
BATCH_SIZE = 25  # games per progress update

# League-average bullpen fallback
DEFAULT_BP_K = 0.253
DEFAULT_BP_BB = 0.084
DEFAULT_BP_HR = 0.024


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


def fetch_backtest_games(
    seasons: list[int],
    max_games: int | None = None,
) -> pd.DataFrame:
    """Fetch games with full 9-man lineups, boxscores, and opposing starters.

    Returns one row per batter per game, with batting_order 1-9 and actuals.
    """
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH starters AS (
        SELECT
            pb.pitcher_id,
            pb.game_pk,
            pb.team_id AS pitcher_team_id
        FROM staging.pitching_boxscores pb
        WHERE pb.is_starter = TRUE
    ),
    game_season AS (
        SELECT game_pk, season, home_team_id, away_team_id
        FROM production.dim_game
        WHERE game_type = 'R' AND season IN ({season_list})
    )
    SELECT
        fl.game_pk,
        gs.season,
        fl.player_id AS batter_id,
        fl.batting_order,
        fl.team_id,
        bb.hits AS actual_h,
        bb.runs AS actual_r,
        bb.rbi AS actual_rbi,
        bb.strikeouts AS actual_k,
        bb.walks AS actual_bb,
        bb.home_runs AS actual_hr,
        bb.doubles AS actual_2b,
        bb.triples AS actual_3b,
        bb.total_bases AS actual_tb,
        bb.plate_appearances AS actual_pa,
        st.pitcher_id AS opp_starter_id,
        st.pitcher_team_id AS opp_team_id
    FROM production.fact_lineup fl
    JOIN game_season gs ON fl.game_pk = gs.game_pk
    JOIN staging.batting_boxscores bb
      ON fl.player_id = bb.batter_id AND fl.game_pk = bb.game_pk
    LEFT JOIN starters st
      ON fl.game_pk = st.game_pk AND st.pitcher_team_id != fl.team_id
    WHERE fl.is_starter = TRUE
      AND fl.batting_order BETWEEN 1 AND 9
      AND bb.plate_appearances >= 1
    ORDER BY fl.game_pk, fl.team_id, fl.batting_order
    """
    logger.info("Fetching backtest game data for seasons %s...", seasons)
    df = read_sql(query)
    logger.info("Raw rows: %d", len(df))

    if df.empty:
        return df

    # Filter to games that have exactly 9 batters per team-side
    counts = df.groupby(["game_pk", "team_id"]).size().reset_index(name="n_batters")
    full_sides = counts[counts["n_batters"] == 9][["game_pk", "team_id"]]
    df = df.merge(full_sides, on=["game_pk", "team_id"])

    # Filter to sides that have an opposing starter identified
    df = df[df["opp_starter_id"].notna()].copy()
    df["opp_starter_id"] = df["opp_starter_id"].astype(int)

    # Compute HRR actual
    df["actual_hrr"] = df["actual_h"] + df["actual_r"] + df["actual_rbi"]

    n_sides = df.groupby(["game_pk", "team_id"]).ngroups
    logger.info(
        "After filtering to full 9-man lineups with starters: %d rows (%d team-game sides)",
        len(df), n_sides,
    )

    if max_games is not None and n_sides > max_games:
        # Sample at the team-game level
        side_keys = df[["game_pk", "team_id"]].drop_duplicates()
        sampled = side_keys.sample(n=max_games, random_state=42)
        df = df.merge(sampled, on=["game_pk", "team_id"])
        logger.info("Sampled down to %d team-game sides (%d rows)", max_games, len(df))

    return df


def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run the lineup simulator for one team-side of a game.

    Returns a list of 9 dicts (one per batter) with predictions, or None
    if posteriors are missing for too many batters.
    """
    side_df = side_df.sort_values("batting_order")
    batter_ids = side_df["batter_id"].astype(int).tolist()
    opp_starter_id = int(side_df.iloc[0]["opp_starter_id"])
    opp_team_id = int(side_df.iloc[0]["opp_team_id"])
    pid_str = str(opp_starter_id)

    # Check pitcher posteriors
    if pid_str not in posteriors["pitcher_k"].files:
        return None
    if pid_str not in posteriors["pitcher_bb"].files:
        return None
    if pid_str not in posteriors["pitcher_hr"].files:
        return None

    # Build batter sample lists (need all 9)
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
            # Use league-average fallback for missing batters
            rng = np.random.default_rng(bid % 100000)
            batter_k_samples.append(
                np.clip(rng.normal(0.226, 0.03, 2000), 0.05, 0.50)
            )
            batter_bb_samples.append(
                np.clip(rng.normal(0.082, 0.02, 2000), 0.02, 0.25)
            )
            batter_hr_samples.append(
                np.clip(rng.normal(0.031, 0.01, 2000), 0.005, 0.10)
            )

    # Require at least 5 of 9 batters with real posteriors
    if valid_count < 5:
        return None

    # Pitcher rates (posterior means)
    starter_k = float(np.mean(posteriors["pitcher_k"][pid_str]))
    starter_bb = float(np.mean(posteriors["pitcher_bb"][pid_str]))
    starter_hr = float(np.mean(posteriors["pitcher_hr"][pid_str]))

    # BF priors
    bf_info = posteriors["bf_lookup"].get(opp_starter_id)
    if bf_info:
        mu_bf = bf_info["mu_bf"]
        sigma_bf = bf_info["sigma_bf"]
    else:
        mu_bf = 22.0
        sigma_bf = 3.4

    # Bullpen rates
    bp_info = posteriors["bp_lookup"].get(opp_team_id)
    if bp_info:
        bp_k = bp_info["k_rate"]
        bp_bb = bp_info["bb_rate"]
        bp_hr = bp_info["hr_rate"]
    else:
        bp_k = DEFAULT_BP_K
        bp_bb = DEFAULT_BP_BB
        bp_hr = DEFAULT_BP_HR

    # Run simulation
    game_pk = int(side_df.iloc[0]["game_pk"])
    result = simulate_lineup_game(
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
        random_seed=game_pk % (2**31),
    )

    # Extract per-batter predictions
    records = []
    for i, (_, row) in enumerate(side_df.iterrows()):
        bid = int(row["batter_id"])
        bid_str = str(bid)
        has_posterior = (
            bid_str in posteriors["hitter_k"].files
            and bid_str in posteriors["hitter_bb"].files
            and bid_str in posteriors["hitter_hr"].files
        )

        br = result.batter_result(i)
        h_samples = br["h_samples"]
        r_samples = br["r_samples"]
        rbi_samples = br["rbi_samples"]
        k_samples = br["k_samples"]
        bb_samples = br["bb_samples"]
        hrr_samples = h_samples + r_samples + rbi_samples

        rec = {
            "game_pk": int(row["game_pk"]),
            "season": int(row["season"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "opp_starter_id": int(row["opp_starter_id"]),
            "has_posterior": has_posterior,
            # Actuals
            "actual_h": int(row["actual_h"]),
            "actual_r": int(row["actual_r"]),
            "actual_rbi": int(row["actual_rbi"]),
            "actual_hrr": int(row["actual_hrr"]),
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
            "actual_pa": int(row["actual_pa"]),
            # Predicted means
            "pred_h": float(np.mean(h_samples)),
            "pred_r": float(np.mean(r_samples)),
            "pred_rbi": float(np.mean(rbi_samples)),
            "pred_hrr": float(np.mean(hrr_samples)),
            "pred_k": float(np.mean(k_samples)),
            "pred_bb": float(np.mean(bb_samples)),
        }

        # P(over) at each prop line
        for stat, lines in PROP_LINES.items():
            if stat == "hrr":
                samples = hrr_samples
            else:
                samples = br[f"{stat}_samples"]
            for line in lines:
                p_over = float(np.mean(samples > line))
                rec[f"p_over_{stat}_{line}"] = p_over

        records.append(rec)

    return records


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

    # Group by (game_pk, team_id) = one side of a game
    sides = list(games_df.groupby(["game_pk", "team_id"]))
    n_sides = len(sides)
    logger.info("Simulating %d team-game sides with %d sims each...", n_sides, n_sims)

    all_records: list[dict] = []
    n_skipped = 0
    t0 = time.perf_counter()

    for idx, ((game_pk, team_id), side_df) in enumerate(sides):
        recs = simulate_one_side(side_df, posteriors, n_sims=n_sims)
        if recs is None:
            n_skipped += 1
        else:
            all_records.extend(recs)

        if (idx + 1) % BATCH_SIZE == 0 or idx == n_sides - 1:
            elapsed = time.perf_counter() - t0
            pct = 100 * (idx + 1) / n_sides
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d sides (%.1f%%) | %.1f sides/sec | "
                "%d skipped | %d batter-games collected",
                idx + 1, n_sides, pct, rate, n_skipped, len(all_records),
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Backtest complete: %d batter-games in %.1fs (%.1f sides/sec). "
        "Skipped %d sides (missing posteriors).",
        len(all_records), elapsed, n_sides / elapsed if elapsed > 0 else 0,
        n_skipped,
    )

    return pd.DataFrame(all_records)


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

    # --- MAE per stat ---
    print("-" * 72)
    print("MEAN ABSOLUTE ERROR (batters with posteriors only)")
    print("-" * 72)
    stats = ["h", "r", "rbi", "hrr", "k", "bb"]
    mae_results = {}
    for stat in stats:
        actual = df_real[f"actual_{stat}"]
        pred = df_real[f"pred_{stat}"]
        mae = float(np.mean(np.abs(actual - pred)))
        bias = float(np.mean(pred - actual))
        mae_results[stat] = mae
        print(f"  {stat.upper():>4s}:  MAE = {mae:.4f}   Bias = {bias:+.4f}   "
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

    # Pool all stat-line combinations
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    bucket_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

    all_p_over = []
    all_actual_over = []

    for stat in stats:
        for line in PROP_LINES[stat]:
            col = f"p_over_{stat}_{line}"
            if col not in df_real.columns:
                continue
            p = df_real[col].values
            actual = (df_real[f"actual_{stat}"].values > line).astype(float)
            all_p_over.append(p)
            all_actual_over.append(actual)

    all_p = np.concatenate(all_p_over)
    all_a = np.concatenate(all_actual_over)

    print(f"  {'Bucket':>8s}  {'Predicted':>10s}  {'Actual':>10s}  "
          f"{'N':>8s}  {'Gap':>8s}")
    print(f"  {'--------':>8s}  {'----------':>10s}  {'----------':>10s}  "
          f"{'--------':>8s}  {'--------':>8s}")

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        mask = (all_p >= lo) & (all_p < hi)
        n_in = mask.sum()
        if n_in == 0:
            print(f"  {label:>8s}  {'--':>10s}  {'--':>10s}  {0:>8d}  {'--':>8s}")
            continue
        pred_mean = float(all_p[mask].mean())
        actual_mean = float(all_a[mask].mean())
        gap = actual_mean - pred_mean
        print(f"  {label:>8s}  {pred_mean:>10.4f}  {actual_mean:>10.4f}  "
              f"{n_in:>8,}  {gap:>+8.4f}")

    # Also do under side
    print()
    print("  (Under side)")
    all_p_under = 1 - all_p
    all_a_under = 1 - all_a

    for i, label in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        mask = (all_p_under >= lo) & (all_p_under < hi)
        n_in = mask.sum()
        if n_in == 0:
            continue
        pred_mean = float(all_p_under[mask].mean())
        actual_mean = float(all_a_under[mask].mean())
        gap = actual_mean - pred_mean
        print(f"  {label:>8s}  {pred_mean:>10.4f}  {actual_mean:>10.4f}  "
              f"{n_in:>8,}  {gap:>+8.4f}")

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

    for stat in stats:
        lines = PROP_LINES[stat]
        print(f"\n  {stat.upper()}:")
        print(f"    {'Line':>5s}  {'P(over) mean':>13s}  {'Actual %':>10s}  "
              f"{'Gap':>8s}  {'N':>8s}  {'>=63% hit':>10s}  {'N_63':>6s}")

        for line in lines:
            col = f"p_over_{stat}_{line}"
            if col not in df_real.columns:
                continue
            p = df_real[col].values
            actual = (df_real[f"actual_{stat}"].values > line).astype(float)

            avg_p = float(p.mean())
            actual_rate = float(actual.mean())
            gap = actual_rate - avg_p
            n = len(p)

            # Hit rate at >= 63%
            m63 = p >= 0.63
            if m63.sum() > 0:
                hr63 = float(actual[m63].mean())
                n63 = m63.sum()
                hr63_str = f"{hr63:.4f}"
                n63_str = f"{n63:,}"
            else:
                hr63_str = "--"
                n63_str = "0"

            print(f"    {line:>5.1f}  {avg_p:>13.4f}  {actual_rate:>10.4f}  "
                  f"{gap:>+8.4f}  {n:>8,}  {hr63_str:>10s}  {n63_str:>6s}")

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
