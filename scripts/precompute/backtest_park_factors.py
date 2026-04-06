#!/usr/bin/env python
"""
A/B backtest: Lineup simulator with vs without component park factors.

Runs the same 400 games (2024-2025) through the lineup simulator twice:
  A) Baseline: no park factors (current production behavior)
  B) Park factors: K/BB logit lifts + H BABIP adjustment + HR lift per venue

Same seeds, same games, same posteriors. Compares calibration and
65% confidence hit rate for H, K, BB (the three stats park factors
directly affect).

Also reports by park type: extreme parks (Coors, Oracle, Fenway, Petco,
GABP, Trop, etc.) vs neutral parks.

Usage
-----
    python scripts/precompute/backtest_park_factors.py
    python scripts/precompute/backtest_park_factors.py --max-games 200
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

DASHBOARD_DIR = Path(
    r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard"
)

# Stats that park factors affect
PARK_STATS = ["h", "k", "bb"]
ALL_STATS = ["h", "k", "bb", "r", "rbi", "hrr"]

PROP_LINES = {
    "h": [0.5, 1.5, 2.5],
    "k": [0.5, 1.5],
    "bb": [0.5],
    "r": [0.5, 1.5],
    "rbi": [0.5, 1.5],
    "hrr": [0.5, 1.5, 2.5, 3.5],
}

PRIMARY_LINE = {
    "h": 0.5,
    "k": 0.5,
    "bb": 0.5,
    "r": 0.5,
    "rbi": 0.5,
    "hrr": 1.5,
}

N_SIMS = 10_000
BATCH_SIZE = 25

DEFAULT_BP_K = 0.253
DEFAULT_BP_BB = 0.084
DEFAULT_BP_HR = 0.024

# Extreme venues for park-type breakdown
EXTREME_VENUE_IDS = {
    2395,  # Coors Field (COL)
    3289,  # Oracle Park (SF)
    3,     # Fenway Park (BOS)
    2680,  # Petco Park (SD)
    15,    # Chase Field (ARI)
    680,   # Tropicana Field (TB)
    31,    # Great American Ball Park (CIN)
    12,    # Wrigley Field (CHC)
    2602,  # Yankee Stadium (NYY)
    19,    # Camden Yards (BAL) -- new config, extreme H/K
}

VENUE_NAMES = {
    2395: "Coors", 3289: "Oracle", 3: "Fenway", 2680: "Petco",
    15: "Chase", 680: "Trop", 31: "GABP", 12: "Wrigley",
    2602: "Yankee", 19: "Camden",
}


def load_posteriors() -> dict:
    """Load pre-computed posterior samples and supporting data."""
    logger.info("Loading posteriors...")
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

    # Index bullpen rates
    bp = (
        data["bullpen_rates"]
        .sort_values("season")
        .groupby("team_id").last()
        .reset_index()
    )
    data["bp_lookup"] = {
        int(row["team_id"]): {
            "k_rate": float(row["k_rate"]),
            "bb_rate": float(row["bb_rate"]),
            "hr_rate": float(row["hr_rate"]),
        }
        for _, row in bp.iterrows()
    }

    # Index BF priors
    bf = (
        data["bf_priors"]
        .sort_values("season")
        .groupby("pitcher_id").last()
        .reset_index()
    )
    data["bf_lookup"] = {
        int(row["pitcher_id"]): {
            "mu_bf": float(row["mu_bf"]),
            "sigma_bf": float(row["sigma_bf"]),
        }
        for _, row in bf.iterrows()
    }

    # LD% BABIP adjustments
    data["ld_babip_lookup"] = {}
    _LD_COEFF = 0.25
    _LEAGUE_LD = 0.22
    try:
        ld_df = pd.read_parquet(DASHBOARD_DIR / "batter_ld_rate.parquet")
        if not ld_df.empty:
            ld_latest = (
                ld_df.sort_values("season")
                .groupby("player_id").last()
                .reset_index()
            )
            for _, lr in ld_latest.iterrows():
                ld_dev = float(lr["ld_rate_regressed"]) - _LEAGUE_LD
                data["ld_babip_lookup"][int(lr["player_id"])] = ld_dev * _LD_COEFF
    except FileNotFoundError:
        pass

    # Sprint speed BABIP adjustments
    data["speed_babip_lookup"] = {}
    _LEAGUE_SPEED = 27.0
    _SPEED_COEFF = 0.010
    try:
        speed_df = pd.read_parquet(DASHBOARD_DIR / "batter_sprint_speed.parquet")
        if not speed_df.empty:
            speed_latest = (
                speed_df.sort_values("season")
                .groupby("player_id").last()
                .reset_index()
            )
            for _, sr in speed_latest.iterrows():
                speed_dev = float(sr["sprint_speed_regressed"]) - _LEAGUE_SPEED
                data["speed_babip_lookup"][int(sr["player_id"])] = (
                    speed_dev * _SPEED_COEFF
                )
    except FileNotFoundError:
        pass

    logger.info(
        "Loaded: %d hitters, %d pitchers, %d BF priors, %d bullpen, "
        "%d LD BABIP, %d speed BABIP",
        len(set(data["hitter_k"].files)),
        len(set(data["pitcher_k"].files)),
        len(data["bf_lookup"]),
        len(data["bp_lookup"]),
        len(data["ld_babip_lookup"]),
        len(data["speed_babip_lookup"]),
    )
    return data


def load_park_factor_lifts() -> dict[int, dict[str, float]]:
    """Load park factor lifts keyed by venue_id."""
    try:
        lifts_df = pd.read_parquet(DASHBOARD_DIR / "park_factor_lifts.parquet")
    except FileNotFoundError:
        logger.warning("No park_factor_lifts.parquet found -- run precompute_park_factors.py first")
        return {}

    lookup = {}
    for _, row in lifts_df.iterrows():
        lookup[int(row["venue_id"])] = {
            "k_lift": float(row["k_lift"]),
            "bb_lift": float(row["bb_lift"]),
            "hr_lift": float(row["hr_lift"]),
            "h_babip_adj": float(row["h_babip_adj"]),
        }
    logger.info("Loaded park factor lifts for %d venues", len(lookup))
    return lookup


def fetch_backtest_games(
    seasons: list[int],
    max_games: int | None = None,
) -> pd.DataFrame:
    """Fetch games with venue_id, full lineups, boxscores, and opposing starters."""
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
    WITH starters AS (
        SELECT
            pb.pitcher_id,
            pb.game_pk,
            pb.team_id AS pitcher_team_id
        FROM staging.pitching_boxscores pb
        WHERE pb.is_starter = TRUE
          AND pb.batters_faced >= 9
    ),
    game_season AS (
        SELECT game_pk, season, home_team_id, away_team_id, venue_id
        FROM production.dim_game
        WHERE game_type = 'R' AND season IN ({season_list})
    )
    SELECT
        fl.game_pk,
        gs.season,
        gs.venue_id,
        fl.player_id AS batter_id,
        fl.batting_order,
        fl.team_id,
        bb.hits AS actual_h,
        bb.runs AS actual_r,
        bb.rbi AS actual_rbi,
        bb.strikeouts AS actual_k,
        bb.walks AS actual_bb,
        bb.home_runs AS actual_hr,
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
    logger.info("Fetching backtest games for seasons %s with venue_id...", seasons)
    df = read_sql(query)
    logger.info("Raw rows: %d", len(df))

    if df.empty:
        return df

    # Filter to full 9-man lineups
    counts = df.groupby(["game_pk", "team_id"]).size().reset_index(name="n_batters")
    full_sides = counts[counts["n_batters"] == 9][["game_pk", "team_id"]]
    df = df.merge(full_sides, on=["game_pk", "team_id"])

    # Filter to sides with opposing starter
    df = df[df["opp_starter_id"].notna()].copy()
    df["opp_starter_id"] = df["opp_starter_id"].astype(int)
    df["actual_hrr"] = df["actual_h"] + df["actual_r"] + df["actual_rbi"]

    n_sides = df.groupby(["game_pk", "team_id"]).ngroups
    logger.info(
        "After filtering: %d rows (%d team-game sides), %d unique venues",
        len(df), n_sides, df["venue_id"].nunique(),
    )

    if max_games is not None and n_sides > max_games:
        side_keys = df[["game_pk", "team_id"]].drop_duplicates()
        sampled = side_keys.sample(n=max_games, random_state=42)
        df = df.merge(sampled, on=["game_pk", "team_id"])
        n_sides = max_games
        logger.info("Sampled to %d team-game sides (%d rows)", n_sides, len(df))

    return df


def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    park_lifts: dict[int, dict[str, float]] | None,
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run lineup sim for one team-side with optional park factors.

    If park_lifts is None, runs baseline (no park factors).
    """
    side_df = side_df.sort_values("batting_order")
    batter_ids = side_df["batter_id"].astype(int).tolist()
    opp_starter_id = int(side_df.iloc[0]["opp_starter_id"])
    opp_team_id = int(side_df.iloc[0]["opp_team_id"])
    venue_id = int(side_df.iloc[0]["venue_id"])
    pid_str = str(opp_starter_id)

    # Check pitcher posteriors
    if pid_str not in posteriors["pitcher_k"].files:
        return None
    if pid_str not in posteriors["pitcher_bb"].files:
        return None
    if pid_str not in posteriors["pitcher_hr"].files:
        return None

    # Build batter samples
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

    if valid_count < 5:
        return None

    # Pitcher rates
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

    # Per-batter BABIP adjustments (LD% + sprint speed)
    babip_adjs = np.array([
        posteriors["ld_babip_lookup"].get(bid, 0.0)
        + posteriors["speed_babip_lookup"].get(bid, 0.0)
        for bid in batter_ids
    ])

    # Park factor lifts
    if park_lifts is not None:
        pf = park_lifts.get(venue_id, {})
        p_k_lift = pf.get("k_lift", 0.0)
        p_bb_lift = pf.get("bb_lift", 0.0)
        p_hr_lift = pf.get("hr_lift", 0.0)
        p_h_babip = pf.get("h_babip_adj", 0.0)
    else:
        p_k_lift = 0.0
        p_bb_lift = 0.0
        p_hr_lift = 0.0
        p_h_babip = 0.0

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
        batter_babip_adjs=babip_adjs,
        park_k_lift=p_k_lift,
        park_bb_lift=p_bb_lift,
        park_hr_lift=p_hr_lift,
        park_h_babip_adj=p_h_babip,
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
            "venue_id": int(row["venue_id"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "opp_starter_id": int(row["opp_starter_id"]),
            "has_posterior": has_posterior,
            "actual_h": int(row["actual_h"]),
            "actual_r": int(row["actual_r"]),
            "actual_rbi": int(row["actual_rbi"]),
            "actual_hrr": int(row["actual_hrr"]),
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
            "actual_pa": int(row["actual_pa"]),
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


def run_one_variant(
    label: str,
    sides: list,
    posteriors: dict,
    park_lifts: dict[int, dict[str, float]] | None,
    n_sims: int,
) -> pd.DataFrame:
    """Run all sides for one variant (baseline or park factors)."""
    all_records: list[dict] = []
    n_skipped = 0
    t0 = time.perf_counter()
    n_sides = len(sides)

    for idx, ((game_pk, team_id), side_df) in enumerate(sides):
        recs = simulate_one_side(side_df, posteriors, park_lifts, n_sims=n_sims)
        if recs is None:
            n_skipped += 1
        else:
            all_records.extend(recs)

        if (idx + 1) % BATCH_SIZE == 0 or idx == n_sides - 1:
            elapsed = time.perf_counter() - t0
            pct = 100 * (idx + 1) / n_sides
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "[%s] %d/%d (%.0f%%) | %.1f sides/sec | %d skipped | %d batters",
                label, idx + 1, n_sides, pct, rate, n_skipped, len(all_records),
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] Done: %d batter-games in %.1fs. Skipped %d.",
        label, len(all_records), elapsed, n_skipped,
    )
    return pd.DataFrame(all_records)


def compute_comparison(df_base: pd.DataFrame, df_park: pd.DataFrame) -> None:
    """Print head-to-head comparison of baseline vs park-factor model."""

    # Filter to batters with real posteriors
    base = df_base[df_base["has_posterior"]].copy()
    park = df_park[df_park["has_posterior"]].copy()

    n_batters = len(base)
    n_games = base["game_pk"].nunique()
    n_venues = base["venue_id"].nunique()

    print()
    print("=" * 78)
    print("PARK FACTORS A/B BACKTEST RESULTS")
    print("=" * 78)
    print(f"  Batter-games:    {n_batters:,}")
    print(f"  Unique games:    {n_games:,}")
    print(f"  Unique venues:   {n_venues}")
    print(f"  Sims per game:   {N_SIMS:,}")
    print()

    # =====================================================================
    # 1. 65% confidence hit rate
    # =====================================================================
    print()
    print("-" * 78)
    print("65% CONFIDENCE HIT RATE")
    print("-" * 78)

    for label, df in [("Baseline", base), ("Park PF", park)]:
        all_p = []
        all_a = []
        for stat in ALL_STATS:
            for line in PROP_LINES[stat]:
                col = f"p_over_{stat}_{line}"
                if col not in df.columns:
                    continue
                p = df[col].values
                actual = (df[f"actual_{stat}"].values > line).astype(float)
                all_p.append(p)
                all_a.append(actual)

        all_p = np.concatenate(all_p)
        all_a = np.concatenate(all_a)

        # Over side
        m65 = all_p >= 0.65
        if m65.sum() > 0:
            hr65 = float(all_a[m65].mean())
            n65 = m65.sum()
            print(f"  {label:10s}  P(over)>=65%: {hr65:.4f} ({hr65*100:.1f}%) "
                  f"on {n65:,} picks")

        # Under side
        all_p_u = 1 - all_p
        all_a_u = 1 - all_a
        m65u = all_p_u >= 0.65
        if m65u.sum() > 0:
            hr65u = float(all_a_u[m65u].mean())
            n65u = m65u.sum()
            print(f"  {label:10s}  P(under)>=65%: {hr65u:.4f} ({hr65u*100:.1f}%) "
                  f"on {n65u:,} picks")

    # =====================================================================
    # 3. Per-stat 65% hit rate
    # =====================================================================
    print()
    print("-" * 78)
    print("PER-STAT 65% HIT RATE (park-affected stats)")
    print("-" * 78)
    print(f"  {'Stat':>4s}  {'Line':>5s}  {'Base hit%':>10s}  {'Park hit%':>10s}  "
          f"{'Delta':>8s}  {'N_base':>8s}  {'N_park':>8s}")

    for stat in PARK_STATS:
        for line in PROP_LINES[stat]:
            col = f"p_over_{stat}_{line}"
            if col not in base.columns:
                continue

            # Baseline
            p_b = base[col].values
            a_b = (base[f"actual_{stat}"].values > line).astype(float)
            m_b = p_b >= 0.65
            hr_b = float(a_b[m_b].mean()) if m_b.sum() > 0 else 0
            n_b = int(m_b.sum())

            # Park
            p_p = park[col].values
            a_p = (park[f"actual_{stat}"].values > line).astype(float)
            m_p = p_p >= 0.65
            hr_p = float(a_p[m_p].mean()) if m_p.sum() > 0 else 0
            n_p = int(m_p.sum())

            delta = hr_p - hr_b
            print(f"  {stat.upper():>4s}  {line:>5.1f}  {hr_b:>10.4f}  {hr_p:>10.4f}  "
                  f"{delta:>+8.4f}  {n_b:>8,}  {n_p:>8,}")

    # =====================================================================
    # 4. Per-venue breakdown for extreme parks
    # =====================================================================
    print("-" * 78)
    print("PER-VENUE IMPACT (extreme parks)")
    print("-" * 78)

    for vid in sorted(EXTREME_VENUE_IDS):
        b_v = base[base["venue_id"] == vid]
        p_v = park[park["venue_id"] == vid]
        if len(b_v) == 0:
            continue

        vname = VENUE_NAMES.get(vid, str(vid))
        print(f"\n  {vname} (venue {vid}, {len(b_v)} batters)")

        # Show park factor lifts being applied
        pf_lifts = {}
        try:
            lifts_df = pd.read_parquet(DASHBOARD_DIR / "park_factor_lifts.parquet")
            pf_row = lifts_df[lifts_df["venue_id"] == vid]
            if len(pf_row) > 0:
                r = pf_row.iloc[0]
                pf_lifts = {
                    "k_lift": float(r["k_lift"]),
                    "bb_lift": float(r["bb_lift"]),
                    "hr_lift": float(r["hr_lift"]),
                    "h_babip": float(r["h_babip_adj"]),
                }
                print(f"    Lifts: K={pf_lifts['k_lift']:+.4f}  "
                      f"BB={pf_lifts['bb_lift']:+.4f}  "
                      f"HR={pf_lifts['hr_lift']:+.4f}  "
                      f"H_babip={pf_lifts['h_babip']:+.4f}")
        except FileNotFoundError:
            pass

        for stat in PARK_STATS:
            actual = b_v[f"actual_{stat}"]
            bias_b = float(np.mean(b_v[f"pred_{stat}"] - actual))
            bias_p = float(np.mean(p_v[f"pred_{stat}"] - actual))
            print(f"    {stat.upper():>2s}:  Bias {bias_b:+.4f} -> {bias_p:+.4f}")

    print()
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B backtest: lineup sim with/without park factors"
    )
    parser.add_argument(
        "--season", type=int, nargs="+", default=[2024, 2025],
        help="Seasons to backtest (default: 2024 2025)",
    )
    parser.add_argument(
        "--max-games", type=int, default=400,
        help="Max team-game sides to simulate (default: 400)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=N_SIMS,
        help="Monte Carlo sims per game (default: 10000)",
    )
    args = parser.parse_args()

    posteriors = load_posteriors()
    park_lifts_full = load_park_factor_lifts()
    games_df = fetch_backtest_games(args.season, max_games=args.max_games)

    if games_df.empty:
        logger.error("No games found")
        return

    sides = list(games_df.groupby(["game_pk", "team_id"]))

    # Build variant lift dictionaries:
    # 1) Full: all 4 components (K, BB, HR, H_babip)
    # 2) K+HR only: zero out BB and H BABIP
    # 3) Damped 50%: all 4 at half strength
    park_lifts_k_hr = {}
    park_lifts_damped = {}
    for vid, pf in park_lifts_full.items():
        park_lifts_k_hr[vid] = {
            "k_lift": pf["k_lift"],
            "bb_lift": 0.0,
            "hr_lift": pf["hr_lift"],
            "h_babip_adj": 0.0,
        }
        park_lifts_damped[vid] = {
            "k_lift": pf["k_lift"] * 0.5,
            "bb_lift": pf["bb_lift"] * 0.5,
            "hr_lift": pf["hr_lift"] * 0.5,
            "h_babip_adj": pf["h_babip_adj"] * 0.5,
        }

    logger.info("Running 4 variants on %d team-game sides...", len(sides))

    # Variant A: Baseline (no park factors)
    logger.info("=" * 60)
    logger.info("VARIANT A: BASELINE (no park factors)")
    df_base = run_one_variant("Baseline", sides, posteriors, None, args.n_sims)

    # Variant B: Full park factors (all 4 components)
    logger.info("=" * 60)
    logger.info("VARIANT B: FULL PARK FACTORS (K + BB + HR + H_babip)")
    df_full = run_one_variant("Full PF", sides, posteriors, park_lifts_full, args.n_sims)

    # Variant C: K + HR only (skip BB and H BABIP)
    logger.info("=" * 60)
    logger.info("VARIANT C: K + HR ONLY")
    df_k_hr = run_one_variant("K+HR", sides, posteriors, park_lifts_k_hr, args.n_sims)

    # Variant D: Damped 50% (all 4 at half strength)
    logger.info("=" * 60)
    logger.info("VARIANT D: DAMPED 50% (all components at half strength)")
    df_damped = run_one_variant("Damp50", sides, posteriors, park_lifts_damped, args.n_sims)

    if df_base.empty:
        logger.error("No results to compare")
        return

    print("\n\n")
    print("#" * 78)
    print("# VARIANT B vs BASELINE: FULL PARK FACTORS")
    print("#" * 78)
    compute_comparison(df_base, df_full)

    print("\n\n")
    print("#" * 78)
    print("# VARIANT C vs BASELINE: K + HR ONLY (no BB, no H_babip)")
    print("#" * 78)
    compute_comparison(df_base, df_k_hr)

    print("\n\n")
    print("#" * 78)
    print("# VARIANT D vs BASELINE: ALL COMPONENTS AT 50% STRENGTH")
    print("#" * 78)
    compute_comparison(df_base, df_damped)


if __name__ == "__main__":
    main()
