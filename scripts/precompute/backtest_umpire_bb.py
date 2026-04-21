#!/usr/bin/env python
"""
A/B backtest: Lineup simulator with vs without umpire BB lift.

Runs the same 400 games (2024-2025) through the lineup simulator twice:
  A) Baseline: umpire K lift only (current production behavior)
  B) Umpire BB: umpire K lift + umpire BB lift

Same seeds, same games, same posteriors. Compares Brier score,
Brier skill score and 65% confidence hit rate.

Primary evaluation metric: **Brier score** = mean((p_over - actual_binary)^2)

Usage
-----
    python scripts/precompute/backtest_umpire_bb.py
    python scripts/precompute/backtest_umpire_bb.py --max-games 200
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

from scripts.precompute import DASHBOARD_DIR
from scripts.precompute.backtest_harness import (
    ALL_STATS,
    N_SIMS,
    PROP_LINES,
    PRIMARY_LINE,
    fetch_backtest_games as _harness_fetch,
    run_sides_loop,
    simulate_one_side as _harness_simulate_one_side,
)
from scripts.precompute.backtest_lineup_sim import (
    load_posteriors as _load_base_posteriors,
)
from src.data.queries import get_umpire_tendencies
from src.evaluation.runner import setup_logging
from src.utils.constants import (
    BABIP_LD_COEFF,
    BABIP_LEAGUE_LD_RATE,
    BABIP_LEAGUE_SPEED,
    BABIP_SPEED_COEFF,
)

logger = setup_logging(__name__)
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_posteriors() -> dict:
    """Load base posteriors and add LD% / sprint-speed BABIP lookups."""
    data = _load_base_posteriors()

    # LD% BABIP adjustments
    data["ld_babip_lookup"] = {}
    try:
        ld_df = pd.read_parquet(DASHBOARD_DIR / "batter_ld_rate.parquet")
        if not ld_df.empty:
            ld_latest = (
                ld_df.sort_values("season")
                .groupby("player_id").last()
                .reset_index()
            )
            for _, lr in ld_latest.iterrows():
                ld_dev = float(lr["ld_rate_regressed"]) - BABIP_LEAGUE_LD_RATE
                data["ld_babip_lookup"][int(lr["player_id"])] = ld_dev * BABIP_LD_COEFF
    except FileNotFoundError:
        pass

    # Sprint speed BABIP adjustments
    data["speed_babip_lookup"] = {}
    try:
        speed_df = pd.read_parquet(DASHBOARD_DIR / "batter_sprint_speed.parquet")
        if not speed_df.empty:
            speed_latest = (
                speed_df.sort_values("season")
                .groupby("player_id").last()
                .reset_index()
            )
            for _, sr in speed_latest.iterrows():
                speed_dev = float(sr["sprint_speed_regressed"]) - BABIP_LEAGUE_SPEED
                data["speed_babip_lookup"][int(sr["player_id"])] = (
                    speed_dev * BABIP_SPEED_COEFF
                )
    except FileNotFoundError:
        pass

    logger.info(
        "  + LD BABIP: %d, speed BABIP: %d",
        len(data["ld_babip_lookup"]),
        len(data["speed_babip_lookup"]),
    )
    return data


def load_umpire_lifts(train_seasons: list[int]) -> dict[str, dict[str, float]]:
    """Load per-umpire K and BB logit lifts from training seasons.

    Returns dict keyed by umpire name, each value is
    {"k_lift": float, "bb_lift": float}.
    """
    ump_df = get_umpire_tendencies(seasons=train_seasons, min_games=30)
    if ump_df.empty:
        return {}
    lookup: dict[str, dict[str, float]] = {}
    for _, row in ump_df.iterrows():
        lookup[row["hp_umpire_name"]] = {
            "k_lift": float(row["k_logit_lift"]),
            "bb_lift": float(row.get("bb_logit_lift", 0.0)),
        }
    logger.info(
        "Loaded umpire lifts for %d umpires (train seasons: %s)",
        len(lookup), train_seasons,
    )

    # Show extremes
    names = list(lookup.keys())
    bb_lifts = [lookup[n]["bb_lift"] for n in names]
    sorted_idx = np.argsort(bb_lifts)
    logger.info("  Tightest zone (fewest BB): %s (%.4f)",
                names[sorted_idx[0]], bb_lifts[sorted_idx[0]])
    logger.info("  Loosest zone (most BB): %s (%.4f)",
                names[sorted_idx[-1]], bb_lifts[sorted_idx[-1]])

    return lookup


def fetch_backtest_games(
    seasons: list[int],
    max_games: int | None = None,
) -> pd.DataFrame:
    """Fetch games with umpire info via harness ``fetch_backtest_games``.

    The umpire backtest needs special sampling logic (prefer games with
    umpire data), so we fetch more than ``max_games`` and post-filter.
    """
    # Fetch without sampling first so we can apply umpire-aware sampling
    df = _harness_fetch(
        seasons,
        max_games=None,
        extra_select=["gs.venue_id", "du.hp_umpire_name"],
        extra_joins=[
            "LEFT JOIN production.dim_umpire du ON fl.game_pk = du.game_pk",
        ],
    )

    if df.empty:
        return df

    n_sides = df.groupby(["game_pk", "team_id"]).ngroups
    n_with_ump = df[df["hp_umpire_name"].notna()]["game_pk"].nunique()
    logger.info(
        "Umpire games: %d of %d total sides", n_with_ump, n_sides,
    )

    if max_games is not None and n_sides > max_games:
        # Prefer games that have umpire data
        side_keys = df[["game_pk", "team_id"]].drop_duplicates()
        ump_games = set(df[df["hp_umpire_name"].notna()]["game_pk"].unique())
        side_keys["has_ump"] = side_keys["game_pk"].isin(ump_games)
        ump_sides = side_keys[side_keys["has_ump"]]
        if len(ump_sides) > max_games:
            ump_sides = ump_sides.sample(n=max_games, random_state=42)
            sampled = ump_sides
        else:
            remaining = max_games - len(ump_sides)
            non_ump_sides = side_keys[~side_keys["has_ump"]]
            if len(non_ump_sides) > remaining:
                non_ump_sides = non_ump_sides.sample(n=remaining, random_state=42)
            sampled = pd.concat([ump_sides, non_ump_sides])
        df = df.merge(sampled[["game_pk", "team_id"]], on=["game_pk", "team_id"])
        n_sides = len(sampled)
        n_with_ump = df[df["hp_umpire_name"].notna()]["game_pk"].nunique()
        logger.info(
            "Sampled to %d team-game sides (%d rows), %d with umpire",
            n_sides, len(df), n_with_ump,
        )

    return df


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    ump_lifts: dict[str, dict[str, float]],
    use_umpire_bb: bool,
    n_sims: int = N_SIMS,
) -> list[dict] | None:
    """Run lineup sim for one team-side.

    If use_umpire_bb is True, passes both umpire K and BB lifts.
    If False, passes only umpire K lift (BB lift = 0).
    """
    batter_ids = side_df.sort_values("batting_order")["batter_id"].astype(int).tolist()

    # Per-batter BABIP adjustments (LD% + sprint speed)
    babip_adjs = np.array([
        posteriors["ld_babip_lookup"].get(bid, 0.0)
        + posteriors["speed_babip_lookup"].get(bid, 0.0)
        for bid in batter_ids
    ])

    # Umpire lifts
    hp_ump = side_df.iloc[0].get("hp_umpire_name", "")
    ump_info = ump_lifts.get(hp_ump, {}) if pd.notna(hp_ump) and hp_ump else {}
    ump_k = ump_info.get("k_lift", 0.0)
    ump_bb = ump_info.get("bb_lift", 0.0) if use_umpire_bb else 0.0

    ctx = dict(
        batter_babip_adjs=babip_adjs,
        umpire_k_lift=ump_k,
        umpire_bb_lift=ump_bb,
    )

    extra = {
        "hp_umpire_name": hp_ump if pd.notna(hp_ump) else "",
        "has_umpire": bool(pd.notna(hp_ump) and hp_ump and hp_ump in ump_lifts),
        "ump_k_lift": ump_k,
        "ump_bb_lift": ump_bb if use_umpire_bb else 0.0,
    }

    return _harness_simulate_one_side(
        side_df,
        posteriors,
        prop_lines=PROP_LINES,
        n_sims=n_sims,
        context_kwargs=ctx,
        extra_record_fields=extra,
    )


def run_one_variant(
    label: str,
    sides: list,
    posteriors: dict,
    ump_lifts: dict[str, dict[str, float]],
    use_umpire_bb: bool,
    n_sims: int,
) -> pd.DataFrame:
    """Run all sides for one variant."""
    df, _skipped = run_sides_loop(
        label=label,
        sides=sides,
        simulate_fn=lambda sdf: simulate_one_side(
            sdf, posteriors, ump_lifts, use_umpire_bb, n_sims=n_sims,
        ),
    )
    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def brier_score(p_over: np.ndarray, actual_binary: np.ndarray) -> float:
    """Brier score: mean((p - y)^2). Lower is better."""
    return float(np.mean((p_over - actual_binary) ** 2))


def brier_skill_score(
    model_brier: float, climatology_brier: float,
) -> float:
    """BSS = 1 - model_brier / climatology_brier. Higher is better."""
    if climatology_brier == 0:
        return 0.0
    return 1.0 - model_brier / climatology_brier


def compute_comparison(
    df_base: pd.DataFrame,
    df_ump_bb: pd.DataFrame,
    ump_lifts: dict[str, dict[str, float]],
) -> None:
    """Print comprehensive comparison of baseline vs umpire-BB model."""

    # Filter to batters with real posteriors
    base = df_base[df_base["has_posterior"]].copy()
    ump = df_ump_bb[df_ump_bb["has_posterior"]].copy()

    n_batters = len(base)
    n_games = base["game_pk"].nunique()
    n_with_ump = base[base["has_umpire"]].groupby("game_pk").ngroups

    print()
    print("=" * 78)
    print("UMPIRE BB LIFT A/B BACKTEST RESULTS")
    print("=" * 78)
    print(f"  Batter-games:          {n_batters:,}")
    print(f"  Unique games:          {n_games:,}")
    print(f"  Games with umpire:     {n_with_ump:,}")
    print(f"  Sims per game:         {N_SIMS:,}")

    # Report umpire BB lift distribution
    bb_lifts_all = [v["bb_lift"] for v in ump_lifts.values()]
    print(f"  Umpire BB lift range:  [{min(bb_lifts_all):.4f}, {max(bb_lifts_all):.4f}]")
    print(f"  Umpire BB lift std:    {np.std(bb_lifts_all):.4f}")
    print()

    # =================================================================
    # 1. BRIER SCORE at 0.5 line (primary metric)
    # =================================================================
    print("-" * 78)
    print("BRIER SCORE at 0.5 prop line (lower = better)")
    print("-" * 78)
    print(f"  {'Stat':>4s}  {'Line':>5s}  {'Base Brier':>12s}  {'UmpBB Brier':>12s}  "
          f"{'Delta':>10s}  {'BSS_base':>10s}  {'BSS_umpbb':>10s}")
    print(f"  {'----':>4s}  {'-----':>5s}  {'-' * 12:>12s}  {'-' * 12:>12s}  "
          f"{'-' * 10:>10s}  {'-' * 10:>10s}  {'-' * 10:>10s}")

    for stat in ALL_STATS:
        line = PRIMARY_LINE[stat]
        col = f"p_over_{stat}_{line}"
        if col not in base.columns:
            continue

        actual_b = (base[f"actual_{stat}"].values > line).astype(float)
        actual_u = (ump[f"actual_{stat}"].values > line).astype(float)

        p_base = base[col].values
        p_ump = ump[col].values

        bs_base = brier_score(p_base, actual_b)
        bs_ump = brier_score(p_ump, actual_u)

        # Climatology = overall hit rate
        clim = float(np.mean(actual_b))
        clim_brier = clim * (1 - clim)

        bss_base = brier_skill_score(bs_base, clim_brier)
        bss_ump = brier_skill_score(bs_ump, clim_brier)

        delta = bs_ump - bs_base
        marker = " <--" if stat == "bb" else ""
        print(f"  {stat.upper():>4s}  {line:>5.1f}  {bs_base:>12.6f}  {bs_ump:>12.6f}  "
              f"{delta:>+10.6f}  {bss_base:>+10.4f}  {bss_ump:>+10.4f}{marker}")

    # =================================================================
    # 2. BRIER SCORE at ALL lines
    # =================================================================
    print()
    print("-" * 78)
    print("BRIER SCORE at all prop lines (lower = better)")
    print("-" * 78)
    print(f"  {'Stat':>4s}  {'Line':>5s}  {'Base Brier':>12s}  {'UmpBB Brier':>12s}  "
          f"{'Delta':>10s}  {'N':>8s}  {'Hit rate':>9s}")
    print(f"  {'----':>4s}  {'-----':>5s}  {'-' * 12:>12s}  {'-' * 12:>12s}  "
          f"{'-' * 10:>10s}  {'-' * 8:>8s}  {'-' * 9:>9s}")

    for stat in ALL_STATS:
        for line in PROP_LINES[stat]:
            col = f"p_over_{stat}_{line}"
            if col not in base.columns:
                continue

            actual_b = (base[f"actual_{stat}"].values > line).astype(float)
            actual_u = (ump[f"actual_{stat}"].values > line).astype(float)

            p_base = base[col].values
            p_ump = ump[col].values

            bs_base = brier_score(p_base, actual_b)
            bs_ump = brier_score(p_ump, actual_u)
            delta = bs_ump - bs_base
            hit_rate = float(np.mean(actual_b))

            marker = " <--" if stat == "bb" else ""
            print(f"  {stat.upper():>4s}  {line:>5.1f}  {bs_base:>12.6f}  {bs_ump:>12.6f}  "
                  f"{delta:>+10.6f}  {n_batters:>8,}  {hit_rate:>9.3f}{marker}")

    # =================================================================
    # 3. 65% confidence hit rate
    # =================================================================
    print()
    print("-" * 78)
    print("65% CONFIDENCE HIT RATE")
    print("-" * 78)

    for label_name, df in [("Baseline", base), ("Ump BB", ump)]:
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
            print(f"  {label_name:10s}  P(over)>=65%:  {hr65:.4f} ({hr65*100:.1f}%) "
                  f"on {n65:,} picks")

        # Under side
        all_p_u = 1 - all_p
        all_a_u = 1 - all_a
        m65u = all_p_u >= 0.65
        if m65u.sum() > 0:
            hr65u = float(all_a_u[m65u].mean())
            n65u = m65u.sum()
            print(f"  {label_name:10s}  P(under)>=65%: {hr65u:.4f} ({hr65u*100:.1f}%) "
                  f"on {n65u:,} picks")

    # =================================================================
    # 5. Per-stat 65% hit rate (BB focus)
    # =================================================================
    print()
    print("-" * 78)
    print("PER-STAT 65% HIT RATE")
    print("-" * 78)
    print(f"  {'Stat':>4s}  {'Line':>5s}  {'Base hit%':>10s}  {'UmpBB hit%':>10s}  "
          f"{'Delta':>8s}  {'N_base':>8s}  {'N_umpbb':>8s}")

    for stat in ALL_STATS:
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

            # Umpire BB
            p_u = ump[col].values
            a_u = (ump[f"actual_{stat}"].values > line).astype(float)
            m_u = p_u >= 0.65
            hr_u = float(a_u[m_u].mean()) if m_u.sum() > 0 else 0
            n_u = int(m_u.sum())

            delta = hr_u - hr_b
            marker = " <--" if stat == "bb" else ""
            print(f"  {stat.upper():>4s}  {line:>5.1f}  {hr_b:>10.4f}  {hr_u:>10.4f}  "
                  f"{delta:>+8.4f}  {n_b:>8,}  {n_u:>8,}{marker}")

    # =================================================================
    # 6. Umpire-only subset (games with known umpire)
    # =================================================================
    base_ump = base[base["has_umpire"]].copy()
    ump_ump = ump[ump["has_umpire"]].copy()

    if len(base_ump) > 0:
        print()
        print("-" * 78)
        print("UMPIRE-ONLY SUBSET: Brier score (games with known umpire)")
        print(f"  ({len(base_ump):,} batter-games in {base_ump['game_pk'].nunique():,} games)")
        print("-" * 78)
        print(f"  {'Stat':>4s}  {'Line':>5s}  {'Base Brier':>12s}  {'UmpBB Brier':>12s}  "
              f"{'Delta':>10s}")
        print(f"  {'----':>4s}  {'-----':>5s}  {'-' * 12:>12s}  {'-' * 12:>12s}  "
              f"{'-' * 10:>10s}")

        for stat in ALL_STATS:
            line = PRIMARY_LINE[stat]
            col = f"p_over_{stat}_{line}"
            if col not in base_ump.columns:
                continue

            actual_b = (base_ump[f"actual_{stat}"].values > line).astype(float)
            actual_u = (ump_ump[f"actual_{stat}"].values > line).astype(float)

            bs_base = brier_score(base_ump[col].values, actual_b)
            bs_ump = brier_score(ump_ump[col].values, actual_u)
            delta = bs_ump - bs_base
            marker = " <--" if stat == "bb" else ""
            print(f"  {stat.upper():>4s}  {line:>5.1f}  {bs_base:>12.6f}  {bs_ump:>12.6f}  "
                  f"{delta:>+10.6f}{marker}")

    # =================================================================
    # 7. Per-umpire extremes (tightest/loosest zone)
    # =================================================================
    if len(base_ump) > 0:
        print()
        print("-" * 78)
        print("PER-UMPIRE EXTREMES: BB Brier score (tightest vs loosest zone)")
        print("-" * 78)

        # Compute per-umpire BB lift and Brier scores
        ump_results = []
        for ump_name in ump_ump["hp_umpire_name"].unique():
            if not isinstance(ump_name, str) or not ump_name:
                continue
            mask_b = base_ump["hp_umpire_name"] == ump_name
            mask_u = ump_ump["hp_umpire_name"] == ump_name

            if mask_b.sum() < 9:
                continue

            bb_lift_val = ump_lifts.get(ump_name, {}).get("bb_lift", 0.0)

            actual_b = (base_ump.loc[mask_b, "actual_bb"].values > 0.5).astype(float)
            actual_u = (ump_ump.loc[mask_u, "actual_bb"].values > 0.5).astype(float)

            col = "p_over_bb_0.5"
            bs_base_val = brier_score(base_ump.loc[mask_b, col].values, actual_b)
            bs_ump_val = brier_score(ump_ump.loc[mask_u, col].values, actual_u)

            ump_results.append({
                "umpire": ump_name,
                "bb_lift": bb_lift_val,
                "n_batters": int(mask_b.sum()),
                "actual_bb_rate": float(actual_b.mean()),
                "brier_base": bs_base_val,
                "brier_umpbb": bs_ump_val,
                "brier_delta": bs_ump_val - bs_base_val,
            })

        if ump_results:
            ump_df = pd.DataFrame(ump_results).sort_values("bb_lift")
            n_show = min(5, len(ump_df))

            print()
            print("  TIGHTEST ZONE (lowest BB lift, fewest walks):")
            print(f"  {'Umpire':<22s}  {'BB lift':>8s}  {'N':>5s}  "
                  f"{'Act BB%':>8s}  {'Base BS':>9s}  {'UmpBB BS':>9s}  {'Delta':>9s}")
            for _, r in ump_df.head(n_show).iterrows():
                print(f"  {r['umpire']:<22s}  {r['bb_lift']:>+8.4f}  {r['n_batters']:>5d}  "
                      f"{r['actual_bb_rate']:>8.3f}  {r['brier_base']:>9.6f}  "
                      f"{r['brier_umpbb']:>9.6f}  {r['brier_delta']:>+9.6f}")

            print()
            print("  LOOSEST ZONE (highest BB lift, most walks):")
            for _, r in ump_df.tail(n_show).iterrows():
                print(f"  {r['umpire']:<22s}  {r['bb_lift']:>+8.4f}  {r['n_batters']:>5d}  "
                      f"{r['actual_bb_rate']:>8.3f}  {r['brier_base']:>9.6f}  "
                      f"{r['brier_umpbb']:>9.6f}  {r['brier_delta']:>+9.6f}")

            # Improvement count
            n_improved = (ump_df["brier_delta"] < 0).sum()
            n_total = len(ump_df)
            print()
            print(f"  Umpires improved: {n_improved}/{n_total} "
                  f"({100*n_improved/n_total:.0f}%)")
            print(f"  Mean Brier delta: {ump_df['brier_delta'].mean():+.6f}")

    # =================================================================
    # 8. K vs BB orthogonality check
    # =================================================================
    if len(base_ump) > 0:
        print()
        print("-" * 78)
        print("K vs BB LIFT CORRELATION (orthogonality check)")
        print("-" * 78)
        k_lifts_arr = []
        bb_lifts_arr = []
        for ump_name in ump_ump["hp_umpire_name"].unique():
            if not isinstance(ump_name, str) or ump_name not in ump_lifts:
                continue
            k_lifts_arr.append(ump_lifts[ump_name]["k_lift"])
            bb_lifts_arr.append(ump_lifts[ump_name]["bb_lift"])
        if len(k_lifts_arr) > 5:
            corr = float(np.corrcoef(k_lifts_arr, bb_lifts_arr)[0, 1])
            shared_var = corr ** 2
            unique_var = 1 - shared_var
            print(f"  Pearson r(K lift, BB lift): {corr:.4f}")
            print(f"  Shared variance (r^2):     {shared_var:.4f}")
            print(f"  Unique BB variance:        {unique_var:.4f}")

    print()
    print("=" * 78)
    print("BACKTEST COMPLETE")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B backtest: umpire BB lift",
    )
    parser.add_argument(
        "--max-games", type=int, default=400,
        help="Max team-game sides to simulate (default: 400)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=N_SIMS,
        help=f"Monte Carlo sims per game (default: {N_SIMS})",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Load data
    posteriors = load_posteriors()

    # Train umpire lifts on 2022-2023, test on 2024-2025
    train_seasons = [2022, 2023]
    test_seasons = [2024, 2025]
    ump_lifts = load_umpire_lifts(train_seasons)

    # Fetch games from test seasons
    games_df = fetch_backtest_games(test_seasons, max_games=args.max_games)
    if games_df.empty:
        logger.error("No backtest games found")
        return

    # Group by (game_pk, team_id)
    sides = list(games_df.groupby(["game_pk", "team_id"]))
    logger.info("Running %d team-game sides...", len(sides))

    # Run A variant: Baseline (K lift only, no BB lift)
    df_base = run_one_variant(
        "Baseline", sides, posteriors, ump_lifts,
        use_umpire_bb=False, n_sims=args.n_sims,
    )

    # Run B variant: Umpire BB (K lift + BB lift)
    df_ump_bb = run_one_variant(
        "Ump BB", sides, posteriors, ump_lifts,
        use_umpire_bb=True, n_sims=args.n_sims,
    )

    # Compare
    compute_comparison(df_base, df_ump_bb, ump_lifts)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
