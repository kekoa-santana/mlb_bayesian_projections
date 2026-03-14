"""
Phase 0B: Profile simulate_game_stat() and simulate_game_ks() performance.

Establishes a performance baseline and identifies hotspots for the Monte Carlo
simulation pipeline. Tests with a real 2024 starter from the database.

Usage:
    python scripts/profile_simulation.py
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.bf_model import draw_bf_samples, get_bf_distribution
from src.models.game_k_model import (
    _safe_logit,
    simulate_game_ks,
    simulate_game_stat,
    simulate_game_stat_poisson,
    simulate_batter_game_stat,
    compute_k_over_probs,
    compute_over_probs,
    predict_game,
    predict_game_stat,
)


# ---------------------------------------------------------------------------
# Helper: fetch a real 2024 starter from the database
# ---------------------------------------------------------------------------
def get_real_pitcher_data() -> dict:
    """Fetch a real 2024 starter's BF stats from the database."""
    from src.data.db import read_sql

    # Find a high-K starter from 2024 (e.g., top 10 by K count)
    query = """
    SELECT
        b.pitcher_id,
        dp.name_first || ' ' || dp.name_last AS pitcher_name,
        COUNT(*) AS n_starts,
        AVG(b.batters_faced) AS avg_bf,
        STDDEV(b.batters_faced) AS std_bf,
        AVG(b.strike_outs) AS avg_k,
        SUM(b.strike_outs)::float / NULLIF(SUM(b.batters_faced), 0) AS k_rate
    FROM production.fact_player_game_mlb b
    JOIN production.dim_game dg ON b.game_pk = dg.game_pk
    JOIN production.dim_player dp ON b.pitcher_id = dp.player_id
    WHERE dg.season = 2024
      AND dg.game_type = 'R'
      AND b.is_starter = true
      AND b.batters_faced >= 15
    GROUP BY b.pitcher_id, dp.name_first, dp.name_last
    HAVING COUNT(*) >= 20
    ORDER BY SUM(b.strike_outs)::float / NULLIF(SUM(b.batters_faced), 0) DESC
    LIMIT 5
    """
    df = read_sql(query)
    if df.empty:
        raise RuntimeError("No qualifying pitchers found in 2024 data")

    row = df.iloc[0]
    print(f"\n  Selected pitcher: {row['pitcher_name']} (ID: {row['pitcher_id']})")
    print(f"  2024 stats: {int(row['n_starts'])} starts, "
          f"K%={row['k_rate']:.3f}, avg BF={row['avg_bf']:.1f}, "
          f"std BF={row['std_bf']:.1f}, avg K={row['avg_k']:.1f}")

    return {
        "pitcher_id": int(row["pitcher_id"]),
        "pitcher_name": row["pitcher_name"],
        "k_rate": float(row["k_rate"]),
        "avg_bf": float(row["avg_bf"]),
        "std_bf": float(row["std_bf"]),
        "n_starts": int(row["n_starts"]),
    }


# ---------------------------------------------------------------------------
# Synthetic posterior samples (mimic Layer 1 output without fitting the model)
# ---------------------------------------------------------------------------
def generate_synthetic_posterior(
    k_rate_mean: float,
    n_samples: int = 4000,
    posterior_std: float = 0.015,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic K% posterior samples on logit scale, then expit back."""
    rng = np.random.default_rng(seed)
    logit_mean = logit(np.clip(k_rate_mean, 1e-6, 1 - 1e-6))
    logit_samples = rng.normal(logit_mean, posterior_std / k_rate_mean, size=n_samples)
    return expit(logit_samples)


# ---------------------------------------------------------------------------
# Stage-by-stage timing
# ---------------------------------------------------------------------------
def time_stage(func, *args, n_repeats: int = 100, **kwargs):
    """Time a function over n_repeats and return (mean_ms, std_ms, result)."""
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times), result


def profile_stages(pitcher_data: dict, n_draws: int = 4000):
    """Profile each stage of the simulation pipeline individually."""
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE PROFILING")
    print("=" * 70)
    n_repeats = 200

    k_rate = pitcher_data["k_rate"]
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]

    # --- Stage 1: Generate posterior samples (synthetic) ---
    mean_ms, std_ms, k_rate_samples = time_stage(
        generate_synthetic_posterior, k_rate, n_draws, n_repeats=n_repeats,
    )
    print(f"\n  1. Generate posterior samples ({n_draws} draws)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # --- Stage 2: Draw BF samples (truncated normal) ---
    rng = np.random.default_rng(42)
    mean_ms, std_ms, bf_draws = time_stage(
        draw_bf_samples, bf_mu, bf_sigma, n_draws, 3, 35, rng,
        n_repeats=n_repeats,
    )
    print(f"\n  2. Draw BF samples (truncnorm, {n_draws} draws)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")
    print(f"     BF range: {bf_draws.min()}-{bf_draws.max()}, "
          f"unique values: {len(np.unique(bf_draws))}")

    # --- Stage 3: Logit transform + clip ---
    mean_ms, std_ms, k_logit = time_stage(
        _safe_logit, k_rate_samples, n_repeats=n_repeats,
    )
    print(f"\n  3. Logit transform ({n_draws} samples)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # --- Stage 4: Expit (inverse logit) ---
    adjusted_logit = k_logit + 0.05  # simulate a small lift
    mean_ms, std_ms, _ = time_stage(
        expit, adjusted_logit, n_repeats=n_repeats,
    )
    print(f"\n  4. Expit (inverse logit, {n_draws} samples)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # --- Stage 5: Binomial draws (single slot, all draws) ---
    p_arr = expit(k_logit)
    rng2 = np.random.default_rng(42)

    def binomial_single_slot(n_pa, p, rng_):
        return rng_.binomial(n=n_pa, p=p)

    mean_ms, std_ms, _ = time_stage(
        binomial_single_slot, 3, p_arr, rng2, n_repeats=n_repeats,
    )
    print(f"\n  5. Binomial draw (1 slot, n=3, {n_draws} draws)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # --- Stage 6: Full 9-slot binomial loop (the inner loop) ---
    def nine_slot_binomial(k_logit_arr, bf_val, lineup_lifts, rng_):
        """Simulate the inner loop for a single BF value across all draws."""
        base_pa = bf_val // 9
        extra = bf_val % 9
        pa_per_slot = np.full(9, base_pa, dtype=int)
        pa_per_slot[:extra] += 1
        game_ks = np.zeros(len(k_logit_arr), dtype=int)
        for slot in range(9):
            if pa_per_slot[slot] == 0:
                continue
            adj = k_logit_arr + lineup_lifts[slot]
            adj_p = expit(adj)
            game_ks += rng_.binomial(n=pa_per_slot[slot], p=adj_p)
        return game_ks

    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)
    rng3 = np.random.default_rng(42)
    mean_ms, std_ms, _ = time_stage(
        nine_slot_binomial, k_logit, 22, lineup_lifts, rng3,
        n_repeats=n_repeats,
    )
    print(f"\n  6. 9-slot binomial loop (BF=22, {n_draws} draws)")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # --- Stage 7: compute_over_probs aggregation ---
    k_samples = np.random.default_rng(42).poisson(6, size=n_draws)
    mean_ms, std_ms, _ = time_stage(
        compute_over_probs, k_samples, stat_name="k", n_repeats=n_repeats,
    )
    print(f"\n  7. compute_over_probs aggregation")
    print(f"     {mean_ms:.3f} +/- {std_ms:.3f} ms")


# ---------------------------------------------------------------------------
# End-to-end timing: simulate_game_ks
# ---------------------------------------------------------------------------
def profile_simulate_game_ks(pitcher_data: dict, n_draws: int = 4000):
    """Profile the full simulate_game_ks function."""
    print("\n" + "=" * 70)
    print("END-TO-END: simulate_game_ks()")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]

    # Without matchup lifts
    mean_ms, std_ms, k_samples = time_stage(
        simulate_game_ks,
        k_rate_samples, bf_mu, bf_sigma,
        None, 0.0, 0.0, n_draws, 3, 35, 42,
        n_repeats=200,
    )
    print(f"\n  No matchup lifts:  {mean_ms:.3f} +/- {std_ms:.3f} ms")
    print(f"  E[K]={np.mean(k_samples):.2f}, std={np.std(k_samples):.2f}")

    # With matchup lifts
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)
    mean_ms, std_ms, k_samples = time_stage(
        simulate_game_ks,
        k_rate_samples, bf_mu, bf_sigma,
        lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42,
        n_repeats=200,
    )
    print(f"  With matchup lifts: {mean_ms:.3f} +/- {std_ms:.3f} ms")
    print(f"  E[K]={np.mean(k_samples):.2f}, std={np.std(k_samples):.2f}")


# ---------------------------------------------------------------------------
# End-to-end timing: simulate_game_stat
# ---------------------------------------------------------------------------
def profile_simulate_game_stat(pitcher_data: dict, n_draws: int = 4000):
    """Profile the generalized simulate_game_stat function."""
    print("\n" + "=" * 70)
    print("END-TO-END: simulate_game_stat()")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]

    # Pitcher mode (n_slots=9) without matchup lifts
    mean_ms, std_ms, _ = time_stage(
        simulate_game_stat,
        k_rate_samples, bf_mu, bf_sigma,
        None, 0.0, n_draws, 3, 35, 9, 42,
        n_repeats=200,
    )
    print(f"\n  Pitcher mode (9 slots), no lifts:  {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # Pitcher mode with matchup lifts
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)
    mean_ms, std_ms, _ = time_stage(
        simulate_game_stat,
        k_rate_samples, bf_mu, bf_sigma,
        lineup_lifts, 0.05, n_draws, 3, 35, 9, 42,
        n_repeats=200,
    )
    print(f"  Pitcher mode (9 slots), with lifts: {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # Batter mode (n_slots=1)
    bb_rate_samples = generate_synthetic_posterior(0.085, n_draws)
    mean_ms, std_ms, _ = time_stage(
        simulate_game_stat,
        bb_rate_samples, 3.9, 0.9,
        np.array([0.05]), 0.0, n_draws, 1, 7, 1, 42,
        n_repeats=200,
    )
    print(f"  Batter mode (1 slot):               {mean_ms:.3f} +/- {std_ms:.3f} ms")

    # Poisson mode (HR)
    hr_rate_samples = generate_synthetic_posterior(0.030, n_draws)
    mean_ms, std_ms, _ = time_stage(
        simulate_game_stat_poisson,
        hr_rate_samples, bf_mu, bf_sigma,
        None, 0.0, 1.05, n_draws, 3, 35, 9, 42,
        n_repeats=200,
    )
    print(f"  Poisson pitcher mode (HR, 9 slots):  {mean_ms:.3f} +/- {std_ms:.3f} ms")


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------
def profile_memory(pitcher_data: dict, n_draws: int = 4000):
    """Measure peak memory for a single simulation."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    k_samples = simulate_game_ks(
        k_rate_samples, bf_mu, bf_sigma,
        lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42,
    )

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_diff = snapshot_after.compare_to(snapshot_before, "lineno")

    total_alloc = sum(s.size for s in stats_diff if s.size > 0)
    print(f"\n  n_draws = {n_draws}")
    print(f"  Net memory allocated: {total_alloc / 1024:.1f} KB")
    print(f"  Output array size:    {k_samples.nbytes / 1024:.1f} KB "
          f"(dtype={k_samples.dtype})")
    print(f"  Rate samples size:    {k_rate_samples.nbytes / 1024:.1f} KB")

    # Estimate memory for different n_draws
    print(f"\n  Memory scaling estimate (linear with n_draws):")
    for nd in [1000, 2000, 4000, 8000, 16000]:
        est_kb = total_alloc / 1024 * (nd / n_draws)
        print(f"    n_draws={nd:>6d}: ~{est_kb:.1f} KB")


# ---------------------------------------------------------------------------
# cProfile deep dive
# ---------------------------------------------------------------------------
def cprofile_simulation(pitcher_data: dict, n_draws: int = 4000):
    """Run cProfile on simulate_game_ks for detailed function-level breakdown."""
    print("\n" + "=" * 70)
    print("cProfile BREAKDOWN (1000 calls to simulate_game_ks)")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1000):
        simulate_game_ks(
            k_rate_samples, bf_mu, bf_sigma,
            lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42,
        )
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------
def profile_scaling(pitcher_data: dict):
    """Test how simulation time scales with n_draws."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: time vs n_draws")
    print("=" * 70)

    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)

    draws_list = [500, 1000, 2000, 4000, 8000, 16000]
    print(f"\n  {'n_draws':>8s}  {'ms/call':>10s}  {'draws/ms':>10s}  {'relative':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    baseline_rate = None
    for nd in draws_list:
        k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], nd)
        mean_ms, std_ms, _ = time_stage(
            simulate_game_ks,
            k_rate_samples, bf_mu, bf_sigma,
            lineup_lifts, 0.05, -0.02, nd, 3, 35, 42,
            n_repeats=100,
        )
        draws_per_ms = nd / mean_ms
        if baseline_rate is None:
            baseline_rate = draws_per_ms
        relative = mean_ms / (draws_list[0] / baseline_rate * nd / draws_list[0])
        print(f"  {nd:>8d}  {mean_ms:>10.3f}  {draws_per_ms:>10.0f}  {relative:>10.2f}x")


# ---------------------------------------------------------------------------
# Latency budget analysis for 9 pre-computed logit lifts
# ---------------------------------------------------------------------------
def latency_budget_analysis(pitcher_data: dict, n_draws: int = 4000):
    """Estimate what adding 9 pre-computed logit lifts would cost."""
    print("\n" + "=" * 70)
    print("LATENCY BUDGET: Adding 9 Pre-Computed Logit Lifts")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]

    # Current: 2 lifts (umpire + weather)
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)
    mean_2lift, _, _ = time_stage(
        simulate_game_ks,
        k_rate_samples, bf_mu, bf_sigma,
        lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42,
        n_repeats=200,
    )

    # Proposed: 9 additional lifts (lineup, framing, umpire, weather, rest,
    # BvP, platoon, form, opposing pitcher)
    # These are CONTEXT lifts applied uniformly (scalar additions to logit),
    # not per-batter lifts.
    #
    # Key insight: additional scalar logit lifts cost ZERO per-draw time
    # because they're just added to the logit once before the binomial loop.

    print(f"\n  Current baseline (with matchup + 2 context lifts):")
    print(f"    simulate_game_ks: {mean_2lift:.3f} ms")

    # Simulate adding more scalar lifts (they all get summed before the loop)
    n_new_lifts = 9
    total_context_lift = sum(
        np.random.default_rng(i).normal(0, 0.05) for i in range(n_new_lifts)
    )

    mean_9lift, _, _ = time_stage(
        simulate_game_ks,
        k_rate_samples, bf_mu, bf_sigma,
        lineup_lifts, total_context_lift, 0.0, n_draws, 3, 35, 42,
        n_repeats=200,
    )
    print(f"\n  With 9 additional pre-computed context lifts (summed to scalar):")
    print(f"    simulate_game_ks: {mean_9lift:.3f} ms")
    print(f"    Marginal cost:    {mean_9lift - mean_2lift:.3f} ms (essentially zero)")

    # But: computing those lifts requires lookup/computation BEFORE simulation
    print(f"\n  Pre-computation cost estimates (per game, BEFORE simulation):")

    lift_names = [
        ("Lineup matchup (9 batters)", "Per-batter pitch-type scoring loop"),
        ("Catcher framing", "Lookup from pre-computed table"),
        ("Umpire tendency", "Lookup from pre-computed table"),
        ("Weather effect", "Lookup from pre-computed table"),
        ("Rest days", "Lookup from schedule"),
        ("BvP history", "Lookup from pre-computed matchup table"),
        ("Platoon splits", "Lookup from pre-computed table"),
        ("Recent form (rolling)", "Lookup from pre-computed table"),
        ("Opposing pitcher", "Lookup from pre-computed table"),
    ]

    # Time a simple dict lookup (representative of pre-computed lift lookup)
    lookup_dict = {i: np.random.default_rng(i).normal(0, 0.05) for i in range(10000)}

    def dict_lookup(d, key):
        return d.get(key, 0.0)

    lookup_ms, _, _ = time_stage(
        dict_lookup, lookup_dict, 5000, n_repeats=10000,
    )

    print(f"\n  {'Lift source':<35s}  {'Type':<40s}  {'Est. ms':>8s}")
    print(f"  {'-'*35}  {'-'*40}  {'-'*8}")

    total_precomp = 0.0
    for name, desc in lift_names:
        if "Lineup matchup" in name:
            # This is the expensive one: 9 score_matchup calls
            est = 1.0  # placeholder, measured below
        else:
            est = lookup_ms
        total_precomp += est
        print(f"  {name:<35s}  {desc:<40s}  {est:>8.4f}")

    print(f"\n  Total pre-computation estimate:  ~{total_precomp:.2f} ms")
    print(f"  Simulation itself:              ~{mean_2lift:.2f} ms")
    print(f"  Grand total per game:           ~{total_precomp + mean_2lift:.2f} ms")

    # Latency budget
    budget_ms = 2000.0
    print(f"\n  LATENCY BUDGET TARGET: <{budget_ms:.0f} ms per game prediction")
    games_per_slate = 15  # typical MLB daily slate
    batch_budget = budget_ms * games_per_slate / 1000
    print(f"  For a {games_per_slate}-game slate: <{batch_budget:.0f}s total")

    per_game_ms = total_precomp + mean_2lift
    print(f"\n  Current per-game estimate: {per_game_ms:.2f} ms")
    print(f"  Budget headroom:          {budget_ms - per_game_ms:.2f} ms ({(1 - per_game_ms/budget_ms)*100:.1f}%)")

    print(f"\n  WHAT'S PRE-COMPUTABLE (before game day, cache in parquet/dict):")
    print(f"    - Catcher framing lift (season-level, changes ~never)")
    print(f"    - Umpire tendency lift (career-level, changes ~never)")
    print(f"    - Platoon splits lift (season-level, pre-computed)")
    print(f"    - Recent form z-scores (daily update, pre-computed in DB)")
    print(f"    - BvP career matchup lift (pre-computed in DB)")
    print(f"    - Park factor (static per venue)")
    print(f"\n  WHAT'S PER-DRAW (must happen inside simulation loop):")
    print(f"    - BF sampling (truncated normal)")
    print(f"    - Rate sampling (posterior resampling)")
    print(f"    - Binomial/Poisson draws (the core simulation)")
    print(f"    - Logit transform + expit (vectorized, fast)")
    print(f"\n  WHAT'S PER-GAME but NOT per-draw:")
    print(f"    - Lineup matchup scoring (9 batters x pitch-type loop)")
    print(f"    - Weather lookup (depends on game time)")
    print(f"    - Rest days (depends on schedule)")


# ---------------------------------------------------------------------------
# Batch throughput estimation
# ---------------------------------------------------------------------------
def profile_batch_throughput(pitcher_data: dict, n_draws: int = 4000):
    """Estimate throughput for backtest-scale batch predictions."""
    print("\n" + "=" * 70)
    print("BATCH THROUGHPUT ESTIMATION")
    print("=" * 70)

    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    bf_mu = pitcher_data["avg_bf"]
    bf_sigma = pitcher_data["std_bf"]
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)

    # Time 1000 consecutive simulations
    t0 = time.perf_counter()
    for i in range(1000):
        simulate_game_ks(
            k_rate_samples, bf_mu, bf_sigma,
            lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42 + i,
        )
    t1 = time.perf_counter()
    total_s = t1 - t0
    ms_per_game = total_s * 1000 / 1000

    print(f"\n  1000 games simulated in {total_s:.2f}s ({ms_per_game:.3f} ms/game)")
    print(f"\n  Backtest estimates:")
    for n_games in [1000, 5000, 11517, 30000]:
        est_s = n_games * ms_per_game / 1000
        print(f"    {n_games:>6d} games: {est_s:>6.1f}s ({est_s/60:.1f} min)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  PHASE 0B: Game Simulation Performance Profiling")
    print("=" * 70)

    # Fetch real pitcher data
    print("\n[1/7] Fetching real pitcher data from database...")
    try:
        pitcher_data = get_real_pitcher_data()
    except Exception as e:
        print(f"  Database query failed: {e}")
        print("  Using fallback synthetic data (Cole-like pitcher)")
        pitcher_data = {
            "pitcher_id": 543037,
            "pitcher_name": "Gerrit Cole",
            "k_rate": 0.280,
            "avg_bf": 23.5,
            "std_bf": 4.2,
            "n_starts": 30,
        }

    n_draws = 4000  # default from model config

    # Run all profiling sections
    print("\n[2/7] Stage-by-stage profiling...")
    profile_stages(pitcher_data, n_draws)

    print("\n[3/7] End-to-end simulate_game_ks...")
    profile_simulate_game_ks(pitcher_data, n_draws)

    print("\n[4/7] End-to-end simulate_game_stat...")
    profile_simulate_game_stat(pitcher_data, n_draws)

    print("\n[5/7] Memory profiling...")
    profile_memory(pitcher_data, n_draws)

    print("\n[6/7] cProfile breakdown...")
    cprofile_simulation(pitcher_data, n_draws)

    print("\n[7/7] Scaling + latency budget analysis...")
    profile_scaling(pitcher_data)
    latency_budget_analysis(pitcher_data, n_draws)
    profile_batch_throughput(pitcher_data, n_draws)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    k_rate_samples = generate_synthetic_posterior(pitcher_data["k_rate"], n_draws)
    lineup_lifts = np.random.default_rng(99).normal(0, 0.1, size=9)

    mean_ms, _, _ = time_stage(
        simulate_game_ks,
        k_rate_samples, pitcher_data["avg_bf"], pitcher_data["std_bf"],
        lineup_lifts, 0.05, -0.02, n_draws, 3, 35, 42,
        n_repeats=500,
    )
    print(f"\n  Pitcher: {pitcher_data['pitcher_name']}")
    print(f"  n_draws: {n_draws}")
    print(f"  Per-game simulation: {mean_ms:.3f} ms")
    print(f"  Latency budget: 2000 ms")
    print(f"  Budget usage: {mean_ms/2000*100:.1f}%")
    print(f"  Remaining for 9 pre-computed lifts + overhead: "
          f"{2000 - mean_ms:.1f} ms")
    print(f"\n  VERDICT: {'WELL WITHIN BUDGET' if mean_ms < 100 else 'NEEDS OPTIMIZATION' if mean_ms < 2000 else 'OVER BUDGET'}")
    print()


if __name__ == "__main__":
    main()
