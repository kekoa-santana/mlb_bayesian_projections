"""
Validation script for the lineup simulator.

Runs the simulator with realistic test data (league-average batters
vs a league-average starter) and prints summary stats to verify that
R, RBI, and other counting stats land in realistic MLB ranges.

MLB averages (2022-2025, per batter per game):
  R    ~0.55
  RBI  ~0.52
  H    ~1.06
  HR   ~0.14
  BB   ~0.36
  K    ~0.96
  PA   ~3.9  (varies by lineup spot)
  Team runs per game (one side): ~4.5
"""
from __future__ import annotations

import time

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.game_sim.lineup_simulator import simulate_lineup_game


def main() -> None:
    n_sims = 20_000
    seed = 12345

    rng = np.random.default_rng(99)
    k_rates = [0.215, 0.200, 0.190, 0.230, 0.240, 0.250, 0.260, 0.220, 0.235]
    bb_rates = [0.090, 0.100, 0.095, 0.080, 0.075, 0.070, 0.065, 0.080, 0.070]
    hr_rates = [0.035, 0.040, 0.045, 0.038, 0.030, 0.025, 0.020, 0.028, 0.022]

    n_posterior = 2000
    batter_k_samples = []
    batter_bb_samples = []
    batter_hr_samples = []
    for i in range(9):
        batter_k_samples.append(
            np.clip(rng.normal(k_rates[i], 0.02, n_posterior), 0.05, 0.50)
        )
        batter_bb_samples.append(
            np.clip(rng.normal(bb_rates[i], 0.015, n_posterior), 0.02, 0.25)
        )
        batter_hr_samples.append(
            np.clip(rng.normal(hr_rates[i], 0.01, n_posterior), 0.005, 0.10)
        )

    starter_k = 0.226
    starter_bb = 0.082
    starter_hr = 0.031
    starter_bf_mu = 22.0
    starter_bf_sigma = 3.4

    print(f"Running lineup simulation with {n_sims:,} sims...")
    t0 = time.perf_counter()

    result = simulate_lineup_game(
        batter_k_rate_samples=batter_k_samples,
        batter_bb_rate_samples=batter_bb_samples,
        batter_hr_rate_samples=batter_hr_samples,
        starter_k_rate=starter_k,
        starter_bb_rate=starter_bb,
        starter_hr_rate=starter_hr,
        starter_bf_mu=starter_bf_mu,
        starter_bf_sigma=starter_bf_sigma,
        n_sims=n_sims,
        random_seed=seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.2f}s ({n_sims / elapsed:,.0f} sims/sec)")
    print()

    print("=" * 72)
    print("PER-BATTER AVERAGES")
    print("=" * 72)
    header = (
        f"{'Slot':>4} {'PA':>5} {'H':>5} {'1B':>5} {'2B':>5} "
        f"{'3B':>5} {'HR':>5} {'BB':>5} {'K':>5} {'R':>5} "
        f"{'RBI':>5} {'TB':>5}"
    )
    print(header)
    print("-" * len(header))

    all_r = []
    all_rbi = []
    all_pa = []
    for i in range(9):
        s = result.batter_summary(i)
        all_r.append(s["r"]["mean"])
        all_rbi.append(s["rbi"]["mean"])
        all_pa.append(s["pa"]["mean"])
        print(
            f"  {i + 1:>2} "
            f"{s['pa']['mean']:>5.2f} "
            f"{s['h']['mean']:>5.2f} "
            f"{s['single']['mean']:>5.2f} "
            f"{s['double']['mean']:>5.2f} "
            f"{s['triple']['mean']:>5.3f} "
            f"{s['hr']['mean']:>5.2f} "
            f"{s['bb']['mean']:>5.2f} "
            f"{s['k']['mean']:>5.2f} "
            f"{s['r']['mean']:>5.2f} "
            f"{s['rbi']['mean']:>5.2f} "
            f"{s['tb']['mean']:>5.2f}"
        )

    avg_r = np.mean(all_r)
    avg_rbi = np.mean(all_rbi)
    avg_pa = np.mean(all_pa)
    print("-" * len(header))

    print()
    print("=" * 72)
    print("TEAM-LEVEL SUMMARY")
    print("=" * 72)
    ts = result.team_summary()
    print(
        f"  Team R/game:   {ts['team_runs']['mean']:.2f}  "
        f"(std {ts['team_runs']['std']:.2f})"
    )
    print(f"  Team H/game:   {ts['h']['mean']:.2f}")
    print(f"  Team HR/game:  {ts['hr']['mean']:.2f}")
    print(f"  Team BB/game:  {ts['bb']['mean']:.2f}")
    print(f"  Team K/game:   {ts['k']['mean']:.2f}")
    print(f"  Team PA/game:  {ts['pa']['mean']:.1f}")

    print()
    print("=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)

    sum_batter_r = result.r.sum(axis=0)
    r_mismatch = np.abs(
        sum_batter_r.astype(np.int64) - result.team_runs.astype(np.int64)
    )
    print(
        f"  R consistency (sum batter R == team runs): "
        f"max mismatch = {r_mismatch.max()}, "
        f"mean = {r_mismatch.mean():.4f}"
    )

    for i in range(9):
        r_lt_hr = (result.r[i] < result.hr[i]).sum()
        if r_lt_hr > 0:
            print(f"  WARNING: Slot {i+1} has {r_lt_hr} sims where R < HR")

    team_r_mean = ts["team_runs"]["mean"]
    r_check = "PASS" if 3.0 < team_r_mean < 6.5 else "FAIL"
    print(f"  Team R/game in range (3.0-6.5): {team_r_mean:.2f} [{r_check}]")

    avg_r_check = "PASS" if 0.30 < avg_r < 0.80 else "FAIL"
    print(f"  Avg batter R/game in range (0.30-0.80): {avg_r:.3f} [{avg_r_check}]")

    avg_rbi_check = "PASS" if 0.30 < avg_rbi < 0.80 else "FAIL"
    print(
        f"  Avg batter RBI/game in range (0.30-0.80): "
        f"{avg_rbi:.3f} [{avg_rbi_check}]"
    )

    rbi_sum = result.rbi.sum(axis=0).mean()
    r_sum = result.r.sum(axis=0).mean()
    rbi_r_ratio = rbi_sum / r_sum if r_sum > 0 else 0
    ratio_check = "PASS" if 0.85 < rbi_r_ratio < 1.15 else "FAIL"
    print(f"  Team RBI/R ratio: {rbi_r_ratio:.3f} [{ratio_check}]")

    print()
    print(f"  Expected MLB ranges (per batter per game):")
    print(f"    R:   ~0.55  (got {avg_r:.3f})")
    print(f"    RBI: ~0.52  (got {avg_rbi:.3f})")
    print(f"    Team R: ~4.5  (got {team_r_mean:.2f})")

    print()
    print("=" * 72)
    print("PROP LINE EXAMPLE (Leadoff hitter)")
    print("=" * 72)
    for stat in ["h", "r", "rbi", "tb", "hr", "k"]:
        s = result.batter_summary(0)
        if stat in s:
            expected = s[stat]["mean"]
            line_strs = []
            probs = result.batter_over_probs(0, stat)
            for _, row in probs.iterrows():
                if row["line"] <= 3.5:
                    line_strs.append(
                        f"P(>{row['line']:.1f})={row['p_over']:.3f}"
                    )
            print(
                f"  {stat.upper():>3}: E[X] = {expected:.2f}  "
                f"{'  '.join(line_strs)}"
            )

    s = result.batter_summary(0)
    print(f"  HRR: E[X] = {s['hrr']['mean']:.2f}")


if __name__ == "__main__":
    main()
