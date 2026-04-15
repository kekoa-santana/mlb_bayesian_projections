#!/usr/bin/env python
"""RE24 sanity check for the game simulator runner model.

For each of the 24 base-out states, simulates forward using the current
_advance_runners() function and league-average PA outcome probabilities,
then compares empirical run expectancy to Tango's published RE24 table.

A well-calibrated runner model should match Tango within ~0.05 runs on
common states. Large deltas indicate mechanical bugs in _advance_runners
or in the PA outcome distribution mix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.game_sim.simulator import _advance_runners
from src.models.game_sim.pa_outcome_model import (
    PA_STRIKEOUT,
    PA_WALK,
    PA_HBP,
    PA_SINGLE,
    PA_DOUBLE,
    PA_TRIPLE,
    PA_HOME_RUN,
    PA_OUT,
)

# Modern MLB league-average PA outcome mix (2023-2025 average, per FanGraphs).
# Sum to 1.0.
LEAGUE_PROBS = np.array([
    0.225,  # K
    0.085,  # BB
    0.011,  # HBP
    0.140,  # 1B
    0.044,  # 2B
    0.004,  # 3B
    0.032,  # HR
    0.459,  # out (non-K)
])
assert abs(LEAGUE_PROBS.sum() - 1.0) < 1e-6, f"probs sum to {LEAGUE_PROBS.sum()}"

OUTCOME_CODES = np.array([
    PA_STRIKEOUT, PA_WALK, PA_HBP,
    PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN,
    PA_OUT,
])

# Tango RE24 table, 2010-2015 MLB average (from tangotiger.net).
# Rows indexed by (r1, r2, r3) flag tuple; cols by pre-PA outs (0, 1, 2).
TANGO_RE24: dict[tuple[int, int, int], tuple[float, float, float]] = {
    (0, 0, 0): (0.481, 0.254, 0.098),
    (1, 0, 0): (0.859, 0.509, 0.224),
    (0, 1, 0): (1.100, 0.664, 0.319),
    (0, 0, 1): (1.350, 0.950, 0.382),
    (1, 1, 0): (1.437, 0.884, 0.429),
    (1, 0, 1): (1.784, 1.130, 0.478),
    (0, 1, 1): (1.964, 1.376, 0.580),
    (1, 1, 1): (2.292, 1.541, 0.752),
}

STATE_LABELS = {
    (0, 0, 0): "empty",
    (1, 0, 0): "1B",
    (0, 1, 0): "2B",
    (0, 0, 1): "3B",
    (1, 1, 0): "1B,2B",
    (1, 0, 1): "1B,3B",
    (0, 1, 1): "2B,3B",
    (1, 1, 1): "loaded",
}


def simulate_state(
    start_r1: int,
    start_r2: int,
    start_r3: int,
    start_outs: int,
    n_sims: int,
    rng: np.random.Generator,
) -> float:
    """Simulate forward from a base-out state until the inning ends.

    Returns mean runs scored to end of inning across n_sims trials.
    """
    r1 = np.full(n_sims, start_r1, dtype=np.int32)
    r2 = np.full(n_sims, start_r2, dtype=np.int32)
    r3 = np.full(n_sims, start_r3, dtype=np.int32)
    inning_outs = np.full(n_sims, start_outs, dtype=np.int32)
    total_runs = np.zeros(n_sims, dtype=np.int32)
    active = np.ones(n_sims, dtype=bool)

    # Loop until all sims have reached 3 outs. Cap to avoid runaway.
    for _ in range(100):
        if not active.any():
            break

        n_active = int(active.sum())
        draws = rng.choice(OUTCOME_CODES, size=n_active, p=LEAGUE_PROBS)

        r1_a = r1[active]
        r2_a = r2[active]
        r3_a = r3[active]
        io_a = inning_outs[active]

        runs_delta, new_r1, new_r2, new_r3 = _advance_runners(
            draws, r1_a, r2_a, r3_a, io_a, rng,
        )

        # Track outs. K and non-K-out both add an out. Note sac fly: when
        # out_mask fires and runner on 3rd with <2 outs, runs score AND an
        # out is recorded.
        is_out = (draws == PA_STRIKEOUT) | (draws == PA_OUT)
        new_io = io_a + is_out.astype(np.int32)

        r1[active] = new_r1
        r2[active] = new_r2
        r3[active] = new_r3
        inning_outs[active] = new_io
        total_runs[active] += runs_delta

        # Deactivate sims that hit 3 outs
        newly_done = active & (inning_outs >= 3)
        active = active & ~newly_done

    return float(total_runs.mean())


def main() -> None:
    n_sims = 50_000
    rng = np.random.default_rng(42)

    print("=" * 78)
    print(f"RE24 SANITY CHECK  (n_sims per state = {n_sims:,})")
    print("=" * 78)
    print(f"{'state':<10s} {'outs':>5s} {'sim_re':>10s} {'tango':>10s} "
          f"{'delta':>10s} {'flag':>6s}")
    print("-" * 78)

    deltas: list[float] = []
    large_errors: list[str] = []

    state_order = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ]

    for state in state_order:
        r1, r2, r3 = state
        for outs in (0, 1, 2):
            sim_re = simulate_state(r1, r2, r3, outs, n_sims, rng)
            tango_re = TANGO_RE24[state][outs]
            delta = sim_re - tango_re
            deltas.append(delta)
            flag = ""
            if abs(delta) > 0.10:
                flag = "****"
                large_errors.append(f"{STATE_LABELS[state]}/{outs}out: {delta:+.3f}")
            elif abs(delta) > 0.05:
                flag = "**"

            print(f"{STATE_LABELS[state]:<10s} {outs:>5d} "
                  f"{sim_re:>10.3f} {tango_re:>10.3f} {delta:>+10.3f} "
                  f"{flag:>6s}")
        print()

    deltas_arr = np.array(deltas)
    print("-" * 78)
    print(f"Summary across 24 states:")
    print(f"  Mean delta:      {deltas_arr.mean():+.3f} runs")
    print(f"  Median delta:    {np.median(deltas_arr):+.3f} runs")
    print(f"  Abs mean delta:  {np.abs(deltas_arr).mean():.3f} runs")
    print(f"  Max |delta|:     {np.abs(deltas_arr).max():.3f} runs")
    print(f"  States |delta|>0.05: {int((np.abs(deltas_arr) > 0.05).sum())}/24")
    print(f"  States |delta|>0.10: {int((np.abs(deltas_arr) > 0.10).sum())}/24")

    if large_errors:
        print("\nStates with |delta| > 0.10:")
        for msg in large_errors:
            print(f"  - {msg}")

    print()
    # Interpretation hint
    mean_d = deltas_arr.mean()
    if mean_d < -0.10:
        print("DIAG: Sim systematically UNDER Tango. Likely causes:")
        print("  - Runner advancement too conservative (2B→home, 1B→home probs)")
        print("  - Sac fly probs too low")
        print("  - PA outcome mix too low on XBH / HR")
    elif mean_d > 0.10:
        print("DIAG: Sim systematically OVER Tango. Likely causes:")
        print("  - Advancement probs too generous")
        print("  - PA outcome mix too high on XBH / HR")
    else:
        print("DIAG: Mean bias within ~0.10 runs. Individual states to investigate above.")


if __name__ == "__main__":
    main()
