"""Layer 2 BvP matchup lift diagnostic.

Probes raw vs dampened K/BB/HR matchup lifts across a representative sample of
real 2024 starter-lineup pairs. Reports:

1. Distribution (mean, std, |lift| quantiles) of raw lifts
2. Same after dampening (K x 0.55, BB x 0.40, HR x 0.20)
3. Translation to rate-point impact at league-average baselines
4. What percentage of matchups produce a "meaningful" adjustment post-dampen
5. Reliability distribution (how often are we leaning on fallback baselines?)
6. Tail behavior: do the dampeners clamp only the extreme tails or shave the whole distribution?
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql
from src.data.feature_eng import get_hitter_vulnerability, get_pitcher_arsenal
from src.data.league_baselines import get_baselines_dict
from src.models.game_sim._sim_utils import MATCHUP_DAMPEN
from src.models.matchup import score_matchup_for_stat

logging.basicConfig(level=logging.WARNING, format="%(message)s")

SEASON = 2024
TRAIN_SEASONS = [2022, 2023, 2024]
N_GAMES_SAMPLE = 400  # ~3,600 pairs
MIN_BF_STARTER = 15
RNG_SEED = 42

# League-average baseline rates for rate-point translation
BASELINE_K = 0.224
BASELINE_BB = 0.083
BASELINE_HR = 0.030


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _inv_logit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _rate_point_delta(baseline: float, logit_lift: float) -> float:
    """Convert a logit-scale lift into a rate-point delta from baseline."""
    return _inv_logit(_logit(baseline) + logit_lift) - baseline


def load_real_pairs(season: int, n_games: int, rng: random.Random) -> pd.DataFrame:
    """Sample real starter-lineup pairs from a season.

    For each sampled game we take (starting pitcher, 9 distinct batters faced).
    """
    query = f"""
        WITH pitcher_game AS (
            SELECT
                pa.game_pk,
                pa.pitcher_id,
                MAX(pa.pitcher_pa_number) AS bf,
                BOOL_OR(pa.pitcher_pa_number = 1) AS faced_first
            FROM production.fact_pa pa
            JOIN production.dim_game g USING (game_pk)
            WHERE g.season = {season}
              AND g.game_type = 'R'
            GROUP BY pa.game_pk, pa.pitcher_id
        ),
        starters AS (
            SELECT game_pk, pitcher_id
            FROM pitcher_game
            WHERE faced_first
              AND bf >= {MIN_BF_STARTER}
        )
        SELECT
            s.game_pk,
            s.pitcher_id,
            pa.batter_id,
            MIN(pa.pitcher_pa_number) AS first_pa_seen
        FROM starters s
        JOIN production.fact_pa pa
          ON pa.game_pk = s.game_pk AND pa.pitcher_id = s.pitcher_id
        GROUP BY s.game_pk, s.pitcher_id, pa.batter_id
        ORDER BY s.game_pk, first_pa_seen
    """
    df = read_sql(query)
    all_games = df["game_pk"].unique().tolist()
    rng.shuffle(all_games)
    pick = set(all_games[:n_games])
    return df[df["game_pk"].isin(pick)].reset_index(drop=True)


def run_diagnostic() -> None:
    rng = random.Random(RNG_SEED)
    print(f"Loading profiles for season {SEASON}...")
    pitcher_arsenal = get_pitcher_arsenal(SEASON)
    hitter_vuln = get_hitter_vulnerability(SEASON)
    baselines_pt = get_baselines_dict(seasons=TRAIN_SEASONS, recency_weights="equal")

    print(f"Sampling {N_GAMES_SAMPLE} starter games from {SEASON}...")
    pairs_df = load_real_pairs(SEASON, N_GAMES_SAMPLE, rng)
    print(f"  {len(pairs_df):,} pitcher-batter pairs across "
          f"{pairs_df['game_pk'].nunique()} games")

    rows = []
    for _, row in pairs_df.iterrows():
        pid = int(row["pitcher_id"])
        bid = int(row["batter_id"])
        out = {"pitcher_id": pid, "batter_id": bid}
        for stat in ("k", "bb", "hr"):
            res = score_matchup_for_stat(
                stat_name=stat,
                pitcher_id=pid,
                batter_id=bid,
                pitcher_arsenal=pitcher_arsenal,
                hitter_vuln=hitter_vuln,
                baselines_pt=baselines_pt,
            )
            lift = res.get(f"matchup_{stat}_logit_lift", 0.0)
            if isinstance(lift, float) and np.isnan(lift):
                lift = 0.0
            out[f"{stat}_lift"] = float(lift)
            out[f"{stat}_reliability"] = float(res.get("avg_reliability", 0.0))
            out[f"{stat}_n_pt"] = int(res.get("n_pitch_types", 0))
        rows.append(out)

    df = pd.DataFrame(rows)
    print(f"\n  Scored {len(df):,} pairs")
    print(f"  Nonzero K lifts: {(df['k_lift'] != 0).sum():,} "
          f"({(df['k_lift'] != 0).mean() * 100:.1f}%)")
    print(f"  Nonzero BB lifts: {(df['bb_lift'] != 0).sum():,} "
          f"({(df['bb_lift'] != 0).mean() * 100:.1f}%)")
    print(f"  Nonzero HR lifts: {(df['hr_lift'] != 0).sum():,} "
          f"({(df['hr_lift'] != 0).mean() * 100:.1f}%)")

    print("\n" + "=" * 72)
    print("RAW vs DAMPENED LOGIT-SCALE LIFTS")
    print("=" * 72)
    print(f"{'stat':<6} {'mean':>8} {'std':>8} {'p10':>8} {'p50':>8} {'p90':>8} "
          f"{'p99':>8} {'|max|':>8}")
    for stat, baseline in [("k", BASELINE_K), ("bb", BASELINE_BB), ("hr", BASELINE_HR)]:
        raw = df[f"{stat}_lift"].values
        damp = raw * MATCHUP_DAMPEN[stat]
        print(f"\n{stat.upper()} (dampen={MATCHUP_DAMPEN[stat]:.2f}, "
              f"baseline={baseline:.3f})")
        for label, arr in [("raw", raw), ("damp", damp)]:
            abs_arr = np.abs(arr)
            print(f"  {label:<4} {arr.mean():>8.4f} {arr.std():>8.4f} "
                  f"{np.quantile(arr, 0.10):>8.4f} {np.quantile(arr, 0.50):>8.4f} "
                  f"{np.quantile(arr, 0.90):>8.4f} {np.quantile(arr, 0.99):>8.4f} "
                  f"{abs_arr.max():>8.4f}")

    print("\n" + "=" * 72)
    print("RATE-POINT IMPACT AT LEAGUE-AVG BASELINE (percentage points)")
    print("=" * 72)
    for stat, baseline in [("k", BASELINE_K), ("bb", BASELINE_BB), ("hr", BASELINE_HR)]:
        raw = df[f"{stat}_lift"].values
        damp = raw * MATCHUP_DAMPEN[stat]
        raw_rp = np.array([_rate_point_delta(baseline, l) * 100 for l in raw])
        damp_rp = np.array([_rate_point_delta(baseline, l) * 100 for l in damp])
        print(f"\n{stat.upper()} @ baseline {baseline:.1%}")
        for label, arr in [("raw", raw_rp), ("damp", damp_rp)]:
            abs_arr = np.abs(arr)
            print(f"  {label:<4}  mean {arr.mean():>6.2f}pp  "
                  f"std {arr.std():>5.2f}pp  "
                  f"|p50| {np.median(abs_arr):>5.2f}pp  "
                  f"|p90| {np.quantile(abs_arr, 0.90):>5.2f}pp  "
                  f"|p99| {np.quantile(abs_arr, 0.99):>5.2f}pp  "
                  f"|max| {abs_arr.max():>5.2f}pp")

    print("\n" + "=" * 72)
    print("HOW OFTEN DOES DAMPENED LIFT EXCEED A MEANINGFUL THRESHOLD?")
    print("=" * 72)
    thresholds_pp = {"k": 1.0, "bb": 0.5, "hr": 0.25}
    for stat, baseline in [("k", BASELINE_K), ("bb", BASELINE_BB), ("hr", BASELINE_HR)]:
        raw = df[f"{stat}_lift"].values
        damp = raw * MATCHUP_DAMPEN[stat]
        raw_rp = np.array([abs(_rate_point_delta(baseline, l) * 100) for l in raw])
        damp_rp = np.array([abs(_rate_point_delta(baseline, l) * 100) for l in damp])
        thr = thresholds_pp[stat]
        raw_pct = (raw_rp > thr).mean() * 100
        damp_pct = (damp_rp > thr).mean() * 100
        print(f"  {stat.upper():<3} (thr={thr:.2f}pp):  "
              f"raw {raw_pct:>5.1f}%  -->  damp {damp_pct:>5.1f}%")

    print("\n" + "=" * 72)
    print("RELIABILITY DISTRIBUTION (hitter side fallback coverage)")
    print("=" * 72)
    for stat in ("k", "bb", "hr"):
        rel = df[f"{stat}_reliability"].values
        print(f"  {stat.upper()}  mean {rel.mean():.3f}  "
              f"p10 {np.quantile(rel, 0.10):.3f}  "
              f"p50 {np.quantile(rel, 0.50):.3f}  "
              f"p90 {np.quantile(rel, 0.90):.3f}  "
              f"pct_below_0.25 {(rel < 0.25).mean() * 100:.1f}%  "
              f"pct_at_1.0 {(rel >= 0.99).mean() * 100:.1f}%")

    print("\n" + "=" * 72)
    print("TAIL BEHAVIOR: rank correlation of raw lift with rate-point impact")
    print("=" * 72)
    for stat, baseline in [("k", BASELINE_K), ("bb", BASELINE_BB), ("hr", BASELINE_HR)]:
        raw = df[f"{stat}_lift"].values
        damp = raw * MATCHUP_DAMPEN[stat]
        order = np.argsort(raw)
        sorted_raw = raw[order]
        sorted_damp = damp[order]
        n = len(order)
        deciles = [int(n * q) for q in (0.0, 0.1, 0.5, 0.9, 0.99)]
        deciles[-1] = min(deciles[-1], n - 1)
        print(f"\n{stat.upper()} quantile snapshot")
        for q, idx in zip((0.0, 0.1, 0.5, 0.9, 0.99), deciles):
            rp_raw = _rate_point_delta(baseline, sorted_raw[idx]) * 100
            rp_damp = _rate_point_delta(baseline, sorted_damp[idx]) * 100
            print(f"  q{q:.2f}  raw_logit {sorted_raw[idx]:>7.4f} "
                  f"({rp_raw:>6.2f}pp)   "
                  f"damp_logit {sorted_damp[idx]:>7.4f} ({rp_damp:>6.2f}pp)")

    out_path = Path("outputs") / "layer2_lift_diagnostic.parquet"
    out_path.parent.mkdir(exist_ok=True)
    df.to_parquet(out_path)
    print(f"\nSaved raw pair-level output to {out_path}")


if __name__ == "__main__":
    run_diagnostic()
