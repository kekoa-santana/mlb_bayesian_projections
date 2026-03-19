"""Historical validation of the hitter breakout archetype model.

Runs the breakout scoring on past seasons (using only data available
at prediction time) and checks whether flagged players actually broke
out in the following season.

Folds: 2022->2023, 2023->2024, 2024->2025
(Skips 2020 COVID season in delta calculations)
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.data.db import read_sql
from src.data.queries import (
    get_batted_ball_spray,
    get_hitter_observed_profile,
    get_sprint_speed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("breakout_backtest")


# ---------------------------------------------------------------------------
# Helpers (same as breakout_model.py)
# ---------------------------------------------------------------------------

def _pctl(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def _inv_pctl(s: pd.Series) -> pd.Series:
    return 1.0 - s.rank(pct=True, method="average")


# ---------------------------------------------------------------------------
# Feature loading (DB-only, no dashboard parquets)
# ---------------------------------------------------------------------------

def load_and_score(pred_season: int, min_pa: int = 200) -> pd.DataFrame:
    """Load features for ``pred_season`` and compute breakout scores.

    Uses only data available at end of ``pred_season`` — no leakage
    from the outcome season.  YoY observed deltas proxy for Bayesian
    trajectory projections.
    """
    prior_season = pred_season - 1
    # Skip 2020 (COVID-shortened)
    if prior_season == 2020:
        prior_season = 2019

    # 1. Batting advanced stats
    adv = read_sql(
        f"""
        SELECT batter_id, pa, barrel_pct, woba, xwoba, k_pct, bb_pct, wrc_plus
        FROM production.fact_batting_advanced
        WHERE season = {pred_season} AND pa >= {min_pa}
        """,
        {},
    )

    # 2. Prior season for YoY deltas
    adv_prior = read_sql(
        f"""
        SELECT batter_id, k_pct AS k_pct_prior, bb_pct AS bb_pct_prior
        FROM production.fact_batting_advanced
        WHERE season = {prior_season} AND pa >= 100
        """,
        {},
    )

    # 3. Observed profile (z_contact, chase, exit velo, hard-hit)
    obs = get_hitter_observed_profile(pred_season)

    # 4. Spray data
    spray = get_batted_ball_spray(pred_season)

    # 5. OAA
    oaa = read_sql(
        f"""
        SELECT player_id AS batter_id, SUM(outs_above_average) AS oaa
        FROM production.fact_fielding_oaa
        WHERE season = {pred_season}
        GROUP BY player_id
        """,
        {},
    )

    # 6. Sprint speed
    sprint = get_sprint_speed(pred_season)

    # 7. Age + name
    demo = read_sql(
        f"""
        SELECT player_id AS batter_id,
               player_name AS batter_name,
               ({pred_season} - EXTRACT(YEAR FROM birth_date))::int AS age
        FROM production.dim_player
        """,
        {},
    )

    # --- Assemble ---
    base = adv.copy()
    base = base.merge(
        obs[["batter_id", "z_contact_pct", "chase_rate",
             "avg_exit_velo", "hard_hit_pct"]],
        on="batter_id", how="left",
    )
    if not spray.empty:
        base = base.merge(spray[["batter_id", "pull_pct"]], on="batter_id", how="left")
    if "pull_pct" not in base.columns:
        base["pull_pct"] = np.nan
    base = base.merge(oaa, on="batter_id", how="left")
    if not sprint.empty:
        base = base.merge(
            sprint[["player_id", "sprint_speed"]].rename(
                columns={"player_id": "batter_id"}
            ),
            on="batter_id", how="left",
        )
    if "sprint_speed" not in base.columns:
        base["sprint_speed"] = np.nan
    base = base.merge(demo, on="batter_id", how="left")
    base["oaa"] = base["oaa"].fillna(0).astype(float)

    # YoY deltas (trajectory proxy)
    base = base.merge(adv_prior, on="batter_id", how="left")
    base["delta_k_rate"] = base["k_pct"] - base["k_pct_prior"]
    base["delta_bb_rate"] = base["bb_pct"] - base["bb_pct_prior"]

    # Fill NaN with medians
    fill_cols = [
        "barrel_pct", "z_contact_pct", "pull_pct", "hard_hit_pct",
        "avg_exit_velo", "chase_rate", "sprint_speed", "bb_pct",
        "woba", "xwoba", "k_pct", "delta_k_rate", "delta_bb_rate",
    ]
    for col in fill_cols:
        if col in base.columns:
            base[col] = base[col].fillna(base[col].median())
    base["age"] = base["age"].fillna(28)

    # --- Score archetypes ---
    base["hitter_fit"] = (
        0.30 * _pctl(base["barrel_pct"])
        + 0.25 * _pctl(base["z_contact_pct"])
        + 0.20 * _pctl(base["pull_pct"])
        + 0.15 * _pctl(base["hard_hit_pct"])
        + 0.10 * _pctl(base["avg_exit_velo"])
    )

    power = 0.60 * _pctl(base["avg_exit_velo"]) + 0.40 * _pctl(base["hard_hit_pct"])
    discipline = 0.60 * _inv_pctl(base["chase_rate"]) + 0.40 * _pctl(base["bb_pct"])
    speed = _pctl(base["sprint_speed"])
    defense = _pctl(base["oaa"])
    contact = _pctl(base["z_contact_pct"])
    base["all_around_fit"] = (
        0.25 * power + 0.20 * discipline + 0.20 * speed
        + 0.20 * defense + 0.15 * contact
    )

    base["primary_fit"] = base[["hitter_fit", "all_around_fit"]].max(axis=1)
    base["breakout_type"] = np.where(
        base["hitter_fit"] >= base["all_around_fit"], "Hitter", "All-Around",
    )

    # --- Room to grow ---
    age_mult = np.clip(1.0 - (base["age"] - 23) / 12.0, 0.10, 1.0)
    woba_pctl = _pctl(base["woba"])
    gap = (base["primary_fit"] - woba_pctl).clip(0, 1)
    gap_score = (gap / 0.35).clip(0, 1)
    k_improving = _inv_pctl(base["delta_k_rate"].fillna(0))
    bb_improving = _pctl(base["delta_bb_rate"].fillna(0))
    trajectory = 0.50 * k_improving + 0.50 * bb_improving
    base["room_to_grow"] = (0.55 * gap_score + 0.45 * trajectory) * age_mult

    # --- Breakout score ---
    base["breakout_score"] = base["primary_fit"] * base["room_to_grow"]

    # Tiers
    score_pctl = _pctl(base["breakout_score"])
    base["breakout_tier"] = np.where(
        score_pctl >= 0.90, "High",
        np.where(score_pctl >= 0.75, "Medium", ""),
    )

    return base


def load_outcomes(outcome_season: int, min_pa: int = 150) -> pd.DataFrame:
    """Load outcome season batting stats."""
    return read_sql(
        f"""
        SELECT batter_id, pa AS pa_next, woba AS woba_next,
               wrc_plus AS wrc_next, k_pct AS k_next, bb_pct AS bb_next
        FROM production.fact_batting_advanced
        WHERE season = {outcome_season} AND pa >= {min_pa}
        """,
        {},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    folds = [(2022, 2023), (2023, 2024), (2024, 2025)]
    all_results: list[pd.DataFrame] = []

    for pred_yr, out_yr in folds:
        logger.info("=" * 50)
        logger.info("Scoring %d -> validating %d", pred_yr, out_yr)

        scores = load_and_score(pred_yr)
        outcomes = load_outcomes(out_yr)

        merged = scores.merge(outcomes, on="batter_id", how="inner")
        merged["woba_delta"] = merged["woba_next"] - merged["woba"]
        merged["wrc_delta"] = merged["wrc_next"] - merged["wrc_plus"]
        merged["pred_season"] = pred_yr
        merged["out_season"] = out_yr

        # "Broke out" = wOBA improved by ≥25 pts AND outcome wOBA ≥ .310
        merged["broke_out_strict"] = (
            (merged["woba_delta"] >= 0.025) & (merged["woba_next"] >= 0.310)
        )
        # Looser: any wOBA improvement ≥ 20 pts
        merged["broke_out_loose"] = merged["woba_delta"] >= 0.020

        all_results.append(merged)
        logger.info(
            "  %d players matched (of %d scored, %d outcomes)",
            len(merged), len(scores), len(outcomes),
        )

    results = pd.concat(all_results, ignore_index=True)

    # ===================================================================
    # Report
    # ===================================================================
    print()
    print("=" * 70)
    print("BREAKOUT MODEL HISTORICAL VALIDATION")
    print("=" * 70)
    print(f"Folds: {folds}")
    print(f"Total player-seasons: {len(results)}")
    print(f"Breakout definition (strict): wOBA delta >= .025 AND outcome wOBA >= .310")
    print(f"Breakout definition (loose):  wOBA delta >= .020")

    # --- By tier ---
    print()
    print("-" * 70)
    print("BREAKOUT RATES BY TIER")
    print("-" * 70)
    print(f"  {'Tier':>8}   {'n':>4}  {'Strict%':>8}  {'Loose%':>8}  "
          f"{'Avg wOBA chg':>10}  {'Avg wRC+ chg':>10}")
    for tier in ["High", "Medium", ""]:
        label = tier if tier else "None"
        sub = results[results["breakout_tier"] == tier]
        n = len(sub)
        strict = sub["broke_out_strict"].mean() * 100
        loose = sub["broke_out_loose"].mean() * 100
        woba_d = sub["woba_delta"].mean()
        wrc_d = sub["wrc_delta"].mean()
        print(f"  {label:>8}  {n:>5}  {strict:>7.1f}%  {loose:>7.1f}%  "
              f"{woba_d:>+10.3f}  {wrc_d:>+10.1f}")

    # Baseline: all players
    strict_all = results["broke_out_strict"].mean() * 100
    loose_all = results["broke_out_loose"].mean() * 100
    print(f"  {'ALL':>8}  {len(results):>5}  {strict_all:>7.1f}%  {loose_all:>7.1f}%  "
          f"{results['woba_delta'].mean():>+10.3f}  {results['wrc_delta'].mean():>+10.1f}")

    # --- Lift over baseline ---
    print()
    base_strict = results[results["breakout_tier"] == ""]["broke_out_strict"].mean()
    high_strict = results[results["breakout_tier"] == "High"]["broke_out_strict"].mean()
    med_strict = results[results["breakout_tier"] == "Medium"]["broke_out_strict"].mean()
    if base_strict > 0:
        print(f"  High tier lift over baseline: {high_strict/base_strict:.1f}x (strict)")
        print(f"  Medium tier lift over baseline: {med_strict/base_strict:.1f}x (strict)")

    # --- Correlation ---
    print()
    print("-" * 70)
    print("CORRELATION: breakout_score vs. actual improvement")
    print("-" * 70)
    r_woba, p_woba = stats.spearmanr(results["breakout_score"], results["woba_delta"])
    r_wrc, p_wrc = stats.spearmanr(results["breakout_score"], results["wrc_delta"])
    print(f"  Spearman (breakout_score vs wOBA delta): r={r_woba:.3f}, p={p_woba:.4f}")
    print(f"  Spearman (breakout_score vs wRC+ delta): r={r_wrc:.3f}, p={p_wrc:.4f}")

    # By archetype
    for atype in ["Hitter", "All-Around"]:
        sub = results[results["breakout_type"] == atype]
        r_a, p_a = stats.spearmanr(sub["breakout_score"], sub["woba_delta"])
        print(f"  Spearman ({atype} breakout_score vs wOBA delta): r={r_a:.3f}, p={p_a:.4f}")

    # --- By archetype tier ---
    print()
    print("-" * 70)
    print("BY ARCHETYPE")
    print("-" * 70)
    for atype in ["Hitter", "All-Around"]:
        print(f"\n  {atype}:")
        for tier in ["High", "Medium", ""]:
            label = tier if tier else "None"
            sub = results[(results["breakout_type"] == atype) & (results["breakout_tier"] == tier)]
            n = len(sub)
            if n == 0:
                continue
            strict = sub["broke_out_strict"].mean() * 100
            woba_d = sub["woba_delta"].mean()
            print(f"    {label:>8}: n={n:>4}  strict_breakout={strict:>5.1f}%  avg_wOBA_delta={woba_d:+.3f}")

    # --- Per-fold breakdown ---
    print()
    print("-" * 70)
    print("PER-FOLD BREAKDOWN")
    print("-" * 70)
    for pred_yr, out_yr in folds:
        fold = results[results["pred_season"] == pred_yr]
        print(f"\n  {pred_yr} -> {out_yr} (n={len(fold)}):")
        for tier in ["High", "Medium", ""]:
            label = tier if tier else "None"
            sub = fold[fold["breakout_tier"] == tier]
            n = len(sub)
            strict = sub["broke_out_strict"].mean() * 100
            woba_d = sub["woba_delta"].mean()
            print(f"    {label:>8}: n={n:>3}  strict_breakout={strict:>5.1f}%  avg_wOBA_delta={woba_d:+.3f}")

    # --- Named hits and misses ---
    print()
    print("-" * 70)
    print("HIGH-TIER HITS (correctly predicted breakouts)")
    print("-" * 70)
    hits = results[
        (results["breakout_tier"] == "High") & (results["broke_out_strict"])
    ].sort_values("breakout_score", ascending=False)
    for _, r in hits.head(20).iterrows():
        print(
            f"  {r.batter_name:<22} {int(r.pred_season)}->{int(r.out_season)}  "
            f"age={int(r.age):>2}  wOBA {r.woba:.3f}->{r.woba_next:.3f} "
            f"({r.woba_delta:+.3f})  type={r.breakout_type}"
        )

    print()
    print("-" * 70)
    print("HIGH-TIER MISSES (top scored but didn't break out)")
    print("-" * 70)
    misses = results[
        (results["breakout_tier"] == "High") & (~results["broke_out_strict"])
    ].sort_values("breakout_score", ascending=False)
    for _, r in misses.head(15).iterrows():
        print(
            f"  {r.batter_name:<22} {int(r.pred_season)}->{int(r.out_season)}  "
            f"age={int(r.age):>2}  wOBA {r.woba:.3f}->{r.woba_next:.3f} "
            f"({r.woba_delta:+.3f})  type={r.breakout_type}"
        )

    # --- Quintile analysis ---
    print()
    print("-" * 70)
    print("QUINTILE ANALYSIS (breakout_score)")
    print("-" * 70)
    results["quintile"] = pd.qcut(
        results["breakout_score"], 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    )
    for q in ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]:
        sub = results[results["quintile"] == q]
        strict = sub["broke_out_strict"].mean() * 100
        woba_d = sub["woba_delta"].mean()
        print(f"  {q:>10}: n={len(sub):>4}  strict_breakout={strict:>5.1f}%  "
              f"avg_wOBA_delta={woba_d:+.3f}  avg_age={sub.age.mean():.1f}")


if __name__ == "__main__":
    main()
