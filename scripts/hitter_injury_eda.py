#!/usr/bin/env python
"""
EDA: Hitter injury prediction + projection stat stability analysis.

Part 1: Which hitter attributes predict next-season IL stints?
Part 2: Which hitter stats are stable enough for Bayesian projection?
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("hitter_eda")


def main() -> None:
    # ================================================================
    # PART 1: HITTER INJURY PREDICTION
    # ================================================================
    logger.info("Loading hitter IL data...")
    il_data = read_sql("""
        WITH hitter_ids AS (
            SELECT DISTINCT player_id FROM production.fact_player_game_mlb
            WHERE player_role = 'batter' AND season BETWEEN 2019 AND 2025
        )
        SELECT pst.player_id, pst.season, pst.days_on_status,
               pst.injury_description, pst.status_type,
               CASE
                   WHEN LOWER(injury_description) LIKE '%%hamstring%%' THEN 'hamstring'
                   WHEN LOWER(injury_description) LIKE '%%oblique%%'
                        OR LOWER(injury_description) LIKE '%%abdomin%%' THEN 'core'
                   WHEN LOWER(injury_description) LIKE '%%knee%%'
                        OR LOWER(injury_description) LIKE '%%acl%%'
                        OR LOWER(injury_description) LIKE '%%meniscus%%' THEN 'knee'
                   WHEN LOWER(injury_description) LIKE '%%shoulder%%' THEN 'shoulder'
                   WHEN LOWER(injury_description) LIKE '%%wrist%%'
                        OR LOWER(injury_description) LIKE '%%hand%%'
                        OR LOWER(injury_description) LIKE '%%thumb%%' THEN 'hand_wrist'
                   WHEN LOWER(injury_description) LIKE '%%back%%' THEN 'back'
                   WHEN LOWER(injury_description) LIKE '%%ankle%%'
                        OR LOWER(injury_description) LIKE '%%foot%%' THEN 'foot_ankle'
                   WHEN LOWER(injury_description) LIKE '%%quad%%'
                        OR LOWER(injury_description) LIKE '%%calf%%'
                        OR LOWER(injury_description) LIKE '%%groin%%'
                        OR LOWER(injury_description) LIKE '%%hip%%' THEN 'lower_body'
                   ELSE 'other'
               END as injury_type
        FROM production.fact_player_status_timeline pst
        JOIN hitter_ids hi ON pst.player_id = hi.player_id
        WHERE pst.status_type LIKE 'IL%%' AND pst.season BETWEEN 2019 AND 2025
    """, {})

    print(f"Hitter IL stints: {len(il_data)}")
    print(f"\nBy injury type:")
    type_counts = il_data.groupby("injury_type").agg(
        stints=("player_id", "count"),
        players=("player_id", "nunique"),
        avg_days=("days_on_status", "mean"),
    ).sort_values("stints", ascending=False).round(1)
    print(type_counts)

    # Hitter features
    logger.info("Loading hitter features...")
    hitter_features = read_sql("""
        SELECT fg.player_id as batter_id, fg.season,
               SUM(bat_pa) as pa, SUM(bat_ab) as ab,
               SUM(bat_h) as h, SUM(bat_hr) as hr,
               SUM(bat_r) as runs, SUM(bat_rbi) as rbi,
               SUM(bat_bb) as bb, SUM(bat_k) as k,
               SUM(bat_sb) as sb, SUM(bat_cs) as cs,
               COUNT(DISTINCT fg.game_pk) as games,
               SUM(bat_k)::float / NULLIF(SUM(bat_pa), 0) as k_rate,
               SUM(bat_bb)::float / NULLIF(SUM(bat_pa), 0) as bb_rate,
               SUM(bat_hr)::float / NULLIF(SUM(bat_pa), 0) as hr_rate,
               SUM(bat_sb)::float / NULLIF(COUNT(DISTINCT fg.game_pk), 0) as sb_per_game
        FROM production.fact_player_game_mlb fg
        WHERE fg.player_role = 'batter' AND fg.season BETWEEN 2019 AND 2024
        GROUP BY fg.player_id, fg.season
        HAVING SUM(bat_pa) >= 200
    """, {})

    # Statcast quality
    statcast = read_sql("""
        SELECT batter_id, season, pa as pa_sc,
               barrel_pct, hard_hit_pct, xwoba, xba, xslg,
               k_pct, bb_pct, woba, wrc_plus
        FROM production.fact_batting_advanced
        WHERE season BETWEEN 2019 AND 2024 AND pa >= 200
    """, {})

    # Sprint speed from staging table
    sprint = read_sql("""
        SELECT player_id as batter_id, season,
               sprint_speed
        FROM staging.statcast_sprint_speed
        WHERE season BETWEEN 2019 AND 2024
              AND sprint_speed IS NOT NULL
    """, {})

    # Age
    ages = read_sql("""
        SELECT player_id as batter_id,
               EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date))::int as current_age
        FROM production.dim_player WHERE birth_date IS NOT NULL
    """, {})

    # Merge
    merged = hitter_features.merge(statcast, on=["batter_id", "season"], how="left")
    merged = merged.merge(sprint, on=["batter_id", "season"], how="left")
    merged = merged.merge(ages, on="batter_id", how="left")
    # Approximate age at time of season
    merged["age"] = merged["current_age"] - (2025 - merged["season"])

    # IL outcome: season N features -> season N+1 injury
    il_agg = il_data.groupby(["player_id", "season"]).agg(
        il_stints=("player_id", "count"),
        il_days=("days_on_status", "sum"),
    ).reset_index()
    il_agg["predict_season"] = il_agg["season"] - 1

    merged = merged.merge(
        il_agg[["player_id", "predict_season", "il_stints", "il_days"]].rename(
            columns={"player_id": "batter_id", "predict_season": "season"}
        ),
        on=["batter_id", "season"], how="left",
    )
    merged["injured_next"] = (merged["il_stints"].fillna(0) > 0).astype(int)

    # Prior IL
    any_prior = il_data.groupby("player_id")["season"].apply(
        lambda s: {yr: int(any(x < yr for x in s)) for yr in range(2019, 2025)}
    ).explode().reset_index()
    any_prior.columns = ["batter_id", "season", "has_prior_il"]
    any_prior["season"] = any_prior["season"].astype(int)
    any_prior["has_prior_il"] = any_prior["has_prior_il"].astype(int)
    merged = merged.merge(any_prior, on=["batter_id", "season"], how="left")
    merged["has_prior_il"] = merged["has_prior_il"].fillna(0)

    # YoY deltas
    merged = merged.sort_values(["batter_id", "season"])
    for col in ["k_rate", "bb_rate", "hr_rate", "sprint_speed", "barrel_pct", "hard_hit_pct"]:
        if col in merged.columns:
            merged[f"{col}_delta"] = merged.groupby("batter_id")[col].diff()

    analysis = merged[merged["season"] <= 2023].copy()
    print(f"\nAnalysis: {len(analysis)} hitter-seasons, injury rate {analysis['injured_next'].mean():.1%}")

    # Correlations
    print("\n" + "=" * 65)
    print("  HITTER FEATURES vs NEXT-SEASON IL STINT")
    print("=" * 65)

    features = [
        ("age", "Age"),
        ("games", "Games played"),
        ("pa", "Plate appearances"),
        ("k_rate", "K rate"),
        ("bb_rate", "BB rate"),
        ("hr_rate", "HR rate"),
        ("sb_per_game", "SB per game"),
        ("sprint_speed", "Sprint speed"),
        ("barrel_pct", "Barrel %"),
        ("hard_hit_pct", "Hard hit %"),
        ("xwoba", "xwOBA"),
        ("has_prior_il", "Prior IL history"),
        ("k_rate_delta", "K rate change YoY"),
        ("bb_rate_delta", "BB rate change YoY"),
        ("sprint_speed_delta", "Sprint speed change YoY"),
        ("barrel_pct_delta", "Barrel % change YoY"),
        ("hard_hit_pct_delta", "Hard hit % change YoY"),
    ]

    results = []
    for col, label in features:
        if col not in analysis.columns:
            continue
        valid = analysis[[col, "injured_next"]].dropna()
        if len(valid) < 50:
            continue
        r = np.corrcoef(valid[col], valid["injured_next"])[0, 1]
        inj = valid[valid["injured_next"] == 1][col]
        healthy = valid[valid["injured_next"] == 0][col]
        results.append({"feature": col, "label": label, "r": r,
                         "mean_inj": inj.mean(), "mean_healthy": healthy.mean(),
                         "diff": inj.mean() - healthy.mean(), "n": len(valid)})

    results_df = pd.DataFrame(results).sort_values("r", key=abs, ascending=False)
    print(f"\n{'Feature':<28s} {'r':>7s} {'Injured':>9s} {'Healthy':>9s} {'Diff':>8s}")
    print("-" * 60)
    for _, r in results_df.iterrows():
        flag = " ***" if abs(r["r"]) >= 0.05 else " *" if abs(r["r"]) >= 0.03 else ""
        print(f"  {r['label']:<26s} {r['r']:+.4f} {r['mean_inj']:9.3f} {r['mean_healthy']:9.3f} {r['diff']:+8.3f}{flag}")

    # Bucketed
    print("\n" + "=" * 65)
    print("  BUCKETED ANALYSIS")
    print("=" * 65)

    for col in ["age", "sprint_speed", "games", "has_prior_il"]:
        if col not in analysis.columns:
            continue
        valid = analysis[[col, "injured_next"]].dropna()
        if len(valid) < 100:
            continue

        if col == "has_prior_il":
            for val, label in [(0, "No prior IL"), (1, "Prior IL")]:
                sub = valid[valid[col] == val]
                print(f"  {col} = {label}: {sub['injured_next'].mean():.1%} ({len(sub)})")
            print()
            continue

        try:
            valid["q"] = pd.qcut(valid[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            valid["q"] = pd.qcut(valid[col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])

        qtable = valid.groupby("q", observed=False)["injured_next"].agg(["mean", "count"]).reset_index()
        ranges = valid.groupby("q", observed=False)[col].agg(["min", "max"]).reset_index()
        q1 = qtable[qtable["q"] == "Q1"]["mean"].values[0]
        q4 = qtable[qtable["q"] == "Q4"]["mean"].values[0]
        print(f"\n  {col}:")
        for _, qr in qtable.iterrows():
            rng = ranges[ranges["q"] == qr["q"]]
            lo, hi = rng["min"].values[0], rng["max"].values[0]
            bar = "#" * int(qr["mean"] * 100)
            print(f"    {qr['q']}: {qr['mean']:5.1%} ({qr['count']:4.0f}p)  [{lo:.1f}-{hi:.1f}]  {bar}")
        print(f"    Q4/Q1 lift: {q4 / max(q1, 0.01):.2f}x")

    # ================================================================
    # PART 2: HITTER STAT STABILITY (YoY correlation)
    # ================================================================
    print("\n" + "=" * 65)
    print("  HITTER STAT YEAR-TO-YEAR STABILITY (for projection model)")
    print("=" * 65)

    # Build year pairs
    all_data = merged[["batter_id", "season", "k_rate", "bb_rate", "hr_rate",
                        "sb_per_game", "sprint_speed", "barrel_pct", "hard_hit_pct",
                        "xwoba", "xba", "xslg", "woba", "wrc_plus",
                        "games", "pa"]].copy()

    current = all_data.copy()
    prior = all_data.copy()
    prior["season"] = prior["season"] + 1
    pairs = current.merge(prior, on=["batter_id", "season"], suffixes=("", "_prior"))

    print(f"\n  Year-pairs: {len(pairs)} (min 200 PA both seasons)")
    print(f"\n  {'Stat':<20s} {'YoY r':>7s} {'Projectable?'}")
    print("  " + "-" * 45)

    stats_to_check = [
        ("k_rate", "K%"),
        ("bb_rate", "BB%"),
        ("hr_rate", "HR/PA"),
        ("sb_per_game", "SB/game"),
        ("sprint_speed", "Sprint speed"),
        ("barrel_pct", "Barrel %"),
        ("hard_hit_pct", "Hard hit %"),
        ("xwoba", "xwOBA"),
        ("xba", "xBA"),
        ("xslg", "xSLG"),
        ("woba", "wOBA"),
        ("wrc_plus", "wRC+"),
    ]

    for col, label in stats_to_check:
        if col not in pairs.columns or f"{col}_prior" not in pairs.columns:
            continue
        valid = pairs[[col, f"{col}_prior"]].dropna()
        if len(valid) < 50:
            continue
        r = np.corrcoef(valid[col], valid[f"{col}_prior"])[0, 1]
        tier = "STRONG" if r >= 0.60 else "MODERATE" if r >= 0.40 else "WEAK"
        print(f"  {label:<20s} {r:+.3f}   {tier}")

    analysis.to_parquet("data/cached/_hitter_injury_analysis.parquet", index=False)
    results_df.to_csv("outputs/hitter_injury_correlations.csv", index=False)
    logger.info("Saved hitter analysis data")


if __name__ == "__main__":
    main()
