#!/usr/bin/env python
"""
EDA: Pitcher biomechanical features vs next-season arm injury.

Builds a pitcher-season dataset with Statcast biomechanics (velocity,
usage, extension, release point, spin, arm angle) and IL stint outcomes,
then computes correlations and group comparisons.
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
logger = logging.getLogger("injury_eda")


def build_injury_outcomes() -> pd.DataFrame:
    """Build pitcher-season arm injury flags from IL data."""
    il_data = read_sql("""
        WITH pitcher_ids AS (
            SELECT DISTINCT player_id FROM production.fact_player_game_mlb
            WHERE player_role = 'pitcher' AND season BETWEEN 2019 AND 2025
        )
        SELECT pst.player_id, pst.season, pst.days_on_status, pst.injury_description,
               CASE
                   WHEN LOWER(pst.injury_description) LIKE '%%elbow%%'
                        OR LOWER(pst.injury_description) LIKE '%%ucl%%'
                        OR LOWER(pst.injury_description) LIKE '%%tommy john%%' THEN 'elbow'
                   WHEN LOWER(pst.injury_description) LIKE '%%shoulder%%'
                        OR LOWER(pst.injury_description) LIKE '%%rotator%%'
                        OR LOWER(pst.injury_description) LIKE '%%labrum%%' THEN 'shoulder'
                   WHEN LOWER(pst.injury_description) LIKE '%%forearm%%'
                        OR LOWER(pst.injury_description) LIKE '%%flexor%%' THEN 'forearm'
                   WHEN LOWER(pst.injury_description) LIKE '%%lat%%'
                        AND LOWER(pst.injury_description) NOT LIKE '%%plate%%' THEN 'lat'
                   ELSE 'other'
               END as injury_type
        FROM production.fact_player_status_timeline pst
        JOIN pitcher_ids pi ON pst.player_id = pi.player_id
        WHERE pst.status_type LIKE 'IL%%' AND pst.season BETWEEN 2019 AND 2025
    """, {})

    il_data["is_arm"] = il_data["injury_type"].isin(["elbow", "shoulder", "forearm", "lat"])

    logger.info("Total IL stints: %d, arm: %d", len(il_data), il_data["is_arm"].sum())
    logger.info("By type:\n%s", il_data.groupby("injury_type")["player_id"].count().sort_values(ascending=False))

    arm = il_data[il_data["is_arm"]].groupby(["player_id", "season"]).agg(
        arm_il_stints=("is_arm", "sum"),
        arm_il_days=("days_on_status", "sum"),
    ).reset_index()

    return arm


def build_bio_features() -> pd.DataFrame:
    """Build pitcher-season biomechanical features from Statcast."""
    logger.info("Building biomechanical features (this takes a minute)...")
    bio = read_sql("""
        SELECT fp.pitcher_id, dg.season,
               AVG(fp.release_speed) as avg_velo,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fp.release_speed) as p95_velo,
               STDDEV(fp.release_speed) as velo_sd,
               AVG(CASE WHEN fp.pitch_type IN ('FF','SI','FC') THEN fp.release_speed END) as fb_velo,
               AVG(CASE WHEN fp.pitch_type IN ('SL','ST','SV','CU','KC') THEN fp.release_speed END) as brk_velo,
               AVG(CASE WHEN fp.pitch_type IN ('CH','FS') THEN fp.release_speed END) as off_velo,
               COUNT(CASE WHEN fp.pitch_type IN ('FF','SI','FC') THEN 1 END)::float / COUNT(*) as fb_pct,
               COUNT(CASE WHEN fp.pitch_type IN ('SL','ST','SV') THEN 1 END)::float / COUNT(*) as slider_pct,
               COUNT(CASE WHEN fp.pitch_type IN ('CU','KC') THEN 1 END)::float / COUNT(*) as curve_pct,
               COUNT(CASE WHEN fp.pitch_type IN ('CH','FS') THEN 1 END)::float / COUNT(*) as ch_pct,
               COUNT(CASE WHEN fp.pitch_type IN ('SL','ST','SV','CU','KC') THEN 1 END)::float / COUNT(*) as brk_pct,
               COUNT(DISTINCT CASE WHEN fp.pitch_type IS NOT NULL THEN fp.pitch_type END) as n_pitch_types,
               AVG(sps.release_extension) as avg_extension,
               STDDEV(sps.release_extension) as extension_sd,
               AVG(sps.release_pos_x) as avg_rel_x,
               AVG(sps.release_pos_z) as avg_rel_z,
               STDDEV(sps.release_pos_x) as rel_x_sd,
               STDDEV(sps.release_pos_z) as rel_z_sd,
               AVG(fp.pfx_x) as avg_pfx_x,
               AVG(fp.pfx_z) as avg_pfx_z,
               AVG(fp.release_spin_rate) as avg_spin,
               COUNT(*) as total_pitches,
               COUNT(DISTINCT fp.game_pk) as games
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        LEFT JOIN production.sat_pitch_shape sps ON fp.pitch_id = sps.pitch_id
        WHERE dg.game_type = 'R'
          AND dg.season BETWEEN 2019 AND 2024
          AND fp.release_speed IS NOT NULL
          AND CAST(fp.release_speed AS TEXT) != 'NaN'
        GROUP BY fp.pitcher_id, dg.season
        HAVING COUNT(*) >= 500
    """, {})

    logger.info("Pitcher-seasons with 500+ pitches: %d (%d unique pitchers)",
                len(bio), bio["pitcher_id"].nunique())

    # Arm angle proxy
    bio["arm_angle_proxy"] = np.degrees(np.arctan2(
        bio["avg_rel_z"] - 5.0, np.abs(bio["avg_rel_x"])
    ))

    # YoY deltas
    bio = bio.sort_values(["pitcher_id", "season"])
    for col in ["avg_velo", "fb_velo", "brk_velo", "avg_extension", "avg_spin",
                "slider_pct", "brk_pct", "avg_rel_x", "avg_rel_z"]:
        bio[f"{col}_delta"] = bio.groupby("pitcher_id")[col].diff()

    return bio


def main() -> None:
    arm_stints = build_injury_outcomes()
    bio = build_bio_features()

    # Merge: season N mechanics -> season N+1 injury
    arm_stints["predict_season"] = arm_stints["season"] - 1
    merged = bio.merge(
        arm_stints[["player_id", "predict_season", "arm_il_stints", "arm_il_days"]].rename(
            columns={"player_id": "pitcher_id", "predict_season": "season"}
        ),
        on=["pitcher_id", "season"],
        how="left",
    )
    merged["injured_next"] = (merged["arm_il_stints"].fillna(0) > 0).astype(int)

    # Prior injury history
    all_arm = arm_stints.sort_values(["player_id", "season"])
    all_arm["cum_stints"] = all_arm.groupby("player_id")["arm_il_stints"].cumsum()
    all_arm["prior_arm"] = all_arm.groupby("player_id")["cum_stints"].shift(fill_value=0)
    all_arm["any_prior_arm"] = (all_arm["prior_arm"] > 0).astype(int)

    merged = merged.merge(
        all_arm[["player_id", "season", "any_prior_arm"]].rename(
            columns={"player_id": "pitcher_id"}
        ),
        on=["pitcher_id", "season"],
        how="left",
    )
    merged["any_prior_arm"] = merged["any_prior_arm"].fillna(0)

    # Filter to seasons where we can observe outcome
    analysis = merged[merged["season"] <= 2023].copy()
    logger.info("Analysis: %d pitcher-seasons, injury rate %.1f%% (%d injuries)",
                len(analysis), analysis["injured_next"].mean() * 100, analysis["injured_next"].sum())

    # ============================================================
    # CORRELATIONS: each feature vs next-season arm injury
    # ============================================================
    print("\n" + "=" * 65)
    print("  FEATURE CORRELATIONS WITH NEXT-SEASON ARM INJURY")
    print("=" * 65)

    features = [
        # Static levels
        ("avg_velo", "Avg velocity (all)"),
        ("fb_velo", "Fastball velocity"),
        ("p95_velo", "95th pctl velocity"),
        ("brk_velo", "Breaking ball velocity"),
        ("off_velo", "Offspeed velocity"),
        ("slider_pct", "Slider usage %"),
        ("brk_pct", "Breaking ball usage %"),
        ("curve_pct", "Curveball usage %"),
        ("ch_pct", "Changeup usage %"),
        ("fb_pct", "Fastball usage %"),
        ("n_pitch_types", "Distinct pitch types"),
        ("avg_extension", "Release extension"),
        ("arm_angle_proxy", "Arm angle (proxy)"),
        ("avg_spin", "Avg spin rate"),
        ("avg_pfx_x", "Horizontal movement"),
        ("avg_pfx_z", "Vertical movement"),
        ("rel_x_sd", "Release point X variability"),
        ("rel_z_sd", "Release point Z variability"),
        ("extension_sd", "Extension variability"),
        ("total_pitches", "Total pitches (workload)"),
        ("any_prior_arm", "Prior arm injury history"),
        # Deltas (within-pitcher changes)
        ("avg_velo_delta", "Velocity change YoY"),
        ("fb_velo_delta", "FB velocity change YoY"),
        ("brk_velo_delta", "Breaking velo change YoY"),
        ("avg_extension_delta", "Extension change YoY"),
        ("avg_spin_delta", "Spin rate change YoY"),
        ("slider_pct_delta", "Slider usage change YoY"),
        ("brk_pct_delta", "Breaking usage change YoY"),
    ]

    results = []
    for col, label in features:
        if col not in analysis.columns:
            continue
        valid = analysis[[col, "injured_next"]].dropna()
        if len(valid) < 50:
            continue
        r = np.corrcoef(valid[col], valid["injured_next"])[0, 1]
        # Group comparison
        inj = valid[valid["injured_next"] == 1][col]
        healthy = valid[valid["injured_next"] == 0][col]
        diff = inj.mean() - healthy.mean()
        results.append({"feature": col, "label": label, "r": r, "diff": diff,
                         "mean_inj": inj.mean(), "mean_healthy": healthy.mean(), "n": len(valid)})

    results_df = pd.DataFrame(results).sort_values("r", key=abs, ascending=False)

    print(f"\n{'Feature':<30s} {'r':>7s} {'Injured':>9s} {'Healthy':>9s} {'Diff':>8s}")
    print("-" * 65)
    for _, r in results_df.iterrows():
        flag = " ***" if abs(r["r"]) >= 0.05 else " *" if abs(r["r"]) >= 0.03 else ""
        print(f"  {r['label']:<28s} {r['r']:+.4f} {r['mean_inj']:9.3f} {r['mean_healthy']:9.3f} {r['diff']:+8.3f}{flag}")

    # ============================================================
    # BUCKETED ANALYSIS: quartile injury rates
    # ============================================================
    print("\n" + "=" * 65)
    print("  BUCKETED ANALYSIS (Q1=lowest, Q4=highest)")
    print("=" * 65)

    bucket_features = [
        "fb_velo", "brk_velo", "slider_pct", "brk_pct", "ch_pct",
        "n_pitch_types", "avg_extension", "arm_angle_proxy",
        "fb_velo_delta", "avg_velo_delta", "any_prior_arm",
    ]

    for col in bucket_features:
        if col not in analysis.columns:
            continue
        valid = analysis[[col, "injured_next"]].dropna()
        if len(valid) < 100:
            continue

        if col == "any_prior_arm":
            # Binary feature
            for val, label in [(0, "No prior"), (1, "Prior arm IL")]:
                sub = valid[valid[col] == val]
                rate = sub["injured_next"].mean()
                print(f"  {col} = {label}: {rate:.1%} injury rate ({len(sub)} pitchers)")
            print()
            continue

        try:
            valid["q"] = pd.qcut(valid[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            valid["q"] = pd.qcut(valid[col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        qtable = valid.groupby("q")["injured_next"].agg(["mean", "count"]).reset_index()
        q1_rate = qtable[qtable["q"] == "Q1"]["mean"].values[0] if "Q1" in qtable["q"].values else 0
        q4_rate = qtable[qtable["q"] == "Q4"]["mean"].values[0] if "Q4" in qtable["q"].values else 0
        lift = q4_rate / q1_rate if q1_rate > 0 else float("inf")

        ranges = valid.groupby("q")[col].agg(["min", "max"]).reset_index()
        print(f"\n  {col}:")
        for _, qr in qtable.iterrows():
            rng = ranges[ranges["q"] == qr["q"]]
            lo = rng["min"].values[0] if not rng.empty else 0
            hi = rng["max"].values[0] if not rng.empty else 0
            bar = "#" * int(qr["mean"] * 100)
            print(f"    {qr['q']}: {qr['mean']:5.1%} ({qr['count']:4.0f}p)  [{lo:.1f}-{hi:.1f}]  {bar}")
        print(f"    Q4/Q1 lift: {lift:.2f}x")

    # Save
    analysis.to_parquet("data/cached/_injury_analysis.parquet", index=False)
    results_df.to_csv("outputs/injury_feature_correlations.csv", index=False)
    logger.info("Saved analysis data and correlations")


if __name__ == "__main__":
    main()
