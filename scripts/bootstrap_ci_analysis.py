"""Bootstrap CI Analysis — Test whether bootstrapped CIs from year N
contain the actual year N+1 value (calibration test).

Phase 1: Individual Statcast metrics (2018-2025)
Phase 2: Composite features / interaction terms

Good calibration = 80% CI contains the truth ~80% of the time.
"""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from src.data.db import read_sql

# ===================================================================
# Phase 1: Bootstrap on individual Statcast metrics
# ===================================================================

def get_per_game_metrics(season: int) -> pd.DataFrame:
    """Load per-game Statcast metrics for hitters."""
    df = read_sql(f"""
        WITH game_pitch AS (
            SELECT
                fp.batter_id,
                fp.game_pk,
                COUNT(*) FILTER (WHERE fp.is_bip) AS bip,
                COUNT(*) FILTER (WHERE fp.is_swing) AS swings,
                COUNT(*) FILTER (WHERE fp.is_whiff) AS whiffs,
                COUNT(*) FILTER (
                    WHERE (fp.plate_x < -0.83 OR fp.plate_x > 0.83
                        OR fp.plate_z < 1.5 OR fp.plate_z > 3.5)
                    AND fp.is_swing
                ) AS oz_swings,
                COUNT(*) FILTER (
                    WHERE (fp.plate_x < -0.83 OR fp.plate_x > 0.83
                        OR fp.plate_z < 1.5 OR fp.plate_z > 3.5)
                    AND fp.is_swing AND NOT fp.is_whiff
                ) AS oz_contacts
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season = {season} AND dg.game_type = 'R'
            GROUP BY fp.batter_id, fp.game_pk
            HAVING COUNT(*) >= 3
        ),
        game_bb AS (
            SELECT
                pa.batter_id,
                pa.game_pk,
                AVG(sb.launch_speed) AS avg_exit_velo,
                AVG(CASE WHEN sb.hard_hit THEN 1.0 ELSE 0.0 END) AS hard_hit_pct,
                AVG(CASE WHEN sb.launch_speed >= 98 AND sb.launch_angle BETWEEN 26 AND 30
                    THEN 1.0 ELSE 0.0 END) AS barrel_pct,
                AVG(CASE WHEN sb.xwoba::text != 'NaN' THEN sb.xwoba::float ELSE NULL END) AS xwoba,
                AVG(CASE WHEN sb.xba::text != 'NaN' THEN sb.xba::float ELSE NULL END) AS xba
            FROM production.sat_batted_balls sb
            JOIN production.fact_pa pa ON sb.pa_id = pa.pa_id
            JOIN production.dim_game dg ON pa.game_pk = dg.game_pk
            WHERE dg.season = {season} AND dg.game_type = 'R'
            GROUP BY pa.batter_id, pa.game_pk
        )
        SELECT
            g.batter_id,
            g.game_pk,
            g.bip,
            g.swings,
            g.whiffs,
            g.oz_swings,
            g.oz_contacts,
            CASE WHEN g.swings > 0 THEN g.whiffs::float / g.swings ELSE NULL END AS whiff_rate,
            CASE WHEN g.oz_swings > 0 THEN g.oz_contacts::float / g.oz_swings ELSE NULL END AS o_contact_pct,
            b.avg_exit_velo,
            b.hard_hit_pct,
            b.barrel_pct,
            b.xwoba,
            b.xba
        FROM game_pitch g
        LEFT JOIN game_bb b ON g.batter_id = b.batter_id AND g.game_pk = b.game_pk
    """, {})
    df["season"] = season
    return df


def get_pitcher_per_game_metrics(season: int) -> pd.DataFrame:
    """Load per-game Statcast metrics for pitchers."""
    df = read_sql(f"""
        SELECT
            fp.pitcher_id,
            fp.game_pk,
            COUNT(*) AS pitches,
            COUNT(*) FILTER (WHERE fp.is_swing) AS swings,
            COUNT(*) FILTER (WHERE fp.is_whiff) AS whiffs,
            COUNT(*) FILTER (WHERE fp.is_called_strike OR fp.is_whiff) AS csw,
            CASE WHEN COUNT(*) FILTER (WHERE fp.is_swing) > 0
                THEN COUNT(*) FILTER (WHERE fp.is_whiff)::float /
                     COUNT(*) FILTER (WHERE fp.is_swing)
                ELSE NULL END AS swstr_pct,
            CASE WHEN COUNT(*) > 0
                THEN (COUNT(*) FILTER (WHERE fp.is_called_strike OR fp.is_whiff))::float /
                     COUNT(*)
                ELSE NULL END AS csw_pct
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = {season} AND dg.game_type = 'R'
        GROUP BY fp.pitcher_id, fp.game_pk
        HAVING COUNT(*) >= 10
    """, {})
    df["season"] = season
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                 ci: float = 0.80) -> tuple[float, float, float]:
    """Bootstrap a metric and return (mean, lo, hi) for given CI."""
    if len(values) < 10:
        m = np.nanmean(values)
        return m, m, m
    boot_means = np.array([
        np.nanmean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.nanmean(values), lo, hi


def bootstrap_season(game_df: pd.DataFrame, metric: str,
                     min_games: int = 50, n_boot: int = 1000,
                     ci: float = 0.80) -> pd.DataFrame:
    """Bootstrap per-player season CI for a metric."""
    results = []
    for pid, g in game_df.groupby(game_df.columns[0]):  # batter_id or pitcher_id
        vals = g[metric].dropna().values
        if len(vals) < min_games:
            continue
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, ci=ci)
        results.append({
            "player_id": pid,
            "metric": metric,
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_games": len(vals),
        })
    return pd.DataFrame(results)


# ===================================================================
# Run Phase 1: Individual metrics
# ===================================================================

print("=" * 80)
print("PHASE 1: Bootstrap CIs on Individual Statcast Metrics")
print("Testing: does the 80% CI from year N contain year N+1 actual?")
print("=" * 80)

np.random.seed(42)

hitter_metrics = ["whiff_rate", "barrel_pct", "hard_hit_pct",
                  "avg_exit_velo", "xba", "o_contact_pct"]
pitcher_metrics = ["swstr_pct", "csw_pct"]

# Cache game-level data
print("\nLoading game-level data...")
hitter_games = {}
pitcher_games = {}
for s in range(2018, 2026):
    print(f"  {s}...", end=" ", flush=True)
    hitter_games[s] = get_per_game_metrics(s)
    pitcher_games[s] = get_pitcher_per_game_metrics(s)
    print(f"({len(hitter_games[s])} hitter games, {len(pitcher_games[s])} pitcher games)")

# Calibration test: for each year pair (N, N+1)
calibration_results = []

for year_n in range(2018, 2025):
    year_n1 = year_n + 1

    # Hitter metrics
    for metric in hitter_metrics:
        if metric not in hitter_games[year_n].columns:
            continue

        # Bootstrap CIs for year N
        ci_df = bootstrap_season(hitter_games[year_n], metric, min_games=40, n_boot=500)
        if ci_df.empty:
            continue

        # Get year N+1 actuals (season-level mean)
        actual_n1 = (
            hitter_games[year_n1]
            .groupby("batter_id")[metric]
            .agg(["mean", "count"])
            .rename(columns={"mean": "actual_n1", "count": "n_games_n1"})
            .reset_index()
        )
        actual_n1 = actual_n1[actual_n1["n_games_n1"] >= 40]

        # Join and check containment
        merged = ci_df.merge(
            actual_n1.rename(columns={"batter_id": "player_id"}),
            on="player_id", how="inner",
        )
        if len(merged) < 20:
            continue

        contained = (
            (merged["actual_n1"] >= merged["ci_lo"])
            & (merged["actual_n1"] <= merged["ci_hi"])
        )
        containment_rate = contained.mean()
        ci_width = (merged["ci_hi"] - merged["ci_lo"]).mean()

        calibration_results.append({
            "year": f"{year_n}->{year_n1}",
            "metric": metric,
            "type": "hitter",
            "phase": "individual",
            "containment_80ci": containment_rate,
            "avg_ci_width": ci_width,
            "n_players": len(merged),
        })

    # Pitcher metrics
    for metric in pitcher_metrics:
        if metric not in pitcher_games[year_n].columns:
            continue

        ci_df = bootstrap_season(pitcher_games[year_n], metric, min_games=20, n_boot=500)
        if ci_df.empty:
            continue

        actual_n1 = (
            pitcher_games[year_n1]
            .groupby("pitcher_id")[metric]
            .agg(["mean", "count"])
            .rename(columns={"mean": "actual_n1", "count": "n_games_n1"})
            .reset_index()
        )
        actual_n1 = actual_n1[actual_n1["n_games_n1"] >= 20]

        merged = ci_df.merge(
            actual_n1.rename(columns={"pitcher_id": "player_id"}),
            on="player_id", how="inner",
        )
        if len(merged) < 20:
            continue

        contained = (
            (merged["actual_n1"] >= merged["ci_lo"])
            & (merged["actual_n1"] <= merged["ci_hi"])
        )

        calibration_results.append({
            "year": f"{year_n}->{year_n1}",
            "metric": metric,
            "type": "pitcher",
            "phase": "individual",
            "containment_80ci": contained.mean(),
            "avg_ci_width": (merged["ci_hi"] - merged["ci_lo"]).mean(),
            "n_players": len(merged),
        })

results_df = pd.DataFrame(calibration_results)

# Average across years
print("\nPHASE 1 RESULTS: Individual Metric Bootstrap Calibration")
print("-" * 80)
print(f"{'Metric':20s} {'Type':8s} {'Containment':>12s} {'Target':>8s} {'CI Width':>10s} {'N Years':>8s}")
print("-" * 80)

avg = results_df.groupby(["metric", "type"]).agg(
    mean_containment=("containment_80ci", "mean"),
    mean_ci_width=("avg_ci_width", "mean"),
    n_years=("year", "count"),
).sort_values("mean_containment", ascending=False)

for (metric, ptype), row in avg.iterrows():
    cal_status = "GOOD" if 0.70 <= row["mean_containment"] <= 0.90 else (
        "WIDE" if row["mean_containment"] > 0.90 else "NARROW"
    )
    print(f"{metric:20s} {ptype:8s} {row['mean_containment']:11.1%}  {'80%':>8s} {row['mean_ci_width']:10.4f} {int(row['n_years']):>8d}  {cal_status}")


# ===================================================================
# Phase 2: Composite / Interaction Features
# ===================================================================

print()
print("=" * 80)
print("PHASE 2: Bootstrap CIs on Composite Features")
print("Testing: do composites produce tighter, better-calibrated CIs?")
print("=" * 80)

composite_results = []

for year_n in range(2018, 2025):
    year_n1 = year_n + 1
    gn = hitter_games[year_n].copy()
    gn1 = hitter_games[year_n1].copy()

    # Compute composites per game
    # Power composite (additive weighted z within game data)
    for df in [gn, gn1]:
        # Power: ISO not available per-game, use barrel% * hard_hit% * exit_velo
        if all(c in df.columns for c in ["barrel_pct", "hard_hit_pct", "avg_exit_velo"]):
            # Standardize within season
            for c in ["barrel_pct", "hard_hit_pct", "avg_exit_velo"]:
                mu, sd = df[c].mean(), df[c].std()
                df[f"{c}_z"] = (df[c] - mu) / sd if sd > 1e-9 else 0
            df["power_composite"] = (
                0.40 * df["barrel_pct_z"].fillna(0)
                + 0.30 * df["hard_hit_pct_z"].fillna(0)
                + 0.30 * df["avg_exit_velo_z"].fillna(0)
            )

        # Hit composite: xBA + whiff_rate(inv)
        if all(c in df.columns for c in ["xba", "whiff_rate"]):
            for c in ["xba", "whiff_rate"]:
                mu, sd = df[c].mean(), df[c].std()
                df[f"{c}_z"] = (df[c] - mu) / sd if sd > 1e-9 else 0
            df["hit_composite"] = (
                0.50 * df["xba_z"].fillna(0)
                + 0.50 * (-df["whiff_rate_z"].fillna(0))  # inverted
            )

        # Power interaction (multiplicative): barrel% * exit_velo
        if "barrel_pct" in df.columns and "avg_exit_velo" in df.columns:
            df["power_interaction"] = df["barrel_pct"].fillna(0) * df["avg_exit_velo"].fillna(85) / 100

    composites = ["power_composite", "hit_composite", "power_interaction"]
    for metric in composites:
        if metric not in gn.columns:
            continue

        ci_df = bootstrap_season(gn, metric, min_games=40, n_boot=500)
        if ci_df.empty:
            continue

        actual_n1 = (
            gn1.groupby("batter_id")[metric]
            .agg(["mean", "count"])
            .rename(columns={"mean": "actual_n1", "count": "n_games_n1"})
            .reset_index()
        )
        actual_n1 = actual_n1[actual_n1["n_games_n1"] >= 40]

        merged = ci_df.merge(
            actual_n1.rename(columns={"batter_id": "player_id"}),
            on="player_id", how="inner",
        )
        if len(merged) < 20:
            continue

        contained = (
            (merged["actual_n1"] >= merged["ci_lo"])
            & (merged["actual_n1"] <= merged["ci_hi"])
        )

        composite_results.append({
            "year": f"{year_n}->{year_n1}",
            "metric": metric,
            "type": "hitter",
            "phase": "composite",
            "containment_80ci": contained.mean(),
            "avg_ci_width": (merged["ci_hi"] - merged["ci_lo"]).mean(),
            "n_players": len(merged),
        })

comp_df = pd.DataFrame(composite_results)

print("\nPHASE 2 RESULTS: Composite Feature Bootstrap Calibration")
print("-" * 80)
print(f"{'Metric':25s} {'Containment':>12s} {'Target':>8s} {'CI Width':>10s} {'N Years':>8s}")
print("-" * 80)

if not comp_df.empty:
    avg2 = comp_df.groupby("metric").agg(
        mean_containment=("containment_80ci", "mean"),
        mean_ci_width=("avg_ci_width", "mean"),
        n_years=("year", "count"),
    ).sort_values("mean_containment", ascending=False)

    for metric, row in avg2.iterrows():
        cal_status = "GOOD" if 0.70 <= row["mean_containment"] <= 0.90 else (
            "WIDE" if row["mean_containment"] > 0.90 else "NARROW"
        )
        print(f"{metric:25s} {row['mean_containment']:11.1%}  {'80%':>8s} {row['mean_ci_width']:10.4f} {int(row['n_years']):>8d}  {cal_status}")

# ===================================================================
# Summary comparison
# ===================================================================

print()
print("=" * 80)
print("COMPARISON: Individual vs Composite Calibration")
print("=" * 80)

all_results = pd.concat([results_df, comp_df], ignore_index=True)
summary = all_results.groupby(["metric", "phase"]).agg(
    containment=("containment_80ci", "mean"),
    ci_width=("avg_ci_width", "mean"),
).reset_index()

# Compare power composite vs individual barrel%
for individual, composite in [
    ("barrel_pct", "power_composite"),
    ("barrel_pct", "power_interaction"),
    ("xba", "hit_composite"),
]:
    ind_row = summary[summary["metric"] == individual]
    comp_row = summary[summary["metric"] == composite]
    if not ind_row.empty and not comp_row.empty:
        ind_cal = ind_row["containment"].iloc[0]
        comp_cal = comp_row["containment"].iloc[0]
        ind_width = ind_row["ci_width"].iloc[0]
        comp_width = comp_row["ci_width"].iloc[0]
        print(f"\n{individual} vs {composite}:")
        print(f"  Calibration: {ind_cal:.1%} vs {comp_cal:.1%} (target: 80%)")
        print(f"  CI Width:    {ind_width:.4f} vs {comp_width:.4f}")
        better = "COMPOSITE" if abs(comp_cal - 0.80) < abs(ind_cal - 0.80) else "INDIVIDUAL"
        print(f"  Winner: {better}")
