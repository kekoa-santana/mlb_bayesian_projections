"""Year-over-Year Stability of Scouting Tool Grades.

Compute historical scouting grades for every player-season (2018-2025),
then measure YoY correlation for each tool grade. This tells us which
grades are projectable (high r) vs volatile (low r).

Also tests: is the grade composite more stable than its individual components?
"""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.data.db import read_sql
from src.data.feature_eng import (
    get_cached_hitter_observed_profile,
    get_cached_sprint_speed,
)
from src.data.queries import get_hitter_traditional_stats
from src.models.scouting_grades import (
    _compute_hitter_pop_stats,
    _compute_pitcher_pop_stats,
    _grade_hit, _grade_power, _grade_speed, _grade_fielding,
    _grade_discipline, _grade_stuff, _grade_command, _grade_durability,
    _composite_z,
)

# ===================================================================
# Compute historical grades for hitters
# ===================================================================

print("Computing historical hitter grades (2018-2025)...")
print("=" * 80)

hitter_grades_all = []

for season in range(2018, 2026):
    print(f"  {season}...", end=" ", flush=True)

    # Load batting advanced
    batting = read_sql(f"""
        SELECT batter_id, pa, k_pct, bb_pct, xba, barrel_pct,
               hard_hit_pct, woba, wrc_plus, xwoba
        FROM production.fact_batting_advanced
        WHERE season = {season} AND pa >= 200
    """, {})

    if batting.empty:
        print("no data")
        continue

    # Load observed profile
    try:
        obs = get_cached_hitter_observed_profile(season)
        for c in ["z_contact_pct", "avg_exit_velo", "whiff_rate", "chase_rate"]:
            if c in obs.columns:
                batting = batting.merge(obs[["batter_id", c]], on="batter_id", how="left")
    except Exception:
        pass

    # Load sprint speed
    try:
        spd = get_cached_sprint_speed(season)
        if not spd.empty and "sprint_speed" in spd.columns:
            batting = batting.merge(
                spd[["player_id", "sprint_speed"]].rename(columns={"player_id": "batter_id"}),
                on="batter_id", how="left",
            )
    except Exception:
        pass

    # Load ISO
    try:
        trad = get_hitter_traditional_stats(season)
        if not trad.empty and "iso" in trad.columns:
            batting = batting.merge(trad[["batter_id", "iso"]], on="batter_id", how="left")
    except Exception:
        pass

    # Load fielding
    try:
        oaa = read_sql(f"""
            SELECT player_id, SUM(outs_above_average) as outs_above_average
            FROM production.fact_fielding_oaa
            WHERE season = {season}
            GROUP BY player_id
        """, {})
        if not oaa.empty:
            oaa["fielding_score"] = oaa["outs_above_average"].rank(pct=True)
            batting = batting.merge(
                oaa[["player_id", "fielding_score"]].rename(columns={"player_id": "batter_id"}),
                on="batter_id", how="left",
            )
    except Exception:
        pass

    batting["fielding_score"] = batting.get("fielding_score", pd.Series(0.5, index=batting.index)).fillna(0.5)
    batting["fielding_combined"] = batting["fielding_score"]

    # O-contact%
    try:
        oc = read_sql(f"""
            SELECT fp.batter_id,
                COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                    OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing AND NOT is_whiff)::float
                / NULLIF(COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                    OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing), 0) AS o_contact_pct
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season = {season} AND dg.game_type = 'R'
            GROUP BY fp.batter_id
            HAVING COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83
                OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing) >= 50
        """, {})
        if not oc.empty:
            batting = batting.merge(oc, on="batter_id", how="left")
    except Exception:
        pass

    # Load health scores
    try:
        health = read_sql(f"""
            SELECT player_id,
                   SUM(CASE WHEN status_type LIKE '%%IL%%' THEN
                       COALESCE(duration_days, 15) ELSE 0 END) AS il_days
            FROM production.fact_player_status_timeline
            WHERE season = {season}
            GROUP BY player_id
        """, {})
        if not health.empty:
            # Convert IL days to 0-1 health score (0 days = 1.0, 60+ days = 0.2)
            health["health_score"] = (1.0 - health["il_days"].clip(0, 120) / 150).clip(0.1, 1.0)
            batting = batting.merge(
                health[["player_id", "health_score", "il_days"]].rename(
                    columns={"player_id": "batter_id"}),
                on="batter_id", how="left",
            )
    except Exception:
        pass
    batting["health_score"] = batting.get("health_score", pd.Series(0.7, index=batting.index)).fillna(0.7)

    # Projected BB rate (use observed as proxy for historical)
    batting["projected_bb_rate"] = batting["bb_pct"]

    # Compute population stats for this season
    pop = {}
    for col in ["k_pct", "bb_pct", "xba", "barrel_pct", "hard_hit_pct"]:
        if col in batting.columns:
            vals = batting[col].dropna()
            if len(vals) > 30:
                pop[col] = (vals.mean(), vals.std())
    for col in ["z_contact_pct", "avg_exit_velo"]:
        if col in batting.columns:
            vals = batting[col].dropna()
            if len(vals) > 30:
                pop[col] = (vals.mean(), vals.std())
    if "o_contact_pct" in batting.columns:
        vals = batting["o_contact_pct"].dropna()
        if len(vals) > 30:
            pop["o_contact_pct"] = (vals.mean(), vals.std())
    if "iso" in batting.columns:
        vals = batting["iso"].dropna()
        if len(vals) > 30:
            pop["iso"] = (vals.mean(), vals.std())

    # Compute grades
    grades = pd.DataFrame({"batter_id": batting["batter_id"], "season": season})
    grades["grade_hit"] = _grade_hit(batting, pop)
    grades["grade_power"] = _grade_power(batting, pop)
    grades["grade_speed"] = _grade_speed(batting, pop)
    grades["grade_fielding"] = _grade_fielding(batting)
    grades["grade_discipline"] = _grade_discipline(batting, pop)
    grades["health_score"] = batting["health_score"].values

    # Also store raw components for comparison
    for col in ["barrel_pct", "hard_hit_pct", "avg_exit_velo", "iso",
                "xba", "whiff_rate", "chase_rate", "sprint_speed",
                "z_contact_pct", "o_contact_pct", "k_pct", "bb_pct"]:
        if col in batting.columns:
            grades[col] = batting[col].values

    hitter_grades_all.append(grades)
    print(f"{len(grades)} hitters")

hitter_grades = pd.concat(hitter_grades_all, ignore_index=True)
print(f"\nTotal: {len(hitter_grades)} hitter-seasons")

# ===================================================================
# Compute historical grades for pitchers
# ===================================================================

print("\nComputing historical pitcher grades (2018-2025)...")
print("=" * 80)

pitcher_grades_all = []

for season in range(2018, 2026):
    print(f"  {season}...", end=" ", flush=True)

    pitching = read_sql(f"""
        SELECT pitcher_id, batters_faced, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, xwoba_against, barrel_pct_against
        FROM production.fact_pitching_advanced
        WHERE season = {season} AND batters_faced >= 100
    """, {})

    if pitching.empty:
        print("no data")
        continue

    # Projected rates (use observed as proxy for historical)
    pitching["projected_bb_rate"] = pitching["bb_pct"]
    pitching["projected_fip"] = 4.0  # neutral default for historical

    # Compute FIP from K/BB/HR if available
    try:
        fip_data = read_sql(f"""
            SELECT bb.pitcher_id,
                   SUM(bb.earned_runs)::float / NULLIF(SUM(bb.outs_recorded) / 3.0, 0) AS era,
                   SUM(bb.strikeouts)::float / NULLIF(SUM(bb.outs_recorded) / 3.0, 0) * 9 AS k9,
                   SUM(bb.walks)::float / NULLIF(SUM(bb.outs_recorded) / 3.0, 0) * 9 AS bb9,
                   SUM(bb.home_runs)::float / NULLIF(SUM(bb.outs_recorded) / 3.0, 0) * 9 AS hr9
            FROM staging.pitching_boxscores bb
            JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
            WHERE dg.season = {season} AND dg.game_type = 'R'
            GROUP BY bb.pitcher_id
            HAVING SUM(bb.outs_recorded) >= 30
        """, {})
        if not fip_data.empty:
            fip_data["projected_fip"] = (
                (13 * fip_data["hr9"] + 3 * fip_data["bb9"] - 2 * fip_data["k9"]) / 3 + 3.2
            ).clip(1.5, 8.0)
            pitching = pitching.merge(
                fip_data[["pitcher_id", "projected_fip"]],
                on="pitcher_id", how="left", suffixes=("_old", ""),
            )
            if "projected_fip_old" in pitching.columns:
                pitching["projected_fip"] = pitching["projected_fip"].fillna(
                    pitching["projected_fip_old"]
                )
                pitching.drop(columns=["projected_fip_old"], inplace=True)
    except Exception:
        pass

    # Health
    try:
        health = read_sql(f"""
            SELECT player_id,
                   SUM(CASE WHEN status_type LIKE '%%IL%%' THEN
                       COALESCE(duration_days, 15) ELSE 0 END) AS il_days
            FROM production.fact_player_status_timeline
            WHERE season = {season}
            GROUP BY player_id
        """, {})
        if not health.empty:
            health["health_score"] = (1.0 - health["il_days"].clip(0, 120) / 150).clip(0.1, 1.0)
            pitching = pitching.merge(
                health[["player_id", "health_score"]].rename(
                    columns={"player_id": "pitcher_id"}),
                on="pitcher_id", how="left",
            )
    except Exception:
        pass
    pitching["health_score"] = pitching.get("health_score", pd.Series(0.7, index=pitching.index)).fillna(0.7)

    # Determine role by BF threshold
    pitching["role"] = np.where(pitching["batters_faced"] >= 300, "SP", "RP")

    # Population stats (combined for simplicity — the grade function handles role split)
    pop = {}
    for col in ["k_pct", "bb_pct", "swstr_pct", "csw_pct",
                "zone_pct", "chase_pct", "xwoba_against", "barrel_pct_against"]:
        if col in pitching.columns:
            vals = pitching[col].dropna()
            if len(vals) > 30:
                pop[col] = (vals.mean(), vals.std())

    grades = pd.DataFrame({"pitcher_id": pitching["pitcher_id"], "season": season})
    grades["role"] = pitching["role"].values
    grades["grade_stuff"] = _grade_stuff(pitching, pop)
    grades["grade_command"] = _grade_command(pitching, pop)
    grades["grade_durability"] = _grade_durability(pitching)
    grades["health_score"] = pitching["health_score"].values

    # Raw components
    for col in ["swstr_pct", "csw_pct", "k_pct", "bb_pct", "zone_pct",
                "chase_pct", "xwoba_against", "projected_fip"]:
        if col in pitching.columns:
            grades[col] = pitching[col].values

    pitcher_grades_all.append(grades)
    print(f"{len(grades)} pitchers")

pitcher_grades = pd.concat(pitcher_grades_all, ignore_index=True)
print(f"\nTotal: {len(pitcher_grades)} pitcher-seasons")

# ===================================================================
# YoY Correlation Analysis
# ===================================================================

print()
print("=" * 80)
print("YEAR-OVER-YEAR CORRELATION: Tool Grades vs Raw Components")
print("=" * 80)


def yoy_correlation(df, id_col, metric, min_seasons=2):
    """Compute avg YoY correlation for a metric across all season pairs."""
    correlations = []
    for year_n in range(2018, 2025):
        year_n1 = year_n + 1
        n_data = df[df["season"] == year_n][[id_col, metric]].dropna()
        n1_data = df[df["season"] == year_n1][[id_col, metric]].dropna()
        merged = n_data.merge(n1_data, on=id_col, suffixes=("_n", "_n1"))
        if len(merged) >= 30:
            r, p = pearsonr(merged[f"{metric}_n"], merged[f"{metric}_n1"])
            correlations.append(r)
    if correlations:
        return np.mean(correlations), len(correlations)
    return np.nan, 0


# Hitter grades + components
print("\nHITTER GRADES (YoY correlation — higher = more projectable):")
print("-" * 70)
print(f"{'Metric':25s} {'YoY r':>8s} {'Seasons':>8s} {'Projectable?':>14s}")
print("-" * 70)

hitter_metrics = [
    # Grades
    ("grade_hit", "GRADE"),
    ("grade_power", "GRADE"),
    ("grade_speed", "GRADE"),
    ("grade_fielding", "GRADE"),
    ("grade_discipline", "GRADE"),
    ("health_score", "GRADE"),
    # Raw components
    ("xba", "component"),
    ("barrel_pct", "component"),
    ("hard_hit_pct", "component"),
    ("avg_exit_velo", "component"),
    ("iso", "component"),
    ("whiff_rate", "component"),
    ("chase_rate", "component"),
    ("z_contact_pct", "component"),
    ("o_contact_pct", "component"),
    ("sprint_speed", "component"),
    ("k_pct", "component"),
    ("bb_pct", "component"),
]

for metric, mtype in hitter_metrics:
    if metric in hitter_grades.columns:
        r, n = yoy_correlation(hitter_grades, "batter_id", metric)
        if not np.isnan(r):
            proj = "HIGH" if r >= 0.70 else "MODERATE" if r >= 0.50 else "LOW"
            marker = "***" if mtype == "GRADE" else "   "
            print(f"{marker} {metric:22s} {r:8.3f} {n:8d} {proj:>14s}")

# Pitcher grades + components
print()
print("PITCHER GRADES (YoY correlation):")
print("-" * 70)
print(f"{'Metric':25s} {'YoY r':>8s} {'Seasons':>8s} {'Projectable?':>14s}")
print("-" * 70)

pitcher_metrics = [
    ("grade_stuff", "GRADE"),
    ("grade_command", "GRADE"),
    ("grade_durability", "GRADE"),
    ("health_score", "GRADE"),
    ("swstr_pct", "component"),
    ("csw_pct", "component"),
    ("k_pct", "component"),
    ("bb_pct", "component"),
    ("zone_pct", "component"),
    ("chase_pct", "component"),
    ("xwoba_against", "component"),
    ("projected_fip", "component"),
]

for metric, mtype in pitcher_metrics:
    if metric in pitcher_grades.columns:
        r, n = yoy_correlation(pitcher_grades, "pitcher_id", metric)
        if not np.isnan(r):
            proj = "HIGH" if r >= 0.70 else "MODERATE" if r >= 0.50 else "LOW"
            marker = "***" if mtype == "GRADE" else "   "
            print(f"{marker} {metric:22s} {r:8.3f} {n:8d} {proj:>14s}")

# Compare grade vs best component
print()
print("=" * 80)
print("KEY QUESTION: Are grades more stable than their best component?")
print("=" * 80)

comparisons = [
    ("grade_hit", ["xba", "whiff_rate", "z_contact_pct", "o_contact_pct", "k_pct"], "hitter"),
    ("grade_power", ["barrel_pct", "hard_hit_pct", "avg_exit_velo", "iso"], "hitter"),
    ("grade_speed", ["sprint_speed"], "hitter"),
    ("grade_discipline", ["bb_pct", "chase_rate"], "hitter"),
    ("grade_fielding", [], "hitter"),
    ("grade_stuff", ["swstr_pct", "csw_pct", "xwoba_against"], "pitcher"),
    ("grade_command", ["bb_pct", "zone_pct", "chase_pct"], "pitcher"),
]

for grade, components, ptype in comparisons:
    df = hitter_grades if ptype == "hitter" else pitcher_grades
    id_col = "batter_id" if ptype == "hitter" else "pitcher_id"

    if grade not in df.columns:
        continue

    grade_r, _ = yoy_correlation(df, id_col, grade)
    best_comp_r = 0
    best_comp_name = "none"
    for comp in components:
        if comp in df.columns:
            comp_r, _ = yoy_correlation(df, id_col, comp)
            if not np.isnan(comp_r) and comp_r > best_comp_r:
                best_comp_r = comp_r
                best_comp_name = comp

    if not np.isnan(grade_r):
        winner = "GRADE" if grade_r > best_comp_r else "COMPONENT"
        delta = ((grade_r / best_comp_r - 1) * 100) if best_comp_r > 0 else 0
        print(f"  {grade:20s}: r={grade_r:.3f} vs best component ({best_comp_name}: r={best_comp_r:.3f}) -> {winner} ({delta:+.1f}%)")
