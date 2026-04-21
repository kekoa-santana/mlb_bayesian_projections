"""Analyze whether scouting grade composites predict next-year stats
better than the raw stats they're projecting.

If grade_X in year N -> stat_Y in year N+1 has higher R² than
raw stat_Y in year N -> stat_Y in year N+1, then the grade composite
contains more signal about true talent and should inform the Bayesian prior.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from src.data.db import read_sql
from src.data.feature_eng import get_cached_hitter_observed_profile
from src.data.queries import get_hitter_traditional_stats
from src.models.scouting_grades import _compute_hitter_pop_stats, _composite_z

seasons = [2019, 2020, 2021, 2022, 2023, 2024]
all_pairs = []

for season in seasons:
    next_season = season + 1
    print(f"Processing {season} -> {next_season}...", flush=True)

    # Year N stats
    batting_n = read_sql(
        f"SELECT batter_id, pa, k_pct, bb_pct, xba, barrel_pct, hard_hit_pct, woba "
        f"FROM production.fact_batting_advanced "
        f"WHERE season = {season} AND pa >= 200", {}
    )

    # Year N+1 stats (targets)
    batting_n1 = read_sql(
        f"SELECT batter_id, k_pct AS k_pct_n1, bb_pct AS bb_pct_n1, "
        f"woba AS woba_n1, barrel_pct AS barrel_pct_n1 "
        f"FROM production.fact_batting_advanced "
        f"WHERE season = {next_season} AND pa >= 200", {}
    )

    if batting_n.empty or batting_n1.empty:
        continue

    # Observed profiles
    try:
        obs = get_cached_hitter_observed_profile(season)
    except Exception:
        continue

    trad = get_hitter_traditional_stats(season)
    trad_n1 = get_hitter_traditional_stats(next_season)

    df = batting_n.copy()
    if not obs.empty:
        for c in ["z_contact_pct", "avg_exit_velo", "whiff_rate", "chase_rate"]:
            if c in obs.columns:
                df = df.merge(obs[["batter_id", c]], on="batter_id", how="left")

    if not trad.empty and "iso" in trad.columns:
        df = df.merge(trad[["batter_id", "iso"]], on="batter_id", how="left")

    # O-contact%
    try:
        oc = read_sql(
            f"SELECT fp.batter_id, "
            f"COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83 "
            f"OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing AND NOT is_whiff)::float "
            f"/ NULLIF(COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83 "
            f"OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing), 0) AS o_contact_pct "
            f"FROM production.fact_pitch fp "
            f"JOIN production.dim_game dg ON fp.game_pk = dg.game_pk "
            f"WHERE dg.season = {season} AND dg.game_type = 'R' "
            f"GROUP BY fp.batter_id "
            f"HAVING COUNT(*) FILTER (WHERE (plate_x < -0.83 OR plate_x > 0.83 "
            f"OR plate_z < 1.5 OR plate_z > 3.5) AND is_swing) >= 50", {}
        )
        if not oc.empty:
            df = df.merge(oc, on="batter_id", how="left")
    except Exception:
        pass

    # Population stats — use single season to avoid old-data errors
    # (the 3-year pool in _compute_hitter_pop_stats can fail for 2017/2018)
    pop = {}
    for col in ["k_pct", "bb_pct", "xba", "barrel_pct", "hard_hit_pct"]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 30:
                pop[col] = (vals.mean(), vals.std())
    if not obs.empty:
        for col in ["z_contact_pct", "avg_exit_velo"]:
            if col in obs.columns:
                vals = obs[col].dropna()
                if len(vals) > 30:
                    pop[col] = (vals.mean(), vals.std())
    if "o_contact_pct" in df.columns:
        vals = df["o_contact_pct"].dropna()
        if len(vals) > 30:
            pop["o_contact_pct"] = (vals.mean(), vals.std())
    if "iso" in df.columns:
        vals = df["iso"].dropna()
        if len(vals) > 30:
            pop["iso"] = (vals.mean(), vals.std())

    # Hit composite z
    hit_specs = []
    if "k_pct" in pop:
        hit_specs.append(("k_pct", 0.30, True, *pop["k_pct"]))
    if "z_contact_pct" in pop:
        hit_specs.append(("z_contact_pct", 0.25, False, *pop["z_contact_pct"]))
    if "o_contact_pct" in pop:
        hit_specs.append(("o_contact_pct", 0.20, False, *pop["o_contact_pct"]))
    if "xba" in pop:
        hit_specs.append(("xba", 0.25, False, *pop["xba"]))
    df["hit_z"] = _composite_z(df, hit_specs)

    # Power composite z
    pow_specs = []
    if "iso" in pop:
        pow_specs.append(("iso", 0.35, False, *pop["iso"]))
    if "barrel_pct" in pop:
        pow_specs.append(("barrel_pct", 0.25, False, *pop["barrel_pct"]))
    if "hard_hit_pct" in pop:
        pow_specs.append(("hard_hit_pct", 0.20, False, *pop["hard_hit_pct"]))
    if "avg_exit_velo" in pop:
        pow_specs.append(("avg_exit_velo", 0.20, False, *pop["avg_exit_velo"]))
    df["power_z"] = _composite_z(df, pow_specs)

    # Discipline composite z
    disc_specs = []
    if "bb_pct" in pop:
        disc_specs.append(("bb_pct", 0.35, False, *pop["bb_pct"]))
    if "chase_rate" in df.columns:
        vals = df["chase_rate"].dropna()
        if len(vals) > 30:
            disc_specs.append(("chase_rate", 0.35, True, vals.mean(), vals.std()))
    df["discipline_z"] = _composite_z(df, disc_specs)

    # Merge N+1 targets
    df = df.merge(batting_n1, on="batter_id", how="inner")
    if not trad_n1.empty and "iso" in trad_n1.columns:
        df = df.merge(
            trad_n1[["batter_id", "iso"]].rename(columns={"iso": "iso_n1"}),
            on="batter_id", how="left",
        )

    if len(df) < 50:
        continue

    # Compute correlations
    comparisons = [
        ("hit_z", "k_pct_n1", "hit_z -> K%"),
        ("k_pct", "k_pct_n1", "raw K% -> K%"),
        ("power_z", "iso_n1", "power_z -> ISO"),
        ("iso", "iso_n1", "raw ISO -> ISO"),
        ("power_z", "barrel_pct_n1", "power_z -> barrel%"),
        ("barrel_pct", "barrel_pct_n1", "raw barrel% -> barrel%"),
        ("discipline_z", "bb_pct_n1", "discipline_z -> BB%"),
        ("bb_pct", "bb_pct_n1", "raw BB% -> BB%"),
        ("hit_z", "woba_n1", "hit_z -> wOBA"),
        ("k_pct", "woba_n1", "raw K% -> wOBA"),
        ("power_z", "woba_n1", "power_z -> wOBA"),
        ("iso", "woba_n1", "raw ISO -> wOBA"),
        ("discipline_z", "woba_n1", "discipline_z -> wOBA"),
        ("bb_pct", "woba_n1", "raw BB% -> wOBA"),
    ]

    for predictor, target, label in comparisons:
        if predictor in df.columns and target in df.columns:
            valid = df[[predictor, target]].dropna()
            if len(valid) >= 30:
                r, p = pearsonr(valid[predictor], valid[target])
                all_pairs.append({
                    "season": f"{season}->{next_season}",
                    "predictor": label,
                    "r": r,
                    "r_sq": r ** 2,
                    "n": len(valid),
                    "p_value": p,
                })

results = pd.DataFrame(all_pairs)

# Average across all season pairs
avg = results.groupby("predictor").agg(
    mean_r=("r", "mean"),
    mean_r_sq=("r_sq", "mean"),
    n_seasons=("season", "count"),
    avg_n=("n", "mean"),
).sort_values("mean_r_sq", ascending=False)

print()
print("YEAR-OVER-YEAR PREDICTION: Scouting Grade Composites vs Raw Stats")
print("Each row: predictor in year N -> target in year N+1 (avg across 2019-2024)")
print("=" * 90)
print(f"{'Predictor -> Target':40s}  {'Mean r':>7s}  {'Mean R²':>8s}  {'Seasons':>7s}  {'Avg N':>6s}")
print("-" * 90)

for idx, row in avg.iterrows():
    print(
        f"{idx:40s}  {row['mean_r']:+7.3f}  {row['mean_r_sq']:8.3f}"
        f"  {int(row['n_seasons']):7d}  {row['avg_n']:6.0f}"
    )

print()
print("KEY FINDINGS:")
comparisons = [
    ("hit_z -> K%", "raw K% -> K%", "Hit grade vs raw K% -> next-year K%"),
    ("power_z -> ISO", "raw ISO -> ISO", "Power grade vs raw ISO -> next-year ISO"),
    ("power_z -> barrel%", "raw barrel% -> barrel%", "Power grade vs raw barrel% -> next-year barrel%"),
    ("discipline_z -> BB%", "raw BB% -> BB%", "Discipline grade vs raw BB% -> next-year BB%"),
    ("power_z -> wOBA", "raw ISO -> wOBA", "Power grade vs raw ISO -> next-year wOBA"),
    ("hit_z -> wOBA", "raw K% -> wOBA", "Hit grade vs raw K% -> next-year wOBA"),
    ("discipline_z -> wOBA", "raw BB% -> wOBA", "Discipline grade vs raw BB% -> next-year wOBA"),
]
for grade_key, raw_key, desc in comparisons:
    if grade_key in avg.index and raw_key in avg.index:
        g_r2 = avg.loc[grade_key, "mean_r_sq"]
        r_r2 = avg.loc[raw_key, "mean_r_sq"]
        winner = "GRADE WINS" if g_r2 > r_r2 else "RAW WINS"
        delta = ((g_r2 / r_r2 - 1) * 100) if r_r2 > 0 else 0
        print(f"  {desc}:")
        print(f"    Grade R²={g_r2:.3f} vs Raw R²={r_r2:.3f} -> {winner} ({delta:+.1f}%)")
