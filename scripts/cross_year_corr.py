"""Cross-year predictive correlations analysis."""
import pandas as pd
import numpy as np
from itertools import product

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

h = pd.read_parquet('data/cached/hitter_full_stats.parquet')
p = pd.read_parquet('data/cached/pitcher_full_stats.parquet')

hf = h[h['pa'] >= 200].copy()
pf = p[p['batters_faced'] >= 200].copy()

year_pairs = [(y, y+1) for y in range(2019, 2025)]


def compute_cross_year_corr(df, id_col, predictors, outcomes, year_pairs, min_pairs=30):
    results = {}
    counts = {}
    for pred, out in product(predictors, outcomes):
        rs = []
        for y_n, y_n1 in year_pairs:
            dn = df[df['season'] == y_n][[id_col, pred]].dropna().rename(columns={pred: 'pred_val'})
            dn1 = df[df['season'] == y_n1][[id_col, out]].dropna().rename(columns={out: 'out_val'})
            merged = dn.merge(dn1, on=id_col)
            if len(merged) < min_pairs:
                continue
            r = merged['pred_val'].corr(merged['out_val'])
            if not np.isnan(r):
                rs.append(r)
        if rs:
            results[(pred, out)] = np.mean(rs)
            counts[(pred, out)] = len(rs)
    return results, counts


def self_corr(df, id_col, stats, year_pairs, min_pairs=30):
    results = {}
    for stat in stats:
        rs = []
        for y_n, y_n1 in year_pairs:
            dn = df[df['season'] == y_n][[id_col, stat]].dropna()
            dn1 = df[df['season'] == y_n1][[id_col, stat]].dropna()
            merged = dn.merge(dn1, on=id_col, suffixes=('_n', '_n1'))
            if len(merged) >= min_pairs:
                r = merged[f'{stat}_n'].corr(merged[f'{stat}_n1'])
                if not np.isnan(r):
                    rs.append(r)
        if rs:
            results[stat] = np.mean(rs)
    return results


# ============================================================
# HITTER
# ============================================================
print("=" * 80)
print("HITTER CROSS-YEAR PREDICTIVE CORRELATIONS (>= 200 PA both years)")
print("=" * 80)

h_predictors = [c for c in [
    'whiff_rate', 'o_swing_pct', 'z_contact_pct', 'o_contact_pct', 'contact_rate',
    'avg_exit_velo', 'hard_hit_pct', 'barrel_pct', 'sweet_spot_pct',
    'k_rate', 'bb_rate', 'hr_rate', 'swing_pct', 'z_swing_pct', 'csw_pct',
    'gb_pct', 'fb_pct', 'xwoba_avg', 'avg_launch_angle', 'la_std',
    'bip_pct', 'foul_pct', 'zone_pct', 'pitches_per_pa',
    'first_pitch_swing_pct', 'max_exit_velo',
    'barrel_pct_bb', 'hard_hit_pct_bb', 'sweet_spot_pct_bb', 'xwoba_avg_bb'
] if c in hf.columns]

h_outcomes = [c for c in [
    'k_rate', 'bb_rate', 'hr_rate', 'xwoba_avg', 'barrel_pct', 'hard_hit_pct',
    'xwoba_avg_bb', 'barrel_pct_bb', 'hard_hit_pct_bb'
] if c in hf.columns]

h_results, h_counts = compute_cross_year_corr(hf, 'batter_id', h_predictors, h_outcomes, year_pairs)

h_matrix = pd.DataFrame(index=h_predictors, columns=h_outcomes, dtype=float)
for (pred, out), r in h_results.items():
    h_matrix.loc[pred, out] = r

print("\n--- FULL CORRELATION MATRIX (Year N predictor -> Year N+1 outcome) ---")
print(h_matrix.to_string())

print("\n\n--- TOP 5 PREDICTORS FOR EACH HITTER OUTCOME ---")
for out in h_outcomes:
    col = h_matrix[out].dropna().sort_values(key=abs, ascending=False)
    print(f"\n  {out} (Year N+1):")
    for i, (pred, r) in enumerate(col.head(5).items()):
        n_pairs = h_counts.get((pred, out), 0)
        print(f"    {i+1}. {pred:25s} r = {r:+.3f}  ({n_pairs} year-pairs)")

# ============================================================
# PITCHER
# ============================================================
print("\n\n" + "=" * 80)
print("PITCHER CROSS-YEAR PREDICTIVE CORRELATIONS (>= 200 BF both years)")
print("=" * 80)

p_predictors = [c for c in [
    'avg_velo', 'whiff_rate', 'k_rate', 'bb_rate', 'csw_pct',
    'gb_pct_against', 'fb_pct_against', 'hr_per_bf',
    'contact_rate_against', 'o_swing_pct', 'o_contact_pct', 'z_contact_pct',
    'swing_pct_against', 'z_swing_pct', 'zone_pct',
    'avg_exit_velo_against', 'barrel_pct_against', 'hard_hit_pct_against',
    'xwoba_against', 'avg_la_against', 'foul_pct',
    'first_pitch_swing_pct', 'k_bb_ratio', 'k_per_9', 'bb_per_9', 'hr_per_9'
] if c in pf.columns]

p_outcomes = [c for c in [
    'k_rate', 'bb_rate', 'hr_per_bf', 'xwoba_against',
    'barrel_pct_against', 'hard_hit_pct_against', 'gb_pct_against'
] if c in pf.columns]

p_results, p_counts = compute_cross_year_corr(pf, 'pitcher_id', p_predictors, p_outcomes, year_pairs)

p_matrix = pd.DataFrame(index=p_predictors, columns=p_outcomes, dtype=float)
for (pred, out), r in p_results.items():
    p_matrix.loc[pred, out] = r

print("\n--- FULL CORRELATION MATRIX (Year N predictor -> Year N+1 outcome) ---")
print(p_matrix.to_string())

print("\n\n--- TOP 5 PREDICTORS FOR EACH PITCHER OUTCOME ---")
for out in p_outcomes:
    col = p_matrix[out].dropna().sort_values(key=abs, ascending=False)
    print(f"\n  {out} (Year N+1):")
    for i, (pred, r) in enumerate(col.head(5).items()):
        n_pairs = p_counts.get((pred, out), 0)
        print(f"    {i+1}. {pred:25s} r = {r:+.3f}  ({n_pairs} year-pairs)")

# ============================================================
# SPECIFIC CROSS-STAT PREDICTIONS
# ============================================================
print("\n\n" + "=" * 80)
print("SPECIFIC CROSS-STAT PREDICTIONS (baseball analyst questions)")
print("=" * 80)

specific = [
    (hf, 'batter_id', 'xwoba_avg', 'xwoba_avg', 'Hitter: YrN xwOBA -> YrN+1 xwOBA'),
    (hf, 'batter_id', 'xwoba_avg_bb', 'xwoba_avg_bb', 'Hitter: YrN xwOBA(bb) -> YrN+1 xwOBA(bb)'),
    (hf, 'batter_id', 'avg_exit_velo', 'hr_rate', 'Hitter: YrN avg_exit_velo -> YrN+1 HR/PA'),
    (hf, 'batter_id', 'avg_exit_velo', 'xwoba_avg', 'Hitter: YrN avg_exit_velo -> YrN+1 xwOBA'),
    (hf, 'batter_id', 'o_swing_pct', 'bb_rate', 'Hitter: YrN chase_rate -> YrN+1 BB%'),
    (hf, 'batter_id', 'o_swing_pct', 'k_rate', 'Hitter: YrN chase_rate -> YrN+1 K%'),
    (hf, 'batter_id', 'whiff_rate', 'k_rate', 'Hitter: YrN whiff_rate -> YrN+1 K%'),
    (hf, 'batter_id', 'z_contact_pct', 'k_rate', 'Hitter: YrN z_contact_pct -> YrN+1 K%'),
    (hf, 'batter_id', 'hard_hit_pct', 'hr_rate', 'Hitter: YrN hard_hit_pct -> YrN+1 HR/PA'),
    (hf, 'batter_id', 'gb_pct', 'hr_rate', 'Hitter: YrN gb_pct -> YrN+1 HR/PA'),
    (pf, 'pitcher_id', 'gb_pct_against', 'hr_per_bf', 'Pitcher: YrN gb_pct -> YrN+1 HR/BF'),
    (pf, 'pitcher_id', 'avg_velo', 'k_rate', 'Pitcher: YrN avg_velo -> YrN+1 K%'),
    (pf, 'pitcher_id', 'whiff_rate', 'k_rate', 'Pitcher: YrN whiff_rate -> YrN+1 K%'),
    (pf, 'pitcher_id', 'avg_exit_velo_against', 'hr_per_bf', 'Pitcher: YrN avg_EV_against -> YrN+1 HR/BF'),
    (pf, 'pitcher_id', 'avg_exit_velo_against', 'xwoba_against', 'Pitcher: YrN avg_EV_against -> YrN+1 xwOBA_against'),
    (pf, 'pitcher_id', 'o_swing_pct', 'bb_rate', 'Pitcher: YrN chase_rate -> YrN+1 BB%'),
    (pf, 'pitcher_id', 'z_contact_pct', 'k_rate', 'Pitcher: YrN z_contact_pct -> YrN+1 K%'),
]

print(f"\n{'Label':<55s} {'Avg r':>7s}  {'N':>3s}  Per-year r values")
print("-" * 140)

for df, id_col, pred, out, label in specific:
    rs = []
    for y_n, y_n1 in year_pairs:
        dn = df[df['season'] == y_n][[id_col, pred]].dropna().rename(columns={pred: 'pred_val'})
        dn1 = df[df['season'] == y_n1][[id_col, out]].dropna().rename(columns={out: 'out_val'})
        merged = dn.merge(dn1, on=id_col)
        if len(merged) >= 30:
            r = merged['pred_val'].corr(merged['out_val'])
            if not np.isnan(r):
                rs.append((y_n, y_n1, r, len(merged)))
    if rs:
        avg_r = np.mean([x[2] for x in rs])
        detail = ', '.join([f"{y0}->{y1}: {r:.3f}(n={n})" for y0, y1, r, n in rs])
        print(f"{label:<55s} {avg_r:>+7.3f}  {len(rs):>3d}  {detail}")
    else:
        print(f"{label:<55s} {'N/A':>7s}")

# ============================================================
# SELF-PREDICTION (stability)
# ============================================================
print("\n\n" + "=" * 80)
print("SELF-PREDICTION (Year N stat -> Year N+1 same stat) for context")
print("=" * 80)

print("\nHITTERS (YoY stability):")
h_self_stats = [c for c in [
    'k_rate', 'bb_rate', 'hr_rate', 'xwoba_avg', 'barrel_pct', 'hard_hit_pct',
    'whiff_rate', 'o_swing_pct', 'avg_exit_velo', 'z_contact_pct', 'csw_pct',
    'gb_pct', 'sweet_spot_pct', 'contact_rate', 'swing_pct', 'max_exit_velo',
    'avg_launch_angle', 'xwoba_avg_bb', 'barrel_pct_bb', 'hard_hit_pct_bb'
] if c in hf.columns]
h_self = self_corr(hf, 'batter_id', h_self_stats, year_pairs)
for stat in h_self_stats:
    if stat in h_self:
        print(f"  {stat:25s} r = {h_self[stat]:+.3f}")

print("\nPITCHERS (YoY stability):")
p_self_stats = [c for c in [
    'k_rate', 'bb_rate', 'hr_per_bf', 'xwoba_against', 'barrel_pct_against',
    'hard_hit_pct_against', 'whiff_rate', 'o_swing_pct', 'avg_velo',
    'z_contact_pct', 'csw_pct', 'gb_pct_against', 'contact_rate_against',
    'avg_exit_velo_against', 'zone_pct', 'fb_pct_against'
] if c in pf.columns]
p_self = self_corr(pf, 'pitcher_id', p_self_stats, year_pairs)
for stat in p_self_stats:
    if stat in p_self:
        print(f"  {stat:25s} r = {p_self[stat]:+.3f}")
