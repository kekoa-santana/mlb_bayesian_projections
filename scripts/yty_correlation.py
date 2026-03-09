"""
Year-to-year correlation analysis for MLB hitters and pitchers.
Computes comprehensive discipline/contact/damage layer stats and
analyzes self-correlations and cross-stat correlations.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

engine = create_engine('postgresql://kekoa:goatez@localhost:5433/mlb_fantasy')

###############################################################################
# STEP 1: Build comprehensive hitter season stats
###############################################################################
print("=" * 80)
print("STEP 1: Building comprehensive season-level stats from pitch data + caches")
print("=" * 80)

# Query pitch-level aggregates per hitter per season
hitter_pitch_sql = """
SELECT
    fp.batter_id,
    dg.season,
    COUNT(*) AS total_pitches,
    SUM(CASE WHEN fp.is_swing THEN 1 ELSE 0 END) AS total_swings,
    SUM(CASE WHEN fp.is_whiff THEN 1 ELSE 0 END) AS total_whiffs,
    SUM(CASE WHEN fp.is_called_strike THEN 1 ELSE 0 END) AS total_called_strikes,
    SUM(CASE WHEN fp.is_bip THEN 1 ELSE 0 END) AS total_bip,
    SUM(CASE WHEN fp.is_foul THEN 1 ELSE 0 END) AS total_fouls,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS in_zone_pitches,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing THEN 1 ELSE 0 END) AS z_swing,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS out_zone_pitches,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 AND fp.is_swing THEN 1 ELSE 0 END) AS o_swing,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing AND NOT fp.is_whiff THEN 1 ELSE 0 END) AS z_contact,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 AND fp.is_swing AND NOT fp.is_whiff THEN 1 ELSE 0 END) AS o_contact,
    SUM(CASE WHEN fp.pitch_number = 1 AND fp.is_swing THEN 1 ELSE 0 END) AS first_pitch_swings,
    SUM(CASE WHEN fp.pitch_number = 1 THEN 1 ELSE 0 END) AS first_pitches
FROM production.fact_pitch fp
JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
WHERE dg.game_type = 'R' AND dg.season BETWEEN 2018 AND 2025
GROUP BY fp.batter_id, dg.season
"""
print("Querying hitter pitch-level aggregates...")
hitter_pitch = pd.read_sql(hitter_pitch_sql, engine)
print(f"  Got {len(hitter_pitch)} hitter-seasons from pitch data")

# Query batted ball aggregates per hitter per season
hitter_bb_sql = """
SELECT
    fp.batter_id,
    dg.season,
    AVG(CASE WHEN sbb.launch_speed != 'NaN' THEN sbb.launch_speed END) AS avg_exit_velo,
    AVG(CASE WHEN sbb.launch_angle != 'NaN' THEN sbb.launch_angle END) AS avg_launch_angle,
    STDDEV(CASE WHEN sbb.launch_angle != 'NaN' THEN sbb.launch_angle END) AS la_std,
    AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba_avg,
    SUM(CASE WHEN sbb.hard_hit = true THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS hard_hit_pct,
    SUM(CASE WHEN sbb.ideal_contact = true THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS barrel_pct,
    SUM(CASE WHEN sbb.sweet_spot = true THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS sweet_spot_pct_bb,
    COUNT(*) AS batted_balls,
    SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END)::float
        / NULLIF(SUM(CASE WHEN sbb.launch_angle != 'NaN' THEN 1 ELSE 0 END), 0) AS sweet_spot_pct,
    SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
        / NULLIF(SUM(CASE WHEN sbb.launch_angle != 'NaN' THEN 1 ELSE 0 END), 0) AS gb_pct,
    SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
        / NULLIF(SUM(CASE WHEN sbb.launch_angle != 'NaN' THEN 1 ELSE 0 END), 0) AS fb_pct,
    MAX(CASE WHEN sbb.launch_speed != 'NaN' THEN sbb.launch_speed END) AS max_exit_velo
FROM production.fact_pitch fp
JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
WHERE dg.game_type = 'R' AND dg.season BETWEEN 2018 AND 2025
  AND fp.is_bip = true
GROUP BY fp.batter_id, dg.season
"""
print("Querying hitter batted ball aggregates...")
hitter_bb = pd.read_sql(hitter_bb_sql, engine)
print(f"  Got {len(hitter_bb)} hitter-seasons from batted ball data")

# Load cached season totals for PA/K/BB/HR counts
print("Loading cached season totals...")
hitter_seasons = []
for yr in range(2018, 2026):
    df = pd.read_parquet(f'data/cached/season_totals_age_{yr}.parquet')
    hitter_seasons.append(df)
hitter_base = pd.concat(hitter_seasons, ignore_index=True)
print(f"  Got {len(hitter_base)} hitter-seasons from cache")

# Merge everything
hitters = hitter_base.merge(hitter_pitch, on=['batter_id', 'season'], how='left')
hitters = hitters.merge(hitter_bb, on=['batter_id', 'season'], how='left', suffixes=('', '_bb'))

# Compute derived rates
hitters['whiff_rate'] = hitters['total_whiffs'] / hitters['total_swings']
hitters['swing_pct'] = hitters['total_swings'] / hitters['total_pitches']
hitters['contact_rate'] = 1 - hitters['whiff_rate']
hitters['csw_pct'] = (hitters['total_called_strikes'] + hitters['total_whiffs']) / hitters['total_pitches']
hitters['o_swing_pct'] = hitters['o_swing'] / hitters['out_zone_pitches']
hitters['z_swing_pct'] = hitters['z_swing'] / hitters['in_zone_pitches']
hitters['o_contact_pct'] = hitters['o_contact'] / hitters['o_swing']
hitters['z_contact_pct'] = hitters['z_contact'] / hitters['z_swing']
hitters['zone_pct'] = hitters['in_zone_pitches'] / hitters['total_pitches']
hitters['first_pitch_swing_pct'] = hitters['first_pitch_swings'] / hitters['first_pitches']
hitters['bip_pct'] = hitters['total_bip'] / hitters['pa']
hitters['foul_pct'] = hitters['total_fouls'] / hitters['total_swings']
hitters['pitches_per_pa'] = hitters['total_pitches'] / hitters['pa']

# Use BB-derived columns if base ones missing
for col in ['avg_exit_velo', 'avg_launch_angle', 'xwoba_avg']:
    bb_col = col + '_bb'
    if bb_col in hitters.columns:
        hitters[col] = hitters[col].fillna(hitters[bb_col])

print(f"\nFinal hitter dataset: {len(hitters)} rows, {len(hitters.columns)} columns")

###############################################################################
# Build pitcher stats
###############################################################################
print("\n" + "=" * 80)
print("Building pitcher season-level stats")
print("=" * 80)

pitcher_pitch_sql = """
SELECT
    fp.pitcher_id,
    dg.season,
    COUNT(*) AS total_pitches,
    SUM(CASE WHEN fp.is_swing THEN 1 ELSE 0 END) AS total_swings,
    SUM(CASE WHEN fp.is_whiff THEN 1 ELSE 0 END) AS total_whiffs,
    SUM(CASE WHEN fp.is_called_strike THEN 1 ELSE 0 END) AS total_called_strikes,
    SUM(CASE WHEN fp.is_bip THEN 1 ELSE 0 END) AS total_bip,
    SUM(CASE WHEN fp.is_foul THEN 1 ELSE 0 END) AS total_fouls,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS in_zone_pitches,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing THEN 1 ELSE 0 END) AS z_swing,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS out_zone_pitches,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 AND fp.is_swing THEN 1 ELSE 0 END) AS o_swing,
    SUM(CASE WHEN fp.zone BETWEEN 1 AND 9 AND fp.is_swing AND NOT fp.is_whiff THEN 1 ELSE 0 END) AS z_contact,
    SUM(CASE WHEN fp.zone NOT BETWEEN 1 AND 9 AND fp.is_swing AND NOT fp.is_whiff THEN 1 ELSE 0 END) AS o_contact,
    AVG(CASE WHEN fp.release_speed != 'NaN' THEN fp.release_speed END) AS avg_velo,
    SUM(CASE WHEN fp.pitch_number = 1 AND fp.is_swing THEN 1 ELSE 0 END) AS first_pitch_swings,
    SUM(CASE WHEN fp.pitch_number = 1 THEN 1 ELSE 0 END) AS first_pitches
FROM production.fact_pitch fp
JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
WHERE dg.game_type = 'R' AND dg.season BETWEEN 2018 AND 2025
GROUP BY fp.pitcher_id, dg.season
"""
print("Querying pitcher pitch-level aggregates...")
pitcher_pitch = pd.read_sql(pitcher_pitch_sql, engine)
print(f"  Got {len(pitcher_pitch)} pitcher-seasons")

pitcher_bb_sql = """
SELECT
    fp.pitcher_id,
    dg.season,
    AVG(CASE WHEN sbb.launch_speed != 'NaN' THEN sbb.launch_speed END) AS avg_exit_velo_against,
    AVG(CASE WHEN sbb.launch_angle != 'NaN' THEN sbb.launch_angle END) AS avg_la_against,
    AVG(CASE WHEN sbb.xwoba != 'NaN' THEN sbb.xwoba END) AS xwoba_against,
    SUM(CASE WHEN sbb.hard_hit = true THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS hard_hit_pct_against,
    SUM(CASE WHEN sbb.ideal_contact = true THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS barrel_pct_against,
    COUNT(*) AS batted_balls,
    SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle < 10 THEN 1 ELSE 0 END)::float
        / NULLIF(SUM(CASE WHEN sbb.launch_angle != 'NaN' THEN 1 ELSE 0 END), 0) AS gb_pct_against,
    SUM(CASE WHEN sbb.launch_angle != 'NaN' AND sbb.launch_angle > 25 THEN 1 ELSE 0 END)::float
        / NULLIF(SUM(CASE WHEN sbb.launch_angle != 'NaN' THEN 1 ELSE 0 END), 0) AS fb_pct_against
FROM production.fact_pitch fp
JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
JOIN production.sat_batted_balls sbb ON fp.pa_id = sbb.pa_id
WHERE dg.game_type = 'R' AND dg.season BETWEEN 2018 AND 2025
  AND fp.is_bip = true
GROUP BY fp.pitcher_id, dg.season
"""
print("Querying pitcher batted ball aggregates...")
pitcher_bb = pd.read_sql(pitcher_bb_sql, engine)
print(f"  Got {len(pitcher_bb)} pitcher-seasons")

# Load cached pitcher season totals
pitcher_seasons = []
for yr in range(2018, 2026):
    df = pd.read_parquet(f'data/cached/pitcher_season_totals_age_{yr}.parquet')
    pitcher_seasons.append(df)
pitcher_base = pd.concat(pitcher_seasons, ignore_index=True)

# Merge
pitchers = pitcher_base.merge(pitcher_pitch, on=['pitcher_id', 'season'], how='left')
pitchers = pitchers.merge(pitcher_bb, on=['pitcher_id', 'season'], how='left')

# Compute derived rates
pitchers['whiff_rate'] = pitchers['total_whiffs'] / pitchers['total_swings']
pitchers['swing_pct_against'] = pitchers['total_swings'] / pitchers['total_pitches']
pitchers['contact_rate_against'] = 1 - pitchers['whiff_rate']
pitchers['csw_pct'] = (pitchers['total_called_strikes'] + pitchers['total_whiffs']) / pitchers['total_pitches']
pitchers['o_swing_pct'] = pitchers['o_swing'] / pitchers['out_zone_pitches']
pitchers['z_swing_pct'] = pitchers['z_swing'] / pitchers['in_zone_pitches']
pitchers['o_contact_pct'] = pitchers['o_contact'] / pitchers['o_swing']
pitchers['z_contact_pct'] = pitchers['z_contact'] / pitchers['z_swing']
pitchers['zone_pct'] = pitchers['in_zone_pitches'] / pitchers['total_pitches']
pitchers['first_pitch_swing_pct'] = pitchers['first_pitch_swings'] / pitchers['first_pitches']
pitchers['foul_pct'] = pitchers['total_fouls'] / pitchers['total_swings']
pitchers['k_per_9'] = pitchers['k'] / pitchers['ip'].replace(0, np.nan) * 9
pitchers['bb_per_9'] = pitchers['bb'] / pitchers['ip'].replace(0, np.nan) * 9
pitchers['k_bb_ratio'] = pitchers['k_rate'] / pitchers['bb_rate'].replace(0, np.nan)

print(f"\nFinal pitcher dataset: {len(pitchers)} rows, {len(pitchers.columns)} columns")

# Save
hitters.to_parquet('data/cached/hitter_full_stats.parquet')
pitchers.to_parquet('data/cached/pitcher_full_stats.parquet')
print("\nSaved full stats to data/cached/")

###############################################################################
# STEP 2: HITTER year-to-year self-correlations
###############################################################################
print("\n" + "=" * 80)
print("STEP 2: HITTER Year-to-Year Self-Correlations (>=200 PA both years)")
print("=" * 80)

hitter_rate_cols = [
    # Discipline layer
    'o_swing_pct', 'z_swing_pct', 'swing_pct', 'zone_pct',
    'bb_rate', 'k_rate', 'csw_pct', 'first_pitch_swing_pct', 'pitches_per_pa',
    # Contact layer
    'whiff_rate', 'contact_rate', 'o_contact_pct', 'z_contact_pct', 'foul_pct',
    # Damage layer
    'avg_exit_velo', 'max_exit_velo', 'barrel_pct', 'hard_hit_pct',
    'sweet_spot_pct', 'avg_launch_angle', 'la_std', 'gb_pct', 'fb_pct',
    # Outcome stats
    'hr_rate', 'xwoba_avg', 'bip_pct',
]

def compute_yty_correlations(df, id_col, rate_cols, min_threshold, threshold_col):
    """Compute year-to-year correlations for consecutive seasons."""
    results = {}
    pair_counts = {}
    for col in rate_cols:
        if col not in df.columns:
            continue
        rs = []
        ns = []
        for yr in range(2019, 2026):
            y1 = df[(df['season'] == yr - 1) & (df[threshold_col] >= min_threshold)][[id_col, col]].dropna()
            y2 = df[(df['season'] == yr) & (df[threshold_col] >= min_threshold)][[id_col, col]].dropna()
            merged = y1.merge(y2, on=id_col, suffixes=('_y1', '_y2'))
            if len(merged) >= 20:
                r = merged[f'{col}_y1'].corr(merged[f'{col}_y2'])
                rs.append(r)
                ns.append(len(merged))
        if rs:
            # Weighted average correlation by sample size
            weights = np.array(ns)
            avg_r = np.average(rs, weights=weights)
            results[col] = avg_r
            pair_counts[col] = int(np.mean(ns))
    return results, pair_counts

hitter_corrs, hitter_ns = compute_yty_correlations(
    hitters, 'batter_id', hitter_rate_cols, 200, 'pa'
)

# Organize by skill layer
discipline_stats = ['o_swing_pct', 'z_swing_pct', 'swing_pct', 'bb_rate', 'k_rate',
                    'csw_pct', 'first_pitch_swing_pct', 'pitches_per_pa', 'zone_pct']
contact_stats = ['whiff_rate', 'contact_rate', 'o_contact_pct', 'z_contact_pct', 'foul_pct']
damage_stats = ['avg_exit_velo', 'max_exit_velo', 'barrel_pct', 'hard_hit_pct',
                'sweet_spot_pct', 'avg_launch_angle', 'la_std', 'gb_pct', 'fb_pct']
outcome_stats = ['hr_rate', 'xwoba_avg', 'bip_pct']

def print_layer(name, stats, corrs, ns):
    print(f"\n  --- {name} ---")
    items = [(s, corrs.get(s, np.nan), ns.get(s, 0)) for s in stats if s in corrs]
    items.sort(key=lambda x: -x[1])
    for stat, r, n in items:
        bar = '#' * int(abs(r) * 40)
        print(f"    {stat:<25s}  r = {r:+.3f}  (n~{n:>3d})  |{bar}")

print("\nHITTER YEAR-TO-YEAR CORRELATIONS (sorted by r within each layer)")
print_layer("DISCIPLINE LAYER (most stable)", discipline_stats, hitter_corrs, hitter_ns)
print_layer("CONTACT LAYER", contact_stats, hitter_corrs, hitter_ns)
print_layer("DAMAGE LAYER", damage_stats, hitter_corrs, hitter_ns)
print_layer("OUTCOME STATS", outcome_stats, hitter_corrs, hitter_ns)

# Also print overall sorted
print("\n  --- ALL HITTER STATS SORTED BY YTY CORRELATION ---")
all_sorted = sorted(hitter_corrs.items(), key=lambda x: -x[1])
for stat, r in all_sorted:
    layer = 'DISC' if stat in discipline_stats else 'CONT' if stat in contact_stats else 'DMG' if stat in damage_stats else 'OUT'
    print(f"    {stat:<25s}  r = {r:+.3f}  [{layer}]  (n~{hitter_ns[stat]:>3d})")

###############################################################################
# STEP 3: PITCHER year-to-year self-correlations
###############################################################################
print("\n" + "=" * 80)
print("STEP 3: PITCHER Year-to-Year Self-Correlations (>=200 BF both years)")
print("=" * 80)

pitcher_rate_cols = [
    # Discipline layer (induced)
    'o_swing_pct', 'z_swing_pct', 'swing_pct_against', 'zone_pct',
    'bb_rate', 'k_rate', 'csw_pct', 'first_pitch_swing_pct',
    # Contact layer (suppression)
    'whiff_rate', 'contact_rate_against', 'o_contact_pct', 'z_contact_pct', 'foul_pct',
    # Damage layer (allowed)
    'avg_exit_velo_against', 'barrel_pct_against', 'hard_hit_pct_against',
    'xwoba_against', 'gb_pct_against', 'fb_pct_against', 'avg_la_against',
    # Other
    'avg_velo', 'k_per_9', 'bb_per_9', 'k_bb_ratio', 'hr_per_bf', 'hr_per_9',
]

pitcher_corrs, pitcher_ns = compute_yty_correlations(
    pitchers, 'pitcher_id', pitcher_rate_cols, 200, 'batters_faced'
)

p_discipline = ['o_swing_pct', 'z_swing_pct', 'swing_pct_against', 'bb_rate', 'k_rate',
                'csw_pct', 'first_pitch_swing_pct', 'zone_pct']
p_contact = ['whiff_rate', 'contact_rate_against', 'o_contact_pct', 'z_contact_pct', 'foul_pct']
p_damage = ['avg_exit_velo_against', 'barrel_pct_against', 'hard_hit_pct_against',
            'xwoba_against', 'gb_pct_against', 'fb_pct_against', 'avg_la_against']
p_other = ['avg_velo', 'k_per_9', 'bb_per_9', 'k_bb_ratio', 'hr_per_bf', 'hr_per_9']

print("\nPITCHER YEAR-TO-YEAR CORRELATIONS")
print_layer("DISCIPLINE LAYER (induced)", p_discipline, pitcher_corrs, pitcher_ns)
print_layer("CONTACT SUPPRESSION LAYER", p_contact, pitcher_corrs, pitcher_ns)
print_layer("DAMAGE ALLOWED LAYER", p_damage, pitcher_corrs, pitcher_ns)
print_layer("OTHER / VELOCITY", p_other, pitcher_corrs, pitcher_ns)

print("\n  --- ALL PITCHER STATS SORTED BY YTY CORRELATION ---")
all_sorted_p = sorted(pitcher_corrs.items(), key=lambda x: -x[1])
for stat, r in all_sorted_p:
    layer = 'DISC' if stat in p_discipline else 'CONT' if stat in p_contact else 'DMG' if stat in p_damage else 'OTH'
    print(f"    {stat:<25s}  r = {r:+.3f}  [{layer}]  (n~{pitcher_ns[stat]:>3d})")

###############################################################################
# STEP 4: HITTER cross-stat correlations (Year N stat X -> Year N+1 stat Y)
###############################################################################
print("\n" + "=" * 80)
print("STEP 4: HITTER Cross-Stat Correlations (Year N -> Year N+1)")
print("=" * 80)

def compute_cross_correlations(df, id_col, predictor_cols, target_cols, min_threshold, threshold_col):
    """For each (predictor, target) pair, compute weighted avg YTY r."""
    results = []
    for pred in predictor_cols:
        if pred not in df.columns:
            continue
        for tgt in target_cols:
            if tgt not in df.columns:
                continue
            rs = []
            ns = []
            for yr in range(2019, 2026):
                y1 = df[(df['season'] == yr - 1) & (df[threshold_col] >= min_threshold)][[id_col, pred]].dropna()
                y2 = df[(df['season'] == yr) & (df[threshold_col] >= min_threshold)][[id_col, tgt]].dropna()
                y1 = y1.rename(columns={pred: 'pred_val'})
                y2 = y2.rename(columns={tgt: 'tgt_val'})
                merged = y1.merge(y2, on=id_col)
                if len(merged) >= 20:
                    r = merged['pred_val'].corr(merged['tgt_val'])
                    rs.append(r)
                    ns.append(len(merged))
            if rs:
                weights = np.array(ns)
                avg_r = np.average(rs, weights=weights)
                results.append({
                    'predictor': pred,
                    'target': tgt,
                    'r': avg_r,
                    'n': int(np.mean(ns)),
                    'self_r': hitter_corrs.get(tgt, np.nan),
                    'lift': avg_r - hitter_corrs.get(tgt, 0)
                })
    return pd.DataFrame(results)

# Key cross-stat pairs to test
h_predictors = ['avg_exit_velo', 'max_exit_velo', 'barrel_pct', 'hard_hit_pct',
                'whiff_rate', 'contact_rate', 'o_swing_pct', 'z_swing_pct',
                'o_contact_pct', 'z_contact_pct', 'csw_pct', 'sweet_spot_pct',
                'xwoba_avg', 'fb_pct', 'gb_pct', 'swing_pct', 'foul_pct',
                'avg_launch_angle', 'pitches_per_pa', 'first_pitch_swing_pct']
h_targets = ['k_rate', 'bb_rate', 'hr_rate', 'xwoba_avg', 'barrel_pct', 'hard_hit_pct']

cross_h = compute_cross_correlations(hitters, 'batter_id', h_predictors, h_targets, 200, 'pa')

# For each target, show top predictors
for tgt in h_targets:
    sub = cross_h[cross_h['target'] == tgt].sort_values('r', ascending=False, key=abs)
    self_r = hitter_corrs.get(tgt, np.nan)
    print(f"\n  Target: {tgt} (self-correlation: {self_r:+.3f})")
    print(f"  {'Predictor':<25s}  {'r':>7s}  {'vs self':>8s}  {'n':>5s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*8}  {'-'*5}")
    for _, row in sub.head(15).iterrows():
        if row['predictor'] == tgt:
            marker = ' <-- SELF'
        elif abs(row['r']) > abs(self_r):
            marker = ' ***'
        else:
            marker = ''
        print(f"  {row['predictor']:<25s}  {row['r']:+.3f}    {row['r'] - self_r:+.3f}   {row['n']:>4.0f}{marker}")

###############################################################################
# STEP 5: PITCHER cross-stat correlations
###############################################################################
print("\n" + "=" * 80)
print("STEP 5: PITCHER Cross-Stat Correlations (Year N -> Year N+1)")
print("=" * 80)

p_predictors = ['avg_velo', 'whiff_rate', 'contact_rate_against', 'o_swing_pct',
                'z_swing_pct', 'o_contact_pct', 'z_contact_pct', 'csw_pct',
                'zone_pct', 'barrel_pct_against', 'hard_hit_pct_against',
                'xwoba_against', 'gb_pct_against', 'fb_pct_against',
                'avg_exit_velo_against', 'foul_pct', 'swing_pct_against',
                'first_pitch_swing_pct', 'avg_la_against']
p_targets = ['k_rate', 'bb_rate', 'hr_per_bf', 'xwoba_against',
             'barrel_pct_against', 'hard_hit_pct_against']

def compute_cross_correlations_pitcher(df, id_col, predictor_cols, target_cols, min_threshold, threshold_col, self_corrs):
    results = []
    for pred in predictor_cols:
        if pred not in df.columns:
            continue
        for tgt in target_cols:
            if tgt not in df.columns:
                continue
            rs = []
            ns = []
            for yr in range(2019, 2026):
                y1 = df[(df['season'] == yr - 1) & (df[threshold_col] >= min_threshold)][[id_col, pred]].dropna()
                y2 = df[(df['season'] == yr) & (df[threshold_col] >= min_threshold)][[id_col, tgt]].dropna()
                y1 = y1.rename(columns={pred: 'pred_val'})
                y2 = y2.rename(columns={tgt: 'tgt_val'})
                merged = y1.merge(y2, on=id_col)
                if len(merged) >= 20:
                    r = merged['pred_val'].corr(merged['tgt_val'])
                    rs.append(r)
                    ns.append(len(merged))
            if rs:
                weights = np.array(ns)
                avg_r = np.average(rs, weights=weights)
                results.append({
                    'predictor': pred,
                    'target': tgt,
                    'r': avg_r,
                    'n': int(np.mean(ns)),
                    'self_r': self_corrs.get(tgt, np.nan),
                })
    return pd.DataFrame(results)

cross_p = compute_cross_correlations_pitcher(
    pitchers, 'pitcher_id', p_predictors, p_targets, 200, 'batters_faced', pitcher_corrs
)

for tgt in p_targets:
    sub = cross_p[cross_p['target'] == tgt].sort_values('r', ascending=False, key=abs)
    self_r = pitcher_corrs.get(tgt, np.nan)
    print(f"\n  Target: {tgt} (self-correlation: {self_r:+.3f})")
    print(f"  {'Predictor':<25s}  {'r':>7s}  {'vs self':>8s}  {'n':>5s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*8}  {'-'*5}")
    for _, row in sub.head(15).iterrows():
        if row['predictor'] == tgt:
            marker = ' <-- SELF'
        elif abs(row['r']) > abs(self_r):
            marker = ' ***'
        else:
            marker = ''
        print(f"  {row['predictor']:<25s}  {row['r']:+.3f}    {row['r'] - self_r:+.3f}   {row['n']:>4.0f}{marker}")

###############################################################################
# SUMMARY: Layer stability comparison
###############################################################################
print("\n" + "=" * 80)
print("SUMMARY: Skill Layer Stability (Hitters)")
print("=" * 80)

for layer_name, stats in [("DISCIPLINE", discipline_stats), ("CONTACT", contact_stats),
                           ("DAMAGE", damage_stats), ("OUTCOME", outcome_stats)]:
    rs = [hitter_corrs[s] for s in stats if s in hitter_corrs]
    if rs:
        print(f"  {layer_name:<12s}  avg r = {np.mean(rs):+.3f}  (range: {min(rs):+.3f} to {max(rs):+.3f})")

print("\nSUMMARY: Skill Layer Stability (Pitchers)")
for layer_name, stats in [("DISCIPLINE", p_discipline), ("CONTACT", p_contact),
                           ("DAMAGE", p_damage), ("OTHER/VELO", p_other)]:
    rs = [pitcher_corrs[s] for s in stats if s in pitcher_corrs]
    if rs:
        print(f"  {layer_name:<12s}  avg r = {np.mean(rs):+.3f}  (range: {min(rs):+.3f} to {max(rs):+.3f})")

print("\nDone!")
