"""
Overdispersion test: Binomial vs Beta-Binomial for MLB player rate stats.
Computes per-player-season overdispersion factor = observed_var / binomial_expected_var.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import sqlalchemy
from sqlalchemy import create_engine, text

DB_URL = "postgresql://kekoa:goatez@localhost:5433/mlb_fantasy"
engine = create_engine(DB_URL)

# -----------------------------------------------------------------
# 1. Pull per-game data
# -----------------------------------------------------------------
print("Fetching batter game data (2022-2025)...")
batter_query = text("""
    SELECT
        fpg.player_id,
        fpg.season,
        fpg.bat_pa,
        fpg.bat_k,
        fpg.bat_bb,
        fpg.bat_hr
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg USING (game_pk)
    WHERE dg.game_type = 'R'
      AND fpg.season BETWEEN 2022 AND 2025
      AND fpg.player_role = 'batter'
      AND fpg.bat_pa IS NOT NULL
      AND fpg.bat_pa > 0
""")

with engine.connect() as conn:
    batter_df = pd.read_sql(batter_query, conn)

print(f"Batter rows: {len(batter_df):,}  |  players: {batter_df['player_id'].nunique():,}")

print("Fetching pitcher game data (2022-2025)...")
pitcher_query = text("""
    SELECT
        fpg.player_id,
        fpg.season,
        fpg.pit_bf,
        fpg.pit_k,
        fpg.pit_bb,
        fpg.pit_hr,
        fpg.pit_is_starter
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg USING (game_pk)
    WHERE dg.game_type = 'R'
      AND fpg.season BETWEEN 2022 AND 2025
      AND fpg.player_role = 'pitcher'
      AND fpg.pit_bf IS NOT NULL
      AND fpg.pit_bf > 0
""")

with engine.connect() as conn:
    pitcher_df = pd.read_sql(pitcher_query, conn)

print(f"Pitcher rows: {len(pitcher_df):,}  |  players: {pitcher_df['player_id'].nunique():,}")


# -----------------------------------------------------------------
# 2. Overdispersion computation
# -----------------------------------------------------------------
def compute_overdispersion(df, n_col, k_col, min_total_n=100, min_games=50, label=""):
    """
    For each player-season (with sufficient data):
      p_hat = sum(k_i) / sum(n_i)  (season rate)
      Binomial expected variance per game: E[Var(k_i)] = n_i * p_hat * (1 - p_hat)
      Observed sample variance of per-game k counts: Var(k_i)
      overdispersion = observed_var / mean(expected_var_per_game)

    If data is exactly Binomial, overdispersion ~ 1.0.
    If overdispersion > 1, there is extra-Binomial variance.
    """
    results = []
    groups = df.groupby(["player_id", "season"])

    for (pid, season), grp in groups:
        grp = grp.dropna(subset=[n_col, k_col])
        n_arr = grp[n_col].values.astype(float)
        k_arr = grp[k_col].values.astype(float)

        total_n = n_arr.sum()
        n_games = len(grp)

        if total_n < min_total_n or n_games < min_games:
            continue

        p_hat = k_arr.sum() / total_n
        if p_hat <= 0 or p_hat >= 1:
            continue

        # Per-game Binomial expected variance: n_i * p * (1-p)
        binom_var_per_game = n_arr * p_hat * (1.0 - p_hat)

        # Observed sample variance of per-game K counts
        obs_var = np.var(k_arr, ddof=1)
        # Expected per-game variance under Binomial (average across games)
        exp_var = np.mean(binom_var_per_game)

        if exp_var <= 0:
            continue

        disp = obs_var / exp_var

        results.append({
            "player_id": pid,
            "season": season,
            "total_n": total_n,
            "n_games": n_games,
            "p_hat": p_hat,
            "obs_var": obs_var,
            "exp_var": exp_var,
            "overdispersion": disp,
        })

    result_df = pd.DataFrame(results)
    print(f"\n{label}: {len(result_df)} qualifying player-seasons")
    return result_df


def summarize_overdispersion(disp_df, label):
    d = disp_df["overdispersion"]
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  N player-seasons : {len(d):,}")
    print(f"  Mean             : {d.mean():.3f}")
    print(f"  Median           : {d.median():.3f}")
    print(f"  Std              : {d.std():.3f}")
    print(f"  P10              : {d.quantile(0.10):.3f}")
    print(f"  P25              : {d.quantile(0.25):.3f}")
    print(f"  P75              : {d.quantile(0.75):.3f}")
    print(f"  P90              : {d.quantile(0.90):.3f}")
    print(f"  P95              : {d.quantile(0.95):.3f}")
    print(f"  % > 1.1          : {(d > 1.1).mean()*100:.1f}%")
    print(f"  % > 1.2          : {(d > 1.2).mean()*100:.1f}%")
    print(f"  % > 1.5          : {(d > 1.5).mean()*100:.1f}%")
    print(f"  % > 2.0          : {(d > 2.0).mean()*100:.1f}%")
    m = d.median()
    if m < 1.1:
        interp = "Binomial is fine"
    elif m < 1.2:
        interp = "Mild overdispersion -- Binomial slightly misspecified"
    elif m < 1.5:
        interp = "MODERATE overdispersion -- Beta-Binomial would help"
    else:
        interp = "STRONG overdispersion -- Binomial significantly misspecified"
    print(f"  >>> Interpretation: {interp}")
    return d


# -----------------------------------------------------------------
# 3. Batter analyses
# -----------------------------------------------------------------
bk_disp = compute_overdispersion(
    batter_df, n_col="bat_pa", k_col="bat_k",
    min_total_n=100, min_games=50, label="Batter K%"
)
bk_summary = summarize_overdispersion(bk_disp, "Batter K%")

bbb_disp = compute_overdispersion(
    batter_df, n_col="bat_pa", k_col="bat_bb",
    min_total_n=100, min_games=50, label="Batter BB%"
)
bbb_summary = summarize_overdispersion(bbb_disp, "Batter BB%")

bhr_disp = compute_overdispersion(
    batter_df, n_col="bat_pa", k_col="bat_hr",
    min_total_n=100, min_games=50, label="Batter HR/PA"
)
bhr_summary = summarize_overdispersion(bhr_disp, "Batter HR/PA")


# -----------------------------------------------------------------
# 4. Pitcher analyses (starters)
# -----------------------------------------------------------------

# Identify starter seasons: majority of appearances as starter
starter_games = pitcher_df[pitcher_df["pit_is_starter"] == True]
total_apps = pitcher_df.groupby(["player_id", "season"]).size().rename("total_apps")
starter_apps = starter_games.groupby(["player_id", "season"]).size().rename("starter_apps")
role_df = pd.concat([total_apps, starter_apps], axis=1).fillna(0)
role_df["starter_pct"] = role_df["starter_apps"] / role_df["total_apps"]
starter_seasons_idx = role_df[role_df["starter_pct"] >= 0.5].index

pitcher_starters = pitcher_df[
    pitcher_df.set_index(["player_id", "season"]).index.isin(starter_seasons_idx)
    & (pitcher_df["pit_is_starter"] == True)
]

print(f"\nStarter pool: {pitcher_starters['player_id'].nunique()} unique pitchers, "
      f"{len(pitcher_starters):,} start rows")

pk_disp = compute_overdispersion(
    pitcher_starters, n_col="pit_bf", k_col="pit_k",
    min_total_n=100, min_games=15, label="Pitcher K% (starters)"
)
pk_summary = summarize_overdispersion(pk_disp, "Pitcher K% (starters, >=15 GS, >=100 BF)")

pbb_disp = compute_overdispersion(
    pitcher_starters, n_col="pit_bf", k_col="pit_bb",
    min_total_n=100, min_games=15, label="Pitcher BB% (starters)"
)
pbb_summary = summarize_overdispersion(pbb_disp, "Pitcher BB% (starters, >=15 GS, >=100 BF)")


# -----------------------------------------------------------------
# 5. Year-by-year breakdown
# -----------------------------------------------------------------
print("\n\n" + "="*62)
print("  BATTER K%  -- Overdispersion by Season")
print("="*62)
for season in sorted(bk_disp["season"].unique()):
    sub = bk_disp[bk_disp["season"] == season]["overdispersion"]
    print(f"  {season}:  N={len(sub):4d}  mean={sub.mean():.3f}  "
          f"median={sub.median():.3f}  P90={sub.quantile(0.9):.3f}")

print("\n" + "="*62)
print("  PITCHER K% -- Overdispersion by Season")
print("="*62)
for season in sorted(pk_disp["season"].unique()):
    sub = pk_disp[pk_disp["season"] == season]["overdispersion"]
    print(f"  {season}:  N={len(sub):4d}  mean={sub.mean():.3f}  "
          f"median={sub.median():.3f}  P90={sub.quantile(0.9):.3f}")


# -----------------------------------------------------------------
# 6. Overdispersion by player sub-type
# -----------------------------------------------------------------
print("\n\n" + "="*62)
print("  BATTER K% -- By K-rate tier")
print("="*62)
bk_disp["k_tier"] = pd.qcut(bk_disp["p_hat"], q=3, labels=["low_K", "mid_K", "high_K"])
for tier, grp in bk_disp.groupby("k_tier"):
    d = grp["overdispersion"]
    print(f"  {str(tier):8s}  p=[{grp['p_hat'].min():.3f},{grp['p_hat'].max():.3f}]  "
          f"N={len(d):4d}  mean={d.mean():.3f}  median={d.median():.3f}")

print("\n" + "="*62)
print("  PITCHER K% -- By K-rate tier")
print("="*62)
pk_disp["k_tier"] = pd.qcut(pk_disp["p_hat"], q=3, labels=["low_K", "mid_K", "high_K"])
for tier, grp in pk_disp.groupby("k_tier"):
    d = grp["overdispersion"]
    print(f"  {str(tier):8s}  p=[{grp['p_hat'].min():.3f},{grp['p_hat'].max():.3f}]  "
          f"N={len(d):4d}  mean={d.mean():.3f}  median={d.median():.3f}")

# Relievers comparison
print("\n" + "="*62)
print("  PITCHER K% -- Starters vs Relievers")
print("="*62)
reliever_seasons_idx = role_df[role_df["starter_pct"] < 0.5].index
pitcher_relievers = pitcher_df[
    pitcher_df.set_index(["player_id", "season"]).index.isin(reliever_seasons_idx)
]
pk_rel_disp = compute_overdispersion(
    pitcher_relievers, n_col="pit_bf", k_col="pit_k",
    min_total_n=80, min_games=30, label="Pitcher K% (relievers)"
)
if len(pk_rel_disp) > 0:
    d = pk_rel_disp["overdispersion"]
    print(f"  Relievers: N={len(d):4d}  mean={d.mean():.3f}  median={d.median():.3f}  P90={d.quantile(0.9):.3f}")
d = pk_disp["overdispersion"]
print(f"  Starters : N={len(d):4d}  mean={d.mean():.3f}  median={d.median():.3f}  P90={d.quantile(0.9):.3f}")


# -----------------------------------------------------------------
# 7. Beta-Binomial phi (concentration) estimation
# -----------------------------------------------------------------
print("\n\n" + "="*62)
print("  BETA-BINOMIAL CONCENTRATION (phi) ESTIMATES")
print("  For Beta-Binomial: Var(k_i) = n_i*p*(1-p) * [1 + (n_i-1)/(phi+1)]")
print("  => overdispersion factor = 1 + (n_mean-1)/(phi+1)")
print("  => phi = (n_mean - 1) / (disp - 1) - 1")
print("  phi < 10: strong OD  |  10-30: moderate  |  >100: near-Binomial")
print("="*62)


def estimate_phi(disp_df):
    """Method-of-moments estimate of Beta-Binomial concentration phi."""
    d = disp_df["overdispersion"].values
    n_mean = disp_df["total_n"].values / disp_df["n_games"].values

    valid = d > 1.001
    if valid.sum() < 5:
        return np.nan, np.nan, 0

    phi_vals = (n_mean[valid] - 1) / (d[valid] - 1) - 1
    phi_vals = phi_vals[np.isfinite(phi_vals) & (phi_vals > 0)]
    if len(phi_vals) == 0:
        return np.nan, np.nan, 0
    return np.median(phi_vals), np.mean(phi_vals), len(phi_vals)


for name, df in [
    ("Batter K%",    bk_disp),
    ("Batter BB%",   bbb_disp),
    ("Batter HR/PA", bhr_disp),
    ("Pitcher K%",   pk_disp),
    ("Pitcher BB%",  pbb_disp),
]:
    if len(df) == 0:
        continue
    phi_med, phi_mean, n_valid = estimate_phi(df)
    if np.isnan(phi_med):
        print(f"  {name:18s}  phi: insufficient data")
        continue
    severity = "near-Binomial" if phi_med > 100 else (
               "mild OD" if phi_med > 30 else (
               "moderate OD" if phi_med > 10 else "strong OD"))
    print(f"  {name:18s}  phi_med={phi_med:8.1f}  phi_mean={phi_mean:8.1f}  "
          f"n_valid={n_valid:4d}  => {severity}")


# -----------------------------------------------------------------
# 8. Calibration implication
# -----------------------------------------------------------------
print("\n\n" + "="*62)
print("  CALIBRATION IMPLICATION")
print("  What actual coverage does a Binomial 95% CI achieve,")
print("  given the observed overdispersion? (Normal approximation)")
print("  actual_coverage ~ Phi(1.96 / sqrt(disp_median))")
print("="*62)

all_stats_list = [
    ("Batter K%",    bk_disp),
    ("Batter BB%",   bbb_disp),
    ("Batter HR/PA", bhr_disp),
    ("Pitcher K%",   pk_disp),
    ("Pitcher BB%",  pbb_disp),
]

for name, df in all_stats_list:
    if len(df) == 0:
        continue
    m = df["overdispersion"].median()
    approx_coverage = 2 * stats.norm.cdf(1.96 / np.sqrt(m)) - 1
    sigma_inflation = np.sqrt(m)
    print(f"  {name:18s}  disp_med={m:.3f}  sigma_inflation={sigma_inflation:.3f}x  "
          f"Binomial 95% CI ~> {approx_coverage*100:.1f}% actual coverage")


# -----------------------------------------------------------------
# 9. Pearson chi-squared dispersion test
# -----------------------------------------------------------------
print("\n\n" + "="*62)
print("  PEARSON CHI-SQUARED DISPERSION STATISTIC")
print("  X2/df = mean(chi2 per game). Under Binomial, ratio ~ 1.0.")
print("  Ratio > 1.0 confirms overdispersion independent of sample-var method.")
print("="*62)


def pearson_dispersion(df, n_col, k_col, min_total_n=100, min_games=50, label=""):
    results = []
    for (pid, season), grp in df.groupby(["player_id", "season"]):
        grp = grp.dropna(subset=[n_col, k_col])
        n_arr = grp[n_col].values.astype(float)
        k_arr = grp[k_col].values.astype(float)
        total_n = n_arr.sum()
        n_games = len(grp)
        if total_n < min_total_n or n_games < min_games:
            continue
        p_hat = k_arr.sum() / total_n
        if p_hat <= 0 or p_hat >= 1:
            continue
        expected = n_arr * p_hat
        variance = n_arr * p_hat * (1 - p_hat)
        valid = variance > 0
        chi2 = np.sum((k_arr[valid] - expected[valid])**2 / variance[valid])
        df_chi = valid.sum() - 1
        if df_chi <= 0:
            continue
        ratio = chi2 / df_chi
        results.append({"player_id": pid, "season": season, "chi2_ratio": ratio, "p_hat": p_hat})

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        r = rdf["chi2_ratio"]
        print(f"\n  {label}")
        print(f"  N={len(r):4d}  mean={r.mean():.3f}  median={r.median():.3f}  "
              f"P10={r.quantile(0.1):.3f}  P90={r.quantile(0.9):.3f}")
        print(f"  % chi2_ratio > 1.2: {(r>1.2).mean()*100:.1f}%  "
              f"% chi2_ratio > 1.5: {(r>1.5).mean()*100:.1f}%")
    return rdf


pearson_dispersion(batter_df, "bat_pa", "bat_k", label="Batter K%")
pearson_dispersion(batter_df, "bat_pa", "bat_bb", label="Batter BB%")
pearson_dispersion(batter_df, "bat_pa", "bat_hr", label="Batter HR/PA")
pearson_dispersion(pitcher_starters, "pit_bf", "pit_k", min_games=15, label="Pitcher K% (starters)")
pearson_dispersion(pitcher_starters, "pit_bf", "pit_bb", min_games=15, label="Pitcher BB% (starters)")

print("\n\nAnalysis complete.")
