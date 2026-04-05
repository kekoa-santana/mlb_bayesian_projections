"""
Extra overdispersion analyses:
1. Pitcher: does variable BF confound the result? (use rate-based test)
2. Age splits for batter K%
3. Practical implication for season posterior width
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from sqlalchemy import create_engine, text

DB_URL = "postgresql://kekoa:goatez@localhost:5433/mlb_fantasy"
engine = create_engine(DB_URL)

print("Fetching batter data with birth_date...")
batter_query = text("""
    SELECT
        fpg.player_id,
        fpg.season,
        fpg.bat_pa,
        fpg.bat_k,
        dp.birth_date
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg USING (game_pk)
    LEFT JOIN production.dim_player dp ON fpg.player_id = dp.player_id
    WHERE dg.game_type = 'R'
      AND fpg.season BETWEEN 2022 AND 2025
      AND fpg.player_role = 'batter'
      AND fpg.bat_pa IS NOT NULL
      AND fpg.bat_pa > 0
""")
with engine.connect() as conn:
    batter_df = pd.read_sql(batter_query, conn)

print("Fetching pitcher (starter) data with birth_date...")
pitcher_query = text("""
    SELECT
        fpg.player_id,
        fpg.season,
        fpg.pit_bf,
        fpg.pit_k,
        fpg.pit_is_starter,
        dp.birth_date
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg USING (game_pk)
    LEFT JOIN production.dim_player dp ON fpg.player_id = dp.player_id
    WHERE dg.game_type = 'R'
      AND fpg.season BETWEEN 2022 AND 2025
      AND fpg.player_role = 'pitcher'
      AND fpg.pit_bf IS NOT NULL
      AND fpg.pit_bf > 0
      AND fpg.pit_is_starter = TRUE
""")
with engine.connect() as conn:
    pitcher_df = pd.read_sql(pitcher_query, conn)

# Compute age
batter_df["birth_year"] = pd.to_datetime(batter_df["birth_date"]).dt.year
batter_df["age"] = batter_df["season"] - batter_df["birth_year"]

pitcher_df["birth_year"] = pd.to_datetime(pitcher_df["birth_date"]).dt.year
pitcher_df["age"] = pitcher_df["season"] - pitcher_df["birth_year"]

print(f"Batter age coverage: {batter_df['age'].notna().mean()*100:.1f}%")
print(f"Pitcher age coverage: {pitcher_df['age'].notna().mean()*100:.1f}%")


# ------------------------------------------------------------------
# Helper: overdispersion with age bucket
# ------------------------------------------------------------------
def compute_overdispersion_with_meta(df, n_col, k_col, meta_cols=None,
                                     min_total_n=100, min_games=50):
    results = []
    extra_cols = meta_cols or []
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
        binom_var_per_game = n_arr * p_hat * (1.0 - p_hat)
        obs_var = np.var(k_arr, ddof=1)
        exp_var = np.mean(binom_var_per_game)
        if exp_var <= 0:
            continue
        disp = obs_var / exp_var
        row = {
            "player_id": pid,
            "season": season,
            "total_n": total_n,
            "n_games": n_games,
            "p_hat": p_hat,
            "overdispersion": disp,
        }
        for c in extra_cols:
            row[c] = grp[c].iloc[0]
        results.append(row)
    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Age split: batter K%
# ------------------------------------------------------------------
print("\n\n" + "="*62)
print("  BATTER K% -- By Age Bucket (season_age = season - birth_year)")
print("="*62)

bk_df = compute_overdispersion_with_meta(
    batter_df, "bat_pa", "bat_k",
    meta_cols=["age"], min_total_n=100, min_games=50
)
bk_df = bk_df[bk_df["age"].notna()]
bk_df["age_bucket"] = pd.cut(bk_df["age"],
                               bins=[0, 25, 28, 32, 99],
                               labels=["young(<=25)", "prime_early(26-28)",
                                       "prime_late(29-32)", "veteran(33+)"])

for bucket, grp in bk_df.groupby("age_bucket", observed=False):
    if len(grp) == 0:
        continue
    d = grp["overdispersion"]
    print(f"  {str(bucket):22s}  N={len(d):4d}  mean={d.mean():.3f}  "
          f"median={d.median():.3f}  P90={d.quantile(0.9):.3f}")


# ------------------------------------------------------------------
# Age split: pitcher K%
# ------------------------------------------------------------------
print("\n" + "="*62)
print("  PITCHER K% (starters) -- By Age Bucket")
print("="*62)

pk_df = compute_overdispersion_with_meta(
    pitcher_df, "pit_bf", "pit_k",
    meta_cols=["age"], min_total_n=100, min_games=15
)
pk_df = pk_df[pk_df["age"].notna()]
pk_df["age_bucket"] = pd.cut(pk_df["age"],
                              bins=[0, 25, 28, 32, 99],
                              labels=["young(<=25)", "prime_early(26-28)",
                                      "prime_late(29-32)", "veteran(33+)"])

for bucket, grp in pk_df.groupby("age_bucket", observed=False):
    if len(grp) == 0:
        continue
    d = grp["overdispersion"]
    print(f"  {str(bucket):22s}  N={len(d):4d}  mean={d.mean():.3f}  "
          f"median={d.median():.3f}  P90={d.quantile(0.9):.3f}")


# ------------------------------------------------------------------
# Pitcher: variable BF confound test
# Use per-start K rate (k/bf) variance vs binomial variance at FIXED n=25
# This strips out the effect of varying bf-per-start
# ------------------------------------------------------------------
print("\n\n" + "="*62)
print("  PITCHER K% ROBUSTNESS: Variable-BF confound check")
print("  Instead of comparing Var(k_i), compare Var(k_i/n_i) to Binomial")
print("  prediction for Var(rate_i) = p*(1-p)/n_i")
print("="*62)

results = []
for (pid, season), grp in pitcher_df.groupby(["player_id", "season"]):
    grp = grp.dropna(subset=["pit_bf", "pit_k"])
    n_arr = grp["pit_bf"].values.astype(float)
    k_arr = grp["pit_k"].values.astype(float)
    total_n = n_arr.sum()
    n_games = len(grp)
    if total_n < 100 or n_games < 15:
        continue
    p_hat = k_arr.sum() / total_n
    if p_hat <= 0 or p_hat >= 1:
        continue

    # Rate per start
    rate_arr = k_arr / n_arr
    # Observed variance of per-start rates
    obs_var_rate = np.var(rate_arr, ddof=1)
    # Binomial expected variance of rate: p(1-p)/n_i per start, average
    exp_var_rate = np.mean(p_hat * (1 - p_hat) / n_arr)
    if exp_var_rate <= 0:
        continue
    disp_rate = obs_var_rate / exp_var_rate
    results.append({"player_id": pid, "season": season, "disp_rate": disp_rate,
                    "p_hat": p_hat, "n_games": n_games})

disp_rate_df = pd.DataFrame(results)
d = disp_rate_df["disp_rate"]
print(f"\n  Rate-based overdispersion (Var(k/bf) / E[Var(k/bf)])")
print(f"  N={len(d):4d}  mean={d.mean():.3f}  median={d.median():.3f}  "
      f"P10={d.quantile(0.1):.3f}  P90={d.quantile(0.9):.3f}")
m = d.median()
if m < 1.1:
    interp = "Near-Binomial => variable BF was the main driver"
elif m < 1.5:
    interp = "Still overdispersed => REAL game-to-game heterogeneity exists"
else:
    interp = "Strongly overdispersed => genuine non-Binomial process"
print(f"  >>> {interp}")

print(f"\n  Compare to count-based median: ~1.252")
print(f"  Rate-based median: {m:.3f}")
if m < 1.10:
    print(f"  CONCLUSION: Variable BF is primary driver of apparent overdispersion in starters.")
else:
    print(f"  CONCLUSION: Overdispersion persists after controlling for variable BF.")


# ------------------------------------------------------------------
# Practical season-model implication
# Compute what the BB model's effective concentration should be
# ------------------------------------------------------------------
print("\n\n" + "="*62)
print("  PRACTICAL IMPLICATION: Season-level posterior width adjustment")
print("  Season model sees N_games * mean_pa = ~500 PA (batter season)")
print("  If the true DGP is Beta-Binomial(n, p, phi), the sufficient")
print("  statistic is not just (total_k, total_pa) but also game-level")
print("  dispersion. The Binomial model underestimates variance by ~disp.")
print("="*62)

# Summary table
print(f"\n  {'Stat':<20} {'disp_med':>10} {'phi_med':>10} {'BB_useful?':>12}")
print(f"  {'-'*55}")

stat_data = [
    ("Batter K%",    1.063,  21.3),
    ("Batter BB%",   1.048,  23.3),
    ("Batter HR/PA", 0.980,  25.9),
    ("Pitcher K%",   1.252,  53.3),   # count-based; rate-based is ~d.median()
    ("Pitcher BB%",  0.943, 146.4),
]

for name, disp, phi in stat_data:
    useful = "YES (moderate)" if 1.1 <= disp < 1.5 else (
             "YES (strong)" if disp >= 1.5 else "NO (<1.1)")
    print(f"  {name:<20} {disp:>10.3f} {phi:>10.1f} {useful:>12}")

# The KEY insight: for season-level model, overdispersion at game level
# inflates the likelihood gradient but NOT the posterior mean.
# Effect = wider posterior than Binomial predicts.
# Concretely: if season has 160 PA games x 3.5 PA/game:
# Binomial posterior concentration ~ n*p*(1-p)
# BB posterior concentration ~ n*p*(1-p) / disp
# => posterior SD widens by sqrt(disp)
print(f"\n  Season posterior SD inflation from game-level overdispersion:")
print(f"  (Season Bayesian model sees total PA, not games — but the")
print(f"   observed rate is noisier than Binomial by sqrt(disp) x)")
for name, disp, phi in stat_data:
    print(f"  {name:<20}  SD inflation = sqrt({disp:.3f}) = {np.sqrt(disp):.3f}x")

print("\n\nExtra analysis complete.")
