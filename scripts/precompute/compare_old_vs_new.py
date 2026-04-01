"""Compare old batter sim vs new lineup sim on the same completed games."""
import sys
sys.path.insert(0, "C:/Users/kekoa/Documents/data_analytics/player_profiles")

import numpy as np
import pandas as pd
from src.models.game_sim.lineup_simulator import simulate_lineup_game

DD = "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"

# Load old predictions with actuals
gp = pd.read_parquet(f"{DD}/game_props.parquet")
final = gp[
    (gp["game_status"] == "final")
    & gp["actual"].notna()
    & (gp["player_type"] == "batter")
].copy()
if "line_mid" in final.columns:
    final["line"] = final["line"].fillna(final["line_mid"])
    final["p_over"] = final["p_over"].fillna(final["p_over_mid"])
final = final[final["p_over"].notna()].copy()

print(f"Old predictions with actuals: {len(final)} batter props")
print(f"Dates: {sorted(final['game_date'].unique())}")
print(f"Games: {final['game_pk'].nunique()}")

# Load posteriors
hk = np.load(f"{DD}/hitter_k_samples.npz")
hbb = np.load(f"{DD}/hitter_bb_samples.npz")
hhr = np.load(f"{DD}/hitter_hr_samples.npz")
pk = np.load(f"{DD}/pitcher_k_samples.npz")
pbb = np.load(f"{DD}/pitcher_bb_samples.npz")
phr = np.load(f"{DD}/pitcher_hr_samples.npz")
bf = pd.read_parquet(f"{DD}/bf_priors.parquet")

# Build game info: group by (game_pk, team)
games_info = {}
for gpk in final["game_pk"].unique():
    gdf = final[final["game_pk"] == gpk]
    for team in gdf["team"].unique():
        team_df = gdf[gdf["team"] == team]
        games_info[(gpk, team)] = {
            "players": team_df["player_id"].unique().tolist(),
            "opponent": team_df["opponent"].iloc[0],
            "props": team_df,
        }

print(f"Team-game sides: {len(games_info)}")

# Run new sim on each and compare
results = []
n_sims = 10000
skipped = processed = 0

for (gpk, team), info in games_info.items():
    players = info["players"]
    props = info["props"]

    # Find opposing pitcher from pitcher-type rows
    opp_pitchers = gp[
        (gp["game_pk"] == gpk)
        & (gp["player_type"] == "pitcher")
        & (gp["team"] == info["opponent"])
    ]
    if opp_pitchers.empty:
        skipped += 1
        continue
    opp_pid = int(opp_pitchers["player_id"].iloc[0])
    opp_pid_str = str(opp_pid)

    if opp_pid_str not in pk:
        skipped += 1
        continue

    # Build lineup posteriors
    lineup_k, lineup_bb, lineup_hr = [], [], []
    valid_slots = {}
    for i, pid in enumerate(players[:9]):
        bid = str(pid)
        if bid in hk and bid in hbb and bid in hhr:
            lineup_k.append(hk[bid])
            lineup_bb.append(hbb[bid])
            lineup_hr.append(hhr[bid])
            valid_slots[i] = pid
        else:
            lineup_k.append(np.full(500, 0.226))
            lineup_bb.append(np.full(500, 0.082))
            lineup_hr.append(np.full(500, 0.031))

    while len(lineup_k) < 9:
        lineup_k.append(np.full(500, 0.226))
        lineup_bb.append(np.full(500, 0.082))
        lineup_hr.append(np.full(500, 0.031))

    if not valid_slots:
        skipped += 1
        continue

    starter_k = float(np.mean(pk[opp_pid_str]))
    starter_bb = float(np.mean(pbb[opp_pid_str])) if opp_pid_str in pbb else 0.082
    starter_hr = float(np.mean(phr[opp_pid_str])) if opp_pid_str in phr else 0.031

    bf_row = bf[bf["pitcher_id"] == opp_pid]
    bf_mu = float(bf_row.iloc[0]["mu_bf"]) if len(bf_row) > 0 else 22.0
    bf_sigma = float(bf_row.iloc[0]["sigma_bf"]) if len(bf_row) > 0 else 4.5

    try:
        lsim = simulate_lineup_game(
            batter_k_rate_samples=lineup_k,
            batter_bb_rate_samples=lineup_bb,
            batter_hr_rate_samples=lineup_hr,
            starter_k_rate=starter_k,
            starter_bb_rate=starter_bb,
            starter_hr_rate=starter_hr,
            starter_bf_mu=bf_mu,
            starter_bf_sigma=bf_sigma,
            n_sims=n_sims,
            random_seed=42 + gpk % 10000,
        )
    except Exception:
        skipped += 1
        continue

    for slot_idx, pid in valid_slots.items():
        player_props = props[props["player_id"] == pid]
        summary = lsim.batter_summary(slot_idx)

        for _, row in player_props.iterrows():
            stat = row["stat"]
            stat_key = stat.lower()
            line = row["line"]
            actual = row["actual"]
            old_p = row["p_over"]

            if stat_key not in summary:
                continue

            over_df = lsim.batter_over_probs(slot_idx, stat_key, [line])
            new_p = float(over_df["p_over"].iloc[0]) if not over_df.empty else None

            if new_p is not None:
                results.append({
                    "game_pk": gpk,
                    "player_id": pid,
                    "stat": stat,
                    "line": line,
                    "actual": actual,
                    "over_hit": actual > line,
                    "old_p_over": old_p,
                    "new_p_over": new_p,
                })

    processed += 1
    if processed % 20 == 0:
        print(f"  Processed {processed} team-games...")

print(f"\nProcessed: {processed}, skipped: {skipped}")
print(f"Total comparisons: {len(results)}")

df = pd.DataFrame(results)

print()
print("=" * 70)
print("HEAD-TO-HEAD: Same games, same lines, >= 63% confidence")
print("=" * 70)

old_strong = df[df["old_p_over"] >= 0.63]
new_strong = df[df["new_p_over"] >= 0.63]

oh = int(old_strong["over_hit"].sum())
nh = int(new_strong["over_hit"].sum())
print(f"Old model >= 63%: {oh}/{len(old_strong)} ({oh/len(old_strong)*100:.1f}%)" if len(old_strong) else "Old: n/a")
print(f"New model >= 63%: {nh}/{len(new_strong)} ({nh/len(new_strong)*100:.1f}%)" if len(new_strong) else "New: n/a")

print()
print("Per stat (>= 63% confidence):")
for stat in ["H", "K", "TB", "HRR", "BB", "R", "RBI"]:
    os = old_strong[old_strong["stat"] == stat]
    ns = new_strong[new_strong["stat"] == stat]
    o_str = f"{int(os['over_hit'].sum())}/{len(os)} ({os['over_hit'].mean()*100:.1f}%)" if len(os) else "n/a"
    n_str = f"{int(ns['over_hit'].sum())}/{len(ns)} ({ns['over_hit'].mean()*100:.1f}%)" if len(ns) else "n/a"
    print(f"  {stat:>3s}: Old {o_str:>20s}  |  New {n_str:>20s}")

print()
print("Calibration (all stats):")
for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]:
    om = df[(df["old_p_over"] >= lo) & (df["old_p_over"] < hi)]
    nm = df[(df["new_p_over"] >= lo) & (df["new_p_over"] < hi)]
    o_str = f"{om['over_hit'].mean()*100:.1f}% (n={len(om)})" if len(om) >= 5 else f"n/a (n={len(om)})"
    n_str = f"{nm['over_hit'].mean()*100:.1f}% (n={len(nm)})" if len(nm) >= 5 else f"n/a (n={len(nm)})"
    print(f"  {lo:.0%}-{hi:.0%}:  Old {o_str:>20s}  |  New {n_str:>20s}")
