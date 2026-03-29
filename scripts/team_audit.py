"""Team Audit Report — smell-test validation for every team.

For each of the 30 teams, shows:
  - Team TDD score breakdown (talent/projections/ELO components)
  - Starting lineup: 9 hitters with position, diamond rating, tool grades, wRC+
  - Rotation: top 5 SP with diamond rating, tool grades, K%, BB%, ERA
  - Bullpen: relievers with role, diamond rating, tool grades
  - Who boosts the team most / least
  - Flagged concerns (low-PA players driving grades, big wRC+/diamond mismatches)
"""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np

DASHBOARD_DIR = __import__("pathlib").Path(
    "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
)

# Load all data
h = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
p = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")
tr = pd.read_parquet(DASHBOARD_DIR / "team_rankings.parquet")
tp = pd.read_parquet(DASHBOARD_DIR / "team_profiles.parquet")
pt = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")
roster = pd.read_parquet(DASHBOARD_DIR / "roster.parquet")

# Filter roster to active + IL
active_ids = set(
    roster.loc[
        roster["roster_status"].isin(["active", "il_7", "il_10", "il_15", "il_60"]),
        "player_id",
    ]
)

# Map players to teams
h_team = h.merge(
    pt[["player_id", "team_abbr"]].rename(columns={"player_id": "batter_id"}),
    on="batter_id", how="left",
)
h_team = h_team[h_team["batter_id"].isin(active_ids)]

p_team = p.merge(
    pt[["player_id", "team_abbr"]].rename(columns={"player_id": "pitcher_id"}),
    on="pitcher_id", how="left",
)
p_team = p_team[p_team["pitcher_id"].isin(active_ids)]

# Sort teams by rank
teams_sorted = tr.sort_values("rank")

for _, team_row in teams_sorted.iterrows():
    abbr = team_row["abbreviation"]
    tdd = team_row.get("tdd_score", 0)
    tier = team_row.get("tier", "")
    rank = int(team_row.get("rank", 0))
    proj_w = team_row.get("projected_wins", 81)
    elo = team_row.get("composite_elo", 1500)
    elo = f"{elo:.0f}" if pd.notna(elo) else "N/A"

    # Get team profile
    tp_row = tp[tp["abbreviation"] == abbr]
    lu_d = tp_row["lineup_diamond"].iloc[0] if not tp_row.empty and "lineup_diamond" in tp_row.columns else 0
    rot_d = tp_row["rotation_diamond"].iloc[0] if not tp_row.empty and "rotation_diamond" in tp_row.columns else 0
    bp_d = tp_row["bullpen_diamond"].iloc[0] if not tp_row.empty and "bullpen_diamond" in tp_row.columns else 0
    def_s = tp_row["defense_score"].iloc[0] if not tp_row.empty else 0.5
    fld_g = tp_row["team_fielding_grade"].iloc[0] if not tp_row.empty and "team_fielding_grade" in tp_row.columns else 50

    print()
    print("=" * 130)
    print(f"#{rank:2d}  {abbr}  TDD: {tdd:.1f}  {tier}  |  Proj W: {proj_w:.0f}  ELO: {elo}")
    print(f"     Lineup: {lu_d:.1f}  Rotation: {rot_d:.1f}  Bullpen: {bp_d:.1f}  Defense: {def_s:.2f}  Fielding Grade: {fld_g:.0f}")
    print("-" * 130)

    # === LINEUP ===
    team_h = h_team[h_team["team_abbr"] == abbr].copy()
    if team_h.empty:
        print("  [No hitters found for this team]")
    else:
        team_h = team_h.sort_values("tools_rating", ascending=False)
        print(f"  LINEUP ({len(team_h)} hitters on active roster)")
        print(f"  {'Name':22s} {'Pos':3s} {'Age':>3s} {'DR':>4s}  {'H':>3s} {'P':>3s} {'Sp':>3s} {'F':>3s} {'D':>3s}  {'PA':>4s} {'wRC+':>5s} {'ProjwRC+':>8s} {'Flag':s}")

        for i, (_, r) in enumerate(team_h.iterrows()):
            wrc = r.get("wrc_plus", np.nan)
            wrc_s = f"{wrc:.0f}" if pd.notna(wrc) else "--"
            proj_wrc = r.get("projected_wrc_plus_mean", np.nan)
            proj_s = f"{proj_wrc:.0f}" if pd.notna(proj_wrc) else "--"
            pa = int(r.get("pa", 0))

            # Flags
            flags = []
            if pa < 300:
                flags.append(f"LOW PA({pa})")
            if pd.notna(wrc) and pd.notna(r.get("tools_rating")):
                if wrc > 140 and r["tools_rating"] < 6.0:
                    flags.append("UNDERGRADED")
                if wrc < 90 and r["tools_rating"] > 6.5:
                    flags.append("OVERGRADED")
            flag_s = " ".join(flags)

            marker = "*" if i < 9 else " "  # top 9 = starters
            print(f"  {marker}{r['batter_name']:21s} {r['position']:3s} {int(r['age']):3d} {r['tools_rating']:4.1f}  {int(r['grade_hit']):3d} {int(r['grade_power']):3d} {int(r['grade_speed']):3d} {int(r['grade_fielding']):3d} {int(r['grade_discipline']):3d}  {pa:4d} {wrc_s:>5s} {proj_s:>8s} {flag_s}")

        # Top booster / worst drag
        if len(team_h) >= 2:
            best = team_h.iloc[0]
            worst_starter = team_h.head(9).iloc[-1] if len(team_h) >= 9 else team_h.iloc[-1]
            avg_dr = team_h.head(9)["tools_rating"].mean()
            print(f"  >> Lineup avg DR: {avg_dr:.1f} | Best: {best['batter_name']} ({best['tools_rating']:.1f}) | Floor: {worst_starter['batter_name']} ({worst_starter['tools_rating']:.1f})")

    # === ROTATION ===
    team_sp = p_team[(p_team["team_abbr"] == abbr) & (p_team["role"] == "SP")].copy()
    team_sp = team_sp.sort_values("tools_rating", ascending=False)
    print()
    print(f"  ROTATION ({len(team_sp)} SP)")
    print(f"  {'Name':22s} {'Age':>3s} {'DR':>4s}  {'St':>3s} {'Cm':>3s} {'Du':>3s}  {'K%':>5s} {'BB%':>5s} {'ERA':>5s} {'Stuff+':>6s} {'Flag':s}")

    for _, r in team_sp.head(7).iterrows():
        era = f"{r['projected_era']:.2f}" if pd.notna(r.get("projected_era")) else "--"
        sp = f"{r['arsenal_stuff_plus']:.0f}" if pd.notna(r.get("arsenal_stuff_plus")) else "--"
        flags = []
        if pd.notna(r.get("projected_era")) and r["projected_era"] < 3.0 and r["tools_rating"] < 6.0:
            flags.append("UNDERGRADED")
        if pd.notna(r.get("projected_era")) and r["projected_era"] > 5.0 and r["tools_rating"] > 6.5:
            flags.append("OVERGRADED")
        flag_s = " ".join(flags)
        print(f"  {r['pitcher_name']:22s} {int(r['age']):3d} {r['tools_rating']:4.1f}  {int(r['grade_stuff']):3d} {int(r['grade_command']):3d} {int(r['grade_durability']):3d}  {r['k_pct']:.1%} {r['bb_pct']:.1%} {era:>5s} {sp:>6s} {flag_s}")

    if len(team_sp) >= 2:
        avg_sp = team_sp.head(5)["tools_rating"].mean()
        print(f"  >> Rotation avg DR: {avg_sp:.1f}")

    # === BULLPEN ===
    team_rp = p_team[(p_team["team_abbr"] == abbr) & (p_team["role"] == "RP")].copy()
    team_rp = team_rp.sort_values("tools_rating", ascending=False)
    print()
    print(f"  BULLPEN ({len(team_rp)} RP)")
    print(f"  {'Name':22s} {'Role':4s} {'Age':>3s} {'DR':>4s}  {'St':>3s} {'Cm':>3s}  {'K%':>5s} {'BB%':>5s}")

    for _, r in team_rp.head(8).iterrows():
        role_d = r.get("role_detail", "RP")
        if pd.isna(role_d):
            role_d = "RP"
        print(f"  {r['pitcher_name']:22s} {str(role_d):4s} {int(r['age']):3d} {r['tools_rating']:4.1f}  {int(r['grade_stuff']):3d} {int(r['grade_command']):3d}  {r['k_pct']:.1%} {r['bb_pct']:.1%}")

    if len(team_rp) >= 2:
        avg_rp = team_rp.head(5)["tools_rating"].mean()
        print(f"  >> Bullpen avg DR: {avg_rp:.1f}")

    print()

print("=" * 130)
print("AUDIT COMPLETE")
