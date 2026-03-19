"""Historical validation of the pitcher breakout archetype model."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.data.db import read_sql
from src.data.queries import get_pitcher_efficiency, get_pitcher_fly_ball_data

logging.basicConfig(level=logging.WARNING, format="%(name)-20s %(message)s")
logger = logging.getLogger("pitcher_breakout_bt")


def _pctl(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def _inv_pctl(s: pd.Series) -> pd.Series:
    return 1.0 - s.rank(pct=True, method="average")


def score_pitchers_historical(season: int, min_bf: int = 200) -> pd.DataFrame:
    prior = season - 1
    if prior == 2020:
        prior = 2019

    adv = read_sql(
        f"SELECT pitcher_id, batters_faced, k_pct, bb_pct, "
        f"swstr_pct, csw_pct, zone_pct, chase_pct, contact_pct, "
        f"xwoba_against, barrel_pct_against, hard_hit_pct_against, woba_against "
        f"FROM production.fact_pitching_advanced "
        f"WHERE season = {season} AND batters_faced >= {min_bf}", {},
    )

    trad = read_sql(
        f"SELECT player_id AS pitcher_id, "
        f"SUM(pit_ip) AS ip, SUM(pit_k) AS k_total, "
        f"SUM(pit_bb) AS bb_total, SUM(pit_hr) AS hr_allowed, "
        f"SUM(pit_er) AS earned_runs "
        f"FROM production.fact_player_game_mlb "
        f"WHERE player_role = 'pitcher' AND season = {season} "
        f"GROUP BY player_id HAVING SUM(pit_bf) >= {min_bf}", {},
    )
    trad["era"] = trad["earned_runs"] * 9.0 / trad["ip"].clip(lower=1)
    trad["fip"] = (
        (13 * trad["hr_allowed"] + 3 * trad["bb_total"] - 2 * trad["k_total"])
        / trad["ip"].clip(lower=1)
    ) + 3.20

    fb = get_pitcher_fly_ball_data(season)
    eff = get_pitcher_efficiency(season)

    obs = read_sql(
        f"SELECT fp.pitcher_id, "
        f"SUM(fp.is_whiff::int)::float / NULLIF(SUM(fp.is_swing::int), 0) AS whiff_rate, "
        f"AVG(fp.release_speed) AS avg_velo "
        f"FROM production.fact_pitch fp "
        f"JOIN production.dim_game dg ON fp.game_pk = dg.game_pk "
        f"WHERE dg.season = {season} AND dg.game_type = 'R' "
        f"AND fp.pitch_type IS NOT NULL AND fp.release_speed != 'NaN' "
        f"GROUP BY fp.pitcher_id HAVING SUM(fp.is_swing::int) >= 50", {},
    )

    demo = read_sql(
        f"SELECT player_id AS pitcher_id, player_name AS pitcher_name, "
        f"({season} - EXTRACT(YEAR FROM birth_date))::int AS age "
        f"FROM production.dim_player", {},
    )

    adv_prior = read_sql(
        f"SELECT pitcher_id, k_pct AS k_prior, bb_pct AS bb_prior "
        f"FROM production.fact_pitching_advanced "
        f"WHERE season = {prior} AND batters_faced >= 100", {},
    )

    base = adv.copy()
    base = base.merge(trad[["pitcher_id", "ip", "hr_allowed", "era", "fip"]], on="pitcher_id", how="left")
    if not fb.empty:
        base = base.merge(fb[["pitcher_id", "fly_balls", "home_runs", "hr_per_fb"]], on="pitcher_id", how="left")
    for c in ["fly_balls", "home_runs", "hr_per_fb"]:
        if c not in base.columns:
            base[c] = np.nan
    base = base.merge(obs[["pitcher_id", "whiff_rate", "avg_velo"]], on="pitcher_id", how="left")
    if not eff.empty:
        eff_cols = [c for c in ["pitcher_id", "first_strike_pct", "putaway_rate"] if c in eff.columns]
        base = base.merge(eff[eff_cols], on="pitcher_id", how="left")
    for c in ["first_strike_pct", "putaway_rate"]:
        if c not in base.columns:
            base[c] = np.nan
    base = base.merge(demo, on="pitcher_id", how="left")
    base = base.merge(adv_prior, on="pitcher_id", how="left")
    base["delta_k"] = base["k_pct"] - base["k_prior"]
    base["delta_bb"] = base["bb_pct"] - base["bb_prior"]

    # xFIP
    has_fb = base["fly_balls"].notna() & (base["fly_balls"] > 0) & (base["ip"] > 0)
    if has_fb.any():
        lg_hr_fb = base.loc[has_fb, "home_runs"].sum() / base.loc[has_fb, "fly_balls"].sum()
    else:
        lg_hr_fb = 0.10
    base["xfip"] = base["fip"].copy()
    base.loc[has_fb, "xfip"] = (
        base.loc[has_fb, "fip"]
        - 13 * (base.loc[has_fb, "hr_allowed"] - base.loc[has_fb, "fly_balls"] * lg_hr_fb)
        / base.loc[has_fb, "ip"]
    )
    base["era_xfip_gap"] = base["era"] - base["xfip"]
    base["fip_xfip_gap"] = base["fip"] - base["xfip"]

    # Fill NaN
    fill_cols = [
        "swstr_pct", "csw_pct", "zone_pct", "chase_pct", "contact_pct",
        "xwoba_against", "barrel_pct_against", "first_strike_pct", "putaway_rate",
        "hr_per_fb", "whiff_rate", "avg_velo", "k_pct", "bb_pct",
        "era", "fip", "xfip", "era_xfip_gap", "fip_xfip_gap", "delta_k", "delta_bb",
    ]
    for c in fill_cols:
        if c in base.columns:
            base[c] = base[c].fillna(base[c].median())
    base["age"] = base["age"].fillna(28)

    # Stuff + Command
    stuff = (
        0.25 * _pctl(base["whiff_rate"]) + 0.20 * _pctl(base["swstr_pct"])
        + 0.15 * _pctl(base["csw_pct"]) + 0.15 * _pctl(base["avg_velo"])
        + 0.15 * _inv_pctl(base["xwoba_against"]) + 0.10 * _inv_pctl(base["barrel_pct_against"])
    )
    cmd = (
        0.30 * _inv_pctl(base["bb_pct"]) + 0.20 * _pctl(base["chase_pct"])
        + 0.20 * _pctl(base["zone_pct"]) + 0.15 * _pctl(base["first_strike_pct"])
        + 0.15 * _pctl(base["putaway_rate"])
    )

    # Archetypes
    base["next_step"] = np.sqrt(stuff.clip(0.01) * cmd.clip(0.01))
    s_c_gap = (stuff - cmd).clip(0, 1)
    base["pure_stuff"] = 0.45 * stuff + 0.30 * s_c_gap + 0.25 * _inv_pctl(base["contact_pct"])
    skill_floor = 0.50 * stuff + 0.50 * cmd
    q_gate = ((skill_floor - 0.30) / 0.30).clip(0.15, 1.0)
    base["regression"] = (
        0.30 * _pctl(base["era_xfip_gap"]) + 0.25 * _pctl(base["hr_per_fb"])
        + 0.25 * skill_floor + 0.20 * _pctl(base["fip_xfip_gap"])
    ) * q_gate

    fits = base[["next_step", "pure_stuff", "regression"]]
    base["primary_fit"] = fits.max(axis=1)
    base["breakout_type"] = fits.idxmax(axis=1).map(
        {"next_step": "Next Step", "pure_stuff": "Pure Stuff", "regression": "Regression"}
    )

    # Room to grow
    age_mult = np.clip(1.0 - (base["age"] - 24) / 14.0, 0.15, 1.0)
    era_q = _inv_pctl(base["era"])
    raw_gap = (skill_floor - era_q).clip(0, 1)
    s_gate = (skill_floor / 0.55).clip(0.30, 1.0)
    gap_sc = (raw_gap * s_gate / 0.30).clip(0, 1)
    k_imp = _pctl(base["delta_k"].fillna(0))
    bb_imp = _inv_pctl(base["delta_bb"].fillna(0))
    traj = 0.50 * k_imp + 0.50 * bb_imp
    luck = 0.60 * _pctl(base["hr_per_fb"]) + 0.40 * _pctl(base["era_xfip_gap"])
    raw_rtg = 0.45 * gap_sc + 0.25 * traj + 0.30 * luck
    base["room_to_grow"] = raw_rtg * age_mult
    base["breakout_score"] = base["primary_fit"] * base["room_to_grow"]
    sc_pctl = _pctl(base["breakout_score"])
    base["tier"] = np.where(sc_pctl >= 0.90, "High", np.where(sc_pctl >= 0.75, "Medium", ""))
    base["stuff_score"] = stuff
    base["command_score"] = cmd

    return base


def main() -> None:
    folds = [(2022, 2023), (2023, 2024), (2024, 2025)]
    all_res: list[pd.DataFrame] = []

    for pred, out in folds:
        logger.info("Scoring %d -> %d", pred, out)
        scores = score_pitchers_historical(pred)

        outcomes = read_sql(
            f"SELECT pitcher_id, batters_faced AS bf_next, k_pct AS k_next, bb_pct AS bb_next "
            f"FROM production.fact_pitching_advanced "
            f"WHERE season = {out} AND batters_faced >= 150", {},
        )
        trad_out = read_sql(
            f"SELECT player_id AS pitcher_id, "
            f"SUM(pit_er)*9.0/NULLIF(SUM(pit_ip),0) AS era_next, "
            f"SUM(pit_ip) AS ip_next "
            f"FROM production.fact_player_game_mlb "
            f"WHERE player_role = 'pitcher' AND season = {out} "
            f"GROUP BY player_id HAVING SUM(pit_bf) >= 150", {},
        )
        outcomes = outcomes.merge(trad_out, on="pitcher_id", how="inner")
        m = scores.merge(outcomes, on="pitcher_id", how="inner")
        m["era_delta"] = m["era_next"] - m["era"]
        m["pred"] = pred
        m["out"] = out
        m["broke_out"] = (m["era_delta"] <= -0.75) & (m["era_next"] <= 4.00)
        all_res.append(m)

    res = pd.concat(all_res, ignore_index=True)

    print("=" * 65)
    print("PITCHER BREAKOUT MODEL VALIDATION")
    print(f"Folds: {folds} | Total: {len(res)}")
    print("Breakout def: ERA drops 0.75+ AND outcome ERA <= 4.00")
    print()

    for tier in ["High", "Medium", ""]:
        lab = tier if tier else "None"
        s = res[res["tier"] == tier]
        n = len(s)
        bo = s["broke_out"].mean() * 100
        era_d = s["era_delta"].mean()
        print(f"  {lab:>8}: n={n:>4}  breakout_rate={bo:>5.1f}%  avg_ERA_chg={era_d:+.2f}")

    base_rate = res[res["tier"] == ""]["broke_out"].mean()
    high_rate = res[res["tier"] == "High"]["broke_out"].mean()
    if base_rate > 0:
        print(f"\n  High tier lift: {high_rate / base_rate:.1f}x over baseline")

    r, p = stats.spearmanr(res["breakout_score"], -res["era_delta"])
    print(f"  Spearman (score vs ERA improvement): r={r:.3f}, p={p:.4f}")

    print("\nBy archetype (High tier):")
    for at in ["Next Step", "Pure Stuff", "Regression"]:
        h = res[(res["breakout_type"] == at) & (res["tier"] == "High")]
        if len(h) > 0:
            print(
                f"  {at:>12}: n={len(h):>3}  breakout={h.broke_out.mean() * 100:.1f}%  "
                f"era_chg={h.era_delta.mean():+.2f}"
            )

    print("\nPer-fold:")
    for pred, out in folds:
        fold = res[res["pred"] == pred]
        h = fold[fold["tier"] == "High"]
        print(
            f"  {pred}->{out}: n={len(fold)}  High: n={len(h)}  "
            f"breakout={h.broke_out.mean() * 100:.1f}%  era_chg={h.era_delta.mean():+.2f}"
        )

    print("\nHIGH-TIER HITS:")
    hits = res[(res["tier"] == "High") & (res["broke_out"])].sort_values(
        "breakout_score", ascending=False,
    )
    for _, r2 in hits.head(15).iterrows():
        print(
            f"  {r2.pitcher_name:<22} {int(r2.pred)}->{int(r2.out)}  "
            f"age={int(r2.age)}  ERA {r2.era:.2f}->{r2.era_next:.2f} ({r2.era_delta:+.2f})  "
            f"type={r2.breakout_type}"
        )

    print("\nHIGH-TIER MISSES (top 10):")
    misses = res[(res["tier"] == "High") & (~res["broke_out"])].sort_values(
        "breakout_score", ascending=False,
    )
    for _, r2 in misses.head(10).iterrows():
        print(
            f"  {r2.pitcher_name:<22} {int(r2.pred)}->{int(r2.out)}  "
            f"age={int(r2.age)}  ERA {r2.era:.2f}->{r2.era_next:.2f} ({r2.era_delta:+.2f})  "
            f"type={r2.breakout_type}"
        )

    # Quintile analysis
    print("\nQUINTILE ANALYSIS:")
    res["quintile"] = pd.qcut(
        res["breakout_score"], 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"],
    )
    for q in ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]:
        s = res[res["quintile"] == q]
        bo = s["broke_out"].mean() * 100
        era_d = s["era_delta"].mean()
        print(f"  {q:>10}: n={len(s):>4}  breakout={bo:>5.1f}%  avg_ERA_chg={era_d:+.2f}  avg_age={s.age.mean():.1f}")


if __name__ == "__main__":
    main()
