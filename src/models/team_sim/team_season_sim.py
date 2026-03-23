"""Team-level Monte Carlo season simulator.

For each team, runs N season simulations:
1. Draw injuries for each player from their injury distribution
2. Cascade PA/IP to backups via depth chart
3. Compute team BaseRuns RS from hitter counting projections
4. Compute team RA from pitcher counting projections
5. Convert to wins via Pythagorean expectation

Output: full win distribution per team (mean, p10, p50, p90).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.models.team_sim.depth_cascade import (
    REPLACEMENT_HITTER_WRC_PLUS,
    cascade_hitter_pa,
    cascade_pitcher_ip,
)
from src.models.team_sim.injury_model import (
    build_team_injury_params,
    draw_games_missed,
)

logger = logging.getLogger(__name__)

PYTH_EXP = 1.83
LG_RS_PER_GAME = 4.50   # league average ~2024-2025
LG_RA_PER_GAME = 4.50


def _baseruns_from_counting(
    h: float, bb: float, hbp: float, hr: float, tb: float, pa: float,
) -> float:
    """Compute BaseRuns from counting stats."""
    if pa <= 0:
        return 0.0
    A = h + bb + hbp - hr
    B = (1.4 * tb - 0.6 * h - 3 * hr + 0.1 * (bb + hbp)) * 1.02
    sf = pa * 0.008
    C = (pa - bb - hbp - sf) - h
    D = hr
    if B + C <= 0:
        return D
    return A * B / (B + C) + D


def _pythagorean_wins(rs: float, ra: float, games: int = 162) -> float:
    """Pythagorean win expectation."""
    if rs <= 0 or ra <= 0:
        return games / 2
    wpct = rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)
    return wpct * games


def simulate_team_season(
    team_hitters: pd.DataFrame,
    team_pitchers: pd.DataFrame,
    injury_params: pd.DataFrame,
    n_sims: int = 1000,
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo season simulations for one team.

    Parameters
    ----------
    team_hitters : pd.DataFrame
        Columns: player_id, position, projected_pa, value_score,
        total_h_mean, total_hr_mean, total_bb_mean, total_tb_mean,
        total_hbp_mean, is_starter (bool).
    team_pitchers : pd.DataFrame
        Columns: player_id, role, projected_ip, value_score,
        total_runs_mean.
    injury_params : pd.DataFrame
        Columns: player_id, il_prob, is_pitcher.
    n_sims : int
    random_seed : int

    Returns
    -------
    dict with keys: wins (array), rs (array), ra (array),
        plus summary stats (mean, p10, p50, p90).
    """
    rng = np.random.default_rng(random_seed)

    # Build injury probability lookup
    il_probs = dict(zip(
        injury_params["player_id"].astype(int),
        injury_params["il_prob"],
    ))
    all_pids = set(team_hitters["player_id"]) | set(team_pitchers["player_id"])

    # Pre-draw all injuries: (n_sims, n_players) -> games_missed
    pid_list = sorted(all_pids)
    injury_draws = {}
    for pid in pid_list:
        prob = il_probs.get(pid, 0.50)
        injury_draws[pid] = draw_games_missed(prob, n_sims, rng)

    # Split hitters into starters/bench
    h_starters = team_hitters[team_hitters.get("is_starter", True) == True].copy()
    h_bench = team_hitters[team_hitters.get("is_starter", True) == False].copy()
    if h_starters.empty:
        h_starters = team_hitters.nlargest(9, "value_score").copy()
        h_bench = team_hitters[~team_hitters.index.isin(h_starters.index)].copy()

    # Split pitchers into SP/RP
    if "role" in team_pitchers.columns:
        p_starters = team_pitchers[team_pitchers["role"] == "SP"].nlargest(5, "value_score").copy()
        p_relievers = team_pitchers[~team_pitchers.index.isin(p_starters.index)].copy()
    else:
        p_starters = team_pitchers.nlargest(5, "value_score").copy()
        p_relievers = team_pitchers[~team_pitchers.index.isin(p_starters.index)].copy()

    # Target team totals for normalization (applied AFTER cascade)
    TARGET_TEAM_PA = 6100.0
    TARGET_TEAM_IP = 1458.0

    # Hitter counting stats lookup (raw, unscaled)
    h_stats = {}
    for _, row in team_hitters.iterrows():
        pid = int(row["player_id"])
        pa = float(row.get("projected_pa", row.get("total_pa_mean", 400)))
        h_stats[pid] = {
            "pa": pa,
            "h": float(row.get("total_h_mean", pa * 0.250)),
            "hr": float(row.get("total_hr_mean", pa * 0.030)),
            "bb": float(row.get("total_bb_mean", pa * 0.082)),
            "hbp": float(row.get("total_hbp_mean", pa * 0.008)),
            "tb": float(row.get("total_tb_mean", pa * 0.400)),
        }

    # Pitcher runs lookup (raw, unscaled)
    p_stats = {}
    for _, row in team_pitchers.iterrows():
        pid = int(row["player_id"])
        ip = float(row.get("projected_ip", row.get("projected_ip_mean", 50)))
        p_stats[pid] = {
            "ip": ip,
            "runs": float(row.get("total_runs_mean", ip * LG_RA_PER_GAME / 9)),
        }

    # Simulate
    wins = np.zeros(n_sims)
    rs_arr = np.zeros(n_sims)
    ra_arr = np.zeros(n_sims)

    for sim in range(n_sims):
        # Get this sim's injury draws
        sim_injuries = {pid: int(injury_draws[pid][sim]) for pid in pid_list}

        # --- Offense: cascade PA and compute BaseRuns ---
        adj_h = cascade_hitter_pa(h_starters, h_bench, sim_injuries)

        total_h, total_hr, total_bb, total_hbp, total_tb, total_pa = 0, 0, 0, 0, 0, 0
        for _, row in adj_h.iterrows():
            pid = int(row["player_id"])
            adj_pa = float(row["adjusted_pa"])
            if adj_pa <= 0:
                continue

            if row.get("is_replacement", False):
                # Replacement level: ~75 wRC+ ≈ league_avg * 0.85
                scale = 0.85
                total_pa += adj_pa
                total_h += adj_pa * 0.230 * scale
                total_hr += adj_pa * 0.025 * scale
                total_bb += adj_pa * 0.075
                total_hbp += adj_pa * 0.008
                total_tb += adj_pa * 0.340 * scale
            elif pid in h_stats:
                s = h_stats[pid]
                frac = adj_pa / max(s["pa"], 1)
                total_pa += adj_pa
                total_h += s["h"] * frac
                total_hr += s["hr"] * frac
                total_bb += s["bb"] * frac
                total_hbp += s["hbp"] * frac
                total_tb += s["tb"] * frac

        # Normalize to target team PA: scale all counting stats so
        # total_pa matches a realistic team season (~6100 PA)
        if total_pa > 0:
            pa_scale = TARGET_TEAM_PA / total_pa
            total_h *= pa_scale
            total_hr *= pa_scale
            total_bb *= pa_scale
            total_hbp *= pa_scale
            total_tb *= pa_scale
            total_pa = TARGET_TEAM_PA

        rs = _baseruns_from_counting(total_h, total_bb, total_hbp, total_hr, total_tb, total_pa)

        # --- Pitching: cascade IP and compute RA ---
        adj_p = cascade_pitcher_ip(p_starters, p_relievers, sim_injuries)

        total_runs_allowed, total_ip = 0, 0
        for _, row in adj_p.iterrows():
            pid = int(row["player_id"])
            adj_ip = float(row["adjusted_ip"])
            if adj_ip <= 0:
                continue

            if row.get("is_replacement", False):
                # Replacement-level pitcher: ~5.20 ERA
                total_runs_allowed += adj_ip * 5.20 / 9
                total_ip += adj_ip
            elif pid in p_stats:
                s = p_stats[pid]
                frac = adj_ip / max(s["ip"], 1)
                total_runs_allowed += s["runs"] * frac
                total_ip += adj_ip

        # Normalize pitcher RA to target team IP (~1458)
        if total_ip > 0:
            ip_scale = TARGET_TEAM_IP / total_ip
            ra = total_runs_allowed * ip_scale
        else:
            ra = LG_RA_PER_GAME * 162

        w = _pythagorean_wins(rs, ra)
        wins[sim] = w
        rs_arr[sim] = rs
        ra_arr[sim] = ra

    return {
        "wins": wins,
        "rs": rs_arr,
        "ra": ra_arr,
        "wins_mean": float(np.mean(wins)),
        "wins_median": float(np.median(wins)),
        "wins_p10": float(np.percentile(wins, 10)),
        "wins_p90": float(np.percentile(wins, 90)),
        "wins_std": float(np.std(wins)),
        "rs_mean": float(np.mean(rs_arr)),
        "ra_mean": float(np.mean(ra_arr)),
    }


def simulate_all_teams(
    team_rosters: dict[str, dict],
    n_sims: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run season simulations for all 30 teams.

    Parameters
    ----------
    team_rosters : dict[str, dict]
        Keyed by team_abbr. Each value has:
        - "hitters": DataFrame with counting projections
        - "pitchers": DataFrame with counting projections
        - "injury_params": DataFrame with il_prob per player
    n_sims : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per team with win distribution columns.
    """
    t0 = time.time()
    rows = []

    for i, (abbr, data) in enumerate(sorted(team_rosters.items())):
        seed = random_seed + i * 1000
        result = simulate_team_season(
            team_hitters=data["hitters"],
            team_pitchers=data["pitchers"],
            injury_params=data["injury_params"],
            n_sims=n_sims,
            random_seed=seed,
        )
        result["team_abbr"] = abbr
        rows.append(result)

    df = pd.DataFrame([{
        "team_abbr": r["team_abbr"],
        "sim_wins_mean": r["wins_mean"],
        "sim_wins_median": r["wins_median"],
        "sim_wins_p10": r["wins_p10"],
        "sim_wins_p90": r["wins_p90"],
        "sim_wins_std": r["wins_std"],
        "sim_rs_mean": r["rs_mean"],
        "sim_ra_mean": r["ra_mean"],
    } for r in rows])

    elapsed = time.time() - t0
    logger.info(
        "Team sim: %d teams x %d seasons in %.1fs (mean wins range: %.0f-%.0f)",
        len(df), n_sims, elapsed,
        df["sim_wins_mean"].min(), df["sim_wins_mean"].max(),
    )
    return df
