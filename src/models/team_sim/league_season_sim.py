"""Full-league Monte Carlo season simulator.

Simulates all 2,430 games of an MLB season N times using the actual
schedule.  Every game produces exactly 1 win and 1 loss, enforcing
the zero-sum constraint (total wins = 2,430 across all teams).

v2: Poisson game-score simulation with SP rotation and park factors.
Each game simulates actual runs scored via Poisson draws adjusted for
the matchup (offense vs that day's opposing SP), park factor, and
home advantage.

v3: Injury cascading.  When ``team_rosters`` are provided, each sim
iteration draws injuries for all 30 teams, cascades PA/IP to backups,
and recomputes RS/RA via BaseRuns before running that season's games.
Injury-prone / thin-depth teams get wider win distributions while
maintaining zero-sum.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.models.team_sim.depth_cascade import (
    cascade_hitter_pa,
    cascade_pitcher_ip,
)
from src.models.team_sim.injury_model import draw_games_missed
from src.models.team_sim.team_season_sim import _baseruns_from_counting

logger = logging.getLogger(__name__)

HOME_ADVANTAGE_RUNS = 0.30  # ~0.3 runs/game home edge (~54% home win rate)
LG_AVG_RPG = 4.50

# 2H weighting: weight the second half of the season more heavily.
# Teams that improved at the deadline carry that momentum into next year.
# Backtested: 40/60 matches full-year accuracy while capturing trade-deadline signal.
H1_WEIGHT = 0.35
H2_WEIGHT = 0.65


def _pythagorean_wpct(rpg: float, ra_per_game: float) -> float:
    """PythagenPat win% from RS/RA per game."""
    if rpg <= 0 or ra_per_game <= 0:
        return 0.500
    exp = (rpg + ra_per_game) ** 0.287
    return rpg ** exp / (rpg ** exp + ra_per_game ** exp)


# ── Team strength (for backwards compat / simple mode) ──────────────

def compute_2h_weighted_rs_ra(
    game_results: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute 40/60 first-half/second-half weighted RS/RA per team.

    Teams that improved after the trade deadline carry that signal
    into next-season projections more heavily than early-season play.

    Parameters
    ----------
    game_results : pd.DataFrame
        Output of ``get_game_results()``.
    season : int
        Season to compute.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, rpg, ra_per_game (2H-weighted).
    """
    sg = game_results[
        (game_results["season"] == season) & (game_results["game_type"] == "R")
    ].sort_values(["game_date", "game_pk"])

    team_games: list[dict] = []
    for _, row in sg.iterrows():
        team_games.append({
            "team_id": int(row["home_team_id"]),
            "rs": int(row["home_runs"]),
            "ra": int(row["away_runs"]),
            "game_date": row["game_date"],
        })
        team_games.append({
            "team_id": int(row["away_team_id"]),
            "rs": int(row["away_runs"]),
            "ra": int(row["home_runs"]),
            "game_date": row["game_date"],
        })

    tg = pd.DataFrame(team_games).sort_values(["team_id", "game_date"])
    tg["game_num"] = tg.groupby("team_id").cumcount() + 1

    rows = []
    for tid, grp in tg.groupby("team_id"):
        mid = len(grp) // 2
        h1 = grp[grp["game_num"] <= mid]
        h2 = grp[grp["game_num"] > mid]
        rpg = H1_WEIGHT * h1["rs"].mean() + H2_WEIGHT * h2["rs"].mean()
        ra = H1_WEIGHT * h1["ra"].mean() + H2_WEIGHT * h2["ra"].mean()
        rows.append({"team_id": int(tid), "rpg": rpg, "ra_per_game": ra})

    return pd.DataFrame(rows)


def build_team_strength(
    team_stats: pd.DataFrame,
    waa_deltas: dict[int, float] | None = None,
    talent_scores: dict[int, float] | None = None,
    team_history: dict[int, dict] | None = None,
    blend_proj_wt: float = 0.35,
    regress_pct: float = 0.40,
    waa_dampen: float = 0.55,
    waa_cap: float = 8.0,
    talent_weight: float = 0.20,
) -> dict[int, float]:
    """Build team strength dict from RS/RA data + talent composite.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Columns: team_id, rpg, ra_per_game.
    waa_deltas : dict, optional
        team_id -> raw WAA delta from roster turnover.
    talent_scores : dict, optional
        team_id -> talent composite (0-1).
    team_history : dict, optional
        team_id -> {"avg_rpg_3yr": float, "avg_ra_3yr": float}.
        3-year team averages for team-specific regression.
    talent_weight : float
        Weight on talent composite (default 0.20).

    Returns dict[team_id, Pythagorean win%].
    """
    lg_rpg = team_stats["rpg"].mean()
    lg_ra = team_stats["ra_per_game"].mean()
    obs_wt = 1.0 - blend_proj_wt

    strength = {}
    for _, row in team_stats.iterrows():
        tid = int(row["team_id"])
        rpg = row["rpg"]
        ra = row["ra_per_game"]

        # Team-specific regression target: blend team's 3-year avg with
        # league mean.  Teams with a multi-year track record regress
        # toward their own baseline, not the league.
        if team_history and tid in team_history:
            th = team_history[tid]
            # 60% team history + 40% league mean
            regress_rpg = 0.60 * th["avg_rpg_3yr"] + 0.40 * lg_rpg
            regress_ra = 0.60 * th["avg_ra_3yr"] + 0.40 * lg_ra
        else:
            regress_rpg = lg_rpg
            regress_ra = lg_ra

        proj_rpg = rpg * (1 - regress_pct) + regress_rpg * regress_pct
        proj_ra = ra * (1 - regress_pct) + regress_ra * regress_pct

        blend_rpg = blend_proj_wt * proj_rpg + obs_wt * rpg
        blend_ra = blend_proj_wt * proj_ra + obs_wt * ra

        base_wpct = _pythagorean_wpct(blend_rpg, blend_ra)

        # WAA roster adjustment
        if waa_deltas and tid in waa_deltas:
            waa_wins = max(-waa_cap, min(waa_cap, waa_deltas[tid] * waa_dampen))
            base_wpct += waa_wins / 162.0

        strength[tid] = max(0.250, min(0.750, base_wpct))

    # Blend with talent composite
    if talent_scores:
        talent_wpcts = {
            tid: 0.400 + score * 0.200
            for tid, score in talent_scores.items()
        }
        rs_wt = 1.0 - talent_weight
        for tid in strength:
            if tid in talent_wpcts:
                strength[tid] = (
                    rs_wt * strength[tid]
                    + talent_weight * talent_wpcts[tid]
                )
                strength[tid] = max(0.250, min(0.750, strength[tid]))

    return strength


def compute_team_history(
    game_results: pd.DataFrame,
    season: int,
    n_years: int = 3,
) -> dict[int, dict]:
    """Compute multi-year team RS/RA averages for regression targets.

    Parameters
    ----------
    game_results : pd.DataFrame
        Output of get_game_results().
    season : int
        Most recent completed season.
    n_years : int
        Number of years to average (default 3).

    Returns
    -------
    dict[int, {"avg_rpg_3yr": float, "avg_ra_3yr": float}]
    """
    rg = game_results[game_results["game_type"] == "R"].copy()

    # Use last n_years, skip 2020 (COVID)
    valid_seasons = [s for s in range(season - n_years + 1, season + 1) if s != 2020]

    team_totals: dict[int, list[tuple[float, float, int]]] = {}
    for s in valid_seasons:
        sg = rg[rg["season"] == s]
        if sg.empty:
            continue
        for side, rs_col, ra_col in [
            ("home_team_id", "home_runs", "away_runs"),
            ("away_team_id", "away_runs", "home_runs"),
        ]:
            for tid, grp in sg.groupby(side):
                tid = int(tid)
                team_totals.setdefault(tid, []).append(
                    (grp[rs_col].sum(), grp[ra_col].sum(), len(grp))
                )

    result = {}
    for tid, seasons_data in team_totals.items():
        total_rs = sum(d[0] for d in seasons_data)
        total_ra = sum(d[1] for d in seasons_data)
        total_g = sum(d[2] for d in seasons_data)
        if total_g > 0:
            result[tid] = {
                "avg_rpg_3yr": total_rs / total_g,
                "avg_ra_3yr": total_ra / total_g,
            }

    return result


# ── Injury cascade helpers ─────────────────────────────────────────

# Standard full-season targets for normalization (same as team_season_sim)
_TARGET_TEAM_PA = 6100.0
_TARGET_TEAM_IP = 1458.0
_LG_RA_PER_9 = 4.50


def _precompute_roster_data(
    team_rosters: dict[int, dict],
) -> dict[int, dict]:
    """Pre-extract rate stats and roster splits for all teams.

    Runs once before the sim loop so per-iteration work is minimal.

    Parameters
    ----------
    team_rosters : dict[int, dict]
        team_id -> {"hitters": DataFrame, "pitchers": DataFrame,
                     "injury_params": DataFrame}

    Returns
    -------
    dict[int, dict]
        team_id -> pre-extracted roster data (starters, bench, rates, etc.)
    """
    precomp: dict[int, dict] = {}
    for tid, data in team_rosters.items():
        hitters = data["hitters"]
        pitchers = data["pitchers"]

        # --- Hitter starter/bench split ---
        if "is_starter" in hitters.columns:
            h_starters = hitters[hitters["is_starter"] == True].copy()
            h_bench = hitters[hitters["is_starter"] != True].copy()
        else:
            h_starters = hitters.nlargest(9, "value_score").copy()
            h_bench = hitters[~hitters.index.isin(h_starters.index)].copy()
        if h_starters.empty:
            h_starters = hitters.nlargest(9, "value_score").copy()
            h_bench = hitters[~hitters.index.isin(h_starters.index)].copy()

        # --- Pitcher SP/RP split ---
        if "role" in pitchers.columns:
            p_starters = pitchers[pitchers["role"] == "SP"].nlargest(
                5, "value_score"
            ).copy()
            p_relievers = pitchers[
                ~pitchers.index.isin(p_starters.index)
            ].copy()
        else:
            p_starters = pitchers.nlargest(5, "value_score").copy()
            p_relievers = pitchers[
                ~pitchers.index.isin(p_starters.index)
            ].copy()

        # --- Hitter counting stat lookup ---
        h_stats: dict[int, dict] = {}
        for _, row in hitters.iterrows():
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

        # --- Pitcher counting stat lookup ---
        p_stats: dict[int, dict] = {}
        for _, row in pitchers.iterrows():
            pid = int(row["player_id"])
            ip = float(
                row.get("projected_ip", row.get("projected_ip_mean", 50))
            )
            p_stats[pid] = {
                "ip": ip,
                "runs": float(
                    row.get("total_runs_mean", ip * _LG_RA_PER_9 / 9)
                ),
            }

        # --- Replacement-level rates (from bench quality) ---
        bench_h = hitters.sort_values("value_score").head(
            max(len(hitters) - 9, 3)
        )
        if len(bench_h) > 0 and "total_h_mean" in bench_h.columns:
            bh_pa = bench_h["projected_pa"].clip(1)
            repl_h_rate = float((bench_h["total_h_mean"] / bh_pa).mean())
            repl_hr_rate = float((bench_h["total_hr_mean"] / bh_pa).mean())
            repl_tb_rate = float((bench_h["total_tb_mean"] / bh_pa).mean())
        else:
            repl_h_rate, repl_hr_rate, repl_tb_rate = 0.200, 0.020, 0.290

        bench_p = pitchers.sort_values("value_score").head(
            max(len(pitchers) - 5, 3)
        )
        if len(bench_p) > 0 and "total_runs_mean" in bench_p.columns:
            bp_ip = bench_p["projected_ip"].clip(1)
            repl_ra_per_9 = float(
                (bench_p["total_runs_mean"] / bp_ip * 9).mean()
            )
        else:
            repl_ra_per_9 = 5.20

        # --- IL prob lookup ---
        injury_params = data["injury_params"]
        il_probs: dict[int, float] = dict(
            zip(
                injury_params["player_id"].astype(int),
                injury_params["il_prob"],
            )
        )
        all_pids = sorted(
            set(hitters["player_id"].astype(int))
            | set(pitchers["player_id"].astype(int))
        )

        precomp[tid] = {
            "h_starters": h_starters,
            "h_bench": h_bench,
            "p_starters": p_starters,
            "p_relievers": p_relievers,
            "h_stats": h_stats,
            "p_stats": p_stats,
            "repl_h_rate": repl_h_rate,
            "repl_hr_rate": repl_hr_rate,
            "repl_tb_rate": repl_tb_rate,
            "repl_ra_per_9": repl_ra_per_9,
            "il_probs": il_probs,
            "all_pids": all_pids,
        }

    return precomp


def _draw_team_profiles_for_sim(
    all_teams: list[int],
    precomp: dict[int, dict],
    rng: np.random.Generator,
) -> dict[int, dict]:
    """Draw one sim's injuries for all teams and return adjusted profiles.

    For each team: draw injuries -> cascade PA/IP -> BaseRuns RS ->
    pitcher RA -> RS/G, RA/G.

    Parameters
    ----------
    all_teams : list[int]
        Team IDs participating in the sim.
    precomp : dict
        Output of ``_precompute_roster_data()``.
    rng : np.random.Generator

    Returns
    -------
    dict[int, {"rs_per_game": float, "ra_per_game": float}]
    """
    profiles: dict[int, dict] = {}

    for tid in all_teams:
        if tid not in precomp:
            profiles[tid] = {"rs_per_game": 4.5, "ra_per_game": 4.5}
            continue

        pc = precomp[tid]

        # Draw injuries for every player on this team
        sim_injuries: dict[int, int] = {}
        for pid in pc["all_pids"]:
            prob = pc["il_probs"].get(pid, 0.50)
            # Single draw (n_sims=1), take scalar
            sim_injuries[pid] = int(draw_games_missed(prob, 1, rng)[0])

        # --- Offense: cascade PA, compute BaseRuns ---
        adj_h = cascade_hitter_pa(
            pc["h_starters"], pc["h_bench"], sim_injuries,
        )

        tot_h, tot_hr, tot_bb, tot_hbp, tot_tb, tot_pa = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        h_stats = pc["h_stats"]
        for _, row in adj_h.iterrows():
            pid = int(row["player_id"])
            adj_pa = float(row["adjusted_pa"])
            if adj_pa <= 0:
                continue
            if row.get("is_replacement", False):
                tot_pa += adj_pa
                tot_h += adj_pa * pc["repl_h_rate"]
                tot_hr += adj_pa * pc["repl_hr_rate"]
                tot_bb += adj_pa * 0.075
                tot_hbp += adj_pa * 0.008
                tot_tb += adj_pa * pc["repl_tb_rate"]
            elif pid in h_stats:
                s = h_stats[pid]
                frac = adj_pa / max(s["pa"], 1)
                tot_pa += adj_pa
                tot_h += s["h"] * frac
                tot_hr += s["hr"] * frac
                tot_bb += s["bb"] * frac
                tot_hbp += s["hbp"] * frac
                tot_tb += s["tb"] * frac

        if tot_pa > 0:
            pa_scale = _TARGET_TEAM_PA / tot_pa
            tot_h *= pa_scale
            tot_hr *= pa_scale
            tot_bb *= pa_scale
            tot_hbp *= pa_scale
            tot_tb *= pa_scale
            tot_pa = _TARGET_TEAM_PA

        rs_season = _baseruns_from_counting(
            tot_h, tot_bb, tot_hbp, tot_hr, tot_tb, tot_pa,
        )

        # --- Pitching: cascade IP, compute RA ---
        adj_p = cascade_pitcher_ip(
            pc["p_starters"], pc["p_relievers"], sim_injuries,
        )

        tot_runs_allowed, tot_ip = 0.0, 0.0
        p_stats = pc["p_stats"]
        for _, row in adj_p.iterrows():
            pid = int(row["player_id"])
            adj_ip = float(row["adjusted_ip"])
            if adj_ip <= 0:
                continue
            if row.get("is_replacement", False):
                tot_runs_allowed += adj_ip * pc["repl_ra_per_9"] / 9
                tot_ip += adj_ip
            elif pid in p_stats:
                s = p_stats[pid]
                frac = adj_ip / max(s["ip"], 1)
                tot_runs_allowed += s["runs"] * frac
                tot_ip += adj_ip

        if tot_ip > 0:
            ip_scale = _TARGET_TEAM_IP / tot_ip
            ra_season = tot_runs_allowed * ip_scale
        else:
            ra_season = _LG_RA_PER_9 * 162

        profiles[tid] = {
            "rs_per_game": rs_season / 162,
            "ra_per_game": ra_season / 162,
        }

    return profiles


# ── Poisson league simulator ────────────────────────────────────────

def simulate_league_season(
    schedule: pd.DataFrame,
    team_strength: dict[int, float] | None = None,
    team_profiles: dict[int, dict] | None = None,
    team_rosters: dict[int, dict] | None = None,
    rotations: dict[int, list[float]] | None = None,
    venue_factors: dict[int, float] | None = None,
    n_sims: int = 1_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate a full MLB season N times.

    Uses Poisson game-score simulation when team_profiles are provided
    (v2), otherwise falls back to Bernoulli Log5 (v1).

    When ``team_rosters`` is provided alongside ``team_profiles`` (v3),
    each sim iteration draws injuries for all 30 teams, cascades PA/IP
    to backups, and recomputes team RS/RA via BaseRuns before simulating
    that season's games.  This gives injury-prone / thin-depth teams
    wider win distributions while maintaining zero-sum.

    Parameters
    ----------
    schedule : pd.DataFrame
        Columns: home_team_id, away_team_id.  Optionally venue_id.
    team_strength : dict, optional
        team_id -> win% (v1 Bernoulli mode).
    team_profiles : dict, optional
        team_id -> {"rs_per_game", "ra_per_game"} (v2 Poisson mode).
        Used as static fallback when ``team_rosters`` is None, or as
        the default for teams missing from ``team_rosters``.
    team_rosters : dict, optional
        team_id -> {"hitters": DataFrame, "pitchers": DataFrame,
        "injury_params": DataFrame}.  When provided, enables per-sim
        injury cascading in Poisson mode.
    rotations : dict, optional
        team_id -> [ra9_sp1, ra9_sp2, ..., ra9_sp5].
    venue_factors : dict, optional
        venue_id -> park run factor (1.0 = neutral).
    n_sims : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per team with win distribution + run totals.
    """
    use_poisson = team_profiles is not None or team_rosters is not None
    use_injury_cascade = team_rosters is not None and use_poisson

    if not use_poisson and team_strength is None:
        raise ValueError("Provide either team_strength or team_profiles")

    # Ensure we have base team_profiles even in injury mode (fallback)
    if team_profiles is None and team_rosters is not None:
        team_profiles = {}  # will be overridden per-sim by injury draws

    t0 = time.time()
    rng = np.random.default_rng(random_seed)

    # Pre-extract roster data for injury cascade (runs once)
    roster_precomp: dict[int, dict] | None = None
    if use_injury_cascade:
        roster_precomp = _precompute_roster_data(team_rosters)
        logger.info(
            "Injury cascade enabled: %d teams with roster data",
            len(roster_precomp),
        )

    home_ids = schedule["home_team_id"].values.astype(int)
    away_ids = schedule["away_team_id"].values.astype(int)
    venue_ids = (
        schedule["venue_id"].values.astype(int)
        if "venue_id" in schedule.columns
        else np.zeros(len(schedule), dtype=int)
    )
    n_games = len(schedule)

    all_teams = sorted(set(home_ids) | set(away_ids))
    tid_to_idx = {tid: i for i, tid in enumerate(all_teams)}
    n_teams = len(all_teams)

    # Track wins and runs
    wins = np.zeros((n_sims, n_teams), dtype=np.int32)
    rs_total = np.zeros((n_sims, n_teams), dtype=np.float64)
    ra_total = np.zeros((n_sims, n_teams), dtype=np.float64)

    if use_poisson:
        # ── Poisson mode (v2/v3) ──
        # Track rotation position per team: resets each sim
        lg_ra9 = LG_AVG_RPG

        for sim in range(n_sims):
            # Reset rotation counters each season
            rotation_idx: dict[int, int] = {tid: 0 for tid in all_teams}

            # v3: Draw injuries and recompute team RS/RA for this sim
            if use_injury_cascade:
                sim_profiles = _draw_team_profiles_for_sim(
                    all_teams, roster_precomp, rng,
                )
            else:
                sim_profiles = team_profiles

            for g in range(n_games):
                h_tid = home_ids[g]
                a_tid = away_ids[g]

                h_prof = sim_profiles.get(h_tid, {"rs_per_game": 4.5, "ra_per_game": 4.5})
                a_prof = sim_profiles.get(a_tid, {"rs_per_game": 4.5, "ra_per_game": 4.5})

                # Park factor
                pf = 1.0
                if venue_factors and venue_ids[g] in venue_factors:
                    pf = venue_factors[venue_ids[g]]
                pf_sqrt = np.sqrt(pf)

                # SP rotation: get today's SP RA/9
                if rotations:
                    h_rot = rotations.get(h_tid)
                    a_rot = rotations.get(a_tid)
                    if h_rot:
                        h_sp_ra9 = h_rot[rotation_idx[h_tid] % len(h_rot)]
                        rotation_idx[h_tid] += 1
                    else:
                        h_sp_ra9 = h_prof["ra_per_game"] / 9 * 9  # team avg
                    if a_rot:
                        a_sp_ra9 = a_rot[rotation_idx[a_tid] % len(a_rot)]
                        rotation_idx[a_tid] += 1
                    else:
                        a_sp_ra9 = a_prof["ra_per_game"] / 9 * 9
                else:
                    h_sp_ra9 = h_prof["ra_per_game"]
                    a_sp_ra9 = a_prof["ra_per_game"]

                # Offense-defense interaction (odds ratio method):
                # λ_home = home_RS/G × (away_pitching / lg_avg) × park × home_adj
                # away_pitching factor: if SP has 3.0 RA/9 vs 4.5 lg avg,
                # opposing offense is suppressed by 3.0/4.5 = 0.67
                lambda_home = (
                    h_prof["rs_per_game"]
                    * (a_sp_ra9 / lg_ra9)
                    * pf_sqrt
                    + HOME_ADVANTAGE_RUNS / 2
                )
                lambda_away = (
                    a_prof["rs_per_game"]
                    * (h_sp_ra9 / lg_ra9)
                    * pf_sqrt
                    - HOME_ADVANTAGE_RUNS / 2
                )

                # Floor lambdas
                lambda_home = max(1.5, lambda_home)
                lambda_away = max(1.5, lambda_away)

                # Draw game scores
                h_runs = rng.poisson(lambda_home)
                a_runs = rng.poisson(lambda_away)

                # Tie? Simulate extras (coin flip weighted by strength)
                if h_runs == a_runs:
                    h_runs += rng.poisson(0.5)
                    a_runs += rng.poisson(0.5)
                    if h_runs == a_runs:
                        # Still tied, give it to better lambda
                        if rng.random() < lambda_home / (lambda_home + lambda_away):
                            h_runs += 1
                        else:
                            a_runs += 1

                h_idx = tid_to_idx[h_tid]
                a_idx = tid_to_idx[a_tid]

                rs_total[sim, h_idx] += h_runs
                rs_total[sim, a_idx] += a_runs
                ra_total[sim, h_idx] += a_runs
                ra_total[sim, a_idx] += h_runs

                if h_runs > a_runs:
                    wins[sim, h_idx] += 1
                else:
                    wins[sim, a_idx] += 1

    else:
        # ── Bernoulli mode (v1 fallback) ──
        game_probs = np.empty(n_games, dtype=np.float64)
        for g in range(n_games):
            p_home = team_strength.get(home_ids[g], 0.500)
            p_away = team_strength.get(away_ids[g], 0.500)
            num = p_home * (1 - p_away)
            den = num + p_away * (1 - p_home)
            raw = num / den if den > 0 else 0.5
            game_probs[g] = max(0.05, min(0.95, raw + 0.040))

        draws = rng.random((n_sims, n_games))
        home_wins = draws < game_probs[np.newaxis, :]
        home_idx = np.array([tid_to_idx[tid] for tid in home_ids])
        away_idx = np.array([tid_to_idx[tid] for tid in away_ids])

        for g in range(n_games):
            mask = home_wins[:, g]
            wins[mask, home_idx[g]] += 1
            wins[~mask, away_idx[g]] += 1

    elapsed = time.time() - t0
    mode = "Poisson+injury" if use_injury_cascade else (
        "Poisson" if use_poisson else "Bernoulli"
    )
    logger.info(
        "League sim (%s): %d sims x %d games in %.1fs",
        mode, n_sims, n_games, elapsed,
    )

    # Summarize
    rows = []
    for i, tid in enumerate(all_teams):
        w = wins[:, i]
        row = {
            "team_id": tid,
            "sim_wins_mean": float(np.mean(w)),
            "sim_wins_median": float(np.median(w)),
            "sim_wins_std": float(np.std(w)),
            "sim_wins_p10": float(np.percentile(w, 10)),
            "sim_wins_p90": float(np.percentile(w, 90)),
            "sim_wins_min": int(np.min(w)),
            "sim_wins_max": int(np.max(w)),
            "playoff_pct": float(np.mean(w >= 88)),
        }
        if use_poisson:
            rs = rs_total[:, i]
            ra = ra_total[:, i]
            row["sim_rs_mean"] = float(np.mean(rs))
            row["sim_ra_mean"] = float(np.mean(ra))
            row["sim_run_diff"] = float(np.mean(rs - ra))
        rows.append(row)

    return pd.DataFrame(rows)
