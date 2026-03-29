"""Full-league Monte Carlo season simulator.

Simulates all 2,430 games of an MLB season N times using the actual
schedule.  Every game produces exactly 1 win and 1 loss, enforcing
the zero-sum constraint (total wins = 2,430 across all teams).

v2: Poisson game-score simulation with SP rotation and park factors.
Each game simulates actual runs scored via Poisson draws adjusted for
the matchup (offense vs that day's opposing SP), park factor, and
home advantage.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOME_ADVANTAGE_RUNS = 0.30  # ~0.3 runs/game home edge (~54% home win rate)
LG_AVG_RPG = 4.50

# 2H weighting: weight the second half of the season more heavily.
# Teams that improved at the deadline carry that momentum into next year.
# Backtested: 40/60 matches full-year MAE while capturing trade-deadline signal.
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


# ── Team offense/pitching profiles for Poisson sim ──────────────────

def build_team_profiles(
    team_stats: pd.DataFrame,
    waa_deltas: dict[int, float] | None = None,
    blend_proj_wt: float = 0.35,
    regress_pct: float = 0.40,
    waa_dampen: float = 0.55,
    waa_cap: float = 8.0,
) -> dict[int, dict]:
    """Build per-team offense RS/G and pitching RA/G profiles.

    Returns dict[team_id, {"rs_per_game": float, "ra_per_game": float}].
    """
    lg_rpg = team_stats["rpg"].mean()
    lg_ra = team_stats["ra_per_game"].mean()
    obs_wt = 1.0 - blend_proj_wt

    profiles = {}
    for _, row in team_stats.iterrows():
        tid = int(row["team_id"])
        rpg = row["rpg"]
        ra = row["ra_per_game"]

        proj_rpg = rpg * (1 - regress_pct) + lg_rpg * regress_pct
        proj_ra = ra * (1 - regress_pct) + lg_ra * regress_pct

        blend_rpg = blend_proj_wt * proj_rpg + obs_wt * rpg
        blend_ra = blend_proj_wt * proj_ra + obs_wt * ra

        # Apply WAA as RS/RA shift (~10 runs per win, ~162 games)
        if waa_deltas and tid in waa_deltas:
            waa_wins = max(-waa_cap, min(waa_cap, waa_deltas[tid] * waa_dampen))
            # Split WAA evenly between offense boost and pitching boost
            run_adj = waa_wins * 10.0 / 162.0  # runs per game
            blend_rpg += run_adj / 2
            blend_ra -= run_adj / 2

        profiles[tid] = {
            "rs_per_game": max(2.5, blend_rpg),
            "ra_per_game": max(2.5, blend_ra),
        }

    return profiles


# ── SP rotation builder ─────────────────────────────────────────────

def build_rotations(
    sp_data: pd.DataFrame,
    roster: pd.DataFrame,
    lg_ra9: float = LG_AVG_RPG,
) -> dict[int, list[float]]:
    """Build 5-man rotation RA/9 arrays per team.

    Parameters
    ----------
    sp_data : pd.DataFrame
        SP counting sim projections.  Needs pitcher_id, role,
        projected_ip_mean, total_runs_mean.
    roster : pd.DataFrame
        Active roster with player_id, org_id (team_id).
    lg_ra9 : float
        League average RA/9 for replacement-level fill.

    Returns
    -------
    dict[int, list[float]]
        team_id -> list of 5 RA/9 values (best to worst).
    """
    sp = sp_data[sp_data["role"] == "SP"].copy()
    sp["ra9"] = sp["total_runs_mean"] / sp["projected_ip_mean"].clip(1) * 9

    # Map pitcher to team via roster
    sp = sp.merge(
        roster[["player_id", "org_id"]].rename(columns={"org_id": "team_id"}),
        left_on="pitcher_id", right_on="player_id", how="inner",
    )

    # Sort by projected IP desc (best starters pitch most)
    sp = sp.sort_values(["team_id", "projected_ip_mean"], ascending=[True, False])

    repl_ra9 = lg_ra9 * 1.15  # replacement SP ~ 15% worse than average

    rotations: dict[int, list[float]] = {}
    for tid, grp in sp.groupby("team_id"):
        ra9_list = grp["ra9"].head(5).tolist()
        # Pad to 5 with replacement level
        while len(ra9_list) < 5:
            ra9_list.append(repl_ra9)
        rotations[int(tid)] = ra9_list

    return rotations


# ── Poisson league simulator ────────────────────────────────────────

def simulate_league_season(
    schedule: pd.DataFrame,
    team_strength: dict[int, float] | None = None,
    team_profiles: dict[int, dict] | None = None,
    rotations: dict[int, list[float]] | None = None,
    venue_factors: dict[int, float] | None = None,
    n_sims: int = 1_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate a full MLB season N times.

    Uses Poisson game-score simulation when team_profiles are provided
    (v2), otherwise falls back to Bernoulli Log5 (v1).

    Parameters
    ----------
    schedule : pd.DataFrame
        Columns: home_team_id, away_team_id.  Optionally venue_id.
    team_strength : dict, optional
        team_id -> win% (v1 Bernoulli mode).
    team_profiles : dict, optional
        team_id -> {"rs_per_game", "ra_per_game"} (v2 Poisson mode).
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
    use_poisson = team_profiles is not None

    if not use_poisson and team_strength is None:
        raise ValueError("Provide either team_strength or team_profiles")

    t0 = time.time()
    rng = np.random.default_rng(random_seed)

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
        # ── Poisson mode (v2) ──
        # Track rotation position per team: resets each sim
        lg_ra9 = LG_AVG_RPG

        for sim in range(n_sims):
            # Reset rotation counters each season
            rotation_idx: dict[int, int] = {tid: 0 for tid in all_teams}

            for g in range(n_games):
                h_tid = home_ids[g]
                a_tid = away_ids[g]

                h_prof = team_profiles.get(h_tid, {"rs_per_game": 4.5, "ra_per_game": 4.5})
                a_prof = team_profiles.get(a_tid, {"rs_per_game": 4.5, "ra_per_game": 4.5})

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
    logger.info(
        "League sim: %d sims x %d games in %.1fs",
        n_sims, n_games, elapsed,
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
