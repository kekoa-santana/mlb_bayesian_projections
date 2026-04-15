"""Precompute: Team ELO, profiles, power rankings, depth chart, roster."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.team")


# Columns copied from team_power_rankings into team_rankings.parquet so the
# dashboard can read a single unified file.
_POWER_MERGE_COLS = [
    "team_id",
    "power_score", "power_rank", "power_tier",
    "wins_component", "form_component", "depth_component",
    "trajectory_component", "elo_component",
    "projected_wins", "schedule_adjusted_wins", "sim_wins",
    "games_played", "preseason_win_pct", "posterior_win_pct",
    "pythag_win_pct", "avg_opp_elo",
    "breakout_count", "regression_count", "net_trajectory",
]


def _merge_power_into_team_rankings(power_rankings_df: pd.DataFrame) -> None:
    """Merge power rankings columns into team_rankings.parquet.

    The dashboard reads ``team_rankings.parquet`` and renders its
    ``rank``, ``tdd_score``, and ``tier`` columns.  Power rankings are
    the authoritative source of those values, so this helper copies
    the merged columns and sorts by the new rank.  Overwrites any
    prior copies so re-runs do not produce ``*_pw`` duplicates.
    """
    tr_path = DASHBOARD_DIR / "team_rankings.parquet"
    if not tr_path.exists():
        return
    tr = pd.read_parquet(tr_path)

    merge_cols = [c for c in _POWER_MERGE_COLS if c in power_rankings_df.columns]
    drop_cols = [c for c in merge_cols if c != "team_id" and c in tr.columns]
    tr = tr.drop(columns=drop_cols, errors="ignore")
    tr = tr.merge(power_rankings_df[merge_cols], on="team_id", how="left")

    if "power_score" in tr.columns:
        ps = tr["power_score"]
        ps_min, ps_max = ps.min(), ps.max()
        if ps_max - ps_min > 1e-9:
            tr["tdd_score"] = (
                1.0 + (ps - ps_min) / (ps_max - ps_min) * 9.0
            ).round(1)
        tr["rank"] = tr["power_rank"]
        tr["tier"] = tr["power_tier"]
        tr = tr.sort_values("rank").reset_index(drop=True)

    tr.to_parquet(tr_path, index=False)
    logger.info("Merged power rankings into team_rankings.parquet: %d teams", len(tr))


def run_team_elo() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Compute team ELO ratings.

    Returns
    -------
    elo_current, elo_preseason, elo_history  (any may be None on failure)
    """
    logger.info("=" * 60)
    logger.info("Computing team ELO ratings...")

    try:
        from src.data.team_queries import (
            get_game_results,
            get_team_info,
            get_venue_run_factors,
        )
        from src.models.team_elo import (
            apply_roster_talent_adjustment,
            compute_elo_history,
            get_current_ratings,
            project_preseason_elo,
        )

        elo_games = get_game_results()
        elo_venue = get_venue_run_factors()
        elo_team_info = get_team_info()
        logger.info("ELO input: %d games, %d venue factors", len(elo_games), len(elo_venue))

        elo_ratings, elo_history = compute_elo_history(elo_games, elo_venue)

        # Current (end-of-2025) ratings
        elo_current = get_current_ratings(elo_ratings, elo_team_info)
        elo_current.to_parquet(DASHBOARD_DIR / "team_elo.parquet", index=False)
        logger.info("Saved team_elo.parquet: %d teams", len(elo_current))

        # Pre-season 2026 ratings (regressed, then roster-adjusted)
        elo_preseason_ratings = project_preseason_elo(elo_ratings)

        # Apply roster talent adjustment if data is available
        roster_path = DASHBOARD_DIR / "roster.parquet"
        pt_path = DASHBOARD_DIR / "player_teams.parquet"
        hr_path = DASHBOARD_DIR / "hitters_rankings.parquet"
        pr_path = DASHBOARD_DIR / "pitchers_rankings.parquet"
        if roster_path.exists() and pt_path.exists() and hr_path.exists() and pr_path.exists():
            _cur_roster = pd.read_parquet(roster_path)
            _prev_roster = pd.read_parquet(pt_path)
            _h_rank = pd.read_parquet(hr_path)
            _p_rank = pd.read_parquet(pr_path)
            logger.info(
                "Roster talent adjustment: %d current, %d previous, %d hitters, %d pitchers",
                len(_cur_roster), len(_prev_roster), len(_h_rank), len(_p_rank),
            )
            apply_roster_talent_adjustment(
                elo_preseason_ratings,
                current_roster=_cur_roster,
                previous_roster=_prev_roster,
                hitter_quality=_h_rank,
                pitcher_quality=_p_rank,
            )
        else:
            logger.warning(
                "Skipping roster talent adjustment -- missing parquets: %s",
                [p.name for p in [roster_path, pt_path, hr_path, pr_path] if not p.exists()],
            )

        elo_preseason = get_current_ratings(elo_preseason_ratings, elo_team_info)
        elo_preseason.to_parquet(DASHBOARD_DIR / "team_elo_preseason.parquet", index=False)
        logger.info("Saved team_elo_preseason.parquet: %d teams (roster-adjusted)", len(elo_preseason))

        # History (last 3 seasons for dashboard charts)
        recent_history = elo_history[elo_history["season"] >= FROM_SEASON - 2]
        recent_history.to_parquet(DASHBOARD_DIR / "team_elo_history.parquet", index=False)
        logger.info("Saved team_elo_history.parquet: %d rows", len(recent_history))

        return elo_current, elo_preseason, elo_history

    except Exception:
        logger.exception("Failed to compute team ELO")
        return None, None, None


def run_series_elo(*, from_season: int = FROM_SEASON) -> None:
    """Compute series-based ELO ratings.

    Detects series from game results, computes a single ELO rating
    per team based on series outcomes, and saves current + preseason
    projections to parquet.
    """
    logger.info("=" * 60)
    logger.info("Computing series-based ELO ratings...")

    try:
        from src.data.team_queries import get_series_results, get_team_info
        from src.models.series_elo import (
            compute_series_elo,
            detect_series,
            get_current_ratings,
            project_preseason_elo,
        )

        games = get_series_results(include_postseason=True)
        team_info = get_team_info()
        logger.info("Series ELO input: %d games", len(games))

        # Detect series from game-level data
        series_data = detect_series(games)
        logger.info(
            "Series detected: %d total, %d regular season, %d postseason",
            len(series_data),
            int((series_data["game_type"] == "R").sum()),
            int((series_data["game_type"] != "R").sum()),
        )

        # Compute series ELO
        ratings, history = compute_series_elo(series_data)

        # Pre-season 2026 projection (regressed)
        preseason_ratings = project_preseason_elo(ratings)
        preseason = get_current_ratings(preseason_ratings, team_info)
        preseason.to_parquet(
            DASHBOARD_DIR / "team_series_elo_preseason.parquet", index=False,
        )
        logger.info("Saved team_series_elo_preseason.parquet: %d teams", len(preseason))

        # Log top 10
        for _, row in preseason.head(10).iterrows():
            logger.info(
                "  %2d. %-4s  %.1f  (%d series)",
                row["series_rank"],
                row.get("abbreviation", "???"),
                row["series_mu"],
                row.get("series_count", 0),
            )

    except Exception:
        logger.exception("Failed to compute series ELO")


def run_team_profiles(
    *,
    elo_history: pd.DataFrame | None = None,
    elo_current: pd.DataFrame | None = None,
    pitcher_roles_df: pd.DataFrame | None = None,
    from_season: int = FROM_SEASON,
) -> None:
    """Build team profiles and rankings."""
    logger.info("=" * 60)
    logger.info("Building team profiles and rankings...")

    try:
        from src.models.team_profiles import build_all_team_profiles
        from src.models.team_rankings import compute_roster_waa_delta, rank_teams

        team_profiles = build_all_team_profiles(
            season=from_season,
            projection_season=from_season + 1,
            elo_history=elo_history,
            pitcher_roles_df=pitcher_roles_df,
        )
        team_profiles.to_parquet(DASHBOARD_DIR / "team_profiles.parquet", index=False)
        logger.info("Saved team_profiles.parquet: %d teams", len(team_profiles))

        # Roster WAA delta: compare prior season rosters to current
        waa_deltas = None
        try:
            roster_path = DASHBOARD_DIR / "roster.parquet"
            cur_roster = pd.read_parquet(roster_path) if roster_path.exists() else None
            waa_deltas = compute_roster_waa_delta(
                from_season, from_season + 1, current_roster=cur_roster,
            )
            logger.info(
                "Roster WAA deltas: %d teams, range [%.1f, %.1f]",
                len(waa_deltas),
                waa_deltas["waa_delta"].min(),
                waa_deltas["waa_delta"].max(),
            )
        except Exception:
            logger.warning("Could not compute roster WAA deltas", exc_info=True)

        # Compute 2H-weighted RS/RA for observed component
        obs_2h = None
        try:
            from src.models.team_sim.league_season_sim import compute_2h_weighted_rs_ra
            from src.data.team_queries import get_game_results as _get_gr_rt
            obs_2h = compute_2h_weighted_rs_ra(_get_gr_rt(), from_season)
            logger.info("Using 2H-weighted RS/RA for rank_teams")
        except Exception:
            logger.warning("Could not compute 2H RS/RA for rank_teams", exc_info=True)

        team_rankings_df = rank_teams(
            team_profiles,
            elo_ratings=elo_current,
            waa_deltas=waa_deltas,
            observed_rs_ra=obs_2h,
        )
        team_rankings_df.to_parquet(DASHBOARD_DIR / "team_rankings.parquet", index=False)
        logger.info("Saved team_rankings.parquet: %d teams", len(team_rankings_df))

        # Log top 10
        for _, row in team_rankings_df.head(10).iterrows():
            logger.info(
                "  %2d. %-4s  %.3f (%s)  Proj W: %.0f",
                row["rank"], row.get("abbreviation", "???"),
                row["composite_score"], row["tier"],
                row.get("projected_wins", 81),
            )

    except Exception:
        logger.exception("Failed to build team profiles/rankings")


def run_team_power(
    *,
    league_sim_df: pd.DataFrame | None = None,
) -> None:
    """Build projection-integrated power rankings.

    Parameters
    ----------
    league_sim_df : pd.DataFrame or None
        League season sim results (from ``run_league_sim``).  Passed through
        to ``build_power_rankings`` so it can use sim wins without reading
        ``league_sim.parquet`` from disk.
    """
    logger.info("=" * 60)
    logger.info("Building projection-integrated power rankings...")

    try:
        _elo_pre = None
        if (DASHBOARD_DIR / "team_elo_preseason.parquet").exists():
            _elo_pre = pd.read_parquet(DASHBOARD_DIR / "team_elo_preseason.parquet")

        _profiles = None
        if (DASHBOARD_DIR / "team_profiles.parquet").exists():
            _profiles = pd.read_parquet(DASHBOARD_DIR / "team_profiles.parquet")

        _cur_roster_pw = pd.read_parquet(DASHBOARD_DIR / "roster.parquet") \
            if (DASHBOARD_DIR / "roster.parquet").exists() else None
        _h_proj_pw = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet") \
            if (DASHBOARD_DIR / "hitter_projections.parquet").exists() else None
        _p_proj_pw = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet") \
            if (DASHBOARD_DIR / "pitcher_projections.parquet").exists() else None

        if all(x is not None for x in [
            _elo_pre, _profiles, _cur_roster_pw, _h_proj_pw, _p_proj_pw,
        ]):
            from src.models.team_rankings import build_power_rankings
            from src.models.in_season_wins import load_current_team_records

            # Current-season records for in-season wins blend
            try:
                _team_records = load_current_team_records(FROM_SEASON + 1)
            except Exception:
                logger.warning("Could not load current team records", exc_info=True)
                _team_records = None

            power_rankings_df = build_power_rankings(
                elo_ratings=_elo_pre,
                profiles=_profiles,
                current_roster=_cur_roster_pw,
                hitter_projections=_h_proj_pw,
                pitcher_projections=_p_proj_pw,
                league_sim_df=league_sim_df,
                team_records=_team_records,
            )
            # Save power rankings standalone (backward compat)
            power_rankings_df.to_parquet(
                DASHBOARD_DIR / "team_power_rankings.parquet", index=False,
            )

            _merge_power_into_team_rankings(power_rankings_df)

            logger.info("Saved team_power_rankings.parquet: %d teams", len(power_rankings_df))
            for _, row in power_rankings_df.head(10).iterrows():
                logger.info(
                    "  %2d. %-4s  %.3f (%s)  Wins=%.2f Form=%.2f Depth=%.2f Traj=%.2f  B:%d R:%d",
                    row["power_rank"], row["abbreviation"],
                    row["power_score"], row["power_tier"],
                    row["wins_component"], row["form_component"],
                    row["depth_component"], row["trajectory_component"],
                    row["breakout_count"], row["regression_count"],
                )
        else:
            missing = [
                name for name, val in [
                    ("elo_preseason", _elo_pre), ("profiles", _profiles),
                    ("roster", _cur_roster_pw),
                    ("hitter_projections", _h_proj_pw),
                    ("pitcher_projections", _p_proj_pw),
                ] if val is None
            ]
            logger.warning("Skipping power rankings -- missing data: %s", missing)

    except Exception:
        logger.exception("Failed to build power rankings")


def run_league_sim(
    *,
    n_sims: int = 1000,
    random_seed: int = 42,
    from_season: int = FROM_SEASON,
) -> pd.DataFrame | None:
    """Run full-league Bernoulli season sim on actual schedule.

    Feeds the same team strengths (Pythagorean + WAA) used by rank_teams
    into a zero-sum league simulator.  The sim mean wins REPLACE
    projected_wins in team_rankings.parquet so there is one source of
    truth.  Also writes playoff probabilities and win distributions.

    Returns
    -------
    result : pd.DataFrame or None
        League sim results (one row per team), or None on failure.
    """
    logger.info("=" * 60)
    logger.info("Running league season simulator (%d sims)...", n_sims)

    try:
        from src.models.team_sim.league_season_sim import (
            build_team_strength,
            compute_2h_weighted_rs_ra,
            compute_team_history,
            simulate_league_season,
        )
        from src.models.team_sim.injury_model import build_team_injury_params
        from src.models.team_rankings import compute_roster_waa_delta, _assign_tier

        tr_path = DASHBOARD_DIR / "team_rankings.parquet"
        tp_path = DASHBOARD_DIR / "team_profiles.parquet"
        if not tr_path.exists() or not tp_path.exists():
            logger.warning("Skipping league sim -- need team_rankings + team_profiles first")
            return None

        tr = pd.read_parquet(tr_path)
        tp = pd.read_parquet(tp_path)

        # Build team strength from the same RS/RA data rank_teams uses
        # Need rpg + ra_per_game from profiles, plus WAA
        waa_deltas = None
        roster_path = DASHBOARD_DIR / "roster.parquet"
        if roster_path.exists():
            try:
                cur_roster = pd.read_parquet(roster_path)
                waa_df = compute_roster_waa_delta(
                    from_season, from_season + 1, current_roster=cur_roster,
                )
                waa_deltas = dict(zip(waa_df["team_id"], waa_df["waa_delta"]))
            except Exception:
                logger.warning("Could not compute WAA for league sim", exc_info=True)

        # Use 2H-weighted RS/RA (40% 1H + 60% 2H) for team strength.
        # Teams that improved at the deadline carry that signal forward.
        from src.data.team_queries import get_game_results as _get_gr
        _game_results = _get_gr()
        _2h_stats = compute_2h_weighted_rs_ra(_game_results, from_season)

        # Build talent composite per team: 45% offense + 35% pitching + 20% defense
        _talent_scores: dict[int, float] = {}
        for _, _t in tr.iterrows():
            _talent_scores[int(_t["team_id"])] = (
                0.45 * _t.get("offense_score", 0.5)
                + 0.35 * _t.get("pitching_score", 0.5)
                + 0.20 * _t.get("defense_score", 0.5)
            )
        logger.info(
            "Talent composites: min=%.3f, max=%.3f",
            min(_talent_scores.values()), max(_talent_scores.values()),
        )

        # Compute 3-year team RS/RA history for team-specific regression
        _team_hist = compute_team_history(_game_results, from_season, n_years=3)
        logger.info("Team history: %d teams with 3-year baselines", len(_team_hist))

        if not _2h_stats.empty:
            logger.info("Using 2H-weighted RS/RA + talent blend + team-specific regression")
            team_strength = build_team_strength(
                _2h_stats, waa_deltas=waa_deltas, talent_scores=_talent_scores,
                team_history=_team_hist,
            )
        else:
            logger.info("Falling back to full-season RS/RA from profiles")
            team_strength = build_team_strength(
                tp, waa_deltas=waa_deltas, talent_scores=_talent_scores,
                team_history=_team_hist,
            )
        # ── Breakout model trajectory adjustment ──
        # Uses the full XGBoost breakout model scores (Power Surge, Diamond
        # in the Rough, Stuff Dominant, ERA Correction, Command Leap)
        # weighted by projected PA/IP.  Each team gets a net breakout upside
        # adjustment relative to league average.
        _TRAJ_SCALE = 1.5   # wins per unit of above-average breakout score
        _TRAJ_CAP = 3.0     # max ±wins adjustment
        try:
            _hr = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
            _pr = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")
            _hcs = pd.read_parquet(DASHBOARD_DIR / "hitter_counting_sim.parquet")
            _pcs = pd.read_parquet(DASHBOARD_DIR / "pitcher_counting_sim.parquet")
            _rost = pd.read_parquet(DASHBOARD_DIR / "roster.parquet")

            # Hitter: PA-weighted breakout score
            _hb = _hr[["batter_id", "breakout_score"]].merge(
                _hcs[["batter_id", "total_pa_mean"]], on="batter_id", how="left",
            ).merge(
                _rost[["player_id", "org_id"]], left_on="batter_id",
                right_on="player_id", how="left",
            )
            _hb["weighted_brk"] = (
                _hb["breakout_score"] * _hb["total_pa_mean"].fillna(200) / 600
            )

            # Pitcher: IP-weighted breakout score
            _pb = _pr[["pitcher_id", "breakout_score"]].merge(
                _pcs[["pitcher_id", "projected_ip_mean"]], on="pitcher_id", how="left",
            ).merge(
                _rost[["player_id", "org_id"]], left_on="pitcher_id",
                right_on="player_id", how="left",
            )
            _pb["weighted_brk"] = (
                _pb["breakout_score"] * _pb["projected_ip_mean"].fillna(50) / 180
            )

            # Sum per team
            team_brk: dict[int, float] = {}
            for tid, grp in _hb.groupby("org_id"):
                team_brk[int(tid)] = grp["weighted_brk"].sum()
            for tid, grp in _pb.groupby("org_id"):
                team_brk[int(tid)] = team_brk.get(int(tid), 0) + grp["weighted_brk"].sum()

            # Convert to win adjustment relative to league average
            lg_avg_brk = np.mean(list(team_brk.values())) if team_brk else 0
            for tid in team_strength:
                raw = (team_brk.get(tid, lg_avg_brk) - lg_avg_brk) * _TRAJ_SCALE
                capped = max(-_TRAJ_CAP, min(_TRAJ_CAP, raw))
                team_strength[tid] += capped / 162.0
                team_strength[tid] = max(0.250, min(0.750, team_strength[tid]))
            logger.info(
                "Breakout trajectory applied: lg_avg=%.2f, range [%.2f, %.2f], "
                "win adj range [%+.1f, %+.1f]",
                lg_avg_brk, min(team_brk.values()), max(team_brk.values()),
                (min(team_brk.values()) - lg_avg_brk) * _TRAJ_SCALE,
                (max(team_brk.values()) - lg_avg_brk) * _TRAJ_SCALE,
            )
        except Exception:
            logger.warning("Could not compute breakout trajectory", exc_info=True)

        logger.info(
            "Team strengths: min=%.3f, max=%.3f, spread=%.3f",
            min(team_strength.values()), max(team_strength.values()),
            max(team_strength.values()) - min(team_strength.values()),
        )

        # Fetch schedule for projection season
        target_season = from_season + 1
        schedule = None

        # Try dim_game first (for backtesting / if season has started)
        from src.data.db import read_sql
        sched = read_sql(
            "SELECT home_team_id, away_team_id, venue_id "
            "FROM production.dim_game "
            "WHERE season = :s AND game_type = 'R' "
            "ORDER BY game_date, game_pk",
            {"s": target_season},
        )
        if len(sched) >= 2400:
            schedule = sched
            logger.info("Using dim_game schedule: %d games", len(schedule))

        # Fall back to MLB API
        if schedule is None or len(schedule) < 2400:
            logger.info("Fetching %d schedule from MLB API...", target_season)
            import json
            import urllib.request

            url = (
                f"https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&season={target_season}&gameType=R"
                f"&hydrate=team&fields=dates,date,games,gamePk,"
                f"teams,away,home,team,id"
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                rows = []
                for date_entry in data.get("dates", []):
                    for game in date_entry.get("games", []):
                        rows.append({
                            "home_team_id": game["teams"]["home"]["team"]["id"],
                            "away_team_id": game["teams"]["away"]["team"]["id"],
                            "venue_id": 0,  # no venue in this endpoint
                        })
                schedule = pd.DataFrame(rows)
                logger.info("Fetched %d games from MLB API", len(schedule))
            except Exception:
                logger.exception("Failed to fetch schedule from MLB API")
                return None

        if schedule is None or schedule.empty:
            logger.warning("No schedule available for %d", target_season)
            return None

        # ── Build team_rosters for injury cascading (Poisson mode) ──
        # Reuses the same data that run_team_sim() loads, keyed by team_id
        # instead of team_abbr.
        league_rosters: dict[int, dict] | None = None
        league_profiles: dict[int, dict] | None = None
        try:
            _hcs_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
            _pcs_path = DASHBOARD_DIR / "pitcher_counting_sim.parquet"
            _pt_path = DASHBOARD_DIR / "player_teams.parquet"
            _pe_path = DASHBOARD_DIR / "hitter_position_eligibility.parquet"
            _hs_path = DASHBOARD_DIR / "health_scores.parquet"

            _hr_path = DASHBOARD_DIR / "hitters_rankings.parquet"
            _pr_path = DASHBOARD_DIR / "pitchers_rankings.parquet"
            _needed = [_hcs_path, _pcs_path, _pt_path, _hr_path, _pr_path]
            if all(p.exists() for p in _needed):
                _lr_hr_src = pd.read_parquet(_hr_path)
                _lr_pr = pd.read_parquet(_pr_path)
                _lr_hcs = pd.read_parquet(_hcs_path)
                _lr_pcs = pd.read_parquet(_pcs_path)
                _lr_teams = pd.read_parquet(_pt_path)
                _lr_health = (
                    pd.read_parquet(_hs_path) if _hs_path.exists()
                    else None
                )

                # Build team_id <-> team_abbr mapping from team_rankings
                _abbr_to_tid: dict[str, int] = dict(
                    zip(tr["abbreviation"], tr["team_id"].astype(int))
                )

                league_rosters = {}
                league_profiles = {}
                for abbr in _lr_teams["team_abbr"].unique():
                    tid = _abbr_to_tid.get(abbr)
                    if tid is None:
                        continue
                    t_ids = set(_lr_teams[_lr_teams["team_abbr"] == abbr]["player_id"])

                    # Hitters
                    t_hr = _lr_hr_src[_lr_hr_src["batter_id"].isin(t_ids)]
                    t_hcs = _lr_hcs[_lr_hcs["batter_id"].isin(t_ids)]
                    h_m = t_hr[["batter_id", "position", "tdd_value_score", "pa", "age"]].merge(
                        t_hcs[["batter_id", "total_h_mean", "total_hr_mean",
                                "total_bb_mean", "total_hbp_mean", "total_tb_mean",
                                "total_pa_mean"]],
                        on="batter_id", how="inner",
                    ).rename(columns={
                        "batter_id": "player_id",
                        "tdd_value_score": "value_score",
                    })
                    h_m["projected_pa"] = h_m["total_pa_mean"]
                    h_m = h_m.sort_values("value_score", ascending=False).reset_index(drop=True)
                    h_m["is_starter"] = h_m.index < 9
                    h_m["games"] = (h_m["pa"].fillna(0) / 3.85).clip(20, 162).astype(int)

                    # Pitchers
                    t_pr = _lr_pr[_lr_pr["pitcher_id"].isin(t_ids)]
                    t_pcs = _lr_pcs[_lr_pcs["pitcher_id"].isin(t_ids)]
                    bf_col = "batters_faced" if "batters_faced" in t_pr.columns else None
                    p_cols = ["pitcher_id", "role", "tdd_value_score", "age"]
                    if bf_col:
                        p_cols.append(bf_col)
                    p_m = t_pr[p_cols].merge(
                        t_pcs[["pitcher_id", "total_runs_mean", "projected_ip_mean"]],
                        on="pitcher_id", how="inner",
                    ).rename(columns={
                        "pitcher_id": "player_id",
                        "tdd_value_score": "value_score",
                    })
                    p_m["projected_ip"] = p_m["projected_ip_mean"]
                    if bf_col and bf_col in p_m.columns:
                        p_m["games"] = (p_m[bf_col].fillna(0) / 25).clip(10, 35).astype(int)
                    else:
                        p_m["games"] = 30

                    if h_m.empty and p_m.empty:
                        continue

                    # Injury params
                    all_players = pd.concat([
                        h_m[["player_id", "age", "position", "games"]].rename(
                            columns={"position": "primary_position"}),
                        p_m[["player_id", "age", "role", "games"]].rename(
                            columns={"role": "primary_position"}),
                    ], ignore_index=True)
                    injury_params = build_team_injury_params(
                        all_players, health_scores=_lr_health,
                    )

                    league_rosters[tid] = {
                        "hitters": h_m,
                        "pitchers": p_m,
                        "injury_params": injury_params,
                    }

                logger.info(
                    "Built league rosters for %d teams (injury cascade mode)",
                    len(league_rosters),
                )
            else:
                missing = [p.name for p in _needed if not p.exists()]
                logger.info(
                    "Skipping injury cascade (missing: %s) -- using Bernoulli",
                    ", ".join(missing),
                )
        except Exception:
            logger.warning(
                "Could not build team rosters for injury cascade",
                exc_info=True,
            )

        # Run sim: prefer Poisson+injury, fall back to Bernoulli
        if league_rosters and len(league_rosters) >= 25:
            result = simulate_league_season(
                schedule,
                team_profiles=league_profiles or None,
                team_rosters=league_rosters,
                n_sims=n_sims, random_seed=random_seed,
            )
        else:
            result = simulate_league_season(
                schedule, team_strength=team_strength,
                n_sims=n_sims, random_seed=random_seed,
            )

        # Save standalone league sim results
        # Map team_id to abbreviation
        tid_abbr = dict(zip(tr["team_id"], tr["abbreviation"]))
        result["team_abbr"] = result["team_id"].map(tid_abbr)
        result.to_parquet(DASHBOARD_DIR / "league_sim.parquet", index=False)
        logger.info("Saved league_sim.parquet: %d teams", len(result))

        # Apply in-season Beta-Binomial blend so team_rankings.parquet's
        # projected_wins reflects the current-season record, not just
        # the preseason sim.
        from src.models.in_season_wins import (
            apply_in_season_blend,
            load_current_team_records,
        )
        try:
            _records = load_current_team_records(from_season + 1)
        except Exception:
            logger.warning("Could not load current records for blend", exc_info=True)
            _records = None
        blended = apply_in_season_blend(result, _records)

        sim_wins_map = dict(zip(blended["team_id"], blended["blended_wins"]))
        sim_p10_map = dict(zip(result["team_id"], result["sim_wins_p10"]))
        sim_p90_map = dict(zip(result["team_id"], result["sim_wins_p90"]))
        sim_std_map = dict(zip(result["team_id"], result["sim_wins_std"]))
        playoff_map = dict(zip(result["team_id"], result["playoff_pct"]))

        tr["projected_wins"] = tr["team_id"].map(sim_wins_map).round(1)
        tr["wins_p10"] = tr["team_id"].map(sim_p10_map)
        tr["wins_p90"] = tr["team_id"].map(sim_p90_map)
        tr["wins_std"] = tr["team_id"].map(sim_std_map)
        tr["playoff_pct"] = tr["team_id"].map(playoff_map)

        # Re-derive tiers from new projected wins
        tr["tier"] = tr["projected_wins"].apply(_assign_tier)

        # Re-normalize (blended wins should already sum to ~2430)
        raw_total = tr["projected_wins"].sum()
        if abs(raw_total - 2430) > 5:
            tr["projected_wins"] = (tr["projected_wins"] * 2430 / raw_total).round(1)
            tr["tier"] = tr["projected_wins"].apply(_assign_tier)

        tr.to_parquet(tr_path, index=False)
        logger.info("Updated team_rankings.parquet with blended sim wins")

        # Log top 10
        for _, row in tr.sort_values("projected_wins", ascending=False).head(10).iterrows():
            logger.info(
                "  %-4s  %5.1fW (p10=%2.0f p90=%2.0f) %s  PO=%.0f%%",
                row.get("abbreviation", "???"),
                row["projected_wins"],
                row.get("wins_p10", 0), row.get("wins_p90", 0),
                row["tier"],
                row.get("playoff_pct", 0) * 100,
            )

        return result

    except Exception:
        logger.exception("Failed to run league season simulator")
        return None


def run_team_sim(
    *,
    health_df: pd.DataFrame | None = None,
    n_sims: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame | None:
    """Run Monte Carlo team season sim with injury cascading.

    Parameters
    ----------
    health_df : pd.DataFrame or None
        Pre-loaded health scores (columns: player_id, health_score,
        health_label).  When provided the disk read of
        ``health_scores.parquet`` is skipped.
    n_sims : int
        Number of Monte Carlo simulations per team.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    results : pd.DataFrame or None
        Team sim results (one row per team), or None on failure.
    """
    logger.info("=" * 60)
    logger.info("Running team season simulator (%d sims)...", n_sims)

    try:
        from src.models.team_sim.injury_model import build_team_injury_params
        from src.models.team_sim.team_season_sim import simulate_all_teams

        # Load all needed data
        hr = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
        pr = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")
        hcs = pd.read_parquet(DASHBOARD_DIR / "hitter_counting_sim.parquet")
        pcs = pd.read_parquet(DASHBOARD_DIR / "pitcher_counting_sim.parquet")
        teams_df = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")
        pe = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")

        health = health_df
        if health is None:
            health_path = DASHBOARD_DIR / "health_scores.parquet"
            if health_path.exists():
                health = pd.read_parquet(health_path)

        # Build per-team data
        team_rosters: dict[str, dict] = {}
        for abbr in teams_df["team_abbr"].unique():
            t_ids = set(teams_df[teams_df["team_abbr"] == abbr]["player_id"])

            # Hitters
            t_hr = hr[hr["batter_id"].isin(t_ids)]
            t_hcs = hcs[hcs["batter_id"].isin(t_ids)]
            h_m = t_hr[["batter_id", "position", "tdd_value_score", "pa", "age"]].merge(
                t_hcs[["batter_id", "total_h_mean", "total_hr_mean", "total_bb_mean",
                        "total_hbp_mean", "total_tb_mean", "total_pa_mean"]],
                on="batter_id", how="inner",
            ).rename(columns={"batter_id": "player_id", "tdd_value_score": "value_score"})
            h_m["projected_pa"] = h_m["total_pa_mean"]
            h_m = h_m.sort_values("value_score", ascending=False).reset_index(drop=True)
            h_m["is_starter"] = h_m.index < 9
            h_m["games"] = (h_m["pa"].fillna(0) / 3.85).clip(20, 162).astype(int)

            # Pitchers
            t_pr = pr[pr["pitcher_id"].isin(t_ids)]
            t_pcs = pcs[pcs["pitcher_id"].isin(t_ids)]
            bf_col = "batters_faced" if "batters_faced" in t_pr.columns else None
            p_cols = ["pitcher_id", "role", "tdd_value_score", "age"]
            if bf_col:
                p_cols.append(bf_col)
            p_m = t_pr[p_cols].merge(
                t_pcs[["pitcher_id", "total_runs_mean", "projected_ip_mean"]],
                on="pitcher_id", how="inner",
            ).rename(columns={"pitcher_id": "player_id", "tdd_value_score": "value_score"})
            p_m["projected_ip"] = p_m["projected_ip_mean"]
            if bf_col and bf_col in p_m.columns:
                p_m["games"] = (p_m[bf_col].fillna(0) / 25).clip(10, 35).astype(int)
            else:
                p_m["games"] = 30

            # Injury params
            all_players = pd.concat([
                h_m[["player_id", "age", "position", "games"]].rename(
                    columns={"position": "primary_position"}),
                p_m[["player_id", "age", "role", "games"]].rename(
                    columns={"role": "primary_position"}),
            ], ignore_index=True)
            injury_params = build_team_injury_params(all_players, health_scores=health)

            team_rosters[abbr] = {
                "hitters": h_m,
                "pitchers": p_m,
                "injury_params": injury_params,
                "position_eligibility": pe[pe["player_id"].isin(t_ids)],
            }

        # Run simulation
        results = simulate_all_teams(team_rosters, n_sims=n_sims, random_seed=random_seed)
        results.to_parquet(DASHBOARD_DIR / "team_sim_wins.parquet", index=False)
        logger.info("Saved team_sim_wins.parquet: %d teams", len(results))

        # Log top 10
        for _, row in results.sort_values("sim_wins_mean", ascending=False).head(10).iterrows():
            logger.info(
                "  %-4s  %.1fW (p10=%.0f, p90=%.0f, std=%.1f)",
                row["team_abbr"], row["sim_wins_mean"],
                row["sim_wins_p10"], row["sim_wins_p90"], row["sim_wins_std"],
            )

        return results

    except Exception:
        logger.exception("Failed to run team season simulator")
        return None


def _load_oaa_by_position(from_season: int) -> dict[tuple[int, str], float]:
    """Load position-specific OAA percentiles from the last 2 seasons.

    Returns
    -------
    dict[(player_id, position), oaa_percentile]
        0 = worst at that position, 1 = best.  Missing entries default to
        0.3 at call-site (below average — unproven at position).
    """
    from src.data.db import read_sql as _read_sql

    min_season = from_season - 1  # 2 seasons of data
    oaa_raw = _read_sql(
        """
        SELECT player_id, position,
               SUM(outs_above_average) AS total_oaa,
               COUNT(*) AS seasons
        FROM production.fact_fielding_oaa
        WHERE season >= :min_season
        GROUP BY player_id, position
        """,
        {"min_season": min_season},
    )
    if oaa_raw.empty:
        return {}

    oaa_raw["player_id"] = oaa_raw["player_id"].astype(int)

    # Normalize per position: 0 = worst, 1 = best
    oaa_lookup: dict[tuple[int, str], float] = {}
    for pos, grp in oaa_raw.groupby("position"):
        pos_min = grp["total_oaa"].min()
        pos_max = grp["total_oaa"].max()
        rng = pos_max - pos_min
        if rng == 0:
            # All players identical at this position
            for _, row in grp.iterrows():
                oaa_lookup[(int(row["player_id"]), pos)] = 0.5
        else:
            for _, row in grp.iterrows():
                pctl = (row["total_oaa"] - pos_min) / rng
                oaa_lookup[(int(row["player_id"]), pos)] = float(pctl)

    return oaa_lookup


def run_depth_chart(
    *,
    hitter_proj: pd.DataFrame | None = None,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute probable starters (PA-based position assignment with OAA).

    Algorithm
    ---------
    Pass 1 — Position candidates: For each field position, collect all
        ranked players whose lineup priors show starts there.  The player
        with the most starts is the incumbent.
    Pass 2 — Resolve conflicts: When multiple candidates compete for a
        position, score each candidate AT that position using 60% offense,
        30% position-specific OAA, and 10% experience (starts).  Winner
        stays; losers cascade to their next-most-played position.
    Pass 3 — Secondary eligibility: Unfilled positions get the best
        remaining unassigned player from dim_roster secondary_positions,
        scored by offense only.
    Pass 4 — DH: First assign any player whose effective primary is DH,
        then fall back to best remaining bat.
    """
    from src.data.queries import get_lineup_priors, get_lineup_priors_by_hand

    logger.info("=" * 60)
    logger.info("Computing probable starters...")
    try:
        from src.data.db import read_sql as _read_sql

        # Get lineup priors (position + batting order history for all players)
        lineup_priors = get_lineup_priors(from_season)
        lineup_priors_by_hand = get_lineup_priors_by_hand(from_season)

        # Get current active roster with secondary positions
        roster = _read_sql("""
            SELECT dr.player_id, dp.player_name, dr.org_id,
                   dr.primary_position, dr.secondary_positions,
                   dr.is_starter, dr.roster_status, dr.last_game_date,
                   dt.abbreviation AS team_abbr
            FROM production.dim_roster dr
            JOIN production.dim_player dp ON dr.player_id = dp.player_id
            JOIN production.dim_team dt ON dr.org_id = dt.team_id
            WHERE dr.level = 'MLB'
              AND dr.roster_status IN ('active', 'nri',
                                       'il_7', 'il_10', 'il_15', 'il_60')
              AND dr.primary_position NOT IN ('SP', 'RP', 'P')
        """)

        if not roster.empty and "secondary_positions" in roster.columns:
            roster["secondary_positions"] = roster["secondary_positions"].apply(
                lambda x: list(x) if x is not None else []
            )

        # Load position-specific OAA percentiles (last 2 seasons)
        oaa_lookup = _load_oaa_by_position(from_season)
        logger.info("Loaded OAA data: %d (player, position) entries", len(oaa_lookup))

        # Build player offense_score lookup — layered fallback:
        #   1. hitters_rankings offense_score (0-1, best)
        #   2. hitters_rankings tdd_value_score as fallback
        #   3. hitter_projections composite_score (shifted +1, only if no rankings)
        #   4. prospect_rankings tdd_prospect_score (scaled to 0-0.4 range
        #      so prospects rank below established MLB players but above
        #      completely unknown players at 0.0)
        offense_lookup: dict[int, float] = {}
        quality_lookup: dict[int, float] = {}
        try:
            _h_rank = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
            if not _h_rank.empty:
                for _, r in _h_rank.iterrows():
                    pid = int(r.get("batter_id", 0))
                    # Prefer offense_score for position-specific scoring
                    if "offense_score" in _h_rank.columns:
                        offense_lookup[pid] = float(r.get("offense_score", 0.5))
                    # quality_lookup tracks who is "ranked" (for Pass 1-2 gating)
                    if "tdd_value_score" in _h_rank.columns:
                        quality_lookup[pid] = float(r.get("tdd_value_score", 0.5))
                    elif "offense_score" in _h_rank.columns:
                        quality_lookup[pid] = float(r.get("offense_score", 0.5))
        except FileNotFoundError:
            pass
        if not quality_lookup and hitter_proj is not None and not hitter_proj.empty:
            for _, r in hitter_proj.iterrows():
                pid = int(r.get("batter_id", 0))
                val = float(r.get("composite_score", 0)) + 1.0
                quality_lookup[pid] = val
                offense_lookup[pid] = val

        # MiLB / prospect fallback for players not in MLB rankings
        try:
            _prospects = pd.read_parquet(DASHBOARD_DIR / "prospect_rankings.parquet")
            if not _prospects.empty and "tdd_prospect_score" in _prospects.columns:
                for _, r in _prospects.iterrows():
                    pid = int(r.get("player_id", 0))
                    if pid not in quality_lookup:
                        # Scale prospect score (0-1) to 0-0.4 range
                        quality_lookup[pid] = float(r.get("tdd_prospect_score", 0)) * 0.4
                        offense_lookup[pid] = float(r.get("tdd_prospect_score", 0)) * 0.4
        except FileNotFoundError:
            pass

        # Default OAA for players without data at a position
        _OAA_DEFAULT = 0.3

        # Field positions filled first; DH is a fallback assigned last.
        FIELD_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        ALL_POSITIONS = FIELD_POSITIONS + ["DH"]
        _IL_STATUSES = {"il_7", "il_10", "il_15", "il_60"}

        def _availability_penalty(row: pd.Series) -> float:
            """Penalty for IL / NRI status."""
            status = row.get("roster_status", "")
            if status in _IL_STATUSES:
                return -0.5
            if status == "nri":
                return -0.2
            return 0.0

        def _player_eligible_positions(row: pd.Series) -> list[str]:
            """All positions a player can play (primary + secondary)."""
            positions: list[str] = []
            pp = row.get("primary_position", "")
            if pp in ALL_POSITIONS:
                positions.append(pp)
            for sp in row.get("secondary_positions", []):
                if sp in ALL_POSITIONS and sp not in positions:
                    positions.append(sp)
            # Everyone is DH-eligible as a last resort
            if "DH" not in positions:
                positions.append("DH")
            return positions

        def _make_entry(team_abbr: str, pos: str, prow: pd.Series,
                        bo_lookup: dict[int, int]) -> dict:
            pid = int(prow["player_id"])
            return {
                "team_abbr": team_abbr,
                "position": pos,
                "player_id": pid,
                "player_name": prow["player_name"],
                "batting_order": bo_lookup.get(pid, 5),
                "roster_status": prow.get("roster_status", "active"),
            }

        def _assign_starters(
            roster_df: pd.DataFrame,
            priors_df: pd.DataFrame,
        ) -> list[dict]:
            """Run multi-pass lineup assignment for all teams given a set of priors."""
            starters: list[dict] = []
            for team_abbr, team_roster in roster_df.groupby("team_abbr"):
                team_pids = set(team_roster["player_id"])
                team_roster = team_roster.copy()
                team_roster["_avail_pen"] = team_roster.apply(
                    _availability_penalty, axis=1,
                )

                elig_map: dict[int, list[str]] = {}
                roster_rows: dict[int, pd.Series] = {}
                for _, prow in team_roster.iterrows():
                    pid = int(prow["player_id"])
                    elig_map[pid] = _player_eligible_positions(prow)
                    roster_rows[pid] = prow

                # Build lineup-prior lookups for this team's players
                # (uses data from ANY team — catches traded players)
                team_priors = (
                    priors_df[priors_df["player_id"].isin(team_pids)].copy()
                    if not priors_df.empty
                    else pd.DataFrame()
                )

                bo_lookup: dict[int, int] = {}
                # Per-player per-position starts: {(player_id, position): starts}
                starts_at_pos: dict[tuple[int, str], int] = {}
                # Effective primary = position with most starts in prior season
                effective_primary: dict[int, str] = {}
                # All positions played per player, sorted by starts desc
                positions_by_starts: dict[int, list[str]] = {}

                if not team_priors.empty:
                    for _, pr in team_priors.drop_duplicates("player_id").iterrows():
                        pid = int(pr["player_id"])
                        bo_lookup[pid] = int(pr.get("batting_order_mode", 5))

                    # Build starts_at_pos and positions_by_starts
                    pos_rows = team_priors[
                        team_priors["position"].isin(ALL_POSITIONS)
                    ].sort_values("starts", ascending=False)

                    for _, pr in pos_rows.iterrows():
                        pid = int(pr["player_id"])
                        pos = pr["position"]
                        starts = int(pr["starts"])
                        starts_at_pos[(pid, pos)] = starts

                        if pid not in positions_by_starts:
                            positions_by_starts[pid] = []
                        positions_by_starts[pid].append(pos)

                    # First entry per player is their most-started position
                    top_pos = pos_rows.drop_duplicates("player_id")
                    for _, pr in top_pos.iterrows():
                        effective_primary[int(pr["player_id"])] = pr["position"]

                # ── Pass 1: Build candidate lists per field position ──────
                # For each position, collect all RANKED players who have starts
                # there from lineup priors.
                pos_candidates: dict[str, list[int]] = {p: [] for p in FIELD_POSITIONS}

                ranked_pids = {
                    pid for pid in team_pids if pid in quality_lookup
                }

                for pid in ranked_pids:
                    for pos in positions_by_starts.get(pid, []):
                        if pos in FIELD_POSITIONS:
                            pos_candidates[pos].append(pid)

                # Sort candidates per position by starts at that position (desc)
                for pos in FIELD_POSITIONS:
                    pos_candidates[pos].sort(
                        key=lambda p: starts_at_pos.get((p, pos), 0),
                        reverse=True,
                    )

                # ── Pass 2: Resolve conflicts with position-specific scoring ─
                # Score = 60% offense + 30% position OAA pctl + 10% experience
                assigned: set[int] = set()
                position_filled: dict[str, dict] = {}

                # Compute max starts per position for experience normalization
                max_starts_at_pos: dict[str, int] = {}
                for pos in FIELD_POSITIONS:
                    starts_vals = [
                        starts_at_pos.get((pid, pos), 0)
                        for pid in pos_candidates[pos]
                    ]
                    max_starts_at_pos[pos] = max(starts_vals) if starts_vals else 1

                def _position_score(pid: int, pos: str) -> float:
                    """Score a player AT a specific position."""
                    off = offense_lookup.get(pid, 0.0) + roster_rows[pid]["_avail_pen"]
                    oaa_pct = oaa_lookup.get((pid, pos), _OAA_DEFAULT)
                    exp_starts = starts_at_pos.get((pid, pos), 0)
                    max_s = max_starts_at_pos.get(pos, 1)
                    exp = exp_starts / max_s if max_s > 0 else 0.0
                    return 0.60 * off + 0.30 * oaa_pct + 0.10 * exp

                # Process positions in order.  When a player loses a contest
                # they cascade to their next-most-played position.
                # We iterate until no more assignments can be made.
                unresolved = set(FIELD_POSITIONS)
                max_iters = len(FIELD_POSITIONS) * 2  # safety cap
                for _ in range(max_iters):
                    if not unresolved:
                        break
                    made_assignment = False
                    still_unresolved: set[str] = set()
                    for pos in sorted(unresolved):
                        # Remove already-assigned players from candidates
                        pos_candidates[pos] = [
                            p for p in pos_candidates[pos] if p not in assigned
                        ]
                        candidates = pos_candidates[pos]
                        if not candidates:
                            still_unresolved.add(pos)
                            continue
                        if len(candidates) == 1:
                            # Uncontested — assign
                            winner = candidates[0]
                        else:
                            # Contested — score each candidate at this position
                            scored = [
                                (pid, _position_score(pid, pos))
                                for pid in candidates
                            ]
                            scored.sort(key=lambda x: x[1], reverse=True)
                            winner = scored[0][0]

                        assigned.add(winner)
                        position_filled[pos] = _make_entry(
                            team_abbr, pos, roster_rows[winner], bo_lookup,
                        )
                        made_assignment = True

                        # Losers stay in other positions' candidate lists
                        # (they were never removed from those)

                    unresolved = still_unresolved
                    if not made_assignment:
                        break  # no progress — remaining positions have no candidates

                # ── Pass 3: Fill remaining from secondary eligibility ─────
                # Any unfilled field position gets the best remaining unassigned
                # player who has that position in their secondary_positions list.
                # Score by offense only (no OAA data for positions they haven't
                # played).
                for pos in FIELD_POSITIONS:
                    if pos in position_filled:
                        continue
                    best_pid: int | None = None
                    best_off = -999.0
                    for pid in ranked_pids:
                        if pid in assigned:
                            continue
                        if pos in elig_map.get(pid, []):
                            off = (
                                offense_lookup.get(pid, 0.0)
                                + roster_rows[pid]["_avail_pen"]
                            )
                            if off > best_off:
                                best_off = off
                                best_pid = pid
                    # Also check unranked players
                    if best_pid is None:
                        for _, prow in team_roster.iterrows():
                            pid = int(prow["player_id"])
                            if pid in assigned:
                                continue
                            if pos in elig_map.get(pid, []):
                                best_pid = pid
                                break
                    if best_pid is not None:
                        assigned.add(best_pid)
                        position_filled[pos] = _make_entry(
                            team_abbr, pos, roster_rows[best_pid], bo_lookup,
                        )

                # ── Pass 4: DH — effective DH primary first, then best bat ─
                if "DH" not in position_filled:
                    # First: anyone whose effective primary IS DH
                    for pid in ranked_pids - assigned:
                        eff_pos = effective_primary.get(pid, "")
                        if eff_pos == "DH":
                            assigned.add(pid)
                            position_filled["DH"] = _make_entry(
                                team_abbr, "DH", roster_rows[pid], bo_lookup,
                            )
                            break
                if "DH" not in position_filled:
                    # Fallback: best remaining unassigned hitter by offense
                    best_pid = None
                    best_off = -999.0
                    for pid in team_pids - assigned:
                        off = (
                            offense_lookup.get(pid, 0.0)
                            + roster_rows.get(pid, pd.Series({"_avail_pen": 0}))
                            .get("_avail_pen", 0.0)
                        )
                        if off > best_off:
                            best_off = off
                            best_pid = pid
                    if best_pid is not None:
                        assigned.add(best_pid)
                        position_filled["DH"] = _make_entry(
                            team_abbr, "DH", roster_rows[best_pid], bo_lookup,
                        )

                starters.extend(position_filled.values())
            return starters

        # --- Run assignment: overall (default) ---
        all_starters = _assign_starters(roster, lineup_priors)

        probable_df = pd.DataFrame(all_starters)
        if not probable_df.empty:
            probable_df.to_parquet(
                DASHBOARD_DIR / "probable_starters.parquet", index=False,
            )
            logger.info(
                "Saved probable starters: %d players across %d teams",
                len(probable_df), probable_df["team_abbr"].nunique(),
            )
        else:
            logger.warning("No probable starters computed")

        # --- Run assignment: by opposing pitcher hand (vs RHP / vs LHP) ---
        if not lineup_priors_by_hand.empty:
            hand_starters: list[dict] = []
            for vs_hand in ["R", "L"]:
                hand_priors = lineup_priors_by_hand[
                    lineup_priors_by_hand["vs_hand"] == vs_hand
                ].copy()
                entries = _assign_starters(roster, hand_priors)
                for e in entries:
                    e["vs_hand"] = vs_hand
                hand_starters.extend(entries)

            hand_df = pd.DataFrame(hand_starters)
            if not hand_df.empty:
                hand_df.to_parquet(
                    DASHBOARD_DIR / "probable_starters_by_hand.parquet", index=False,
                )
                logger.info(
                    "Saved platoon starters: %d entries (%d vs RHP, %d vs LHP)",
                    len(hand_df),
                    (hand_df["vs_hand"] == "R").sum(),
                    (hand_df["vs_hand"] == "L").sum(),
                )

        if not lineup_priors.empty:
            lineup_priors.to_parquet(
                DASHBOARD_DIR / "lineup_priors.parquet", index=False,
            )
            logger.info("Saved lineup priors: %d rows", len(lineup_priors))
    except Exception:
        logger.exception("Failed to compute probable starters")


def run_roster(*, from_season: int = FROM_SEASON) -> None:
    """Export full MLB roster enriched with actual lineup positions.

    Reads dim_roster for the full roster (team, career positions, status)
    then overlays current-season fact_lineup data to add:

    - ``lineup_position``: position the player has actually started most
      often this season (None for bench / pitchers without starts).
    - ``is_depth_starter``: True for the primary everyday starter at each
      position per team (9 per team).

    Career ``secondary_positions`` from dim_roster are preserved — lineup
    data only informs where the player is playing NOW, not what they can
    play.
    """
    logger.info("=" * 60)
    logger.info("Exporting MLB roster from dim_roster...")
    try:
        from src.data.db import read_sql as _read_sql_roster
        roster_full = _read_sql_roster("""
            SELECT dr.player_id, dp.player_name, dr.org_id,
                   dr.primary_position, dr.secondary_positions,
                   dr.is_starter, dr.roster_status, dr.level,
                   dr.last_game_date,
                   dt.abbreviation AS team_abbr,
                   dt.team_name, dt.league, dt.division
            FROM production.dim_roster dr
            JOIN production.dim_player dp ON dr.player_id = dp.player_id
            JOIN production.dim_team dt ON dr.org_id = dt.team_id
            WHERE dr.level = 'MLB'
              AND dr.roster_status IN (
                  'active', 'nri',
                  'il_7', 'il_10', 'il_15', 'il_60'
              )
        """)
        # Convert secondary_positions array to Python list for parquet
        if not roster_full.empty and "secondary_positions" in roster_full.columns:
            roster_full["secondary_positions"] = roster_full["secondary_positions"].apply(
                lambda x: list(x) if x is not None else []
            )

        # ── Overlay actual lineup positions from current season ──────
        current_season = from_season + 1
        lineup_pos = _read_sql_roster("""
            SELECT fl.player_id, dt.abbreviation AS team_abbr,
                   fl.position, COUNT(*) AS games_started
            FROM production.fact_lineup fl
            JOIN production.dim_team dt ON fl.team_id = dt.team_id
            WHERE fl.season = :season
              AND fl.is_starter = true
              AND fl.position NOT IN ('P')
            GROUP BY fl.player_id, dt.abbreviation, fl.position
        """, {"season": current_season})

        if lineup_pos.empty:
            logger.info(
                "No %d lineup data — falling back to %d",
                current_season, from_season,
            )
            current_season = from_season
            lineup_pos = _read_sql_roster("""
                SELECT fl.player_id, dt.abbreviation AS team_abbr,
                       fl.position, COUNT(*) AS games_started
                FROM production.fact_lineup fl
                JOIN production.dim_team dt ON fl.team_id = dt.team_id
                WHERE fl.season = :season
                  AND fl.is_starter = true
                  AND fl.position NOT IN ('P')
                GROUP BY fl.player_id, dt.abbreviation, fl.position
            """, {"season": current_season})

        roster_full["lineup_position"] = None
        roster_full["is_depth_starter"] = False

        if not lineup_pos.empty:
            # Each player's most-started position this season
            idx = lineup_pos.groupby("player_id")["games_started"].idxmax()
            player_best = lineup_pos.loc[idx].set_index("player_id")

            # Determine the primary starter at each (team, position):
            # the player with the most games started there.
            ALL_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
            starter_pids: set[int] = set()
            for (team, pos), grp in lineup_pos.groupby(["team_abbr", "position"]):
                if pos not in ALL_POSITIONS:
                    continue
                top = grp.sort_values("games_started", ascending=False).iloc[0]
                pid = int(top["player_id"])
                # Only mark as starter if this is also the player's
                # most-started position (avoid double-counting)
                if pid in player_best.index:
                    if player_best.loc[pid, "position"] == pos:
                        starter_pids.add(pid)

            for i, row in roster_full.iterrows():
                pid = int(row["player_id"])
                if pid in player_best.index:
                    roster_full.at[i, "lineup_position"] = player_best.loc[
                        pid, "position"
                    ]
                if pid in starter_pids:
                    roster_full.at[i, "is_depth_starter"] = True

        if not roster_full.empty:
            roster_full.to_parquet(
                DASHBOARD_DIR / "roster.parquet", index=False,
            )
            n_starters = roster_full["is_depth_starter"].sum()
            n_with_pos = roster_full["lineup_position"].notna().sum()
            logger.info(
                "Saved roster.parquet: %d players (%d starters, %d with "
                "lineup positions) across %d teams",
                len(roster_full), n_starters, n_with_pos,
                roster_full["team_abbr"].nunique(),
            )
        else:
            logger.warning("dim_roster query returned no rows")
    except Exception:
        logger.exception("Failed to export roster")
