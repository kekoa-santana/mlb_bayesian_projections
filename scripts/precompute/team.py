"""Precompute: Team ELO, profiles, power rankings, depth chart, roster."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.team")


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

        # Current (end-of-2025) ratings
        current = get_current_ratings(ratings, team_info)
        current.to_parquet(DASHBOARD_DIR / "team_series_elo.parquet", index=False)
        logger.info("Saved team_series_elo.parquet: %d teams", len(current))

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
    from_season: int = FROM_SEASON,
) -> None:
    """Build team profiles and rankings."""
    logger.info("=" * 60)
    logger.info("Building team profiles and rankings...")

    try:
        from src.models.team_profiles import build_all_team_profiles
        from src.models.team_rankings import rank_teams

        team_profiles = build_all_team_profiles(
            season=from_season,
            projection_season=from_season + 1,
            elo_history=elo_history,
        )
        team_profiles.to_parquet(DASHBOARD_DIR / "team_profiles.parquet", index=False)
        logger.info("Saved team_profiles.parquet: %d teams", len(team_profiles))

        team_rankings_df = rank_teams(
            team_profiles,
            elo_ratings=elo_current,
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


def run_team_power() -> None:
    """Build projection-integrated power rankings."""
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
        _h_rank_pw = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet") \
            if (DASHBOARD_DIR / "hitters_rankings.parquet").exists() else None
        _p_rank_pw = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet") \
            if (DASHBOARD_DIR / "pitchers_rankings.parquet").exists() else None
        _h_proj_pw = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet") \
            if (DASHBOARD_DIR / "hitter_projections.parquet").exists() else None
        _p_proj_pw = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet") \
            if (DASHBOARD_DIR / "pitcher_projections.parquet").exists() else None

        # Glicko paths (optional -- gracefully degraded if missing)
        _bg_path = DASHBOARD_DIR / "batter_glicko.parquet"
        _pg_path = DASHBOARD_DIR / "pitcher_glicko.parquet"

        if all(x is not None for x in [
            _elo_pre, _profiles, _cur_roster_pw,
            _h_rank_pw, _p_rank_pw, _h_proj_pw, _p_proj_pw,
        ]):
            from src.models.team_rankings import build_power_rankings
            power_rankings_df = build_power_rankings(
                elo_ratings=_elo_pre,
                profiles=_profiles,
                current_roster=_cur_roster_pw,
                hitter_rankings=_h_rank_pw,
                pitcher_rankings=_p_rank_pw,
                hitter_projections=_h_proj_pw,
                pitcher_projections=_p_proj_pw,
                batter_glicko_path=_bg_path if _bg_path.exists() else None,
                pitcher_glicko_path=_pg_path if _pg_path.exists() else None,
            )
            power_rankings_df.to_parquet(
                DASHBOARD_DIR / "team_power_rankings.parquet", index=False,
            )
            logger.info("Saved team_power_rankings.parquet: %d teams", len(power_rankings_df))
            for _, row in power_rankings_df.head(10).iterrows():
                logger.info(
                    "  %2d. %-4s  %.3f (%s)  ELO=%.2f Proj=%.2f Prof=%.2f Glk=%.2f  B:%d R:%d",
                    row["power_rank"], row["abbreviation"],
                    row["power_score"], row["power_tier"],
                    row["elo_component"], row["projection_component"],
                    row["profile_component"], row.get("glicko_component", 0.0),
                    row["breakout_count"], row["regression_count"],
                )
        else:
            missing = [
                name for name, val in [
                    ("elo_preseason", _elo_pre), ("profiles", _profiles),
                    ("roster", _cur_roster_pw), ("hitter_rankings", _h_rank_pw),
                    ("pitcher_rankings", _p_rank_pw), ("hitter_projections", _h_proj_pw),
                    ("pitcher_projections", _p_proj_pw),
                ] if val is None
            ]
            logger.warning("Skipping power rankings -- missing data: %s", missing)

    except Exception:
        logger.exception("Failed to build power rankings")


def run_team_sim(
    *,
    n_sims: int = 1000,
    random_seed: int = 42,
) -> None:
    """Run Monte Carlo team season sim with injury cascading."""
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

        health = None
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
            h_m["games"] = (h_m["pa"] / 3.85).clip(20, 162).astype(int)

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
                p_m["games"] = (p_m[bf_col] / 25).clip(10, 35).astype(int)
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

    except Exception:
        logger.exception("Failed to run team season simulator")


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
    from src.data.queries import get_lineup_priors

    logger.info("=" * 60)
    logger.info("Computing probable starters...")
    try:
        from src.data.db import read_sql as _read_sql

        # Get lineup priors (position + batting order history for all players)
        lineup_priors = get_lineup_priors(from_season)

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

        all_starters: list[dict] = []

        for team_abbr, team_roster in roster.groupby("team_abbr"):
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
                lineup_priors[lineup_priors["player_id"].isin(team_pids)].copy()
                if not lineup_priors.empty
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

            all_starters.extend(position_filled.values())

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

        if not lineup_priors.empty:
            lineup_priors.to_parquet(
                DASHBOARD_DIR / "lineup_priors.parquet", index=False,
            )
            logger.info("Saved lineup priors: %d rows", len(lineup_priors))
    except Exception:
        logger.exception("Failed to compute probable starters")


def run_roster() -> None:
    """Export full MLB roster from dim_roster."""
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
        if not roster_full.empty:
            roster_full.to_parquet(
                DASHBOARD_DIR / "roster.parquet", index=False,
            )
            logger.info(
                "Saved roster.parquet: %d players across %d teams",
                len(roster_full), roster_full["team_abbr"].nunique(),
            )
        else:
            logger.warning("dim_roster query returned no rows")
    except Exception:
        logger.exception("Failed to export roster")
