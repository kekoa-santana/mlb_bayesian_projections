"""
Comprehensive team profiles — six dimensions of team quality.

Builds profiles from a mix of existing projection parquets and live
database queries, producing one row per team with sub-scores for:

a. Offense (style, depth, platoon balance, projected totals)
b. Pitching (rotation strength, bullpen depth, staff style)
c. Defense (team OAA, catcher framing, premium positions)
d. Organizational philosophy (build vs buy, farm system)
e. Health & depth (IL days, starter-backup gap, age trajectory)
f. Schedule context (opponent ELO, division competitiveness)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = better)."""
    return series.rank(pct=True, method="average")


def _safe_load(filename: str) -> pd.DataFrame | None:
    """Load a parquet from dashboard dir, returning None if missing."""
    path = DASHBOARD_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Missing %s — skipping", filename)
    return None


def _build_roster_stability(
    current_roster: pd.DataFrame,
    previous_roster: pd.DataFrame,
) -> pd.DataFrame:
    """Measure roster continuity between seasons.

    Parameters
    ----------
    current_roster : pd.DataFrame
        2026 roster (from roster.parquet) with player_id, team_abbr,
        roster_status.  Filtered to MLB active/IL before calling.
    previous_roster : pd.DataFrame
        2025 roster (from player_teams.parquet) with player_id, team_abbr.

    Returns
    -------
    pd.DataFrame
        team_abbr, roster_continuity (0-1 percentile), players_retained,
        players_new, players_lost.
    """
    rows: list[dict] = []

    all_teams = sorted(current_roster["team_abbr"].dropna().unique())
    for team_abbr in all_teams:
        curr_ids = set(
            current_roster.loc[
                current_roster["team_abbr"] == team_abbr, "player_id"
            ]
        )
        prev_ids = set(
            previous_roster.loc[
                previous_roster["team_abbr"] == team_abbr, "player_id"
            ]
        )

        retained = len(curr_ids & prev_ids)
        new = len(curr_ids - prev_ids)
        lost = len(prev_ids - curr_ids)
        denom = retained + new + lost
        raw_continuity = retained / denom if denom > 0 else 0.0

        rows.append({
            "team_abbr": team_abbr,
            "players_retained": retained,
            "players_new": new,
            "players_lost": lost,
            "raw_continuity": raw_continuity,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "team_abbr", "roster_continuity",
                "players_retained", "players_new", "players_lost",
            ]
        )

    df = pd.DataFrame(rows)
    # Normalize to percentile across 30 teams (higher = more stable)
    df["roster_continuity"] = _pctl(df["raw_continuity"])
    df.drop(columns=["raw_continuity"], inplace=True)

    logger.info(
        "Roster stability: median retained=%d, range=[%d, %d]",
        int(df["players_retained"].median()),
        df["players_retained"].min(),
        df["players_retained"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# a. Offense Profile
# ---------------------------------------------------------------------------
def _build_offense_profile(
    player_teams: pd.DataFrame,
    hitter_rankings: pd.DataFrame | None,
    hitter_counting: pd.DataFrame | None,
    team_offense: pd.DataFrame | None,
    park_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Offense sub-scores per team."""
    teams = player_teams[["team_id"]].drop_duplicates().copy()
    if teams.empty:
        return pd.DataFrame()

    # --- Projected totals from counting sim (aggregated by team) ---
    if hitter_counting is not None and "batter_id" in hitter_counting.columns:
        # Map batter_id -> team_id via player_teams
        pid_col = "batter_id" if "batter_id" in player_teams.columns else "player_id"
        counting_merged = hitter_counting.merge(
            player_teams[[pid_col, "team_id"]].rename(columns={pid_col: "batter_id"}),
            on="batter_id",
            how="inner",
        )

        stat_cols = {
            "R": "runs" if "runs" in counting_merged.columns else "r",
            "HR": "hr" if "hr" in counting_merged.columns else "home_runs",
            "SB": "sb" if "sb" in counting_merged.columns else "stolen_bases",
        }
        agg_dict = {}
        for label, col in stat_cols.items():
            if col in counting_merged.columns:
                agg_dict[col] = "sum"

        if agg_dict:
            team_totals = counting_merged.groupby("team_id").agg(agg_dict).reset_index()
            for label, col in stat_cols.items():
                if col in team_totals.columns:
                    team_totals.rename(columns={col: f"proj_{label}"}, inplace=True)
            teams = teams.merge(team_totals, on="team_id", how="left")

    # --- Lineup depth from hitter rankings ---
    if hitter_rankings is not None:
        pid_col = "player_id" if "player_id" in hitter_rankings.columns else "batter_id"
        score_col = "tdd_value_score" if "tdd_value_score" in hitter_rankings.columns else "composite_score"
        if score_col in hitter_rankings.columns:
            pt_pid = "batter_id" if "batter_id" in player_teams.columns else "player_id"
            rank_merged = hitter_rankings.merge(
                player_teams[[pt_pid, "team_id"]].rename(columns={pt_pid: pid_col}),
                on=pid_col,
                how="inner",
            )
            median_score = rank_merged[score_col].median()

            # Merge projected PA for weighting (from counting sim)
            pa_weight_col = "total_pa_mean"
            if hitter_counting is not None and pa_weight_col in hitter_counting.columns:
                pa_id = "batter_id" if "batter_id" in hitter_counting.columns else "player_id"
                rank_merged = rank_merged.merge(
                    hitter_counting[[pa_id, pa_weight_col]].rename(columns={pa_id: pid_col}),
                    on=pid_col, how="left",
                )
            if pa_weight_col not in rank_merged.columns:
                rank_merged[pa_weight_col] = 1.0  # fallback: equal weight
            rank_merged[pa_weight_col] = rank_merged[pa_weight_col].fillna(
                rank_merged[pa_weight_col].median()
            ).clip(50)

            def _pa_weighted_score(g: pd.DataFrame) -> pd.Series:
                top9 = g.nlargest(9, score_col)
                pa_w = top9[pa_weight_col].values
                scores = top9[score_col].values
                wtd_avg = float(np.average(scores, weights=pa_w)) if pa_w.sum() > 0 else scores.mean()
                return pd.Series({
                    "lineup_depth": (g[score_col] >= median_score).sum() / min(len(g), 9),
                    "avg_hitter_score": wtd_avg,
                    "lineup_floor": g.nlargest(min(len(g), 8), score_col)[score_col].iloc[-1] if len(g) >= 7 else 0.0,
                })

            depth = rank_merged.groupby("team_id").apply(
                _pa_weighted_score, include_groups=False,
            ).reset_index()
            teams = teams.merge(depth, on="team_id", how="left")

    # --- Offense style classification ---
    if team_offense is not None:
        recent = team_offense.sort_values("season", ascending=False).drop_duplicates("team_id")
        recent["iso"] = (
            (recent.get("total_bases", recent.get("home_runs", 0) * 4))
            / recent["plate_appearances"].clip(1)
            if "total_bases" in recent.columns
            else recent["home_runs"] / recent["plate_appearances"].clip(1) * 4
        )
        recent["k_rate"] = recent["strikeouts"] / recent["plate_appearances"].clip(1)
        recent["bb_rate"] = recent["walks"] / recent["plate_appearances"].clip(1)
        recent["sb_per_game"] = recent["sb"] / recent["games"].clip(1)
        recent["hr_per_game"] = recent["home_runs"] / recent["games"].clip(1)
        recent["rpg"] = recent["runs"] / recent["games"].clip(1)

        # --- Park factor normalization ---
        # Divide park-dependent rates by park factor to get park-neutral values.
        # rpg and hr_per_game are inflated/deflated by home park environment.
        # K/BB rates and SB are not meaningfully park-dependent.
        if park_factors is not None and not park_factors.empty:
            recent = recent.merge(
                park_factors[["team_id", "run_pf", "hr_pf"]],
                on="team_id",
                how="left",
            )
            recent["run_pf"] = recent["run_pf"].fillna(1.0)
            recent["hr_pf"] = recent["hr_pf"].fillna(1.0)
            recent["rpg"] = recent["rpg"] / recent["run_pf"]
            recent["hr_per_game"] = recent["hr_per_game"] / recent["hr_pf"]
            logger.info("Offense park-adjusted: rpg and hr_per_game normalized")

        def _classify_style(r: pd.Series) -> str:
            hr_pctl = _pctl(recent["hr_per_game"]).loc[r.name]
            k_pctl = _pctl(recent["k_rate"]).loc[r.name]
            sb_pctl = _pctl(recent["sb_per_game"]).loc[r.name]
            bb_pctl = _pctl(recent["bb_rate"]).loc[r.name]
            if hr_pctl > 0.75 and k_pctl > 0.50:
                return "Power"
            if sb_pctl > 0.75 and hr_pctl < 0.50:
                return "Smallball"
            if k_pctl < 0.25 and bb_pctl > 0.50:
                return "Contact"
            return "Balanced"

        recent["offense_style"] = recent.apply(_classify_style, axis=1)
        style_cols = ["team_id", "offense_style", "rpg", "hr_per_game", "sb_per_game"]
        style_cols = [c for c in style_cols if c in recent.columns]
        teams = teams.merge(recent[style_cols], on="team_id", how="left")

    # Offense composite score (percentile-based)
    score_cols = [c for c in ["lineup_depth", "avg_hitter_score", "rpg", "lineup_floor"] if c in teams.columns]
    if score_cols:
        for c in score_cols:
            teams[f"{c}_pctl"] = _pctl(teams[c].fillna(teams[c].median()))
        pctl_cols = [f"{c}_pctl" for c in score_cols]
        teams["offense_score"] = teams[pctl_cols].mean(axis=1)
    else:
        teams["offense_score"] = 0.5

    return teams


# ---------------------------------------------------------------------------
# b. Pitching Profile
# ---------------------------------------------------------------------------
def _build_pitching_profile(
    player_teams: pd.DataFrame,
    pitcher_rankings: pd.DataFrame | None,
    pitcher_roles: pd.DataFrame | None,
    team_pitching: pd.DataFrame | None,
    park_factors: pd.DataFrame | None = None,
    pitcher_counting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pitching sub-scores per team (rotation + bullpen split)."""
    pt_pid = "player_id" if "player_id" in player_teams.columns else "batter_id"
    teams = player_teams[["team_id"]].drop_duplicates().copy()

    # --- Rotation strength (top 5 SP by value score) ---
    pr_merged = None  # keep reference for Glicko/ERA joins below
    if pitcher_rankings is not None:
        pid_col = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"
        score_col = "tdd_value_score" if "tdd_value_score" in pitcher_rankings.columns else "composite_score"
        role_col = "role" if "role" in pitcher_rankings.columns else "position"

        if score_col in pitcher_rankings.columns:
            pr_merged = pitcher_rankings.merge(
                player_teams[["team_id", pt_pid]].rename(columns={pt_pid: pid_col}),
                on=pid_col,
                how="inner",
            )
            # Merge projected IP for weighting
            ip_weight_col = "projected_ip_mean"
            if pitcher_counting is not None and ip_weight_col in pitcher_counting.columns:
                p_id = "pitcher_id" if "pitcher_id" in pitcher_counting.columns else "player_id"
                pr_merged = pr_merged.merge(
                    pitcher_counting[[p_id, ip_weight_col]].rename(columns={p_id: pid_col}),
                    on=pid_col, how="left",
                )
            if ip_weight_col not in pr_merged.columns:
                pr_merged[ip_weight_col] = 1.0
            pr_merged[ip_weight_col] = pr_merged[ip_weight_col].fillna(
                pr_merged[ip_weight_col].median()
            ).clip(10)

            # SP rotation (IP-weighted)
            sp_mask = pr_merged[role_col].isin(["SP"]) if role_col in pr_merged.columns else pd.Series(True, index=pr_merged.index)
            sp_data = pr_merged[sp_mask]

            def _ip_weighted_rot(g: pd.DataFrame) -> float:
                top5 = g.nlargest(5, score_col)
                w = top5[ip_weight_col].values
                s = top5[score_col].values
                return float(np.average(s, weights=w)) if w.sum() > 0 else s.mean()

            rot_strength = sp_data.groupby("team_id").apply(
                _ip_weighted_rot, include_groups=False,
            ).reset_index(name="rotation_strength")
            teams = teams.merge(rot_strength, on="team_id", how="left")

    # --- Bullpen depth (role-weighted quality) ---
    if pitcher_roles is not None and pitcher_rankings is not None:
        pid_col_r = "pitcher_id" if "pitcher_id" in pitcher_roles.columns else "player_id"
        roles_merged = pitcher_roles.merge(
            player_teams[["team_id", pt_pid]].rename(columns={pt_pid: pid_col_r}),
            on=pid_col_r,
            how="inner",
        )
        # Join value scores
        score_col_r = "tdd_value_score" if "tdd_value_score" in pitcher_rankings.columns else "composite_score"
        pid_col_pr = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"
        if score_col_r in pitcher_rankings.columns:
            roles_merged = roles_merged.merge(
                pitcher_rankings[[pid_col_pr, score_col_r]].rename(columns={pid_col_pr: pid_col_r}),
                on=pid_col_r,
                how="left",
            )
            # Role weights: CL 1.5x, SU 1.0x, MR 0.5x
            role_w_col = "role" if "role" in roles_merged.columns else "predicted_role"
            if role_w_col in roles_merged.columns:
                role_weights = {"CL": 1.5, "SU": 1.0, "MR": 0.5}
                roles_merged["role_weight"] = roles_merged[role_w_col].map(role_weights).fillna(0.5)
                roles_merged["weighted_score"] = roles_merged[score_col_r].fillna(0) * roles_merged["role_weight"]

                bp_depth = roles_merged.groupby("team_id").apply(
                    lambda g: g["weighted_score"].sum() / g["role_weight"].sum() if g["role_weight"].sum() > 0 else 0,
                    include_groups=False,
                ).reset_index(name="bullpen_depth")
                teams = teams.merge(bp_depth, on="team_id", how="left")

    # --- Glicko ratings per role (SP / RP) ---
    pitcher_glicko = _safe_load("pitcher_glicko.parquet")
    if pitcher_glicko is not None and pitcher_rankings is not None:
        pid_col_g = "pitcher_id"
        role_col_g = "role" if "role" in pitcher_rankings.columns else "position"
        # Map pitcher_id -> team_id via player_teams
        glicko_team = pitcher_glicko[[pid_col_g, "mu"]].merge(
            player_teams[["team_id", pt_pid]].rename(columns={pt_pid: pid_col_g}),
            on=pid_col_g,
            how="inner",
        )
        # Map pitcher_id -> role via pitcher_rankings
        pid_col_pr2 = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"
        glicko_team = glicko_team.merge(
            pitcher_rankings[[pid_col_pr2, role_col_g]].rename(columns={pid_col_pr2: pid_col_g}),
            on=pid_col_g,
            how="inner",
        )

        # SP aggregate Glicko per team
        sp_glicko = glicko_team[glicko_team[role_col_g] == "SP"]
        if not sp_glicko.empty:
            sp_glicko_agg = sp_glicko.groupby("team_id")["mu"].mean().reset_index(name="sp_glicko_mu")
            sp_glicko_agg["sp_glicko_pctl"] = _pctl(sp_glicko_agg["sp_glicko_mu"])
            teams = teams.merge(sp_glicko_agg[["team_id", "sp_glicko_pctl"]], on="team_id", how="left")
            logger.info("SP Glicko aggregated for %d teams", sp_glicko_agg.shape[0])

        # RP aggregate Glicko per team
        rp_glicko = glicko_team[glicko_team[role_col_g] == "RP"]
        if not rp_glicko.empty:
            rp_glicko_agg = rp_glicko.groupby("team_id")["mu"].mean().reset_index(name="rp_glicko_mu")
            rp_glicko_agg["rp_glicko_pctl"] = _pctl(rp_glicko_agg["rp_glicko_mu"])
            teams = teams.merge(rp_glicko_agg[["team_id", "rp_glicko_pctl"]], on="team_id", how="left")
            logger.info("RP Glicko aggregated for %d teams", rp_glicko_agg.shape[0])

    # --- SP park-adjusted ERA (top 5 SP per team) ---
    if pr_merged is not None and "observed_era" in pr_merged.columns:
        role_col_era = "role" if "role" in pr_merged.columns else "position"
        sp_era_data = pr_merged[pr_merged[role_col_era] == "SP"].copy()
        if not sp_era_data.empty:
            sp_era_agg = sp_era_data.groupby("team_id").apply(
                lambda g: g.nsmallest(5, "observed_era")["observed_era"].mean(),
                include_groups=False,
            ).reset_index(name="sp_era_avg")
            # Lower ERA = better, so invert the percentile
            sp_era_agg["sp_era_pctl"] = 1 - _pctl(sp_era_agg["sp_era_avg"])
            teams = teams.merge(sp_era_agg[["team_id", "sp_era_pctl"]], on="team_id", how="left")
            logger.info("SP ERA aggregated for %d teams", sp_era_agg.shape[0])

    # --- Bullpen K rate from team_bullpen_rates ---
    bp_rates = _safe_load("team_bullpen_rates.parquet")
    if bp_rates is not None and not bp_rates.empty:
        # Use most recent season
        recent_bp = bp_rates.sort_values("season", ascending=False).drop_duplicates("team_id")
        recent_bp["bp_k_rate_pctl"] = _pctl(recent_bp["k_rate"])
        teams = teams.merge(
            recent_bp[["team_id", "bp_k_rate_pctl"]],
            on="team_id",
            how="left",
        )
        logger.info("Bullpen K rate loaded for %d teams", recent_bp.shape[0])

    # --- Staff style classification ---
    if team_pitching is not None:
        recent_p = team_pitching.sort_values("season", ascending=False).drop_duplicates("team_id")
        recent_p["ra_per_game"] = recent_p["runs_allowed"] / recent_p["games"].clip(1)
        recent_p["k_per_game"] = recent_p["strikeouts_by"] / recent_p["games"].clip(1)

        # --- Park factor normalization ---
        # Runs allowed is inflated at hitter-friendly parks, so divide by run_pf.
        # K per game is not meaningfully park-dependent (no K park factor in DB),
        # so leave as-is.
        if park_factors is not None and not park_factors.empty:
            recent_p = recent_p.merge(
                park_factors[["team_id", "run_pf"]],
                on="team_id",
                how="left",
            )
            recent_p["run_pf"] = recent_p["run_pf"].fillna(1.0)
            recent_p["ra_per_game"] = recent_p["ra_per_game"] / recent_p["run_pf"]
            logger.info("Pitching park-adjusted: ra_per_game normalized")

        def _classify_staff(r: pd.Series) -> str:
            k_pctl = _pctl(recent_p["k_per_game"]).loc[r.name]
            if k_pctl > 0.75:
                return "Strikeout"
            if k_pctl < 0.25:
                return "Contact-mgmt"
            return "Balanced"

        recent_p["pitching_style"] = recent_p.apply(_classify_staff, axis=1)
        teams = teams.merge(
            recent_p[["team_id", "pitching_style", "ra_per_game", "k_per_game"]],
            on="team_id", how="left",
        )

    # --- Rotation score ---
    rot_parts: list[tuple[str, float]] = []
    if "rotation_strength" in teams.columns:
        teams["rotation_strength_pctl"] = _pctl(teams["rotation_strength"].fillna(teams["rotation_strength"].median()))
        rot_parts.append(("rotation_strength_pctl", 0.40))
    if "sp_glicko_pctl" in teams.columns:
        teams["sp_glicko_pctl"] = teams["sp_glicko_pctl"].fillna(0.5)
        rot_parts.append(("sp_glicko_pctl", 0.30))
    if "sp_era_pctl" in teams.columns:
        teams["sp_era_pctl"] = teams["sp_era_pctl"].fillna(0.5)
        rot_parts.append(("sp_era_pctl", 0.30))

    if rot_parts:
        total_w = sum(w for _, w in rot_parts)
        teams["rotation_score"] = sum(
            (w / total_w) * teams[col] for col, w in rot_parts
        )
    else:
        teams["rotation_score"] = 0.5

    # --- Bullpen score ---
    bp_parts: list[tuple[str, float]] = []
    if "bullpen_depth" in teams.columns:
        teams["bullpen_depth_pctl"] = _pctl(teams["bullpen_depth"].fillna(teams["bullpen_depth"].median()))
        bp_parts.append(("bullpen_depth_pctl", 0.40))
    if "rp_glicko_pctl" in teams.columns:
        teams["rp_glicko_pctl"] = teams["rp_glicko_pctl"].fillna(0.5)
        bp_parts.append(("rp_glicko_pctl", 0.30))
    if "bp_k_rate_pctl" in teams.columns:
        teams["bp_k_rate_pctl"] = teams["bp_k_rate_pctl"].fillna(0.5)
        bp_parts.append(("bp_k_rate_pctl", 0.30))

    if bp_parts:
        total_w = sum(w for _, w in bp_parts)
        teams["bullpen_score"] = sum(
            (w / total_w) * teams[col] for col, w in bp_parts
        )
    else:
        teams["bullpen_score"] = 0.5

    # --- Pitching composite (backward compat) ---
    teams["pitching_score"] = 0.55 * teams["rotation_score"] + 0.45 * teams["bullpen_score"]

    return teams


# ---------------------------------------------------------------------------
# c. Defense Profile
# ---------------------------------------------------------------------------
def _build_defense_profile(
    player_teams: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Defense sub-scores from OAA and catcher framing."""
    from src.data.db import read_sql

    teams = player_teams[["team_id"]].drop_duplicates().copy()

    # Team OAA (sum all fielders)
    try:
        oaa = read_sql("""
            SELECT
                dt.team_id,
                SUM(f.outs_above_average) AS team_oaa
            FROM production.fact_fielding_oaa f
            JOIN production.dim_team dt ON f.team_name = dt.team_name
            WHERE f.season = :season
            GROUP BY dt.team_id
        """, {"season": season})
        teams = teams.merge(oaa, on="team_id", how="left")
    except Exception as e:
        logger.warning("OAA load failed: %s", e)
        teams["team_oaa"] = 0.0

    # Catcher framing
    try:
        framing = read_sql("""
            SELECT
                dt.team_id,
                SUM(cf.runs_extra_strikes) AS catcher_framing_runs
            FROM production.fact_catcher_framing cf
            JOIN production.dim_team dt ON cf.team_name = dt.team_name
            WHERE cf.season = :season
            GROUP BY dt.team_id
        """, {"season": season})
        teams = teams.merge(framing, on="team_id", how="left")
    except Exception as e:
        logger.warning("Catcher framing load failed: %s", e)
        teams["catcher_framing_runs"] = 0.0

    # Defense composite
    teams["team_oaa"] = teams["team_oaa"].fillna(0)
    teams["catcher_framing_runs"] = teams["catcher_framing_runs"].fillna(0)
    teams["oaa_pctl"] = _pctl(teams["team_oaa"])
    teams["framing_pctl"] = _pctl(teams["catcher_framing_runs"])
    teams["defense_score"] = 0.7 * teams["oaa_pctl"] + 0.3 * teams["framing_pctl"]

    return teams


# ---------------------------------------------------------------------------
# d. Organizational Philosophy
# ---------------------------------------------------------------------------
def _build_org_philosophy(
    season: int,
    roster_comp: pd.DataFrame | None,
    prospect_rankings: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build vs buy + farm system quality."""
    rows: list[dict] = []

    if roster_comp is not None and not roster_comp.empty:
        for tid, grp in roster_comp.groupby("team_id"):
            total = len(grp)
            homegrown = grp["acquisition_type"].isin(["Draft", "Signed"]).sum()
            acquired = grp["acquisition_type"].isin(["Free Agent", "Trade", "Waiver"]).sum()
            rows.append({
                "team_id": tid,
                "pct_homegrown": homegrown / max(total, 1),
                "pct_acquired": acquired / max(total, 1),
                "roster_size": total,
            })

    if not rows:
        return pd.DataFrame(columns=["team_id", "pct_homegrown", "org_score"])

    df = pd.DataFrame(rows)

    # Farm system score from prospect rankings
    if prospect_rankings is not None and not prospect_rankings.empty:
        org_col = "org" if "org" in prospect_rankings.columns else None
        score_col = "tdd_prospect_score" if "tdd_prospect_score" in prospect_rankings.columns else None
        rank_col = "overall_rank" if "overall_rank" in prospect_rankings.columns else None

        if org_col and score_col:
            # Map org abbreviations to team_ids
            from src.data.team_queries import get_team_info
            team_info = get_team_info()
            org_to_tid = dict(zip(team_info["abbreviation"], team_info["team_id"].astype(int)))

            prospect_rankings = prospect_rankings.copy()
            prospect_rankings["team_id"] = prospect_rankings[org_col].map(org_to_tid)
            farm = prospect_rankings.dropna(subset=["team_id"]).groupby("team_id").agg(
                farm_avg_score=(score_col, "mean"),
                farm_top_score=(score_col, "max"),
                n_ranked_prospects=(score_col, "count"),
            ).reset_index()
            if rank_col and rank_col in prospect_rankings.columns:
                top100 = prospect_rankings[prospect_rankings[rank_col] <= 100].groupby("team_id").size().reset_index(name="top_100_count")
                farm = farm.merge(top100, on="team_id", how="left")
                farm["top_100_count"] = farm["top_100_count"].fillna(0).astype(int)

            df = df.merge(farm, on="team_id", how="left")

    # Org composite
    score_parts = []
    if "pct_homegrown" in df.columns:
        df["homegrown_pctl"] = _pctl(df["pct_homegrown"])
        score_parts.append("homegrown_pctl")
    if "farm_avg_score" in df.columns:
        df["farm_pctl"] = _pctl(df["farm_avg_score"].fillna(0))
        score_parts.append("farm_pctl")
    if score_parts:
        df["org_score"] = df[score_parts].mean(axis=1)
    else:
        df["org_score"] = 0.5

    return df


# ---------------------------------------------------------------------------
# e. Health & Depth
# ---------------------------------------------------------------------------
def _build_health_depth(
    season: int,
    il_history: pd.DataFrame | None,
    age_profile: pd.DataFrame | None,
    roster_stability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Player-level health aggregation + age trajectory + roster stability.

    Uses individual player health scores from rankings parquets
    (forward-looking durability) rather than raw team IL days
    (backward-looking burden).  Falls back to IL-days if rankings
    are unavailable.

    When *roster_stability* is provided (from ``_build_roster_stability``),
    the ``roster_continuity`` percentile is blended into the composite
    with a small weight (15%) as a confidence modifier.
    """
    from pathlib import Path
    DASHBOARD_DIR = Path(
        "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
    )

    # ── 1. Aggregate player-level health scores per team ──
    # Load rankings + player-team mapping
    team_health: dict[int, float] = {}
    try:
        player_teams = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")
        hitter_r = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
        pitcher_r = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")

        # Map player_id -> team_id
        from src.data.db import read_sql as _rs
        team_map = _rs(
            "SELECT team_id, abbreviation FROM production.dim_team"
        )
        abbr_to_tid = dict(zip(team_map["abbreviation"], team_map["team_id"]))

        # Collect (team_abbr, health_score) pairs
        health_rows: list[tuple[str, float]] = []
        if "health_score" in hitter_r.columns:
            merged_h = hitter_r[["batter_id", "health_score"]].merge(
                player_teams.rename(columns={"player_id": "batter_id"})[["batter_id", "team_abbr"]],
                on="batter_id", how="inner",
            )
            for _, r in merged_h.iterrows():
                if pd.notna(r["health_score"]):
                    health_rows.append((r["team_abbr"], r["health_score"]))

        if "health_score" in pitcher_r.columns:
            merged_p = pitcher_r[["pitcher_id", "health_score"]].merge(
                player_teams.rename(columns={"player_id": "pitcher_id"})[["pitcher_id", "team_abbr"]],
                on="pitcher_id", how="inner",
            )
            for _, r in merged_p.iterrows():
                if pd.notna(r["health_score"]):
                    health_rows.append((r["team_abbr"], r["health_score"]))

        if health_rows:
            health_df = pd.DataFrame(health_rows, columns=["team_abbr", "health_score"])
            team_avg = health_df.groupby("team_abbr")["health_score"].mean()
            for abbr, score in team_avg.items():
                tid = abbr_to_tid.get(abbr)
                if tid:
                    team_health[tid] = score
            logger.info("Player-level health aggregated for %d teams", len(team_health))
    except Exception as e:
        logger.warning("Could not load player health scores: %s", e)

    # ── 2. Fallback: IL days (if player health unavailable) ──
    il_burden: dict[int, float] = {}
    il_days_map: dict[int, int] = {}
    if il_history is not None and not il_history.empty:
        recent_il = il_history[il_history["season"] == season].copy()
        if not recent_il.empty:
            recent_il["il_burden_pctl"] = 1 - _pctl(recent_il["total_il_days"])
            for _, r in recent_il.iterrows():
                il_burden[r["team_id"]] = r["il_burden_pctl"]
                il_val = r["total_il_days"]
                il_days_map[r["team_id"]] = int(il_val) if pd.notna(il_val) else 0

    # ── 3. Age trajectory ──
    age_data: dict[int, dict] = {}
    if age_profile is not None and not age_profile.empty:
        age = age_profile.copy()

        def _age_trajectory(r: pd.Series) -> str:
            if r["pct_under_27"] > 0.40:
                return "Ascending"
            if r["pct_over_31"] > 0.35:
                return "Declining"
            return "Peak"

        age["age_trajectory"] = age.apply(_age_trajectory, axis=1)
        for _, r in age.iterrows():
            age_data[r["team_id"]] = {
                "avg_age": r["avg_age"],
                "age_trajectory": r["age_trajectory"],
            }

    # ── 4. Combine into health_depth_score ──
    all_tids = set(team_health) | set(il_burden) | set(age_data)
    if not all_tids:
        return pd.DataFrame(columns=["team_id", "health_depth_score"])

    rows = []
    for tid in all_tids:
        row: dict = {"team_id": tid}

        # Primary: player-level health (70% weight)
        # Fallback: IL burden percentile
        if tid in team_health:
            row["player_health"] = team_health[tid]
        elif tid in il_burden:
            row["player_health"] = il_burden[tid]
        else:
            row["player_health"] = 0.5

        row["total_il_days"] = il_days_map.get(tid, 0)
        row["avg_age"] = age_data.get(tid, {}).get("avg_age", None)
        row["age_trajectory"] = age_data.get(tid, {}).get("age_trajectory", "Peak")
        rows.append(row)

    result = pd.DataFrame(rows)

    # Age percentile (younger = better for depth)
    if "avg_age" in result.columns and result["avg_age"].notna().any():
        result["age_pctl"] = 1 - _pctl(result["avg_age"].fillna(result["avg_age"].median()))
    else:
        result["age_pctl"] = 0.5

    # ── 5. Roster stability (confidence modifier) ──
    # Merge roster_continuity + detail columns if available.
    if roster_stability is not None and not roster_stability.empty:
        # Need abbr -> team_id mapping to join
        try:
            from src.data.db import read_sql as _rs2
            _tm = _rs2("SELECT team_id, abbreviation FROM production.dim_team")
            _abbr_tid = dict(zip(_tm["abbreviation"], _tm["team_id"]))
            rs = roster_stability.copy()
            rs["team_id"] = rs["team_abbr"].map(_abbr_tid)
            rs = rs.dropna(subset=["team_id"])
            rs["team_id"] = rs["team_id"].astype(int)
            result = result.merge(
                rs[["team_id", "roster_continuity", "players_retained",
                    "players_new", "players_lost"]],
                on="team_id",
                how="left",
            )
            logger.info("Roster stability merged into health & depth")
        except Exception as e:
            logger.warning("Roster stability merge failed: %s", e)

    if "roster_continuity" not in result.columns:
        result["roster_continuity"] = 0.5
        result["players_retained"] = np.nan
        result["players_new"] = np.nan
        result["players_lost"] = np.nan

    result["roster_continuity"] = result["roster_continuity"].fillna(0.5)

    # Composite: 70% player health + 30% age trajectory
    # roster_continuity demoted to metadata only — research shows it's not
    # predictive for one-year forecasts (retained as column for display)
    result["health_depth_score"] = (
        0.70 * result["player_health"]
        + 0.30 * result["age_pctl"]
    )

    return result


# ---------------------------------------------------------------------------
# f. Schedule Context
# ---------------------------------------------------------------------------
def _build_schedule_context(
    elo_history: pd.DataFrame | None,
    team_info: pd.DataFrame | None,
) -> pd.DataFrame:
    """Schedule difficulty from ELO history."""
    if elo_history is None or elo_history.empty:
        return pd.DataFrame(columns=["team_id", "schedule_score"])

    # Use the most recent season
    max_season = elo_history["season"].max()
    recent = elo_history[elo_history["season"] == max_season].copy()

    if recent.empty:
        return pd.DataFrame(columns=["team_id", "schedule_score"])

    # Avg opponent composite ELO faced
    sched = recent.groupby("team_id").agg(
        avg_opp_elo=("effective_elo", "mean"),  # opponent's effective rating when faced
        games_played=("game_pk", "nunique"),
    ).reset_index()

    # Big game record (vs top-10 opponents)
    top10_elo = recent.groupby("opponent_id")["composite_elo"].mean().nlargest(10).index
    big_games = recent[recent["opponent_id"].isin(top10_elo)]
    if not big_games.empty:
        big_record = big_games.groupby("team_id").agg(
            big_game_wins=("win", "sum"),
            big_game_total=("win", "count"),
        ).reset_index()
        big_record["big_game_pct"] = big_record["big_game_wins"] / big_record["big_game_total"].clip(1)
        sched = sched.merge(big_record[["team_id", "big_game_pct"]], on="team_id", how="left")

    # Division competitiveness
    if team_info is not None and "division" in team_info.columns:
        team_div = team_info[["team_id", "division"]].copy()
        team_div["team_id"] = team_div["team_id"].astype(int)
        sched = sched.merge(team_div, on="team_id", how="left")

        if "division" in sched.columns:
            # Get each team's final composite ELO
            team_final_elo = recent.groupby("team_id")["composite_elo"].last().reset_index()
            team_final_elo = team_final_elo.merge(team_div, on="team_id", how="left")

            div_strength = team_final_elo.groupby("division")["composite_elo"].mean().reset_index(
                name="div_avg_elo"
            )
            sched = sched.merge(div_strength, on="division", how="left")

    sched["schedule_score"] = 0.5  # neutral default
    if "avg_opp_elo" in sched.columns:
        sched["schedule_score"] = _pctl(sched["avg_opp_elo"])

    return sched


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------
def build_all_team_profiles(
    season: int,
    projection_season: int,
    elo_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build comprehensive team profiles (one row per team, all sub-scores).

    Parameters
    ----------
    season : int
        Most recent completed season for observed data.
    projection_season : int
        Season being projected (typically season + 1).
    elo_history : pd.DataFrame, optional
        ELO history from ``compute_elo_history()``.

    Returns
    -------
    pd.DataFrame
        One row per team with all sub-scores.
    """
    from src.data.team_queries import (
        get_team_age_profile,
        get_team_il_history,
        get_team_info,
        get_team_park_factors,
        get_team_roster_composition,
        get_team_season_offense,
        get_team_season_pitching,
    )

    logger.info("Building team profiles for %d (projecting %d)...", season, projection_season)

    # Load shared data
    team_info = get_team_info()
    player_teams = _safe_load("player_teams.parquet")
    if player_teams is None:
        logger.error("player_teams.parquet required — cannot build profiles")
        return pd.DataFrame()

    # Map player_teams to include team_id
    if "team_id" not in player_teams.columns and "team_abbr" in player_teams.columns:
        abbr_to_id = dict(zip(team_info["abbreviation"], team_info["team_id"].astype(int)))
        player_teams["team_id"] = player_teams["team_abbr"].map(abbr_to_id)
        player_teams = player_teams.dropna(subset=["team_id"])
        player_teams["team_id"] = player_teams["team_id"].astype(int)

    # Load optional parquets
    hitter_rankings = _safe_load("hitters_rankings.parquet")
    pitcher_rankings = _safe_load("pitchers_rankings.parquet")
    hitter_counting = _safe_load("hitter_counting_sim.parquet")
    pitcher_roles = _safe_load("pitcher_roles.parquet")
    prospect_rankings = _safe_load("prospect_rankings.parquet")

    # Load from DB
    team_offense = get_team_season_offense([season])
    team_pitching = get_team_season_pitching([season])
    il_history = get_team_il_history([season])
    age_profile = get_team_age_profile(season)

    try:
        roster_comp = get_team_roster_composition(season)
    except Exception as e:
        logger.warning("Roster composition failed: %s", e)
        roster_comp = None

    # Park factors for normalizing rate stats
    try:
        park_factors = get_team_park_factors(seasons=[season - 2, season - 1, season])
    except Exception as e:
        logger.warning("Park factor load failed: %s — using neutral factors", e)
        park_factors = None

    # Build each dimension
    logger.info("Building offense profile...")
    offense = _build_offense_profile(
        player_teams, hitter_rankings, hitter_counting, team_offense,
        park_factors=park_factors,
    )

    logger.info("Building pitching profile...")
    pitcher_counting = _safe_load("pitcher_counting_sim.parquet")
    pitching = _build_pitching_profile(
        player_teams, pitcher_rankings, pitcher_roles, team_pitching,
        park_factors=park_factors,
        pitcher_counting=pitcher_counting,
    )

    logger.info("Building defense profile...")
    defense = _build_defense_profile(player_teams, season)

    logger.info("Building organizational philosophy...")
    org = _build_org_philosophy(season, roster_comp, prospect_rankings)

    # Roster stability (2026 roster vs 2025 player_teams)
    roster_stability = None
    try:
        current_roster = _safe_load("roster.parquet")
        if current_roster is not None and player_teams is not None:
            # Filter to MLB-level, active or IL players only
            mlb_roster = current_roster[
                current_roster["roster_status"].str.contains(
                    r"active|il", case=False, na=False
                )
            ].copy()
            roster_stability = _build_roster_stability(mlb_roster, player_teams)
            logger.info("Roster stability computed for %d teams", len(roster_stability))
    except Exception as e:
        logger.warning("Roster stability computation failed: %s", e)

    logger.info("Building health & depth...")
    health = _build_health_depth(season, il_history, age_profile, roster_stability)

    logger.info("Building schedule context...")
    schedule = _build_schedule_context(elo_history, team_info)

    # --- BaseRuns: projected RS/RA from counting sims ---
    # BaseRuns formula: RS = A*B/(B+C) + D
    # A = H+BB+HBP-HR, B = (1.4*TB - 0.6*H - 3*HR + 0.1*(BB+HBP))*1.02
    # C = AB-H (outs on BIP), D = HR
    # Produces forward-looking projected runs, not retrospective
    try:
        pt_pid = "batter_id" if "batter_id" in player_teams.columns else "player_id"
        if hitter_counting is not None and "total_h_mean" in hitter_counting.columns:
            h_id = "batter_id" if "batter_id" in hitter_counting.columns else "player_id"
            hc = hitter_counting.merge(
                player_teams[[pt_pid, "team_id"]].rename(columns={pt_pid: h_id}),
                on=h_id, how="inner",
            )
            team_rs = hc.groupby("team_id").agg(
                h=("total_h_mean", "sum"),
                bb=("total_bb_mean", "sum"),
                hbp=("total_hbp_mean", "sum") if "total_hbp_mean" in hc.columns else ("total_bb_mean", lambda x: 0),
                hr=("total_hr_mean", "sum"),
                tb=("total_tb_mean", "sum") if "total_tb_mean" in hc.columns else ("total_hr_mean", lambda x: 0),
                pa=("total_pa_mean", "sum"),
            ).reset_index()

            # Estimate TB if not available: TB = 1B + 2*2B + 3*3B + 4*HR
            if "total_tb_mean" not in hc.columns:
                tb_cols = {}
                for c in ["total_1b_mean", "total_2b_mean", "total_3b_mean", "total_hr_mean"]:
                    if c in hc.columns:
                        tb_cols[c] = hc.groupby("team_id")[c].sum()
                if tb_cols:
                    tb_df = pd.DataFrame(tb_cols).fillna(0)
                    team_rs["tb"] = (
                        tb_df.get("total_1b_mean", 0)
                        + 2 * tb_df.get("total_2b_mean", 0)
                        + 3 * tb_df.get("total_3b_mean", 0)
                        + 4 * tb_df.get("total_hr_mean", 0)
                    ).values

            if "total_hbp_mean" not in hc.columns:
                team_rs["hbp"] = team_rs["pa"] * 0.008  # ~0.8% of PA

            # BaseRuns
            A = team_rs["h"] + team_rs["bb"] + team_rs["hbp"] - team_rs["hr"]
            B = (1.4 * team_rs["tb"] - 0.6 * team_rs["h"] - 3 * team_rs["hr"]
                 + 0.1 * (team_rs["bb"] + team_rs["hbp"])) * 1.02
            sf_est = team_rs["pa"] * 0.008
            C = (team_rs["pa"] - team_rs["bb"] - team_rs["hbp"] - sf_est) - team_rs["h"]
            D = team_rs["hr"]
            team_rs["baseruns_rs"] = A * B / (B + C).clip(1) + D

            # Normalize to per-game (standard ~38.5 PA/game for a team)
            STD_TEAM_PA_PER_GAME = 38.5
            team_rs["proj_rs_per_game"] = (
                team_rs["baseruns_rs"] / team_rs["pa"] * STD_TEAM_PA_PER_GAME * 162
                / 162
            )
            offense = offense.merge(
                team_rs[["team_id", "baseruns_rs", "proj_rs_per_game"]],
                on="team_id", how="left",
            )
            logger.info("BaseRuns RS computed for %d teams (mean=%.0f runs)",
                        len(team_rs), team_rs["baseruns_rs"].mean())

        # Pitcher side: use simulated runs directly (already calibrated)
        if pitcher_counting is not None and "total_runs_mean" in pitcher_counting.columns:
            p_id = "pitcher_id" if "pitcher_id" in pitcher_counting.columns else "player_id"
            pc = pitcher_counting.merge(
                player_teams[[pt_pid, "team_id"]].rename(columns={pt_pid: p_id}),
                on=p_id, how="inner",
            )
            team_ra = pc.groupby("team_id").agg(
                runs_allowed=("total_runs_mean", "sum"),
                ip=("projected_ip_mean", "sum") if "projected_ip_mean" in pc.columns else ("total_outs_mean", lambda x: x.sum() / 3),
            ).reset_index()
            team_ra["proj_ra_per_game"] = team_ra["runs_allowed"] / (team_ra["ip"] / 9).clip(1)
            pitching = pitching.merge(
                team_ra[["team_id", "proj_ra_per_game"]],
                on="team_id", how="left",
            )
            logger.info("Projected RA/game computed for %d teams (mean=%.2f)",
                        len(team_ra), team_ra["proj_ra_per_game"].mean())
    except Exception as e:
        logger.warning("BaseRuns computation failed: %s", e)

    # Merge all dimensions
    result = offense[["team_id", "offense_score"]].copy()
    for extra_col in ["offense_style", "lineup_depth", "lineup_floor", "rpg", "hr_per_game", "sb_per_game",
                       "proj_R", "proj_HR", "proj_SB", "avg_hitter_score",
                       "baseruns_rs", "proj_rs_per_game"]:
        if extra_col in offense.columns:
            result[extra_col] = offense[extra_col]

    for df, key_cols in [
        (pitching, ["pitching_score", "rotation_score", "bullpen_score",
                     "rotation_strength", "bullpen_depth",
                     "pitching_style", "ra_per_game", "k_per_game",
                     "proj_ra_per_game"]),
        (defense, ["defense_score", "team_oaa", "catcher_framing_runs"]),
        (org, ["org_score", "pct_homegrown", "farm_avg_score", "n_ranked_prospects", "top_100_count"]),
        (health, ["health_depth_score", "total_il_days", "avg_age", "age_trajectory",
                  "roster_continuity", "players_retained", "players_new"]),
        (schedule, ["schedule_score", "avg_opp_elo", "big_game_pct", "div_avg_elo"]),
    ]:
        available = [c for c in key_cols if c in df.columns]
        if available and "team_id" in df.columns:
            result = result.merge(df[["team_id"] + available], on="team_id", how="left")

    # Add team info
    result = result.merge(
        team_info[["team_id", "abbreviation", "team_name", "division", "league"]],
        on="team_id", how="left",
    )

    logger.info("Team profiles built: %d teams", len(result))
    return result
