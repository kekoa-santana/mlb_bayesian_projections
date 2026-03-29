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

# When a player is on IL, their roster slot is filled by a replacement.
# Effective contribution = availability * their_score + (1-avail) * replacement.
# These represent approximate replacement-level scores (0-1 scale).
_REPLACEMENT_HITTER = 0.30
_REPLACEMENT_PITCHER = 0.30


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
        # Use offense_score (not overall composite) so team offense
        # measures actual hitting quality, not fielding/versatility
        score_col = next(
                (c for c in ("offense_score", "current_value_score", "tdd_value_score", "composite_score")
                 if c in hitter_rankings.columns), "composite_score"
            )
        if score_col in hitter_rankings.columns:
            pt_pid = "batter_id" if "batter_id" in player_teams.columns else "player_id"
            merge_cols = [pt_pid, "team_id"]
            if "availability" in player_teams.columns:
                merge_cols.append("availability")
            rank_merged = hitter_rankings.merge(
                player_teams[merge_cols].rename(columns={pt_pid: pid_col}),
                on=pid_col,
                how="inner",
            )

            # Save raw (full-health) score for ceiling calculation
            rank_merged["_raw_score"] = rank_merged[score_col].copy()

            # Blend IL player scores with replacement level.
            # IL-60: 10% their score + 90% replacement — someone else fills
            # the slot, but they'll return and add value when healthy.
            if "availability" in rank_merged.columns:
                avail = rank_merged["availability"]
                rank_merged[score_col] = (
                    avail * rank_merged[score_col]
                    + (1 - avail) * _REPLACEMENT_HITTER
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

            # Scouting grade columns to aggregate at team level
            _GRADE_COLS = ["grade_hit", "grade_power", "grade_speed",
                           "grade_fielding", "grade_discipline", "tools_rating"]

            def _pa_weighted_score(g: pd.DataFrame) -> pd.Series:
                top9 = g.nlargest(9, score_col)
                pa_w = top9[pa_weight_col].values
                scores = top9[score_col].values
                wtd_avg = float(np.average(scores, weights=pa_w)) if pa_w.sum() > 0 else scores.mean()
                result = {
                    "lineup_depth": (g[score_col] >= median_score).sum() / min(len(g), 9),
                    "avg_hitter_score": wtd_avg,
                    "lineup_floor": g.nlargest(min(len(g), 8), score_col)[score_col].iloc[-1] if len(g) >= 7 else 0.0,
                }
                # Ceiling: full-health top-9 (no IL discount)
                if "_raw_score" in g.columns:
                    top9_ceil = g.nlargest(9, "_raw_score")
                    ceil_scores = top9_ceil["_raw_score"].values
                    ceil_w = top9_ceil[pa_weight_col].values
                    result["avg_hitter_score_ceiling"] = (
                        float(np.average(ceil_scores, weights=ceil_w))
                        if ceil_w.sum() > 0 else ceil_scores.mean()
                    )
                # Aggregate scouting grades for top-9 lineup
                for gc in _GRADE_COLS:
                    if gc in top9.columns and top9[gc].notna().any():
                        vals = top9[gc].dropna()
                        w = pa_w[:len(vals)] if len(vals) == len(pa_w) else None
                        key = f"lineup_{gc}" if gc != "tools_rating" else "lineup_diamond"
                        result[key] = float(np.average(vals, weights=w)) if w is not None and len(w) > 0 else vals.mean()
                    else:
                        key = f"lineup_{gc}" if gc != "tools_rating" else "lineup_diamond"
                        result[key] = np.nan
                return pd.Series(result)

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

    # Offense composite score (percentile-based, weighted)
    # avg_hitter_score and rpg capture actual offensive quality;
    # lineup_floor and lineup_depth are tiebreakers, not drivers.
    _OFF_WEIGHTS = {
        "avg_hitter_score": 0.40,
        "rpg": 0.30,
        "lineup_floor": 0.20,
        "lineup_depth": 0.10,
    }
    score_cols = [c for c in _OFF_WEIGHTS if c in teams.columns]
    if score_cols:
        for c in score_cols:
            teams[f"{c}_pctl"] = _pctl(teams[c].fillna(teams[c].median()))
        weights = np.array([_OFF_WEIGHTS[c] for c in score_cols])
        weights = weights / weights.sum()  # renormalize if any cols missing
        pctl_cols = [f"{c}_pctl" for c in score_cols]
        teams["offense_score"] = sum(
            w * teams[pc] for w, pc in zip(weights, pctl_cols)
        )
    else:
        teams["offense_score"] = 0.5

    # Offense ceiling: same formula but using full-health avg_hitter_score
    if "avg_hitter_score_ceiling" in teams.columns:
        ceil_weights = _OFF_WEIGHTS.copy()
        ceil_weights["avg_hitter_score"] = ceil_weights.pop("avg_hitter_score")
        teams["_ceil_hitter_pctl"] = _pctl(
            teams["avg_hitter_score_ceiling"].fillna(teams["avg_hitter_score_ceiling"].median())
        )
        # Use ceiling hitter score + same rpg/floor/depth
        ceil_pctls = []
        ceil_w = []
        for c, w in _OFF_WEIGHTS.items():
            if c == "avg_hitter_score":
                ceil_pctls.append(teams["_ceil_hitter_pctl"])
            elif f"{c}_pctl" in teams.columns:
                ceil_pctls.append(teams[f"{c}_pctl"])
            else:
                continue
            ceil_w.append(w)
        if ceil_pctls:
            cw = np.array(ceil_w)
            cw = cw / cw.sum()
            teams["offense_ceiling"] = sum(
                w * p for w, p in zip(cw, ceil_pctls)
            )
        teams.drop(columns=["_ceil_hitter_pctl"], inplace=True, errors="ignore")
    else:
        teams["offense_ceiling"] = teams["offense_score"]

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
    pitcher_fip: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pitching sub-scores per team (rotation + bullpen split)."""
    pt_pid = "player_id" if "player_id" in player_teams.columns else "batter_id"
    teams = player_teams[["team_id"]].drop_duplicates().copy()

    # --- Rotation strength (top 5 SP by value score) ---
    pr_merged = None  # keep reference for Glicko/ERA joins below
    if pitcher_rankings is not None:
        pid_col = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"
        score_col = next(
            (c for c in ("current_value_score", "tdd_value_score", "composite_score")
             if c in pitcher_rankings.columns), "composite_score"
        )
        role_col = "role" if "role" in pitcher_rankings.columns else "position"

        if score_col in pitcher_rankings.columns:
            merge_cols = ["team_id", pt_pid]
            if "availability" in player_teams.columns:
                merge_cols.append("availability")
            pr_merged = pitcher_rankings.merge(
                player_teams[merge_cols].rename(columns={pt_pid: pid_col}),
                on=pid_col,
                how="inner",
            )

            # Save raw for ceiling
            pr_merged["_raw_score"] = pr_merged[score_col].copy()

            # Blend IL pitcher scores with replacement level
            if "availability" in pr_merged.columns:
                avail = pr_merged["availability"]
                pr_merged[score_col] = (
                    avail * pr_merged[score_col]
                    + (1 - avail) * _REPLACEMENT_PITCHER
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

            # SP rotation (IP-weighted) + scouting grade aggregation
            sp_mask = pr_merged[role_col].isin(["SP"]) if role_col in pr_merged.columns else pd.Series(True, index=pr_merged.index)
            sp_data = pr_merged[sp_mask]

            _P_GRADE_COLS = ["grade_stuff", "grade_command", "grade_durability", "tools_rating"]
            _REPLACEMENT_GRADE = 4.0  # replacement-level diamond rating
            _MIN_SP = 5  # every team needs at least 5 SP
            _REPLACEMENT_SP = 0.40  # replacement-level tdd_value

            def _ip_weighted_rot(g: pd.DataFrame) -> pd.Series:
                top5 = g.nlargest(5, score_col)
                w = top5[ip_weight_col].values
                s = top5[score_col].values

                # If fewer than 5 SP, fill with replacement-level grades.
                # A team with only 2 elite SP still needs 3 more arms.
                n_actual = len(g)
                if n_actual < _MIN_SP:
                    n_fill = _MIN_SP - n_actual
                    s = np.append(s, [_REPLACEMENT_SP] * n_fill)
                    w = np.append(w, [100.0] * n_fill)  # neutral IP weight

                base_strength = float(np.average(s, weights=w)) if w.sum() > 0 else s.mean()

                # Rotation depth bonus: SPs beyond the top 5 provide insurance
                # against injuries and mid-season fatigue. Steep diminishing
                # returns: SP6 at 50% credit, SP7 at 25%, SP8 at 12.5%.
                # Uses full-health scores (_raw_score) since depth = who can
                # step in when healthy, not current IL-discounted value.
                depth_bonus = 0.0
                depth_col = "_raw_score" if "_raw_score" in g.columns else score_col
                if n_actual > _MIN_SP:
                    extras = g.nlargest(n_actual, depth_col).iloc[_MIN_SP:]
                    for i, (_, sp_row) in enumerate(extras.iterrows()):
                        discount = 0.5 ** (i + 1)  # 0.50, 0.25, 0.125, ...
                        # Credit = how much better than replacement × discount
                        above_repl = max(0, sp_row[depth_col] - _REPLACEMENT_SP)
                        depth_bonus += discount * above_repl
                    # Scale so max realistic bonus ≈ 0.03-0.05
                    depth_bonus *= 0.15

                result = {
                    "rotation_strength": base_strength + depth_bonus,
                    "rotation_sp_count": n_actual,
                    "rotation_depth_bonus": round(depth_bonus, 4),
                }

                # Ceiling: full-health rotation
                if "_raw_score" in g.columns:
                    ceil_top5 = g.nlargest(5, "_raw_score")
                    cs = ceil_top5["_raw_score"].values
                    cw = ceil_top5[ip_weight_col].values
                    n_c = len(ceil_top5)
                    if n_c < _MIN_SP:
                        cs = np.append(cs, [0.40] * (_MIN_SP - n_c))
                        cw = np.append(cw, [100.0] * (_MIN_SP - n_c))
                    result["rotation_strength_ceiling"] = (
                        float(np.average(cs, weights=cw)) if cw.sum() > 0 else cs.mean()
                    )
                # Aggregate pitcher scouting grades for top-5 rotation
                for gc in _P_GRADE_COLS:
                    if gc in top5.columns and top5[gc].notna().any():
                        vals = list(top5[gc].dropna())
                        # Fill to 5 with replacement grade
                        while len(vals) < _MIN_SP:
                            vals.append(_REPLACEMENT_GRADE if gc == "tools_rating" else 40)
                        wv = w[:len(vals)]
                        key = f"rotation_{gc}" if gc != "tools_rating" else "rotation_diamond"
                        result[key] = float(np.average(vals, weights=wv))
                    else:
                        key = f"rotation_{gc}" if gc != "tools_rating" else "rotation_diamond"
                        result[key] = np.nan
                return pd.Series(result)

            rot_strength = sp_data.groupby("team_id").apply(
                _ip_weighted_rot, include_groups=False,
            ).reset_index()
            teams = teams.merge(rot_strength, on="team_id", how="left")

    # --- Bullpen depth (role-weighted quality) ---
    if pitcher_roles is not None and pitcher_rankings is not None:
        pid_col_r = "pitcher_id" if "pitcher_id" in pitcher_roles.columns else "player_id"
        bp_merge_cols = ["team_id", pt_pid]
        if "availability" in player_teams.columns:
            bp_merge_cols.append("availability")
        roles_merged = pitcher_roles.merge(
            player_teams[bp_merge_cols].rename(columns={pt_pid: pid_col_r}),
            on=pid_col_r,
            how="inner",
        )
        # Join value scores
        score_col_r = next(
            (c for c in ("current_value_score", "tdd_value_score", "composite_score")
             if c in pitcher_rankings.columns), "composite_score"
        )
        pid_col_pr = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"
        if score_col_r in pitcher_rankings.columns:
            roles_merged = roles_merged.merge(
                pitcher_rankings[[pid_col_pr, score_col_r]].rename(columns={pid_col_pr: pid_col_r}),
                on=pid_col_r,
                how="left",
            )
            # Blend IL bullpen scores with replacement level
            if "availability" in roles_merged.columns:
                avail = roles_merged["availability"]
                roles_merged[score_col_r] = (
                    avail * roles_merged[score_col_r]
                    + (1 - avail) * _REPLACEMENT_PITCHER
                )
            # Role weights: CL 1.5x, SU 1.0x, MR 0.5x
            role_w_col = "role" if "role" in roles_merged.columns else "predicted_role"
            if role_w_col in roles_merged.columns:
                role_weights = {"CL": 1.5, "SU": 1.0, "MR": 0.5}
                roles_merged["role_weight"] = roles_merged[role_w_col].map(role_weights).fillna(0.5)
                roles_merged["weighted_score"] = roles_merged[score_col_r].fillna(0) * roles_merged["role_weight"]

                # Also merge scouting grades for bullpen aggregation
                _BP_GRADE_COLS = ["grade_stuff", "grade_command", "tools_rating"]
                bp_grade_available = [gc for gc in _BP_GRADE_COLS if gc in pitcher_rankings.columns]
                if bp_grade_available:
                    roles_merged = roles_merged.merge(
                        pitcher_rankings[[pid_col_pr] + bp_grade_available].rename(columns={pid_col_pr: pid_col_r}),
                        on=pid_col_r, how="left",
                    )

                _STANDARD_RP = 7  # standardize all bullpens to 7 relievers

                def _bp_agg(g: pd.DataFrame) -> pd.Series:
                    rw = g["role_weight"]
                    n_actual = len(g)
                    result = {
                        "bullpen_depth": g["weighted_score"].sum() / rw.sum() if rw.sum() > 0 else 0,
                        "bullpen_rp_count": n_actual,
                    }
                    for gc in _BP_GRADE_COLS:
                        if gc in g.columns and g[gc].notna().any():
                            vals = list(g[gc].dropna())
                            w = list(rw.loc[g[gc].notna().values if hasattr(g[gc].notna(), 'values') else g.index])
                            # Standardize to _STANDARD_RP relievers:
                            # - If fewer, pad with replacement-level (penalizes thin pens)
                            # - If more, use only top _STANDARD_RP by role weight
                            if len(vals) > _STANDARD_RP:
                                # Sort by value descending, keep top N
                                paired = sorted(zip(vals, w), key=lambda x: x[0], reverse=True)
                                vals = [p[0] for p in paired[:_STANDARD_RP]]
                                w = [p[1] for p in paired[:_STANDARD_RP]]
                            while len(vals) < _STANDARD_RP:
                                vals.append(4.0 if gc == "tools_rating" else 40)
                                w.append(0.5)  # MR weight
                            key = f"bullpen_{gc}" if gc != "tools_rating" else "bullpen_diamond"
                            result[key] = float(np.average(vals, weights=w))
                        else:
                            key = f"bullpen_{gc}" if gc != "tools_rating" else "bullpen_diamond"
                            result[key] = np.nan
                    return pd.Series(result)

                bp_depth = roles_merged.groupby("team_id").apply(
                    _bp_agg, include_groups=False,
                ).reset_index()
                teams = teams.merge(bp_depth, on="team_id", how="left")

    # --- FIP per role (SP / RP) ---
    # FIP measures pitcher talent independent of defense/luck.
    # Replaces Glicko + ERA + BP K-rate as the backward-looking signal.
    if pitcher_fip is not None and pitcher_rankings is not None:
        pid_col_f = "pitcher_id"
        role_col_f = "role" if "role" in pitcher_rankings.columns else "position"
        pid_col_pr2 = "player_id" if "player_id" in pitcher_rankings.columns else "pitcher_id"

        # Use most recent season FIP
        recent_fip = pitcher_fip.sort_values("season", ascending=False).drop_duplicates("pitcher_id")

        # Map pitcher -> team + role
        fip_team = recent_fip[["pitcher_id", "fip", "ip"]].merge(
            player_teams[["team_id", pt_pid]].rename(columns={pt_pid: pid_col_f}),
            on=pid_col_f,
            how="inner",
        )
        fip_team = fip_team.merge(
            pitcher_rankings[[pid_col_pr2, role_col_f]].rename(columns={pid_col_pr2: pid_col_f}),
            on=pid_col_f,
            how="inner",
        )

        # SP FIP: IP-weighted avg of top 5 SP (lower = better)
        sp_fip = fip_team[fip_team[role_col_f] == "SP"]
        if not sp_fip.empty:
            def _sp_fip_agg(g: pd.DataFrame) -> float:
                top5 = g.nlargest(5, "ip")
                return float(np.average(top5["fip"], weights=top5["ip"]))

            sp_fip_agg = sp_fip.groupby("team_id").apply(
                _sp_fip_agg, include_groups=False,
            ).reset_index(name="sp_fip_avg")
            # Lower FIP = better, so invert the percentile
            sp_fip_agg["sp_fip_pctl"] = 1 - _pctl(sp_fip_agg["sp_fip_avg"])
            teams = teams.merge(sp_fip_agg[["team_id", "sp_fip_pctl"]], on="team_id", how="left")
            logger.info("SP FIP aggregated for %d teams (range %.2f–%.2f)",
                        len(sp_fip_agg), sp_fip_agg["sp_fip_avg"].min(), sp_fip_agg["sp_fip_avg"].max())

        # RP FIP: IP-weighted avg of all RP (lower = better)
        rp_fip = fip_team[fip_team[role_col_f] == "RP"]
        if not rp_fip.empty:
            def _rp_fip_agg(g: pd.DataFrame) -> float:
                return float(np.average(g["fip"], weights=g["ip"]))

            rp_fip_agg = rp_fip.groupby("team_id").apply(
                _rp_fip_agg, include_groups=False,
            ).reset_index(name="rp_fip_avg")
            rp_fip_agg["bp_fip_pctl"] = 1 - _pctl(rp_fip_agg["rp_fip_avg"])
            teams = teams.merge(rp_fip_agg[["team_id", "bp_fip_pctl"]], on="team_id", how="left")
            logger.info("RP FIP aggregated for %d teams (range %.2f–%.2f)",
                        len(rp_fip_agg), rp_fip_agg["rp_fip_avg"].min(), rp_fip_agg["rp_fip_avg"].max())
    else:
        logger.warning("Pitcher FIP data not available — rotation/bullpen scores will be projection-only")

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
    # rotation_strength (projected talent, 60%) + sp_fip_pctl (backward talent, 40%)
    rot_parts: list[tuple[str, float]] = []
    if "rotation_strength" in teams.columns:
        teams["rotation_strength_pctl"] = _pctl(teams["rotation_strength"].fillna(teams["rotation_strength"].median()))
        rot_parts.append(("rotation_strength_pctl", 0.60))
    if "sp_fip_pctl" in teams.columns:
        teams["sp_fip_pctl"] = teams["sp_fip_pctl"].fillna(0.5)
        rot_parts.append(("sp_fip_pctl", 0.40))

    if rot_parts:
        total_w = sum(w for _, w in rot_parts)
        teams["rotation_score"] = sum(
            (w / total_w) * teams[col] for col, w in rot_parts
        )
    else:
        teams["rotation_score"] = 0.5

    # --- Bullpen score ---
    # bullpen_depth (projected talent, 50%) + bp_fip_pctl (backward talent, 50%)
    bp_parts: list[tuple[str, float]] = []
    if "bullpen_depth" in teams.columns:
        teams["bullpen_depth_pctl"] = _pctl(teams["bullpen_depth"].fillna(teams["bullpen_depth"].median()))
        bp_parts.append(("bullpen_depth_pctl", 0.50))
    if "bp_fip_pctl" in teams.columns:
        teams["bp_fip_pctl"] = teams["bp_fip_pctl"].fillna(0.5)
        bp_parts.append(("bp_fip_pctl", 0.50))

    if bp_parts:
        total_w = sum(w for _, w in bp_parts)
        teams["bullpen_score"] = sum(
            (w / total_w) * teams[col] for col, w in bp_parts
        )
    else:
        teams["bullpen_score"] = 0.5

    # --- Pitching composite (backward compat) ---
    teams["pitching_score"] = 0.55 * teams["rotation_score"] + 0.45 * teams["bullpen_score"]

    # Pitching ceiling: use full-health rotation strength
    if "rotation_strength_ceiling" in teams.columns:
        # Recompute rotation_score_ceiling from raw ceiling strength
        if "rotation_strength" in teams.columns:
            ceil_rot_pctl = _pctl(teams["rotation_strength_ceiling"].fillna(
                teams["rotation_strength_ceiling"].median()
            ))
            teams["pitching_ceiling"] = 0.55 * ceil_rot_pctl + 0.45 * teams["bullpen_score"]
        else:
            teams["pitching_ceiling"] = teams["pitching_score"]
    else:
        teams["pitching_ceiling"] = teams["pitching_score"]

    return teams


# ---------------------------------------------------------------------------
# c. Defense Profile
# ---------------------------------------------------------------------------
def _build_defense_profile(
    player_teams: pd.DataFrame,
    season: int,
    pitcher_rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Defense sub-scores from OAA, catcher framing, and pitcher-defense synergy."""
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

    # Infield vs outfield OAA split
    try:
        oaa_split = read_sql("""
            SELECT
                dt.team_id,
                SUM(CASE WHEN f.position IN ('SS','2B','3B','1B')
                    THEN f.outs_above_average ELSE 0 END) AS infield_oaa,
                SUM(CASE WHEN f.position IN ('CF','RF','LF')
                    THEN f.outs_above_average ELSE 0 END) AS outfield_oaa
            FROM production.fact_fielding_oaa f
            JOIN production.dim_team dt ON f.team_name = dt.team_name
            WHERE f.season = :season
            GROUP BY dt.team_id
        """, {"season": season})
        teams = teams.merge(oaa_split, on="team_id", how="left")
        teams["infield_oaa"] = teams["infield_oaa"].fillna(0)
        teams["outfield_oaa"] = teams["outfield_oaa"].fillna(0)
    except Exception as e:
        logger.warning("OAA split load failed: %s", e)
        teams["infield_oaa"] = 0.0
        teams["outfield_oaa"] = 0.0

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

    # --- Pitcher-defense synergy ---
    # GB-heavy staff + strong infield = bonus
    # FB-heavy staff + strong outfield = bonus
    # Mismatches = penalty
    teams["pitcher_defense_synergy"] = 0.0
    if pitcher_rankings is not None and "gb_pct" in pitcher_rankings.columns:
        try:
            pid_col = "pitcher_id" if "pitcher_id" in pitcher_rankings.columns else "player_id"
            pt_pid = "batter_id" if "batter_id" in player_teams.columns else "player_id"
            bf_col = "batters_faced" if "batters_faced" in pitcher_rankings.columns else None

            pr_team = pitcher_rankings.merge(
                player_teams[[pt_pid, "team_id"]].rename(columns={pt_pid: pid_col}),
                on=pid_col, how="inner",
            )

            # BF-weighted staff GB%/FB%
            if bf_col and bf_col in pr_team.columns:
                w = pr_team[bf_col].fillna(100).clip(10)
            else:
                w = pd.Series(1.0, index=pr_team.index)

            pr_team["_w"] = w
            staff_bb = pr_team.groupby("team_id").apply(
                lambda g: pd.Series({
                    "staff_gb_pct": np.average(g["gb_pct"].fillna(0.44), weights=g["_w"]),
                    "staff_fb_pct": np.average(
                        g["fb_pct"].fillna(0.35) if "fb_pct" in g.columns
                        else 1.0 - g["gb_pct"].fillna(0.44),
                        weights=g["_w"],
                    ),
                }),
                include_groups=False,
            ).reset_index()
            teams = teams.merge(staff_bb, on="team_id", how="left")
            teams["staff_gb_pct"] = teams["staff_gb_pct"].fillna(0.44)
            teams["staff_fb_pct"] = teams["staff_fb_pct"].fillna(0.35)

            # Percentile-rank infield and outfield OAA
            if_pctl = _pctl(teams["infield_oaa"])
            of_pctl = _pctl(teams["outfield_oaa"])

            # Synergy: how well does the staff's batted-ball profile
            # align with the team's defensive strengths?
            # GB contribution: staff_gb_pct * infield_oaa_pctl
            # FB contribution: staff_fb_pct * outfield_oaa_pctl
            # Normalize so a perfectly aligned team scores ~0.5+
            gb_alignment = teams["staff_gb_pct"] * if_pctl
            fb_alignment = teams["staff_fb_pct"] * of_pctl
            # Weight GB heavier since ground balls are more fieldable
            raw_synergy = 0.60 * gb_alignment + 0.40 * fb_alignment
            teams["pitcher_defense_synergy"] = _pctl(raw_synergy)

            logger.info(
                "Pitcher-defense synergy: staff GB%% range [%.1f%%, %.1f%%], "
                "synergy range [%.2f, %.2f]",
                teams["staff_gb_pct"].min() * 100,
                teams["staff_gb_pct"].max() * 100,
                teams["pitcher_defense_synergy"].min(),
                teams["pitcher_defense_synergy"].max(),
            )
        except Exception as e:
            logger.warning("Pitcher-defense synergy failed: %s", e)

    # Defense composite: OAA + framing + synergy
    teams["team_oaa"] = teams["team_oaa"].fillna(0)
    teams["catcher_framing_runs"] = teams["catcher_framing_runs"].fillna(0)
    teams["oaa_pctl"] = _pctl(teams["team_oaa"])
    teams["framing_pctl"] = _pctl(teams["catcher_framing_runs"])

    # Positional-weighted team fielding from per-position scouting grades.
    # A 55-grade fielding SS is more valuable than a 55-grade fielding LF.
    _POS_FIELDING_MULT = {
        "C": 1.15, "SS": 1.15,
        "2B": 1.05, "3B": 1.05, "CF": 1.05,
        "LF": 0.95, "RF": 0.95,
        "1B": 0.85, "DH": 0.85,
    }
    elig = _safe_load("hitter_position_eligibility.parquet")
    if elig is not None and "grade_fielding_at_pos" in elig.columns:
        # Map players to teams
        pt_pid = "batter_id" if "batter_id" in player_teams.columns else "player_id"
        elig_team = elig.merge(
            player_teams[["team_id", pt_pid]].rename(columns={pt_pid: "player_id"}),
            on="player_id", how="inner",
        )
        # Apply positional multiplier
        elig_team["weighted_grade"] = (
            elig_team["grade_fielding_at_pos"]
            * elig_team["position"].map(_POS_FIELDING_MULT).fillna(1.0)
        )
        # Primary position only for team aggregation
        primary = elig_team[elig_team["is_primary"]].copy()
        if not primary.empty:
            team_fielding_grade = primary.groupby("team_id")["weighted_grade"].mean().reset_index(
                name="team_fielding_grade",
            )
            teams = teams.merge(team_fielding_grade, on="team_id", how="left")
            teams["team_fielding_grade"] = teams["team_fielding_grade"].fillna(50)
            # Percentile-rank for blending into defense_score
            fielding_grade_pctl = _pctl(teams["team_fielding_grade"])
        else:
            fielding_grade_pctl = pd.Series(0.5, index=teams.index)
            teams["team_fielding_grade"] = 50
    else:
        fielding_grade_pctl = pd.Series(0.5, index=teams.index)
        teams["team_fielding_grade"] = 50

    # Blend: 35% scouting fielding grades (positional-weighted) + 30% raw OAA
    # + 20% catcher framing + 15% pitcher-defense synergy
    teams["defense_score"] = (
        0.35 * fielding_grade_pctl
        + 0.30 * teams["oaa_pctl"]
        + 0.20 * teams["framing_pctl"]
        + 0.15 * teams["pitcher_defense_synergy"]
    )

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

    # ── 6. Roster depth metrics (positional coverage + gaps + bench) ──
    # Uses career-weighted position eligibility and per-position OAA to
    # build a realistic depth chart, then scores roster resilience.
    try:
        player_teams_df = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")
        hitter_r = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
        pitcher_r = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")
        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")

        from src.data.db import read_sql as _rs3
        _tm3 = _rs3("SELECT team_id, abbreviation FROM production.dim_team")
        _abbr_tid3 = dict(zip(_tm3["abbreviation"], _tm3["team_id"]))

        h_score_col = next(
            (c for c in ("current_value_score", "tdd_value_score")
             if c in hitter_r.columns), None
        )
        p_score_col = next(
            (c for c in ("current_value_score", "tdd_value_score")
             if c in pitcher_r.columns), None
        )

        FIELDING_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        ALL_LINEUP_SPOTS = FIELDING_POSITIONS + ["DH"]

        depth_rows = []
        if h_score_col and p_score_col:
            h_pid = "batter_id" if "batter_id" in hitter_r.columns else "player_id"
            p_pid = "pitcher_id" if "pitcher_id" in pitcher_r.columns else "player_id"
            pt_pid = "batter_id" if "batter_id" in player_teams_df.columns else "player_id"

            # Map players to teams
            h_teams = hitter_r.merge(
                player_teams_df[[pt_pid, "team_abbr"]].rename(columns={pt_pid: h_pid}),
                on=h_pid, how="inner",
            )
            p_teams = pitcher_r[[p_pid, p_score_col]].merge(
                player_teams_df[["player_id", "team_abbr"]].rename(
                    columns={"player_id": p_pid}),
                on=p_pid, how="inner",
            )

            # Build per-player position eligibility with OAA
            elig_pid = "batter_id" if "batter_id" in elig.columns else "player_id"
            if elig_pid != "player_id":
                elig = elig.rename(columns={elig_pid: "player_id"})
            oaa_col = "oaa" if "oaa" in elig.columns else None

            for abbr in h_teams["team_abbr"].unique():
                tid = _abbr_tid3.get(abbr)
                if tid is None:
                    continue

                team_h = h_teams[h_teams["team_abbr"] == abbr].copy()
                team_pids = set(team_h[h_pid])

                # Player eligibility for this team
                team_elig = elig[elig["player_id"].isin(team_pids)].copy()

                # Build score + OAA lookup per (player, position)
                score_lookup = dict(zip(team_h[h_pid], team_h[h_score_col]))
                oaa_lookup = {}
                if oaa_col and not team_elig.empty:
                    for _, erow in team_elig.iterrows():
                        oaa_lookup[(erow["player_id"], erow["position"])] = (
                            erow[oaa_col] if pd.notna(erow[oaa_col]) else 0
                        )

                # Eligibility sets: which positions can each player play?
                player_positions: dict[int, set[str]] = {}
                for _, erow in team_elig.iterrows():
                    pid = erow["player_id"]
                    pos = erow["position"]
                    if pos in ALL_LINEUP_SPOTS:
                        player_positions.setdefault(pid, set()).add(pos)
                # Add DH eligibility for everyone
                for pid in team_pids:
                    player_positions.setdefault(pid, set()).add("DH")

                # --- Positional assignment (3-pass) ---
                assigned: dict[str, int] = {}    # position -> player_id
                used: set[int] = set()

                # Pass 1: Lock single-fielding-position players
                for pid, positions in player_positions.items():
                    field_pos = positions & set(FIELDING_POSITIONS)
                    if len(field_pos) == 1:
                        pos = field_pos.pop()
                        if pos not in assigned:
                            assigned[pos] = pid
                            used.add(pid)
                        elif score_lookup.get(pid, 0) > score_lookup.get(assigned[pos], 0):
                            # Better player bumps the existing one
                            old = assigned[pos]
                            assigned[pos] = pid
                            used.discard(old)
                            used.add(pid)

                # Pass 2: Fill remaining positions by best OAA at each spot
                open_positions = [p for p in FIELDING_POSITIONS if p not in assigned]
                for pos in open_positions:
                    candidates = [
                        pid for pid, positions in player_positions.items()
                        if pos in positions and pid not in used
                    ]
                    if not candidates:
                        continue
                    # Sort by OAA at this position (desc), then by value (desc)
                    candidates.sort(key=lambda pid: (
                        oaa_lookup.get((pid, pos), 0),
                        score_lookup.get(pid, 0),
                    ), reverse=True)
                    assigned[pos] = candidates[0]
                    used.add(candidates[0])

                # Pass 3: DH = best remaining bat
                if "DH" not in assigned:
                    remaining = [
                        pid for pid in team_pids if pid not in used
                    ]
                    if remaining:
                        remaining.sort(
                            key=lambda pid: score_lookup.get(pid, 0),
                            reverse=True,
                        )
                        assigned["DH"] = remaining[0]
                        used.add(remaining[0])

                # --- Scoring ---
                # Positional coverage: starter value at each filled position
                pos_values = [
                    score_lookup.get(assigned.get(pos), 0)
                    for pos in ALL_LINEUP_SPOTS
                ]
                filled = sum(1 for v in pos_values if v > 0)
                positional_avg = np.mean(pos_values) if pos_values else 0
                weakest_link = min(pos_values) if pos_values else 0
                coverage_pct = filled / len(ALL_LINEUP_SPOTS)

                # Multi-position backup: positions with a qualified backup
                backup_count = 0
                multi_pos_players = 0
                for pos in FIELDING_POSITIONS:
                    starter = assigned.get(pos)
                    backups = [
                        pid for pid, positions in player_positions.items()
                        if pos in positions and pid != starter
                        and pid in team_pids
                    ]
                    if backups:
                        backup_count += 1
                for pid, positions in player_positions.items():
                    field_pos = positions & set(FIELDING_POSITIONS)
                    if len(field_pos) >= 2:
                        multi_pos_players += 1
                versatility = (
                    0.60 * (backup_count / len(FIELDING_POSITIONS))
                    + 0.40 * min(multi_pos_players / 5, 1.0)
                )

                # Traditional depth metrics (keep existing)
                h_vals = team_h[h_score_col].sort_values(ascending=False).values
                n_h = len(h_vals)
                if n_h >= 5:
                    top3_share = h_vals[:3].sum() / (h_vals.sum() + 1e-9)
                    starter_avg = h_vals[:min(9, n_h)].mean()
                    bench_avg = h_vals[9:min(15, n_h)].mean() if n_h > 9 else 0.30
                    h_gap = starter_avg - bench_avg
                    h_bench_q = bench_avg
                else:
                    top3_share = 0.33
                    h_gap = 0.20
                    h_bench_q = 0.30

                # Pitcher depth
                p_vals = (
                    p_teams[p_teams["team_abbr"] == abbr][p_score_col]
                    .sort_values(ascending=False).values
                )
                n_p = len(p_vals)
                p_bench_q = p_vals[5:min(10, n_p)].mean() if n_p > 5 else 0.30
                roster_breadth = n_h + n_p

                depth_rows.append({
                    "team_id": tid,
                    # Traditional
                    "concentration": top3_share,
                    "replacement_gap": h_gap,
                    "bench_quality": h_bench_q,
                    "pitcher_bench": p_bench_q,
                    "roster_breadth": roster_breadth,
                    # Positional
                    "positional_avg": positional_avg,
                    "weakest_link": weakest_link,
                    "coverage_pct": coverage_pct,
                    "versatility": versatility,
                })

        if depth_rows:
            depth_df = pd.DataFrame(depth_rows)

            # Traditional depth percentiles
            conc_pctl = 1.0 - _pctl(depth_df["concentration"])
            gap_pctl = 1.0 - _pctl(depth_df["replacement_gap"])
            bench_pctl = _pctl(depth_df["bench_quality"])
            pbench_pctl = _pctl(depth_df["pitcher_bench"])

            # Positional depth percentiles
            posavg_pctl = _pctl(depth_df["positional_avg"])
            weakest_pctl = _pctl(depth_df["weakest_link"])
            coverage_pctl = _pctl(depth_df["coverage_pct"])
            versatility_pctl = _pctl(depth_df["versatility"])

            # Positional coverage sub-score
            depth_df["positional_fit"] = (
                0.50 * posavg_pctl
                + 0.30 * weakest_pctl
                + 0.20 * coverage_pctl
            )

            # Combined roster depth: traditional (40%) + positional (40%) + versatility (20%)
            depth_df["roster_depth_score"] = (
                # Traditional (40%)
                0.12 * conc_pctl
                + 0.10 * gap_pctl
                + 0.10 * bench_pctl
                + 0.08 * pbench_pctl
                # Positional fit (40%)
                + 0.40 * depth_df["positional_fit"]
                # Bench versatility (20%)
                + 0.20 * versatility_pctl
            )

            result = result.merge(
                depth_df[["team_id", "roster_depth_score", "concentration",
                          "replacement_gap", "bench_quality", "pitcher_bench",
                          "roster_breadth", "positional_fit", "versatility"]],
                on="team_id", how="left",
            )
            logger.info(
                "Roster depth computed for %d teams (positional + traditional)",
                len(depth_df),
            )
    except Exception as e:
        logger.warning("Could not compute roster depth metrics: %s", e)

    if "roster_depth_score" not in result.columns:
        result["roster_depth_score"] = 0.50
    result["roster_depth_score"] = result["roster_depth_score"].fillna(0.50)

    # Composite: health (50%) + age (20%) + roster depth (30%)
    # Walk-forward validation (2022-2025) showed depth adds signal
    # beyond simple health + age, especially for mid-tier teams.
    result["health_depth_score"] = (
        0.50 * result["player_health"]
        + 0.20 * result["age_pctl"]
        + 0.30 * result["roster_depth_score"]
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
        get_pitcher_season_fip,
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

    # Filter player_teams by current roster status (active + IL only, exclude
    # NRI/suspended/unavailable). This ensures team aggregations reflect the
    # projected Opening Day roster, not everyone who appeared in 2025 boxscores.
    #
    # IL availability weighting: players on IL contribute less to team strength.
    # IL-60 players (Strider, etc.) are essentially unavailable for months and
    # should not inflate a team's projected pitching/offense scores.
    _IL_AVAILABILITY = {
        "active": 1.0,
        "il_7": 0.90,    # back in ~1 week
        "il_10": 0.75,   # back in ~2 weeks
        "il_15": 0.50,   # back in ~3 weeks
        "il_60": 0.10,   # out for months, minimal contribution
    }
    roster = _safe_load("roster.parquet")
    if roster is not None and "roster_status" in roster.columns:
        active_mask = roster["roster_status"].isin(_IL_AVAILABILITY.keys())
        roster_cols = ["player_id", "roster_status"]
        if "team_abbr" in roster.columns:
            roster_cols.append("team_abbr")
        roster_filtered = roster.loc[active_mask, roster_cols].copy()
        roster_filtered["availability"] = roster_filtered["roster_status"].map(_IL_AVAILABILITY)

        active_ids = set(roster_filtered["player_id"])
        before_n = len(player_teams)
        player_teams = player_teams[player_teams["player_id"].isin(active_ids)].copy()

        # Override team assignment with roster's current team.
        # player_teams reflects 2025 boxscores; roster reflects 2026 roster.
        # Offseason acquisitions (e.g. Valdez HOU→DET) need the roster team.
        if "team_abbr" in roster_filtered.columns:
            roster_teams = roster_filtered[["player_id", "team_abbr"]].drop_duplicates("player_id")
            player_teams = player_teams.drop(columns=["team_abbr"], errors="ignore")
            player_teams = player_teams.merge(
                roster_teams, on="player_id", how="left",
            )
            n_reassigned = (player_teams["team_abbr"].notna()).sum()
            logger.info("Team assignments updated from roster for %d players", n_reassigned)

        # Merge availability weight into player_teams
        player_teams = player_teams.merge(
            roster_filtered[["player_id", "availability", "roster_status"]],
            on="player_id", how="left",
        )
        player_teams["availability"] = player_teams["availability"].fillna(1.0)

        n_il = (player_teams["availability"] < 1.0).sum()
        logger.info(
            "Roster filter: %d -> %d players (%d on IL with reduced availability)",
            before_n, len(player_teams), n_il,
        )
    else:
        logger.warning("roster.parquet not found — using unfiltered player_teams")
        player_teams["availability"] = 1.0

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

    # Merge GB%/FB% from pitcher_projections into pitcher_rankings for synergy
    pitcher_proj = _safe_load("pitcher_projections.parquet")
    if pitcher_rankings is not None and pitcher_proj is not None:
        pid_col = "pitcher_id" if "pitcher_id" in pitcher_proj.columns else "player_id"
        gb_cols = [pid_col] + [c for c in ["gb_pct", "fb_pct"] if c in pitcher_proj.columns]
        if len(gb_cols) > 1:
            pitcher_rankings = pitcher_rankings.merge(
                pitcher_proj[gb_cols], on=pid_col, how="left",
            )
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
    try:
        pitcher_fip = get_pitcher_season_fip(seasons=[season], min_ip=10)
    except Exception as e:
        logger.warning("Pitcher FIP query failed: %s", e)
        pitcher_fip = None
    pitching = _build_pitching_profile(
        player_teams, pitcher_rankings, pitcher_roles, team_pitching,
        park_factors=park_factors,
        pitcher_counting=pitcher_counting,
        pitcher_fip=pitcher_fip,
    )

    logger.info("Building defense profile...")
    defense = _build_defense_profile(player_teams, season, pitcher_rankings=pitcher_rankings)

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
        # Pad missing IP to a full 162-game season with replacement-level pitching
        # so that teams with fewer projected arms aren't artificially deflated.
        if pitcher_counting is not None and "total_runs_mean" in pitcher_counting.columns:
            SEASON_IP = 162.0 * 9  # 1458 innings per team-season
            REPLACEMENT_RA_PER_9 = 5.50  # replacement-level total runs/9 IP

            p_id = "pitcher_id" if "pitcher_id" in pitcher_counting.columns else "player_id"
            pc = pitcher_counting.merge(
                player_teams[[pt_pid, "team_id"]].rename(columns={pt_pid: p_id}),
                on=p_id, how="inner",
            )
            ip_col = "projected_ip_mean" if "projected_ip_mean" in pc.columns else None
            team_ra = pc.groupby("team_id").agg(
                runs_allowed=("total_runs_mean", "sum"),
                ip=(ip_col, "sum") if ip_col else ("total_outs_mean", lambda x: x.sum() / 3),
            ).reset_index()

            # Pad missing innings with replacement-level pitching
            missing_ip = (SEASON_IP - team_ra["ip"]).clip(lower=0)
            team_ra["runs_allowed"] += missing_ip / 9 * REPLACEMENT_RA_PER_9
            team_ra["ip"] = team_ra["ip"].clip(lower=SEASON_IP)

            team_ra["proj_ra_per_game"] = team_ra["runs_allowed"] / 162.0

            # Force league-mean RA to equal league-mean RS so total W sums to 2430
            if "proj_rs_per_game" in offense.columns:
                lg_rs = offense["proj_rs_per_game"].mean()
                lg_ra = team_ra["proj_ra_per_game"].mean()
                if lg_ra > 0 and lg_rs > 0:
                    team_ra["proj_ra_per_game"] *= lg_rs / lg_ra
                    logger.info("Normalized RA/game: lg_rs=%.3f, raw_ra=%.3f, scale=%.3f",
                                lg_rs, lg_ra, lg_rs / lg_ra)

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
    for extra_col in ["offense_ceiling",
                       "offense_style", "lineup_depth", "lineup_floor", "rpg", "hr_per_game", "sb_per_game",
                       "proj_R", "proj_HR", "proj_SB", "avg_hitter_score",
                       "baseruns_rs", "proj_rs_per_game",
                       # Lineup scouting grades
                       "lineup_diamond", "lineup_grade_hit", "lineup_grade_power",
                       "lineup_grade_speed", "lineup_grade_fielding", "lineup_grade_discipline"]:
        if extra_col in offense.columns:
            result[extra_col] = offense[extra_col]

    for df, key_cols in [
        (pitching, ["pitching_score", "pitching_ceiling",
                     "rotation_score", "bullpen_score",
                     "rotation_strength", "rotation_strength_ceiling",
                     "rotation_depth_bonus",
                     "bullpen_depth",
                     "sp_fip_pctl", "bp_fip_pctl",
                     "pitching_style", "ra_per_game", "k_per_game",
                     "proj_ra_per_game",
                     # Rotation + bullpen scouting grades
                     "rotation_diamond", "rotation_grade_stuff", "rotation_grade_command",
                     "rotation_grade_durability", "rotation_sp_count",
                     "bullpen_diamond", "bullpen_grade_stuff", "bullpen_grade_command",
                     "bullpen_rp_count"]),
        (defense, ["defense_score", "team_oaa", "catcher_framing_runs",
                   "infield_oaa", "outfield_oaa", "staff_gb_pct", "staff_fb_pct",
                   "pitcher_defense_synergy", "team_fielding_grade"]),
        (org, ["org_score", "pct_homegrown", "farm_avg_score", "n_ranked_prospects", "top_100_count"]),
        (health, ["health_depth_score", "roster_depth_score",
                  "concentration", "replacement_gap", "bench_quality",
                  "pitcher_bench", "roster_breadth",
                  "positional_fit", "versatility",
                  "total_il_days", "avg_age", "age_trajectory",
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

    # Rescale lineup/rotation/bullpen diamond ratings to use the full 1-10 range.
    # Raw averages compress to ~4-6 since individual DRs cluster around 5.
    # Min-max rescale across teams so the best lineup = 10, worst = 1.
    for col in ("lineup_diamond", "rotation_diamond", "bullpen_diamond"):
        if col in result.columns:
            vals = result[col].dropna()
            if len(vals) > 1:
                lo, hi = vals.min(), vals.max()
                if hi > lo:
                    result[col] = 1.0 + 9.0 * (result[col] - lo) / (hi - lo)

    logger.info("Team profiles built: %d teams", len(result))
    return result
