"""
TDD Positional Player Rankings — 2026 value composite.

Ranks MLB players within each position by blending observed production
with Bayesian projections, fielding value (OAA), catcher framing,
and projected playing time.

Positions: C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP, RP.

Blend: ~40% Bayesian projection / ~60% recent observed for 2026 value,
since the season starts in ~10 days.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"
DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")

# ---------------------------------------------------------------------------
# Hitter sub-component weights (sum to 1.0)
# ---------------------------------------------------------------------------
_HITTER_WEIGHTS = {
    "offense": 0.55,
    "fielding": 0.20,
    "playing_time": 0.15,
    "trajectory": 0.10,
}

# Offense sub-weights: how to blend projection vs observed
_PROJ_WEIGHT = 0.40
_OBS_WEIGHT = 0.60

# ---------------------------------------------------------------------------
# Pitcher sub-component weights (sum to 1.0)
# ---------------------------------------------------------------------------
_PITCHER_WEIGHTS = {
    "stuff": 0.50,
    "command": 0.20,
    "workload": 0.15,
    "trajectory": 0.15,
}

# All hitter positions
HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
PITCHER_ROLES = ["SP", "RP"]


def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0–1, higher = better)."""
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    """Inverse percentile rank (lower raw value = higher score)."""
    return 1.0 - series.rank(pct=True, method="average")


# ===================================================================
# Position assignment
# ===================================================================

def _assign_hitter_positions(season: int = 2025, min_starts: int = 20) -> pd.DataFrame:
    """Assign primary position to each hitter based on lineup starts.

    Uses MODE of started positions in the most recent season. Falls back
    to ``dim_player.primary_position_code`` for players with few starts.

    Parameters
    ----------
    season : int
        Season to determine positions from.
    min_starts : int
        Minimum starts at a position to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position.
    """
    from src.data.db import read_sql

    # Primary method: most-started position from lineup data
    lineup_pos = read_sql(f"""
        SELECT player_id,
               MODE() WITHIN GROUP (ORDER BY position) as position,
               count(*) as starts
        FROM production.fact_lineup
        WHERE season = {season} AND is_starter = true AND position != 'P'
        GROUP BY player_id
        HAVING count(*) >= {min_starts}
    """, {})

    # Fallback: dim_player for players not in lineup data
    fallback = read_sql("""
        SELECT player_id, primary_position as position
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
          AND active = true
    """, {})

    if not lineup_pos.empty:
        assigned = lineup_pos[["player_id", "position"]].copy()
        # Add fallback players not in lineup
        missing = fallback[~fallback["player_id"].isin(assigned["player_id"])]
        assigned = pd.concat([assigned, missing[["player_id", "position"]]], ignore_index=True)
    else:
        assigned = fallback[["player_id", "position"]].copy()

    return assigned


def _assign_pitcher_roles() -> pd.DataFrame:
    """Assign SP/RP role from pitcher projections.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, role ('SP' or 'RP').
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    proj["role"] = np.where(proj["is_starter"] == 1, "SP", "RP")
    return proj[["pitcher_id", "role"]].copy()


# ===================================================================
# Hitter ranking
# ===================================================================

def _build_hitter_offense_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Blend projected and observed offensive value into a single score.

    Parameters
    ----------
    proj : pd.DataFrame
        Hitter projections (from dashboard parquet).
    observed : pd.DataFrame
        Observed season stats from ``fact_batting_advanced``.

    Returns
    -------
    pd.DataFrame
        player_id + offense_score columns.
    """
    # Merge projection + observed on batter_id
    merged = proj[["batter_id", "projected_k_rate", "projected_bb_rate",
                    "projected_hr_per_fb", "composite_score"]].merge(
        observed[["batter_id", "pa", "k_pct", "bb_pct", "woba", "wrc_plus",
                   "barrel_pct", "hard_hit_pct", "xwoba"]],
        on="batter_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["batter_id", "offense_score"])

    # --- Projected component (Bayesian rates) ---
    proj_k_score = _inv_pctl(merged["projected_k_rate"])
    proj_bb_score = _pctl(merged["projected_bb_rate"])
    proj_hr_score = _pctl(merged["projected_hr_per_fb"])
    projected = 0.35 * proj_k_score + 0.30 * proj_bb_score + 0.35 * proj_hr_score

    # --- Observed component (Statcast + traditional) ---
    obs_woba = _pctl(merged["woba"].fillna(merged["woba"].median()))
    obs_xwoba = _pctl(merged["xwoba"].fillna(merged["xwoba"].median()))
    obs_barrel = _pctl(merged["barrel_pct"].fillna(0))
    obs_hh = _pctl(merged["hard_hit_pct"].fillna(0))
    observed_score = 0.35 * obs_woba + 0.30 * obs_xwoba + 0.20 * obs_barrel + 0.15 * obs_hh

    # Blend
    merged["offense_score"] = _PROJ_WEIGHT * projected + _OBS_WEIGHT * observed_score

    return merged[["batter_id", "offense_score"]]


def _build_hitter_fielding_score(season: int = 2025) -> pd.DataFrame:
    """Build fielding score from OAA data.

    Parameters
    ----------
    season : int
        Season for OAA data.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, fielding_score.
    """
    from src.data.db import read_sql

    oaa = read_sql(f"""
        SELECT player_id, outs_above_average, fielding_runs_prevented
        FROM production.fact_fielding_oaa
        WHERE season = {season}
    """, {})

    if oaa.empty:
        return pd.DataFrame(columns=["player_id", "fielding_score"])

    # Percentile rank OAA (higher = better)
    oaa["fielding_score"] = _pctl(oaa["outs_above_average"])
    return oaa[["player_id", "fielding_score"]]


def _build_catcher_framing_score(season: int = 2025) -> pd.DataFrame:
    """Build catcher framing score from framing data.

    Parameters
    ----------
    season : int
        Season for framing data.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, framing_score.
    """
    from src.data.db import read_sql

    framing = read_sql(f"""
        SELECT player_id, runs_extra_strikes, strike_rate_diff,
               shadow_zone_strike_rate
        FROM production.fact_catcher_framing
        WHERE season = {season}
    """, {})

    if framing.empty:
        return pd.DataFrame(columns=["player_id", "framing_score"])

    # Percentile rank runs_extra_strikes (higher = better)
    framing["framing_score"] = _pctl(framing["runs_extra_strikes"])
    return framing[["player_id", "framing_score"]]


def _build_hitter_playing_time_score() -> pd.DataFrame:
    """Score projected playing time from counting projections + health.

    Blends 70% projected PA percentile with 30% health score percentile
    when health scores are available.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, pt_score, health_score, health_label.
    """
    counting = pd.read_parquet(DASHBOARD_DIR / "hitter_counting.parquet")
    pa_pctl = _pctl(counting["projected_pa_mean"])

    # Health scores: prefer columns already on counting parquet,
    # fall back to standalone health_scores.parquet
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        health_path = DASHBOARD_DIR / "health_scores.parquet"
        if health_path.exists():
            health_df = pd.read_parquet(health_path)
            counting = counting.merge(
                health_df[["player_id", "health_score", "health_label"]].rename(
                    columns={"player_id": "batter_id"}
                ),
                on="batter_id",
                how="left",
            )
            has_health = True

    if has_health:
        counting["health_score"] = counting["health_score"].fillna(0.85)
        counting["health_label"] = counting["health_label"].fillna("Unknown")
        health_pctl = _pctl(counting["health_score"])
        counting["pt_score"] = 0.70 * pa_pctl + 0.30 * health_pctl
    else:
        counting["pt_score"] = pa_pctl
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["batter_id", "pt_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_hitter_trajectory_score() -> pd.DataFrame:
    """Score trajectory using posterior certainty and projected quality.

    Rewards players whose Bayesian posteriors are tight (proven track
    record) and whose projected rates are strong. Age still matters
    but is secondary — a 33-year-old with 8 years of elite data should
    score well, not get destroyed by age alone.

    Components
    ----------
    - **Projection certainty** (50%): tighter posterior SD = more proven.
      Directly leverages what the Bayesian model already knows about
      year-over-year consistency.
    - **Age upside** (20%): younger players get a modest bump.
    - **Projected rate quality** (30%): how good are the projected rates
      themselves (low K%, high BB%).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, trajectory_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Posterior certainty: tighter SD = more proven/consistent
    # Average across K% and BB% SDs, then invert (lower SD = higher score)
    k_certainty = _inv_pctl(proj["projected_k_rate_sd"].fillna(proj["projected_k_rate_sd"].median()))
    bb_certainty = _inv_pctl(proj["projected_bb_rate_sd"].fillna(proj["projected_bb_rate_sd"].median()))
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # Projected rate quality: how good are the projected rates?
    proj_k_quality = _inv_pctl(proj["projected_k_rate"])
    proj_bb_quality = _pctl(proj["projected_bb_rate"])
    rate_quality = 0.50 * proj_k_quality + 0.50 * proj_bb_quality

    # Age upside: younger = more room to grow (but modest weight)
    age_score = _inv_pctl(proj["age"].fillna(30))

    proj["trajectory_score"] = (
        0.50 * certainty_score
        + 0.30 * rate_quality
        + 0.20 * age_score
    )
    return proj[["batter_id", "trajectory_score"]]


def rank_hitters(
    season: int = 2025,
    projection_season: int = 2026,
    min_pa: int = 100,
) -> pd.DataFrame:
    """Rank all hitters by position for 2026 value.

    Parameters
    ----------
    season : int
        Most recent completed season (for observed stats).
    projection_season : int
        Target projection season.
    min_pa : int
        Minimum PA in observed season to qualify.

    Returns
    -------
    pd.DataFrame
        Positional rankings with composite score, sub-scores,
        and key stats. Sorted by position then rank.
    """
    from src.data.db import read_sql

    # Load projections
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Load observed stats
    observed = read_sql(f"""
        SELECT batter_id, pa, k_pct, bb_pct, woba, wrc_plus,
               xwoba, barrel_pct, hard_hit_pct
        FROM production.fact_batting_advanced
        WHERE season = {season} AND pa >= {min_pa}
    """, {})

    # Position assignments
    positions = _assign_hitter_positions(season=season)

    # Build sub-scores
    offense = _build_hitter_offense_score(proj, observed)
    fielding = _build_hitter_fielding_score(season=season)
    framing = _build_catcher_framing_score(season=season)
    playing_time = _build_hitter_playing_time_score()
    trajectory = _build_hitter_trajectory_score()

    # Start from projections as base (has batter_id, name, age, etc.)
    base = proj[["batter_id", "batter_name", "age", "batter_stand",
                 "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
                 "projected_k_rate_sd", "projected_bb_rate_sd",
                 "composite_score"]].copy()

    # Merge observed stats for display
    base = base.merge(
        observed[["batter_id", "pa", "woba", "wrc_plus", "xwoba",
                   "barrel_pct", "hard_hit_pct"]],
        on="batter_id", how="inner",
    )

    # Merge position
    base = base.merge(
        positions.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="inner",
    )

    # Merge sub-scores
    base = base.merge(offense, on="batter_id", how="left")
    base = base.merge(
        fielding.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base = base.merge(
        framing.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base = base.merge(playing_time, on="batter_id", how="left")
    base = base.merge(trajectory, on="batter_id", how="left")

    # Fill missing fielding/framing with neutral
    base["fielding_score"] = base["fielding_score"].fillna(0.50)
    base["framing_score"] = base["framing_score"].fillna(0.50)
    base["offense_score"] = base["offense_score"].fillna(0.50)
    base["pt_score"] = base["pt_score"].fillna(0.50)
    base["trajectory_score"] = base["trajectory_score"].fillna(0.50)
    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Composite score ---
    # For catchers: blend framing into fielding component
    is_catcher = base["position"] == "C"
    base["fielding_combined"] = base["fielding_score"]
    base.loc[is_catcher, "fielding_combined"] = (
        0.50 * base.loc[is_catcher, "fielding_score"]
        + 0.50 * base.loc[is_catcher, "framing_score"]
    )

    # For DH: only half the fielding weight redirects to offense.
    # The other half is lost — a positional penalty reflecting zero
    # defensive value. DH max = 0.90 vs 1.00 for position players.
    is_dh = base["position"] == "DH"
    dh_offense_wt = _HITTER_WEIGHTS["offense"] + _HITTER_WEIGHTS["fielding"] / 2
    base["tdd_value_score"] = (
        _HITTER_WEIGHTS["offense"] * base["offense_score"]
        + _HITTER_WEIGHTS["fielding"] * base["fielding_combined"]
        + _HITTER_WEIGHTS["playing_time"] * base["pt_score"]
        + _HITTER_WEIGHTS["trajectory"] * base["trajectory_score"]
    )
    # Recalc for DH (half fielding weight lost as positional penalty)
    base.loc[is_dh, "tdd_value_score"] = (
        dh_offense_wt * base.loc[is_dh, "offense_score"]
        + _HITTER_WEIGHTS["playing_time"] * base.loc[is_dh, "pt_score"]
        + _HITTER_WEIGHTS["trajectory"] * base.loc[is_dh, "trajectory_score"]
    )

    # --- Two-way player bonus ---
    # Players who also pitch (e.g. Ohtani) get credit for pitching value.
    # Check both Bayesian pitcher projections and raw pitching stats,
    # since a two-way player may not appear in projections (e.g. missed
    # a full season pitching due to injury).
    base["two_way_bonus"] = 0.0
    base["is_two_way"] = False
    try:
        from src.data.db import read_sql as _read_sql

        pitcher_proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

        # Also check raw pitching advanced stats for recent seasons
        recent_pitching = _read_sql(f"""
            SELECT pitcher_id, k_pct, bb_pct, swstr_pct, batters_faced
            FROM production.fact_pitching_advanced
            WHERE season >= {season - 2} AND batters_faced >= 100
        """, {})
        # Aggregate across recent seasons (PA-weighted)
        if not recent_pitching.empty:
            rp_agg = (
                recent_pitching.groupby("pitcher_id")
                .apply(
                    lambda g: pd.Series({
                        "k_pct": np.average(g["k_pct"], weights=g["batters_faced"]),
                        "bb_pct": np.average(g["bb_pct"], weights=g["batters_faced"]),
                        "total_bf": g["batters_faced"].sum(),
                    }),
                    include_groups=False,
                )
                .reset_index()
            )
        else:
            rp_agg = pd.DataFrame(columns=["pitcher_id", "k_pct", "bb_pct", "total_bf"])

        # Find two-way players: hitters who appear in pitcher data
        hitter_ids = set(base["batter_id"])
        proj_pitcher_ids = set(pitcher_proj["pitcher_id"]) & hitter_ids
        raw_pitcher_ids = set(rp_agg["pitcher_id"]) & hitter_ids
        two_way_ids = proj_pitcher_ids | raw_pitcher_ids

        if two_way_ids:
            # Use projections if available, raw stats as fallback
            for pid in two_way_ids:
                if pid in proj_pitcher_ids:
                    pp = pitcher_proj[pitcher_proj["pitcher_id"] == pid].iloc[0]
                    k_val = pp["projected_k_rate"]
                    bb_val = pp["projected_bb_rate"]
                    ref_k = pitcher_proj["projected_k_rate"]
                    ref_bb = pitcher_proj["projected_bb_rate"]
                else:
                    pp_raw = rp_agg[rp_agg["pitcher_id"] == pid].iloc[0]
                    k_val = pp_raw["k_pct"]
                    bb_val = pp_raw["bb_pct"]
                    ref_k = rp_agg["k_pct"]
                    ref_bb = rp_agg["bb_pct"]

                k_pctl = (ref_k < k_val).mean()
                bb_pctl = (ref_bb > bb_val).mean()
                pitcher_value = 0.60 * k_pctl + 0.40 * bb_pctl
                # Scale bonus: elite pitcher value (0.8+) adds up to ~0.15
                bonus = pitcher_value * 0.18
                mask = base["batter_id"] == pid
                base.loc[mask, "two_way_bonus"] = bonus
                base.loc[mask, "is_two_way"] = True
                pname = base.loc[mask, "batter_name"].iloc[0]
                logger.info(
                    "Two-way bonus: %s — pitcher value=%.3f, bonus=+%.3f",
                    pname, pitcher_value, bonus,
                )
    except Exception:
        logger.exception("Could not compute two-way player bonus")

    base["tdd_value_score"] = base["tdd_value_score"] + base["two_way_bonus"]

    # --- Rank within each position ---
    base = base.sort_values("tdd_value_score", ascending=False)
    base["pos_rank"] = base.groupby("position").cumcount() + 1

    # Overall hitter rank
    base["overall_rank"] = base["tdd_value_score"].rank(ascending=False, method="min").astype(int)

    # Select output columns
    output_cols = [
        "pos_rank", "overall_rank", "batter_id", "batter_name", "position",
        "age", "batter_stand", "is_two_way",
        # Composite
        "tdd_value_score",
        # Sub-scores
        "offense_score", "fielding_combined", "framing_score",
        "pt_score", "trajectory_score", "two_way_bonus",
        # Health
        "health_score", "health_label",
        # Observed
        "pa", "woba", "wrc_plus", "xwoba", "barrel_pct", "hard_hit_pct",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
        "projected_k_rate_sd", "projected_bb_rate_sd",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["position", "pos_rank"])

    for pos in HITTER_POSITIONS:
        pos_df = result[result["position"] == pos]
        if not pos_df.empty:
            logger.info(
                "%s: %d ranked — #1 %s (%.3f)",
                pos, len(pos_df),
                pos_df.iloc[0]["batter_name"],
                pos_df.iloc[0]["tdd_value_score"],
            )

    return result


# ===================================================================
# Pitcher ranking
# ===================================================================

def _build_pitcher_stuff_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Blend projected and observed stuff/quality metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, stuff_score.
    """
    merged = proj[["pitcher_id", "projected_k_rate", "projected_hr_per_bf",
                    "whiff_rate"]].merge(
        observed[["pitcher_id", "swstr_pct", "csw_pct", "xwoba_against",
                   "barrel_pct_against", "hard_hit_pct_against"]],
        on="pitcher_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "stuff_score"])

    # Projected: K rate (higher = better), HR/BF (lower = better)
    proj_k = _pctl(merged["projected_k_rate"])
    proj_hr = _inv_pctl(merged["projected_hr_per_bf"])
    projected = 0.65 * proj_k + 0.35 * proj_hr

    # Observed: SwStr%, CSW%, xwOBA-against (lower = better),
    # barrel%-against (lower = better)
    obs_swstr = _pctl(merged["swstr_pct"].fillna(0))
    obs_csw = _pctl(merged["csw_pct"].fillna(0))
    obs_xwoba = _inv_pctl(merged["xwoba_against"].fillna(merged["xwoba_against"].median()))
    obs_barrel = _inv_pctl(merged["barrel_pct_against"].fillna(0))
    observed_score = 0.30 * obs_swstr + 0.30 * obs_csw + 0.25 * obs_xwoba + 0.15 * obs_barrel

    merged["stuff_score"] = _PROJ_WEIGHT * projected + _OBS_WEIGHT * observed_score
    return merged[["pitcher_id", "stuff_score"]]


def _build_pitcher_command_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Score command/control from projected BB% and observed metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, command_score.
    """
    merged = proj[["pitcher_id", "projected_bb_rate"]].merge(
        observed[["pitcher_id", "bb_pct", "zone_pct", "chase_pct"]],
        on="pitcher_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "command_score"])

    # Lower BB% = better command
    proj_bb = _inv_pctl(merged["projected_bb_rate"])
    obs_bb = _inv_pctl(merged["bb_pct"].fillna(merged["bb_pct"].median()))
    obs_zone = _pctl(merged["zone_pct"].fillna(0))
    obs_chase = _pctl(merged["chase_pct"].fillna(0))

    projected_cmd = proj_bb
    observed_cmd = 0.40 * obs_bb + 0.30 * obs_zone + 0.30 * obs_chase

    merged["command_score"] = _PROJ_WEIGHT * projected_cmd + _OBS_WEIGHT * observed_cmd
    return merged[["pitcher_id", "command_score"]]


def _build_pitcher_workload_score() -> pd.DataFrame:
    """Score projected workload from counting projections + health.

    Blends 70% projected BF percentile with 30% health score percentile
    when health scores are available.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, workload_score, health_score, health_label.
    """
    counting = pd.read_parquet(DASHBOARD_DIR / "pitcher_counting.parquet")
    bf_pctl = _pctl(counting["projected_bf_mean"])

    # Health scores: prefer columns already on counting parquet,
    # fall back to standalone health_scores.parquet
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        health_path = DASHBOARD_DIR / "health_scores.parquet"
        if health_path.exists():
            health_df = pd.read_parquet(health_path)
            counting = counting.merge(
                health_df[["player_id", "health_score", "health_label"]].rename(
                    columns={"player_id": "pitcher_id"}
                ),
                on="pitcher_id",
                how="left",
            )
            has_health = True

    if has_health:
        counting["health_score"] = counting["health_score"].fillna(0.85)
        counting["health_label"] = counting["health_label"].fillna("Unknown")
        health_pctl = _pctl(counting["health_score"])
        counting["workload_score"] = 0.70 * bf_pctl + 0.30 * health_pctl
    else:
        counting["workload_score"] = bf_pctl
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["pitcher_id", "workload_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_pitcher_trajectory_score() -> pd.DataFrame:
    """Score trajectory using posterior certainty and projected quality.

    Same philosophy as hitter trajectory: reward proven track records
    (tight posteriors) and strong projected rates, with a modest
    age upside factor.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, trajectory_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

    # Posterior certainty: tighter SD = more proven
    k_certainty = _inv_pctl(proj["projected_k_rate_sd"].fillna(proj["projected_k_rate_sd"].median()))
    bb_certainty = _inv_pctl(proj["projected_bb_rate_sd"].fillna(proj["projected_bb_rate_sd"].median()))
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # Projected rate quality
    proj_k_quality = _pctl(proj["projected_k_rate"])  # higher K% = better for pitchers
    proj_bb_quality = _inv_pctl(proj["projected_bb_rate"])  # lower BB% = better
    rate_quality = 0.60 * proj_k_quality + 0.40 * proj_bb_quality

    # Age upside
    age_score = _inv_pctl(proj["age"].fillna(28))

    proj["trajectory_score"] = (
        0.50 * certainty_score
        + 0.30 * rate_quality
        + 0.20 * age_score
    )
    return proj[["pitcher_id", "trajectory_score"]]


def rank_pitchers(
    season: int = 2025,
    projection_season: int = 2026,
    min_bf: int = 50,
) -> pd.DataFrame:
    """Rank all pitchers by role (SP/RP) for 2026 value.

    Parameters
    ----------
    season : int
        Most recent completed season.
    projection_season : int
        Target projection season.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        Pitchers ranked by role with composite score, sub-scores.
    """
    from src.data.db import read_sql

    # Load projections
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

    # Load observed stats
    observed = read_sql(f"""
        SELECT pitcher_id, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, contact_pct, xwoba_against,
               barrel_pct_against, hard_hit_pct_against,
               batters_faced, woba_against
        FROM production.fact_pitching_advanced
        WHERE season = {season} AND batters_faced >= {min_bf}
    """, {})

    # Role assignment
    roles = _assign_pitcher_roles()

    # Build sub-scores
    stuff = _build_pitcher_stuff_score(proj, observed)
    command = _build_pitcher_command_score(proj, observed)
    workload = _build_pitcher_workload_score()
    trajectory = _build_pitcher_trajectory_score()

    # Base from projections
    base = proj[["pitcher_id", "pitcher_name", "age", "pitch_hand",
                 "is_starter", "projected_k_rate", "projected_bb_rate",
                 "projected_hr_per_bf", "projected_k_rate_sd",
                 "projected_bb_rate_sd", "composite_score"]].copy()

    # Merge observed for display
    base = base.merge(
        observed[["pitcher_id", "batters_faced", "k_pct", "bb_pct",
                   "swstr_pct", "csw_pct", "xwoba_against", "woba_against"]],
        on="pitcher_id", how="inner",
    )

    # Merge role
    base = base.merge(roles, on="pitcher_id", how="left")
    missing_role = base["role"].isna()
    base.loc[missing_role, "role"] = np.where(
        base.loc[missing_role, "is_starter"] == 1, "SP", "RP"
    )

    # Merge sub-scores
    base = base.merge(stuff, on="pitcher_id", how="left")
    base = base.merge(command, on="pitcher_id", how="left")
    base = base.merge(workload, on="pitcher_id", how="left")
    base = base.merge(trajectory, on="pitcher_id", how="left")

    # Fill missing with neutral
    for col in ["stuff_score", "command_score", "workload_score", "trajectory_score"]:
        base[col] = base[col].fillna(0.50)
    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Composite ---
    base["tdd_value_score"] = (
        _PITCHER_WEIGHTS["stuff"] * base["stuff_score"]
        + _PITCHER_WEIGHTS["command"] * base["command_score"]
        + _PITCHER_WEIGHTS["workload"] * base["workload_score"]
        + _PITCHER_WEIGHTS["trajectory"] * base["trajectory_score"]
    )

    # Rank within role
    base = base.sort_values("tdd_value_score", ascending=False)
    base["role_rank"] = base.groupby("role").cumcount() + 1

    # Overall pitcher rank
    base["overall_rank"] = base["tdd_value_score"].rank(ascending=False, method="min").astype(int)

    # Select output
    output_cols = [
        "role_rank", "overall_rank", "pitcher_id", "pitcher_name", "role",
        "age", "pitch_hand",
        # Composite
        "tdd_value_score",
        # Sub-scores
        "stuff_score", "command_score", "workload_score", "trajectory_score",
        # Health
        "health_score", "health_label",
        # Observed
        "batters_faced", "k_pct", "bb_pct", "swstr_pct", "csw_pct",
        "xwoba_against", "woba_against",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_bf",
        "projected_k_rate_sd", "projected_bb_rate_sd",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["role", "role_rank"])

    for role in PITCHER_ROLES:
        role_df = result[result["role"] == role]
        if not role_df.empty:
            logger.info(
                "%s: %d ranked — #1 %s (%.3f)",
                role, len(role_df),
                role_df.iloc[0]["pitcher_name"],
                role_df.iloc[0]["tdd_value_score"],
            )

    return result


# ===================================================================
# Combined entry point
# ===================================================================

def rank_all(
    season: int = 2025,
    projection_season: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Run all positional rankings.

    Parameters
    ----------
    season : int
        Most recent completed season for observed data.
    projection_season : int
        Target projection season.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'hitters', 'pitchers'. Each sorted by position/role then rank.
    """
    logger.info("Building 2026 positional rankings...")
    hitters = rank_hitters(season=season, projection_season=projection_season)
    pitchers = rank_pitchers(season=season, projection_season=projection_season)
    logger.info(
        "Rankings complete: %d hitters, %d pitchers",
        len(hitters), len(pitchers),
    )
    return {"hitters": hitters, "pitchers": pitchers}
