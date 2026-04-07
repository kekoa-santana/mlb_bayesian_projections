"""
TDD Positional Player Rankings -- 2026 value composite.

Ranks MLB players within each position by blending observed production
with Bayesian projections, fielding value (OAA), catcher framing,
and projected playing time.

Positions: C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP, RP.

Two-score architecture:
  - ``current_value_score``: production-dominant, feeds team rankings and
    projected wins.  Scouting grades are a minor input that shrinks as
    sample size grows (configurable via config/model.yaml ``rankings:``).
  - ``talent_upside_score``: scouting-dominant, feeds dynasty rankings and
    prospect evaluation.  Uses the same regressed diamond rating blend that
    was previously the sole ``tdd_value_score``.

``tdd_value_score`` is retained as a backward-compatible alias for
``current_value_score`` so that team profiles / team rankings / power
rankings continue to work without modification.

Sub-modules (structural refactor, zero behavioral changes):
  - ``_ranking_utils``: shared utility functions and constants
  - ``player_positions``: position assignment logic
  - ``_hitter_scores``: hitter sub-score builders
  - ``_pitcher_scores``: pitcher sub-score builders
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Re-export shared constants and utilities so external callers that import
# from player_rankings continue to work unchanged.
# ---------------------------------------------------------------------------
from src.models._ranking_utils import (  # noqa: F401 — re-exports
    CACHE_DIR,
    DASHBOARD_DIR,
    HITTER_POSITIONS,
    PITCHER_ROLES,
    _HITTER_WEIGHTS,
    _OBS_WEIGHT_BASE,
    _POS_ADJUSTMENT,
    _POS_TIER,
    _POSITIONAL_OFFENSE_ADJ,
    _PROJ_WEIGHT_BASE,
    _RANK_CFG,
    _RP_WEIGHTS,
    _SP_WEIGHTS,
    _dynamic_blend_weights,
    _exposure_conditioned_scouting_weight,
    _hitter_age_factor,
    _inv_pctl,
    _inv_zscore_pctl,
    _load_glicko_scores,
    _load_ranking_config,
    _pctl,
    _pitcher_age_factor,
    _stat_family_trust,
    _zscore_pctl,
)

# Re-export position functions
from src.models.player_positions import (  # noqa: F401 — re-exports
    _assign_hitter_positions,
    _assign_pitcher_roles,
    _get_career_weighted_positions,
    _verify_positions_mlb_api,
    get_hitter_position_eligibility,
)

# Re-export hitter score builders
from src.models._hitter_scores import (  # noqa: F401 — re-exports
    _build_catcher_framing_score,
    _build_confirmed_breakout_score,
    _build_hitter_baserunning_score,
    _build_hitter_fielding_score,
    _build_hitter_offense_score,
    _build_hitter_platoon_modifier,
    _build_hitter_playing_time_score,
    _build_hitter_trajectory_score,
    _build_roster_construction_score,
    _build_versatility_score,
)

# Re-export pitcher score builders
from src.models._pitcher_scores import (  # noqa: F401 — re-exports
    _build_pitcher_command_score,
    _build_pitcher_innings_durability,
    _build_pitcher_stuff_score,
    _build_pitcher_trajectory_score,
    _build_pitcher_velo_trend,
    _build_pitcher_workload_score,
    _compute_stuff_plus,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Hitter ranking orchestrator
# ===================================================================

def rank_hitters(
    season: int = 2025,
    projection_season: int = 2026,
    min_pa: int = 100,
    health_df: pd.DataFrame | None = None,
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
    health_df : pd.DataFrame or None
        Pre-loaded health scores (columns: player_id, health_score,
        health_label).  Passed through to ``_build_hitter_playing_time_score``
        to avoid a disk read when the pipeline already has it in memory.

    Returns
    -------
    pd.DataFrame
        Positional rankings with composite score, sub-scores,
        and key stats. Sorted by position then rank.
    """
    from src.data.db import read_sql

    # Load projections (defensive: parquet / upstream must be one row per batter)
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")
    _n_proj = len(proj)
    if "pa" in proj.columns:
        proj = proj.sort_values("pa", ascending=False)
    proj = proj.drop_duplicates("batter_id", keep="first").reset_index(drop=True)
    if len(proj) < _n_proj:
        logger.warning(
            "Deduped hitter_projections: %d -> %d rows (duplicate batter_id)",
            _n_proj, len(proj),
        )

    # Load observed stats
    observed = read_sql(f"""
        SELECT batter_id, pa, k_pct, bb_pct, woba, wrc_plus,
               xwoba, xba, xslg, barrel_pct, hard_hit_pct
        FROM production.fact_batting_advanced
        WHERE season = {season} AND pa >= {min_pa}
    """, {})

    # Park-adjust wOBA (xwOBA/barrel%/hard_hit% are already park-neutral)
    try:
        from src.data.park_factors import compute_multi_stat_park_factors, get_player_park_adjustments
        from src.data.queries import get_hitter_team_venue
        park_factors = compute_multi_stat_park_factors(
            seasons=[season - 2, season - 1, season],
        )
        player_venues = get_hitter_team_venue(season)
        if not park_factors.empty and not player_venues.empty:
            pf_map = get_player_park_adjustments(
                park_factors,
                player_venues.rename(columns={"batter_id": "player_id"}),
            )
            pf_df = pd.DataFrame([
                {"batter_id": pid, "pf_r": v.get("pf_r", 1.0)}
                for pid, v in pf_map.items()
            ])
            observed = observed.merge(pf_df, on="batter_id", how="left")
            observed["pf_r"] = observed["pf_r"].fillna(1.0)
            observed["woba_raw"] = observed["woba"]
            observed["woba"] = observed["woba"] / observed["pf_r"]
            logger.info("Park-adjusted wOBA for %d hitters", (observed["pf_r"] != 1.0).sum())
    except Exception:
        logger.warning("Could not load park factors -- using raw wOBA", exc_info=True)

    # Multi-year observed stats (recency-weighted) for production and damage
    # buckets.  Single-season observed overweights outlier years (Springer's
    # career-best at 35, Grisham's breakout at 28).  Multi-year smoothing
    # uses the same 3/2/1 pattern as fielding.
    try:
        obs_multi = read_sql(f"""
            SELECT batter_id, season, pa, wrc_plus,
                   xwoba, barrel_pct, hard_hit_pct
            FROM production.fact_batting_advanced
            WHERE season BETWEEN {season - 2} AND {season} AND pa >= 50
        """, {})
        if not obs_multi.empty:
            # Flatten recency weights for age 33+: a 35-year-old's
            # career-best most-recent season should not dominate the
            # blend the way a 26-year-old's breakout should.
            # Standard: 3/2/1.  Age 33+: 2/2/1 (recent year loses edge).
            obs_multi = obs_multi.merge(
                proj[["batter_id", "age"]].drop_duplicates("batter_id"),
                on="batter_id", how="left",
            )
            is_aging = obs_multi["age"].fillna(28) >= 33
            base_wt = obs_multi["season"].map(
                {season: 3, season - 1: 2, season - 2: 1}
            ).fillna(1)
            aging_wt = obs_multi["season"].map(
                {season: 2, season - 1: 2, season - 2: 1}
            ).fillna(1)
            obs_multi["recency_wt"] = np.where(is_aging, aging_wt, base_wt)
            obs_multi.drop(columns=["age"], inplace=True)
            obs_multi["total_wt"] = obs_multi["recency_wt"] * obs_multi["pa"]

            def _wtd_avg(g, col):
                valid = g[g[col].notna()]
                if valid.empty or valid["total_wt"].sum() == 0:
                    return np.nan
                return np.average(valid[col], weights=valid["total_wt"])

            multiyear = obs_multi.groupby("batter_id").agg(
                wrc_plus_multiyear=("wrc_plus", lambda g: _wtd_avg(obs_multi.loc[g.index], "wrc_plus")),
                xwoba_multiyear=("xwoba", lambda g: _wtd_avg(obs_multi.loc[g.index], "xwoba")),
                barrel_pct_multiyear=("barrel_pct", lambda g: _wtd_avg(obs_multi.loc[g.index], "barrel_pct")),
                hard_hit_pct_multiyear=("hard_hit_pct", lambda g: _wtd_avg(obs_multi.loc[g.index], "hard_hit_pct")),
                obs_years=("season", "nunique"),
            ).reset_index()
            observed = observed.merge(multiyear, on="batter_id", how="left")
            logger.info("Multi-year observed stats: %d hitters with 2+ years",
                        (observed["obs_years"].fillna(1) >= 2).sum())
    except Exception:
        logger.warning("Could not load multi-year observed stats", exc_info=True)

    _n_obs = len(observed)
    observed = observed.sort_values("pa", ascending=False).drop_duplicates(
        "batter_id", keep="first",
    ).reset_index(drop=True)
    if len(observed) < _n_obs:
        logger.warning(
            "Deduped fact_batting_advanced rows: %d -> %d (duplicate batter_id)",
            _n_obs, len(observed),
        )

    # Position assignments
    positions = _assign_hitter_positions(season=season)

    # Load aggressiveness data for discipline modifier
    from src.data.queries import get_hitter_aggressiveness
    aggressiveness = get_hitter_aggressiveness(season)

    # Preseason: allow sim wRC+ from hitter_counting_sim. In-season
    # (season == projection_season) skip -- weekly refresh does not rebuild sim.
    _in_season = season == projection_season
    use_hitter_sim = not _in_season

    # Load current IL player IDs for offense dampening exemption.
    # Players on IL with < 10 PA get "active starter" treatment instead
    # of bench-player dampening, since they'll return to the lineup.
    _il_ids: set[int] = set()
    if _in_season:
        try:
            from src.data.db import read_sql
            _il_df = read_sql("""
                SELECT DISTINCT player_id
                FROM production.fact_player_status_timeline
                WHERE season = :season
                  AND status_type LIKE '%%IL%%'
                  AND (status_end_date IS NULL
                       OR status_end_date >= CURRENT_DATE - INTERVAL '7 days')
            """, {"season": season})
            _il_ids = set(_il_df["player_id"])
            logger.info("Loaded %d IL player IDs for dampening exemption", len(_il_ids))
        except Exception:
            logger.warning("Could not load IL data; no dampening exemptions")

    # Build sub-scores
    offense = _build_hitter_offense_score(
        proj, observed, aggressiveness=aggressiveness,
        use_sim_wrc_plus=use_hitter_sim,
        il_player_ids=_il_ids if _in_season else None,
    )
    baserunning = _build_hitter_baserunning_score(season=season)
    platoon = _build_hitter_platoon_modifier(season=season)
    fielding = _build_hitter_fielding_score(season=season, in_season=_in_season)
    framing = _build_catcher_framing_score(season=season, in_season=_in_season)
    playing_time = _build_hitter_playing_time_score(health_df=health_df)
    trajectory = _build_hitter_trajectory_score()
    confirmed_breakout = _build_confirmed_breakout_score(season=season - 1)

    # Start from projections as base (has batter_id, name, age, etc.)
    _base_cols = ["batter_id", "batter_name", "age", "batter_stand",
                  "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
                  "projected_k_rate_sd", "projected_bb_rate_sd",
                  "composite_score"]
    if "projected_woba" in proj.columns:
        _base_cols.append("projected_woba")
    base = proj[_base_cols].copy()

    # Merge observed stats for display (include park-adjusted columns if available)
    obs_cols = ["batter_id", "pa", "woba", "wrc_plus", "xwoba",
                "xba", "xslg", "barrel_pct", "hard_hit_pct"]
    for extra in ["woba_raw", "pf_r"]:
        if extra in observed.columns:
            obs_cols.append(extra)
    base = base.merge(observed[obs_cols], on="batter_id", how="left")

    # Flag whether displayed stats are projected-only, early-season, or observed
    base["stats_type"] = np.where(
        base["pa"].isna(), "projected",
        np.where(base["pa"] < 100, "early_season", "observed"),
    )

    # Merge aggressiveness data for discipline modifier and display
    if not aggressiveness.empty:
        _agg = aggressiveness[
            ["batter_id", "chase_rate", "two_strike_whiff_rate"]
        ].drop_duplicates("batter_id", keep="first")
        base = base.merge(_agg, on="batter_id", how="left")

    # Merge position (left join -- players without position data default to DH)
    base = base.merge(
        positions.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base["position"] = base["position"].fillna("DH")

    # Merge sub-scores
    base = base.merge(offense, on="batter_id", how="left")
    base = base.merge(baserunning, on="batter_id", how="left")
    base = base.merge(
        platoon.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
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
    base = base.merge(confirmed_breakout, on="batter_id", how="left")

    # Merge sim-based counting projections for display (same guard as offense)
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    if use_hitter_sim and sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        sim_cols = ["batter_id", "total_k_mean", "total_bb_mean", "total_hr_mean",
                     "total_h_mean", "total_r_mean", "total_rbi_mean", "total_sb_mean",
                     "projected_woba_mean", "projected_wrc_plus_mean",
                     "projected_wraa_mean", "projected_ops_mean", "projected_avg_mean",
                     "dk_season_mean", "espn_season_mean", "total_games_mean", "total_pa_mean"]
        sim_cols = [c for c in sim_cols if c in sim_df.columns]
        sim_df = sim_df[sim_cols].drop_duplicates("batter_id", keep="first")
        base = base.merge(sim_df, on="batter_id", how="left")
        logger.info("Merged hitter sim projections for %d batters",
                     base["projected_wrc_plus_mean"].notna().sum() if "projected_wrc_plus_mean" in base.columns else 0)

    # Breakout archetype data (GMM-derived)
    breakout_path = DASHBOARD_DIR / "hitter_breakout_candidates.parquet"
    if breakout_path.exists():
        breakout_df = pd.read_parquet(breakout_path)
        breakout_cols = [
            "batter_id", "breakout_type", "breakout_score",
            "breakout_tier", "breakout_hole", "gmm_fit",
            "prob_power_surge", "prob_diamond_in_the_rough",
        ]
        available_bc = [c for c in breakout_cols if c in breakout_df.columns]
        _bo = breakout_df[available_bc].drop_duplicates("batter_id", keep="first")
        base = base.merge(_bo, on="batter_id", how="left")
    else:
        for col in ["breakout_type", "breakout_tier", "breakout_hole"]:
            base[col] = ""
        for col in ["breakout_score", "gmm_fit"]:
            base[col] = np.nan

    # Fill missing with neutral
    base["fielding_score"] = base["fielding_score"].fillna(0.50)
    base["framing_score"] = base["framing_score"].fillna(0.50)
    base["offense_score"] = base["offense_score"].fillna(0.50)
    base["baserunning_score"] = base["baserunning_score"].fillna(0.50)
    base["platoon_score"] = base["platoon_score"].fillna(0.50)

    # --- Postseason performance boost (player-specific) ---
    # Apply individual postseason performance as an additive adjustment
    # to offense_score.  Elite October performers get up to +5%, poor
    # performers get up to -5%.  Heavy sample-size dampening prevents
    # small-sample noise (e.g. 2 IP in October barely moves the needle).
    try:
        from src.data.queries import get_postseason_batter_stats
        from src.models.postseason_boost import compute_postseason_scores

        ps_seasons = list(range(max(2018, projection_season - 3), projection_season))
        ps_bat = get_postseason_batter_stats(ps_seasons)
        if not ps_bat.empty:
            ps_hitter_scores, _ = compute_postseason_scores(
                ps_bat, pd.DataFrame(), season=projection_season - 1,
            )
            if not ps_hitter_scores.empty:
                base = base.merge(
                    ps_hitter_scores[["batter_id", "postseason_score",
                                      "postseason_pa", "best_round"]].rename(
                        columns={"best_round": "ps_best_round"}
                    ),
                    on="batter_id", how="left",
                )
                base["postseason_score"] = base["postseason_score"].fillna(0.50)
                ps_adj = (base["postseason_score"] - 0.50) * 0.10  # +/-5% max
                base["offense_score"] = np.clip(
                    base["offense_score"] + ps_adj, 0, 1,
                )
                n_affected = (ps_adj.abs() > 0.001).sum()
                logger.info(
                    "Postseason offense adjustment: %d hitters affected "
                    "(max +%.3f, min %.3f)",
                    n_affected, ps_adj.max(), ps_adj.min(),
                )
    except Exception:
        logger.warning("Could not compute hitter postseason boost", exc_info=True)
        base["postseason_score"] = 0.50

    base["pt_score"] = base["pt_score"].fillna(0.50)
    base["trajectory_score"] = base["trajectory_score"].fillna(0.50)

    # Breakout + confirmed breakout blend into trajectory.
    # XGBoost breakout (future-facing) gets 10%, confirmed breakout
    # (recent-season validation with trajectory direction) gets 15%.
    # Trajectory's own components (age + YoY + certainty) still dominate
    # at 75%, avoiding the circular dependency that existed at 35%.
    base["breakout_score"] = base["breakout_score"].fillna(0.0)
    breakout_pctl = _pctl(base["breakout_score"].clip(lower=0))
    base["confirmed_breakout_score"] = base["confirmed_breakout_score"].fillna(0.50)
    cb_pctl = _pctl(base["confirmed_breakout_score"])
    raw_trajectory = base["trajectory_score"].copy()  # save for upside calc
    base["trajectory_score"] = (
        0.70 * base["trajectory_score"]
        + 0.15 * cb_pctl
        + 0.15 * breakout_pctl
    )

    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Health & role scores (replace old playing_time composite) ---
    base["health_adj"] = base["health_score"].fillna(0.50)
    # Role score: everyday (140+ games) = 1.0, platoon (70) = 0.5, bench (30) = 0.21
    if "total_games_mean" in base.columns:
        base["role_score"] = (base["total_games_mean"].fillna(0) / 140).clip(0, 1)
    else:
        # Fallback to pt_score which already captures playing time
        base["role_score"] = base["pt_score"]

    # --- Versatility & roster construction ---
    base["versatility_score"] = _build_versatility_score(base)
    base["roster_value_score"] = _build_roster_construction_score(base)

    # --- Glicko-2 opponent-adjusted performance ---
    glicko = _load_glicko_scores("batter")
    if not glicko.empty:
        base = base.merge(glicko, on="batter_id", how="left")
    if "glicko_score" not in base.columns:
        base["glicko_score"] = 0.5  # neutral default
    base["glicko_score"] = base["glicko_score"].fillna(0.5)

    # --- Composite score ---
    # For catchers: blend framing into fielding component
    is_catcher = base["position"] == "C"
    base["fielding_combined"] = base["fielding_score"]
    base.loc[is_catcher, "fielding_combined"] = (
        0.50 * base.loc[is_catcher, "fielding_score"]
        + 0.50 * base.loc[is_catcher, "framing_score"]
    )

    # Percentile-rank fielding and baserunning, then compress toward 0.50
    # so their effective spread matches offense.  After early-season PA
    # dampening, offense IQR ≈ 0.14 while raw percentile IQR ≈ 0.47.
    # Without compression, 13% fielding weight has ~44% effective impact.
    # Sigmoid compression: preserves rank order but pulls extremes toward
    # center.  A 0.067 fielder → ~0.25 (not devastated), a 0.95 → ~0.80
    # (still rewarded, not dominant).
    base["fielding_combined"] = _pctl(base["fielding_combined"])
    base["baserunning_score"] = _pctl(base["baserunning_score"])
    # Compress: blend 60% percentile + 40% center (0.50)
    base["fielding_combined"] = 0.60 * base["fielding_combined"] + 0.40 * 0.50
    base["baserunning_score"] = 0.60 * base["baserunning_score"] + 0.40 * 0.50

    # Positional adjustment (FanGraphs-standard run values, 0-1 scale)
    base["positional_adj"] = base["position"].map(_POSITIONAL_OFFENSE_ADJ).fillna(0.333)

    # For DH: redistribute fielding weight to offense + baserunning.
    # DHs are evaluated purely on production -- no artificial cap.
    is_dh = base["position"] == "DH"
    dh_offense_wt = _HITTER_WEIGHTS["offense"] + _HITTER_WEIGHTS["fielding"] * 0.70
    dh_baserunning_wt = _HITTER_WEIGHTS["baserunning"] + _HITTER_WEIGHTS["fielding"] * 0.30

    # Dynamic fielding dampening for elite hitters.
    # If you're so good with the bat, teams will hide you at DH/1B/corner OF.
    # Worst fielder costs ~-15 runs; best hitter adds ~+50 runs -- fielding
    # range is ~30% of offense range, so elite bats should have reduced
    # fielding penalty.  Kicks in above 0.65 offense, reduces fielding
    # weight by up to 75% at 1.00 offense.  Redistributed weight goes to
    # offense.  Threshold lowered from 0.70 and ceiling raised from 0.50
    # because early-season dampening compresses offense (IQR ~0.14),
    # making 0.70 harder to reach while fielding spread stays wide.
    offense_excess = (base["offense_score"] - 0.65).clip(0) / 0.35
    fielding_dampening = 0.75 * offense_excess
    eff_fielding_wt = _HITTER_WEIGHTS["fielding"] * (1 - fielding_dampening)
    eff_offense_boost = _HITTER_WEIGHTS["fielding"] * fielding_dampening

    # Standard composite (non-DH)
    base["tdd_value_score"] = (
        (_HITTER_WEIGHTS["offense"] + eff_offense_boost) * base["offense_score"]
        + _HITTER_WEIGHTS["baserunning"] * base["baserunning_score"]
        + eff_fielding_wt * base["fielding_combined"]
        + _HITTER_WEIGHTS["positional_adj"] * base["positional_adj"]
        + _HITTER_WEIGHTS["role"] * base["role_score"]
        + _HITTER_WEIGHTS["trajectory"] * base["trajectory_score"]
        + _HITTER_WEIGHTS["versatility"] * base["versatility_score"]
    )
    # DH composite: fielding + positional weight fully redistributed
    dh_pos_wt = _HITTER_WEIGHTS["positional_adj"]  # DH positional_adj = 0.0, so this zeroes out
    base.loc[is_dh, "tdd_value_score"] = (
        dh_offense_wt * base.loc[is_dh, "offense_score"]
        + dh_baserunning_wt * base.loc[is_dh, "baserunning_score"]
        + dh_pos_wt * base.loc[is_dh, "positional_adj"]
        + _HITTER_WEIGHTS["role"] * base.loc[is_dh, "role_score"]
        + _HITTER_WEIGHTS["trajectory"] * base.loc[is_dh, "trajectory_score"]
        + _HITTER_WEIGHTS["versatility"] * base.loc[is_dh, "versatility_score"]
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
                # Scale bonus: elite pitcher value (0.8+) adds up to ~0.10.
                # Reduced from 0.18 (2026-04-06): old multiplier created a
                # 0.131 gap between Ohtani (.994) and #2 (.863) — larger
                # than the spread of the next 9 players combined.  At 0.12,
                # Ohtani's bonus is ~0.09 (still clearly #1, gap ~0.09).
                bonus = pitcher_value * 0.12
                mask = base["batter_id"] == pid
                base.loc[mask, "two_way_bonus"] = bonus
                base.loc[mask, "is_two_way"] = True
                pname = base.loc[mask, "batter_name"].iloc[0]
                logger.info(
                    "Two-way bonus: %s -- pitcher value=%.3f, bonus=+%.3f",
                    pname, pitcher_value, bonus,
                )
    except Exception:
        logger.exception("Could not compute two-way player bonus")

    base["tdd_value_score"] = base["tdd_value_score"] + base["two_way_bonus"]

    # --- Two-score architecture: current_value + talent_upside ---
    # Save the production composite before scouting blending.
    # This is the weighted sum of offense, fielding, trajectory, health,
    # role, glicko, etc. -- the "what has he done / what will he do" signal.
    production_composite = base["tdd_value_score"].copy()

    # Initialize both scores to production composite (scouting may fail)
    base["current_value_score"] = production_composite
    base["talent_upside_score"] = production_composite

    try:
        from src.models.scouting_grades import grade_hitter_tools
        # In-season: use prior season for scouting grades.  Current-season
        # data is too sparse (everyone has < 150 PA, so _composite_z
        # confidence = 0 and every grade = 50).  Build a grade-specific
        # base with prior-season batting stats so the PA-confidence ramp
        # has real data to work with.
        _grade_season = season - 1 if _in_season else season
        if _in_season:
            from src.data.db import read_sql as _grade_sql
            _prior_obs = _grade_sql(f"""
                SELECT batter_id, pa, k_pct, bb_pct, woba, wrc_plus,
                       xba, barrel_pct, hard_hit_pct
                FROM production.fact_batting_advanced
                WHERE season = {season - 1} AND pa >= 50
            """, {})
            _grade_cols = ["batter_id", "batter_name", "age", "position",
                          "projected_k_rate", "projected_bb_rate",
                          "fielding_combined", "chase_rate",
                          "two_strike_whiff_rate"]
            _grade_cols = [c for c in _grade_cols if c in base.columns]
            _grade_base = base[_grade_cols].copy()
            _grade_base = _grade_base.merge(
                _prior_obs, on="batter_id", how="left",
            )
        else:
            _grade_base = base
        scouting = grade_hitter_tools(_grade_base, season=_grade_season)
        if not scouting.empty:
            base = base.merge(scouting, on="batter_id", how="left")
            logger.info("Scouting grades computed for %d hitters", scouting["tools_rating"].notna().sum())

            # Regress diamond rating by PA reliability (same for both scores)
            dr_norm = (base["tools_rating"] / 10.0).clip(0, 1)

            pa_col = base["pa"].fillna(0) if "pa" in base.columns else base.get("total_pa", 400)
            cfg = _RANK_CFG["hitter"]
            reliability = ((pa_col - cfg["pa_ramp_min"]) / (cfg["pa_ramp_max"] - cfg["pa_ramp_min"])).clip(0, 1)

            # Low-PA players: regress diamond rating toward 0.50 (league avg)
            dr_regressed = reliability * dr_norm + (1 - reliability) * 0.50

            # Health penalty on DR: injury-prone players get dampened for
            # current_value (availability matters for near-term), but NOT
            # for talent_upside (tools and health are separate -- an injured
            # player's tools don't change, only their playing time does).
            health = base["health_adj"] if "health_adj" in base.columns else 0.50
            health_penalty = np.where(health < 0.40, 0.70 + 0.75 * health, 1.0)
            dr_talent = dr_regressed.copy()       # no health penalty for talent
            dr_regressed = dr_regressed * health_penalty  # health penalty for current value

            # ----- talent_upside_score: scouting-dominant -----
            # Tool grades carry heavy weight for forward-looking assessment.
            # Uses dr_talent (no health penalty -- tools and health are separate).
            # No positional multiplier -- raw talent regardless of position.
            upside_w = cfg["upside_scout_weight"]
            base["talent_upside_score"] = (
                upside_w * dr_talent + (1 - upside_w) * production_composite
            )

            # ----- current_value_score: production-dominant -----
            # Scouting grades are a feature, not the override.  The weight
            # shrinks with sample size: 30% at 150 PA -> 10% at 600+ PA.
            # These defaults are configurable in config/model.yaml and will
            # be learned by Phase 2 walk-forward validation.
            scout_w = _exposure_conditioned_scouting_weight(
                pa_col,
                min_exp=cfg["pa_ramp_min"],
                max_exp=cfg["pa_ramp_max"],
                weight_ceil=cfg["scout_weight_ceil"],
                weight_floor=cfg["scout_weight_floor"],
            )
            base["current_value_score"] = (
                scout_w * dr_regressed + (1 - scout_w) * production_composite
            )

            n_scouted = scouting["tools_rating"].notna().sum()
            logger.info(
                "Two-score split: %d hitters -- scout weight range [%.0f%%, %.0f%%]",
                n_scouted, cfg["scout_weight_floor"] * 100, cfg["scout_weight_ceil"] * 100,
            )
    except Exception:
        logger.warning("Could not compute hitter scouting grades", exc_info=True)

    # --- Upside adjustments: breakout trajectory + harsher age curve ---
    # talent_upside_score is the 2-3yr forward view (dynasty rankings).
    # Swap sustain trajectory for breakout: developing players should be
    # rewarded for room to improve, not penalized for not being elite yet.
    # Apply harsher age curve: 2-3yr horizon amplifies decline risk.
    upside_trajectory = 0.65 * raw_trajectory + 0.35 * breakout_pctl
    traj_swap = _HITTER_WEIGHTS["trajectory"] * (upside_trajectory - base["trajectory_score"])
    base["talent_upside_score"] = base["talent_upside_score"] + traj_swap

    # Harsher age: peak 25-27, young players boosted, 30+ penalized on forward horizon
    age = base["age"].fillna(28)
    upside_age_mult = (1.0 + (26 - age) * 0.02).clip(0.76, 1.10)
    # age 21: 1.10, age 24: 1.04, age 26: 1.00, age 30: 0.92, age 33: 0.86, age 38: 0.76
    base["talent_upside_score"] = base["talent_upside_score"] * upside_age_mult

    # DH penalty removed: the DH composite already handles the value gap by
    # excluding fielding weight (redistributed to offense/baserunning) and
    # positional_adj = 0.0 (5% zero contribution).  The old 10% penalty on
    # top was double-penalizing, dropping elite DH bats (Alvarez, Schwarber)
    # 100+ ranks below consensus.

    # tdd_value_score = current_value_score (backward compat for team consumption)
    base["tdd_value_score"] = base["current_value_score"]

    # --- Rank within each position ---
    base = base.sort_values("current_value_score", ascending=False)
    base["pos_rank"] = base.groupby("position").cumcount() + 1

    # --- Overall rank: directly from current_value_score ---
    # No positional adjustment -- the displayed score IS the ranking order.
    # Positional value is captured by pos_rank within each position.
    base["overall_rank"] = (
        base["current_value_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .fillna(len(base))
        .astype(int)
    )

    # --- Talent-based ranking (scouting-dominant, no positional adj) ---
    base["talent_rank"] = (
        base["talent_upside_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .fillna(len(base))
        .astype(int)
    )

    # Select output columns
    output_cols = [
        "pos_rank", "overall_rank", "talent_rank",
        "batter_id", "batter_name", "position",
        "age", "batter_stand", "is_two_way",
        # Two-score architecture
        "current_value_score", "talent_upside_score",
        "tdd_value_score",
        # Sub-scores
        "offense_score", "contact_skill", "decision_skill", "damage_skill", "production_skill",
        "baserunning_score", "platoon_score",
        "fielding_combined", "framing_score", "pt_score",
        "health_adj", "role_score", "versatility_score", "roster_value_score",
        "positional_adj", "trajectory_score", "two_way_bonus",
        # Health
        "health_score", "health_label",
        # Observed (stats_type: "projected" | "early_season" | "observed")
        "stats_type",
        "pa", "woba", "woba_raw", "pf_r", "wrc_plus", "xwoba", "xba", "xslg",
        "barrel_pct", "hard_hit_pct",
        "chase_rate", "two_strike_whiff_rate",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
        "projected_k_rate_sd", "projected_bb_rate_sd",
        "projected_woba",
        # Confirmed breakout (trajectory-direction validated)
        "confirmed_breakout_score",
        # Breakout archetype (GMM-derived)
        "breakout_type", "breakout_score", "breakout_tier",
        "breakout_hole", "gmm_fit",
        "prob_power_surge", "prob_diamond_in_the_rough",
        # Sim-based season projections
        "total_k_mean", "total_bb_mean", "total_hr_mean", "total_h_mean",
        "total_r_mean", "total_rbi_mean", "total_sb_mean",
        "projected_woba_mean", "projected_wrc_plus_mean", "projected_wraa_mean",
        "projected_ops_mean", "projected_avg_mean",
        "dk_season_mean", "espn_season_mean", "total_games_mean", "total_pa_mean",
        # Glicko-2 opponent-adjusted rating
        "glicko_score", "glicko_mu", "glicko_phi",
        # Scouting grades (20-80) + diamond rating (0-10)
        "grade_hit", "grade_power", "grade_speed", "grade_fielding",
        "grade_discipline", "tools_rating",
        # Postseason performance
        "postseason_score", "postseason_pa", "ps_best_round",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["position", "pos_rank"])

    for pos in HITTER_POSITIONS:
        pos_df = result[result["position"] == pos]
        if not pos_df.empty:
            logger.info(
                "%s: %d ranked -- #1 %s (%.3f)",
                pos, len(pos_df),
                pos_df.iloc[0]["batter_name"],
                pos_df.iloc[0]["tdd_value_score"],
            )

    return result


# ===================================================================
# Pitcher ranking orchestrator
# ===================================================================

def rank_pitchers(
    season: int = 2025,
    projection_season: int = 2026,
    min_bf: int = 50,
    health_df: pd.DataFrame | None = None,
    pitcher_roles_df: pd.DataFrame | None = None,
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
    health_df : pd.DataFrame or None
        Pre-loaded health scores (columns: player_id, health_score,
        health_label).  Passed through to ``_build_pitcher_workload_score``
        to avoid a disk read when the pipeline already has it in memory.
    pitcher_roles_df : pd.DataFrame or None
        Pre-loaded reliever roles (columns: pitcher_id, role, ...).
        When provided the disk read of ``pitcher_roles.parquet`` is skipped.

    Returns
    -------
    pd.DataFrame
        Pitchers ranked by role with composite score, sub-scores.
    """
    from src.data.db import read_sql

    # Load projections
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    _n_pp = len(proj)
    if "batters_faced" in proj.columns:
        proj = proj.sort_values("batters_faced", ascending=False)
    proj = proj.drop_duplicates("pitcher_id", keep="first").reset_index(drop=True)
    if len(proj) < _n_pp:
        logger.warning(
            "Deduped pitcher_projections: %d -> %d rows (duplicate pitcher_id)",
            _n_pp, len(proj),
        )

    # Load observed stats
    observed = read_sql(f"""
        SELECT pitcher_id, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, contact_pct, xwoba_against,
               barrel_pct_against, hard_hit_pct_against,
               batters_faced, woba_against
        FROM production.fact_pitching_advanced
        WHERE season = {season} AND batters_faced >= {min_bf}
    """, {})
    _n_po = len(observed)
    observed = observed.sort_values("batters_faced", ascending=False).drop_duplicates(
        "pitcher_id", keep="first",
    ).reset_index(drop=True)
    if len(observed) < _n_po:
        logger.warning(
            "Deduped fact_pitching_advanced: %d -> %d (duplicate pitcher_id)",
            _n_po, len(observed),
        )

    # Role assignment -- prefer in-memory roles, then disk, then heuristic
    rp_roles = pitcher_roles_df
    if rp_roles is None:
        roles_path = DASHBOARD_DIR / "pitcher_roles.parquet"
        if roles_path.exists():
            rp_roles = pd.read_parquet(roles_path)
    if rp_roles is not None and not rp_roles.empty:
        # Map CL/SU/MR -> RP for composite weights, keep detail for display
        roles = rp_roles[["pitcher_id", "role"]].copy()
        roles["role_detail"] = roles["role"]
        roles["role"] = roles["role"].map(
            {"CL": "RP", "SU": "RP", "MR": "RP"}
        ).fillna(roles["role"])
        # Add SP pitchers not in reliever roles
        sp_roles = _assign_pitcher_roles()
        sp_only = sp_roles[
            (sp_roles["role"] == "SP")
            & (~sp_roles["pitcher_id"].isin(roles["pitcher_id"]))
        ].copy()
        sp_only["role_detail"] = "SP"
        roles = pd.concat([roles, sp_only], ignore_index=True)
        logger.info("Roles from sim: %s", roles["role_detail"].value_counts().to_dict())
    else:
        roles = _assign_pitcher_roles()
        roles["role_detail"] = roles["role"]

    roles = roles.drop_duplicates("pitcher_id", keep="first")

    # Load run values and efficiency for enhanced scoring
    from src.data.queries import get_pitcher_run_values, get_pitcher_efficiency
    run_values = get_pitcher_run_values(season)
    efficiency = get_pitcher_efficiency(season)

    use_pitcher_sim = season != projection_season

    # Build sub-scores
    stuff = _build_pitcher_stuff_score(proj, observed, run_values=run_values)
    command = _build_pitcher_command_score(proj, observed, efficiency=efficiency)
    workload = _build_pitcher_workload_score(
        health_df=health_df, use_counting_sim=use_pitcher_sim,
    )
    durability = _build_pitcher_innings_durability()
    trajectory = _build_pitcher_trajectory_score(season=season)

    # Base from projections (include ERA/FIP if available)
    base_cols = ["pitcher_id", "pitcher_name", "age", "pitch_hand",
                 "is_starter", "projected_k_rate", "projected_bb_rate",
                 "projected_hr_per_bf", "projected_k_rate_sd",
                 "projected_bb_rate_sd", "composite_score",
                 "projected_era", "projected_era_sd",
                 "projected_era_2_5", "projected_era_97_5",
                 "projected_fip", "projected_fip_sd",
                 "observed_era", "observed_fip"]
    base_cols = [c for c in base_cols if c in proj.columns]
    base = proj[base_cols].copy()

    # Merge observed for display (left join -- include projected-only pitchers)
    base = base.merge(
        observed[["pitcher_id", "batters_faced", "k_pct", "bb_pct",
                   "swstr_pct", "csw_pct", "xwoba_against", "woba_against"]],
        on="pitcher_id", how="left",
    )

    # Merge run values for display
    if not run_values.empty:
        rv_display = run_values[["pitcher_id", "weighted_rv_per_100"]].drop_duplicates()
        base = base.merge(rv_display, on="pitcher_id", how="left")

    # Merge efficiency for display
    if not efficiency.empty:
        _eff = efficiency[
            ["pitcher_id", "first_strike_pct", "putaway_rate"]
        ].drop_duplicates("pitcher_id", keep="first")
        base = base.merge(_eff, on="pitcher_id", how="left")

    # Merge role
    base = base.merge(roles, on="pitcher_id", how="left")
    missing_role = base["role"].isna()
    base.loc[missing_role, "role"] = np.where(
        base.loc[missing_role, "is_starter"] == 1, "SP", "RP"
    )
    if "role_detail" not in base.columns:
        base["role_detail"] = base["role"]
    base["role_detail"] = base["role_detail"].fillna(base["role"])

    # Merge sim-based counting projections for display (same guard as workload)
    sim_path = DASHBOARD_DIR / "pitcher_counting_sim.parquet"
    if use_pitcher_sim and sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        sim_cols = ["pitcher_id", "total_k_mean", "total_bb_mean", "total_sv_mean",
                     "total_hld_mean", "projected_ip_mean", "projected_era_mean",
                     "projected_fip_era_mean", "projected_whip_mean",
                     "dk_season_mean", "espn_season_mean", "total_games_mean"]
        sim_cols = [c for c in sim_cols if c in sim_df.columns]
        sim_df = sim_df[sim_cols].drop_duplicates("pitcher_id", keep="first")
        base = base.merge(sim_df, on="pitcher_id", how="left")
        logger.info("Merged sim projections for %d pitchers", base["total_k_mean"].notna().sum())

    # Merge sub-scores
    base = base.merge(stuff, on="pitcher_id", how="left")
    base = base.merge(command, on="pitcher_id", how="left")
    base = base.merge(workload, on="pitcher_id", how="left")
    base = base.merge(durability, on="pitcher_id", how="left")
    base = base.merge(trajectory, on="pitcher_id", how="left")

    # Pitcher breakout archetype data (GMM-derived)
    p_breakout_path = DASHBOARD_DIR / "pitcher_breakout_candidates.parquet"
    if p_breakout_path.exists():
        p_bo = pd.read_parquet(p_breakout_path)
        p_bo_cols = [
            "pitcher_id", "breakout_type", "breakout_score",
            "breakout_tier", "breakout_hole", "gmm_fit",
            "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        ]
        available_pbc = [c for c in p_bo_cols if c in p_bo.columns]
        _pbo = p_bo[available_pbc].drop_duplicates("pitcher_id", keep="first")
        base = base.merge(_pbo, on="pitcher_id", how="left")
    else:
        for col in ["breakout_type", "breakout_tier", "breakout_hole"]:
            base[col] = ""
        for col in ["breakout_score", "gmm_fit"]:
            base[col] = np.nan

    # Fill missing with neutral
    for col in ["stuff_score", "command_score", "workload_score", "trajectory_score"]:
        base[col] = base[col].fillna(0.50)
    base["durability_score"] = base["durability_score"].fillna(0.50)
    base["velo_delta"] = base["velo_delta"].fillna(0)
    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Postseason performance boost (player-specific) ---
    try:
        from src.data.queries import get_postseason_pitcher_stats
        from src.models.postseason_boost import compute_postseason_scores

        ps_seasons = list(range(max(2018, projection_season - 3), projection_season))
        ps_pit = get_postseason_pitcher_stats(ps_seasons)
        if not ps_pit.empty:
            _, ps_pitcher_scores = compute_postseason_scores(
                pd.DataFrame(), ps_pit, season=projection_season - 1,
            )
            if not ps_pitcher_scores.empty:
                base = base.merge(
                    ps_pitcher_scores[["pitcher_id", "postseason_score",
                                       "postseason_bf", "best_round"]].rename(
                        columns={"best_round": "ps_best_round"}
                    ),
                    on="pitcher_id", how="left",
                )
                base["postseason_score"] = base["postseason_score"].fillna(0.50)
                ps_adj = (base["postseason_score"] - 0.50) * 0.10  # +/-5% max
                base["stuff_score"] = np.clip(
                    base["stuff_score"] + ps_adj, 0, 1,
                )
                n_affected = (ps_adj.abs() > 0.001).sum()
                logger.info(
                    "Postseason stuff adjustment: %d pitchers affected "
                    "(max +%.3f, min %.3f)",
                    n_affected, ps_adj.max(), ps_adj.min(),
                )
    except Exception:
        logger.warning("Could not compute pitcher postseason boost", exc_info=True)
        base["postseason_score"] = 0.50

    # --- Health & role scores ---
    base["health_adj"] = base["health_score"].fillna(0.50)
    # Role score: SP uses starts (min(starts/30, 1)), RP uses appearances (min(apps/60, 1))
    if "total_games_mean" in base.columns:
        # SP: projected games ~= starts for starters, use /30 target
        # RP: projected games ~= appearances for relievers, use /60 target
        sp_mask = base["role"] == "SP"
        rp_mask = base["role"] == "RP"
        base["role_score"] = 0.5  # default
        base.loc[sp_mask, "role_score"] = (
            base.loc[sp_mask, "total_games_mean"].fillna(0) / 30
        ).clip(0, 1)
        base.loc[rp_mask, "role_score"] = (
            base.loc[rp_mask, "total_games_mean"].fillna(0) / 60
        ).clip(0, 1)
    elif "batters_faced" in base.columns:
        # Fallback: use BF as proxy (SP ~700 BF/season, RP ~250)
        base["role_score"] = _pctl(base["batters_faced"].fillna(0))
    else:
        base["role_score"] = 0.5

    # Sustain/upside blend into trajectory (mirrors hitter pattern).
    # Without this, elite established pitchers (Crochet, Yamamoto) get
    # penalized for having "no room to break out" -- same problem hitters
    # had with Soto/Judge before the sustain fix.
    #   - Elite pitchers: stuff_score provides a high trajectory floor
    #   - Developing pitchers: breakout_pctl drives the score
    base["breakout_score"] = base["breakout_score"].fillna(0.0)
    p_breakout_pctl = _pctl(base["breakout_score"].clip(lower=0))
    sustain_upside = np.maximum(p_breakout_pctl, base["stuff_score"])
    base["trajectory_score"] = (
        0.65 * base["trajectory_score"] + 0.35 * sustain_upside
    )

    # Blend innings durability into SP workload (30% durability, 70% base workload)
    is_sp = base["role"] == "SP"
    base.loc[is_sp, "workload_score"] = (
        0.70 * base.loc[is_sp, "workload_score"]
        + 0.30 * _pctl(base.loc[is_sp, "durability_score"])
    )

    # --- Glicko-2 opponent-adjusted performance ---
    glicko = _load_glicko_scores("pitcher")
    if not glicko.empty:
        base = base.merge(glicko, on="pitcher_id", how="left")
    if "glicko_score" not in base.columns:
        base["glicko_score"] = 0.5
    base["glicko_score"] = base["glicko_score"].fillna(0.5)

    # --- Composite (role-specific weights) ---
    is_sp = base["role"] == "SP"
    is_rp = base["role"] == "RP"

    # SP composite
    base.loc[is_sp, "tdd_value_score"] = (
        _SP_WEIGHTS["stuff"] * base.loc[is_sp, "stuff_score"]
        + _SP_WEIGHTS["command"] * base.loc[is_sp, "command_score"]
        + _SP_WEIGHTS["workload"] * base.loc[is_sp, "workload_score"]
        + _SP_WEIGHTS["health"] * base.loc[is_sp, "health_adj"]
        + _SP_WEIGHTS["role"] * base.loc[is_sp, "role_score"]
        + _SP_WEIGHTS["trajectory"] * base.loc[is_sp, "trajectory_score"]
        + _SP_WEIGHTS["glicko"] * base.loc[is_sp, "glicko_score"]
    )
    # RP composite
    base.loc[is_rp, "tdd_value_score"] = (
        _RP_WEIGHTS["stuff"] * base.loc[is_rp, "stuff_score"]
        + _RP_WEIGHTS["command"] * base.loc[is_rp, "command_score"]
        + _RP_WEIGHTS["workload"] * base.loc[is_rp, "workload_score"]
        + _RP_WEIGHTS["health"] * base.loc[is_rp, "health_adj"]
        + _RP_WEIGHTS["role"] * base.loc[is_rp, "role_score"]
        + _RP_WEIGHTS["trajectory"] * base.loc[is_rp, "trajectory_score"]
        + _RP_WEIGHTS["glicko"] * base.loc[is_rp, "glicko_score"]
    )
    # Fallback for any unassigned role
    neither = ~is_sp & ~is_rp
    if neither.any():
        base.loc[neither, "tdd_value_score"] = (
            _SP_WEIGHTS["stuff"] * base.loc[neither, "stuff_score"]
            + _SP_WEIGHTS["command"] * base.loc[neither, "command_score"]
            + _SP_WEIGHTS["workload"] * base.loc[neither, "workload_score"]
            + _SP_WEIGHTS["health"] * base.loc[neither, "health_adj"]
            + _SP_WEIGHTS["role"] * base.loc[neither, "role_score"]
            + _SP_WEIGHTS["trajectory"] * base.loc[neither, "trajectory_score"]
            + _SP_WEIGHTS["glicko"] * base.loc[neither, "glicko_score"]
        )

    # --- Two-score architecture: current_value + talent_upside ---
    # Save production composite before scouting blending.
    production_composite = base["tdd_value_score"].copy()
    base["current_value_score"] = production_composite
    base["talent_upside_score"] = production_composite

    try:
        from src.models.scouting_grades import grade_pitcher_tools
        scouting = grade_pitcher_tools(base, season=season)
        if not scouting.empty:
            base = base.merge(scouting, on="pitcher_id", how="left")
            logger.info("Scouting grades computed for %d pitchers", scouting["tools_rating"].notna().sum())

            # Regress diamond rating by BF reliability (same for both scores)
            dr_norm = (base["tools_rating"] / 10.0).clip(0, 1)

            cfg = _RANK_CFG["pitcher"]
            bf_col = base["batters_faced"] if "batters_faced" in base.columns else 400
            reliability = ((bf_col - cfg["bf_ramp_min"]) / (cfg["bf_ramp_max"] - cfg["bf_ramp_min"])).clip(0, 1)

            # Low-IP pitchers: regress diamond rating toward 0.50
            dr_regressed = reliability * dr_norm + (1 - reliability) * 0.50

            # Health penalty: injury-prone pitchers get DR dampened.
            health = base["health_adj"] if "health_adj" in base.columns else 0.50
            health_penalty = np.where(health < 0.40, 0.70 + 0.75 * health, 1.0)
            dr_regressed = dr_regressed * health_penalty

            # ----- talent_upside_score: scouting-dominant -----
            upside_w = cfg["upside_scout_weight"]
            base["talent_upside_score"] = (
                upside_w * dr_regressed + (1 - upside_w) * production_composite
            )

            # ----- current_value_score: production-dominant -----
            scout_w = _exposure_conditioned_scouting_weight(
                bf_col,
                min_exp=cfg["bf_ramp_min"],
                max_exp=cfg["bf_ramp_max"],
                weight_ceil=cfg["scout_weight_ceil"],
                weight_floor=cfg["scout_weight_floor"],
            )
            base["current_value_score"] = (
                scout_w * dr_regressed + (1 - scout_w) * production_composite
            )

            n_scouted = scouting["tools_rating"].notna().sum()
            logger.info(
                "Two-score split: %d pitchers -- scout weight range [%.0f%%, %.0f%%]",
                n_scouted, cfg["scout_weight_floor"] * 100, cfg["scout_weight_ceil"] * 100,
            )
    except Exception:
        logger.warning("Could not compute pitcher scouting grades", exc_info=True)

    # tdd_value_score = current_value_score (backward compat for team consumption)
    base["tdd_value_score"] = base["current_value_score"]

    # Rank within role
    base = base.sort_values("current_value_score", ascending=False)
    base["role_rank"] = base.groupby("role").cumcount() + 1

    # Overall pitcher rank
    base["overall_rank"] = (
        base["current_value_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .fillna(len(base))
        .astype(int)
    )

    # Talent-based ranking (scouting-dominant)
    base["talent_rank"] = (
        base["talent_upside_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .fillna(len(base))
        .astype(int)
    )

    # Select output
    output_cols = [
        "role_rank", "overall_rank", "talent_rank",
        "pitcher_id", "pitcher_name", "role",
        "role_detail", "age", "pitch_hand",
        # Two-score architecture
        "current_value_score", "talent_upside_score",
        "tdd_value_score",
        # Sub-scores
        "stuff_score", "arsenal_stuff_plus", "command_score", "workload_score",
        "health_adj", "role_score",
        "trajectory_score",
        "durability_score", "velo_delta",
        # Health
        "health_score", "health_label",
        # Observed
        "batters_faced", "k_pct", "bb_pct", "swstr_pct", "csw_pct",
        "xwoba_against", "woba_against",
        "weighted_rv_per_100", "first_strike_pct", "putaway_rate",
        "observed_era", "observed_fip",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_bf",
        "projected_k_rate_sd", "projected_bb_rate_sd",
        "projected_era", "projected_era_sd",
        "projected_era_2_5", "projected_era_97_5",
        "projected_fip", "projected_fip_sd",
        # Breakout archetype (GMM-derived)
        "breakout_type", "breakout_score", "breakout_tier",
        "breakout_hole", "gmm_fit",
        "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        # Sim-based season projections
        "total_k_mean", "total_bb_mean", "total_sv_mean", "total_hld_mean",
        "projected_ip_mean", "projected_era_mean", "projected_fip_era_mean",
        "projected_whip_mean", "dk_season_mean", "espn_season_mean",
        "total_games_mean",
        # Glicko-2 opponent-adjusted rating
        "glicko_score", "glicko_mu", "glicko_phi",
        # Scouting grades (20-80) + diamond rating (0-10)
        "grade_stuff", "grade_command", "grade_durability", "tools_rating",
        # Postseason performance
        "postseason_score", "postseason_bf", "ps_best_round",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["role", "role_rank"])

    for role in PITCHER_ROLES:
        role_df = result[result["role"] == role]
        if not role_df.empty:
            logger.info(
                "%s: %d ranked -- #1 %s (%.3f)",
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
    health_df: pd.DataFrame | None = None,
    pitcher_roles_df: pd.DataFrame | None = None,
    min_pa: int | None = None,
    min_bf: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all positional rankings.

    Parameters
    ----------
    season : int
        Most recent completed season for observed data.  During the
        in-season weekly refresh this equals ``projection_season``
        (e.g. both 2026).
    projection_season : int
        Target projection season.
    health_df : pd.DataFrame or None
        Pre-loaded health scores.  Passed through to both hitter and
        pitcher ranking functions to avoid disk reads.
    pitcher_roles_df : pd.DataFrame or None
        Pre-loaded reliever roles.  Passed through to pitcher ranking.
    min_pa : int or None
        Minimum PA for hitters.  Defaults to 100 (preseason) or 10
        when ``season == projection_season`` (in-season).
    min_bf : int or None
        Minimum BF for pitchers.  Defaults to 50 (preseason) or 10
        when ``season == projection_season`` (in-season).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'hitters', 'pitchers'. Each sorted by position/role then rank.
    """
    in_season = season == projection_season
    if min_pa is None:
        min_pa = 10 if in_season else 100
    if min_bf is None:
        min_bf = 10 if in_season else 50

    logger.info(
        "Building %d positional rankings (min_pa=%d, min_bf=%d)...",
        projection_season, min_pa, min_bf,
    )
    hitters = rank_hitters(
        season=season, projection_season=projection_season,
        min_pa=min_pa, health_df=health_df,
    )
    pitchers = rank_pitchers(
        season=season, projection_season=projection_season,
        min_bf=min_bf,
        health_df=health_df, pitcher_roles_df=pitcher_roles_df,
    )
    # --- Filter to rostered players only (active + IL) ---
    roster_path = DASHBOARD_DIR / "roster.parquet"
    if roster_path.exists():
        roster = pd.read_parquet(roster_path)
        valid_statuses = {"active", "il_7", "il_10", "il_15", "il_60"}
        rostered_ids = set(
            roster.loc[roster["roster_status"].isin(valid_statuses), "player_id"]
        )

        h_before = len(hitters)
        hitters = hitters[hitters["batter_id"].isin(rostered_ids)].copy()
        if len(hitters) < h_before:
            logger.info(
                "Roster filter: %d -> %d hitters (%d removed)",
                h_before, len(hitters), h_before - len(hitters),
            )
            hitters = hitters.sort_values("current_value_score", ascending=False)
            hitters["pos_rank"] = hitters.groupby("position").cumcount() + 1
            hitters["overall_rank"] = (
                hitters["current_value_score"]
                .rank(ascending=False, method="min", na_option="bottom")
                .fillna(len(hitters))
                .astype(int)
            )
            hitters["talent_rank"] = (
                hitters["talent_upside_score"]
                .rank(ascending=False, method="min", na_option="bottom")
                .fillna(len(hitters))
                .astype(int)
            )
            hitters = hitters.sort_values(["position", "pos_rank"])

        p_before = len(pitchers)
        pitchers = pitchers[pitchers["pitcher_id"].isin(rostered_ids)].copy()
        if len(pitchers) < p_before:
            logger.info(
                "Roster filter: %d -> %d pitchers (%d removed)",
                p_before, len(pitchers), p_before - len(pitchers),
            )
            pitchers = pitchers.sort_values("current_value_score", ascending=False)
            pitchers["role_rank"] = pitchers.groupby("role").cumcount() + 1
            pitchers["overall_rank"] = (
                pitchers["current_value_score"]
                .rank(ascending=False, method="min", na_option="bottom")
                .fillna(len(pitchers))
                .astype(int)
            )
            pitchers["talent_rank"] = (
                pitchers["talent_upside_score"]
                .rank(ascending=False, method="min", na_option="bottom")
                .fillna(len(pitchers))
                .astype(int)
            )
            pitchers = pitchers.sort_values(["role", "role_rank"])
    else:
        logger.warning("roster.parquet not found -- skipping roster filter")

    logger.info(
        "Rankings complete: %d hitters, %d pitchers",
        len(hitters), len(pitchers),
    )
    return {"hitters": hitters, "pitchers": pitchers}
