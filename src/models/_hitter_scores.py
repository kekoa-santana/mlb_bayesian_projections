"""Hitter sub-score builders for the player rankings system.

Extracted from ``player_rankings.py`` -- zero behavioral changes.

Each ``_build_hitter_*`` function computes one component of the hitter
composite ranking score.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models._ranking_utils import (
    DASHBOARD_DIR,
    _POS_TIER,
    _OBS_WEIGHT_BASE,
    _hitter_age_factor,
    _inv_pctl,
    _inv_zscore_pctl,
    _pctl,
    _stat_family_trust,
    _zscore_pctl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confirmed breakout score
# ---------------------------------------------------------------------------

def _build_confirmed_breakout_score(season: int = 2025) -> pd.DataFrame:
    """Score players whose recent performance exceeds their baseline AND
    is backed by quality metrics (not BABIP luck).

    Unlike the XGBoost breakout model (which predicts *future* breakouts),
    this captures players who *already* broke out or sustained elite
    production in the most recent season.  The Bayesian projected_woba
    compresses these players toward the mean (std=0.018 vs true-talent
    0.025); this score provides a correction signal.

    Components
    ----------
    - **Performance level** (25%): observed_woba vs league average.
      How good was the player, not just how much did they improve.
    - **Trajectory direction** (25%): slope of wOBA over the 2-3 seasons
      *before* the breakout year.  Ascending trajectory (Henderson, Witt)
      scores high; decline-then-spike (Springer) scores low.
    - **Statcast backing** (25%): xwOBA >= wOBA means the production is
      skill-backed, not BABIP luck.  Also incorporates barrel% and
      hard_hit% as quality indicators.
    - **Age credibility** (15%): young player's high performance is more
      sustainable.  A 24-year-old breakout is more believable than a
      35-year-old career-best.
    - **Improvement signal** (10%): observed_woba - career_woba.
      Positive = above career baseline.  Down-weighted because trajectory
      direction captures the development arc more precisely.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, confirmed_breakout_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Load prior-season observed stats with xwOBA/barrel% for backing
    try:
        from src.data.db import read_sql
        obs_adv = read_sql("""
            SELECT batter_id, woba, xwoba, barrel_pct, hard_hit_pct, pa
            FROM production.fact_batting_advanced
            WHERE season = :season AND pa >= 100
        """, {"season": season})
    except Exception:
        logger.warning("Could not load fact_batting_advanced for confirmed breakout")
        obs_adv = pd.DataFrame(columns=["batter_id", "woba", "xwoba",
                                         "barrel_pct", "hard_hit_pct", "pa"])

    # Load multi-year wOBA history for trajectory direction
    try:
        from src.data.db import read_sql
        woba_history = read_sql("""
            SELECT batter_id, season, woba, pa
            FROM production.fact_batting_advanced
            WHERE season BETWEEN :start_season AND :end_season
              AND pa >= 50
        """, {"start_season": season - 4, "end_season": season})
    except Exception:
        logger.warning("Could not load wOBA history for trajectory direction")
        woba_history = pd.DataFrame(columns=["batter_id", "season", "woba", "pa"])

    df = proj[["batter_id", "observed_woba", "career_woba", "age"]].copy()
    df = df.merge(obs_adv, on="batter_id", how="left")

    # Fill missing with neutral values
    df["observed_woba"] = df["observed_woba"].fillna(0.315)
    df["career_woba"] = df["career_woba"].fillna(0.315)
    df["xwoba"] = df["xwoba"].fillna(df["woba"]).fillna(df["observed_woba"])
    df["woba"] = df["woba"].fillna(df["observed_woba"])
    df["barrel_pct"] = df["barrel_pct"].fillna(df["barrel_pct"].median())
    df["hard_hit_pct"] = df["hard_hit_pct"].fillna(df["hard_hit_pct"].median())
    age = df["age"].fillna(28)

    # --- 1. Performance level (25%): how good, not just improvement ---
    perf_level = _zscore_pctl(df["observed_woba"])

    # --- 2. Trajectory direction (25%): pre-breakout slope ---
    # Compute slope of wOBA over the 2-3 seasons BEFORE the breakout year.
    # Positive slope = ascending arc (real development).
    # Negative slope = decline-then-spike (suspicious outlier).
    traj_scores = _compute_trajectory_direction(
        df["batter_id"], woba_history, breakout_season=season,
    )
    df = df.merge(traj_scores, on="batter_id", how="left")
    trajectory_dir = df["trajectory_direction"].fillna(0.5)

    # --- 3. Statcast backing (25%): is the production real? ---
    xwoba_gap = df["xwoba"] - df["woba"]
    backing_xwoba = _zscore_pctl(xwoba_gap)
    backing_barrel = _pctl(df["barrel_pct"])
    backing_hh = _pctl(df["hard_hit_pct"])
    statcast_backing = 0.50 * backing_xwoba + 0.25 * backing_barrel + 0.25 * backing_hh

    # --- 4. Age credibility (15%) ---
    age_cred = _hitter_age_factor(age)

    # --- 5. Improvement signal (10%): above career baseline ---
    improvement = df["observed_woba"] - df["career_woba"]
    improvement_score = _zscore_pctl(improvement)

    # --- Composite ---
    df["confirmed_breakout_score"] = (
        0.25 * perf_level
        + 0.25 * trajectory_dir
        + 0.25 * statcast_backing
        + 0.15 * age_cred
        + 0.10 * improvement_score
    )

    return df[["batter_id", "confirmed_breakout_score"]]


def _compute_trajectory_direction(
    batter_ids: pd.Series,
    woba_history: pd.DataFrame,
    breakout_season: int,
) -> pd.DataFrame:
    """Compute pre-breakout wOBA slope for each player.

    Looks at the 2-4 seasons *before* the breakout year and fits a
    PA-weighted linear slope.  Maps slope to 0-1 via sigmoid:
    - Positive slope (ascending arc) -> high score (~0.75-0.90)
    - Flat slope -> neutral (~0.50)
    - Negative slope (decline-then-spike) -> low score (~0.10-0.25)

    Examples (2025 breakout season):
    - Witt:     2022 .306 → 2023 .342 → 2024 .397  slope=+.045/yr → ~0.88
    - Raleigh:  2022 .323 → 2023 .324 → 2024 .319  slope=-.002/yr → ~0.48
    - Springer: 2022 .346 → 2023 .317 → 2024 .296  slope=-.025/yr → ~0.15

    Parameters
    ----------
    batter_ids : pd.Series
        Player IDs to score.
    woba_history : pd.DataFrame
        Multi-year wOBA with columns: batter_id, season, woba, pa.
    breakout_season : int
        The season whose production is being evaluated.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, trajectory_direction (0-1 score).
    """
    # Only use seasons BEFORE the breakout year for slope calculation
    prior = woba_history[
        (woba_history["season"] < breakout_season)
        & (woba_history["batter_id"].isin(batter_ids))
    ].copy()

    results = {}
    for bid in batter_ids.unique():
        player_hist = prior[prior["batter_id"] == bid].sort_values("season")
        if len(player_hist) < 2:
            # Not enough history — neutral slope (maps to 0.50 via sigmoid)
            results[bid] = 0.0
            continue

        # Use most recent 3 seasons (or whatever is available)
        player_hist = player_hist.tail(3)

        # PA-weighted linear regression: slope of wOBA vs season
        seasons = player_hist["season"].values.astype(float)
        woba_vals = player_hist["woba"].values.astype(float)
        pa_weights = player_hist["pa"].values.astype(float)
        pa_weights = pa_weights / pa_weights.sum()

        # Weighted mean
        s_mean = np.average(seasons, weights=pa_weights)
        w_mean = np.average(woba_vals, weights=pa_weights)

        # Weighted slope = sum(w * (s - s_mean) * (woba - w_mean)) / sum(w * (s - s_mean)^2)
        denom = np.sum(pa_weights * (seasons - s_mean) ** 2)
        if denom < 1e-10:
            results[bid] = 0.0  # neutral slope
            continue
        slope = np.sum(pa_weights * (seasons - s_mean) * (woba_vals - w_mean)) / denom
        results[bid] = slope

    slope_series = pd.Series(results)

    # Map slope to 0-1 via sigmoid.
    # Scale factor: slope of ±0.030/yr maps to ~0.85/0.15 (strong signal).
    # A slope of +0.045 (Witt) → ~0.90, -0.025 (Springer) → ~0.15.
    k = 40.0  # steepness: 1/(1+exp(-40*0.030)) ≈ 0.77, good spread
    direction_score = 1.0 / (1.0 + np.exp(-k * slope_series))

    return pd.DataFrame({
        "batter_id": slope_series.index,
        "trajectory_direction": direction_score.values,
    })


def _build_hitter_offense_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    aggressiveness: pd.DataFrame | None = None,
    use_sim_wrc_plus: bool = True,
    il_player_ids: set[int] | None = None,
    xwoba_talent: pd.DataFrame | None = None,
    baserunning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Production-anchored hitter offense with soft pathway credit.

    Replaces the prior flat five-skill sum which linearly added inputs
    (contact, decision) and outputs (damage, production) -- a structure
    that penalized specialists by giving each input skill its own full
    weight slot even when the player had already produced elite outcomes
    via a different route.

    New composition:

    1. **Production (60%)**: wRC+ bottom-line run creation.  The outcome
       anchor.  Two hitters with the same wRC+ get equal production
       credit regardless of pathway.
    2. **Soft pathway (25%)**: 0.66 * max(damage, contact) + 0.34 * mean.
       Symmetric between the two routes to value -- neither power-first
       nor contact-first is preferred.  A specialist gets ~90% of the
       credit a balanced hitter earns at the same peak level; a hitter
       strong on both gets a small bonus for having two routes.
    3. **Decision (10%)**: BB% + chase + 2-strike whiff.  Standalone
       because plate discipline adds independent value -- a patient
       slugger beats an impatient slugger at the same damage level.
    4. **Speed (5%)**: Offensive speed bonus (infield hits, BABIP boost).
       Baserunning between bases is credited separately in the composite
       via baserunning_score at its own weight.

    The function also exposes storytelling columns (pathway_lead and
    offense_archetype) describing each player's primary route to value.

    Parameters
    ----------
    proj : pd.DataFrame
        Hitter projections (from dashboard parquet).
    observed : pd.DataFrame
        Observed season stats from ``fact_batting_advanced``.
    aggressiveness : pd.DataFrame, optional
        From ``get_hitter_aggressiveness()`` -- chase_rate, two_strike_whiff_rate.
    use_sim_wrc_plus : bool
        If False, do not load ``hitter_counting_sim.parquet`` for the damage /
        production buckets (avoids stale preseason sim during in-season
        rankings when ``season == projection_season``).
    xwoba_talent : pd.DataFrame, optional
        Unused — retained for call-site compatibility.  The xwOBA talent
        anchor has been removed in favor of the flat skill composite.
    baserunning : pd.DataFrame, optional
        Pre-computed baserunning scores.  Columns: batter_id, baserunning_score.
        When provided, speed is included as 10% of offense.

    Returns
    -------
    pd.DataFrame
        batter_id + offense_score + sub-bucket columns.
    """
    # Merge projection + observed on batter_id
    obs_cols = ["batter_id", "pa", "k_pct", "bb_pct", "woba", "wrc_plus",
                "barrel_pct", "hard_hit_pct", "xwoba", "sweet_spot_pct",
                # Multi-year recency-weighted columns (when available)
                "wrc_plus_multiyear", "xwoba_multiyear",
                "barrel_pct_multiyear", "hard_hit_pct_multiyear"]
    obs_cols = [c for c in obs_cols if c in observed.columns]
    proj_cols = ["batter_id", "projected_k_rate", "projected_bb_rate",
                  "projected_hr_per_fb", "projected_gb_rate", "projected_woba",
                  "career_woba", "composite_score"]
    # Include prior-season PA for career reliability weighting
    if "pa" in proj.columns:
        proj_cols.append("pa")  # will be renamed to _proj_pa below
    if "age" in proj.columns:
        proj_cols.append("age")
    proj_sub = proj[proj_cols].drop_duplicates("batter_id", keep="first")
    # Rename projections PA to avoid collision with observed PA
    if "pa" in proj_sub.columns:
        proj_sub = proj_sub.rename(columns={"pa": "_proj_pa"})
    obs_sub = observed[obs_cols].drop_duplicates("batter_id", keep="first")
    merged = proj_sub.merge(
        obs_sub,
        on="batter_id", how="left",
    )

    if merged.empty:
        return pd.DataFrame(columns=["batter_id", "offense_score",
                                      "contact_skill", "decision_skill",
                                      "damage_skill", "production_skill"])

    pa = merged["pa"].fillna(0)

    # Merge aggressiveness (chase, 2-strike whiff) if provided
    if aggressiveness is not None and not aggressiveness.empty:
        agg_cols = ["batter_id"]
        for c in ("chase_rate", "two_strike_whiff_rate"):
            if c in aggressiveness.columns:
                agg_cols.append(c)
        if len(agg_cols) > 1:
            agg_sub = aggressiveness[agg_cols].drop_duplicates("batter_id", keep="first")
            merged = merged.merge(agg_sub, on="batter_id", how="left")

    # Optionally load sim wRC+ for damage bucket
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    has_sim = False
    if use_sim_wrc_plus and sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        if "projected_wrc_plus_mean" in sim_df.columns:
            sim_df = sim_df[["batter_id", "projected_wrc_plus_mean"]].drop_duplicates(
                "batter_id", keep="first",
            )
            merged = merged.merge(sim_df, on="batter_id", how="left")
            has_sim = merged["projected_wrc_plus_mean"].notna().any()
            if has_sim:
                logger.info("Offense using sim wRC+ for damage bucket (%d hitters)",
                            merged["projected_wrc_plus_mean"].notna().sum())
    elif not use_sim_wrc_plus:
        logger.info(
            "Skipping hitter_counting_sim wRC+ (in-season / stale-sim guard)",
        )

    # =================================================================
    # Bucket 1: CONTACT SKILL (stabilizes ~150 PA)
    # Projected K% is the most stable hitter stat (r=0.80 YoY).
    # Observed K% confirms or challenges the projection quickly.
    # =================================================================
    proj_contact = _inv_zscore_pctl(merged["projected_k_rate"])
    obs_k = _inv_zscore_pctl(merged["k_pct"]) if "k_pct" in merged.columns else proj_contact
    obs_k = obs_k.fillna(proj_contact)  # NaN-safe: left-joined players use projection

    contact_trust = _stat_family_trust(pa, min_pa=60, full_pa=350)
    merged["contact_skill"] = (1 - contact_trust) * proj_contact + contact_trust * obs_k

    # =================================================================
    # Bucket 2: SWING DECISIONS (stabilizes ~200 PA)
    # BB% projection + observed chase/discipline metrics.
    # Chase rate is extremely stable (r=0.84 YoY).
    # =================================================================
    proj_decisions = _zscore_pctl(merged["projected_bb_rate"])

    obs_decision_parts = [proj_decisions]  # fallback: just projection
    obs_decision_weights = [1.0]

    if "chase_rate" in merged.columns:
        obs_chase = _inv_zscore_pctl(merged["chase_rate"].fillna(
            merged["chase_rate"].median()))
        obs_decision_parts = [obs_chase]
        obs_decision_weights = [0.55]
        if "two_strike_whiff_rate" in merged.columns:
            obs_2s = _inv_zscore_pctl(merged["two_strike_whiff_rate"].fillna(
                merged["two_strike_whiff_rate"].median()))
            obs_decision_parts.append(obs_2s)
            obs_decision_weights.append(0.45)

    obs_decisions = sum(w * p for w, p in zip(obs_decision_weights, obs_decision_parts))

    decision_trust = _stat_family_trust(pa, min_pa=100, full_pa=450)
    merged["decision_skill"] = (1 - decision_trust) * proj_decisions + decision_trust * obs_decisions

    # =================================================================
    # Bucket 3: DAMAGE ON CONTACT (stabilizes ~300 PA)
    # xwOBA + barrel% + hard_hit% + sweet_spot%.  Noisier metrics
    # that need more BIP to stabilize.  Sim wRC+ serves as projected
    # anchor when available; falls back to HR/FB projection.
    # Removed xBA and xSLG (redundant with xwOBA, ~80% shared variance).
    # =================================================================
    if has_sim:
        proj_damage = _zscore_pctl(merged["projected_wrc_plus_mean"].fillna(100))
    else:
        # Without sim wRC+, use Bayesian projected wOBA.  Age-dependent
        # rho in hitter_model.py handles development vs aging regression.
        proj_damage = _zscore_pctl(merged["projected_woba"].fillna(0.310))

    # Use multi-year recency-weighted Statcast when available (smooths
    # outlier single seasons like Springer's 2025 career-best at 35).
    # Falls back to single-season when multi-year data is absent.
    _xwoba_col = "xwoba_multiyear" if "xwoba_multiyear" in merged.columns else "xwoba"
    _barrel_col = "barrel_pct_multiyear" if "barrel_pct_multiyear" in merged.columns else "barrel_pct"
    _hardhit_col = "hard_hit_pct_multiyear" if "hard_hit_pct_multiyear" in merged.columns else "hard_hit_pct"

    obs_damage_parts = []
    obs_damage_weights = []
    if _xwoba_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_xwoba_col].fillna(merged[_xwoba_col].median())))
        obs_damage_weights.append(0.35)
    if _barrel_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_barrel_col].fillna(0)))
        obs_damage_weights.append(0.30)
    if _hardhit_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_hardhit_col].fillna(0)))
        obs_damage_weights.append(0.20)
    if "sweet_spot_pct" in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged["sweet_spot_pct"].fillna(0)))
        obs_damage_weights.append(0.15)

    if obs_damage_parts:
        # Renormalize weights in case some columns are missing
        total_w = sum(obs_damage_weights)
        obs_damage = sum(
            (w / total_w) * p for w, p in zip(obs_damage_weights, obs_damage_parts)
        )
    else:
        obs_damage = proj_damage

    # Damage trust: PA-based ramp only.  Multi-year recency-weighted
    # Statcast columns already smooth outlier seasons; age regression
    # is handled by trajectory_score in the overall composite.
    damage_trust = _stat_family_trust(pa, min_pa=150, full_pa=550)
    merged["damage_skill"] = (1 - damage_trust) * proj_damage + damage_trust * obs_damage

    # =================================================================
    # Bucket 4: PRODUCTION (stabilizes ~250 PA)
    # Observed wRC+ -- the bottom line.  Anchors the score to actual
    # run production so that high-process / low-results players
    # (e.g. low-K% / low-chase but .280 wOBA) cannot score elite.
    # Projected component uses sim wRC+ when available.
    # =================================================================
    if has_sim:
        proj_production = _zscore_pctl(merged["projected_wrc_plus_mean"].fillna(100))
    else:
        # Without sim wRC+, use Bayesian projected wOBA.
        proj_production = _zscore_pctl(merged["projected_woba"].fillna(0.310))

    # Use multi-year recency-weighted wRC+ when available (same smoothing
    # as damage bucket -- prevents single-season outliers from dominating).
    _wrc_col = "wrc_plus_multiyear" if "wrc_plus_multiyear" in merged.columns else "wrc_plus"
    if _wrc_col in merged.columns:
        obs_production = _zscore_pctl(merged[_wrc_col].fillna(100))
    else:
        obs_production = proj_production

    production_trust = _stat_family_trust(pa, min_pa=100, full_pa=500)

    merged["production_skill"] = (
        (1 - production_trust) * proj_production
        + production_trust * obs_production
    )

    # =================================================================
    # Combine: production-anchored with soft pathway credit.
    #
    # Production (60%): wRC+ bottom line -- the outcome.  Two hitters
    #   who produce the same runs get equal anchor value regardless of
    #   HOW they produce.
    # Pathway (25%): soft-max of damage_skill and contact_skill.  Takes
    #   0.66 * max + 0.34 * mean so a pure specialist gets ~90% of the
    #   credit of a balanced hitter at the same peak level, and a hitter
    #   with both elite skills gets a small bonus for having two routes.
    #   Neither route is preferred; they are symmetric alternatives.
    # Decision (10%): BB% + chase + 2K-whiff.  Standalone because patience
    #   adds value beyond both pathways -- a patient slugger beats an
    #   impatient slugger at the same damage level.
    # Speed (5%): residual offensive speed bonus.  Baserunning value
    #   between bases is already credited separately in the composite;
    #   this small weight captures infield-hit / BABIP boost that
    #   production_skill partially misses.
    #
    # Design replaces the prior flat five-skill sum which penalized
    # specialists (e.g. power-first Caminero) by giving contact / decision
    # their own full-value slots even when damage alone had already
    # produced elite outcomes.
    # =================================================================
    _W_PRODUCTION = 0.60
    _W_PATHWAY = 0.25
    _W_DECISION = 0.10
    _W_SPEED = 0.05
    _PATHWAY_MAX_MIX = 0.66  # 0.66*max + 0.34*mean; higher = more specialist-friendly

    # Merge baserunning if provided
    if baserunning is not None and not baserunning.empty:
        br_cols = ["batter_id", "baserunning_score"]
        br_cols = [c for c in br_cols if c in baserunning.columns]
        if "baserunning_score" in baserunning.columns:
            br_sub = baserunning[br_cols].drop_duplicates("batter_id", keep="first")
            merged = merged.merge(br_sub, on="batter_id", how="left")

    _speed = merged.get("baserunning_score", pd.Series(0.5, index=merged.index)).fillna(0.5)

    # Soft-pathway: best of damage-or-contact with a mean tempering term.
    # Symmetric between the two routes -- neither power nor contact is
    # structurally preferred.  A "teetering on both" hitter who has damage
    # 0.75 and contact 0.70 gets ~0.74 (slightly above their max), which
    # is the small bonus for having two genuine pathways.
    _pathway_max = np.maximum(merged["damage_skill"], merged["contact_skill"])
    _pathway_mean = (merged["damage_skill"] + merged["contact_skill"]) / 2.0
    merged["pathway_score"] = _PATHWAY_MAX_MIX * _pathway_max + (1 - _PATHWAY_MAX_MIX) * _pathway_mean

    # Storytelling: which route is the player leaning on, and by how much.
    # pathway_lead: "damage" if power-first, "contact" if contact-first.
    # offense_archetype: "power-first" / "contact-first" / "balanced"
    #   based on the gap between the two skills.
    _pathway_gap = merged["damage_skill"] - merged["contact_skill"]
    merged["pathway_lead"] = np.where(_pathway_gap >= 0, "damage", "contact")
    merged["offense_archetype"] = np.where(
        _pathway_gap > 0.15, "power-first",
        np.where(_pathway_gap < -0.15, "contact-first", "balanced"),
    )

    merged["offense_score"] = (
        _W_PRODUCTION * merged["production_skill"]
        + _W_PATHWAY * merged["pathway_score"]
        + _W_DECISION * merged["decision_skill"]
        + _W_SPEED * _speed
    )

    # Dampening for true unknowns (no Statcast history, no current PA).
    # Trust ramps in individual buckets already handle small samples;
    # this catches players with zero data at all.
    pa_safe = pa.fillna(0)
    _is_unknown = pa_safe < 50
    dampening = np.where(_is_unknown, 0.20, 0.0)
    merged["offense_score"] = merged["offense_score"] * (1 - dampening) + 0.50 * dampening

    return merged[["batter_id", "offense_score", "contact_skill",
                    "decision_skill", "damage_skill", "production_skill",
                    "pathway_score", "pathway_lead", "offense_archetype"]]


def _build_hitter_baserunning_score(season: int = 2025) -> pd.DataFrame:
    """Score baserunning from speed, hustle, SB volume, efficiency, and utilization.

    Components
    ----------
    - **Sprint speed** (25%): raw foot speed from Statcast
    - **HP to 1B hustle** (15%): home-to-first time (faster = better).
      Captures effort on routine plays beyond raw speed.
    - **SB volume** (25%): projected SB per PA -- aggressive baserunning.
    - **SB efficiency** (15%): success rate regressed toward league avg (~78%).
      Smart baserunning -- not just fast, but picks good spots.
    - **Speed utilization** (20%): SB rate relative to sprint-speed expectation.
      A player who steals more than their speed predicts gets credit for
      baserunning IQ and aggressiveness beyond raw ability.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, baserunning_score.
    """
    from src.data.db import read_sql

    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Sprint speed + HP-to-1B from Statcast.  3-year window so we pick up
    # data even when current season hasn't published yet (Savant releases
    # sprint speed mid-season).  DISTINCT ON + ORDER BY season DESC prefers
    # the most recent available year per player.
    speed_df = read_sql(f"""
        SELECT DISTINCT ON (player_id)
               player_id AS batter_id, sprint_speed, hp_to_1b
        FROM staging.statcast_sprint_speed
        WHERE season BETWEEN {season - 2} AND {season}
        ORDER BY player_id, season DESC
    """, {})

    # Observed SB/CS (2-year window for stability)
    sb_obs = read_sql(f"""
        SELECT pg.player_id AS batter_id,
               SUM(pg.bat_sb) AS obs_sb, SUM(pg.bat_cs) AS obs_cs
        FROM production.fact_player_game_mlb pg
        JOIN production.dim_game dg ON pg.game_pk = dg.game_pk
        WHERE dg.season BETWEEN {season - 1} AND {season}
          AND dg.game_type = 'R'
        GROUP BY pg.player_id
    """, {})

    # Projected SB from sim
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "hitter_counting.parquet"
    if sim_path.exists():
        counting = pd.read_parquet(sim_path)
        sb_col = "total_sb_mean"
        pa_col = "total_pa_mean" if "total_pa_mean" in counting.columns else "projected_pa_mean"
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        sb_col = "total_sb_mean"
        pa_col = "projected_pa_mean"
    else:
        counting = pd.DataFrame()

    # Assemble
    merged = proj[["batter_id", "sprint_speed"]].copy()
    merged = merged.merge(speed_df, on="batter_id", how="left", suffixes=("_proj", ""))
    # Prefer Statcast sprint_speed over projection parquet
    if "sprint_speed_proj" in merged.columns:
        merged["sprint_speed"] = merged["sprint_speed"].fillna(merged["sprint_speed_proj"])
        merged.drop(columns=["sprint_speed_proj"], inplace=True)
    merged = merged.merge(sb_obs, on="batter_id", how="left")
    if not counting.empty:
        merged = merged.merge(
            counting[["batter_id", sb_col, pa_col]], on="batter_id", how="left",
        )

    if merged.empty:
        return pd.DataFrame(columns=["batter_id", "baserunning_score"])

    # --- Component 1: Sprint speed (25%) ---
    speed_med = merged["sprint_speed"].median()
    speed_score = _pctl(merged["sprint_speed"].fillna(speed_med))

    # --- Component 2: HP-to-1B hustle (15%) ---
    # Lower = faster = better.  Captures effort beyond raw speed.
    hp1b_med = merged["hp_to_1b"].median() if "hp_to_1b" in merged.columns else 4.45
    hp1b_score = _inv_pctl(merged["hp_to_1b"].fillna(hp1b_med))

    # --- Component 3: SB volume rate (25%) ---
    if sb_col in merged.columns and pa_col in merged.columns:
        sb_rate = merged[sb_col] / merged[pa_col].clip(lower=1)
    else:
        sb_rate = merged["obs_sb"].fillna(0) / 500  # rough fallback
    sb_volume_score = _pctl(sb_rate)

    # --- Component 4: SB efficiency (15%) ---
    # Regress toward league average (~78%) using Bayesian shrinkage.
    #
    # Gate: only percentile-rank among players with >= 10 attempts.
    # Non-stealers with 1-2 attempts regress to ~78% and flood the
    # distribution at the league mean, causing high-volume stealers
    # with honest 76-77% rates to score worse than non-stealers.
    # Witt (76.8% on 90 attempts) was scoring 14th percentile because
    # 200+ non-stealers at exactly 78% crowded him out.
    _LEAGUE_SB_SUCCESS = 0.78
    _SB_REGRESS_N = 10
    obs_sb = merged["obs_sb"].fillna(0)
    obs_cs = merged["obs_cs"].fillna(0)
    obs_att = obs_sb + obs_cs
    regressed_success = (
        (obs_sb + _SB_REGRESS_N * _LEAGUE_SB_SUCCESS)
        / (obs_att + _SB_REGRESS_N)
    )
    has_attempts = obs_att >= 10
    # Percentile-rank only among players with meaningful volume
    qualified_mask = has_attempts.values
    sb_eff_score = pd.Series(0.50, index=merged.index)
    if qualified_mask.sum() >= 5:
        sb_eff_score.loc[qualified_mask] = _zscore_pctl(
            regressed_success[qualified_mask]
        ).values

    # --- Component 5: Speed utilization (20%) ---
    # How much does this player steal relative to their sprint speed?
    # Fit linear model: expected SB rate = f(sprint_speed)
    valid = merged[merged["sprint_speed"].notna() & (sb_rate > 0)].copy()
    if len(valid) >= 20:
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(valid["sprint_speed"].values, sb_rate[valid.index].values, 1)
        expected_sb_rate = P.polyval(merged["sprint_speed"].fillna(speed_med).values, coef)
        sb_over_expected = sb_rate - pd.Series(expected_sb_rate, index=merged.index)
        utilization_score = _pctl(sb_over_expected)
    else:
        utilization_score = pd.Series(0.50, index=merged.index)

    merged["baserunning_score"] = (
        0.25 * speed_score
        + 0.15 * hp1b_score
        + 0.25 * sb_volume_score
        + 0.15 * pd.Series(sb_eff_score, index=merged.index)
        + 0.20 * utilization_score
    )

    return merged[["batter_id", "baserunning_score"]]


def _build_hitter_platoon_modifier(season: int = 2025, min_pa_side: int = 30) -> pd.DataFrame:
    """Score platoon balance with regressed splits and exposure weighting.

    Uses 3-year PA-weighted platoon data with handedness-specific regression:
    - LHH: regress toward league avg OPS with 1000 PA shrinkage
    - RHH: regress toward league avg OPS with 2200 PA shrinkage
    Hitters do NOT learn to close platoon splits over time (research consensus).

    Exposure-weighted: weak vs LHP (28% of games) penalized less than
    weak vs RHP (72% of games).

    Parameters
    ----------
    season : int
        Most recent season for platoon data.
    min_pa_side : int
        Minimum total PA vs each side across all years to be scored.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, platoon_score (0-1, 1 = balanced).
    """
    from src.data.db import read_sql

    # 3-year window for stability
    splits = read_sql(f"""
        SELECT player_id, platoon_side, season, pa, ops
        FROM production.fact_platoon_splits
        WHERE season BETWEEN {season - 2} AND {season}
          AND player_role = 'batter'
    """, {})

    if splits.empty:
        return pd.DataFrame(columns=["player_id", "platoon_score"])

    # Get batter handedness for regression calibration
    handedness = read_sql("""
        SELECT player_id, COALESCE(bat_side, 'R') AS bat_side
        FROM production.dim_player
    """, {})

    # PA-weighted average OPS per player per side (across years)
    def _pa_weighted_ops(g: pd.DataFrame) -> pd.Series:
        total_pa = g["pa"].sum()
        if total_pa == 0:
            return pd.Series({"ops": np.nan, "pa": 0})
        return pd.Series({
            "ops": np.average(g["ops"], weights=g["pa"]),
            "pa": total_pa,
        })

    vlh = (
        splits[splits["platoon_side"] == "vLH"]
        .groupby("player_id")
        .apply(_pa_weighted_ops, include_groups=False)
        .rename(columns={"ops": "ops_vlh", "pa": "pa_vlh"})
    )
    vrh = (
        splits[splits["platoon_side"] == "vRH"]
        .groupby("player_id")
        .apply(_pa_weighted_ops, include_groups=False)
        .rename(columns={"ops": "ops_vrh", "pa": "pa_vrh"})
    )

    merged = vlh.join(vrh, how="inner").reset_index()
    merged = merged.merge(handedness, on="player_id", how="left")
    merged["bat_side"] = merged["bat_side"].fillna("R")

    # Only score players with enough PA on both sides
    has_both = (merged["pa_vlh"] >= min_pa_side) & (merged["pa_vrh"] >= min_pa_side)
    scored = merged[has_both].copy()

    if scored.empty:
        return pd.DataFrame(columns=["player_id", "platoon_score"])

    # League average OPS (~.710-.730 range)
    lg_avg_ops = 0.720

    # Handedness-specific regression (Tango research):
    # LHH splits more variable -> less PA needed -> shrinkage = 1000
    # RHH splits more stable -> need more data to trust extreme -> shrinkage = 2200
    shrinkage = np.where(scored["bat_side"] == "L", 1000, 2200)

    # Regress each side toward league average
    scored["reg_ops_vlh"] = (
        scored["ops_vlh"] * scored["pa_vlh"] + lg_avg_ops * shrinkage
    ) / (scored["pa_vlh"] + shrinkage)
    scored["reg_ops_vrh"] = (
        scored["ops_vrh"] * scored["pa_vrh"] + lg_avg_ops * shrinkage
    ) / (scored["pa_vrh"] + shrinkage)

    # Regressed gap
    scored["ops_gap"] = (scored["reg_ops_vlh"] - scored["reg_ops_vrh"]).abs()

    # Exposure weighting: penalty is proportional to how often you face your weak side
    # LHP ~28% of games, RHP ~72%
    weak_vs_lhp = scored["reg_ops_vlh"] < scored["reg_ops_vrh"]
    # If weak vs LHP (28% exposure), gap impact is muted
    # If weak vs RHP (72% exposure), gap impact is amplified
    exposure = np.where(weak_vs_lhp, 0.28, 0.72)
    scored["adjusted_gap"] = scored["ops_gap"] * exposure * 2  # rescale to preserve range

    # Convert to 0-1 score: no gap = 1.0, 250+ adjusted gap = 0.0
    scored["platoon_score"] = (1.0 - scored["adjusted_gap"] / 0.250).clip(0, 1)

    # Players without enough PA get neutral
    neutral = merged[~has_both][["player_id"]].copy()
    neutral["platoon_score"] = 0.50

    result = pd.concat([scored[["player_id", "platoon_score"]], neutral], ignore_index=True)
    return result


def _build_hitter_fielding_score(
    season: int = 2025,
    in_season: bool = False,
) -> pd.DataFrame:
    """Build fielding score from multi-signal defensive composite.

    Blends four signals with position-aware weights:
    - **OAA** (primary): 3-year rolling, recency-weighted, regressed to
      position-specific prior.  IF k=3, OF k=2.
    - **Arm strength** (supplementary): ``arm_overall`` mph from Savant.
      Available 2020+.  Matters for OF (throwing out runners) and IF
      (turning DPs, preventing advances).
    - **Catch probability** (OF only): difficulty-adjusted catch rate from
      Savant star-level breakdowns.  Captures elite range beyond OAA.
    - **Sprint speed** (range proxy): ``sprint_speed`` ft/s from Statcast.
      Correlated with defensive range for both IF and OF.

    Position-group weights (when all signals available):
    - OF: OAA 50%, catch_prob 15%, arm 15%, speed 20%
    - IF: OAA 55%, arm 20%, speed 25%
    - 1B: OAA 65%, speed 35% (arm irrelevant)
    - C:  OAA-only (framing handled separately)

    When a signal is missing for a player, its weight is redistributed
    proportionally to available signals.

    Parameters
    ----------
    season : int
        Most recent season for defensive data.
    in_season : bool
        If True, exclude current season (too few games to be reliable)
        and use prior 3 full seasons instead.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, fielding_score.
    """
    from src.data.db import read_sql

    _IF_POSITIONS = {"SS", "2B", "3B", "1B"}
    _OF_POSITIONS = {"LF", "CF", "RF"}

    # ------------------------------------------------------------------
    # 1. OAA — primary signal (existing logic)
    # ------------------------------------------------------------------
    anchor = season - 1 if in_season else season
    oaa = read_sql(f"""
        SELECT player_id, season, position, outs_above_average
        FROM production.fact_fielding_oaa
        WHERE season BETWEEN {anchor - 2} AND {anchor}
    """, {})

    if oaa.empty:
        return pd.DataFrame(columns=["player_id", "fielding_score"])

    oaa["weight"] = oaa["season"].map(
        {anchor: 3, anchor - 1: 2, anchor - 2: 1}
    ).fillna(1)

    pos_mode = (
        oaa.groupby("player_id")["position"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "IF")
    )

    weighted = oaa.groupby("player_id").apply(
        lambda g: np.average(g["outs_above_average"], weights=g["weight"]),
        include_groups=False,
    ).rename("weighted_oaa")

    n_seasons = oaa.groupby("player_id")["season"].nunique().rename("n_seasons")

    result = pd.DataFrame({"weighted_oaa": weighted, "n_seasons": n_seasons})
    result["position_group"] = pos_mode

    result["k"] = result["position_group"].apply(
        lambda p: 2 if p in _OF_POSITIONS else 3
    )

    _POSITION_OAA_PRIOR = {
        "CF": 2.8, "SS": 1.1, "2B": 0.2, "3B": -0.5,
        "1B": -0.9, "RF": -1.1, "LF": -1.4,
    }
    result["pos_prior"] = result["position_group"].map(_POSITION_OAA_PRIOR).fillna(0.0)
    result["reliability"] = result["n_seasons"] / (result["n_seasons"] + result["k"])
    result["regressed_oaa"] = (
        result["reliability"] * result["weighted_oaa"]
        + (1 - result["reliability"]) * result["pos_prior"]
    )
    result["oaa_pctl"] = _pctl(result["regressed_oaa"]).clip(lower=0.10)

    result = result.reset_index()

    # ------------------------------------------------------------------
    # 2. Arm strength — recency-weighted (2-year window, data starts 2020)
    # ------------------------------------------------------------------
    arm = read_sql(f"""
        SELECT player_id, season, arm_overall
        FROM production.fact_arm_strength
        WHERE season BETWEEN {anchor - 1} AND {anchor}
          AND arm_overall IS NOT NULL
    """, {})

    if not arm.empty:
        arm["w"] = arm["season"].map({anchor: 2, anchor - 1: 1}).fillna(1)
        arm_avg = arm.groupby("player_id").apply(
            lambda g: np.average(g["arm_overall"], weights=g["w"]),
            include_groups=False,
        ).rename("arm_mph")
        result = result.merge(
            arm_avg.reset_index(), on="player_id", how="left"
        )
    if "arm_mph" not in result.columns:
        result["arm_mph"] = np.nan

    result["arm_pctl"] = _pctl(result["arm_mph"])

    # ------------------------------------------------------------------
    # 3. Catch probability — OF only, recency-weighted
    # ------------------------------------------------------------------
    cp = read_sql(f"""
        SELECT player_id, season, oaa AS cp_oaa,
               pct_5star, pct_4star
        FROM production.fact_catch_probability
        WHERE season BETWEEN {anchor - 1} AND {anchor}
    """, {})

    if not cp.empty:
        # Composite: 60% catch-prob OAA + 25% 5-star% + 15% 4-star%
        for col in ["pct_5star", "pct_4star"]:
            cp[col] = pd.to_numeric(cp[col], errors="coerce")
        cp["cp_composite"] = (
            0.60 * cp["cp_oaa"].fillna(0)
            + 0.25 * (cp["pct_5star"].fillna(0) / 100.0) * 30  # scale to OAA-like range
            + 0.15 * (cp["pct_4star"].fillna(0) / 100.0) * 20
        )
        cp["w"] = cp["season"].map({anchor: 2, anchor - 1: 1}).fillna(1)
        cp_avg = cp.groupby("player_id").apply(
            lambda g: np.average(g["cp_composite"], weights=g["w"]),
            include_groups=False,
        ).rename("cp_score")
        result = result.merge(
            cp_avg.reset_index(), on="player_id", how="left"
        )
    if "cp_score" not in result.columns:
        result["cp_score"] = np.nan

    result["cp_pctl"] = _pctl(result["cp_score"])

    # ------------------------------------------------------------------
    # 4. Sprint speed — range proxy, recency-weighted
    # ------------------------------------------------------------------
    ss = read_sql(f"""
        SELECT player_id, season, sprint_speed
        FROM staging.statcast_sprint_speed
        WHERE season BETWEEN {anchor - 1} AND {anchor}
          AND sprint_speed IS NOT NULL
    """, {})

    if not ss.empty:
        ss["w"] = ss["season"].map({anchor: 2, anchor - 1: 1}).fillna(1)
        ss_avg = ss.groupby("player_id").apply(
            lambda g: np.average(g["sprint_speed"], weights=g["w"]),
            include_groups=False,
        ).rename("speed_fps")
        result = result.merge(
            ss_avg.reset_index(), on="player_id", how="left"
        )
    if "speed_fps" not in result.columns:
        result["speed_fps"] = np.nan

    result["speed_pctl"] = _pctl(result["speed_fps"])

    # ------------------------------------------------------------------
    # 5. Position-aware composite with missing-data redistribution
    # ------------------------------------------------------------------
    # Target weights per position group (sum to 1.0)
    _WEIGHT_PROFILES = {
        "OF": {"oaa": 0.50, "arm": 0.15, "cp": 0.15, "speed": 0.20},
        "IF": {"oaa": 0.55, "arm": 0.20, "cp": 0.00, "speed": 0.25},
        "1B": {"oaa": 0.65, "arm": 0.00, "cp": 0.00, "speed": 0.35},
        "C":  {"oaa": 1.00, "arm": 0.00, "cp": 0.00, "speed": 0.00},
    }

    def _pos_group_key(pos: str) -> str:
        if pos in _OF_POSITIONS:
            return "OF"
        if pos == "1B":
            return "1B"
        if pos == "C":
            return "C"
        return "IF"

    scores = []
    for _, row in result.iterrows():
        profile = _WEIGHT_PROFILES[_pos_group_key(row["position_group"])]

        # Determine which signals are available for this player
        signals = {"oaa": row["oaa_pctl"]}  # OAA always available (we're in the OAA result set)
        if pd.notna(row["arm_mph"]) and profile["arm"] > 0:
            signals["arm"] = row["arm_pctl"]
        if pd.notna(row["cp_score"]) and profile["cp"] > 0:
            signals["cp"] = row["cp_pctl"]
        if pd.notna(row["speed_fps"]) and profile["speed"] > 0:
            signals["speed"] = row["speed_pctl"]

        # Redistribute missing weights proportionally to available signals
        total_avail = sum(profile[k] for k in signals)
        if total_avail > 0:
            composite = sum(
                (profile[k] / total_avail) * signals[k]
                for k in signals
            )
        else:
            composite = row["oaa_pctl"]

        scores.append(composite)

    result["fielding_score"] = scores

    n_with_arm = result["arm_mph"].notna().sum()
    n_with_cp = result["cp_score"].notna().sum()
    n_with_speed = result["speed_fps"].notna().sum()
    logger.info(
        f"Fielding composite: {len(result)} players, "
        f"{n_with_arm} with arm, {n_with_cp} with catch prob, "
        f"{n_with_speed} with speed"
    )

    return result[["player_id", "fielding_score"]]


def _build_catcher_framing_score(
    season: int = 2025,
    in_season: bool = False,
) -> pd.DataFrame:
    """Build catcher framing score from multi-year framing data.

    Uses 3-year rolling with recency weights (3/2/1) and regression
    toward league average, matching the pattern in fielding.  Single-season
    framing is noisy (Will Smith: +9.1 in 2021, -7.6 in 2025).

    Parameters
    ----------
    season : int
        Most recent season for framing data.
    in_season : bool
        If True, exclude current season framing (too few games) and use
        prior 3 full seasons instead.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, framing_score.
    """
    from src.data.db import read_sql

    anchor = season - 1 if in_season else season
    framing = read_sql(f"""
        SELECT player_id, season, runs_extra_strikes
        FROM production.fact_catcher_framing
        WHERE season BETWEEN {anchor - 2} AND {anchor}
    """, {})

    if framing.empty:
        return pd.DataFrame(columns=["player_id", "framing_score"])

    # Recency weights
    framing["weight"] = framing["season"].map(
        {anchor: 3, anchor - 1: 2, anchor - 2: 1}
    ).fillna(1)

    # Weighted average per catcher
    weighted = framing.groupby("player_id").apply(
        lambda g: np.average(g["runs_extra_strikes"], weights=g["weight"]),
        include_groups=False,
    ).rename("weighted_framing")

    n_seasons = framing.groupby("player_id")["season"].nunique().rename("n_seasons")
    result = pd.DataFrame({"weighted_framing": weighted, "n_seasons": n_seasons})

    # Regress toward 0 (league-average framing) based on sample size
    # k=2: 1 year -> 33% trust, 2 years -> 50%, 3 years -> 60%
    k = 2
    result["reliability"] = result["n_seasons"] / (result["n_seasons"] + k)
    result["regressed_framing"] = result["weighted_framing"] * result["reliability"]

    result["framing_score"] = _pctl(result["regressed_framing"])
    return result.reset_index()[["player_id", "framing_score"]]


def _build_hitter_playing_time_score(
    health_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score projected playing time from counting projections + health.

    Prefers sim-based ``hitter_counting_sim.parquet`` (PA + games from
    game sim). Falls back to old ``hitter_counting.parquet``.

    Parameters
    ----------
    health_df : pd.DataFrame or None
        Pre-loaded health scores (columns: player_id, health_score,
        health_label).  When provided the disk read of
        ``health_scores.parquet`` is skipped.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, pt_score, health_score, health_label.
    """
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "hitter_counting.parquet"

    if sim_path.exists():
        counting = pd.read_parquet(sim_path)
        pa_col = "total_pa_mean" if "total_pa_mean" in counting.columns else "projected_pa_mean"
        games_col = "total_games_mean"
        if pa_col in counting.columns and games_col in counting.columns:
            pa_pctl = _pctl(counting[pa_col].fillna(0))
            games_pctl = _pctl(counting[games_col].fillna(0))
            base_pt = 0.60 * pa_pctl + 0.40 * games_pctl
        else:
            base_pt = _pctl(counting[pa_col].fillna(0))
        logger.info("Playing time score from sim parquet: %d hitters", len(counting))
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        base_pt = _pctl(counting["projected_pa_mean"])
    else:
        return pd.DataFrame(columns=["batter_id", "pt_score"])

    # Rename for downstream compatibility
    if "projected_pa_mean" not in counting.columns and "total_pa_mean" in counting.columns:
        counting["projected_pa_mean"] = counting["total_pa_mean"]

    # Health scores: prefer columns already on counting parquet,
    # then in-memory health_df, then disk fallback
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        if health_df is None:
            health_path = DASHBOARD_DIR / "health_scores.parquet"
            if health_path.exists():
                health_df = pd.read_parquet(health_path)
        if health_df is not None and not health_df.empty:
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
        counting["pt_score"] = 0.70 * base_pt + 0.30 * health_pctl
    else:
        counting["pt_score"] = base_pt
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["batter_id", "pt_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_hitter_trajectory_score() -> pd.DataFrame:
    """Score trajectory using age, observed improvement, certainty, and trend.

    Redesigned 2026-04-06 to fix two systemic issues:
    1. Certainty at 55% rewarded proven veterans and penalized young
       breakout players (Henderson, J-Rod) whose wide posteriors are a
       *feature* (room to grow), not a flaw.
    2. Trend used Bayesian projected delta (projected - observed within
       same season), which is circular and misses actual YoY improvement.

    New components
    --------------
    - **Age factor** (50%): forward-looking trajectory curve.  Youth (20-24)
      gets upside credit, prime (25-29) holds flat, 30+ declines steeply.
      Dominant weight because age is the single strongest predictor of
      future value trajectory.
    - **YoY observed improvement** (20%): did the player's actual wOBA/wRC+
      improve from prior year?  Captures breakouts and declines.
    - **Projection certainty** (15%): lower CV = more proven.
    - **Season trend** (15%): Bayesian projected delta signal (tiebreaker).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, trajectory_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # --- Age factor (35%): research-aligned curve ---
    age = proj["age"].fillna(30)
    age_factor = _hitter_age_factor(age)

    # --- YoY observed improvement (20%) ---
    # Compare most recent observed season to prior career baseline.
    # Positive = player is currently above their career baseline = improving.
    #
    # For young players (age <= 25), use a higher career baseline floor
    # (0.300 vs 0.315) so developmental seasons don't artificially depress
    # the gap.  Young players with low career wOBA due to limited MLB time
    # were getting penalized: their "improvement" looked small even when
    # ascending, because the career baseline included partial/bad early
    # seasons.
    if "observed_woba" in proj.columns and "career_woba" in proj.columns:
        _obs = proj["observed_woba"].fillna(proj["career_woba"])
        _career = proj["career_woba"].fillna(0.315)
        yoy_improvement = (_obs - _career).fillna(0.0)
    elif "delta_woba" in proj.columns:
        yoy_improvement = -proj["delta_woba"].fillna(0)  # negative delta = improving
    else:
        yoy_improvement = pd.Series(0.0, index=proj.index)
    yoy_score = _zscore_pctl(yoy_improvement)

    # --- Projection certainty (20%): CV-based ---
    k_mean = proj["projected_k_rate"].clip(0.01)
    bb_mean = proj["projected_bb_rate"].clip(0.01)
    k_cv = proj["projected_k_rate_sd"].fillna(k_mean.median() * 0.15) / k_mean
    bb_cv = proj["projected_bb_rate_sd"].fillna(bb_mean.median() * 0.15) / bb_mean
    k_certainty = _inv_pctl(k_cv)
    bb_certainty = _inv_pctl(bb_cv)
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # --- Season trend (15%): Bayesian projected delta ---
    delta_k = proj.get("delta_k_rate", pd.Series(0.0, index=proj.index))
    delta_bb = proj.get("delta_bb_rate", pd.Series(0.0, index=proj.index))
    k_trend = _inv_pctl(delta_k.fillna(0))
    bb_trend = _pctl(delta_bb.fillna(0))
    trend_score = 0.50 * k_trend + 0.50 * bb_trend

    proj["trajectory_score"] = (
        0.50 * age_factor
        + 0.20 * yoy_score
        + 0.15 * certainty_score
        + 0.15 * trend_score
    )
    return proj[["batter_id", "trajectory_score"]]


def _build_versatility_score(base: pd.DataFrame) -> pd.Series:
    """Positional versatility bonus (0-1 scale).

    Players eligible at multiple premium positions (SS, C, CF) are more
    valuable than single-position players at low-value spots (1B/DH).

    Parameters
    ----------
    base : pd.DataFrame
        Hitter base DataFrame (must contain ``batter_id``).

    Returns
    -------
    pd.Series
        Versatility score aligned with ``base.index``.
    """
    try:
        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")
        # Count positions and sum tier values per player
        player_elig = elig.groupby("player_id").agg(
            n_positions=("position", "nunique"),
            tier_sum=("position", lambda x: sum(_POS_TIER.get(p, 0) for p in x)),
        )
        # Normalize: 1 position = 0.0, 4+ positions with premium = 1.0
        player_elig["versatility"] = (
            (player_elig["n_positions"] - 1) * 0.15
            + player_elig["tier_sum"] * 0.05
        ).clip(0, 1)
        return base["batter_id"].map(player_elig["versatility"]).fillna(0.0)
    except FileNotFoundError:
        logger.warning("Position eligibility parquet not found -- versatility scores set to 0")
        return pd.Series(0.0, index=base.index)


def _build_roster_construction_score(base: pd.DataFrame) -> pd.Series:
    """Roster scarcity bonus -- players at thin positions get a boost.

    For each team, counts how many ranked players are eligible at each
    position.  A player at a position with thin depth (1-2 eligible)
    gets a small value boost; deep positions (3+) get no bonus.

    Parameters
    ----------
    base : pd.DataFrame
        Hitter base DataFrame (must contain ``batter_id``, ``position``).

    Returns
    -------
    pd.Series
        Roster construction score aligned with ``base.index``.
    """
    try:
        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")
        player_teams_df = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")

        # Map player -> team
        pid_team = dict(zip(player_teams_df["player_id"], player_teams_df["team_abbr"]))

        # For each team-position combo, count eligible ranked players
        ranked_pids = set(base["batter_id"])
        elig_ranked = elig[elig["player_id"].isin(ranked_pids)].copy()
        elig_ranked["team"] = elig_ranked["player_id"].map(pid_team)

        team_pos_depth = elig_ranked.groupby(["team", "position"])["player_id"].nunique()

        # For each player, find their primary position's depth on their team
        scores: dict[int, float] = {}
        for _, row in base.iterrows():
            pid = row["batter_id"]
            pos = row["position"]
            team = pid_team.get(pid, "")
            depth = team_pos_depth.get((team, pos), 1)
            # Thin depth (1-2 players) = higher value; deep (4+) = no bonus
            scores[pid] = max(0, (3 - depth) / 3)  # 1 player = 0.67, 2 = 0.33, 3+ = 0

        return base["batter_id"].map(scores).fillna(0.0)
    except Exception:
        logger.warning("Could not compute roster construction scores -- set to 0")
        return pd.Series(0.0, index=base.index)
