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
    - **Performance level** (35%): observed_woba vs league average.
      How good was the player, not just how much did they improve.
    - **Improvement signal** (25%): observed_woba - career_woba.
      Positive = above career baseline = breakout or sustained peak.
    - **Statcast backing** (25%): xwOBA ≥ wOBA means the production is
      skill-backed, not BABIP luck.  Also incorporates barrel% and
      hard_hit% as quality indicators.
    - **Age credibility** (15%): young player's high performance is more
      sustainable.  A 24-year-old breakout is more believable than a
      35-year-old career-best.

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

    # --- 1. Performance level (35%): how good, not just improvement ---
    perf_level = _zscore_pctl(df["observed_woba"])

    # --- 2. Improvement signal (25%): above career baseline ---
    improvement = df["observed_woba"] - df["career_woba"]
    improvement_score = _zscore_pctl(improvement)

    # --- 3. Statcast backing (25%): is the production real? ---
    # xwOBA - wOBA gap: positive = unlucky (deserved better), near zero = real
    # Negative = lucky (BABIP-inflated, will regress)
    xwoba_gap = df["xwoba"] - df["woba"]
    # Reward: xwOBA ≥ wOBA (production is real or even unlucky)
    # Penalize: xwOBA << wOBA (lucky, will regress)
    backing_xwoba = _zscore_pctl(xwoba_gap)
    # Also factor in absolute quality metrics
    backing_barrel = _pctl(df["barrel_pct"])
    backing_hh = _pctl(df["hard_hit_pct"])
    statcast_backing = 0.50 * backing_xwoba + 0.25 * backing_barrel + 0.25 * backing_hh

    # --- 4. Age credibility (25%) ---
    # Young breakouts (≤28) stick at ~60% rate; 33+ career-bests regress
    # at ~80%.  This component ensures a 35-year-old's career-best (.409
    # Springer) scores much lower than a 24-year-old's emergence.
    age_cred = _hitter_age_factor(age)

    # --- Composite ---
    df["confirmed_breakout_score"] = (
        0.30 * perf_level
        + 0.25 * improvement_score
        + 0.20 * statcast_backing
        + 0.25 * age_cred
    )

    return df[["batter_id", "confirmed_breakout_score"]]


def _build_hitter_offense_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    aggressiveness: pd.DataFrame | None = None,
    use_sim_wrc_plus: bool = True,
    il_player_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Decomposed hitter offense: contact + decisions + damage.

    Three skill buckets, each with its own reliability curve:

    1. **Contact skill** (stabilizes ~150 PA): K% projected, K% observed.
       Fast-stabilizing -- the most projectable hitter trait.
    2. **Swing decisions** (stabilizes ~200 PA): BB% projected, chase rate,
       two-strike whiff rate.  Medium stability -- discipline metrics.
    3. **Damage on contact** (stabilizes ~300 PA): xwOBA, barrel%, hard_hit%,
       sweet_spot%.  Slow-stabilizing -- requires more BIP for signal.

    Each bucket blends projected + observed using its own trust ramp, then
    buckets are combined.  Replaces the old monolithic offense blend that
    mixed all signals at one global PA reliability.

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

    # Age discount on observed stats: older players' outlier seasons get
    # less trust, regressing more toward projections.  A 36-year-old's
    # career-best is far more likely regression-bound than a 26-year-old's.
    # Applies to damage + production (the two observation-heavy buckets).
    # Contact + decisions are already projection-anchored via K%/BB%.
    age = merged["age"].fillna(28)
    # Two-phase age trust: steepened 2026-04-06 to fix aging veteran
    # inflation (Trout #5, Springer #11 vs consensus #50+, #47).
    # Old curve: 33→0.91, 35→0.79 (too gentle).
    # New curve: 33→0.88, 35→0.74.  Moderate penalty that moves vets
    # down without over-penalizing through percentile redistribution.
    phase1 = (1.0 - ((age - 30).clip(0) * 0.04)).clip(0.88, 1.0)  # 30-33: 4%/yr
    phase2 = (0.88 - ((age - 33).clip(0) * 0.07)).clip(0.45, 0.88)  # 33+: 7%/yr
    age_trust = np.where(age <= 33, phase1, phase2)
    # age 30: 1.00, age 32: 0.92, age 33: 0.88, age 35: 0.74, age 37: 0.60, age 39: 0.46

    # Projection-deviation dampening: when observed stats diverge sharply
    # from the Bayesian projection, reduce trust in observed.  The model
    # has multi-year context -- a huge deviation signals an outlier season
    # (positive or negative) that is unlikely to repeat.
    # Works both ways: a career-worst year for a good player AND a
    # career-best year for an aging player both get dampened.
    #
    # Computed on RAW stat scale (not percentiles) to avoid tail
    # compression -- a 44-point wRC+ gap is meaningful even if both
    # values are above the 90th percentile.
    damage_deviation = (obs_damage - proj_damage).abs()
    damage_dev_trust = (1.0 - damage_deviation).clip(0.50, 1.0)

    # Aging spike penalty: if a 33+ year-old's most recent season sharply
    # outperforms their multi-year baseline, it's almost certainly an
    # outlier that will regress.  A 26-year-old breakout is believable;
    # a 36-year-old career-best after two bad years is not.
    # The penalty scales with age: harsher at 35+ than at 33.
    aging_spike_penalty = pd.Series(1.0, index=merged.index)
    if "xwoba" in merged.columns and "xwoba_multiyear" in merged.columns:
        single = merged["xwoba"].fillna(0)
        multi = merged["xwoba_multiyear"].fillna(single)
        spike = (single - multi).clip(0)  # only penalize positive spikes
        # Age-scaled severity: 33 = 15% per unit, 35 = 25%, 37 = 35%
        age_severity = (0.15 + ((age - 33).clip(0) * 0.05)).clip(0.15, 0.40)
        is_aging_spike = (age >= 33) & (spike > 0.015)
        spike_factor = (1.0 - (spike - 0.015) / 0.030 * age_severity).clip(0.35, 1.0)
        aging_spike_penalty = np.where(is_aging_spike, spike_factor, 1.0)

    damage_trust = (
        _stat_family_trust(pa, min_pa=150, full_pa=550)
        * age_trust * damage_dev_trust * aging_spike_penalty
    )
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

    # Production deviation: use raw wRC+ gap normalized by population SD.
    # Percentile deviation compresses the tails (165 vs 121 wRC+ both map
    # to >93rd pctl, hiding the 44-point gap).  Raw z-score preserves it.
    if has_sim and _wrc_col in merged.columns:
        raw_wrc_gap = (merged[_wrc_col].fillna(100) - merged["projected_wrc_plus_mean"].fillna(100)).abs()
        wrc_std = merged["wrc_plus"].std()
        if wrc_std > 0:
            production_deviation_z = raw_wrc_gap / wrc_std
        else:
            production_deviation_z = pd.Series(0.0, index=merged.index)
        # z=0: trust 1.0, z=1 (~30 wRC+ gap): 0.70, z=1.67+: 0.50 floor
        production_dev_trust = (1.0 - production_deviation_z * 0.30).clip(0.50, 1.0)
    else:
        production_dev_trust = pd.Series(1.0, index=merged.index)

    # Aging spike penalty for production (same logic as damage bucket)
    aging_spike_prod = pd.Series(1.0, index=merged.index)
    if "wrc_plus" in merged.columns and "wrc_plus_multiyear" in merged.columns:
        wrc_single = merged["wrc_plus"].fillna(100)
        wrc_multi = merged["wrc_plus_multiyear"].fillna(wrc_single)
        wrc_spike = (wrc_single - wrc_multi).clip(0)
        # Age-scaled wRC+ spike penalty (same pattern as damage)
        age_severity_wrc = (0.15 + ((age - 33).clip(0) * 0.05)).clip(0.15, 0.40)
        is_wrc_spike = (age >= 33) & (wrc_spike > 10)
        spike_factor_wrc = (1.0 - (wrc_spike - 10) / 15 * age_severity_wrc).clip(0.35, 1.0)
        aging_spike_prod = np.where(is_wrc_spike, spike_factor_wrc, 1.0)

    production_trust = (
        _stat_family_trust(pa, min_pa=100, full_pa=500)
        * age_trust * production_dev_trust * aging_spike_prod
    )

    merged["production_skill"] = (
        (1 - production_trust) * proj_production
        + production_trust * obs_production
    )

    # =================================================================
    # Combine buckets into offense_score
    # Two-layer blend:
    #   70% -- projected wOBA percentile (total production truth).
    #          Prevents the bucket decomposition from creating artificial
    #          spread between players with similar total output (e.g. a
    #          low-walk contact hitter vs a high-walk power hitter).
    #   30% -- skill-bucket composite (profile differentiation).
    #          Rewards/penalizes specific skill profiles on top of the
    #          production anchor.  Decision skill reduced to 5% -- walk
    #          rate is already captured in projected wOBA; higher weight
    #          double-penalizes aggressive hitters whose results are fine.
    # =================================================================
    # Blend projected + career wOBA for the anchor.  Projected wOBA is
    # compressed by Bayesian shrinkage (std=0.018); career wOBA preserves
    # demonstrated ability (std=0.031).  50/50 blend keeps projection
    # signal while restoring meaningful spread.
    # Note: career_woba is now recency-weighted (3/2/1 by season) in
    # hitter_projections.py, which already discounts old prime years.
    # Age-dependent blend weight was tested but caused population-level
    # zscore_pctl shifts that penalized ALL players (young included)
    # because the distribution shape changed.
    _proj_woba = merged["projected_woba"].fillna(0.310)
    _career_woba = merged["career_woba"].fillna(_proj_woba) if "career_woba" in merged.columns else _proj_woba
    # Career wOBA reliability ramp: small-career players (Jahmai Jones,
    # 150 PA) shouldn't have career_woba weighted equally with veterans.
    # Tango stabilization for wOBA ~300-500 PA.  Uses prior-season PA
    # from the projections parquet as a proxy for career exposure.
    # At 150 PA: career weight = 0.50 * (150/800) = 9%.
    # At 400 PA: career weight = 0.50 * (400/800) = 25%.
    # At 800 PA: career weight = 0.50 * 1.0 = 50% (full blend).
    _proj_pa = merged["_proj_pa"].fillna(400) if "_proj_pa" in merged.columns else pd.Series(400, index=merged.index)
    _career_reliability = (_proj_pa / 800).clip(0, 1)
    _career_weight = 0.50 * _career_reliability
    _blended_woba = (1.0 - _career_weight) * _proj_woba + _career_weight * _career_woba
    woba_anchor = _zscore_pctl(_blended_woba)

    bucket_composite = (
        0.175 * merged["contact_skill"]
        + 0.05 * merged["decision_skill"]
        + 0.425 * merged["damage_skill"]
        + 0.35 * merged["production_skill"]
    )

    merged["offense_score"] = 0.70 * woba_anchor + 0.30 * bucket_composite

    # Small-sample dampening toward league average.  Three regimes:
    #
    # 1. < 10 PA and NOT on IL: heavy dampening (30%).  These players
    #    aren't starting — bench, minors, or fringe roster.  Their
    #    projections from Bayesian regression shouldn't carry full weight.
    # 2. 10-49 PA, OR < 10 PA but ON IL: light dampening (10%).
    #    Active starters in early season, or injured starters who will
    #    return.  Projections carry the signal; heavier dampening here
    #    compresses stars toward average and lets fielding/baserunning
    #    dominate the composite.
    # 3. 50-400 PA: ramp from 20% (at 50 PA) to 0% (at 400+ PA).
    pa_safe = pa.fillna(0)
    _il = il_player_ids or set()
    _on_il = merged["batter_id"].isin(_il)
    _is_active = (pa_safe >= 10) | _on_il  # starters or IL-exempt
    dampening = np.where(
        ~_is_active,
        0.30,                                          # regime 1: bench/inactive
        np.where(
            pa_safe < 50,
            0.10,                                      # regime 2: early season / IL
            0.20 * (1.0 - ((pa_safe - 50) / 350).clip(0, 1)),  # regime 3: ramp
        ),
    )
    merged["offense_score"] = merged["offense_score"] * (1 - dampening) + 0.50 * dampening

    return merged[["batter_id", "offense_score", "contact_skill",
                    "decision_skill", "damage_skill", "production_skill"]]


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

    # Sprint speed + HP-to-1B from Statcast (fall back to prior season
    # when current season data isn't available yet -- e.g. early April)
    speed_df = read_sql(f"""
        SELECT DISTINCT ON (player_id)
               player_id AS batter_id, sprint_speed, hp_to_1b
        FROM staging.statcast_sprint_speed
        WHERE season BETWEEN {season - 1} AND {season}
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
    # Regress toward league average (~78%) using Bayesian shrinkage
    _LEAGUE_SB_SUCCESS = 0.78
    _SB_REGRESS_N = 10  # regress with 10 pseudo-attempts at league avg
    obs_sb = merged["obs_sb"].fillna(0)
    obs_cs = merged["obs_cs"].fillna(0)
    obs_att = obs_sb + obs_cs
    regressed_success = (
        (obs_sb + _SB_REGRESS_N * _LEAGUE_SB_SUCCESS)
        / (obs_att + _SB_REGRESS_N)
    )
    # Only score efficiency for players with enough attempts
    has_attempts = obs_att >= 3
    sb_eff_score = _pctl(regressed_success)
    # Neutral (0.50) for players with < 3 attempts
    sb_eff_score = np.where(has_attempts, sb_eff_score, 0.50)

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
    """Build fielding score from multi-year OAA with position-aware regression.

    Uses 3-year rolling OAA with recency weights (3/2/1), then regresses
    toward 0 (position average) based on sample size and position group.
    OF metrics are more stable (k=2), IF noisier (k=3).

    Parameters
    ----------
    season : int
        Most recent season for OAA data.
    in_season : bool
        If True, exclude current season OAA (too few games to be reliable)
        and use prior 3 full seasons instead.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, fielding_score.
    """
    from src.data.db import read_sql

    # In-season: current-year OAA from ~1 week is noise (gets 3x weight and
    # wildly distorts scores).  Use prior 3 full seasons instead.
    anchor = season - 1 if in_season else season
    oaa = read_sql(f"""
        SELECT player_id, season, position, outs_above_average
        FROM production.fact_fielding_oaa
        WHERE season BETWEEN {anchor - 2} AND {anchor}
    """, {})

    if oaa.empty:
        return pd.DataFrame(columns=["player_id", "fielding_score"])

    # Recency weights: most recent season = 3, year-1 = 2, year-2 = 1
    oaa["weight"] = oaa["season"].map(
        {season: 3, season - 1: 2, season - 2: 1}
    ).fillna(1)

    # Position-group regression constants (higher k = more regression)
    # OF OAA is more stable (larger sample of chances); IF is noisier
    _IF_POSITIONS = {"SS", "2B", "3B", "1B"}
    _OF_POSITIONS = {"LF", "CF", "RF"}

    # Determine primary position group per player (mode of positions played)
    pos_mode = (
        oaa.groupby("player_id")["position"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "IF")
    )

    # Weighted average OAA per player
    weighted = oaa.groupby("player_id").apply(
        lambda g: np.average(g["outs_above_average"], weights=g["weight"]),
        include_groups=False,
    ).rename("weighted_oaa")

    # Count seasons with data per player (for reliability)
    n_seasons = oaa.groupby("player_id")["season"].nunique().rename("n_seasons")

    result = pd.DataFrame({"weighted_oaa": weighted, "n_seasons": n_seasons})
    result["position_group"] = pos_mode

    # Regression constant: k=2 for OF (more stable), k=3 for IF (noisier)
    result["k"] = result["position_group"].apply(
        lambda p: 2 if p in _OF_POSITIONS else 3
    )

    # Position-specific OAA priors (empirical means, 2023-2025).
    # Unknown fielders regress toward their POSITION average, not 0.
    # This prevents unproven 1B from getting neutral (50th pctl) scores
    # when the typical 1B is below average defensively.
    _POSITION_OAA_PRIOR = {
        "CF": 2.8, "SS": 1.1, "2B": 0.2, "3B": -0.5,
        "1B": -0.9, "RF": -1.1, "LF": -1.4,
    }
    result["pos_prior"] = result["position_group"].map(_POSITION_OAA_PRIOR).fillna(0.0)

    # Reliability: n_seasons / (n_seasons + k)
    # 1 year IF: 1/4=0.25, 3 years IF: 3/6=0.50
    # 1 year OF: 1/3=0.33, 3 years OF: 3/5=0.60
    result["reliability"] = result["n_seasons"] / (result["n_seasons"] + result["k"])
    result["regressed_oaa"] = (
        result["reliability"] * result["weighted_oaa"]
        + (1 - result["reliability"]) * result["pos_prior"]
    )

    # Percentile rank the regressed OAA, with floor so worst defenders
    # aren't zeroed out (even -15 OAA provides some value vs empty position)
    result["fielding_score"] = _pctl(result["regressed_oaa"]).clip(lower=0.10)

    return result.reset_index()[["player_id", "fielding_score"]]


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
    - **Age factor** (35%): younger players get upside credit.  The single
      strongest predictor of future value trajectory.
    - **YoY observed improvement** (30%): did the player's actual wOBA/wRC+
      improve from prior year?  Captures breakouts (Henderson, Raleigh)
      and declines (Trout, Springer) that the Bayesian delta misses.
    - **Projection certainty** (20%): lower CV = more proven.  Reduced
      from 55% — still rewards stability but no longer dominates.
    - **Season trend** (15%): Bayesian projected delta signal (kept as
      minor tiebreaker).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, trajectory_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # --- Age factor (35%): research-aligned curve ---
    age = proj["age"].fillna(30)
    age_factor = _hitter_age_factor(age)

    # --- YoY observed improvement (30%) ---
    # Compare most recent observed season to prior.  Uses delta_woba
    # (projected - observed) as a proxy when direct YoY isn't available,
    # but prefer actual observed_woba vs career_woba gap.
    # Positive = player is currently above their career baseline = improving.
    if "observed_woba" in proj.columns and "career_woba" in proj.columns:
        _obs = proj["observed_woba"].fillna(proj["career_woba"])
        _career = proj["career_woba"].fillna(0.315)
        yoy_improvement = _obs - _career  # positive = above career baseline
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
        0.35 * age_factor
        + 0.30 * yoy_score
        + 0.20 * certainty_score
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
