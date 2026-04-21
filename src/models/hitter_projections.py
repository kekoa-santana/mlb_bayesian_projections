"""
Composite hitter projections — 6-dimension scoring.

Fits K% and BB% Bayesian models (stable, projectable stats), then enriches
with observed profile stats (whiff rate, chase rate, exit velo, hard-hit%,
sprint speed, FB%) for a comprehensive composite score.

Dimensions
----------
1. Contact ability (20%): whiff_rate + z_contact_pct
2. Plate discipline (20%): chase_rate + projected BB% delta
3. Raw power (25%): avg_exit_velo + hard_hit_pct
4. Speed (15%): sprint_speed
5. Batted ball profile (5%): fb_pct
6. Projected trajectory (15%): K% delta + BB% delta
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_hitter_data,
    get_cached_hitter_observed_profile,
    get_cached_sprint_speed,
)
from src.models.hitter_model import (
    STAT_CONFIGS,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)
from src.models.projection_utils import (
    compute_composite,
    find_breakouts_and_regressions as _find_breakouts_and_regressions,
    fit_all_models_generic,
    project_rate_samples,
)

logger = logging.getLogger(__name__)

# Bayesian-projected stats (hierarchical + AR(1))
ALL_STATS = ["k_rate", "bb_rate", "gb_rate", "fb_rate", "hr_per_fb", "woba", "chase_rate"]

# ---------------------------------------------------------------------------
# wOBA spread calibration — INVESTIGATED, NOT APPLIED
# ---------------------------------------------------------------------------
# The logit-normal hierarchical model compresses projected_woba (std~0.018)
# below empirical true-talent spread (std~0.025).  Walk-forward backtest
# (6 folds, 1445 player-seasons) showed:
#   - Widening priors (mu_pop_sigma 0.25->0.35, StudentT(4) alpha) caused
#     ESS collapse (r_hat=1.29, ESS=12) without fixing compression.
#   - Post-hoc stretch 1.4x restores spread but costs +7.6% CRPS.
#   - No-stretch model is already well-calibrated: z-std=1.04, 95% cov=93.8%.
# Decision: keep compressed projections (optimal for betting/CRPS), fix
# ranking spread via career_woba blend + zscore_pctl in _hitter_scores.py.
_WOBA_SPREAD_STRETCH = 1.0  # 1.0 = disabled

# --------------------------------------------------------------------------
# Composite dimension weights and components
# --------------------------------------------------------------------------
# Each dimension: (weight, [(col, sign, source)])
#   sign: +1 means higher percentile = better, -1 means lower = better
#   source: "observed" or "projected_delta"
COMPOSITE_DIMENSIONS: dict[str, tuple[float, list[tuple[str, int, str]]]] = {
    "contact": (0.20, [
        ("whiff_rate", -1, "observed"),       # lower whiff = better
        ("z_contact_pct", +1, "observed"),    # higher zone contact = better
    ]),
    "discipline": (0.20, [
        ("chase_rate", -1, "observed"),       # lower chase = better
        ("delta_bb_rate", +1, "projected_delta"),  # improving BB% = better
    ]),
    "power": (0.25, [
        ("avg_exit_velo", +1, "observed"),    # higher EV = better
        ("hard_hit_pct", +1, "observed"),     # higher HH% = better
    ]),
    "speed": (0.15, [
        ("sprint_speed", +1, "observed"),     # faster = better
    ]),
    "batted_ball": (0.05, [
        ("fb_pct", +1, "observed"),           # more fly balls = power indicator
    ]),
    "trajectory": (0.15, [
        ("delta_k_rate", -1, "projected_delta"),  # decreasing K% = better
        ("delta_bb_rate", +1, "projected_delta"),  # increasing BB% = better
    ]),
}


def fit_all_models(
    seasons: list[int],
    min_pa: int = 100,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    stats: list[str] | None = None,
    extract_season: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit K% and BB% hitter projection models.

    Parameters
    ----------
    seasons : list[int]
        Training seasons.
    min_pa : int
        Minimum PA per player-season.
    draws, tune, chains, random_seed
        MCMC parameters.
    stats : list[str] | None
        Stats to fit. Defaults to ALL_STATS (k_rate, bb_rate).
    extract_season : int | None
        If set, pre-extract forward-projected rate samples for all hitters
        in this season immediately after fitting, then free the MCMC trace
        to save memory.  Results stored in ``results[stat]["rate_samples"]``.

    Returns
    -------
    dict[str, dict]
        Keyed by stat name.  Always contains "data", "convergence",
        "posteriors".  Contains "rate_samples" (dict[int, ndarray]) when
        *extract_season* is set, otherwise "trace" (InferenceData).
    """
    if stats is None:
        stats = ALL_STATS

    return fit_all_models_generic(
        stats=stats,
        data_builder_fn=build_multi_season_hitter_data,
        data_builder_kwargs={"seasons": seasons, "min_pa": min_pa},
        prepare_data_fn=prepare_hitter_data,
        model_fitter_fn=fit_hitter_model,
        check_convergence_fn=check_convergence,
        extract_posteriors_fn=extract_posteriors,
        extract_samples_fn=extract_rate_samples,
        id_col="batter_id",
        player_type="hitter",
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        extract_season=extract_season,
    )


def _enrich_with_observed(
    base: pd.DataFrame,
    from_season: int,
) -> pd.DataFrame:
    """Merge observed pitch-level stats and sprint speed into base."""
    # Pitch-level observed profile
    try:
        obs = get_cached_hitter_observed_profile(from_season)
        merge_cols = ["whiff_rate", "chase_rate", "z_contact_pct",
                      "avg_exit_velo", "fb_pct", "hard_hit_pct"]
        base = base.merge(
            obs[["batter_id"] + merge_cols],
            on="batter_id",
            how="left",
            suffixes=("", "_obs"),
        )
    except Exception as e:
        logger.warning("Could not load hitter observed profile: %s", e)
        for col in ["whiff_rate", "chase_rate", "z_contact_pct",
                     "avg_exit_velo", "fb_pct", "hard_hit_pct"]:
            if col not in base.columns:
                base[col] = np.nan

    # Sprint speed
    try:
        sprint = get_cached_sprint_speed(from_season)
        base = base.merge(
            sprint[["player_id", "sprint_speed"]].rename(
                columns={"player_id": "batter_id"}
            ),
            on="batter_id",
            how="left",
        )
    except Exception as e:
        logger.warning("Could not load sprint speed: %s", e)
        if "sprint_speed" not in base.columns:
            base["sprint_speed"] = np.nan

    return base


def _compute_composite(base: pd.DataFrame) -> pd.DataFrame:
    """Compute 6-dimension composite score using z-score normalization."""
    return compute_composite(base, COMPOSITE_DIMENSIONS)


def project_forward(
    model_results: dict[str, dict[str, Any]],
    from_season: int,
    min_pa: int = 150,
    random_seed: int = 42,
    calibration_t: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Forward-project K%/BB% and build 6-dimension composite.

    Parameters
    ----------
    model_results : dict
        Output of ``fit_all_models``.
    from_season : int
        Season to project from (most recent training season).
    min_pa : int
        Minimum PA in from_season to include in projections.
    random_seed : int
        For reproducibility.
    calibration_t : dict, optional
        Per-stat calibration factors {stat_name: T}.
        T < 1.0 narrows intervals, T > 1.0 widens.
        From ``config/model.yaml`` calibration section.

    Returns
    -------
    pd.DataFrame
        One row per player with projected K%/BB%, observed stats,
        per-stat deltas, and composite improvement score.
    """
    # Start with player info from any stat (they all use the same data)
    first_stat = list(model_results.keys())[0]
    first_data = model_results[first_stat]["data"]
    first_df = first_data["df"]

    # Get players from the projection season with enough PA
    keep_cols = ["batter_id", "batter_name", "batter_stand", "season", "age",
                  "age_bucket", "pa"]
    if "skill_tier" in first_df.columns:
        keep_cols.append("skill_tier")
    base = first_df[
        (first_df["season"] == from_season) & (first_df["pa"] >= min_pa)
    ][keep_cols].copy()

    # Fallback: players from prior season who aren't in from_season yet.
    # These get preseason-style projections with wider uncertainty.
    prior_season = from_season - 1
    current_ids = set(base["batter_id"])
    prior_df = first_df[
        (first_df["season"] == prior_season)
        & (first_df["pa"] >= min_pa)
        & ~first_df["batter_id"].isin(current_ids)
    ][keep_cols].copy()
    if len(prior_df) > 0:
        prior_df["age"] = prior_df["age"] + 1
        prior_df["season"] = from_season
        base = pd.concat([base, prior_df], ignore_index=True)
        logger.info(
            "Added %d fallback hitters from %d (not yet in %d)",
            len(prior_df), prior_season, from_season,
        )

    if len(base) == 0:
        logger.warning("No players found in season %d with >= %d PA",
                       from_season, min_pa)
        return pd.DataFrame()

    # For each Bayesian stat (K%, BB%), extract observed + projected
    for stat, res in model_results.items():
        cfg = STAT_CONFIGS[stat]
        data = res["data"]
        pre_extracted = res.get("rate_samples") or {}
        trace = res.get("trace")

        stat_df = data["df"]
        stat_season = stat_df[stat_df["season"] == from_season]

        obs_col = f"observed_{stat}"
        proj_col = f"projected_{stat}"
        delta_col = f"delta_{stat}"
        sd_col = f"projected_{stat}_sd"
        ci_lo_col = f"projected_{stat}_2_5"
        ci_hi_col = f"projected_{stat}_97_5"

        obs_map = dict(zip(stat_season["batter_id"], stat_season[cfg.rate_col]))
        base[obs_col] = base["batter_id"].map(obs_map)

        # Recency-weighted career average (3/2/1 weighting by season).
        # All-time PA-weighted averages let prime-year stats prop up aging
        # veterans (e.g. Trout career .413 wOBA inflates his offense score
        # even as he declines).  Recency weighting (most recent 3x, oldest
        # 1x) credits recent performance more, which better reflects
        # current talent for both rising young players and aging vets.
        career_col = f"career_{stat}"
        _career_df = stat_df[
            stat_df["batter_id"].isin(base["batter_id"])
        ].copy()
        if not _career_df.empty:
            _max_season = _career_df["season"].max()
            # Recency weight: most recent season = 3, oldest = 1
            _career_df["_recency_w"] = (
                1.0 + 2.0 * ((_career_df["season"] - _career_df["season"].min())
                              / max(1, _max_season - _career_df["season"].min()))
            )
            _career_df["_w"] = _career_df[cfg.trials_col] * _career_df["_recency_w"]
            career_sum = _career_df.groupby("batter_id").apply(
                lambda g: (g[cfg.rate_col] * g["_w"]).sum() / g["_w"].sum()
                if g["_w"].sum() > 0 else np.nan,
                include_groups=False,
            )
        else:
            career_sum = pd.Series(dtype=float)
        base[career_col] = base["batter_id"].map(career_sum)

        # Forward-project each player
        _cal_t = (calibration_t or {}).get(stat, 1.0)

        proj_means, proj_sds, proj_lo, proj_hi, _ = project_rate_samples(
            ids=base["batter_id"],
            pre_extracted=pre_extracted,
            trace=trace,
            data=data,
            from_season=from_season,
            id_col="batter_id",
            extract_samples_fn=extract_rate_samples,
            calibration_t=_cal_t,
            random_seed=random_seed,
            collect_samples=False,
        )

        base[proj_col] = base["batter_id"].map(proj_means)
        base[sd_col] = base["batter_id"].map(proj_sds)
        base[ci_lo_col] = base["batter_id"].map(proj_lo)
        base[ci_hi_col] = base["batter_id"].map(proj_hi)

        base[delta_col] = base[proj_col] - base[obs_col]

    # Enrich with observed stats (whiff, chase, EV, sprint speed, etc.)
    base = _enrich_with_observed(base, from_season)

    # Compute 6-dimension composite
    base = _compute_composite(base)

    # Sort by composite score (biggest improvers first)
    base = base.sort_values("composite_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Projected %d players from %d forward",
        len(base), from_season,
    )
    return base


def find_breakouts_and_regressions(
    projections: pd.DataFrame,
    n_top: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split projections into breakout candidates and regression risks.

    Parameters
    ----------
    projections : pd.DataFrame
        Output of ``project_forward``.
    n_top : int
        Number of players per category.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (breakouts, regressions) sorted by |composite_score|.
    """
    return _find_breakouts_and_regressions(projections, n_top=n_top)
