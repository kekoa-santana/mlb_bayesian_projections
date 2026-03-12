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

logger = logging.getLogger(__name__)

# Only project stable stats with the Bayesian model
ALL_STATS = ["k_rate", "bb_rate", "hr_rate"]

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

    Returns
    -------
    dict[str, dict]
        Keyed by stat name, each containing:
        "data", "trace", "convergence", "posteriors".
    """
    if stats is None:
        stats = ALL_STATS

    df = build_multi_season_hitter_data(seasons, min_pa=min_pa)
    logger.info("Loaded %d player-seasons for projection", len(df))

    results: dict[str, dict[str, Any]] = {}

    for stat in stats:
        logger.info("=" * 50)
        logger.info("Fitting %s model", stat)

        data = prepare_hitter_data(df, stat)
        model, trace = fit_hitter_model(
            data,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        conv = check_convergence(trace, stat)
        posteriors = extract_posteriors(trace, data)

        results[stat] = {
            "data": data,
            "trace": trace,
            "convergence": conv,
            "posteriors": posteriors,
        }

        logger.info(
            "%s: converged=%s, r_hat=%.4f",
            stat, conv["converged"], conv["max_rhat"],
        )

    return results


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
    base["composite_score"] = 0.0
    base["_total_weight"] = 0.0

    for dim_name, (weight, components) in COMPOSITE_DIMENSIONS.items():
        dim_z_scores = []

        for col, sign, source in components:
            if col not in base.columns:
                continue
            vals = base[col].astype(float)
            mu, sd = vals.mean(), vals.std()
            if pd.isna(sd) or np.isclose(sd, 0.0):
                continue
            z = (vals - mu) / sd * sign
            dim_z_scores.append(z)

        if not dim_z_scores:
            continue

        # Average z-scores within dimension (NaN-safe)
        dim_df = pd.concat(dim_z_scores, axis=1)
        dim_avg = dim_df.mean(axis=1)

        has_value = dim_avg.notna()
        base["composite_score"] += weight * dim_avg.fillna(0)
        base["_total_weight"] += has_value.astype(float) * weight

    # Re-scale so players with partial stats are comparable
    base["composite_score"] = np.where(
        base["_total_weight"] > 0,
        base["composite_score"] / base["_total_weight"],
        0.0,
    )
    base.drop(columns=["_total_weight"], inplace=True)

    return base


def project_forward(
    model_results: dict[str, dict[str, Any]],
    from_season: int,
    min_pa: int = 200,
    random_seed: int = 42,
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

    if len(base) == 0:
        logger.warning("No players found in season %d with >= %d PA",
                       from_season, min_pa)
        return pd.DataFrame()

    # For each Bayesian stat (K%, BB%), extract observed + projected
    for stat, res in model_results.items():
        cfg = STAT_CONFIGS[stat]
        data = res["data"]
        trace = res["trace"]

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

        # Career PA-weighted average across all training seasons
        career_col = f"career_{stat}"
        career_group = stat_df[
            stat_df["batter_id"].isin(base["batter_id"])
        ].groupby("batter_id")
        career_sum = career_group.apply(
            lambda g: (g[cfg.rate_col] * g[cfg.trials_col]).sum()
            / g[cfg.trials_col].sum()
            if g[cfg.trials_col].sum() > 0 else np.nan,
            include_groups=False,
        )
        base[career_col] = base["batter_id"].map(career_sum)

        # Forward-project each player
        proj_means = {}
        proj_sds = {}
        proj_lo = {}
        proj_hi = {}

        for batter_id in base["batter_id"]:
            try:
                samples = extract_rate_samples(
                    trace, data, batter_id, from_season,
                    project_forward=True, random_seed=random_seed,
                )
                proj_means[batter_id] = float(np.mean(samples))
                proj_sds[batter_id] = float(np.std(samples))
                proj_lo[batter_id] = float(np.percentile(samples, 2.5))
                proj_hi[batter_id] = float(np.percentile(samples, 97.5))
            except ValueError:
                continue

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
    breakouts = projections.head(n_top).copy()
    regressions = projections.tail(n_top).iloc[::-1].copy()

    return breakouts.reset_index(drop=True), regressions.reset_index(drop=True)
