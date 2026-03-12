"""
Composite pitcher projections — 4-dimension scoring.

Fits K% and BB% Bayesian models (stable, projectable stats), then enriches
with observed profile stats (whiff rate, avg velo, extension, zone%, GB%)
for a comprehensive composite score.

Dimensions
----------
1. Stuff (35%): whiff_rate + avg_velo + release_extension
2. Command (25%): projected BB% delta + zone_pct
3. Ground ball profile (15%): gb_pct
4. Projected trajectory (25%): K% delta + BB% delta
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_pitcher_data,
    get_cached_pitcher_observed_profile,
)
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)

# Only project stable stats with the Bayesian model
ALL_STATS = ["k_rate", "bb_rate", "hr_per_bf"]

# --------------------------------------------------------------------------
# Composite dimension weights and components
# --------------------------------------------------------------------------
COMPOSITE_DIMENSIONS: dict[str, tuple[float, list[tuple[str, int, str]]]] = {
    "stuff": (0.35, [
        ("whiff_rate", +1, "observed"),          # higher whiff = better stuff
        ("avg_velo", +1, "observed"),            # harder = better
        ("release_extension", +1, "observed"),   # more extension = better
    ]),
    "command": (0.25, [
        ("delta_bb_rate", -1, "projected_delta"),  # decreasing BB% = better
        ("zone_pct", +1, "observed"),              # more zone pitches = better
    ]),
    "ground_ball": (0.15, [
        ("gb_pct", +1, "observed"),              # more GBs = fewer HRs
    ]),
    "trajectory": (0.25, [
        ("delta_k_rate", +1, "projected_delta"),   # increasing K% = better
        ("delta_bb_rate", -1, "projected_delta"),   # decreasing BB% = better
    ]),
}


def fit_all_models(
    seasons: list[int],
    min_bf: int = 100,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    stats: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit K% and BB% pitcher projection models.

    Parameters
    ----------
    seasons : list[int]
        Training seasons.
    min_bf : int
        Minimum BF per player-season.
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

    df = build_multi_season_pitcher_data(seasons, min_bf=min_bf)
    logger.info("Loaded %d pitcher-seasons for projection", len(df))

    results: dict[str, dict[str, Any]] = {}

    for stat in stats:
        logger.info("=" * 50)
        logger.info("Fitting pitcher %s model", stat)

        data = prepare_pitcher_data(df, stat)
        model, trace = fit_pitcher_model(
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
            "pitcher %s: converged=%s, r_hat=%.4f",
            stat, conv["converged"], conv["max_rhat"],
        )

    return results


def _enrich_with_observed(
    base: pd.DataFrame,
    from_season: int,
) -> pd.DataFrame:
    """Merge observed pitch-level stats into base."""
    try:
        obs = get_cached_pitcher_observed_profile(from_season)
        merge_cols = ["whiff_rate", "avg_velo", "release_extension",
                      "zone_pct", "gb_pct"]
        base = base.merge(
            obs[["pitcher_id"] + merge_cols],
            on="pitcher_id",
            how="left",
            suffixes=("", "_obs"),
        )
    except Exception as e:
        logger.warning("Could not load pitcher observed profile: %s", e)
        for col in ["whiff_rate", "avg_velo", "release_extension",
                     "zone_pct", "gb_pct"]:
            if col not in base.columns:
                base[col] = np.nan

    return base


def _compute_composite(base: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-dimension composite score using z-score normalization."""
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

        dim_df = pd.concat(dim_z_scores, axis=1)
        dim_avg = dim_df.mean(axis=1)

        has_value = dim_avg.notna()
        base["composite_score"] += weight * dim_avg.fillna(0)
        base["_total_weight"] += has_value.astype(float) * weight

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
    min_bf: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Forward-project K%/BB% and build 4-dimension composite.

    Parameters
    ----------
    model_results : dict
        Output of ``fit_all_models``.
    from_season : int
        Season to project from (most recent training season).
    min_bf : int
        Minimum BF in from_season to include in projections.
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per pitcher with projected K%/BB%, observed stats,
        per-stat deltas, and composite improvement score.
    """
    first_stat = list(model_results.keys())[0]
    first_data = model_results[first_stat]["data"]
    first_df = first_data["df"]

    keep_cols = ["pitcher_id", "pitcher_name", "pitch_hand", "season", "age",
                  "age_bucket", "batters_faced", "is_starter"]
    if "skill_tier" in first_df.columns:
        keep_cols.append("skill_tier")
    base = first_df[
        (first_df["season"] == from_season) & (first_df["batters_faced"] >= min_bf)
    ][keep_cols].copy()

    if len(base) == 0:
        logger.warning("No pitchers found in season %d with >= %d BF",
                        from_season, min_bf)
        return pd.DataFrame()

    # For each Bayesian stat (K%, BB%), extract observed + projected
    for stat, res in model_results.items():
        cfg = PITCHER_STAT_CONFIGS[stat]
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

        obs_map = dict(zip(stat_season["pitcher_id"], stat_season[cfg.rate_col]))
        base[obs_col] = base["pitcher_id"].map(obs_map)

        # Career BF-weighted average
        career_col = f"career_{stat}"
        career_group = stat_df[
            stat_df["pitcher_id"].isin(base["pitcher_id"])
        ].groupby("pitcher_id")
        career_sum = career_group.apply(
            lambda g: (g[cfg.rate_col] * g["batters_faced"]).sum()
            / g["batters_faced"].sum()
            if g["batters_faced"].sum() > 0 else np.nan,
            include_groups=False,
        )
        base[career_col] = base["pitcher_id"].map(career_sum)

        proj_means = {}
        proj_sds = {}
        proj_lo = {}
        proj_hi = {}

        for pitcher_id in base["pitcher_id"]:
            try:
                samples = extract_rate_samples(
                    trace, data, pitcher_id, from_season,
                    project_forward=True, random_seed=random_seed,
                )
                proj_means[pitcher_id] = float(np.mean(samples))
                proj_sds[pitcher_id] = float(np.std(samples))
                proj_lo[pitcher_id] = float(np.percentile(samples, 2.5))
                proj_hi[pitcher_id] = float(np.percentile(samples, 97.5))
            except ValueError:
                continue

        base[proj_col] = base["pitcher_id"].map(proj_means)
        base[sd_col] = base["pitcher_id"].map(proj_sds)
        base[ci_lo_col] = base["pitcher_id"].map(proj_lo)
        base[ci_hi_col] = base["pitcher_id"].map(proj_hi)
        base[delta_col] = base[proj_col] - base[obs_col]

    # Enrich with observed stats (whiff, velo, extension, zone%, GB%)
    base = _enrich_with_observed(base, from_season)

    # Compute 4-dimension composite
    base = _compute_composite(base)

    base = base.sort_values("composite_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Projected %d pitchers from %d forward",
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
        Number of pitchers per category.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (breakouts, regressions) sorted by |composite_score|.
    """
    breakouts = projections.head(n_top).copy()
    regressions = projections.tail(n_top).iloc[::-1].copy()

    return breakouts.reset_index(drop=True), regressions.reset_index(drop=True)
