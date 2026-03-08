"""
Composite pitcher projections combining K%, BB%, and HR/BF.

Fits all stat models, forward-projects posteriors, and produces
a unified projection DataFrame with per-stat deltas and a weighted
composite improvement score.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import build_multi_season_pitcher_data
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)

# Stats to fit
ALL_STATS = ["k_rate", "bb_rate", "hr_per_bf"]

# Composite weights — signs: positive delta = improvement for pitcher
# K%: higher is better for pitcher (+1)
# BB%: lower is better for pitcher (-1)
# HR/BF: lower is better for pitcher (-1)
COMPOSITE_WEIGHTS: dict[str, tuple[float, int]] = {
    "k_rate":    (0.40, +1),
    "bb_rate":   (0.30, -1),
    "hr_per_bf": (0.30, -1),
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
    """Fit all pitcher projection models.

    Parameters
    ----------
    seasons : list[int]
        Training seasons.
    min_bf : int
        Minimum BF per player-season.
    draws, tune, chains, random_seed
        MCMC parameters.
    stats : list[str] | None
        Stats to fit. Defaults to ALL_STATS.

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


def project_forward(
    model_results: dict[str, dict[str, Any]],
    from_season: int,
    min_bf: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Forward-project all stats and build composite projections.

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
        One row per pitcher with observed/projected values for each stat,
        per-stat deltas, and composite improvement score.
    """
    first_stat = list(model_results.keys())[0]
    first_data = model_results[first_stat]["data"]
    first_df = first_data["df"]

    # Get pitchers from the projection season with enough BF
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

    # For each stat, extract observed + projected
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

    # Composite improvement score — handle missing stats gracefully
    base["composite_score"] = 0.0
    base["_total_weight"] = 0.0
    for stat, (weight, sign) in COMPOSITE_WEIGHTS.items():
        delta_col = f"delta_{stat}"
        if delta_col in base.columns:
            stat_sd = base[delta_col].std()
            if stat_sd > 0:
                normalized_delta = base[delta_col] / stat_sd
            else:
                normalized_delta = pd.Series(0.0, index=base.index)
            has_value = base[delta_col].notna()
            base["composite_score"] += (
                weight * sign * normalized_delta.fillna(0)
            )
            base["_total_weight"] += has_value.astype(float) * weight

    # Re-scale so players with partial stats are comparable
    base["composite_score"] = np.where(
        base["_total_weight"] > 0,
        base["composite_score"] / base["_total_weight"],
        0.0,
    )
    base.drop(columns=["_total_weight"], inplace=True)

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
