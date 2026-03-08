"""
Composite hitter projections combining K%, BB%, HR/PA, and xwOBA.

Fits all four stat models, forward-projects posteriors, and produces
a unified projection DataFrame with per-stat deltas and a weighted
composite improvement score.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import build_multi_season_hitter_data
from src.models.hitter_model import (
    STAT_CONFIGS,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)

logger = logging.getLogger(__name__)

# Stats to fit (order matters for composite)
ALL_STATS = ["k_rate", "bb_rate", "hr_rate", "xwoba"]

# Composite weights — xwOBA weighted heavier as best single predictor
# of run production. Signs: positive delta = improvement for hitter.
COMPOSITE_WEIGHTS: dict[str, tuple[float, int]] = {
    # stat: (weight, sign_for_improvement)
    # sign: -1 means lower is better (K%), +1 means higher is better
    "k_rate": (0.15, -1),
    "bb_rate": (0.15, +1),
    "hr_rate": (0.20, +1),
    "xwoba":  (0.50, +1),
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
    """Fit all hitter projection models.

    Parameters
    ----------
    seasons : list[int]
        Training seasons.
    min_pa : int
        Minimum PA per player-season.
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


def project_forward(
    model_results: dict[str, dict[str, Any]],
    from_season: int,
    min_pa: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Forward-project all stats and build composite projections.

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
        One row per player with observed/projected values for each stat,
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

    # For each stat, extract observed + projected
    for stat, res in model_results.items():
        cfg = STAT_CONFIGS[stat]
        data = res["data"]
        trace = res["trace"]

        stat_df = data["df"]
        stat_season = stat_df[stat_df["season"] == from_season]

        # Observed values
        obs_col = f"observed_{stat}"
        proj_col = f"projected_{stat}"
        delta_col = f"delta_{stat}"
        sd_col = f"projected_{stat}_sd"
        ci_lo_col = f"projected_{stat}_2_5"
        ci_hi_col = f"projected_{stat}_97_5"

        obs_map = dict(zip(stat_season["batter_id"], stat_season[cfg.rate_col]))
        base[obs_col] = base["batter_id"].map(obs_map)

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
