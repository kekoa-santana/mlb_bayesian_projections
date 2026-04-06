"""
Shared utilities for hitter and pitcher projection modules.

Extracts duplicated logic from hitter_projections.py and pitcher_projections.py:
- compute_composite: weighted z-score composite scoring
- find_breakouts_and_regressions: top/bottom composite split
- fit_all_models_generic: Bayesian model fitting loop with trace management
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_composite(
    df: pd.DataFrame,
    dimensions: dict[str, tuple[float, list[tuple[str, int, str]]]],
    composite_col: str = "composite_score",
) -> pd.DataFrame:
    """Compute weighted z-score composite from dimension columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns referenced in *dimensions*.
    dimensions : dict
        Mapping of dimension name to (weight, [(col, sign, source)]).
        sign: +1 means higher percentile = better, -1 means lower = better.
        source: "observed" or "projected_delta" (informational only).
    composite_col : str
        Name of the output composite column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with *composite_col* added.
    """
    df[composite_col] = 0.0
    df["_total_weight"] = 0.0

    for dim_name, (weight, components) in dimensions.items():
        dim_z_scores = []

        for col, sign, source in components:
            if col not in df.columns:
                continue
            vals = df[col].astype(float)
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
        df[composite_col] += weight * dim_avg.fillna(0)
        df["_total_weight"] += has_value.astype(float) * weight

    # Re-scale so players with partial stats are comparable
    df[composite_col] = np.where(
        df["_total_weight"] > 0,
        df[composite_col] / df["_total_weight"],
        0.0,
    )
    df.drop(columns=["_total_weight"], inplace=True)

    return df


def find_breakouts_and_regressions(
    projections: pd.DataFrame,
    n_top: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split projections into breakout candidates and regression risks.

    Parameters
    ----------
    projections : pd.DataFrame
        Output of ``project_forward`` (sorted by composite_score descending).
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


def fit_all_models_generic(
    stats: list[str],
    data_builder_fn: Callable[..., pd.DataFrame],
    data_builder_kwargs: dict[str, Any],
    prepare_data_fn: Callable[[pd.DataFrame, str], dict[str, Any]],
    model_fitter_fn: Callable[..., tuple[Any, Any]],
    check_convergence_fn: Callable,
    extract_posteriors_fn: Callable,
    extract_samples_fn: Callable,
    id_col: str,
    player_type: str,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    extract_season: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit Bayesian projection models for a list of stats.

    Generic loop body shared by hitter and pitcher projection pipelines.
    Handles model fitting, convergence checking, posterior extraction,
    optional pre-extraction of rate samples, trace stripping, and memory
    cleanup.

    Parameters
    ----------
    stats : list[str]
        Stat names to fit (e.g. ["k_rate", "bb_rate"]).
    data_builder_fn : callable
        Builds the multi-season DataFrame (e.g. build_multi_season_hitter_data).
    data_builder_kwargs : dict
        Keyword arguments forwarded to *data_builder_fn* (e.g. seasons, min_pa).
    prepare_data_fn : callable
        Prepares model data dict from (df, stat) (e.g. prepare_hitter_data).
    model_fitter_fn : callable
        Fits the model, returns (model, trace) (e.g. fit_hitter_model).
    check_convergence_fn : callable
        Checks convergence from (trace, stat).
    extract_posteriors_fn : callable
        Extracts posteriors from (trace, data).
    extract_samples_fn : callable
        Extracts rate samples for one player. Called with
        (trace, data, **{id_col: pid}, season=..., project_forward=True,
        random_seed=...).
    id_col : str
        Player ID column name ("batter_id" or "pitcher_id").
    player_type : str
        Label for logging ("hitter" or "pitcher").
    draws, tune, chains, random_seed
        MCMC sampling parameters.
    extract_season : int | None
        If set, pre-extract forward-projected rate samples for all players
        in this season immediately after fitting, then free the MCMC trace
        to save memory.

    Returns
    -------
    dict[str, dict]
        Keyed by stat name.  Always contains "data", "convergence",
        "posteriors".  Contains "rate_samples" (dict[int, ndarray]) when
        *extract_season* is set, otherwise "trace" (InferenceData).
    """
    df = data_builder_fn(**data_builder_kwargs)
    logger.info("Loaded %d %s-seasons for projection", len(df), player_type)

    results: dict[str, dict[str, Any]] = {}

    for stat in stats:
        logger.info("=" * 50)
        logger.info("Fitting %s %s model", player_type, stat)

        data = prepare_data_fn(df, stat)
        model, trace = model_fitter_fn(
            data,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        conv = check_convergence_fn(trace, stat)
        posteriors = extract_posteriors_fn(trace, data)

        # Pre-extract rate samples so we can free the trace entirely
        rate_samples: dict[int, np.ndarray] | None = None
        if extract_season is not None:
            rate_samples = {}
            stat_df = data["df"]
            active_ids = stat_df[
                stat_df["season"] == extract_season
            ][id_col].unique()
            for pid in active_ids:
                try:
                    rate_samples[int(pid)] = extract_samples_fn(
                        trace, data,
                        **{id_col: int(pid)},
                        season=extract_season,
                        project_forward=True,
                        random_seed=random_seed,
                    )
                except ValueError:
                    continue
            logger.info(
                "Pre-extracted %s samples for %d %ss",
                stat, len(rate_samples), player_type,
            )

        results[stat] = {
            "data": data,
            "convergence": conv,
            "posteriors": posteriors,
        }
        if rate_samples is not None:
            results[stat]["rate_samples"] = rate_samples
        else:
            # Keep stripped trace for backward compat (backtests etc.)
            for group in ("posterior_predictive", "sample_stats", "observed_data"):
                if hasattr(trace, group):
                    delattr(trace, group)
            results[stat]["trace"] = trace

        # Free model + trace (if pre-extracted) to reclaim memory
        del model
        if rate_samples is not None:
            del trace
        gc.collect()

        logger.info(
            "%s %s: converged=%s, r_hat=%.4f",
            player_type, stat, conv["converged"], conv["max_rhat"],
        )

    return results
