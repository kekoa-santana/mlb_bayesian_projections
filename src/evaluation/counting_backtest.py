"""
Walk-forward backtesting for counting stat projections.

Tests hitter counting stats (K, BB, HR, SB) and pitcher counting stats
(K, BB, Outs) across walk-forward folds.  Compares Bayesian rate × playing
time distributions vs Marcel baseline.

Metrics: correlation, MAPE, 80/95% coverage.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_hitter_data,
    build_multi_season_hitter_extended,
    build_multi_season_pitcher_data,
    build_multi_season_pitcher_extended,
)
from src.models.counting_projections import (
    HITTER_COUNTING_STATS,
    PITCHER_COUNTING_STATS,
    marcel_counting_hitter,
    marcel_counting_pitcher,
    project_hitter_counting,
    project_pitcher_counting,
)
from src.models.pa_model import compute_hitter_pa_priors

logger = logging.getLogger(__name__)

# Mapping from counting stat name → actual column in extended data
HITTER_ACTUAL_MAP = {
    "total_k": "k",
    "total_bb": "bb",
    "total_hr": "hr",
    "total_sb": "sb",
}

PITCHER_ACTUAL_MAP = {
    "total_k": "k",
    "total_bb": "bb",
    "total_outs": "outs",
    "total_er": "earned_runs",
}


def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lo_80: np.ndarray | None = None,
    hi_80: np.ndarray | None = None,
    lo_95: np.ndarray | None = None,
    hi_95: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute regression metrics for counting stat predictions."""
    n = len(actual)
    if n == 0:
        return {}

    residuals = actual - predicted

    # Correlation
    if np.std(actual) > 0 and np.std(predicted) > 0:
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        corr = 0.0

    # MAPE (avoid div/0 for players with 0 actuals)
    nonzero = actual > 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(residuals[nonzero]) / actual[nonzero]))
    else:
        mape = float("nan")

    metrics = {
        "n": n,
        "correlation": corr,
        "mape": mape,
    }

    # Coverage
    if lo_80 is not None and hi_80 is not None:
        cov_80 = float(np.mean((actual >= lo_80) & (actual <= hi_80)))
        metrics["coverage_80"] = cov_80
    if lo_95 is not None and hi_95 is not None:
        cov_95 = float(np.mean((actual >= lo_95) & (actual <= hi_95)))
        metrics["coverage_95"] = cov_95

    return metrics


# ---------------------------------------------------------------------------
# Hitter counting stat backtest (single fold)
# ---------------------------------------------------------------------------
def walk_forward_hitter_counting(
    train_seasons: list[int],
    test_season: int,
    stats: list[str] | None = None,
    min_pa_train: int = 100,
    min_pa_test: int = 200,
    n_draws: int = 4000,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train on historical seasons, project counting stats, evaluate on test season.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons for training.
    test_season : int
        Season to evaluate.
    stats : list[str] | None
        Counting stats to evaluate (defaults to all HITTER_COUNTING_STATS).
    min_pa_train, min_pa_test : int
        Minimum PA thresholds.
    n_draws : int
        Monte Carlo samples for counting distributions.
    draws, tune, chains : int
        MCMC sampling parameters for rate models.
    random_seed : int

    Returns
    -------
    dict
        Per-stat metrics, comparison DataFrames, Marcel comparison.
    """
    from src.models.hitter_projections import fit_all_models as fit_hitter_models

    if stats is None:
        stats = list(HITTER_COUNTING_STATS.keys())

    last_train = max(train_seasons)
    all_seasons = sorted(set(train_seasons) | {test_season})
    logger.info(
        "Hitter counting backtest: train=%s, test=%d, stats=%s",
        train_seasons, test_season, stats,
    )

    # Build extended data (has sb, games, hr, etc.)
    hitter_ext = build_multi_season_hitter_extended(all_seasons, min_pa=1)

    # Build standard hitter data for Bayesian rate models
    hitter_data = build_multi_season_hitter_data(all_seasons, min_pa=min_pa_train)

    # Fit Bayesian rate models on train data only
    rate_results = fit_hitter_models(
        seasons=train_seasons,
        min_pa=min_pa_train,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )

    # PA priors from train data
    train_ext = hitter_ext[hitter_ext["season"].isin(train_seasons)]
    pa_priors = compute_hitter_pa_priors(
        train_ext, from_season=last_train, min_pa=min_pa_train,
    )

    # Project counting stats (using train data + train-period rate models)
    # We pass from_season=last_train so it projects from the last training season
    bayes_proj = project_hitter_counting(
        rate_model_results=rate_results,
        pa_priors=pa_priors,
        hitter_extended=train_ext,
        from_season=last_train,
        n_draws=n_draws,
        min_pa=min_pa_train,
        random_seed=random_seed,
    )

    # Marcel baseline (also using only train data)
    marcel_proj = marcel_counting_hitter(
        train_ext, from_season=last_train, min_pa=min_pa_train,
    )

    # Get test season actuals
    test_df = hitter_ext[
        (hitter_ext["season"] == test_season)
        & (hitter_ext["pa"] >= min_pa_test)
    ].copy()

    if test_df.empty:
        logger.warning("No test data for season %d", test_season)
        return {"stats": {}, "test_season": test_season, "n_players": 0}

    # Find common players (in both training projections and test actuals)
    common_bayes = set(bayes_proj["batter_id"]) & set(test_df["batter_id"])
    common_marcel = set(marcel_proj["batter_id"]) & set(test_df["batter_id"])
    common = common_bayes & common_marcel
    logger.info("Common players: %d (Bayes: %d, Marcel: %d, test: %d)",
                len(common), len(common_bayes), len(common_marcel), len(test_df))

    stat_results = {}
    for stat_name in stats:
        cfg = HITTER_COUNTING_STATS[stat_name]
        actual_col = HITTER_ACTUAL_MAP[stat_name]

        if actual_col not in test_df.columns:
            logger.warning("Column %s not in test data, skipping %s", actual_col, stat_name)
            continue

        # Build comparison frame
        test_actual = test_df[test_df["batter_id"].isin(common)][
            ["batter_id", "batter_name", "pa", "games", actual_col]
        ].copy()
        test_actual = test_actual.rename(columns={actual_col: "actual"})

        # Merge Bayes projections
        bayes_cols = ["batter_id"] + [
            c for c in bayes_proj.columns
            if c.startswith(f"{stat_name}_")
        ]
        bayes_sub = bayes_proj[bayes_proj["batter_id"].isin(common)][bayes_cols].copy()

        comp = test_actual.merge(bayes_sub, on="batter_id", how="inner")

        # Merge Marcel
        marcel_col = f"marcel_{actual_col}"
        if marcel_col in marcel_proj.columns:
            comp = comp.merge(
                marcel_proj[marcel_proj["batter_id"].isin(common)][["batter_id", marcel_col]],
                on="batter_id", how="inner",
            )
        else:
            comp[marcel_col] = np.nan

        if comp.empty:
            logger.warning("No data for %s after merge", stat_name)
            continue

        actual_vals = comp["actual"].values.astype(float)
        bayes_pred = comp[f"{stat_name}_mean"].values
        marcel_pred = comp[marcel_col].values if marcel_col in comp.columns else np.full(len(comp), np.nan)

        # Bayes metrics (with coverage from p10/p90 and p2.5/p97.5)
        lo_80 = comp[f"{stat_name}_p10"].values if f"{stat_name}_p10" in comp.columns else None
        hi_80 = comp[f"{stat_name}_p90"].values if f"{stat_name}_p90" in comp.columns else None
        lo_95 = comp[f"{stat_name}_p2_5"].values if f"{stat_name}_p2_5" in comp.columns else None
        hi_95 = comp[f"{stat_name}_p97_5"].values if f"{stat_name}_p97_5" in comp.columns else None

        bayes_metrics = _compute_metrics(actual_vals, bayes_pred, lo_80, hi_80, lo_95, hi_95)

        # Marcel metrics (no coverage — point estimate only)
        marcel_metrics = _compute_metrics(actual_vals, marcel_pred)

        logger.info(
            "%s: Bayes corr=%.3f, coverage 80/95=%.0f%%/%.0f%%",
            stat_name,
            bayes_metrics.get("correlation", 0),
            bayes_metrics.get("coverage_80", 0) * 100,
            bayes_metrics.get("coverage_95", 0) * 100,
        )

        stat_results[stat_name] = {
            "bayes": bayes_metrics,
            "marcel": marcel_metrics,
            "comparison_df": comp,
        }

    return {
        "stats": stat_results,
        "test_season": test_season,
        "n_players": len(common),
    }


# ---------------------------------------------------------------------------
# Pitcher counting stat backtest (single fold)
# ---------------------------------------------------------------------------
def walk_forward_pitcher_counting(
    train_seasons: list[int],
    test_season: int,
    stats: list[str] | None = None,
    min_bf_train: int = 100,
    min_bf_test: int = 200,
    n_draws: int = 4000,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train on historical seasons, project pitcher counting stats, evaluate.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons for training.
    test_season : int
        Season to evaluate.
    stats : list[str] | None
        Counting stats to evaluate (defaults to all PITCHER_COUNTING_STATS).
    min_bf_train, min_bf_test : int
        Minimum BF thresholds.
    n_draws : int
        Monte Carlo samples for counting distributions.
    draws, tune, chains : int
        MCMC sampling parameters.
    random_seed : int

    Returns
    -------
    dict
        Per-stat metrics, comparison DataFrames.
    """
    from src.models.pitcher_projections import fit_all_models as fit_pitcher_models

    if stats is None:
        stats = list(PITCHER_COUNTING_STATS.keys())

    last_train = max(train_seasons)
    all_seasons = sorted(set(train_seasons) | {test_season})
    logger.info(
        "Pitcher counting backtest: train=%s, test=%d, stats=%s",
        train_seasons, test_season, stats,
    )

    # Build extended pitcher data
    pitcher_ext = build_multi_season_pitcher_extended(all_seasons, min_bf=9)

    # Fit Bayesian rate models on train data only
    rate_results = fit_pitcher_models(
        seasons=train_seasons,
        min_bf=min_bf_train,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )

    # Project counting stats from last training season
    train_ext = pitcher_ext[pitcher_ext["season"].isin(train_seasons)]

    bayes_proj = project_pitcher_counting(
        rate_model_results=rate_results,
        pitcher_extended=train_ext,
        from_season=last_train,
        n_draws=n_draws,
        min_bf=min_bf_train,
        random_seed=random_seed,
    )

    # Marcel baseline
    marcel_proj = marcel_counting_pitcher(
        train_ext, from_season=last_train, min_bf=min_bf_train,
    )

    # Test actuals
    test_df = pitcher_ext[
        (pitcher_ext["season"] == test_season)
        & (pitcher_ext["batters_faced"] >= min_bf_test)
    ].copy()

    if test_df.empty:
        logger.warning("No test data for season %d", test_season)
        return {"stats": {}, "test_season": test_season, "n_players": 0}

    common_bayes = set(bayes_proj["pitcher_id"]) & set(test_df["pitcher_id"])
    common_marcel = set(marcel_proj["pitcher_id"]) & set(test_df["pitcher_id"])
    common = common_bayes & common_marcel
    logger.info("Common pitchers: %d (Bayes: %d, Marcel: %d, test: %d)",
                len(common), len(common_bayes), len(common_marcel), len(test_df))

    stat_results = {}
    for stat_name in stats:
        cfg = PITCHER_COUNTING_STATS[stat_name]
        actual_col = PITCHER_ACTUAL_MAP[stat_name]

        if actual_col not in test_df.columns:
            logger.warning("Column %s not in test data, skipping %s", actual_col, stat_name)
            continue

        # Build comparison
        id_col = "pitcher_id"
        name_col = "pitcher_name"
        test_actual = test_df[test_df[id_col].isin(common)][
            [id_col, name_col, "batters_faced", "games", actual_col]
        ].copy()
        test_actual = test_actual.rename(columns={actual_col: "actual"})

        bayes_cols = [id_col] + [
            c for c in bayes_proj.columns if c.startswith(f"{stat_name}_")
        ]
        bayes_sub = bayes_proj[bayes_proj[id_col].isin(common)][bayes_cols].copy()

        comp = test_actual.merge(bayes_sub, on=id_col, how="inner")

        marcel_col = f"marcel_{actual_col}"
        if marcel_col in marcel_proj.columns:
            comp = comp.merge(
                marcel_proj[marcel_proj[id_col].isin(common)][[id_col, marcel_col]],
                on=id_col, how="inner",
            )
        else:
            comp[marcel_col] = np.nan

        if comp.empty:
            logger.warning("No data for %s after merge", stat_name)
            continue

        actual_vals = comp["actual"].values.astype(float)
        bayes_pred = comp[f"{stat_name}_mean"].values
        marcel_pred = comp[marcel_col].values

        lo_80 = comp.get(f"{stat_name}_p10", pd.Series(dtype=float)).values if f"{stat_name}_p10" in comp.columns else None
        hi_80 = comp.get(f"{stat_name}_p90", pd.Series(dtype=float)).values if f"{stat_name}_p90" in comp.columns else None
        lo_95 = comp.get(f"{stat_name}_p2_5", pd.Series(dtype=float)).values if f"{stat_name}_p2_5" in comp.columns else None
        hi_95 = comp.get(f"{stat_name}_p97_5", pd.Series(dtype=float)).values if f"{stat_name}_p97_5" in comp.columns else None

        bayes_metrics = _compute_metrics(actual_vals, bayes_pred, lo_80, hi_80, lo_95, hi_95)
        marcel_metrics = _compute_metrics(actual_vals, marcel_pred)

        logger.info(
            "%s: Bayes corr=%.3f, coverage 80/95=%.0f%%/%.0f%%",
            stat_name,
            bayes_metrics.get("correlation", 0),
            bayes_metrics.get("coverage_80", 0) * 100,
            bayes_metrics.get("coverage_95", 0) * 100,
        )

        stat_results[stat_name] = {
            "bayes": bayes_metrics,
            "marcel": marcel_metrics,
            "comparison_df": comp,
        }

    return {
        "stats": stat_results,
        "test_season": test_season,
        "n_players": len(common),
    }


# ---------------------------------------------------------------------------
# Multi-fold runners
# ---------------------------------------------------------------------------
DEFAULT_HITTER_FOLDS = [
    {"train_seasons": list(range(2018, 2022)), "test_season": 2022},
    {"train_seasons": list(range(2018, 2023)), "test_season": 2023},
    {"train_seasons": list(range(2018, 2024)), "test_season": 2024},
    {"train_seasons": list(range(2018, 2025)), "test_season": 2025},
]

DEFAULT_PITCHER_FOLDS = DEFAULT_HITTER_FOLDS


def run_hitter_counting_backtest(
    stats: list[str] | None = None,
    folds: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run walk-forward counting stat backtest for hitters across folds.

    Parameters
    ----------
    stats : list[str] | None
        Counting stats to test. Defaults to all HITTER_COUNTING_STATS.
    folds : list[dict]
        Each has 'train_seasons' and 'test_season'.
    **kwargs
        Passed to walk_forward_hitter_counting.

    Returns
    -------
    pd.DataFrame
        Summary metrics per stat per fold.
    """
    if folds is None:
        folds = DEFAULT_HITTER_FOLDS
    if stats is None:
        stats = list(HITTER_COUNTING_STATS.keys())

    rows = []
    for fold in folds:
        result = walk_forward_hitter_counting(
            train_seasons=fold["train_seasons"],
            test_season=fold["test_season"],
            stats=stats,
            **kwargs,
        )
        for stat_name, sr in result["stats"].items():
            rows.append({
                "stat": stat_name,
                "test_season": result["test_season"],
                "n_players": sr["bayes"].get("n", 0),
                "bayes_corr": sr["bayes"].get("correlation", float("nan")),
                "marcel_corr": sr["marcel"].get("correlation", float("nan")),
                "bayes_mape": sr["bayes"].get("mape", float("nan")),
                "marcel_mape": sr["marcel"].get("mape", float("nan")),
                "coverage_80": sr["bayes"].get("coverage_80", float("nan")),
                "coverage_95": sr["bayes"].get("coverage_95", float("nan")),
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        logger.info("\n=== Hitter Counting Backtest Summary ===\n%s", summary.to_string())
    return summary


def run_pitcher_counting_backtest(
    stats: list[str] | None = None,
    folds: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run walk-forward counting stat backtest for pitchers across folds.

    Parameters
    ----------
    stats : list[str] | None
        Counting stats to test. Defaults to all PITCHER_COUNTING_STATS.
    folds : list[dict]
        Each has 'train_seasons' and 'test_season'.
    **kwargs
        Passed to walk_forward_pitcher_counting.

    Returns
    -------
    pd.DataFrame
        Summary metrics per stat per fold.
    """
    if folds is None:
        folds = DEFAULT_PITCHER_FOLDS
    if stats is None:
        stats = list(PITCHER_COUNTING_STATS.keys())

    rows = []
    for fold in folds:
        result = walk_forward_pitcher_counting(
            train_seasons=fold["train_seasons"],
            test_season=fold["test_season"],
            stats=stats,
            **kwargs,
        )
        for stat_name, sr in result["stats"].items():
            rows.append({
                "stat": stat_name,
                "test_season": result["test_season"],
                "n_players": sr["bayes"].get("n", 0),
                "bayes_corr": sr["bayes"].get("correlation", float("nan")),
                "marcel_corr": sr["marcel"].get("correlation", float("nan")),
                "bayes_mape": sr["bayes"].get("mape", float("nan")),
                "marcel_mape": sr["marcel"].get("mape", float("nan")),
                "coverage_80": sr["bayes"].get("coverage_80", float("nan")),
                "coverage_95": sr["bayes"].get("coverage_95", float("nan")),
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        logger.info("\n=== Pitcher Counting Backtest Summary ===\n%s", summary.to_string())
    return summary
