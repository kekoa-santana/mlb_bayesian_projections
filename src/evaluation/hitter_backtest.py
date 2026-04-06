"""
Walk-forward backtesting for the generalized hitter projection model.

Supports all stats in STAT_CONFIGS (K%, BB%, HR/PA, wOBA, xwOBA).
Evaluates Bayesian projections vs Marcel baseline across folds,
with CRPS, PPC diagnostics, and Bayes-Marcel ensemble.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import brier_score_loss

from src.data.feature_eng import build_multi_season_hitter_data
from src.evaluation.baselines import marcel_rate_projection
from src.evaluation.ensemble import (
    apply_ensemble,
    compute_ensemble_metrics,
    fit_ensemble_weight,
)
from src.evaluation.metrics import (
    compute_bayes_vs_marcel_crps,
    compute_coverage,
    compute_coverage_from_sd,
    compute_crps_single,
    compute_posterior_calibration_t,
    extract_ppc_summary,
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


# ---------------------------------------------------------------------------
# Marcel baseline (generalized)
# ---------------------------------------------------------------------------
def marcel_hitter(
    df_history: pd.DataFrame,
    stat: str,
    league_avg: float | None = None,
) -> pd.DataFrame:
    """Marcel projection for any rate stat.

    Weights recent seasons 5/4/3, regresses toward league mean.

    Parameters
    ----------
    df_history : pd.DataFrame
        Multi-season data with batter_id, season, pa, and the stat's
        count and rate columns.
    stat : str
        Stat key from STAT_CONFIGS.
    league_avg : float | None
        League-average rate. Defaults to weighted mean of history.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, marcel_{stat}, reliability, weighted_pa.
    """
    result = marcel_rate_projection(
        df_history,
        id_col="batter_id",
        trials_col="pa",
        stat_configs={stat: STAT_CONFIGS[stat]},
    )
    return result[stat]


# ---------------------------------------------------------------------------
# Walk-forward for a single stat
# ---------------------------------------------------------------------------
def walk_forward_stat_backtest(
    train_seasons: list[int],
    test_season: int,
    stat: str,
    min_pa_train: int = 50,
    min_pa_test: int = 100,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train on historical seasons, project forward, evaluate on test season.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to train on.
    test_season : int
        Season to evaluate against.
    stat : str
        Stat key from STAT_CONFIGS.
    min_pa_train, min_pa_test
        Minimum PA thresholds.
    draws, tune, chains, random_seed
        MCMC sampling parameters.

    Returns
    -------
    dict
        Evaluation metrics, comparison DataFrame, convergence info,
        proj_samples, and ppc_summary.
    """
    cfg = STAT_CONFIGS[stat]
    logger.info("Walk-forward %s: train=%s, test=%d", stat, train_seasons, test_season)

    # Load data
    all_seasons = sorted(set(train_seasons) | {test_season})
    df_all = build_multi_season_hitter_data(all_seasons, min_pa=min_pa_train)

    df_train = df_all[df_all["season"].isin(train_seasons)]
    df_test = df_all[
        (df_all["season"] == test_season) & (df_all["pa"] >= min_pa_test)
    ]

    # Common batters
    common = set(df_train["batter_id"]) & set(df_test["batter_id"])
    logger.info("Common batters: %d", len(common))
    if len(common) < 10:
        logger.warning("Very few common batters — results may be unreliable")

    # Fit model on train data only (common batters for evaluation)
    df_model = df_train[df_train["batter_id"].isin(common)].copy()
    data = prepare_hitter_data(df_model, stat)
    model, trace = fit_hitter_model(
        data, draws=draws, tune=tune, chains=chains, random_seed=random_seed,
    )

    conv = check_convergence(trace, stat)

    # Extract forward-projected posteriors from last training season
    last_train = max(train_seasons)
    proj_means: dict[int, float] = {}
    proj_sds: dict[int, float] = {}
    proj_lo: dict[int, float] = {}
    proj_hi: dict[int, float] = {}
    proj_samples: dict[int, np.ndarray] = {}

    for batter_id in common:
        try:
            samples = extract_rate_samples(
                trace, data, batter_id, last_train,
                project_forward=True, random_seed=random_seed,
            )
            proj_means[batter_id] = float(np.mean(samples))
            proj_sds[batter_id] = float(np.std(samples))
            proj_lo[batter_id] = float(np.percentile(samples, 2.5))
            proj_hi[batter_id] = float(np.percentile(samples, 97.5))
            proj_samples[batter_id] = samples
        except ValueError:
            continue

    # Marcel baseline
    marcel = marcel_hitter(
        df_train[df_train["batter_id"].isin(common)], stat,
    )

    # Build comparison DataFrame
    test_actuals = df_test[df_test["batter_id"].isin(common)][
        ["batter_id", "batter_name", "pa", cfg.rate_col]
    ].rename(columns={cfg.rate_col: f"actual_{stat}", "pa": "actual_pa"})

    comp = test_actuals.copy()
    comp[f"bayes_{stat}"] = comp["batter_id"].map(proj_means)
    comp[f"bayes_{stat}_sd"] = comp["batter_id"].map(proj_sds)
    comp["ci_95_lo"] = comp["batter_id"].map(proj_lo)
    comp["ci_95_hi"] = comp["batter_id"].map(proj_hi)
    comp = comp.merge(marcel[["batter_id", f"marcel_{stat}", "weighted_pa"]],
                      on="batter_id", how="inner")
    comp = comp.dropna(subset=[f"bayes_{stat}", f"actual_{stat}"])

    # Compute metrics
    actual = comp[f"actual_{stat}"].values
    bayes = comp[f"bayes_{stat}"].values
    marcel_pred = comp[f"marcel_{stat}"].values

    ci_lo = comp["ci_95_lo"].values
    ci_hi = comp["ci_95_hi"].values
    coverage_95 = compute_coverage(actual, ci_lo, ci_hi)

    # 80% CI coverage (approximate from mean ± 1.282 * SD)
    bayes_sd = comp[f"bayes_{stat}_sd"].values if f"bayes_{stat}_sd" in comp.columns else None
    if bayes_sd is not None:
        coverage_80 = compute_coverage_from_sd(actual, bayes, bayes_sd, level=0.80)
    else:
        coverage_80 = float("nan")

    # Compute optimal calibration T from 80% coverage
    optimal_cal_t = compute_posterior_calibration_t(coverage_80)

    # Brier score (above league average?) — reuse stored samples
    league_avg = cfg.league_avg
    actual_above = (actual > league_avg).astype(float)
    bayes_prob_above = np.array([
        float(np.mean(proj_samples[bid] > league_avg))
        if bid in proj_samples else 0.5
        for bid in comp["batter_id"]
    ])

    # Marcel Brier via beta approximation
    strength = np.maximum(comp["weighted_pa"].values.astype(float), 1.0)
    alpha = 1.0 + (marcel_pred * strength)
    beta_param = 1.0 + ((1.0 - marcel_pred) * strength)
    if cfg.likelihood == "binomial":
        marcel_prob_above = beta_dist.sf(league_avg, alpha, beta_param)
    else:
        marcel_prob_above = (marcel_pred > league_avg).astype(float)

    bayes_brier = float(brier_score_loss(actual_above, bayes_prob_above))
    marcel_brier = float(brier_score_loss(actual_above, marcel_prob_above))

    # CRPS: Bayes via stored samples, Marcel via Beta sampling
    rng = np.random.default_rng(random_seed + 1)
    bayes_crps, marcel_crps = compute_bayes_vs_marcel_crps(
        comp, stat, proj_samples,
        id_col="batter_id", weight_col="weighted_pa",
        rng=rng, compute_crps_single_fn=compute_crps_single,
        likelihood=cfg.likelihood,
    )

    # PPC: posterior predictive check
    ppc_summary = extract_ppc_summary(trace, data, cfg, label=stat) or {}

    logger.info(
        "%s: 80%% CI coverage: %.1f%%, 95%% CI coverage: %.1f%% (optimal T=%.3f)",
        stat, coverage_80 * 100, coverage_95 * 100, optimal_cal_t,
    )
    logger.info(
        "%s: CRPS Bayes=%.4f, Marcel=%.4f", stat, bayes_crps, marcel_crps,
    )

    return {
        "stat": stat,
        "test_season": test_season,
        "n_players": len(comp),
        "coverage_80": coverage_80,
        "coverage_95": coverage_95,
        "calibration_t": optimal_cal_t,
        "bayes_brier": bayes_brier,
        "marcel_brier": marcel_brier,
        "bayes_crps": bayes_crps,
        "marcel_crps": marcel_crps,
        "convergence": conv,
        "comparison_df": comp,
        "proj_samples": proj_samples,
        "ppc_summary": ppc_summary,
    }


# ---------------------------------------------------------------------------
# Multi-fold, multi-stat backtest
# ---------------------------------------------------------------------------
def run_hitter_backtest(
    stats: list[str] | None = None,
    folds: list[dict[str, Any]] | None = None,
    **sampling_kwargs: Any,
) -> pd.DataFrame:
    """Run walk-forward backtesting across stats and folds.

    Includes expanding-window ensemble: fold 1 uses w=0.5, later folds
    fit w on residuals from prior folds (no leakage).

    Parameters
    ----------
    stats : list[str] | None
        Stats to backtest. Defaults to all in STAT_CONFIGS.
    folds : list[dict]
        Each dict has 'train_seasons' and 'test_season'.
    **sampling_kwargs
        Passed to ``walk_forward_stat_backtest``.

    Returns
    -------
    pd.DataFrame
        Summary metrics across all stats and folds.
    """
    if stats is None:
        stats = list(STAT_CONFIGS.keys())
    if folds is None:
        folds = [
            {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
            {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
            {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
        ]

    results = []
    for stat in stats:
        cfg = STAT_CONFIGS[stat]

        # First pass: run all folds, store full result dicts
        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            logger.info(
                "=== %s: train=%s → test=%d ===",
                stat, fold["train_seasons"], fold["test_season"],
            )
            metrics = walk_forward_stat_backtest(
                train_seasons=fold["train_seasons"],
                test_season=fold["test_season"],
                stat=stat,
                **sampling_kwargs,
            )
            fold_results.append(metrics)

        # Second pass: expanding-window ensemble
        for i, (fold, metrics) in enumerate(zip(folds, fold_results)):
            comp = metrics["comparison_df"]
            proj_samps = metrics["proj_samples"]

            # Fit ensemble weight from prior folds (no leakage)
            if i == 0:
                w = 0.5  # no prior data
            else:
                # Pool residuals from all prior folds
                prior_actuals = []
                prior_bayes = []
                prior_marcel = []
                for j in range(i):
                    prior_comp = fold_results[j]["comparison_df"]
                    prior_actuals.append(prior_comp[f"actual_{stat}"].values)
                    prior_bayes.append(prior_comp[f"bayes_{stat}"].values)
                    prior_marcel.append(prior_comp[f"marcel_{stat}"].values)
                w = fit_ensemble_weight(
                    np.concatenate(prior_actuals),
                    np.concatenate(prior_bayes),
                    np.concatenate(prior_marcel),
                )

            # Apply ensemble
            comp_ens = apply_ensemble(comp, w, stat)
            ens_metrics = compute_ensemble_metrics(
                comp_ens, stat, cfg.league_avg, proj_samps, id_col="batter_id",
            )

            logger.info(
                "%s fold %d: ensemble w=%.2f, Brier=%.4f, coverage_95=%.3f",
                stat, i + 1, w, ens_metrics["ensemble_brier"],
                ens_metrics["ensemble_coverage_95"],
            )

            results.append({
                "stat": stat,
                "test_season": metrics["test_season"],
                "n_players": metrics["n_players"],
                "coverage_80": metrics["coverage_80"],
                "coverage_95": metrics["coverage_95"],
                "calibration_t": metrics["calibration_t"],
                "bayes_brier": metrics["bayes_brier"],
                "marcel_brier": metrics["marcel_brier"],
                "bayes_crps": metrics["bayes_crps"],
                "marcel_crps": metrics["marcel_crps"],
                "ensemble_w": w,
                "ensemble_brier": ens_metrics["ensemble_brier"],
                "ensemble_coverage_95": ens_metrics["ensemble_coverage_95"],
                "converged": metrics["convergence"]["converged"],
            })

    summary = pd.DataFrame(results)
    logger.info("\n=== Hitter Backtest Summary ===\n%s", summary.to_string())
    return summary
