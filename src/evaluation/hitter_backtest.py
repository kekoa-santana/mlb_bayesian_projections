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
from src.evaluation.ensemble import (
    apply_ensemble,
    compute_ensemble_metrics,
    fit_ensemble_weight,
)
from src.evaluation.metrics import (
    compute_crps_single,
    compute_ppc_pvalues,
    summarize_ppc_calibration,
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
    cfg = STAT_CONFIGS[stat]
    weights = {0: 5, 1: 4, 2: 3}
    available_seasons = sorted(df_history["season"].unique(), reverse=True)

    if league_avg is None:
        if cfg.likelihood == "binomial":
            total_count = df_history[cfg.count_col].sum()
            total_pa = df_history[cfg.trials_col].sum()
            league_avg = total_count / total_pa if total_pa > 0 else cfg.league_avg
        else:
            league_avg = df_history[cfg.rate_col].mean()

    # Regression PA constant — more stable stats need less regression
    regression_pa = {
        "k_rate": 1200, "bb_rate": 1200,
        "gb_rate": 800, "fb_rate": 800, "hr_per_fb": 1500,
    }
    reg_pa = regression_pa.get(stat, 1200)

    records = []
    for batter_id, group in df_history.groupby("batter_id"):
        weighted_val = 0.0
        weighted_pa = 0.0

        for offset, season in enumerate(available_seasons):
            if offset > 2:
                break
            w = weights.get(offset, 0)
            row = group[group["season"] == season]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            pa = float(r[cfg.trials_col])

            if cfg.likelihood == "binomial":
                weighted_val += w * float(r[cfg.count_col])
            else:
                weighted_val += w * float(r[cfg.rate_col]) * pa
            weighted_pa += w * pa

        if weighted_pa == 0:
            continue

        raw_rate = weighted_val / weighted_pa
        reliability = weighted_pa / (weighted_pa + reg_pa)
        marcel_rate = reliability * raw_rate + (1 - reliability) * league_avg

        records.append({
            "batter_id": batter_id,
            f"marcel_{stat}": marcel_rate,
            "reliability": reliability,
            "weighted_pa": weighted_pa,
        })

    return pd.DataFrame(records)


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

    bayes_mae = float(np.mean(np.abs(actual - bayes)))
    marcel_mae = float(np.mean(np.abs(actual - marcel_pred)))
    bayes_rmse = float(np.sqrt(np.mean((actual - bayes) ** 2)))
    marcel_rmse = float(np.sqrt(np.mean((actual - marcel_pred) ** 2)))

    ci_lo = comp["ci_95_lo"].values
    ci_hi = comp["ci_95_hi"].values
    coverage_95 = float(np.mean((actual >= ci_lo) & (actual <= ci_hi)))

    mae_imp = (marcel_mae - bayes_mae) / marcel_mae * 100 if marcel_mae > 0 else 0
    rmse_imp = (marcel_rmse - bayes_rmse) / marcel_rmse * 100 if marcel_rmse > 0 else 0

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
    bayes_crps_vals = []
    marcel_crps_vals = []
    for _, row in comp.iterrows():
        bid = int(row["batter_id"])
        act = float(row[f"actual_{stat}"])

        # Bayes CRPS
        if bid in proj_samples:
            bayes_crps_vals.append(compute_crps_single(act, proj_samples[bid]))

        # Marcel CRPS via Beta(alpha, beta) draws
        m_rate = float(row[f"marcel_{stat}"])
        w_pa = float(row["weighted_pa"])
        if cfg.likelihood == "binomial":
            a = 1.0 + m_rate * max(w_pa, 1.0)
            b = 1.0 + (1.0 - m_rate) * max(w_pa, 1.0)
            marcel_draws = rng.beta(a, b, size=4000)
        else:
            # Normal approximation for continuous stats
            marcel_draws = rng.normal(m_rate, max(0.01, abs(m_rate) * 0.1), size=4000)
        marcel_crps_vals.append(compute_crps_single(act, marcel_draws))

    bayes_crps = float(np.mean(bayes_crps_vals)) if bayes_crps_vals else np.nan
    marcel_crps = float(np.mean(marcel_crps_vals)) if marcel_crps_vals else np.nan

    # PPC: posterior predictive check
    ppc_summary: dict[str, Any] = {}
    try:
        if hasattr(trace, "posterior_predictive") and "obs" in trace.posterior_predictive:
            ppc_obs = trace.posterior_predictive["obs"].values  # (chains, draws, n_obs)
            n_chains, n_draws_per, n_obs = ppc_obs.shape
            ppc_flat = ppc_obs.reshape(n_chains * n_draws_per, n_obs)

            # Observed data for PPC comparison
            if cfg.likelihood == "binomial":
                observed_for_ppc = data["counts"]
            else:
                observed_for_ppc = data["y_obs"]

            pvalues = compute_ppc_pvalues(ppc_flat, observed_for_ppc)
            ppc_summary = summarize_ppc_calibration(pvalues)
            logger.info(
                "%s PPC: KS stat=%.3f (p=%.3f), outliers=%.1f%%",
                stat, ppc_summary["ks_stat"], ppc_summary["ks_pvalue"],
                ppc_summary["pct_outliers"],
            )
    except Exception as e:
        logger.warning("PPC computation failed for %s: %s", stat, e)

    logger.info(
        "%s: Bayes MAE=%.4f, Marcel MAE=%.4f (improvement: %.1f%%)",
        stat, bayes_mae, marcel_mae, mae_imp,
    )
    logger.info(
        "%s: 95%% CI coverage: %.1f%%", stat, coverage_95 * 100,
    )
    logger.info(
        "%s: CRPS Bayes=%.4f, Marcel=%.4f", stat, bayes_crps, marcel_crps,
    )

    return {
        "stat": stat,
        "test_season": test_season,
        "n_players": len(comp),
        "bayes_mae": bayes_mae,
        "marcel_mae": marcel_mae,
        "mae_improvement_pct": mae_imp,
        "bayes_rmse": bayes_rmse,
        "marcel_rmse": marcel_rmse,
        "rmse_improvement_pct": rmse_imp,
        "coverage_95": coverage_95,
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
                "%s fold %d: ensemble w=%.2f, MAE=%.4f (Bayes=%.4f, Marcel=%.4f)",
                stat, i + 1, w, ens_metrics["ensemble_mae"],
                metrics["bayes_mae"], metrics["marcel_mae"],
            )

            results.append({
                "stat": stat,
                "test_season": metrics["test_season"],
                "n_players": metrics["n_players"],
                "bayes_mae": metrics["bayes_mae"],
                "marcel_mae": metrics["marcel_mae"],
                "mae_improvement_pct": metrics["mae_improvement_pct"],
                "bayes_rmse": metrics["bayes_rmse"],
                "marcel_rmse": metrics["marcel_rmse"],
                "rmse_improvement_pct": metrics["rmse_improvement_pct"],
                "coverage_95": metrics["coverage_95"],
                "bayes_brier": metrics["bayes_brier"],
                "marcel_brier": metrics["marcel_brier"],
                "bayes_crps": metrics["bayes_crps"],
                "marcel_crps": metrics["marcel_crps"],
                "ensemble_w": w,
                "ensemble_mae": ens_metrics["ensemble_mae"],
                "ensemble_rmse": ens_metrics["ensemble_rmse"],
                "ensemble_brier": ens_metrics["ensemble_brier"],
                "ensemble_coverage_95": ens_metrics["ensemble_coverage_95"],
                "converged": metrics["convergence"]["converged"],
            })

    summary = pd.DataFrame(results)
    logger.info("\n=== Hitter Backtest Summary ===\n%s", summary.to_string())
    return summary
