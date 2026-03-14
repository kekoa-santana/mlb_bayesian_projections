"""
Step 15: Walk-forward calibration and evaluation for game-level K predictions.

Combines Layer 1 (pitcher K% posteriors), Layer 2 (matchup lifts), and
Step 13 (BF distribution) into a game-level backtest, then evaluates:
- Calibration: P(over X.5) vs actual over rates
- Brier scores per K line
- RMSE/MAE on expected K
- Coverage of posterior credible intervals
- Comparison vs Poisson and no-matchup baselines
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import compute_ece, compute_mce, compute_temperature

from src.data.feature_eng import (
    build_multi_season_pitcher_k_data,
    get_cached_game_batter_ks,
    get_cached_game_lineups,
    get_cached_pitcher_game_logs,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.queries import (
    get_umpire_k_tendencies,
    get_weather_effects,
)
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_k_model import (
    compute_k_over_probs,
    extract_pitcher_k_rate_samples,
    predict_game_batch,
    predict_game_batch_stat,
    simulate_game_ks,
)
from src.models.pitcher_k_rate_model import (
    fit_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    extract_rate_samples as extract_generalized_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)


def _build_umpire_lift_lookup(
    train_seasons: list[int],
    test_season: int,
) -> dict[int, float]:
    """Build game_pk → umpire K logit lift for the test season.

    Computes umpire tendencies from training seasons, then maps each
    test-season game to its HP umpire's logit lift.

    Returns
    -------
    dict[int, float]
        {game_pk: umpire_k_logit_lift}.
    """
    from src.data.db import read_sql

    ump_tendencies = get_umpire_k_tendencies(
        seasons=train_seasons, min_games=30,
    )
    if ump_tendencies.empty:
        return {}

    # Build name → lift lookup
    ump_lift_map = dict(zip(
        ump_tendencies["hp_umpire_name"],
        ump_tendencies["k_logit_lift"],
    ))

    # Get umpire assignments for test season
    ump_assignments = read_sql(f"""
        SELECT du.game_pk, du.hp_umpire_name
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    result: dict[int, float] = {}
    for _, row in ump_assignments.iterrows():
        gpk = int(row["game_pk"])
        name = row["hp_umpire_name"]
        lift = ump_lift_map.get(name, 0.0)
        if pd.notna(lift):
            result[gpk] = float(lift)
    logger.info("Umpire lifts: %d games mapped (%.1f%% non-zero)",
                len(result),
                100 * sum(1 for v in result.values() if abs(v) > 0.001) / max(len(result), 1))
    return result


def _build_weather_lift_lookup(
    train_seasons: list[int],
    test_season: int,
) -> dict[int, float]:
    """Build game_pk → weather K logit lift for the test season.

    Computes weather multipliers from training seasons, converts to logit
    lifts, then maps each test-season outdoor game to its weather lift.

    Returns
    -------
    dict[int, float]
        {game_pk: weather_k_logit_lift}.
    """
    from scipy.special import logit as _logit_fn
    from src.data.db import read_sql

    wx_effects = get_weather_effects(seasons=train_seasons)
    if wx_effects.empty:
        return {}

    # Build (temp_bucket, wind_category) → k_multiplier lookup
    overall_k = float(wx_effects.iloc[0]["overall_k_rate"])
    wx_map: dict[tuple[str, str], float] = {}
    for _, row in wx_effects.iterrows():
        k_mult = float(row["k_multiplier"])
        adj_k = overall_k * k_mult
        lift = float(
            _logit_fn(np.clip(adj_k, 1e-6, 1 - 1e-6))
            - _logit_fn(np.clip(overall_k, 1e-6, 1 - 1e-6))
        )
        wx_map[(row["temp_bucket"], row["wind_category"])] = lift

    # Get weather for test season games (temp_bucket derived from temperature)
    weather_data = read_sql(f"""
        SELECT dw.game_pk, dw.is_dome, dw.wind_category,
            CASE
                WHEN dw.temperature < 55 THEN 'cold'
                WHEN dw.temperature BETWEEN 55 AND 69 THEN 'cool'
                WHEN dw.temperature BETWEEN 70 AND 84 THEN 'warm'
                ELSE 'hot'
            END AS temp_bucket
        FROM production.dim_weather dw
        JOIN production.dim_game dg ON dw.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    result: dict[int, float] = {}
    for _, row in weather_data.iterrows():
        gpk = int(row["game_pk"])
        if row.get("is_dome"):
            result[gpk] = 0.0
        else:
            key = (row.get("temp_bucket", ""), row.get("wind_category", ""))
            result[gpk] = wx_map.get(key, 0.0)
    logger.info("Weather lifts: %d games mapped (%.1f%% non-zero)",
                len(result),
                100 * sum(1 for v in result.values() if abs(v) > 0.001) / max(len(result), 1))
    return result


def _build_umpire_lift_lookup_multi(
    train_seasons: list[int],
    test_season: int,
) -> dict[str, dict[int, float]]:
    """Build game_pk -> logit lift for K, BB, and HR per umpire.

    Uses ``get_umpire_tendencies()`` (extended version) to compute lifts
    for all three stats, then maps each test-season game to its HP
    umpire's stat-specific logit lifts.

    Returns
    -------
    dict[str, dict[int, float]]
        ``{"k": {game_pk: lift}, "bb": {game_pk: lift}, "hr": {game_pk: lift}}``.
    """
    from src.data.db import read_sql
    from src.data.queries import get_umpire_tendencies

    ump_tendencies = get_umpire_tendencies(
        seasons=train_seasons, min_games=30,
    )
    if ump_tendencies.empty:
        return {"k": {}, "bb": {}, "hr": {}}

    # Build name -> lift lookups for each stat
    stat_cols = {"k": "k_logit_lift", "bb": "bb_logit_lift", "hr": "hr_logit_lift"}
    ump_lift_maps: dict[str, dict[str, float]] = {}
    for stat, col in stat_cols.items():
        ump_lift_maps[stat] = dict(zip(
            ump_tendencies["hp_umpire_name"],
            ump_tendencies[col],
        ))

    # Get umpire assignments for test season
    ump_assignments = read_sql(f"""
        SELECT du.game_pk, du.hp_umpire_name
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    result: dict[str, dict[int, float]] = {"k": {}, "bb": {}, "hr": {}}
    for _, row in ump_assignments.iterrows():
        gpk = int(row["game_pk"])
        name = row["hp_umpire_name"]
        for stat in ("k", "bb", "hr"):
            lift = ump_lift_maps[stat].get(name, 0.0)
            if pd.notna(lift):
                result[stat][gpk] = float(lift)
    logger.info(
        "Umpire lifts (multi): %d games, K non-zero=%.1f%%, BB non-zero=%.1f%%, "
        "HR non-zero=%.1f%%",
        len(ump_assignments),
        100 * sum(1 for v in result["k"].values() if abs(v) > 0.001) / max(len(result["k"]), 1),
        100 * sum(1 for v in result["bb"].values() if abs(v) > 0.001) / max(len(result["bb"]), 1),
        100 * sum(1 for v in result["hr"].values() if abs(v) > 0.001) / max(len(result["hr"]), 1),
    )
    return result


def _build_weather_lift_lookup_multi(
    train_seasons: list[int],
    test_season: int,
) -> dict[str, dict[int, float]]:
    """Build game_pk -> logit lift for K and HR from weather conditions.

    Returns
    -------
    dict[str, dict[int, float]]
        ``{"k": {game_pk: lift}, "hr": {game_pk: lift}}``.
        BB is not included because weather has negligible effect on BB rates.
    """
    from scipy.special import logit as _logit_fn
    from src.data.db import read_sql

    wx_effects = get_weather_effects(seasons=train_seasons)
    if wx_effects.empty:
        return {"k": {}, "hr": {}}

    overall_k = float(wx_effects.iloc[0]["overall_k_rate"])
    overall_hr = float(wx_effects.iloc[0]["overall_hr_rate"])

    # Build (temp_bucket, wind_category) -> logit lift for K and HR
    wx_map_k: dict[tuple[str, str], float] = {}
    wx_map_hr: dict[tuple[str, str], float] = {}
    for _, row in wx_effects.iterrows():
        key = (row["temp_bucket"], row["wind_category"])
        # K lift
        k_mult = float(row["k_multiplier"])
        adj_k = overall_k * k_mult
        wx_map_k[key] = float(
            _logit_fn(np.clip(adj_k, 1e-6, 1 - 1e-6))
            - _logit_fn(np.clip(overall_k, 1e-6, 1 - 1e-6))
        )
        # HR lift: log(hr_multiplier) on logit scale
        hr_mult = float(row["hr_multiplier"])
        adj_hr = overall_hr * hr_mult
        wx_map_hr[key] = float(
            _logit_fn(np.clip(adj_hr, 1e-6, 1 - 1e-6))
            - _logit_fn(np.clip(overall_hr, 1e-6, 1 - 1e-6))
        )

    # Get weather for test season games
    weather_data = read_sql(f"""
        SELECT dw.game_pk, dw.is_dome, dw.wind_category,
            CASE
                WHEN dw.temperature < 55 THEN 'cold'
                WHEN dw.temperature BETWEEN 55 AND 69 THEN 'cool'
                WHEN dw.temperature BETWEEN 70 AND 84 THEN 'warm'
                ELSE 'hot'
            END AS temp_bucket
        FROM production.dim_weather dw
        JOIN production.dim_game dg ON dw.game_pk = dg.game_pk
        WHERE dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    result_k: dict[int, float] = {}
    result_hr: dict[int, float] = {}
    for _, row in weather_data.iterrows():
        gpk = int(row["game_pk"])
        if row.get("is_dome"):
            result_k[gpk] = 0.0
            result_hr[gpk] = 0.0
        else:
            key = (row.get("temp_bucket", ""), row.get("wind_category", ""))
            result_k[gpk] = wx_map_k.get(key, 0.0)
            result_hr[gpk] = wx_map_hr.get(key, 0.0)
    logger.info(
        "Weather lifts (multi): %d games, K non-zero=%.1f%%, HR non-zero=%.1f%%",
        len(weather_data),
        100 * sum(1 for v in result_k.values() if abs(v) > 0.001) / max(len(result_k), 1),
        100 * sum(1 for v in result_hr.values() if abs(v) > 0.001) / max(len(result_hr), 1),
    )
    return {"k": result_k, "hr": result_hr}


def _build_baselines_pt(
    pitcher_arsenal: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Build league-average baselines per pitch type from arsenal data.

    Parameters
    ----------
    pitcher_arsenal : pd.DataFrame
        Full pitcher arsenal profiles for a season.

    Returns
    -------
    dict[str, dict[str, float]]
        {pitch_type: {"whiff_rate": float}}.
    """
    agg = pitcher_arsenal.groupby("pitch_type").agg(
        total_whiffs=("whiffs", "sum"),
        total_swings=("swings", "sum"),
    ).reset_index()
    agg["whiff_rate"] = agg["total_whiffs"] / agg["total_swings"].replace(0, np.nan)

    baselines: dict[str, dict[str, float]] = {}
    for _, row in agg.iterrows():
        pt = row["pitch_type"]
        wr = row["whiff_rate"]
        if pd.notna(wr):
            baselines[pt] = {"whiff_rate": float(wr)}
    return baselines


def build_game_k_predictions(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    starters_only: bool = True,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build game-level K predictions for a single train→test fold.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons for training the pitcher K% model and BF priors.
    test_season : int
        Season to predict.
    draws : int
        MCMC posterior draws.
    tune : int
        MCMC tuning steps.
    chains : int
        MCMC chains.
    n_mc_draws : int
        Monte Carlo draws per game.
    starters_only : bool
        Only predict starter games.
    min_bf_game : int
        Minimum BF in a game to include (filters short outings).
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per game with predictions and actuals.
    """
    logger.info(
        "Building game K predictions: train=%s, test=%d",
        train_seasons, test_season,
    )

    # 1. Train pitcher K% model on training seasons
    df_model = build_multi_season_pitcher_k_data(train_seasons, min_bf=10)
    data = prepare_pitcher_model_data(df_model)
    _, trace = fit_pitcher_k_rate_model(
        data, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    # 2. Extract posterior samples for each pitcher (last training season)
    last_train = max(train_seasons)
    df = data["df"]
    pitcher_ids_in_model = df[df["season"] == last_train]["pitcher_id"].unique()

    pitcher_posteriors: dict[int, np.ndarray] = {}
    for pid in pitcher_ids_in_model:
        try:
            samples = extract_pitcher_k_rate_samples(
                trace, data, pid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            pitcher_posteriors[pid] = samples
        except ValueError:
            continue

    logger.info("Extracted posteriors for %d pitchers", len(pitcher_posteriors))

    # 3. Build BF priors from training seasons
    game_logs_frames = []
    for s in train_seasons:
        game_logs_frames.append(get_cached_pitcher_game_logs(s))
    all_game_logs = pd.concat(game_logs_frames, ignore_index=True)
    bf_priors = compute_pitcher_bf_priors(all_game_logs)

    # Remap bf_priors season to test_season so predict_game_batch can
    # look up by (pitcher_id, test_season). The priors themselves are
    # computed only from training data — this just fixes the key.
    bf_last_train = bf_priors[bf_priors["season"] == last_train].copy()
    bf_last_train["season"] = test_season
    bf_priors = pd.concat([bf_priors, bf_last_train], ignore_index=True)

    # 4. Load test season data
    test_game_logs = get_cached_pitcher_game_logs(test_season)
    if starters_only:
        test_game_logs = test_game_logs[test_game_logs["is_starter"] == True].copy()  # noqa: E712
    test_game_logs = test_game_logs[test_game_logs["batters_faced"] >= min_bf_game].copy()
    test_game_logs["season"] = test_season

    # 5. Load matchup data from LAST TRAINING SEASON (not test season)
    # to prevent data leakage — full test-season profiles would include
    # future game outcomes (whiff rates, usage) not available at
    # prediction time.
    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    baselines_pt = _build_baselines_pt(pitcher_arsenal)
    # Lineup identity from test season is OK (lineups are public pre-game)
    game_batter_ks = get_cached_game_batter_ks(test_season)

    # 5b. Real starting lineups from fact_lineup (preferred over batter_ks)
    game_lineups = get_cached_game_lineups(test_season)
    logger.info("Loaded %d lineup rows for test season %d",
                len(game_lineups), test_season)

    # 5c. Umpire and weather lifts (trained on training seasons only)
    umpire_lifts = _build_umpire_lift_lookup(train_seasons, test_season)
    weather_lifts = _build_weather_lift_lookup(train_seasons, test_season)

    # 6. Batch predict
    predictions = predict_game_batch(
        game_records=test_game_logs,
        pitcher_posteriors=pitcher_posteriors,
        bf_priors=bf_priors,
        pitcher_arsenal=pitcher_arsenal,
        hitter_vuln=hitter_vuln,
        baselines_pt=baselines_pt,
        game_batter_ks=game_batter_ks,
        game_lineups=game_lineups,
        umpire_lifts=umpire_lifts,
        weather_lifts=weather_lifts,
        n_draws=n_mc_draws,
    )

    logger.info(
        "Predictions: %d games, mean expected K=%.2f, mean actual K=%.2f",
        len(predictions),
        predictions["expected_k"].mean() if len(predictions) else 0,
        predictions["actual_k"].mean() if len(predictions) else 0,
    )

    return predictions


def compute_calibration_by_line(
    predictions: pd.DataFrame,
    lines: list[float] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration: predicted P(over) vs actual over frequency.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_k_predictions`` or ``predict_game_batch``.
    lines : list[float] or None
        K lines to evaluate. Default: [3.5, 4.5, 5.5, 6.5, 7.5].
    n_bins : int
        Number of probability bins for calibration.

    Returns
    -------
    pd.DataFrame
        Columns: line, bin_center, predicted_prob, actual_freq, n_games,
        calibration_error.
    """
    if lines is None:
        lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    records = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue

        p_over = predictions[col].values
        actual_over = (predictions["actual_k"] > line).astype(float).values

        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            if j == n_bins - 1:
                mask = (p_over >= lo) & (p_over <= hi)
            else:
                mask = (p_over >= lo) & (p_over < hi)

            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue

            records.append({
                "line": line,
                "bin_center": bin_centers[j],
                "predicted_prob": float(p_over[mask].mean()),
                "actual_freq": float(actual_over[mask].mean()),
                "n_games": int(n_in_bin),
                "calibration_error": abs(
                    float(p_over[mask].mean()) - float(actual_over[mask].mean())
                ),
            })

    return pd.DataFrame(records)


def compute_game_k_metrics(
    predictions: pd.DataFrame,
    lines: list[float] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics for game K predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_k_predictions``.
    lines : list[float] or None
        Lines for Brier score. Default: [3.5, 4.5, 5.5, 6.5, 7.5].

    Returns
    -------
    dict
        Keys: rmse_expected, mae_expected, brier_scores, avg_brier,
        calibration_df, coverage_50, coverage_80, coverage_90,
        log_score, n_games.
    """
    if lines is None:
        lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    n_games = len(predictions)
    if n_games == 0:
        return {
            "rmse_expected": np.nan,
            "mae_expected": np.nan,
            "brier_scores": {},
            "avg_brier": np.nan,
            "calibration_df": pd.DataFrame(),
            "coverage_50": np.nan,
            "coverage_80": np.nan,
            "coverage_90": np.nan,
            "log_score": np.nan,
            "n_games": 0,
        }

    # RMSE and MAE on expected K
    errors = predictions["expected_k"] - predictions["actual_k"]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # Brier scores per line
    brier_scores: dict[float, float] = {}
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        y_true = (predictions["actual_k"] > line).astype(float).values
        y_prob = predictions[col].values
        brier_scores[line] = float(brier_score_loss(y_true, y_prob))

    avg_brier = float(np.mean(list(brier_scores.values()))) if brier_scores else np.nan

    # Calibration
    calibration_df = compute_calibration_by_line(predictions, lines)

    # Coverage: what fraction of actual Ks fall within [expected - z*std, expected + z*std]
    expected = predictions["expected_k"].values
    std = predictions["std_k"].values
    actual = predictions["actual_k"].values

    def _coverage(z: float) -> float:
        lo = expected - z * std
        hi = expected + z * std
        return float(np.mean((actual >= lo) & (actual <= hi)))

    coverage_50 = _coverage(0.6745)  # 50% CI
    coverage_80 = _coverage(1.2816)  # 80% CI
    coverage_90 = _coverage(1.6449)  # 90% CI

    # Log score (average negative log probability of actual outcome)
    log_scores = []
    for _, row in predictions.iterrows():
        actual_k = int(row["actual_k"])
        mu = row["expected_k"]
        if mu > 0:
            log_p = poisson.logpmf(actual_k, mu)
            log_scores.append(log_p)
    log_score = float(np.mean(log_scores)) if log_scores else np.nan

    # ECE and temperature per line + pooled
    ece_per_line: dict[float, float] = {}
    temperature_per_line: dict[float, float] = {}
    all_probs = []
    all_outcomes = []

    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        p_over = predictions[col].values
        actual_over = (predictions["actual_k"] > line).astype(float).values
        ece_per_line[line] = compute_ece(p_over, actual_over)
        temperature_per_line[line] = compute_temperature(p_over, actual_over)
        all_probs.append(p_over)
        all_outcomes.append(actual_over)

    if all_probs:
        pooled_probs = np.concatenate(all_probs)
        pooled_outcomes = np.concatenate(all_outcomes)
        pooled_ece = compute_ece(pooled_probs, pooled_outcomes)
        pooled_temperature = compute_temperature(pooled_probs, pooled_outcomes)
    else:
        pooled_ece = np.nan
        pooled_temperature = np.nan

    return {
        "rmse_expected": rmse,
        "mae_expected": mae,
        "brier_scores": brier_scores,
        "avg_brier": avg_brier,
        "calibration_df": calibration_df,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "log_score": log_score,
        "n_games": n_games,
        "ece_per_line": ece_per_line,
        "temperature_per_line": temperature_per_line,
        "pooled_ece": pooled_ece,
        "pooled_temperature": pooled_temperature,
    }


def compare_to_baselines(
    predictions: pd.DataFrame,
    lines: list[float] | None = None,
) -> pd.DataFrame:
    """Compare full model to simpler baselines.

    Baselines
    ---------
    1. Naive: pitcher_season_k_rate * actual_bf (perfect BF knowledge)
    2. Poisson: K ~ Poisson(k_rate * avg_bf) — no uncertainty, no matchup
    3. Model-no-matchup: full model but assume all matchup lifts = 0
       (approximated by using pitcher_k_rate_mean * bf_mu)
    4. Full model: as given in predictions

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_k_predictions``.
    lines : list[float] or None
        Lines for Brier evaluation.

    Returns
    -------
    pd.DataFrame
        One row per baseline with metric columns.
    """
    if lines is None:
        lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    if len(predictions) == 0:
        return pd.DataFrame()

    actual = predictions["actual_k"].values
    actual_bf = predictions["actual_bf"].values
    k_rate_mean = predictions["pitcher_k_rate_mean"].values
    bf_mu = predictions["bf_mu"].values

    results = []

    # 1. Naive: k_rate * actual_bf
    naive_expected = k_rate_mean * actual_bf
    naive_rmse = float(np.sqrt(np.mean((naive_expected - actual) ** 2)))
    naive_mae = float(np.mean(np.abs(naive_expected - actual)))
    naive_briers = {}
    for line in lines:
        p_over = 1.0 - poisson.cdf(int(line), naive_expected)
        y_true = (actual > line).astype(float)
        naive_briers[line] = float(brier_score_loss(y_true, np.clip(p_over, 0, 1)))
    results.append({
        "baseline": "naive",
        "rmse": naive_rmse,
        "mae": naive_mae,
        "avg_brier": float(np.mean(list(naive_briers.values()))),
    })

    # 2. Poisson: k_rate * mean_bf (no actual BF knowledge)
    poisson_expected = k_rate_mean * bf_mu
    poisson_rmse = float(np.sqrt(np.mean((poisson_expected - actual) ** 2)))
    poisson_mae = float(np.mean(np.abs(poisson_expected - actual)))
    poisson_briers = {}
    for line in lines:
        p_over = 1.0 - poisson.cdf(int(line), np.clip(poisson_expected, 0.01, None))
        y_true = (actual > line).astype(float)
        poisson_briers[line] = float(brier_score_loss(y_true, np.clip(p_over, 0, 1)))
    results.append({
        "baseline": "poisson",
        "rmse": poisson_rmse,
        "mae": poisson_mae,
        "avg_brier": float(np.mean(list(poisson_briers.values()))),
    })

    # 3. Model-no-matchup: posterior K% * estimated BF (no lineup)
    no_matchup_expected = k_rate_mean * bf_mu
    no_matchup_rmse = float(np.sqrt(np.mean((no_matchup_expected - actual) ** 2)))
    no_matchup_mae = float(np.mean(np.abs(no_matchup_expected - actual)))
    no_matchup_briers = {}
    for line in lines:
        p_over = 1.0 - poisson.cdf(int(line), np.clip(no_matchup_expected, 0.01, None))
        y_true = (actual > line).astype(float)
        no_matchup_briers[line] = float(brier_score_loss(y_true, np.clip(p_over, 0, 1)))
    results.append({
        "baseline": "model_no_matchup",
        "rmse": no_matchup_rmse,
        "mae": no_matchup_mae,
        "avg_brier": float(np.mean(list(no_matchup_briers.values()))),
    })

    # 4. Full model
    full_expected = predictions["expected_k"].values
    full_rmse = float(np.sqrt(np.mean((full_expected - actual) ** 2)))
    full_mae = float(np.mean(np.abs(full_expected - actual)))
    full_briers = {}
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col in predictions.columns:
            y_true = (actual > line).astype(float)
            full_briers[line] = float(brier_score_loss(y_true, predictions[col].values))
    results.append({
        "baseline": "full_model",
        "rmse": full_rmse,
        "mae": full_mae,
        "avg_brier": float(np.mean(list(full_briers.values()))) if full_briers else np.nan,
    })

    return pd.DataFrame(results)


def run_full_game_k_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run walk-forward backtest across multiple folds.

    Parameters
    ----------
    folds : list of (train_seasons, test_season) or None
        Default folds:
        - Train [2020-2022] → Test 2023
        - Train [2020-2023] → Test 2024
        - Train [2020-2024] → Test 2025
    draws, tune, chains : int
        MCMC parameters.
    n_mc_draws : int
        Monte Carlo draws per game.
    min_bf_game : int
        Minimum BF per game to include.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Summary metrics across folds.
    """
    if folds is None:
        folds = [
            ([2020, 2021, 2022], 2023),
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]

    fold_results = []
    all_predictions = []

    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info("Fold: train=%s → test=%d", train_seasons, test_season)

        predictions = build_game_k_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_mc_draws=n_mc_draws,
            min_bf_game=min_bf_game,
            random_seed=random_seed,
        )

        if len(predictions) == 0:
            logger.warning("No predictions for fold train=%s test=%d", train_seasons, test_season)
            continue

        metrics = compute_game_k_metrics(predictions)
        baselines = compare_to_baselines(predictions)

        fold_rec = {
            "test_season": test_season,
            "n_games": metrics["n_games"],
            "rmse": metrics["rmse_expected"],
            "mae": metrics["mae_expected"],
            "avg_brier": metrics["avg_brier"],
            "coverage_50": metrics["coverage_50"],
            "coverage_80": metrics["coverage_80"],
            "coverage_90": metrics["coverage_90"],
            "log_score": metrics["log_score"],
            "pooled_ece": metrics["pooled_ece"],
            "pooled_temperature": metrics["pooled_temperature"],
        }

        # Add baseline comparison
        if len(baselines) > 0:
            for _, brow in baselines.iterrows():
                name = brow["baseline"]
                fold_rec[f"{name}_rmse"] = brow["rmse"]
                fold_rec[f"{name}_avg_brier"] = brow["avg_brier"]

        fold_results.append(fold_rec)
        predictions["fold_test_season"] = test_season
        all_predictions.append(predictions)

        logger.info(
            "Fold results: RMSE=%.3f, MAE=%.3f, Brier=%.4f, "
            "Coverage(50/80/90)=%.2f/%.2f/%.2f",
            metrics["rmse_expected"], metrics["mae_expected"],
            metrics["avg_brier"],
            metrics["coverage_50"], metrics["coverage_80"],
            metrics["coverage_90"],
        )

    results_df = pd.DataFrame(fold_results)

    if len(all_predictions) > 0:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)
        overall = compute_game_k_metrics(all_pred_df)
        logger.info(
            "Overall: RMSE=%.3f, MAE=%.3f, Brier=%.4f, n=%d",
            overall["rmse_expected"], overall["mae_expected"],
            overall["avg_brier"], overall["n_games"],
        )

    return results_df


# ---------------------------------------------------------------------------
# Game-level multi-stat predictions (BB, HR, Outs)
# ---------------------------------------------------------------------------

# Config for additional game-level stats beyond K
GAME_STAT_CONFIGS: dict[str, dict[str, Any]] = {
    "bb": {
        "model_stat": "bb_rate",
        "actual_col": "walks",
        "model_type": "binomial",
        "default_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
    },
    "hr": {
        "model_stat": "hr_per_bf",
        "actual_col": "home_runs",
        "model_type": "poisson",
        "default_lines": [0.5, 1.5, 2.5],
    },
    "outs": {
        "model_stat": None,  # no Bayesian model — use historical rate
        "actual_col": "outs",
        "model_type": "binomial",
        "default_lines": [14.5, 17.5, 20.5, 23.5],
    },
}


def build_game_stat_predictions(
    stat_name: str,
    train_seasons: list[int],
    test_season: int,
    test_game_logs: pd.DataFrame,
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
    game_batter_ks: pd.DataFrame,
    game_lineups: pd.DataFrame,
    umpire_lifts: dict[int, float],
    weather_lifts: dict[int, float],
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build game-level predictions for a non-K stat (BB, HR, Outs).

    Uses the generalized pitcher model for BB and HR. For Outs, uses
    historical rate-based posteriors (no Bayesian model).

    Parameters
    ----------
    stat_name : str
        One of 'bb', 'hr', 'outs'.
    train_seasons, test_season : list[int], int
        Training and test seasons.
    test_game_logs : pd.DataFrame
        Pre-filtered test season game logs (starters, min BF applied).
    bf_priors : pd.DataFrame
        Pre-computed BF priors (already remapped to test_season).
    pitcher_arsenal, hitter_vuln, baselines_pt
        Matchup data from last training season.
    game_batter_ks, game_lineups
        Lineup data for test season.
    umpire_lifts, weather_lifts
        Context lifts per game_pk.
    draws, tune, chains, n_mc_draws, random_seed
        Sampling parameters.

    Returns
    -------
    pd.DataFrame
        Predictions with actual_{stat}, expected_{stat}, std_{stat}, etc.
    """
    cfg = GAME_STAT_CONFIGS[stat_name]
    last_train = max(train_seasons)

    # Build combined context lifts (umpire + weather)
    all_game_pks = set(umpire_lifts.keys()) | set(weather_lifts.keys())
    context_lifts = {
        gpk: umpire_lifts.get(gpk, 0.0) + weather_lifts.get(gpk, 0.0)
        for gpk in all_game_pks
    }

    if cfg["model_stat"] is not None:
        # Fit generalized pitcher model for this stat
        from src.data.feature_eng import build_multi_season_pitcher_data

        model_stat = cfg["model_stat"]
        logger.info("Fitting pitcher %s model for game-level %s predictions",
                     model_stat, stat_name)

        df_pitcher = build_multi_season_pitcher_data(train_seasons, min_bf=10)
        data = prepare_pitcher_data(df_pitcher, model_stat)
        _, trace = fit_pitcher_model(
            data, draws=draws, tune=tune, chains=chains,
            random_seed=random_seed + hash(stat_name) % 1000,
        )

        # Extract posteriors from last training season
        df = data["df"]
        pids_in_model = df[df["season"] == last_train]["pitcher_id"].unique()

        pitcher_posteriors: dict[int, np.ndarray] = {}
        for pid in pids_in_model:
            try:
                samples = extract_generalized_rate_samples(
                    trace, data, pid, last_train,
                    project_forward=True, random_seed=random_seed,
                )
                pitcher_posteriors[pid] = samples
            except ValueError:
                continue

        logger.info("Extracted %s posteriors for %d pitchers",
                     stat_name, len(pitcher_posteriors))
    else:
        # Outs: use historical outs/BF rate as pseudo-posterior
        pitcher_posteriors = {}
        all_logs = []
        for s in train_seasons:
            all_logs.append(get_cached_pitcher_game_logs(s))
        all_log_df = pd.concat(all_logs, ignore_index=True)

        for pid, grp in all_log_df.groupby("pitcher_id"):
            total_outs = grp["outs"].sum()
            total_bf = grp["batters_faced"].sum()
            if total_bf > 0:
                rate = total_outs / total_bf
                # Create pseudo-posterior with reasonable spread
                rng = np.random.default_rng(random_seed + int(pid) % 10000)
                std = max(0.02, rate * 0.05)  # 5% CV floor
                samples = rng.normal(rate, std, size=2000)
                samples = np.clip(samples, 0.3, 1.0)
                pitcher_posteriors[int(pid)] = samples

        logger.info("Built outs rate pseudo-posteriors for %d pitchers",
                     len(pitcher_posteriors))

    # Predict using predict_game_batch_stat
    predictions = predict_game_batch_stat(
        stat_name=stat_name,
        game_records=test_game_logs,
        pitcher_posteriors=pitcher_posteriors,
        bf_priors=bf_priors,
        pitcher_arsenal=pitcher_arsenal,
        hitter_vuln=hitter_vuln,
        baselines_pt=baselines_pt,
        game_batter_ks=game_batter_ks,
        game_lineups=game_lineups,
        context_lifts=context_lifts,
        model_type=cfg["model_type"],
        default_lines=cfg["default_lines"],
        actual_col=cfg["actual_col"],
        n_draws=n_mc_draws,
    )

    logger.info(
        "Game %s predictions: %d games, mean expected=%.2f, mean actual=%.2f",
        stat_name, len(predictions),
        predictions[f"expected_{stat_name}"].mean() if len(predictions) else 0,
        predictions[f"actual_{stat_name}"].mean() if len(predictions) else 0,
    )
    return predictions


def compute_game_stat_metrics(
    predictions: pd.DataFrame,
    stat_name: str,
) -> dict[str, float]:
    """Compute RMSE, MAE, and correlation for a game-level stat prediction.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_stat_predictions``.
    stat_name : str
        Stat name (bb, hr, outs).

    Returns
    -------
    dict
        rmse, mae, correlation, n_games.
    """
    n = len(predictions)
    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "correlation": np.nan, "n_games": 0}

    actual = predictions[f"actual_{stat_name}"].values.astype(float)
    expected = predictions[f"expected_{stat_name}"].values.astype(float)

    errors = expected - actual
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    if np.std(actual) > 0 and np.std(expected) > 0:
        corr = float(np.corrcoef(actual, expected)[0, 1])
    else:
        corr = 0.0

    return {"rmse": rmse, "mae": mae, "correlation": corr, "n_games": n}


def run_full_game_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    additional_stats: list[str] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest for K + additional game-level stats.

    Parameters
    ----------
    folds : list of (train_seasons, test_season) or None
    additional_stats : list[str] or None
        Extra stats beyond K: 'bb', 'hr', 'outs'. Default: all three.
    draws, tune, chains, n_mc_draws, min_bf_game, random_seed
        Sampling parameters.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (k_results, stat_results) — K fold summary and additional stat summary.
    """
    if folds is None:
        folds = [
            ([2020, 2021, 2022], 2023),
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]
    if additional_stats is None:
        additional_stats = ["bb", "hr", "outs"]

    k_fold_results = []
    stat_fold_results = []
    all_k_predictions = []

    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info("Fold: train=%s → test=%d", train_seasons, test_season)
        last_train = max(train_seasons)

        # --- K predictions (existing pipeline) ---
        predictions = build_game_k_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_mc_draws=n_mc_draws,
            min_bf_game=min_bf_game,
            random_seed=random_seed,
        )

        if len(predictions) == 0:
            logger.warning("No K predictions for fold")
            continue

        metrics = compute_game_k_metrics(predictions)
        baselines = compare_to_baselines(predictions)

        fold_rec: dict[str, Any] = {
            "test_season": test_season,
            "n_games": metrics["n_games"],
            "rmse": metrics["rmse_expected"],
            "mae": metrics["mae_expected"],
            "avg_brier": metrics["avg_brier"],
            "coverage_50": metrics["coverage_50"],
            "coverage_80": metrics["coverage_80"],
            "coverage_90": metrics["coverage_90"],
            "log_score": metrics["log_score"],
            "pooled_ece": metrics["pooled_ece"],
            "pooled_temperature": metrics["pooled_temperature"],
        }
        if len(baselines) > 0:
            for _, brow in baselines.iterrows():
                name = brow["baseline"]
                fold_rec[f"{name}_rmse"] = brow["rmse"]
                fold_rec[f"{name}_avg_brier"] = brow["avg_brier"]

        k_fold_results.append(fold_rec)
        predictions["fold_test_season"] = test_season
        all_k_predictions.append(predictions)

        logger.info(
            "K: RMSE=%.3f, MAE=%.3f, Brier=%.4f",
            metrics["rmse_expected"], metrics["mae_expected"],
            metrics["avg_brier"],
        )

        # --- Additional stat predictions ---
        # Reuse shared infrastructure: BF priors, lineups, matchup data
        game_logs_frames = []
        for s in train_seasons:
            game_logs_frames.append(get_cached_pitcher_game_logs(s))
        all_game_logs = pd.concat(game_logs_frames, ignore_index=True)
        bf_priors = compute_pitcher_bf_priors(all_game_logs)
        bf_last_train = bf_priors[bf_priors["season"] == last_train].copy()
        bf_last_train["season"] = test_season
        bf_priors = pd.concat([bf_priors, bf_last_train], ignore_index=True)

        test_game_logs = get_cached_pitcher_game_logs(test_season)
        test_game_logs = test_game_logs[test_game_logs["is_starter"] == True].copy()  # noqa: E712
        test_game_logs = test_game_logs[test_game_logs["batters_faced"] >= min_bf_game].copy()
        test_game_logs["season"] = test_season

        pitcher_arsenal = get_pitcher_arsenal(last_train)
        hitter_vuln = get_hitter_vulnerability(last_train)
        baselines_pt = _build_baselines_pt(pitcher_arsenal)
        game_batter_ks = get_cached_game_batter_ks(test_season)
        game_lineups = get_cached_game_lineups(test_season)
        umpire_lifts = _build_umpire_lift_lookup(train_seasons, test_season)
        weather_lifts = _build_weather_lift_lookup(train_seasons, test_season)

        for stat_name in additional_stats:
            try:
                stat_preds = build_game_stat_predictions(
                    stat_name=stat_name,
                    train_seasons=train_seasons,
                    test_season=test_season,
                    test_game_logs=test_game_logs,
                    bf_priors=bf_priors,
                    pitcher_arsenal=pitcher_arsenal,
                    hitter_vuln=hitter_vuln,
                    baselines_pt=baselines_pt,
                    game_batter_ks=game_batter_ks,
                    game_lineups=game_lineups,
                    umpire_lifts=umpire_lifts,
                    weather_lifts=weather_lifts,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    n_mc_draws=n_mc_draws,
                    random_seed=random_seed,
                )
                stat_metrics = compute_game_stat_metrics(stat_preds, stat_name)
                stat_fold_results.append({
                    "stat": stat_name,
                    "test_season": test_season,
                    **stat_metrics,
                })
                logger.info(
                    "%s: RMSE=%.3f, MAE=%.3f, corr=%.3f, n=%d",
                    stat_name.upper(),
                    stat_metrics["rmse"], stat_metrics["mae"],
                    stat_metrics["correlation"], stat_metrics["n_games"],
                )
            except Exception as e:
                logger.error("Failed to predict game %s: %s", stat_name, e)
                stat_fold_results.append({
                    "stat": stat_name,
                    "test_season": test_season,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "correlation": np.nan,
                    "n_games": 0,
                })

    k_results = pd.DataFrame(k_fold_results)
    stat_results = pd.DataFrame(stat_fold_results)

    if len(all_k_predictions) > 0:
        all_pred_df = pd.concat(all_k_predictions, ignore_index=True)
        overall = compute_game_k_metrics(all_pred_df)
        logger.info(
            "Overall K: RMSE=%.3f, MAE=%.3f, Brier=%.4f, n=%d",
            overall["rmse_expected"], overall["mae_expected"],
            overall["avg_brier"], overall["n_games"],
        )

    return k_results, stat_results
