"""
Walk-forward backtest for the batter game simulator.

Evaluates batter game-level stat predictions (K, BB, H, HR, TB) against
actual outcomes from fact_player_game_mlb.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import compute_log_loss, compute_sharpness

from src.data.db import read_sql
from src.data.feature_eng import (
    build_multi_season_hitter_data,
    build_multi_season_pitcher_data,
    get_cached_game_lineups,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.queries import (
    get_batter_game_actuals,
    get_team_bullpen_rates,
    get_team_reliever_roster,
    get_umpire_tendencies,
)
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_sim.batter_simulator import simulate_batter_game
from src.models.hitter_model import (
    extract_rate_samples as extract_hitter_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)
from src.models.matchup import compute_bullpen_matchup_lifts, score_matchup_for_stat
from src.models.pitcher_model import (
    extract_rate_samples as extract_pitcher_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)



def build_batter_sim_predictions(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    min_pa_game: int = 1,
    random_seed: int = 42,
    enable_bullpen_matchup: bool = False,
) -> pd.DataFrame:
    """Build batter game predictions for one fold.

    Parameters
    ----------
    train_seasons : list[int]
        Training seasons.
    test_season : int
        Season to predict.
    draws, tune, chains : int
        MCMC parameters.
    n_sims : int
        MC sims per batter-game.
    min_pa_game : int
        Minimum PA in actual game to include.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per batter-game with predictions and actuals.
    """
    last_train = max(train_seasons)
    logger.info("Building batter sim predictions: train=%s, test=%d",
                train_seasons, test_season)

    # --- 1. Fit hitter K% model ---
    logger.info("Fitting hitter K%% model...")
    df_hitter = build_multi_season_hitter_data(train_seasons, min_pa=50)
    data_k = prepare_hitter_data(df_hitter, "k_rate")
    _, trace_k = fit_hitter_model(
        data_k, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    df_k = data_k["df"]
    batter_ids_k = df_k[df_k["season"] == last_train]["batter_id"].unique()
    batter_k_posteriors: dict[int, np.ndarray] = {}
    for bid in batter_ids_k:
        try:
            samples = extract_hitter_rate_samples(
                trace_k, data_k, bid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            batter_k_posteriors[bid] = samples
        except ValueError:
            continue
    logger.info("Batter K posteriors: %d", len(batter_k_posteriors))

    # --- 2. Fit hitter BB% model ---
    logger.info("Fitting hitter BB%% model...")
    data_bb = prepare_hitter_data(df_hitter, "bb_rate")
    _, trace_bb = fit_hitter_model(
        data_bb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 1,
    )

    df_bb = data_bb["df"]
    batter_ids_bb = df_bb[df_bb["season"] == last_train]["batter_id"].unique()
    batter_bb_posteriors: dict[int, np.ndarray] = {}
    for bid in batter_ids_bb:
        try:
            samples = extract_hitter_rate_samples(
                trace_bb, data_bb, bid, last_train,
                project_forward=True, random_seed=random_seed + 1,
            )
            batter_bb_posteriors[bid] = samples
        except ValueError:
            continue
    logger.info("Batter BB posteriors: %d", len(batter_bb_posteriors))

    # --- 3. Build batter HR/PA pseudo-posteriors from observed rates ---
    logger.info("Building batter HR pseudo-posteriors...")
    batter_hr_posteriors: dict[int, np.ndarray] = {}
    hr_data = df_hitter[df_hitter["season"].isin(train_seasons)].copy()
    for bid, grp in hr_data.groupby("batter_id"):
        total_hr = grp["hr"].sum() if "hr" in grp.columns else 0
        total_pa = grp["pa"].sum()
        if total_pa >= 50:
            rate = total_hr / total_pa
            rng_hr = np.random.default_rng(random_seed + int(bid) % 10000)
            std = max(0.005, rate * 0.15)
            samples = rng_hr.normal(rate, std, size=2000)
            samples = np.clip(samples, 0.001, 0.10)
            batter_hr_posteriors[int(bid)] = samples
    logger.info("Batter HR posteriors: %d", len(batter_hr_posteriors))

    # --- 4. Fit pitcher models (for opposing starter quality lifts) ---
    logger.info("Fitting pitcher K%% model...")
    df_pitcher = build_multi_season_pitcher_data(train_seasons, min_bf=10)
    data_pk = prepare_pitcher_data(df_pitcher, "k_rate")
    _, trace_pk = fit_pitcher_model(
        data_pk, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 10,
    )

    df_pk = data_pk["df"]
    pitcher_k_posteriors: dict[int, np.ndarray] = {}
    for pid in df_pk[df_pk["season"] == last_train]["pitcher_id"].unique():
        try:
            samples = extract_pitcher_rate_samples(
                trace_pk, data_pk, pid, last_train,
                project_forward=True, random_seed=random_seed + 10,
            )
            pitcher_k_posteriors[pid] = samples
        except ValueError:
            continue

    logger.info("Fitting pitcher BB%% model...")
    data_pbb = prepare_pitcher_data(df_pitcher, "bb_rate")
    _, trace_pbb = fit_pitcher_model(
        data_pbb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 11,
    )
    pitcher_bb_posteriors: dict[int, np.ndarray] = {}
    for pid in data_pbb["df"][data_pbb["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            samples = extract_pitcher_rate_samples(
                trace_pbb, data_pbb, pid, last_train,
                project_forward=True, random_seed=random_seed + 11,
            )
            pitcher_bb_posteriors[pid] = samples
        except ValueError:
            continue

    logger.info("Fitting pitcher HR model...")
    data_phr = prepare_pitcher_data(df_pitcher, "hr_per_bf")
    _, trace_phr = fit_pitcher_model(
        data_phr, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 12,
    )
    pitcher_hr_posteriors: dict[int, np.ndarray] = {}
    for pid in data_phr["df"][data_phr["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            samples = extract_pitcher_rate_samples(
                trace_phr, data_phr, pid, last_train,
                project_forward=True, random_seed=random_seed + 12,
            )
            pitcher_hr_posteriors[pid] = samples
        except ValueError:
            continue
    logger.info("Pitcher posteriors: K=%d, BB=%d, HR=%d",
                len(pitcher_k_posteriors), len(pitcher_bb_posteriors),
                len(pitcher_hr_posteriors))

    # --- 5. BF priors for starter workload ---
    from src.data.feature_eng import get_cached_pitcher_game_logs
    game_logs_frames = []
    for s in train_seasons:
        game_logs_frames.append(get_cached_pitcher_game_logs(s))
    all_game_logs = pd.concat(game_logs_frames, ignore_index=True)
    bf_priors = compute_pitcher_bf_priors(all_game_logs)

    # --- 6. Team bullpen rates ---
    bullpen_rates = get_team_bullpen_rates(train_seasons)
    # Use most recent season per team
    bullpen_latest = (
        bullpen_rates
        .sort_values("season")
        .groupby("team_id")
        .last()
        .reset_index()
    )

    # --- 6b. Reliever rosters for bullpen matchup lifts ---
    reliever_roster_df = pd.DataFrame()
    matchup_cache: dict[tuple[int, int, str], float] = {}
    if enable_bullpen_matchup:
        reliever_roster_df = get_team_reliever_roster(train_seasons)
        # Use most recent season per team-pitcher
        reliever_roster_df = (
            reliever_roster_df
            .sort_values("season")
            .groupby(["team_id", "pitcher_id"])
            .last()
            .reset_index()
        )
        logger.info("Loaded reliever rosters: %d relievers across %d teams",
                     len(reliever_roster_df),
                     reliever_roster_df["team_id"].nunique())

    # --- 7. Matchup data ---
    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    from src.data.league_baselines import get_baselines_dict
    baselines_pt = get_baselines_dict(seasons=train_seasons, recency_weights="equal")

    # --- 8. Load test season actuals ---
    logger.info("Loading test season %d batter actuals...", test_season)
    actuals = get_batter_game_actuals(test_season)
    actuals = actuals[actuals["bat_pa"] >= min_pa_game].copy()
    logger.info("Test batter-games: %d", len(actuals))

    # --- 9. Valid batters (have all posteriors) ---
    valid_batters = (
        set(batter_k_posteriors.keys())
        & set(batter_bb_posteriors.keys())
        & set(batter_hr_posteriors.keys())
    )
    logger.info("Valid batters: %d", len(valid_batters))

    # --- 10. Simulate each batter-game ---
    results = []
    n_skipped = 0

    # Group actuals by game for efficiency
    for _, row in actuals.iterrows():
        batter_id = int(row["batter_id"])
        game_pk = int(row["game_pk"])
        batting_order = int(row["batting_order"])
        opp_starter_id = row.get("opp_starter_id")
        opp_team_id = row.get("opp_team_id")

        if batter_id not in valid_batters:
            n_skipped += 1
            continue

        if pd.isna(opp_starter_id):
            n_skipped += 1
            continue
        opp_starter_id = int(opp_starter_id)

        # Get pitcher rates
        starter_k = float(np.mean(pitcher_k_posteriors.get(
            opp_starter_id, np.array([0.226])
        )))
        starter_bb = float(np.mean(pitcher_bb_posteriors.get(
            opp_starter_id, np.array([0.082])
        )))
        starter_hr = float(np.mean(pitcher_hr_posteriors.get(
            opp_starter_id, np.array([0.031])
        )))

        # Starter BF distribution
        bf_row = bf_priors[
            (bf_priors["pitcher_id"] == opp_starter_id)
            & (bf_priors["season"] == last_train)
        ]
        if len(bf_row) > 0:
            starter_bf_mu = float(bf_row.iloc[0]["mu_bf"])
            starter_bf_sigma = float(bf_row.iloc[0]["sigma_bf"])
        else:
            starter_bf_mu = 22.0
            starter_bf_sigma = 4.5

        # Matchup lifts
        matchup_k = 0.0
        matchup_bb = 0.0
        matchup_hr = 0.0
        try:
            for stat, lift_var in [("k", "matchup_k"), ("bb", "matchup_bb"), ("hr", "matchup_hr")]:
                res = score_matchup_for_stat(
                    stat, opp_starter_id, batter_id,
                    pitcher_arsenal, hitter_vuln, baselines_pt,
                )
                lift_key = f"matchup_{stat}_logit_lift"
                val = res.get(lift_key, 0.0)
                if isinstance(val, float) and not np.isnan(val):
                    if stat == "k":
                        matchup_k = val
                    elif stat == "bb":
                        matchup_bb = val
                    else:
                        matchup_hr = val
        except Exception:
            pass

        # Bullpen rates
        bp_matchup_k = 0.0
        bp_matchup_bb = 0.0
        bp_matchup_hr = 0.0
        if pd.notna(opp_team_id):
            opp_tid = int(opp_team_id)
            bp_row = bullpen_latest[bullpen_latest["team_id"] == opp_tid]
            if len(bp_row) > 0:
                bp_k = float(bp_row.iloc[0]["k_rate"])
                bp_bb = float(bp_row.iloc[0]["bb_rate"])
                bp_hr = float(bp_row.iloc[0]["hr_rate"])
            else:
                bp_k, bp_bb, bp_hr = 0.253, 0.084, 0.024

            # Per-batter bullpen matchup lifts
            if enable_bullpen_matchup and not reliever_roster_df.empty:
                team_relievers = reliever_roster_df[
                    reliever_roster_df["team_id"] == opp_tid
                ]
                if len(team_relievers) > 0:
                    bp_lifts = compute_bullpen_matchup_lifts(
                        batter_id, team_relievers,
                        pitcher_arsenal, hitter_vuln, baselines_pt,
                        _cache=matchup_cache,
                    )
                    bp_matchup_k = bp_lifts["bullpen_matchup_k_lift"]
                    bp_matchup_bb = bp_lifts["bullpen_matchup_bb_lift"]
                    bp_matchup_hr = bp_lifts["bullpen_matchup_hr_lift"]
        else:
            bp_k, bp_bb, bp_hr = 0.253, 0.084, 0.024

        # Run simulation
        try:
            sim_result = simulate_batter_game(
                batter_k_rate_samples=batter_k_posteriors[batter_id],
                batter_bb_rate_samples=batter_bb_posteriors[batter_id],
                batter_hr_rate_samples=batter_hr_posteriors[batter_id],
                batting_order=batting_order,
                starter_k_rate=starter_k,
                starter_bb_rate=starter_bb,
                starter_hr_rate=starter_hr,
                starter_bf_mu=starter_bf_mu,
                starter_bf_sigma=starter_bf_sigma,
                matchup_k_lift=matchup_k,
                matchup_bb_lift=matchup_bb,
                matchup_hr_lift=matchup_hr,
                bullpen_k_rate=bp_k,
                bullpen_bb_rate=bp_bb,
                bullpen_hr_rate=bp_hr,
                bullpen_matchup_k_lift=bp_matchup_k,
                bullpen_matchup_bb_lift=bp_matchup_bb,
                bullpen_matchup_hr_lift=bp_matchup_hr,
                n_sims=n_sims,
                random_seed=random_seed + game_pk % 10000 + batter_id % 1000,
            )
        except Exception as e:
            logger.warning("Sim failed batter %d game %d: %s",
                           batter_id, game_pk, e)
            n_skipped += 1
            continue

        summary = sim_result.summary()

        rec = {
            "game_pk": game_pk,
            "batter_id": batter_id,
            "batting_order": batting_order,
            "opp_starter_id": opp_starter_id,
            "test_season": test_season,
            "expected_k": summary["k"]["mean"],
            "std_k": summary["k"]["std"],
            "expected_bb": summary["bb"]["mean"],
            "std_bb": summary["bb"]["std"],
            "expected_h": summary["h"]["mean"],
            "std_h": summary["h"]["std"],
            "expected_hr": summary["hr"]["mean"],
            "std_hr": summary["hr"]["std"],
            "expected_double": summary["double"]["mean"],
            "expected_triple": summary["triple"]["mean"],
            "expected_tb": summary["tb"]["mean"],
            "std_tb": summary["tb"]["std"],
            "expected_pa": summary["pa"]["mean"],
            "actual_k": int(row["bat_k"]),
            "actual_bb": int(row["bat_bb"]),
            "actual_h": int(row["bat_h"]),
            "actual_hr": int(row["bat_hr"]),
            "actual_double": int(row["bat_2b"]),
            "actual_triple": int(row["bat_3b"]),
            "actual_tb": int(row["bat_tb"]),
            "actual_pa": int(row["bat_pa"]),
        }

        # Prop line probs
        k_over = sim_result.over_probs("k", [0.5, 1.5])
        for _, prow in k_over.iterrows():
            col = f"p_k_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        h_over = sim_result.over_probs("h", [0.5, 1.5, 2.5])
        for _, prow in h_over.iterrows():
            col = f"p_h_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        hr_over = sim_result.over_probs("hr", [0.5])
        for _, prow in hr_over.iterrows():
            col = f"p_hr_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        tb_over = sim_result.over_probs("tb", [0.5, 1.5, 2.5, 3.5])
        for _, prow in tb_over.iterrows():
            col = f"p_tb_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        double_over = sim_result.over_probs("double", [0.5])
        for _, prow in double_over.iterrows():
            col = f"p_double_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        triple_over = sim_result.over_probs("triple", [0.5])
        for _, prow in triple_over.iterrows():
            col = f"p_triple_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        results.append(rec)

    logger.info("Completed: %d batter-games predicted, %d skipped",
                len(results), n_skipped)
    return pd.DataFrame(results)


def compute_batter_sim_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Compute metrics for batter sim predictions."""
    n = len(predictions)
    if n == 0:
        return {"n_games": 0}

    metrics: dict[str, Any] = {"n_games": n}

    for stat in ("k", "bb", "h", "hr", "double", "triple", "tb"):
        exp_col = f"expected_{stat}"
        act_col = f"actual_{stat}"
        if exp_col not in predictions.columns or act_col not in predictions.columns:
            continue

        expected = predictions[exp_col].values.astype(float)
        actual = predictions[act_col].values.astype(float)
        errors = expected - actual

        metrics[f"{stat}_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        metrics[f"{stat}_mae"] = float(np.mean(np.abs(errors)))
        metrics[f"{stat}_bias"] = float(np.mean(errors))

        if np.std(actual) > 0 and np.std(expected) > 0:
            metrics[f"{stat}_corr"] = float(np.corrcoef(actual, expected)[0, 1])

    # Brier scores and log loss for prop lines
    prop_cols = {
        "k_0.5": ("p_k_over_0_5", "actual_k", 0.5),
        "k_1.5": ("p_k_over_1_5", "actual_k", 1.5),
        "h_0.5": ("p_h_over_0_5", "actual_h", 0.5),
        "h_1.5": ("p_h_over_1_5", "actual_h", 1.5),
        "hr_0.5": ("p_hr_over_0_5", "actual_hr", 0.5),
        "tb_0.5": ("p_tb_over_0_5", "actual_tb", 0.5),
        "tb_1.5": ("p_tb_over_1_5", "actual_tb", 1.5),
        "tb_2.5": ("p_tb_over_2_5", "actual_tb", 2.5),
        "tb_3.5": ("p_tb_over_3_5", "actual_tb", 3.5),
        "double_0.5": ("p_double_over_0_5", "actual_double", 0.5),
        "triple_0.5": ("p_triple_over_0_5", "actual_triple", 0.5),
    }
    brier_scores: dict[str, float] = {}
    log_losses: dict[str, float] = {}
    for label, (prob_col, act_col, line) in prop_cols.items():
        if prob_col in predictions.columns and act_col in predictions.columns:
            y_true = (predictions[act_col] > line).astype(float).values
            y_prob = predictions[prob_col].values
            brier_scores[label] = float(brier_score_loss(y_true, y_prob))
            log_losses[label] = compute_log_loss(y_prob, y_true)
    metrics["brier_scores"] = brier_scores
    metrics["log_losses"] = log_losses

    # Per-stat aggregated Brier and log loss
    for stat_prefix in ("k", "h", "hr", "tb", "double", "triple"):
        stat_briers = {k: v for k, v in brier_scores.items()
                       if k.startswith(stat_prefix + "_")}
        stat_lls = {k: v for k, v in log_losses.items()
                    if k.startswith(stat_prefix + "_")}
        if stat_briers:
            metrics[f"{stat_prefix}_avg_brier"] = float(
                np.mean(list(stat_briers.values()))
            )
        if stat_lls:
            metrics[f"{stat_prefix}_avg_log_loss"] = float(
                np.mean(list(stat_lls.values()))
            )

    # Sharpness per stat group — pool all prop line probabilities for each
    stat_prob_groups: dict[str, list[str]] = {
        "k": ["p_k_over_0_5", "p_k_over_1_5"],
        "h": ["p_h_over_0_5", "p_h_over_1_5", "p_h_over_2_5"],
        "hr": ["p_hr_over_0_5"],
        "tb": ["p_tb_over_0_5", "p_tb_over_1_5",
               "p_tb_over_2_5", "p_tb_over_3_5"],
    }
    for stat_prefix, prob_cols_list in stat_prob_groups.items():
        probs_all: list[np.ndarray] = []
        for col in prob_cols_list:
            if col in predictions.columns:
                probs_all.append(predictions[col].values.astype(float))
        if probs_all:
            sharp = compute_sharpness(np.concatenate(probs_all))
        else:
            sharp = compute_sharpness(np.array([]))
        for key, val in sharp.items():
            metrics[f"{stat_prefix}_sharpness_{key}"] = val

    # Coverage: credible interval coverage from mean + std for K and H
    for stat in ("k", "h"):
        exp_col = f"expected_{stat}"
        std_col = f"std_{stat}" if f"std_{stat}" in predictions.columns else None
        act_col = f"actual_{stat}"
        if (
            exp_col in predictions.columns
            and std_col is not None
            and act_col in predictions.columns
        ):
            expected = predictions[exp_col].values.astype(float)
            std = predictions[std_col].values.astype(float)
            actual = predictions[act_col].values.astype(float)
            for ci_name, z in [("50", 0.6745), ("80", 1.2816), ("90", 1.6449)]:
                lo = expected - z * std
                hi = expected + z * std
                metrics[f"{stat}_coverage_{ci_name}"] = float(
                    np.mean((actual >= lo) & (actual <= hi))
                )

    return metrics


def run_full_batter_sim_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    max_batters_per_fold: int | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward batter sim backtest.

    Parameters
    ----------
    folds : list or None
        Default: test 2024, 2025.
    max_batters_per_fold : int or None
        Cap on batter-games per fold for fast iteration.
    """
    if folds is None:
        folds = [
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]

    fold_results = []
    all_predictions = []

    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info("BATTER FOLD: train=%s → test=%d", train_seasons, test_season)

        predictions = build_batter_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_sims=n_sims,
            random_seed=random_seed,
        )

        if max_batters_per_fold and len(predictions) > max_batters_per_fold:
            predictions = predictions.head(max_batters_per_fold)

        if len(predictions) == 0:
            continue

        metrics = compute_batter_sim_metrics(predictions)

        fold_rec = {"test_season": test_season, "n_games": metrics["n_games"]}
        for stat in ("k", "bb", "h", "hr", "double", "triple", "tb"):
            for m in ("rmse", "mae", "bias", "corr"):
                key = f"{stat}_{m}"
                if key in metrics:
                    fold_rec[key] = metrics[key]

        # Brier scores
        if "brier_scores" in metrics:
            for label, score in metrics["brier_scores"].items():
                fold_rec[f"brier_{label}"] = score

        # Log losses
        if "log_losses" in metrics:
            for label, score in metrics["log_losses"].items():
                fold_rec[f"logloss_{label}"] = score

        # Per-stat avg Brier and log loss
        for stat_prefix in ("k", "h", "hr", "tb", "double", "triple"):
            for m in ("avg_brier", "avg_log_loss"):
                key = f"{stat_prefix}_{m}"
                if key in metrics:
                    fold_rec[key] = metrics[key]

        # Sharpness
        for stat_prefix in ("k", "h", "hr", "tb"):
            for skey in ("mean_confidence", "pct_actionable_60",
                         "pct_actionable_65", "pct_actionable_70",
                         "entropy"):
                mkey = f"{stat_prefix}_sharpness_{skey}"
                if mkey in metrics:
                    fold_rec[mkey] = metrics[mkey]

        # Coverage
        for stat_prefix in ("k", "h"):
            for ci in ("50", "80", "90"):
                key = f"{stat_prefix}_coverage_{ci}"
                if key in metrics:
                    fold_rec[key] = metrics[key]

        fold_results.append(fold_rec)
        predictions["fold_test_season"] = test_season
        all_predictions.append(predictions)

        logger.info(
            "Batter fold: K RMSE=%.3f, K LogLoss=%.4f, "
            "H RMSE=%.3f, H LogLoss=%.4f, "
            "HR RMSE=%.3f, TB RMSE=%.3f, n=%d",
            fold_rec.get("k_rmse", np.nan),
            fold_rec.get("k_avg_log_loss", np.nan),
            fold_rec.get("h_rmse", np.nan),
            fold_rec.get("h_avg_log_loss", np.nan),
            fold_rec.get("hr_rmse", np.nan),
            fold_rec.get("tb_rmse", np.nan),
            metrics["n_games"],
        )
        logger.info(
            "  Coverage: K(50%%=%.1f%%, 80%%=%.1f%%, 90%%=%.1f%%) "
            "H(50%%=%.1f%%, 80%%=%.1f%%, 90%%=%.1f%%)",
            100 * fold_rec.get("k_coverage_50", np.nan),
            100 * fold_rec.get("k_coverage_80", np.nan),
            100 * fold_rec.get("k_coverage_90", np.nan),
            100 * fold_rec.get("h_coverage_50", np.nan),
            100 * fold_rec.get("h_coverage_80", np.nan),
            100 * fold_rec.get("h_coverage_90", np.nan),
        )
        logger.info(
            "  Sharpness: K(conf=%.3f, act60=%.1f%%) "
            "H(conf=%.3f, act60=%.1f%%)",
            fold_rec.get("k_sharpness_mean_confidence", np.nan),
            fold_rec.get("k_sharpness_pct_actionable_60", np.nan),
            fold_rec.get("h_sharpness_mean_confidence", np.nan),
            fold_rec.get("h_sharpness_pct_actionable_60", np.nan),
        )

    summary = pd.DataFrame(fold_results)
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Overall metrics
    if len(pred_df) > 0:
        overall = compute_batter_sim_metrics(pred_df)
        logger.info("=" * 60)
        logger.info("OVERALL BATTER RESULTS (%d batter-games)", overall["n_games"])
        for stat in ("k", "bb", "h", "hr", "double", "triple", "tb"):
            rmse_key = f"{stat}_rmse"
            if rmse_key in overall:
                logger.info(
                    "  %s: RMSE=%.3f, MAE=%.3f, bias=%.3f, corr=%.3f",
                    stat.upper(),
                    overall[rmse_key],
                    overall.get(f"{stat}_mae", np.nan),
                    overall.get(f"{stat}_bias", np.nan),
                    overall.get(f"{stat}_corr", np.nan),
                )
        if "brier_scores" in overall:
            logger.info("  Brier scores:")
            for label, score in sorted(overall["brier_scores"].items()):
                logger.info("    %s: %.4f", label, score)
        if "log_losses" in overall:
            logger.info("  Log losses:")
            for label, score in sorted(overall["log_losses"].items()):
                logger.info("    %s: %.4f", label, score)
        for stat_prefix in ("k", "h"):
            for ci in ("50", "80", "90"):
                key = f"{stat_prefix}_coverage_{ci}"
                if key in overall:
                    logger.info("  %s Coverage %s%%: %.1f%%",
                                stat_prefix.upper(), ci,
                                100 * overall[key])
        for stat_prefix in ("k", "h", "hr", "tb"):
            conf_key = f"{stat_prefix}_sharpness_mean_confidence"
            if conf_key in overall:
                logger.info(
                    "  %s Sharpness: conf=%.3f, act60=%.1f%%, entropy=%.3f",
                    stat_prefix.upper(),
                    overall[conf_key],
                    overall.get(f"{stat_prefix}_sharpness_pct_actionable_60",
                                np.nan),
                    overall.get(f"{stat_prefix}_sharpness_entropy", np.nan),
                )

    return summary, pred_df
