#!/usr/bin/env python
"""
A/B comparison: old stamina-only exit offset vs new unified BF offset.

Runs the game sim backtest for a single test season using both approaches
and compares BF accuracy, K/Outs Brier scores, and distributional calibration.

Usage
-----
    python scripts/compare_bf_improvements.py
    python scripts/compare_bf_improvements.py --test-season 2024
    python scripts/compare_bf_improvements.py --test-season 2024 --draws 500 --n-sims 3000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_comparison(
    test_season: int = 2025,
    draws: int = 500,
    tune: int = 300,
    chains: int = 2,
    n_sims: int = 3000,
    random_seed: int = 42,
) -> None:
    """Run A/B comparison between old and new BF calibration."""
    from src.data.db import read_sql
    from src.data.feature_eng import (
        build_multi_season_pitcher_data,
        build_multi_season_pitcher_k_data,
        get_cached_game_lineups,
        get_cached_pitcher_game_logs,
        get_hitter_vulnerability,
        get_pitcher_arsenal,
    )
    from src.data.queries import (
        get_batter_pitch_count_features,
        get_bullpen_trailing_workload,
        get_exit_model_training_data,
        get_pitcher_exit_tendencies,
        get_pitcher_pitch_count_features,
        get_tto_adjustment_profiles,
        get_umpire_tendencies,
    )
    from src.models.bf_model import compute_pitcher_bf_priors
    from src.models.game_sim.exit_model import ExitModel
    from src.models.game_sim.pa_outcome_model import GameContext
    from src.models.game_sim.pitch_count_model import build_pitch_count_features
    from src.models.game_sim.simulator import (
        compute_exit_offset,
        compute_stamina_offset,
        simulate_game,
    )
    from src.models.game_sim.tto_model import build_all_tto_lifts
    from src.models.matchup import score_matchup_for_stat
    from src.models.pitcher_k_rate_model import (
        fit_pitcher_k_rate_model,
        prepare_pitcher_model_data,
    )
    from src.models.pitcher_model import (
        extract_rate_samples as extract_generalized_rate_samples,
        fit_pitcher_model,
        prepare_pitcher_data,
    )
    from src.models.posterior_utils import extract_pitcher_k_rate_samples

    train_seasons = list(range(2018, test_season))
    last_train = max(train_seasons)

    logger.info("=" * 60)
    logger.info("BF Improvement A/B Test: train=%s, test=%d", train_seasons, test_season)
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # 1-3. Fit pitcher models (K, BB, HR)
    # ---------------------------------------------------------------
    logger.info("Fitting pitcher K%% model...")
    df_k = build_multi_season_pitcher_k_data(train_seasons, min_bf=10)
    data_k = prepare_pitcher_model_data(df_k)
    _, trace_k = fit_pitcher_k_rate_model(
        data_k, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    df_k_df = data_k["df"]
    pids_k = df_k_df[df_k_df["season"] == last_train]["pitcher_id"].unique()
    k_posteriors: dict[int, np.ndarray] = {}
    for pid in pids_k:
        try:
            samples = extract_pitcher_k_rate_samples(
                trace_k, data_k, pid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            k_posteriors[pid] = samples
        except ValueError:
            continue
    logger.info("K posteriors: %d pitchers", len(k_posteriors))

    logger.info("Fitting pitcher BB model...")
    df_bb = build_multi_season_pitcher_data(train_seasons, min_bf=10)
    data_bb = prepare_pitcher_data(df_bb, "bb_rate")
    _, trace_bb = fit_pitcher_model(
        data_bb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 1,
    )
    df_bb_df = data_bb["df"]
    bb_posteriors: dict[int, np.ndarray] = {}
    for pid in df_bb_df[df_bb_df["season"] == last_train]["pitcher_id"].unique():
        try:
            bb_posteriors[pid] = extract_generalized_rate_samples(
                trace_bb, data_bb, pid, last_train,
                project_forward=True, random_seed=random_seed + 1,
            )
        except ValueError:
            continue
    logger.info("BB posteriors: %d pitchers", len(bb_posteriors))

    logger.info("Fitting pitcher HR model...")
    data_hr = prepare_pitcher_data(df_bb, "hr_per_bf")
    _, trace_hr = fit_pitcher_model(
        data_hr, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 2,
    )
    hr_posteriors: dict[int, np.ndarray] = {}
    for pid in data_hr["df"][data_hr["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            hr_posteriors[pid] = extract_generalized_rate_samples(
                trace_hr, data_hr, pid, last_train,
                project_forward=True, random_seed=random_seed + 2,
            )
        except ValueError:
            continue
    logger.info("HR posteriors: %d pitchers", len(hr_posteriors))

    # ---------------------------------------------------------------
    # 4. Train exit model
    # ---------------------------------------------------------------
    logger.info("Training exit model...")
    exit_train_data = get_exit_model_training_data(train_seasons)
    exit_tendencies = get_pitcher_exit_tendencies(train_seasons)
    exit_model = ExitModel()
    exit_metrics = exit_model.train(exit_train_data, exit_tendencies)
    logger.info("Exit model AUC: %.4f", exit_metrics["auc"])

    # ---------------------------------------------------------------
    # 5. Load supporting data
    # ---------------------------------------------------------------
    logger.info("Loading supporting data...")
    pitcher_pc_features = get_pitcher_pitch_count_features(train_seasons)
    batter_pc_features = get_batter_pitch_count_features(train_seasons)

    pitcher_pc_latest = (
        pitcher_pc_features.sort_values("season")
        .groupby("pitcher_id").last().reset_index()
    )
    pitcher_pc_latest["season"] = last_train

    batter_pc_latest = (
        batter_pc_features.sort_values("season")
        .groupby("batter_id").last().reset_index()
    )
    batter_pc_latest["season"] = last_train

    tto_profiles = get_tto_adjustment_profiles(train_seasons)
    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    from src.data.league_baselines import get_baselines_dict
    baselines_pt = get_baselines_dict(seasons=train_seasons, recency_weights="equal")

    pitcher_tend_latest = (
        exit_tendencies.sort_values("season")
        .groupby("pitcher_id").last().reset_index()
    )

    # BF priors
    bf_game_logs = pd.concat(
        [get_cached_pitcher_game_logs(s) for s in train_seasons],
        ignore_index=True,
    )
    bf_priors = compute_pitcher_bf_priors(bf_game_logs)
    bf_priors_latest = (
        bf_priors.sort_values("season")
        .groupby("pitcher_id").last().reset_index()
    )
    logger.info("BF priors: %d pitchers", len(bf_priors_latest))

    # Bullpen trailing workload
    bullpen_workload = get_bullpen_trailing_workload([test_season])
    bullpen_wl_lookup: dict[tuple[int, int], float] = {}
    if not bullpen_workload.empty:
        for _, bw_row in bullpen_workload.iterrows():
            key = (int(bw_row["team_id"]), int(bw_row["game_pk"]))
            bullpen_wl_lookup[key] = float(bw_row["bullpen_trailing_ip"])
        logger.info("Bullpen workload: %d team-games", len(bullpen_wl_lookup))

    # ---------------------------------------------------------------
    # 6. Load test season data
    # ---------------------------------------------------------------
    logger.info("Loading test season %d data...", test_season)
    test_game_logs = get_cached_pitcher_game_logs(test_season)
    test_game_logs = test_game_logs[
        (test_game_logs["is_starter"] == True)  # noqa: E712
        & (test_game_logs["batters_faced"] >= 15)
    ].copy()
    test_game_logs["season"] = test_season

    game_lineups = get_cached_game_lineups(test_season)

    actuals = read_sql(f"""
        SELECT player_id AS pitcher_id, game_pk, team_id,
               pit_k, pit_bb, pit_h, pit_hr, pit_bf,
               pit_pitches, pit_ip, pit_er
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = TRUE
          AND season = {int(test_season)}
          AND pit_bf >= 15
    """)

    valid_pids = (
        set(k_posteriors.keys())
        & set(bb_posteriors.keys())
        & set(hr_posteriors.keys())
    )
    logger.info(
        "Test games: %d, pitchers with all posteriors: %d",
        len(test_game_logs), len(valid_pids),
    )

    # ---------------------------------------------------------------
    # 7. Run both approaches per game
    # ---------------------------------------------------------------
    results_old: list[dict] = []
    results_new: list[dict] = []
    n_skipped = 0

    from src.evaluation.game_sim_validation import _compute_lineup_matchup_lifts

    for i, (_, game_row) in enumerate(test_game_logs.iterrows()):
        pitcher_id = int(game_row["pitcher_id"])
        game_pk = int(game_row["game_pk"])

        if pitcher_id not in valid_pids:
            n_skipped += 1
            continue

        # Get opposing lineup
        game_lu = game_lineups[game_lineups["game_pk"] == game_pk]
        if len(game_lu) == 0:
            n_skipped += 1
            continue

        teams = game_lu["team_id"].unique()
        opposing_team = None
        for t in teams:
            team_players = game_lu[game_lu["team_id"] == t]["player_id"].values
            if pitcher_id not in team_players:
                opposing_team = t
                break
        if opposing_team is None:
            n_skipped += 1
            continue

        batter_ids = (
            game_lu[game_lu["team_id"] == opposing_team]
            .sort_values("batting_order")["player_id"]
            .tolist()[:9]
        )
        if len(batter_ids) < 9:
            n_skipped += 1
            continue

        # Matchup lifts
        matchup_lifts, matchup_reliabilities = _compute_lineup_matchup_lifts(
            pitcher_id, batter_ids,
            pitcher_arsenal, hitter_vuln, baselines_pt,
        )

        # TTO lifts
        tto_lifts = build_all_tto_lifts(tto_profiles, pitcher_id, last_train)

        # Pitcher tendencies
        tend_row = pitcher_tend_latest[pitcher_tend_latest["pitcher_id"] == pitcher_id]
        if len(tend_row) > 0:
            avg_pitches = float(tend_row.iloc[0]["avg_pitches"])
            avg_ip = (
                float(tend_row.iloc[0]["avg_ip"])
                if "avg_ip" in tend_row.columns
                and pd.notna(tend_row.iloc[0].get("avg_ip"))
                else 5.28
            )
            team_avg_p = (
                float(tend_row.iloc[0]["team_avg_pitches"])
                if "team_avg_pitches" in tend_row.columns
                and pd.notna(tend_row.iloc[0].get("team_avg_pitches"))
                else 88.0
            )
        else:
            avg_pitches = 88.0
            avg_ip = 5.28
            team_avg_p = 88.0

        # Pitch count features
        pitcher_adj, batter_adjs = build_pitch_count_features(
            pitcher_features=pitcher_pc_latest,
            batter_features=batter_pc_latest,
            pitcher_id=pitcher_id,
            batter_ids=batter_ids,
            season=last_train,
        )

        # --- OLD approach: stamina-only offset ---
        old_offset = compute_stamina_offset(avg_ip)

        # --- NEW approach: unified offset with BF prior + bullpen + lineup ---
        bf_row = bf_priors_latest[bf_priors_latest["pitcher_id"] == pitcher_id]
        mu_bf = float(bf_row.iloc[0]["mu_bf"]) if len(bf_row) > 0 else None

        actual_row_team = actuals[
            (actuals["pitcher_id"] == pitcher_id)
            & (actuals["game_pk"] == game_pk)
        ]
        pitcher_team_id = (
            int(actual_row_team.iloc[0]["team_id"])
            if len(actual_row_team) > 0 else None
        )
        bullpen_ip = (
            bullpen_wl_lookup.get((pitcher_team_id, game_pk))
            if pitcher_team_id is not None else None
        )
        lineup_ppa_agg = float(np.mean(batter_adjs))

        new_offset = compute_exit_offset(
            mu_bf=mu_bf,
            pitcher_avg_ip=avg_ip,
            bullpen_trailing_ip=bullpen_ip,
            lineup_ppa_aggregate=lineup_ppa_agg,
        )

        # Actuals
        if len(actual_row_team) > 0:
            ar = actual_row_team.iloc[0]
            actual_k = int(ar.get("pit_k", 0))
            actual_bb = int(ar.get("pit_bb", 0))
            actual_h = int(ar.get("pit_h", 0))
            actual_hr = int(ar.get("pit_hr", 0))
            actual_bf = int(ar.get("pit_bf", 0))
            actual_ip = float(ar.get("pit_ip", 0))
        else:
            actual_k = int(game_row.get("strike_outs", 0))
            actual_bb = int(game_row.get("walks", 0))
            actual_h = int(game_row.get("hits", 0))
            actual_hr = int(game_row.get("home_runs", 0))
            actual_bf = int(game_row.get("batters_faced", 0))
            actual_ip = float(game_row.get("innings_pitched", 0))
        actual_outs = int(actual_ip) * 3 + round((actual_ip % 1) * 10)

        # Run both sims
        shared_kwargs = dict(
            pitcher_k_rate_samples=k_posteriors[pitcher_id],
            pitcher_bb_rate_samples=bb_posteriors[pitcher_id],
            pitcher_hr_rate_samples=hr_posteriors[pitcher_id],
            lineup_matchup_lifts=matchup_lifts,
            tto_lifts=tto_lifts,
            pitcher_ppa_adj=pitcher_adj,
            batter_ppa_adjs=batter_adjs,
            exit_model=exit_model,
            pitcher_avg_pitches=avg_pitches,
            game_context=GameContext(),
            lineup_matchup_reliabilities=matchup_reliabilities,
            manager_pull_tendency=team_avg_p,
            n_sims=n_sims,
            random_seed=random_seed + game_pk % 10000,
        )

        try:
            sim_old = simulate_game(
                **shared_kwargs,
                exit_calibration_offset=old_offset,
            )
            sim_new = simulate_game(
                **shared_kwargs,
                exit_calibration_offset=new_offset,
            )
        except Exception as e:
            logger.warning("Sim failed for game %d pitcher %d: %s", game_pk, pitcher_id, e)
            n_skipped += 1
            continue

        # Build result records
        for sim, results_list, label in [
            (sim_old, results_old, "old"),
            (sim_new, results_new, "new"),
        ]:
            summary = sim.summary()
            k_over = sim.over_probs("k", [3.5, 4.5, 5.5, 6.5, 7.5])
            outs_over = sim.over_probs("outs", [14.5, 15.5, 16.5, 17.5, 18.5])

            rec = {
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "approach": label,
                "expected_k": summary["k"]["mean"],
                "expected_bb": summary["bb"]["mean"],
                "expected_h": summary["h"]["mean"],
                "expected_hr": summary["hr"]["mean"],
                "expected_bf": summary["bf"]["mean"],
                "expected_outs": summary["outs"]["mean"],
                "std_bf": summary["bf"]["std"],
                "std_k": summary["k"]["std"],
                "std_outs": summary["outs"]["std"],
                "actual_k": actual_k,
                "actual_bb": actual_bb,
                "actual_h": actual_h,
                "actual_hr": actual_hr,
                "actual_bf": actual_bf,
                "actual_outs": actual_outs,
            }
            if mu_bf is not None:
                rec["mu_bf"] = mu_bf
            if bullpen_ip is not None:
                rec["bullpen_trailing_ip"] = bullpen_ip
            rec["lineup_ppa_agg"] = lineup_ppa_agg
            rec["exit_offset_old"] = old_offset
            rec["exit_offset_new"] = new_offset

            for _, prow in k_over.iterrows():
                col = f"p_over_{prow['line']:.1f}".replace(".", "_")
                rec[col] = prow["p_over"]
            for _, prow in outs_over.iterrows():
                col = f"p_outs_over_{prow['line']:.1f}".replace(".", "_")
                rec[col] = prow["p_over"]

            results_list.append(rec)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d games", i + 1, len(test_game_logs))

    logger.info("Completed: %d games predicted, %d skipped", len(results_old), n_skipped)

    # ---------------------------------------------------------------
    # 8. Compare metrics
    # ---------------------------------------------------------------
    df_old = pd.DataFrame(results_old)
    df_new = pd.DataFrame(results_new)

    if df_old.empty:
        logger.error("No results — cannot compare")
        return

    print("\n" + "=" * 70)
    print(f"  BF IMPROVEMENT A/B COMPARISON — {test_season} ({len(df_old)} games)")
    print("=" * 70)

    # BF metrics
    for label, df in [("OLD (stamina)", df_old), ("NEW (unified)", df_new)]:
        bf_error = df["expected_bf"].values - df["actual_bf"].values
        bf_abs_error = np.abs(bf_error)
        bf_corr = np.corrcoef(df["actual_bf"], df["expected_bf"])[0, 1]

        print(f"\n  {label}:")
        print(f"    BF bias:     {np.mean(bf_error):+.3f}")
        print(f"    BF MAE:      {np.mean(bf_abs_error):.3f}")
        print(f"    BF corr:     {bf_corr:.4f}")

        # K metrics
        k_briers = []
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            col = f"p_over_{line:.1f}".replace(".", "_")
            if col in df.columns:
                y_true = (df["actual_k"] > line).astype(float).values
                y_prob = df[col].values
                brier = brier_score_loss(y_true, y_prob)
                k_briers.append(brier)
        if k_briers:
            print(f"    K avg Brier: {np.mean(k_briers):.5f}")

        # Outs metrics
        outs_briers = []
        for line in [14.5, 15.5, 16.5, 17.5, 18.5]:
            col = f"p_outs_over_{line:.1f}".replace(".", "_")
            if col in df.columns and "actual_outs" in df.columns:
                y_true = (df["actual_outs"] > line).astype(float).values
                y_prob = df[col].values
                outs_briers.append(brier_score_loss(y_true, y_prob))
        if outs_briers:
            print(f"    Outs avg Brier: {np.mean(outs_briers):.5f}")

        # Stat biases
        for stat in ("k", "bb", "h", "hr", "outs"):
            bias = float(np.mean(df[f"expected_{stat}"] - df[f"actual_{stat}"]))
            print(f"    {stat:>4s} bias:  {bias:+.3f}")

        # BF coverage
        exp_bf = df["expected_bf"].values
        std_bf = df["std_bf"].values
        act_bf = df["actual_bf"].values
        for ci_name, z in [("50%", 0.6745), ("80%", 1.2816), ("90%", 1.6449)]:
            lo = exp_bf - z * std_bf
            hi = exp_bf + z * std_bf
            cov = np.mean((act_bf >= lo) & (act_bf <= hi))
            print(f"    BF {ci_name} coverage: {cov:.3f}")

    # Delta analysis
    print(f"\n  DELTA (NEW - OLD):")
    for stat in ("bf", "k", "bb", "h", "hr", "outs"):
        old_bias = np.mean(df_old[f"expected_{stat}"] - df_old[f"actual_{stat}"])
        new_bias = np.mean(df_new[f"expected_{stat}"] - df_new[f"actual_{stat}"])
        print(f"    {stat:>4s} bias: {old_bias:+.3f} -> {new_bias:+.3f}  (d={new_bias-old_bias:+.3f})")

    # Offset analysis
    if "exit_offset_old" in df_new.columns:
        old_offsets = df_new["exit_offset_old"].values
        new_offsets = df_new["exit_offset_new"].values
        delta_offsets = new_offsets - old_offsets
        print(f"\n  Exit offset distribution:")
        print(f"    Old: mean={np.mean(old_offsets):.3f}, std={np.std(old_offsets):.3f}")
        print(f"    New: mean={np.mean(new_offsets):.3f}, std={np.std(new_offsets):.3f}")
        print(f"    Delta: mean={np.mean(delta_offsets):+.3f}, std={np.std(delta_offsets):.3f}")
        print(f"    Delta range: [{np.min(delta_offsets):+.3f}, {np.max(delta_offsets):+.3f}]")

    # Save results
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    combined = pd.concat([df_old, df_new], ignore_index=True)
    out_path = output_dir / f"bf_ab_comparison_{test_season}.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BF exit offset approaches")
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--n-sims", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_comparison(
        test_season=args.test_season,
        draws=args.draws,
        tune=args.tune,
        n_sims=args.n_sims,
        random_seed=args.seed,
    )
