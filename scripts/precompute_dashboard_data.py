#!/usr/bin/env python
"""
Pre-compute all data needed by the Streamlit dashboard.

Fits composite hitter + pitcher models (K% and BB% only), enriches with
observed profiles, extracts posterior K% samples, computes BF priors,
and saves everything to tdd-dashboard/data/dashboard/.

Usage
-----
    python scripts/precompute_dashboard_data.py          # full quality
    python scripts/precompute_dashboard_data.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Dashboard repo location — precompute writes directly to tdd-dashboard
DASHBOARD_REPO = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")

from src.data.feature_eng import (
    build_multi_season_hitter_extended,
    build_multi_season_pitcher_extended,
    build_multi_season_pitcher_k_data,
    get_cached_hitter_observed_profile,
    get_cached_pitcher_observed_profile,
    get_cached_sprint_speed,
    get_hitter_strength,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.queries import (
    get_batter_pitch_count_features,
    get_exit_model_training_data,
    get_game_batter_ks,
    get_game_lineups,
    get_hitter_aggressiveness,
    get_hitter_traditional_stats,
    get_park_factors,
    get_hitter_team_venue,
    get_pitcher_efficiency,
    get_pitcher_exit_tendencies,
    get_pitcher_game_logs,
    get_pitcher_pitch_count_features,
    get_pitcher_traditional_stats,
    get_lineup_priors,
    get_player_teams,
    get_team_bullpen_rates,
    get_tto_adjustment_profiles,
    get_umpire_k_tendencies,
    get_weather_effects,
)
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.health_score import compute_health_scores
from src.models.counting_projections import (
    project_hitter_counting,
    project_pitcher_counting,
    project_pitcher_counting_sim,
)
from src.models.game_k_model import extract_pitcher_k_rate_samples
from src.models.game_sim.exit_model import ExitModel
from src.models.hitter_model import extract_rate_samples as extract_hitter_rate_samples
from src.models.pitcher_model import extract_rate_samples
from src.models.hitter_projections import (
    fit_all_models as fit_hitter_models,
    project_forward as project_hitter_forward,
)
from src.models.pa_model import compute_hitter_pa_priors
from src.models.pitcher_k_rate_model import (
    check_pitcher_convergence,
    fit_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)
from src.models.pitcher_projections import (
    fit_all_models as fit_pitcher_models,
    project_forward as project_pitcher_forward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")

SEASONS = list(range(2018, 2026))
FROM_SEASON = 2025
DASHBOARD_DIR = DASHBOARD_REPO / "data" / "dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute dashboard data")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws for fast iteration",
    )
    return parser.parse_args()


def precompute_backtest_summaries() -> None:
    """Copy backtest results to dashboard directory and compute confidence tiers."""
    import shutil
    from src.evaluation.confidence_tiers import assign_confidence_tiers, tiers_to_dataframe

    outputs_dir = PROJECT_ROOT / "outputs"
    dashboard_dir = DASHBOARD_DIR

    # Copy game prop backtest summary if it exists
    summary_path = outputs_dir / "game_prop_backtest_summary.parquet"
    if summary_path.exists():
        dest = dashboard_dir / "backtest_game_prop_summary.parquet"
        shutil.copy2(summary_path, dest)
        logger.info("Copied game prop backtest summary to %s", dest)

        # Compute confidence tiers
        summary = pd.read_parquet(summary_path)
        tiers = assign_confidence_tiers(summary)
        tiers_df = tiers_to_dataframe(tiers)
        tier_path = dashboard_dir / "backtest_confidence_tiers.parquet"
        tiers_df.to_parquet(tier_path, index=False)
        logger.info("Saved confidence tiers to %s (%d props)", tier_path, len(tiers_df))
    else:
        logger.info("No game prop backtest summary found at %s — skipping tier computation", summary_path)

    # Copy calibration data if available
    for pred_file in outputs_dir.glob("game_prop_predictions_*.parquet"):
        dest = dashboard_dir / pred_file.name
        shutil.copy2(pred_file, dest)
        logger.info("Copied %s to dashboard", pred_file.name)

    # Copy existing season backtest CSVs as standardized parquets
    for csv_file in outputs_dir.glob("*backtest*.csv"):
        try:
            df = pd.read_csv(csv_file)
            dest = dashboard_dir / f"backtest_{csv_file.stem}.parquet"
            df.to_parquet(dest, index=False)
            logger.info("Converted %s to parquet", csv_file.name)
        except Exception as e:
            logger.warning("Failed to convert %s: %s", csv_file.name, e)


def main() -> None:
    args = parse_args()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        draws, tune, chains = 500, 250, 2
        logger.info("QUICK mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    else:
        draws, tune, chains = 2000, 1000, 4
        logger.info("FULL mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)

    # =================================================================
    # 0. Pre-cache observed profiles (needed by projection enrichment)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Pre-caching observed profiles for %d...", FROM_SEASON)
    try:
        obs_h = get_cached_hitter_observed_profile(FROM_SEASON)
        logger.info("Hitter observed profile: %d rows", len(obs_h))
    except Exception as e:
        logger.warning("Could not cache hitter observed profile: %s", e)

    try:
        sprint = get_cached_sprint_speed(FROM_SEASON)
        logger.info("Sprint speed: %d rows", len(sprint))
    except Exception as e:
        logger.warning("Could not cache sprint speed: %s", e)

    try:
        obs_p = get_cached_pitcher_observed_profile(FROM_SEASON)
        logger.info("Pitcher observed profile: %d rows", len(obs_p))
    except Exception as e:
        logger.warning("Could not cache pitcher observed profile: %s", e)

    # =================================================================
    # 1. Hitter composite projections (K% + BB% projected, observed enriched)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting hitter composite models (K%%, BB%%)...")

    # Load posterior calibration T from config
    import yaml
    with open(PROJECT_ROOT / "config" / "model.yaml") as f:
        _cfg = yaml.safe_load(f)
    _cal = _cfg.get("calibration", {})
    hitter_cal_t = _cal.get("hitter", {})
    pitcher_cal_t = _cal.get("pitcher", {})
    if any(v != 1.0 for v in hitter_cal_t.values()):
        logger.info("Hitter calibration T: %s", hitter_cal_t)
    if any(v != 1.0 for v in pitcher_cal_t.values()):
        logger.info("Pitcher calibration T: %s", pitcher_cal_t)

    hitter_results = fit_hitter_models(
        seasons=SEASONS, min_pa=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    hitter_proj = project_hitter_forward(
        hitter_results, from_season=FROM_SEASON, min_pa=200,
        calibration_t=hitter_cal_t,
    )
    hitter_proj.to_parquet(DASHBOARD_DIR / "hitter_projections.parquet", index=False)
    logger.info("Saved hitter projections: %d players", len(hitter_proj))

    # =================================================================
    # 2. Pitcher composite projections (K% + BB% projected, observed enriched)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting pitcher composite models (K%%, BB%%)...")
    pitcher_results = fit_pitcher_models(
        seasons=SEASONS, min_bf=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    # Get ERA-FIP gap data for ERA derivation
    from src.data.feature_eng import get_pitcher_era_fip_data
    era_fip_data = get_pitcher_era_fip_data(FROM_SEASON)
    logger.info("Loaded ERA-FIP data for %d pitchers", len(era_fip_data))

    # Train XGBoost priors for ERA-FIP gap and BABIP
    from src.models.xgb_priors import train_all_pitcher_priors, predict_pitcher_priors
    xgb_models = train_all_pitcher_priors(seasons=SEASONS, min_ip=40.0)
    xgb_preds: dict[str, dict[int, float]] = {}
    for target, bundle in xgb_models.items():
        if bundle.get("model") is not None:
            xgb_preds[target] = predict_pitcher_priors(bundle, FROM_SEASON)
            logger.info(
                "XGBoost %s: %d predictions (train RMSE=%.4f)",
                target, len(xgb_preds[target]), bundle["train_rmse"],
            )

    pitcher_proj = project_pitcher_forward(
        pitcher_results, from_season=FROM_SEASON, min_bf=200,
        era_fip_data=era_fip_data,
        calibration_t=pitcher_cal_t,
        xgb_priors=xgb_preds,
    )
    pitcher_proj.to_parquet(DASHBOARD_DIR / "pitcher_projections.parquet", index=False)
    logger.info("Saved pitcher projections: %d players", len(pitcher_proj))

    # =================================================================
    # 2b. Counting stat projections (K, BB, HR, SB for hitters; K, BB, Outs for pitchers)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing counting stat projections...")

    # Health/durability scores (replaces blunt age penalties)
    logger.info("Computing health/durability scores...")
    health_df = compute_health_scores(FROM_SEASON)
    health_df.to_parquet(DASHBOARD_DIR / "health_scores.parquet", index=False)
    logger.info("Saved health scores: %d players", len(health_df))

    # Hitter counting stats (with park factor adjustment for HR)
    hitter_ext = build_multi_season_hitter_extended(SEASONS, min_pa=1)
    pa_priors = compute_hitter_pa_priors(
        hitter_ext, from_season=FROM_SEASON, min_pa=100,
        health_scores=health_df,
    )

    # Load park factors for HR adjustment
    logger.info("Loading park factors for HR adjustment...")
    park_factors = get_park_factors(FROM_SEASON)
    hitter_venues = get_hitter_team_venue(FROM_SEASON)
    logger.info("Park factors: %d venue-hand combos, %d hitter-venue mappings",
                len(park_factors), len(hitter_venues))

    # Save park factor data for dashboard display
    hitter_venues.to_parquet(DASHBOARD_DIR / "hitter_venues.parquet", index=False)
    park_factors.to_parquet(DASHBOARD_DIR / "park_factors.parquet", index=False)

    hitter_counting = project_hitter_counting(
        rate_model_results=hitter_results,
        pa_priors=pa_priors,
        hitter_extended=hitter_ext,
        from_season=FROM_SEASON,
        n_draws=4000,
        min_pa=200,
        random_seed=42,
        park_factors=park_factors,
        hitter_venues=hitter_venues,
    )
    # Pitcher counting stats
    pitcher_ext = build_multi_season_pitcher_extended(SEASONS, min_bf=1)
    pitcher_counting = project_pitcher_counting(
        rate_model_results=pitcher_results,
        pitcher_extended=pitcher_ext,
        from_season=FROM_SEASON,
        n_draws=4000,
        min_bf=200,
        random_seed=42,
        health_scores=health_df,
    )

    hitter_counting.to_parquet(DASHBOARD_DIR / "hitter_counting.parquet", index=False)
    logger.info("Saved hitter counting projections: %d players", len(hitter_counting))
    pitcher_counting.to_parquet(DASHBOARD_DIR / "pitcher_counting.parquet", index=False)
    logger.info("Saved pitcher counting projections: %d players", len(pitcher_counting))

    # =================================================================
    # 2c. Sim-based pitcher counting stats + reliever roles & rankings
    # =================================================================
    logger.info("=" * 60)
    logger.info("Running sim-based pitcher season projections...")

    try:
        from src.models.reliever_roles import classify_reliever_roles
        from src.models.reliever_rankings import rank_relievers

        # 2c-i. Classify reliever roles
        logger.info("Classifying reliever roles...")
        reliever_roles = classify_reliever_roles(
            seasons=list(range(max(2020, SEASONS[0]), SEASONS[-1] + 1)),
            current_season=FROM_SEASON,
            min_games=10,
        )
        reliever_roles.to_parquet(DASHBOARD_DIR / "pitcher_roles.parquet", index=False)
        logger.info("Saved reliever roles: %d relievers", len(reliever_roles))

        # 2c-ii. Build posteriors dict from existing rate samples
        # (reuses pitcher_results from Section 2, which has k_rate/bb_rate traces)
        logger.info("Building posterior rate samples for season sim...")
        pitcher_posteriors: dict[int, dict[str, np.ndarray]] = {}

        # Get active pitchers from the projection season
        active_pitchers = pitcher_ext[
            (pitcher_ext["season"] == FROM_SEASON)
            & (pitcher_ext["batters_faced"] >= 50)
        ][["pitcher_id", "pitcher_name", "batters_faced", "is_starter"]].drop_duplicates("pitcher_id")

        pitcher_name_lookup = dict(
            zip(active_pitchers["pitcher_id"], active_pitchers["pitcher_name"])
        )

        rng_post = np.random.default_rng(42)
        for _, prow in active_pitchers.iterrows():
            pid = int(prow["pitcher_id"])
            rates: dict[str, np.ndarray] = {}

            # K rate from composite pitcher model
            if "k_rate" in pitcher_results:
                try:
                    rates["k_rate"] = extract_rate_samples(
                        pitcher_results["k_rate"]["trace"],
                        pitcher_results["k_rate"]["data"],
                        pitcher_id=pid, season=FROM_SEASON,
                        project_forward=True,
                    )
                except (ValueError, KeyError):
                    rates["k_rate"] = rng_post.beta(5, 20, size=8000)
            else:
                rates["k_rate"] = rng_post.beta(5, 20, size=8000)

            # BB rate
            if "bb_rate" in pitcher_results:
                try:
                    rates["bb_rate"] = extract_rate_samples(
                        pitcher_results["bb_rate"]["trace"],
                        pitcher_results["bb_rate"]["data"],
                        pitcher_id=pid, season=FROM_SEASON,
                        project_forward=True,
                    )
                except (ValueError, KeyError):
                    rates["bb_rate"] = rng_post.beta(3, 30, size=8000)
            else:
                rates["bb_rate"] = rng_post.beta(3, 30, size=8000)

            # HR rate — synthetic Beta if no Bayesian model
            if "hr_per_bf" in pitcher_results:
                try:
                    rates["hr_rate"] = extract_rate_samples(
                        pitcher_results["hr_per_bf"]["trace"],
                        pitcher_results["hr_per_bf"]["data"],
                        pitcher_id=pid, season=FROM_SEASON,
                        project_forward=True,
                    )
                except (ValueError, KeyError):
                    rates["hr_rate"] = rng_post.beta(2, 60, size=8000)
            else:
                # Synthetic Beta from observed HR/BF
                bf = int(prow["batters_faced"])
                hr_obs = pitcher_ext[
                    (pitcher_ext["pitcher_id"] == pid)
                    & (pitcher_ext["season"] == FROM_SEASON)
                ]["hr"].values
                hr_count = int(hr_obs[0]) if len(hr_obs) > 0 else int(bf * 0.03)
                rates["hr_rate"] = rng_post.beta(
                    hr_count + 1, bf - hr_count + 1, size=8000,
                ).astype(np.float32)

            pitcher_posteriors[pid] = rates

        logger.info("Built posteriors for %d pitchers", len(pitcher_posteriors))

        # 2c-iii. Build starter-specific priors (n_starts, avg_pitches)
        logger.info("Building starter-specific priors...")
        starter_ids = set(
            active_pitchers[active_pitchers["is_starter"] == 1]["pitcher_id"]
        )
        starter_prior_rows = []
        for _, prow in active_pitchers.iterrows():
            pid = int(prow["pitcher_id"])
            if pid not in starter_ids:
                continue

            player_hist = pitcher_ext[
                (pitcher_ext["pitcher_id"] == pid)
                & (pitcher_ext["season"] <= FROM_SEASON)
                & (pitcher_ext["is_starter"] == 1)
            ].sort_values("season", ascending=False).head(3)

            if player_hist.empty:
                starter_prior_rows.append({
                    "pitcher_id": pid,
                    "n_starts_mu": 30.0,
                    "n_starts_sigma": 6.0,
                    "avg_pitches": 88.0,
                })
                continue

            # Weighted starts (5/4/3)
            weights = [5, 4, 3][:len(player_hist)]
            adj_games = player_hist["games"].values.copy().astype(float)
            for i, s in enumerate(player_hist["season"].values):
                if s == 2020:
                    adj_games[i] = min(adj_games[i] * (162 / 60), 36)
            wtd_starts = sum(g * w for g, w in zip(adj_games, weights)) / sum(weights)
            # Regress toward pop mean
            n = len(player_hist)
            rel = n / (n + 1.5)
            mu = rel * wtd_starts + (1 - rel) * 30.0

            # Sigma from variation (floor at 3)
            if len(adj_games) >= 2:
                sigma = max(float(np.std(adj_games, ddof=1)), 3.0)
            else:
                sigma = 6.0

            starter_prior_rows.append({
                "pitcher_id": pid,
                "n_starts_mu": mu,
                "n_starts_sigma": sigma,
                "avg_pitches": 88.0,
            })

        starter_priors_df = pd.DataFrame(starter_prior_rows) if starter_prior_rows else None

        # 2c-iv. Train exit model (or reuse if already trained in section 9c)
        logger.info("Training exit model for season sim...")
        sim_exit_model = ExitModel()
        try:
            exit_training_data = get_exit_model_training_data(SEASONS)
            exit_tend = get_pitcher_exit_tendencies(SEASONS)
            sim_exit_model.train(exit_training_data, exit_tend)
            logger.info("Exit model trained for season sim")
        except Exception as e:
            logger.warning("Exit model training failed, using fallback: %s", e)

        # 2c-v. Run sim-based projections
        import time
        t0 = time.time()

        pitcher_counting_sim = project_pitcher_counting_sim(
            posteriors=pitcher_posteriors,
            roles=reliever_roles,
            exit_model=sim_exit_model,
            starter_priors=starter_priors_df,
            health_scores=health_df,
            pitcher_names=pitcher_name_lookup,
            n_seasons=200,
            random_seed=42,
        )
        elapsed = time.time() - t0
        pitcher_counting_sim.to_parquet(
            DASHBOARD_DIR / "pitcher_counting_sim.parquet", index=False,
        )
        logger.info(
            "Saved sim-based pitcher projections: %d pitchers in %.1fs",
            len(pitcher_counting_sim), elapsed,
        )

        # 2c-vi. Reliever rankings
        logger.info("Building reliever rankings...")
        rp_rankings = rank_relievers(
            sim_df=pitcher_counting_sim,
            roles_df=reliever_roles,
            season=FROM_SEASON,
        )
        rp_rankings.to_parquet(
            DASHBOARD_DIR / "reliever_rankings.parquet", index=False,
        )
        logger.info("Saved reliever rankings: %d relievers", len(rp_rankings))

    except Exception:
        logger.exception("Failed to run sim-based pitcher projections")

    # =================================================================
    # 2d. Sim-based hitter counting stats
    # =================================================================
    logger.info("=" * 60)
    logger.info("Running sim-based hitter season projections...")

    try:
        from src.models.counting_projections import project_hitter_counting_sim
        from src.models.hitter_model import extract_rate_samples as extract_hitter_rates

        # Build hitter posteriors
        logger.info("Building hitter posterior rate samples...")
        hitter_posteriors: dict[int, dict[str, np.ndarray]] = {}
        hitter_name_lookup: dict[int, str] = {}

        active_hitters = hitter_ext[
            (hitter_ext["season"] == FROM_SEASON)
            & (hitter_ext["pa"] >= 50)
        ][["batter_id", "batter_name", "pa", "hr", "sb", "games"]].drop_duplicates("batter_id")

        hitter_name_lookup = dict(
            zip(active_hitters["batter_id"], active_hitters["batter_name"])
        )

        rng_h = np.random.default_rng(42)
        for _, hrow in active_hitters.iterrows():
            bid = int(hrow["batter_id"])
            rates: dict[str, np.ndarray] = {}

            for stat_key in ["k_rate", "bb_rate"]:
                if stat_key in hitter_results:
                    try:
                        rates[stat_key] = extract_hitter_rates(
                            hitter_results[stat_key]["trace"],
                            hitter_results[stat_key]["data"],
                            batter_id=bid, season=FROM_SEASON,
                            project_forward=True,
                        )
                        continue
                    except (ValueError, KeyError):
                        pass
                if stat_key == "k_rate":
                    rates[stat_key] = rng_h.beta(5, 18, size=8000)
                else:
                    rates[stat_key] = rng_h.beta(3, 30, size=8000)

            # HR: synthetic Beta
            pa_val = int(hrow["pa"])
            hr_val = int(hrow.get("hr", pa_val * 0.03))
            rates["hr_rate"] = rng_h.beta(hr_val + 1, pa_val - hr_val + 1, size=8000)

            hitter_posteriors[bid] = rates

        logger.info("Built hitter posteriors for %d batters", len(hitter_posteriors))

        # SB rates from recent history
        sb_rates: dict[int, tuple[float, float]] = {}
        for _, hrow in active_hitters.iterrows():
            bid = int(hrow["batter_id"])
            games = max(int(hrow.get("games", 100)), 1)
            sb = int(hrow.get("sb", 0))
            sb_per_g = sb / games
            sb_rates[bid] = (sb_per_g, max(sb_per_g * 0.30, 0.01))

        # Run sim
        import time as _time
        t0_h = _time.time()
        hitter_counting_sim = project_hitter_counting_sim(
            posteriors=hitter_posteriors,
            pa_priors=pa_priors,
            sb_rates=sb_rates,
            health_scores=health_df,
            batter_names=hitter_name_lookup,
            n_seasons=200,
            random_seed=42,
        )
        elapsed_h = _time.time() - t0_h
        hitter_counting_sim.to_parquet(
            DASHBOARD_DIR / "hitter_counting_sim.parquet", index=False,
        )
        logger.info(
            "Saved sim-based hitter projections: %d batters in %.1fs",
            len(hitter_counting_sim), elapsed_h,
        )

    except Exception:
        logger.exception("Failed to run sim-based hitter projections")

    # =================================================================
    # 3. Pitcher K% model (for posterior samples → Game K sim)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting pitcher K%% model for posterior samples...")
    df_pitcher = build_multi_season_pitcher_k_data(SEASONS, min_bf=1)
    pitcher_data = prepare_pitcher_model_data(df_pitcher)
    _model, pitcher_trace = fit_pitcher_k_rate_model(
        pitcher_data, draws=draws, tune=tune, chains=chains,
    )
    conv = check_pitcher_convergence(pitcher_trace)
    logger.info("Pitcher K%% convergence: %s (r_hat=%.4f)",
                "OK" if conv["converged"] else "ISSUES", conv["max_rhat"])

    # Extract forward-projected K% samples for each pitcher active in FROM_SEASON
    active = df_pitcher[
        (df_pitcher["season"] == FROM_SEASON) & (df_pitcher["batters_faced"] >= 50)
    ]["pitcher_id"].unique()

    k_samples_dict: dict[str, np.ndarray] = {}
    for pid in active:
        try:
            samples = extract_pitcher_k_rate_samples(
                pitcher_trace, pitcher_data,
                pitcher_id=int(pid),
                season=FROM_SEASON,
                project_forward=True,
            )
            k_samples_dict[str(int(pid))] = samples
        except ValueError:
            continue

    np.savez_compressed(
        DASHBOARD_DIR / "pitcher_k_samples.npz",
        **k_samples_dict,
    )
    logger.info("Saved K%% posterior samples for %d pitchers", len(k_samples_dict))

    # Save preseason K samples snapshot for in-season conjugate updating
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_dir / "pitcher_k_samples_preseason.npz",
        **k_samples_dict,
    )
    logger.info("Saved preseason K%% samples snapshot")

    # =================================================================
    # 3b. BB% and HR/BF posterior samples (from composite model)
    # =================================================================
    for stat_name, npz_name in [("bb_rate", "pitcher_bb_samples"), ("hr_per_bf", "pitcher_hr_samples")]:
        if stat_name not in pitcher_results:
            logger.warning("No %s model in pitcher_results — skipping %s samples", stat_name, stat_name)
            continue
        stat_data = pitcher_results[stat_name]["data"]
        stat_trace = pitcher_results[stat_name]["trace"]
        stat_df = stat_data["df"]
        stat_active = stat_df[
            (stat_df["season"] == FROM_SEASON) & (stat_df["batters_faced"] >= 50)
        ]["pitcher_id"].unique()

        samples_dict: dict[str, np.ndarray] = {}
        for pid in stat_active:
            try:
                samples = extract_rate_samples(
                    stat_trace, stat_data,
                    pitcher_id=int(pid),
                    season=FROM_SEASON,
                    project_forward=True,
                )
                samples_dict[str(int(pid))] = samples
            except ValueError:
                continue

        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **samples_dict)
        logger.info("Saved %s posterior samples for %d pitchers", stat_name, len(samples_dict))

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **samples_dict)
        logger.info("Saved preseason %s samples snapshot", stat_name)

    # =================================================================
    # 4. BF priors
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing BF priors...")
    game_logs_list = []
    for s in SEASONS:
        gl = get_pitcher_game_logs(s)
        game_logs_list.append(gl)
    game_logs = pd.concat(game_logs_list, ignore_index=True)

    # Compute pitcher P/PA from game logs for the projection season
    # (starters only, need BF > 0 to avoid div-by-zero)
    _starter_logs = game_logs[
        (game_logs["is_starter"] == True)  # noqa: E712
        & (game_logs["batters_faced"] >= 3)
        & (game_logs["number_of_pitches"].notna())
        & (game_logs["number_of_pitches"] > 0)
    ]
    if not _starter_logs.empty:
        _ppa = (
            _starter_logs.groupby("pitcher_id")
            .apply(
                lambda g: g["number_of_pitches"].sum() / g["batters_faced"].sum(),
                include_groups=False,
            )
            .reset_index(name="pitches_per_pa")
        )
        logger.info("Computed P/PA for %d pitchers (mean=%.2f)", len(_ppa), _ppa["pitches_per_pa"].mean())
    else:
        _ppa = None

    bf_priors = compute_pitcher_bf_priors(game_logs, pitcher_ppa=_ppa)
    bf_priors.to_parquet(DASHBOARD_DIR / "bf_priors.parquet", index=False)
    logger.info("Saved BF priors: %d pitcher-seasons", len(bf_priors))

    # =================================================================
    # 4b. Umpire K-rate tendencies
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing umpire K-rate tendencies...")
    umpire_tendencies = get_umpire_k_tendencies(
        seasons=list(range(2021, 2026)), min_games=30,
    )
    umpire_tendencies.to_parquet(DASHBOARD_DIR / "umpire_tendencies.parquet", index=False)
    logger.info("Saved umpire tendencies: %d umpires", len(umpire_tendencies))

    # =================================================================
    # 4c. Weather effects on K and HR rates
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing weather effects...")
    weather_effects = get_weather_effects(seasons=SEASONS)
    weather_effects.to_parquet(DASHBOARD_DIR / "weather_effects.parquet", index=False)
    logger.info("Saved weather effects: %d combinations", len(weather_effects))

    # =================================================================
    # 4d. Player team mapping (2025 regular season + 2026 spring training override)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing player-team mapping...")
    player_teams = get_player_teams(FROM_SEASON)

    # Override with 2026 spring training teams for players who changed teams
    try:
        from src.data.db import read_sql as _read_sql
        st_teams = _read_sql("""
            WITH st_appearances AS (
                SELECT bb.batter_id AS player_id, bb.team_id, COUNT(*) AS games
                FROM staging.batting_boxscores bb
                JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
                WHERE dg.season = :season AND dg.game_type = 'S'
                GROUP BY bb.batter_id, bb.team_id
                UNION ALL
                SELECT pb.pitcher_id AS player_id, pb.team_id, COUNT(*) AS games
                FROM staging.pitching_boxscores pb
                JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
                WHERE dg.season = :season AND dg.game_type = 'S'
                GROUP BY pb.pitcher_id, pb.team_id
            ),
            primary_st AS (
                SELECT player_id, team_id,
                       SUM(games) AS total_games,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY SUM(games) DESC) AS rn
                FROM st_appearances GROUP BY player_id, team_id
            )
            SELECT p.player_id,
                   COALESCE(dt.abbreviation, '') AS team_abbr,
                   COALESCE(dt.team_name, '') AS team_name
            FROM primary_st p
            LEFT JOIN production.dim_team dt ON p.team_id = dt.team_id
            WHERE p.rn = 1
        """, {"season": FROM_SEASON + 1})
        if not st_teams.empty:
            # Only override for players already in our projections
            st_lookup = dict(zip(st_teams["player_id"], st_teams["team_abbr"]))
            st_name_lookup = dict(zip(st_teams["player_id"], st_teams["team_name"]))
            n_updated = 0
            for idx, row in player_teams.iterrows():
                pid = row["player_id"]
                if pid in st_lookup and st_lookup[pid] and st_lookup[pid] != row["team_abbr"]:
                    player_teams.at[idx, "team_abbr"] = st_lookup[pid]
                    player_teams.at[idx, "team_name"] = st_name_lookup.get(pid, "")
                    n_updated += 1
            # Also add players who are in spring training but not in 2025 regular season
            existing_pids = set(player_teams["player_id"])
            new_rows = st_teams[~st_teams["player_id"].isin(existing_pids)]
            if not new_rows.empty:
                player_teams = pd.concat([player_teams, new_rows], ignore_index=True)
            logger.info("Spring training override: %d team changes, %d new players",
                        n_updated, len(new_rows))
    except Exception as e:
        logger.warning("Could not load spring training teams: %s", e)

    player_teams.to_parquet(DASHBOARD_DIR / "player_teams.parquet", index=False)
    logger.info("Saved player teams: %d players", len(player_teams))

    # =================================================================
    # 4e. Game browser data (lineups + batter Ks for historical games)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing game browser data for %d...", FROM_SEASON)
    game_lineups = get_game_lineups(FROM_SEASON)
    game_lineups.to_parquet(DASHBOARD_DIR / "game_lineups.parquet", index=False)
    logger.info("Saved game lineups: %d rows (%d games)",
                len(game_lineups), game_lineups["game_pk"].nunique())

    game_batter_ks = get_game_batter_ks(FROM_SEASON)
    game_batter_ks.to_parquet(DASHBOARD_DIR / "game_batter_ks.parquet", index=False)
    logger.info("Saved game batter Ks: %d rows", len(game_batter_ks))

    # Pitcher game logs (used by Game Browser for game selection)
    pitcher_game_logs = get_pitcher_game_logs(FROM_SEASON)
    pitcher_game_logs.to_parquet(DASHBOARD_DIR / "pitcher_game_logs.parquet", index=False)
    logger.info("Saved pitcher game logs: %d rows", len(pitcher_game_logs))

    # Game info (date, teams) — used by Game Browser enrichment
    from src.data.db import read_sql as _read_sql_gi
    game_info = _read_sql_gi("""
        SELECT game_pk, game_date, season,
               home_team_id, away_team_id,
               home_team_name, away_team_name
        FROM production.dim_game
        WHERE game_type = 'R' AND season = :season
    """, {"season": FROM_SEASON})
    game_info.to_parquet(DASHBOARD_DIR / "game_info.parquet", index=False)
    logger.info("Saved game info: %d games", len(game_info))

    # =================================================================
    # 5. Matchup profiles (arsenal, vulnerability, strength)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Loading matchup profiles for %d...", FROM_SEASON)

    pitcher_arsenal = get_pitcher_arsenal(FROM_SEASON)
    pitcher_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal.parquet", index=False)
    logger.info("Saved pitcher arsenal: %d rows", len(pitcher_arsenal))

    hitter_vuln = get_hitter_vulnerability(FROM_SEASON)
    hitter_vuln.to_parquet(DASHBOARD_DIR / "hitter_vuln.parquet", index=False)
    logger.info("Saved hitter vulnerability: %d rows", len(hitter_vuln))

    hitter_str = get_hitter_strength(FROM_SEASON)
    hitter_str.to_parquet(DASHBOARD_DIR / "hitter_str.parquet", index=False)
    logger.info("Saved hitter strength: %d rows", len(hitter_str))

    # Career-aggregated vulnerability (weighted across all seasons)
    vuln_frames = []
    for s in SEASONS:
        v = get_hitter_vulnerability(s)
        vuln_frames.append(v)
    all_vuln = pd.concat(vuln_frames, ignore_index=True)

    # Aggregate: sum raw counts, recompute rates
    career_vuln = all_vuln.groupby(["batter_id", "batter_stand", "pitch_type"]).agg(
        pitches=("pitches", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
        out_of_zone_pitches=("out_of_zone_pitches", "sum"),
        chase_swings=("chase_swings", "sum"),
        called_strikes=("called_strikes", "sum"),
        csw=("csw", "sum"),
        bip=("bip", "sum"),
        hard_hits=("hard_hits", "sum"),
        barrels_proxy=("barrels_proxy", "sum"),
    ).reset_index()
    career_vuln["whiff_rate"] = career_vuln["whiffs"] / career_vuln["swings"].replace(0, np.nan)
    career_vuln["chase_rate"] = career_vuln["chase_swings"] / career_vuln["out_of_zone_pitches"].replace(0, np.nan)
    career_vuln["csw_pct"] = career_vuln["csw"] / career_vuln["pitches"].replace(0, np.nan)
    career_vuln["pitch_family"] = career_vuln["pitch_type"].map(
        hitter_vuln.set_index("pitch_type")["pitch_family"].to_dict()
        if "pitch_family" in hitter_vuln.columns else {}
    )
    # xwoba_contact: weighted average by BIP across seasons
    xwoba_rows = all_vuln[all_vuln["xwoba_contact"].notna() & (all_vuln["bip"] > 0)]
    if not xwoba_rows.empty:
        xwoba_career = xwoba_rows.groupby(["batter_id", "pitch_type"]).apply(
            lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
            if g["bip"].sum() > 0 else np.nan,
            include_groups=False,
        ).reset_index(name="xwoba_contact")
        career_vuln = career_vuln.merge(xwoba_career, on=["batter_id", "pitch_type"], how="left")
    else:
        career_vuln["xwoba_contact"] = np.nan

    career_vuln.to_parquet(DASHBOARD_DIR / "hitter_vuln_career.parquet", index=False)
    logger.info("Saved career hitter vulnerability: %d rows", len(career_vuln))

    # Career-aggregated strength
    str_frames = []
    for s in SEASONS:
        st_df = get_hitter_strength(s)
        str_frames.append(st_df)
    all_str = pd.concat(str_frames, ignore_index=True)

    career_str = all_str.groupby(["batter_id", "batter_stand", "pitch_type"]).agg(
        bip=("bip", "sum"),
        barrels_proxy=("barrels_proxy", "sum"),
        hard_hits=("hard_hits", "sum"),
    ).reset_index()
    career_str["barrel_rate_contact"] = career_str["barrels_proxy"] / career_str["bip"].replace(0, np.nan)
    career_str["hard_hit_rate"] = career_str["hard_hits"] / career_str["bip"].replace(0, np.nan)
    career_str["pitch_family"] = career_str["pitch_type"].map(
        hitter_str.set_index("pitch_type")["pitch_family"].to_dict()
        if "pitch_family" in hitter_str.columns else {}
    )
    # xwoba_contact: weighted average by BIP
    xwoba_str_rows = all_str[all_str["xwoba_contact"].notna() & (all_str["bip"] > 0)]
    if not xwoba_str_rows.empty:
        xwoba_str_career = xwoba_str_rows.groupby(["batter_id", "pitch_type"]).apply(
            lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
            if g["bip"].sum() > 0 else np.nan,
            include_groups=False,
        ).reset_index(name="xwoba_contact")
        career_str = career_str.merge(xwoba_str_career, on=["batter_id", "pitch_type"], how="left")
    else:
        career_str["xwoba_contact"] = np.nan

    career_str.to_parquet(DASHBOARD_DIR / "hitter_str_career.parquet", index=False)
    logger.info("Saved career hitter strength: %d rows", len(career_str))

    # =================================================================
    # 5b. Location grid data (pitcher heatmaps + hitter zone profiles)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing location grids for %d...", FROM_SEASON)

    from src.data.queries import get_pitcher_location_grid, get_hitter_zone_grid

    pitcher_loc = get_pitcher_location_grid(FROM_SEASON)
    pitcher_loc.to_parquet(DASHBOARD_DIR / "pitcher_location_grid.parquet", index=False)
    logger.info("Saved pitcher location grid: %d rows (%d pitchers)",
                len(pitcher_loc), pitcher_loc["pitcher_id"].nunique())

    hitter_zone = get_hitter_zone_grid(FROM_SEASON)
    hitter_zone.to_parquet(DASHBOARD_DIR / "hitter_zone_grid.parquet", index=False)
    logger.info("Saved hitter zone grid: %d rows (%d batters)",
                len(hitter_zone), hitter_zone["batter_id"].nunique())

    # Career-aggregated hitter zone grid (sum counts across all seasons)
    zone_frames = []
    for s in SEASONS:
        try:
            zf = get_hitter_zone_grid(s)
            zone_frames.append(zf)
        except Exception as e:
            logger.warning("Hitter zone grid for %d failed: %s", s, e)
    if zone_frames:
        all_zones = pd.concat(zone_frames, ignore_index=True)
        sum_cols = ["pitches", "swings", "whiffs", "called_strikes", "bip",
                    "xwoba_sum", "xwoba_count", "hard_hits", "barrels"]
        sum_cols = [c for c in sum_cols if c in all_zones.columns]
        career_zones = all_zones.groupby(
            ["batter_id", "batter_name", "batter_stand", "pitch_type", "grid_row", "grid_col"]
        )[sum_cols].sum().reset_index()
        # Keep most recent name per batter
        latest_names = all_zones.sort_values("pitches", ascending=False).drop_duplicates(
            "batter_id"
        )[["batter_id", "batter_name"]]
        career_zones = career_zones.drop(columns=["batter_name"]).merge(
            latest_names, on="batter_id", how="left"
        )
        career_zones.to_parquet(DASHBOARD_DIR / "hitter_zone_grid_career.parquet", index=False)
        logger.info("Saved career hitter zone grid: %d rows (%d batters)",
                    len(career_zones), career_zones["batter_id"].nunique())

    # =================================================================
    # 5c. Archetype-based matchup profiles
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing archetype-based matchup data for %d...", FROM_SEASON)

    try:
        from src.data.pitch_archetypes import (
            get_pitch_archetype_offerings,
            get_pitch_archetype_clusters,
        )
        from src.data.league_baselines import get_baselines_by_archetype_stand

        pitcher_offerings = get_pitch_archetype_offerings(FROM_SEASON)
        pitcher_offerings.to_parquet(DASHBOARD_DIR / "pitcher_offerings.parquet", index=False)
        logger.info("Saved pitcher offerings with archetypes: %d rows", len(pitcher_offerings))

        cluster_metadata = get_pitch_archetype_clusters()
        cluster_metadata.to_parquet(DASHBOARD_DIR / "pitcher_cluster_metadata.parquet", index=False)
        logger.info("Saved pitcher cluster metadata: %d archetypes", len(cluster_metadata))

        baselines_arch = get_baselines_by_archetype_stand(FROM_SEASON)
        baselines_arch.to_parquet(DASHBOARD_DIR / "baselines_arch.parquet", index=False)
        logger.info("Saved league baselines by archetype: %d rows", len(baselines_arch))
    except Exception as e:
        logger.warning("Archetype pitcher data failed: %s", e)

    try:
        from src.data.feature_eng import get_hitter_vulnerability_by_archetype

        hitter_vuln_arch = get_hitter_vulnerability_by_archetype(FROM_SEASON)
        hitter_vuln_arch.to_parquet(DASHBOARD_DIR / "hitter_vuln_arch.parquet", index=False)
        logger.info("Saved hitter vulnerability by archetype: %d rows", len(hitter_vuln_arch))

        # Career-aggregated archetype vulnerability
        arch_frames = []
        for s in SEASONS:
            try:
                df = get_hitter_vulnerability_by_archetype(s)
                arch_frames.append(df)
            except Exception as e:
                logger.warning("Archetype vuln for %d failed: %s", s, e)
        if arch_frames:
            all_arch = pd.concat(arch_frames, ignore_index=True)
            sum_cols = ["swings", "whiffs", "out_of_zone_pitches", "chase_swings", "csw"]
            sum_cols = [c for c in sum_cols if c in all_arch.columns]
            career_arch = all_arch.groupby(
                ["batter_id", "pitch_archetype"]
            )[sum_cols].sum().reset_index()
            if "swings" in career_arch.columns:
                career_arch["whiff_rate"] = (
                    career_arch["whiffs"] / career_arch["swings"].clip(lower=1)
                )
            if "out_of_zone_pitches" in career_arch.columns:
                career_arch["chase_rate"] = (
                    career_arch["chase_swings"] / career_arch["out_of_zone_pitches"].clip(lower=1)
                )
            career_arch.to_parquet(DASHBOARD_DIR / "hitter_vuln_arch_career.parquet", index=False)
            logger.info("Saved career hitter archetype vulnerability: %d rows", len(career_arch))
    except Exception as e:
        logger.warning("Archetype hitter data failed: %s", e)

    # =================================================================
    # 5d. Player-level archetype clustering (hitter + pitcher archetypes)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Exporting player archetype assignments...")
    try:
        from src.data.player_clustering import export_for_dashboard as export_archetypes

        arch_paths = export_archetypes(
            export_season=FROM_SEASON,
            output_dir=DASHBOARD_DIR,
            force_rebuild=True,
        )
        logger.info("Exported player archetypes: %s", list(arch_paths.keys()))
    except Exception as e:
        logger.warning("Player archetype export failed: %s", e)

    # =================================================================
    # 5e. Archetype matchup matrix (pitcher type vs hitter type outcomes)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building archetype matchup matrix...")
    try:
        from src.data.archetype_matchups import (
            build_archetype_matchup_matrix,
            export_archetype_matchups_for_dashboard,
        )

        arch_matrix = build_archetype_matchup_matrix(force_rebuild=True)
        logger.info("Built archetype matchup matrix: %d pairs", len(arch_matrix))

        arch_path = export_archetype_matchups_for_dashboard(output_dir=DASHBOARD_DIR)
        logger.info("Exported archetype matchup matrix to %s", arch_path)
    except Exception as e:
        logger.warning("Archetype matchup matrix failed: %s", e)

    # =================================================================
    # 6. Traditional / actual stats (2025 season from boxscores + Statcast)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing traditional stats (2025)...")
    hitter_trad = get_hitter_traditional_stats(FROM_SEASON)
    hitter_trad.to_parquet(DASHBOARD_DIR / "hitter_traditional.parquet", index=False)
    logger.info("Saved hitter traditional stats: %d players", len(hitter_trad))

    pitcher_trad = get_pitcher_traditional_stats(FROM_SEASON)
    pitcher_trad.to_parquet(DASHBOARD_DIR / "pitcher_traditional.parquet", index=False)
    logger.info("Saved pitcher traditional stats: %d players", len(pitcher_trad))

    # =================================================================
    # 6b. Hitter aggressiveness & pitcher efficiency profiles
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing hitter aggressiveness profiles...")
    hitter_agg = get_hitter_aggressiveness(FROM_SEASON)
    hitter_agg.to_parquet(DASHBOARD_DIR / "hitter_aggressiveness.parquet", index=False)
    logger.info("Saved hitter aggressiveness: %d players", len(hitter_agg))

    logger.info("Computing pitcher efficiency profiles...")
    pitcher_eff = get_pitcher_efficiency(FROM_SEASON)
    pitcher_eff.to_parquet(DASHBOARD_DIR / "pitcher_efficiency.parquet", index=False)
    logger.info("Saved pitcher efficiency: %d players", len(pitcher_eff))

    # =================================================================
    # 6c. Multi-season data (concat all seasons for season selector)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building multi-season datasets (2018-2025)...")

    # Copy full_stats parquets to dashboard dir (already have all seasons)
    import shutil
    for fname in ["hitter_full_stats.parquet", "pitcher_full_stats.parquet"]:
        src = PROJECT_ROOT / "data" / "cached" / fname
        if src.exists():
            shutil.copy2(src, DASHBOARD_DIR / fname)
            _fs = pd.read_parquet(DASHBOARD_DIR / fname)
            logger.info("Copied %s: %d rows", fname, len(_fs))
        else:
            logger.warning("Missing %s — run feature_eng first", fname)

    # Multi-season traditional stats
    trad_frames_h, trad_frames_p = [], []
    for s in SEASONS:
        try:
            trad_frames_h.append(get_hitter_traditional_stats(s))
        except Exception as e:
            logger.warning("Hitter trad stats %d failed: %s", s, e)
        try:
            trad_frames_p.append(get_pitcher_traditional_stats(s))
        except Exception as e:
            logger.warning("Pitcher trad stats %d failed: %s", s, e)
    if trad_frames_h:
        all_hitter_trad = pd.concat(trad_frames_h, ignore_index=True)
        all_hitter_trad.to_parquet(DASHBOARD_DIR / "hitter_traditional_all.parquet", index=False)
        logger.info("Saved hitter_traditional_all: %d rows", len(all_hitter_trad))
    if trad_frames_p:
        all_pitcher_trad = pd.concat(trad_frames_p, ignore_index=True)
        all_pitcher_trad.to_parquet(DASHBOARD_DIR / "pitcher_traditional_all.parquet", index=False)
        logger.info("Saved pitcher_traditional_all: %d rows", len(all_pitcher_trad))

    # Multi-season aggressiveness + efficiency
    agg_frames, eff_frames = [], []
    for s in SEASONS:
        try:
            agg_frames.append(get_hitter_aggressiveness(s))
        except Exception as e:
            logger.warning("Hitter aggressiveness %d failed: %s", s, e)
        try:
            eff_frames.append(get_pitcher_efficiency(s))
        except Exception as e:
            logger.warning("Pitcher efficiency %d failed: %s", s, e)
    if agg_frames:
        all_agg = pd.concat(agg_frames, ignore_index=True)
        all_agg.to_parquet(DASHBOARD_DIR / "hitter_aggressiveness_all.parquet", index=False)
        logger.info("Saved hitter_aggressiveness_all: %d rows", len(all_agg))
    if eff_frames:
        all_eff = pd.concat(eff_frames, ignore_index=True)
        all_eff.to_parquet(DASHBOARD_DIR / "pitcher_efficiency_all.parquet", index=False)
        logger.info("Saved pitcher_efficiency_all: %d rows", len(all_eff))

    # Multi-season arsenal (already loaded per-season during career vuln)
    arsenal_frames = []
    for s in SEASONS:
        try:
            arsenal_frames.append(get_pitcher_arsenal(s))
        except Exception as e:
            logger.warning("Pitcher arsenal %d failed: %s", s, e)
    if arsenal_frames:
        all_arsenal = pd.concat(arsenal_frames, ignore_index=True)
        all_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal_all.parquet", index=False)
        logger.info("Saved pitcher_arsenal_all: %d rows", len(all_arsenal))

    # Multi-season vuln + strength (concat the frames already loaded for career)
    if vuln_frames:
        all_vuln_seasons = pd.concat(vuln_frames, ignore_index=True)
        all_vuln_seasons.to_parquet(DASHBOARD_DIR / "hitter_vuln_all.parquet", index=False)
        logger.info("Saved hitter_vuln_all: %d rows", len(all_vuln_seasons))
    if str_frames:
        all_str_seasons = pd.concat(str_frames, ignore_index=True)
        all_str_seasons.to_parquet(DASHBOARD_DIR / "hitter_str_all.parquet", index=False)
        logger.info("Saved hitter_str_all: %d rows", len(all_str_seasons))

    # Multi-season location grids (add season column since queries don't include it)
    pitcher_loc_frames, hitter_zone_frames = [], []
    for s in SEASONS:
        try:
            _pl = get_pitcher_location_grid(s)
            _pl["season"] = s
            pitcher_loc_frames.append(_pl)
        except Exception as e:
            logger.warning("Pitcher location grid %d failed: %s", s, e)
        try:
            _hz = get_hitter_zone_grid(s)
            _hz["season"] = s
            hitter_zone_frames.append(_hz)
        except Exception as e:
            logger.warning("Hitter zone grid %d failed: %s", s, e)
    if pitcher_loc_frames:
        all_pitcher_loc = pd.concat(pitcher_loc_frames, ignore_index=True)
        all_pitcher_loc.to_parquet(DASHBOARD_DIR / "pitcher_location_grid_all.parquet", index=False)
        logger.info("Saved pitcher_location_grid_all: %d rows", len(all_pitcher_loc))
    if hitter_zone_frames:
        all_hitter_zone = pd.concat(hitter_zone_frames, ignore_index=True)
        all_hitter_zone.to_parquet(DASHBOARD_DIR / "hitter_zone_grid_all.parquet", index=False)
        logger.info("Saved hitter_zone_grid_all: %d rows", len(all_hitter_zone))

    logger.info("Multi-season datasets complete.")

    # =================================================================
    # 6d. MiLB translated prospect data
    # =================================================================
    logger.info("=" * 60)
    logger.info("Copying MiLB translated prospect data...")

    milb_files = {
        "milb_translated_batters.parquet": "MiLB translated batters",
        "milb_translated_pitchers.parquet": "MiLB translated pitchers",
        "milb_batter_factors.parquet": "MiLB batter translation factors",
        "milb_pitcher_factors.parquet": "MiLB pitcher translation factors",
    }
    cached_dir = PROJECT_ROOT / "data" / "cached"
    for fname, label in milb_files.items():
        src_path = cached_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, DASHBOARD_DIR / fname)
            _df = pd.read_parquet(src_path)
            logger.info("Copied %s: %d rows", label, len(_df))
        else:
            logger.warning("Missing %s — run build_milb_translations.py first", fname)

    # =================================================================
    # 6e. Prospect readiness scores + FanGraphs rankings
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building prospect readiness scores...")

    try:
        from src.models.mlb_readiness import train_readiness_model, score_prospects
        from src.data.db import read_sql

        target_season = FROM_SEASON + 1  # e.g. 2026

        # Train model and score prospects
        bundle = train_readiness_model()
        prospects = score_prospects(projection_season=target_season)
        logger.info(
            "Readiness model: AUC=%.3f, scored %d prospects",
            bundle["train_auc"], len(prospects),
        )

        # Load FanGraphs rankings for the target season
        rankings = read_sql(
            "SELECT player_id, player_name, org, position, overall_rank, "
            "org_rank, future_value, risk, eta, source "
            "FROM production.dim_prospect_ranking "
            f"WHERE season = {target_season}",
            {},
        )
        if not rankings.empty:
            # Prefer fg_report, deduplicate
            rankings = rankings.sort_values(
                "source", ascending=True  # fg_report before fg_updated
            ).drop_duplicates("player_id", keep="first")
            logger.info("FanGraphs rankings: %d prospects for %d", len(rankings), target_season)
        else:
            logger.warning("No FanGraphs rankings for season %d", target_season)

        # Merge readiness scores with rankings
        if not prospects.empty:
            # Select readiness columns
            readiness_cols = [
                "player_id", "name", "pos_group", "primary_position",
                "max_level", "max_level_num", "readiness_score", "readiness_tier",
                "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "k_bb_diff", "sb_rate",
                "youngest_age_rel", "min_age", "career_milb_pa",
                "n_above", "total_at_pos_in_org",
            ]
            available_cols = [c for c in readiness_cols if c in prospects.columns]
            prospect_out = prospects[available_cols].copy()

            # Merge FG rankings for extra columns (org, org_rank, risk, eta)
            if not rankings.empty:
                fg_cols = ["player_id", "org", "overall_rank", "org_rank", "risk", "eta"]
                fg_available = [c for c in fg_cols if c in rankings.columns]
                prospect_out = prospect_out.merge(
                    rankings[fg_available], on="player_id", how="left",
                )

            prospect_out.to_parquet(
                DASHBOARD_DIR / "prospect_readiness.parquet", index=False,
            )
            logger.info("Saved prospect_readiness.parquet: %d rows", len(prospect_out))
        else:
            logger.warning("No prospect readiness scores generated")

    except Exception:
        logger.exception("Failed to build prospect readiness scores")

    # =================================================================
    # 6f. TDD Prospect Rankings (composite score)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building TDD prospect rankings...")

    try:
        from src.models.prospect_ranking import rank_prospects, rank_pitching_prospects

        prospect_rankings = rank_prospects(projection_season=FROM_SEASON + 1)
        if not prospect_rankings.empty:
            prospect_rankings.to_parquet(
                DASHBOARD_DIR / "prospect_rankings.parquet", index=False,
            )
            logger.info(
                "Saved prospect_rankings.parquet: %d rows", len(prospect_rankings),
            )
        else:
            logger.warning("No batting prospect rankings generated")

        pitching_prospect_rankings = rank_pitching_prospects(
            projection_season=FROM_SEASON + 1,
        )
        if not pitching_prospect_rankings.empty:
            pitching_prospect_rankings.to_parquet(
                DASHBOARD_DIR / "pitching_prospect_rankings.parquet", index=False,
            )
            logger.info(
                "Saved pitching_prospect_rankings.parquet: %d rows",
                len(pitching_prospect_rankings),
            )
        else:
            logger.warning("No pitching prospect rankings generated")

    except Exception:
        logger.exception("Failed to build prospect rankings")

    # =================================================================
    # 6f-ii. Prospect-to-MLB player comps
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building prospect-to-MLB player comps...")

    try:
        from src.models.prospect_comps import find_all_comps

        comps = find_all_comps(projection_season=FROM_SEASON + 1)
        for key, cdf in comps.items():
            if not cdf.empty:
                fname = f"prospect_comps_{key}.parquet"
                cdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(cdf))

    except Exception:
        logger.exception("Failed to build prospect comps")

    # =================================================================
    # 6g. MLB Positional Rankings (2026 value)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building MLB positional rankings...")

    try:
        from src.models.player_rankings import rank_all

        rankings = rank_all(
            season=FROM_SEASON, projection_season=FROM_SEASON + 1,
        )
        for key, rdf in rankings.items():
            if not rdf.empty:
                fname = f"{key}_rankings.parquet"
                rdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(rdf))

    except Exception:
        logger.exception("Failed to build MLB positional rankings")

    # =================================================================
    # 6h. Hitter position eligibility (multi-position depth chart)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building hitter position eligibility...")

    try:
        from src.models.player_rankings import get_hitter_position_eligibility

        elig = get_hitter_position_eligibility(season=FROM_SEASON, min_starts=10)
        if not elig.empty:
            elig.to_parquet(
                DASHBOARD_DIR / "hitter_position_eligibility.parquet",
                index=False,
            )
            n_players = elig["player_id"].nunique()
            n_rows = len(elig)
            logger.info(
                "Saved hitter_position_eligibility.parquet: "
                "%d rows (%d unique players)",
                n_rows, n_players,
            )
        else:
            logger.warning("No position eligibility data generated")

    except Exception:
        logger.exception("Failed to build position eligibility")

    # =================================================================
    # 6i. Hitter breakout archetypes
    # =================================================================
    logger.info("=" * 60)
    logger.info("Scoring hitter breakout archetypes...")

    try:
        from src.models.breakout_model import score_breakout_candidates

        breakouts = score_breakout_candidates(season=FROM_SEASON, min_pa=200)
        if not breakouts.empty:
            breakouts.to_parquet(
                DASHBOARD_DIR / "hitter_breakout_candidates.parquet",
                index=False,
            )
            logger.info(
                "Saved hitter_breakout_candidates.parquet: %d rows",
                len(breakouts),
            )
        else:
            logger.warning("No breakout candidates generated")

    except Exception:
        logger.exception("Failed to compute hitter breakout archetypes")

    # =================================================================
    # 6j. Pitcher breakout archetypes
    # =================================================================
    logger.info("=" * 60)
    logger.info("Scoring pitcher breakout archetypes...")

    try:
        from src.models.pitcher_breakout_model import score_pitcher_breakout_candidates

        p_breakouts = score_pitcher_breakout_candidates(season=FROM_SEASON, min_bf=200)
        if not p_breakouts.empty:
            p_breakouts.to_parquet(
                DASHBOARD_DIR / "pitcher_breakout_candidates.parquet",
                index=False,
            )
            logger.info(
                "Saved pitcher_breakout_candidates.parquet: %d rows",
                len(p_breakouts),
            )
        else:
            logger.warning("No pitcher breakout candidates generated")

    except Exception:
        logger.exception("Failed to compute pitcher breakout archetypes")

    # =================================================================
    # 7. Save preseason snapshot (frozen projections for end-of-season comparison)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Saving preseason snapshot...")
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date
    snapshot_date = date.today().isoformat()
    target_season = FROM_SEASON + 1  # projecting INTO this season

    hitter_proj["snapshot_date"] = snapshot_date
    hitter_proj["target_season"] = target_season
    pitcher_proj["snapshot_date"] = snapshot_date
    pitcher_proj["target_season"] = target_season

    h_path = snapshot_dir / f"hitter_projections_{target_season}_preseason.parquet"
    p_path = snapshot_dir / f"pitcher_projections_{target_season}_preseason.parquet"
    hitter_proj.to_parquet(h_path, index=False)
    pitcher_proj.to_parquet(p_path, index=False)
    logger.info("Saved preseason snapshot: %s, %s", h_path.name, p_path.name)

    # Remove snapshot columns from live projections (they stay in the snapshot files)
    hitter_proj.drop(columns=["snapshot_date", "target_season"], inplace=True)
    pitcher_proj.drop(columns=["snapshot_date", "target_season"], inplace=True)

    # =================================================================
    # 8. Backtest summaries (copy outputs → dashboard, compute confidence tiers)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Processing backtest summaries...")
    precompute_backtest_summaries()

    # =================================================================
    # 9. Game Simulator Data (Layer 3 v2)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Generating game simulator data...")

    # 9a. Hitter K% and BB% posterior samples
    for stat_name, npz_name in [("k_rate", "hitter_k_samples"), ("bb_rate", "hitter_bb_samples")]:
        if stat_name not in hitter_results:
            logger.warning("No %s model in hitter_results — skipping %s", stat_name, npz_name)
            continue
        stat_data = hitter_results[stat_name]["data"]
        stat_trace = hitter_results[stat_name]["trace"]
        stat_df = stat_data["df"]
        stat_active = stat_df[
            (stat_df["season"] == FROM_SEASON) & (stat_df["pa"] >= 50)
        ]["batter_id"].unique()

        h_samples_dict: dict[str, np.ndarray] = {}
        for bid in stat_active:
            try:
                samples = extract_hitter_rate_samples(
                    stat_trace, stat_data,
                    batter_id=int(bid),
                    season=FROM_SEASON,
                    project_forward=True,
                )
                h_samples_dict[str(int(bid))] = samples
            except ValueError:
                continue

        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **h_samples_dict)
        logger.info("Saved hitter %s posterior samples for %d batters", stat_name, len(h_samples_dict))

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **h_samples_dict)
        logger.info("Saved preseason hitter %s samples snapshot", stat_name)

    # 9b. Hitter HR synthetic Beta samples (no Bayesian model — use Beta posterior)
    logger.info("Generating synthetic hitter HR samples...")
    hr_season = hitter_ext[hitter_ext["season"] == FROM_SEASON].copy()
    hr_samples_dict: dict[str, np.ndarray] = {}
    rng_hr = np.random.default_rng(42)
    for _, row in hr_season.iterrows():
        bid = int(row["batter_id"])
        pa = int(row.get("pa", row.get("plate_appearances", 0)))
        hr = int(row.get("hr", row.get("home_runs", 0)))
        if pa < 50:
            continue
        # Beta(hr + 1, pa - hr + 1) posterior
        hr_rate_samples = rng_hr.beta(hr + 1, pa - hr + 1, size=10_000)
        hr_samples_dict[str(bid)] = hr_rate_samples.astype(np.float32)
    np.savez_compressed(DASHBOARD_DIR / "hitter_hr_samples.npz", **hr_samples_dict)
    logger.info("Saved hitter HR Beta posterior samples for %d batters", len(hr_samples_dict))

    # 9c. Train + save exit model
    logger.info("Training pitcher exit model...")
    try:
        exit_training = get_exit_model_training_data(SEASONS)
        exit_tendencies = get_pitcher_exit_tendencies(SEASONS)
        exit_model = ExitModel()
        exit_metrics = exit_model.train(exit_training, exit_tendencies)
        logger.info(
            "Exit model: AUC=%.4f, n=%d",
            exit_metrics["auc"], exit_metrics["n_samples"],
        )
        exit_model.save(DASHBOARD_DIR / "exit_model.pkl")
        exit_tendencies.to_parquet(
            DASHBOARD_DIR / "pitcher_exit_tendencies.parquet", index=False,
        )
        logger.info("Saved exit model + tendencies")
    except Exception:
        logger.exception("Failed to train exit model")

    # 9d. Pitch count features
    logger.info("Computing pitch count features...")
    try:
        pitcher_pc = get_pitcher_pitch_count_features(SEASONS)
        pitcher_pc.to_parquet(
            DASHBOARD_DIR / "pitcher_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved pitcher pitch count features: %d rows", len(pitcher_pc))

        batter_pc = get_batter_pitch_count_features(SEASONS)
        batter_pc.to_parquet(
            DASHBOARD_DIR / "batter_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved batter pitch count features: %d rows", len(batter_pc))
    except Exception:
        logger.exception("Failed to compute pitch count features")

    # 9e. TTO profiles
    logger.info("Computing TTO adjustment profiles...")
    try:
        tto_profiles = get_tto_adjustment_profiles(SEASONS)
        tto_profiles.to_parquet(
            DASHBOARD_DIR / "tto_profiles.parquet", index=False,
        )
        logger.info("Saved TTO profiles: %d rows", len(tto_profiles))
    except Exception:
        logger.exception("Failed to compute TTO profiles")

    # 9f. Team bullpen rates
    logger.info("Computing team bullpen rates...")
    try:
        bullpen_rates = get_team_bullpen_rates(SEASONS)
        bullpen_rates.to_parquet(
            DASHBOARD_DIR / "team_bullpen_rates.parquet", index=False,
        )
        logger.info("Saved team bullpen rates: %d rows", len(bullpen_rates))
    except Exception:
        logger.exception("Failed to compute team bullpen rates")

    logger.info("Game simulator data complete.")

    # =================================================================
    # 8. Probable starters (greedy position assignment from lineup priors)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing probable starters...")
    try:
        from src.data.db import read_sql as _read_sql

        # Get lineup priors (position + batting order history for all players)
        lineup_priors = get_lineup_priors(FROM_SEASON)

        # Get current active roster with team info
        roster = _read_sql("""
            SELECT dr.player_id, dp.player_name, dr.org_id,
                   dr.primary_position, dr.is_starter,
                   dt.abbreviation AS team_abbr
            FROM production.dim_roster dr
            JOIN production.dim_player dp ON dr.player_id = dp.player_id
            JOIN production.dim_team dt ON dr.org_id = dt.team_id
            WHERE dr.level = 'MLB'
              AND dr.roster_status IN ('active', 'nri')
              AND dr.primary_position NOT IN ('SP', 'RP', 'P')
        """)

        POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

        all_starters = []
        for team_abbr, team_roster in roster.groupby("team_abbr"):
            team_pids = set(team_roster["player_id"])

            # Get this team's players' lineup priors (from ANY team in 2025)
            team_priors = lineup_priors[
                lineup_priors["player_id"].isin(team_pids)
            ].copy()

            assigned = set()
            team_starters = []

            # Greedy: for each position, pick the available player with the
            # most starts there
            for pos in POSITIONS:
                pos_candidates = team_priors[
                    (team_priors["position"] == pos)
                    & (~team_priors["player_id"].isin(assigned))
                ].sort_values("starts", ascending=False)

                if not pos_candidates.empty:
                    best = pos_candidates.iloc[0]
                    pid = int(best["player_id"])
                    assigned.add(pid)
                    prow = team_roster[team_roster["player_id"] == pid].iloc[0]
                    team_starters.append({
                        "team_abbr": team_abbr,
                        "position": pos,
                        "player_id": pid,
                        "player_name": prow["player_name"],
                        "batting_order": int(best["batting_order_mode"]),
                        "starts_at_position": int(best["starts"]),
                        "pct_at_position": float(best["pct"]),
                    })
                else:
                    # Fallback: use dim_roster primary_position
                    fallback = team_roster[
                        (team_roster["primary_position"] == pos)
                        & (~team_roster["player_id"].isin(assigned))
                    ]
                    if not fallback.empty:
                        fb = fallback.iloc[0]
                        pid = int(fb["player_id"])
                        assigned.add(pid)
                        team_starters.append({
                            "team_abbr": team_abbr,
                            "position": pos,
                            "player_id": pid,
                            "player_name": fb["player_name"],
                            "batting_order": 5,  # default middle-order
                            "starts_at_position": 0,
                            "pct_at_position": 0.0,
                        })

            all_starters.extend(team_starters)

        probable_df = pd.DataFrame(all_starters)
        if not probable_df.empty:
            probable_df.to_parquet(
                DASHBOARD_DIR / "probable_starters.parquet", index=False,
            )
            logger.info(
                "Saved probable starters: %d players across %d teams",
                len(probable_df), probable_df["team_abbr"].nunique(),
            )
        else:
            logger.warning("No probable starters computed")

        # Also save the full lineup priors for reference
        if not lineup_priors.empty:
            lineup_priors.to_parquet(
                DASHBOARD_DIR / "lineup_priors.parquet", index=False,
            )
            logger.info("Saved lineup priors: %d rows", len(lineup_priors))
    except Exception:
        logger.exception("Failed to compute probable starters")

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Dashboard pre-computation complete!")
    logger.info("  Hitter projections:  %d players", len(hitter_proj))
    logger.info("  Pitcher projections: %d players", len(pitcher_proj))
    logger.info("  Hitter counting:     %d players", len(hitter_counting))
    logger.info("  Pitcher counting:    %d players", len(pitcher_counting))
    logger.info("  Hitter traditional:  %d players", len(hitter_trad))
    logger.info("  Pitcher traditional: %d players", len(pitcher_trad))
    logger.info("  Hitter aggressiveness:%d players", len(hitter_agg))
    logger.info("  Pitcher efficiency:  %d players", len(pitcher_eff))
    logger.info("  K%% samples:          %d pitchers", len(k_samples_dict))
    logger.info("  BF priors:           %d pitcher-seasons", len(bf_priors))
    logger.info("  Pitcher arsenal:     %d rows", len(pitcher_arsenal))
    logger.info("  Hitter vulnerability:%d rows", len(hitter_vuln))
    logger.info("  Hitter strength:     %d rows", len(hitter_str))
    logger.info("  MiLB prospects:      milb_translated_batters, milb_translated_pitchers, factors")
    logger.info("  Multi-season files:  hitter/pitcher trad_all, agg_all, eff_all, arsenal_all, vuln_all, str_all, loc_all, zone_all, full_stats")
    logger.info("  Output dir:          %s", DASHBOARD_DIR)


if __name__ == "__main__":
    main()
