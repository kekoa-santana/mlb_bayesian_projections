#!/usr/bin/env python
"""
Pre-compute all data needed by the Streamlit dashboard.

Fits composite hitter + pitcher models, extracts projections and
posterior K% samples, computes BF priors, and saves everything to
data/dashboard/.

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

from src.data.feature_eng import (
    build_multi_season_pitcher_k_data,
    get_pitcher_arsenal,
    get_hitter_vulnerability,
    get_hitter_strength,
)
from src.data.queries import get_pitcher_game_logs
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_k_model import extract_pitcher_k_rate_samples
from src.models.hitter_projections import (
    fit_all_models as fit_hitter_models,
    project_forward as project_hitter_forward,
)
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
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute dashboard data")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws for fast iteration",
    )
    return parser.parse_args()


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
    # 1. Hitter composite projections
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting hitter composite models...")
    hitter_results = fit_hitter_models(
        seasons=SEASONS, min_pa=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    hitter_proj = project_hitter_forward(
        hitter_results, from_season=FROM_SEASON, min_pa=200,
    )
    hitter_proj.to_parquet(DASHBOARD_DIR / "hitter_projections.parquet", index=False)
    logger.info("Saved hitter projections: %d players", len(hitter_proj))

    # =================================================================
    # 2. Pitcher composite projections
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting pitcher composite models...")
    pitcher_results = fit_pitcher_models(
        seasons=SEASONS, min_bf=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    pitcher_proj = project_pitcher_forward(
        pitcher_results, from_season=FROM_SEASON, min_bf=200,
    )
    pitcher_proj.to_parquet(DASHBOARD_DIR / "pitcher_projections.parquet", index=False)
    logger.info("Saved pitcher projections: %d players", len(pitcher_proj))

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

    bf_priors = compute_pitcher_bf_priors(game_logs)
    bf_priors.to_parquet(DASHBOARD_DIR / "bf_priors.parquet", index=False)
    logger.info("Saved BF priors: %d pitcher-seasons", len(bf_priors))

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

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Dashboard pre-computation complete!")
    logger.info("  Hitter projections:  %d players", len(hitter_proj))
    logger.info("  Pitcher projections: %d players", len(pitcher_proj))
    logger.info("  K%% samples:          %d pitchers", len(k_samples_dict))
    logger.info("  BF priors:           %d pitcher-seasons", len(bf_priors))
    logger.info("  Pitcher arsenal:     %d rows", len(pitcher_arsenal))
    logger.info("  Hitter vulnerability:%d rows", len(hitter_vuln))
    logger.info("  Hitter strength:     %d rows", len(hitter_str))
    logger.info("  Output dir:          %s", DASHBOARD_DIR)


if __name__ == "__main__":
    main()
