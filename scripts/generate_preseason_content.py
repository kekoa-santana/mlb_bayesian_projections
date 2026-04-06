#!/usr/bin/env python
"""
Generate pre-season 2026 content cards using Bayesian K% projections.

Trains pitcher and hitter K% models on 2018-2025, projects posteriors
forward into 2026, and produces Twitter/X-ready PNG cards.

Usage
-----
    python scripts/generate_preseason_content.py          # full quality
    python scripts/generate_preseason_content.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_eng import (
    build_multi_season_k_data,
    build_multi_season_pitcher_k_data,
)
from src.models.posterior_utils import extract_pitcher_k_rate_samples
from src.models.k_rate_model import (
    extract_player_posteriors,
    fit_k_rate_model,
    prepare_model_data,
)
from src.models.pitcher_k_rate_model import (
    check_pitcher_convergence,
    extract_pitcher_posteriors,
    fit_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)
from src.viz.projections import (
    enrich_with_team_info,
    find_movers,
    plot_hitter_k_movers,
    plot_pitcher_card,
    plot_pitcher_k_movers,
    project_posteriors_forward,
)
from src.viz.theme import apply_theme, format_pct, save_card

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_content")

TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
PROJECT_FROM = 2025
SEASON_LABEL = "2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pre-season content")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use fewer draws for fast iteration",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/content/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    if args.quick:
        draws, tune, chains = 500, 250, 2
        logger.info("QUICK mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    else:
        draws, tune, chains = 2000, 1000, 4
        logger.info("FULL mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)

    apply_theme()
    output_paths: list[Path] = []

    # =================================================================
    # 1. Pitcher K% model
    # =================================================================
    logger.info("Building pitcher data for %s", TRAIN_SEASONS)
    df_pitcher = build_multi_season_pitcher_k_data(TRAIN_SEASONS, min_bf=9)

    logger.info("Preparing and fitting pitcher K%% model (%d player-seasons)",
                len(df_pitcher))
    pitcher_data = prepare_pitcher_model_data(df_pitcher)
    _pitcher_model, pitcher_trace = fit_pitcher_k_rate_model(
        pitcher_data, draws=draws, tune=tune, chains=chains,
    )

    convergence = check_pitcher_convergence(pitcher_trace)
    logger.info("Pitcher convergence: %s", "OK" if convergence["converged"] else "ISSUES")

    # Project forward
    logger.info("Projecting pitcher posteriors from %d → %s", PROJECT_FROM, SEASON_LABEL)
    pitcher_proj = project_posteriors_forward(
        pitcher_trace, pitcher_data, PROJECT_FROM,
        id_col="pitcher_id", name_col="pitcher_name",
        hand_col="pitch_hand", trials_col="batters_faced",
    )

    # Merge is_starter flag from source data
    starter_map = (
        df_pitcher[df_pitcher["season"] == PROJECT_FROM]
        .set_index("pitcher_id")["is_starter"]
    )
    pitcher_proj["is_starter"] = (
        pitcher_proj["pitcher_id"].map(starter_map).fillna(0).astype(int)
    )

    # Enrich with team
    pitcher_proj = enrich_with_team_info(pitcher_proj, id_col="pitcher_id")

    # Find movers (starters only, min 200 BF, top 8)
    p_improvers, p_decliners = find_movers(
        pitcher_proj, n_top=8, min_trials=200,
        player_type="pitcher", role_filter="starter",
    )

    logger.info("Pitcher improvers:\n%s",
                p_improvers[["pitcher_name", "team", "observed_k_rate",
                             "projected_k_rate_mean", "delta"]].to_string())
    logger.info("Pitcher decliners:\n%s",
                p_decliners[["pitcher_name", "team", "observed_k_rate",
                             "projected_k_rate_mean", "delta"]].to_string())

    # Chart 1: Pitcher K% movers
    fig = plot_pitcher_k_movers(p_improvers, p_decliners, SEASON_LABEL)
    path = save_card(fig, "pitcher_k_movers_2026", output_dir)
    output_paths.append(path)
    logger.info("Saved: %s", path)

    # =================================================================
    # 2. Hitter K% model
    # =================================================================
    logger.info("Building hitter data for %s", TRAIN_SEASONS)
    df_hitter = build_multi_season_k_data(TRAIN_SEASONS, min_pa=1)

    logger.info("Preparing and fitting hitter K%% model (%d player-seasons)",
                len(df_hitter))
    hitter_data = prepare_model_data(df_hitter)
    _hitter_model, hitter_trace = fit_k_rate_model(
        hitter_data, draws=draws, tune=tune, chains=chains,
    )

    # Project forward
    logger.info("Projecting hitter posteriors from %d → %s", PROJECT_FROM, SEASON_LABEL)
    hitter_proj = project_posteriors_forward(
        hitter_trace, hitter_data, PROJECT_FROM,
        id_col="batter_id", name_col="batter_name",
        hand_col="batter_stand", trials_col="pa",
    )
    hitter_proj = enrich_with_team_info(hitter_proj, id_col="batter_id")

    h_improvers, h_decliners = find_movers(
        hitter_proj, n_top=8, min_trials=200,
        player_type="hitter",
        id_col="batter_id", trials_col="pa",
    )

    logger.info("Hitter improvers (less K):\n%s",
                h_improvers[["batter_name", "team", "observed_k_rate",
                             "projected_k_rate_mean", "delta"]].to_string())
    logger.info("Hitter decliners (more K):\n%s",
                h_decliners[["batter_name", "team", "observed_k_rate",
                             "projected_k_rate_mean", "delta"]].to_string())

    # Chart 2: Hitter K% movers
    fig = plot_hitter_k_movers(h_improvers, h_decliners, SEASON_LABEL)
    path = save_card(fig, "hitter_k_movers_2026", output_dir)
    output_paths.append(path)
    logger.info("Saved: %s", path)

    # =================================================================
    # 3. Individual pitcher cards
    # =================================================================
    # Generate cards for top improver, top decliner, and a marquee pick
    card_candidates = []
    if len(p_improvers) > 0:
        card_candidates.append(("top_improver", p_improvers.iloc[0]))
    if len(p_decliners) > 0:
        card_candidates.append(("top_decliner", p_decliners.iloc[0]))

    # Add a 3rd card: 2nd improver if available, else 2nd decliner
    if len(p_improvers) > 1:
        card_candidates.append(("notable_improver", p_improvers.iloc[1]))
    elif len(p_decliners) > 1:
        card_candidates.append(("notable_decliner", p_decliners.iloc[1]))

    for tag, row in card_candidates:
        pitcher_id = int(row["pitcher_id"])
        name = str(row["pitcher_name"])
        team = str(row.get("team", ""))
        hand = str(row.get("pitch_hand", ""))
        obs_k = float(row["observed_k_rate"])

        logger.info("Generating card for %s (id=%d)", name, pitcher_id)

        try:
            samples = extract_pitcher_k_rate_samples(
                pitcher_trace, pitcher_data,
                pitcher_id=pitcher_id,
                season=PROJECT_FROM,
                project_forward=True,
            )
        except ValueError as e:
            logger.warning("Skipping card for %s: %s", name, e)
            continue

        summary = {
            "mean": float(np.mean(samples)),
            "ci_lo": float(np.percentile(samples, 2.5)),
            "ci_hi": float(np.percentile(samples, 97.5)),
        }

        fig = plot_pitcher_card(
            pitcher_name=name,
            team=team,
            hand=hand,
            observed_k_rate=obs_k,
            projected_samples=samples,
            projected_summary=summary,
            season_label=SEASON_LABEL,
        )

        # Build filename from name
        name_slug = name.replace(", ", "_").replace(" ", "_").lower()
        path = save_card(fig, f"pitcher_card_{name_slug}_{tag}", output_dir, aspect="1:1")
        output_paths.append(path)
        logger.info("Saved: %s", path)

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Content generation complete. %d files produced:", len(output_paths))
    for p in output_paths:
        logger.info("  %s", p)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
