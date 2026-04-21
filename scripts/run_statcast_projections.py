"""Run all Statcast metric projections + compute grade CIs.

Expected runtime: ~30-50 minutes for all 10 models.

Usage:
    myenv/Scripts/python scripts/run_statcast_projections.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import time

import pandas as pd

from src.data.paths import dashboard_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("statcast_projections")

DASHBOARD_DIR = dashboard_dir()


def main():
    logger.info("=" * 60)
    logger.info("STATCAST METRIC PROJECTIONS - Bayesian AR(1)")
    logger.info("=" * 60)

    t_start = time.time()

    from src.models.statcast_projections import (
        fit_all_statcast_models,
        STATCAST_CONFIGS,
        compute_projected_grades_with_ci,
    )

    hitter_metrics = [k for k, v in STATCAST_CONFIGS.items() if v.player_type == "hitter"]
    pitcher_metrics = [k for k, v in STATCAST_CONFIGS.items() if v.player_type == "pitcher"]

    logger.info("Hitter metrics: %s", hitter_metrics)
    logger.info("Pitcher metrics: %s", pitcher_metrics)

    # Fit hitter models
    logger.info("=" * 60)
    logger.info("FITTING HITTER STATCAST MODELS...")
    hitter_proj = fit_all_statcast_models(
        seasons=list(range(2020, 2026)),
        metrics=hitter_metrics,
        draws=1000, tune=500, chains=2,
    )
    if not hitter_proj.empty:
        hitter_proj.to_parquet(DASHBOARD_DIR / "hitter_statcast_projections.parquet", index=False)
        logger.info("Saved hitter_statcast_projections.parquet: %d players", len(hitter_proj))

    # Fit pitcher models
    logger.info("=" * 60)
    logger.info("FITTING PITCHER STATCAST MODELS...")
    pitcher_proj = fit_all_statcast_models(
        seasons=list(range(2020, 2026)),
        metrics=pitcher_metrics,
        draws=1000, tune=500, chains=2,
    )
    if not pitcher_proj.empty:
        pitcher_proj.to_parquet(DASHBOARD_DIR / "pitcher_statcast_projections.parquet", index=False)
        logger.info("Saved pitcher_statcast_projections.parquet: %d players", len(pitcher_proj))

    # Compute grade CIs
    logger.info("=" * 60)
    logger.info("COMPUTING PROJECTED GRADE CIs...")

    hitter_rankings = pd.read_parquet(DASHBOARD_DIR / "hitters_rankings.parquet")
    pitcher_rankings = pd.read_parquet(DASHBOARD_DIR / "pitchers_rankings.parquet")

    hitter_ci, pitcher_ci = compute_projected_grades_with_ci(
        hitter_proj, pitcher_proj, hitter_rankings, pitcher_rankings,
    )

    if not hitter_ci.empty:
        hitter_ci.to_parquet(DASHBOARD_DIR / "hitter_grade_ci.parquet", index=False)
        logger.info("Saved hitter_grade_ci.parquet: %d players", len(hitter_ci))
        for name in ["Ohtani", "Judge", "Witt", "Soto"]:
            match = hitter_rankings[hitter_rankings["batter_name"].str.contains(name, na=False)]
            if not match.empty:
                pid = match["batter_id"].iloc[0]
                ci_row = hitter_ci[hitter_ci["player_id"] == pid]
                if not ci_row.empty:
                    r = ci_row.iloc[0]
                    logger.info(
                        "  %s: DR=%.1f (%.1f-%.1f)  Hit=%d(%d-%d) Power=%d(%d-%d)",
                        name, r.get("tools_rating", 0),
                        r.get("tools_rating_lo", 0), r.get("tools_rating_hi", 0),
                        r.get("grade_hit", 0), r.get("grade_hit_lo", 0), r.get("grade_hit_hi", 0),
                        r.get("grade_power", 0), r.get("grade_power_lo", 0), r.get("grade_power_hi", 0),
                    )

    if not pitcher_ci.empty:
        pitcher_ci.to_parquet(DASHBOARD_DIR / "pitcher_grade_ci.parquet", index=False)
        logger.info("Saved pitcher_grade_ci.parquet: %d players", len(pitcher_ci))
        for name in ["Skubal", "Skenes", "Glasnow", "Sale"]:
            match = pitcher_rankings[pitcher_rankings["pitcher_name"].str.contains(name, na=False)]
            if not match.empty:
                pid = match["pitcher_id"].iloc[0]
                ci_row = pitcher_ci[pitcher_ci["player_id"] == pid]
                if not ci_row.empty:
                    r = ci_row.iloc[0]
                    logger.info(
                        "  %s: DR=%.1f (%.1f-%.1f)  Stuff=%d(%d-%d) Cmd=%d(%d-%d)",
                        name, r.get("tools_rating", 0),
                        r.get("tools_rating_lo", 0), r.get("tools_rating_hi", 0),
                        r.get("grade_stuff", 0), r.get("grade_stuff_lo", 0), r.get("grade_stuff_hi", 0),
                        r.get("grade_command", 0), r.get("grade_command_lo", 0), r.get("grade_command_hi", 0),
                    )

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
