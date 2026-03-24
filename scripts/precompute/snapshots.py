"""Precompute: Preseason snapshot + backtest summaries."""
from __future__ import annotations

import logging
import shutil
from datetime import date

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, PROJECT_ROOT

logger = logging.getLogger("precompute.snapshots")


def run_projection_snapshots(
    *,
    hitter_proj: pd.DataFrame,
    pitcher_proj: pd.DataFrame,
    from_season: int = FROM_SEASON,
) -> None:
    """Save preseason snapshot (frozen projections for end-of-season comparison)."""
    logger.info("=" * 60)
    logger.info("Saving preseason snapshot...")
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not hitter_proj.empty and not pitcher_proj.empty:
        snapshot_date = date.today().isoformat()
        target_season = from_season + 1  # projecting INTO this season

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
    else:
        logger.warning("Skipping projection snapshot -- projections not available")


def run_backtest_summaries() -> None:
    """Copy backtest results to dashboard directory and compute confidence tiers."""
    from src.evaluation.confidence_tiers import assign_confidence_tiers, tiers_to_dataframe

    logger.info("=" * 60)
    logger.info("Processing backtest summaries...")

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
        logger.info("No game prop backtest summary found at %s -- skipping tier computation", summary_path)

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
