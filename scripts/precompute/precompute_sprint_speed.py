#!/usr/bin/env python
"""
Precompute per-batter sprint speed with Bayesian shrinkage.

Loads cached sprint speed data (from Statcast), applies light Bayesian
regression toward league average (27.0 ft/sec), and saves a dashboard-ready
parquet.

Sprint speed is already a seasonal aggregate from Baseball Savant
(average of all competitive runs), so it requires only light shrinkage.
YoY correlation is r=0.90, meaning it is very stable. Shrinkage is based
on the number of observed seasons per player, not raw run counts.

Output: data/dashboard/batter_sprint_speed.parquet
    Columns: player_id, season, sprint_speed, sprint_speed_regressed
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from precompute import DASHBOARD_DIR
from src.data.feature_eng import get_cached_sprint_speed

logger = logging.getLogger(__name__)

# League average sprint speed (2022-2025 Statcast data, ~27.0-27.3)
LEAGUE_SPRINT_SPEED = 27.0

# Shrinkage constant in "equivalent seasons". With 1 season of data,
# the player gets 67% weight on their observed speed; with 2+ seasons,
# weight rises to 80%+. This is intentionally light because Savant's
# sprint_speed is already an aggregated seasonal metric.
SHRINKAGE_K = 0.5

# Seasons to compute sprint speed for
SEASONS = [2022, 2023, 2024, 2025]


def compute_sprint_speeds() -> pd.DataFrame:
    """Load cached sprint speed data and apply Bayesian shrinkage.

    Shrinkage is based on number of observed seasons per player. Players
    with more seasons of data get less regression toward league average.

    Returns DataFrame with columns:
        player_id, season, sprint_speed, sprint_speed_regressed
    """
    all_dfs: list[pd.DataFrame] = []

    for season in SEASONS:
        try:
            df = get_cached_sprint_speed(season)
        except Exception as e:
            logger.warning("Could not load sprint speed for %d: %s", season, e)
            continue

        if df.empty:
            logger.warning("No sprint speed data for %d", season)
            continue

        df = df[["player_id", "sprint_speed"]].copy()
        df["season"] = season
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No sprint speed data found across any season")
        return pd.DataFrame(
            columns=["player_id", "season", "sprint_speed", "sprint_speed_regressed"]
        )

    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop rows with missing sprint speed
    combined = combined.dropna(subset=["sprint_speed"])

    # Count seasons per player (used for shrinkage weight)
    season_counts = (
        combined.groupby("player_id")["season"]
        .nunique()
        .reset_index(name="n_seasons")
    )
    combined = combined.merge(season_counts, on="player_id")

    # Bayesian shrinkage toward league average based on season count.
    # regressed = (n / (n + K)) * observed + (K / (n + K)) * league_avg
    # With K=0.5: 1 season -> 67% weight, 2 -> 80%, 3 -> 86%, 4 -> 89%
    reliability = combined["n_seasons"] / (combined["n_seasons"] + SHRINKAGE_K)
    combined["sprint_speed_regressed"] = (
        reliability * combined["sprint_speed"]
        + (1 - reliability) * LEAGUE_SPRINT_SPEED
    )

    # Round for storage
    combined["sprint_speed_regressed"] = combined["sprint_speed_regressed"].round(3)

    # Select output columns
    result = combined[
        ["player_id", "season", "sprint_speed", "sprint_speed_regressed"]
    ].copy()

    # Log summary stats
    logger.info(
        "Sprint speed summary: N=%d player-seasons, "
        "raw_mean=%.2f, raw_std=%.2f, "
        "regressed_mean=%.2f, regressed_std=%.2f",
        len(result),
        result["sprint_speed"].mean(),
        result["sprint_speed"].std(),
        result["sprint_speed_regressed"].mean(),
        result["sprint_speed_regressed"].std(),
    )

    return result


def run() -> None:
    """Compute and save per-batter sprint speed data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    df = compute_sprint_speeds()

    if df.empty:
        logger.warning("No sprint speed data to save")
        return

    out_path = DASHBOARD_DIR / "batter_sprint_speed.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(
        "Saved %d player-seasons to %s",
        len(df), out_path,
    )

    # Show distribution of regressed sprint speed for latest season
    latest = df[df["season"] == df["season"].max()]
    if not latest.empty:
        pcts = latest["sprint_speed_regressed"].quantile(
            [0.1, 0.25, 0.5, 0.75, 0.9]
        )
        logger.info("Latest season sprint speed distribution (regressed):")
        for q, v in pcts.items():
            logger.info("  P%d: %.2f ft/sec", int(q * 100), v)

        # Show BABIP adjustment distribution
        babip_adjs = (latest["sprint_speed_regressed"] - LEAGUE_SPRINT_SPEED) * 0.010
        logger.info("Implied BABIP adjustments (latest season):")
        for q, v in babip_adjs.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items():
            logger.info("  P%d: %+.4f", int(q * 100), v)


if __name__ == "__main__":
    run()
