#!/usr/bin/env python
"""
Precompute per-batter line drive rate (LD%) with Bayesian shrinkage.

Queries launch angle data from the database, computes LD% (launch angles
10-25 degrees), and regresses toward league average (~22%) based on
sample size.

Output: data/dashboard/batter_ld_rate.parquet
    Columns: player_id, season, ld_rate, ld_rate_regressed, n_bip
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
from src.data.db import read_sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# League average LD% (2022-2025 Statcast data)
LEAGUE_LD_RATE = 0.22

# Shrinkage constant: BIP needed for full weight on individual LD%.
# With 200 BIP, the individual rate gets ~57% weight.
# With 400 BIP, ~73%. With 100 BIP, ~40%.
SHRINKAGE_K = 150

# Seasons to compute LD% for
SEASONS = [2023, 2024, 2025]


def compute_ld_rates() -> pd.DataFrame:
    """Query database for launch angle data and compute per-batter LD%.

    Returns DataFrame with columns:
        player_id, season, ld_rate, ld_rate_regressed, n_bip
    """
    season_list = ", ".join(str(s) for s in SEASONS)

    query = f"""
    SELECT
        fpa.batter_id AS player_id,
        dg.season,
        COUNT(*) AS n_bip,
        SUM(CASE WHEN sbb.launch_angle >= 10 AND sbb.launch_angle <= 25
                 THEN 1 ELSE 0 END) AS n_ld
    FROM production.fact_pa fpa
    JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
    JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id
    WHERE dg.season IN ({season_list})
      AND dg.game_type = 'R'
      AND sbb.launch_angle IS NOT NULL
      AND sbb.launch_angle != 'NaN'
    GROUP BY fpa.batter_id, dg.season
    HAVING COUNT(*) >= 30
    ORDER BY dg.season, fpa.batter_id
    """

    logger.info("Querying LD%% data for seasons %s...", SEASONS)
    df = read_sql(query)
    logger.info("Raw LD data: %d player-seasons", len(df))

    if df.empty:
        logger.error("No launch angle data found in database")
        return pd.DataFrame(
            columns=["player_id", "season", "ld_rate", "ld_rate_regressed", "n_bip"]
        )

    # Compute raw LD%
    df["ld_rate"] = df["n_ld"] / df["n_bip"]

    # Bayesian shrinkage toward league average
    # regressed = (n_bip / (n_bip + K)) * observed + (K / (n_bip + K)) * league
    reliability = df["n_bip"] / (df["n_bip"] + SHRINKAGE_K)
    df["ld_rate_regressed"] = (
        reliability * df["ld_rate"]
        + (1 - reliability) * LEAGUE_LD_RATE
    )

    # Round for storage
    df["ld_rate"] = df["ld_rate"].round(4)
    df["ld_rate_regressed"] = df["ld_rate_regressed"].round(4)

    # Drop intermediate column
    df = df[["player_id", "season", "ld_rate", "ld_rate_regressed", "n_bip"]].copy()

    # Log summary stats
    logger.info(
        "LD%% summary: mean=%.3f, std=%.3f, regressed_mean=%.3f, regressed_std=%.3f",
        df["ld_rate"].mean(), df["ld_rate"].std(),
        df["ld_rate_regressed"].mean(), df["ld_rate_regressed"].std(),
    )
    logger.info("  BIP per player-season: mean=%.0f, median=%.0f", df["n_bip"].mean(), df["n_bip"].median())

    return df


def run() -> None:
    """Compute and save per-batter LD% data."""
    df = compute_ld_rates()

    if df.empty:
        logger.warning("No LD data to save")
        return

    out_path = DASHBOARD_DIR / "batter_ld_rate.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(
        "Saved %d player-seasons to %s",
        len(df), out_path,
    )

    # Show distribution of regressed LD% for latest season
    latest = df[df["season"] == df["season"].max()]
    if not latest.empty:
        pcts = latest["ld_rate_regressed"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        logger.info("Latest season LD%% distribution (regressed):")
        for q, v in pcts.items():
            logger.info("  P%d: %.3f", int(q * 100), v)


if __name__ == "__main__":
    run()
