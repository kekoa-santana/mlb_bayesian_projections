#!/usr/bin/env python
"""
Apply preseason injury adjustments to existing counting stat parquets.

This is a quick standalone script — no model fitting required.
It reads the existing counting projections, scales them by each player's
games-available fraction, and overwrites the parquets.

Usage
-----
    python scripts/apply_injury_adjustments.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    inj_path = DASHBOARD_DIR / "preseason_injuries.parquet"
    if not inj_path.exists():
        logger.error("No preseason injuries file found at %s", inj_path)
        logger.error("Run: python scripts/build_preseason_injuries.py")
        sys.exit(1)

    inj_df = pd.read_parquet(inj_path)
    inj_df = inj_df[inj_df["player_id"].notna() & (inj_df["est_missed_games"] > 0)]
    inj_frac = dict(zip(
        inj_df["player_id"].astype(int),
        ((162 - inj_df["est_missed_games"].clip(upper=162)) / 162).values,
    ))
    logger.info("Injury adjustments for %d players", len(inj_frac))

    # ── Hitter counting ──────────────────────────────────────────────
    h_path = DASHBOARD_DIR / "hitter_counting.parquet"
    if h_path.exists():
        hc = pd.read_parquet(h_path)
        scale_cols = [
            c for c in hc.columns
            if c.startswith("total_") or c in ("projected_pa_mean", "projected_games_mean")
        ]
        n_adjusted = 0
        for pid, frac in inj_frac.items():
            mask = hc["batter_id"] == pid
            if mask.any():
                hc.loc[mask, scale_cols] *= frac
                name = hc.loc[mask, "batter_name"].iloc[0]
                missed = int(round(162 * (1 - frac)))
                logger.info("  %s: %.0f%% season (~%dG missed)", name, frac * 100, missed)
                n_adjusted += 1
        hc.to_parquet(h_path, index=False)
        logger.info("Adjusted %d hitters, saved to %s", n_adjusted, h_path)
    else:
        logger.warning("No hitter counting file found at %s", h_path)

    # ── Pitcher counting ──────────────────────────────────────────────
    p_path = DASHBOARD_DIR / "pitcher_counting.parquet"
    if p_path.exists():
        pc = pd.read_parquet(p_path)
        scale_cols = [
            c for c in pc.columns
            if c.startswith("total_") or c in ("projected_bf_mean", "projected_games_mean")
        ]
        n_adjusted = 0
        for pid, frac in inj_frac.items():
            mask = pc["pitcher_id"] == pid
            if mask.any():
                pc.loc[mask, scale_cols] *= frac
                name = pc.loc[mask, "pitcher_name"].iloc[0]
                missed = int(round(162 * (1 - frac)))
                logger.info("  %s: %.0f%% season (~%dG missed)", name, frac * 100, missed)
                n_adjusted += 1
        pc.to_parquet(p_path, index=False)
        logger.info("Adjusted %d pitchers, saved to %s", n_adjusted, p_path)
    else:
        logger.warning("No pitcher counting file found at %s", p_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
