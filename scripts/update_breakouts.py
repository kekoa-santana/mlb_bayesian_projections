"""Generate breakout candidate parquets and refresh rankings.

Run this after the main precompute has been run at least once
(needs projection parquets to exist for rankings).

Usage:
    python scripts/update_breakouts.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_breakouts")

DASHBOARD_DIR = Path(
    "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
)
FROM_SEASON = 2025


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Hitter breakout candidates ---
    logger.info("=" * 60)
    logger.info("Scoring hitter breakout archetypes (GMM k=2)...")
    from src.models.breakout_model import score_breakout_candidates

    h = score_breakout_candidates(season=FROM_SEASON, min_pa=200)
    h.to_parquet(DASHBOARD_DIR / "hitter_breakout_candidates.parquet", index=False)
    n_hc = (h["breakout_tier"] == "Breakout Candidate").sum()
    logger.info("Saved hitter_breakout_candidates.parquet: %d rows (%d candidates)", len(h), n_hc)

    # --- 2. Pitcher breakout candidates ---
    logger.info("=" * 60)
    logger.info("Scoring pitcher breakout archetypes (GMM k=3)...")
    from src.models.pitcher_breakout_model import score_pitcher_breakout_candidates

    p = score_pitcher_breakout_candidates(season=FROM_SEASON, min_bf=200)
    p.to_parquet(DASHBOARD_DIR / "pitcher_breakout_candidates.parquet", index=False)
    n_pc = (p["breakout_tier"] == "Breakout Candidate").sum()
    logger.info("Saved pitcher_breakout_candidates.parquet: %d rows (%d candidates)", len(p), n_pc)

    # --- 3. Refresh rankings (picks up breakout columns) ---
    logger.info("=" * 60)
    logger.info("Refreshing MLB positional rankings...")
    from src.models.player_rankings import rank_all

    rankings = rank_all(season=FROM_SEASON, projection_season=FROM_SEASON + 1)
    for key, rdf in rankings.items():
        if not rdf.empty:
            fname = f"{key}_rankings.parquet"
            rdf.to_parquet(DASHBOARD_DIR / fname, index=False)
            logger.info("Saved %s: %d rows", fname, len(rdf))

    logger.info("=" * 60)
    logger.info("Done. Dashboard parquets updated.")


if __name__ == "__main__":
    main()
