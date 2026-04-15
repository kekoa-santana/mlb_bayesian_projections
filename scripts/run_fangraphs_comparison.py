"""
Compare TDD player rankings against FanGraphs Depth Charts projected fWAR.

Usage:
    python scripts/run_fangraphs_comparison.py
    python scripts/run_fangraphs_comparison.py --hitters data/fangraphs/batters.csv
    python scripts/run_fangraphs_comparison.py --pitchers data/fangraphs/pitchers.csv
    python scripts/run_fangraphs_comparison.py --top-n 100

Prerequisites:
    1. Download Depth Charts CSV exports from FanGraphs:
       - Batters:  https://www.fangraphs.com/projections?type=depthcharts&stats=bat&pos=all
       - Pitchers: https://www.fangraphs.com/projections?type=depthcharts&stats=pit&pos=all
    2. Place in data/fangraphs/ as:
       - data/fangraphs/depthcharts_batters.csv
       - data/fangraphs/depthcharts_pitchers.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.fangraphs import load_hitter_projections, load_pitcher_projections
from src.evaluation.fangraphs_comparison import (
    compare_hitter_rankings,
    compare_pitcher_rankings,
    format_comparison_report,
    save_comparison_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")


def _load_tdd_rankings() -> dict[str, pd.DataFrame]:
    """Load latest TDD rankings from dashboard parquets."""
    results = {}

    hitter_path = DASHBOARD_DIR / "hitters_rankings.parquet"
    if hitter_path.exists():
        results["hitters"] = pd.read_parquet(hitter_path)
        logger.info("Loaded %d hitter rankings from %s", len(results["hitters"]), hitter_path)
    else:
        logger.warning("Hitter rankings not found at %s", hitter_path)

    pitcher_path = DASHBOARD_DIR / "pitchers_rankings.parquet"
    if pitcher_path.exists():
        results["pitchers"] = pd.read_parquet(pitcher_path)
        logger.info("Loaded %d pitcher rankings from %s", len(results["pitchers"]), pitcher_path)
    else:
        logger.warning("Pitcher rankings not found at %s", pitcher_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TDD rankings vs FanGraphs Depth Charts fWAR"
    )
    parser.add_argument(
        "--hitters", type=str, default=None,
        help="Path to FanGraphs hitter projections CSV",
    )
    parser.add_argument(
        "--pitchers", type=str, default=None,
        help="Path to FanGraphs pitcher projections CSV",
    )
    parser.add_argument(
        "--top-n", type=int, default=150,
        help="Compare top N players from each system (default: 150)",
    )
    parser.add_argument(
        "--save-csv", action="store_true", default=True,
        help="Save merged comparison to outputs/ (default: True)",
    )
    args = parser.parse_args()

    # Load TDD rankings
    tdd = _load_tdd_rankings()
    if not tdd:
        logger.error("No TDD rankings found. Run precompute first.")
        sys.exit(1)

    # Hitter comparison
    if "hitters" in tdd:
        try:
            fg_hitters = load_hitter_projections(args.hitters)
            result = compare_hitter_rankings(tdd["hitters"], fg_hitters, top_n=args.top_n)
            print(format_comparison_report(result))
            if args.save_csv:
                save_comparison_csv(result)
        except FileNotFoundError as e:
            logger.warning("Skipping hitters: %s", e)

    # Pitcher comparison
    if "pitchers" in tdd:
        try:
            fg_pitchers = load_pitcher_projections(args.pitchers)
            result = compare_pitcher_rankings(tdd["pitchers"], fg_pitchers, top_n=args.top_n)
            print(format_comparison_report(result))
            if args.save_csv:
                save_comparison_csv(result)
        except FileNotFoundError as e:
            logger.warning("Skipping pitchers: %s", e)


if __name__ == "__main__":
    main()
