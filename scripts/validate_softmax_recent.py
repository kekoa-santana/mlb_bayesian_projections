"""Run the new (softmax + bullpen tiers + BIP) game sim on recent 2026 games.

Produces predictions compatible with the dashboard Props Lab validator so
hit rates can be compared against the current (old-sim) system. Output
goes to ``data/dashboard/softmax_predictions.parquet`` in the tdd-dashboard
data dir.

Usage
-----
    python scripts/validate_softmax_recent.py --days 13
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.data.feature_eng import CACHE_DIR
from src.evaluation.game_sim_validation import build_game_sim_predictions
from src.evaluation.batter_sim_validation import build_batter_sim_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _invalidate_stale_game_caches(season: int, min_game_date: date) -> None:
    """Delete pitcher_game_logs / game_lineups caches whose newest row is
    older than *min_game_date*. Next access rebuilds from the DB.

    Needed because `_load_or_build` has no mtime/freshness check; a cache
    built before today's ETL will silently omit yesterday's games from any
    backtest that extends to the present.
    """
    stale_caches = ["pitcher_game_logs", "game_lineups"]
    for name in stale_caches:
        path = CACHE_DIR / f"{name}_{season}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["game_date"])
            max_date = pd.to_datetime(df["game_date"]).max().date()
        except Exception as e:
            logger.warning("Cache freshness check failed for %s (%s); deleting", path.name, e)
            path.unlink()
            continue
        if max_date < min_game_date:
            logger.info(
                "Cache %s max game_date %s < required %s; rebuilding",
                path.name, max_date, min_game_date,
            )
            path.unlink()
        else:
            logger.info("Cache %s is fresh (max game_date %s)", path.name, max_date)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=13,
                        help="Number of recent days to include (default: 13)")
    parser.add_argument("--n-sims", type=int, default=5000,
                        help="Monte Carlo sims per game (default: 5000)")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tdd-dashboard" / "data" / "dashboard",
    )
    args = parser.parse_args()

    train_seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    logger.info(
        "Running softmax sim for test_season=%d, last %d days, n_sims=%d",
        args.test_season, args.days, args.n_sims,
    )

    # Bust stale game-log / lineup caches if they don't cover yesterday.
    # Prevents silent gaps when the cache was written before today's ETL.
    _invalidate_stale_game_caches(
        season=args.test_season,
        min_game_date=date.today() - timedelta(days=1),
    )

    # --- Pitcher side ---
    pitcher_preds = build_game_sim_predictions(
        train_seasons=train_seasons,
        test_season=args.test_season,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        n_sims=args.n_sims,
    )
    logger.info("Pitcher predictions: %d rows", len(pitcher_preds))

    # --- Batter side ---
    batter_preds = build_batter_sim_predictions(
        train_seasons=train_seasons,
        test_season=args.test_season,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        n_sims=args.n_sims,
    )
    logger.info("Batter predictions: %d rows", len(batter_preds))

    if pitcher_preds.empty and batter_preds.empty:
        logger.error("No predictions returned")
        return

    # Collect all game_pks and look up dates once
    all_pks = set()
    if not pitcher_preds.empty:
        all_pks.update(pitcher_preds["game_pk"].unique().tolist())
    if not batter_preds.empty:
        all_pks.update(batter_preds["game_pk"].unique().tolist())
    dates_df = read_sql(
        """
        SELECT game_pk, game_date::text AS game_date
        FROM production.dim_game
        WHERE season = :season AND game_pk = ANY(:pks)
        """,
        {"season": args.test_season, "pks": list(all_pks)},
    )

    today = date.today()
    cutoff = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")

    def _attach_and_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.merge(dates_df, on="game_pk", how="left")
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
        df = df[df["game_date"] >= cutoff].copy()
        logger.info(
            "%s after %d-day filter (>= %s): %d rows across %d games",
            label, args.days, cutoff, len(df), df["game_pk"].nunique(),
        )
        return df

    pitcher_preds = _attach_and_filter(pitcher_preds, "Pitcher")
    batter_preds = _attach_and_filter(batter_preds, "Batter")

    args.dashboard_dir.mkdir(parents=True, exist_ok=True)
    p_path = args.dashboard_dir / "softmax_pitcher_predictions.parquet"
    b_path = args.dashboard_dir / "softmax_batter_predictions.parquet"
    if not pitcher_preds.empty:
        pitcher_preds.to_parquet(p_path, index=False)
        logger.info("Saved pitcher predictions (%d) → %s", len(pitcher_preds), p_path)
    if not batter_preds.empty:
        batter_preds.to_parquet(b_path, index=False)
        logger.info("Saved batter predictions (%d) → %s", len(batter_preds), b_path)


if __name__ == "__main__":
    main()
