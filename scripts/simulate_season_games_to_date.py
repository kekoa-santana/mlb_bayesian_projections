#!/usr/bin/env python
"""
Simulate every *completed* regular-season game in a season through a cutoff date,
then compare simulated final scores (aggregated from Monte Carlo draws) to actuals.

Uses the same full-game engine as ``run_full_game_backtest.py`` (fit on prior
seasons, simulate with ``simulate_full_game_both_teams``).

Requires PostgreSQL (``production`` schema) and cached lineup parquet for the
target season (see ``get_cached_game_lineups``).

Usage
-----
    python scripts/simulate_season_games_to_date.py --season 2026
    python scripts/simulate_season_games_to_date.py --season 2026 --through 2026-04-07 --quick
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_eng import get_cached_game_lineups
from src.data.team_queries import get_game_results
from src.evaluation.full_game_sim_batch import fit_full_game_sim_bundle, simulate_full_games

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def load_completed_games_through(
    season: int,
    through: str,
) -> pd.DataFrame:
    """Regular-season games with final scores and both SP ids through *through*."""
    games = get_game_results(seasons=[season], include_postseason=False)
    if games.empty:
        return games

    games = games.dropna(subset=["home_runs", "away_runs", "home_sp_id", "away_sp_id"])
    cutoff = _parse_date(through)
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.normalize()
    games = games[games["game_date"] <= cutoff]

    games = games.rename(columns={"home_sp_id": "home_sp", "away_sp_id": "away_sp"})
    for c in ("home_runs", "away_runs", "home_sp", "away_sp"):
        games[c] = games[c].astype(int)
    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return games


def aggregate_final_scores(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable final lines and error vs actual (Monte Carlo means / medians).

    Convention: ``away-home`` (road score first, home second), matching MLB logs.
    """
    out = sim_df.copy()
    out["actual_final_score"] = (
        out["actual_away_runs"].astype(str) + "-" + out["actual_home_runs"].astype(str)
    )
    out["pred_final_mean_score"] = (
        np.round(out["pred_away_runs_mean"]).astype(int).astype(str)
        + "-"
        + np.round(out["pred_home_runs_mean"]).astype(int).astype(str)
    )
    if "pred_away_runs_p50" in out.columns and "pred_home_runs_p50" in out.columns:
        out["pred_final_median_score"] = (
            np.round(out["pred_away_runs_p50"]).astype(int).astype(str)
            + "-"
            + np.round(out["pred_home_runs_p50"]).astype(int).astype(str)
        )
    out["err_total_runs_mean"] = out["pred_total_runs_mean"] - out["actual_total_runs"]
    out["err_home_runs_mean"] = out["pred_home_runs_mean"] - out["actual_home_runs"]
    out["err_away_runs_mean"] = out["pred_away_runs_mean"] - out["actual_away_runs"]
    out["abs_err_total_mean"] = out["err_total_runs_mean"].abs()
    return out


def print_summary(df: pd.DataFrame) -> None:
    """Log aggregate calibration / error stats."""
    n = len(df)
    if n == 0:
        logger.warning("No games in output; nothing to summarize.")
        return
    mae_total = float(df["abs_err_total_mean"].mean())
    rmse_total = float(np.sqrt((df["err_total_runs_mean"] ** 2).mean()))
    home_win = df["actual_home_win"].values
    p_home = df["pred_home_win_prob"].values
    pred_winner_home = (p_home > 0.5).astype(int)
    acc = float((pred_winner_home == home_win).mean())
    logger.info("Games: %d", n)
    logger.info("Total runs — MAE: %.3f  RMSE: %.3f", mae_total, rmse_total)
    logger.info("Home win (threshold 0.5) accuracy: %.3f", acc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate completed season games through a date vs actual finals",
    )
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2026)")
    parser.add_argument(
        "--through",
        type=str,
        default=None,
        help="Last game date to include (YYYY-MM-DD). Default: today (UTC date).",
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=2020,
        help="First training season (inclusive). Train seasons are train_start .. season-1.",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer MCMC draws and sims")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for per-game output (default: outputs/season_sim_<season>_through_<date>.csv)",
    )
    args = parser.parse_args()

    season = args.season
    through = args.through or date.today().isoformat()
    train_seasons = list(range(args.train_start, season))
    if not train_seasons:
        raise SystemExit(f"No training seasons before {season}; use --train-start < {season}")

    if args.quick:
        draws, tune, chains, n_sims = 500, 250, 2, 2000
    else:
        draws, tune, chains, n_sims = 1000, 500, 2, 5000

    games = load_completed_games_through(season, through)
    logger.info(
        "Loaded %d completed games for season %d through %s",
        len(games), season, through,
    )
    if games.empty:
        raise SystemExit("No games matched filters (check DB and --through date).")

    logger.info("Fitting models on seasons %s ...", train_seasons)
    bundle = fit_full_game_sim_bundle(
        train_seasons,
        draws=draws, tune=tune, chains=chains,
        random_seed=42,
    )

    lineups = get_cached_game_lineups(season)
    preds = simulate_full_games(
        bundle,
        games,
        lineups,
        n_sims=n_sims,
        random_seed=42,
        include_run_quantiles=True,
    )

    out = aggregate_final_scores(preds)
    out["season"] = season
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = (
        Path(args.output)
        if args.output
        else out_dir / f"season_{season}_sim_through_{through}.csv"
    )
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), out_path)

    print_summary(out)


if __name__ == "__main__":
    main()
