#!/usr/bin/env python
"""
Walk-forward backtest for the full-game (both-teams) simulator.

Evaluates win probabilities, run totals, and run lines against
actual game outcomes using simulate_full_game_both_teams().

Usage
-----
    python scripts/run_full_game_backtest.py          # full quality
    python scripts/run_full_game_backtest.py --quick   # fast iteration
    python scripts/run_full_game_backtest.py --single-fold 2025
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.data.feature_eng import get_cached_game_lineups
from src.evaluation.full_game_sim_batch import fit_full_game_sim_bundle, simulate_full_games

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _get_game_scores(season: int) -> pd.DataFrame:
    """Get home/away scores for all regular-season games."""
    return read_sql(f"""
        WITH team_scores AS (
            SELECT fpg.game_pk, fpg.team_id,
                   SUM(COALESCE(fpg.bat_r, 0)) AS team_runs
            FROM production.fact_player_game_mlb fpg
            JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
            WHERE fpg.season = {int(season)} AND dg.game_type = 'R'
            GROUP BY fpg.game_pk, fpg.team_id
        )
        SELECT dg.game_pk, dg.game_date,
               dg.home_team_id, dg.away_team_id,
               COALESCE(h.team_runs, 0) AS home_runs,
               COALESCE(a.team_runs, 0) AS away_runs
        FROM production.dim_game dg
        LEFT JOIN team_scores h
            ON dg.game_pk = h.game_pk AND dg.home_team_id = h.team_id
        LEFT JOIN team_scores a
            ON dg.game_pk = a.game_pk AND dg.away_team_id = a.team_id
        WHERE dg.season = {int(season)} AND dg.game_type = 'R'
    """)


def _get_game_starters(season: int) -> pd.DataFrame:
    """Get starting pitchers per game (home + away)."""
    return read_sql(f"""
        SELECT fpg.game_pk, fpg.player_id AS pitcher_id,
               fpg.team_id,
               dg.home_team_id, dg.away_team_id
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = TRUE
          AND fpg.season = {int(season)}
          AND dg.game_type = 'R'
          AND fpg.pit_bf >= 3
    """)


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def run_one_fold(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run full-game backtest for one fold.

    Returns
    -------
    pd.DataFrame
        One row per game with predicted and actual outcomes.
    """
    logger.info("Full-game backtest: train=%s, test=%d", train_seasons, test_season)

    bundle = fit_full_game_sim_bundle(
        train_seasons,
        draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    logger.info("Loading test season %d...", test_season)
    game_scores = _get_game_scores(test_season)
    game_starters = _get_game_starters(test_season)
    game_lineups = get_cached_game_lineups(test_season)

    starter_map: dict[int, dict[str, int]] = {}
    for _, row in game_starters.iterrows():
        gpk = int(row["game_pk"])
        pid = int(row["pitcher_id"])
        tid = int(row["team_id"])
        if gpk not in starter_map:
            starter_map[gpk] = {}
        if tid == int(row["home_team_id"]):
            starter_map[gpk]["home"] = pid
        elif tid == int(row["away_team_id"]):
            starter_map[gpk]["away"] = pid

    logger.info("Games with scores: %d, with starter rows: %d",
                len(game_scores), len(starter_map))

    gs = game_scores.copy()
    gs["home_sp"] = gs["game_pk"].map(lambda g: starter_map.get(int(g), {}).get("home"))
    gs["away_sp"] = gs["game_pk"].map(lambda g: starter_map.get(int(g), {}).get("away"))
    gs = gs.dropna(subset=["home_sp", "away_sp"])
    gs["home_sp"] = gs["home_sp"].astype(int)
    gs["away_sp"] = gs["away_sp"].astype(int)

    preds = simulate_full_games(
        bundle, gs, game_lineups,
        n_sims=n_sims, random_seed=random_seed,
        include_run_quantiles=False,
    )
    preds["test_season"] = test_season
    preds = preds.rename(columns={
        "pred_home_runs_mean": "pred_home_runs",
        "pred_away_runs_mean": "pred_away_runs",
        "pred_total_runs_mean": "pred_total_runs",
        "pred_home_margin_mean": "pred_home_margin",
    })
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute full-game validation metrics."""
    n = len(df)
    if n == 0:
        return {}

    metrics: dict = {"n_games": n}

    # --- Win probability ---
    home_win = df["actual_home_win"].values
    home_prob = df["pred_home_win_prob"].values

    # Brier score
    metrics["win_brier"] = float(brier_score_loss(home_win, home_prob))

    # Log loss
    eps = 1e-6
    clipped = np.clip(home_prob, eps, 1 - eps)
    ll = -(home_win * np.log(clipped) + (1 - home_win) * np.log(1 - clipped))
    metrics["win_log_loss"] = float(np.mean(ll))

    # Accuracy at 50% threshold
    pred_home = (home_prob > 0.5).astype(int)
    metrics["win_accuracy"] = float(np.mean(pred_home == home_win))

    # Calibration by bin
    bins = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]
    for i in range(len(bins) - 1):
        mask = (home_prob >= bins[i]) & (home_prob < bins[i + 1])
        if mask.sum() >= 20:
            metrics[f"win_cal_{bins[i]:.2f}_{bins[i+1]:.2f}"] = float(
                home_win[mask].mean()
            )

    # --- Run totals ---
    total_err = df["pred_total_runs"] - df["actual_total_runs"]
    metrics["total_runs_bias"] = float(total_err.mean())
    metrics["total_runs_corr"] = float(
        df["pred_total_runs"].corr(df["actual_total_runs"])
    )

    # Over/under calibration
    for line, col in [(7.5, "p_over_7_5"), (8.5, "p_over_8_5"), (9.5, "p_over_9_5")]:
        actual_over = (df["actual_total_runs"] > line).astype(int)
        pred_over = df[col]
        metrics[f"ou_{line}_pred_rate"] = float(pred_over.mean())
        metrics[f"ou_{line}_actual_rate"] = float(actual_over.mean())
        metrics[f"ou_{line}_brier"] = float(brier_score_loss(actual_over, pred_over))

    # --- Margin / run line ---
    margin_err = df["pred_home_margin"] - (df["actual_home_runs"] - df["actual_away_runs"])
    metrics["margin_corr"] = float(
        df["pred_home_margin"].corr(df["actual_home_runs"] - df["actual_away_runs"])
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-game walk-forward backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws and sims")
    parser.add_argument("--single-fold", type=int, default=None,
                        help="Run one test season only")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_sims = 500, 250, 2, 2000
    else:
        draws, tune, chains, n_sims = 1000, 500, 2, 5000

    if args.single_fold:
        test_season = args.single_fold
        folds = [(list(range(2020, test_season)), test_season)]
    else:
        folds = [
            (list(range(2020, 2023)), 2023),
            (list(range(2020, 2024)), 2024),
            (list(range(2020, 2025)), 2025),
        ]

    logger.info("=" * 70)
    logger.info("FULL-GAME BACKTEST")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC sims per game: %d", n_sims)
    logger.info("  Folds: %d", len(folds))
    logger.info("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    all_preds = []
    all_metrics = []

    for train_seasons, test_season in folds:
        preds = run_one_fold(
            train_seasons, test_season,
            draws=draws, tune=tune, chains=chains,
            n_sims=n_sims, random_seed=42,
        )
        all_preds.append(preds)

        m = compute_metrics(preds)
        m["test_season"] = test_season
        all_metrics.append(m)

    # Save
    predictions = pd.concat(all_preds, ignore_index=True)
    predictions.to_csv(out_dir / "full_game_backtest_predictions.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "full_game_backtest_summary.csv", index=False)

    # Print results
    _print_results(metrics_df)


def _print_results(metrics: pd.DataFrame) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("FULL-GAME BACKTEST RESULTS")
    print("=" * 70)

    for _, row in metrics.iterrows():
        s = int(row["test_season"])
        n = int(row["n_games"])
        print(f"\nTest Season {s} ({n} games):")
        print(f"  Win Prob:  Brier={row['win_brier']:.4f}  "
              f"LogLoss={row['win_log_loss']:.4f}  "
              f"Accuracy={row['win_accuracy']:.3f}")
        print(f"  Runs:      Bias={row['total_runs_bias']:+.3f}  "
              f"Corr={row['total_runs_corr']:.3f}")
        print(f"  Margin:    Corr={row['margin_corr']:.3f}")

        for line in [7.5, 8.5, 9.5]:
            pred = row.get(f"ou_{line}_pred_rate", float("nan"))
            actual = row.get(f"ou_{line}_actual_rate", float("nan"))
            brier = row.get(f"ou_{line}_brier", float("nan"))
            print(f"  O/U {line}:   pred={pred:.3f}  actual={actual:.3f}  "
                  f"Brier={brier:.4f}")

        # Calibration bins
        cal_cols = [c for c in row.index if c.startswith("win_cal_")]
        if cal_cols:
            print("  Win calibration:")
            for c in sorted(cal_cols):
                lo, hi = c.replace("win_cal_", "").split("_")
                print(f"    P({lo}-{hi}): actual_win_rate={row[c]:.3f}")

    # Averages
    print("\n--- Averages ---")
    print(f"  Win Brier:    {metrics['win_brier'].mean():.4f}")
    print(f"  Win Accuracy: {metrics['win_accuracy'].mean():.3f}")
    print(f"  Margin Corr:  {metrics['margin_corr'].mean():.3f}")

    # Baselines
    print("\n--- Baselines ---")
    print("  Coin flip Brier: 0.2500")
    print("  Home-field-only accuracy: ~0.540")
    print()


if __name__ == "__main__":
    main()
