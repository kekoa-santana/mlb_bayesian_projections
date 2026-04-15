#!/usr/bin/env python
"""
Run the market comparison evaluator.

Two modes:
  1. ``--today``  — compare today's fresh sim predictions vs the latest
     Bovada snapshot for today. No actuals required.
  2. ``--archive`` — evaluate the accumulated archive against historical
     odds + final scores. Requires games to have completed and results
     to be back-filled into the archive.

Outputs a terminal report with per-game edges on all three surfaces
(moneyline, run line, total). When actuals are available, also prints
aggregate hit rate / ROI by edge bucket.

Usage
-----
    python scripts/run_market_comparison.py --today
    python scripts/run_market_comparison.py --archive --from 2026-04-10 --to 2026-05-01
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

from src.evaluation.market_comparison import (
    compute_bias_correction,
    compute_edges,
    compute_results,
    extract_closing_odds,
    summarize_performance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = PROJECT_ROOT.parent / "tdd-dashboard" / "data" / "dashboard"
ARCHIVE_PATH = DASHBOARD_DIR / "sim_predictions_archive.parquet"
ODDS_HISTORY_PATH = DASHBOARD_DIR / "game_odds_history.parquet"
TODAYS_PREDS_PATH = DASHBOARD_DIR / "todays_game_predictions.parquet"


def _print_game_report(edges_df: pd.DataFrame, min_edge: float = 0.03) -> None:
    """Print per-game edges across ML / RL / total."""
    if edges_df.empty:
        print("No games to report.")
        return

    edges_df = edges_df.sort_values("total_mean", ascending=False)

    print()
    print("=" * 118)
    print("PER-GAME EDGES (ML / RUN LINE / TOTAL)")
    print("=" * 118)

    for _, r in edges_df.iterrows():
        match = f"{r['away_abbr']} @ {r['home_abbr']}"
        score = f"{r['away_runs_mean']:.1f} - {r['home_runs_mean']:.1f}"

        # ML
        a_edge = r.get("ml_edge_away", float("nan"))
        h_edge = r.get("ml_edge_home", float("nan"))
        best_ml = max(a_edge, h_edge) if not (pd.isna(a_edge) or pd.isna(h_edge)) else float("nan")
        ml_pick = ""
        if not pd.isna(best_ml) and best_ml >= min_edge:
            side = r["away_abbr"] if a_edge >= h_edge else r["home_abbr"]
            ml_pick = f"  ML pick: {side} ({best_ml:+.1%})"

        # Run line
        rl_pick = ""
        if pd.notna(r.get("spread_home_line")):
            p_home_cov = r["p_home_cover_rl"]
            home_line = r["spread_home_line"]
            fav_side = r["home_abbr"] if home_line < 0 else r["away_abbr"]
            rl_pick = f"  RL: {r['home_abbr']}{home_line:+.1f} P({fav_side} cover)={p_home_cov if home_line<0 else 1-p_home_cov:.1%}"

        # Total
        tot_pick = ""
        if pd.notna(r.get("total_line")):
            p_over = r["p_over"]
            edge_over = r.get("total_edge_over", float("nan"))
            edge_under = r.get("total_edge_under", float("nan"))
            best_tot = max(edge_over, edge_under) if not (pd.isna(edge_over) or pd.isna(edge_under)) else float("nan")
            side = "OVER" if edge_over >= edge_under else "UNDER"
            label = f"  TOT pick: {side} ({best_tot:+.1%})" if not pd.isna(best_tot) and best_tot >= min_edge else ""
            tot_pick = (
                f"  Total model={r['total_mean']:.2f} vs book={r['total_line']:.1f} "
                f"P(over)={p_over:.1%}{label}"
            )

        print(f"\n{match}")
        print(f"  Score: {score}  margin={r['margin_mean']:+.2f}")
        if ml_pick:
            print(ml_pick)
        if rl_pick:
            print(rl_pick)
        if tot_pick:
            print(tot_pick)

    # Summary counts
    def _count_picks(df: pd.DataFrame, a_col: str, b_col: str) -> int:
        best = np.maximum(
            df[a_col].fillna(-np.inf), df[b_col].fillna(-np.inf),
        )
        return int((best >= min_edge).sum())

    print()
    print("-" * 118)
    print(
        f"Picks at ≥{min_edge:.0%} edge:  "
        f"ML={_count_picks(edges_df, 'ml_edge_away', 'ml_edge_home')}  "
        f"Total={_count_picks(edges_df, 'total_edge_over', 'total_edge_under')}  "
        f"(of {len(edges_df)} games)"
    )


def _print_performance(perf: dict[str, pd.DataFrame]) -> None:
    """Print aggregate hit rate / ROI by surface and edge bucket."""
    if not perf:
        print("\nNo completed games to evaluate.")
        return

    print()
    print("=" * 80)
    print("AGGREGATE PERFORMANCE (completed games)")
    print("=" * 80)
    for surface, df in perf.items():
        print(f"\n{surface.upper()} — assume flat $110 to win $100:")
        print(f"{'edge≥':>8s} {'n':>6s} {'W':>5s} {'L':>5s} {'Push':>5s} "
              f"{'Hit%':>8s} {'ROI':>8s}")
        for _, r in df.iterrows():
            print(
                f"{r['edge_min']:>+8.1%} {int(r['n_bets']):>6d} "
                f"{int(r['wins']):>5d} {int(r['losses']):>5d} "
                f"{int(r['pushes']):>5d} "
                f"{r['hit_rate']:>7.1%}  {r['roi']:>+7.1%}"
                if not np.isnan(r["hit_rate"]) else
                f"{r['edge_min']:>+8.1%} {int(r['n_bets']):>6d}  "
                f"{'--':>5s} {'--':>5s} {'--':>5s} {'--':>8s} {'--':>8s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sim predictions to Bovada lines",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--today", action="store_true",
        help="Compare today's fresh predictions to Bovada",
    )
    mode.add_argument(
        "--archive", action="store_true",
        help="Evaluate accumulated archive against historical odds",
    )
    parser.add_argument(
        "--from", dest="date_from", default=None,
        help="Start date for --archive mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to", dest="date_to", default=None,
        help="End date for --archive mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--source", default="bovada",
        help="Odds source to compare against (bovada/dk)",
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.03,
        help="Minimum edge threshold for picks (default 0.03 = 3%%)",
    )
    parser.add_argument(
        "--before", default=None,
        help="ISO UTC cutoff for odds snapshot. In --today mode, defaults "
             "to earliest game first pitch minus 30 min (guarantees "
             "pre-game lines).",
    )
    parser.add_argument(
        "--bias-correct", action="store_true",
        help="Learn and apply bias corrections from the archive",
    )
    args = parser.parse_args()

    # Load odds history once (used by both modes)
    if not ODDS_HISTORY_PATH.exists():
        logger.error("Missing %s", ODDS_HISTORY_PATH)
        sys.exit(1)
    odds_history = pd.read_parquet(ODDS_HISTORY_PATH)

    # Optional bias correction
    bias = {"total": 0.0, "margin": 0.0}
    if args.bias_correct and ARCHIVE_PATH.exists():
        archive = pd.read_parquet(ARCHIVE_PATH)
        bias = compute_bias_correction(archive)
        logger.info(
            "Bias correction from archive: total=%.3f, margin=%.3f",
            bias["total"], bias["margin"],
        )

    if args.today:
        if not TODAYS_PREDS_PATH.exists():
            logger.error("Missing %s", TODAYS_PREDS_PATH)
            sys.exit(1)

        preds = pd.read_parquet(TODAYS_PREDS_PATH)
        game_date = str(date.today())

        # Default --before to earliest game first pitch minus 30min so we
        # always read pre-game lines instead of in-game markets.
        before_cutoff = args.before
        if before_cutoff is None:
            todays_games_path = DASHBOARD_DIR / "todays_games.parquet"
            if todays_games_path.exists():
                games = pd.read_parquet(todays_games_path)
                if not games.empty and "game_datetime_utc" in games.columns:
                    earliest = pd.to_datetime(games["game_datetime_utc"]).min()
                    before_cutoff = (
                        earliest - pd.Timedelta(minutes=30)
                    ).isoformat()
                    logger.info(
                        "Auto cutoff: %s (earliest first pitch − 30min)",
                        before_cutoff,
                    )

        odds = extract_closing_odds(
            odds_history, game_date=game_date, source=args.source,
            before_time_utc=before_cutoff,
        )
        if odds.empty:
            logger.error(
                "No %s odds snapshot for %s — nothing to compare",
                args.source, game_date,
            )
            sys.exit(1)

        edges = compute_edges(
            preds, odds,
            total_bias_correction=bias["total"],
            margin_bias_correction=bias["margin"],
        )
        _print_game_report(edges, min_edge=args.min_edge)
        return

    # --archive mode
    if not ARCHIVE_PATH.exists():
        logger.error(
            "Archive parquet missing at %s — run the daily pipeline first",
            ARCHIVE_PATH,
        )
        sys.exit(1)

    archive = pd.read_parquet(ARCHIVE_PATH)
    if args.date_from:
        archive = archive[archive["game_date"] >= args.date_from]
    if args.date_to:
        archive = archive[archive["game_date"] <= args.date_to]

    logger.info("Evaluating %d archived games", len(archive))

    all_edges: list[pd.DataFrame] = []
    for game_date, preds_for_date in archive.groupby("game_date"):
        odds = extract_closing_odds(
            odds_history, game_date=str(game_date), source=args.source,
        )
        if odds.empty:
            continue
        edges = compute_edges(
            preds_for_date, odds,
            total_bias_correction=bias["total"],
            margin_bias_correction=bias["margin"],
        )
        all_edges.append(edges)

    if not all_edges:
        logger.error("No overlap between archive and odds history")
        sys.exit(1)

    all_edges_df = pd.concat(all_edges, ignore_index=True)

    # Load actuals (if archive has them backfilled)
    if "actual_away" in archive.columns and "actual_home" in archive.columns:
        actuals = archive[["game_pk", "actual_away", "actual_home"]].rename(
            columns={"actual_away": "away_score", "actual_home": "home_score"}
        )
        results_df = compute_results(all_edges_df, actuals)
        perf = summarize_performance(results_df, edge_buckets=(0.0, 0.03, 0.05, 0.10))
        _print_game_report(results_df, min_edge=args.min_edge)
        _print_performance(perf)
    else:
        _print_game_report(all_edges_df, min_edge=args.min_edge)
        print("\n[archive has no actuals yet — back-fill after games complete]")


if __name__ == "__main__":
    main()
