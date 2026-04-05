#!/usr/bin/env python
"""Day-forward validation of game prop predictions.

Reads the accumulated game_props.parquet from the dashboard repo
(which already contains predictions + backfilled actuals) and
computes Brier / ECE / calibration without re-running the full backtest.

Usage
-----
python scripts/validate_daily_props.py                    # all completed props
python scripts/validate_daily_props.py --days 7            # last 7 days only
python scripts/validate_daily_props.py --side pitcher       # pitcher only
python scripts/validate_daily_props.py --stat K BB          # specific stats
python scripts/validate_daily_props.py --stratify           # stratified by context
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_ece, compute_temperature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(
    r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard"
)

# game_props.parquet uses generic "expected" / "actual" / "p_over" columns
# with a "stat" column to distinguish K, BB, HR, H, Outs, TB, etc.
# and "player_type" for pitcher vs batter.

# Standard lines per stat for Brier / ECE evaluation.
_STAT_LINES: dict[str, list[float]] = {
    "K":    [3.5, 4.5, 5.5, 6.5, 7.5],   # pitcher K
    "BB":   [0.5, 1.5, 2.5, 3.5],
    "HR":   [0.5, 1.5],
    "H":    [3.5, 4.5, 5.5, 6.5, 7.5],
    "Outs": [14.5, 15.5, 16.5, 17.5, 18.5],
}
_BATTER_LINES: dict[str, list[float]] = {
    "K":  [0.5, 1.5, 2.5],
    "BB": [0.5, 1.5],
    "HR": [0.5, 1.5],
    "H":  [0.5, 1.5, 2.5],
    "TB": [0.5, 1.5, 2.5],
}


def _compute_prop_metrics(
    df: pd.DataFrame,
    lines: list[float],
) -> dict[str, float]:
    """Compute Brier, ECE, temperature for a single prop group."""
    n = len(df)
    if n < 20:
        return {}

    brier_vals: list[float] = []
    ece_vals: list[float] = []
    temp_vals: list[float] = []

    for line in lines:
        col = f"p_over_{line:.1f}"
        if col not in df.columns:
            continue

        y_prob = df[col].dropna().values
        y_true_full = df.loc[df[col].notna(), "actual"]
        y_true = (y_true_full > line).astype(float).values

        if len(y_prob) < 20 or y_true.std() == 0:
            continue

        y_prob = np.clip(y_prob, 0, 1)
        brier_vals.append(float(brier_score_loss(y_true, y_prob)))
        ece_vals.append(compute_ece(y_prob, y_true))
        temp_vals.append(compute_temperature(y_prob, y_true))

    return {
        "n": n,
        "avg_brier": float(np.mean(brier_vals)) if brier_vals else np.nan,
        "avg_ece": float(np.mean(ece_vals)) if ece_vals else np.nan,
        "avg_temp": float(np.mean(temp_vals)) if temp_vals else np.nan,
    }


def _stratified_daily(
    df: pd.DataFrame,
    lines: list[float],
    strata: list[str],
    stat: str,
    side: str,
    n_bins: int = 3,
    min_group: int = 30,
) -> list[dict]:
    """Compute metrics per stratum bin for a single (stat, side) group."""
    labels = ["low", "mid", "high"][:n_bins]
    rows: list[dict] = []

    for col in strata:
        if col not in df.columns:
            continue
        series = df[col]
        if series.isna().all():
            continue

        if series.dtype == object or series.nunique() <= n_bins:
            bin_col = series.copy()
        else:
            valid = series.dropna()
            if len(valid) < n_bins * min_group:
                continue
            try:
                bin_col = pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")
            except ValueError:
                continue

        for bin_label, idx in df.groupby(bin_col).groups.items():
            group = df.loc[idx]
            if len(group) < min_group:
                continue
            m = _compute_prop_metrics(group, lines)
            if not m:
                continue
            raw = series.loc[idx].dropna()
            rows.append({
                "stat": stat,
                "side": side,
                "stratum": col,
                "bin": str(bin_label),
                "bin_n": m["n"],
                "bin_min": float(raw.min()) if len(raw) else np.nan,
                "bin_max": float(raw.max()) if len(raw) else np.nan,
                **m,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-forward prop validation")
    parser.add_argument("--days", type=int, default=None,
                        help="Only include last N days of completed games")
    parser.add_argument("--side", choices=["pitcher", "batter", "all"], default="all")
    parser.add_argument("--stat", nargs="+", default=None,
                        help="Stats to validate (K, BB, HR, H, Outs, TB)")
    parser.add_argument("--stratify", action="store_true",
                        help="Show stratified metrics by context columns")
    args = parser.parse_args()

    props_path = DASHBOARD_DIR / "game_props.parquet"
    if not props_path.exists():
        logger.error("game_props.parquet not found at %s", props_path)
        sys.exit(1)

    df = pd.read_parquet(props_path)
    logger.info("Loaded %d rows from game_props.parquet", len(df))

    # Filter to completed games with actuals
    if "actual" not in df.columns:
        logger.error("No 'actual' column — run confident_picks backfill first")
        sys.exit(1)

    df = df[df["actual"].notna()].copy()
    logger.info("%d rows with actuals", len(df))

    if df.empty:
        logger.warning("No completed predictions to validate")
        sys.exit(0)

    # Date filter
    if args.days is not None and "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        cutoff = df["game_date"].max() - pd.Timedelta(days=args.days)
        df = df[df["game_date"] >= cutoff]
        logger.info("Filtered to last %d days: %d rows", args.days, len(df))

    # Side filter
    if args.side != "all" and "player_type" in df.columns:
        df = df[df["player_type"] == args.side]

    # Stat filter
    if args.stat is not None and "stat" in df.columns:
        df = df[df["stat"].isin(args.stat)]

    if df.empty:
        logger.warning("No rows after filters")
        sys.exit(0)

    # Compute metrics per (player_type, stat)
    summary_rows: list[dict] = []

    for (ptype, stat), group in df.groupby(["player_type", "stat"]):
        line_map = _BATTER_LINES if ptype == "batter" else _STAT_LINES
        lines = line_map.get(stat, [0.5, 1.5])

        m = _compute_prop_metrics(group, lines)
        if not m:
            continue

        summary_rows.append({
            "side": ptype,
            "stat": stat,
            **m,
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        print("\n" + "=" * 75)
        print("DAY-FORWARD PROP VALIDATION")
        print("=" * 75)
        display = ["side", "stat", "n",
                    "avg_brier", "avg_ece", "avg_temp"]
        avail = [c for c in display if c in summary.columns]
        print(summary[avail].to_string(index=False))
    else:
        logger.warning("No groups had enough data for metrics")

    # Stratified metrics
    if args.stratify:
        strata = ["umpire_k_lift", "umpire_bb_lift", "weather_k_lift"]
        strat_rows: list[dict] = []

        for (ptype, stat), group in df.groupby(["player_type", "stat"]):
            line_map = _BATTER_LINES if ptype == "batter" else _STAT_LINES
            lines = line_map.get(stat, [0.5, 1.5])
            strat_rows.extend(
                _stratified_daily(group, lines, strata, stat, ptype)
            )

        if strat_rows:
            strat_df = pd.DataFrame(strat_rows)
            print("\n" + "=" * 75)
            print("STRATIFIED METRICS")
            print("=" * 75)
            strat_cols = ["side", "stat", "stratum", "bin", "bin_n",
                          "avg_brier", "avg_ece", "avg_temp"]
            avail = [c for c in strat_cols if c in strat_df.columns]
            print(strat_df[avail].to_string(index=False))
        else:
            logger.info("No stratum columns with enough data for stratification")


if __name__ == "__main__":
    main()
