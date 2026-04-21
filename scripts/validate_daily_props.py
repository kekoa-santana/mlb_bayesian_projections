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

from src.data.paths import dashboard_dir
from src.evaluation.metrics import compute_ece, compute_temperature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()

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


def _compute_point_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Point-prediction quality: expected vs actual count."""
    mask = df["expected"].notna() & df["actual"].notna()
    if mask.sum() < 10:
        return {}
    exp = df.loc[mask, "expected"].astype(float).values
    act = df.loc[mask, "actual"].astype(float).values
    diff = exp - act
    return {
        "n_point": int(mask.sum()),
        "mean_exp": float(exp.mean()),
        "mean_act": float(act.mean()),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
    }


def _compute_pick_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Pick hit rate: blind-bet the model's favored side at each row's
    default line. Also breaks out by confidence quartile."""
    mask = (
        df["p_over"].notna()
        & df["actual"].notna()
        & df["line"].notna()
    )
    if mask.sum() < 20:
        return {}
    p = df.loc[mask, "p_over"].astype(float).values
    actual = df.loc[mask, "actual"].astype(float).values
    line = df.loc[mask, "line"].astype(float).values
    # Drop exact coin flips (no pick)
    nontrivial = p != 0.5
    p = p[nontrivial]
    actual = actual[nontrivial]
    line = line[nontrivial]
    if len(p) == 0:
        return {}
    pick_over = p > 0.5
    actual_over = actual > line
    win = pick_over == actual_over
    # Confidence quartiles on |p - 0.5|
    conf = np.abs(p - 0.5)
    edges = np.quantile(conf, [0.25, 0.5, 0.75])
    bucket = np.digitize(conf, edges)  # 0..3
    out = {
        "n_picks": int(len(p)),
        "pick_hit_rate": float(win.mean()),
    }
    for b in range(4):
        idx = bucket == b
        if idx.sum() >= 10:
            out[f"hit_q{b + 1}"] = float(win[idx].mean())
    return out


def _compute_per_line_metrics(
    df: pd.DataFrame, lines: list[float],
) -> list[dict]:
    """Per-line Brier / ECE / calibration breakdown."""
    rows: list[dict] = []
    for line in lines:
        col = f"p_over_{line:.1f}"
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["actual"].notna()
        if mask.sum() < 20:
            continue
        p = np.clip(df.loc[mask, col].astype(float).values, 0, 1)
        y = (df.loc[mask, "actual"].astype(float).values > line).astype(float)
        if y.std() == 0:
            continue
        rows.append({
            "line": line,
            "n": int(mask.sum()),
            "mean_pred": float(p.mean()),
            "mean_actual": float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "ece": compute_ece(p, y),
            "temp": compute_temperature(p, y),
        })
    return rows


def _compute_calibration_bins(
    df: pd.DataFrame, lines: list[float], n_bins: int = 10,
) -> list[dict]:
    """Pool predictions across all stored lines and bin into n_bins
    calibration buckets. Each row shows predicted-vs-empirical hit rate."""
    preds: list[float] = []
    hits: list[int] = []
    for line in lines:
        col = f"p_over_{line:.1f}"
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["actual"].notna()
        if mask.sum() == 0:
            continue
        p = np.clip(df.loc[mask, col].astype(float).values, 0, 1)
        y = (df.loc[mask, "actual"].astype(float).values > line).astype(int)
        preds.extend(p.tolist())
        hits.extend(y.tolist())
    if len(preds) < 50:
        return []
    preds_arr = np.asarray(preds)
    hits_arr = np.asarray(hits)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            idx = (preds_arr >= lo) & (preds_arr <= hi)
        else:
            idx = (preds_arr >= lo) & (preds_arr < hi)
        n = int(idx.sum())
        if n == 0:
            continue
        rows.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": n,
            "mean_pred": float(preds_arr[idx].mean()),
            "empirical": float(hits_arr[idx].mean()),
        })
    return rows


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
    parser.add_argument("--detail", action="store_true",
                        help="Show per-line breakdown and calibration bins")
    parser.add_argument("--props-path", type=str, default=None,
                        help="Override path to the props parquet file "
                             "(defaults to dashboard game_props.parquet)")
    args = parser.parse_args()

    props_path = (
        Path(args.props_path) if args.props_path
        else DASHBOARD_DIR / "game_props.parquet"
    )
    if not props_path.exists():
        logger.error("props parquet not found at %s", props_path)
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
    point_rows: list[dict] = []
    pick_rows: list[dict] = []

    for (ptype, stat), group in df.groupby(["player_type", "stat"]):
        line_map = _BATTER_LINES if ptype == "batter" else _STAT_LINES
        lines = line_map.get(stat, [0.5, 1.5])

        m = _compute_prop_metrics(group, lines)
        if m:
            summary_rows.append({"side": ptype, "stat": stat, **m})

        pm = _compute_point_metrics(group)
        if pm:
            point_rows.append({"side": ptype, "stat": stat, **pm})

        km = _compute_pick_metrics(group)
        if km:
            pick_rows.append({"side": ptype, "stat": stat, **km})

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        print("\n" + "=" * 75)
        print("PROBABILITY CALIBRATION (Brier / ECE / Temperature)")
        print("=" * 75)
        print("  Brier  -- lower is better (0.25 = coin flip)")
        print("  ECE    -- expected calibration error, lower is better")
        print("  Temp   -- >1 overconfident, <1 underconfident, 1 = calibrated")
        display = ["side", "stat", "n",
                    "avg_brier", "avg_ece", "avg_temp"]
        avail = [c for c in display if c in summary.columns]
        print(summary[avail].to_string(index=False))
    else:
        logger.warning("No groups had enough data for metrics")

    if point_rows:
        point_df = pd.DataFrame(point_rows)
        print("\n" + "=" * 75)
        print("POINT PREDICTION (Expected vs Actual)")
        print("=" * 75)
        print("  bias   -- mean(expected - actual); positive = model runs hot")
        print("  mae    -- mean absolute error")
        print("  rmse   -- root mean squared error")
        display = ["side", "stat", "n_point",
                    "mean_exp", "mean_act", "bias", "mae", "rmse"]
        avail = [c for c in display if c in point_df.columns]
        print(point_df[avail].round(3).to_string(index=False))

    if pick_rows:
        pick_df = pd.DataFrame(pick_rows)
        print("\n" + "=" * 75)
        print("PICK HIT RATE (blind-bet the model's favored side)")
        print("=" * 75)
        print("  pick_hit_rate -- fraction of rows where favored side won")
        print("  hit_q1..q4    -- hit rate by confidence quartile (q4 = most confident)")
        display = ["side", "stat", "n_picks",
                    "pick_hit_rate", "hit_q1", "hit_q2", "hit_q3", "hit_q4"]
        avail = [c for c in display if c in pick_df.columns]
        print(pick_df[avail].round(3).to_string(index=False))

    if args.detail:
        print("\n" + "=" * 75)
        print("PER-LINE BREAKDOWN")
        print("=" * 75)
        for (ptype, stat), group in df.groupby(["player_type", "stat"]):
            line_map = _BATTER_LINES if ptype == "batter" else _STAT_LINES
            lines = line_map.get(stat, [0.5, 1.5])
            rows = _compute_per_line_metrics(group, lines)
            if not rows:
                continue
            print(f"\n{ptype} {stat}:")
            print(pd.DataFrame(rows).round(4).to_string(index=False))

        print("\n" + "=" * 75)
        print("CALIBRATION BINS (predicted vs empirical hit rate)")
        print("=" * 75)
        for (ptype, stat), group in df.groupby(["player_type", "stat"]):
            line_map = _BATTER_LINES if ptype == "batter" else _STAT_LINES
            lines = line_map.get(stat, [0.5, 1.5])
            rows = _compute_calibration_bins(group, lines)
            if not rows:
                continue
            print(f"\n{ptype} {stat}:")
            print(pd.DataFrame(rows).round(3).to_string(index=False))

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
