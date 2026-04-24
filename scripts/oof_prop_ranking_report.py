#!/usr/bin/env python
"""Rank game-level props by walk-forward (OOF-style) backtest metrics.

Reads ``outputs/game_prop_backtest_summary.parquet`` produced by::

    python scripts/run_game_prop_backtest.py [--side ...] [--folds ...]

Ranks props by mean ``ece`` and ``avg_brier`` across folds (lower is better),
prints per-line Brier columns when present (``brier_0_5``, etc.), and suggests
a betting confidence band using the same logic as ``confidence_tiers.py``.

Usage
-----
python scripts/oof_prop_ranking_report.py
python scripts/oof_prop_ranking_report.py --summary path/to/summary.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SUMMARY = ROOT / "outputs" / "game_prop_backtest_summary.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description="OOF-style prop ranking from backtest summary")
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = p.parse_args()

    if not args.summary.exists():
        print(f"No summary at {args.summary}\n")
        print("Generate it with walk-forward folds, e.g.:")
        print("  python scripts/run_game_prop_backtest.py --side all")
        print("  python scripts/run_game_prop_backtest.py --side batter --folds 2025:2021-2024 2024:2020-2023")
        sys.exit(1)

    df = pd.read_parquet(args.summary)
    if "prop" not in df.columns:
        print("Expected column 'prop' in summary parquet")
        sys.exit(1)

    # Aggregate across folds: mean ECE / Brier / sharpness
    agg_cols = ["ece", "avg_brier", "avg_log_loss", "mce", "temperature",
                "sharpness_pct_actionable_60", "sharpness_pct_actionable_70",
                "n_games"]
    avail = [c for c in agg_cols if c in df.columns]
    g = df.groupby("prop", as_index=False)[avail].mean(numeric_only=True)

    # Per-line Brier: any column brier_X_Y
    brier_cols = [c for c in df.columns if c.startswith("brier_") and c != "brier_scores"]
    if brier_cols:
        bmean = df.groupby("prop", as_index=False)[brier_cols].mean(numeric_only=True)
        g = g.merge(bmean, on="prop", how="left")

    g = g.sort_values(["ece", "avg_brier"], ascending=[True, True], na_position="last")

    print("=" * 88)
    print("OOF / walk-forward prop ranking (mean across folds; lower ECE & Brier = better)")
    print("=" * 88)
    show = ["prop", "ece", "avg_brier", "avg_log_loss", "temperature",
            "sharpness_pct_actionable_60", "sharpness_pct_actionable_70", "n_games"]
    show = [c for c in show if c in g.columns]
    print(g[show].to_string(index=False))

    if brier_cols:
        print("\n" + "=" * 88)
        print("Per-line mean Brier (across folds)")
        print("=" * 88)
        line_show = ["prop"] + sorted(brier_cols)
        line_show = [c for c in line_show if c in g.columns]
        print(g[line_show].to_string(index=False))

    # Suggested tiers (mirror confidence_tiers.py thresholds)
    HIGH_ECE = 0.05
    LOW_ECE = 0.10

    print("\n" + "=" * 88)
    print("Suggested betting confidence (calibration-first; tune with your bankroll)")
    print("=" * 88)
    for _, row in g.iterrows():
        prop = row["prop"]
        ece = row.get("ece", np.nan)
        br = row.get("avg_brier", np.nan)
        if np.isnan(ece):
            tier = "UNKNOWN (no ECE)"
        elif ece < HIGH_ECE:
            tier = "HIGH - use model vs market with normal stake sizing if edge clears your rule"
        elif ece < LOW_ECE:
            tier = "MEDIUM - require larger edge or secondary confirmation"
        else:
            tier = "LOW - paper flags these; avoid or only tiny stake / alt lines"

        act70 = row.get("sharpness_pct_actionable_70", np.nan)
        act60 = row.get("sharpness_pct_actionable_60", np.nan)
        extra = ""
        if not np.isnan(act70):
            extra = f"  (pct predictions with |p-0.5|>=0.35: ~{act70:.1f}%)"
        print(f"  {prop:<18} ECE={ece:.4f}  avg_Brier={br:.4f}  -> {tier}{extra}")

    print("\nReference: published walk-forward (2023-2025) in docs/paper_v1_calibrated_game_props.md")
    print("  Tier 1 (ECE<0.04): batter H, batter K, batter BB, pitcher K")
    print("  Tier 2: pitcher BB, pitcher H")
    print("  Tier 3: pitcher HR, pitcher outs; batter HR excluded")


if __name__ == "__main__":
    main()
