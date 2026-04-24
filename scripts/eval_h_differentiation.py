"""Evaluate batter H model differentiation: 0 hits vs 1 hit vs 2+ hits.

Compares the model's ability to distinguish between batters who will
get 0, 1, or 2+ hits using the current game_props predictions vs actuals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.paths import dashboard_dir

DASHBOARD_DIR = dashboard_dir()


def main() -> None:
    gp = pd.read_parquet(DASHBOARD_DIR / "game_props.parquet")
    bh = gp[(gp["player_type"] == "batter") & (gp["stat"] == "H")].copy()
    bh = bh.dropna(subset=["actual"])

    print(f"Batter H rows with actuals: {len(bh)}")
    print(f"Date range: {bh['game_date'].min()} to {bh['game_date'].max()}")
    print()

    actual = bh["actual"].values
    expected = bh["expected"].values

    # Actual distribution
    print("=" * 70)
    print("ACTUAL HIT DISTRIBUTION")
    print("=" * 70)
    for h_val in range(6):
        n = (actual == h_val).sum()
        pct = n / len(actual) * 100
        bar = "#" * int(pct)
        print(f"  {h_val} hits: {n:>5d} ({pct:>5.1f}%) {bar}")

    zero_rate = (actual == 0).mean()
    one_rate = (actual == 1).mean()
    multi_rate = (actual >= 2).mean()
    print(f"\n  0 hits: {zero_rate:.3f}  |  1 hit: {one_rate:.3f}  |  2+ hits: {multi_rate:.3f}")

    # Model predictions
    p05 = bh["p_over_0.5"].values if "p_over_0.5" in bh.columns else None
    p15 = bh["p_over_1.5"].values if "p_over_1.5" in bh.columns else None
    p25 = bh["p_over_2.5"].values if "p_over_2.5" in bh.columns else None

    if p05 is None or p15 is None:
        print("Missing p_over columns")
        return

    # Derived probabilities
    p_zero = 1 - p05        # P(H = 0)
    p_one = p05 - p15       # P(H = 1)
    p_multi = p15            # P(H >= 2)

    print()
    print("=" * 70)
    print("MODEL vs ACTUAL: PREDICTED PROBABILITY BY OUTCOME")
    print("=" * 70)
    print(f"{'Category':<15} {'Pred':>8} {'Actual':>8} {'Bias':>8}")
    print("-" * 40)
    print(f"{'P(0 hits)':<15} {np.nanmean(p_zero):>8.3f} {zero_rate:>8.3f} {np.nanmean(p_zero)-zero_rate:>+8.3f}")
    print(f"{'P(1 hit)':<15} {np.nanmean(p_one):>8.3f} {one_rate:>8.3f} {np.nanmean(p_one)-one_rate:>+8.3f}")
    print(f"{'P(2+ hits)':<15} {np.nanmean(p_multi):>8.3f} {multi_rate:>8.3f} {np.nanmean(p_multi)-multi_rate:>+8.3f}")

    # Brier scores
    print()
    print("=" * 70)
    print("BRIER SCORES (lower = better, 0.25 = coin flip)")
    print("=" * 70)
    mask = np.isfinite(p05) & np.isfinite(p15)
    if mask.sum() > 100:
        act_0 = (actual[mask] == 0).astype(float)
        act_ge1 = (actual[mask] >= 1).astype(float)
        act_ge2 = (actual[mask] >= 2).astype(float)

        br_ge1 = brier_score_loss(act_ge1, p05[mask])
        br_ge2 = brier_score_loss(act_ge2, p15[mask])
        print(f"  H >= 1 (over 0.5): Brier = {br_ge1:.4f}")
        print(f"  H >= 2 (over 1.5): Brier = {br_ge2:.4f}")
        if p25 is not None:
            mask3 = mask & np.isfinite(p25)
            act_ge3 = (actual[mask3] >= 3).astype(float)
            br_ge3 = brier_score_loss(act_ge3, p25[mask3])
            print(f"  H >= 3 (over 2.5): Brier = {br_ge3:.4f}")

    # Calibration by predicted P(2+ hits) quintile
    print()
    print("=" * 70)
    print("CALIBRATION: P(2+ HITS) BY MODEL QUINTILE")
    print("=" * 70)
    df = bh[mask].copy()
    df["p_multi"] = p15[mask]
    df["actual_multi"] = (actual[mask] >= 2).astype(float)
    df["actual_zero"] = (actual[mask] == 0).astype(float)
    df["actual_one"] = (actual[mask] == 1).astype(float)

    df["quintile"] = pd.qcut(df["p_multi"], 5, labels=False, duplicates="drop")
    cal = df.groupby("quintile").agg(
        n=("actual", "size"),
        pred_multi=("p_multi", "mean"),
        pred_expected=("expected", "mean"),
        actual_mean_h=("actual", "mean"),
        rate_0h=("actual_zero", "mean"),
        rate_1h=("actual_one", "mean"),
        rate_2plus=("actual_multi", "mean"),
    ).reset_index()

    print(f"{'Q':>2} {'n':>6} {'pred_2+':>8} {'act_2+':>8} {'pred_H':>7} {'act_H':>6} "
          f"{'0H%':>6} {'1H%':>6} {'2+H%':>6}")
    print("-" * 70)
    for _, r in cal.iterrows():
        print(f"{int(r['quintile']):>2} {int(r['n']):>6} {r['pred_multi']:>8.3f} {r['rate_2plus']:>8.3f} "
              f"{r['pred_expected']:>7.3f} {r['actual_mean_h']:>6.3f} "
              f"{r['rate_0h']:>6.3f} {r['rate_1h']:>6.3f} {r['rate_2plus']:>6.3f}")

    # Discrimination: top vs bottom quintile
    top = cal.iloc[-1]
    bot = cal.iloc[0]
    print(f"\n  Top quintile 2+H rate: {top['rate_2plus']:.3f} ({top['pred_multi']:.3f} predicted)")
    print(f"  Bottom quintile 2+H rate: {bot['rate_2plus']:.3f} ({bot['pred_multi']:.3f} predicted)")
    ratio = top["rate_2plus"] / bot["rate_2plus"] if bot["rate_2plus"] > 0 else float("inf")
    print(f"  Ratio: {ratio:.2f}x")

    # Decile breakdown for finer resolution
    print()
    print("=" * 70)
    print("CALIBRATION: P(2+ HITS) BY MODEL DECILE")
    print("=" * 70)
    df["decile"] = pd.qcut(df["p_multi"], 10, labels=False, duplicates="drop")
    cal10 = df.groupby("decile").agg(
        n=("actual", "size"),
        pred_multi=("p_multi", "mean"),
        pred_expected=("expected", "mean"),
        actual_mean_h=("actual", "mean"),
        rate_0h=("actual_zero", "mean"),
        rate_1h=("actual_one", "mean"),
        rate_2plus=("actual_multi", "mean"),
    ).reset_index()

    print(f"{'D':>2} {'n':>6} {'pred_2+':>8} {'act_2+':>8} {'pred_H':>7} {'act_H':>6} "
          f"{'0H%':>6} {'1H%':>6} {'2+H%':>6}")
    print("-" * 70)
    for _, r in cal10.iterrows():
        print(f"{int(r['decile']):>2} {int(r['n']):>6} {r['pred_multi']:>8.3f} {r['rate_2plus']:>8.3f} "
              f"{r['pred_expected']:>7.3f} {r['actual_mean_h']:>6.3f} "
              f"{r['rate_0h']:>6.3f} {r['rate_1h']:>6.3f} {r['rate_2plus']:>6.3f}")

    # By batting order
    print()
    print("=" * 70)
    print("DIFFERENTIATION BY BATTING ORDER")
    print("=" * 70)
    print(f"{'#':>2} {'n':>6} {'pred_H':>7} {'act_H':>6} {'pred_2+':>8} {'act_2+':>8} {'0H%':>6}")
    print("-" * 55)
    for bo in range(1, 10):
        sub = df[df["batting_order"] == bo] if "batting_order" in df.columns else pd.DataFrame()
        if len(sub) < 20:
            continue
        print(f"{bo:>2} {len(sub):>6} {sub['expected'].mean():>7.3f} {sub['actual'].mean():>6.3f} "
              f"{sub['p_multi'].mean():>8.3f} {sub['actual_multi'].mean():>8.3f} "
              f"{sub['actual_zero'].mean():>6.3f}")

    # ROC-like: if we pick top N% by P(2+H), what fraction actually get 2+?
    print()
    print("=" * 70)
    print("PICK ACCURACY: TOP N% BY P(2+ HITS)")
    print("=" * 70)
    df_sorted = df.sort_values("p_multi", ascending=False)
    for pct in [5, 10, 20, 30, 50]:
        n_pick = int(len(df_sorted) * pct / 100)
        top_slice = df_sorted.head(n_pick)
        hit_rate = top_slice["actual_multi"].mean()
        avg_pred = top_slice["p_multi"].mean()
        print(f"  Top {pct:>2}%: n={n_pick:>5d}  pred_2+={avg_pred:.3f}  actual_2+={hit_rate:.3f}  "
              f"lift={hit_rate/multi_rate:.2f}x vs baseline {multi_rate:.3f}")


if __name__ == "__main__":
    main()
