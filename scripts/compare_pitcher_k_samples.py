"""Three-way parity check on pitcher K-rate posterior samples.

Compares:
  - LEGACY:  production pitcher_k_samples.npz (legacy pitcher_k_rate_model,
             min_bf=9 multi-season, random-walk AR)
  - MB100:   sandbox pitcher_k_samples_mb100.npz (generalized pitcher_model,
             min_bf=100, AR(1))
  - MB50:    sandbox pitcher_k_samples.npz (generalized pitcher_model,
             min_bf=50, AR(1))

Reports:
  1. Coverage pairwise deltas
  2. LEGACY vs MB50: numerical disagreement (adoption check)
  3. MB100 vs MB50 on HIGH-BF pitchers: contamination check
     (do heavy-BF pitchers' posteriors shift when we lower the training floor?)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEGACY = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard/pitcher_k_samples.npz")
SANDBOX = Path("C:/Users/kekoa/Documents/data_analytics/sandbox_dashboard/data/dashboard")
MB100 = SANDBOX / "pitcher_k_samples_mb100.npz"
MB50 = SANDBOX / "pitcher_k_samples.npz"


def summarize(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "q025": float(np.quantile(samples, 0.025)),
        "q975": float(np.quantile(samples, 0.975)),
        "ci95_width": float(np.quantile(samples, 0.975) - np.quantile(samples, 0.025)),
    }


def pairwise_deltas(a_npz, b_npz, label_a: str, label_b: str) -> pd.DataFrame:
    common = sorted(set(a_npz.files) & set(b_npz.files), key=int)
    rows = []
    for pid in common:
        a = summarize(a_npz[pid])
        b = summarize(b_npz[pid])
        rows.append({
            "pitcher_id": int(pid),
            f"{label_a}_mean": a["mean"],
            f"{label_b}_mean": b["mean"],
            "mean_delta": b["mean"] - a["mean"],
            f"{label_a}_ci95": a["ci95_width"],
            f"{label_b}_ci95": b["ci95_width"],
            "ci95_delta": b["ci95_width"] - a["ci95_width"],
        })
    return pd.DataFrame(rows)


def print_delta_stats(df: pd.DataFrame, label: str) -> None:
    abs_delta = df["mean_delta"].abs()
    print(f"  Median |dmean|:    {abs_delta.median():.4f}")
    print(f"  Mean   |dmean|:    {abs_delta.mean():.4f}")
    print(f"  Max    |dmean|:    {abs_delta.max():.4f}")
    print(f"  90th %ile |dmean|: {np.quantile(abs_delta, 0.9):.4f}")
    print(f"  99th %ile |dmean|: {np.quantile(abs_delta, 0.99):.4f}")
    print(f"  Signed median:     {df['mean_delta'].median():+.4f}  (bias if nonzero)")
    print(f"  CI95 width median d: {df['ci95_delta'].median():+.4f}  (negative = tighter)")


def main() -> None:
    legacy = np.load(LEGACY)
    mb100 = np.load(MB100)
    mb50 = np.load(MB50)

    print("=" * 70)
    print("THREE-WAY PARITY CHECK")
    print("=" * 70)
    print(f"  LEGACY:  {len(legacy.files):>4} pitchers  (legacy model, min_bf=9)")
    print(f"  MB100:   {len(mb100.files):>4} pitchers  (generalized, min_bf=100)")
    print(f"  MB50:    {len(mb50.files):>4} pitchers  (generalized, min_bf=50)")
    print()

    # Coverage matrix
    ids_legacy = set(legacy.files)
    ids_mb100 = set(mb100.files)
    ids_mb50 = set(mb50.files)
    print("Pairwise coverage (rows = superset, cols = subset)")
    print(f"  LEGACY & MB50:    {len(ids_legacy & ids_mb50):>4}  (LEGACY only: {len(ids_legacy - ids_mb50)}, MB50 only: {len(ids_mb50 - ids_legacy)})")
    print(f"  MB100  & MB50:    {len(ids_mb100 & ids_mb50):>4}  (MB100 only: {len(ids_mb100 - ids_mb50)}, MB50 only: {len(ids_mb50 - ids_mb100)})")
    print()

    # --- LEGACY vs MB50 (adoption check) ---
    print("-" * 70)
    print("ADOPTION CHECK: LEGACY vs MB50 (common pitchers)")
    print("-" * 70)
    df_leg_mb50 = pairwise_deltas(legacy, mb50, "legacy", "mb50")
    print_delta_stats(df_leg_mb50, "LEGACY vs MB50")
    print()

    # --- MB100 vs MB50 (contamination check) ---
    print("-" * 70)
    print("CONTAMINATION CHECK: MB100 vs MB50 (high-BF pitchers shared between both)")
    print("-" * 70)
    df_contam = pairwise_deltas(mb100, mb50, "mb100", "mb50")
    print(f"  Common pitchers between MB100 and MB50: {len(df_contam)}")
    print(f"  (Expected: all of MB100 should be in MB50 since MB50 is a superset)")
    print()
    print_delta_stats(df_contam, "MB100 vs MB50")
    print()
    print("  INTERPRETATION:")
    print("    If median |dmean| is within MC noise floor (~0.005-0.010 for quick mode),")
    print("    then lowering min_bf does NOT materially shift high-BF pitcher posteriors.")
    print("    This validates the 'partial pooling protects us' claim.")
    print()

    # Top contamination cases
    print("  Top 10 largest MB100 vs MB50 shifts (potential contamination):")
    top_contam = df_contam.reindex(df_contam["mean_delta"].abs().sort_values(ascending=False).index).head(10)
    print(top_contam[["pitcher_id", "mb100_mean", "mb50_mean", "mean_delta", "mb100_ci95", "mb50_ci95"]].to_string(index=False))
    print()

    # Write CSVs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    df_leg_mb50.to_csv(out_dir / "pitcher_k_parity_legacy_vs_mb50.csv", index=False)
    df_contam.to_csv(out_dir / "pitcher_k_parity_mb100_vs_mb50.csv", index=False)
    print(f"Wrote diffs to {out_dir}/pitcher_k_parity_*.csv")


if __name__ == "__main__":
    main()
