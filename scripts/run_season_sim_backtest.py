#!/usr/bin/env python
"""
Walk-forward backtest for the season simulator pipeline.

Tests the full end-to-end: PyMC hierarchical models -> posterior extraction ->
PA-by-PA game simulator -> season counting stats + fantasy scoring.

Compares sim-based projections against old rate x BF and Marcel baselines.

Usage
-----
    # Quick mode (~15-20 min)
    python scripts/run_season_sim_backtest.py --quick

    # Full quality (~45-75 min)
    python scripts/run_season_sim_backtest.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.season_sim_backtest import walk_forward_season_sim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("season_sim_backtest")


def _print_summary(summary: pd.DataFrame) -> None:
    """Print formatted backtest results."""
    print(f"\n{'=' * 75}")
    print("  SEASON SIMULATOR BACKTEST RESULTS")
    print(f"{'=' * 75}")

    # Overall stats
    overall = summary[summary["role"] == "ALL"]
    if not overall.empty:
        print("\n--- OVERALL ---")
        for _, r in overall.iterrows():
            stat = r["stat"]
            mae = f"MAE={r['sim_mae']:5.1f}" if pd.notna(r.get("sim_mae")) else ""
            bias = f"bias={r['sim_bias']:+5.1f}" if pd.notna(r.get("sim_bias")) else ""
            corr = f"r={r['sim_corr']:.3f}" if pd.notna(r.get("sim_corr")) else ""
            cov80 = f"80%CI={r['sim_cov80']:.0%}" if pd.notna(r.get("sim_cov80")) else ""
            vs_m = ""
            if pd.notna(r.get("mae_vs_marcel_pct")) and r["mae_vs_marcel_pct"] != 0:
                vs_m = f"vs Marcel: {r['mae_vs_marcel_pct']:+.1f}%"
            vs_o = ""
            if pd.notna(r.get("mae_vs_old_pct")) and r["mae_vs_old_pct"] != 0:
                vs_o = f"vs old: {r['mae_vs_old_pct']:+.1f}%"
            parts = [p for p in [mae, bias, corr, cov80, vs_m, vs_o] if p]
            print(f"  {stat:15s}  {r['n']:3.0f}p  {'  '.join(parts)}")

    # SP/RP split
    for role in ["SP", "RP"]:
        split = summary[summary["role"] == role]
        if split.empty:
            continue
        print(f"\n--- {role} ---")
        for _, r in split.iterrows():
            stat = r["stat"]
            mae = f"MAE={r['sim_mae']:5.1f}" if pd.notna(r.get("sim_mae")) else ""
            bias = f"bias={r['sim_bias']:+5.1f}" if pd.notna(r.get("sim_bias")) else ""
            corr = f"r={r['sim_corr']:.3f}" if pd.notna(r.get("sim_corr")) else ""
            cov80 = f"80%CI={r['sim_cov80']:.0%}" if pd.notna(r.get("sim_cov80")) else ""
            vs_m = ""
            if pd.notna(r.get("mae_vs_marcel_pct")) and r["mae_vs_marcel_pct"] != 0:
                vs_m = f"vs Marcel: {r['mae_vs_marcel_pct']:+.1f}%"
            parts = [p for p in [mae, bias, corr, cov80, vs_m] if p]
            print(f"  {stat:15s}  {r['n']:3.0f}p  {'  '.join(parts)}")

    # Verdict
    print(f"\n{'=' * 75}")
    print("  VERDICT")
    print(f"{'=' * 75}")
    k_all = summary[(summary["stat"] == "total_k") & (summary["role"] == "ALL")]
    if not k_all.empty:
        r = k_all.iloc[0]
        beats_marcel = r.get("mae_vs_marcel_pct", 0) > 0
        beats_old = r.get("mae_vs_old_pct", 0) > 0
        print(f"  K:  Sim {'BEATS' if beats_marcel else 'LOSES TO'} Marcel "
              f"({r.get('mae_vs_marcel_pct', 0):+.1f}%), "
              f"{'BEATS' if beats_old else 'LOSES TO'} old counting "
              f"({r.get('mae_vs_old_pct', 0):+.1f}%)")

    ip_all = summary[(summary["stat"] == "projected_ip") & (summary["role"] == "ALL")]
    if not ip_all.empty:
        r = ip_all.iloc[0]
        print(f"  IP: Sim MAE={r['sim_mae']:.1f}, bias={r['sim_bias']:+.1f}, "
              f"80%CI={r.get('sim_cov80', 0):.0%}")

    k_sp = summary[(summary["stat"] == "total_k") & (summary["role"] == "SP")]
    k_rp = summary[(summary["stat"] == "total_k") & (summary["role"] == "RP")]
    if not k_sp.empty and not k_rp.empty:
        sp_v = k_sp.iloc[0].get("mae_vs_marcel_pct", 0)
        rp_v = k_rp.iloc[0].get("mae_vs_marcel_pct", 0)
        print(f"  K split: SP {sp_v:+.1f}% vs Marcel, RP {rp_v:+.1f}% vs Marcel")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Season sim backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws for fast iteration")
    args = parser.parse_args()

    if args.quick:
        sampling = dict(draws=500, tune=250, chains=2, random_seed=42, n_seasons=100)
        logger.info("QUICK mode: draws=500, tune=250, chains=2, n_seasons=100")
    else:
        sampling = dict(draws=2000, tune=1000, chains=4, random_seed=42, n_seasons=200)
        logger.info("FULL mode: draws=2000, tune=1000, chains=4, n_seasons=200")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Single fold: train 2018-2024, test 2025
    result = walk_forward_season_sim(
        train_seasons=list(range(2018, 2025)),
        test_season=2025,
        **sampling,
    )

    summary = result["summary"]
    _print_summary(summary)

    # Save
    summary.to_csv(out_dir / "season_sim_backtest_summary.csv", index=False)
    logger.info("Saved summary to outputs/season_sim_backtest_summary.csv")

    # Print timings
    timings = result["timings"]
    total = sum(timings.values())
    logger.info("Timings: %s", {k: f"{v:.0f}s" for k, v in timings.items()})
    logger.info("Total: %.0fs (%.1f min)", total, total / 60)


if __name__ == "__main__":
    main()
