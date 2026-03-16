#!/usr/bin/env python
"""Run game prop backtest with all Phase 1 features ENABLED.

Compares against outputs/baseline_calibration.csv to measure impact.

Features enabled:
- 1B+1C: Umpire BB/HR + Weather HR (auto-enabled in backtest functions)
- 1H: Catcher framing (auto-enabled in backtest functions)
- 2A: Derived H/Outs (auto-enabled via derived=True on configs)
- 1G: Lineup-aware pitcher props (lineup_proneness_weight=0.5)
- 1J: Opposing pitcher for batter props (opposing_pitcher_weight=0.5)

Also wired into backtest loop:
- 1A: TTO-adjusted game sim (via tto_profiles → build_tto_logit_lifts)
- 1I: Days rest adjustments (via rest_df → apply_rest_to_bf + rate lifts)

Usage
-----
python scripts/run_features_backtest.py
python scripts/run_features_backtest.py --draws 500 --tune 250 --chains 2
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.game_prop_validation import (
    BATTER_PROP_CONFIGS,
    PITCHER_PROP_CONFIGS,
    GamePropConfig,
    run_full_game_prop_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

AGG_METRICS = [
    "n_games", "rmse", "mae", "avg_brier", "crps",
    "ece", "mce", "temperature",
    "coverage_50", "coverage_80", "coverage_90", "coverage_95",
]


def enable_features(configs: dict[str, GamePropConfig], side: str) -> dict[str, GamePropConfig]:
    """Return copies of configs with Phase 1 features enabled."""
    enabled = {}
    for name, config in configs.items():
        c = copy.copy(config)
        if side == "pitcher":
            # 1G: Lineup-aware pitcher props
            c.lineup_proneness_weight = 0.5
        elif side == "batter":
            # 1J: Opposing pitcher quality for batter props
            c.opposing_pitcher_weight = 0.5
        enabled[name] = c
    return enabled


def load_baseline() -> pd.DataFrame | None:
    """Load baseline calibration CSV if it exists."""
    path = OUTPUT_DIR / "baseline_calibration.csv"
    if path.exists():
        return pd.read_csv(path)
    logger.warning("No baseline found at %s", path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run game prop backtest with Phase 1 features enabled"
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enable features
    pitcher_configs = enable_features(PITCHER_PROP_CONFIGS, "pitcher")
    batter_configs = enable_features(BATTER_PROP_CONFIGS, "batter")

    configs_to_run: list[tuple[str, GamePropConfig]] = []
    for name, config in pitcher_configs.items():
        configs_to_run.append((f"pitcher_{name}", config))
    for name, config in batter_configs.items():
        configs_to_run.append((f"batter_{name}", config))

    aggregated_rows: list[dict] = []

    for label, config in configs_to_run:
        logger.info("=" * 70)
        logger.info("FEATURES BACKTEST: %s", label)
        logger.info("=" * 70)

        try:
            summary_df, predictions_df = run_full_game_prop_backtest(
                config=config,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                n_mc_draws=args.mc_draws,
                random_seed=args.seed,
            )

            if summary_df.empty:
                logger.warning("No results for %s — skipping", label)
                continue

            agg_row: dict = {
                "prop": label,
                "side": config.side,
                "stat": config.stat_name,
                "n_folds": len(summary_df),
            }
            for metric in AGG_METRICS:
                if metric in summary_df.columns:
                    if metric == "n_games":
                        agg_row[metric] = int(summary_df[metric].sum())
                    else:
                        agg_row[metric] = float(summary_df[metric].mean())
                else:
                    agg_row[metric] = np.nan

            aggregated_rows.append(agg_row)

            logger.info(
                "  %s: MAE=%.3f  Brier=%.4f  ECE=%.4f  Cov80=%.2f",
                label,
                agg_row["mae"], agg_row["avg_brier"],
                agg_row["ece"], agg_row["coverage_80"],
            )

        except Exception:
            logger.exception("FAILED backtest for %s", label)
            continue

    if not aggregated_rows:
        logger.error("No backtests completed")
        return

    # Save results
    features_df = pd.DataFrame(aggregated_rows)
    features_df["generated_at"] = datetime.utcnow().isoformat(timespec="seconds")

    csv_path = OUTPUT_DIR / "features_calibration.csv"
    features_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("Saved features calibration to %s", csv_path)

    # Compare against baseline
    baseline_df = load_baseline()

    print("\n" + "=" * 100)
    print("FEATURES vs BASELINE COMPARISON")
    print("=" * 100)

    if baseline_df is not None:
        compare_metrics = ["mae", "avg_brier", "ece", "coverage_80", "coverage_90"]
        comparison_rows = []

        for _, row in features_df.iterrows():
            prop = row["prop"]
            baseline_row = baseline_df[baseline_df["prop"] == prop]
            if baseline_row.empty:
                continue

            comp = {"prop": prop}
            for m in compare_metrics:
                feat_val = row.get(m, np.nan)
                base_val = baseline_row[m].iloc[0] if m in baseline_row.columns else np.nan
                comp[f"{m}_baseline"] = base_val
                comp[f"{m}_features"] = feat_val
                if pd.notna(feat_val) and pd.notna(base_val) and base_val != 0:
                    comp[f"{m}_delta"] = feat_val - base_val
                    comp[f"{m}_pct_change"] = ((feat_val - base_val) / abs(base_val)) * 100
                else:
                    comp[f"{m}_delta"] = np.nan
                    comp[f"{m}_pct_change"] = np.nan

            # Calibration gate check
            brier_delta = comp.get("avg_brier_delta", 0)
            cov80 = comp.get("coverage_80_features", 0.80)
            comp["brier_gate"] = "PASS" if brier_delta <= 0.005 else "FAIL"
            comp["cov80_gate"] = "PASS" if 0.76 <= cov80 <= 0.88 else "WARN"

            comparison_rows.append(comp)

        if comparison_rows:
            for comp in comparison_rows:
                prop = comp["prop"]
                mae_d = comp.get("mae_pct_change", 0)
                brier_d = comp.get("avg_brier_delta", 0)
                ece_d = comp.get("ece_delta", 0)
                cov80 = comp.get("coverage_80_features", 0)
                bgate = comp.get("brier_gate", "?")
                cgate = comp.get("cov80_gate", "?")

                mae_arrow = "v" if mae_d < 0 else "^" if mae_d > 0 else "="
                brier_arrow = "v" if brier_d < 0 else "^" if brier_d > 0 else "="

                print(
                    f"  {prop:15s}  "
                    f"MAE {mae_arrow} {mae_d:+.1f}%  "
                    f"Brier {brier_arrow} {brier_d:+.4f}  "
                    f"ECE d{ece_d:+.4f}  "
                    f"Cov80={cov80:.1%}  "
                    f"[Brier:{bgate} Cov80:{cgate}]"
                )

            # Save comparison
            comp_df = pd.DataFrame(comparison_rows)
            comp_path = OUTPUT_DIR / "features_vs_baseline.csv"
            comp_df.to_csv(comp_path, index=False, float_format="%.6f")
            logger.info("Saved comparison to %s", comp_path)

    print("=" * 100)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
