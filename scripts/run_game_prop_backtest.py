#!/usr/bin/env python
"""Run game-level prop backtests for all stat types.

Usage
-----
python scripts/run_game_prop_backtest.py                        # legacy engine, all props
python scripts/run_game_prop_backtest.py --side pitcher         # pitcher only
python scripts/run_game_prop_backtest.py --stat k bb            # K and BB only
python scripts/run_game_prop_backtest.py --side batter --stat k # batter K only
python scripts/run_game_prop_backtest.py --stratify             # stratified metrics

Engine comparison (pitcher props only):

    --engine legacy   # default; existing behavior, game_k_model.predict_game_batch_stat
    --engine pa_sim   # PA-by-PA simulator (game_sim.simulator.simulate_game)
    --engine both     # run both, emit *_legacy.csv, *_pa_sim.csv, *_diff.csv

Limit folds for a smoke test:

    --folds 2024:2020-2023 2025:2021-2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.game_prop_validation import (
    BATTER_PROP_CONFIGS,
    PITCHER_PROP_CONFIGS,
    compute_stratified_metrics,
    run_full_game_prop_backtest,
)
from src.evaluation.pa_sim_prop_adapter import (
    compute_engine_diff,
    run_pa_sim_prop_backtest,
)
from src.evaluation.runner import setup_logging

logger = setup_logging(__name__)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _parse_folds(fold_specs: list[str] | None) -> list[tuple[list[int], int]] | None:
    """Parse ``TEST:START-END`` fold specs into ``(train_seasons, test_season)``."""
    if not fold_specs:
        return None
    folds: list[tuple[list[int], int]] = []
    for spec in fold_specs:
        test_s, train_range = spec.split(":")
        start, end = train_range.split("-")
        test_season = int(test_s)
        train = list(range(int(start), int(end) + 1))
        folds.append((train, test_season))
    return folds


def _run_legacy(
    args: argparse.Namespace,
    configs_to_run: list[tuple[str, object]],
    folds: list[tuple[list[int], int]] | None,
    suffix: str = "",
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run the legacy engine; returns (summaries, stratified, per-prop preds)."""
    summaries: list[pd.DataFrame] = []
    stratified: list[pd.DataFrame] = []
    predictions_by_prop: dict[str, pd.DataFrame] = {}

    for label, config in configs_to_run:
        logger.info("=" * 60)
        logger.info("[legacy] Running backtest: %s", label)
        logger.info("=" * 60)

        try:
            summary_df, predictions_df = run_full_game_prop_backtest(
                config=config,
                folds=folds,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                n_mc_draws=args.mc_draws,
                random_seed=args.seed,
            )

            pred_path = OUTPUT_DIR / (
                f"game_prop_predictions_{label}{suffix}.parquet"
            )
            predictions_df.to_parquet(pred_path, index=False)
            logger.info("Saved predictions to %s", pred_path)

            predictions_by_prop[label] = predictions_df

            summary_df["prop"] = label
            summary_df["side"] = config.side
            summary_df["stat"] = config.stat_name
            summary_df["engine"] = "legacy"
            summaries.append(summary_df)

            if args.stratify and len(predictions_df) > 0:
                strat_df = compute_stratified_metrics(config, predictions_df)
                if len(strat_df) > 0:
                    strat_df["prop"] = label
                    strat_df["engine"] = "legacy"
                    stratified.append(strat_df)

        except Exception:
            logger.exception("[legacy] Failed backtest for %s", label)
            continue

    return summaries, stratified, predictions_by_prop


def _run_pa_sim(
    args: argparse.Namespace,
    configs_to_run: list[tuple[str, object]],
    folds: list[tuple[list[int], int]] | None,
    suffix: str = "",
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run the PA-sim engine; returns (summaries, stratified, per-prop preds).

    The PA sim covers pitcher K/BB/H/HR/Outs. Batter props and any
    non-pitcher-stat configs are silently skipped with a log message.
    """
    # Filter to pitcher props the PA sim supports.
    pa_sim_stats: list[str] = []
    for label, config in configs_to_run:
        if config.side != "pitcher":
            logger.info("[pa_sim] skipping non-pitcher prop %s", label)
            continue
        if config.stat_name not in ("k", "bb", "h", "hr", "outs"):
            logger.info("[pa_sim] skipping unsupported stat %s", label)
            continue
        pa_sim_stats.append(config.stat_name)

    if not pa_sim_stats:
        logger.warning("[pa_sim] no compatible props to run")
        return [], [], {}

    default_folds = folds or [
        ([2020, 2021, 2022], 2023),
        ([2020, 2021, 2022, 2023], 2024),
        ([2020, 2021, 2022, 2023, 2024], 2025),
    ]

    summary_df, preds_by_stat = run_pa_sim_prop_backtest(
        folds=default_folds,
        stats=tuple(pa_sim_stats),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        n_sims=args.mc_draws,
        random_seed=args.seed,
    )

    summaries: list[pd.DataFrame] = []
    stratified: list[pd.DataFrame] = []
    predictions_by_prop: dict[str, pd.DataFrame] = {}

    for stat, df in preds_by_stat.items():
        label = f"pitcher_{stat}"
        predictions_by_prop[label] = df
        pred_path = OUTPUT_DIR / (
            f"game_prop_predictions_{label}{suffix}.parquet"
        )
        df.to_parquet(pred_path, index=False)
        logger.info("[pa_sim] Saved predictions to %s", pred_path)

    if not summary_df.empty:
        summary_df["prop"] = "pitcher_" + summary_df["stat_name"]
        summary_df["side"] = "pitcher"
        summary_df["stat"] = summary_df["stat_name"]
        summaries.append(summary_df)

        if args.stratify:
            for stat, df in preds_by_stat.items():
                cfg = PITCHER_PROP_CONFIGS[stat]
                try:
                    strat_df = compute_stratified_metrics(cfg, df)
                except Exception:
                    logger.exception(
                        "[pa_sim] stratified metrics failed for %s", stat,
                    )
                    continue
                if len(strat_df) > 0:
                    strat_df["prop"] = f"pitcher_{stat}"
                    strat_df["engine"] = "pa_sim"
                    stratified.append(strat_df)

    return summaries, stratified, predictions_by_prop


def _write_diff(
    legacy_preds: dict[str, pd.DataFrame],
    pa_sim_preds: dict[str, pd.DataFrame],
    suffix: str,
) -> pd.DataFrame:
    """Build the per-(stat, line) Brier/slope/ECE diff CSV."""
    frames: list[pd.DataFrame] = []
    for label, legacy_df in legacy_preds.items():
        if label not in pa_sim_preds:
            continue
        stat = label.replace("pitcher_", "")
        cfg = PITCHER_PROP_CONFIGS.get(stat)
        if cfg is None:
            continue
        diff = compute_engine_diff(
            legacy_df, pa_sim_preds[label], stat, cfg.default_lines,
        )
        if not diff.empty:
            diff["prop"] = label
            frames.append(diff)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    path = OUTPUT_DIR / f"game_prop_backtest_diff{suffix}.csv"
    combined.to_csv(path, index=False)
    logger.info("Saved engine diff to %s", path)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Game prop backtests")
    parser.add_argument("--side", choices=["pitcher", "batter", "all"], default="all")
    parser.add_argument("--stat", nargs="+", default=None,
                        help="Stats to backtest (k, bb, hr, h, outs)")
    parser.add_argument("--engine", choices=["legacy", "pa_sim", "both"],
                        default="legacy",
                        help="Prop engine. pa_sim/both only apply to "
                             "pitcher props.")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Fold specs like 2024:2020-2023. Omit for "
                             "defaults (3 walk-forward folds).")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true",
                        help="Compute stratified metrics by context columns")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folds = _parse_folds(args.folds)

    configs_to_run: list[tuple[str, object]] = []
    if args.side in ("pitcher", "all"):
        for name, config in PITCHER_PROP_CONFIGS.items():
            if args.stat is None or name in args.stat:
                configs_to_run.append((f"pitcher_{name}", config))
    if args.side in ("batter", "all"):
        for name, config in BATTER_PROP_CONFIGS.items():
            if args.stat is None or name in args.stat:
                configs_to_run.append((f"batter_{name}", config))

    all_summaries: list[pd.DataFrame] = []
    all_stratified: list[pd.DataFrame] = []
    legacy_preds: dict[str, pd.DataFrame] = {}
    pa_sim_preds: dict[str, pd.DataFrame] = {}

    if args.engine in ("legacy", "both"):
        suffix = "_legacy" if args.engine == "both" else ""
        s, st, preds = _run_legacy(args, configs_to_run, folds, suffix)
        all_summaries.extend(s)
        all_stratified.extend(st)
        legacy_preds = preds

    if args.engine in ("pa_sim", "both"):
        suffix = "_pa_sim" if args.engine == "both" else ""
        s, st, preds = _run_pa_sim(args, configs_to_run, folds, suffix)
        all_summaries.extend(s)
        all_stratified.extend(st)
        pa_sim_preds = preds

    if args.engine == "both":
        _write_diff(legacy_preds, pa_sim_preds, suffix="")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        if args.engine == "both":
            summary_path = OUTPUT_DIR / "game_prop_backtest_summary_both.parquet"
        else:
            summary_path = OUTPUT_DIR / (
                f"game_prop_backtest_summary_{args.engine}.parquet"
                if args.engine != "legacy"
                else "game_prop_backtest_summary.parquet"
            )
        combined.to_parquet(summary_path, index=False)
        logger.info("Saved combined summary to %s", summary_path)

        print("\n" + "=" * 80)
        print("GAME PROP BACKTEST SUMMARY")
        print("=" * 80)
        display_cols = ["engine", "prop", "test_season", "n_games",
                        "avg_brier", "avg_log_loss", "ece",
                        "coverage_80", "coverage_90",
                        "sharpness_mean_confidence", "sharpness_pct_actionable_60",
                        "sharpness_entropy"]
        available = [c for c in display_cols if c in combined.columns]
        print(combined[available].to_string(index=False))
    else:
        logger.warning("No backtests completed successfully")

    if all_stratified:
        strat_combined = pd.concat(all_stratified, ignore_index=True)
        strat_path = OUTPUT_DIR / (
            f"game_prop_stratified_summary_{args.engine}.parquet"
            if args.engine != "legacy"
            else "game_prop_stratified_summary.parquet"
        )
        strat_combined.to_parquet(strat_path, index=False)
        logger.info("Saved stratified summary to %s", strat_path)

        print("\n" + "=" * 80)
        print("STRATIFIED METRICS")
        print("=" * 80)
        strat_cols = ["engine", "prop", "stratum", "bin", "bin_n",
                      "avg_brier", "avg_log_loss", "ece",
                      "temperature"]
        avail = [c for c in strat_cols if c in strat_combined.columns]
        print(strat_combined[avail].to_string(index=False))


if __name__ == "__main__":
    main()
