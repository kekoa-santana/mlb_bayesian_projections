"""Walk-forward backtest for XGBoost breakout models.

Trains on expanding windows, predicts each held-out fold, reports:
- AUC, Precision@25, calibration lift per fold
- Mean metrics across all folds
- Feature importance stability

Usage:
    python scripts/run_breakout_backtest.py [--type hitter|pitcher|both]

Precompute groups: none (standalone script)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import setup_logging

logger = setup_logging("breakout_backtest")
def _load_config() -> dict:
    cfg_path = Path("config/model.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f).get("breakout", {})
    return {}


def _build_folds(cfg: dict) -> list[tuple[int, int]]:
    start = cfg.get("folds_start", 2001)
    end = cfg.get("folds_end", 2025)
    skip_covid = cfg.get("skip_covid_deltas", True)
    folds = []
    for y in range(start, end):
        if skip_covid and y in (2019, 2020):
            continue
        folds.append((y, y + 1))
    return folds


def run_hitter_backtest(folds: list[tuple[int, int]], cfg: dict) -> pd.DataFrame:
    """Run walk-forward hitter breakout backtest."""
    from src.data.queries import get_hitter_breakout_features
    from src.models.breakout_engine import BreakoutConfig, walk_forward_validate
    from src.models.breakout_model import HITTER_FEATURES

    hitter_cfg = cfg.get("hitter", {})
    xgb_cfg = cfg.get("xgb", {})

    config = BreakoutConfig(
        player_type="hitter",
        id_col="batter_id",
        name_col="batter_name",
        feature_cols=HITTER_FEATURES,
        breakout_metric="est_woba",
        breakout_direction="gain",
        breakout_threshold=hitter_cfg.get("woba_gain_threshold", 0.020),
        outcome_bound=hitter_cfg.get("woba_outcome_floor", 0.300),
        outcome_is_floor=True,
        n_archetypes=hitter_cfg.get("n_archetypes", 2),
        age_window=tuple(hitter_cfg.get("age_window", [24, 27])),
        xgb_params=xgb_cfg,
    )

    def outcome_loader(season: int, min_pa: int) -> pd.DataFrame:
        return get_hitter_breakout_features(season, min_pa)

    logger.info("=" * 60)
    logger.info("HITTER BREAKOUT BACKTEST (%d folds)", len(folds))
    logger.info("=" * 60)

    results = walk_forward_validate(
        config,
        feature_loader=get_hitter_breakout_features,
        outcome_loader=outcome_loader,
        folds=folds,
        min_train_folds=5,
    )
    return results


def run_pitcher_backtest(folds: list[tuple[int, int]], cfg: dict) -> pd.DataFrame:
    """Run walk-forward pitcher breakout backtest."""
    from src.data.queries import get_pitcher_breakout_features
    from src.models.breakout_engine import BreakoutConfig, walk_forward_validate
    from src.models.pitcher_breakout_model import PITCHER_FEATURES

    pitcher_cfg = cfg.get("pitcher", {})
    xgb_cfg = cfg.get("xgb", {})

    config = BreakoutConfig(
        player_type="pitcher",
        id_col="pitcher_id",
        name_col="pitcher_name",
        feature_cols=PITCHER_FEATURES,
        breakout_metric="era",
        breakout_direction="drop",
        breakout_threshold=pitcher_cfg.get("era_drop_threshold", 0.75),
        outcome_bound=pitcher_cfg.get("era_outcome_ceiling", 4.00),
        outcome_is_floor=False,
        n_archetypes=pitcher_cfg.get("n_archetypes", 3),
        age_window=tuple(pitcher_cfg.get("age_window", [23, 27])),
        xgb_params=xgb_cfg,
    )

    def outcome_loader(season: int, min_bf: int) -> pd.DataFrame:
        return get_pitcher_breakout_features(season, min_bf)

    logger.info("=" * 60)
    logger.info("PITCHER BREAKOUT BACKTEST (%d folds)", len(folds))
    logger.info("=" * 60)

    results = walk_forward_validate(
        config,
        feature_loader=get_pitcher_breakout_features,
        outcome_loader=outcome_loader,
        folds=folds,
        min_train_folds=5,
    )
    return results


def _print_summary(results: pd.DataFrame, label: str) -> None:
    """Print backtest summary statistics."""
    if results.empty:
        logger.warning("No results for %s backtest", label)
        return

    print(f"\n{'='*60}")
    print(f"  {label} BREAKOUT BACKTEST RESULTS")
    print(f"{'='*60}")
    print(results.to_string(index=False))

    print(f"\n--- Summary ({len(results)} folds) ---")
    for col in ["auc", "precision_at_25", "base_rate", "top_quartile_rate", "lift"]:
        if col in results.columns:
            vals = results[col].dropna()
            if not vals.empty:
                print(f"  {col:>20s}: mean={vals.mean():.3f}  "
                      f"std={vals.std():.3f}  "
                      f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    # Save results
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"breakout_backtest_{label.lower()}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Breakout model walk-forward backtest")
    parser.add_argument(
        "--type", choices=["hitter", "pitcher", "both"], default="both",
        help="Which model to backtest (default: both)",
    )
    args = parser.parse_args()

    cfg = _load_config()
    folds = _build_folds(cfg)
    logger.info("Loaded %d training folds (%s → %s)",
                len(folds), folds[0], folds[-1])

    if args.type in ("hitter", "both"):
        hitter_results = run_hitter_backtest(folds, cfg)
        _print_summary(hitter_results, "HITTER")

    if args.type in ("pitcher", "both"):
        pitcher_results = run_pitcher_backtest(folds, cfg)
        _print_summary(pitcher_results, "PITCHER")


if __name__ == "__main__":
    main()
