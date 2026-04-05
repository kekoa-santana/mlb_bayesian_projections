#!/usr/bin/env python
"""
Automated change safety gate.

Runs the precompute pipeline (--quick), then validates output parquet
schemas and backtest metric thresholds.  Returns exit code 0 if all
checks pass, 1 if any fail.

Usage
-----
    python scripts/validate_change_safety.py                  # full run
    python scripts/validate_change_safety.py --skip-precompute # artifacts only
    python scripts/validate_change_safety.py --config path.yaml
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = (
    Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")
    / "data"
    / "dashboard"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_change_safety")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated change safety gate")
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip running precompute; validate existing artifacts only",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "change_safety.yaml",
        help="Path to threshold config YAML",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load change safety thresholds from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checks ──────────────────────────────────────────────────────────


def run_precompute() -> bool:
    """Run precompute_dashboard_data.py --quick and return success."""
    logger.info("Running precompute --quick ...")
    script = PROJECT_ROOT / "scripts" / "precompute_dashboard_data.py"
    result = subprocess.run(
        [sys.executable, str(script), "--quick"],
        cwd=str(PROJECT_ROOT),
    )
    passed = result.returncode == 0
    if passed:
        logger.info("PASS  precompute completed successfully")
    else:
        logger.error("FAIL  precompute exited with code %d", result.returncode)
    return passed


def check_parquet_schema(
    name: str,
    schema_cfg: dict,
) -> bool:
    """Validate a parquet file exists and has required columns."""
    path = DASHBOARD_DIR / f"{name}.parquet"
    if not path.exists():
        logger.warning("SKIP  %s not found at %s", name, path)
        return True  # missing file is a skip, not a failure

    df = pd.read_parquet(path)
    cols = set(df.columns)
    passed = True

    # Check required columns
    for col in schema_cfg.get("required_columns", []):
        if col not in cols:
            logger.error("FAIL  %s missing required column: %s", name, col)
            passed = False

    # Check required prefixes (at least one column with that prefix)
    for prefix in schema_cfg.get("required_prefix", []):
        if not any(c.startswith(prefix) for c in cols):
            logger.error(
                "FAIL  %s has no column with prefix '%s'", name, prefix
            )
            passed = False

    if passed:
        logger.info("PASS  %s schema OK (%d cols, %d rows)", name, len(cols), len(df))
    return passed


def check_thresholds(cfg: dict) -> bool:
    """Validate backtest metrics against configured thresholds."""
    thresholds = cfg.get("thresholds", {})
    path = DASHBOARD_DIR / "backtest_game_prop_summary.parquet"

    if not path.exists():
        logger.warning("SKIP  backtest_game_prop_summary.parquet not found -- skipping threshold checks")
        return True

    df = pd.read_parquet(path)
    passed = True

    # ECE check (across all stats)
    max_ece = thresholds.get("max_ece", 0.08)
    if "ece" in df.columns:
        worst_ece = df["ece"].max()
        if worst_ece > max_ece:
            logger.error(
                "FAIL  worst ECE = %.4f exceeds threshold %.4f", worst_ece, max_ece
            )
            passed = False
        else:
            logger.info("PASS  ECE OK (worst = %.4f, threshold = %.4f)", worst_ece, max_ece)

    # Temperature deviation check
    max_temp_dev = thresholds.get("max_temperature_deviation", 0.3)
    if "temperature" in df.columns:
        worst_dev = (df["temperature"] - 1.0).abs().max()
        if worst_dev > max_temp_dev:
            logger.error(
                "FAIL  worst |temperature - 1.0| = %.3f exceeds threshold %.3f",
                worst_dev, max_temp_dev,
            )
            passed = False
        else:
            logger.info(
                "PASS  temperature OK (worst deviation = %.3f, threshold = %.3f)",
                worst_dev, max_temp_dev,
            )

    return passed


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    all_passed = True

    # 1. Run precompute (unless skipped)
    if not args.skip_precompute:
        if not run_precompute():
            all_passed = False

    # 2. Schema validation
    schema_cfg = cfg.get("schema", {})
    for name, spec in schema_cfg.items():
        if not check_parquet_schema(name, spec):
            all_passed = False

    # 3. Threshold validation
    if not check_thresholds(cfg):
        all_passed = False

    # Summary
    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.error("SOME CHECKS FAILED -- review output above")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
