"""Shared CLI / IO / logging boilerplate for backtest runner scripts.

These helpers capture the repeated patterns across ``scripts/run_*_backtest.py``
(argparse setup, logging.basicConfig, MCMC sampling dicts, outputs dir creation,
and CSV writes) so individual runners only contain the domain-specific fold
logic and verdict printing.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def setup_logging(name: str) -> logging.Logger:
    """Configure root logging once and return a named logger.

    Only calls ``logging.basicConfig`` if the root logger has no handlers yet,
    so re-imports in the same process don't stack handlers.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
    return logging.getLogger(name)


def quick_full_sampling(quick: bool) -> dict:
    """Return the canonical quick/full PyMC sampling kwargs dict.

    Used by rate-model backtests (hitter, pitcher, counting, season sim).
    """
    if quick:
        return dict(draws=500, tune=250, chains=2, random_seed=42)
    return dict(draws=2000, tune=1000, chains=4, random_seed=42)


def quick_full_game_mcmc(quick: bool) -> tuple[int, int, int]:
    """Return ``(draws, tune, chains)`` for game-level engine backtests.

    Distinct from :func:`quick_full_sampling` — game-level backtests
    (game_k, game_sim, batter_sim) use a lighter spec because Monte Carlo
    draws dominate runtime. Callers pick their own ``n_mc`` / ``n_sims``
    count on top.
    """
    return (500, 250, 2) if quick else (1000, 500, 2)


# Canonical walk-forward fold windows for rate-model backtests.
# Matched across hitter and pitcher runners so metrics are comparable.
RATE_FOLDS: list[dict] = [
    {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
]

COUNTING_FOLDS: list[dict] = [
    {"train_seasons": list(range(2018, 2022)), "test_season": 2022},
    {"train_seasons": list(range(2018, 2023)), "test_season": 2023},
    {"train_seasons": list(range(2018, 2024)), "test_season": 2024},
    {"train_seasons": list(range(2018, 2025)), "test_season": 2025},
]


def ensure_out_dir() -> Path:
    """Create (if needed) and return the ``outputs/`` directory."""
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def save_csv(df: pd.DataFrame, filename: str, logger: logging.Logger) -> Path:
    """Write ``df`` to ``outputs/<filename>`` (no index) and log the save."""
    path = ensure_out_dir() / filename
    df.to_csv(path, index=False)
    logger.info("Saved to outputs/%s", filename)
    return path


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    stat: bool = False,
    skip_counting: bool = False,
    player_type: bool = False,
) -> None:
    """Attach the shared backtest CLI flags to ``parser``.

    Always adds ``--quick``. Opt-in flags mirror the semantics of the six
    canonical runner scripts exactly.
    """
    parser.add_argument("--quick", action="store_true")
    if stat:
        parser.add_argument(
            "--stat",
            type=str,
            default=None,
            help="Single rate stat to backtest (e.g. k_rate, bb_rate)",
        )
    if skip_counting:
        parser.add_argument(
            "--skip-counting",
            action="store_true",
            help="Skip counting stat backtest",
        )
    if player_type:
        parser.add_argument(
            "--type",
            choices=["hitter", "pitcher"],
            help="Player type to backtest",
        )
