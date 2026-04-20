"""Shared CLI / IO / logging boilerplate for backtest runner scripts.

These helpers capture the repeated patterns across ``scripts/run_*_backtest.py``
(argparse setup, logging.basicConfig, MCMC sampling dicts, outputs dir creation,
and CSV writes) so individual runners only contain the domain-specific fold
logic and verdict printing.

Also provides ``fit_hitter_posteriors()`` — a shared model-fitting utility
used by batter_sim_validation, full_game_sim_batch, and game_sim_validation
to avoid triplicating hitter model fitting code.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Shared model-fitting utilities
# ---------------------------------------------------------------------------


def fit_hitter_posteriors(
    train_seasons: list[int],
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    min_pa: int = 50,
    random_seed: int = 42,
) -> dict[str, dict[int, np.ndarray]]:
    """Fit hitter K%, BB% models + HR pseudo-posteriors for one fold.

    Imports are deferred to avoid pulling PyMC into lightweight CLI
    scripts that only use the logging/IO helpers.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons for training models.
    draws, tune, chains : int
        MCMC sampling parameters.
    min_pa : int
        Minimum PA across training seasons to include a batter.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict[str, dict[int, np.ndarray]]
        ``{"k": {bid: samples}, "bb": {bid: samples}, "hr": {bid: samples}}``
    """
    from src.data.feature_eng import build_multi_season_hitter_data
    from src.models.hitter_model import (
        extract_rate_samples as extract_hitter_rate_samples,
        fit_hitter_model,
        prepare_hitter_data,
    )

    logger = logging.getLogger(__name__)
    last_train = max(train_seasons)

    df_hitter = build_multi_season_hitter_data(train_seasons, min_pa=min_pa)

    # --- K% ---
    logger.info("Fitting hitter K%% model...")
    data_k = prepare_hitter_data(df_hitter, "k_rate")
    _, trace_k = fit_hitter_model(
        data_k, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    batter_k: dict[int, np.ndarray] = {}
    for bid in data_k["df"][data_k["df"]["season"] == last_train]["batter_id"].unique():
        try:
            batter_k[int(bid)] = extract_hitter_rate_samples(
                trace_k, data_k, bid, last_train,
                project_forward=True, random_seed=random_seed,
            )
        except ValueError:
            continue
    logger.info("Batter K posteriors: %d", len(batter_k))

    # --- BB% ---
    logger.info("Fitting hitter BB%% model...")
    data_bb = prepare_hitter_data(df_hitter, "bb_rate")
    _, trace_bb = fit_hitter_model(
        data_bb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 1,
    )

    batter_bb: dict[int, np.ndarray] = {}
    for bid in data_bb["df"][data_bb["df"]["season"] == last_train]["batter_id"].unique():
        try:
            batter_bb[int(bid)] = extract_hitter_rate_samples(
                trace_bb, data_bb, bid, last_train,
                project_forward=True, random_seed=random_seed + 1,
            )
        except ValueError:
            continue
    logger.info("Batter BB posteriors: %d", len(batter_bb))

    # --- HR pseudo-posteriors (parametric bootstrap, no MCMC) ---
    logger.info("Building batter HR pseudo-posteriors...")
    batter_hr: dict[int, np.ndarray] = {}
    hr_data = df_hitter[df_hitter["season"].isin(train_seasons)].copy()
    for bid, grp in hr_data.groupby("batter_id"):
        total_hr = grp["hr"].sum() if "hr" in grp.columns else 0
        total_pa = grp["pa"].sum()
        if total_pa >= min_pa:
            rate = total_hr / total_pa
            rng_hr = np.random.default_rng(random_seed + int(bid) % 10000)
            std = max(0.005, rate * 0.15)
            samples = rng_hr.normal(rate, std, size=2000)
            samples = np.clip(samples, 0.001, 0.10)
            batter_hr[int(bid)] = samples
    logger.info("Batter HR posteriors: %d", len(batter_hr))

    return {"k": batter_k, "bb": batter_bb, "hr": batter_hr}
