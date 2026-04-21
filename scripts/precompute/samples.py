"""Precompute: Posterior sample extraction (npz files)."""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.samples")

# Maximum posterior draws saved per player.  1,000 is sufficient for KDE
# display and game-level Monte Carlo while keeping NPZ files small (~24 MB
# total vs ~189 MB at 8,000 draws).
_MAX_DASHBOARD_SAMPLES = 1_000


def _downsample(arr: np.ndarray, max_n: int = _MAX_DASHBOARD_SAMPLES) -> np.ndarray:
    """Thin a posterior sample array to *max_n* draws (evenly spaced)."""
    if arr.shape[0] <= max_n:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_n, dtype=int)
    return arr[idx]


def run_pitcher_samples(
    *,
    pitcher_results: dict,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
) -> None:
    """Extract and save pitcher posterior samples (K%, BB%, HR/BF)."""
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not pitcher_results:
        logger.warning("pitcher_results not available (pitcher_models skipped) -- skipping pitcher_samples")
        return

    # Prefer the generalized pitcher_model pipeline for all three rate stats.
    # pitcher_k_samples.npz is consumed by update_in_season and confident_picks;
    # historically it was produced by a separate legacy pitcher_k_rate_model
    # fit below. Single-source-of-truth is the generalized path.
    k_samples_from_generalized = False
    for stat_name, npz_name in [
        ("k_rate", "pitcher_k_samples"),
        ("bb_rate", "pitcher_bb_samples"),
        ("hr_per_bf", "pitcher_hr_samples"),
    ]:
        pre = pitcher_results.get(stat_name, {}).get("rate_samples", {})
        if not pre:
            logger.warning("No %s pre-extracted samples -- skipping %s", stat_name, npz_name)
            continue

        samples_dict = {str(pid): _downsample(arr) for pid, arr in pre.items()}
        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **samples_dict)
        logger.info(
            "Saved %s posterior samples for %d pitchers (source=pitcher_model)",
            stat_name, len(samples_dict),
        )

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **samples_dict)
        logger.info("Saved preseason %s samples snapshot", stat_name)

        if stat_name == "k_rate":
            k_samples_from_generalized = True

    # Free pre-extracted samples -- already saved to disk above
    for stat_key in list(pitcher_results.keys()):
        pitcher_results[stat_key].pop("rate_samples", None)
    gc.collect()

    if not k_samples_from_generalized:
        raise RuntimeError(
            "Generalized pitcher_model did not produce K-rate samples. "
            "Check pitcher_results['k_rate']['rate_samples'] from models.py."
        )


def run_hitter_samples(
    *,
    hitter_results: dict,
) -> None:
    """Extract and save hitter K% and BB% posterior samples."""
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not hitter_results:
        logger.warning("hitter_results not available (hitter_models skipped) -- skipping hitter_samples")
        return

    for stat_name, npz_name in [("k_rate", "hitter_k_samples"), ("bb_rate", "hitter_bb_samples")]:
        pre = hitter_results.get(stat_name, {}).get("rate_samples", {})
        if not pre:
            logger.warning("No %s pre-extracted samples -- skipping %s", stat_name, npz_name)
            continue

        h_samples_dict = {str(pid): _downsample(arr) for pid, arr in pre.items()}
        np.savez_compressed(DASHBOARD_DIR / f"{npz_name}.npz", **h_samples_dict)
        logger.info("Saved hitter %s posterior samples for %d batters", stat_name, len(h_samples_dict))

        np.savez_compressed(snapshot_dir / f"{npz_name}_preseason.npz", **h_samples_dict)
        logger.info("Saved preseason hitter %s samples snapshot", stat_name)
