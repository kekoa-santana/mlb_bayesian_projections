"""
Multi-year forward projections using the AR(1) talent evolution model.

Projects player rates 2-5 years forward by iterating the AR(1) process:
  rate_{t+1} = rho * rate_t + (1-rho) * population_mean + noise

Each forward year adds uncertainty (wider posteriors) as the signal
from current data decays. Combined with aging curves and playing time
decay, produces full career trajectory distributions.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# AR(1) persistence parameter (from model: rho ~ Beta(8,2), mean 0.8)
_RHO_MEAN = 0.80
_RHO_SD = 0.06

# Population means for regression target
_POP_MEANS = {
    "k_rate": {"hitter": 0.222, "pitcher": 0.220},
    "bb_rate": {"hitter": 0.083, "pitcher": 0.080},
    "hr_rate": {"hitter": 0.032, "pitcher": 0.030},
}

# Age-related playing time decay (fraction of prior year's games)
# Starts declining at 32 for hitters, 33 for pitchers
_AGE_GAMES_DECAY = {
    "hitter": {30: 1.00, 31: 0.98, 32: 0.95, 33: 0.92, 34: 0.88,
               35: 0.83, 36: 0.77, 37: 0.70, 38: 0.62, 39: 0.53, 40: 0.44},
    "pitcher": {30: 1.00, 31: 1.00, 32: 0.97, 33: 0.94, 34: 0.90,
                35: 0.85, 36: 0.78, 37: 0.70, 38: 0.60, 39: 0.50, 40: 0.40},
}

# Innovation noise per forward year (rate-scale SD added per year)
_INNOVATION_SD = {
    "k_rate": 0.020,
    "bb_rate": 0.012,
    "hr_rate": 0.008,
}


def project_rates_forward(
    current_samples: np.ndarray,
    stat: str,
    player_type: str,
    n_years: int = 3,
    random_seed: int = 42,
) -> list[np.ndarray]:
    """Project rate posterior samples N years forward via AR(1).

    Parameters
    ----------
    current_samples : np.ndarray
        Current-year posterior samples (from Bayesian model).
    stat : str
        Rate stat name (k_rate, bb_rate, hr_rate).
    player_type : str
        'hitter' or 'pitcher'.
    n_years : int
        Number of years to project forward.
    random_seed : int

    Returns
    -------
    list[np.ndarray]
        One array per future year, each with same length as input.
    """
    rng = np.random.default_rng(random_seed)
    pop_mean = _POP_MEANS.get(stat, {}).get(player_type, 0.15)
    innovation_sd = _INNOVATION_SD.get(stat, 0.015)

    projections = []
    prev = current_samples.copy()

    for year in range(n_years):
        # Draw rho for this step (accounts for rho uncertainty)
        rho = np.clip(rng.normal(_RHO_MEAN, _RHO_SD, size=len(prev)), 0.5, 0.98)

        # AR(1): regress toward population mean
        mean_next = rho * prev + (1 - rho) * pop_mean

        # Add innovation noise (increases with each year)
        noise = rng.normal(0, innovation_sd * (1 + 0.2 * year), size=len(prev))
        next_samples = np.clip(mean_next + noise, 0.001, 0.600)

        projections.append(next_samples)
        prev = next_samples

    return projections


def project_multi_year(
    current_posteriors: dict[int, dict[str, np.ndarray]],
    player_info: pd.DataFrame,
    player_type: str = "hitter",
    n_years: int = 3,
    current_season: int = 2026,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Project all players N years forward.

    Parameters
    ----------
    current_posteriors : dict[int, dict[str, np.ndarray]]
        player_id -> {k_rate, bb_rate, hr_rate} current samples.
    player_info : pd.DataFrame
        Columns: player_id, player_name, age, games (current season).
    player_type : str
        'hitter' or 'pitcher'.
    n_years : int
        Years to project forward (1-5).
    current_season : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per (player, year) with projected rates and games.
    """
    rng = np.random.default_rng(random_seed)
    age_decay = _AGE_GAMES_DECAY[player_type]

    rows = []
    for pid, posteriors in current_posteriors.items():
        info = player_info[player_info["player_id"] == pid]
        if info.empty:
            continue

        info_row = info.iloc[0]
        base_age = int(info_row.get("age", 28))
        base_games = float(info_row.get("games", 140 if player_type == "hitter" else 30))
        name = info_row.get("player_name", "")

        # Project rates forward
        rate_projections: dict[str, list[np.ndarray]] = {}
        for stat in ["k_rate", "bb_rate", "hr_rate"]:
            if stat in posteriors:
                rate_projections[stat] = project_rates_forward(
                    posteriors[stat], stat, player_type,
                    n_years=n_years, random_seed=random_seed + pid,
                )

        # Build rows per future year
        for y in range(n_years):
            season = current_season + y + 1
            age = base_age + y + 1

            # Playing time decay
            decay = age_decay.get(age, age_decay.get(min(age, 40), 0.40))
            proj_games = base_games * decay

            row = {
                "player_id": pid,
                "player_name": name,
                "season": season,
                "age": age,
                "projected_games": round(proj_games, 1),
            }

            for stat, projs in rate_projections.items():
                samples = projs[y]
                row[f"{stat}_mean"] = float(np.mean(samples))
                row[f"{stat}_sd"] = float(np.std(samples))
                row[f"{stat}_p10"] = float(np.percentile(samples, 10))
                row[f"{stat}_p90"] = float(np.percentile(samples, 90))

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Multi-year projections: %d players x %d years = %d rows",
            df["player_id"].nunique(), n_years, len(df),
        )
    return df
